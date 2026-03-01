"""Step 7: Stacking — Ridge + LGBM meta-learner + combo.

Meta-features (369): 3x41 base + 3x41 pairwise |diffs| + 3x41 pairwise prods.
Ridge and LGBM meta-learners trained with 5-fold OOF.
Final combo: alpha * rank(meta) + (1-alpha) * rank(blend).

Output: submissions/stacking.parquet

Runtime: ~2 minutes.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from utils import SEED, DATA_DIR, N_FOLDS, compute_macro_auc, to_ranks, verify_submission

N_META_FOLDS = 5


def build_meta_features(oof_a, oof_b, oof_c):
    """Build 369-dimensional meta-feature matrix from 3 models."""
    return np.hstack([
        oof_a, oof_b, oof_c,
        np.abs(oof_a - oof_b), np.abs(oof_a - oof_c), np.abs(oof_b - oof_c),
        oof_a * oof_b, oof_a * oof_c, oof_b * oof_c,
    ]).astype(np.float32)


def stack_ridge(X_train, y_train, X_test, target_cols):
    """Ridge per-target stacking with alpha selection."""
    n_train, n_targets = y_train.shape
    n_test = X_test.shape[0]
    oof_preds = np.zeros((n_train, n_targets), dtype=np.float32)
    test_preds = np.zeros((n_test, n_targets), dtype=np.float32)

    kf = KFold(n_splits=N_META_FOLDS, shuffle=True, random_state=SEED)
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    t0 = time.time()

    for t_idx in range(n_targets):
        y_t = y_train[:, t_idx]
        best_alpha, best_oof_auc, best_oof, best_test = 1.0, 0, None, None

        for alpha in alphas:
            fold_oof = np.zeros(n_train, dtype=np.float32)
            fold_test = np.zeros(n_test, dtype=np.float32)
            for tr_idx, val_idx in kf.split(range(n_train)):
                model = Ridge(alpha=alpha, random_state=SEED)
                model.fit(X_train[tr_idx], y_t[tr_idx])
                fold_oof[val_idx] = model.predict(X_train[val_idx])
                fold_test += model.predict(X_test) / N_META_FOLDS

            if y_t.sum() >= 2 and (len(y_t) - y_t.sum()) >= 2:
                auc = roc_auc_score(y_t, fold_oof)
                if auc > best_oof_auc:
                    best_oof_auc = auc
                    best_alpha = alpha
                    best_oof = fold_oof.copy()
                    best_test = fold_test.copy()

        if best_oof is not None:
            oof_preds[:, t_idx] = best_oof
            test_preds[:, t_idx] = best_test

        if (t_idx + 1) % 10 == 0 or t_idx == n_targets - 1:
            elapsed = time.time() - t0
            print(f"    Ridge: {t_idx+1}/{n_targets} targets ({elapsed:.1f}s)", flush=True)

    return oof_preds, test_preds


def stack_lgbm_meta(X_train, y_train, X_test, target_cols):
    """LGBM per-target stacking."""
    import lightgbm as lgb

    n_train, n_targets = y_train.shape
    n_test = X_test.shape[0]
    oof_preds = np.zeros((n_train, n_targets), dtype=np.float32)
    test_preds = np.zeros((n_test, n_targets), dtype=np.float32)

    kf = KFold(n_splits=N_META_FOLDS, shuffle=True, random_state=SEED)
    params = dict(objective="binary", metric="auc", num_leaves=8, max_depth=3,
                  learning_rate=0.05, n_estimators=200, subsample=0.8,
                  colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0,
                  min_child_samples=1000, random_state=SEED, verbose=-1, n_jobs=-1)

    t0 = time.time()
    for t_idx in range(n_targets):
        y_t = y_train[:, t_idx]
        for tr_idx, val_idx in kf.split(range(n_train)):
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train[tr_idx], y_t[tr_idx],
                      eval_set=[(X_train[val_idx], y_t[val_idx])],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
            oof_preds[val_idx, t_idx] = model.predict_proba(X_train[val_idx])[:, 1]
            test_preds[:, t_idx] += model.predict_proba(X_test)[:, 1] / N_META_FOLDS

        if (t_idx + 1) % 10 == 0 or t_idx == n_targets - 1:
            elapsed = time.time() - t0
            print(f"    LGBM meta: {t_idx+1}/{n_targets} targets ({elapsed:.1f}s)", flush=True)

    return oof_preds, test_preds


def optimize_rank_blend_3(oof_nn, oof_lgbm, oof_pb, y, target_cols, step=0.10):
    """Per-target weight optimization for 3-model rank blend."""
    oof_ranks = [to_ranks(oof_nn), to_ranks(oof_lgbm), to_ranks(oof_pb)]
    n_targets = len(target_cols)
    weights = np.zeros((n_targets, 3))
    for i in range(n_targets):
        y_t = y[:, i]
        if y_t.sum() < 2 or (len(y_t) - y_t.sum()) < 2:
            weights[i] = [1/3, 1/3, 1/3]
            continue
        best_auc, best_w = 0.0, [1/3, 1/3, 1/3]
        for w0 in np.arange(0, 1.01, step):
            for w1 in np.arange(0, 1.01 - w0, step):
                w2 = 1.0 - w0 - w1
                if w2 < -0.001:
                    continue
                blended = w0*oof_ranks[0][:, i] + w1*oof_ranks[1][:, i] + w2*oof_ranks[2][:, i]
                auc = roc_auc_score(y_t, blended)
                if auc > best_auc:
                    best_auc = auc
                    best_w = [w0, w1, w2]
        weights[i] = best_w
    return weights, oof_ranks


def main():
    t0 = time.time()
    print("=" * 60)
    print("Step 7: Stacking (Ridge + LGBM meta + combo)")
    print("=" * 60)

    # Load targets
    train_tgt = pl.read_parquet(f"{DATA_DIR}train_target.parquet")
    target_cols = [c for c in train_tgt.columns if c.startswith("target_")]
    y = train_tgt.select(target_cols).to_numpy().astype(np.float32)
    n_train, n_targets = y.shape

    # Load predictions
    print("\n[1/5] Loading predictions...")
    d = np.load("blend_artifacts/blend_data.npz")
    oof_nn = d["oof_nn"].astype(np.float32)
    test_nn = d["test_nn"].astype(np.float32)
    oof_lgbm = d["oof_lgbm"].astype(np.float32)
    test_lgbm = d["test_lgbm"].astype(np.float32)
    oof_pb = d["oof_pb"].astype(np.float32)
    test_pb = d["test_pb"].astype(np.float32)

    n_test = test_nn.shape[0]
    nn_auc, _ = compute_macro_auc(y, oof_nn, target_cols)
    lgbm_auc, _ = compute_macro_auc(y, oof_lgbm, target_cols)
    pb_auc, _ = compute_macro_auc(y, oof_pb, target_cols)
    print(f"  NN: {nn_auc:.4f}, LGBM: {lgbm_auc:.4f}, PyBoost: {pb_auc:.4f}")

    # Rank blend baseline
    print("\n  Computing rank blend baseline...")
    blend_weights, oof_ranks = optimize_rank_blend_3(oof_nn, oof_lgbm, oof_pb, y, target_cols)
    test_ranks = [to_ranks(test_nn), to_ranks(test_lgbm), to_ranks(test_pb)]

    baseline_oof = np.zeros_like(oof_nn)
    baseline_test = np.zeros_like(test_nn)
    for i in range(n_targets):
        w = blend_weights[i]
        baseline_oof[:, i] = w[0]*oof_ranks[0][:, i] + w[1]*oof_ranks[1][:, i] + w[2]*oof_ranks[2][:, i]
        baseline_test[:, i] = w[0]*test_ranks[0][:, i] + w[1]*test_ranks[1][:, i] + w[2]*test_ranks[2][:, i]

    baseline_auc, _ = compute_macro_auc(y, baseline_oof, target_cols)
    print(f"  Rank blend baseline: {baseline_auc:.4f}")

    # Build meta-features
    print("\n[2/5] Building meta-features...")
    X_meta_train = build_meta_features(oof_nn, oof_lgbm, oof_pb)
    X_meta_test = build_meta_features(test_nn, test_lgbm, test_pb)
    print(f"  Meta-features: {X_meta_train.shape[1]}")

    # Ridge stacking
    print("\n[3/5] Ridge stacking...")
    ridge_oof, ridge_test = stack_ridge(X_meta_train, y, X_meta_test, target_cols)
    ridge_auc, _ = compute_macro_auc(y, ridge_oof, target_cols)
    print(f"  Ridge OOF: {ridge_auc:.4f} (vs baseline: {ridge_auc - baseline_auc:+.4f})")

    # LGBM stacking
    print("\n[4/5] LGBM meta stacking...")
    lgbm_meta_oof, lgbm_meta_test = stack_lgbm_meta(X_meta_train, y, X_meta_test, target_cols)
    lgbm_meta_auc, _ = compute_macro_auc(y, lgbm_meta_oof, target_cols)
    print(f"  LGBM meta OOF: {lgbm_meta_auc:.4f} (vs baseline: {lgbm_meta_auc - baseline_auc:+.4f})")

    # Combo
    print("\n[5/5] Optimizing combo...")
    if lgbm_meta_auc > ridge_auc:
        meta_oof, meta_test, meta_name = lgbm_meta_oof, lgbm_meta_test, "LGBM meta"
    else:
        meta_oof, meta_test, meta_name = ridge_oof, ridge_test, "Ridge"

    meta_oof_rank = to_ranks(meta_oof)
    best_combo_auc, best_alpha = 0, 0.5
    for alpha in np.arange(0, 1.05, 0.05):
        combo = alpha * meta_oof_rank + (1 - alpha) * baseline_oof
        auc, _ = compute_macro_auc(y, combo, target_cols)
        if auc > best_combo_auc:
            best_combo_auc = auc
            best_alpha = alpha

    meta_test_rank = to_ranks(meta_test)
    combo_test = best_alpha * meta_test_rank + (1 - best_alpha) * baseline_test
    print(f"  Combo: {meta_name} {best_alpha:.0%} + blend {1-best_alpha:.0%}, "
          f"OOF {best_combo_auc:.4f}")

    # Summary
    print(f"\n{'='*60}")
    results = [
        ("Rank blend", baseline_auc, baseline_test),
        ("Ridge meta", ridge_auc, ridge_test),
        ("LGBM meta", lgbm_meta_auc, lgbm_meta_test),
        ("Combo", best_combo_auc, combo_test),
    ]
    best_name, best_auc, best_preds = max(results, key=lambda x: x[1])
    for name, auc, _ in sorted(results, key=lambda x: -x[1]):
        marker = " <<<" if name == best_name else ""
        print(f"  {name:<20s} {auc:.4f}{marker}")

    # Save
    test_ids = pl.read_parquet(f"{DATA_DIR}test_main_features.parquet", columns=["customer_id"])
    sample = pl.read_parquet(f"{DATA_DIR}sample_submit.parquet")
    predict_cols = [c.replace("target_", "predict_") for c in target_cols]
    submit = pl.DataFrame({"customer_id": test_ids["customer_id"]}).hstack(
        pl.DataFrame(best_preds.astype(np.float64), schema=predict_cols)
    )
    verify_submission(submit, sample)
    Path("submissions").mkdir(exist_ok=True)
    submit.write_parquet("submissions/stacking.parquet")
    print(f"\n  Saved: submissions/stacking.parquet")
    print(f"  Best: {best_name} (OOF {best_auc:.4f})")
    print(f"\nDone in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)
    np.random.seed(SEED)
    main()
