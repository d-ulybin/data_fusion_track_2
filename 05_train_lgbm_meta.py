"""Step 5: Train LGBM with cross-target meta-features (L8).

Uses OOF predictions from NN, LGBM, and PyBoost as additional features.
For each target_i: base features + 120 meta-features (40 from each model,
excluding the own target to prevent leakage).

Output: checkpoints_lgbm_meta/lgbm_predictions.npz

Runtime: ~30-40 minutes.
"""

import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from utils import SEED, DATA_DIR, N_FOLDS, compute_macro_auc

FEATURES_DIR = Path("features")
CHECKPOINT_DIR = Path("checkpoints_lgbm_meta")

# Same Optuna-tuned params as L7
LGBM_PARAMS = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.050216,
    num_leaves=34,
    max_depth=10,
    min_child_samples=102,
    n_estimators=1500,
    subsample=0.521715,
    colsample_bytree=0.205092,
    reg_alpha=8.146025,
    reg_lambda=7.761833,
    min_split_gain=0.424512,
    subsample_freq=2,
    random_state=SEED,
    verbose=-1,
    n_jobs=-1,
)
EARLY_STOPPING_ROUNDS = 100


def load_model_predictions():
    """Load OOF and test predictions from all 3 models."""
    print("\n  Loading OOF predictions from 3 models...")

    # NN
    nn_dir = Path("checkpoints_nn")
    n_train_check = np.load(nn_dir / "fold_0.npz")["val_preds"].shape[1]
    oof_parts, test_parts = {}, []
    for fi in range(N_FOLDS):
        d = np.load(nn_dir / f"fold_{fi}.npz")
        for idx, pred in zip(d["val_idx"], d["val_preds"]):
            oof_parts[int(idx)] = pred
        test_parts.append(d["test_preds"])
    n_train = max(oof_parts.keys()) + 1
    nn_oof = np.zeros((n_train, n_train_check), dtype=np.float32)
    for idx, pred in oof_parts.items():
        nn_oof[idx] = pred
    nn_test = np.mean(test_parts, axis=0).astype(np.float32)
    print(f"    NN: OOF {nn_oof.shape}, test {nn_test.shape}")

    # LGBM
    d = np.load("checkpoints_lgbm/lgbm_predictions.npz")
    lgbm_oof = d["oof_preds"].astype(np.float32)
    lgbm_test = d["test_preds"].astype(np.float32)
    print(f"    LGBM: OOF {lgbm_oof.shape}, test {lgbm_test.shape}")

    # PyBoost
    d = np.load("checkpoints_pyboost/pyboost_predictions.npz")
    pb_oof = d["oof_preds"].astype(np.float32)
    pb_test = d["test_preds"].astype(np.float32)
    print(f"    PyBoost: OOF {pb_oof.shape}, test {pb_test.shape}")

    meta_train = np.hstack([lgbm_oof, nn_oof, pb_oof])
    meta_test = np.hstack([lgbm_test, nn_test, pb_test])
    print(f"    Combined meta: {meta_train.shape}")
    return meta_train, meta_test


def main():
    t0 = time.time()
    print("=" * 60)
    print("Step 5: LGBM with cross-target meta-features (L8)")
    print("=" * 60)

    # 1. Load base features
    print("\n[1/4] Loading features...")
    with open(FEATURES_DIR / "meta.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_names"]
    cat_feature_names = meta["cat_cols"]
    target_cols = meta["target_cols"]
    cat_indices = [feature_cols.index(c) for c in cat_feature_names]

    train_feat = pl.read_parquet(FEATURES_DIR / "train_features.parquet")
    test_feat = pl.read_parquet(FEATURES_DIR / "test_features.parquet")
    train_tgt = pl.read_parquet(FEATURES_DIR / "targets.parquet")

    X_train_base = train_feat.drop("customer_id").to_numpy().astype(np.float32)
    X_test_base = test_feat.drop("customer_id").to_numpy().astype(np.float32)
    y_train = train_tgt.select(target_cols).to_numpy().astype(np.float32)

    n_targets = len(target_cols)
    n_base = len(feature_cols)

    # 2. Load meta predictions
    meta_train, meta_test = load_model_predictions()

    # Combine
    X_train_full = np.hstack([X_train_base, meta_train])
    X_test_full = np.hstack([X_test_base, meta_test])
    del X_train_base, X_test_base, meta_train, meta_test; gc.collect()

    n_total = X_train_full.shape[1]
    print(f"\n  Features: {n_base} base + {n_total - n_base} meta = {n_total} total")

    # 3. Check cache
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CHECKPOINT_DIR / "lgbm_predictions.npz"
    if cache_file.exists():
        print(f"\n  Predictions exist at {cache_file}! Delete to retrain.")
        return

    # Precompute own-target column indices to mask
    # For target_i: mask columns [n_base+i, n_base+41+i, n_base+82+i]
    own_cols = np.array([
        [n_base + i, n_base + n_targets + i, n_base + 2 * n_targets + i]
        for i in range(n_targets)
    ])

    # 4. Train
    print(f"\n[2/4] Training {N_FOLDS}-Fold × {n_targets} targets (with meta)...", flush=True)
    kf = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    n_train, n_test = X_train_full.shape[0], X_test_full.shape[0]

    oof_preds = np.zeros((n_train, n_targets))
    test_preds_sum = np.zeros((n_test, n_targets))
    fold_aucs = []

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(np.arange(n_train), y_train)):
        t_fold = time.time()
        print(f"\n  ── Fold {fold_idx+1}/{N_FOLDS} ──", flush=True)

        X_tr = X_train_full[tr_idx]
        X_val = X_train_full[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        fold_test_preds = np.zeros((n_test, n_targets))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for i, col in enumerate(target_cols):
                cols_to_mask = own_cols[i]

                # Mask own-target meta-features to NaN
                saved_tr = X_tr[:, cols_to_mask].copy()
                saved_val = X_val[:, cols_to_mask].copy()
                saved_test = X_test_full[:, cols_to_mask].copy()
                X_tr[:, cols_to_mask] = np.nan
                X_val[:, cols_to_mask] = np.nan
                X_test_full[:, cols_to_mask] = np.nan

                model = lgb.LGBMClassifier(**LGBM_PARAMS)
                model.fit(
                    X_tr, y_tr[:, i],
                    eval_set=[(X_val, y_val[:, i])],
                    eval_metric="auc",
                    callbacks=[
                        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                        lgb.log_evaluation(period=0),
                    ],
                    categorical_feature=cat_indices,
                )
                oof_preds[val_idx, i] = model.predict_proba(X_val)[:, 1]
                fold_test_preds[:, i] = model.predict_proba(X_test_full)[:, 1]

                # Restore
                X_tr[:, cols_to_mask] = saved_tr
                X_val[:, cols_to_mask] = saved_val
                X_test_full[:, cols_to_mask] = saved_test
                del model, saved_tr, saved_val, saved_test

                if (i + 1) % 10 == 0 or i == n_targets - 1:
                    print(f"    {i+1}/{n_targets} targets done", flush=True)

        fold_auc, _ = compute_macro_auc(y_val, oof_preds[val_idx], target_cols)
        test_preds_sum += fold_test_preds
        fold_aucs.append(fold_auc)
        del X_tr, X_val, y_tr, y_val; gc.collect()
        print(f"  Fold {fold_idx+1} AUC={fold_auc:.4f} ({(time.time()-t_fold)/60:.1f} min)", flush=True)

    test_preds_avg = test_preds_sum / N_FOLDS

    # Results
    oof_auc, _ = compute_macro_auc(y_train, oof_preds, target_cols)
    print(f"\n[3/4] Results:")
    print(f"  Per-fold AUC: {['%.4f' % a for a in fold_aucs]}")
    print(f"  OOF Macro ROC-AUC: {oof_auc:.4f}")

    np.savez(cache_file, oof_preds=oof_preds, test_preds=test_preds_avg,
             fold_aucs=np.array(fold_aucs))
    print(f"  Saved: {cache_file}")

    # Save submission
    print("\n[4/4] Saving submission...")
    from utils import verify_submission
    sample = pl.read_parquet(f"{DATA_DIR}sample_submit.parquet")
    predict_cols = [c.replace("target_", "predict_") for c in target_cols]
    submit = pl.DataFrame({"customer_id": test_feat["customer_id"]}).hstack(
        pl.DataFrame(test_preds_avg.astype(np.float64), schema=predict_cols)
    )
    verify_submission(submit, sample)
    Path("submissions").mkdir(exist_ok=True)
    submit.write_parquet("submissions/lgbm_meta.parquet")
    print(f"  Saved: submissions/lgbm_meta.parquet")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min. OOF={oof_auc:.4f}")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)
    np.random.seed(SEED)
    main()
