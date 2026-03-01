"""Step 3: Train LightGBM (Optuna-tuned params, 5-fold).

Loads features from features/, trains 41 per-target binary classifiers.

Output: checkpoints_lgbm/lgbm_predictions.npz (oof_preds, test_preds)

Runtime: ~20-30 minutes.
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
CHECKPOINT_DIR = Path("checkpoints_lgbm")

# Optuna-tuned params (L7, 30 trials)
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


def main():
    t0 = time.time()
    print("=" * 60)
    print("Step 3: Train LightGBM (5-fold × 41 targets)")
    print("=" * 60)

    # 1. Load features
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

    X_train = train_feat.drop("customer_id").to_numpy().astype(np.float32)
    X_test = test_feat.drop("customer_id").to_numpy().astype(np.float32)
    y_train = train_tgt.select(target_cols).to_numpy().astype(np.float32)
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"  Features: {len(cat_feature_names)} cat, {len(feature_cols) - len(cat_feature_names)} num")

    # 2. Check cache
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CHECKPOINT_DIR / "lgbm_predictions.npz"
    if cache_file.exists():
        print(f"\n  Predictions exist at {cache_file}! Delete to retrain.")
        return

    # 3. Train
    n_train, n_test = X_train.shape[0], X_test.shape[0]
    n_targets = len(target_cols)
    oof_preds = np.zeros((n_train, n_targets))
    test_preds_sum = np.zeros((n_test, n_targets))
    fold_aucs = []

    kf = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    print(f"\n[2/4] Training {N_FOLDS}-Fold × {n_targets} targets...", flush=True)

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(np.arange(n_train), y_train)):
        t_fold = time.time()
        print(f"\n  ── Fold {fold_idx+1}/{N_FOLDS} "
              f"(train={len(tr_idx):,}, val={len(val_idx):,}) ──", flush=True)

        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        fold_test_preds = np.zeros((n_test, n_targets))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for i, col in enumerate(target_cols):
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
                fold_test_preds[:, i] = model.predict_proba(X_test)[:, 1]
                del model
                if (i + 1) % 10 == 0 or i == n_targets - 1:
                    print(f"    {i+1}/{n_targets} targets done", flush=True)

        fold_auc, _ = compute_macro_auc(y_val, oof_preds[val_idx], target_cols)
        test_preds_sum += fold_test_preds
        fold_aucs.append(fold_auc)
        del X_tr, X_val, y_tr, y_val; gc.collect()
        print(f"  Fold {fold_idx+1} AUC={fold_auc:.4f} ({(time.time()-t_fold)/60:.1f} min)", flush=True)

    test_preds_avg = test_preds_sum / N_FOLDS

    # 4. Results
    oof_auc, _ = compute_macro_auc(y_train, oof_preds, target_cols)
    print(f"\n[3/4] Results:")
    print(f"  Per-fold AUC: {['%.4f' % a for a in fold_aucs]}")
    print(f"  OOF Macro ROC-AUC: {oof_auc:.4f}")

    # Save predictions
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
    submit.write_parquet("submissions/lgbm.parquet")
    print(f"  Saved: submissions/lgbm.parquet")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min. OOF={oof_auc:.4f}")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)
    np.random.seed(SEED)
    main()
