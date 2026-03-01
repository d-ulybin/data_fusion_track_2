"""Step 4: Train PyBoost (SketchBoost, requires NVIDIA GPU with CUDA).

Loads features from features/, trains SketchBoost multi-output model.

Output: checkpoints_pyboost/pyboost_predictions.npz (oof_preds, test_preds)

Runtime: ~1-2 hours on T4 GPU.
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from utils import SEED, DATA_DIR, N_FOLDS, compute_macro_auc

FEATURES_DIR = Path("features")
CHECKPOINT_DIR = Path("checkpoints_pyboost")

# Optuna-tuned params (25 trials on fold 0)
PARAMS = dict(
    ntrees=5000,
    lr=0.0566,
    max_depth=7,
    min_data_in_leaf=88,
    lambda_l2=2.076,
    subsample=0.88,
    colsample=0.76,
    max_bin=64,
    es=200,
    verbose=100,
    gd_steps=1,
    use_hess=False,
)


def main():
    t0 = time.time()
    print("=" * 60)
    print("Step 4: Train PyBoost (SketchBoost)")
    print("=" * 60)

    # Check CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            print("\nERROR: PyBoost requires NVIDIA GPU with CUDA.")
            print("  This script uses cupy + py-boost which only work on CUDA GPUs.")
            print("  Skipping PyBoost training.")
            return
    except ImportError:
        print("\nERROR: torch not found. Install PyTorch with CUDA support.")
        return

    try:
        import cupy as cp
        from py_boost import SketchBoost
    except ImportError:
        print("\nERROR: cupy and/or py-boost not installed.")
        print("  pip install py-boost cupy-cuda12x")
        return

    # 1. Load features
    print("\n[1/4] Loading features...")
    with open(FEATURES_DIR / "meta.json") as f:
        meta = json.load(f)
    target_cols = meta["target_cols"]

    train_feat = pl.read_parquet(FEATURES_DIR / "train_features.parquet")
    test_feat = pl.read_parquet(FEATURES_DIR / "test_features.parquet")
    train_tgt = pl.read_parquet(FEATURES_DIR / "targets.parquet")

    X_train = train_feat.drop("customer_id").to_numpy().astype(np.float32)
    X_test = test_feat.drop("customer_id").to_numpy().astype(np.float32)
    y = train_tgt.select(target_cols).to_numpy().astype(np.float32)
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    # 2. Check cache
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CHECKPOINT_DIR / "pyboost_predictions.npz"
    if cache_file.exists():
        print(f"\n  Predictions exist at {cache_file}! Delete to retrain.")
        return

    # 3. Train
    print(f"\n[2/4] Training SketchBoost {N_FOLDS}-fold...")
    mskf = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros_like(y, dtype=np.float64)
    test_preds = np.zeros((X_test.shape[0], len(target_cols)), dtype=np.float64)
    fold_scores = []

    for fold_idx, (tr_idx, val_idx) in enumerate(mskf.split(X_train, y)):
        t_fold = time.time()
        print(f"\n  ── Fold {fold_idx+1}/{N_FOLDS} ──", flush=True)

        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = SketchBoost('bce', **PARAMS)
        model.fit(X_tr, y_tr, eval_sets=[{'X': X_val, 'y': y_val}])

        val_raw = model.predict(X_val)
        test_raw = model.predict(X_test)
        val_prob = 1.0 / (1.0 + np.exp(-val_raw))
        test_prob = 1.0 / (1.0 + np.exp(-test_raw))

        oof_preds[val_idx] = val_prob
        test_preds += test_prob / N_FOLDS

        fold_auc = roc_auc_score(y_val, val_prob, average='macro')
        fold_scores.append(fold_auc)
        print(f"  Fold {fold_idx+1} AUC={fold_auc:.5f} ({(time.time()-t_fold)/60:.1f} min)", flush=True)

        del model, X_tr, X_val, y_tr, y_val, val_raw, test_raw, val_prob, test_prob
        cp.get_default_memory_pool().free_all_blocks(); gc.collect()

    # 4. Results
    macro_auc = float(np.mean([
        roc_auc_score(y[:, i], oof_preds[:, i]) for i in range(len(target_cols))
    ]))
    print(f"\n[3/4] Results:")
    print(f"  Fold AUCs: {['%.5f' % s for s in fold_scores]}")
    print(f"  OOF Macro AUC: {macro_auc:.5f}")

    # Save
    np.savez_compressed(cache_file,
                        oof_preds=oof_preds, test_preds=test_preds,
                        fold_scores=fold_scores, macro_auc=macro_auc)
    print(f"  Saved: {cache_file}")

    # Save submission
    print("\n[4/4] Saving submission...")
    from utils import verify_submission
    sample = pl.read_parquet(f"{DATA_DIR}sample_submit.parquet")
    predict_cols = [c.replace("target_", "predict_") for c in target_cols]
    submit = pl.DataFrame({"customer_id": test_feat["customer_id"]}).hstack(
        pl.DataFrame(test_preds.astype(np.float64), schema=predict_cols)
    )
    verify_submission(submit, sample)
    Path("submissions").mkdir(exist_ok=True)
    submit.write_parquet("submissions/pyboost.parquet")
    print(f"  Saved: submissions/pyboost.parquet")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min. OOF={macro_auc:.5f}")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)
    np.random.seed(SEED)
    main()
