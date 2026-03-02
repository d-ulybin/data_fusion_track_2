"""Shared constants and helper functions."""

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

SEED = 1234
DATA_DIR = "/kaggle/input/datafusion26/"
N_FOLDS = 5


def get_device():
    """Auto-detect best available torch device: cuda > mps > cpu."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_macro_auc(y_true, y_pred, target_cols):
    """Compute macro ROC-AUC with per-target breakdown."""
    aucs = {}
    for i, col in enumerate(target_cols):
        y_t = y_true[:, i]
        if y_t.sum() >= 2 and (len(y_t) - y_t.sum()) >= 2:
            aucs[col] = roc_auc_score(y_t, y_pred[:, i])
    return float(np.mean(list(aucs.values()))), aucs


def to_ranks(arr):
    """Convert predictions to per-column ranks."""
    return np.column_stack([rankdata(arr[:, i]) for i in range(arr.shape[1])])


def verify_submission(submit, sample):
    """Assert submission matches sample format."""
    assert submit.shape == sample.shape, f"Shape mismatch: {submit.shape} vs {sample.shape}"
    assert submit.columns == sample.columns, "Column mismatch"
    for col in submit.columns:
        assert submit[col].dtype == sample[col].dtype, f"Dtype mismatch for {col}"
    print(f"  Format verified: {submit.shape[0]:,} rows, {submit.shape[1]} cols")
