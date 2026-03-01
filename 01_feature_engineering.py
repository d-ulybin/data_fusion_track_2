"""Step 1: Feature Engineering.

Loads raw parquet data, builds all features, saves to features/ directory.
Subsequent scripts load from features/ instead of raw data.

Runtime: ~3-5 minutes.
"""

import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix, hstack as sparse_hstack
from sklearn.decomposition import TruncatedSVD

from utils import SEED, DATA_DIR

FEATURES_DIR = Path("features")
NULL_PCA_COMPONENTS = 20
INDIVIDUAL_NULL_THRESHOLD = 0.05

# ── Feature Engineering Config ────────────────────────────────────

NULL_GROUPS_MAIN = {
    "null_group_A": "num_feature_1",
    "null_group_B": "num_feature_8",
    "null_group_C": "num_feature_11",
    "null_group_D": "num_feature_38",
    "null_group_E": "num_feature_72",
}

NULL_GROUPS_EXTRA = {
    "null_extra_E4": "num_feature_155",
    "null_extra_E5": "num_feature_140",
    "null_extra_E8": "num_feature_139",
    "null_extra_E9": "num_feature_222",
    "null_extra_E10": "num_feature_134",
    "null_extra_E11": "num_feature_158",
    "null_extra_E12": "num_feature_240",
    "null_extra_E13": "num_feature_194",
    "null_extra_E14": "num_feature_197",
    "null_extra_E15": "num_feature_167",
}

NUM_DIFFS = [
    ("num_feature_7", "num_feature_42"),
    ("num_feature_33", "num_feature_42"),
    ("num_feature_36", "num_feature_42"),
    ("num_feature_7", "num_feature_132"),
    ("num_feature_33", "num_feature_29"),
    ("num_feature_7", "num_feature_57"),
    ("num_feature_33", "num_feature_132"),
    ("num_feature_7", "num_feature_125"),
    ("num_feature_29", "num_feature_42"),
    ("num_feature_41", "num_feature_42"),
    ("num_feature_56", "num_feature_34"),
    ("num_feature_7", "num_feature_34"),
    ("num_feature_33", "num_feature_13"),
    ("num_feature_7", "num_feature_118"),
    ("num_feature_36", "num_feature_34"),
    ("num_feature_29", "num_feature_13"),
    ("num_feature_118", "num_feature_42"),
    ("num_feature_125", "num_feature_118"),
]

CAT_INTERACTIONS = [
    ("cat_feature_66", "cat_feature_46"),
    ("cat_feature_66", "cat_feature_39"),
    ("cat_feature_66", "cat_feature_48"),
    ("cat_feature_66", "cat_feature_9"),
    ("cat_feature_66", "cat_feature_52"),
]

RATIO_FEATURES = [
    ("num_feature_62", "num_feature_79"),
    ("num_feature_27", "num_feature_79"),
    ("num_feature_76", "num_feature_79"),
    ("num_feature_62", "num_feature_86"),
    ("num_feature_116", "num_feature_124"),
    ("num_feature_36", "num_feature_79"),
    ("num_feature_41", "num_feature_79"),
    ("num_feature_83", "num_feature_108"),
    ("num_feature_83", "num_feature_103"),
]

DUPLICATE_CATS = [
    "cat_feature_24",
    "cat_feature_25",
    "cat_feature_26",
    "cat_feature_29",
    "cat_feature_50",
    "cat_feature_63",
]


# ── Helper functions ──────────────────────────────────────────────


def filter_extra_features(train_extra, test_extra):
    """Remove extra features that are >99% null or constant."""
    extra_cols = [c for c in train_extra.columns if c != "customer_id"]
    n_rows = train_extra.height
    to_drop = set()
    for col in extra_cols:
        s = train_extra[col]
        if s.null_count() / n_rows > 0.99:
            to_drop.add(col)
            continue
        non_null = s.drop_nulls()
        if len(non_null) == 0 or non_null.n_unique() <= 1:
            to_drop.add(col)
    keep_cols = ["customer_id"] + [c for c in extra_cols if c not in to_drop]
    print(f"  Extra: {len(extra_cols)} total, {len(to_drop)} removed, "
          f"{len(keep_cols) - 1} kept")
    return train_extra.select(keep_cols), test_extra.select(keep_cols)


def deduplicate_extra_features(train_extra, test_extra):
    """Remove exact duplicate columns using MD5 hashing."""
    extra_cols = [c for c in train_extra.columns if c.startswith("num_feature")]
    seen_hashes = {}
    dup_cols = []
    BATCH = 200
    for start in range(0, len(extra_cols), BATCH):
        batch_cols = extra_cols[start:start + BATCH]
        batch_np = train_extra.select(batch_cols).to_numpy()
        for i, col in enumerate(batch_cols):
            h = hashlib.md5(batch_np[:, i].tobytes()).hexdigest()
            if h in seen_hashes:
                dup_cols.append(col)
            else:
                seen_hashes[h] = col
        del batch_np
    if dup_cols:
        train_extra = train_extra.drop(dup_cols)
        test_extra = test_extra.drop(dup_cols)
        print(f"  Dedup: removed {len(dup_cols)} → {len(extra_cols) - len(dup_cols)} unique")
    return train_extra, test_extra


def null_pattern_pca(train_extra_raw, test_extra_raw, n_components=20):
    """PCA of binary null-indicator matrix from all extra columns (sparse)."""
    all_cols = [c for c in train_extra_raw.columns if c.startswith("num_feature")]
    n_train = train_extra_raw.height
    BATCH = 500
    sparse_blocks = []
    for start in range(0, len(all_cols), BATCH):
        batch_cols = all_cols[start:start + BATCH]
        train_np = train_extra_raw.select(batch_cols).to_numpy()
        test_np = test_extra_raw.select(batch_cols).to_numpy()
        combined = np.vstack([train_np, test_np])
        del train_np, test_np
        null_bits = np.isnan(combined).astype(np.float32)
        sparse_blocks.append(csr_matrix(null_bits))
        del combined, null_bits
    null_sparse = sparse_hstack(sparse_blocks, format="csr")
    del sparse_blocks; gc.collect()

    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    pca_features = svd.fit_transform(null_sparse)
    var_explained = svd.explained_variance_ratio_.sum()
    print(f"  Null PCA: {null_sparse.shape[1]} cols → {n_components} components, "
          f"var explained: {var_explained:.3f}")
    del null_sparse; gc.collect()

    return pca_features[:n_train].astype(np.float32), pca_features[n_train:].astype(np.float32)


def add_null_indicators(df, groups_dict):
    """Add binary null-indicator features from named groups."""
    new_cols = []
    for name, ref_col in groups_dict.items():
        if ref_col in df.columns:
            new_cols.append(pl.col(ref_col).is_null().cast(pl.Int8).alias(name))
    if new_cols:
        df = df.with_columns(new_cols)
    return df


def select_null_indicator_features(train_main, num_cols, train_tgt, target_cols, threshold):
    """Select num features whose null indicator has |corr| > threshold with any target."""
    null_arrays, valid_cols = [], []
    for col in num_cols:
        arr = train_main[col].is_null().cast(pl.Float32).to_numpy()
        if arr.std() < 1e-8:
            continue
        null_arrays.append(arr)
        valid_cols.append(col)
    if not null_arrays:
        return []

    null_matrix = np.column_stack(null_arrays).astype(np.float32)
    tgt_np = train_tgt.select(target_cols).to_numpy().astype(np.float32)

    null_mean = null_matrix.mean(axis=0, keepdims=True)
    null_std = null_matrix.std(axis=0, keepdims=True)
    null_std[null_std < 1e-8] = 1.0
    null_matrix = (null_matrix - null_mean) / null_std

    tgt_mean = tgt_np.mean(axis=0, keepdims=True)
    tgt_std = tgt_np.std(axis=0, keepdims=True)
    tgt_std[tgt_std < 1e-8] = 1.0
    tgt_np = (tgt_np - tgt_mean) / tgt_std

    corr_matrix = (null_matrix.T @ tgt_np) / null_matrix.shape[0]
    max_abs_corr = np.abs(corr_matrix).max(axis=1)
    return [col for col, c in zip(valid_cols, max_abs_corr) if c > threshold]


def add_individual_null_indicators(df, selected_cols):
    """Add is_null indicators for selected numerical features."""
    exprs = [pl.col(c).is_null().cast(pl.Int8).alias(f"is_null_{c}") for c in selected_cols]
    return df.with_columns(exprs)


def add_frequency_encoding(train_df, test_df, cat_cols):
    """Frequency-encode categoricals using train+test combined counts."""
    total = train_df.height + test_df.height
    freq_cols = []
    for col in cat_cols:
        fname = f"freq_{col}"
        combined = pl.concat([train_df.select(col), test_df.select(col)])
        freq_map = (
            combined[col].value_counts()
            .with_columns((pl.col("count") / total).alias(fname))
            .select([col, fname])
        )
        train_df = train_df.join(freq_map, on=col, how="left")
        test_df = test_df.join(freq_map, on=col, how="left")
        freq_cols.append(fname)
    return train_df, test_df, freq_cols


def add_cat_interaction_freqs(train_df, test_df, interactions):
    """Add frequency-encoded category interaction features."""
    total = train_df.height + test_df.height
    freq_cols = []
    for c1, c2 in interactions:
        fname = f"freq_{c1}_x_{c2}"
        combo = pl.col(c1).cast(pl.Int64) * 1_000_000 + pl.col(c2).cast(pl.Int64)
        combined = pl.concat([
            train_df.select(combo.alias("_combo")),
            test_df.select(combo.alias("_combo")),
        ])
        freq_map = (
            combined["_combo"].value_counts()
            .with_columns((pl.col("count") / total).alias(fname))
            .select(["_combo", fname])
        )
        train_df = (train_df.with_columns(combo.alias("_combo"))
                    .join(freq_map, on="_combo", how="left").drop("_combo"))
        test_df = (test_df.with_columns(combo.alias("_combo"))
                   .join(freq_map, on="_combo", how="left").drop("_combo"))
        freq_cols.append(fname)
    return train_df, test_df, freq_cols


def add_numerical_diffs(df, diff_pairs):
    """Add difference features for numerical pairs."""
    exprs = []
    for a, b in diff_pairs:
        if a in df.columns and b in df.columns:
            a_id = a.replace("num_feature_", "")
            b_id = b.replace("num_feature_", "")
            exprs.append((pl.col(a) - pl.col(b)).alias(f"diff_{a_id}_minus_{b_id}"))
    if exprs:
        df = df.with_columns(exprs)
    return df


def add_ratio_features(df, ratio_pairs):
    """Add ratio features a/b."""
    exprs = []
    for a, b in ratio_pairs:
        name = f"ratio_{a.replace('num_feature_', '')}_{b.replace('num_feature_', '')}"
        exprs.append(
            pl.when(pl.col(b).abs() > 1e-8)
            .then(pl.col(a) / pl.col(b))
            .otherwise(None)
            .alias(name)
        )
    return df.with_columns(exprs)


def add_null_count(df, cols, name, batch_size=300):
    """Add per-row null count for given columns."""
    parts = []
    for i in range(0, len(cols), batch_size):
        batch = cols[i:i + batch_size]
        parts.append(pl.sum_horizontal([pl.col(c).is_null().cast(pl.UInt16) for c in batch]))
    total = parts[0]
    for p in parts[1:]:
        total = total + p
    return df.with_columns(total.alias(name))


def add_row_mean(df, num_cols):
    """Add per-row mean of numerical features."""
    return df.with_columns(
        pl.mean_horizontal([pl.col(c) for c in num_cols]).alias("row_mean_main")
    )


def add_row_stats(df, num_cols, prefix):
    """Add row-level std and skew for numerical columns."""
    arr = df.select(num_cols).to_numpy()
    row_mean = np.nanmean(arr, axis=1, keepdims=True)
    diff = arr - row_mean
    m2 = np.nanmean(diff ** 2, axis=1)
    m3 = np.nanmean(diff ** 3, axis=1)
    row_std = np.sqrt(np.maximum(m2, 0)).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_skew = np.where(m2 > 1e-16, m3 / (m2 ** 1.5 + 1e-16), 0).astype(np.float32)
    del arr, diff, m2, m3; gc.collect()
    return df.with_columns([
        pl.Series(f"{prefix}_row_std", row_std),
        pl.Series(f"{prefix}_row_skew", row_skew),
    ])


def remove_duplicate_cats(train_df, test_df, dup_cols):
    """Remove duplicate categorical features."""
    existing = [c for c in dup_cols if c in train_df.columns]
    if existing:
        train_df = train_df.drop(existing)
        test_df = test_df.drop(existing)
        print(f"  Removed {len(existing)} duplicate cats")
    return train_df, test_df


# ── Main pipeline ─────────────────────────────────────────────────


def main():
    t0 = time.time()
    print("=" * 60)
    print("Step 1: Feature Engineering")
    print("=" * 60)

    # 1. Load raw data
    print("\n[1/8] Loading raw data...")
    train_main = pl.read_parquet(f"{DATA_DIR}train_main_features.parquet")
    test_main = pl.read_parquet(f"{DATA_DIR}test_main_features.parquet")
    train_extra_raw = pl.read_parquet(f"{DATA_DIR}train_extra_features.parquet")
    test_extra_raw = pl.read_parquet(f"{DATA_DIR}test_extra_features.parquet")
    train_tgt = pl.read_parquet(f"{DATA_DIR}train_target.parquet")
    print(f"  Train: main {train_main.shape}, extra {train_extra_raw.shape}")

    cat_cols = sorted([c for c in train_main.columns if c.startswith("cat_feature")])
    num_cols_main = sorted([c for c in train_main.columns if c.startswith("num_feature")])
    target_cols = [c for c in train_tgt.columns if c.startswith("target_")]

    # 2. Null pattern PCA (on raw extra, before filtering)
    print("\n[2/8] Null Pattern PCA...")
    train_null_pca, test_null_pca = null_pattern_pca(
        train_extra_raw, test_extra_raw, NULL_PCA_COMPONENTS
    )

    # 3. Filter + dedup extra
    print("\n[3/8] Filter + dedup extra features...")
    train_extra, test_extra = filter_extra_features(train_extra_raw, test_extra_raw)
    del train_extra_raw, test_extra_raw; gc.collect()
    train_extra, test_extra = deduplicate_extra_features(train_extra, test_extra)

    # 4. Remove duplicate cats + cast
    print("\n[4/8] Prepare categoricals...")
    train_main, test_main = remove_duplicate_cats(train_main, test_main, DUPLICATE_CATS)
    cat_cols = sorted([c for c in train_main.columns if c.startswith("cat_feature")])
    train_main = train_main.with_columns(pl.col(cat_cols).cast(pl.Int32))
    test_main = test_main.with_columns(pl.col(cat_cols).cast(pl.Int32))

    # 5. Feature engineering
    print("\n[5/8] Engineering features...")

    # Null indicators (groups)
    train_main = add_null_indicators(train_main, NULL_GROUPS_MAIN)
    test_main = add_null_indicators(test_main, NULL_GROUPS_MAIN)
    train_extra = add_null_indicators(train_extra, NULL_GROUPS_EXTRA)
    test_extra = add_null_indicators(test_extra, NULL_GROUPS_EXTRA)
    print(f"  + {len(NULL_GROUPS_MAIN) + len(NULL_GROUPS_EXTRA)} null group indicators")

    # Null counts
    train_main = add_null_count(train_main, num_cols_main, "null_count_main")
    test_main = add_null_count(test_main, num_cols_main, "null_count_main")
    extra_num_cols = [c for c in train_extra.columns if c.startswith("num_feature")]
    train_extra = add_null_count(train_extra, extra_num_cols, "null_count_extra")
    test_extra = add_null_count(test_extra, extra_num_cols, "null_count_extra")
    print(f"  + 2 null count features")

    # Frequency encoding
    train_main, test_main, freq_cols = add_frequency_encoding(train_main, test_main, cat_cols)
    print(f"  + {len(freq_cols)} frequency features")

    # Category interaction frequencies
    train_main, test_main, interact_cols = add_cat_interaction_freqs(
        train_main, test_main, CAT_INTERACTIONS
    )
    print(f"  + {len(interact_cols)} interaction features")

    # Numerical diffs
    train_main = add_numerical_diffs(train_main, NUM_DIFFS)
    test_main = add_numerical_diffs(test_main, NUM_DIFFS)
    diff_cols = [c for c in train_main.columns if c.startswith("diff_")]
    print(f"  + {len(diff_cols)} diff features")

    # Ratio features
    train_main = add_ratio_features(train_main, RATIO_FEATURES)
    test_main = add_ratio_features(test_main, RATIO_FEATURES)
    print(f"  + {len(RATIO_FEATURES)} ratio features")

    # Row mean
    train_main = add_row_mean(train_main, num_cols_main)
    test_main = add_row_mean(test_main, num_cols_main)
    print(f"  + 1 row_mean feature")

    # Individual null indicators (correlation-selected)
    null_ind_cols = select_null_indicator_features(
        train_main, num_cols_main, train_tgt, target_cols, INDIVIDUAL_NULL_THRESHOLD
    )
    train_main = add_individual_null_indicators(train_main, null_ind_cols)
    test_main = add_individual_null_indicators(test_main, null_ind_cols)
    print(f"  + {len(null_ind_cols)} individual null indicators (|corr| > {INDIVIDUAL_NULL_THRESHOLD})")

    # Row stats
    train_main = add_row_stats(train_main, num_cols_main, "main")
    test_main = add_row_stats(test_main, num_cols_main, "main")
    extra_num_for_stats = [c for c in train_extra.columns if c.startswith("num_feature")]
    train_extra = add_row_stats(train_extra, extra_num_for_stats, "extra")
    test_extra = add_row_stats(test_extra, extra_num_for_stats, "extra")
    print(f"  + 4 row stats (main + extra std/skew)")

    # 6. Join main + extra + null PCA
    print("\n[6/8] Joining features...")
    train_feat = train_main.join(train_extra, on="customer_id")
    test_feat = test_main.join(test_extra, on="customer_id")
    del train_main, train_extra, test_main, test_extra; gc.collect()

    null_pca_cols = [f"null_pca_{i}" for i in range(NULL_PCA_COMPONENTS)]
    train_feat = train_feat.hstack(pl.DataFrame(train_null_pca, schema=null_pca_cols))
    test_feat = test_feat.hstack(pl.DataFrame(test_null_pca, schema=null_pca_cols))
    del train_null_pca, test_null_pca

    feature_cols = [c for c in train_feat.columns if c != "customer_id"]
    cat_feature_names = [c for c in feature_cols if c.startswith("cat_feature")]
    num_feature_names = [c for c in feature_cols if c not in cat_feature_names]

    print(f"  Total: {len(feature_cols)} features "
          f"({len(cat_feature_names)} cat, {len(num_feature_names)} num)")

    # 7. Save targets
    print("\n[7/8] Saving targets...")
    FEATURES_DIR.mkdir(exist_ok=True)
    targets = train_tgt.select(["customer_id"] + target_cols)
    targets.write_parquet(FEATURES_DIR / "targets.parquet")

    # 8. Save features + metadata
    print("\n[8/8] Saving features...")
    train_feat.write_parquet(FEATURES_DIR / "train_features.parquet")
    test_feat.write_parquet(FEATURES_DIR / "test_features.parquet")

    meta = {
        "cat_cols": cat_feature_names,
        "num_cols": num_feature_names,
        "feature_names": feature_cols,
        "target_cols": target_cols,
    }
    with open(FEATURES_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed / 60:.1f} min.")
    print(f"  Saved to {FEATURES_DIR}/")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Train: {train_feat.shape}, Test: {test_feat.shape}")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
