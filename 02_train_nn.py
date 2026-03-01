"""Step 2: Train Neural Network (TabM + PLR + ASL + Mixup + SWA).

Loads features from features/ directory, trains DAE for semi-supervised
bottleneck embeddings, then trains TabM with PLR encoding.

Output: checkpoints_nn/fold_{0-4}.npz (val_preds, test_preds, val_idx)

Runtime: ~2-3 hours on GPU, ~5-8 hours on MPS/CPU.
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset

from utils import SEED, DATA_DIR, N_FOLDS, compute_macro_auc, get_device

DEVICE = get_device()
FEATURES_DIR = Path("features")
CHECKPOINT_DIR = Path("checkpoints_nn")

# ── Hyperparameters ───────────────────────────────────────────────
BATCH_SIZE = 1024
EPOCHS = 50
LR = 5e-4
WEIGHT_DECAY = 1e-3
PATIENCE = 10
GRAD_CLIP = 1.0
HIDDEN_DIM = 384

# DAE
DAE_BOTTLENECK_DIM = 256
DAE_EPOCHS = 20
DAE_LR = 1e-3
DAE_BATCH_SIZE = 4096
DAE_CORRUPTION_RATE = 0.15
DAE_WEIGHT_DECAY = 1e-5
DAE_GRAD_CLIP = 1.0

# TabM + ASL + Mixup + PLR
K_ENSEMBLE = 16
ASL_GAMMA_NEG = 4
ASL_GAMMA_POS = 1
ASL_CLIP = 0.05
MIXUP_ALPHA = 0.2
PLR_N_BINS = 24
TOP_K_SWA = 5


# ══════════════════════════════════════════════════════════════════
#  Denoising Autoencoder
# ══════════════════════════════════════════════════════════════════


class DenoisingAutoencoder(nn.Module):
    """Symmetric DAE: Input → 1024 → 512 → 256 (bottleneck) → 512 → 1024 → Input."""

    def __init__(self, input_dim, bottleneck_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(512, bottleneck_dim), nn.BatchNorm1d(bottleneck_dim), nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(1024, input_dim),
        )
        for module in [self.encoder, self.decoder]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                    nn.init.zeros_(layer.bias)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def apply_swap_noise(clean, non_null_mask, corruption_rate):
    """Apply swap noise: replace values with random other rows in the batch."""
    corrupt_mask = (torch.rand_like(clean) < corruption_rate) & non_null_mask
    perm = torch.randperm(clean.shape[0], device=clean.device)
    return torch.where(corrupt_mask, clean[perm], clean)


def train_dae(all_num_data, null_mask_np, device):
    """Train DAE on train+test numerical data (semi-supervised)."""
    n_samples, input_dim = all_num_data.shape
    print(f"    DAE: {n_samples:,} × {input_dim}, bottleneck={DAE_BOTTLENECK_DIM}")

    model = DenoisingAutoencoder(input_dim, DAE_BOTTLENECK_DIM).to(device)
    use_compile = hasattr(torch, "compile")
    if use_compile:
        backend = "aot_eager" if device.type == "mps" else "inductor"
        model = torch.compile(model, backend=backend)

    optimizer = torch.optim.AdamW(model.parameters(), lr=DAE_LR, weight_decay=DAE_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=DAE_EPOCHS)

    data_tensor = torch.from_numpy(all_num_data).to(device)
    non_null_tensor = torch.from_numpy((1 - null_mask_np).astype(np.float32)).to(device)
    del null_mask_np; gc.collect()

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    n_batches = (n_samples + DAE_BATCH_SIZE - 1) // DAE_BATCH_SIZE

    # Check for saved checkpoint
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    ckpt_final = CHECKPOINT_DIR / "dae_final.pt"
    if ckpt_final.exists():
        print(f"    Loading DAE from {ckpt_final}")
        ckpt = torch.load(ckpt_final, map_location=device, weights_only=True)
        orig_model = model._orig_mod if use_compile else model
        orig_model.load_state_dict(ckpt["model_state_dict"])
        del data_tensor, non_null_tensor, ckpt
        return model

    for epoch in range(DAE_EPOCHS):
        model.train()
        total_loss = 0.0
        perm_indices = torch.randperm(n_samples, device=device)
        for b in range(n_batches):
            start = b * DAE_BATCH_SIZE
            end = min(start + DAE_BATCH_SIZE, n_samples)
            idx = perm_indices[start:end]
            clean = data_tensor[idx]
            non_null = non_null_tensor[idx]

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                corrupted = apply_swap_noise(clean, non_null > 0.5, DAE_CORRUPTION_RATE)
                reconstructed = model(corrupted)
                diff_sq = (reconstructed.float() - clean.float()) ** 2 * non_null
                loss = diff_sq.sum() / non_null.sum().clamp(min=1)

            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), DAE_GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), DAE_GRAD_CLIP)
                optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"      Epoch {epoch+1:2d}/{DAE_EPOCHS}  loss={total_loss/n_batches:.6f}",
                  flush=True)

    orig_model = model._orig_mod if use_compile else model
    torch.save({"epoch": DAE_EPOCHS, "model_state_dict": orig_model.state_dict(),
                "loss": total_loss / n_batches}, ckpt_final)
    del data_tensor, non_null_tensor
    return model


@torch.no_grad()
def extract_dae_embeddings(model, all_num_data, device, batch_size=4096):
    """Extract bottleneck embeddings from trained DAE."""
    model.eval()
    embeddings = []
    for start in range(0, all_num_data.shape[0], batch_size):
        end = min(start + batch_size, all_num_data.shape[0])
        batch = torch.from_numpy(all_num_data[start:end]).to(device)
        embeddings.append(model.encode(batch).cpu().numpy())
    return np.vstack(embeddings)


# ══════════════════════════════════════════════════════════════════
#  TabM Model
# ══════════════════════════════════════════════════════════════════


class LinearBE(nn.Module):
    """Batch Ensemble Linear: shared W + per-member rank-1 adapters."""
    def __init__(self, in_dim, out_dim, k):
        super().__init__()
        self.k = k
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.r = nn.Parameter(torch.ones(k, in_dim))
        self.s = nn.Parameter(torch.ones(k, out_dim))
        self.b = nn.Parameter(torch.zeros(k, out_dim))

    def forward(self, x):
        x = x * self.r.unsqueeze(0)
        B, k, D = x.shape
        x = self.linear(x.reshape(B * k, D)).reshape(B, k, -1)
        return x * self.s.unsqueeze(0) + self.b.unsqueeze(0)


class ResidualBlockBE(nn.Module):
    """Batch Ensemble residual block with zero-init."""
    def __init__(self, dim, k, dropout=0.3):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.lin1 = LinearBE(dim, dim, k)
        self.bn2 = nn.BatchNorm1d(dim)
        self.lin2 = LinearBE(dim, dim, k)
        self.drop = nn.Dropout(dropout)
        self.act = nn.SiLU()
        nn.init.zeros_(self.lin2.linear.weight)

    def forward(self, x):
        B, k, D = x.shape
        h = self.bn1(x.reshape(B * k, D)).reshape(B, k, D)
        h = self.act(h)
        h = self.lin1(h)
        B, k, D = h.shape
        h = self.bn2(h.reshape(B * k, D)).reshape(B, k, D)
        h = self.act(h)
        h = self.drop(h)
        h = self.lin2(h)
        return x + h


class AsymmetricLoss(nn.Module):
    """ASL: down-weights easy negatives via probability shifting."""
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        pos_term = (1 - p).pow(self.gamma_pos) * torch.log(p.clamp(min=self.eps))
        p_m = (p - self.clip).clamp(min=self.eps) if self.clip > 0 else p
        neg_term = p_m.pow(self.gamma_neg) * torch.log((1 - p_m).clamp(min=self.eps))
        return (-(targets * pos_term) - (1 - targets) * neg_term).mean()


class PiecewiseLinearEncoding(nn.Module):
    """PLR: learnable piecewise-linear function per feature (d_embedding=1)."""
    def __init__(self, n_features, n_bins=24):
        super().__init__()
        self.n_features = n_features
        self.n_bins = n_bins
        self.register_buffer('edges', torch.zeros(n_features, n_bins + 1))
        self.weight = nn.Parameter(torch.empty(n_features, n_bins))
        self.bias = nn.Parameter(torch.zeros(n_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def set_bins(self, X_train_np):
        quantiles = np.linspace(0, 1, self.n_bins + 1)
        for i in range(self.n_features):
            col = X_train_np[:, i]
            col_valid = col[~np.isnan(col)] if np.any(np.isnan(col)) else col
            if len(col_valid) < 2:
                edges = np.linspace(-3.0, 3.0, self.n_bins + 1)
            else:
                edges = np.unique(np.quantile(col_valid, quantiles))
                if len(edges) < self.n_bins + 1:
                    edges = np.linspace(edges[0], edges[-1], self.n_bins + 1)
                else:
                    edges = edges[:self.n_bins + 1]
            self.edges[i] = torch.from_numpy(edges.astype(np.float32))

    def forward(self, x):
        left = self.edges[:, :-1]
        right = self.edges[:, 1:]
        width = (right - left).clamp(min=1e-8)
        ple = ((x.unsqueeze(2) - left) / width).clamp(0, 1)
        return (ple * self.weight).sum(dim=2) + self.bias


class TabularNet(nn.Module):
    """TabM: Batch Ensemble MLP with PLR + categorical embeddings."""
    def __init__(self, cat_cardinalities, n_numerical, n_targets, k=K_ENSEMBLE):
        super().__init__()
        self.k = k
        self.n_targets = n_targets
        self.plr = PiecewiseLinearEncoding(n_numerical, PLR_N_BINS)

        self.embeddings = nn.ModuleList()
        total_embed_dim = 0
        for n_cat in cat_cardinalities:
            dim = min(50, max(2, (n_cat + 1) // 2))
            self.embeddings.append(nn.Embedding(n_cat + 2, dim))
            total_embed_dim += dim

        input_dim = total_embed_dim + n_numerical
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM), nn.BatchNorm1d(HIDDEN_DIM), nn.SiLU(),
        )
        self.res_blocks = nn.ModuleList([
            ResidualBlockBE(HIDDEN_DIM, k, dropout=0.4),
            ResidualBlockBE(HIDDEN_DIM, k, dropout=0.4),
            ResidualBlockBE(HIDDEN_DIM, k, dropout=0.3),
        ])
        head_dim = HIDDEN_DIM // 2
        self.head_bn1 = nn.BatchNorm1d(HIDDEN_DIM)
        self.head_drop1 = nn.Dropout(0.2)
        self.head_lin1 = LinearBE(HIDDEN_DIM, head_dim, k)
        self.head_bn2 = nn.BatchNorm1d(head_dim)
        self.head_drop2 = nn.Dropout(0.1)
        self.head_lin2 = LinearBE(head_dim, n_targets, k)
        self.act = nn.SiLU()

        for layer in self.input_proj:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)
        nn.init.kaiming_normal_(self.head_lin1.linear.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.head_lin2.linear.weight, nonlinearity="relu")

    def _embed_and_project(self, x_cat, x_num):
        x_num = self.plr(x_num)
        embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        return self.input_proj(torch.cat(embeds + [x_num], dim=1))

    def _head(self, x):
        B, k, D = x.shape
        x = self.head_bn1(x.reshape(B * k, D)).reshape(B, k, D)
        x = self.act(x); x = self.head_drop1(x); x = self.head_lin1(x)
        B, k, D2 = x.shape
        x = self.head_bn2(x.reshape(B * k, D2)).reshape(B, k, D2)
        x = self.act(x); x = self.head_drop2(x)
        return self.head_lin2(x)

    def forward(self, x_cat, x_num):
        x = self._embed_and_project(x_cat, x_num).unsqueeze(1).expand(-1, self.k, -1)
        for block in self.res_blocks:
            x = block(x)
        logits = self._head(x)
        if self.training:
            return logits
        return torch.sigmoid(logits).mean(dim=1)

    def forward_mixup(self, x_cat, x_num, lam, perm, mixup_layer):
        x = self._embed_and_project(x_cat, x_num).unsqueeze(1).expand(-1, self.k, -1)
        if mixup_layer == 0:
            x = lam * x + (1 - lam) * x[perm]
        for i, block in enumerate(self.res_blocks):
            x = block(x)
            if mixup_layer == i + 1:
                x = lam * x + (1 - lam) * x[perm]
        return self._head(x)


# ══════════════════════════════════════════════════════════════════
#  Data helpers
# ══════════════════════════════════════════════════════════════════


def to_tensors(df_feat, cat_cols, num_cols, cat_cardinalities, df_tgt=None, target_cols=None):
    """Convert Polars DataFrame to PyTorch tensors."""
    cat_arr = df_feat.select(cat_cols).to_numpy()
    cat_np = np.zeros_like(cat_arr, dtype=np.int64)
    for i, card in enumerate(cat_cardinalities):
        col = cat_arr[:, i]
        null_mask = np.isnan(col) if np.issubdtype(col.dtype, np.floating) else np.zeros(len(col), dtype=bool)
        values = np.clip(np.nan_to_num(col, nan=0).astype(np.int64), 0, card)
        values[null_mask] = card + 1
        cat_np[:, i] = values

    num_arr = df_feat.select(num_cols).fill_null(0.0).to_numpy().astype(np.float32)
    num_arr = np.nan_to_num(num_arr, nan=0.0, posinf=0.0, neginf=0.0)

    if df_tgt is not None and target_cols is not None:
        tgt_arr = df_tgt.select(target_cols).to_numpy().astype(np.float32)
        return torch.from_numpy(cat_np), torch.from_numpy(num_arr), torch.from_numpy(tgt_arr)
    return torch.from_numpy(cat_np), torch.from_numpy(num_arr)


def quantile_normalize(train_num, val_num, test_num):
    """Quantile-transform to N(0,1). Fit on train only."""
    n = train_num.shape[0]
    qt = QuantileTransformer(n_quantiles=min(1000, n), output_distribution="normal",
                             subsample=min(100_000, n), random_state=SEED)
    tr_np = train_num.numpy()
    qt.fit(tr_np)
    return (
        torch.from_numpy(np.nan_to_num(qt.transform(tr_np), nan=0.0).astype(np.float32)),
        torch.from_numpy(np.nan_to_num(qt.transform(val_num.numpy()), nan=0.0).astype(np.float32)),
        torch.from_numpy(np.nan_to_num(qt.transform(test_num.numpy()), nan=0.0).astype(np.float32)),
    )


# ══════════════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════════════


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n_batches = 0, 0
    k = model.k
    for x_cat, x_num, y in loader:
        x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)
        optimizer.zero_grad()
        mixup_layer = np.random.randint(0, 4)
        lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
        perm = torch.randperm(x_cat.shape[0], device=device)
        logits = model.forward_mixup(x_cat, x_num, lam, perm, mixup_layer)
        y_k = y.unsqueeze(1).expand(-1, k, -1)
        y_mixed = lam * y_k + (1 - lam) * y[perm].unsqueeze(1).expand(-1, k, -1)
        loss = criterion(logits, y_mixed)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for x_cat, x_num, y in loader:
        probs = model(x_cat.to(device), x_num.to(device))
        all_preds.append(probs.cpu())
        all_targets.append(y)
    return torch.cat(all_preds).numpy(), torch.cat(all_targets).numpy()


@torch.no_grad()
def predict_model(model, loader, device):
    model.eval()
    all_preds = []
    for batch in loader:
        probs = model(batch[0].to(device), batch[1].to(device))
        all_preds.append(probs.cpu())
    return torch.cat(all_preds).numpy()


def train_one_fold(fold_idx, tr_cat, tr_num, tr_y, val_cat, val_num, val_y,
                   test_cat, test_num, cat_cardinalities, n_numerical,
                   target_cols, device):
    train_loader = DataLoader(TensorDataset(tr_cat, tr_num, tr_y),
                              batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(val_cat, val_num, val_y),
                            batch_size=BATCH_SIZE * 2, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_cat, test_num),
                             batch_size=BATCH_SIZE * 2, shuffle=False)

    model = TabularNet(cat_cardinalities, n_numerical, len(target_cols)).to(device)
    if fold_idx == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model: {n_params:,} parameters (k={K_ENSEMBLE})")

    model.plr.set_bins(tr_num.numpy())
    criterion = AsymmetricLoss(ASL_GAMMA_NEG, ASL_GAMMA_POS, ASL_CLIP)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_auc, patience_counter = 0, 0
    top_states = []

    for epoch in range(EPOCHS):
        t1 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_preds, val_targets = evaluate_model(model, val_loader, device)
        macro_auc, _ = compute_macro_auc(val_targets, val_preds, target_cols)
        scheduler.step(macro_auc)

        improved = ""
        if macro_auc > best_auc:
            best_auc = macro_auc
            patience_counter = 0
            improved = " *"
        else:
            patience_counter += 1

        state_copy = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        top_states.append((macro_auc, state_copy))
        top_states.sort(key=lambda x: x[0], reverse=True)
        if len(top_states) > TOP_K_SWA:
            top_states.pop()

        print(f"    Ep {epoch+1:2d}/{EPOCHS}  loss={train_loss:.4f}  "
              f"val_auc={macro_auc:.4f}  lr={optimizer.param_groups[0]['lr']:.1e}  "
              f"{time.time()-t1:.1f}s{improved}", flush=True)

        if patience_counter >= PATIENCE:
            print(f"    Early stop at epoch {epoch+1}")
            break

    # SWA: average top-K checkpoints
    avg_state = {}
    for key in top_states[0][1]:
        avg_state[key] = sum(s[key] for _, s in top_states) / len(top_states)
    model.load_state_dict(avg_state)
    model.to(device)

    val_preds, _ = evaluate_model(model, val_loader, device)
    swa_auc, _ = compute_macro_auc(val_targets, val_preds, target_cols)
    test_preds = predict_model(model, test_loader, device)

    print(f"  Fold {fold_idx+1} best={best_auc:.4f}, SWA={swa_auc:.4f}", flush=True)
    return val_preds, test_preds, swa_auc


# ══════════════════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()
    print("=" * 60)
    print(f"Step 2: Train NN (TabM + PLR + ASL + Mixup) on {DEVICE}")
    print("=" * 60)

    # 1. Load features
    print("\n[1/5] Loading features...")
    with open(FEATURES_DIR / "meta.json") as f:
        meta = json.load(f)
    cat_cols = meta["cat_cols"]
    num_cols_all = meta["num_cols"]
    target_cols = meta["target_cols"]

    train_feat = pl.read_parquet(FEATURES_DIR / "train_features.parquet")
    test_feat = pl.read_parquet(FEATURES_DIR / "test_features.parquet")
    train_tgt = pl.read_parquet(FEATURES_DIR / "targets.parquet")
    print(f"  Features: {len(cat_cols)} cat, {len(num_cols_all)} num")

    # Cat cardinalities
    cat_cardinalities = []
    for col in cat_cols:
        combined = pl.concat([train_feat[col], test_feat[col]])
        cat_cardinalities.append(int(combined.drop_nulls().n_unique()))

    # 2. DAE pretraining
    print(f"\n[2/5] DAE Pretraining ({DAE_EPOCHS} epochs)...", flush=True)
    ENGINEERED_PREFIXES = ("is_null_", "null_count_", "null_pca_", "freq_",
                           "diff_", "ratio_", "interact_", "row_", "main_row_", "extra_row_")
    dae_input_cols = [c for c in num_cols_all if not any(c.startswith(p) for p in ENGINEERED_PREFIXES)]
    print(f"  DAE input: {len(dae_input_cols)} raw features")

    train_num_raw = train_feat.select(dae_input_cols).to_numpy()
    test_num_raw = test_feat.select(dae_input_cols).to_numpy()
    train_null_mask = np.isnan(train_num_raw).astype(np.uint8)
    test_null_mask = np.isnan(test_num_raw).astype(np.uint8)
    all_null_mask = np.vstack([train_null_mask, test_null_mask])
    del train_null_mask, test_null_mask

    all_num_data = np.vstack([
        np.nan_to_num(train_num_raw, nan=0.0).astype(np.float32),
        np.nan_to_num(test_num_raw, nan=0.0).astype(np.float32),
    ])
    del train_num_raw, test_num_raw; gc.collect()

    dae_mean = all_num_data.mean(axis=0)
    dae_std = all_num_data.std(axis=0)
    dae_std[dae_std < 1e-8] = 1.0
    all_num_data = (all_num_data - dae_mean) / dae_std

    dae_model = train_dae(all_num_data, all_null_mask, DEVICE)

    print("  Extracting DAE embeddings...")
    all_embeddings = extract_dae_embeddings(dae_model, all_num_data, DEVICE)
    n_train = train_feat.height
    train_embeddings = all_embeddings[:n_train]
    test_embeddings = all_embeddings[n_train:]
    print(f"  DAE embeddings: {all_embeddings.shape[1]} dims")

    del all_num_data, all_null_mask, dae_model, all_embeddings; gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()

    # Add DAE embeddings
    dae_cols = [f"dae_{i}" for i in range(DAE_BOTTLENECK_DIM)]
    train_feat = train_feat.hstack(pl.DataFrame(train_embeddings, schema=dae_cols))
    test_feat = test_feat.hstack(pl.DataFrame(test_embeddings, schema=dae_cols))
    del train_embeddings, test_embeddings

    all_num_cols = [c for c in train_feat.columns if c != "customer_id" and c not in cat_cols]
    print(f"  Total numerical features (with DAE): {len(all_num_cols)}")

    # 3. Convert test
    print("\n[3/5] Converting test to tensors...")
    test_cat, test_num = to_tensors(test_feat, cat_cols, all_num_cols, cat_cardinalities)

    # 4. K-Fold CV
    print(f"\n[4/5] Training {N_FOLDS}-Fold CV...", flush=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    n_train = train_feat.height
    n_test = test_feat.height
    n_targets = len(target_cols)
    oof_preds = np.zeros((n_train, n_targets))
    test_preds_sum = np.zeros((n_test, n_targets))
    fold_aucs = []

    # Check for fold checkpoints
    resume_fold = 0
    for fi in range(N_FOLDS):
        ckpt_path = CHECKPOINT_DIR / f"fold_{fi}.npz"
        if ckpt_path.exists():
            ckpt = np.load(ckpt_path)
            oof_preds[ckpt["val_idx"]] = ckpt["val_preds"]
            test_preds_sum += ckpt["test_preds"]
            fold_aucs.append(float(ckpt["fold_auc"]))
            print(f"  Loaded fold {fi+1} (AUC={float(ckpt['fold_auc']):.4f})", flush=True)
            resume_fold = fi + 1
        else:
            break

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(np.arange(n_train))):
        if fold_idx < resume_fold:
            continue
        print(f"\n  ── Fold {fold_idx+1}/{N_FOLDS} ──", flush=True)

        tr_feat = train_feat[tr_idx.tolist()]
        val_feat_fold = train_feat[val_idx.tolist()]
        tr_tgt = train_tgt[tr_idx.tolist()]
        val_tgt = train_tgt[val_idx.tolist()]

        tr_cat, tr_num, tr_y = to_tensors(tr_feat, cat_cols, all_num_cols, cat_cardinalities, tr_tgt, target_cols)
        val_cat, val_num, val_y = to_tensors(val_feat_fold, cat_cols, all_num_cols, cat_cardinalities, val_tgt, target_cols)

        tr_num, val_num, test_num_normed = quantile_normalize(tr_num, val_num, test_num.clone())
        del tr_feat, val_feat_fold, tr_tgt, val_tgt; gc.collect()

        val_preds, test_preds, fold_auc = train_one_fold(
            fold_idx, tr_cat, tr_num, tr_y, val_cat, val_num, val_y,
            test_cat, test_num_normed, cat_cardinalities, len(all_num_cols),
            target_cols, DEVICE,
        )

        oof_preds[val_idx] = val_preds
        test_preds_sum += test_preds
        fold_aucs.append(fold_auc)

        np.savez(CHECKPOINT_DIR / f"fold_{fold_idx}.npz",
                 val_idx=val_idx, val_preds=val_preds,
                 test_preds=test_preds, fold_auc=fold_auc)

        del tr_cat, tr_num, tr_y, val_cat, val_num, val_y, test_num_normed; gc.collect()
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    # 5. Evaluation
    print(f"\n[5/5] Evaluation...", flush=True)
    oof_y = train_tgt.select(target_cols).to_numpy().astype(np.float32)
    oof_auc, _ = compute_macro_auc(oof_y, oof_preds, target_cols)
    print(f"  Per-fold AUC: {['%.4f' % a for a in fold_aucs]}")
    print(f"  OOF Macro ROC-AUC: {oof_auc:.4f}")

    # Save submission
    test_preds_avg = test_preds_sum / N_FOLDS
    from utils import verify_submission
    sample = pl.read_parquet(f"{DATA_DIR}sample_submit.parquet")
    predict_cols = [c.replace("target_", "predict_") for c in target_cols]
    submit = pl.DataFrame({"customer_id": test_feat["customer_id"]}).hstack(
        pl.DataFrame(test_preds_avg.astype(np.float64), schema=predict_cols)
    )
    verify_submission(submit, sample)
    Path("submissions").mkdir(exist_ok=True)
    submit.write_parquet("submissions/nn.parquet")
    print(f"  Saved: submissions/nn.parquet")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min. OOF={oof_auc:.4f}")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    main()
