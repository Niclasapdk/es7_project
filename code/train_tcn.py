#!/usr/bin/env python3
# train_tcn.py
import os, math, argparse, json, time, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Utilities
# ---------------------------

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_device(batch, device):
    return tuple(t.to(device, non_blocking=True) for t in batch)

def numel(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def human_time(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h: return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

# ---------------------------
# Data
# ---------------------------

class IQWindows(Dataset):
    """
    Dataset expecting sequences [N, T, 2]. If T>W, chunks with hop H.
    If T<W:
      - with drop_last=True, yields 0 windows (strict)
      - with drop_last=False, yields ONE zero-padded window starting at 0
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, W: int, H: int, drop_last=True):
        assert X.ndim == 3 and Y.ndim == 3 and X.shape == Y.shape, "X/Y must be [N, T, 2]"
        self.W = int(W); self.H = int(H)
        self.drop_last = bool(drop_last)

        N, T, C = X.shape
        assert C == 2, "last dim must be I/Q=2"
        self.src = X.astype(np.float32, copy=False)
        self.tgt = Y.astype(np.float32, copy=False)

        self.index = []
        if T == W:
            self.index = [(i, 0) for i in range(N)]
        elif T > W:
            for i in range(N):
                starts = list(range(0, T - W + 1, self.H))
                if not starts and not self.drop_last:
                    starts = [0]
                for s in starts:
                    self.index.append((i, s))
        else:  # T < W
            if not self.drop_last:
                # one padded window per sequence
                self.index = [(i, 0) for i in range(N)]
            else:
                self.index = []  # strict: produce none

        self.T = T

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, s = self.index[k]
        x = self.src[i, s:s+self.W, :]
        y = self.tgt[i, s:s+self.W, :]
        if x.shape[0] < self.W:
            pad = self.W - x.shape[0]
            x = np.pad(x, ((0, pad), (0, 0)), mode="constant")
            y = np.pad(y, ((0, pad), (0, 0)), mode="constant")
        return torch.from_numpy(x), torch.from_numpy(y)

def _find_npz_keys(npz):
    """
    Locate train/val/test arrays inside an .npz with flexible naming.
    Returns dict: {"train": (Xtr, Ytr), "val": (Xva, Yva), "test": (Xte, Yte)}
    Accepted patterns include:
      - X_train/Y_train, X_val/Y_val, X_test/Y_test
      - train_X/train_Y, val_X/val_Y, test_X/test_Y
      - X_tr/Y_tr, X_va/Y_va, X_te/Y_te
      - Xtr/Ytr, Xva/Yva, (optional Xte/Yte)
    If no explicit test is present, val is split 50/50 into val/test.
    """
    keys = set(npz.files)

    # Common complete triplets
    triplets = [
        ("X_train","Y_train","X_val","Y_val","X_test","Y_test"),
        ("train_X","train_Y","val_X","val_Y","test_X","test_Y"),
        ("X_tr","Y_tr","X_va","Y_va","X_te","Y_te"),
        ("Xtr","Ytr","Xva","Yva","Xte","Yte"),
    ]
    for (xtr,ytr,xva,yva,xte,yte) in triplets:
        if {xtr,ytr,xva,yva,xte,yte}.issubset(keys):
            return {
                "train": (npz[xtr], npz[ytr]),
                "val":   (npz[xva], npz[yva]),
                "test":  (npz[xte], npz[yte]),
            }

    # Handle the "no test set" variants explicitly
    pairs_with_no_test = [
        ("Xtr","Ytr","Xva","Yva"),
        ("X_tr","Y_tr","X_va","Y_va"),
        ("X_train","Y_train","X_val","Y_val"),
        ("train_X","train_Y","val_X","val_Y"),
    ]
    for (xtr,ytr,xva,yva) in pairs_with_no_test:
        if {xtr,ytr,xva,yva}.issubset(keys):
            Xtr, Ytr = npz[xtr], npz[ytr]
            Xva, Yva = npz[xva], npz[yva]
            # If test exists with another naming, use it
            for xte,yte in [("Xte","Yte"), ("X_te","Y_te"), ("X_test","Y_test"), ("test_X","test_Y")]:
                if {xte,yte}.issubset(keys):
                    return {
                        "train": (Xtr, Ytr),
                        "val":   (Xva, Yva),
                        "test":  (npz[xte], npz[yte]),
                    }
            # Otherwise split val into val/test (50/50)
            Nva = Xva.shape[0]
            mid = max(1, Nva // 2)
            return {
                "train": (Xtr, Ytr),
                "val":   (Xva[:mid], Yva[:mid]),
                "test":  (Xva[mid:], Yva[mid:]),
            }

    # Fallback: try generic X/Y (we'll split 80/10/10)
    if "X" in keys and "Y" in keys:
        X = npz["X"]; Y = npz["Y"]
        N = X.shape[0]
        ntr = int(0.8 * N); nva = int(0.1 * N)
        return {
            "train": (X[:ntr], Y[:ntr]),
            "val":   (X[ntr:ntr+nva], Y[ntr:ntr+nva]),
            "test":  (X[ntr+nva:], Y[ntr+nva:]),
        }

    raise ValueError(f"Could not infer dataset keys from npz keys={sorted(keys)}")


def _canonicalize_bt2(a: np.ndarray) -> np.ndarray:
    """
    Convert various IQ layouts to [N, T, 2] float32:
      - complex: [N, T] complex64/128  -> stack real/imag -> [N, T, 2]
      - channel-first: [N, 2, T]       -> transpose -> [N, T, 2]
      - single sequence: [T, 2]        -> add batch -> [1, T, 2]
      - real only: [N, T]              -> add zero Q -> [N, T, 2]
    """
    a = np.asarray(a)
    # Complex → split to I/Q
    if np.iscomplexobj(a):
        a = np.stack([a.real, a.imag], axis=-1)

    # Now ensure last dim is 2 (I/Q)
    if a.ndim == 3 and a.shape[-1] == 2:
        pass  # [N, T, 2] OK
    elif a.ndim == 3 and a.shape[1] == 2 and a.shape[-1] != 2:
        # [N, 2, T] → [N, T, 2]
        a = np.transpose(a, (0, 2, 1))
    elif a.ndim == 2 and a.shape[-1] == 2:
        # [T, 2] → [1, T, 2]
        a = a[None, ...]
    elif a.ndim == 2:
        # [N, T] real → add zero Q
        a = np.stack([a, np.zeros_like(a)], axis=-1)
    elif a.ndim == 1:
        # [T] → [1, T, 2] with zero Q
        a = a[None, :, None]
        a = np.concatenate([a, np.zeros_like(a)], axis=-1)
    else:
        raise ValueError(f"Unsupported array shape {a.shape} for IQ data")

    # Final sanity
    if a.shape[-1] != 2:
        raise ValueError(f"Expected last dim=2 after canonicalize, got {a.shape}")
    return a.astype(np.float32, copy=False)


def _align_xy_bt2(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Make X and Y have the same [N, T, 2] by trimming to common N,T.
    """
    if X.ndim != 3 or Y.ndim != 3 or X.shape[-1] != 2 or Y.shape[-1] != 2:
        raise ValueError(f"Expected [N,T,2], got X {X.shape}, Y {Y.shape}")
    N = min(X.shape[0], Y.shape[0])
    T = min(X.shape[1], Y.shape[1])
    if X.shape[0] != Y.shape[0] or X.shape[1] != Y.shape[1]:
        # Trim with a friendly note in stdout (optional)
        print(f"[align] Trimming X {X.shape} / Y {Y.shape} to common (N={N}, T={T})")
    return X[:N, :T, :], Y[:N, :T, :]


def load_npz_dataset(path: Path, W: int, H: int, batch: int, workers: int = 4, drop_last=True):
    npz = np.load(str(path))
    parts = _find_npz_keys(npz)

    def canon_pair(X, Y):
        Xc = _canonicalize_bt2(X)
        Yc = _canonicalize_bt2(Y)
        Xc, Yc = _align_xy_bt2(Xc, Yc)
        return Xc, Yc

    Xtr, Ytr = canon_pair(parts["train"][0], parts["train"][1])
    Xva, Yva = canon_pair(parts["val"][0],   parts["val"][1])
    Xte, Yte = canon_pair(parts["test"][0],  parts["test"][1])

    # Diagnostics
    print(f"[data] shapes: train {Xtr.shape}, val {Xva.shape}, test {Xte.shape} | W={W} H={H}")

    # Preferred: strict windows for training, tolerant for eval
    ds_tr = IQWindows(Xtr, Ytr, W, H, drop_last=True)
    ds_va = IQWindows(Xva, Yva, W, H, drop_last=False)
    ds_te = IQWindows(Xte, Yte, W, H, drop_last=False)

    # If strict train produced 0 windows (e.g., T<W), fall back to tolerant mode
    if len(ds_tr) == 0:
        print("[data] WARNING: train produced 0 windows with drop_last=True; retrying with drop_last=False (padded).")
        ds_tr = IQWindows(Xtr, Ytr, W, H, drop_last=False)

    # Final sanity check
    print(f"[data] windows: train {len(ds_tr)}, val {len(ds_va)}, test {len(ds_te)}")

    if len(ds_tr) == 0:
        raise RuntimeError("Training dataset has 0 windows. Consider lowering --W or using --H<=W and ensure T>=1.")

    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True,  num_workers=workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True, drop_last=False)
    dl_te = DataLoader(ds_te, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True, drop_last=False)
    return ds_tr, ds_va, ds_te, dl_tr, dl_va, dl_te

# ---------------------------
# Model: Causal TCN
# ---------------------------

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        pad = (kernel_size - 1) * dilation  # left padding only (causal)
        super().__init__(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
    def forward(self, x):
        out = super().forward(x)
        # Remove the extra right drift to keep causality/alignment
        cut = (self.kernel_size[0] - 1) * self.dilation[0]
        if cut > 0:
            out = out[..., :-cut]
        return out

class TCNBlock(nn.Module):
    def __init__(self, ch, k, dilation, dropout=0.0):
        super().__init__()
        self.conv1 = CausalConv1d(ch, ch, k, dilation=dilation)
        self.conv2 = CausalConv1d(ch, ch, k, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(ch)
        self.norm2 = nn.BatchNorm1d(ch)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.relu(h, inplace=True)
        h = self.dropout(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.relu(h, inplace=True)
        h = self.dropout(h)

        return x + h  # residual

class TCN(nn.Module):
    def __init__(self, in_ch=2, ch=128, out_ch=2, k=5, blocks=10, dropout=0.0):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, ch, 1)
        layers = []
        for b in range(blocks):
            d = 2 ** b
            layers.append(TCNBlock(ch, k, dilation=d, dropout=dropout))
        self.tcn = nn.Sequential(*layers)
        self.out = nn.Conv1d(ch, out_ch, 1)

    def forward(self, x_bt2):
        # x: [B, W, 2] -> [B, 2, W]
        x = x_bt2.transpose(1, 2)
        h = self.inp(x)
        h = self.tcn(h)
        y = self.out(h)
        # -> [B, W, 2]
        y_bt2 = y.transpose(1, 2)
        return y_bt2, None

def receptive_field(kernel: int, blocks: int) -> int:
    # RF = 1 + (k-1) * sum_{i=0}^{B-1} 2^i = 1 + (k-1)*(2^B - 1)
    return 1 + (kernel - 1) * (2 ** blocks - 1)

# ---------------------------
# Losses & Metrics (stable)
# ---------------------------

def pairwise_diff(x):
    # x: [B, T, C] -> differences along T (length T-1, pad back to T)
    d = x[:, 1:, :] - x[:, :-1, :]
    d = F.pad(d, (0, 0, 1, 0))
    return d

class TimeDomainLoss(nn.Module):
    """
    L = L1(ŷ_H, y_H) + alpha * L1(Δŷ_H, Δy_H)
    """
    def __init__(self, alpha=0.05):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()

    def forward(self, yhat, y, H):
        if H <= 0 or H > y.shape[1]:
            H = y.shape[1]
        yhat_h = yhat[:, -H:, :]
        y_h = y[:, -H:, :]
        base = self.l1(yhat_h, y_h)
        d1 = pairwise_diff(yhat_h)
        d2 = pairwise_diff(y_h)
        delta = self.l1(d1, d2)
        return base + self.alpha * delta

# ---- Metric helpers (compat + stability) ----
def snr_db(signal: torch.Tensor, noise: torch.Tensor, eps: float = 1e-12) -> float:
    s = signal.float()
    n = noise.float()
    p_sig = s.pow(2).sum()
    p_noi = n.pow(2).sum().clamp_min(eps)
    return 10.0 * torch.log10((p_sig + eps) / p_noi)

@torch.no_grad()
def evm_stats(yhat: torch.Tensor, y: torch.Tensor,
              ref_thresh: float = 1e-8, eps: float = 1e-12):
    """
    EVM using global sums + mask (batchwise power gate).
    Returns: (evm_pct, evm_db, coverage in [0,1])
    """
    yh = yhat.float(); yt = y.float()
    B = yt.shape[0]
    yh2 = yh.reshape(B, -1)
    yt2 = yt.reshape(B, -1)
    ref_pow_b = (yt2.pow(2)).sum(dim=1)  # [B]
    mask = ref_pow_b > ref_thresh
    if not mask.any():
        return float("nan"), float("nan"), 0.0
    num = ((yh2 - yt2).pow(2)).sum(dim=1)[mask].sum()
    den = ref_pow_b[mask].sum().clamp_min(eps)
    evm_lin = math.sqrt(float((num + eps) / (den + eps)))
    evm_pct = 100.0 * evm_lin
    evm_db  = 20.0 * math.log10(max(eps, evm_lin))
    cov = float(mask.float().mean().item())
    return evm_pct, evm_db, cov

@torch.no_grad()
def estimate_lag(yt: torch.Tensor, yh: torch.Tensor, search: int) -> int:
    """
    Estimate best alignment lag between yt (target) and yh (prediction).
    Returns lag >= 0 meaning: yh shifted left by 'lag' aligns with yt.
    We search lags in [0..search].
    """
    if search <= 0:
        return 0
    # Use first sample, I channel; normalize to zero-mean to avoid DC effects
    a = yt[0, :, 0] - yt[0, :, 0].mean()
    b = yh[0, :, 0] - yh[0, :, 0].mean()
    # Brute-force small search (fast for H<=4096)
    best_lag = 0
    best_score = -1e9
    T = a.numel()
    for lag in range(0, search + 1):
        t = T - lag
        if t <= 8:
            break
        s = (a[:t] * b[lag:lag+t]).sum().item()
        if s > best_score:
            best_score = s
            best_lag = lag
    return best_lag

@torch.no_grad()
def evaluate(model, loss_fn, dl, device, H, ref_thresh=1e-8, eps=1e-12, align_search=None):
    """
    Stable evaluation with:
      - global power sums (no mean of ratios)
      - masking of near-zero reference batches
      - small non-negative lag search to compensate causal delay
    """
    model.eval()
    tot_loss = 0.0
    n_loss = 0
    err_pow = torch.zeros((), dtype=torch.float64, device=device)
    ref_pow = torch.zeros((), dtype=torch.float64, device=device)
    sig_pow = torch.zeros((), dtype=torch.float64, device=device)
    nse_in_pow = torch.zeros((), dtype=torch.float64, device=device)
    nse_out_pow = torch.zeros((), dtype=torch.float64, device=device)
    kept_batches = 0
    total_batches = 0

    for x, y in dl:
        total_batches += 1
        x, y = to_device((x, y), device)
        with torch.cuda.amp.autocast(enabled=False):
            yhat, _ = model(x)
        loss = loss_fn(yhat, y, H)
        bs = x.size(0)
        tot_loss += loss.item() * bs
        n_loss += bs

        Huse = min(H, y.shape[1])
        yt = y[:, -Huse:, :]
        yh = yhat[:, -Huse:, :]
        xx = x[:, -Huse:, :]

        # Estimate a small non-negative lag (best shift left for yh)
        if align_search is not None and align_search > 0:
            lag = estimate_lag(yt, yh, align_search)
            if lag > 0:
                yt = yt[:, :Huse-lag, :]
                yh = yh[:, lag:, :]
                xx = xx[:, :Huse-lag, :]
                if yt.shape[1] < 16:
                    continue  # too short after shift

        # Batch mask by reference power
        ref_b = (yt.float().pow(2).sum(dim=(-1, -2))).mean()
        if ref_b.item() <= ref_thresh:
            continue

        err_pow += (yh.float() - yt.float()).pow(2).sum(dtype=torch.float64)
        ref_pow += (yt.float()).pow(2).sum(dtype=torch.float64)
        sig_pow += (yt.float()).pow(2).sum(dtype=torch.float64)
        nse_in_pow += (xx.float() - yt.float()).pow(2).sum(dtype=torch.float64)
        nse_out_pow += (yh.float() - yt.float()).pow(2).sum(dtype=torch.float64)
        kept_batches += 1

    mean_loss = tot_loss / max(1, n_loss)

    if ref_pow.item() <= eps or kept_batches == 0:
        evm_pct = float("nan"); evm_db = float("nan")
        snr_in = float("nan"); snr_out = float("nan")
    else:
        evm_lin = math.sqrt(float((err_pow + eps) / (ref_pow + eps)))
        evm_pct = 100.0 * evm_lin
        evm_db  = 20.0 * math.log10(max(eps, evm_lin))
        snr_in  = 10.0 * math.log10(float((sig_pow + eps) / (nse_in_pow + eps)))
        snr_out = 10.0 * math.log10(float((sig_pow + eps) / (nse_out_pow + eps)))

    return {
        "loss": mean_loss,
        "evm_pct": evm_pct,
        "evm_db": evm_db,
        "snr_in": snr_in,
        "snr_out": snr_out,
        "n": n_loss,
        "ref_cov": 100.0 * kept_batches / max(1, total_batches),
    }

# ---------------------------
# Training
# ---------------------------

def make_optimizer(model, lr, wd):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

def make_scheduler(opt, sched, epochs, steps_per_epoch, warmup_epochs, base_lr):
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(0, int(warmup_epochs * steps_per_epoch))

    if sched == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * t))
        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    elif sched == "none":
        return torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0)
    else:
        raise ValueError(f"Unknown --sched {sched}")

def train_one_epoch(model, loss_fn, dl, opt, scaler, device, H, accum_steps, max_steps=0):
    model.train()
    t0 = time.time()
    running = 0.0
    steps = 0
    samples = 0
    for it, (x, y) in enumerate(dl):
        x, y = to_device((x, y), device)
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            yhat, _ = model(x)
            loss = loss_fn(yhat, y, H)
            loss = loss / accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (it + 1) % accum_steps == 0:
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            steps += 1

        running += loss.item() * x.size(0) * accum_steps
        samples += x.size(0)

        if max_steps and steps >= max_steps:
            break

    dt = human_time(time.time() - t0)
    return running / max(1, samples), steps, dt

# ---------------------------
# CLI / Main
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Causal TCN denoiser training")
    p.add_argument("--data", type=str, required=True, help="Path to .npz dataset")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--W", type=int, default=2048, help="window length")
    p.add_argument("--H", type=int, default=512,  help="hop/emit length")
    p.add_argument("--width", type=int, default=192, help="TCN channels")
    p.add_argument("--blocks", type=int, default=10, help="TCN residual blocks")
    p.add_argument("--kernel", type=int, default=5, help="TCN kernel size")
    p.add_argument("--lr", type=float, default=4e-3)
    p.add_argument("--wd", type=float, default=5e-3)
    p.add_argument("--sched", type=str, default="cosine", choices=["cosine","none"])
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=0, help="max optimizer steps per epoch (after grad accumulation)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out", type=str, default="tcn_denoiser.pt")
    p.add_argument("--alpha", type=float, default=0.05, help="delta-L1 weight")
    p.add_argument("--align_search", type=int, default=-1, help="override lag search window; default=-1 uses RF/8 capped to 256")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | seed {args.seed}")

    # Data
    data_path = Path(args.data)
    ds_tr, ds_va, ds_te, dl_tr, dl_va, dl_te = load_npz_dataset(
        data_path, W=args.W, H=args.H, batch=args.batch, workers=args.workers, drop_last=True)

    # Model
    model = TCN(in_ch=2, ch=args.width, out_ch=2, k=args.kernel, blocks=args.blocks, dropout=0.05).to(device)
    nparams = numel(model)
    print(f"Params: {nparams/1e6:.2f}M | W {args.W} H {args.H} | width {args.width} blocks {args.blocks} k {args.kernel}")

    # Loss / Opt / Sched
    loss_fn = TimeDomainLoss(alpha=args.alpha)
    opt = make_optimizer(model, lr=args.lr, wd=args.wd)

    steps_per_epoch = max(1, len(dl_tr) // max(1, args.accum_steps))
    scheduler = make_scheduler(opt, args.sched, args.epochs, steps_per_epoch, args.warmup_epochs, args.lr)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Align search default: RF/8 limited to 256, non-negative (causal)
    RF = receptive_field(args.kernel, args.blocks)
    if args.align_search is None or args.align_search < 0:
        align_search = min(256, max(0, RF // 8))
    else:
        align_search = max(0, args.align_search)

    best_val = float("inf")
    best_ckpt = args.out

    # Warmup print
    print(f"Receptive field (samples): {RF} | align_search: {align_search} | steps/epoch: {steps_per_epoch}")

    # Training loop
    global_step = 0
    t_train0 = time.time()
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_steps, tr_dt = train_one_epoch(
            model, loss_fn, dl_tr, opt, scaler, device, args.H, args.accum_steps, args.max_steps)
        global_step += tr_steps

        # Scheduler step per-epoch (cosine + warmup uses step count)
        if isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
            scheduler.step()

        # Eval
        with torch.no_grad():
            va = evaluate(model, loss_fn, dl_va, device, args.H, align_search=align_search)
        lr_now = opt.param_groups[0]["lr"]
        print(f"Epoch {ep:03d} | train {tr_loss:.6f} | val {va['loss']:.6f} "
              f"| EVM% {va['evm_pct']:.2f} ({va['evm_db']:.2f} dB) "
              f"| SNR_in {va['snr_in']:.2f} → SNR_out {va['snr_out']:.2f} "
              f"| Δ { (va['snr_out'] - va['snr_in']) if (not math.isnan(va['snr_in']) and not math.isnan(va['snr_out'])) else float('nan'):+.2f} dB "
              f"| cov {va['ref_cov']:.1f}% | lr {lr_now:.2e} | {tr_dt}")

        # Save best by val loss
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save({"model": model.state_dict(),
                        "args": vars(args),
                        "epoch": ep,
                        "val": va}, best_ckpt)
            print(f"  ↳ saved -> {best_ckpt}")

    # Final test
    with torch.no_grad():
        te = evaluate(model, loss_fn, dl_te, device, args.H, align_search=align_search)
    print("\n=== TEST === "
          f" loss {te['loss']:.6f} | EVM% {te['evm_pct']:.2f} ({te['evm_db']:.2f} dB) "
          f"| SNR_in {te['snr_in']:.2f} → SNR_out {te['snr_out']:.2f} "
          f"| Δ {(te['snr_out'] - te['snr_in']) if (not math.isnan(te['snr_in']) and not math.isnan(te['snr_out'])) else float('nan'):+.2f} dB "
          f"| cov {te['ref_cov']:.1f}%")

if __name__ == "__main__":
    main()
