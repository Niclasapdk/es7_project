#!/usr/bin/env python3
# train_tcn.py — Causal TCN denoiser (residual head) with robust EVM/SNR metrics
# Fixes & safeguards:
#  • Auto-fit receptive field to W (reduces blocks if RF > W)
#  • Residual prediction (yhat = x - n̂) for identity-safe denoising
#  • GroupNorm by default (stable with small per-GPU batches)
#  • Identity baseline metrics check before training (catches X/Y mispairing)
#  • Optional EMA for evaluation
#  • Scheduler stepped per optimizer step, optional grad clipping
#  • Same masked metrics/loss & DDP support

import os, math, argparse, time, random
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler

# =========================
# Utilities
# =========================

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

def is_ddp() -> bool:
    return dist.is_available() and dist.is_initialized()

def ddp_rank() -> int:
    return dist.get_rank() if is_ddp() else 0

def ddp_world_size() -> int:
    return dist.get_world_size() if is_ddp() else 1

def only_rank0_print(*args, **kwargs):
    if ddp_rank() == 0: print(*args, **kwargs)

def all_reduce_sum_(t: torch.Tensor) -> torch.Tensor:
    if is_ddp(): dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t

# =========================
# Data
# =========================

class IQWindows(Dataset):
    """Sliding windows [W] over sequences [N, T, 2]. Assumes W <= T (clamped at load)."""
    def __init__(self, X: np.ndarray, Y: np.ndarray, W: int, H: int):
        assert X.ndim == 3 and Y.ndim == 3 and X.shape == Y.shape, "X/Y must be [N, T, 2]"
        self.W = int(W); self.H = int(H)
        N, T, C = X.shape; assert C == 2
        self.src = X.astype(np.float32, copy=False)
        self.tgt = Y.astype(np.float32, copy=False)
        self.T = T
        self.index = []
        if T == W:
            self.index = [(i, 0) for i in range(N)]
        else:
            for i in range(N):
                for s in range(0, T - W + 1, self.H):
                    self.index.append((i, s))

    def __len__(self): return len(self.index)

    def __getitem__(self, k):
        i, s = self.index[k]
        x = self.src[i, s:s+self.W, :]
        y = self.tgt[i, s:s+self.W, :]
        return torch.from_numpy(x), torch.from_numpy(y)

def _find_npz_keys(npz):
    keys = set(npz.files)
    triplets = [
        ("X_train","Y_train","X_val","Y_val","X_test","Y_test"),
        ("train_X","train_Y","val_X","val_Y","test_X","test_Y"),
        ("X_tr","Y_tr","X_va","Y_va","X_te","Y_te"),
        ("Xtr","Ytr","Xva","Yva","Xte","Yte"),
    ]
    for (xtr,ytr,xva,yva,xte,yte) in triplets:
        if {xtr,ytr,xva,yva,xte,yte}.issubset(keys):
            return {"train": (npz[xtr], npz[ytr]), "val": (npz[xva], npz[yva]), "test": (npz[xte], npz[yte])}
    pairs_no_test = [
        ("Xtr","Ytr","Xva","Yva"), ("X_tr","Y_tr","X_va","Y_va"),
        ("X_train","Y_train","X_val","Y_val"), ("train_X","train_Y","val_X","val_Y"),
    ]
    for (xtr,ytr,xva,yva) in pairs_no_test:
        if {xtr,ytr,xva,yva}.issubset(keys):
            Xtr, Ytr = npz[xtr], npz[ytr]
            Xva, Yva = npz[xva], npz[yva]
            mid = max(1, Xva.shape[0] // 2)
            return {"train": (Xtr, Ytr), "val": (Xva[:mid], Yva[:mid]), "test": (Xva[mid:], Yva[mid:])}
    if "X" in keys and "Y" in keys:
        X, Y = npz["X"], npz["Y"]; N = X.shape[0]
        ntr, nva = int(0.8*N), int(0.1*N)
        return {"train": (X[:ntr],Y[:ntr]), "val": (X[ntr:ntr+nva],Y[ntr:ntr+nva]), "test": (X[ntr+nva:],Y[ntr+nva:])}
    raise ValueError(f"Could not infer dataset keys from npz keys={sorted(keys)}")

def _canonicalize_bt2(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if np.iscomplexobj(a): a = np.stack([a.real, a.imag], axis=-1)
    if a.ndim == 3 and a.shape[-1] == 2: pass
    elif a.ndim == 3 and a.shape[1] == 2: a = np.transpose(a, (0,2,1))
    elif a.ndim == 2 and a.shape[-1] == 2: a = a[None, ...]
    elif a.ndim == 2: a = np.stack([a, np.zeros_like(a)], axis=-1)
    elif a.ndim == 1:
        a = a[None, :, None]; a = np.concatenate([a, np.zeros_like(a)], axis=-1)
    else: raise ValueError(f"Unsupported array shape {a.shape}")
    return a.astype(np.float32, copy=False)

def _align_xy_bt2(X: np.ndarray, Y: np.ndarray):
    N = min(X.shape[0], Y.shape[0]); T = min(X.shape[1], Y.shape[1])
    return X[:N, :T, :], Y[:N, :T, :]

def load_npz_dataset(path: Path, W: int, H: int, batch: int, workers: int = 4, prefetch: int = 4):
    npz = np.load(str(path))
    parts = _find_npz_keys(npz)

    def canon_pair(X, Y):
        Xc = _canonicalize_bt2(X); Yc = _canonicalize_bt2(Y)
        return _align_xy_bt2(Xc, Yc)

    Xtr, Ytr = canon_pair(*parts["train"])  # [N,T,2]
    Xva, Yva = canon_pair(*parts["val"])    # [N,T,2]
    Xte, Yte = canon_pair(*parts["test"])   # [N,T,2]

    T_min = min(Xtr.shape[1], Xva.shape[1], Xte.shape[1])
    W_eff = min(W, T_min)
    if W_eff < W:
        only_rank0_print(f"[data] Requested W={W} exceeds dataset min T={T_min}. Clamping W -> {W_eff}.")
        W = W_eff

    only_rank0_print(f"[data] shapes: train {Xtr.shape}, val {Xva.shape}, test {Xte.shape} | W={W} H={H}")

    ds_tr = IQWindows(Xtr, Ytr, W, H)
    ds_va = IQWindows(Xva, Yva, W, H)
    ds_te = IQWindows(Xte, Yte, W, H)

    if is_ddp():
        smp_tr = DistributedSampler(ds_tr, shuffle=True)
        smp_va = DistributedSampler(ds_va, shuffle=False)
        smp_te = DistributedSampler(ds_te, shuffle=False)
    else:
        smp_tr = smp_va = smp_te = None

    pin = True
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=(smp_tr is None), sampler=smp_tr,
                       num_workers=workers, pin_memory=pin, drop_last=True,
                       persistent_workers=(workers>0), prefetch_factor=prefetch)
    dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False, sampler=smp_va,
                       num_workers=max(1, workers//2), pin_memory=pin, drop_last=False,
                       persistent_workers=(workers>0), prefetch_factor=max(2, prefetch//2))
    dl_te = DataLoader(ds_te, batch_size=batch, shuffle=False, sampler=smp_te,
                       num_workers=max(1, workers//2), pin_memory=pin, drop_last=False,
                       persistent_workers=(workers>0), prefetch_factor=max(2, prefetch//2))

    only_rank0_print(f"[data] windows: train {len(ds_tr)}, val {len(ds_va)}, test {len(ds_te)}")
    return ds_tr, ds_va, ds_te, dl_tr, dl_va, dl_te, W

# =========================
# Model: Causal TCN
# =========================

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        pad = (kernel_size - 1) * dilation
        super().__init__(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
    def forward(self, x):
        out = super().forward(x)
        cut = (self.kernel_size[0] - 1) * self.dilation[0]
        if cut > 0: out = out[..., :-cut]
        return out

class TCNBlock(nn.Module):
    def __init__(self, ch, k, dilation, dropout=0.0, use_gn=True, gn_groups=8):
        super().__init__()
        self.conv1 = CausalConv1d(ch, ch, k, dilation=dilation)
        self.conv2 = CausalConv1d(ch, ch, k, dilation=dilation)
        if use_gn:
            self.norm1 = nn.GroupNorm(gn_groups, ch)
            self.norm2 = nn.GroupNorm(gn_groups, ch)
        else:
            self.norm1 = nn.BatchNorm1d(ch)
            self.norm2 = nn.BatchNorm1d(ch)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        h = self.conv1(x); h = self.norm1(h); h = F.relu(h, inplace=True); h = self.dropout(h)
        h = self.conv2(h); h = self.norm2(h); h = F.relu(h, inplace=True); h = self.dropout(h)
        return x + h

class TCN(nn.Module):
    def __init__(self, in_ch=2, ch=128, out_ch=2, k=5, blocks=6, dropout=0.05,
                 use_ckpt=False, use_gn=True, predict_noise=True):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, ch, 1)
        self.blocks = nn.ModuleList([TCNBlock(ch, k, dilation=2**b, dropout=dropout, use_gn=use_gn) for b in range(blocks)])
        self.out = nn.Conv1d(ch, out_ch, 1)
        self.use_ckpt = use_ckpt
        self.predict_noise = predict_noise
    def forward(self, x_bt2):
        from torch.utils.checkpoint import checkpoint
        x = x_bt2.transpose(1, 2)   # [B,T,2] -> [B,2,T]
        h = self.inp(x)
        for blk in self.blocks:
            h = checkpoint(blk, h, use_reentrant=False) if (self.use_ckpt and self.training) else blk(h)
        head = self.out(h)                  # [B,2,T]
        if self.predict_noise:
            yhat = x - head                 # residual removal
        else:
            yhat = head                     # direct prediction
        return yhat.transpose(1, 2), None   # -> [B,T,2]

def receptive_field(kernel: int, blocks: int) -> int:
    return 1 + (kernel - 1) * (2 ** blocks - 1)

def fit_blocks_to_W(kernel: int, blocks: int, W: int) -> Tuple[int,int]:
    """Reduce blocks until RF <= W."""
    b = blocks
    while b > 1 and receptive_field(kernel, b) > W:
        b -= 1
    return b, receptive_field(kernel, b)

# =========================
# Losses & Metrics
# =========================

def pairwise_diff(x):
    d = x[:, 1:, :] - x[:, :-1, :]
    return F.pad(d, (0, 0, 1, 0))

class MaskedTimeLoss(nn.Module):
    """Masked L1 + ΔL1 with optional EVM-normalized and spectral terms."""
    def __init__(self, alpha=0.03, beta_evm_norm=0.01, spec_weight=0.005, eps=1e-12):
        super().__init__()
        self.alpha = alpha
        self.beta = beta_evm_norm
        self.spec_w = spec_weight
        self.eps = eps
    def forward(self, yhat, y, H):
        mask = (y.pow(2).sum(dim=-1, keepdim=True) > self.eps).float()  # [B,T,1]
        m = mask
        denom = m.sum() * y.size(-1) + 1e-12

        base = ((yhat - y).abs() * m).sum() / denom
        d1 = ((pairwise_diff(yhat) - pairwise_diff(y)).abs() * m).sum() / denom

        num = ((yhat - y) * m).pow(2).sum(dim=(-1, -2))
        den = (y * m).pow(2).sum(dim=(-1, -2)).clamp_min(self.eps)
        evm_norm = (num / den).mean()

        yhm = yhat * m; ym = y * m
        yh_c = torch.complex(yhm[..., 0], yhm[..., 1])
        y_c  = torch.complex(ym[..., 0],  ym[..., 1])
        YH = torch.fft.fft(yh_c, dim=1); Y = torch.fft.fft(y_c, dim=1)
        spec = (YH.abs() - Y.abs()).abs().mean()

        return base + self.alpha*d1 + self.beta*evm_norm + self.spec_w*spec

@torch.no_grad()
def evaluate(model, loss_fn, dl, device, H, eps=1e-12, use_ema=None):
    model_eval = model
    if use_ema is not None:
        use_ema.apply(model_eval)
    model_eval.eval()

    tot_loss = torch.zeros((), dtype=torch.float64, device=device)
    n_loss   = torch.zeros((), dtype=torch.float64, device=device)
    err_pow  = torch.zeros((), dtype=torch.float64, device=device)
    ref_pow  = torch.zeros((), dtype=torch.float64, device=device)
    sig_pow  = torch.zeros((), dtype=torch.float64, device=device)
    nse_in   = torch.zeros((), dtype=torch.float64, device=device)
    nse_out  = torch.zeros((), dtype=torch.float64, device=device)
    kept_s   = torch.zeros((), dtype=torch.float64, device=device)
    total_s  = torch.zeros((), dtype=torch.float64, device=device)

    for x, y in dl:
        x, y = to_device((x, y), device)
        with autocast("cuda", enabled=False):
            yhat, _ = model_eval(x)
        loss = loss_fn(yhat, y, H)
        bs = x.size(0)
        tot_loss += loss.detach() * bs
        n_loss += bs

        mask = (y.pow(2).sum(dim=-1, keepdim=True) > eps).float()
        valid = (mask.sum(dim=(1, 2)) > 0)
        if valid.any():
            m = mask[valid]
            yt = y[valid] * m
            yh = yhat[valid] * m
            xx = x[valid] * m
            err_pow  += (yh - yt).pow(2).sum(dtype=torch.float64)
            ref_pow  += (yt).pow(2).sum(dtype=torch.float64)
            sig_pow  += (yt).pow(2).sum(dtype=torch.float64)
            nse_in   += (xx - yt).pow(2).sum(dtype=torch.float64)
            nse_out  += (yh - yt).pow(2).sum(dtype=torch.float64)
            kept_s   += valid.sum()
        total_s += bs

    for t in (tot_loss, n_loss, err_pow, ref_pow, sig_pow, nse_in, nse_out, kept_s, total_s):
        all_reduce_sum_(t)

    mean_loss = (tot_loss / torch.clamp_min(n_loss, 1)).item()

    if use_ema is not None:
        use_ema.restore(model_eval)

    if ref_pow.item() <= eps or kept_s.item() == 0:
        evm_pct = float("nan"); evm_db = float("nan"); snr_in = float("nan"); snr_out = float("nan"); cov = 0.0
    else:
        evm_lin = math.sqrt(float((err_pow) / (ref_pow + 1e-12)))
        evm_pct = 100.0 * evm_lin
        evm_db  = 20.0 * math.log10(max(1e-12, evm_lin))
        snr_in  = 10.0 * math.log10(float((sig_pow + 1e-12) / (nse_in + 1e-12)))
        snr_out = 10.0 * math.log10(float((sig_pow + 1e-12) / (nse_out + 1e-12)))
        cov = 100.0 * float(kept_s.item() / max(1.0, total_s.item()))

    return {"loss": mean_loss, "evm_pct": evm_pct, "evm_db": evm_db, "snr_in": snr_in, "snr_out": snr_out, "n": int(n_loss.item()), "ref_cov": cov}

# =========================
# EMA helper
# =========================

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        base = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
        self.shadow = {n: p.detach().clone() for n, p in base.named_parameters() if p.requires_grad}
        self.base_ref = base
        self.back = {}
    @torch.no_grad()
    def update(self, model: nn.Module):
        base = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
        for n, p in base.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
    @torch.no_grad()
    def apply(self, model: nn.Module):
        base = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
        self.back = {}
        for n, p in base.named_parameters():
            if p.requires_grad:
                self.back[n] = p.data.clone()
                p.data.copy_(self.shadow[n])
    @torch.no_grad()
    def restore(self, model: nn.Module):
        if not self.back: return
        base = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
        for n, p in base.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.back[n])
        self.back = {}

# =========================
# Optim / Train
# =========================

def make_optimizer(model, lr, wd):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

def make_scheduler(opt, sched, epochs, steps_per_epoch, warmup_epochs):
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

def train_one_epoch(model, loss_fn, dl, opt, scaler, device, H, accum_steps,
                    scheduler=None, ema: Optional[EMA]=None, max_steps=0, clip: float=0.0):
    model.train()
    t0 = time.time()
    running = 0.0; steps = 0; samples = 0

    if is_ddp() and isinstance(dl.sampler, DistributedSampler):
        pass  # caller sets epoch externally

    for it, (x, y) in enumerate(dl):
        x, y = to_device((x, y), device)
        with autocast("cuda", enabled=(scaler is not None)):
            yhat, _ = model(x)
            loss = loss_fn(yhat, y, H) / accum_steps
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (it + 1) % accum_steps == 0:
            if clip and clip > 0:
                if scaler is not None:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            if scaler is not None:
                scaler.step(opt); scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            steps += 1
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update(model)

        running += loss.item() * x.size(0) * accum_steps
        samples += x.size(0)
        if max_steps and steps >= max_steps:
            break

    dt = human_time(time.time() - t0)
    tr_loss_t = torch.tensor(running / max(1, samples), device=device, dtype=torch.float64)
    all_reduce_sum_(tr_loss_t)
    tr_loss = (tr_loss_t / ddp_world_size()).item()
    return tr_loss, steps, dt

# =========================
# Distributed init
# =========================

def init_distributed(backend: str = "nccl", port: Optional[int] = None):
    if dist.is_initialized(): return 0
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    if world > 1:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "127.0.0.1")
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(port or 29500)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, rank=rank, world_size=world)
    return local_rank

# =========================
# CLI / Main
# =========================

def parse_args():
    p = argparse.ArgumentParser(description="Causal TCN denoiser (residual) with DDP + EMA + robust EVM/SNR")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=256, help="per-GPU batch size")
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--use_gn", action="store_true", help="use GroupNorm instead of BatchNorm (overrides default)")
    p.add_argument("--bn", action="store_true", help="force BatchNorm instead of GroupNorm")
    p.add_argument("--predict_direct", action="store_true", help="predict clean directly (default is residual noise)")
    p.add_argument("--clip", type=float, default=1.0, help="grad clip norm (0 to disable)")
    p.add_argument("--W", type=int, default=2048)
    p.add_argument("--H", type=int, default=512)
    p.add_argument("--width", type=int, default=192)
    p.add_argument("--blocks", type=int, default=10)
    p.add_argument("--kernel", type=int, default=7)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=5e-3)
    p.add_argument("--sched", type=str, default="cosine", choices=["cosine","none"])
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--prefetch", type=int, default=4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out", type=str, default="tcn_denoiser.pt")
    p.add_argument("--backend", type=str, default="nccl")
    # EMA + loss extras
    p.add_argument("--ema", action="store_true")
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--eval_use_ema", action="store_true", help="swap EMA weights during evaluation")
    p.add_argument("--beta_evm_norm", type=float, default=0.01)
    p.add_argument("--spec_weight", type=float, default=0.005)
    # Resume
    p.add_argument("--resume", type=str, default="", help="path to checkpoint to load model weights from")
    p.add_argument("--resume_all", action="store_true", help="also load optimizer/scheduler state")
    return p.parse_args()

def identity_metrics_check(dl, device, H, eps=1e-12):
    """Compute metrics with yhat=x to sanity-check dataset/metrics."""
    class _Id(nn.Module):
        def forward(self, x): return x, None
    loss_fn = MaskedTimeLoss()
    return evaluate(_Id().to(device), loss_fn, dl, device, H, eps=eps, use_ema=None)

def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    local_rank = init_distributed(backend=args.backend)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available(): torch.cuda.set_device(device)

    if ddp_rank() == 0:
        only_rank0_print(f"Device: {device} | seed {args.seed} | world_size {ddp_world_size()} | rank {ddp_rank()}")

    data_path = Path(args.data).expanduser()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    ds_tr, ds_va, ds_te, dl_tr, dl_va, dl_te, W_eff = load_npz_dataset(
        data_path, W=args.W, H=args.H, batch=args.batch, workers=args.workers, prefetch=args.prefetch)

    # Auto-fit blocks to W (avoid RF >> W which wrecks training)  ← fix for your logs
    blocks_fit, RF = fit_blocks_to_W(args.kernel, args.blocks, W_eff)
    if blocks_fit != args.blocks:
        only_rank0_print(f"[model] RF too large for W={W_eff}: requested blocks={args.blocks} → using blocks={blocks_fit}")
    predict_noise = (not args.predict_direct)
    use_gn = True
    if args.bn: use_gn = False
    if args.use_gn: use_gn = True

    model = TCN(in_ch=2, ch=args.width, out_ch=2, k=args.kernel, blocks=blocks_fit,
                dropout=args.dropout, use_ckpt=args.grad_ckpt, use_gn=use_gn, predict_noise=predict_noise).to(device)
    if args.compile and hasattr(torch, "compile"): model = torch.compile(model)

    nparams = numel(model)
    only_rank0_print(f"Params: {nparams/1e6:.2f}M | W {W_eff} H {args.H} | width {args.width} blocks {blocks_fit} k {args.kernel}")
    only_rank0_print(f"Receptive field (samples): {RF} | steps/epoch: {max(1, len(dl_tr)//max(1,args.accum_steps))}")

    loss_fn = MaskedTimeLoss(alpha=0.03, beta_evm_norm=args.beta_evm_norm, spec_weight=args.spec_weight)
    opt = make_optimizer(model, lr=args.lr, weight_decay=args.wd)

    steps_per_epoch = max(1, len(dl_tr) // max(1, args.accum_steps))
    scheduler = make_scheduler(opt, args.sched, args.epochs, steps_per_epoch, args.warmup_epochs)
    scaler = GradScaler("cuda", enabled=args.amp)

    # DDP wrap
    if ddp_world_size() > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device.index] if device.type=="cuda" else None,
                                                    output_device=device.index if device.type=="cuda" else None,
                                                    find_unused_parameters=False)

    # EMA
    ema = EMA(model, decay=args.ema_decay) if args.ema else None

    # Resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        (model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model).load_state_dict(ckpt["model"], strict=True)
        if args.resume_all and "opt" in ckpt and "sched" in ckpt:
            try:
                opt.load_state_dict(ckpt["opt"])
                scheduler.load_state_dict(ckpt["sched"])
            except Exception as e:
                only_rank0_print(f"[resume_all] could not load opt/sched: {e}")
        only_rank0_print(f"Resumed from {args.resume} (epoch {ckpt.get('epoch','?')})")

    # --- Identity baseline sanity check (catches mispaired X/Y immediately)
    id_va = identity_metrics_check(dl_va, device, args.H)
    only_rank0_print(f"[baseline yhat=x] EVM% {id_va['evm_pct']:.2f} ({id_va['evm_db']:.2f} dB) | "
                     f"SNR_in {id_va['snr_in']:.2f} → SNR_out {id_va['snr_out']:.2f} | "
                     f"Δ {(id_va['snr_out'] - id_va['snr_in']) if (not math.isnan(id_va['snr_in']) and not math.isnan(id_va['snr_out'])) else float('nan'):+.2f} dB")

    best_val = float("inf"); best_ckpt = args.out

    for ep in range(1, args.epochs + 1):
        if is_ddp() and isinstance(dl_tr.sampler, DistributedSampler): dl_tr.sampler.set_epoch(ep)

        tr_loss, tr_steps, tr_dt = train_one_epoch(model, loss_fn, dl_tr, opt, scaler, device,
                                                   args.H, args.accum_steps, scheduler=scheduler, ema=ema,
                                                   max_steps=args.max_steps, clip=args.clip)

        with torch.no_grad():
            va = evaluate(model, loss_fn, dl_va, device, args.H, use_ema=(ema if args.eval_use_ema else None))

        lr_now = opt.param_groups[0]["lr"]
        if ddp_rank() == 0:
            only_rank0_print(
                f"Epoch {ep:03d} | train {tr_loss:.6f} | val {va['loss']:.6f} "
                f"| EVM% {va['evm_pct']:.2f} ({va['evm_db']:.2f} dB) "
                f"| SNR_in {va['snr_in']:.2f} → SNR_out {va['snr_out']:.2f} "
                f"| Δ {(va['snr_out'] - va['snr_in']) if (not math.isnan(va['snr_in']) and not math.isnan(va['snr_out'])) else float('nan'):+.2f} dB "
                f"| cov {va['ref_cov']:.1f}% | lr {lr_now:.2e} | {tr_dt}"
            )
            if va["loss"] < best_val:
                best_val = va["loss"]
                state = {
                    "model": (model.module.state_dict() if isinstance(model, nn.parallel.DistributedDataParallel) else model.state_dict()),
                    "args": vars(args),
                    "epoch": ep,
                    "val": va,
                    "opt": opt.state_dict(),
                    "sched": scheduler.state_dict() if hasattr(scheduler, 'state_dict') else {},
                }
                torch.save(state, best_ckpt)
                only_rank0_print(f"  ↳ saved -> {best_ckpt}")

    with torch.no_grad():
        te = evaluate(model, loss_fn, dl_te, device, args.H, use_ema=(ema if args.eval_use_ema else None))
    if ddp_rank() == 0:
        print("=== TEST === "
              f" loss {te['loss']:.6f} | EVM% {te['evm_pct']:.2f} ({te['evm_db']:.2f} dB) "
              f"| SNR_in {te['snr_in']:.2f} → SNR_out {te['snr_out']:.2f} "
              f"| Δ {(te['snr_out'] - te['snr_in']) if (not math.isnan(te['snr_in']) and not math.isnan(te['snr_out'])) else float('nan'):+.2f} dB "
              f"| cov {te['ref_cov']:.1f}%")

    if dist.is_initialized(): dist.destroy_process_group()

if __name__ == "__main__":
    main()
