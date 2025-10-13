#!/usr/bin/env python3
# train_tcn.py — DDP-ready Causal TCN denoiser with robust EVM/SNR + hardened loader
# (from-scratch friendly; EMA-safe resume; dual-best saving; clean EMA export)

import os, math, argparse, time, random
from pathlib import Path
from typing import Optional, Iterable, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler

# ---------------------------
# Utilities
# ---------------------------

def set_seed(seed: int = 1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def to_device(batch, device): return tuple(t.to(device, non_blocking=True) for t in batch)

def numel(model: nn.Module) -> int: return sum(p.numel() for p in model.parameters())

def is_ddp() -> bool: return dist.is_available() and dist.is_initialized()
def ddp_rank() -> int: return dist.get_rank() if is_ddp() else 0
def ddp_world_size() -> int: return dist.get_world_size() if is_ddp() else 1
def only_rank0_print(*args, **kwargs):
    if ddp_rank() == 0: print(*args, **kwargs, flush=True)

def all_reduce_sum_(t: torch.Tensor) -> torch.Tensor:
    if is_ddp(): dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t

# ---------------------------
# Data
# ---------------------------

class IQWindows(Dataset):
    """Sliding windows [W] over sequences [N, T, 2]."""
    def __init__(self, X: np.ndarray, Y: np.ndarray, W: int, H: int):
        if not (X.ndim == 3 and Y.ndim == 3 and X.shape[-1] == 2 and Y.shape[-1] == 2):
            raise ValueError(f"IQWindows expects [N,T,2] but got X {X.shape}, Y {Y.shape}")
        if X.shape != Y.shape:
            raise ValueError(f"X and Y must match exactly, got {X.shape} vs {Y.shape}")

        self.W = int(W); self.H = int(H)
        N, T, C = X.shape; self.T = T
        self.src = X.astype(np.float32, copy=False)
        self.tgt = Y.astype(np.float32, copy=False)

        self.index = []
        if T == W:
            self.index = [(i, 0) for i in range(N)]
        else:
            for i in range(N):
                for s in range(0, T - W + 1, self.H):
                    self.index.append((i, s))

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        n, s = self.index[i]; e = s + self.W
        x = self.src[n, s:e, :]    # [W,2]
        y = self.tgt[n, s:e, :]
        return torch.from_numpy(x), torch.from_numpy(y)

def _canonicalize_bt2(a: np.ndarray) -> np.ndarray:
    """Convert many RF dataset shapes into [N,T,2] float32."""
    a = np.asarray(a)
    if np.iscomplexobj(a):
        a = np.stack([a.real, a.imag], axis=-1)  # (..., 2)

    if a.ndim == 3 and a.shape[-1] == 2:
        pass
    elif a.ndim == 3 and a.shape[1] == 2:  # [N,2,T] -> [N,T,2]
        a = np.transpose(a, (0, 2, 1))
    elif a.ndim == 2 and a.shape[-1] == 2:  # [T,2] -> [1,T,2]
        a = a[None, ...]
    elif a.ndim == 2:  # [N,T] real -> add imag=0 -> [N,T,2]
        a = np.stack([a, np.zeros_like(a)], axis=-1)
    elif a.ndim == 1:  # [T] -> [1,T,2] with imag=0
        a = np.stack([a, np.zeros_like(a)], axis=-1)[None, ...]
    else:
        raise ValueError(f"Cannot canonicalize array of shape {a.shape} to [N,T,2]")

    if a.shape[-1] != 2:
        if a.shape[-1] == 1:
            a = np.concatenate([a, np.zeros_like(a)], axis=-1)
        else:
            raise ValueError(f"Last dimension must be 2 after canonicalization, got {a.shape}")
    return a.astype(np.float32, copy=False)

def _align_xy(X: np.ndarray, Y: np.ndarray, label: str) -> Tuple[np.ndarray, np.ndarray]:
    """If X and Y differ slightly in N or T, slice to common min dimensions."""
    if X.shape == Y.shape: return X, Y
    Nx, Tx, _ = X.shape; Ny, Ty, _ = Y.shape
    N = min(Nx, Ny); T = min(Tx, Ty)
    only_rank0_print(f"[data:{label}] Mismatch X {X.shape} vs Y {Y.shape} -> slicing to (N={N}, T={T})")
    return X[:N, :T, :], Y[:N, :T, :]

def _find_npz_keys(npz):
    keys = set(npz.files)
    triplets = [
        ("X_train","Y_train","X_val","Y_val","X_test","Y_test"),
        ("train_X","train_Y","val_X","val_Y","test_X","test_Y"),
        ("Xtr","Ytr","Xva","Yva","Xte","Yte"),
        ("X_tr","Y_tr","X_va","Y_va","X_te","Y_te"),
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

def load_npz_dataset(path: str, W: int, H: int, batch: int, workers: int, prefetch: int):
    path = str(path); npz = np.load(path, allow_pickle=False)
    sets = _find_npz_keys(npz)

    Xtr, Ytr = _canonicalize_bt2(sets["train"][0]), _canonicalize_bt2(sets["train"][1])
    Xva, Yva = _canonicalize_bt2(sets["val"][0]),   _canonicalize_bt2(sets["val"][1])
    Xte, Yte = _canonicalize_bt2(sets["test"][0]),  _canonicalize_bt2(sets["test"][1])

    Xtr, Ytr = _align_xy(Xtr, Ytr, "train")
    Xva, Yva = _align_xy(Xva, Yva, "val")
    Xte, Yte = _align_xy(Xte, Yte, "test")

    T_min = min(Xtr.shape[1], Xva.shape[1], Xte.shape[1])
    W_eff = min(int(W), int(T_min))
    if W_eff < W: only_rank0_print(f"[data] Requested W={W} exceeds dataset min T={T_min}. Clamping W -> {W_eff}.")

    ds_tr = IQWindows(Xtr, Ytr, W_eff, H)
    ds_va = IQWindows(Xva, Yva, W_eff, H)
    ds_te = IQWindows(Xte, Yte, W_eff, H)

    only_rank0_print(f"[data] shapes: train {Xtr.shape}, val {Xva.shape}, test {Xte.shape} | W={W_eff} H={H}")

    sampler_tr = DistributedSampler(ds_tr, shuffle=True) if ddp_world_size() > 1 else None
    sampler_va = DistributedSampler(ds_va, shuffle=False) if ddp_world_size() > 1 else None
    sampler_te = DistributedSampler(ds_te, shuffle=False) if ddp_world_size() > 1 else None

    dl_kwargs = dict(pin_memory=True, persistent_workers=(workers > 0), num_workers=workers)
    if workers > 0:
        dl_kwargs["prefetch_factor"] = prefetch

    dl_tr = DataLoader(ds_tr, batch_size=batch, sampler=sampler_tr,
                       shuffle=(sampler_tr is None), drop_last=True, **dl_kwargs)
    dl_va = DataLoader(ds_va, batch_size=batch, sampler=sampler_va,
                       shuffle=False, drop_last=False, **dl_kwargs)
    dl_te = DataLoader(ds_te, batch_size=batch, sampler=sampler_te,
                       shuffle=False, drop_last=False, **dl_kwargs)
    return ds_tr, ds_va, ds_te, dl_tr, dl_va, dl_te, W_eff

# ---------------------------
# Model
# ---------------------------

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        pad = (kernel_size - 1) * dilation
        super().__init__(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
    def forward(self, x):
        out = super().forward(x)
        cut = (self.kernel_size[0] - 1) * self.dilation[0]
        return out[..., :-cut] if cut > 0 else out

class TCNBlock(nn.Module):
    def __init__(self, ch, k, dilation, dropout=0.0, use_bn=False, gn_groups=8):
        super().__init__()
        self.conv1 = CausalConv1d(ch, ch, k, dilation=dilation)
        self.conv2 = CausalConv1d(ch, ch, k, dilation=dilation)
        if use_bn:
            self.norm1 = nn.BatchNorm1d(ch); self.norm2 = nn.BatchNorm1d(ch)
        else:
            self.norm1 = nn.GroupNorm(gn_groups, ch); self.norm2 = nn.GroupNorm(gn_groups, ch)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        h = self.conv1(x); h = self.norm1(h); h = F.relu(h, inplace=True); h = self.dropout(h)
        h = self.conv2(h); h = self.norm2(h); h = F.relu(h, inplace=True); h = self.dropout(h)
        return x + h

class TCN(nn.Module):
    def __init__(self, in_ch=2, ch=192, out_ch=2, k=5, blocks=10, dropout=0.05,
                 use_ckpt=False, use_bn=False, residual=True, separable=False, sep2d=False):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, ch, 1)
        block_cls = TCNBlock
        if separable and sep2d:
            block_cls = SepTCNBlock2D
        self.blocks = nn.ModuleList([block_cls(ch, k, dilation=2**b, dropout=dropout, use_bn=use_bn)
                                     for b in range(blocks)])
        self.out = nn.Conv1d(ch, out_ch, 1)
        self.use_ckpt = use_ckpt
        self.residual = residual


    def forward(self, x_bt2):
        from torch.utils.checkpoint import checkpoint
        x2t = x_bt2.transpose(1, 2)   # [B,2,T]
        h = self.inp(x2t)
        for blk in self.blocks:
            h = checkpoint(blk, h, use_reentrant=False) if (self.use_ckpt and self.training) else blk(h)
        y = self.out(h).transpose(1, 2)  # [B,T,2]
        return (y + x_bt2 if self.residual else y), None

class CausalDWConv2d(torch.nn.Conv2d):
    # x4: [B,C,T,1]; kernel: (k,1); groups=C (depthwise)
    def __init__(self, ch, k, dilation=1, bias=False):
        pad = (k - 1) * dilation
        super().__init__(ch, ch, kernel_size=(k,1),
                         padding=(pad,0), dilation=(dilation,1),
                         groups=ch, bias=bias)
        self._cut = (k - 1) * dilation
    def forward(self, x4):
        y4 = super().forward(x4)                 # [B,C,T+pad,1]
        return y4[:, :, :-self._cut, :] if self._cut > 0 else y4

class SepTCNBlock2D(torch.nn.Module):
    def __init__(self, ch, k, dilation, dropout=0.0, use_bn=False, gn_groups=8):
        super().__init__()
        self.dw1 = CausalDWConv2d(ch, k, dilation=dilation, bias=False)
        self.pw1 = torch.nn.Conv2d(ch, ch, kernel_size=1, bias=True)
        self.dw2 = CausalDWConv2d(ch, k, dilation=dilation, bias=False)
        self.pw2 = torch.nn.Conv2d(ch, ch, kernel_size=1, bias=True)
        if use_bn:
            self.norm1 = torch.nn.BatchNorm2d(ch); self.norm2 = torch.nn.BatchNorm2d(ch)
        else:
            self.norm1 = torch.nn.GroupNorm(gn_groups, ch); self.norm2 = torch.nn.GroupNorm(gn_groups, ch)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):                         # x: [B,C,T]
        z = x.unsqueeze(-1).contiguous(memory_format=torch.channels_last)  # [B,C,T,1] (NHWC-friendly)
        h = self.dw1(z); h = self.pw1(h); h = self.norm1(h); h = F.relu(h, inplace=True); h = self.dropout(h)
        h = self.dw2(h); h = self.pw2(h); h = self.norm2(h); h = F.relu(h, inplace=True); h = self.dropout(h)
        h = h.squeeze(-1)                         # [B,C,T]
        return x + h

# ---------------------------
# Loss & Metrics
# ---------------------------

def pairwise_diff(x):  # [B,T,2]
    d = x[:, 1:, :] - x[:, :-1, :]
    return F.pad(d, (0,0,1,0))

class MaskedTimeLoss(nn.Module):
    def __init__(self, alpha=0.05, beta_evm_norm=0.02, spec_weight=0.01, eps=1e-12):
        super().__init__()
        self.alpha = alpha; self.beta = beta_evm_norm; self.spec_w = spec_weight; self.eps = eps
    def forward(self, yhat, y, H):
        m = (y.pow(2).sum(dim=-1, keepdim=True) > self.eps).float()
        denom = m.sum() * y.size(-1) + 1e-12
        base = ((yhat - y).abs() * m).sum() / denom
        delta = ((pairwise_diff(yhat) - pairwise_diff(y)).abs() * m).sum() / denom
        num = ((yhat - y) * m).pow(2).sum(dim=(-1, -2))
        den = (y * m).pow(2).sum(dim=(-1, -2)).clamp_min(self.eps)
        evm_norm = (num / den).mean()
        yh_c = torch.complex((yhat*m)[...,0], (yhat*m)[...,1]); y_c = torch.complex((y*m)[...,0], (y*m)[...,1])
        spec = (torch.fft.fft(yh_c, dim=1).abs() - torch.fft.fft(y_c, dim=1).abs()).abs().mean()
        return base + self.alpha*delta + self.beta*evm_norm + self.spec_w*spec

@torch.no_grad()
def evaluate(model, loss_fn, dl, device, H, eps=1e-12, use_ema=None):
    model_eval = model
    if use_ema is not None: use_ema.apply(model_eval)
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
        bs = x.size(0); tot_loss += loss.detach() * bs; n_loss += bs
        m = (y.pow(2).sum(dim=-1, keepdim=True) > eps).float()
        valid = (m.sum(dim=(1, 2)) > 0)
        if valid.any():
            m = m[valid]; yt = y[valid]*m; yh = yhat[valid]*m; xx = x[valid]*m
            err_pow += (yh - yt).pow(2).sum(dtype=torch.float64)
            ref_pow += yt.pow(2).sum(dtype=torch.float64)
            sig_pow += yt.pow(2).sum(dtype=torch.float64)
            nse_in  += (xx - yt).pow(2).sum(dtype=torch.float64)
            nse_out += (yh - yt).pow(2).sum(dtype=torch.float64)
            kept_s  += m.sum(dtype=torch.float64)
            total_s += torch.tensor(float(m.numel()), dtype=torch.float64, device=device)

    for t in (tot_loss, n_loss, err_pow, ref_pow, sig_pow, nse_in, nse_out, kept_s, total_s): all_reduce_sum_(t)
    mean_loss = (tot_loss / torch.clamp_min(n_loss, 1)).item()

    if use_ema is not None: use_ema.restore(model_eval)

    if ref_pow.item() <= eps or kept_s.item() == 0:
        evm_pct = float("nan"); evm_db = float("nan"); snr_in = float("nan"); snr_out = float("nan"); cov = 0.0
    else:
        evm_lin = math.sqrt(float((err_pow) / (ref_pow + 1e-12)))
        evm_pct = 100.0 * evm_lin; evm_db = 20.0 * math.log10(max(1e-12, evm_lin))
        snr_in  = 10.0 * math.log10(float((sig_pow + 1e-12) / (nse_in + 1e-12)))
        snr_out = 10.0 * math.log10(float((sig_pow + 1e-12) / (nse_out + 1e-12)))
        cov = 100.0 * float(kept_s.item() / max(1.0, total_s.item()))
    return {"loss": mean_loss, "evm_pct": evm_pct, "evm_db": evm_db, "snr_in": snr_in, "snr_out": snr_out, "n": int(n_loss.item()), "ref_cov": cov}

# ---------------------------
# EMA helper
# ---------------------------

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        base = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
        self.shadow = {n: p.detach().clone() for n, p in base.named_parameters() if p.requires_grad}
        self.back: Dict[str, torch.Tensor] = {}
    @torch.no_grad()
    def update(self, model: nn.Module):
        base = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
        for n, p in base.named_parameters():
            if p.requires_grad: self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
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
            if p.requires_grad: p.data.copy_(self.back[n])
        self.back = {}

# ---------------------------
# Optim / Train
# ---------------------------

def make_optimizer(model: nn.Module, lr: float, wd: Optional[float] = None, **kwargs: Any):
    if wd is None: wd = kwargs.pop("weight_decay", 0.0)
    else: _ = kwargs.pop("weight_decay", None)
    betas = kwargs.pop("betas", (0.9, 0.99)); eps = kwargs.pop("eps", 1e-8)
    no_decay: Iterable[str] = ["bias", "norm", "bn", "gn", "running_mean", "running_var"]
    decay_params, nodecay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if any(nd in n.lower() for nd in no_decay) or p.dim() == 1: nodecay_params.append(p)
        else: decay_params.append(p)
    param_groups = [{"params": decay_params, "weight_decay": wd}, {"params": nodecay_params, "weight_decay": 0.0}]
    return torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)

def make_scheduler(opt, sched, epochs, steps_per_epoch, warmup_epochs):
    total_steps = max(1, epochs * steps_per_epoch); warmup_steps = max(0, int(warmup_epochs * steps_per_epoch))
    if sched == "cosine":
        def lr_lambda(step):
            if step < warmup_steps: return (step + 1) / max(1, warmup_steps)
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * t))
        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    elif sched == "none": return torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0)
    else: raise ValueError(f"Unknown --sched {sched}")

def train_one_epoch(model, loss_fn, dl, opt, scaler, device, H, accum_steps=1, scheduler=None, ema: Optional[EMA]=None, max_steps=0):
    model.train(); start = time.time()
    running = 0.0; steps = 0
    opt.zero_grad(set_to_none=True)
    for it, (x, y) in enumerate(dl):
        x, y = to_device((x, y), device)
        with autocast("cuda", enabled=(scaler is not None)):
            yhat, _ = model(x); loss = loss_fn(yhat, y, H) / max(1, accum_steps)
        if scaler is not None: scaler.scale(loss).backward()
        else: loss.backward()
        if ((it + 1) % accum_steps) == 0:
            if scaler is not None:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            opt.zero_grad(set_to_none=True)
            if scheduler is not None: scheduler.step()
            if ema is not None: ema.update(model)
            steps += 1
            if max_steps > 0 and steps >= max_steps: break
        running += float(loss.detach())
    elapsed = time.time() - start
    return running / max(1, (it + 1)), steps, elapsed

# ---------------------------
# DDP init
# ---------------------------

def init_distributed(backend="nccl", port: Optional[int] = None):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"]); world = int(os.environ.get("SLURM_NTASKS", "1"))
        local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
    else:
        rank, world, local_rank = 0, 1, 0
    if world > 1:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "127.0.0.1")
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(port or 29500)
        if torch.cuda.is_available(): torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, rank=rank, world_size=world)
    return local_rank

# ---------------------------
# CLI / Main
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Causal TCN denoiser training (DDP + EMA + robust EVM/SNR)")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=256, help="per-GPU batch size")
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--separable", action="store_true", help="use depthwise-separable TCN blocks")
    p.add_argument("--sep2d", action="store_true", help="use Conv2d-based depthwise blocks (fast CPU)")
    # Norm & residual
    p.add_argument("--use_bn", action="store_true", help="use BatchNorm instead of GroupNorm")
    p.add_argument("--residual", dest="residual", action=argparse.BooleanOptionalAction, default=True)
    # Windows
    p.add_argument("--W", type=int, default=2048)
    p.add_argument("--H", type=int, default=512)
    # Model
    p.add_argument("--width", type=int, default=192)
    p.add_argument("--blocks", type=int, default=10)
    p.add_argument("--kernel", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.05)
    # Optim
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--wd", type=float, default=5e-3)
    p.add_argument("--sched", type=str, default="cosine", choices=["cosine","none"])
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=0)
    # IO
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--prefetch", type=int, default=4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out", type=str, default="tcn_denoiser.pt")
    p.add_argument("--backend", type=str, default="nccl")
    # EMA + loss extras
    p.add_argument("--ema", action="store_true")
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--eval_use_ema", action="store_true", help="swap EMA weights during evaluation")
    p.add_argument("--beta_evm_norm", type=float, default=0.02)
    p.add_argument("--spec_weight", type=float, default=0.01)
    # Resume
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--resume_all", action="store_true")
    return p.parse_args()

def _save_ckpt(base_model, opt, scheduler, scaler, args, ema_obj: Optional[EMA], path: Path, extra: Dict[str, Any]):
    ck = {
        "model": base_model.state_dict(),
        "opt": opt.state_dict(),
        "sched": scheduler.state_dict(),
        "scaler": (scaler.state_dict() if scaler is not None else {}),
        "args": vars(args),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    if ema_obj is not None:
        ck["ema_state"] = {k: v.detach().clone() for k, v in ema_obj.shadow.items()}
        ck["ema_decay"] = ema_obj.decay
        ck["eval_used_ema"] = bool(args.eval_use_ema)
    ck.update(extra or {})
    torch.save(ck, path)

def _export_ema_model(base_model, ema_obj: EMA, path: Path):
    # Make a pure 'model' file whose weights == EMA shadow (for inference).
    model_sd = base_model.state_dict()
    for n, p in ema_obj.shadow.items():
        if n in model_sd:
            model_sd[n] = p.detach().clone().cpu()
    torch.save({"model": model_sd}, path)

def main():
    args = parse_args()
    set_seed(args.seed); torch.backends.cudnn.benchmark = True
    local_rank = init_distributed(backend=args.backend)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    if ddp_rank() == 0:
        only_rank0_print(f"Device: {device} | seed {args.seed} | world_size {ddp_world_size()} | rank {ddp_rank()}")

    # Data
    data_path = Path(args.data).expanduser().resolve()
    ds_tr, ds_va, ds_te, dl_tr, dl_va, dl_te, W_eff = load_npz_dataset(
        data_path, W=args.W, H=args.H, batch=args.batch, workers=args.workers, prefetch=args.prefetch)

    # Model
    model = TCN(in_ch=2, ch=args.width, out_ch=2, k=args.kernel, blocks=args.blocks,
                dropout=args.dropout, separable=args.separable, sep2d=args.sep2d , use_ckpt=args.grad_ckpt, use_bn=args.use_bn, residual=args.residual).to(device)
    if args.compile and hasattr(torch, "compile"): model = torch.compile(model)
    only_rank0_print(f"Params: {numel(model)/1e6:.2f}M | W {W_eff} H {args.H} | width {args.width} blocks {args.blocks} k {args.kernel}")

    # Loss/optim/sched
    loss_fn = MaskedTimeLoss(alpha=0.05, beta_evm_norm=args.beta_evm_norm, spec_weight=args.spec_weight)
    opt = make_optimizer(model, lr=args.lr, wd=args.wd)
    steps_per_epoch = max(1, len(dl_tr) // max(1, args.accum_steps))
    scheduler = make_scheduler(opt, args.sched, args.epochs, steps_per_epoch, args.warmup_epochs)
    scaler = GradScaler("cuda", enabled=args.amp)

    # DDP wrap
    if ddp_world_size() > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type=="cuda" else None,
            output_device=device.index if device.type=="cuda" else None,
            find_unused_parameters=False
        )

    # EMA
    ema = EMA(model, decay=args.ema_decay) if args.ema else None

    # Resume (safe: no ckpt leakage when not resuming)
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        (model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model).load_state_dict(ckpt["model"], strict=True)
        if ema is not None and "ema_state" in ckpt:
            for n, p in ema.shadow.items():
                if n in ckpt["ema_state"]:
                    p.copy_(ckpt["ema_state"][n].to(p.device))
            only_rank0_print("[resume] Restored EMA shadow from checkpoint.")
        elif ema is not None:
            only_rank0_print("[resume] No ema_state in checkpoint; will create/save it now.")

        if args.resume_all:
            try:
                if "opt" in ckpt: opt.load_state_dict(ckpt["opt"])
                if "sched" in ckpt: scheduler.load_state_dict(ckpt["sched"])
                if "scaler" in ckpt and scaler is not None: scaler.load_state_dict(ckpt["scaler"])
                if "rng_state" in ckpt:
                    torch.set_rng_state(ckpt["rng_state"]["torch"])
                    if torch.cuda.is_available() and ckpt["rng_state"]["cuda"]:
                        torch.cuda.set_rng_state_all(ckpt["rng_state"]["cuda"])
                    np.random.set_state(ckpt["rng_state"]["numpy"])
                    random.setstate(ckpt["rng_state"]["python"])
            except Exception as e:
                only_rank0_print(f"[resume] Warning: could not fully load optimizer/scheduler/RNG: {e}")

    best_raw = float("inf"); best_ema = float("inf")
    out_base = Path(args.out)
    out_raw_ckpt = out_base.with_name(f"{out_base.stem}_raw_best{out_base.suffix or '.pt'}")
    out_ema_ckpt = out_base.with_name(f"{out_base.stem}_ema_best{out_base.suffix or '.pt'}")
    out_ema_model = out_base.with_name(f"{out_base.stem}_ema_model{out_base.suffix or '.pt'}")

    for epoch in range(1, args.epochs + 1):
        if isinstance(dl_tr.sampler, DistributedSampler): dl_tr.sampler.set_epoch(epoch)

        train_loss, steps_done, sec = train_one_epoch(
            model, loss_fn, dl_tr, opt, scaler, device, args.H,
            accum_steps=args.accum_steps, scheduler=scheduler, ema=ema, max_steps=args.max_steps)

        with torch.no_grad():
            val_raw = evaluate(model, loss_fn, dl_va, device, args.H, use_ema=None)
            val_ema = evaluate(model, loss_fn, dl_va, device, args.H, use_ema=ema) if ema is not None else None

        if ddp_rank() == 0:
            # Print both
            d_raw = (val_raw['snr_out'] - val_raw['snr_in']) if (not math.isnan(val_raw['snr_in']) and not math.isnan(val_raw['snr_out'])) else float('nan')
            print(f"Epoch {epoch:03d} | train {train_loss:.6f} | valRAW {val_raw['loss']:.6f} | "
                  f"EVM% {val_raw['evm_pct']:.2f} ({val_raw['evm_db']:.2f} dB) | "
                  f"SNR_in {val_raw['snr_in']:.2f} → SNR_out {val_raw['snr_out']:.2f} | Δ {d_raw:+.2f} dB | "
                  f"cov {val_raw['ref_cov']:.1f}% | lr {scheduler.get_last_lr()[0]:.2e} | {sec/60:.02f} min")

            if val_ema is not None:
                d_ema = (val_ema['snr_out'] - val_ema['snr_in']) if (not math.isnan(val_ema['snr_in']) and not math.isnan(val_ema['snr_out'])) else float('nan')
                print(f"            |        | valEMA {val_ema['loss']:.6f} | "
                      f"EVM% {val_ema['evm_pct']:.2f} ({val_ema['evm_db']:.2f} dB) | "
                      f"SNR_in {val_ema['snr_in']:.2f} → SNR_out {val_ema['snr_out']:.2f} | Δ {d_ema:+.2f} dB | "
                      f"cov {val_ema['ref_cov']:.1f}%")

            base_model = (model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model)

            # Save best RAW checkpoint
            if val_raw['loss'] < best_raw:
                best_raw = val_raw['loss']
                _save_ckpt(base_model, opt, scheduler, scaler, args, ema, out_raw_ckpt, {"best_metric": "raw_loss", "best_value": best_raw})
                print(f"  ↳ saved RAW best -> {out_raw_ckpt}")

            # Save best EMA checkpoint + export pure EMA model
            if val_ema is not None and val_ema['loss'] < best_ema:
                best_ema = val_ema['loss']
                _save_ckpt(base_model, opt, scheduler, scaler, args, ema, out_ema_ckpt, {"best_metric": "ema_loss", "best_value": best_ema})
                _export_ema_model(base_model, ema, out_ema_model)
                print(f"  ↳ saved EMA best -> {out_ema_ckpt}")
                print(f"  ↳ exported EMA model (for inference) -> {out_ema_model}")

    with torch.no_grad():
        te_raw = evaluate(model, loss_fn, dl_te, device, args.H, use_ema=None)
        te_ema = evaluate(model, loss_fn, dl_te, device, args.H, use_ema=ema) if ema is not None else None

    if ddp_rank() == 0:
        d_raw = (te_raw['snr_out'] - te_raw['snr_in']) if (not math.isnan(te_raw['snr_in']) and not math.isnan(te_raw['snr_out'])) else float('nan')
        print("=== TEST (RAW) === "
              f" loss {te_raw['loss']:.6f} | EVM% {te_raw['evm_pct']:.2f} ({te_raw['evm_db']:.2f} dB) "
              f"| SNR_in {te_raw['snr_in']:.2f} → SNR_out {te_raw['snr_out']:.2f} | Δ {d_raw:+.2f} dB "
              f"| cov {te_raw['ref_cov']:.1f}%")
        if te_ema is not None:
            d_ema = (te_ema['snr_out'] - te_ema['snr_in']) if (not math.isnan(te_ema['snr_in']) and not math.isnan(te_ema['snr_out'])) else float('nan')
            print("=== TEST (EMA) === "
                  f" loss {te_ema['loss']:.6f} | EVM% {te_ema['evm_pct']:.2f} ({te_ema['evm_db']:.2f} dB) "
                  f"| SNR_in {te_ema['snr_in']:.2f} → SNR_out {te_ema['snr_out']:.2f} | Δ {d_ema:+.2f} dB "
                  f"| cov {te_ema['ref_cov']:.1f}%")

    if dist.is_initialized(): dist.destroy_process_group()

if __name__ == "__main__":
    main()
