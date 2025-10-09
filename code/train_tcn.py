#!/usr/bin/env python3
# train_tcn_ddp.py
# DDP-ready Causal TCN denoiser with EVM% / SNR_in/out / ΔSNR reporting
# Usage examples are shown below the main() function docstring.

import os, math, argparse, json, time, random, socket
from pathlib import Path
from typing import Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

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
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def is_ddp() -> bool:
    return dist.is_available() and dist.is_initialized()


def ddp_rank() -> int:
    return dist.get_rank() if is_ddp() else 0


def ddp_world_size() -> int:
    return dist.get_world_size() if is_ddp() else 1


def only_rank0_print(*args, **kwargs):
    if ddp_rank() == 0:
        print(*args, **kwargs)


def all_reduce_sum_(t: torch.Tensor) -> torch.Tensor:
    if is_ddp():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


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
            for xte,yte in [("Xte","Yte"), ("X_te","Y_te"), ("X_test","Y_test"), ("test_X","test_Y")]:
                if {xte,yte}.issubset(keys):
                    return {
                        "train": (Xtr, Ytr),
                        "val":   (Xva, Yva),
                        "test":  (npz[xte], npz[yte]),
                    }
            Nva = Xva.shape[0]
            mid = max(1, Nva // 2)
            return {
                "train": (Xtr, Ytr),
                "val":   (Xva[:mid], Yva[:mid]),
                "test":  (Xva[mid:], Yva[mid:]),
            }

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
    if np.iscomplexobj(a):
        a = np.stack([a.real, a.imag], axis=-1)

    if a.ndim == 3 and a.shape[-1] == 2:
        pass
    elif a.ndim == 3 and a.shape[1] == 2 and a.shape[-1] != 2:
        a = np.transpose(a, (0, 2, 1))
    elif a.ndim == 2 and a.shape[-1] == 2:
        a = a[None, ...]
    elif a.ndim == 2:
        a = np.stack([a, np.zeros_like(a)], axis=-1)
    elif a.ndim == 1:
        a = a[None, :, None]
        a = np.concatenate([a, np.zeros_like(a)], axis=-1)
    else:
        raise ValueError(f"Unsupported array shape {a.shape} for IQ data")

    if a.shape[-1] != 2:
        raise ValueError(f"Expected last dim=2 after canonicalize, got {a.shape}")
    return a.astype(np.float32, copy=False)


def _align_xy_bt2(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if X.ndim != 3 or Y.ndim != 3 or X.shape[-1] != 2 or Y.shape[-1] != 2:
        raise ValueError(f"Expected [N,T,2], got X {X.shape}, Y {Y.shape}")
    N = min(X.shape[0], Y.shape[0])
    T = min(X.shape[1], Y.shape[1])
    if X.shape[0] != Y.shape[0] or X.shape[1] != Y.shape[1]:
        only_rank0_print(f"[align] Trimming X {X.shape} / Y {Y.shape} to common (N={N}, T={T})")
    return X[:N, :T, :], Y[:N, :T, :]


def load_npz_dataset(path: Path, W: int, H: int, batch: int, workers: int = 4, prefetch: int = 4, drop_last=True):
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

    only_rank0_print(f"[data] shapes: train {Xtr.shape}, val {Xva.shape}, test {Xte.shape} | W={W} H={H}")

    ds_tr = IQWindows(Xtr, Ytr, W, H, drop_last=True)
    ds_va = IQWindows(Xva, Yva, W, H, drop_last=False)
    ds_te = IQWindows(Xte, Yte, W, H, drop_last=False)

    if len(ds_tr) == 0:
        only_rank0_print("[data] WARNING: train produced 0 windows with drop_last=True; retrying with drop_last=False (padded).")
        ds_tr = IQWindows(Xtr, Ytr, W, H, drop_last=False)

    if len(ds_tr) == 0:
        raise RuntimeError("Training dataset has 0 windows. Consider lowering --W or using --H<=W and ensure T>=1.")

    # Samplers for DDP
    if is_ddp():
        smp_tr = DistributedSampler(ds_tr, shuffle=True)
        smp_va = DistributedSampler(ds_va, shuffle=False)
        smp_te = DistributedSampler(ds_te, shuffle=False)
    else:
        smp_tr = None
        smp_va = None
        smp_te = None

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
    return ds_tr, ds_va, ds_te, dl_tr, dl_va, dl_te


# ---------------------------
# Model: Causal TCN
# ---------------------------

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        pad = (kernel_size - 1) * dilation
        super().__init__(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
    def forward(self, x):
        out = super().forward(x)
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

        return x + h


class TCN(nn.Module):
    def __init__(self, in_ch=2, ch=128, out_ch=2, k=5, blocks=10, dropout=0.05):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, ch, 1)
        layers = []
        for b in range(blocks):
            d = 2 ** b
            layers.append(TCNBlock(ch, k, dilation=d, dropout=dropout))
        self.tcn = nn.Sequential(*layers)
        self.out = nn.Conv1d(ch, out_ch, 1)

    def forward(self, x_bt2):
        x = x_bt2.transpose(1, 2)
        h = self.inp(x)
        h = self.tcn(h)
        y = self.out(h)
        y_bt2 = y.transpose(1, 2)
        return y_bt2, None


def receptive_field(kernel: int, blocks: int) -> int:
    return 1 + (kernel - 1) * (2 ** blocks - 1)


# ---------------------------
# Losses & Metrics
# ---------------------------

def pairwise_diff(x):
    d = x[:, 1:, :] - x[:, :-1, :]
    d = F.pad(d, (0, 0, 1, 0))
    return d


class TimeDomainLoss(nn.Module):
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


@torch.no_grad()
def estimate_lag(yt: torch.Tensor, yh: torch.Tensor, search: int) -> int:
    if search <= 0:
        return 0
    a = yt[0, :, 0] - yt[0, :, 0].mean()
    b = yh[0, :, 0] - yh[0, :, 0].mean()
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
    model.eval()

    # Distributed accumulators on device
    tot_loss = torch.zeros((), dtype=torch.float64, device=device)
    n_loss   = torch.zeros((), dtype=torch.float64, device=device)

    err_pow   = torch.zeros((), dtype=torch.float64, device=device)
    ref_pow   = torch.zeros((), dtype=torch.float64, device=device)
    sig_pow   = torch.zeros((), dtype=torch.float64, device=device)
    nse_in_pw = torch.zeros((), dtype=torch.float64, device=device)
    nse_outpw = torch.zeros((), dtype=torch.float64, device=device)

    kept_batches = torch.zeros((), dtype=torch.float64, device=device)
    total_batches = torch.zeros((), dtype=torch.float64, device=device)

    for x, y in dl:
        total_batches += 1
        x, y = to_device((x, y), device)
        with torch.cuda.amp.autocast(enabled=False):
            yhat, _ = model(x)
        loss = loss_fn(yhat, y, H)
        bs = x.size(0)
        tot_loss += loss.detach() * bs
        n_loss += bs

        Huse = min(H, y.shape[1])
        yt = y[:, -Huse:, :]
        yh = yhat[:, -Huse:, :]
        xx = x[:, -Huse:, :]

        if align_search is not None and align_search > 0:
            lag = estimate_lag(yt, yh, align_search)
            if lag > 0:
                yt = yt[:, :Huse-lag, :]
                yh = yh[:, lag:, :]
                xx = xx[:, :Huse-lag, :]
                if yt.shape[1] < 16:
                    continue

        ref_b = (yt.float().pow(2).sum(dim=(-1, -2))).mean()
        if ref_b.item() <= ref_thresh:
            continue

        err_pow   += (yh.float() - yt.float()).pow(2).sum(dtype=torch.float64)
        ref_pow   += (yt.float()).pow(2).sum(dtype=torch.float64)
        sig_pow   += (yt.float()).pow(2).sum(dtype=torch.float64)
        nse_in_pw += (xx.float() - yt.float()).pow(2).sum(dtype=torch.float64)
        nse_outpw += (yh.float() - yt.float()).pow(2).sum(dtype=torch.float64)
        kept_batches += 1

    # All-reduce across ranks
    for t in (tot_loss, n_loss, err_pow, ref_pow, sig_pow, nse_in_pw, nse_outpw, kept_batches, total_batches):
        all_reduce_sum_(t)

    mean_loss = (tot_loss / torch.clamp_min(n_loss, 1)).item()

    if ref_pow.item() <= eps or kept_batches.item() == 0:
        evm_pct = float("nan"); evm_db = float("nan")
        snr_in = float("nan"); snr_out = float("nan")
        cov = 0.0
    else:
        evm_lin = math.sqrt(float((err_pow + eps) / (ref_pow + eps)))
        evm_pct = 100.0 * evm_lin
        evm_db  = 20.0 * math.log10(max(eps, evm_lin))
        snr_in  = 10.0 * math.log10(float((sig_pow + eps) / (nse_in_pw + eps)))
        snr_out = 10.0 * math.log10(float((sig_pow + eps) / (nse_outpw + eps)))
        cov = 100.0 * float(kept_batches.item() / max(1.0, total_batches.item()))

    return {
        "loss": mean_loss,
        "evm_pct": evm_pct,
        "evm_db": evm_db,
        "snr_in": snr_in,
        "snr_out": snr_out,
        "n": int(n_loss.item()),
        "ref_cov": cov,
    }


# ---------------------------
# Training
# ---------------------------

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


def train_one_epoch(model, loss_fn, dl, opt, scaler, device, H, accum_steps, max_steps=0):
    model.train()
    t0 = time.time()
    running = 0.0
    steps = 0
    samples = 0

    # In DDP, set the epoch on the sampler to reshuffle shards consistently
    if is_ddp() and isinstance(dl.sampler, DistributedSampler):
        # caller must set epoch before invoking this function for reproducibility
        pass

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

    # Average training loss across ranks for logging
    tr_loss_t = torch.tensor(running / max(1, samples), device=device, dtype=torch.float64)
    all_reduce_sum_(tr_loss_t)
    tr_loss = (tr_loss_t / ddp_world_size()).item()

    return tr_loss, steps, dt


# ---------------------------
# Distributed init helpers
# ---------------------------

def init_distributed(backend: str = "nccl", port: Optional[int] = None):
    """Initialize torch.distributed using env variables from torchrun or SLURM.
    Safe to call even for single-process training.
    """
    if dist.is_initialized():
        return

    # Already provided by torchrun or srun --mpi=pmi2 on many clusters
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))

    if world > 1:
        if "MASTER_ADDR" not in os.environ:
            # Best-effort: set master to rank0 node
            host = os.environ.get("SLURM_LAUNCH_NODE_IPADDR") or os.environ.get("SLURM_NODELIST") or "127.0.0.1"
            os.environ["MASTER_ADDR"] = str(host).split()[0]
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(port or 29500)

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, rank=rank, world_size=world)

    return local_rank


# ---------------------------
# CLI / Main
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Causal TCN denoiser training (DDP-ready)")
    p.add_argument("--data", type=str, required=True, help="Path to .npz dataset")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=256, help="per-GPU batch size")
    p.add_argument("--accum_steps", type=int, default=1, help="gradient accumulation steps")
    p.add_argument("--amp", action="store_true", help="use Automatic Mixed Precision")
    p.add_argument("--compile", action="store_true", help="torch.compile() the model")
    p.add_argument("--W", type=int, default=2048, help="window length")
    p.add_argument("--H", type=int, default=512,  help="hop/emit length")
    p.add_argument("--width", type=int, default=192, help="TCN channels")
    p.add_argument("--blocks", type=int, default=10, help="TCN residual blocks")
    p.add_argument("--kernel", type=int, default=5, help="TCN kernel size")
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=4e-3)
    p.add_argument("--wd", type=float, default=5e-3)
    p.add_argument("--sched", type=str, default="cosine", choices=["cosine","none"])
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=0, help="max optimizer steps per epoch (after grad accumulation)")
    p.add_argument("--workers", type=int, default=16, help="DataLoader workers per process (GPU)")
    p.add_argument("--prefetch", type=int, default=4, help="prefetch_factor per worker")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out", type=str, default="tcn_denoiser.pt")
    p.add_argument("--alpha", type=float, default=0.05, help="delta-L1 weight in loss")
    p.add_argument("--align_search", type=int, default=-1, help="override lag search window; default=-1 uses RF/8 capped to 256")
    p.add_argument("--backend", type=str, default="nccl", help="DDP backend")
    return p.parse_args()


def main():
    """
    Launch with torchrun (single node, 8 GPUs):
      torchrun --nproc_per_node=8 train_tcn_ddp.py --data <dataset.npz> --epochs 200 --batch 512 --workers 16 --amp --compile

    Under SLURM (multi-node), inside your container:
      srun --ntasks-per-node=8 --gpus-per-node=8 --cpus-per-task=16 \
           python -u -m torch.distributed.run --nproc_per_node=$SLURM_GPUS_ON_NODE \
           --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID train_tcn_ddp.py --data <dataset.npz> ...
    """
    args = parse_args()
    set_seed(args.seed)

    torch.backends.cudnn.benchmark = True

    # DDP init (no-op if world_size==1)
    local_rank = init_distributed(backend=args.backend)

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    if ddp_rank() == 0:
        only_rank0_print(f"Device: {device} | seed {args.seed} | world_size {ddp_world_size()} | rank {ddp_rank()}")

    # Data
    data_path = Path(args.data)
    ds_tr, ds_va, ds_te, dl_tr, dl_va, dl_te = load_npz_dataset(
        data_path, W=args.W, H=args.H, batch=args.batch, workers=args.workers, prefetch=args.prefetch, drop_last=True)

    # Model
    model = TCN(in_ch=2, ch=args.width, out_ch=2, k=args.kernel, blocks=args.blocks, dropout=args.dropout).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    nparams = numel(model)
    only_rank0_print(f"Params: {nparams/1e6:.2f}M | W {args.W} H {args.H} | width {args.width} blocks {args.blocks} k {args.kernel}")

    # Loss / Opt / Sched
    loss_fn = TimeDomainLoss(alpha=args.alpha)
    opt = make_optimizer(model, lr=args.lr, wd=args.wd)

    steps_per_epoch = max(1, len(dl_tr) // max(1, args.accum_steps))
    scheduler = make_scheduler(opt, args.sched, args.epochs, steps_per_epoch, args.warmup_epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Wrap with DDP
    if ddp_world_size() > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device.index] if device.type=="cuda" else None,
                                                    output_device=device.index if device.type=="cuda" else None,
                                                    find_unused_parameters=False)

    RF = receptive_field(args.kernel, args.blocks)
    align_search = min(256, max(0, RF // 8)) if (args.align_search is None or args.align_search < 0) else max(0, args.align_search)
    if ddp_rank() == 0:
        only_rank0_print(f"Receptive field (samples): {RF} | align_search: {align_search} | steps/epoch: {steps_per_epoch}")

    best_val = float("inf")
    best_ckpt = args.out

    global_step = 0
    t_train0 = time.time()
    for ep in range(1, args.epochs + 1):
        # Ensure different shuffles per epoch on each rank
        if is_ddp() and isinstance(dl_tr.sampler, DistributedSampler):
            dl_tr.sampler.set_epoch(ep)

        tr_loss, tr_steps, tr_dt = train_one_epoch(model, loss_fn, dl_tr, opt, scaler, device, args.H, args.accum_steps, args.max_steps)
        global_step += tr_steps

        if isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
            scheduler.step()

        # Eval (all ranks compute on their shards; evaluate() all-reduces metrics)
        with torch.no_grad():
            va = evaluate(model, loss_fn, dl_va, device, args.H, align_search=align_search)

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
                }
                torch.save(state, best_ckpt)
                only_rank0_print(f"  ↳ saved -> {best_ckpt}")

    # Final test (all-reduced across ranks)
    with torch.no_grad():
        te = evaluate(model, loss_fn, dl_te, device, args.H, align_search=align_search)
    if ddp_rank() == 0:
        print("\n=== TEST === "
              f" loss {te['loss']:.6f} | EVM% {te['evm_pct']:.2f} ({te['evm_db']:.2f} dB) "
              f"| SNR_in {te['snr_in']:.2f} → SNR_out {te['snr_out']:.2f} "
              f"| Δ {(te['snr_out'] - te['snr_in']) if (not math.isnan(te['snr_in']) and not math.isnan(te['snr_out'])) else float('nan'):+.2f} dB "
              f"| cov {te['ref_cov']:.1f}%")


if __name__ == "__main__":
    main()
