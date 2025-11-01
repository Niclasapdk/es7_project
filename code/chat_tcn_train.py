#!/usr/bin/env python3
# Train a causal TCN denoiser on I/Q with a fixed notch prefilter (residual scheme).
# This version REMOVES the AGC/rotator and ADDS a cosine scheduler (with warmup) and CAWR option.

from __future__ import annotations
import os, math, time, argparse
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# ---------------- I/Q helpers ----------------
def ensure_iq(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.float32: x = x.float()
    if x.ndim != 3 or x.shape[-1] != 2:
        raise RuntimeError(f"Expected [B,T,2] IQ, got {tuple(x.shape)}")
    return x.contiguous()

def complex_from_iq(x: torch.Tensor) -> torch.Tensor:
    x = ensure_iq(x)
    return torch.complex(x[...,0], x[...,1])

def iq_from_complex(z: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(z): raise RuntimeError("Expected complex")
    return torch.stack([z.real, z.imag], dim=-1)

# ---------------- Dataset ----------------
class IQWindows(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        assert X.shape == Y.shape and X.ndim == 3 and X.shape[-1] == 2, f"Bad shapes {X.shape} vs {Y.shape}"
        self.X = X; self.Y = Y
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i: int):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])

def load_npz(path: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    npz = np.load(path)
    k = set(npz.keys())
    if {"Xtr","Ytr","Xva","Yva"}.issubset(k):
        Xtr,Ytr,Xva,Yva = npz["Xtr"],npz["Ytr"],npz["Xva"],npz["Yva"]
        mid = max(1, Xva.shape[0]//2)
        return {"train":(Xtr,Ytr),"val":(Xva[:mid],Yva[:mid]),"test":(Xva[mid:],Yva[mid:])}
    if {"X","Y"}.issubset(k):
        X,Y = npz["X"], npz["Y"]
        N = X.shape[0]; ntr=int(0.8*N); nva=int(0.1*N)
        return {"train":(X[:ntr],Y[:ntr]), "val":(X[ntr:ntr+nva],Y[ntr:ntr+nva]), "test":(X[ntr+nva:],Y[ntr+nva:])}
    raise ValueError(f"Unexpected keys in {path}: {sorted(k)}")

# ---------------- Notch prefilter ----------------
@torch.no_grad()
def notch_tonal_multi(x, fs, peaks=2, q=800.0, depth_db=90.0):
    def _as_complex(x):
        if x.ndim==3 and x.shape[-1]==2:
            z = torch.complex(x[...,0].float(), x[...,1].float()); back = lambda z: torch.stack([z.real,z.imag],-1)
            return z, back
        if torch.is_complex(x): return x, (lambda z:z)
        xr = x.float(); return xr, (lambda z:z.real)

    z,back = _as_complex(x)
    B,T = z.shape
    Z = torch.fft.fft(z, n=T, dim=1)
    mag = Z.abs(); mag[:,0] = 0.0
    freqs = torch.fft.fftfreq(T, d=1.0/fs, device=z.device)

    mask = torch.ones_like(Z, dtype=torch.float32)
    depth = 10.0**(-depth_db/20.0)
    for b in range(B):
        m = mag[b].clone()
        for _ in range(int(max(1,peaks))):
            k0 = int(torch.argmax(m).item())
            f0 = freqs[k0].item()
            bw_hz = max(abs(f0)/max(q,1.0), fs/T)
            df = (freqs - f0).abs()
            notch = 1.0 - (1.0 - depth) * (1.0/(1.0 + (df/bw_hz)**2))
            mask[b] = mask[b] * notch.float()
            m[max(0,k0-4):min(T,k0+5)] = 0.0
    zf = torch.fft.ifft(Z*mask, n=T, dim=1)
    return back(zf)

# ---------------- Model (causal TCN, residual) ----------------
class CausalConv1d(nn.Conv1d):
    def __init__(self, C_in, C_out, k, d=1):
        pad = (k-1)*d
        super().__init__(C_in, C_out, k, padding=pad, dilation=d)
        self._pad = pad
    def forward(self, x):
        y = super().forward(x)
        if self._pad: y = y[..., :-self._pad]
        return y

class TCNBlock(nn.Module):
    def __init__(self, C, k, d, dropout=0.0):
        super().__init__()
        self.conv1 = CausalConv1d(C, C, k, d)
        self.conv2 = CausalConv1d(C, C, k, d)
        self.norm1 = nn.BatchNorm1d(C)
        self.norm2 = nn.BatchNorm1d(C)
        self.drop = nn.Dropout(dropout)
        self.act  = nn.GELU()
    def forward(self, x):
        y = self.conv1(x); y = self.norm1(y); y = self.act(y); y = self.drop(y)
        y = self.conv2(y); y = self.norm2(y); y = self.drop(y)
        return self.act(x + y)

class ResidualTCN(nn.Module):
    def __init__(self, in_ch=4, hid=64, blocks=8, k=7, dropout=0.05):
        super().__init__()
        self.inp = CausalConv1d(in_ch, hid, k=3, d=1)
        self.tcn = nn.Sequential(*[TCNBlock(hid, k=k, d=2**b, dropout=dropout) for b in range(blocks)])
        self.out = CausalConv1d(hid, 2, k=3, d=1)
    def forward(self, x):  # x: [B,C,T]
        h = self.inp(x); h = self.tcn(h); r = self.out(h); return r

# ---------------- Losses (raw) ----------------
def spectral_loss(y_true: torch.Tensor, y_pred: torch.Tensor,
                  fs: float, inband_hz: float, guard_hz: float,
                  w_in: float, w_guard: float, w_out: float) -> torch.Tensor:
    yt = complex_from_iq(y_true); yp = complex_from_iq(y_pred)
    B,T = yt.shape
    Yt = torch.fft.fft(yt, n=T, dim=1)
    Yp = torch.fft.fft(yp, n=T, dim=1)
    freqs = torch.fft.fftfreq(T, d=1.0/fs, device=yt.device)
    BW = float(inband_hz)
    m_in    = (freqs >= -BW/2) & (freqs <= +BW/2)
    m_guard = ((freqs > +BW/2) & (freqs <= +BW/2+guard_hz)) | ((freqs < -BW/2) & (freqs >= -BW/2-guard_hz))
    m_out   = ~(m_in | m_guard)
    err = (Yp - Yt).abs()
    return (w_in*err[:,m_in].mean() + w_guard*err[:,m_guard].mean() + w_out*err[:,m_out].mean())

def first_diff_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    dy = y_pred[:,1:,:] - y_pred[:,:-1,:]
    return dy.abs().mean()

class CompositeLoss(nn.Module):
    def __init__(self, fs, inband_hz, guard_hz,
                 spec_w=0.3, w_in=1.0, w_guard=1.0, w_out=1.0,
                 smooth_w=0.05, time_w=1.0):
        super().__init__()
        self.fs=fs; self.inband=inband_hz; self.guard=guard_hz
        self.spec_w=spec_w; self.w_in=w_in; self.w_guard=w_guard; self.w_out=w_out
        self.smooth_w=smooth_w; self.time_w=time_w
    def forward(self, y_pred, y_true):
        l_time   = F.l1_loss(y_pred, y_true)
        l_spec   = spectral_loss(y_true, y_pred, self.fs, self.inband, self.guard,
                                 self.w_in, self.w_guard, self.w_out)
        l_smooth = first_diff_loss(y_true, y_pred)
        return self.time_w*l_time + self.spec_w*l_spec + self.smooth_w*l_smooth

# ---------------- Metrics (RAW only) ----------------
@torch.no_grad()
def snr_in_out_raw(x_in: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor) -> Tuple[float,float]:
    yt = complex_from_iq(y_true); xp = complex_from_iq(x_in); yp = complex_from_iq(y_pred)
    s = (yt.abs()**2).sum(dim=1).clamp_min(1e-12)
    n_in  = ((xp - yt).abs()**2).sum(dim=1).clamp_min(1e-12)
    n_out = ((yp - yt).abs()**2).sum(dim=1).clamp_min(1e-12)
    snr_in  = 10.0*torch.log10((s/n_in).clamp_min(1e-12)).mean().item()
    snr_out = 10.0*torch.log10((s/n_out).clamp_min(1e-12)).mean().item()
    return snr_in, snr_out

@torch.no_grad()
def evm_rms_pct_raw(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    yt = complex_from_iq(y_true); yp = complex_from_iq(y_pred)
    err_pow = ((yp-yt).abs()**2).sum(dim=1)
    ref_pow = (yt.abs()**2).sum(dim=1).clamp_min(1e-12)
    evm = torch.sqrt(err_pow/ref_pow).mean().item()
    return 100.0*evm

# ---------------- EMA ----------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay=float(decay)
        self.shadow={n:p.detach().clone() for n,p in (model.module if hasattr(model,"module") else model).named_parameters() if p.requires_grad}
        self.backup={}
    @torch.no_grad()
    def update(self, model):
        m = model.module if hasattr(model,"module") else model
        for n,p in m.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0-self.decay)
    @torch.no_grad()
    def apply(self, model):
        self.backup={}
        m = model.module if hasattr(model,"module") else model
        for n,p in m.named_parameters():
            if p.requires_grad:
                self.backup[n]=p.data.detach().clone()
                p.data.copy_(self.shadow[n].to(p.device))
    @torch.no_grad()
    def restore(self, model):
        if not self.backup: return
        m = model.module if hasattr(model,"module") else model
        for n,p in m.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup={}
    @torch.no_grad()
    def state_dict_for(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        m = model.module if hasattr(model,"module") else model
        sd = m.state_dict()
        out = {}
        for k, v in sd.items():
            if k in self.shadow:
                out[k] = self.shadow[k].detach().clone().to(v.device)
            else:
                out[k] = v.detach().clone()
        return out

# ---------------- DDP utils ----------------
def ddp_init() -> Tuple[int,int,int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK","0"))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local)
        return world, rank, local
    return 1, 0, 0

def is_master(rank:int)->bool: return (rank==0)

# ---------------- Scheduler helpers ----------------
def configure_scheduler(opt, args, steps_per_epoch, total_steps):
    if args.scheduler == "cosine":
        warm_steps = int(args.warmup_frac * total_steps)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps - warm_steps))
        return {"type":"cosine", "obj":sched, "warm_steps":warm_steps}
    elif args.scheduler == "cawr":
        # CosineAnnealingWarmRestarts; we optionally warm up the first few steps
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=max(1, args.cawr_T0), T_mult=max(1, args.cawr_Tmult)
        )
        warm_steps = int(args.warmup_frac * steps_per_epoch * args.cawr_T0) if args.warmup_frac>0 else 0
        return {"type":"cawr", "obj":sched, "warm_steps":warm_steps}
    else:
        return {"type":"none", "obj":None, "warm_steps":0}

# ---------------- Train / Eval ----------------
def make_model(in_ch:int, hid:int, blocks:int, k:int, dropout:float) -> nn.Module:
    return ResidualTCN(in_ch=in_ch, hid=hid, blocks=blocks, k=k, dropout=dropout)

def make_loader(X: np.ndarray, Y: np.ndarray, batch: int, shuffle: bool,
                rank:int, world:int, num_workers:int=4):
    ds = IQWindows(X,Y)
    if world>1:
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=shuffle, drop_last=False)
        return DataLoader(ds, batch_size=batch, sampler=sampler, num_workers=num_workers, pin_memory=True)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

def linear_warmup(step:int, warm:int):
    return min(1.0, (step+1)/max(1,warm))

def train(args):
    world, rank, local = ddp_init()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    master = is_master(rank)
    torch.backends.cudnn.benchmark = True

    data = load_npz(args.data)
    Xtr,Ytr = data["train"]; Xva,Yva = data["val"]

    train_loader = make_loader(Xtr,Ytr,args.batch, True,  rank, world, args.workers)
    val_loader   = make_loader(Xva,Yva,max(1,args.batch//2), False, rank, world, args.workers)

    model = make_model(in_ch=4, hid=args.width, blocks=args.blocks, k=args.kernel, dropout=args.dropout).to(device)
    if world>1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local], find_unused_parameters=False)

    loss_fn = CompositeLoss(args.fs, args.inband, args.guard,
                            spec_w=args.spec_w, w_in=args.spec_w_in, w_guard=args.spec_w_guard, w_out=args.spec_w_out,
                            smooth_w=args.smooth_w, time_w=args.time_w)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9,0.999))

    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.epochs * steps_per_epoch
    sched_info = configure_scheduler(opt, args, steps_per_epoch, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    ema = EMA(model, decay=args.ema) if args.ema>0 else None

    best_val = float("inf")
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(1, args.epochs+1):
        if world>1 and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        t0=time.time()
        for i, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)   # [B,T,2]
            yb = yb.to(device, non_blocking=True)

            # aggressive notch base (use CLI params)
            base = notch_tonal_multi(xb, fs=args.fs, peaks=args.notch_peaks,
                                     q=args.notch_q, depth_db=args.notch_depth)

            jam = xb.permute(0,2,1)                 # [B,2,T]
            bas = base.permute(0,2,1)               # [B,2,T]
            inp = torch.cat([jam, bas], dim=1)      # [B,4,T]

            with torch.cuda.amp.autocast(enabled=args.amp):
                resid = model(inp).permute(0,2,1)   # [B,T,2]
                yhat  = base + resid                # residual scheme
                loss  = loss_fn(yhat, yb)           # RAW loss (no alignment)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip>0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt); scaler.update()

            # ---- Scheduler step ----
            if sched_info["type"] == "cosine":
                if step < sched_info["warm_steps"]:
                    for pg in opt.param_groups:
                        pg["lr"] = args.lr * linear_warmup(step, sched_info["warm_steps"])
                else:
                    sched_info["obj"].step()
            elif sched_info["type"] == "cawr":
                if step < sched_info["warm_steps"]:
                    for pg in opt.param_groups:
                        pg["lr"] = args.lr * linear_warmup(step, sched_info["warm_steps"])
                # CAWR wants epoch progress as a float: epoch-1 + i/steps_per_epoch
                sched_info["obj"].step(epoch - 1 + (i / steps_per_epoch))

            if ema: ema.update(model)
            step += 1

        # ----- validation -----
        model.eval()
        if ema: ema.apply(model)  # use EMA params for val forward
        val_loss = 0.0; snr_in_sum=0.0; snr_out_sum=0.0; evm_sum=0.0; n_batches=0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
                base = notch_tonal_multi(xb, fs=args.fs, peaks=args.notch_peaks,
                                         q=args.notch_q, depth_db=args.notch_depth)
                jam = xb.permute(0,2,1); bas = base.permute(0,2,1)
                inp = torch.cat([jam, bas], dim=1)
                resid = model(inp).permute(0,2,1)
                yhat  = base + resid

                val_loss += loss_fn(yhat, yb).item()
                si, so = snr_in_out_raw(xb, yb, yhat); snr_in_sum += si; snr_out_sum += so
                evm_sum += evm_rms_pct_raw(yb, yhat)
                n_batches += 1
        if ema: ema.restore(model)

        # reduce across ranks
        if world>1:
            t = torch.tensor([val_loss, snr_in_sum, snr_out_sum, evm_sum, n_batches], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            val_loss, snr_in_sum, snr_out_sum, evm_sum, n_batches = t.tolist()

        val_loss /= max(1,n_batches)
        snr_in  = snr_in_sum/max(1,n_batches)
        snr_out = snr_out_sum/max(1,n_batches)
        evm     = evm_sum/max(1,n_batches)

        if is_master(rank):
            dt=time.time()-t0
            print(f"Epoch {epoch:03d} | val {val_loss:.6f} | SNR_in {snr_in:+.2f} dB → SNR_out {snr_out:+.2f} dB | EVM {evm:.2f}% | {dt:.1f}s")
            if val_loss < best_val:
                best_val = val_loss
                ck = {
                    "model": (ema.state_dict_for(model) if ema else (model.module.state_dict() if hasattr(model,"module") else model.state_dict())),
                    "model_raw": (model.module.state_dict() if hasattr(model,"module") else model.state_dict()),
                    "args": vars(args),
                    "best_val": best_val,
                    "epoch": epoch
                }
                path = Path(args.ckpt_dir)/"best.pt"
                torch.save(ck, path)
                print("  ↳ saved best ->", path)

    if world>1:
        dist.barrier(); dist.destroy_process_group()

# ---------------- CLI ----------------
def build_argparser():
    ap = argparse.ArgumentParser("Notch+TCN trainer — cosine scheduler variants")
    # data / io
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--ckpt-dir", type=str, default="ckpts_notch")
    # signal params
    ap.add_argument("--fs", type=float, default=4.092e6)
    ap.add_argument("--inband", type=float, default=2.046e6)
    ap.add_argument("--guard", type=float, default=150e3)
    # notch
    ap.add_argument("--notch-peaks", type=int, default=2)
    ap.add_argument("--notch-q", type=float, default=800.0)
    ap.add_argument("--notch-depth", type=float, default=90.0)
    # model
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--kernel", type=int, default=7)
    ap.add_argument("--dropout", type=float, default=0.05)
    # loss weights
    ap.add_argument("--time-w", type=float, default=1.0)
    ap.add_argument("--spec-w", type=float, default=0.3)
    ap.add_argument("--spec-w-in", type=float, default=1.0)
    ap.add_argument("--spec-w-guard", type=float, default=1.0)
    ap.add_argument("--spec-w-out", type=float, default=1.0)
    ap.add_argument("--smooth-w", type=float, default=0.05)
    # opt + scheduler
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.02, help="fraction of total steps (cosine) or first T0 (CAWR)")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scheduler", type=str, default="cawr", choices=["cosine","cawr","none"])
    ap.add_argument("--cawr-T0", type=int, default=5, help="epochs between first restart")
    ap.add_argument("--cawr-Tmult", type=int, default=2, help="restart period multiplier")
    return ap

def main():
    args = build_argparser().parse_args()
    train(args)

if __name__ == "__main__":
    main()
