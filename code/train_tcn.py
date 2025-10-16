#!/usr/bin/env python3
"""
train_tcn.py — Causal TCN denoiser for GNSS L1 (pre‑correlation), with optional DSP prefilter
Features
- Causal TCN (dilated residual blocks)
- Optional DSP assist: frequency‑domain soft gate ("stft_gate") or single‑bin Gaussian notch ("notch")
- Input modes: raw | dsp | dualpath  (2‑ch raw IQ, 2‑ch DSP IQ, or 4‑ch concat)
- Band‑aware spectral loss (in‑band/guard/out‑of‑band weights)
- DDP (torchrun) + AMP support; metrics reduced across ranks; only rank 0 saves
"""

import os, time, argparse, math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

# ---------------------------
# DDP helpers
# ---------------------------

def ddp_env():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size, rank, local_rank

def ddp_init_if_needed(backend="nccl"):
    world_size, rank, local_rank = ddp_env()
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return world_size, rank, local_rank

def ddp_barrier():
    if dist.is_initialized():
        dist.barrier()

def ddp_all_reduce_tensor(t: torch.Tensor, op=dist.ReduceOp.SUM):
    if dist.is_initialized():
        dist.all_reduce(t, op=op)
    return t

# ---------------------------
# Utilities
# ---------------------------

def complex_from_iq(x: torch.Tensor) -> torch.Tensor:
    """x: [..., 2] float32 -> complex64"""
    return torch.view_as_complex(x.to(torch.float32))

def iq_from_complex(z: torch.Tensor) -> torch.Tensor:
    """z: complex64 -> [..., 2] float32"""
    return torch.view_as_real(z).to(torch.float32)

def center_and_rms_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """DC removal + unit RMS per example (per-block). x: [B,T,2]"""
    x = x - x.mean(dim=1, keepdim=True)
    rms = torch.sqrt((x.pow(2).sum(dim=-1).mean(dim=1, keepdim=True)) + eps)  # [B,1]
    return x / rms.unsqueeze(-1)

# ---------------------------
# Data
# ---------------------------

def _canonicalize_bt2(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 2 and a.shape[1] == 2:
        a = a[None, ...]
    if not (a.ndim == 3 and a.shape[-1] == 2):
        raise ValueError(f"Expected [N,T,2] but got {a.shape}")
    return a.astype(np.float32, copy=False)

def _find_sets(npz: Dict[str, np.ndarray]):
    keys = set(npz.keys())
    if {"Xtr","Ytr","Xva","Yva"}.issubset(keys):
        Xtr,Ytr,Xva,Yva = npz["Xtr"],npz["Ytr"],npz["Xva"],npz["Yva"]
        nmid = max(1, Xva.shape[0]//2)
        return {"train":(Xtr,Ytr), "val":(Xva[:nmid],Yva[:nmid]), "test":(Xva[nmid:],Yva[nmid:])}
    if {"X","Y"}.issubset(keys):
        X,Y = npz["X"],npz["Y"]; N=X.shape[0]
        ntr, nva = int(0.8*N), int(0.1*N)
        return {"train":(X[:ntr],Y[:ntr]), "val":(X[ntr:ntr+nva],Y[ntr:ntr+nva]), "test":(X[ntr+nva:],Y[ntr+nva:])}
    raise ValueError(f"Could not infer dataset keys from: {sorted(keys)}")

class IQWindows(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, W: int, H: int):
        self.X = _canonicalize_bt2(X); self.Y = _canonicalize_bt2(Y)
        if self.X.shape != self.Y.shape: raise ValueError("X and Y shapes must match.")
        self.W = int(W); self.H = int(H)
        N,T,_ = self.X.shape; self.T=T
        self.index = []
        if T == self.W:
            self.index = [(i,0) for i in range(N)]
        else:
            for i in range(N):
                for s in range(0, T - self.W + 1, self.H):
                    self.index.append((i,s))

    def __len__(self): return len(self.index)

    def __getitem__(self, i: int):
        n,s = self.index[i]; e=s+self.W
        x = self.X[n, s:e, :]; y = self.Y[n, s:e, :]
        return torch.from_numpy(x), torch.from_numpy(y)

def load_npz_dataset(path: str, W: int, H: int, batch: int, workers: int, *, world_size:int=1, rank:int=0):
    npz = np.load(path, allow_pickle=False)
    sets = _find_sets(npz)
    Xtr,Ytr = sets["train"]; Xva,Yva = sets["val"]; Xte,Yte = sets["test"]

    T_min = min(Xtr.shape[1], Xva.shape[1], Xte.shape[1])
    W_eff = min(int(W), int(T_min))
    if W_eff < W:
        print(f"[data] Requested W={W} exceeds dataset min T={T_min}. Clamping W -> {W_eff}.")

    ds_tr = IQWindows(Xtr, Ytr, W_eff, H)
    ds_va = IQWindows(Xva, Yva, W_eff, W_eff)

    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler_tr = DistributedSampler(ds_tr, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        sampler_va = DistributedSampler(ds_va, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    else:
        sampler_tr = None; sampler_va = None

    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=(sampler_tr is None),
                       sampler=sampler_tr, num_workers=workers, pin_memory=True,
                       drop_last=True, persistent_workers=(workers>0), prefetch_factor=4)
    dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False,
                       sampler=sampler_va, num_workers=workers, pin_memory=True,
                       drop_last=False, persistent_workers=(workers>0), prefetch_factor=4)

    return ds_tr, ds_va, dl_tr, dl_va, W_eff

# ---------------------------
# Prefilter (frequency-domain, per-window)
# ---------------------------

def _freq_bins(n: int, fs: float, device) -> torch.Tensor:
    return torch.linspace(0.0, fs/2.0, n//2 + 1, device=device)

def _band_weights(freqs: torch.Tensor, inband_hz: float, guard_hz: float, w_in: float, w_guard: float, w_out: float):
    w = torch.full_like(freqs, w_out)
    w = torch.where(freqs <= guard_hz, torch.full_like(freqs, w_guard), w)
    w = torch.where(freqs <= inband_hz, torch.full_like(freqs, w_in), w)
    return w

def stft_gate_prefilter(x: torch.Tensor, fs: float, inband_hz: float, guard_hz: float,
                        max_depth_in: float, max_depth_out: float, gate_k: float = 3.0) -> torch.Tensor:
    B,T = x.shape
    X = torch.fft.rfft(x, n=T, dim=1)  # [B, F]
    mag = X.abs()
    floor = torch.median(mag, dim=1, keepdim=True).values.clamp_min(1e-12)
    ratio = (mag / floor)
    excess = torch.clamp(ratio - gate_k, min=0.0)

    freqs = _freq_bins(T, fs, x.device)[None, :]
    Lin = 10.0 ** (-max_depth_in / 20.0)
    Lout = 10.0 ** (-max_depth_out / 20.0)

    in_mask   = (freqs <= inband_hz).to(x.dtype)
    att_in  = (1.0 - (1.0 - Lin)  * torch.tanh(excess / (gate_k*2.0)))
    att_out = (1.0 - (1.0 - Lout) * torch.tanh(excess / (gate_k*2.0)))
    att = att_in * in_mask + att_out * (1 - in_mask)

    Xf = X * att
    xf = torch.fft.irfft(Xf, n=T, dim=1)
    return xf

def notch_prefilter(x: torch.Tensor, fs: float, inband_hz: float, guard_hz: float,
                    max_depth_in: float, max_depth_out: float, q: float = 600.0) -> torch.Tensor:
    B,T = x.shape
    X = torch.fft.rfft(x, n=T, dim=1)
    mag = X.abs()
    peak_bin = torch.argmax(mag[:,1:], dim=1) + 1
    freqs = _freq_bins(T, fs, x.device)

    peak_freq = freqs[peak_bin]
    cap_db = torch.where(peak_freq <= guard_hz,
                         torch.where(peak_freq <= inband_hz,
                                     torch.full_like(peak_freq, max_depth_in),
                                     torch.full_like(peak_freq, max_depth_out)),
                         torch.full_like(peak_freq, max_depth_out))
    cap = 10.0 ** (-cap_db / 20.0)

    df = (fs/2.0) / (T//2)
    f0 = peak_freq.clamp_min(1.0)
    sigma_bins = (f0 / (q * df)).clamp_min(1.5)

    F = X.shape[1]
    idx = torch.arange(F, device=x.device)[None, :].repeat(B,1)
    pb = peak_bin[:,None].to(torch.float32)
    sb = sigma_bins[:,None]
    gauss = torch.exp(-0.5 * ((idx - pb)/sb)**2)
    att = 1.0 - (1.0 - cap[:,None]) * gauss

    Xf = X * att
    xf = torch.fft.irfft(Xf, n=T, dim=1)
    return xf

def apply_prefilter(x_btc2: torch.Tensor, fs: float, mode: str, inband_hz: float, guard_hz: float,
                    max_depth_in: float, max_depth_out: float) -> torch.Tensor:
    xc = complex_from_iq(x_btc2)
    xc = complex_from_iq(center_and_rms_norm(iq_from_complex(xc)))
    if mode == "none":
        return iq_from_complex(xc)
    elif mode == "stft_gate":
        xf = stft_gate_prefilter(xc, fs, inband_hz, guard_hz, max_depth_in, max_depth_out)
        return iq_from_complex(xf)
    elif mode == "notch":
        xf = notch_prefilter(xc, fs, inband_hz, guard_hz, max_depth_in, max_depth_out)
        return iq_from_complex(xf)
    else:
        raise ValueError(f"Unknown prefilter mode: {mode}")

# ---------------------------
# Model (Causal TCN)
# ---------------------------

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        padding = (kernel_size - 1) * dilation
        super().__init__(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.remove = padding
    def forward(self, x):
        out = super().forward(x)
        if self.remove > 0:
            out = out[..., :-self.remove]
        return out

class ResidualBlock(nn.Module):
    def __init__(self, ch, k, d, gn_groups=8, dropout=0.0):
        super().__init__()
        self.conv1 = CausalConv1d(ch, ch, k, d)
        self.gn1 = nn.GroupNorm(num_groups=min(gn_groups, ch), num_channels=ch)
        self.conv2 = CausalConv1d(ch, ch, k, d)
        self.gn2 = nn.GroupNorm(num_groups=min(gn_groups, ch), num_channels=ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.conv1(x)
        y = F.relu(self.gn1(y))
        y = self.drop(y)
        y = self.conv2(y)
        y = self.gn2(y)
        return F.relu(x + y)

class TCN(nn.Module):
    def __init__(self, in_ch: int, ch: int = 160, k: int = 7, n_blocks: int = 10, out_ch: int = 2):
        super().__init__()
        self.inp = CausalConv1d(in_ch, ch, 1, 1)
        blocks = []
        for i in range(n_blocks):
            d = 2 ** i
            blocks.append(ResidualBlock(ch, k, d))
        self.blocks = nn.Sequential(*blocks)
        self.out = nn.Conv1d(ch, out_ch, 1)
    def forward(self, x_btc):
        x = x_btc.permute(0,2,1)
        h = self.inp(x)
        h = self.blocks(h)
        y = self.out(h)
        y = y.permute(0,2,1)
        return y

# ---------------------------
# Loss & Metrics
# ---------------------------

def evm_pct_and_db(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-12):
    yt = complex_from_iq(y_true); yp = complex_from_iq(y_pred)
    num = torch.sum(torch.conj(yp) * yt, dim=1, keepdim=True)
    den = torch.sum(torch.conj(yp) * yp, dim=1, keepdim=True).clamp_min(eps)
    alpha = num / den
    err = alpha * yp - yt
    evm = torch.sqrt(torch.sum(torch.abs(err)**2, dim=1) / torch.sum(torch.abs(yt)**2, dim=1).clamp_min(eps))
    evm_pct = evm * 100.0
    evm_db = 20.0 * torch.log10(evm.clamp_min(eps))
    return evm_pct.mean(), evm_db.mean()

def snr_db(y_true: torch.Tensor, y_pred: torch.Tensor, x_in: torch.Tensor, eps: float = 1e-12):
    yt = complex_from_iq(y_true); yp = complex_from_iq(y_pred); xx = complex_from_iq(x_in)
    n_in = torch.sum(torch.abs(xx - yt)**2, dim=1)
    s   = torch.sum(torch.abs(yt)**2, dim=1).clamp_min(eps)
    snr_in = 10.0 * torch.log10((s / n_in.clamp_min(eps)).clamp_min(eps))
    n_out = torch.sum(torch.abs(yp - yt)**2, dim=1)
    snr_out = 10.0 * torch.log10((s / n_out.clamp_min(eps)).clamp_min(eps))
    return snr_in.mean(), snr_out.mean()

def spectral_loss(y_true: torch.Tensor, y_pred: torch.Tensor, fs: float, inband_hz: float, guard_hz: float,
                  w_in: float, w_guard: float, w_out: float, eps: float = 1e-9) -> torch.Tensor:
    yt = complex_from_iq(y_true); yp = complex_from_iq(y_pred)
    B,T = yt.shape
    YT = torch.fft.rfft(yt, n=T, dim=1)
    YP = torch.fft.rfft(yp, n=T, dim=1)
    freqs = _freq_bins(T, fs, yt.device)[None, :]
    W = _band_weights(freqs, inband_hz, guard_hz, w_in, w_guard, w_out)
    diff = (YP.abs() - YT.abs())
    return torch.mean(W * (diff ** 2))

def first_diff_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    yt = y_true[:,1:,:] - y_true[:,:-1,:]
    yp = y_pred[:,1:,:] - y_pred[:,:-1,:]
    return F.l1_loss(yp, yt)

class CompositeLoss(nn.Module):
    def __init__(self, fs: float, inband_hz: float, guard_hz: float,
                 spec_weight: float, w_in: float, w_guard: float, w_out: float,
                 smooth_weight: float, evm_norm_weight: float):
        super().__init__()
        self.fs=fs; self.inband=inband_hz; self.guard=guard_hz
        self.spec_weight=spec_weight; self.w_in=w_in; self.w_guard=w_guard; self.w_out=w_out
        self.smooth_weight=smooth_weight; self.evm_norm_weight=evm_norm_weight

        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        l_time = F.l1_loss(y_pred, y_true)
        l_spec = spectral_loss(y_true, y_pred, self.fs, self.inband, self.guard, self.w_in, self.w_guard, self.w_out)
        l_smooth = first_diff_loss(y_true, y_pred)
        evm_pct, _ = evm_pct_and_db(y_true, y_pred); evm_norm = (evm_pct / 100.0)
        return l_time + self.spec_weight*l_spec + self.smooth_weight*l_smooth + self.evm_norm_weight*evm_norm

# ---------------------------
# Train / Eval
# ---------------------------

def make_model(in_ch: int, ch: int, k: int, n_blocks: int, device):
    model = TCN(in_ch=in_ch, ch=ch, k=k, n_blocks=n_blocks, out_ch=2)
    return model.to(device)

def train(args):
    world_size, rank, local_rank = ddp_init_if_needed(backend="nccl" if not args.cpu else "gloo")
    use_ddp = world_size > 1
    if not args.cpu and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    else:
        device = torch.device("cpu")
    if rank == 0:
        print(f"Using device: {device} | world_size={world_size} rank={rank} local_rank={local_rank}")

    ds_tr, ds_va, dl_tr, dl_va, W_eff = load_npz_dataset(args.data, args.W, args.H, args.batch, args.workers,
                                                         world_size=world_size, rank=rank)

    in_ch = 2 if args.input_mode in ("raw","dsp") else 4
    model = make_model(in_ch, args.width, args.kernel, args.blocks, device)

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=None if device.type=="cpu" else [local_rank],
                                                          output_device=None if device.type=="cpu" else local_rank,
                                                          find_unused_parameters=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.99), weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda" and args.amp))

    loss_fn = CompositeLoss(fs=args.fs, inband_hz=args.inband_hz, guard_hz=args.guard_hz,
                            spec_weight=args.spec_weight, w_in=args.spec_w_in, w_guard=args.spec_w_guard, w_out=args.spec_w_out,
                            smooth_weight=args.smooth_weight, evm_norm_weight=args.evm_norm_weight)

    best_val = float("inf")

    for epoch in range(1, args.epochs+1):
        if use_ddp and hasattr(dl_tr.sampler, "set_epoch"):
            dl_tr.sampler.set_epoch(epoch)
        model.train()
        t0 = time.time(); run_loss = 0.0; m = 0
        for x, y in dl_tr:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            if args.prefilter != "none" or args.input_mode != "raw":
                dsp_x = apply_prefilter(x, args.fs, args.prefilter, args.inband_hz, args.guard_hz,
                                        args.prefilter_max_depth_in, args.prefilter_max_depth_out)
            else:
                dsp_x = x

            if args.input_mode == "raw":
                xin = x
            elif args.input_mode == "dsp":
                xin = dsp_x
            elif args.input_mode == "dualpath":
                xin = torch.cat([x, dsp_x], dim=-1)
            else:
                raise ValueError("invalid --input-mode")

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda" and args.amp)):
                yhat = model(xin)
                loss = loss_fn(yhat, y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            run_loss += float(loss.detach().cpu()); m += 1

        model.eval()
        tot_loss = torch.tensor([0.0], device=device); tot_n = torch.tensor([0.0], device=device)
        evm_pct_s, evm_db_s, snr_in_s, snr_out_s = torch.tensor([0.0], device=device), torch.tensor([0.0], device=device), torch.tensor([0.0], device=device), torch.tensor([0.0], device=device)
        cnt = torch.tensor([0.0], device=device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type=="cuda" and args.amp)):
            for x, y in dl_va:
                x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                if args.input_mode == "dualpath" or args.input_mode == "dsp" or args.prefilter != "none":
                    dsp_x = apply_prefilter(x, args.fs, args.prefilter, args.inband_hz, args.guard_hz,
                                            args.prefilter_max_depth_in, args.prefilter_max_depth_out)
                else:
                    dsp_x = x
                if args.input_mode == "raw":
                    xin = x
                elif args.input_mode == "dsp":
                    xin = dsp_x
                else:
                    xin = torch.cat([x, dsp_x], dim=-1)
                yhat = model(xin)
                l = loss_fn(yhat, y)
                tot_loss += l.detach()
                B = torch.tensor([x.size(0)], device=device, dtype=torch.float32)
                tot_n += B
                evm_pct, evm_db = evm_pct_and_db(y, yhat); ein, eout = snr_db(y, yhat, x[...,:2])
                evm_pct_s += evm_pct.detach(); evm_db_s += evm_db.detach()
                snr_in_s  += ein.detach();     snr_out_s += eout.detach()
                cnt += 1.0

        for t in (tot_loss, tot_n, evm_pct_s, evm_db_s, snr_in_s, snr_out_s, cnt):
            ddp_all_reduce_tensor(t, op=dist.ReduceOp.SUM)

        if rank == 0:
            val_loss = (tot_loss / torch.clamp_min(tot_n, 1.0)).item()
            evm_pct_m = (evm_pct_s / torch.clamp_min(cnt, 1.0)).item()
            evm_db_m  = (evm_db_s  / torch.clamp_min(cnt, 1.0)).item()
            snr_in_m  = (snr_in_s  / torch.clamp_min(cnt, 1.0)).item()
            snr_out_m = (snr_out_s / torch.clamp_min(cnt, 1.0)).item()
            dt = time.time() - t0
            print(f"Epoch {epoch:03d} | train {run_loss/max(1,m):.6f} | val {val_loss:.6f} | "
                  f"EVM% {evm_pct_m:.2f} ({evm_db_m:.2f} dB) | "
                  f"SNR_in {snr_in_m:.2f} → SNR_out {snr_out_m:.2f} | {dt/60:.2f} min")

            if val_loss < best_val:
                best_val = val_loss
                ckpt = {"model": model.module.state_dict() if use_ddp else model.state_dict(), "args": vars(args)}
                outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
                torch.save(ckpt, str(outp))
                print(f"  ↳ saved -> {outp}")

    if dist.is_initialized():
        ddp_barrier()
        if rank == 0:
            print("Training done (DDP).")
    else:
        print("Training done.")

# ---------------------------
# CLI
# ---------------------------

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to .npz dataset (Xtr/Ytr/Xva/Yva)")
    # Core data/time
    ap.add_argument("--fs", "--sample_rate", dest="fs", type=float, default=4.092e6, help="Sampling rate (Hz)")
    ap.add_argument("--W", type=int, default=4092, help="Window length (samples)")
    ap.add_argument("--H", type=int, default=4092, help="Hop (samples) for train windows")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    # Model
    ap.add_argument("--width", type=int, default=160, help="TCN hidden channels")
    ap.add_argument("--blocks", type=int, default=10, help="Number of residual dilated blocks")
    ap.add_argument("--kernel", type=int, default=7, help="Kernel size")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", type=str, default="tcn_denoiser.pt")
    ap.add_argument("--cpu", action="store_true", help="Force CPU")
    ap.add_argument("--amp", action="store_true", help="Mixed precision training")
    # Prefilter + inputs
    ap.add_argument("--prefilter", type=str, default="none", choices=["none","stft_gate","notch"])
    ap.add_argument("--input-mode","--input_mode", dest="input_mode", type=str, default="raw", choices=["raw","dsp","dualpath"])
    ap.add_argument("--inband-hz","--inband_hz", dest="inband_hz", type=float, default=1.2e6)
    ap.add_argument("--guard-hz","--guard_hz", dest="guard_hz", type=float, default=1.8e6)
    ap.add_argument("--prefilter-max-depth-in","--prefilter_max_depth_in", dest="prefilter_max_depth_in", type=float, default=18.0, help="Max attenuation in-band (dB)")
    ap.add_argument("--prefilter-max-depth-out","--prefilter_max_depth_out", dest="prefilter_max_depth_out", type=float, default=30.0, help="Max attenuation out-of-band (dB)")
    # Loss weights (hyphen + underscore aliases)
    ap.add_argument("--spec-weight","--spec_weight", dest="spec_weight", type=float, default=0.04)
    ap.add_argument("--spec-w-in","--spec_w_in", dest="spec_w_in", type=float, default=0.5)
    ap.add_argument("--spec-w-guard","--spec_w_guard", dest="spec_w_guard", type=float, default=1.0)
    ap.add_argument("--spec-w-out","--spec_w_out", dest="spec_w_out", type=float, default=2.0)
    ap.add_argument("--smooth-weight","--smooth_weight", dest="smooth_weight", type=float, default=0.05)
    ap.add_argument("--evm-norm-weight","--beta_evm_norm","--evm_norm_weight", dest="evm_norm_weight", type=float, default=0.02)
    # Backward‑compat harmless no‑ops so old sbatches don't crash
    ap.add_argument("--compile", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--grad_ckpt", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--use_bn", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--residual", dest="residual", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--no-residual", dest="residual", action="store_false", help=argparse.SUPPRESS)
    ap.set_defaults(residual=True)
    ap.add_argument("--wd", type=float, default=1e-4, help=argparse.SUPPRESS)
    ap.add_argument("--sched", type=str, default="none", choices=["none","cosine"], help=argparse.SUPPRESS)
    ap.add_argument("--warmup_epochs", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--max_steps", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--prefetch", type=int, default=4, help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=1337, help=argparse.SUPPRESS)
    ap.add_argument("--ema", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--ema_decay", type=float, default=0.999, help=argparse.SUPPRESS)
    ap.add_argument("--eval_use_ema", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--resume", type=str, default="", help=argparse.SUPPRESS)
    ap.add_argument("--resume_all", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--backend", type=str, default="nccl", help=argparse.SUPPRESS)
    return ap

def main():
    args = build_argparser().parse_args()
    train(args)

if __name__ == "__main__":
    main()
