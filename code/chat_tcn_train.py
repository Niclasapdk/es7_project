#!/usr/bin/env python3
# Train a causal TCN denoiser on I/Q with a fixed notch prefilter (residual scheme).
# This build includes:
# - AMP 2.0 API
# - Spec-weight ramp (fixed to keep final value)
# - Optional fractional-delay + gain alignment in the loss
# - Optional Top-K spectral peak emphasis
# - Optional per-seq RMS norm
# - EMA
# - Cosine or CAWR scheduler
# - DDP support
# - QUIET aligned,in-band EVM diagnostic with --diag-ai-evm {off,epoch,batch}

from __future__ import annotations
import os, math, time, argparse, warnings
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")

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

# ---------------- optional per-sequence normalization ----------------
def perseq_rms_norm(x_iq: torch.Tensor, y_iq: torch.Tensor, eps: float = 1e-8):
    """Normalize both input and target by target RMS per sequence. x_iq,y_iq: [B,T,2]"""
    yr = y_iq[...,0]; yi = y_iq[...,1]
    rms = torch.sqrt((yr**2 + yi**2).mean(dim=1, keepdim=True).clamp_min(eps))  # [B,1]
    rms = rms.unsqueeze(-1)  # [B,1,1]
    return x_iq / rms, y_iq / rms

# ---------------- Dataset ----------------
class IQWindows(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        assert X.shape == Y.shape and X.ndim == 3 and X.shape[-1] == 2, f"Bad shapes {X.shape} vs {Y.shape}"
        self.X = X.astype(np.float32); self.Y = Y.astype(np.float32)
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
# ---------------- Notch prefilter ----------------
@torch.no_grad()
def notch_tonal_multi(x: torch.Tensor,
                      fs: float,
                      peaks: int = 2,
                      q: float = 800.0,
                      depth_db: float = 90.0) -> torch.Tensor:
    """
    Frequency-domain multi-tone *soft* notch on complex IQ.
    Finds 'peaks' largest spectral bins (excl. DC) and ATTENUATES a window
    of half-width ~T/q bins around each peak up to depth_db (dB) at the center
    with a raised-cosine taper back to 0 dB at the edges.
    """
    import math

    def _as_complex(x_):
        if x_.ndim == 3 and x_.shape[-1] == 2:
            z = torch.complex(x_[..., 0].float(), x_[..., 1].float())
            back = lambda z_: torch.stack([z_.real, z_.imag], dim=-1)
            return z, back
        if torch.is_complex(x_):
            return x_, (lambda z_: z_)
        xr = x_.float()
        return xr, (lambda z_: z_.real)

    z, back = _as_complex(x)
    B, T = z.shape
    X = torch.fft.fft(z, dim=1)            # [B,T], complex
    mag = X.abs()
    mag[:, 0] = 0.0                        # ignore DC

    # Build multiplicative mask (1 = pass, a = depth at center)
    M = torch.ones_like(X, dtype=X.real.dtype)
    half = max(1, int(T / max(1.0, q)))    # half-window in bins
    a = float(10.0 ** (-depth_db / 20.0))  # linear gain at peak

    peaks = max(0, int(peaks))
    for _ in range(peaks):
        idx = torch.argmax(mag, dim=1)     # [B]
        # Zero out this region in 'mag' so we don't re-pick it next iteration
        for b in range(B):
            k = int(idx[b].item())
            for center in (k, (-k) % T):   # handle ±k like before
                for o in range(-half, half + 1):
                    t = abs(o) / float(half)
                    # raised-cosine from 'a' at center to 1.0 at edges
                    g = a + (1.0 - a) * (1.0 - math.cos(math.pi * t)) * 0.5
                    j = (center + o) % T
                    # elementwise min so total attenuation never exceeds 'a'
                    M[b, j] = torch.minimum(M[b, j], torch.tensor(g, device=M.device, dtype=M.dtype))
            # also suppress the region in 'mag' to avoid reselecting
            rng = [(k + o) % T for o in range(-half, half + 1)]
            mag[b, rng] = 0.0
            k2 = (-k) % T
            rng2 = [(k2 + o) % T for o in range(-half, half + 1)]
            mag[b, rng2] = 0.0

    X = X * M                              # apply soft notch
    zf = torch.fft.ifft(X, dim=1)
    return back(zf)

# ---------------- Model ----------------
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
        h = self.inp(x)
        h = self.tcn(h)
        r = self.out(h)    # residual [B,2,T]
        return r

# ---------------- Losses ----------------
def spectral_loss(y_true: torch.Tensor, y_pred: torch.Tensor, fs: float,
                  inband_hz: float, guard_hz: float,
                  w_in: float=1.0, w_guard: float=1.0, w_out: float=1.0) -> torch.Tensor:
    """Region-weighted L1 spectrum error over full band."""
    yt = complex_from_iq(y_true); yp = complex_from_iq(y_pred)
    B,T = yt.shape
    Yt = torch.fft.fft(yt, dim=1); Yp = torch.fft.fft(yp, dim=1)
    freqs = torch.fft.fftfreq(T, d=1.0/fs).to(yt.device)  # [-fs/2, fs/2)
    Yt = torch.fft.fftshift(Yt, dim=1); Yp = torch.fft.fftshift(Yp, dim=1); freqs = torch.fft.fftshift(freqs)
    BW = inband_hz
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
                 smooth_w=0.05, time_w=1.0,
                 align_maxlag: int = 0, align_w: float = 0.0, gain_align: bool = False,
                 align_frac_steps: int = 0,
                 peak_w: float = 0.0, peak_k: int = 0, peak_region: str = "guard"):
        super().__init__()
        self.fs=fs; self.inband=inband_hz; self.guard=guard_hz
        self.spec_w=spec_w; self.w_in=w_in; self.w_guard=w_guard; self.w_out=w_out
        self.smooth_w=smooth_w; self.time_w=time_w
        self.align_maxlag = int(max(0, align_maxlag))
        self.align_w = float(max(0.0, align_w))
        self.gain_align = bool(gain_align)
        self.align_frac_steps = int(max(0, align_frac_steps))
        self.peak_w = float(max(0.0, peak_w))
        self.peak_k = int(max(0, peak_k))
        assert peak_region in ("guard","in","both")
        self.peak_region = peak_region

    def set_spec_weight(self, w: float): self.spec_w = float(max(0.0, w))

    @staticmethod
    def _frac_delay(z: torch.Tensor, fs: float, tau: float) -> torch.Tensor:
        # z: [B,T] complex. Apply fractional delay tau (samples) via FFT phase ramp.
        B,T = z.shape
        W = 2.0*math.pi*torch.fft.fftfreq(T, d=1.0/fs).to(z.device)
        Z = torch.fft.fft(z, dim=1)
        phase = torch.exp(-1j*W * tau).unsqueeze(0)
        Zs = Z * phase
        return torch.fft.ifft(Zs, dim=1)

    def _best_align(self, y_pred_iq: torch.Tensor, y_true_iq: torch.Tensor,
                    maxlag: int, gain_align: bool, frac_steps: int):
        # Returns IQ aligned prediction [B,T,2]
        if maxlag <= 0 and frac_steps <= 0: return y_pred_iq
        yp = torch.complex(y_pred_iq[...,0], y_pred_iq[...,1])
        yt = torch.complex(y_true_iq[...,0], y_true_iq[...,1])
        B,T = yp.shape

        fracs = [0.0]
        if frac_steps > 0:
            step = 0.5 / frac_steps
            fracs = [i*step for i in range(-frac_steps, frac_steps+1)]

        best = None; best_val = None
        for L in range(-maxlag, maxlag+1):
            for f in fracs:
                tau = float(L) + float(f)
                if abs(f) > 1e-12:
                    y_shift = self._frac_delay(yp, self.fs, tau)
                else:
                    if L > 0:
                        y_shift = torch.cat([torch.zeros(B, L, device=yp.device, dtype=yp.dtype), yp[:, :-L]], dim=1)
                    elif L < 0:
                        y_shift = torch.cat([yp[:, -L:], torch.zeros(B, -L, device=yp.device, dtype=yp.dtype)], dim=1)
                    else:
                        y_shift = yp
                if gain_align:
                    pwr = (y_shift.conj()*y_shift).real.sum(dim=1).float().clamp_min(1e-12)
                    num = (yt.conj()*y_shift).sum(dim=1)
                    g = num / pwr
                    y_al = y_shift * g.unsqueeze(1)
                else:
                    y_al = y_shift
                score = -((y_al - yt).abs()**2).sum(dim=1)  # [B]
                if best is None:
                    best = y_al; best_val = score
                else:
                    mask = (score > best_val)
                    best = torch.where(mask.unsqueeze(1), y_al, best)
                    best_val = torch.where(mask, score, best_val)
        return torch.stack([best.real, best.imag], dim=-1)

    def _peak_emphasis(self, y_true_iq: torch.Tensor, y_pred_iq: torch.Tensor) -> torch.Tensor:
        # Penalize top-K residual spectral spikes in chosen region (guard/in/both).
        yt = complex_from_iq(y_true_iq); yp = complex_from_iq(y_pred_iq)
        B,T = yt.shape
        Yt = torch.fft.fftshift(torch.fft.fft(yt, dim=1), dim=1)
        Yp = torch.fft.fftshift(torch.fft.fft(yp, dim=1), dim=1)
        freqs = torch.fft.fftshift(torch.fft.fftfreq(T, d=1.0/self.fs)).to(yt.device)

        BW = self.inband
        m_in    = (freqs >= -BW/2) & (freqs <= +BW/2)
        m_guard = ((freqs > +BW/2) & (freqs <= +BW/2 + self.guard)) | ((freqs < -BW/2) & (freqs >= -BW/2 - self.guard))
        if self.peak_region == "guard":
            m = m_guard
        elif self.peak_region == "in":
            m = m_in
        else:
            m = (m_in | m_guard)

        diff = (Yp - Yt).abs()[:, m]  # [B,M]
        if diff.numel()==0 or self.peak_k<=0: return diff.new_tensor(0.0)
        k = min(self.peak_k, diff.shape[1])
        topk = torch.topk(diff, k=k, dim=1).values  # [B,k]
        return topk.mean()

    def forward(self, y_pred, y_true):
        l_time   = F.l1_loss(y_pred, y_true)
        l_spec   = spectral_loss(y_true, y_pred, self.fs, self.inband, self.guard,
                                 self.w_in, self.w_guard, self.w_out)
        l_smooth = first_diff_loss(y_true, y_pred)
        loss = self.time_w*l_time + self.spec_w*l_spec + self.smooth_w*l_smooth

        if self.align_w > 0.0 and (self.align_maxlag > 0 or self.align_frac_steps > 0):
            y_al = self._best_align(y_pred, y_true, self.align_maxlag, self.gain_align, self.align_frac_steps)
            l_align = F.l1_loss(y_al, y_true)
            loss = loss + self.align_w * l_align

        if self.peak_w > 0.0 and self.peak_k > 0:
            l_peak = self._peak_emphasis(y_true, y_pred)
            loss = loss + self.peak_w * l_peak

        return loss

# ---------------- Metrics (RAW + aligned/in-band diag) ----------------
@torch.no_grad()
def snr_in_out_raw(x_in: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor) -> Tuple[float,float]:
    yt = complex_from_iq(y_true); xp = complex_from_iq(x_in); yp = complex_from_iq(y_pred)
    s = (yt.abs()**2).sum(dim=1).clamp_min(1e-12)
    n_in  = ((xp - yt).abs()**2).sum(dim=1).clamp_min(1e-12)
    n_out = ((yp - yt).abs()**2).sum(dim=1).clamp_min(1e-12)
    snr_in  = (10.0*torch.log10(s/n_in)).mean().item()
    snr_out = (10.0*torch.log10(s/n_out)).mean().item()
    return float(snr_in), float(snr_out)

@torch.no_grad()
def evm_rms_pct_raw(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    yt = complex_from_iq(y_true); yp = complex_from_iq(y_pred)
    err_pow = ((yp - yt).abs()**2).mean(dim=1).clamp_min(1e-12)
    ref_pow = (yt.abs()**2).mean(dim=1).clamp_min(1e-12)
    evm = torch.sqrt(err_pow/ref_pow).mean().item()
    return 100.0*evm

def bandlimit_inband(y_iq: torch.Tensor, fs: float, inband_hz: float, guard_hz: float) -> torch.Tensor:
    z = complex_from_iq(y_iq)                       # [B,T]
    B,T = z.shape
    F = torch.fft.fft(z, dim=1)
    freqs = torch.fft.fftfreq(T, d=1.0/fs).to(z.device)
    mask = (freqs >= -inband_hz/2) & (freqs <= +inband_hz/2)
    F = F * mask.unsqueeze(0)
    zf = torch.fft.ifft(F, dim=1)
    return iq_from_complex(zf)

@torch.no_grad()
def align_best_frac_gain(y_pred_iq: torch.Tensor, y_true_iq: torch.Tensor,
                         fs: float, maxlag: int = 12, frac_steps: int = 5, gain_align: bool = True) -> torch.Tensor:
    B,T,_ = y_pred_iq.shape
    yp = torch.complex(y_pred_iq[...,0], y_pred_iq[...,1])
    yt = torch.complex(y_true_iq[...,0], y_true_iq[...,1])
    def frac_delay(z, tau):
        W = 2.0*math.pi*torch.fft.fftfreq(T, d=1.0/fs).to(z.device)
        Z = torch.fft.fft(z, dim=1)
        phase = torch.exp(-1j*W * tau).unsqueeze(0)
        return torch.fft.ifft(Z*phase, dim=1)

    fracs = [i*(0.5/max(1,frac_steps)) for i in range(-frac_steps, frac_steps+1)] if frac_steps>0 else [0.0]
    best=None; best_val=None
    for L in range(-maxlag, maxlag+1):
        for f in fracs:
            tau = float(L)+float(f)
            if abs(f) > 1e-12:
                y_shift = frac_delay(yp, tau)
            else:
                if L>0:  y_shift = torch.cat([torch.zeros(B, L, device=yp.device, dtype=yp.dtype), yp[:, :-L]], dim=1)
                elif L<0:y_shift = torch.cat([yp[:, -L:], torch.zeros(B, -L, device=yp.device, dtype=yp.dtype)], dim=1)
                else:    y_shift = yp
            if gain_align:
                pwr = (y_shift.conj()*y_shift).real.sum(dim=1).float().clamp_min(1e-12)
                num = (yt.conj()*y_shift).sum(dim=1)
                g = num / pwr
                y_al = y_shift * g.unsqueeze(1)
            else:
                y_al = y_shift
            score = -((y_al - yt).abs()**2).sum(dim=1)
            if best is None:
                best = y_al; best_val = score
            else:
                mask = (score > best_val)
                best = torch.where(mask.unsqueeze(1), y_al, best)
                best_val = torch.where(mask, score, best_val)
    return iq_from_complex(best)

@torch.no_grad()
def evm_aligned_inband_pct(y_true: torch.Tensor, y_pred: torch.Tensor, fs: float, inband: float, guard: float,
                           maxlag=12, frac_steps=5, gain_align=True) -> float:
    y_pred_al = align_best_frac_gain(y_pred, y_true, fs, maxlag, frac_steps, gain_align)
    yt_in  = bandlimit_inband(y_true, fs, inband, guard)
    yp_in  = bandlimit_inband(y_pred_al, fs, inband, guard)
    yt = complex_from_iq(yt_in); yp = complex_from_iq(yp_in)
    err_pow = ((yp - yt).abs()**2).mean(dim=1).clamp_min(1e-12)
    ref_pow = (yt.abs()**2).mean(dim=1).clamp_min(1e-12)
    return 100.0*torch.sqrt(err_pow/ref_pow).mean().item()

# ---------------- EMA ----------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay=float(decay)
        self.shadow={}
        for n,p in (model.module if hasattr(model,"module") else model).named_parameters():
            if p.requires_grad: self.shadow[n]=p.detach().clone()
        self.backup={}
    @torch.no_grad()
    def update(self, model):
        m = model.module if hasattr(model,"module") else model
        for n,p in m.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0-self.decay)
    @torch.no_grad()
    def apply(self, model):
        self.backup={}
        m = model.module if hasattr(model,"module") else model
        for n,p in m.named_parameters():
            if p.requires_grad and n in self.shadow:
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
    def state_dict_for(self, model):
        return {k:v.cpu() for k,v in self.shadow.items()}

# ---------------- DDP utils ----------------
def ddp_init():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        world = int(os.environ["WORLD_SIZE"]); rank  = int(os.environ["RANK"])
        local = int(os.environ.get("LOCAL_RANK","0"))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local)
        return world, rank, local
    return 1, 0, 0

def is_master(rank:int)->bool: return (rank==0)

# ---------------- Scheduler helpers ----------------
def linear_warmup(step: int, warm_steps: int) -> float:
    return min(1.0, (step+1)/max(1, warm_steps))

def configure_scheduler(opt, args, steps_per_epoch, total_steps):
    if args.scheduler == "cosine":
        warm_steps = int(args.warmup_frac * total_steps)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps - warm_steps))
        return {"type":"cosine", "obj":sched, "warm_steps":warm_steps}
    elif args.scheduler == "cawr":
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
    else:
        return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

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
                            smooth_w=args.smooth_w, time_w=args.time_w,
                            align_maxlag=args.align_maxlag, align_w=args.align_w, gain_align=args.gain_align,
                            align_frac_steps=args.align_frac_steps,
                            peak_w=args.peak_w, peak_k=args.peak_k, peak_region=args.peak_region)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9,0.999))

    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.epochs * steps_per_epoch
    sched_info = configure_scheduler(opt, args, steps_per_epoch, total_steps)

    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    ema = EMA(model, decay=args.ema) if args.ema>0 else None

    best_val = float("inf")
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(1, args.epochs+1):
        if world>1 and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        # ---- spec loss ramp (fixed: keep final after ramp) ----
        if args.spec_w_ramp_epochs > 0:
            if epoch <= args.spec_w_ramp_epochs:
                t_prog = epoch / float(max(1, args.spec_w_ramp_epochs))
                curr_spec = args.spec_w + t_prog * (args.spec_w_final - args.spec_w)
            else:
                curr_spec = args.spec_w_final
            loss_fn.set_spec_weight(curr_spec)
        else:
            loss_fn.set_spec_weight(args.spec_w)

        # ---- optional peak ramp ----
        if args.peak_w > 0.0 and args.peak_k > 0:
            if args.peak_ramp_epochs > 0:
                phase = min(1.0, epoch / float(max(1, args.peak_ramp_epochs)))
                loss_fn.peak_w = args.peak_w * phase
            else:
                loss_fn.peak_w = args.peak_w

        model.train()
        t0=time.time()
        for i, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)   # [B,T,2]
            yb = yb.to(device, non_blocking=True)
            if args.perseq_norm:
                xb, yb = perseq_rms_norm(xb, yb)

            base = notch_tonal_multi(xb, args.fs, peaks=args.notch_peaks,
                                     q=args.notch_q, depth_db=args.notch_depth)

            jam = xb.permute(0,2,1)                 # [B,2,T]
            bas = base.permute(0,2,1)               # [B,2,T]
            inp = torch.cat([jam, bas], dim=1)      # [B,4,T]

            with torch.amp.autocast('cuda', enabled=args.amp):
                resid = model(inp).permute(0,2,1)   # [B,T,2]
                yhat  = base + resid                # residual scheme
                loss  = loss_fn(yhat, yb)

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
                sched_info["obj"].step(epoch-1 + i/float(max(1,steps_per_epoch)))
            step += 1

            if ema: ema.update(model)

        # ---- Validation ----
        with torch.no_grad():
            # keep same spectral weight during evaluation for consistent logging
            if args.spec_w_ramp_epochs > 0 and epoch > args.spec_w_ramp_epochs:
                loss_fn.set_spec_weight(args.spec_w_final)
            else:
                loss_fn.set_spec_weight(loss_fn.spec_w)

            if ema: ema.apply(model)
            model.eval()
            val_loss=0.0; snr_in_sum=0.0; snr_out_sum=0.0; evm_sum=0.0; n_batches=0
            # diagnostic accumulators (local)
            evm_ai_sum_local = 0.0
            evm_ai_batches_local = 0.0

            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                if args.perseq_norm:
                    xb, yb = perseq_rms_norm(xb, yb)

                base = notch_tonal_multi(xb, args.fs, peaks=args.notch_peaks, q=args.notch_q, depth_db=args.notch_depth)
                jam = xb.permute(0,2,1); bas = base.permute(0,2,1)
                inp = torch.cat([jam, bas], dim=1)
                resid = model(inp).permute(0,2,1)
                yhat  = base + resid

                val_loss += loss_fn(yhat, yb).item()
                si, so = snr_in_out_raw(xb, yb, yhat); snr_in_sum += si; snr_out_sum += so
                evm_sum += evm_rms_pct_raw(yb, yhat)
                n_batches += 1

                # optional aligned,in-band diagnostic EVM
                if args.diag_ai_evm != "off":
                    evm_ai = evm_aligned_inband_pct(yb, yhat, args.fs, args.inband, args.guard,
                                                    maxlag=max(8, args.align_maxlag),
                                                    frac_steps=max(3, args.align_frac_steps),
                                                    gain_align=True)
                    evm_ai_sum_local += float(evm_ai)
                    evm_ai_batches_local += 1.0
                    if args.diag_ai_evm == "batch" and master:
                        print(f"      ↳ (aligned,in-band) EVM {evm_ai:.2f}%")

        if ema: ema.restore(model)

        # reduce across ranks (main metrics)
        if world>1:
            t = torch.tensor([val_loss, snr_in_sum, snr_out_sum, evm_sum, n_batches], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            val_loss, snr_in_sum, snr_out_sum, evm_sum, n_batches = t.tolist()

        val_loss /= max(1,n_batches)
        snr_in  = snr_in_sum/max(1,n_batches)
        snr_out = snr_out_sum/max(1,n_batches)
        evm     = evm_sum/max(1,n_batches)

        if master:
            dt=time.time()-t0
            print(f"Epoch {epoch:03d} | val {val_loss:.6f} | SNR_in {snr_in:+.2f} dB → SNR_out {snr_out:+.2f} dB | EVM {evm:.2f}% | {dt:.1f}s")

        # ---- reduce + print epoch-avg aligned,in-band EVM if requested ----
        if args.diag_ai_evm != "off":
            if world > 1:
                t_ai = torch.tensor([evm_ai_sum_local, evm_ai_batches_local], device=device)
                dist.all_reduce(t_ai, op=dist.ReduceOp.SUM)
                evm_ai_sum_tot, evm_ai_batches_tot = t_ai.tolist()
            else:
                evm_ai_sum_tot, evm_ai_batches_tot = evm_ai_sum_local, evm_ai_batches_local
            if args.diag_ai_evm == "epoch" and master and evm_ai_batches_tot > 0:
                evm_ai_avg = evm_ai_sum_tot / evm_ai_batches_tot
                print(f"      ↳ (aligned,in-band) EVM (epoch avg) {evm_ai_avg:.2f}%")

        # ---- save best ----
        if master and (val_loss < best_val):
            best_val = val_loss
            ck = {
                "model": (ema.state_dict_for(model) if (ema is not None) else (model.module.state_dict() if hasattr(model,'module') else model.state_dict())),
                "args": vars(args),
                "val": val_loss,
                "epoch": epoch,
            }
            Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
            torch.save(ck, Path(args.ckpt_dir)/"best.pt")
            print("  ↳ saved best ->", Path(args.ckpt_dir)/"best.pt")

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
    # normalization + aligned head
    ap.add_argument("--perseq-norm", action="store_true", help="per-window RMS normalization using clean RMS")
    ap.add_argument("--align-maxlag", type=int, default=0, help="max samples for lag search (0 disables)")
    ap.add_argument("--align-w", type=float, default=0.0, help="extra L1 weight on lag+gain aligned head")
    ap.add_argument("--gain-align", action="store_true", help="estimate complex gain for aligned head")
    ap.add_argument("--align-frac-steps", type=int, default=0, help=">0 enables fractional-delay alignment candidates")
    # spectral weight ramp
    ap.add_argument("--spec-w-final", type=float, default=0.6, help="target spec weight after ramp")
    ap.add_argument("--spec-w-ramp-epochs", type=int, default=10, help="epochs to ramp spec_w (0 disables)")
    # top-K spectral peak emphasis
    ap.add_argument("--peak-w", type=float, default=0.0, help="extra weight for top-K spectral residuals")
    ap.add_argument("--peak-k", type=int, default=0, help="K bins per batch to emphasize (0 disables)")
    ap.add_argument("--peak-region", type=str, default="guard", choices=["guard","in","both"])
    ap.add_argument("--peak-ramp-epochs", type=int, default=0, help="epochs to ramp peak_w from 0→peak_w (0 disables)")
    # opt + scheduler
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scheduler", type=str, default="cawr", choices=["cosine","cawr","none"])
    ap.add_argument("--cawr-T0", type=int, default=5, help="epochs between first restart")
    ap.add_argument("--cawr-Tmult", type=int, default=2, help="restart period multiplier")
    ap.add_argument("--warmup-frac", type=float, default=0.06, help="fraction of steps for LR warmup")
    # diagnostics
    ap.add_argument("--diag-ai-evm", type=str, default="off", choices=["off","epoch","batch"],
                    help="Aligned,in-band EVM logging: off (default), once per epoch, or per-batch.")
    return ap

def main():
    args = build_argparser().parse_args()
    train(args)

if __name__ == "__main__":
    main()
