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


def compute_curriculum_difficulty_from_xy(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Estimate effective SNR_in per window from X (jammed+noise) and Y (clean),
    then define difficulty = -SNR_in_dB (smaller = easier).
    X, Y: [N, T, 2] float arrays.
    """
    assert X.shape == Y.shape and X.ndim == 3 and X.shape[-1] == 2
    # Use float64 for accumulation
    x = X.astype(np.float64)
    y = Y.astype(np.float64)

    # Signal power: mean(|Y|^2) over time
    sig_pow = (y[..., 0]**2 + y[..., 1]**2).mean(axis=1)

    # Interference+noise power: mean(|X-Y|^2) over time
    e = x - y
    in_pow = (e[..., 0]**2 + e[..., 1]**2).mean(axis=1)

    eps = 1e-12
    in_pow = np.maximum(in_pow, eps)
    sig_pow = np.maximum(sig_pow, eps)

    snr_lin = sig_pow / in_pow
    snr_lin = np.maximum(snr_lin, eps)
    snr_db = 10.0 * np.log10(snr_lin)

    difficulty = -snr_db.astype(np.float32)
    return difficulty


class IQWindowsCurriculum(Dataset):
    """
    Curriculum wrapper around IQ-style windows:
    - 'difficulty' is a 1D array, same length as X, smaller = easier.
    - We sort indices by difficulty and, for each epoch, only sample
      from the easiest prefix.
    - Dataset length stays fixed at N so DistributedSampler works.
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray,
                 difficulty: np.ndarray, min_frac: float = 0.3):
        assert X.shape[0] == difficulty.shape[0], \
            f"Difficulty len {difficulty.shape[0]} != {X.shape[0]} samples"
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)
        self.N = self.X.shape[0]
        self.difficulty = difficulty.astype(np.float32)
        self.order = np.argsort(self.difficulty)  # easiest → hardest
        self.min_frac = float(min_frac)
        self.curr_frac = self.min_frac

    def set_curriculum_progress(self, frac: float):
        """Set fraction of data to sample from (in [min_frac, 1])."""
        frac = float(frac)
        frac = max(self.min_frac, min(1.0, frac))
        self.curr_frac = frac

    def _active_n(self) -> int:
        return max(1, int(round(self.curr_frac * self.N)))

    def __len__(self) -> int:
        # Keep length constant for DDP / sampler
        return self.N

    def __getitem__(self, i: int):
        # Map index into the easy prefix
        active_n = self._active_n()
        j = int(i % active_n)
        idx = int(self.order[j])
        x = torch.from_numpy(self.X[idx])
        y = torch.from_numpy(self.Y[idx])
        return x, y

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
            return x_, torch.view_as_real
        raise RuntimeError(f"Unsupported x shape {tuple(x_.shape)}")

    z, back = _as_complex(x)             # [B,T]
    B,T = z.shape
    X = torch.fft.fft(z, dim=1)          # [B,T]
    mag = X.abs()                        # [B,T]

    # ignore DC
    mag[:, 0] = 0.0

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
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        pad = (kernel_size - 1) * dilation
        super().__init__(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
    def forward(self, x):
        y = super().forward(x)
        k = self.kernel_size[0]
        d = self.dilation[0]
        cut = (k-1)*d
        if cut>0: y = y[...,:-cut]
        return y

class TCNBlock(nn.Module):
    def __init__(self, ch, k, d, dropout=0.0):
        super().__init__()
        self.conv1 = CausalConv1d(ch, ch, k, dilation=d)
        self.conv2 = CausalConv1d(ch, ch, k, dilation=d)
        self.drop = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(ch)
        self.bn2 = nn.BatchNorm1d(ch)
        self.act = nn.ReLU()
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act(y)
        y = self.drop(y)
        return x + y

class TCN(nn.Module):
    def __init__(self, in_ch, hid, blocks, k, dropout):
        super().__init__()
        self.in_proj = CausalConv1d(in_ch, hid, 1)
        layers=[]
        d=1
        for _ in range(blocks):
            layers.append(TCNBlock(hid, k, d, dropout))
            d*=2
        self.tcn = nn.Sequential(*layers)
        self.out_proj = CausalConv1d(hid, 2, 1)
    def forward(self, x):
        # x: [B,C,T]
        h = self.in_proj(x)
        h = self.tcn(h)
        y = self.out_proj(h)
        return y

def make_model(in_ch=4, hid=64, blocks=8, k=7, dropout=0.05) -> nn.Module:
    return TCN(in_ch, hid, blocks, k, dropout)

# ---------------- Losses ----------------
def bandlimit_inband(x_iq: torch.Tensor, fs: float, inband_hz: float, guard_hz: float) -> torch.Tensor:
    z = complex_from_iq(x_iq)        # [B,T]
    B,T = z.shape
    Z = torch.fft.fft(z, dim=1)     # [B,T]
    freqs = torch.fft.fftfreq(T, d=1.0/fs).to(z.device)  # [-fs/2,fs/2)
    Zs = torch.fft.fftshift(Z, dim=1); freqs = torch.fft.fftshift(freqs)
    BW = inband_hz
    m_in = (freqs >= -BW/2) & (freqs <= +BW/2)
    Zs = Zs * m_in
    Z = torch.fft.ifftshift(Zs, dim=1)
    zf = torch.fft.ifft(Z, dim=1)
    return iq_from_complex(zf)

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
        self.fs = fs
        self.inband_hz = inband_hz
        self.guard_hz = guard_hz
        self.spec_w = float(spec_w)
        self.w_in = float(w_in)
        self.w_guard = float(w_guard)
        self.w_out = float(w_out)
        self.smooth_w = float(smooth_w)
        self.time_w = float(time_w)
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
        W = 2.0*math.pi*torch.fft.fftfreq(T, d=1.0).to(z.device)
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
            for k in range(1, frac_steps+1):
                f = k / float(frac_steps+1)
                fracs.extend([+f, -f])

        best = None
        best_val = None

        for L in range(-maxlag, maxlag+1):
            for f in fracs:
                tau = float(L) + f
                if abs(tau) > 1e-6:
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

        BW = self.inband_hz
        m_in    = (freqs >= -BW/2) & (freqs <= +BW/2)
        m_guard = ((freqs > +BW/2) & (freqs <= +BW/2+self.guard_hz)) | ((freqs < -BW/2) & (freqs >= -BW/2-self.guard_hz))
        if self.peak_region == "guard":
            m = m_guard
        elif self.peak_region == "in":
            m = m_in
        else:
            m = m_guard | m_in

        # Residual
        R = (Yp - Yt).abs()  # [B,T]
        R = R[:, m]          # [B, M]
        if R.numel() == 0:
            return torch.tensor(0.0, device=yt.device)

        k = min(self.peak_k, R.shape[1])
        if k <= 0:
            return torch.tensor(0.0, device=yt.device)
        vals, _ = torch.topk(R, k, dim=1)   # [B,k]
        return vals.mean()

    def forward(self, y_true_iq: torch.Tensor, y_pred_iq: torch.Tensor) -> torch.Tensor:
        # Time loss
        time_loss = F.l1_loss(y_pred_iq, y_true_iq)

        # Smoothness
        smooth_loss = first_diff_loss(y_true_iq, y_pred_iq)

        # Alignment
        align_loss = torch.tensor(0.0, device=y_true_iq.device)
        if self.align_w > 0.0 and (self.align_maxlag > 0 or self.align_frac_steps > 0):
            yp_al = self._best_align(y_pred_iq, y_true_iq, self.align_maxlag, self.gain_align, self.align_frac_steps)
            align_loss = F.mse_loss(yp_al, y_true_iq)

        # Spectral loss
        spec_loss = spectral_loss(y_true_iq, y_pred_iq, self.fs, self.inband_hz, self.guard_hz,
                                  self.w_in, self.w_guard, self.w_out)

        peak_loss = torch.tensor(0.0, device=y_true_iq.device)
        if self.peak_w > 0.0 and self.peak_k > 0:
            peak_loss = self._peak_emphasis(y_true_iq, y_pred_iq)

        total = (self.time_w * time_loss
                 + self.smooth_w * smooth_loss
                 + self.align_w * align_loss
                 + self.spec_w * spec_loss
                 + self.peak_w * peak_loss)
        return total

# ---------------- Metrics ----------------
def snr_in_out_raw(x_iq: torch.Tensor, y_true_iq: torch.Tensor, y_pred_iq: torch.Tensor):
    # SNR_in: signal vs (x - y_true) (jammer+noise)
    yt = complex_from_iq(y_true_iq)
    xt = complex_from_iq(x_iq)
    yp = complex_from_iq(y_pred_iq)
    sig_pow = (yt.abs()**2).mean(dim=1).clamp_min(1e-12)
    in_res = xt - yt
    in_pow = (in_res.abs()**2).mean(dim=1).clamp_min(1e-12)
    out_res = yp - yt
    out_pow = (out_res.abs()**2).mean(dim=1).clamp_min(1e-12)
    snr_in = 10.0*torch.log10(sig_pow/in_pow).mean().item()
    snr_out = 10.0*torch.log10(sig_pow/out_pow).mean().item()
    return snr_in, snr_out

def evm_rms_pct_raw(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    yt = complex_from_iq(y_true); yp = complex_from_iq(y_pred)
    err = yt - yp
    evm = (err.abs()**2).mean().sqrt()
    ref = (yt.abs()**2).mean().sqrt().clamp_min(1e-12)
    return 100.0*(evm/ref).item()

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
            if p.requires_grad:
                self.shadow[n]=p.data.detach().clone()
        self.backup={}
    @torch.no_grad()
    def update(self, model):
        m = model.module if hasattr(model,"module") else model
        for n,p in m.named_parameters():
            if p.requires_grad:
                if n not in self.shadow:
                    self.shadow[n]=p.data.detach().clone()
                else:
                    new = (1.0-self.decay)*p.data + self.decay*self.shadow[n]
                    self.shadow[n] = new.detach().clone()
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

def make_cosine_sched(optimizer, T_max:int, eta_min:float=0.0):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

def make_cawr_sched(optimizer, T_0:int, T_mult:int, eta_min:float=0.0):
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)

# ---------------- Alignment helper ----------------
def align_best_frac_gain(y_pred_iq: torch.Tensor, y_true_iq: torch.Tensor,
                         fs: float, maxlag: int, frac_steps: int, gain_align: bool):
    loss_align = CompositeLoss(fs, fs, 0.0, align_maxlag=maxlag, align_w=0.0,
                               gain_align=gain_align, align_frac_steps=frac_steps)
    return loss_align._best_align(y_pred_iq, y_true_iq, maxlag, gain_align, frac_steps)

# ---------------- Data loaders ----------------
def make_loader(X: np.ndarray, Y: np.ndarray, batch: int, shuffle: bool,
                rank:int, world:int, num_workers:int=4):
    ds = IQWindows(X,Y)
    if world>1:
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=shuffle, drop_last=False)
        return DataLoader(ds, batch_size=batch, sampler=sampler, num_workers=num_workers, pin_memory=True)
    else:
        return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

def make_loader_from_ds(ds: Dataset, batch: int, shuffle: bool,
                        rank: int, world: int, num_workers: int = 4):
    if world>1:
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=shuffle, drop_last=False)
        return DataLoader(ds, batch_size=batch, sampler=sampler, num_workers=num_workers, pin_memory=True)
    else:
        return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

# ---------------- Train ----------------
def train(args):
    world, rank, local = ddp_init()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    master = is_master(rank)
    torch.backends.cudnn.benchmark = True

    data = load_npz(args.data)
    Xtr,Ytr = data["train"]; Xva,Yva = data["val"]

    # ---- build train loader (optionally with curriculum) ----
    if getattr(args, "curriculum", False):
        diff_tr = compute_curriculum_difficulty_from_xy(Xtr, Ytr)
        train_ds = IQWindowsCurriculum(Xtr, Ytr, diff_tr, min_frac=args.curriculum_min_frac)
    else:
        train_ds = IQWindows(Xtr, Ytr)

    train_loader = make_loader_from_ds(train_ds, args.batch, True, rank, world, args.workers)
    val_loader   = make_loader(Xva, Yva, max(1,args.batch//2), False, rank, world, args.workers)

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

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Scheduler setup
    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.epochs * steps_per_epoch
    warm_steps = int(args.warmup_frac * total_steps)
    if args.scheduler == "cosine":
        sched = make_cosine_sched(opt, T_max=max(1,total_steps-warm_steps), eta_min=0.0)
        sched_info = {"type":"cosine", "obj":sched, "warm_steps":warm_steps}
    elif args.scheduler == "cawr":
        sched = make_cawr_sched(opt, T_0=max(1,args.cawr_T0*steps_per_epoch), T_mult=args.cawr_Tmult, eta_min=0.0)
        sched_info = {"type":"cawr", "obj":sched, "warm_steps":warm_steps}
    else:
        sched_info = {"type":"none", "obj":None, "warm_steps":warm_steps}

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    ema = EMA(model, decay=args.ema) if args.ema>0 else None

    best_val = float("inf")
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(1, args.epochs+1):
        # ---- curriculum schedule: grow from min_frac → 1.0 over curriculum_epochs ----
        if getattr(args, "curriculum", False) and hasattr(train_loader.dataset, "set_curriculum_progress"):
            if args.curriculum_epochs > 1:
                t = min(1.0, (epoch - 1) / float(max(1, args.curriculum_epochs - 1)))
            else:
                t = 1.0
            frac = args.curriculum_min_frac + t * (1.0 - args.curriculum_min_frac)
            train_loader.dataset.set_curriculum_progress(frac)
            if master and hasattr(train_loader.dataset, "_active_n"):
                active_n = train_loader.dataset._active_n()
                print(f"[curriculum] epoch {epoch}: using easiest {active_n}/{Xtr.shape[0]} samples (frac={frac:.3f})")

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

            with torch.cuda.amp.autocast(enabled=args.amp):
                yhat = model(inp).permute(0,2,1)    # [B,T,2]
                loss = loss_fn(yb, yhat)

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

        # ---- validation ----
        model.eval()
        if ema: ema.apply(model)

        val_loss = 0.0
        snr_in_sum = 0.0
        snr_out_sum = 0.0
        evm_sum = 0.0
        n_batches = 0
        evm_ai_sum_local = 0.0
        evm_ai_batches_local = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                if args.perseq_norm:
                    xb, yb = perseq_rms_norm(xb, yb)

                base = notch_tonal_multi(xb, args.fs, peaks=args.notch_peaks,
                                         q=args.notch_q, depth_db=args.notch_depth)
                jam = xb.permute(0,2,1)
                bas = base.permute(0,2,1)
                inp = torch.cat([jam, bas], dim=1)

                with torch.cuda.amp.autocast(enabled=args.amp):
                    yhat = model(inp).permute(0,2,1)

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
            msg = (f"Epoch {epoch:03d} | val {val_loss:.6f} | "
                   f"SNR_in {snr_in:.2f} dB → SNR_out {snr_out:.2f} dB | "
                   f"EVM {evm:.2f}% | {dt:.1f}s")
            print(msg)

        # optional aligned,in-band EVM aggregated over validation
        if args.diag_ai_evm != "off":
            if world>1:
                t2 = torch.tensor([evm_ai_sum_local, evm_ai_batches_local], device=device)
                dist.all_reduce(t2, op=dist.ReduceOp.SUM)
                evm_ai_sum, evm_ai_batches = t2.tolist()
            else:
                evm_ai_sum, evm_ai_batches = evm_ai_sum_local, evm_ai_batches_local
            if evm_ai_batches > 0 and master:
                evm_ai_mean = evm_ai_sum / max(1.0, evm_ai_batches)
                print(f"      ↳ (aligned,in-band) EVM (epoch avg) {evm_ai_mean:.2f}%")

        # ---- save best ----
        if master and (val_loss < best_val):
            best_val = val_loss
            ck = {
                "model": (ema.state_dict_for(model) if (ema is not None) else
                          (model.module.state_dict() if hasattr(model,'module') else model.state_dict())),
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
    # curriculum learning
    ap.add_argument("--curriculum", action="store_true",
                    help="Enable easy→hard curriculum based on SNR_in estimated from X/Y.")
    ap.add_argument("--curriculum-epochs", type=int, default=30,
                    help="Epochs to grow from easiest subset to full train set.")
    ap.add_argument("--curriculum-min-frac", type=float, default=0.3,
                    help="Starting fraction of easiest samples (0<frac<=1).")
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
    # alignment
    ap.add_argument("--align-maxlag", type=int, default=0, help="max integer lag for align term (0 disables)")
    ap.add_argument("--align-w", type=float, default=0.0, help="weight of alignment term")
    ap.add_argument("--gain-align", action="store_true", help="include gain alignment in alignment term")
    ap.add_argument("--align-frac-steps", type=int, default=0, help="fractional delay steps (0 disables)")
    # peak emphasis
    ap.add_argument("--peak-w", type=float, default=0.0, help="weight for peak emphasis term (0 disables)")
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
