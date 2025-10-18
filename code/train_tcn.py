#!/usr/bin/env python3
"""
train_tcn.py — Causal TCN denoiser for baseband IQ with optional DSP prefilter.

Additions in this build
- Aligned & unaligned EVM metrics (timing + complex LS scale/phase)
- EMA weights (--ema, --ema-decay, --eval-use-ema, --save-ema)
- Cosine LR scheduler with optional warmup (--sched cosine, --warmup-epochs, --min-lr)
- CSV logs include both aligned & unaligned EVM + SNR_out(EVM_aligned)
"""

from __future__ import annotations
import os, time, math, argparse, csv, warnings
from pathlib import Path
from typing import Dict, Tuple, Union, Sequence, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

# Use TF32 on Ampere+ where available
torch.set_float32_matmul_precision("high")

warnings.filterwarnings("ignore", message="The pynvml package is deprecated")

# ---------------------------
# DDP helpers
# ---------------------------

def _ddp_env():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank       = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size, rank, local_rank

def ddp_init_if_needed(backend: str = "nccl"):
    world_size, rank, local_rank = _ddp_env()
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return world_size, rank, local_rank

def ddp_barrier():
    if dist.is_initialized():
        dist.barrier()

def ddp_all_reduce_(t: torch.Tensor, op=dist.ReduceOp.SUM):
    if dist.is_initialized():
        dist.all_reduce(t, op=op)
    return t

# ---------------------------
# IQ / complex utilities
# ---------------------------                                           # back to [B,T,2]

def _ensure_iq_last_contig(x: torch.Tensor) -> torch.Tensor:
    if x.ndim >= 3 and x.shape[-1] != 2 and x.shape[1] == 2:
        x = x.transpose(1, -1)
    return x.contiguous()

def complex_from_iq(x: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(x):
        return x
    x = _ensure_iq_last_contig(x).to(torch.float32)
    if x.shape[-1] != 2:
        raise RuntimeError(f"Expected I/Q in last dim=2, got shape {tuple(x.shape)}")
    return torch.complex(x[..., 0], x[..., 1])

def iq_from_complex(z: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(z):
        raise RuntimeError("Expected complex tensor")
    return torch.stack([z.real, z.imag], dim=-1)

# ---------------------------
# Dataset
# ---------------------------

def _canonicalize_bt2(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 2 and a.shape[1] == 2:
        a = a[None, ...]
    if not (a.ndim == 3 and a.shape[-1] == 2):
        raise ValueError(f"Expected [N,T,2] but got {a.shape}")
    return a.astype(np.float32, copy=False)

def _find_sets(npz: Dict[str, np.ndarray]):
    k = set(npz.keys())
    if {"Xtr","Ytr","Xva","Yva"}.issubset(k):
        Xtr, Ytr, Xva, Yva = npz["Xtr"], npz["Ytr"], npz["Xva"], npz["Yva"]
        mid = max(1, Xva.shape[0] // 2)
        return {"train": (Xtr, Ytr), "val": (Xva[:mid], Yva[:mid]), "test": (Xva[mid:], Yva[mid:])}
    if {"X","Y"}.issubset(k):
        X, Y = npz["X"], npz["Y"]
        N = X.shape[0]
        ntr = int(0.8 * N); nva = int(0.1 * N)
        return {"train": (X[:ntr], Y[:ntr]),
                "val":   (X[ntr:ntr+nva], Y[ntr:ntr+nva]),
                "test":  (X[ntr+nva:], Y[ntr+nva:])}
    raise ValueError(f"Could not infer dataset keys from: {sorted(k)}")

class IQWindows(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, W: int, H: int):
        self.X = _canonicalize_bt2(X)
        self.Y = _canonicalize_bt2(Y)
        if self.X.shape != self.Y.shape:
            raise ValueError("X and Y shapes must match")
        self.W = int(W); self.H = int(H)
        N, T, _ = self.X.shape
        self.index = []
        if T == self.W:
            self.index = [(i, 0) for i in range(N)]
        else:
            for i in range(N):
                for s in range(0, T - self.W + 1, self.H):
                    self.index.append((i, s))

    def __len__(self): return len(self.index)

    def __getitem__(self, i: int):
        n, s = self.index[i]; e = s + self.W
        x = self.X[n, s:e, :]
        y = self.Y[n, s:e, :]
        return torch.from_numpy(x), torch.from_numpy(y)

def load_npz_dataset(path: str, W: int, H: int, batch: int, workers: int,
                     *, world_size: int = 1, rank: int = 0):
    npz = np.load(path, allow_pickle=False)
    sets = _find_sets(npz)
    Xtr, Ytr = sets["train"]
    Xva, Yva = sets["val"]
    Xte, Yte = sets["test"]

    T_min = min(Xtr.shape[1], Xva.shape[1], Xte.shape[1])
    W_eff = min(int(W), int(T_min))
    if W_eff < W and rank == 0:
        print(f"[data] Requested W={W} exceeds dataset min T={T_min}. Clamping W -> {W_eff}.")

    ds_tr = IQWindows(Xtr, Ytr, W_eff, H)
    ds_va = IQWindows(Xva, Yva, W_eff, W_eff)  # non-overlapping windows

    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler_tr = DistributedSampler(ds_tr, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        sampler_va = DistributedSampler(ds_va, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    else:
        sampler_tr = None
        sampler_va = None

    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=(sampler_tr is None),
                       sampler=sampler_tr, num_workers=workers, pin_memory=True,
                       drop_last=True, persistent_workers=(workers > 0), prefetch_factor=4)
    dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False, sampler=sampler_va,
                       num_workers=workers, pin_memory=True, drop_last=False,
                       persistent_workers=(workers > 0), prefetch_factor=4)
    return ds_tr, ds_va, dl_tr, dl_va, W_eff

# ---------------------------
# Prefilter options
# ---------------------------

def stft_gate_prefilter(x: torch.Tensor, fs: float,
                        inband_hz: Union[float, Sequence[float]],
                        guard_hz: float = 0.0,
                        max_depth_out_db: float = 120.0) -> torch.Tensor:
    def _as_complex(x: torch.Tensor):
        if x.ndim == 3 and x.shape[-1] == 2:
            z = complex_from_iq(x)
            def fmt(yc): return iq_from_complex(yc)
            return z, fmt
        if torch.is_complex(x):
            def fmt(yc): return yc
            return x, fmt
        xr = x.to(torch.float32)
        def fmt(yc): return yc.real
        return xr, fmt

    x_c, fmt = _as_complex(x)
    B, T = x_c.shape
    X = torch.fft.fft(x_c, n=T, dim=1)
    freqs = torch.fft.fftfreq(T, d=1.0 / fs, device=x.device)

    if isinstance(inband_hz, (list, tuple)) and len(inband_hz) == 2:
        lo, hi = float(inband_hz[0]) - guard_hz, float(inband_hz[1]) + guard_hz
    else:
        BW = float(inband_hz)
        lo, hi = -(BW / 2 + guard_hz), (BW / 2 + guard_hz)

    passmask = (freqs >= lo) & (freqs <= hi)

    if max_depth_out_db >= 200:
        X[:, ~passmask] = 0
    else:
        att = 10.0 ** (-max_depth_out_db / 20.0)
        X[:, ~passmask] *= att

    x_f = torch.fft.ifft(X, n=T, dim=1)
    return fmt(x_f)

def notch_prefilter(x: torch.Tensor, fs: float,
                    inband_hz: float, guard_hz: float,
                    max_depth_in: float = 40.0, max_depth_out: float = 80.0,
                    q: float = 600.0) -> torch.Tensor:
    def _as_complex(x: torch.Tensor):
        if x.ndim == 3 and x.shape[-1] == 2:
            z = complex_from_iq(x)
            def fmt(yc): return iq_from_complex(yc)
            return z, fmt
        if torch.is_complex(x):
            def fmt(yc): return yc
            return x, fmt
        xr = x.to(torch.float32)
        def fmt(yc): return yc.real
        return xr, fmt

    x_c, fmt = _as_complex(x)
    B, T = x_c.shape
    X = torch.fft.rfft(x_c, n=T, dim=1)
    mag = X.abs()
    peak_bin = torch.argmax(mag[:, 1:], dim=1) + 1  # ignore DC
    freqs = torch.linspace(0.0, fs / 2.0, T // 2 + 1, device=x_c.device)

    peak_freq = freqs[peak_bin]
    cap_db = torch.where(
        peak_freq <= guard_hz,
        torch.where(peak_freq <= inband_hz,
                    torch.full_like(peak_freq, max_depth_in),
                    torch.full_like(peak_freq, max_depth_out)),
        torch.full_like(peak_freq, max_depth_out)
    )
    cap = 10.0 ** (-cap_db / 20.0)

    df = (fs / 2.0) / (T // 2)
    f0 = peak_freq.clamp_min(1.0)
    sigma_bins = (f0 / (q * df)).clamp_min(1.5)

    F = X.shape[1]
    idx = torch.arange(F, device=x_c.device)[None, :].repeat(B, 1)
    pb = peak_bin[:, None].to(torch.float32)
    sb = sigma_bins[:, None]
    gauss = torch.exp(-0.5 * ((idx - pb) / sb) ** 2)
    att = 1.0 - (1.0 - cap[:, None]) * gauss

    Xf = X * att
    x_f = torch.fft.irfft(Xf, n=T, dim=1)
    return fmt(x_f)

def apply_prefilter(x: torch.Tensor, fs: float, prefilter: str,
                    inband_hz: Union[float, Sequence[float]] = 0.0, guard_hz: float = 0.0,
                    max_depth_out_db: float = 120.0,
                    max_depth_in_db: float = 40.0) -> torch.Tensor:
    pf = (prefilter or "none").lower()
    if pf in {"", "none"}:
        return x
    if pf in {"stft_gate", "stft-gate", "fft_gate"}:
        return stft_gate_prefilter(x, fs, inband_hz, guard_hz, max_depth_out_db)
    if pf == "notch":
        return notch_prefilter(x, fs, inband_hz if isinstance(inband_hz, (int, float)) else float(inband_hz[-1]),
                               guard_hz, max_depth_in_db, max_depth_out_db)
    return x

# ---------------------------
# Model (Causal TCN)
# ---------------------------

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        padding = (kernel_size - 1) * dilation
        super().__init__(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.remove = padding
    def forward(self, x):
        y = super().forward(x)
        if self.remove > 0:
            y = y[..., :-self.remove]
        return y

class ResidualBlock(nn.Module):
    def __init__(self, ch, k, d, gn_groups=8, dropout=0.0):
        super().__init__()
        self.c1 = CausalConv1d(ch, ch, k, d)
        self.n1 = nn.GroupNorm(num_groups=min(gn_groups, ch), num_channels=ch)
        self.c2 = CausalConv1d(ch, ch, k, d)
        self.n2 = nn.GroupNorm(num_groups=min(gn_groups, ch), num_channels=ch)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        y = self.c1(x); y = F.relu(self.n1(y)); y = self.drop(y)
        y = self.c2(y); y = self.n2(y)
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
        x = x_btc.permute(0, 2, 1)
        h = self.inp(x)
        h = self.blocks(h)
        y = self.out(h)
        return y.permute(0, 2, 1)

# ---------------------------
# Loss & metrics + alignment
# ---------------------------

Number = Union[int, float]

def _to_hz(v: Number, fs: float) -> float:
    v = float(v)
    return v * (fs / 2.0) if 0.0 <= v <= 1.0 else v

def _canon_band(band: Union[Tuple[Number, Number], Number, dict], fs: float) -> Tuple[float, float]:
    if isinstance(band, (list, tuple)) and len(band) == 2:
        lo, hi = band
    elif isinstance(band, (int, float)):
        lo, hi = 0.0, band
    elif isinstance(band, dict):
        if "lo" in band and "hi" in band:
            lo, hi = band["lo"], band["hi"]
        elif "fc" in band and "bw" in band:
            fc, bw = float(band["fc"]), float(band["bw"])
            lo, hi = fc - 0.5 * bw, fc + 0.5 * bw
        elif "bw" in band:
            lo, hi = 0.0, float(band["bw"])
        else:
            raise ValueError(f"Unsupported band dict keys: {band.keys()}")
    else:
        raise TypeError(f"Unsupported band type: {type(band)}")

    lo_hz = _to_hz(lo, fs); hi_hz = _to_hz(hi, fs)
    nyq = fs / 2.0
    lo_hz = max(0.0, min(nyq, lo_hz))
    hi_hz = max(0.0, min(nyq, hi_hz))
    if hi_hz < lo_hz: lo_hz, hi_hz = hi_hz, lo_hz
    return lo_hz, hi_hz

def _canon_guard(guard: Union[Tuple[Number, Number], Number, dict, None],
                 in_hi_hz: float, fs: float) -> Tuple[float, float]:
    if guard is None:
        return in_hi_hz, in_hi_hz
    if isinstance(guard, (list, tuple)) and len(guard) == 2:
        lo, hi = guard
    elif isinstance(guard, (int, float)):
        lo, hi = in_hi_hz, guard
    elif isinstance(guard, dict):
        if "lo" in guard and "hi" in guard:
            lo, hi = guard["lo"], guard["hi"]
        elif "hi" in guard:
            lo, hi = in_hi_hz, guard["hi"]
        elif "bw" in guard:
            lo, hi = in_hi_hz, in_hi_hz + float(guard["bw"])
        else:
            raise ValueError(f"Unsupported guard dict keys: {guard.keys()}")
    else:
        raise TypeError(f"Unsupported guard type: {type(guard)}")

    lo_hz = _to_hz(lo, fs); hi_hz = _to_hz(hi, fs)
    nyq = fs / 2.0
    lo_hz = max(0.0, min(nyq, lo_hz))
    hi_hz = max(0.0, min(nyq, hi_hz))
    if hi_hz < lo_hz: lo_hz, hi_hz = hi_hz, lo_hz
    lo_hz = max(lo_hz, in_hi_hz)
    hi_hz = max(hi_hz, lo_hz)
    return lo_hz, hi_hz

def spectral_loss(y_true, y_pred, fs, inband, guard, w_in=1.0, w_guard=1.0, w_out=1.0):
    y_true = _ensure_iq_last_contig(y_true)
    y_pred = _ensure_iq_last_contig(y_pred)
    yt = y_true if torch.is_complex(y_true) else complex_from_iq(y_true)
    yp = y_pred if torch.is_complex(y_pred) else complex_from_iq(y_pred)
    assert yt.ndim == 2 and yp.ndim == 2, f"Expected (B,T) complex tensors, got {yt.shape}, {yp.shape}"
    B, T = yt.shape
    YT = torch.fft.fft(yt, n=T, dim=1)
    YP = torch.fft.fft(yp, n=T, dim=1)
    freqs = torch.fft.fftfreq(T, d=1.0 / fs).to(yt.device)
    pos = freqs >= 0
    freqs = freqs[pos]; YT = YT[:, pos]; YP = YP[:, pos]
    PT = (YT.abs() ** 2)
    PP = (YP.abs() ** 2)
    f_lo, f_hi = _canon_band(inband, fs)
    g_lo, g_hi = _canon_guard(guard, f_hi, fs)
    m_in    = (freqs >= f_lo) & (freqs <= f_hi)
    m_guard = (freqs >  f_hi) & (freqs <= g_lo)
    m_out   = (freqs >  g_hi)
    def _mean_mask(x, m):
        if not m.any(): return x.new_zeros(x.size(0))
        return x[:, m].mean(dim=1)
    loss_in    = _mean_mask((PP - PT).abs(), m_in)
    loss_guard = _mean_mask(PP, m_guard)
    loss_out   = _mean_mask(PP, m_out)
    return (w_in * loss_in + w_guard * loss_guard + w_out * loss_out).mean()

def first_diff_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    yt = y_true[:, 1:, :] - y_true[:, :-1, :]
    yp = y_pred[:, 1:, :] - y_pred[:, :-1, :]
    return F.l1_loss(yp, yt)

# ---------- Alignment helpers ----------

@torch.no_grad()
def _best_lag(y: torch.Tensor, yhat: torch.Tensor, maxlag: int = 64) -> Tuple[torch.Tensor, int]:
    """
    Crude per-batch timing alignment via cross-correlation (complex).
    y, yhat: (B,T) complex
    Returns yhat_shifted, chosen_lag (int for info only; if varying per-batch, returns 0)
    """
    B, T = y.shape
    Y = torch.fft.fft(y, n=T, dim=1)
    H = torch.fft.fft(yhat, n=T, dim=1)
    cc = torch.fft.ifft(Y * torch.conj(H), n=T, dim=1)  # circular correlation
    cc = torch.fft.fftshift(cc, dim=1)
    lags = torch.arange(-T//2, T//2, device=y.device)
    m = (lags >= -maxlag) & (lags <= maxlag)
    ccw = cc[:, m]
    lagsw = lags[m]
    peak_idx = torch.argmax(ccw.abs(), dim=1)
    best_lags = lagsw[peak_idx]  # (B,)
    # shift each batch separately along time dim
    yhat_s = torch.empty_like(yhat)
    for b, sft in enumerate(best_lags.tolist()):
        yhat_s[b] = torch.roll(yhat[b], shifts=int(sft), dims=0)
    same = torch.all(best_lags == best_lags[0])
    return yhat_s, int(best_lags[0].item()) if same else 0

def _ls_align(y: torch.Tensor, yhat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable complex LS align: alpha = <yhat,y> / <yhat,yhat>.
    y, yhat: (B,T) complex. Returns (aligned_yhat, alpha).
    """
    num = torch.sum(torch.conj(yhat) * y, dim=1)        # (B,)
    den = torch.sum(torch.conj(yhat) * yhat, dim=1)
    den = den.real.clamp_min(1e-12)                     # keep real, avoid div-by-zero
    alpha = num / den                                   # (B,)
    yhat_al = alpha[:, None] * yhat
    return yhat_al, alpha

@torch.no_grad()
def evm_unaligned(y_true, y_pred):
    yt = y_true if torch.is_complex(y_true) else complex_from_iq(_ensure_iq_last_contig(y_true))
    yp = y_pred if torch.is_complex(y_pred) else complex_from_iq(_ensure_iq_last_contig(y_pred))
    err_pow = (yp - yt).abs().pow(2).sum(dim=1)
    ref_pow = yt.abs().pow(2).sum(dim=1).clamp_min(1e-12)
    evm_rms = torch.sqrt(err_pow / ref_pow)
    evm_pct = 100.0 * evm_rms
    evm_db  = 20.0 * torch.log10(evm_rms.clamp_min(1e-12))
    return evm_pct, evm_db

@torch.no_grad()
def evm_aligned(y_true, y_pred, do_timing: bool = True, maxlag: int = 64):
    yt = y_true if torch.is_complex(y_true) else complex_from_iq(_ensure_iq_last_contig(y_true))
    yp = y_pred if torch.is_complex(y_pred) else complex_from_iq(_ensure_iq_last_contig(y_pred))
    if do_timing:
        yp, _ = _best_lag(yt, yp, maxlag=maxlag)
    yp_s, alpha = _ls_align(yt, yp)
    err_pow = (yp_s - yt).abs().pow(2).sum(dim=1)
    ref_pow = yt.abs().pow(2).sum(dim=1).clamp_min(1e-12)
    evm_rms = torch.sqrt(err_pow / ref_pow)
    evm_pct = 100.0 * evm_rms
    evm_db  = 20.0 * torch.log10(evm_rms.clamp_min(1e-12))
    return evm_pct, evm_db

@torch.no_grad()
def snr_db(y_true, y_pred, x_in, eps: float = 1e-12):
    yt = y_true if torch.is_complex(y_true) else complex_from_iq(_ensure_iq_last_contig(y_true))
    yp = y_pred if torch.is_complex(y_pred) else complex_from_iq(_ensure_iq_last_contig(y_pred))
    xx = x_in   if torch.is_complex(x_in)   else complex_from_iq(_ensure_iq_last_contig(x_in))
    s    = torch.sum(torch.abs(yt) ** 2, dim=1).clamp_min(eps)
    n_in = torch.sum(torch.abs(xx - yt) ** 2, dim=1).clamp_min(eps)
    n_out= torch.sum(torch.abs(yp - yt) ** 2, dim=1).clamp_min(eps)
    snr_in  = 10.0 * torch.log10((s / n_in).clamp_min(eps))
    snr_out = 10.0 * torch.log10((s / n_out).clamp_min(eps))
    return snr_in.mean(), snr_out.mean()

class CompositeLoss(nn.Module):
    def __init__(self, fs: float, inband_hz, guard_hz,
                 spec_weight: float, w_in: float, w_guard: float, w_out: float,
                 smooth_weight: float, evm_norm_weight: float):
        super().__init__()
        self.fs = fs; self.inband = inband_hz; self.guard = guard_hz
        self.spec_weight = spec_weight
        self.w_in = w_in; self.w_guard = w_guard; self.w_out = w_out
        self.smooth_weight = smooth_weight
        self.evm_norm_weight = evm_norm_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # complex views
        yt = complex_from_iq(_ensure_iq_last_contig(y_true))
        yp = complex_from_iq(_ensure_iq_last_contig(y_pred))
        # align (grad-enabled)
        yp_al, _ = _ls_align(yt, yp)
        y_pred_al = iq_from_complex(yp_al)

        l_time   = F.l1_loss(y_pred_al, y_true)
        l_spec   = spectral_loss(y_true, y_pred_al, self.fs, self.inband, self.guard,
                                self.w_in, self.w_guard, self.w_out)
        l_smooth = first_diff_loss(y_true, y_pred_al)

        # optional EVM-normalization term as a CONSTANT (no grads needed)
        with torch.no_grad():
            evm_pct, _ = evm_unaligned(y_true, y_pred_al)
            l_evmn = (evm_pct / 100.0).mean()

        return l_time + self.spec_weight * l_spec + self.smooth_weight * l_smooth + self.evm_norm_weight * l_evmn

# ---------------------------
# EMA utils
# ---------------------------

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}
        for name, p in self._named_params(model):
            self.shadow[name] = p.detach().clone()

    def _named_params(self, model: nn.Module):
        m = model.module if hasattr(model, "module") else model
        for name, p in m.named_parameters():
            if p.requires_grad:
                yield name, p

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in self._named_params(model):
            self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=(1.0 - self.decay))

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        m = model.module if hasattr(model, "module") else model
        for name, p in m.named_parameters():
            if p.requires_grad:
                self.backup[name] = p.data.detach().clone()
                p.data.copy_(self.shadow[name].to(p.device))

    @torch.no_grad()
    def restore(self, model: nn.Module):
        if not self.backup:
            return
        m = model.module if hasattr(model, "module") else model
        for name, p in m.named_parameters():
            if p.requires_grad and name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone().cpu() for k, v in self.shadow.items()}

# ---------------------------
# Scheduler
# ---------------------------

def build_scheduler(optimizer, sched: str, base_lr: float, epochs: int, warmup_epochs: int, min_lr: float):
    if sched == "none":
        return None
    if sched == "cosine":
        def lr_lambda(epoch: int):
            e = epoch
            if warmup_epochs > 0 and e < warmup_epochs:
                return max(1e-8, float(e + 1) / float(warmup_epochs))
            t = (e - warmup_epochs) / max(1, (epochs - warmup_epochs))
            cos_out = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, t))))
            min_scale = min_lr / max(1e-12, base_lr)
            return min_scale + (1.0 - min_scale) * cos_out
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    raise ValueError(f"Unknown scheduler: {sched}")

# ---------------------------
# Train / Eval
# ---------------------------

def make_model(in_ch: int, ch: int, k: int, n_blocks: int, device):
    return TCN(in_ch=in_ch, ch=ch, k=k, n_blocks=n_blocks, out_ch=2).to(device)

def train(args):
    world_size, rank, local_rank = ddp_init_if_needed(backend="gloo" if args.cpu else "nccl")
    use_ddp = world_size > 1

    if not args.cpu and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    if rank == 0:
        print(f"Using device: {device} | world_size={world_size} rank={rank} local_rank={local_rank}")

    # Data
    ds_tr, ds_va, dl_tr, dl_va, W_eff = load_npz_dataset(args.data, args.W, args.H, args.batch, args.workers,
                                                         world_size=world_size, rank=rank)

    in_ch = 2 if args.input_mode in ("raw", "dsp") else 4
    model = make_model(in_ch, args.width, args.kernel, args.blocks, device)

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=None if device.type == "cpu" else [local_rank],
            output_device=None if device.type == "cpu" else local_rank,
            find_unused_parameters=False
        )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    use_amp = (device.type == "cuda" and args.amp)
    scaler  = amp.GradScaler('cuda', enabled=use_amp)

    # EMA
    ema: Optional[EMA] = None
    if args.ema:
        ema = EMA(model, decay=args.ema_decay)

    # Scheduler
    scheduler = build_scheduler(opt, args.sched, args.lr, args.epochs, args.warmup_epochs, args.min_lr)

    loss_fn = CompositeLoss(
        fs=args.fs, inband_hz=args.inband_hz, guard_hz=args.guard_hz,
        spec_weight=args.spec_weight, w_in=args.spec_w_in, w_guard=args.spec_w_guard, w_out=args.spec_w_out,
        smooth_weight=args.smooth_weight, evm_norm_weight=args.evm_norm_weight
    )

    # CSV logging (rank 0 only)
    if rank == 0 and args.log_csv:
        logp = Path(args.log_csv); logp.parent.mkdir(parents=True, exist_ok=True)
        if not logp.exists():
            with logp.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["epoch","train_loss","val_loss",
                            "evm_unalign_pct","evm_unalign_db",
                            "evm_align_pct","evm_align_db",
                            "snr_in","snr_out","snr_out_evm_aligned","minutes","lr"])

    best_val = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        if use_ddp and hasattr(dl_tr.sampler, "set_epoch"):
            dl_tr.sampler.set_epoch(epoch)

        model.train()
        t0 = time.time()
        run_loss_sum = 0.0
        run_loss_cnt = 0

        for step, (x, y) in enumerate(dl_tr, start=1):
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)

            # optional DSP path
            if args.prefilter != "none" or args.input_mode != "raw":
                dsp_x = apply_prefilter(x, args.fs, args.prefilter, args.inband_hz, args.guard_hz,
                                        args.prefilter_max_depth_out_db, args.prefilter_max_depth_in_db)
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
            with amp.autocast('cuda', enabled=use_amp):
                core = model(xin)  # model predicts the correction
                # choose identity base:
                if args.input_mode == "raw":
                    base = x
                elif args.input_mode == "dsp":
                    base = dsp_x
                elif args.input_mode == "dualpath":
                    base = x  # identity to raw path
                else:
                    raise ValueError("invalid --input-mode")

                yhat = base + core
                loss = loss_fn(yhat, y)


            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            if ema is not None:
                ema.update(model)

            # stats
            loss_v = float(loss.detach().cpu())
            run_loss_sum += loss_v; run_loss_cnt += 1
            global_step += 1

            # progress logging (rank 0 only)
            if rank == 0 and (args.progress or args.log_every > 0):
                if args.log_every > 0 and (step % args.log_every == 0 or step == 1):
                    steps_total = len(dl_tr)
                    avg = run_loss_sum / max(1, run_loss_cnt)
                    elapsed = time.time() - t0
                    step_rate = step / max(1e-9, elapsed)
                    rem_steps = steps_total - step
                    eta_min = (rem_steps / max(1e-9, step_rate)) / 60.0
                    cur_lr = opt.param_groups[0]['lr']
                    print(f"Epoch {epoch:03d} [{step:5d}/{steps_total:5d}] "
                          f"loss {avg:.6f} | {step_rate:.1f} it/s | ETA {eta_min:.1f} min | lr {cur_lr:.2e}")

        # Step scheduler per-epoch
        if scheduler is not None:
            scheduler.step()

        # Evaluate periodically
        do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if do_eval:
            model.eval()

            # swap in EMA weights for eval if requested
            using_ema_now = False
            if ema is not None and args.eval_use_ema:
                ema.apply_shadow(model)
                using_ema_now = True

            tot_loss = torch.tensor([0.0], device=device); tot_n = torch.tensor([0.0], device=device)
            evm_u_pct_s = torch.tensor([0.0], device=device); evm_u_db_s = torch.tensor([0.0], device=device)
            evm_a_pct_s = torch.tensor([0.0], device=device); evm_a_db_s = torch.tensor([0.0], device=device)
            snr_in_s  = torch.tensor([0.0], device=device)
            snr_out_s = torch.tensor([0.0], device=device)
            cnt       = torch.tensor([0.0], device=device)
            snr_out_evm_al_s = torch.tensor([0.0], device=device)

            with torch.no_grad(), amp.autocast('cuda', enabled=use_amp):
                for x, y in dl_va:
                    x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                    if args.prefilter != "none" or args.input_mode != "raw":
                        dsp_x = apply_prefilter(x, args.fs, args.prefilter, args.inband_hz, args.guard_hz,
                                                args.prefilter_max_depth_out_db, args.prefilter_max_depth_in_db)
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
                    B = torch.tensor([float(x.size(0))], device=device)
                    tot_n    += B

                    # metrics
                    evm_u_pct, evm_u_db = evm_unaligned(y, yhat)
                    evm_a_pct, evm_a_db = evm_aligned(y, yhat, do_timing=args.align_eval, maxlag=64)
                    snr_in, snr_out = snr_db(y, yhat, x)

                    evm_u_pct_s += evm_u_pct.detach().sum()
                    evm_u_db_s  += evm_u_db.detach().sum()
                    evm_a_pct_s += evm_a_pct.detach().sum()
                    evm_a_db_s  += evm_a_db.detach().sum()
                    snr_in_s    += snr_in.detach()  * B
                    snr_out_s   += snr_out.detach() * B

                    snr_out_evm_al_s += (-20.0 * torch.log10((evm_a_pct / 100.0).clamp_min(1e-12))).detach().sum()
                    cnt       += B

            # restore raw weights if we swapped to EMA
            if using_ema_now and ema is not None:
                ema.restore(model)

            # reduce across ranks
            for t in (tot_loss, tot_n, evm_u_pct_s, evm_u_db_s, evm_a_pct_s, evm_a_db_s,
                      snr_in_s, snr_out_s, snr_out_evm_al_s, cnt):
                ddp_all_reduce_(t, op=dist.ReduceOp.SUM)

            if rank == 0:
                val_loss  = (tot_loss / torch.clamp_min(tot_n, 1.0)).item()
                evm_u_pct_m = (evm_u_pct_s / torch.clamp_min(cnt, 1.0)).item()
                evm_u_db_m  = (evm_u_db_s  / torch.clamp_min(cnt, 1.0)).item()
                evm_a_pct_m = (evm_a_pct_s / torch.clamp_min(cnt, 1.0)).item()
                evm_a_db_m  = (evm_a_db_s  / torch.clamp_min(cnt, 1.0)).item()
                snr_in_m  = (snr_in_s  / torch.clamp_min(cnt, 1.0)).item()
                snr_out_m = (snr_out_s / torch.clamp_min(cnt, 1.0)).item()
                snr_out_evm_al_m = (snr_out_evm_al_s / torch.clamp_min(cnt, 1.0)).item()

                dt_min = (time.time() - t0) / 60.0
                train_loss_avg = run_loss_sum / max(1, run_loss_cnt)
                warn = " ⚠" if abs(snr_out_m - snr_out_evm_al_m) > 0.5 else ""
                cur_lr = opt.param_groups[0]['lr']
                print(
                    f"Epoch {epoch:03d} | train {train_loss_avg:.6f} | val {val_loss:.6f} | "
                    f"EVM {evm_u_pct_m:.2f}% ({evm_u_db_m:.2f} dB) | "
                    f"EVM* {evm_a_pct_m:.2f}% ({evm_a_db_m:.2f} dB) | "
                    f"SNR_in {snr_in_m:.2f} → SNR_out {snr_out_m:.2f} "
                    f"[SNR_out(EVM*) {snr_out_evm_al_m:.2f}]{warn} | {dt_min:.2f} min | lr {cur_lr:.2e}"
                )

                if args.log_csv:
                    with open(args.log_csv, "a", newline="") as f:
                        w = csv.writer(f)
                        w.writerow([
                            epoch,
                            f"{train_loss_avg:.6f}", f"{val_loss:.6f}",
                            f"{evm_u_pct_m:.6f}", f"{evm_u_db_m:.6f}",
                            f"{evm_a_pct_m:.6f}", f"{evm_a_db_m:.6f}",
                            f"{snr_in_m:.6f}", f"{snr_out_m:.6f}", f"{snr_out_evm_al_m:.6f}",
                            f"{dt_min:.6f}", f"{cur_lr:.8f}"
                        ])

                # Save best
                if val_loss < best_val:
                    best_val = val_loss
                    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
                    raw_sd = (model.module.state_dict() if use_ddp else model.state_dict())
                    ckpt = {"model": raw_sd, "args": vars(args)}
                    torch.save(ckpt, str(outp))
                    print(f"  ↳ saved best (raw) -> {outp}")
                    if ema is not None and args.save_ema:
                        ckpt_ema = {"model": ema.state_dict(), "args": vars(args), "ema": True}
                        outp_ema = outp.with_name(outp.stem + "_ema.pt")
                        torch.save(ckpt_ema, str(outp_ema))
                        print(f"  ↳ saved best (EMA) -> {outp_ema}")

        # Periodic checkpoint even if not best
        if rank == 0 and args.save_every > 0 and (epoch % args.save_every == 0):
            outp = Path(args.out)
            ckpt_path = outp.with_name(outp.stem + f"_epoch{epoch}.pt")
            raw_sd = (model.module.state_dict() if use_ddp else model.state_dict())
            ckpt = {"model": raw_sd, "args": vars(args)}
            torch.save(ckpt, str(ckpt_path))
            print(f"  ↳ checkpoint (raw) -> {ckpt_path}")
            if ema is not None and args.save_ema:
                ckpt_ema = {"model": ema.state_dict(), "args": vars(args), "ema": True}
                ckpt_path_ema = outp.with_name(outp.stem + f"_epoch{epoch}_ema.pt")
                torch.save(ckpt_ema, str(ckpt_path_ema))
                print(f"  ↳ checkpoint (EMA) -> {ckpt_path_ema}")

    # Wrap up
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
    ap.add_argument("--data", type=str, required=True, help="Path to .npz (Xtr/Ytr/Xva/Yva) or (X/Y)")
    # Data & loader
    ap.add_argument("--fs", type=float, default=4.092e6, help="Sample rate (Hz)")
    ap.add_argument("--W", type=int, default=4092, help="Window length (samples)")
    ap.add_argument("--H", type=int, default=4092, help="Hop size for training windows (samples)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    # Model
    ap.add_argument("--width", type=int, default=160, help="TCN hidden channels")
    ap.add_argument("--blocks", type=int, default=10, help="Number of residual dilated blocks")
    ap.add_argument("--kernel", type=int, default=7, help="Kernel size")
    # Optim / schedule
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-6, help="Minimum LR at cosine floor")
    ap.add_argument("--sched", type=str, default="none", choices=["none","cosine"], help="LR schedule")
    ap.add_argument("--warmup-epochs", type=int, default=0, help="Linear warmup epochs before cosine")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision")
    ap.add_argument("--cpu", action="store_true", help="Force CPU")
    ap.add_argument("--seed", type=int, default=0)
    # EMA
    ap.add_argument("--ema", action="store_true", help="Track EMA of model weights")
    ap.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay (e.g., 0.999)")
    ap.add_argument("--eval-use-ema", action="store_true", help="Use EMA weights during validation")
    ap.add_argument("--save-ema", action="store_true", help="Also save EMA weights alongside raw")
    # IO
    ap.add_argument("--out", type=str, default="tcn_denoiser.pt", help="Where to save best checkpoint")
    ap.add_argument("--save-every", type=int, default=0, help="Save an extra checkpoint every N epochs (0=off)")
    ap.add_argument("--eval-every", type=int, default=1, help="Run validation every N epochs")
    ap.add_argument("--log-csv", type=str, default="", help="CSV file to append per-epoch logs")
    # Progress
    ap.add_argument("--progress", action="store_true", help="Print intra-epoch progress")
    ap.add_argument("--log-every", type=int, default=0, help="If >0, log a training step every N batches")
    # DSP / Inputs
    ap.add_argument("--input-mode", type=str, choices=["raw", "dsp", "dualpath"], default="raw",
                    help="Feed raw IQ, DSP IQ, or [raw||DSP] concatenation")
    ap.add_argument("--prefilter", type=str, default="none",
                    choices=["none", "stft_gate", "stft-gate", "fft_gate", "notch"])
    ap.add_argument("--inband-hz", type=float, default=2.046e6, help="In-band hi (Hz) or bandwidth (if scalar)")
    ap.add_argument("--guard-hz", type=float, default=2.3e6, help="Guard-band hi (Hz) if scalar; else tuple via code")
    ap.add_argument("--prefilter-max-depth-out-db", type=float, default=120.0)
    ap.add_argument("--prefilter-max-depth-in-db", type=float, default=40.0)
    # Loss weights
    ap.add_argument("--spec-weight", type=float, default=0.2, help="Weight of spectral loss term")
    ap.add_argument("--spec-w-in", type=float, default=1.0)
    ap.add_argument("--spec-w-guard", type=float, default=1.0)
    ap.add_argument("--spec-w-out", type=float, default=1.0)
    ap.add_argument("--smooth-weight", type=float, default=0.0, help="Weight of time-smoothness loss")
    ap.add_argument("--evm-norm-weight", type=float, default=0.2, help="Weight of mean(EVM%)/100")
    # Eval alignment
    ap.add_argument("--align-eval", action="store_true", help="Align timing + complex scale/phase for eval metrics")

    return ap

def main():
    ap = build_argparser()
    args = ap.parse_args()

    # alias compatible names → attributes used in code
    args.spec_w_in    = getattr(args, "spec_w_in")
    args.spec_w_guard = getattr(args, "spec_w_guard")
    args.spec_w_out   = getattr(args, "spec_w_out")

    # Repro
    if args.seed is not None and args.seed >= 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    train(args)

if __name__ == "__main__":
    main()
