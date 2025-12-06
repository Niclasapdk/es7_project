#!/usr/bin/env python3
"""
Notch + Residual TCN jammer-denoiser training script

Features:
- DDP (torchrun) support
- AMP 2.0 (torch.amp.autocast / GradScaler('cuda', ...))
- Multi-tone soft notch prefilter
- Composite loss: time-domain, spectral, smoothness, optional alignment & peak emphasis
- Optional per-window RMS normalization
- Optional curriculum learning (easy → hard based on SNR_in)
- Cosine / CAWR schedulers with warmup
- EMA weights
- Validation SNR_in/out and EVM%, plus optional aligned in-band EVM diagnostics
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


# ---------------- I/Q helpers ----------------

def ensure_iq(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is [B, T, 2] float IQ."""
    if x.ndim != 3 or x.shape[-1] != 2:
        raise RuntimeError(f"Expected [B, T, 2] IQ tensor, got shape {tuple(x.shape)}")
    return x


def complex_from_iq(x: torch.Tensor) -> torch.Tensor:
    """Convert [B, T, 2] → complex [B, T]."""
    x = ensure_iq(x)
    return torch.complex(x[..., 0], x[..., 1])


def iq_from_complex(z: torch.Tensor) -> torch.Tensor:
    """Convert complex [B, T] → [B, T, 2]."""
    return torch.stack([z.real, z.imag], dim=-1)


# ---------------- optional per-sequence normalization ----------------

def perseq_rms_norm(x_iq: torch.Tensor, y_iq: torch.Tensor, eps: float = 1e-8):
    """
    Normalize both input and target by target RMS per sequence.
    x_iq, y_iq: [B, T, 2]
    """
    y_iq = ensure_iq(y_iq)
    yr = y_iq[..., 0]
    yi = y_iq[..., 1]
    rms = torch.sqrt((yr ** 2 + yi ** 2).mean(dim=1, keepdim=True).clamp_min(eps))  # [B, 1]
    rms = rms.unsqueeze(-1)  # [B, 1, 1]
    return x_iq / rms, y_iq / rms


# ---------------- Dataset ----------------

class IQWindows(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        assert X.shape == Y.shape and X.ndim == 3 and X.shape[-1] == 2, \
            f"Bad shapes {X.shape} vs {Y.shape}"
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i: int):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])


def compute_curriculum_difficulty_from_xy(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Estimate effective SNR_in per window from X (jammed+noise) and Y (clean),
    then define difficulty = -SNR_in_dB (smaller = easier).
    X, Y: [N, T, 2] float arrays.
    """
    assert X.shape == Y.shape and X.ndim == 3 and X.shape[-1] == 2
    x = X.astype(np.float64)
    y = Y.astype(np.float64)

    # Signal power: mean(|Y|^2) over time
    sig_pow = (y[..., 0] ** 2 + y[..., 1] ** 2).mean(axis=1)

    # Interference+noise power: mean(|X-Y|^2) over time
    e = x - y
    in_pow = (e[..., 0] ** 2 + e[..., 1] ** 2).mean(axis=1)

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


# ---------------- NPZ loader ----------------

def load_npz(path: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    npz = np.load(path)
    k = set(npz.keys())

    if {"Xtr", "Ytr", "Xva", "Yva"}.issubset(k):
        Xtr = npz["Xtr"]
        Ytr = npz["Ytr"]
        Xva = npz["Xva"]
        Yva = npz["Yva"]
    elif {"Xtr", "Ytr"}.issubset(k):
        Xtr = npz["Xtr"]
        Ytr = npz["Ytr"]
        if {"Xva", "Yva"}.issubset(k):
            Xva = npz["Xva"]
            Yva = npz["Yva"]
        else:
            # simple 90/10 split
            N = Xtr.shape[0]
            n_va = max(1, int(0.1 * N))
            Xva, Yva = Xtr[-n_va:], Ytr[-n_va:]
            Xtr, Ytr = Xtr[:-n_va], Ytr[:-n_va]
    else:
        # fallback: assume "X","Y"
        X = npz["X"]
        Y = npz["Y"]
        N = X.shape[0]
        n_va = max(1, int(0.1 * N))
        Xtr, Ytr = X[:-n_va], Y[:-n_va]
        Xva, Yva = X[-n_va:], Y[-n_va:]

    return {"train": (Xtr, Ytr), "val": (Xva, Yva)}


# ---------------- notch helper ----------------

def _as_complex(x: torch.Tensor) -> torch.Tensor:
    """Interpret a real tensor as complex if needed."""
    if torch.is_complex(x):
        return x
    if x.ndim == 1:
        return torch.complex(x, torch.zeros_like(x))
    if x.ndim == 2:
        # [T,2] or [B,T]?
        if x.shape[-1] == 2:
            return torch.complex(x[..., 0], x[..., 1])
    raise RuntimeError(f"Cannot interpret tensor of shape {x.shape} as complex.")


def notch_tonal_multi(x_iq: torch.Tensor, fs: float,
                      peaks: int = 2, q: float = 800.0, depth_db: float = 90.0) -> torch.Tensor:
    """
    Very simple frequency-domain "soft" multi-tone notch.
    - x_iq: [B, T, 2]
    - fs: sampling rate
    - peaks: number of tones to notch (uses average spectrum over batch)
    - q: quality factor (approx controls width)
    - depth_db: attenuation at tone center (in dB)
    """
    x_iq = ensure_iq(x_iq)
    z = complex_from_iq(x_iq)              # [B, T]
    B, T = z.shape
    X = torch.fft.fft(z, dim=1)           # [B, T]
    freqs = torch.fft.fftfreq(T, d=1.0 / fs).to(X.device)

    # Use average magnitude spectrum to locate main peaks (ignore DC).
    mag = X.abs().mean(dim=0)             # [T]
    # ignore (very near) DC when searching for tonal peaks
    dc_mask = (freqs.abs() < (fs / (10 * T)))  # ~1 bin around DC
    mag_for_peaks = mag.clone()
    mag_for_peaks[dc_mask] = 0.0

    k_peaks = min(peaks, T - 1)
    if k_peaks <= 0:
        return x_iq

    peak_indices = torch.topk(mag_for_peaks, k=k_peaks, dim=-1).indices

    # Build soft notch mask in frequency domain
    depth_lin = 10.0 ** (-depth_db / 20.0)  # amplitude ratio
    notch_mask = torch.ones_like(X)

    for idx in peak_indices:
        f0 = freqs[idx]
        # crude bandwidth from "Q" definition: bw ≈ f0 / q (but ensure non-zero)
        bw = (torch.abs(f0) / max(q, 1.0)).clamp(min=fs / T)
        # Gaussian-ish shape around f0
        w = torch.exp(-0.5 * ((freqs - f0) / bw) ** 2)
        # convert to attenuation factor between depth_lin (at center) and 1.0 (far away)
        att = depth_lin + (1.0 - depth_lin) * (1.0 - w)
        notch_mask = notch_mask * att[None, :]

    Y = X * notch_mask
    y = torch.fft.ifft(Y, dim=1)
    return iq_from_complex(y)


# ---------------- short helpers for bandlimits / losses ----------------

def bandlimit_inband(x_iq: torch.Tensor, fs: float, inband: float, guard: float) -> torch.Tensor:
    """
    Return in-band part (optionally with DC-guard removed) of x_iq.
    """
    x_iq = ensure_iq(x_iq)
    z = complex_from_iq(x_iq)            # [B, T]
    B, T = z.shape
    X = torch.fft.fft(z, dim=1)
    freqs = torch.fft.fftfreq(T, d=1.0 / fs).to(X.device)

    f_in = inband / 2.0
    mask = (freqs >= -f_in) & (freqs <= f_in)

    if guard > 0.0:
        g = guard
        mask_guard = (freqs >= -g) & (freqs <= g)
        mask = mask & ~mask_guard

    Xb = torch.where(mask[None, :], X, torch.zeros_like(X))
    zb = torch.fft.ifft(Xb, dim=1)
    return iq_from_complex(zb)


def spectral_loss(y_true_iq: torch.Tensor,
                  y_pred_iq: torch.Tensor,
                  fs: float,
                  inband: float,
                  guard: float,
                  w_in: float = 1.0,
                  w_guard: float = 1.0,
                  w_out: float = 1.0) -> torch.Tensor:
    """
    L1 loss on magnitude spectra, with weights on in-band / guard / out-of-band.
    """
    y_true_iq = ensure_iq(y_true_iq)
    y_pred_iq = ensure_iq(y_pred_iq)
    yt = complex_from_iq(y_true_iq)
    yp = complex_from_iq(y_pred_iq)
    B, T = yt.shape

    Yt = torch.fft.fft(yt, dim=1)
    Yp = torch.fft.fft(yp, dim=1)
    freqs = torch.fft.fftfreq(T, d=1.0 / fs).to(yt.device)

    mag_t = Yt.abs()
    mag_p = Yp.abs()
    diff = (mag_p - mag_t).abs()         # [B, T]

    f_in = inband / 2.0
    mask_in = (freqs >= -f_in) & (freqs <= f_in)
    if guard > 0.0:
        mask_guard = (freqs >= -guard) & (freqs <= guard)
        mask_inband = mask_in & ~mask_guard
    else:
        mask_guard = torch.zeros_like(mask_in, dtype=torch.bool)
        mask_inband = mask_in

    mask_out = ~mask_in

    loss = 0.0
    if w_in > 0.0 and mask_inband.any():
        loss = loss + w_in * diff[:, mask_inband].mean()
    if w_guard > 0.0 and mask_guard.any():
        loss = loss + w_guard * diff[:, mask_guard].mean()
    if w_out > 0.0 and mask_out.any():
        loss = loss + w_out * diff[:, mask_out].mean()

    return loss


def first_diff_loss(y_true_iq: torch.Tensor,
                    y_pred_iq: torch.Tensor) -> torch.Tensor:
    """
    First-difference (discrete derivative) L1 loss between true and predicted IQ.
    """
    y_true_iq = ensure_iq(y_true_iq)
    y_pred_iq = ensure_iq(y_pred_iq)
    dt_true = y_true_iq[:, 1:, :] - y_true_iq[:, :-1, :]
    dt_pred = y_pred_iq[:, 1:, :] - y_pred_iq[:, :-1, :]
    return (dt_pred - dt_true).abs().mean()


# ---------------- Model: Residual TCN ----------------

class CausalConv1d(nn.Conv1d):
    """1-D causal conv implemented via left padding and trimming."""
    def __init__(self, in_ch: int, out_ch: int, k: int, dilation: int = 1):
        pad = (k - 1) * dilation
        super().__init__(in_ch, out_ch, kernel_size=k, padding=pad, dilation=dilation)
        self._pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        if self._pad > 0:
            return out[..., :-self._pad]
        return out


class TCNBlock(nn.Module):
    def __init__(self, ch: int, k: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(ch, ch, k, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(ch)
        self.act1 = nn.GELU()
        self.conv2 = CausalConv1d(ch, ch, k, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(ch)
        self.act2 = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act1(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act2(y)
        return x + y


class ResidualTCN(nn.Module):
    """
    Simple residual TCN:
      input: [B, C_in, T]
      output: [B, 2, T] (IQ residual)
    """
    def __init__(self, in_ch: int, hid: int, blocks: int, k: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Conv1d(in_ch, hid, kernel_size=1)
        layers = []
        dilation = 1
        for _ in range(blocks):
            layers.append(TCNBlock(hid, k, dilation=dilation, dropout=dropout))
            dilation *= 2
        self.tcn = nn.Sequential(*layers)
        self.out_proj = nn.Conv1d(hid, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.tcn(h)
        out = self.out_proj(h)
        return out


# ---------------- Alignment helpers & EVM/SNR metrics ----------------

@torch.no_grad()
def align_best_frac_gain(y_pred_iq: torch.Tensor,
                         y_true_iq: torch.Tensor,
                         fs: float,
                         maxlag: int,
                         frac_steps: int,
                         gain_align: bool) -> torch.Tensor:
    """
    Simple integer-lag alignment + optional complex gain per sequence.
    frac_steps is currently ignored, but kept for API compatibility.
    """
    del fs  # unused in this simplified version
    del frac_steps

    y_pred_iq = ensure_iq(y_pred_iq)
    y_true_iq = ensure_iq(y_true_iq)
    zp = complex_from_iq(y_pred_iq)
    zt = complex_from_iq(y_true_iq)
    B, T = zp.shape
    device = zp.device

    if maxlag <= 0:
        return y_pred_iq

    lags = range(-maxlag, maxlag + 1)
    eps = 1e-12

    best_err = None
    best_aligned = None

    for lag in lags:
        if lag >= 0:
            zp_seg = zp[:, lag:]
            zt_seg = zt[:, :T - lag]
        else:
            zp_seg = zp[:, :T + lag]
            zt_seg = zt[:, -lag:]

        if gain_align:
            num = (zp_seg.conj() * zt_seg).sum(dim=1)
            den = (zp_seg.conj() * zp_seg).sum(dim=1).clamp_min(eps)
            g = num / den  # [B]
            zp_aligned_seg = g[:, None] * zp_seg
        else:
            zp_aligned_seg = zp_seg

        # pad back to length T
        zp_full = torch.zeros_like(zp)
        if lag >= 0:
            zp_full[:, :T - lag] = zp_aligned_seg
        else:
            zp_full[:, -lag:] = zp_aligned_seg

        err = ((zp_full - zt).abs() ** 2).mean(dim=1)  # [B]

        if best_err is None:
            best_err = err
            best_aligned = zp_full
        else:
            mask = err < best_err
            best_err = torch.minimum(best_err, err)
            best_aligned = torch.where(mask[:, None], zp_full, best_aligned)

    return iq_from_complex(best_aligned)


@torch.no_grad()
def snr_in_out_raw(x_iq: torch.Tensor,
                   y_true_iq: torch.Tensor,
                   y_pred_iq: torch.Tensor,
                   eps: float = 1e-12):
    """
    Compute SNR_in and SNR_out in dB, using mean(|y|^2) / mean(|err|^2).
    """
    x = complex_from_iq(x_iq)
    y = complex_from_iq(y_true_iq)
    yhat = complex_from_iq(y_pred_iq)

    sig_pow = (y.abs() ** 2).mean(dim=1).clamp_min(eps)
    in_err_pow = (x - y).abs().pow(2).mean(dim=1).clamp_min(eps)
    out_err_pow = (yhat - y).abs().pow(2).mean(dim=1).clamp_min(eps)

    snr_in = 10.0 * torch.log10(sig_pow / in_err_pow).mean().item()
    snr_out = 10.0 * torch.log10(sig_pow / out_err_pow).mean().item()
    return snr_in, snr_out


@torch.no_grad()
def evm_rms_pct_raw(y_true_iq: torch.Tensor,
                    y_pred_iq: torch.Tensor,
                    eps: float = 1e-12) -> float:
    """
    Root-mean-square EVM (%) ignoring bandlimits / alignment.
    """
    yt = complex_from_iq(y_true_iq)
    yp = complex_from_iq(y_pred_iq)

    err_pow = (yp - yt).abs().pow(2).mean(dim=1).clamp_min(eps)
    ref_pow = yt.abs().pow(2).mean(dim=1).clamp_min(eps)
    evm = torch.sqrt(err_pow / ref_pow).mean()
    return 100.0 * evm.item()


@torch.no_grad()
def evm_aligned_inband_pct(y_true: torch.Tensor,
                           y_pred: torch.Tensor,
                           fs: float,
                           inband: float,
                           guard: float,
                           maxlag: int = 12,
                           frac_steps: int = 5,
                           gain_align: bool = True) -> float:
    """
    RMS EVM (%) after:
      1) best integer-lag + gain alignment
      2) bandlimiting to inband (with guard)
    """
    y_pred_al = align_best_frac_gain(y_pred, y_true, fs, maxlag, frac_steps, gain_align)
    yt_in = bandlimit_inband(y_true, fs, inband, guard)
    yp_in = bandlimit_inband(y_pred_al, fs, inband, guard)
    yt = complex_from_iq(yt_in)
    yp = complex_from_iq(yp_in)

    err_pow = ((yp - yt).abs() ** 2).mean(dim=1).clamp_min(1e-12)
    ref_pow = (yt.abs() ** 2).mean(dim=1).clamp_min(1e-12)
    return 100.0 * torch.sqrt(err_pow / ref_pow).mean().item()


# ---------------- Composite loss ----------------

class CompositeLoss(nn.Module):
    def __init__(self,
                 fs: float,
                 inband: float,
                 guard: float,
                 spec_w: float = 0.3,
                 w_in: float = 1.0,
                 w_guard: float = 1.0,
                 w_out: float = 1.0,
                 smooth_w: float = 0.05,
                 time_w: float = 1.0,
                 align_maxlag: int = 0,
                 align_w: float = 0.0,
                 gain_align: bool = False,
                 align_frac_steps: int = 0,
                 peak_w: float = 0.0,
                 peak_k: int = 0,
                 peak_region: str = "guard"):
        super().__init__()
        self.fs = float(fs)
        self.inband = float(inband)
        self.guard = float(guard)
        self.spec_w = float(spec_w)
        self.w_in = float(w_in)
        self.w_guard = float(w_guard)
        self.w_out = float(w_out)
        self.smooth_w = float(smooth_w)
        self.time_w = float(time_w)
        self.align_maxlag = int(align_maxlag)
        self.align_w = float(align_w)
        self.gain_align = bool(gain_align)
        self.align_frac_steps = int(align_frac_steps)
        self.peak_w = float(peak_w)
        self.peak_k = int(peak_k)
        self.peak_region = str(peak_region)

    def set_spec_weight(self, w: float):
        self.spec_w = float(w)

    def _peak_emphasis(self, y_true_iq: torch.Tensor, y_pred_iq: torch.Tensor) -> torch.Tensor:
        """
        Emphasize top-K magnitude residuals in chosen frequency region.
        """
        y_true_iq = ensure_iq(y_true_iq)
        y_pred_iq = ensure_iq(y_pred_iq)
        yt = complex_from_iq(y_true_iq)
        yp = complex_from_iq(y_pred_iq)
        B, T = yt.shape

        Yt = torch.fft.fft(yt, dim=1)
        Yp = torch.fft.fft(yp, dim=1)
        freqs = torch.fft.fftfreq(T, d=1.0 / self.fs).to(yt.device)

        resid = (Yp - Yt).abs()  # [B, T]

        f_in = self.inband / 2.0
        mask_in = (freqs >= -f_in) & (freqs <= f_in)
        if self.guard > 0.0:
            mask_guard = (freqs >= -self.guard) & (freqs <= self.guard)
            mask_in = mask_in & ~mask_guard
        else:
            mask_guard = torch.zeros_like(mask_in, dtype=torch.bool)

        if self.peak_region == "guard":
            region = mask_guard
        elif self.peak_region == "in":
            region = mask_in
        else:
            region = mask_guard | mask_in

        if not region.any():
            return torch.tensor(0.0, device=yt.device, dtype=yt.real.dtype)

        r = resid[:, region]  # [B, Kf]
        vals = r.reshape(-1)
        k = min(self.peak_k, vals.numel())
        if k <= 0:
            return torch.tensor(0.0, device=yt.device, dtype=yt.real.dtype)
        topk = torch.topk(vals, k=k, dim=-1).values
        return topk.mean()

    def forward(self, y_pred_iq: torch.Tensor, y_true_iq: torch.Tensor) -> torch.Tensor:
        """
        Composite loss:
          - time-domain L1
          - smoothness in time (first differences)
          - spectral magnitude loss with region weights, optional peak emphasis
          - optional alignment-based term
        """
        y_pred_iq = ensure_iq(y_pred_iq)
        y_true_iq = ensure_iq(y_true_iq)

        loss = 0.0

        if self.time_w > 0.0:
            loss_time = (y_pred_iq - y_true_iq).abs().mean()
            loss = loss + self.time_w * loss_time

        if self.smooth_w > 0.0:
            loss_smooth = first_diff_loss(y_true_iq, y_pred_iq)
            loss = loss + self.smooth_w * loss_smooth

        if self.spec_w > 0.0:
            loss_spec = spectral_loss(
                y_true_iq, y_pred_iq,
                self.fs, self.inband, self.guard,
                self.w_in, self.w_guard, self.w_out
            )
            if self.peak_w > 0.0 and self.peak_k > 0:
                loss_spec = loss_spec + self.peak_w * self._peak_emphasis(y_true_iq, y_pred_iq)
            loss = loss + self.spec_w * loss_spec

        if self.align_w > 0.0 and self.align_maxlag > 0:
            y_pred_al = align_best_frac_gain(
                y_pred_iq, y_true_iq,
                fs=self.fs,
                maxlag=self.align_maxlag,
                frac_steps=self.align_frac_steps,
                gain_align=self.gain_align,
            )
            loss_align = (y_pred_al - y_true_iq).abs().mean()
            loss = loss + self.align_w * loss_align

        return loss


# ---------------- EMA ----------------

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        for n, p in (model.module if hasattr(model, "module") else model).named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.detach().clone()
        self.backup = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        m = model.module if hasattr(model, "module") else model
        for n, p in m.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply(self, model: nn.Module):
        self.backup = {}
        m = model.module if hasattr(model, "module") else model
        for n, p in m.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.data.detach().clone()
                p.data.copy_(self.shadow[n].to(p.device))

    @torch.no_grad()
    def restore(self, model: nn.Module):
        if not self.backup:
            return
        m = model.module if hasattr(model, "module") else model
        for n, p in m.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}

    def state_dict_for(self, model: nn.Module):
        return {k: v.cpu() for k, v in self.shadow.items()}


# ---------------- DDP utils ----------------

def ddp_init():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        world = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local = int(os.environ.get("LOCAL_RANK", "0"))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local)
        return world, rank, local
    return 1, 0, 0


def is_master(rank: int) -> bool:
    return rank == 0


# ---------------- Scheduler helpers ----------------

def linear_warmup(step: int, warm_steps: int) -> float:
    return min(1.0, (step + 1) / max(1, warm_steps))


def configure_scheduler(opt, args, steps_per_epoch: int, total_steps: int):
    if args.scheduler == "cosine":
        warm_steps = int(args.warmup_frac * total_steps)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, total_steps - warm_steps)
        )
        return {"type": "cosine", "obj": sched, "warm_steps": warm_steps}
    elif args.scheduler == "cawr":
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=max(1, args.cawr_T0), T_mult=max(1, args.cawr_Tmult)
        )
        warm_steps = int(args.warmup_frac * steps_per_epoch * args.cawr_T0) if args.warmup_frac > 0 else 0
        return {"type": "cawr", "obj": sched, "warm_steps": warm_steps}
    else:
        return {"type": "none", "obj": None, "warm_steps": 0}


# ---------------- Data loaders & model factory ----------------

def make_model(in_ch: int, hid: int, blocks: int, k: int, dropout: float) -> nn.Module:
    return ResidualTCN(in_ch=in_ch, hid=hid, blocks=blocks, k=k, dropout=dropout)


def make_loader(X: np.ndarray, Y: np.ndarray, batch: int, shuffle: bool,
                rank: int, world: int, num_workers: int = 4):
    ds = IQWindows(X, Y)
    if world > 1:
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank,
                                     shuffle=shuffle, drop_last=False)
        return DataLoader(ds, batch_size=batch, sampler=sampler,
                          num_workers=num_workers, pin_memory=True)
    else:
        return DataLoader(ds, batch_size=batch, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)


def make_loader_from_ds(ds: Dataset, batch: int, shuffle: bool,
                        rank: int, world: int, num_workers: int = 4):
    if world > 1:
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank,
                                     shuffle=shuffle, drop_last=False)
        return DataLoader(ds, batch_size=batch, sampler=sampler,
                          num_workers=num_workers, pin_memory=True)
    else:
        return DataLoader(ds, batch_size=batch, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)


# ---------------- Train ----------------

def train(args):
    world, rank, local = ddp_init()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    master = is_master(rank)
    torch.backends.cudnn.benchmark = True

    data = load_npz(args.data)
    Xtr, Ytr = data["train"]
    Xva, Yva = data["val"]

    # ---- build train loader (optionally with curriculum) ----
    if getattr(args, "curriculum", False):
        diff_tr = compute_curriculum_difficulty_from_xy(Xtr, Ytr)
        train_ds = IQWindowsCurriculum(Xtr, Ytr, diff_tr, min_frac=args.curriculum_min_frac)
    else:
        train_ds = IQWindows(Xtr, Ytr)

    train_loader = make_loader_from_ds(train_ds, args.batch, True, rank, world, args.workers)
    val_loader = make_loader(Xva, Yva, max(1, args.batch // 2), False, rank, world, args.workers)

    model = make_model(in_ch=4, hid=args.width, blocks=args.blocks,
                       k=args.kernel, dropout=args.dropout).to(device)
    if world > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local], find_unused_parameters=False
        )

    loss_fn = CompositeLoss(
        args.fs, args.inband, args.guard,
        spec_w=args.spec_w,
        w_in=args.spec_w_in,
        w_guard=args.spec_w_guard,
        w_out=args.spec_w_out,
        smooth_w=args.smooth_w,
        time_w=args.time_w,
        align_maxlag=args.align_maxlag,
        align_w=args.align_w,
        gain_align=args.gain_align,
        align_frac_steps=args.align_frac_steps,
        peak_w=args.peak_w,
        peak_k=args.peak_k,
        peak_region=args.peak_region,
    )

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999)
    )

    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.epochs * steps_per_epoch
    sched_info = configure_scheduler(opt, args, steps_per_epoch, total_steps)

    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    ema = EMA(model, decay=args.ema) if args.ema > 0 else None

    best_val = float("inf")
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(1, args.epochs + 1):
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

        if world > 1 and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        # ---- spec loss ramp (keep final after ramp) ----
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
        t0 = time.time()
        train_loss_sum = 0.0
        n_train_batches = 0

        for i, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)  # [B, T, 2]
            yb = yb.to(device, non_blocking=True)
            if args.perseq_norm:
                xb, yb = perseq_rms_norm(xb, yb)

            base = notch_tonal_multi(
                xb, args.fs,
                peaks=args.notch_peaks,
                q=args.notch_q,
                depth_db=args.notch_depth,
            )  # [B, T, 2]

            jam = xb.permute(0, 2, 1)   # [B, 2, T]
            bas = base.permute(0, 2, 1) # [B, 2, T]
            inp = torch.cat([jam, bas], dim=1)  # [B, 4, T]

            with torch.amp.autocast('cuda', enabled=args.amp):
                resid = model(inp)                    # [B, 2, T]
                resid = resid.permute(0, 2, 1)       # [B, T, 2]
                yhat = base + resid                  # [B, T, 2]
                loss = loss_fn(yhat, yb)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            # LR scheduler step
            if sched_info["type"] != "none":
                if sched_info["warm_steps"] > 0 and step < sched_info["warm_steps"]:
                    scale = linear_warmup(step, sched_info["warm_steps"])
                    for pg in opt.param_groups:
                        pg["lr"] = args.lr * scale
                else:
                    if sched_info["type"] == "cosine":
                        sched_info["obj"].step()
                    elif sched_info["type"] == "cawr":
                        sched_info["obj"].step(epoch - 1 + i / max(1, steps_per_epoch))

            step += 1
            train_loss_sum += loss.item()
            n_train_batches += 1

            if ema is not None:
                ema.update(model)

        if master:
            dt = time.time() - t0
            train_loss = train_loss_sum / max(1, n_train_batches)
            print(f"Epoch {epoch:03d} | train loss {train_loss:.6f} | {dt:.1f}s")

        # ---------------- Validation ----------------
        if ema is not None:
            ema.apply(model)
        model.eval()

        val_loss = 0.0
        snr_in_sum = 0.0
        snr_out_sum = 0.0
        evm_sum = 0.0
        n_batches = 0

        evm_ai_sum_local = 0.0
        evm_ai_batches_local = 0.0

        t0 = time.time()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                if args.perseq_norm:
                    xb, yb = perseq_rms_norm(xb, yb)

                base = notch_tonal_multi(
                    xb, args.fs,
                    peaks=args.notch_peaks,
                    q=args.notch_q,
                    depth_db=args.notch_depth,
                )
                jam = xb.permute(0, 2, 1)
                bas = base.permute(0, 2, 1)
                inp = torch.cat([jam, bas], dim=1)
                resid = model(inp).permute(0, 2, 1)
                yhat = base + resid

                val_loss += loss_fn(yhat, yb).item()
                si, so = snr_in_out_raw(xb, yb, yhat)
                snr_in_sum += si
                snr_out_sum += so
                evm_sum += evm_rms_pct_raw(yb, yhat)
                n_batches += 1

                if args.diag_ai_evm != "off":
                    evm_ai = evm_aligned_inband_pct(
                        yb, yhat,
                        args.fs, args.inband, args.guard,
                        maxlag=max(8, args.align_maxlag),
                        frac_steps=max(3, args.align_frac_steps),
                        gain_align=True,
                    )
                    evm_ai_sum_local += float(evm_ai)
                    evm_ai_batches_local += 1.0
                    if args.diag_ai_evm == "batch" and master:
                        print(f"      ↳ (aligned,in-band) EVM {evm_ai:.2f}%")

        if ema is not None:
            ema.restore(model)

        # reduce across ranks (main metrics)
        if world > 1:
            t = torch.tensor(
                [val_loss, snr_in_sum, snr_out_sum, evm_sum, n_batches],
                device=device,
            )
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            val_loss, snr_in_sum, snr_out_sum, evm_sum, n_batches = t.tolist()

        val_loss /= max(1, n_batches)
        snr_in = snr_in_sum / max(1, n_batches)
        snr_out = snr_out_sum / max(1, n_batches)
        evm = evm_sum / max(1, n_batches)

        if master:
            dt = time.time() - t0
            print(
                f"Epoch {epoch:03d} | val {val_loss:.6f} | "
                f"SNR_in {snr_in:+.2f} dB → SNR_out {snr_out:+.2f} dB | "
                f"EVM {evm:.2f}% | {dt:.1f}s"
            )

        # epoch-avg aligned,in-band EVM
        if args.diag_ai_evm != "off":
            if world > 1:
                t_ai = torch.tensor(
                    [evm_ai_sum_local, evm_ai_batches_local], device=device
                )
                dist.all_reduce(t_ai, op=dist.ReduceOp.SUM)
                evm_ai_sum_tot, evm_ai_batches_tot = t_ai.tolist()
            else:
                evm_ai_sum_tot, evm_ai_batches_tot = evm_ai_sum_local, evm_ai_batches_local
            if args.diag_ai_evm == "epoch" and master and evm_ai_batches_tot > 0:
                evm_ai_avg = evm_ai_sum_tot / evm_ai_batches_tot
                print(f"      ↳ (aligned,in-band) EVM (epoch avg) {evm_ai_avg:.2f}%")

        # save best
        if master and (val_loss < best_val):
            best_val = val_loss
            ck = {
                "model": (
                    ema.state_dict_for(model)
                    if (ema is not None)
                    else (
                        model.module.state_dict()
                        if hasattr(model, "module")
                        else model.state_dict()
                    )
                ),
                "args": vars(args),
                "val": val_loss,
                "epoch": epoch,
            }
            Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
            out_path = Path(args.ckpt_dir) / "best.pt"
            torch.save(ck, out_path)
            print("  ↳ saved best ->", out_path)

    if world > 1:
        dist.destroy_process_group()


# ---------------- CLI ----------------

def build_argparser():
    ap = argparse.ArgumentParser("Notch+Residual-TCN jammer-denoiser trainer")

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

    # normalization + alignment
    ap.add_argument("--perseq-norm", action="store_true",
                    help="Per-window RMS normalization using clean RMS.")
    ap.add_argument("--align-maxlag", type=int, default=0,
                    help="Max integer lag for align term (0 disables).")
    ap.add_argument("--align-w", type=float, default=0.0,
                    help="Weight of optional alignment term.")
    ap.add_argument("--gain-align", action="store_true",
                    help="Include complex gain alignment in align term.")
    ap.add_argument("--align-frac-steps", type=int, default=0,
                    help="Fractional delay steps (currently ignored).")

    # spectral weight ramp
    ap.add_argument("--spec-w-final", type=float, default=0.6,
                    help="Target spectral weight after ramp.")
    ap.add_argument("--spec-w-ramp-epochs", type=int, default=10,
                    help="Epochs to ramp spec_w to spec-w-final (0 disables).")

    # top-K spectral peak emphasis
    ap.add_argument("--peak-w", type=float, default=0.0,
                    help="Extra weight for top-K spectral residuals.")
    ap.add_argument("--peak-k", type=int, default=0,
                    help="K bins per batch to emphasize (0 disables).")
    ap.add_argument("--peak-region", type=str, default="guard",
                    choices=["guard", "in", "both"])
    ap.add_argument("--peak-ramp-epochs", type=int, default=0,
                    help="Epochs to ramp peak_w from 0→peak_w (0 disables).")

    # opt + scheduler
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scheduler", type=str, default="cawr",
                    choices=["cosine", "cawr", "none"])
    ap.add_argument("--cawr-T0", type=int, default=5,
                    help="Epochs between first restart for CAWR.")
    ap.add_argument("--cawr-Tmult", type=int, default=2,
                    help="Restart period multiplier for CAWR.")
    ap.add_argument("--warmup-frac", type=float, default=0.06,
                    help="Fraction of steps for LR warmup.")

    # diagnostics
    ap.add_argument("--diag-ai-evm", type=str, default="off",
                    choices=["off", "epoch", "batch"],
                    help="Aligned,in-band EVM logging: off, once per epoch, or per-batch.")

    return ap


def main():
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
