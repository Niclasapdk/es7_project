#!/usr/bin/env python3
# visualize_tcn.py — Notch+TCN inference (no alignment) with diagnostics & blind post-AGC
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------------- I/Q helpers ----------------
def ensure_iq(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    assert x.ndim == 3 and x.shape[-1] == 2, f"Expected [B,T,2], got {tuple(x.shape)}"
    return x.contiguous()

def complex_from_iq(x: torch.Tensor) -> torch.Tensor:
    x = ensure_iq(x)
    return torch.complex(x[..., 0], x[..., 1])

def iq_from_complex(z: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(z):
        raise RuntimeError("Expected complex tensor")
    return torch.stack([z.real, z.imag], dim=-1)

# ---------------- Data loading (mirrors trainer) ----------------
def load_npz(path: str):
    npz = np.load(path)
    keys = set(npz.keys())
    if {"Xtr", "Ytr", "Xva", "Yva"}.issubset(keys):
        Xtr, Ytr, Xva, Yva = npz["Xtr"], npz["Ytr"], npz["Xva"], npz["Yva"]
        mid = max(1, Xva.shape[0] // 2)
        return {
            "train": (Xtr, Ytr),
            "val":   (Xva[:mid],  Yva[:mid]),
            "test":  (Xva[mid:],  Yva[mid:])
        }
    if {"X", "Y"}.issubset(keys):
        X, Y = npz["X"], npz["Y"]
        N = X.shape[0]
        ntr = int(0.8 * N)
        nva = int(0.1 * N)
        return {
            "train": (X[:ntr], Y[:ntr]),
            "val":   (X[ntr:ntr+nva], Y[ntr:ntr+nva]),
            "test":  (X[ntr+nva:], Y[ntr+nva:])
        }
    raise ValueError(f"Unexpected keys in {path}: {sorted(keys)}")

# ---------------- Notch prefilter (same as trainer) ----------------
@torch.no_grad()
def notch_prefilter(x: torch.Tensor, fs: float, inband_hz: float, guard_hz: float,
                    max_depth_in_db: float = 40.0, max_depth_out_db: float = 120.0, q: float = 600.0) -> torch.Tensor:
    def _as_complex(x):
        if x.ndim == 3 and x.shape[-1] == 2:
            z = complex_from_iq(x)
            return z, iq_from_complex
        if torch.is_complex(x):
            return x, (lambda z: z)
        xr = x.float()
        return xr, (lambda z: z.real)

    z, back = _as_complex(x)  # [B,T] complex
    B, T = z.shape
    Z = torch.fft.fft(z, n=T, dim=1)
    freqs = torch.fft.fftfreq(T, d=1.0 / fs, device=z.device)

    BW = float(inband_hz)
    lo, hi = -(BW / 2 + guard_hz), (BW / 2 + guard_hz)

    d = torch.zeros_like(freqs).float()
    d = torch.where(freqs < lo, lo - freqs, d)
    d = torch.where(freqs > hi, freqs - hi, d)

    s = (d / (guard_hz + 1e-6)).clamp(min=0.0)
    s = s * s * (3 - 2 * s.clamp(max=1.0))  # smoothstep

    depth_db = max_depth_in_db + (max_depth_out_db - max_depth_in_db) * s
    depth_lin = torch.pow(10.0, -depth_db / 20.0)

    mag = Z.abs()
    thresh = mag.mean(dim=1, keepdim=True) * 5.0
    attn = 1.0 / (1.0 + (mag / (thresh + 1e-6)))
    base = depth_lin[None, :]
    mask = torch.maximum(attn, base)

    Zf = Z * mask
    zf = torch.fft.ifft(Zf, n=T, dim=1)
    return back(zf)

# ---------------- Model (names match checkpoint) ----------------
class CausalConv1d(nn.Conv1d):
    def __init__(self, C_in, C_out, k, d=1):
        pad = (k - 1) * d
        super().__init__(C_in, C_out, k, padding=pad, dilation=d)
        self._pad = pad
    def forward(self, x):
        y = super().forward(x)
        if self._pad:
            y = y[..., :-self._pad]
        return y

class TCNBlock(nn.Module):
    # Keep attribute names conv1/conv2/norm1/norm2 (matches training checkpoint)
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
        self.tcn = nn.Sequential(*[TCNBlock(hid, k, 2**b, dropout) for b in range(blocks)])
        self.out = CausalConv1d(hid, 2, k=3, d=1)
    def forward(self, x):  # x: [B,C,T]
        h = self.inp(x); h = self.tcn(h); r = self.out(h); return r

def build_from_ckpt(ckpt_path: str, device: torch.device, apply_ema: bool = True):
    ck = torch.load(ckpt_path, map_location=device)
    a = ck.get("args", {})
    model = ResidualTCN(
        in_ch=4,
        hid=int(a.get("width", 64)),
        blocks=int(a.get("blocks", 8)),
        k=int(a.get("kernel", 7)),
        dropout=float(a.get("dropout", 0.05)),
    ).to(device)

    model.load_state_dict(ck["model"], strict=True)

    if apply_ema and ck.get("ema", None):
        with torch.no_grad():
            ema_state = ck["ema"]
            for n, p in model.named_parameters():
                if n in ema_state:
                    p.copy_(ema_state[n].to(device))
    model.eval()

    sig = dict(
        fs=float(a.get("fs", 4.092e6)),
        inband=float(a.get("inband", 2.046e6)),
        guard=float(a.get("guard", 150e3)),
        notch_in=float(a.get("notch_in_db", 40.0)),
        notch_out=float(a.get("notch_out_db", 120.0)),
        notch_q=float(a.get("notch_q", 600.0)),
    )
    return model, sig

# ---------------- Metrics & diagnostics ----------------
@torch.no_grad()
def snr_in_out_raw(x_in: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor):
    yt = complex_from_iq(y_true)
    xp = complex_from_iq(x_in)
    yp = complex_from_iq(y_pred)
    s = (yt.abs()**2).sum(dim=1).clamp_min(1e-12)
    n_in  = ((xp - yt).abs()**2).sum(dim=1).clamp_min(1e-12)
    n_out = ((yp - yt).abs()**2).sum(dim=1).clamp_min(1e-12)
    snr_in  = 10.0 * torch.log10((s / n_in)).mean().item()
    snr_out = 10.0 * torch.log10((s / n_out)).mean().item()
    return snr_in, snr_out

@torch.no_grad()
def evm_pct_raw(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    yt = complex_from_iq(y_true)
    yp = complex_from_iq(y_pred)
    evm = torch.sqrt(((yp - yt).abs()**2).sum(dim=1) / (yt.abs()**2).sum(dim=1).clamp_min(1e-12)).mean().item()
    return 100.0 * evm

@torch.no_grad()
def complex_corr_mag_phase(a_iq: torch.Tensor, b_iq: torch.Tensor):
    a = complex_from_iq(a_iq); b = complex_from_iq(b_iq)
    num = torch.sum(a.conj()*b, dim=1)
    den = torch.sqrt((a.abs()**2).sum(dim=1) * (b.abs()**2).sum(dim=1)).clamp_min(1e-12)
    rho = num / den
    rho_m = rho.abs().mean().item()
    rho_p = torch.atan2(rho.mean().imag, rho.mean().real).item()
    return rho_m, rho_p

@torch.no_grad()
def diag_aligned_scores(y_true: torch.Tensor, y_pred: torch.Tensor):
    yt = complex_from_iq(y_true)
    yp = complex_from_iq(y_pred)
    num = torch.sum(yp.conj() * yt, dim=1, keepdim=True)                # [B,1] complex
    den = (yp.abs()**2).sum(dim=1, keepdim=True).clamp_min(1e-12)       # [B,1] real
    alpha = num / den                                                    # [B,1] complex
    ypa = yp * alpha
    s = (yt.abs()**2).sum(dim=1).clamp_min(1e-12)
    n_out = ((ypa - yt).abs()**2).sum(dim=1).clamp_min(1e-12)
    snr_out_a = 10.0 * torch.log10(s / n_out).mean().item()
    evm_a = torch.sqrt(n_out / s).mean().item() * 100.0
    a_mean = alpha.mean()
    a_mag = a_mean.abs().item()
    a_phase = torch.atan2(a_mean.imag, a_mean.real).item()
    return snr_out_a, evm_a, a_mag, a_phase

@torch.no_grad()
def align_ls_to(target_iq: torch.Tensor, pred_iq: torch.Tensor) -> torch.Tensor:
    """One-tap complex LS: return pred aligned to target (no ground-truth needed if target=base)."""
    t = complex_from_iq(target_iq)
    p = complex_from_iq(pred_iq)
    den = (p.abs()**2).sum(dim=1, keepdim=True).clamp_min(1e-12)
    alpha = (torch.sum(p.conj()*t, dim=1, keepdim=True)) / den
    return iq_from_complex(p * alpha)

# ---------------- Plotting ----------------
def db10(x: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(x, 1e-12))

def psd_db(x: torch.Tensor) -> np.ndarray:
    z = x[:, 0].float().cpu().numpy() + 1j * x[:, 1].float().cpu().numpy()
    X = np.fft.fftshift(np.fft.fft(z))
    P = (np.abs(X) ** 2) / max(1, len(z))
    return db10(P)

def plot_sample(idx: int, fs: float, jam: torch.Tensor, base: torch.Tensor, den: torch.Tensor, ref: torch.Tensor, outdir: Path):
    T = jam.shape[0]
    t = np.arange(T) / fs
    fig, ax = plt.subplots(2, 1, figsize=(10, 7))

    ax[0].plot(t, jam[:, 0].cpu().numpy(), label="jammed (I)")
    ax[0].plot(t, base[:, 0].cpu().numpy(), label="notch base (I)", alpha=0.8)
    ax[0].plot(t, den[:, 0].cpu().numpy(), label="denoised (I)", alpha=0.9)
    ax[0].plot(t, ref[:, 0].cpu().numpy(), label="clean ref (I)", alpha=0.9, linestyle="--")
    ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("amplitude")
    ax[0].legend(loc="upper right"); ax[0].grid(True, alpha=0.3)

    freqs = np.fft.fftshift(np.fft.fftfreq(T, d=1.0 / fs))
    ax[1].plot(freqs, psd_db(jam), label="jammed")
    ax[1].plot(freqs, psd_db(base), label="notch base", alpha=0.8)
    ax[1].plot(freqs, psd_db(den), label="denoised", alpha=0.9)
    ax[1].plot(freqs, psd_db(ref), label="clean ref", alpha=0.9, linestyle="--")
    ax[1].set_xlabel("frequency [Hz]"); ax[1].set_ylabel("PSD [dB]")
    ax[1].legend(loc="upper right"); ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = outdir / f"denoise_sample_{idx:04d}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out

# ---------------- Main ----------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser("Inference for notch+TCN (no alignment)")
    ap.add_argument("--ckpt", required=True, type=str, help="Path to best.pt")
    ap.add_argument("--data", required=True, type=str, help="NPZ with X/Y")
    ap.add_argument("--split", default="val", choices=["train","val","test"])
    ap.add_argument("--idx", type=int, default=None, help="If set, evaluate this index only")
    ap.add_argument("--num", type=int, default=8, help="How many samples to dump (ignored if --idx is set)")
    ap.add_argument("--outdir", type=str, default="eval_dumps")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-ema", action="store_true", help="Do not apply EMA weights from checkpoint")
    ap.add_argument("--diag-align", dest="diag_align", action="store_true",
                    help="Also print aligned SNR/EVM for diagnostics")
    ap.add_argument("--rms-norm", dest="rms_norm", choices=["off", "y"], default="off",
                    help="Per-window RMS norm by Y RMS (diagnostic).")
    ap.add_argument("--post-agc", choices=["off","base","gt"], default="off",
                    help="One-tap complex gain on model output: 'base' = align to notch base (blind), 'gt' = align to ground truth (diagnostic)")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, sig = build_from_ckpt(args.ckpt, device, apply_ema=(not args.no_ema))

    splits = load_npz(args.data)
    X, Y = splits[args.split]
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    if args.idx is not None:
        indices = [int(args.idx)]
    else:
        N = X.shape[0]
        indices = list(range(min(args.num, N)))

    # accumulators
    snr_in_list = []; snr_out_list = []; evm_list = []
    snr_out_align_list = []; evm_align_list = []
    base_snr_out_list = []; base_evm_list = []

    for j, idx in enumerate(indices, 1):
        xb = torch.from_numpy(X[idx:idx+1]).to(device)  # [1,T,2]
        yb = torch.from_numpy(Y[idx:idx+1]).to(device)

        # Optional per-sample RMS norm (diagnostic)
        if args.rms_norm == "y":
            r = torch.sqrt((yb.pow(2).sum() / yb.numel()).clamp_min(1e-12))
            xb = xb / r
            yb = yb / r

        # Notch base and model forward
        base = notch_prefilter(xb, sig["fs"], sig["inband"], sig["guard"],
                               max_depth_in_db=sig["notch_in"], max_depth_out_db=sig["notch_out"], q=sig["notch_q"])
        jam = xb.permute(0, 2, 1)          # [1,2,T]
        bas = base.permute(0, 2, 1)        # [1,2,T]
        inp = torch.cat([jam, bas], dim=1) # [1,4,T]
        resid = model(inp).permute(0, 2, 1)
        yhat = base + resid                # [1,T,2]

        # Blind post-AGC (optional)
        yhat_eff = yhat
        if args.post_agc == "base":
            yhat_eff = align_ls_to(base, yhat)   # blind, no GT
        elif args.post_agc == "gt":              # diagnostic
            yhat_eff = align_ls_to(yb, yhat)

        # Metrics: BASE (for reference)
        _, so_b = snr_in_out_raw(xb, yb, base)
        ev_b = evm_pct_raw(yb, base)
        base_snr_out_list.append(so_b); base_evm_list.append(ev_b)

        # Metrics: MODEL (effective output)
        si, so = snr_in_out_raw(xb, yb, yhat_eff)
        ev = evm_pct_raw(yb, yhat_eff)
        snr_in_list.append(si); snr_out_list.append(so); evm_list.append(ev)

        rho_m, rho_p = complex_corr_mag_phase(yb, yhat_eff)

        msg = (f"[{j}/{len(indices)}] idx={idx} "
               f"| BASE: SNR_out={so_b:+.2f} dB, EVM={ev_b:.2f}% "
               f"|| MODEL: SNR_in={si:+.2f} → SNR_out={so:+.2f} dB, EVM={ev:.2f}% "
               f"|| corr(|rho|)={rho_m:.3f}, ∠rho={rho_p:.2f} rad "
               f"{'(post-agc='+args.post_agc+')' if args.post_agc!='off' else ''}")
        if args.diag_align:
            soA, evA, amag, aphase = diag_aligned_scores(yb, yhat_eff)
            snr_out_align_list.append(soA); evm_align_list.append(evA)
            msg += f" || (aligned) SNR_out={soA:+.2f} dB, EVM={evA:.2f}% | |α|≈{amag:.3f}, ∠α≈{aphase:.2f}"
        print(msg)
        rho_xy_m, rho_xy_p = complex_corr_mag_phase(xb, yb)
        rho_by_m, rho_by_p = complex_corr_mag_phase(base, yb)
        print(f"... || corr(X,Y)={rho_xy_m:.3f}, corr(base,Y)={rho_by_m:.3f}")

        _ = plot_sample(idx, sig["fs"], xb[0], base[0], yhat_eff[0], yb[0], outdir)

    # Averages
    if len(indices) > 1:
        line = (f"Avg over {len(indices)} samples: "
                f"BASE SNR_out {np.mean(base_snr_out_list):+.2f} dB | BASE EVM {np.mean(base_evm_list):.2f}%  ||  "
                f"MODEL SNR_in {np.mean(snr_in_list):+.2f} dB → SNR_out {np.mean(snr_out_list):+.2f} dB | "
                f"MODEL EVM {np.mean(evm_list):.2f}%")
        if args.diag_align and snr_out_align_list:
            line += f"  ||  (aligned) SNR_out {np.mean(snr_out_align_list):+.2f} dB | EVM {np.mean(evm_align_list):.2f}%"
        print(line)

if __name__ == "__main__":
    main()
