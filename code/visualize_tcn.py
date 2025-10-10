#!/usr/bin/env python3
# viz_denoise_tcn.py
import argparse, sys, math, json, re, os
from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ----------------------------
# Minimal causal TCN (matches common setups used in our training loops)
# ----------------------------
class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, bias=True):
        padding = 0  # we'll pad manually to keep strict causality
        super().__init__(in_ch, out_ch, kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias)
        self.left_pad = dilation * (kernel_size - 1)

    def forward(self, x):
        # x: [B,C,T]
        x = F.pad(x, (self.left_pad, 0))
        return super().forward(x)

class ResidualBlock(nn.Module):
    def __init__(self, ch, kernel, dilation, dropout=0.0):
        super().__init__()
        self.c1 = nn.utils.weight_norm(CausalConv1d(ch, ch, kernel, dilation=dilation))
        self.c2 = nn.utils.weight_norm(CausalConv1d(ch, ch, kernel, dilation=dilation))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.c1(x)
        y = self.act(y)
        y = self.dropout(y)
        y = self.c2(y)
        y = self.act(y)
        return x + y  # residual

class CausalTCN(nn.Module):
    def __init__(self, in_ch=2, out_ch=2, width=192, blocks=10, kernel=7, dilation_growth=2, dropout=0.0):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, width, kernel_size=1)
        layers = []
        d = 1
        for _ in range(blocks):
            layers.append(ResidualBlock(width, kernel, dilation=d, dropout=dropout))
            d *= dilation_growth
        self.tcn = nn.Sequential(*layers)
        self.out = nn.Conv1d(width, out_ch, kernel_size=1)

    def forward(self, x):
        # x: [B, C=2, T]
        h = self.inp(x)
        h = self.tcn(h)
        y = self.out(h)
        return y

# ----------------------------
# Utilities
# ----------------------------
def complex_from_iq(iq: np.ndarray) -> np.ndarray:
    # iq: [T,2] float -> complex [T]
    return iq[...,0] + 1j*iq[...,1]

def evm_percent(y_hat: np.ndarray, y_ref: np.ndarray) -> float:
    # EVM% = sqrt(sum |e|^2 / sum |ref|^2) * 100
    num = np.sum(np.abs(y_hat - y_ref)**2)
    den = np.sum(np.abs(y_ref)**2) + 1e-12
    return float(math.sqrt(num / den) * 100.0)

def snr_db(sig: np.ndarray, err: np.ndarray) -> float:
    # SNR = P_signal / P_error
    ps = np.mean(np.abs(sig)**2) + 1e-12
    pe = np.mean(np.abs(err)**2) + 1e-12
    return 10.0 * math.log10(ps/pe)

def try_get(d: Dict[str,Any], keys) -> Optional[Any]:
    for k in keys:
        if k in d: return d[k]
    return None

def detect_xy_keys(nzf: np.lib.npyio.NpzFile, split: str) -> Tuple[str,str]:
    # Try common patterns first
    keys = list(nzf.keys())
    split = split.lower()
    x_candidates = [
        f"X{split[:2]}", f"X_{split}", f"X{split}", f"x_{split}", f"x{split}",
        "X", "input", "jammed", "noisy", "J", "Xte", "Xtest", "Xva", "Xval", "Xtr", "Xtrain"
    ]
    y_candidates = [
        f"Y{split[:2]}", f"Y_{split}", f"Y{split}", f"y_{split}", f"y{split}",
        "Y", "target", "clean", "S", "Yte", "Ytest", "Yva", "Yval", "Ytr", "Ytrain"
    ]
    # Filter by existence
    xs = [k for k in x_candidates if k in keys]
    ys = [k for k in y_candidates if k in keys]
    # Choose a pair with matching first dimension
    for xk in xs:
        for yk in ys:
            if nzf[xk].shape[0] == nzf[yk].shape[0]:
                return xk, yk
    # Fallback: try to infer pair by name heuristics
    raise KeyError(f"Could not auto-detect X/Y keys from keys={keys}. Specify --x-key and --y-key.")

def load_pair(npz_path: str, split: str, idx: int, x_key: Optional[str], y_key: Optional[str]) -> Tuple[np.ndarray,np.ndarray]:
    nzf = np.load(npz_path)
    try:
        if x_key is None or y_key is None:
            xk, yk = detect_xy_keys(nzf, split)
        else:
            xk, yk = x_key, y_key
        X = nzf[xk]
        Y = nzf[yk]
    finally:
        nzf.close()
    if idx < 0 or idx >= X.shape[0]:
        raise IndexError(f"idx {idx} out of range 0..{X.shape[0]-1}")
    x = X[idx]  # [T,2]
    y = Y[idx]  # [T,2]
    return x, y

def safe_torch_load(path, map_location):
    ckpt = torch.load(path, map_location=map_location)
    return ckpt

def maybe_build_model_from_ckpt(ckpt, device, default_cfg: dict):
    # Accept either a full Module saved with torch.save(model) or a dict with 'state_dict' and optional config.
    if isinstance(ckpt, nn.Module):
        model = ckpt
        model.to(device)
        model.eval()
        return model

    # Try torch.jit
    if isinstance(ckpt, torch.jit.ScriptModule) or isinstance(ckpt, torch.jit.RecursiveScriptModule):
        model = ckpt.to(device)
        model.eval()
        return model

    # Dict-like: extract cfg
    if isinstance(ckpt, dict):
        cfg = try_get(ckpt, ["model_cfg","cfg","hparams","args"]) or {}
        # merge with defaults
        cfg_norm = {**default_cfg}
        # pull known keys if present
        for k in ["in_ch","out_ch","width","blocks","kernel","dilation_growth","dropout"]:
            if k in cfg: cfg_norm[k] = cfg[k]
            # allow alt names
            if k=="width" and "channels" in cfg: cfg_norm[k] = cfg["channels"]
            if k=="blocks" and "depth" in cfg: cfg_norm[k] = cfg["depth"]

        model = CausalTCN(**cfg_norm).to(device)
        # pick state_dict key
        state = try_get(ckpt, ["state_dict","model_state","model","ema_state","net"])
        if state is None:
            # maybe raw state_dict
            state = ckpt
        # strip possible "module." prefixes
        new_state = {}
        for k,v in state.items():
            new_state[k[7:]] = v if k.startswith("module.") else v
        model.load_state_dict(new_state, strict=False)
        model.eval()
        return model

    # Unknown type
    raise TypeError("Unsupported checkpoint type; expected nn.Module, ScriptModule, or dict with state_dict.")

# ----------------------------
# Main
# ----------------------------
def main():
    p = argparse.ArgumentParser(description="Plot jammed vs denoised vs clean for a single window.")
    p.add_argument("--ckpt", required=True, help="Path to tcn_denoiser.pt")
    p.add_argument("--data", required=True, help="Path to dataset .npz")
    p.add_argument("--split", default="test", help="Which split to auto-detect (test/val/train); used only for key guessing")
    p.add_argument("--idx", type=int, default=0, help="Sample index to visualize")
    p.add_argument("--x-key", default=None, help="Override NPZ key for inputs (jammed)")
    p.add_argument("--y-key", default=None, help="Override NPZ key for targets (clean)")
    p.add_argument("--save", default=None, help="If set, save figure here (e.g., plot.png)")
    p.add_argument("--title", default=None, help="Optional custom plot title")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # Load sample
    x_iq, y_iq = load_pair(args.data, args.split, args.idx, args.x_key, args.y_key)  # [T,2]
    T = x_iq.shape[0]

    device = torch.device(args.device)
    default_cfg = dict(in_ch=2, out_ch=2, width=192, blocks=10, kernel=7, dilation_growth=2, dropout=0.0)

    # Build/load model
    ckpt = safe_torch_load(args.ckpt, map_location=device)
    model = maybe_build_model_from_ckpt(ckpt, device, default_cfg)

    # Torch inference
    with torch.no_grad():
        xin = torch.from_numpy(x_iq.T).float().unsqueeze(0).to(device)  # [1,2,T]
        yhat = model(xin)  # [1,2,T]
        y_hat = yhat.squeeze(0).T.detach().cpu().numpy()  # [T,2]

    # Metrics
    jam = complex_from_iq(x_iq)
    den = complex_from_iq(y_hat)
    ref = complex_from_iq(y_iq)
    evm_in = evm_percent(jam, ref)
    evm_out = evm_percent(den, ref)
    snr_in = snr_db(ref, jam - ref)
    snr_out = snr_db(ref, den - ref)
    delta_snr = snr_out - snr_in

    # Print metrics
    print(f"Sample idx {args.idx}:")
    print(f"  EVM_in  : {evm_in:.2f}%")
    print(f"  EVM_out : {evm_out:.2f}%")
    print(f"  SNR_in  : {snr_in:.2f} dB")
    print(f"  SNR_out : {snr_out:.2f} dB")
    print(f"  ΔSNR    : {delta_snr:+.2f} dB")

    # Plot (time-domain I component; you can change to magnitude if you prefer)
    t = np.arange(T)
    plt.figure(figsize=(11,5.5))
    plt.plot(t, x_iq[:,0], label=f"Jammed (EVM {evm_in:.2f}%, SNR {snr_in:.2f} dB)")
    plt.plot(t, y_hat[:,0], label=f"Denoised (EVM {evm_out:.2f}%, SNR {snr_out:.2f} dB)")
    plt.plot(t, y_iq[:,0], label="Clean (reference)", linewidth=1.5)
    ttl = args.title or f"TCN Denoising | idx {args.idx} | ΔSNR {delta_snr:+.2f} dB"
    plt.title(ttl)
    plt.xlabel("Sample")
    plt.ylabel("I component")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"Saved figure -> {args.save}")
    plt.show()

if __name__ == "__main__":
    main()
