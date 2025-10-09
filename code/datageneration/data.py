#!/usr/bin/env python3
"""
Minimal data generator for synthetic BPSK with optional wide sweeping-CW jammer.

Outputs an NPZ with: Xtr, Ytr, Xva, Yva
 - X* = input (jammed/noisy) blocks, flattened [Re, Im]
 - Y* = target clean BPSK blocks, flattened [Re, Im]

Default split: 90% train / 10% val. You can change SPLIT_VAL below.

Example (PowerShell / Bash):
  python data.py --n 20000 --out artifacts/gnss_synth_sweepcw_20k.npz --seed 1337
"""
from __future__ import annotations
import argparse, json, math, os
from dataclasses import dataclass
from typing import Tuple
import numpy as np

# =========================
# --- USER CONFIG BELOW ---
# =========================
@dataclass(frozen=True)
class GenCfg:
    fs: float = 2.0e6                # sample rate [Hz]
    symrate: float = 200e3           # BPSK symbol rate [sym/s]
    block_len: int = 256             # samples per example

    # Dataset composition
    split_val: float = 0.10          # fraction reserved for validation
    p_jammer: float = 0.80           # probability an example includes sweeping CW
    p_awgn: float = 1.00             # always add AWGN (set <1.0 to occasionally none)

    # Distributions (dB)
    snr_min: float = 5.0             # AWGN SNR range (dB)
    snr_max: float = 30.0

    jsr_min: float = -5.0            # sweeping CW Jammer-to-Signal range (dB)
    jsr_max: float = 25.0

    # Sweeping CW parameters
    # Keep |f0| <= fs/2 - bw/2 to avoid aliasing
    f0_min: float = -300e3
    f0_max: float = +300e3
    sweep_bw_min: float = 250e3
    sweep_bw_max: float = 600e3
    sweep_time_min: float = 0.2e-3   # seconds
    sweep_time_max: float = 2.0e-3

    # Impairments
    max_cfo_hz: float = 5e3          # uniform in [-max, +max]
    enable_cfo: bool = False

    # Random seed default (can be overridden via CLI)
    seed: int = 1337

CFG = GenCfg()

# =========================
# --- IMPLEMENTATION ---
# =========================

def _rms_norm(x: np.ndarray) -> np.ndarray:
    p = np.mean(np.abs(x)**2) + 1e-12
    return (x / np.sqrt(p)).astype(np.complex64)


def _bpsk_baseband(fs: float, symrate: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a *clean* BPSK baseband block (unit RMS before normalization)."""
    sps = fs / symrate
    if abs(sps - round(sps)) > 1e-6:
        # Allow non-integer sps by using float-time resampling of symbols
        # Create a little longer than needed to ensure coverage; then trim.
        n_syms = int(math.ceil(n / sps) + 4)
        syms = (rng.random(n_syms) < 0.5).astype(np.int8)*2 - 1  # ±1
        t = np.arange(n) / fs
        # Map each sample to its symbol index
        idx = np.floor(t * symrate).astype(int)
        idx = np.clip(idx, 0, n_syms-1)
        x = syms[idx].astype(np.float32) + 0j
    else:
        sps_i = int(round(sps))
        n_syms = int(math.ceil(n / sps_i))
        syms = (rng.random(n_syms) < 0.5).astype(np.int8)*2 - 1  # ±1
        x = np.repeat(syms, sps_i)[:n].astype(np.float32) + 0j
    # random phase rotation for variety
    phi0 = 2*np.pi*rng.random()
    x = x * np.exp(1j*phi0)
    return _rms_norm(x)


def _add_awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    sigp = np.mean(np.abs(x)**2)
    snr_lin = 10**(snr_db/10)
    npow = sigp / snr_lin
    n = (rng.normal(0, np.sqrt(npow/2), x.shape) + 1j*rng.normal(0, np.sqrt(npow/2), x.shape)).astype(np.complex64)
    return (x + n).astype(np.complex64)


def _apply_cfo(x: np.ndarray, fs: float, max_cfo_hz: float, rng: np.random.Generator) -> np.ndarray:
    if max_cfo_hz <= 0:
        return x
    N = len(x)
    f = rng.uniform(-max_cfo_hz, max_cfo_hz)
    n = np.arange(N)
    ph = 2*np.pi*f*n/fs + 2*np.pi*rng.random()
    return (x * np.exp(1j*ph).astype(np.complex64)).astype(np.complex64)


def _add_sweeping_cw(x: np.ndarray, fs: float, jsr_db: float, f0: float, bw: float, sweep_time: float, rng: np.random.Generator) -> np.ndarray:
    N = len(x)
    t = np.arange(N) / fs
    T = max(t[-1], 1.0/fs)
    eff_T = float(np.clip(sweep_time, 1e-9, T))
    k = (bw / eff_T) * (1 if rng.random() < 0.5 else -1)  # sweep rate Hz/s (sign random)
    phase0 = 2*np.pi*rng.random()
    # instantaneous frequency: f(t) = f0 + k*t
    phi = 2*np.pi * (f0*t + 0.5*k*t**2) + phase0
    tone = np.exp(1j*phi).astype(np.complex64)
    # scale to target JSR
    sigp = np.mean(np.abs(x)**2)
    jam_pow = sigp * (10**(jsr_db/10))
    tone *= np.sqrt(max(jam_pow, 1e-12))
    return (x + tone).astype(np.complex64)


def _flatten_ri(x: np.ndarray) -> np.ndarray:
    return np.stack([x.real, x.imag], axis=-1).reshape(-1).astype(np.float32)


def _draw_jsr(cfg: GenCfg, rng: np.random.Generator) -> float:
    # Balanced mixture: 40% mild, 40% moderate, 20% strong, within [jsr_min, jsr_max]
    bands = [(-5, 5), (5, 15), (15, 25)]
    probs = [0.4, 0.4, 0.2]
    b = rng.choice(len(bands), p=probs)
    lo, hi = bands[b]
    lo = max(lo, cfg.jsr_min); hi = min(hi, cfg.jsr_max)
    if hi <= lo:  # fallback to uniform
        return rng.uniform(cfg.jsr_min, cfg.jsr_max)
    return rng.uniform(lo, hi)


def _draw_snr(cfg: GenCfg, rng: np.random.Generator) -> float:
    return rng.uniform(cfg.snr_min, cfg.snr_max)


def _draw_sweep_params(cfg: GenCfg, rng: np.random.Generator) -> Tuple[float, float, float]:
    bw = rng.uniform(cfg.sweep_bw_min, cfg.sweep_bw_max)
    # ensure f0 stays within Nyquist margin
    f0_lo = -(cfg.fs/2 - bw/2)
    f0_hi = +(cfg.fs/2 - bw/2)
    f0 = rng.uniform(max(cfg.f0_min, f0_lo), min(cfg.f0_max, f0_hi))
    st = rng.uniform(cfg.sweep_time_min, cfg.sweep_time_max)
    return f0, bw, st


def _make_example(cfg: GenCfg, rng: np.random.Generator):
    n = cfg.block_len
    clean = _bpsk_baseband(cfg.fs, cfg.symrate, n, rng)

    # Optional CFO on the signal (before noise/jammer)
    if cfg.enable_cfo:
        clean = _apply_cfo(clean, cfg.fs, cfg.max_cfo_hz, rng)
    x = clean.copy()

    meta = {
        "snr_db": None, "jsr_db": None,
        "f0": None, "sweep_bw": None, "sweep_time": None,
        "jammer": False
    }

    # Sweeping CW with probability p_jammer
    if rng.random() < cfg.p_jammer:
        jsr = _draw_jsr(cfg, rng)
        f0, bw, st = _draw_sweep_params(cfg, rng)
        x = _add_sweeping_cw(x, cfg.fs, jsr, f0, bw, st, rng)
        meta.update({"jsr_db": float(jsr), "f0": float(f0), "sweep_bw": float(bw), "sweep_time": float(st), "jammer": True})

    # AWGN (usually always on)
    if rng.random() < cfg.p_awgn:
        snr = _draw_snr(cfg, rng)
        x = _add_awgn(x, snr, rng)
        meta["snr_db"] = float(snr)

    # OPTION A: Normalize input-target pairs together
    # Compute RMS from noisy input only
    x_rms = np.sqrt(np.mean(np.abs(x)**2) + 1e-12)
    # Apply the same scaling factor to both input AND target
    x = (x / x_rms).astype(np.complex64)
    y = (clean / x_rms).astype(np.complex64)

    return _flatten_ri(x), _flatten_ri(y), meta


def _build_dataset(n_total: int, cfg: GenCfg, seed: int):
    rng = np.random.default_rng(seed)

    n_val = max(1, int(round(n_total * cfg.split_val)))
    n_train = n_total - n_val

    Xtr = np.zeros((n_train, 2*cfg.block_len), dtype=np.float32)
    Ytr = np.zeros((n_train, 2*cfg.block_len), dtype=np.float32)
    Xva = np.zeros((n_val, 2*cfg.block_len), dtype=np.float32)
    Yva = np.zeros((n_val, 2*cfg.block_len), dtype=np.float32)

    # Optionally collect lightweight metadata for analysis
    Mtr, Mva = [], []

    for i in range(n_train):
        Xtr[i], Ytr[i], m = _make_example(cfg, rng)
        Mtr.append(m)
    for i in range(n_val):
        Xva[i], Yva[i], m = _make_example(cfg, rng)
        Mva.append(m)

    meta = {
        "cfg": {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in CFG.__dict__.items()},
        "seed": seed,
        "n_train": n_train,
        "n_val": n_val,
        "desc": "Synthetic BPSK with optional wide sweeping CW + AWGN (Option A RMS normalization)"
    }
    return (Xtr, Ytr, Xva, Yva, meta, Mtr, Mva)


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic BPSK dataset with sweeping CW. Minimal CLI: only --n, --out, --seed.")
    ap.add_argument("--n", type=int, required=True, help="Total number of examples (train + val)")
    ap.add_argument("--out", type=str, default="artifacts/gnss_synth_sweepcw.npz")
    ap.add_argument("--seed", type=int, default=CFG.seed)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    Xtr, Ytr, Xva, Yva, meta, Mtr, Mva = _build_dataset(args.n, CFG, args.seed)

    # Save arrays
    np.savez(args.out, Xtr=Xtr, Ytr=Ytr, Xva=Xva, Yva=Yva, meta=json.dumps(meta))

    # Also save a sidecar JSONL with per-example metadata (optional, handy for analysis)
    sidecar = args.out + ".jsonl"
    with open(sidecar, "w", encoding="utf-8") as f:
        for m in Mtr:
            f.write(json.dumps({"split": "train", **m})+"\n")
        for m in Mva:
            f.write(json.dumps({"split": "val", **m})+"\n")

    print(f"Saved {args.out}")
    print(f"  Xtr {Xtr.shape}  Ytr {Ytr.shape}")
    print(f"  Xva {Xva.shape}  Yva {Yva.shape}")
    print(f"Sidecar metadata: {sidecar}")

if __name__ == "__main__":
    main()