#!/usr/bin/env python3
# gen_data.py
import math, os, json, argparse
import numpy as np
from dataclasses import dataclass, asdict

# ---------------------------
# Config & RNG
# ---------------------------
@dataclass
class GenConfig:
    fs: float = 2.0e6               # sample rate (Hz)
    symrate: float = 200e3          # BPSK symbol rate (Hz)
    block_len: int = 256            # window length (complex samples)
    n_train: int = 20000
    n_val: int = 2000
    snr_db_range: tuple = (0, 20)          # AWGN SNR (dB)
    jsr_db_range: tuple = (-5, 25)         # Jammer-to-signal ratio (dB)
    f0_range: tuple = (-300e3, 300e3)      # jammer start freq (Hz)
    sweep_bw_range: tuple = (50e3, 500e3)  # sweep span (Hz)
    sweep_time_range: tuple = (0.2e-3, 2e-3) # sweep time (sec) across block
    seed: int = 1337

def set_seed(seed: int):
    rng = np.random.default_rng(seed)
    return rng

# ---------------------------
# Signal helpers
# ---------------------------
def gen_bpsk_symbols(nsym, rng):
    bits = rng.integers(0, 2, nsym, endpoint=False)
    sym = 2*bits - 1  # {0,1}->{-1,+1}
    return sym.astype(np.float32), bits

def upsample_rect(sym, sps):
    return np.repeat(sym, sps)

def add_awgn(x, snr_db, rng):
    # x complex64
    sigp = np.mean(np.abs(x)**2)
    snr_lin = 10**(snr_db/10)
    npow = sigp/snr_lin
    n = (np.sqrt(npow/2)*(rng.standard_normal(x.shape)+1j*rng.standard_normal(x.shape))).astype(np.complex64)
    return (x + n).astype(np.complex64)

def add_sweeping_cw(x, fs, jsr_db, f0, sweep_bw, sweep_time, rng):
    N = len(x)
    t = np.arange(N)/fs
    T = t[-1] if t[-1] > 0 else 1/fs
    eff_T = min(max(sweep_time, 1e-9), T)
    k = sweep_bw / eff_T
    k *= 1 if rng.random() < 0.5 else -1  # random direction
    phase0 = 2*np.pi*rng.random()

    # instantaneous frequency f(t) = f0 + k*t  -> phase = 2π ∫ f dt
    f_inst = f0 + k*t
    phi = 2*np.pi*np.cumsum(f_inst)/fs + phase0
    tone = np.exp(1j*phi).astype(np.complex64)

    # scale jammer to JSR
    sigp = np.mean(np.abs(x)**2)
    jsr_lin = 10**(jsr_db/10)
    jam_pow = sigp*jsr_lin
    tone = tone * np.sqrt(jam_pow)

    return (x + tone).astype(np.complex64)

def make_example(cfg: GenConfig, rng):
    sps = int(round(cfg.fs/cfg.symrate))
    nsym = math.ceil(cfg.block_len / sps)
    sym, bits = gen_bpsk_symbols(nsym, rng)
    x = upsample_rect(sym, sps)[:cfg.block_len].astype(np.float32)
    x = (x + 0j).astype(np.complex64)
    # random carrier phase
    x *= np.exp(1j*2*np.pi*rng.random()).astype(np.complex64)

    clean = x.copy()

    snr_db   = rng.uniform(*cfg.snr_db_range)
    jsr_db   = rng.uniform(*cfg.jsr_db_range)
    f0       = rng.uniform(*cfg.f0_range)
    sweep_bw = rng.uniform(*cfg.sweep_bw_range)
    sweep_T  = rng.uniform(*cfg.sweep_time_range)

    jammed = add_sweeping_cw(clean, cfg.fs, jsr_db, f0, sweep_bw, sweep_T, rng)
    jammed = add_awgn(jammed, snr_db, rng)

    # per-example RMS normalization (stabilizes training)
    denom = np.sqrt(np.mean(np.abs(jammed)**2) + 1e-12)
    jammed_n = jammed/denom
    clean_n  = clean/denom

    # outputs are concatenated [Re, Im]
    X = np.column_stack([jammed_n.real, jammed_n.imag]).astype(np.float32).reshape(-1)
    Y = np.column_stack([clean_n.real,  clean_n.imag ]).astype(np.float32).reshape(-1)
    return X, Y

def build_split(cfg: GenConfig, n, rng):
    X = np.zeros((n, 2*cfg.block_len), dtype=np.float32)
    Y = np.zeros((n, 2*cfg.block_len), dtype=np.float32)
    for i in range(n):
        X[i], Y[i] = make_example(cfg, rng)
    return X, Y

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate BPSK + sweeping CW jamming dataset")
    ap.add_argument("--out", default="artifacts/bpsk_sweep_dataset.npz", help="output NPZ path")
    ap.add_argument("--fs", type=float, default=2.0e6)
    ap.add_argument("--symrate", type=float, default=200e3)
    ap.add_argument("--block_len", type=int, default=256)
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    cfg = GenConfig(fs=args.fs, symrate=args.symrate, block_len=args.block_len,
                    n_train=args.n_train, n_val=args.n_val, seed=args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rng = set_seed(cfg.seed)

    Xtr, Ytr = build_split(cfg, cfg.n_train, rng)
    Xva, Yva = build_split(cfg, cfg.n_val,   rng)

    meta = asdict(cfg)
    np.savez(args.out, Xtr=Xtr, Ytr=Ytr, Xva=Xva, Yva=Yva, meta=json.dumps(meta))
    print(f"Saved dataset to {args.out}")
    print(f"Shapes: Xtr {Xtr.shape}  Ytr {Ytr.shape}  Xva {Xva.shape}  Yva {Yva.shape}")
    print("Note: inputs/targets are per-example RMS-normalized and flattened as [Re,Im]")

if __name__ == "__main__":
    main()
