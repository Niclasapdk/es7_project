#!/usr/bin/env python3
"""
gnss_datagen.py — Ideal GPS L1 C/A dataset generator (pre-correlation complex baseband)
with optional synthetic jammers: tone | swept_cw | pulsed.

It creates an NPZ file with keys:
  - Xtr, Ytr  : training   inputs/targets (observed vs. clean)
  - Xva, Yva  : validation inputs/targets
  - Xte, Yte  : test       inputs/targets
  - meta      : JSON-encoded dict with run configuration and per-split metadata

Conventions
-----------
- Sampling: complex baseband, float32 I/Q packed along the last dim: shape (N, T, 2)
- Targets:  Y* are clean ideal GPS L1 C/A segments (complex baseband)
- Inputs:   X* = Y* + jammer (if enabled) + AWGN (if --snr-db specified)
- Timing:   Blocks are pre-correlation baseband windows of length `block_ms`
- Randomness: Independent seeds per split (seed, seed+1, seed+2)
"""

import argparse, math, json
from typing import Dict, Tuple, Optional
import numpy as np

# ---------------------------
# GPS L1 C/A constants
# ---------------------------
CHIP_RATE = 1.023e6        # C/A code chip rate [chips/s]
NAV_BIT_RATE = 50.0        # Navigation data rate [bits/s] (used implicitly via 20 ms/bit)
CA_LEN = 1023              # C/A code length [chips] per epoch (1 ms)
DEFAULT_SPS = 4            # Default samples per chip → fs = 4.092 MHz
TWOPI = 2.0 * math.pi

# PRN 1..32 G2 tap pairs (ICD-GPS-200). Each PRN uses one pair of G2 taps.
PRN_G2_TAPS: Dict[int, Tuple[int, int]] = {
    1:(2,6), 2:(3,7), 3:(4,8), 4:(5,9), 5:(1,9), 6:(2,10), 7:(1,8), 8:(2,9),
    9:(3,10),10:(2,3),11:(3,4),12:(5,6),13:(6,7),14:(7,8),15:(8,9),16:(9,10),
    17:(1,4),18:(2,5),19:(3,6),20:(4,7),21:(5,8),22:(6,9),23:(1,3),24:(4,6),
    25:(5,7),26:(6,8),27:(7,9),28:(8,10),29:(1,6),30:(2,7),31:(3,8),32:(4,9),
}

# ---------------------------
# C/A code utilities
# ---------------------------
def ca_code(prn: int) -> np.ndarray:
    """
    Generate one epoch (1023 chips) of GPS L1 C/A code for a given PRN.

    Returns
    -------
    chips : np.ndarray[int8], shape (1023,)
        Values are ±1 (NRZ mapping), where +1 corresponds to logic '0' in LFSR XOR sense.
    """
    if prn not in PRN_G2_TAPS:
        raise ValueError(f"PRN {prn} not supported (1..32).")

    # LFSR states for G1 and G2 (10 bits each), initialized to all 1s.
    g1 = np.ones(10, dtype=np.uint8)
    g2 = np.ones(10, dtype=np.uint8)
    tap_a, tap_b = PRN_G2_TAPS[prn]

    out = np.empty(CA_LEN, dtype=np.int8)
    for i in range(CA_LEN):
        # Output taps: G1 last bit XOR selected G2 taps -> C/A chip (before NRZ mapping)
        g1_out = g1[-1]
        g2_out = g2[-tap_a] ^ g2[-tap_b]
        chip = g1_out ^ g2_out

        # NRZ mapping: 0 -> +1, 1 -> -1
        out[i] = 1 if chip == 0 else -1

        # Feedback and shift (indexes below are 0-based; comments 1-based for readability)
        # G1 feedback taps are bits 3 and 10 (1-based) → indexes 2 and 9 (0-based)
        g1_fb = g1[2] ^ g1[9]
        g1[1:] = g1[:-1]; g1[0] = g1_fb

        # G2 feedback taps: 2,3,6,8,9,10 (1-based) → 1,2,5,7,8,9 (0-based)
        g2_fb = g2[1]^g2[2]^g2[5]^g2[7]^g2[8]^g2[9]
        g2[1:] = g2[:-1]; g2[0] = g2_fb

    return out.astype(np.int8)


def tile_ca_samples(prn: int, sps: int, total_samples: int, start_chip_phase: float) -> np.ndarray:
    """
    Sample the PRN's C/A code at `sps` samples per chip and tile to cover `total_samples`,
    starting at a fractional chip phase within [0, 1).

    Parameters
    ----------
    prn : int
    sps : int
        Samples per chip (fs / 1.023e6). Must be integer.
    total_samples : int
    start_chip_phase : float
        Fractional chip offset within an epoch (0..1).

    Returns
    -------
    seq : np.ndarray[float32], shape (total_samples,)
        Repeated PRN sequence sampled at `sps`.
    """
    # Convert fractional chip phase to sample offset within one epoch
    start_sample_offset = int(round(start_chip_phase * sps)) % (CA_LEN * sps)
    ca = ca_code(prn).astype(np.float32)

    # Upsample: repeat each chip 'sps' times → one epoch has CA_LEN * sps samples
    epoch = np.repeat(ca, repeats=sps)

    # Tile enough epochs to cover the requested segment plus the start offset
    need = start_sample_offset + total_samples
    reps = (need + len(epoch) - 1) // len(epoch)
    seq = np.tile(epoch, reps)

    # Return the contiguous slice starting at the desired offset
    return seq[start_sample_offset:start_sample_offset + total_samples]


def build_nav_bits(
    total_samples: int,
    sps: int,
    rng: np.random.Generator,
    start_bit_phase: Optional[float]=None
) -> np.ndarray:
    """
    Build a random 50 bps navigation bit stream (±1) at the given sample rate and duration.
    Each nav bit spans 20 ms (since 50 bps), i.e., 20 * 1 ms C/A epochs.

    Parameters
    ----------
    total_samples : int
    sps : int
        Samples per chip.
    rng : np.random.Generator
        Random generator for bit values and phases.
    start_bit_phase : float or None
        Random fractional phase into a nav bit (0..1). If None, draw uniformly.

    Returns
    -------
    nav : np.ndarray[float32], shape (total_samples,)
        ±1 navigation data sequence aligned with the sample grid.
    """
    samples_per_ms = CA_LEN * sps               # samples in 1 ms (one C/A epoch)
    bit_len_samples = int(round(20 * samples_per_ms))  # 20 ms/bit at 50 bps

    if start_bit_phase is None:
        start_bit_phase = rng.random()

    # Choose a phase within the current nav bit (in samples)
    start_offset = int(round(start_bit_phase * bit_len_samples)) % bit_len_samples

    # Number of nav bits needed to cover the requested sample range
    bits_needed = (start_offset + total_samples + bit_len_samples - 1)//bit_len_samples

    # Draw bits (±1), then repeat each bit for its duration in samples
    bits = rng.choice(np.array([1.0, -1.0], dtype=np.float32), size=bits_needed)
    nav = np.repeat(bits, bit_len_samples)

    return nav[start_offset:start_offset + total_samples]


def synthesize_block(
    fs: float,
    block_len: int,
    prn: int,
    sps: int,
    doppler_hz: float,
    rng: np.random.Generator,
    allow_frac_code_phase: bool=True
) -> np.ndarray:
    """
    Create one complex baseband block of clean GPS L1 C/A (pre-correlation).

    Steps
    -----
    1) Sample C/A code (±1) at `sps`, starting at random chip phase (if allowed).
    2) Generate random nav bits (±1) at 50 bps and multiply with C/A → BPSK(1) spreading.
    3) Impose Doppler and random carrier phase via complex exponential at baseband.

    Returns
    -------
    y : np.ndarray[complex64], shape (block_len,)
        Clean complex baseband signal with Doppler and random carrier phase.
    """
    t = np.arange(block_len, dtype=np.float64) / fs

    start_chip_phase = rng.random() if allow_frac_code_phase else 0.0
    ca_samples = tile_ca_samples(prn, sps, block_len, start_chip_phase)
    nav = build_nav_bits(block_len, sps, rng)

    # Baseband BPSK(1): ±1 (C/A * nav)
    base = ca_samples * nav

    # Random initial carrier phase, then apply Doppler in baseband
    phi0 = rng.uniform(0.0, TWOPI)
    carrier = np.exp(1j * (TWOPI * doppler_hz * t + phi0))

    return (base.astype(np.complex64) * carrier.astype(np.complex64)).astype(np.complex64)

# ---------------------------
# Jammer utilities
# ---------------------------
def tone_jammer(fs: float, block_len: int, f0_hz: float, rng: np.random.Generator) -> np.ndarray:
    """
    Continuous-wave (CW) tone jammer: e^{j(2π f0 t + φ)} with random phase.
    """
    t = np.arange(block_len, dtype=np.float64) / fs
    phi = rng.uniform(0.0, TWOPI)
    return np.exp(1j * (TWOPI * f0_hz * t + phi)).astype(np.complex64)


def swept_cw_jammer(
    fs: float, block_len: int, f_center_hz: float, bw_hz: float,
    sweep_ms: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Linear chirp (swept CW) centered at f_center_hz with sweep bandwidth bw_hz and period sweep_ms.

    Model
    -----
    f(t_eff) = f_start + k * t_eff, where:
      - f_start = f_center - bw/2
      - k = ± bw / T_sweep  (random sweep direction)
      - t_eff = random offset inside the sweep period + time vector

    Returns complex64 phasor at baseband.
    """
    T_sweep = max(sweep_ms * 1e-3, 1e-6)           # guard against zero sweep time
    k = (bw_hz / T_sweep) * (1.0 if rng.random() < 0.5 else -1.0)
    f_start = f_center_hz - 0.5 * bw_hz

    t = np.arange(block_len, dtype=np.float64) / fs
    t0 = rng.uniform(0.0, T_sweep)                  # random phase within sweep
    t_eff = t0 + t

    phi0 = rng.uniform(0.0, TWOPI)
    phase = TWOPI * (f_start * t_eff + 0.5 * k * t_eff**2) + phi0
    return np.exp(1j * phase).astype(np.complex64)


def pulsed_tone_jammer(
    fs: float, block_len: int, f0_hz: float, duty: float, period_ms: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Pulsed CW tone: a tone_jammer gated by a rectangular pulse train with given duty and period.
    """
    tone = tone_jammer(fs, block_len, f0_hz, rng)
    Tper = max(period_ms * 1e-3, 1e-6)              # pulse period [s]

    t = np.arange(block_len, dtype=np.float64) / fs
    t_off = rng.uniform(0.0, Tper)                  # random pulse phase
    gate = (((t + t_off) % Tper) < (duty * Tper)).astype(np.float32)

    return (tone * gate.astype(np.complex64)).astype(np.complex64)


def add_awgn(x: np.ndarray,
             snr_db: float,
             rng: np.random.Generator,
             p_sig_ref: Optional[float] = None) -> np.ndarray:
    """
    Add circularly symmetric complex AWGN to achieve target SNR on a reference signal power.

    Parameters
    ----------
    x : np.ndarray[complex]
        Input block to which noise will be added.
    snr_db : float
        Target SNR in dB, defined as P_ref / P_noise.
    rng : np.random.Generator
        Random generator for the noise.
    p_sig_ref : Optional[float]
        Reference signal power P_ref. If None, P_ref is taken as the power of x.

    Note
    ----
    - SNR is defined here as P_ref / P_noise.
    - Noise is i.i.d. circular complex Gaussian (σ^2 per complex dim = P_noise).
    """
    if p_sig_ref is None:
        p_sig_ref = np.mean(np.abs(x.astype(np.complex128))**2).real

    snr_lin = 10.0 ** (snr_db / 10.0)
    p_noise = p_sig_ref / snr_lin
    sigma = math.sqrt(p_noise / 2.0)                # per real/imag component
    n = (rng.normal(0.0, sigma, x.shape) + 1j * rng.normal(0.0, sigma, x.shape)).astype(np.complex64)
    return (x + n).astype(np.complex64)

def mix_jammer(x_sig: np.ndarray, jam: np.ndarray, jsr_db: float) -> np.ndarray:
    """
    Scale jammer to achieve a specified JSR (P_jam / P_sig) at the block level, then add to signal.

    Parameters
    ----------
    x_sig : np.ndarray[complex]
        Clean signal block.
    jam  : np.ndarray[complex]
        Jammer block (unit-power-ish), same length.
    jsr_db : float
        Target Jammer-to-Signal power ratio in dB (block-wise).

    Returns
    -------
    y : np.ndarray[complex64]
        x_sig + scaled_jammer
    """
    p_sig = np.mean(np.abs(x_sig.astype(np.complex128))**2).real
    p_jam = np.mean(np.abs(jam.astype(np.complex128))**2).real + 1e-20  # avoid division by zero
    target = p_sig * (10.0 ** (jsr_db / 10.0))
    scale = math.sqrt(target / p_jam)
    return (x_sig + scale * jam).astype(np.complex64)

# ---------------------------
# Dataset generation
# ---------------------------
def gen_split(
    n_items: int,
    fs: float,
    block_len: int,
    prn_low: int,
    prn_high: int,
    doppler_max_hz: float,
    snr_db: Optional[float],
    jammer: str,
    jsr_db: Optional[float],
    jsr_min: Optional[float], jsr_max: Optional[float],
    jam_f0_min: float, jam_f0_max: float,
    jam_bw_min: float, jam_bw_max: float,
    jam_sweep_ms_min: float, jam_sweep_ms_max: float,
    jam_duty_min: float, jam_duty_max: float,
    jam_period_ms_min: float, jam_period_ms_max: float,
    seed: int
):
    """
    Synthesize one split (train/val/test) of (X, Y) blocks plus metadata.

    Generation per item
    -------------------
    1) Draw PRN ∈ [prn_low, prn_high], Doppler ∈ [−doppler_max, +doppler_max]
    2) Create clean block Y (complex baseband)
    3) If jammer enabled, draw per-item JSR (fixed or uniform range) and jammer parameters,
       mix jammer to reach JSR (block-wise)
    4) If --snr-db is set, add AWGN to X only (Y remains clean)
    5) Store I/Q in the last dimension

    Notes
    -----
    - `meta['jam_meta']` contains per-item jammer parameters for this split.
    - In the top-level NPZ meta, only a subset of split meta is retained (see main()).
    """
    rng = np.random.default_rng(seed)

    # Ensure fs is an integer multiple of CHIP_RATE so sps is integer
    sps = int(round(fs / CHIP_RATE))
    assert abs(fs - (CHIP_RATE * sps)) < 1.0, "fs must be an integer multiple of 1.023 MHz."

    # Allocate tensors: last dim packs [I, Q]
    X = np.empty((n_items, block_len, 2), dtype=np.float32)
    Y = np.empty_like(X)

    prns_used, dopplers, jam_meta = [], [], []

    for i in range(n_items):
        # Random PRN and Doppler for this example
        prn = int(rng.integers(prn_low, prn_high + 1))
        fd = float(rng.uniform(-doppler_max_hz, doppler_max_hz))

        # Clean signal (target)
        # Clean signal (target)
        clean = synthesize_block(fs, block_len, prn, sps, fd, rng)

        # Reference clean-signal power used for both JSR (jammer scaling)
        # and SNR (thermal noise level), so that AWGN level is independent
        # of whether a jammer is present.
        p_clean = np.mean(np.abs(clean.astype(np.complex128))**2).real

        observed = clean.copy()

        # Choose per-item JSR if a range is provided; else use fixed jsr_db (may be None)
        _jsr_db = jsr_db
        if _jsr_db is None and jsr_min is not None and jsr_max is not None:
            _jsr_db = float(rng.uniform(jsr_min, jsr_max))

        jm = {"type": "none"}
        if jammer != "none" and (_jsr_db is not None):
            # Draw jammer parameters and mix at target JSR
            if jammer == "tone":
                f0 = float(rng.uniform(jam_f0_min, jam_f0_max))
                jam = tone_jammer(fs, block_len, f0, rng)
                observed = mix_jammer(observed, jam, _jsr_db)
                jm = {"type": "tone", "f0_hz": f0, "jsr_db": _jsr_db}

            elif jammer == "swept_cw":
                f0 = float(rng.uniform(jam_f0_min, jam_f0_max))
                bw = float(rng.uniform(jam_bw_min, jam_bw_max))
                sw_ms = float(rng.uniform(jam_sweep_ms_min, jam_sweep_ms_max))
                jam = swept_cw_jammer(fs, block_len, f0, bw, sw_ms, rng)
                observed = mix_jammer(observed, jam, _jsr_db)
                jm = {"type": "swept_cw", "f_center_hz": f0, "bw_hz": bw, "sweep_ms": sw_ms, "jsr_db": _jsr_db}

            elif jammer == "pulsed":
                f0 = float(rng.uniform(jam_f0_min, jam_f0_max))
                duty = float(rng.uniform(jam_duty_min, jam_duty_max))
                per_ms = float(rng.uniform(jam_period_ms_min, jam_period_ms_max))
                jam = pulsed_tone_jammer(fs, block_len, f0, duty, per_ms, rng)
                observed = mix_jammer(observed, jam, _jsr_db)
                jm = {"type": "pulsed", "f0_hz": f0, "duty": duty, "period_ms": per_ms, "jsr_db": _jsr_db}

            else:
                raise ValueError(f"Unknown jammer: {jammer}")

        # Add AWGN after jamming or on clean-only.
        # SNR is defined w.r.t. the CLEAN signal power p_clean,
        # so noise level does not depend on JSR.
        if snr_db is not None:
            observed = add_awgn(observed, snr_db, rng, p_sig_ref=p_clean)


        # Pack complex to float32 I/Q
        X[i, :, 0] = observed.real.astype(np.float32)
        X[i, :, 1] = observed.imag.astype(np.float32)
        Y[i, :, 0] = clean.real.astype(np.float32)
        Y[i, :, 1] = clean.imag.astype(np.float32)

        # Tracking vectors for metadata
        prns_used.append(prn)
        dopplers.append(fd)
        jam_meta.append(jm)

    # Split-level metadata (note _jsr_db is the last sampled JSR if min/max were used)
    meta = {
        "fs": fs,
        "chip_rate": CHIP_RATE,
        "samples_per_chip": sps,
        "block_len": block_len,
        "block_ms": 1000.0 * block_len / fs,
        "prn_range": [prn_low, prn_high],
        "prns_used": prns_used,      # per-item PRNs (not persisted in top-level meta)
        "dopplers": dopplers,        # per-item Dopplers (not persisted in top-level meta)
        "snr_db": snr_db,
        "jammer": jammer,
        "jsr_db": _jsr_db,           # if a range was used, this reflects the last item's draw
        "jam_meta": jam_meta,        # per-item jammer params (not persisted in top-level meta)
        "notes": "Ideal GPS L1 C/A. Jammer added to X only when enabled. Y is clean.",
    }
    return X, Y, meta


def main():
    # -----------------------
    # CLI and argument parse
    # -----------------------
    ap = argparse.ArgumentParser(description="Ideal GPS L1 C/A dataset generator (pre-correlation), with optional jammers.")
    ap.add_argument("--out", type=str, default="artifacts/gnss_l1.npz", help="Output NPZ path.")
    ap.add_argument("--n-train", type=int, default=450000, help="Number of training items.")
    ap.add_argument("--n-val", type=int, default=25000, help="Number of validation items.")
    ap.add_argument("--n-test", type=int, default=25000, help="Number of test items.")
    ap.add_argument("--block-ms", type=float, default=1.0, help="Block length in milliseconds.")
    ap.add_argument("--fs", type=float, default=CHIP_RATE * DEFAULT_SPS, help="Sample rate [Hz]. Must be integer*1.023e6.")
    ap.add_argument("--prn-low", type=int, default=1)
    ap.add_argument("--prn-high", type=int, default=32)
    ap.add_argument("--doppler-max-hz", type=float, default=5000.0, help="Max |Doppler| [Hz] at baseband.")
    ap.add_argument("--snr-db", type=float, default=None, help="If set, add AWGN at this SNR (dB) to X only.")

    # Jammer args (all relative to baseband)
    ap.add_argument("--jammer", type=str, default="none", choices=["none", "tone", "swept_cw", "pulsed"])
    ap.add_argument("--jsr-db", type=float, default=None, help="Fixed JSR in dB (applied to X).")
    ap.add_argument("--jsr-min", type=float, default=None, help="Lower bound for per-item JSR sampling [dB].")
    ap.add_argument("--jsr-max", type=float, default=None, help="Upper bound for per-item JSR sampling [dB].")

    # Tone / swept-CW / pulsed parameter ranges
    ap.add_argument("--jam-f0-min", type=float, default=-6.0e5, help="Min jammer center freq (Hz) relative to baseband.")
    ap.add_argument("--jam-f0-max", type=float, default=+6.0e5, help="Max jammer center freq (Hz).")
    ap.add_argument("--jam-bw-min", type=float, default=2.0e5, help="Min sweep BW for swept_cw (Hz).")
    ap.add_argument("--jam-bw-max", type=float, default=1.2e6, help="Max sweep BW for swept_cw (Hz).")
    ap.add_argument("--jam-sweep-ms-min", type=float, default=1.0, help="Min sweep time (ms) for swept_cw.")
    ap.add_argument("--jam-sweep-ms-max", type=float, default=10.0, help="Max sweep time (ms) for swept_cw.")
    ap.add_argument("--jam-duty-min", type=float, default=0.1, help="Min duty for pulsed jammer (0..1).")
    ap.add_argument("--jam-duty-max", type=float, default=0.5, help="Max duty for pulsed jammer (0..1).")
    ap.add_argument("--jam-period-ms-min", type=float, default=0.2, help="Min pulse period (ms).")
    ap.add_argument("--jam-period-ms-max", type=float, default=2.0, help="Max pulse period (ms).")

    ap.add_argument("--seed", type=int, default=1337, help="Base RNG seed. Each split uses seed+k.")

    args = ap.parse_args()

    # -----------------------
    # Derived parameters
    # -----------------------
    fs = float(args.fs)
    block_len = int(round(args.block_ms * 1e-3 * fs))
    if block_len <= 0:
        raise ValueError("block-ms too small for fs.")

    # Enforce integer samples-per-chip
    sps = fs / CHIP_RATE
    if abs(sps - round(sps)) > 1e-6:
        raise ValueError(f"fs must be an integer multiple of 1.023e6 (got sps={sps}). Try fs=1.023e6*integer.")

    # -----------------------
    # Generate splits
    # -----------------------
    Xtr, Ytr, meta_tr = gen_split(
        args.n_train, fs, block_len, args.prn_low, args.prn_high,
        args.doppler_max_hz, args.snr_db,
        args.jammer, args.jsr_db, args.jsr_min, args.jsr_max,
        args.jam_f0_min, args.jam_f0_max,
        args.jam_bw_min, args.jam_bw_max,
        args.jam_sweep_ms_min, args.jam_sweep_ms_max,
        args.jam_duty_min, args.jam_duty_max,
        args.jam_period_ms_min, args.jam_period_ms_max,
        seed=args.seed + 0
    )

    Xva, Yva, meta_va = gen_split(
        args.n_val, fs, block_len, args.prn_low, args.prn_high,
        args.doppler_max_hz, args.snr_db,
        args.jammer, args.jsr_db, args.jsr_min, args.jsr_max,
        args.jam_f0_min, args.jam_f0_max,
        args.jam_bw_min, args.jam_bw_max,
        args.jam_sweep_ms_min, args.jam_sweep_ms_max,
        args.jam_duty_min, args.jam_duty_max,
        args.jam_period_ms_min, args.jam_period_ms_max,
        seed=args.seed + 1
    )

    Xte, Yte, meta_te = gen_split(
        args.n_test, fs, block_len, args.prn_low, args.prn_high,
        args.doppler_max_hz, args.snr_db,
        args.jammer, args.jsr_db, args.jsr_min, args.jsr_max,
        args.jam_f0_min, args.jam_f0_max,
        args.jam_bw_min, args.jam_bw_max,
        args.jam_sweep_ms_min, args.jam_sweep_ms_max,
        args.jam_duty_min, args.jam_duty_max,
        args.jam_period_ms_min, args.jam_period_ms_max,
        seed=args.seed + 2
    )

    # -----------------------
    # Assemble top-level meta
    # -----------------------
    meta = {
        "generator": "data_gnss.py",
        "fs": fs,
        "chip_rate": CHIP_RATE,
        "samples_per_chip": int(round(fs / CHIP_RATE)),
        "block_len": block_len,
        "block_ms": args.block_ms,
        "prn_low": args.prn_low,
        "prn_high": args.prn_high,
        "doppler_max_hz": args.doppler_max_hz,
        "snr_db": args.snr_db,
        "jammer": args.jammer,
        "jsr_db": args.jsr_db,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "n_test": args.n_test,
        # Keep split meta but drop large per-item lists to keep JSON small
        "meta_tr": {k: v for k, v in meta_tr.items() if k not in ("prns_used", "jam_meta")},
        "meta_va": {k: v for k, v in meta_va.items() if k not in ("prns_used", "jam_meta")},
        "meta_te": {k: v for k, v in meta_te.items() if k not in ("prns_used", "jam_meta")},
    }

    # -----------------------
    # Save NPZ and log
    # -----------------------
    np.savez_compressed(
        args.out,
        Xtr=Xtr, Ytr=Ytr,
        Xva=Xva, Yva=Yva,
        Xte=Xte, Yte=Yte,
        meta=json.dumps(meta)
    )
    print(f"[done] saved -> {args.out}")
    print(f"shapes: train {Xtr.shape}, val {Xva.shape}, test {Xte.shape} | block={args.block_ms:.3f} ms | sps={int(round(fs/CHIP_RATE))}")

    if args.jammer == "none":
        print("No jammer.")
    else:
        if args.jsr_db is None and (args.jsr_min is not None and args.jsr_max is not None):
            print(f"Jammer: {args.jammer} | JSR∈[{args.jsr_min},{args.jsr_max}] dB")
        else:
            print(f"Jammer: {args.jammer} | JSR={args.jsr_db} dB")

    if args.snr_db is None:
        print("No AWGN: X = Y (if no jammer).")
    else:
        print(f"AWGN added at SNR={args.snr_db:.1f} dB (to X).")


if __name__ == "__main__":
    main()
