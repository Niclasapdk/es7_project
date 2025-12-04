#!/usr/bin/env python3
"""
Diagnostic plots for gnss_datagen.py to show that it really generates GPS L1 C/A:

1) C/A autocorrelation + cross-correlation (PRN structure).
2) Code–Doppler correlation (CAF) for one synthetic block.
3) Recovered navigation bits over time after despreading.

Place this file in the same directory as gnss_datagen.py and run:

    python gnss_diag_plots.py
"""

import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------
# Make sure we can import gnss_datagen.py from the same folder
# ---------------------------------------------------------------------
this_dir = pathlib.Path(__file__).resolve().parent
if str(this_dir) not in sys.path:
    sys.path.insert(0, str(this_dir))

import gnss_datagen as gd  # type: ignore

# ---------------------------------------------------------------------
# Optional: tie fs/block_len to an existing NPZ dataset
# ---------------------------------------------------------------------
# Set this to the path of a small NPZ file if you want to match its fs/block_len,
# or leave as None to just use the canonical defaults from gnss_datagen.
NPZ_PATH = None  # e.g. this_dir / "artifacts/gnss_small.npz"

if NPZ_PATH is not None:
    npz_path = pathlib.Path(NPZ_PATH)
    data = np.load(npz_path)
    meta = json.loads(data["meta"].item())
    fs = float(meta["fs"])
    block_len = int(meta["block_len"])
else:
    fs = gd.CHIP_RATE * gd.DEFAULT_SPS      # e.g. 1.023e6 * 4 = 4.092 MHz
    block_ms = 1.0
    block_len = int(round(fs * block_ms * 1e-3))

sps = int(round(fs / gd.CHIP_RATE))
print(f"fs = {fs} Hz, block_len = {block_len} samples, sps = {sps}")

# Choose a PRN and Doppler for the synthetic example
PRN = 1
true_doppler = 1500.0  # Hz (any value within your doppler_max is fine)
rng = np.random.default_rng(1234)

t_block = np.arange(block_len) / fs

# ---------------------------------------------------------------------
# 1) C/A code autocorrelation + cross-correlation
# ---------------------------------------------------------------------
ca1 = gd.ca_code(PRN).astype(np.float64)
ca2 = gd.ca_code(2).astype(np.float64)  # some other PRN

# Full autocorrelation and cross-correlation, normalized by CA_LEN
r_auto = np.correlate(ca1, ca1, mode="full") / gd.CA_LEN
r_cross = np.correlate(ca1, ca2, mode="full") / gd.CA_LEN
lags = np.arange(-gd.CA_LEN + 1, gd.CA_LEN)

fig1, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].stem(lags, r_auto, basefmt=" ")
axes[0].set_title(f"C/A Autocorrelation (PRN {PRN})")
axes[0].set_xlabel("Lag (chips)")
axes[0].set_ylabel("Normalized correlation")

axes[1].stem(lags, r_cross, basefmt=" ")
axes[1].set_title("C/A Cross-correlation (PRN 1 vs 2)")
axes[1].set_xlabel("Lag (chips)")
axes[1].set_ylabel("Normalized correlation")

fig1.tight_layout()
fig1.savefig(this_dir / "diag1_ca_corr.png", dpi=150)
plt.close(fig1)

# ---------------------------------------------------------------------
# 2) Code–Doppler CAF for one synthetic block (include nav bits)
# ---------------------------------------------------------------------
PRN = 1
true_doppler = 1500.0  # Hz
CAF_SEED = 1234        # any seed; used consistently below

# --- generate one clean block with the real generator ---
rng_block = np.random.default_rng(CAF_SEED)
clean_block = gd.synthesize_block(
    fs=fs,
    block_len=block_len,
    prn=PRN,
    sps=sps,
    doppler_hz=true_doppler,
    rng=rng_block,
    allow_frac_code_phase=False,  # chip phase 0
)

t_block = np.arange(block_len, dtype=np.float64) / fs

# --- reconstruct the exact nav sequence used inside synthesize_block ---
# same seed, same call order as in synthesize_block:
rng_nav = np.random.default_rng(CAF_SEED)
# start_chip_phase draw is skipped because allow_frac_code_phase=False
nav = gd.build_nav_bits(block_len, sps, rng_nav)
# (next call inside synthesize_block would be rng.uniform for phi0)

# C/A epoch at sps samples, then combine with nav
ca_epoch = np.repeat(gd.ca_code(PRN).astype(np.float64), sps)
code_nav = ca_epoch * nav  # length = block_len

# Search grid
code_phases = np.arange(0, block_len, sps)  # one value per chip
dopplers = np.linspace(true_doppler - 4000, true_doppler + 4000, 61)

caf = np.zeros((len(dopplers), len(code_phases)), dtype=np.complex128)

for i, fd in enumerate(dopplers):
    # Note +j here so the peak appears at +true_doppler
    carrier = np.exp(1j * 2 * np.pi * fd * t_block)
    for j, tau in enumerate(code_phases):
        shifted = np.roll(code_nav, -tau)
        replica = shifted * carrier
        caf[i, j] = np.vdot(replica, clean_block)

caf_mag = np.abs(caf)
caf_mag_db = 20 * np.log10(caf_mag / caf_mag.max() + 1e-12)

fig2, ax2 = plt.subplots(figsize=(6, 4))
extent = [
    code_phases[0] / sps,   # chips
    code_phases[-1] / sps,
    dopplers[0],
    dopplers[-1],
]

vmax = 0
vmin = -40  # show only top 40 dB

im = ax2.imshow(
    caf_mag_db,
    extent=extent,
    aspect="auto",
    origin="lower",
    vmin=vmin,
    vmax=vmax,
)
ax2.set_xlabel("Code phase (chips)")
ax2.set_ylabel("Doppler (Hz)")
ax2.set_title(f"Code–Doppler Correlation (PRN {PRN})")
fig2.colorbar(im, ax=ax2, label="Correlation (dB rel. max)")
fig2.tight_layout()
fig2.savefig(this_dir / "diag2_caf.png", dpi=150)
plt.close(fig2)


# ---------------------------------------------------------------------
# 3) Recovered navigation bits over time
# ---------------------------------------------------------------------
# Generate several nav bits worth of signal (e.g. 8 bits = 160 ms)
n_bits = 8
bit_period_s = 1.0 / gd.NAV_BIT_RATE  # 0.02 s
total_time_s = n_bits * bit_period_s
nav_len = int(round(fs * total_time_s))

t_nav = np.arange(nav_len) / fs

clean_nav = gd.synthesize_block(
    fs=fs,
    block_len=nav_len,
    prn=PRN,
    sps=sps,
    doppler_hz=true_doppler,
    rng=rng,
    allow_frac_code_phase=False,  # again, start at chip 0
)

# Local C/A code aligned to chip phase 0
code_nav = gd.tile_ca_samples(PRN, sps, nav_len, start_chip_phase=0.0)

# Despread: wipe off code and known Doppler
despread = clean_nav * code_nav * np.exp(-1j * 2 * np.pi * true_doppler * t_nav)

samples_per_bit = int(round(fs * bit_period_s))
n_bits_eff = nav_len // samples_per_bit

# Integrate 20 ms at a time
accums = np.zeros(n_bits_eff, dtype=np.complex128)
for k in range(n_bits_eff):
    start = k * samples_per_bit
    stop = start + samples_per_bit
    accums[k] = despread[start:stop].sum()

# Unknown carrier phase → estimate from first bit and rotate everything
phi_est = np.angle(accums[0])
accums_rot = accums * np.exp(-1j * phi_est)

# Hard decision on the real part → ±1 nav bits
bits_est = np.sign(accums_rot.real)
bit_times_ms = (np.arange(n_bits_eff) + 0.5) * bit_period_s * 1e3

fig3, ax3 = plt.subplots(figsize=(7, 3))
ax3.step(bit_times_ms, bits_est, where="mid", label="Estimated nav bits (±1)")
ax3.set_xlabel("Time (ms)")
ax3.set_ylabel("Bit value")
ax3.set_title(f"Recovered Navigation Bits (PRN {PRN})")
ax3.set_ylim(-1.5, 1.5)
ax3.grid(True, axis="x", linestyle="--", alpha=0.5)
ax3.legend(loc="upper right")
fig3.tight_layout()
fig3.savefig(this_dir / "diag3_nav_bits.png", dpi=150)
plt.close(fig3)

print("Saved plots in", this_dir)
print("  diag1_ca_corr.png   (C/A autocorr + cross-corr)")
print("  diag2_caf.png       (Code–Doppler CAF)")
print("  diag3_nav_bits.png  (Recovered nav bits)")