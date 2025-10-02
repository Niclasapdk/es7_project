# This cell writes a ready-to-run Python script that validates a (clean) BPSK/GNSS-like signal.
# It loads a .npz dataset, builds/uses a PRN replica, and produces 4 diagnostics:
# 1) PSD, 2) Code correlation shape, 3) Delay–Doppler CAF heatmap, 4) C/N0 trend.
# The loader is defensive and tries common key layouts (e.g., X with I/Q channels, or complex 'iq').
# gnss_sanity_check.py
# Validate a clean BPSK/GNSS-like signal via PSD, code correlation, CAF, and C/N0.
# Niclas-friendly version: opinionated defaults for GPS L1 C/A (1.023 Mcps), but configurable.
#
# Usage examples:
#   python gnss_sanity_check.py --file artifacts/bpsk_sweep_dataset.npz --split train --idx 0 --fs 10.23e6
#   python gnss_sanity_check.py --file your_data.npz --key X --idx 0 --fs 10.23e6 --rc 1.023e6 --prn 1
#
# Outputs (PNG) saved next to the input file:
#   *_psd.png, *_corr.png, *_caf.png, *_cn0.png
#
# Notes:
# - If the dataset contains metadata keys like 'fs', 'chip_rate', 'rc', 'prn', 'doppler_hz', 'code_phase_samp', we use them.
# - If there's no true C/A code provided, we generate either the real GPS C/A (if --prn given) or a 1023-chip m-sequence.
# - C/N0 estimator: quick variance-ratio on non-coherent magnitudes of 1-ms prompt correlations, CN0 ≈ SNR_nc / Tcoh.
#   It's a rough sanity metric, good enough to see "flat vs falling" trends and jamming-like effects.
#

import os, sys, argparse, math, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def db(x):
    return 10*np.log10(np.maximum(x, 1e-30))

# ----------------------------
# GPS C/A generator (G1/G2) or fallback m-sequence
# ----------------------------
def ca_code(prn: int) -> np.ndarray:
    """
    Generate 1023-chip GPS L1 C/A code for PRN 1..37 (classic).
    Returns chips as +/-1 numpy array.
    """
    # G2 tap selections for PRNs 1..37 (phase taps), classic table:
    g2_taps = {
        1:(2,6), 2:(3,7), 3:(4,8), 4:(5,9), 5:(1,9), 6:(2,10), 7:(1,8), 8:(2,9), 9:(3,10),10:(2,3),
        11:(3,4),12:(5,6),13:(6,7),14:(7,8),15:(8,9),16:(9,10),17:(1,4),18:(2,5),19:(3,6),20:(4,7),
        21:(5,8),22:(6,9),23:(1,3),24:(4,6),25:(5,7),26:(6,8),27:(7,9),28:(8,10),29:(1,6),30:(2,7),
        31:(3,8),32:(4,9),33:(5,10),34:(4,10),35:(1,7),36:(2,8),37:(4,10)
    }
    if prn not in g2_taps:
        raise ValueError("PRN must be in 1..37 for this simple table.")
    tap1, tap2 = g2_taps[prn]
    # shift register helper (1-indexed taps per GPS convention)
    def step(reg, taps):
        # XOR of given taps (1-indexed)
        fb = 0
        for t in taps:
            fb ^= reg[t-1]
        out = reg[-1]
        reg[1:] = reg[:-1]
        reg[0] = fb
        return out

    # Init registers with ones
    G1 = np.ones(10, dtype=int)
    G2 = np.ones(10, dtype=int)
    chips = np.empty(1023, dtype=int)

    for i in range(1023):
        # G1 output is last bit
        g1_out = G1[-1]
        # G2 output is XOR of the two tap outputs per PRN
        g2_out = G2[tap1-1] ^ G2[tap2-1]
        chips[i] = g1_out ^ g2_out  # 0/1
        # feedbacks: G1 taps at 3,10 ; G2 taps at 2,3,6,8,9,10
        g1_fb = G1[2-1] ^ G1[9]
        g2_fb = G2[1] ^ G2[2] ^ G2[5] ^ G2[7] ^ G2[8] ^ G2[9]
        # Shift
        G1[1:] = G1[:-1]
        G1[0] = g1_fb
        G2[1:] = G2[:-1]
        G2[0] = g2_fb

    # Map 0->+1, 1->-1 (BPSK)
    return 1 - 2*chips

def mseq_1023()->np.ndarray:
    """Generate a 1023-chip m-sequence (maximal LFSR) as +/-1."""
    reg = np.ones(10, dtype=int)
    chips = np.empty(1023, dtype=int)
    for i in range(1023):
        chips[i] = reg[-1]
        fb = reg[2] ^ reg[9]  # taps for a 10-bit maximal LFSR variant
        reg[1:] = reg[:-1]
        reg[0] = fb
    chips = chips ^ 1  # invert to mix it up
    return 1 - 2*chips  # +/-1

# ----------------------------
# Data loader (defensive)
# ----------------------------
def load_iq_from_npz(path, split=None, idx=0, key=None):
    d = np.load(path, allow_pickle=True)
    meta = {}
    # hoover metadata if any
    for mk in ["fs","Fs","sample_rate","rc","chip_rate","prn","doppler_hz","code_phase_samp","center_if_hz"]:
        if mk in d: meta[mk]=float(d[mk])
    # prefer explicit key if provided
    array = None
    if key:
        if key not in d:
            raise KeyError(f"Key '{key}' not found in file. Available: {list(d.keys())}")
        array = d[key]
    else:
        # heuristics: common keys in ML datasets
        for k in ["X","iq","samples","data","signal","S","IQ"]:
            if k in d:
                array = d[k]; key=k; break
        if array is None:
            # if only one array present, take it
            only = [k for k in d.keys() if isinstance(d[k], np.ndarray)]
            if len(only)==1:
                key = only[0]; array = d[key]
            else:
                raise KeyError(f"Could not guess IQ key. Arrays present: {only}")
    # slice by split if 3-way dict-like
    if isinstance(array, np.ndarray):
        arr = array
    elif isinstance(array, np.void) or isinstance(array, dict):
        # unlikely, skip
        arr = np.array(array)
    else:
        arr = array

    # common shapes:
    # (N, L, 2) -> I/Q channels
    # (N, L) complex
    # (L, 2) single example
    if split is not None and split in d:
        # some datasets store splits separately (e.g., X['train'])
        arr = d[split]
    # pick idx if batched
    if arr.ndim==3 and arr.shape[-1]==2:
        # treat as (N, L, 2)
        sel = arr[idx]
        iq = sel[...,0] + 1j*sel[...,1]
    elif arr.ndim==2 and np.iscomplexobj(arr):
        # (N, L) complex
        iq = arr[idx]
    elif arr.ndim==2 and arr.shape[-1]==2:
        # (L, 2)
        iq = arr[:,0] + 1j*arr[:,1]
    elif arr.ndim==1 and np.iscomplexobj(arr):
        iq = arr
    else:
        # last resort: try to view last dim pairs as I/Q
        if arr.ndim>=2 and arr.shape[-1]%2==0:
            sel = arr[idx]
            half = sel.shape[-1]//2
            iq = sel[..., :half] + 1j*sel[..., half:]
        else:
            raise ValueError(f"Don't know how to interpret array with shape {arr.shape}")
    return iq.astype(np.complex64), meta

# ----------------------------
# PSD via Welch (simple)
# ----------------------------
def welch_psd(x, fs, nperseg=4096, noverlap=None):
    if noverlap is None:
        noverlap = nperseg//2
    n = len(x)
    step = nperseg - noverlap
    win = np.hanning(nperseg).astype(np.float64)
    scale = (win**2).sum()
    segs = []
    for start in range(0, n-nperseg+1, step):