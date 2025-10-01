#!/usr/bin/env python3
# data_visualization.py (robust to legacy NPZ without meta_train/meta_val)
#
# Usage examples:
#   python data_visualization.py --file artifacts/bpsk_sweep_dataset.npz --split train --idx 0
#   python data_visualization.py --file artifacts/l1_bpsk_cw_dataset.npz --split val --random 4
#   python data_visualization.py --file artifacts/l1_bpsk_cw_dataset.npz --split train --idx 10 --rf

import argparse, json, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram, get_window

def _safe_json_load(arr, default):
    """Handle json strings saved as 0-d numpy arrays, Python objects, or missing."""
    if arr is None:
        return default
    try:
        # np.savez may store as 0-d array of dtype=object or string
        if hasattr(arr, "item"):
            return json.loads(arr.item())
        else:
            return json.loads(arr.tolist())
    except Exception:
        return default

def load_npz(path):
    data = np.load(path, allow_pickle=True)

    # Core arrays (required)
    Xtr = data["Xtr"]
    Ytr = data["Ytr"]
    Xva = data["Xva"]
    Yva = data["Yva"]

    # Global meta (optional)
    meta = _safe_json_load(data["meta"] if "meta" in data else None, default={})

    # Per-example metas (optional; synthesize if missing)
    if "meta_train" in data:
        meta_train = _safe_json_load(data["meta_train"], default=None)
    else:
        meta_train = None
    if "meta_val" in data:
        meta_val = _safe_json_load(data["meta_val"], default=None)
    else:
        meta_val = None

    if meta_train is None:
        meta_train = [dict() for _ in range(Xtr.shape[0])]
    if meta_val is None:
        meta_val = [dict() for _ in range(Xva.shape[0])]

    return (Xtr, Ytr, Xva, Yva, meta, meta_train, meta_val)

def flat_to_complex(x_flat):
    # x_flat shape: (2*block_len,), layout [Re0, Im0, Re1, Im1, ...]
    x2 = x_flat.reshape(-1, 2)
    return x2[:,0].astype(np.float64) + 1j * x2[:,1].astype(np.float64)

def pick_examples(X, Y, metas, idx=None, n_random=0, rng=None):
    N = X.shape[0]
    if n_random > 0:
        rng = np.random.default_rng(None if rng is None else rng)
        idxs = rng.integers(0, N, size=n_random).tolist()
    elif idx is not None:
        if isinstance(idx, (list, tuple)):
            idxs = [int(i) for i in idx]
        else:
            idxs = [int(idx)]
    else:
        idxs = [0]
    ex = [(flat_to_complex(X[i]), flat_to_complex(Y[i]), metas[i] if i < len(metas) else {}) for i in idxs]
    return idxs, ex

def plot_example(xc, yc, meta, fs, idx, show_rf=False, fc_from_global=None):
    N = len(xc)
    t = np.arange(N) / fs

    # pull fields if present, else show NA
    def _fmt(key, fmt="{:.2f}"):
        v = meta.get(key, None)
        if v is None:
            return "NA"
        try:
            return fmt.format(v)
        except Exception:
            return str(v)

    title = (f"Example #{idx}  |  SNR={_fmt('snr_db')} dB  "
             f"JSR={_fmt('jsr_db')} dB  f0={_fmt('jammer_f0', '{:.0f}')} Hz  "
             f"BW={_fmt('jammer_bw','{:.0f}')} Hz  CFO={_fmt('cfo','{:.1f}')} Hz")
    fig = plt.figure(figsize=(12, 9))
    fig.suptitle(title)

    # 1) Time-domain I/Q
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(t*1e3, xc.real, label="I (noisy)")
    ax1.plot(t*1e3, xc.imag, label="Q (noisy)", alpha=0.8)
    ax1.plot(t*1e3, yc.real, label="I (clean)", linewidth=1, linestyle="--")
    ax1.set_xlabel("Time [ms]"); ax1.set_ylabel("Amplitude")
    ax1.set_title("Time-domain I/Q")
    ax1.grid(True); ax1.legend(loc="upper right", fontsize=8)

    # 2) Constellation
    ax2 = plt.subplot(2, 2, 2)
    step = max(1, N // 2000)
    ax2.scatter(xc.real[::step], xc.imag[::step], s=8, alpha=0.5, label="Noisy/Jammed")
    ax2.scatter(yc.real[::step], yc.imag[::step], s=8, alpha=0.6, label="Clean")
    ax2.axhline(0, color="k", linewidth=0.5); ax2.axvline(0, color="k", linewidth=0.5)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_title("Constellation")
    ax2.grid(True); ax2.legend(loc="upper right", fontsize=8)

    # 3) PSD (Welch) — works for complex signals too
    ax3 = plt.subplot(2, 2, 3)
    nperseg = min(256, N)
    win = get_window("hann", nperseg)
    f, Pxx = welch(xc, fs=fs, window=win, nperseg=nperseg, noverlap=nperseg//2,
                   return_onesided=False, scaling="density")
    ax3.plot(np.fft.fftshift(f)/1e3,
             10*np.log10(np.maximum(np.fft.fftshift(Pxx.real), 1e-20)))
    ax3.set_xlabel("Frequency [kHz]"); ax3.set_ylabel("PSD [dB/Hz]")
    ax3.set_title("PSD (complex baseband)")
    ax3.grid(True)

    # 4) Spectrogram (magnitude) to see any CW / sweep
    ax4 = plt.subplot(2, 2, 4)
    nperseg_s = min(128, N)
    noverlap = nperseg_s // 2
    f2, tt, Sxx = spectrogram(xc, fs=fs, window="hann", nperseg=nperseg_s,
                              noverlap=noverlap, return_onesided=False, mode="magnitude")
    im = ax4.pcolormesh(tt*1e3, np.fft.fftshift(f2)/1e3,
                        20*np.log10(np.maximum(np.fft.fftshift(Sxx, axes=0), 1e-12)),
                        shading="auto")
    ax4.set_xlabel("Time [ms]"); ax4.set_ylabel("Freq [kHz]")
    ax4.set_title("Spectrogram (|X|)")
    fig.colorbar(im, ax=ax4, label="dB")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Optional: reconstructed RF spectrum (needs fc)
    if show_rf:
        fc_meta = meta.get("fc", None)
        fc = fc_meta if (fc_meta is not None) else fc_from_global
        if not fc:
            print("⚠ RF view: 'fc' not found in meta; skipping RF spectrum.")
            return
        rf = np.real(yc * np.exp(1j*2*np.pi*fc*t))
        nfft = 4096 if N >= 4096 else int(2**math.ceil(math.log2(max(N, 512))))
        RF = np.fft.fftshift(np.fft.fft(rf, n=nfft))
        freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1/fs))

        plt.figure(figsize=(10,4))
        plt.plot(freqs/1e3, 20*np.log10(np.abs(RF)+1e-12))
        plt.title(f"Reconstructed RF spectrum — expect tone near ±fc ≈ ±{fc/1e3:.1f} kHz")
        plt.xlabel("Frequency [kHz]"); plt.ylabel("Magnitude [dB]")
        plt.grid(True)
        plt.tight_layout()

def main():
    ap = argparse.ArgumentParser(description="Visualize I/Q dataset (robust loader).")
    ap.add_argument("--file", required=True, help="Path to .npz dataset")
    ap.add_argument("--split", choices=["train","val"], default="train")
    ap.add_argument("--idx", type=int, nargs="*", default=None, help="Index (or multiple) to plot")
    ap.add_argument("--random", type=int, default=0, help="Plot this many random examples instead of --idx")
    ap.add_argument("--rf", action="store_true", help="Also show a reconstructed RF spectrum (requires 'fc' in meta)")
    args = ap.parse_args()

    Xtr, Ytr, Xva, Yva, meta, meta_train, meta_val = load_npz(args.file)
    fs = float(meta.get("fs", 1.0))  # fallback to 1.0 Hz if missing
    fc_global = float(meta.get("fc", 0.0)) if "fc" in meta else None

    if args.split == "train":
        X, Y, M = Xtr, Ytr, meta_train
    else:
        X, Y, M = Xva, Yva, meta_val

    idxs, examples = pick_examples(X, Y, M, idx=args.idx, n_random=args.random)
    print(f"Loaded {args.split}: {X.shape[0]} examples | block_len={X.shape[1]//2} | fs={fs:.0f} Hz")

    for i, (xc, yc, m) in zip(idxs, examples):
        print(f"- Plotting {args.split} example #{i}")
        plot_example(xc, yc, m, fs, i, show_rf=args.rf, fc_from_global=fc_global)

    plt.show()

if __name__ == "__main__":
    main()
