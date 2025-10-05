#!/usr/bin/env python3
"""
Visualize NPZ datasets produced by data.py
- Constellation (noisy vs clean)
- Time-domain I/Q traces
- Optional PSD and spectrogram

Quality-of-life:
- Auto-read fs/block_len from meta['cfg'] if present
- Optional sidecar JSONL (same basename + .jsonl) for per-example SNR/JSR/sweep
- Optional phase alignment so clean BPSK lands at (±1, 0)

Examples (PowerShell / Bash):
  python data_visualization.py --file artifacts/gnss_synth_sweepcw_200k.npz --split train --random 4 --align_phase
  python data_visualization.py --file artifacts/gnss_synth_sweepcw_200k.npz --split val --idx 42 --align_phase --show_psd --show_spec
"""
from __future__ import annotations
import argparse, json, os, sys
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# -----------------
# Helpers & I/O
# -----------------

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    Xtr, Ytr = d['Xtr'], d['Ytr']
    Xva, Yva = d['Xva'], d['Yva']
    meta = json.loads(d['meta'].item()) if 'meta' in d else {}
    # Defaults
    meta_train, meta_val = [], []

    # Try sidecar JSONL (optional)
    sidecar = path + '.jsonl'
    if os.path.isfile(sidecar):
        try:
            with open(sidecar, 'r', encoding='utf-8') as f:
                lines = [json.loads(line) for line in f]
            meta_train = [m for m in lines if m.get('split') == 'train']
            meta_val   = [m for m in lines if m.get('split') == 'val']
            # Trim to array lengths
            meta_train = meta_train[: len(Xtr)]
            meta_val   = meta_val[: len(Xva)]
            print(f"Loaded sidecar meta: {len(meta_train)} train, {len(meta_val)} val")
        except Exception as e:
            print(f"Sidecar read failed ({e}); continuing without it.")

    # Normalize fs and block_len in meta
    cfg = meta.get('cfg', {}) if isinstance(meta.get('cfg', {}), dict) else {}
    if 'fs' not in meta and 'fs' in cfg:
        meta['fs'] = cfg['fs']
    if 'block_len' not in meta and 'block_len' in cfg:
        meta['block_len'] = int(cfg['block_len'])

    return Xtr, Ytr, Xva, Yva, meta, meta_train, meta_val


def to_complex(row: np.ndarray) -> np.ndarray:
    """row is shape (2*block_len,) with [Re, Im] interleaved last dim after reshape.
    Here we expect row shaped (2*L,), where it was produced by stack([Re,Im]).reshape(-1).
    """
    L2 = row.shape[0]
    assert L2 % 2 == 0, "Row length must be even (contains Re,Im pairs)."
    L = L2 // 2
    r = row.reshape(L, 2)
    return (r[:,0] + 1j * r[:,1]).astype(np.complex64)


def derotate_by_clean(x_cplx: np.ndarray, y_cplx: np.ndarray):
    """Rotate both so that the *clean* block's average phase is 0 (align to +I)."""
    theta = -np.angle(np.mean(y_cplx.astype(np.complex64)))
    rot = np.exp(1j * theta).astype(np.complex64)
    return x_cplx * rot, y_cplx * rot, theta


def fmt_meta(meta: dict, key: str, default='NA', fmt='{:.2f}'):
    v = meta.get(key, None)
    if v is None:
        return default
    try:
        return fmt.format(v)
    except Exception:
        return str(v)


def first_of(meta: dict, keys, default='NA', fmt='{:.0f}'):
    for k in keys:
        if k in meta and meta[k] is not None:
            try: return fmt.format(meta[k])
            except Exception: return str(meta[k])
    return default

# -----------------
# Plotters
# -----------------

def plot_example(idx: int, X: np.ndarray, Y: np.ndarray, meta_global: dict, meta_row: dict | None, args):
    fs = float(meta_global.get('fs', 1.0))
    L = int(meta_global.get('block_len', X.shape[1]//2))

    x = to_complex(X[idx])
    y = to_complex(Y[idx])

    theta = None
    if args.align_phase:
        x, y, theta = derotate_by_clean(x, y)

    t = np.arange(L) / fs

    # Figure layout
    nrows = 2 + int(args.show_psd) + int(args.show_spec)
    fig = plt.figure(figsize=(10, 2.6*nrows))
    gs = GridSpec(nrows, 2, figure=fig, height_ratios=[1]*nrows)

    # (1) Time-domain
    ax = fig.add_subplot(gs[0, :])
    ax.plot(t*1e3, x.real, label='I (noisy)')
    ax.plot(t*1e3, x.imag, label='Q (noisy)')
    ax.plot(t*1e3, y.real, '--', label='I (clean)', alpha=0.9)
    ax.set_title('Time-domain I/Q')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', ncol=3)

    # (2) Constellation
    axc = fig.add_subplot(gs[1, 0])
    axc.scatter(x.real, x.imag, s=14, alpha=0.6, label='Noisy/Jammed')
    axc.scatter(y.real, y.imag, s=32, marker='o', c='orange', label='Clean')
    # guides
    lim = 1.6
    axc.set_xlim(-lim, lim); axc.set_ylim(-lim, lim)
    axc.axhline(0, color='k', lw=0.6, alpha=0.4)
    axc.axvline(0, color='k', lw=0.6, alpha=0.4)
    th = np.linspace(0, 2*np.pi, 256)
    axc.plot(np.cos(th), np.sin(th), ls='--', lw=0.8, alpha=0.5)
    axc.set_aspect('equal', adjustable='box')
    axc.set_title('Constellation')
    axc.set_xlabel('I'); axc.set_ylabel('Q')
    axc.legend(loc='upper right')

    # (3) PSD (optional)
    if args.show_psd:
        from numpy.fft import fftshift, fft
        axp = fig.add_subplot(gs[1, 1])
        win = np.hanning(L)
        Xf = fftshift(fft(x * win))
        psd = 20*np.log10(np.abs(Xf)/np.sqrt(np.sum(win**2)))
        freqs = np.linspace(-fs/2, fs/2, L)
        axp.plot(freqs/1e3, psd)
        axp.set_title('PSD (complex baseband)')
        axp.set_xlabel('Freq [kHz]'); axp.set_ylabel('dB (a.u.)')
        axp.grid(True, alpha=0.3)

    # (4) Spectrogram (optional)
    if args.show_spec:
        axs = fig.add_subplot(gs[2 if args.show_psd else 1, :])
        # Using magnitude spectrogram of the complex baseband
        nper = max(16, min(64, L//4))
        nover = nper//2
        Pxx, freqs, bins, im = axs.specgram(x, NFFT=nper, Fs=fs, noverlap=nover, scale='dB', mode='magnitude', cmap='viridis')
        axs.set_title('Spectrogram (|X|)')
        axs.set_xlabel('Time [ms]')
        axs.set_ylabel('Freq [kHz]')
        axs.set_ylim(-fs/2/1e3, fs/2/1e3)
        # Re-label x in ms
        xt = axs.get_xticks()
        axs.set_xticklabels([f"{v*1e3:.2f}" for v in xt])

    # Title with metadata
    M = meta_row or {}
    title = (f"Example #{idx} | "
             f"SNR={fmt_meta(M,'snr_db','NA','{:.2f}')} dB  "
             f"JSR={fmt_meta(M,'jsr_db','NA','{:.2f}')} dB  "
             f"f0={first_of(M,['f0','jammer_f0'],'NA','{:.0f}')} Hz  "
             f"BW={first_of(M,['sweep_bw','jammer_bw'],'NA','{:.0f}')} Hz  "
             f"CFO={fmt_meta(M,'cfo','NA','{:.1f}')} Hz"
             f"{' | aligned' if args.align_phase else ''}")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    if args.save:
        base = os.path.splitext(os.path.basename(args.file))[0]
        out = f"viz_{base}_{args.split}_idx{idx}.png"
        fig.savefig(out, dpi=150)
        print('Saved', out)


# -----------------
# Main
# -----------------

def main():
    ap = argparse.ArgumentParser(description='Visualize synthetic BPSK datasets (constellation, time, optional PSD/spec).')
    ap.add_argument('--file', required=True, help='Path to NPZ dataset')
    ap.add_argument('--split', choices=['train','val'], default='train')
    ap.add_argument('--idx', type=int, default=None, help='Plot this single index')
    ap.add_argument('--random', type=int, default=0, help='Plot N random examples')
    ap.add_argument('--align_phase', action='store_true', help='Rotate so clean constellation aligns to +I (±1,0).')
    ap.add_argument('--show_psd', action='store_true', help='Add a PSD subplot')
    ap.add_argument('--show_spec', action='store_true', help='Add a spectrogram subplot')
    ap.add_argument('--save', action='store_true', help='Save figure(s) as PNG instead of only showing')
    args = ap.parse_args()

    Xtr, Ytr, Xva, Yva, meta, meta_train, meta_val = load_npz(args.file)

    X = Xtr if args.split == 'train' else Xva
    Y = Ytr if args.split == 'train' else Yva
    M = meta_train if args.split == 'train' else meta_val

    n = X.shape[0]
    if n == 0:
        sys.exit('Empty split?')

    if args.idx is not None:
        if not (0 <= args.idx < n):
            sys.exit(f'--idx out of range (0..{n-1})')
        plot_example(args.idx, X, Y, meta, (M[args.idx] if len(M)==n else {}), args)
        plt.show(); return

    k = int(args.random) if args.random else 4
    k = min(k, n)
    idxs = np.random.default_rng(1337).choice(n, size=k, replace=False)
    for i in idxs:
        plot_example(int(i), X, Y, meta, (M[i] if len(M)==n else {}), args)
    plt.show()

if __name__ == '__main__':
    main()
