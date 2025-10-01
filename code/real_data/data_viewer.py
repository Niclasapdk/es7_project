
#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

# -------------------- I/O --------------------
def load_csv_complex(path, limit=None):
    df = pd.read_csv(path)
    if not set(["Time","I","Q"]).issubset(df.columns):
        raise ValueError("CSV must have columns: Time, I, Q")
    if limit is not None:
        df = df.head(int(limit))
    t = df["Time"].to_numpy()
    x = df["I"].to_numpy() + 1j*df["Q"].to_numpy()
    return t, x

# -------------------- Signal conditioning --------------------
def dc_and_lock(x):
    z = x - np.mean(x)
    theta = 0.5 * np.angle(np.mean(z**2))
    y = z * np.exp(-1j*theta)
    return y

def cfo_correct_bpsk(y):
    s2 = y**2
    ph = np.unwrap(np.angle(s2))
    n = np.arange(len(ph))
    mask = np.isfinite(ph)
    if np.count_nonzero(mask) < 10:
        return y, 0.0
    k, b = np.polyfit(n[mask], ph[mask], 1)
    w = 0.5 * k  # per-sample angular frequency
    y_corr = y * np.exp(-1j * w * np.arange(len(y)))
    return y_corr, w

def pca_align_to_I(y):
    X = np.vstack([np.real(y), np.imag(y)])
    C = np.cov(X)
    w, V = np.linalg.eigh(C)
    v = V[:, np.argmax(w)]
    ang = np.arctan2(v[1], v[0])
    return y * np.exp(-1j*ang), ang

def run_segments(z):
    r = np.real(z)
    s = np.sign(r)
    for i in range(1, len(s)):
        if s[i] == 0:
            s[i] = s[i-1]
    changes = np.flatnonzero(np.diff(s)!=0)
    starts = np.concatenate(([0], changes+1))
    ends   = np.concatenate((changes+1, [len(s)]))
    return starts, ends

def pick_symbol_samples(z):
    starts, ends = run_segments(z)
    lens = ends - starts
    if len(lens) == 0:
        return z
    thr = max(2, int(np.median(lens)/5)) if len(lens) > 10 else 2
    keep = lens >= thr
    starts = starts[keep]; ends = ends[keep]
    re = np.real(z)
    idx = []
    for a,b in zip(starts, ends):
        seg = slice(a,b)
        i_local = np.argmax(np.abs(re[seg]))
        idx.append(a + i_local)
    return z[np.array(idx, dtype=int)]

def estimate_A(syms):
    m = np.median(np.abs(np.real(syms)))
    return 1.0 if m == 0 else float(m)

# -------------------- Plots --------------------
def plot_time_domain(t, x, title, outpath, max_samples=None):
    if max_samples is not None and len(t) > max_samples:
        t = t[:max_samples]; x = x[:max_samples]
    fig = plt.figure(figsize=(9,3.5))
    ax = plt.gca()
    ax.plot(t, np.real(x), label="I")
    ax.plot(t, np.imag(x), label="Q")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title(title + " — Time (I & Q)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_constel(s_norm, title, outpath, lim=1.6):
    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()
    for spine in ["left","bottom"]:
        ax.spines[spine].set_position("center")
    for spine in ["right","top"]:
        ax.spines[spine].set_color("none")
    ax.set_aspect("equal", adjustable="box")
    ticks = np.arange(-1.5, 1.6, 0.5)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.grid(False)
    ax.annotate("", xy=(lim,0), xytext=(0,0), arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.annotate("", xy=(0,lim), xytext=(0,0), arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.set_xlabel("I", x=1.0, ha="right"); ax.set_ylabel("Q", y=1.0, va="top")
    circ = plt.Circle((0,0), 1.0, fill=False, linestyle="--", linewidth=1.0)
    ax.add_artist(circ)
    ax.scatter(np.real(s_norm), np.imag(s_norm), s=22, alpha=0.9)
    ax.scatter([1,-1],[0,0], s=140, zorder=3)
    ax.text(-1, 0, " 0", va="center", ha="left", fontsize=12)
    ax.text( 1, 0, " 1", va="center", ha="left", fontsize=12)
    ax.set_title(title + " — symbol-spaced (strict)", pad=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--outdir", default="rf_plots")
    ap.add_argument("--limit", type=int, default=None, help="Read at most N samples from file for processing")
    ap.add_argument("--gate", type=float, default=0.85, help="Keep symbols with |z| >= gate*A and |Q| <= (1-gate)*A")
    ap.add_argument("--time-samples", type=int, default=2000, help="Number of samples to show in time plot")
    args = ap.parse_args()

    p = Path(args.file)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    t, x = load_csv_complex(p, limit=args.limit)

    # --- Save time-domain plot (raw I/Q) ---
    time_png = outdir / f"{p.stem}_time.png"
    plot_time_domain(t, x, p.stem.replace("_"," "), time_png, max_samples=args.time_samples)

    # --- Constellation (strict symbol selection) ---
    y = dc_and_lock(x)
    y, w = cfo_correct_bpsk(y)
    z, ang = pca_align_to_I(y)
    sym = pick_symbol_samples(z)
    A = estimate_A(sym)
    amp = np.abs(sym); qleak = np.abs(np.imag(sym))
    keep = (amp >= args.gate * A) & (qleak <= (1-args.gate) * A)
    sym_kept = sym[keep]
    s_norm = sym_kept / A if A != 0 else sym_kept

    const_png = outdir / f"{p.stem}_bpsk_constellation_symbol_strict.png"
    plot_constel(s_norm, p.stem.replace("_"," "), const_png)

    print(f"Saved:\n  {time_png}\n  {const_png}\n"
          f"  Stats: kept {len(sym_kept)}/{len(sym)} syms, A≈{A:.4g}, CFO(rad/sample)≈{w:.3e}")

if __name__ == "__main__":
    main()
