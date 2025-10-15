#!/usr/bin/env python3
import argparse, json, math
import numpy as np

import matplotlib.pyplot as plt  # only used when --plots

CHIP_RATE = 1.023e6
CA_LEN = 1023
TWOPI = 2*math.pi

# --------- C/A code (PRN 1..32) ----------
PRN_G2_TAPS = {
    1:(2,6),2:(3,7),3:(4,8),4:(5,9),5:(1,9),6:(2,10),7:(1,8),8:(2,9),
    9:(3,10),10:(2,3),11:(3,4),12:(5,6),13:(6,7),14:(7,8),15:(8,9),16:(9,10),
    17:(1,4),18:(2,5),19:(3,6),20:(4,7),21:(5,8),22:(6,9),23:(1,3),24:(4,6),
    25:(5,7),26:(6,8),27:(7,9),28:(8,10),29:(1,6),30:(2,7),31:(3,8),32:(4,9),
}
def ca_code(prn:int)->np.ndarray:
    if prn not in PRN_G2_TAPS: raise ValueError("PRN out of range (1..32)")
    g1 = np.ones(10,dtype=np.uint8); g2 = np.ones(10,dtype=np.uint8)
    ta,tb = PRN_G2_TAPS[prn]
    out = np.empty(CA_LEN, dtype=np.int8)
    for _ in range(CA_LEN):
        g1o = g1[-1]; g2o = g2[-ta]^g2[-tb]
        chip = g1o ^ g2o
        out[_] = 1 if chip==0 else -1
        g1fb = g1[2]^g1[9]; g1[1:] = g1[:-1]; g1[0] = g1fb
        g2fb = g2[1]^g2[2]^g2[5]^g2[7]^g2[8]^g2[9]
        g2[1:] = g2[:-1]; g2[0] = g2fb
    return out.astype(np.int8)

def code_epoch_samples(prn:int, sps:int)->np.ndarray:
    # 1-ms epoch at sps samples/chip, ±1 float32
    return np.repeat(ca_code(prn).astype(np.float32), repeats=sps)

# --------- DSP helpers ----------
def welch_psd(x, fs, nperseg=1024):
    # simple Welch PSD mag^2 (no windowing bells/whistles)
    step = nperseg//2
    if len(x) < nperseg: return np.array([0.0]), np.array([0.0])
    segs = 1 + (len(x)-nperseg)//step
    acc = None
    for i in range(segs):
        s = x[i*step:i*step+nperseg]
        S = np.fft.rfft(s)
        P = (np.abs(S)**2)/ (nperseg*fs)
        acc = P if acc is None else acc+P
    Pxx = acc/segs
    f = np.fft.rfftfreq(nperseg, d=1/fs)
    return f, Pxx

def spectrogram_mag(x, fs, nperseg=512, noverlap=448):
    step = nperseg - noverlap
    cols = 1 + (len(x)-nperseg)//step if len(x)>=nperseg else 0
    if cols <= 0: return np.zeros((0,0)), np.array([]), np.array([])
    spec = []
    for c in range(cols):
        s = x[c*step:c*step+nperseg]
        S = np.fft.fftshift(np.fft.fft(s, n=nperseg))
        spec.append(20*np.log10(np.abs(S)+1e-12))
    spec = np.stack(spec, axis=1)
    freqs = np.fft.fftshift(np.fft.fftfreq(nperseg, d=1/fs))
    times = (np.arange(cols)*step)/fs
    return spec, freqs, times

def circular_corr(x, r):
    # circular correlation via FFT: corr[k] = sum_n x[n]*conj(r[n-k])
    X = np.fft.fft(x)
    R = np.fft.fft(r)
    c = np.fft.ifft(X * np.conj(R))
    return c

def measure_snr_jsr(x_complex, y_complex):
    # X = observed (signal + interference + noise), Y = clean signal
    # INP = interference+noise = X - Y
    inp = x_complex - y_complex
    p_sig = np.mean(np.abs(y_complex)**2).real + 1e-20
    p_inp = np.mean(np.abs(inp)**2).real + 1e-20
    snr_meas = 10*np.log10(p_sig/(p_inp))  # if only noise present: SNR
    jsr_meas = 10*np.log10(p_inp/p_sig)    # if only jammer present: JSR
    return snr_meas, jsr_meas

# --------- CAF (code-Doppler search) ----------
def caf_search(x_complex, fs, sps, prns=range(1,33), fd_min=-5000, fd_max=5000, fd_step=500):
    """
    Returns best (prn, fd, tau_idx, peak_val, peak_norm, corr_vector_for_best_fd)
    peak_norm = peak / len(x)
    """
    N = len(x_complex)
    t = np.arange(N)/fs
    best = {"prn":None,"fd":None,"tau":None,"peak":-1.0,"corr":None}
    for prn in prns:
        ref_epoch = code_epoch_samples(prn, sps)[:N]  # length N (1 ms assumed)
        for fd in np.arange(fd_min, fd_max+1e-9, fd_step):
            derot = np.exp(-1j*TWOPI*fd*t).astype(np.complex64)
            z = x_complex * derot
            # correlate with code (complex because NAV and carrier phase)
            c = circular_corr(z, ref_epoch.astype(np.complex64))
            mag = np.abs(c)
            k = int(np.argmax(mag))
            peak = float(mag[k])
            if peak > best["peak"]:
                best.update(prn=prn, fd=float(fd), tau=int(k), peak=peak, corr=c)
    best["peak_norm"] = best["peak"] / len(x_complex)
    return best

# --------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=str)
    ap.add_argument("--idx", type=int, default=0, help="Example index from test set (Xte/Yte).")
    ap.add_argument("--fd-min", type=float, default=-5000)
    ap.add_argument("--fd-max", type=float, default=5000)
    ap.add_argument("--fd-step", type=float, default=500)
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    meta = json.loads(str(data["meta"]))
    fs = float(meta["fs"])
    sps = int(meta["samples_per_chip"])
    X = data["Xte"]; Y = data["Yte"]
    x = X[args.idx,:,0].astype(np.float32) + 1j*X[args.idx,:,1].astype(np.float32)
    y = Y[args.idx,:,0].astype(np.float32) + 1j*Y[args.idx,:,1].astype(np.float32)

    # Power checks
    snr_meas, jsr_meas = measure_snr_jsr(x, y)

    # CAF blind acquisition
    best = caf_search(x, fs, sps,
                      prns=range(int(meta["prn_low"]), int(meta["prn_high"])+1),
                      fd_min=args.fd_min, fd_max=args.fd_max, fd_step=args.fd_step)

    # Side-lobe estimate (exclude ±2 samples around peak)
    corr_mag = np.abs(best["corr"])
    mask = np.ones_like(corr_mag, dtype=bool)
    m = 2
    mask[(best["tau"]-m):(best["tau"]+m+1)] = False
    sidelobe = float(np.max(corr_mag[mask])) if corr_mag.size>5 else 0.0
    peak_sidelobe_ratio = 20*np.log10((best["peak"]+1e-12)/(sidelobe+1e-12))

    print("=== VALIDATION REPORT ===")
    print(f"File         : {args.npz}")
    print(f"fs           : {fs/1e6:.3f} Msps | sps={sps} | N={x.size} samples (~{1000*x.size/fs:.3f} ms)")
    print(f"SNR_meas(dB) : {snr_meas:6.2f}  (treat as SNR if only AWGN; as -JSR if only jammer)")
    print(f"INR/JSR(dB)  : {jsr_meas:6.2f}  (treat as JSR if no AWGN)")
    print(f"CAF best PRN : {best['prn']}")
    print(f"CAF Doppler  : {best['fd']:6.1f} Hz (grid step {args.fd_step} Hz)")
    print(f"CAF code lag : {best['tau']} samples")
    print(f"CAF peak_norm: {best['peak_norm']:.3f} (expect ~0.9–1.0 at high SNR)")
    print(f"Peak/SL(dB)  : {peak_sidelobe_ratio:5.1f} dB (thumbtack should be >> 10 dB)")

    if args.plots:
        # PSD (observed X)
        f, Pxx = welch_psd(x, fs, nperseg=min(8192, len(x)))
        plt.figure()
        plt.semilogy(f/1e6, Pxx)
        plt.xlabel("Frequency (MHz)"); plt.ylabel("PSD")
        plt.title("Welch PSD of X (observed)")

        # Spectrogram (observed X)
        S, freqs, times = spectrogram_mag(x, fs, nperseg=1024, noverlap=896)
        if S.size>0:
            plt.figure()
            extent = [times[0]*1e3, times[-1]*1e3, freqs[0]/1e6, freqs[-1]/1e6]
            plt.imshow(S, aspect='auto', origin='lower', extent=extent)
            plt.colorbar(label="dB")
            plt.xlabel("Time (ms)"); plt.ylabel("Freq (MHz)")
            plt.title("Spectrogram of X")

        # CAF heatmap for best PRN (magnitude vs code lag & Doppler)
        prn = best["prn"]
        if prn is not None:
            # Build heatmap across Doppler with circular corr mags
            t = np.arange(len(x))/fs
            ref = code_epoch_samples(prn, sps)[:len(x)]
            fds = np.arange(args.fd_min, args.fd_max+1e-9, args.fd_step)
            H = []
            for fd in fds:
                z = x * np.exp(-1j*TWOPI*fd*t)
                c = circular_corr(z, ref.astype(np.complex64))
                H.append(np.abs(c))
            H = np.stack(H, axis=0)
            plt.figure()
            extent = [0, len(x), fds[0], fds[-1]]
            plt.imshow(H, aspect='auto', origin='lower', extent=extent)
            plt.colorbar(label="|corr|")
            plt.xlabel("Code lag (samples)"); plt.ylabel("Doppler (Hz)")
            plt.title(f"CAF magnitude — PRN {prn}")
        plt.show()

if __name__ == "__main__":
    main()
