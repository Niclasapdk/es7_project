import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------
# Helper functions
# ----------------------------------------------------------

def make_prn_like_code(n_chips=1023, oversample=4, seed=0):
    """
    Generate a C/A-like PRN sequence: random +/-1 chips,
    rectangular pulse shaped, oversampled by 'oversample'.
    """
    rng = np.random.default_rng(seed)
    chips = rng.choice([-1.0, 1.0], size=n_chips)
    code = np.repeat(chips, oversample)  # rectangular chips
    return code


def acf(x):
    """
    Discrete auto-correlation, normalised so that max = 1.
    Returns (lags, r).
    """
    r = np.correlate(x, x, mode="full")
    r = r / np.max(np.abs(r))
    lags = np.arange(-len(x) + 1, len(x))
    return lags, r


def xcorr(x, y):
    """
    Discrete cross-correlation, normalised by sequence length.
    Returns (lags, r).
    """
    r = np.correlate(x, y, mode="full") / len(x)
    lags = np.arange(-len(x) + 1, len(x))
    return lags, r


def compute_caf(y, c, fs, fd_grid):
    """
    Compute a simple CAF C(τ, f_D) between received y[n] and code c[n]:

        C(τ, f_D) ≈ sum_n y[n] * c^*(n - τ) * exp(-j 2π f_D n / fs)

    We implement this by, for each Doppler f_D:
      1) mixing y[n] with exp(-j 2π f_D n / fs),
      2) correlating with the local code c[n].

    Returns (lags, C), where:
      - lags are in samples,
      - C has shape (len(fd_grid), len(lags)).
    """
    N = len(y)
    n = np.arange(N)
    lags = np.arange(-N + 1, N)

    C = np.zeros((len(fd_grid), len(lags)), dtype=np.complex128)
    for i_fd, fd in enumerate(fd_grid):
        mix = np.exp(-1j * 2 * np.pi * fd * n / fs)
        ym = y * mix
        # Correlation over delay
        r = np.correlate(ym, c, mode="full")
        C[i_fd, :] = r

    return lags, C


# ----------------------------------------------------------
# Main script
# ----------------------------------------------------------

def main():
    # "GNSS-like" parameters
    n_chips = 1023
    oversample = 4
    fs = 1.023e6 * oversample  # 4.092 MHz

    # Two different PRN-like codes
    code1 = make_prn_like_code(n_chips=n_chips, oversample=oversample, seed=0)
    code2 = make_prn_like_code(n_chips=n_chips, oversample=oversample, seed=1)

    # ------------------------------------------------------
    # 1) Auto-correlation figure
    # ------------------------------------------------------
    lags_acf_samp, r_acf = acf(code1)
    lags_acf_chips = lags_acf_samp / oversample

    plt.figure(figsize=(5, 3))
    plt.plot(lags_acf_chips, r_acf)
    plt.xlabel("Lag $k$ [chips]")
    plt.ylabel("$R_{ss}[k]$ (norm.)")
    plt.title("Auto-correlation of one PRN-like code")
    plt.grid(True)
    # Zoom a bit around the main lobe for a "theory" look
    plt.xlim(-3, 3)
    plt.ylim(-0.1, 1.05)
    plt.tight_layout()
    plt.savefig("acf_prn.png", dpi=300, bbox_inches="tight")

    # ------------------------------------------------------
    # 2) Cross-correlation figure
    # ------------------------------------------------------
    lags_x_samp, r_x = xcorr(code1, code2)
    lags_x_chips = lags_x_samp / oversample

    plt.figure(figsize=(5, 3))
    # Only show a small lag window to get a nice noise-like curve
    mask = np.abs(lags_x_chips) <= 20
    plt.stem(
        lags_x_chips[mask],
        r_x[mask],
        basefmt=" ",
    )
    plt.xlabel("Lag $k$ [chips]")
    plt.ylabel("$R_{s_1 s_2}[k]$")
    plt.title("Cross-correlation of two different PRN-like codes")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("xcorr_prn.png", dpi=300, bbox_inches="tight")

    # ------------------------------------------------------
    # 3) CAF “thumbtack” figure
    # ------------------------------------------------------
    N = len(code1)
    n = np.arange(N)

    # Simulate a single clean satellite with some Doppler
    fd_true = 1500.0  # Hz
    y = code1 * np.exp(1j * 2 * np.pi * fd_true * n / fs)

    # Doppler grid and CAF
    fd_grid = np.linspace(-4000, 4000, 81)  # from -4 kHz to +4 kHz
    lags_caf_samp, C = compute_caf(y, code1, fs, fd_grid)

    # Magnitude, normalised
    C_mag = np.abs(C)
    C_mag /= C_mag.max()

    # Extract a small window around zero delay, e.g. +/- 4 chips
    max_tau_chips = 4
    mask_tau = np.abs(lags_caf_samp) <= max_tau_chips * oversample
    tau_chips = lags_caf_samp[mask_tau] / oversample
    C_win = C_mag[:, mask_tau]

    plt.figure(figsize=(5, 4))
    extent = [tau_chips[0], tau_chips[-1], fd_grid[0], fd_grid[-1]]
    plt.imshow(
        C_win,
        origin="lower",
        aspect="auto",
        extent=extent,
    )
    plt.xlabel("Code delay $\\tau$ [chips]")
    plt.ylabel("Doppler $f_{\\mathrm{D}}$ [Hz]")
    plt.title("CAF magnitude for a single PRN-like signal")
    cbar = plt.colorbar()
    cbar.set_label("$|C(\\tau, f_{\\mathrm{D}})|$ (norm.)")
    plt.tight_layout()
    plt.savefig("caf_thumbtack.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
