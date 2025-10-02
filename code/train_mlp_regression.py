#!/usr/bin/env python3
"""
Residual MLP denoiser for RF interference -> clean IQ regression (paired & aligned).

Changes vs previous:
- Train on RESIDUALS (R = Y - X), predict r_hat; output y_hat = x + r_hat
- Zero-init final layer (identity warm-start)
- No BatchNorm, small dropout
- SmoothL1 (Huber) residual loss + stronger spectral loss (weight=1.0)
- Same matching/alignment and global z-score normalization

Folder layout:
  BASE_PATH/
    Clean_BPSK_reference/*.csv
    BPSK_Interference/<freq>/<gain>/*.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import welch

# ---------------------------
# CONFIG
# ---------------------------
BASE_PATH = "real_data/"

# Windowing
WINDOW = 512
HOP    = 256               # keep <= WINDOW for overlap
LAYOUT = "block"           # 'block' (I then Q) or 'interleave'

# Matching/alignment
DOWN_FAC = 10              # downsample factor for envelope matching (speed)
SMOOTH_ENV = 51           # moving-average length for envelope smoothing (odd)
PSD_NPERSEG = 1024
W_ENV = 0.6               # match score weights
W_PSD = 0.4

# Splits
VAL_FRAC  = 0.15
TEST_FRAC = 0.15
SEED = 42

# Training
HIDDEN = [8128, 4096, 2048]
PDROP = 0.2               # tiny dropout; BN disabled
USE_BN = False
BATCH_SIZE = 512
EPOCHS = 400
LR = 1e-4
WEIGHT_DECAY = 1e-4
SPECTRAL_WEIGHT = 2.0
CKPT_PATH = "mlp_denoiser_best.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(SEED)

# ---------------------------
# I/O
# ---------------------------
def load_measurement_data(base_path):
    data_records = []
    base_path = Path(base_path)

    # Clean
    no_int_path = base_path / "Clean_BPSK_reference"
    if no_int_path.exists():
        for csv in sorted(no_int_path.glob("*.csv")):
            df = pd.read_csv(csv)
            data_records.append(dict(
                data=df[['I','Q']].values.astype(np.float32),
                time=df['Time'].values.astype(np.float32) if 'Time' in df.columns else None,
                has_interference=False,
                interference_freq=None,
                interference_gain=None,
                file_path=str(csv),
                measurement_id=csv.stem
            ))

    # Interference
    with_int_path = base_path / "BPSK_Interference"
    if with_int_path.exists():
        for freq_dir in sorted([d for d in with_int_path.iterdir() if d.is_dir()]):
            f_val = freq_dir.name.replace('freq_', '') if 'freq_' in freq_dir.name else freq_dir.name
            for gain_dir in sorted([g for g in freq_dir.iterdir() if g.is_dir()]):
                g_val = gain_dir.name.replace('gain_', '') if 'gain_' in gain_dir.name else gain_dir.name
                for csv in sorted(gain_dir.glob("*.csv")):
                    df = pd.read_csv(csv)
                    data_records.append(dict(
                        data=df[['I','Q']].values.astype(np.float32),
                        time=df['Time'].values.astype(np.float32) if 'Time' in df.columns else None,
                        has_interference=True,
                        interference_freq=f_val,
                        interference_gain=g_val,
                        file_path=str(csv),
                        measurement_id=csv.stem
                    ))
    return data_records

# ---------------------------
# Matching & alignment utils
# ---------------------------
def to_complex(iq): return iq[:,0].astype(np.float64) + 1j*iq[:,1].astype(np.float64)

def envelope(z):
    e = np.abs(z)
    if SMOOTH_ENV and SMOOTH_ENV > 1:
        k = SMOOTH_ENV
        pad = k//2
        e = np.pad(e, (pad,pad), mode='edge')
        e = np.convolve(e, np.ones(k)/k, mode='valid')
    return e

def psd_shape(z, nperseg=PSD_NPERSEG):
    i = np.real(z); q = np.imag(z)
    f1, Pxx = welch(i, nperseg=min(nperseg, len(i)))
    f2, Pyx = welch(q, nperseg=min(nperseg, len(q)))
    P = Pxx + Pyx
    s = np.sum(P)
    if s > 0: P = P / (s + 1e-12)
    return P.astype(np.float32)

def cosine_sim(a,b):
    num = float(np.dot(a,b))
    den = float(np.linalg.norm(a)*np.linalg.norm(b) + 1e-12)
    return num/den

def peak_xcorr(a,b):
    na = (a - a.mean())/(a.std()+1e-12)
    nb = (b - b.mean())/(b.std()+1e-12)
    r = np.correlate(na, nb, mode='full')
    r /= (len(na))
    return float(np.max(r))

def estimate_lag_gain(z_int, z_clean):
    # lag via envelope; apply to IQ; then scalar gain via LS (real, phase-preserving)
    ea = envelope(z_int); eb = envelope(z_clean)
    if DOWN_FAC > 1:
        ea = ea[::DOWN_FAC]; eb = eb[::DOWN_FAC]
    na = (ea - ea.mean())/(ea.std()+1e-12)
    nb = (eb - eb.mean())/(eb.std()+1e-12)
    r = np.correlate(na, nb, mode='full')
    lag_ds = int(np.argmax(r) - (len(nb)-1))
    lag = lag_ds * DOWN_FAC

    if lag > 0:
        z_int_al = z_int[lag:]
        z_cl_al  = z_clean[:len(z_int_al)]
    elif lag < 0:
        z_cl_al  = z_clean[-lag:]
        z_int_al = z_int[:len(z_cl_al)]
    else:
        z_int_al, z_cl_al = z_int, z_clean
    n = min(len(z_int_al), len(z_cl_al))
    z_int_al = z_int_al[:n]; z_cl_al = z_cl_al[:n]

    num = np.vdot(z_int_al, z_cl_al).real
    den = (np.vdot(z_int_al, z_int_al).real + 1e-12)
    a = num/den
    z_int_fit = a * z_int_al
    return z_int_fit, z_cl_al, lag, a

def match_interference_to_clean(clean_recs, int_recs):
    # Precompute fingerprints for clean
    clean_fps = []
    for r in clean_recs:
        z = to_complex(r['data'])
        clean_fps.append(dict(
            env=envelope(z)[::DOWN_FAC],
            psd=psd_shape(z),
            rec=r
        ))
    pairs = []
    for r in int_recs:
        z = to_complex(r['data'])
        env_i = envelope(z)[::DOWN_FAC]
        psd_i = psd_shape(z)

        best = (-1e9, None)
        for c in clean_fps:
            L = min(len(env_i), len(c['env']))
            if L < 100: continue
            s_env = peak_xcorr(env_i[:L], c['env'][:L])
            Lp = min(len(psd_i), len(c['psd']))
            s_psd = cosine_sim(psd_i[:Lp], c['psd'][:Lp])
            score = W_ENV*s_env + W_PSD*s_psd
            if score > best[0]:
                best = (score, c['rec'])
        if best[1] is not None:
            pairs.append((r, best[1], best[0]))
    return pairs

# ---------------------------
# Windowing
# ---------------------------
def window_iq(iq: np.ndarray, win: int, hop: int, layout: str) -> np.ndarray:
    N = iq.shape[0]
    if N < win: return np.empty((0, 2*win), np.float32)
    starts = range(0, N - win + 1, hop)
    chunks = np.stack([iq[s:s+win] for s in starts], axis=0)  # (W,win,2)
    I, Q = chunks[...,0], chunks[...,1]
    if layout == "interleave":
        flat = np.empty((chunks.shape[0], 2*win), dtype=np.float32)
        flat[:,0::2] = I; flat[:,1::2] = Q
        return flat
    return np.concatenate([I, Q], axis=1).astype(np.float32)

# ---------------------------
# Dataset & Model
# ---------------------------
class PairDataset(Dataset):
    # stores X (input windows) and R (residual targets), both float32
    def __init__(self, X, R):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.R = torch.from_numpy(R.astype(np.float32))
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.R[i]

class ResOutMLP(nn.Module):
    """
    MLP that predicts residual r_hat; returns y_hat = x + r_hat and r_hat.
    - BN disabled (works against near-identity)
    - Zero-initialized head for identity warm-start
    """
    def __init__(self, in_dim, hidden, dropout=0.1, use_bn=False):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h)]
            if use_bn: layers += [nn.BatchNorm1d(h)]
            layers += [nn.ReLU(inplace=True)]
            if dropout > 0: layers += [nn.Dropout(dropout)]
            prev = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, in_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    def forward(self, x):
        r = self.head(self.body(x))
        y_hat = x + r
        return y_hat, r

def spectral_loss(x_hat, y, win):
    """
    Light spectral loss: compare Welch-like normalized power spectra of I/Q.
    x_hat, y: (B, 2*win)
    """
    def unflat(z):
        I = z[:, :win]; Q = z[:, win:]
        return I, Q
    Ih, Qh = unflat(x_hat); Iy, Qy = unflat(y)

    def pshape(t):
        F = torch.fft.rfft(t, dim=1)
        P = (F.real**2 + F.imag**2)
        P = P / (torch.sum(P, dim=1, keepdim=True)+1e-12)
        return P
    Ph = pshape(Ih) + pshape(Qh)
    Py = pshape(Iy) + pshape(Qy)
    return torch.mean((Ph - Py)**2)

# ---------------------------
# Metrics
# ---------------------------
def sdr_db(ref, est):
    num = np.sum(np.abs(ref)**2) + 1e-12
    den = np.sum(np.abs(ref - est)**2) + 1e-12
    return 10*np.log10(num/den)

def evm_pct(ref, est):
    num = np.mean(np.abs(ref - est)**2)
    den = np.mean(np.abs(ref)**2) + 1e-12
    return 100*np.sqrt(num/den)

def flat_to_complex(F, win):
    I = F[:,:win]; Q = F[:,win:]
    return I + 1j*Q

# ---------------------------
# Main
# ---------------------------
def main():
    # Load all records
    all_recs = load_measurement_data(BASE_PATH)
    if not all_recs: raise SystemExit("No data found.")
    clean = [r for r in all_recs if not r['has_interference']]
    ints  = [r for r in all_recs if r['has_interference']]
    print(f"Loaded: clean={len(clean)} | interference={len(ints)}")

    # Match interference -> clean
    pairs = match_interference_to_clean(clean, ints)
    print(f"Matched {len(pairs)} interference files to clean references.")

    # Build aligned windows
    X_in, Y_tg, owner = [], [], []
    for (ir, cr, score) in pairs:
        z_i = to_complex(ir['data'])
        z_c = to_complex(cr['data'])
        z_i_al, z_c_al, lag, gain = estimate_lag_gain(z_i, z_c)

        iq_i = np.stack([np.real(z_i_al), np.imag(z_i_al)], axis=1).astype(np.float32)
        iq_c = np.stack([np.real(z_c_al), np.imag(z_c_al)], axis=1).astype(np.float32)

        Xin = window_iq(iq_i, WINDOW, HOP, LAYOUT)
        Ytg = window_iq(iq_c, WINDOW, HOP, LAYOUT)
        n = min(len(Xin), len(Ytg))
        if n == 0: continue
        X_in.append(Xin[:n]); Y_tg.append(Ytg[:n])
        owner.append(np.full((n,), hash(ir['file_path'])%2**31, dtype=np.int32))

    if not X_in: raise SystemExit("No paired windows created. Adjust WINDOW/HOP or check data.")
    X = np.concatenate(X_in, 0)
    Y = np.concatenate(Y_tg, 0)
    owner = np.concatenate(owner, 0)
    print(f"Paired windows: X={X.shape}  Y={Y.shape}")

    # Split by owner (file-level)
    meas = np.unique(owner)
    rng.shuffle(meas)
    n = len(meas)
    n_test = max(1, int(round(TEST_FRAC * n)))
    n_val  = max(1, int(round(VAL_FRAC  * n)))
    test_ids = set(meas[:n_test])
    val_ids  = set(meas[n_test:n_test+n_val])
    train_ids= set(meas[n_test+n_val:])

    tr_m = np.isin(owner, list(train_ids))
    va_m = np.isin(owner, list(val_ids))
    te_m = np.isin(owner, list(test_ids))

    Xtr, Ytr = X[tr_m], Y[tr_m]
    Xva, Yva = X[va_m], Y[va_m]
    Xte, Yte = X[te_m], Y[te_m]
    print(f"Split -> train {len(Xtr)} | val {len(Xva)} | test {len(Xte)}")

    # Global z-score normalization fit on inputs (apply same to targets)
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True)+1e-8
    Xtr = (Xtr - mu)/sd; Xva = (Xva - mu)/sd; Xte = (Xte - mu)/sd
    Ytr = (Ytr - mu)/sd; Yva = (Yva - mu)/sd; Yte = (Yte - mu)/sd

    # Residual targets
    Rtr, Rva, Rte = Ytr - Xtr, Yva - Xva, Yte - Xte

    # Datasets/loaders
    train_ds = PairDataset(Xtr, Rtr)
    val_ds   = PairDataset(Xva, Rva)
    test_ds  = PairDataset(Xte, Rte)

    tr_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True)
    va_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    te_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # Model/optim
    in_dim = X.shape[1]
    model = ResOutMLP(in_dim, HIDDEN, PDROP, USE_BN).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    huber = nn.SmoothL1Loss(beta=1.0)

    def loss_fn(y_hat, r_hat, x, r_tgt):
        # Residual (SmoothL1) + Spectral on y_hat vs clean (x + r_tgt)
        l_res = huber(r_hat, r_tgt)
        clean = x + r_tgt
        spec = spectral_loss(y_hat, clean, win=WINDOW)
        return l_res + SPECTRAL_WEIGHT*spec, (l_res.item(), spec.item())

    best = (1e9, None)
    for ep in range(1, EPOCHS+1):
        model.train()
        tr_loss=0; lrs=0; lsp=0
        for xb, rb in tr_loader:
            xb=xb.to(device); rb=rb.to(device)
            opt.zero_grad()
            y_hat, r_hat = model(xb)
            loss,(lrsv, lspv) = loss_fn(y_hat, r_hat, xb, rb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = xb.size(0)
            tr_loss += loss.item()*bs
            lrs += lrsv*bs; lsp += lspv*bs
        tr_loss/=len(train_ds); lrs/=len(train_ds); lsp/=len(train_ds)

        # val
        model.eval()
        va_loss=0
        with torch.no_grad():
            for xb, rb in va_loader:
                xb=xb.to(device); rb=rb.to(device)
                y_hat, r_hat = model(xb)
                l,_ = loss_fn(y_hat, r_hat, xb, rb)
                va_loss += l.item()*xb.size(0)
        va_loss/=len(val_ds)
        sched.step(va_loss)

        print(f"Epoch {ep:03d} | train {tr_loss:.5f} (res {lrs:.5f}, spec {lsp:.5f}) | val {va_loss:.5f}")

        if va_loss < best[0]:
            best = (va_loss, {
                "model": model.state_dict(),
                "in_dim": in_dim, "hidden": HIDDEN,
                "dropout": PDROP, "use_bn": USE_BN,
                "window": WINDOW, "hop": HOP, "layout": LAYOUT,
                "mu": mu, "sd": sd
            })
            torch.save(best[1], CKPT_PATH)

    print(f"\nBest val loss: {best[0]:.6f} | saved -> {CKPT_PATH}")

    # ===== Test metrics (SDR/EVM) =====
    state = best[1]
    model.load_state_dict(state["model"])
    model.eval()

    outs = []
    with torch.no_grad():
        for xb, _ in te_loader:
            xb = xb.to(device)
            y_hat, _ = model(xb)
            outs.append(y_hat.cpu().numpy())
    Yhat = np.concatenate(outs,0)

    # de-normalize
    Xte_den = Xte*sd + mu
    Yte_den = Yte*sd + mu
    Yhat_den= Yhat*sd + mu

    xin_c = flat_to_complex(Xte_den, WINDOW)
    y_c   = flat_to_complex(Yte_den, WINDOW)
    yh_c  = flat_to_complex(Yhat_den, WINDOW)

    sdr_in  = np.mean([sdr_db(y, x)    for x,y    in zip(xin_c, y_c)])
    sdr_out = np.mean([sdr_db(y, yhat) for y,yhat in zip(y_c, yh_c)])
    evm_in  = np.mean([evm_pct(y, x)    for x,y    in zip(xin_c, y_c)])
    evm_out = np.mean([evm_pct(y, yhat) for y,yhat in zip(y_c, yh_c)])

    print("\n=== TEST METRICS ===")
    print(f"SDR   in : {sdr_in:6.2f} dB | out : {sdr_out:6.2f} dB | Δ: {sdr_out - sdr_in:+6.2f} dB")
    print(f"EVM%% in : {evm_in:6.2f} %% | out : {evm_out:6.2f} %% | Δ: {evm_out - evm_in:+6.2f} %%")

if __name__ == "__main__":
    main()
