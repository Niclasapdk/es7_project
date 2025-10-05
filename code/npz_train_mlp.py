#!/usr/bin/env python3
"""
npz_train_mlp_cpu.py — simple, stable MLP denoiser for IQ (CPU/CUDA only)

Key points:
- NPZ loader supports: Xtr/Ytr, Xva/Yva [,Xte/Yte]  OR  X_train/...  OR  X/Y[+split]
- Per-window complex alignment (LS gain+phase) so X and Y are comparable
- Global z-score normalization (fit on X_train)
- Residual-learning MLP (predict r = y - x), SmoothL1 + optional light spectral loss
- AdamW optimizer + ReduceLROnPlateau scheduler
- No DirectML, no mixed precision, no special device branches
"""

from pathlib import Path
import argparse, json, numpy as np, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Device (CPU or CUDA if available)
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

SEED = 42
SPECTRAL_WEIGHT_DEFAULT = 2.0
BATCH_SIZE_DEFAULT = 512
EPOCHS_DEFAULT = 300
LR_DEFAULT = 1e-4
WEIGHT_DECAY_DEFAULT = 1e-4
PDROP_DEFAULT = 0.1
HIDDEN_DEFAULT = [2048, 1024, 512]
CKPT_DEFAULT = "mlp_npz_best.pt"

rng = np.random.default_rng(SEED)

# ---------------------------
# NPZ loading & shapes
# ---------------------------
def _to_flat_2win(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[-1] == 2:
            I, Q = arr[...,0], arr[...,1]
        elif arr.shape[1] == 2:
            I, Q = arr[:,0,:], arr[:,1,:]
        else:
            raise ValueError(f"3D array not recognized: {arr.shape}")
        flat = np.concatenate([I, Q], axis=1).astype(np.float32)
        return flat, I.shape[1]
    elif arr.ndim == 2:
        F = arr.shape[1]
        if F % 2 != 0: raise ValueError(f"Feature dim {F} not even; cannot split I/Q.")
        return arr.astype(np.float32), F//2
    else:
        raise ValueError(f"Unsupported shape {arr.shape}")

def _have(D,*ks): return all(k in D for k in ks)

def load_npz_dataset(p: Path):
    D = np.load(p, allow_pickle=True)
    keys = set(D.files)
    print(f"[NPZ] Keys: {sorted(keys)}")

    def flat2(kx,ky):
        Xf, w1 = _to_flat_2win(D[kx]); Yf, w2 = _to_flat_2win(D[ky])
        if w1 != w2: raise ValueError("Window mismatch X vs Y")
        return Xf, Yf, w1

    # X_train / Y_train ...
    if _have(D,'X_train','Y_train','X_val','Y_val'):
        Xtr,Ytr,w = flat2('X_train','Y_train')
        Xva,Yva,_ = flat2('X_val','Y_val')
        Xte=Yte=None
        if _have(D,'X_test','Y_test'):
            Xte,Yte,_ = flat2('X_test','Y_test')
        return Xtr,Ytr,Xva,Yva,Xte,Yte,w

    # Xtr/Xva/Ytr/Yva (+ optional Xte/Yte)
    if _have(D,'Xtr','Ytr','Xva','Yva'):
        Xtr,Ytr,w = flat2('Xtr','Ytr'); Xva,Yva,_ = flat2('Xva','Yva')
        Xte=Yte=None
        if _have(D,'Xte','Yte'):
            Xte,Yte,_ = flat2('Xte','Yte')
        else:
            # carve 10% of val as test
            nva=len(Xva); nte=max(1,int(0.1*nva))
            idx=np.arange(nva); rng.shuffle(idx)
            te,va = idx[:nte], idx[nte:]
            Xte,Yte = Xva[te],Yva[te]
            Xva,Yva = Xva[va],Yva[va]
            print(f"[NPZ] No test split -> carved {len(Xte)} from val as test.")
        return Xtr,Ytr,Xva,Yva,Xte,Yte,w

    # X/Y (+ optional split)
    if _have(D,'X','Y'):
        X,Y,w = flat2('X','Y')
        if 'split' in D:
            sp=D['split']
            if sp.dtype.kind in 'UO':
                m={'train':0,'val':1,'validation':1,'test':2}
                S=np.array([m.get(str(s).lower(),0) for s in sp],dtype=np.int32)
            else:
                S=np.array(sp,dtype=np.int32)
            mtr,mva,mte=(S==0),(S==1),(S==2)
            return X[mtr],Y[mtr],X[mva],Y[mva],X[mte],Y[mte],w
        # random split
        N=len(X); idx=np.arange(N); rng.shuffle(idx)
        nte=max(1,int(0.15*N)); nva=max(1,int(0.15*N))
        te,va,tr = idx[:nte], idx[nte:nte+nva], idx[nte+nva:]
        return X[tr],Y[tr],X[va],Y[va],X[te],Y[te],w

    raise SystemExit("Unrecognized NPZ structure.")

# ---------------------------
# Per-window complex alignment (gain+phase LS)
# ---------------------------
def _flat_to_complex(F, win):
    I = F[:,:win]; Q = F[:,win:]
    return I.astype(np.float64) + 1j*Q.astype(np.float64)

def _complex_to_flat(Z, win, dtype=np.float32):
    I = Z.real.astype(dtype); Q = Z.imag.astype(dtype)
    return np.concatenate([I, Q], axis=1)

def align_X_to_Y_least_squares(X_flat, Y_flat, win):
    Xc = _flat_to_complex(X_flat, win)
    Yc = _flat_to_complex(Y_flat, win)
    num = np.sum(np.conjugate(Xc) * Yc, axis=1)
    den = np.sum(np.conjugate(Xc) * Xc, axis=1) + 1e-12
    a = num / den
    Xc_al = (a[:,None] * Xc)
    return _complex_to_flat(Xc_al, win, dtype=np.float32)

# ---------------------------
# Dataset & Model
# ---------------------------
class PairDataset(Dataset):
    def __init__(self, X, R):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.R = torch.from_numpy(R.astype(np.float32))
    def __len__(self): return self.X.shape[0]
    def __getitem__(self,i): return self.X[i], self.R[i]

class ResOutMLP(nn.Module):
    """Residual-output MLP (ReLU, optional dropout)."""
    def __init__(self, in_dim, hidden, dropout=0.1):
        super().__init__()
        layers=[]; prev=in_dim
        for h in hidden:
            layers += [nn.Linear(prev,h), nn.ReLU(inplace=True)]
            if dropout>0: layers += [nn.Dropout(dropout)]
            prev=h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, in_dim)
        # small random init to avoid zero-grad stall
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.head.bias)
    def forward(self, x):
        r = self.head(self.body(x))
        y_hat = x + r
        return y_hat, r

# ---------------------------
# Losses & Metrics
# ---------------------------
def spectral_loss(x_hat, y, win):
    # CPU float32 rFFT even if model on CUDA
    xh = x_hat.detach().float().cpu()
    yt = y.detach().float().cpu()
    def _unflat(z):
        I = z[:,:win]; Q = z[:,win:]
        return I, Q
    Ih,Qh = _unflat(xh); Iy,Qy = _unflat(yt)
    def _pshape(t):
        F = torch.fft.rfft(t, dim=1)
        P = (F.real**2 + F.imag**2)
        P = P / (torch.sum(P, dim=1, keepdim=True) + 1e-12)
        return P
    Ph = _pshape(Ih) + _pshape(Qh)
    Py = _pshape(Iy) + _pshape(Qy)
    return torch.mean((Ph - Py)**2)

def sdr_db(ref, est):
    num = np.sum(np.abs(ref)**2) + 1e-12
    den = np.sum(np.abs(ref - est)**2) + 1e-12
    return 10*np.log10(num/den)

def evm_pct(ref, est):
    num = np.mean(np.abs(ref - est)**2)
    den = np.mean(np.abs(ref)**2) + 1e-12
    return 100*np.sqrt(num/den)

# ---------------------------
# Train
# ---------------------------
def train_one(args):
    # Load
    Xtr,Ytr,Xva,Yva,Xte,Yte,WIN = load_npz_dataset(Path(args.data))

    # Per-window alignment (recommended)
    Xtr = align_X_to_Y_least_squares(Xtr, Ytr, WIN)
    Xva = align_X_to_Y_least_squares(Xva, Yva, WIN)
    if Xte is not None and Yte is not None:
        Xte = align_X_to_Y_least_squares(Xte, Yte, WIN)

    # Normalize using train X
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-8
    Xtrn = (Xtr - mu)/sd; Xvan = (Xva - mu)/sd
    Ytrn = (Ytr - mu)/sd; Yvan = (Yva - mu)/sd
    if Xte is not None:
        Xten = (Xte - mu)/sd; Yten = (Yte - mu)/sd

    # Residual targets
    Rtr, Rva = Ytrn - Xtrn, Yvan - Xvan
    Rte = Yten - Xten if Xte is not None else None

    # Datasets/Loaders (simple & stable on Windows)
    train_ds = PairDataset(Xtrn, Rtr)
    val_ds   = PairDataset(Xvan, Rva)
    test_ds  = PairDataset(Xten, Rte) if Xte is not None else None

    tr_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    te_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False) if test_ds else None

    # Model
    in_dim = Xtrn.shape[1]
    hidden = HIDDEN_DEFAULT if args.hidden is None else json.loads(args.hidden)
    model = ResOutMLP(in_dim, hidden, dropout=args.pdrop).to(device)

    # Optim/sched
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    huber = nn.SmoothL1Loss(beta=1.0)

    def loss_fn(y_hat, r_hat, x, r_tgt):
        l_res = huber(r_hat, r_tgt)
        spec  = spectral_loss(y_hat, x + r_tgt, win=WIN) if args.spec_weight>0 else torch.tensor(0.0)
        return l_res + args.spec_weight*spec, (l_res.item(), float(spec))

    best = (1e9, None)
    for ep in range(1, args.epochs+1):
        # ---- Train ----
        model.train()
        tr_loss=lrs=lsp=0.0; n=0
        for xb, rb in tr_loader:
            xb=xb.to(device); rb=rb.to(device)
            opt.zero_grad(set_to_none=True)
            y_hat, r_hat = model(xb)
            loss,(lrsv,lspv) = loss_fn(y_hat, r_hat, xb, rb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = xb.size(0)
            tr_loss += loss.item()*bs; lrs += lrsv*bs; lsp += lspv*bs; n += bs
        tr_loss/=n; lrs/=n; lsp/=n

        # ---- Val ----
        model.eval(); va_loss=0.0; m=0
        with torch.no_grad():
            for xb, rb in va_loader:
                xb=xb.to(device); rb=rb.to(device)
                y_hat, r_hat = model(xb)
                l,_ = loss_fn(y_hat, r_hat, xb, rb)
                va_loss += l.item()*xb.size(0); m += xb.size(0)
        va_loss/=m
        sched.step(va_loss)

        print(f"Epoch {ep:03d} | train {tr_loss:.5f} (res {lrs:.5f}, spec {lsp:.5f}) | val {va_loss:.5f}")

        if va_loss < best[0]:
            best = (va_loss, {
                "model": model.state_dict(),
                "in_dim": in_dim,
                "hidden": hidden,
                "window": WIN,
                "mu": mu, "sd": sd
            })
            torch.save(best[1], args.ckpt)

    print(f"\nBest val loss: {best[0]:.6f} | saved -> {args.ckpt}")

    # ---- Test metrics ----
    if te_loader is not None:
        state = best[1]
        model.load_state_dict(state["model"])
        model.eval()
        outs=[]
        with torch.no_grad():
            for xb,_ in te_loader:
                xb=xb.to(device)
                y_hat,_ = model(xb)
                outs.append(y_hat.cpu().numpy())
        Yhat = np.concatenate(outs,0)

        # de-normalize
        Xte_den = Xten*sd + mu
        Yte_den = Yten*sd + mu
        Yhat_den= Yhat*sd + mu

        def flat_to_complex(F, win):
            I = F[:,:win]; Q = F[:,win:]
            return I + 1j*Q

        xin_c = flat_to_complex(Xte_den, WIN)
        y_c   = flat_to_complex(Yte_den, WIN)
        yh_c  = flat_to_complex(Yhat_den, WIN)

        sdr_in  = float(np.mean([sdr_db(y, x)    for x,y    in zip(xin_c, y_c)]))
        sdr_out = float(np.mean([sdr_db(y, yhat) for y,yhat in zip(y_c, yh_c)]))
        evm_in  = float(np.mean([evm_pct(y, x)    for x,y    in zip(xin_c, y_c)]))
        evm_out = float(np.mean([evm_pct(y, yhat) for y,yhat in zip(y_c, yh_c)]))

        print("\n=== TEST METRICS ===")
        print(f"SDR   in : {sdr_in:6.2f} dB | out : {sdr_out:6.2f} dB | Δ: {sdr_out - sdr_in:+6.2f} dB")
        print(f"EVM%  in : {evm_in:6.2f}% | out : {evm_out:6.2f}% | Δ: {evm_out - evm_in:+6.2f}%")

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="NPZ MLP denoiser (CPU/CUDA)")
    p.add_argument("--data", type=str, required=True, help="Path to NPZ dataset")
    p.add_argument("--ckpt", type=str, default=CKPT_DEFAULT)
    p.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE_DEFAULT)
    p.add_argument("--lr", type=float, default=LR_DEFAULT)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY_DEFAULT)
    p.add_argument("--spec_weight", type=float, default=SPECTRAL_WEIGHT_DEFAULT)
    p.add_argument("--pdrop", type=float, default=PDROP_DEFAULT)
    p.add_argument("--hidden", type=str, default=None,
                   help='JSON list, e.g. "[2048,1024,512]"')
    return p.parse_args()

def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    args = parse_args()
    train_one(args)

if __name__ == "__main__":
    main()
