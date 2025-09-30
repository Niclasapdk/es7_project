#!/usr/bin/env python3
# train_mlp.py
import os, json, argparse, numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt

# ---------------------------
# Paths
# ---------------------------
_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA = (_THIS_DIR / ".." / "datageneration" / "artifacts" / "bpsk_sweep_dataset.npz").resolve()

# ---------------------------
# Dataset
# ---------------------------
class IQDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.Y[i]

def load_dataset(npz_path):
    d = np.load(npz_path)
    Xtr, Ytr, Xva, Yva = d["Xtr"], d["Ytr"], d["Xva"], d["Yva"]
    meta = json.loads(d["meta"].item()) if "meta" in d.files else {}
    return (Xtr, Ytr, Xva, Yva, meta)

# ---------------------------
# Model (arbitrary depth MLP with residual)
# ---------------------------
class MLPDenoiser(nn.Module):
    def __init__(self, L, hidden=(256,128), dropout=0.0, use_layernorm=False):
        super().__init__()
        in_dim = 2 * L
        dims = [in_dim, *hidden, in_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if use_layernorm:
                layers.append(nn.LayerNorm(dims[i+1]))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))  # final linear back to 2L
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return x + self.net(x)  # residual to input

# ---------------------------
# Helpers: hidden parser, complex ops, alignment, soft demapper
# ---------------------------
def parse_hidden(arg: str):
    """
    Examples:
      --hidden "512,512,512,256,256,128"
      --hidden "512x5,256x3,128x2"  (repeat shorthand)
    """
    parts = []
    for token in arg.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if "x" in token:
            w, r = token.split("x")
            parts.extend([int(w)] * int(r))
        else:
            parts.append(int(token))
    return tuple(parts)

def flat_to_complex(x_flat):
    """[B, 2L] -> complex [B, L]"""
    B, twoL = x_flat.shape
    L = twoL // 2
    x = x_flat.view(B, L, 2)
    return torch.complex(x[...,0], x[...,1])

def complex_to_flat(z):
    """complex [B, L] -> [B, 2L] float"""
    B, L = z.shape
    out = torch.stack([z.real, z.imag], dim=-1).reshape(B, 2*L)
    return out

@torch.no_grad()
def align_scalar(yhat_c, ytrue_c):
    """Closed-form complex Procrustes scalar per batch item."""
    num = torch.sum(torch.conj(ytrue_c) * yhat_c, dim=1)       # [B]
    den = torch.sum(torch.conj(yhat_c)  * yhat_c,  dim=1) + 1e-12
    return num / den

def apply_align(yhat_flat, ytrue_flat):
    """Align yhat to ytrue with per-item complex scalar (rotation+gain)."""
    yhat_c  = flat_to_complex(yhat_flat)    # [B,L] complex
    ytrue_c = flat_to_complex(ytrue_flat)   # [B,L] complex
    c = align_scalar(yhat_c, ytrue_c)       # [B] complex, no grad
    yhat_al_c = c.unsqueeze(1) * yhat_c
    return complex_to_flat(yhat_al_c)

def bpsk_llr(y_aligned_flat, tau=0.5):
    """Soft demapper logits for BPSK using aligned I as decision variable."""
    z = flat_to_complex(y_aligned_flat)     # [B,L] complex
    I = z.real
    llr = I / (tau + 1e-12)                 # [B,L]
    return llr.unsqueeze(-1)                # [B,L,1]

# ---------------------------
# Train/Eval
# ---------------------------
def train(npz_path, out_dir, epochs, batch_size, lr, device,
          hidden, dropout, use_layernorm, alpha=0.5, beta=0.5, tau=0.5):

    Xtr, Ytr, Xva, Yva, meta = load_dataset(npz_path)
    L = Xtr.shape[1] // 2
    os.makedirs(out_dir, exist_ok=True)

    tr_dl = DataLoader(IQDataset(Xtr, Ytr), batch_size=batch_size, shuffle=True, drop_last=True)
    va_dl = DataLoader(IQDataset(Xva, Yva), batch_size=1024, shuffle=False)

    model = MLPDenoiser(L, hidden=hidden, dropout=dropout, use_layernorm=use_layernorm).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n_tr = Xtr.shape[0]
    n_va = Xva.shape[0]

    best = float("inf")
    for ep in range(1, epochs+1):
        model.train()
        tr_loss = tr_mse = tr_bce = 0.0

        for xb, yb in tr_dl:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()

            yhat    = model(xb)                  # [B, 2L]
            yhat_al = apply_align(yhat, yb)      # [B, 2L]

            mse = ((yhat_al - yb) ** 2).mean()

            ytrue_c   = flat_to_complex(yb)
            bits_true = (ytrue_c.real >= 0).float().unsqueeze(-1)  # [B,L,1]
            logits    = bpsk_llr(yhat_al, tau=tau)                 # [B,L,1]
            bce       = nn.functional.binary_cross_entropy_with_logits(logits, bits_true)

            loss = alpha * mse + beta * bce
            loss.backward()
            opt.step()

            bs = xb.size(0)
            tr_loss += loss.item() * bs
            tr_mse  += mse.item()  * bs
            tr_bce  += bce.item()  * bs

        tr_loss /= n_tr; tr_mse /= n_tr; tr_bce /= n_tr

        # ---- validation ----
        model.eval()
        va_loss = va_mse = va_bce = 0.0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb = xb.to(device); yb = yb.to(device)

                yhat    = model(xb)
                yhat_al = apply_align(yhat, yb)

                mse = ((yhat_al - yb) ** 2).mean()

                ytrue_c   = flat_to_complex(yb)
                bits_true = (ytrue_c.real >= 0).float().unsqueeze(-1)
                logits    = bpsk_llr(yhat_al, tau=tau)
                bce       = nn.functional.binary_cross_entropy_with_logits(logits, bits_true)

                bs = xb.size(0)
                va_loss += (alpha * mse + beta * bce).item() * bs
                va_mse  += mse.item() * bs
                va_bce  += bce.item() * bs

        va_loss /= n_va; va_mse /= n_va; va_bce /= n_va
        sch.step()

        print(
            f"Epoch {ep:02d} | "
            f"train loss {tr_loss:.6e} | train MSE {tr_mse:.6e} | train BCE {tr_bce:.6e} | "
            f"val loss {va_loss:.6e} | val MSE {va_mse:.6e} | val BCE {va_bce:.6e}"
        )

        if va_loss < best:
            best = va_loss
            ckpt = os.path.join(out_dir, "mlp_denoiser.pth")
            torch.save({"state_dict": model.state_dict(),
                        "meta": {"block_len": L, "hidden": hidden, "dataset": os.path.abspath(npz_path)}} , ckpt)
            # also TorchScript for deployment
            model.eval()
            scripted = torch.jit.script(model.cpu())
            scripted.save(os.path.join(out_dir, "mlp_denoiser.ts"))
            model.to(device)

    # ---- Proxy BER on val (aligned) ----
    print("Proxy BER on validation:")
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(Xva).to(device)
        yb = torch.from_numpy(Yva).to(device)
        yhat    = model(xb)
        yhat_al = apply_align(yhat, yb)

        yhat_c = flat_to_complex(yhat_al).cpu().numpy()
        ref_c  = flat_to_complex(yb).cpu().numpy()

    hard = (yhat_c.real >= 0).astype(np.int32)
    ref  = (ref_c.real  >= 0).astype(np.int32)
    ber = (hard != ref).mean()
    print(f"Proxy BER (aligned): {ber:.4f}")

    # ---- Overlay plot: Clean vs Denoised vs Noisy (I component) ----
    try:
        idx = 0
        model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(Xva[idx:idx+1]).to(device)   # noisy/jammed
            yb = torch.from_numpy(Yva[idx:idx+1]).to(device)   # clean target
            yhat    = model(xb)                                # denoised
            yhat_al = apply_align(yhat, yb)                    # align denoised -> clean
            xnoisy_al = apply_align(xb,  yb)                   # align noisy   -> clean

            y_clean_I = flat_to_complex(yb).cpu().numpy()[0].real
            y_deno_I  = flat_to_complex(yhat_al).cpu().numpy()[0].real
            y_noisy_I = flat_to_complex(xnoisy_al).cpu().numpy()[0].real

        Ls = y_clean_I.shape[0]
        fs = meta.get("fs_hz", meta.get("fs", None))
        if fs:
            t = np.arange(Ls) / float(fs) * 1e6  # microseconds
            xlab = "Time (µs)"
        else:
            t = np.arange(Ls)
            xlab = "Sample"

        plt.figure(figsize=(12,4))
        plt.plot(t, y_clean_I, label="Clean", linewidth=2)
        plt.plot(t, y_deno_I,  "--", label="Denoised (aligned)", linewidth=2)
        plt.plot(t, y_noisy_I, ":", label="Noisy/Jammed (aligned)", linewidth=2)
        plt.title("Denoised, Clean, and Noisy Signals (I component)")
        plt.xlabel(xlab); plt.ylabel("Amplitude"); plt.legend()
        plt.tight_layout()
        out_png = os.path.join(out_dir, f"overlay_I_with_noisy_idx{idx:04d}.png")
        plt.savefig(out_png, dpi=200); plt.close()
        print(f"Saved overlay plot to: {out_png}")
    except Exception as e:
        print(f"[overlay plot skipped] {e}")

# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Train MLP denoiser on jammed BPSK dataset")
    ap.add_argument("--data", default=str(_DEFAULT_DATA), help="path to dataset .npz file")
    ap.add_argument("--out_dir", default="artifacts")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--layernorm", action="store_true", help="use LayerNorm after each Linear")
    ap.add_argument("--hidden", type=str, default="512x5,256x3,128x2", help="hidden sizes; supports repeats like 512x5,256x3,128x2")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.75, help="weight for MSE part")
    ap.add_argument("--beta",  type=float, default=0.25, help="weight for BCE(soft-bit) part")
    ap.add_argument("--tau",   type=float, default=0.35, help="temperature for BPSK LLRs")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {data_path}\n"
            f"Tried default relative to this script: {_DEFAULT_DATA}\n"
            f"Tip: run gen_data.py or pass --data /absolute/or/relative/path.npz"
        )

    hidden = parse_hidden(args.hidden)
    device = "cpu" if args.cpu or (not torch.cuda.is_available()) else "cuda"

    train(args.data, args.out_dir, args.epochs, args.batch, args.lr, device,
          hidden, args.dropout, args.layernorm, alpha=args.alpha, beta=args.beta, tau=args.tau)

if __name__ == "__main__":
    main()
