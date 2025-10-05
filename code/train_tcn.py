#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Causal TCN denoiser for complex IQ (time domain), with streaming overlap-add inference.

Expected NPZ formats (auto-detected):
  1) 'X_train','Y_train','X_val','Y_val','X_test','Y_test'  (shape: [N, L, 2])
  2) 'train_X','train_Y','val_X','val_Y','test_X','test_Y'
  3) 'X','Y' (+ optional 'idx_train','idx_val','idx_test'); else it will 80/10/10 split

If your L < window (W), the loader will pad; if L > W it will create rolling windows.
"""

import os, argparse, math, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ---------------------------
# Utilities
# ---------------------------

def device_auto():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_tensor(x):  # np -> torch float32
    return torch.from_numpy(x.astype(np.float32))

def pairwise_diff(x, dim=-2):
    # first difference along time (for ΔL1 loss); expects shape [B, T, C]
    return x.diff(dim=dim)

# ---------------------------
# Data loading
# ---------------------------

def load_npz_auto(path):
    d = np.load(path, allow_pickle=True)
    keys = set(d.keys())

    # --- Your layout: Xtr, Ytr, Xva, Yva (plus optional 'meta')
    if {"Xtr","Ytr","Xva","Yva"}.issubset(keys):
        Xtr, Ytr = d["Xtr"], d["Ytr"]
        Xva, Yva = d["Xva"], d["Yva"]
        # No test set provided -> use val as test (or you can later add Xte/Yte)
        Xte, Yte = Xva, Yva
        print("[load_npz_auto] Using provided train/val. No test found -> reusing val as test.")
        return (Xtr, Ytr), (Xva, Yva), (Xte, Yte)

    # --- Other known layouts
    patterns = [
        ("X_train","Y_train","X_val","Y_val","X_test","Y_test"),
        ("train_X","train_Y","val_X","val_Y","test_X","test_Y"),
    ]
    for pat in patterns:
        if all(k in keys for k in pat):
            Xtr, Ytr = d[pat[0]], d[pat[1]]
            Xva, Yva = d[pat[2]], d[pat[3]]
            Xte, Yte = d[pat[4]], d[pat[5]]
            return (Xtr, Ytr), (Xva, Yva), (Xte, Yte)

    if "X" in keys and "Y" in keys:
        X, Y = d["X"], d["Y"]
        n = len(X)
        if {"idx_train","idx_val","idx_test"}.issubset(keys):
            tr, va, te = d["idx_train"], d["idx_val"], d["idx_test"]
            return (X[tr], Y[tr]), (X[va], Y[va]), (X[te], Y[te])
        # fallback 80/10/10
        n_tr = int(0.8 * n)
        n_va = int(0.1 * n)
        n_te = n - n_tr - n_va
        return (X[:n_tr], Y[:n_tr]), (X[n_tr:n_tr+n_va], Y[n_tr:n_tr+n_va]), (X[n_tr+n_va:], Y[n_tr+n_va:])

    raise ValueError(f"Unrecognized NPZ structure: keys={sorted(keys)}")

def _to_N_L_2(A: np.ndarray, name: str):
    """
    Normalize any IQ-ish array to shape [N, L, 2] (I,Q).
    Accepted inputs:
      - [N, L, 2]               (ok)
      - [N, 2, L]               (transpose)
      - [N, L] with complex     (split real/imag)
      - [N, 2*L] with real      (assume interleaved IQ, reshape to [N, L, 2])
    """
    if A.ndim == 3:
        if A.shape[-1] == 2:
            return A
        if A.shape[1] == 2:
            return np.transpose(A, (0, 2, 1))
        raise ValueError(f"{name}: 3D but cannot infer IQ channel in {A.shape}")

    if A.ndim == 2:
        # complex case: split into I,Q
        if np.iscomplexobj(A):
            return np.stack([A.real, A.imag], axis=-1).astype(np.float32)
        # interleaved real IQ along last axis?
        N, W = A.shape
        if W % 2 == 0:
            L = W // 2
            A2 = A.reshape(N, L, 2)
            return A2
        raise ValueError(f"{name}: 2D real array {A.shape} is not even-width; "
                         f"can't infer interleaved IQ. Provide complex or [N,L,2].")

    raise ValueError(f"{name}: Expected 2D/3D array, got {A.shape}")

def ensure_three_dim_xy(X, Y):
    Xn = _to_N_L_2(X, "X")
    Yn = _to_N_L_2(Y, "Y")
    if Xn.shape != Yn.shape:
        raise ValueError(f"X and Y shapes mismatch after normalization: {Xn.shape} vs {Yn.shape}")
    return Xn, Yn


class WindowSpec:
    __slots__ = ("seq_idx","start","take","pad_left","pad_right")
    def __init__(self, seq_idx, start, take, pad_left=0, pad_right=0):
        self.seq_idx = seq_idx
        self.start = start
        self.take = take
        self.pad_left = pad_left
        self.pad_right = pad_right

class WindowedIQDataset(Dataset):
    """
    Lazy windowing:
      - Does NOT store padded arrays.
      - Keeps only window specs and pads on-the-fly in __getitem__.
    """
    def __init__(self, X, Y, W=2048, H=512):
        X, Y = ensure_three_dim_xy(X, Y)
        assert X.shape == Y.shape and X.shape[-1] == 2
        self.X = X
        self.Y = Y
        self.W = int(W)
        self.H = int(H)
        self.specs = []

        for n in range(X.shape[0]):
            L = X[n].shape[0]
            if L < self.W:
                pad = self.W - L
                pre = pad // 2
                post = pad - pre
                # one centered window with padding
                self.specs.append(WindowSpec(n, start=0, take=L, pad_left=pre, pad_right=post))
            elif L == self.W:
                self.specs.append(WindowSpec(n, start=0, take=L))
            else:
                # rolling windows that emit last H
                start = 0
                while start + self.W <= L:
                    self.specs.append(WindowSpec(n, start=start, take=self.W))
                    start += self.H
                # tail coverage (right-aligned)
                if start < L:
                    self.specs.append(WindowSpec(n, start=L - self.W, take=self.W))

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, i):
        s = self.specs[i]
        x = self.X[s.seq_idx]
        y = self.Y[s.seq_idx]
        if s.pad_left or s.pad_right:
            # centered pad case
            xw = np.pad(x[:s.take], ((s.pad_left, s.pad_right), (0,0)), mode="constant")
            yw = np.pad(y[:s.take], ((s.pad_left, s.pad_right), (0,0)), mode="constant")
        else:
            xw = x[s.start:s.start + self.W]
            yw = y[s.start:s.start + self.W]
        return to_tensor(xw), to_tensor(yw)

# ---------------------------
# Model: complex-aware causal TCN
# ---------------------------

class ComplexMix(nn.Module):
    """
    Per-channel learned complex 2x2 real mixing:
       [[a, -b],
        [b,  a]]  applied to [I,Q]
    Implemented as a 1x1 conv over channels with constraints.
    """
    def __init__(self, channels=2):
        super().__init__()
        assert channels == 2
        # parameters a,b as scalars or 1x1 conv weights
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        # x: [B, T, 2]
        a, b = self.a, self.b
        I = x[..., 0]
        Q = x[..., 1]
        Io = a*I - b*Q
        Qo = b*I + a*Q
        return torch.stack([Io, Qo], dim=-1)

class CausalDepthwiseSepBlock(nn.Module):
    def __init__(self, ch, kernel=5, dilation=1, dropout=0.0):
        super().__init__()
        pad = (kernel - 1) * dilation  # causal padding at left
        self.pad = nn.ConstantPad1d((pad, 0), 0.0)  # pad on left only; operates on [B,C,T]
        self.dw = nn.Conv1d(ch, ch, kernel_size=kernel, dilation=dilation,
                            groups=ch, bias=True)
        self.pw = nn.Conv1d(ch, ch, kernel_size=1, bias=True)
        self.act = nn.SiLU()
        self.norm = nn.GroupNorm(1, ch)
        self.dropout = nn.Dropout(dropout)
        self.res = nn.Conv1d(ch, ch, kernel_size=1, bias=True)

    def forward(self, x):  # x: [B,C,T]
        h = self.pad(x)
        h = self.dw(h)
        h = self.pw(h)
        h = self.norm(h)
        h = self.act(h)
        h = self.dropout(h)
        return self.act(self.res(x) + h)

class CausalTCN(nn.Module):
    def __init__(self, width=128, blocks=10, kernel=5, dil_start=1, dil_max=None, dropout=0.0):
        super().__init__()
        self.mix = ComplexMix()

        self.in1 = nn.Conv1d(2, width, kernel_size=1)
        layers = []
        dil = dil_start
        for i in range(blocks):
            layers.append(CausalDepthwiseSepBlock(width, kernel=kernel, dilation=dil, dropout=dropout))
            dil = min(dil*2, dil_max if dil_max else dil*2)
        self.tcn = nn.Sequential(*layers)
        self.out = nn.Conv1d(width, 2, kernel_size=1)  # residual prediction

    def forward(self, x):
        # x: [B,T,2] -> mix -> [B,2,T]
        x_mix = self.mix(x)
        h = x_mix.permute(0,2,1)
        h = self.in1(h)
        h = self.tcn(h)
        r = self.out(h).permute(0,2,1)  # [B,T,2]
        yhat = x + r  # residual connection to input
        return yhat, r

# ---------------------------
# Losses & Metrics
# ---------------------------

class TimeDomainLoss(nn.Module):
    """
    L = L1(ŷ_H, y_H) + alpha * L1(Δŷ_H, Δy_H)
    where _H means last H samples (causal emission region).
    """
    def __init__(self, alpha=0.05):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()

    def forward(self, yhat, y, H):
        # yhat,y: [B, W, 2]; score last H along W
        if H <= 0 or H > y.shape[1]:
            H = y.shape[1]
        yhat_h = yhat[:, -H:, :]
        y_h = y[:, -H:, :]
        base = self.l1(yhat_h, y_h)
        # small first-difference penalty (keeps high-freq crisp)
        d1 = pairwise_diff(yhat_h)
        d2 = pairwise_diff(y_h)
        delta = self.l1(d1, d2)
        return base + self.alpha * delta

@torch.no_grad()
def evm_stats(y_true, y_pred, eps=1e-12):
    """
    Returns: (EVM_percent, EVM_dB)
      EVM_lin = sqrt( mean(|e|^2) / mean(|ref|^2) )
      EVM%    = 100 * EVM_lin
      EVM dB  = 20*log10(EVM_lin)
    """
    err = y_true - y_pred  # [B,T,2]
    num = (err**2).sum(dim=-1).mean()
    den = (y_true**2).sum(dim=-1).mean().clamp_min(eps)
    evm_lin = torch.sqrt(num / den)
    evm_pct = (100.0 * evm_lin).item()
    evm_db  = (20.0 * torch.log10(evm_lin.clamp_min(eps))).item()
    return evm_pct, evm_db

@torch.no_grad()
def snr_db(signal, observed, eps=1e-12):
    """
    SNR between a clean reference 'signal' and an observed version 'observed':
      noise = observed - signal
      SNR = 10*log10( P_signal / P_noise )
    """
    noise = observed - signal
    p_sig = (signal**2).sum(dim=-1).mean().clamp_min(eps)
    p_nse = (noise**2).sum(dim=-1).mean().clamp_min(eps)
    return (10.0 * torch.log10(p_sig / p_nse)).item()


# ---------------------------
# Training / Evaluation
# ---------------------------

def make_loaders(Xtr, Ytr, Xva, Yva, Xte, Yte, W, H, batch, num_workers=0):
    ds_tr = WindowedIQDataset(Xtr, Ytr, W=W, H=H)
    ds_va = WindowedIQDataset(Xva, Yva, W=W, H=H)
    ds_te = WindowedIQDataset(Xte, Yte, W=W, H=H)
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, drop_last=True, num_workers=num_workers)
    dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False, drop_last=False, num_workers=num_workers)
    dl_te = DataLoader(ds_te, batch_size=batch, shuffle=False, drop_last=False, num_workers=num_workers)
    return dl_tr, dl_va, dl_te

def train_one_epoch(model, loss_fn, opt, dl, device, H, max_steps=0, accum_steps=1, scheduler=None, amp=False):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    run = {"loss": 0.0, "n": 0}
    opt.zero_grad(set_to_none=True)
    for step, (x, y) in enumerate(dl, 1):
        x = x.to(device); y = y.to(device)
        with torch.cuda.amp.autocast(enabled=amp):
            yhat, _ = model(x)
            loss = loss_fn(yhat, y, H) / accum_steps
        scaler.scale(loss).backward()
        if step % accum_steps == 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            if scheduler: scheduler.step()
        bs = x.size(0)
        run["loss"] += loss.item() * accum_steps * bs
        run["n"] += bs
        if step % 50 == 0:
            cur_lr = next(iter(opt.param_groups))["lr"]
            print(f"  step {step:05d} | batch_loss {(loss.item()*accum_steps):.6f} | lr {cur_lr:.2e}")
        if max_steps and step >= max_steps:
            break
    return run["loss"]/max(1,run["n"])



@torch.no_grad()
def evaluate(model, loss_fn, dl, device, H):
    model.eval()
    acc = {"loss":0.0,"evm_pct":0.0,"evm_db":0.0,"snr_in":0.0,"snr_out":0.0,"n":0}
    for x, y in dl:
        x = x.to(device); y = y.to(device)
        yhat, _ = model(x)
        loss = loss_fn(yhat, y, H)

        # emit region
        Huse = min(H, y.shape[1])
        yt = y[:, -Huse:, :]
        yh = yhat[:, -Huse:, :]
        xx = x[:, -Huse:, :]

        evm_pct, evm_dbv = evm_stats(yt, yh)
        snr_in  = snr_db(yt, xx)   # jammed vs clean
        snr_out = snr_db(yt, yh)   # denoised vs clean

        bs = x.size(0)
        acc["loss"]    += loss.item()*bs
        acc["evm_pct"] += evm_pct*bs
        acc["evm_db"]  += evm_dbv*bs
        acc["snr_in"]  += snr_in*bs
        acc["snr_out"] += snr_out*bs
        acc["n"]       += bs

    for k in ("loss","evm_pct","evm_db","snr_in","snr_out"):
        acc[k] = acc[k]/max(1,acc["n"])
    return acc


# ---------------------------
# Streaming overlap-add
# ---------------------------

@torch.no_grad()
def stream_overlap_add(x_np, model, W=2048, H=512, device=None):
    """
    x_np: [L,2] numpy (jammed/noisy input)
    returns: y_hat [L,2] numpy (denoised)
    Causal: each window emits only last H; we OLA them back.
    """
    model.eval()
    device = device or next(model.parameters()).device
    x = x_np.astype(np.float32)
    L = x.shape[0]
    out = np.zeros_like(x, dtype=np.float32)
    weight = np.zeros((L,1), dtype=np.float32)

    # Hann-like synthesis window on emitted region to smooth overlap
    win = np.hanning(2*H)[H:]  # length H, smooth rise
    win = win.astype(np.float32)
    start = 0
    while start < L:
        # take window (left-align if near end)
        s = start - (W - H)
        if s < 0: s = 0
        e = min(s + W, L)
        # pad if needed
        xw = x[s:e]
        if len(xw) < W:
            pad = W - len(xw)
            xw = np.pad(xw, ((W - len(xw), 0),(0,0)), mode='constant')  # left pad to keep causality
        xt = torch.from_numpy(xw).unsqueeze(0).to(device)  # [1,W,2]
        yhat, _ = model(xt)  # [1,W,2]
        yhat = yhat[:, -H:, :].squeeze(0).cpu().numpy()  # [H,2]

        # place emitted H back at [start : start+H]
        end_emit = min(start+H, L)
        span = end_emit - start
        out[start:end_emit] += yhat[:span] * win[:span, None]
        weight[start:end_emit, 0] += win[:span]
        start += H

    # avoid division by zero
    weight[weight==0] = 1.0
    return (out / weight).astype(np.float32)

# ---------------------------
# Main
# ---------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Path to dataset .npz")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--blocks", type=int, default=10)
    p.add_argument("--kernel", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--W", type=int, default=2048, help="window length")
    p.add_argument("--H", type=int, default=512, help="emitted/score hop")
    p.add_argument("--alpha", type=float, default=0.05, help="ΔL1 loss weight")
    p.add_argument("--save", type=str, default="tcn_denoiser.pt")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=200, help="Limit train steps per epoch (0=all)")
    p.add_argument("--wd", type=float, default=1e-2, help="weight decay (AdamW)")
    p.add_argument("--accum_steps", type=int, default=1, help="gradient accumulation steps")
    p.add_argument("--sched", type=str, default="cosine", choices=["none","cosine"])
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--amp", action="store_true", help="enable autocast (CUDA only)")

    args = p.parse_args()

    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = load_npz_auto(args.data)
    dl_tr, dl_va, dl_te = make_loaders(Xtr, Ytr, Xva, Yva, Xte, Yte, args.W, args.H, args.batch, args.num_workers)
    print(f"[dataset] train windows={len(dl_tr.dataset)} | val={len(dl_va.dataset)} | test={len(dl_te.dataset)}")

        # --- Sanity probe: peek one batch ---
    xb, yb = next(iter(dl_tr))
    Hprobe = min(xb.shape[1], args.H)
    x_h = xb[:, -Hprobe:, :]
    y_h = yb[:, -Hprobe:, :]
    def rms(a): return float(torch.sqrt((a**2).mean()))
    print(f"[probe] H={Hprobe} | x_rms={rms(x_h):.6g} | y_rms={rms(y_h):.6g} | "
        f"x_mean={float(x_h.mean()):.3e} | y_mean={float(y_h.mean()):.3e}")
    print(f"[probe] shapes: xb={tuple(xb.shape)} (B,W,2)  yb={tuple(yb.shape)}")

    device = device_auto()
    model = CausalTCN(width=args.width, blocks=args.blocks, kernel=args.kernel,
                      dil_start=1, dil_max=512, dropout=args.dropout).to(device)
    loss_fn = TimeDomainLoss(alpha=args.alpha)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    def make_scheduler(opt, steps_per_epoch, epochs, warmup_epochs=0):
        if args.sched == "none":
            return None
        total = steps_per_epoch * epochs
        warm = steps_per_epoch * warmup_epochs
        def lr_lambda(step):
            if step < warm:
                return max(1e-3, step / max(1, warm))  # linear warmup
            t = (step - warm) / max(1, total - warm)
            return 0.5 * (1 + math.cos(math.pi * t))   # cosine decay
        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # after you build dl_tr:
    steps_per_epoch = len(dl_tr) if args.max_steps == 0 else min(args.max_steps, len(dl_tr))
    scheduler = make_scheduler(opt, steps_per_epoch, args.epochs, args.warmup_epochs)


    best_val = float("inf")
    print(f"Device: {device} | Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    for epoch in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, loss_fn, opt, dl_tr, device, args.H, args.max_steps, args.accum_steps, scheduler, args.amp)

        va = evaluate(model, loss_fn, dl_va, device, args.H)
        dSNR = va["snr_out"] - va["snr_in"]
        print(
            f"Epoch {epoch:03d} | train {tr_loss:.6f} | val {va['loss']:.6f} | "
            f"EVM% {va['evm_pct']:.2f} ({va['evm_db']:.2f} dB) | "
            f"SNR_in {va['snr_in']:.2f} dB → SNR_out {va['snr_out']:.2f} dB | Δ {dSNR:+.2f} dB"
        )


        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save({"model": model.state_dict(),
                        "cfg": vars(args)}, args.save)
            print(f"  ↳ saved -> {args.save}")

    # Final test
    te = evaluate(model, loss_fn, dl_te, device, args.H)
    print(f"Epoch {epoch:03d} | train {tr_loss:.6f} | val {va['loss']:.6f} | "f"EVM% {va['evm_pct']:.2f} ({va['evm_db']:.2f} dB) | "f"SNR_in {va['snr_in']:.2f} dB → SNR_out {va['snr_out']:.2f} dB | Δ {dSNR:+.2f} dB")

if __name__ == "__main__":
    main()
