#!/usr/bin/env python3
# npz_train_mlp.py — wide MLP denoiser with residual target, GELU+LayerNorm, cosine LR + warmup
# Extras: EMA weights, spectral warmup/auto-balance, MR spectral loss, grad clipping, EVM% & SNR metrics.

import os, json, argparse, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Defaults
# ---------------------------
CKPT_DEFAULT = "mlp_denoiser.pt"
EPOCHS_DEFAULT = 50
BATCH_SIZE_DEFAULT = 512
LR_DEFAULT = 1e-3
WEIGHT_DECAY_DEFAULT = 1e-4
SPECTRAL_WEIGHT_DEFAULT = 0.0
PDROP_DEFAULT = 0.0

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ---------------------------
# Dataset
# ---------------------------
class IQDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        # flatten from axis=1 onward, keep N as is
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        if Y.ndim > 2:
            Y = Y.reshape(Y.shape[0], -1)
        self.X = torch.from_numpy(X.astype(np.float32, copy=False))
        self.Y = torch.from_numpy(Y.astype(np.float32, copy=False))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


# ---------------------------
# Model: residual-output MLP (y_hat = x_norm + r_hat)
# ---------------------------
class ResOutMLP(nn.Module):
    """
    Predicts a residual correction r_hat so that y_hat = x_norm + r_hat.
    Body: [Linear -> GELU -> (Dropout)] x L, with a LayerNorm on input.
    """
    def __init__(self, in_dim: int, hidden: list[int], dropout: float = 0.0):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim)
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, in_dim)
        # small init on the head helps stabilize residual learning
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x_n = self.in_norm(x)
        r_hat = self.head(self.body(x_n))
        y_hat = x_n + r_hat  # residual correction applied to normalized input
        # Return prediction, residual, and normalized input (for metrics/loss)
        return y_hat, r_hat, x_n


# ---------------------------
# Spectral losses
# ---------------------------
def spectral_loss(y_hat: torch.Tensor,
                  y_true: torch.Tensor,
                  power: float = 1.0) -> torch.Tensor:
    """
    Single-resolution magnitude-spectrum MSE using rFFT along the last dim.
    """
    y_hat = y_hat.float()
    y_true = y_true.float()
    Yh = torch.fft.rfft(y_hat, dim=-1)
    Yt = torch.fft.rfft(y_true, dim=-1)
    Mh = (Yh.abs() + 1e-8) ** power
    Mt = (Yt.abs() + 1e-8) ** power
    return torch.mean((Mh - Mt) ** 2)


def mr_spectral_loss(y_hat: torch.Tensor,
                     y_true: torch.Tensor,
                     sizes: list[int],
                     logmag: bool = False,
                     power: float = 1.0) -> torch.Tensor:
    """
    Multi-resolution spectral loss: average of per-size magnitude (or log-magnitude) MSEs.
    """
    y_hat = y_hat.float()
    y_true = y_true.float()
    tot = 0.0
    count = 0
    for n in sizes:
        Yh = torch.fft.rfft(y_hat, n=n, dim=-1)
        Yt = torch.fft.rfft(y_true, n=n, dim=-1)
        Mh = (Yh.abs() + 1e-8)
        Mt = (Yt.abs() + 1e-8)
        if logmag:
            Mh = Mh.log()
            Mt = Mt.log()
        else:
            Mh = Mh ** power
            Mt = Mt ** power
        tot = tot + torch.mean((Mh - Mt) ** 2)
        count += 1
    return tot / max(1, count)


def make_spec_fn(args, in_dim: int):
    """
    Returns a function spec_fn(y_hat, y_true) -> scalar tensor based on args.
    If spec_weight <= 0, returns a no-op fn.
    """
    if args.spec_weight <= 0:
        return lambda yh, yt: yh.new_zeros(())

    if args.mr_spec:
        if args.mr_scales:
            try:
                sizes = json.loads(args.mr_scales)
                assert isinstance(sizes, list) and all(isinstance(x, int) and x > 0 for x in sizes)
            except Exception:
                raise ValueError(f"Invalid --mr_scales '{args.mr_scales}'. Use JSON list, e.g. \"[D, D//2, D//4]\".")
        else:
            # sensible defaults from feature dim
            D = in_dim
            sizes = sorted(set([D, max(D // 2, 8), max(D // 4, 8)]), reverse=True)

        def _spec(yh, yt):
            return mr_spectral_loss(yh, yt, sizes=sizes, logmag=args.mr_log, power=1.0)
        return _spec
    else:
        return lambda yh, yt: spectral_loss(yh, yt, power=1.0)


# ---------------------------
# Utils
# ---------------------------
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_hidden(s: str | None, in_dim: int) -> list[int]:
    if s is None:
        # sensible wide default; user can override with --hidden
        return [1024, 2048, 4096]
    try:
        h = json.loads(s)
        assert isinstance(h, list) and all(isinstance(x, int) and x > 0 for x in h)
        return h
    except Exception:
        raise ValueError(f"Invalid --hidden '{s}'. Use JSON list, e.g. \"[1024,2048,4096]\".")


def init_ema(model: nn.Module):
    return [p.detach().clone() for p in model.parameters()]


@torch.no_grad()
def ema_update(model: nn.Module, ema_params, decay: float):
    for p, e in zip(model.parameters(), ema_params):
        e.mul_(decay).add_(p.data, alpha=1.0 - decay)


@torch.no_grad()
def swap_to_ema(model: nn.Module, ema_params):
    backup = [p.detach().clone() for p in model.parameters()]
    for p, e in zip(model.parameters(), ema_params):
        p.copy_(e)
    return backup


@torch.no_grad()
def restore_from_backup(model: nn.Module, backup):
    for p, b in zip(model.parameters(), backup):
        p.copy_(b)


# ---------------------------
# Training / Validation
# ---------------------------
def train_one_epoch(model, opt, scaler, dl, device, spec_w, spec_fn, clip_grad, ema_params, ema_decay):
    model.train()
    mse = nn.MSELoss()
    total, res_sum, spec_sum = 0.0, 0.0, 0.0

    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler is not None):
            y_hat, r_hat, x_norm = model(x)
            r_tgt = y - x_norm
            l_res = mse(r_hat, r_tgt)
            l_spec = spec_fn(y_hat, y) if spec_w > 0 else y_hat.new_zeros(())
            loss = l_res + spec_w * l_spec

        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()

        # EMA update after optimizer step
        if ema_params is not None:
            ema_update(model, ema_params, ema_decay)

        bs = x.shape[0]
        total += loss.detach().item() * bs
        res_sum += l_res.detach().item() * bs
        spec_sum += (l_spec.detach().item() if spec_w > 0 else 0.0) * bs

    n = len(dl.dataset)
    return total / n, res_sum / n, spec_sum / max(1, n)


@torch.no_grad()
def validate(model, dl, device, spec_w, spec_fn):
    model.eval()
    mse = nn.MSELoss()
    total, res_sum, spec_sum = 0.0, 0.0, 0.0

    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y_hat, r_hat, x_norm = model(x)
        r_tgt = y - x_norm
        l_res = mse(r_hat, r_tgt)
        l_spec = spec_fn(y_hat, y) if spec_w > 0 else y_hat.new_zeros(())
        loss = l_res + spec_w * l_spec

        bs = x.shape[0]
        total += loss.item() * bs
        res_sum += l_res.item() * bs
        spec_sum += (l_spec.item() if spec_w > 0 else 0.0) * bs

    n = len(dl.dataset)
    return total / n, res_sum / n, spec_sum / max(1, n)


# ---------------------------
# Metrics (EVM% and SNR dB)
# ---------------------------
@torch.no_grad()
def compute_evm_snr(model: nn.Module, dl: DataLoader, device: torch.device):
    """
    Computes:
      - EVM_in %  between x_norm and y
      - EVM_out % between y_hat and y
      - SNR_in dB  = 10*log10(||y||^2 / ||x_norm - y||^2)
      - SNR_out dB = 10*log10(||y||^2 / ||y_hat - y||^2)
      - Δ = SNR_out - SNR_in
    Aggregated over the entire dataloader.
    """
    model.eval()
    eps = 1e-12
    num_y = 0.0
    err_in = 0.0
    err_out = 0.0

    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y_hat, _, x_norm = model(x)

        # Accumulate in float64 for numeric stability
        y_d = y.double()
        in_err = (x_norm.double() - y_d)
        out_err = (y_hat.double() - y_d)

        num_y += torch.sum(y_d * y_d).item()
        err_in += torch.sum(in_err * in_err).item()
        err_out += torch.sum(out_err * out_err).item()

    # EVM% (RMS)
    evm_in_pct = 100.0 * math.sqrt(max(err_in, 0.0) / max(num_y, eps))
    evm_out_pct = 100.0 * math.sqrt(max(err_out, 0.0) / max(num_y, eps))

    # SNR (dB)
    snr_in = 10.0 * math.log10(max(num_y, eps) / max(err_in, eps))
    snr_out = 10.0 * math.log10(max(num_y, eps) / max(err_out, eps))
    snr_delta = snr_out - snr_in

    return evm_in_pct, evm_out_pct, snr_in, snr_out, snr_delta


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="NPZ MLP denoiser (residual target, GELU+LayerNorm, cosine warmup)")
    p.add_argument("--data", type=str, required=True, help="Path to NPZ dataset")
    p.add_argument("--ckpt", type=str, default=CKPT_DEFAULT)
    p.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE_DEFAULT)
    p.add_argument("--lr", type=float, default=LR_DEFAULT)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY_DEFAULT)
    p.add_argument("--spec_weight", type=float, default=SPECTRAL_WEIGHT_DEFAULT)
    p.add_argument("--pdrop", type=float, default=PDROP_DEFAULT)
    p.add_argument("--hidden", type=str, default=None, help='JSON list, e.g. "[1024,2048,4096]"')
    p.add_argument("--workers", type=int, default=None, help="DataLoader workers (default=min(8, cpu_count))")
    p.add_argument("--prefetch", type=int, default=4, help="DataLoader prefetch_factor")
    p.add_argument("--amp", action="store_true", default=True, help="Enable mixed precision (AMP)")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--compile", action="store_true", default=False, help="Enable torch.compile if available")
    p.add_argument("--no-compile", dest="compile", action="store_false")
    # LR schedule
    p.add_argument("--warmup_epochs", type=int, default=3, help="Linear LR warmup epochs")
    p.add_argument("--eta_min", type=float, default=0.0, help="Cosine min LR (0 -> 0.05*lr)")
    # Extras
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay (0 disables, e.g., 0.999)")
    p.add_argument("--spec_warmup", type=int, default=5, help="Epochs to ramp spec_weight 0→target")
    p.add_argument("--spec_autobalance", action="store_true", help="Scale spec_weight by res/spec ratio each epoch")
    p.add_argument("--clip_grad", type=float, default=0.0, help="Global grad-norm clip (0 disables)")
    # MR spectral loss
    p.add_argument("--mr_spec", action="store_true", help="Use multi-resolution spectral loss")
    p.add_argument("--mr_log", action="store_true", help="Use log-magnitude in MR spectral loss")
    p.add_argument("--mr_scales", type=str, default=None, help='JSON list of FFT sizes, e.g. "[D, D//2, D//4]"')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------
    # Load NPZ
    # ---------------------------
    npz = np.load(args.data, allow_pickle=True)
    keys = list(npz.keys())
    print(f"[NPZ] Keys: {keys}")
    Xtr = npz["Xtr"]; Ytr = npz["Ytr"]
    Xva = npz["Xva"]; Yva = npz["Yva"]
    meta = npz.get("meta", None)
    if meta is not None:
        try:
            print(f"[meta] {meta.item() if hasattr(meta, 'item') else meta}")
        except Exception:
            pass

    tr_set = IQDataset(Xtr, Ytr)
    va_set = IQDataset(Xva, Yva)

    in_dim = tr_set.X.shape[1]
    hidden = parse_hidden(args.hidden, in_dim)

    workers = args.workers
    if workers is None:
        workers = min(8, max(0, (os.cpu_count() or 4) - 1))

    prefetch = args.prefetch if workers > 0 else None

    tr_loader = DataLoader(
        tr_set, batch_size=args.batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, prefetch_factor=prefetch, persistent_workers=(workers > 0)
    )
    va_loader = DataLoader(
        va_set, batch_size=args.batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, prefetch_factor=prefetch, persistent_workers=(workers > 0)
    )

    # ---------------------------
    # Build model
    # ---------------------------
    model = ResOutMLP(in_dim=in_dim, hidden=hidden, dropout=args.pdrop).to(device)
    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="max-autotune")
            print("[info] torch.compile enabled")
        except Exception as e:
            print(f"[warn] torch.compile failed: {e}")

    params = count_params(model)
    print(f"Device: {device.type} | Params: {params/1e6:.2f}M | InDim: {in_dim} | Hidden: {hidden}")

    # ---------------------------
    # Optimizer (weight-decay hygiene) & Scheduler
    # ---------------------------
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.ndim == 1 or "norm" in n.lower() or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.99)
    )

    warmup_epochs = max(0, int(args.warmup_epochs))
    cos_epochs = max(1, args.epochs - warmup_epochs)
    eta_min = (0.05 * args.lr) if args.eta_min == 0.0 else args.eta_min
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cos_epochs, eta_min=eta_min)

    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    # EMA setup
    ema_params = init_ema(model) if args.ema > 0 else None

    # Spectral loss function
    spec_fn = make_spec_fn(args, in_dim)

    # ---------------------------
    # Train
    # ---------------------------
    best_val = float("inf")
    since = time.time()
    last_ratio = 1.0  # for spec_autobalance (res/spec)
    for ep in range(1, args.epochs + 1):
        # LR Warmup
        if ep <= warmup_epochs:
            w = ep / max(1, warmup_epochs)
            for g in opt.param_groups:
                g["lr"] = args.lr * w

        # Spec schedule for this epoch
        spec_w = args.spec_weight
        if args.spec_warmup > 0 and ep <= args.spec_warmup:
            spec_w *= (ep / max(1, args.spec_warmup))
        # Apply auto-balance based on last epoch's ratio
        if args.spec_autobalance:
            spec_w *= max(0.2, min(5.0, last_ratio))

        tr_loss, tr_res, tr_spec = train_one_epoch(
            model, opt, scaler, tr_loader, device,
            spec_w=spec_w, spec_fn=spec_fn,
            clip_grad=args.clip_grad,
            ema_params=ema_params, ema_decay=args.ema if args.ema > 0 else 0.0
        )

        # Update ratio for next epoch (avoid div by zero)
        if tr_spec > 0:
            last_ratio = tr_res / (tr_spec + 1e-12)

        va_loss, va_res, va_spec = validate(
            model, va_loader, device, spec_w=spec_w, spec_fn=spec_fn
        )

        # After warmup, step cosine
        if ep > warmup_epochs:
            cosine.step()

        curr_lr = opt.param_groups[0]["lr"]
        print(
            f"Epoch {ep:03d} | "
            f"train {tr_loss:.5f} (res {tr_res:.5f}, spec {tr_spec:.5f}) | "
            f"val {va_loss:.5f} | lr {curr_lr:.2e} | spec_w_eff {spec_w:.3g}"
        )

        # Save best (prefer EMA weights if enabled)
        if va_loss < best_val:
            best_val = va_loss
            if ema_params is not None:
                backup = swap_to_ema(model, ema_params)
            torch.save(
                {"model": model.state_dict(),
                 "in_dim": in_dim,
                 "hidden": hidden,
                 "args": vars(args),
                 "val_loss": best_val},
                args.ckpt,
            )
            print(f"  ↳ saved -> {args.ckpt}")
            if ema_params is not None:
                restore_from_backup(model, backup)

    total_time = time.time() - since
    print(f"Done. Best val {best_val:.6f} | time {total_time/60:.2f} min")

    # ---------------------------
    # Evaluate best checkpoint on validation set (EVM% and SNR)
    # ---------------------------
    print("=== Evaluating best checkpoint on validation set ===")
    ckpt = torch.load(args.ckpt, map_location=device)
    eval_model = ResOutMLP(in_dim=ckpt.get("in_dim", in_dim),
                           hidden=ckpt.get("hidden", hidden),
                           dropout=args.pdrop).to(device)
    eval_model.load_state_dict(ckpt["model"])
    evm_in, evm_out, snr_in, snr_out, snr_delta = compute_evm_snr(eval_model, va_loader, device)

    print(
        f"=== VALIDATION METRICS ===\n"
        f"EVM_in   : {evm_in:.3f}%\n"
        f"EVM_out  : {evm_out:.3f}%\n"
        f"SNR_in   : {snr_in:.2f} dB\n"
        f"SNR_out  : {snr_out:.2f} dB\n"
        f"Δ SNR    : {snr_delta:+.2f} dB"
    )


if __name__ == "__main__":
    main()
