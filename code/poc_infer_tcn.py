#!/usr/bin/env python3
"""
poc_infer_tcn.py — Proof-of-concept inference for full TCN, separable 1D, or sep2d.
- Loads either a TorchScript .ts OR a PyTorch .pt (EMA/raw) checkpoint.
- Evaluates EVM/SNR on your val split using SAME windowing as training.
- Benchmarks ms per block and samples/sec.

Examples:
  # TorchScript (recommended for CPU speed)
  python poc_infer_tcn.py --data artifacts/gnss_synth_sweepcw_500k.npz \
    --ckpt tcn_sep2d64x6_W256.ts --separable --sep2d \
    --width 64 --blocks 6 --kernel 5 --W 256 --H 256 --device cpu --workers 0

  # PyTorch .pt (EMA pure export)
  python poc_infer_tcn.py --data artifacts/gnss_synth_sweepcw_500k.npz \
    --ckpt tcn_denoiser_ema_model.pt --separable --sep2d \
    --width 64 --blocks 6 --kernel 5 --W 256 --H 256 --device cpu --workers 0
"""

import argparse, os, time, math
import torch
import numpy as np
import train_tcn as tcn  # uses your dataset + model definitions (incl. SepTCNBlock2D)

# ---------- loading ----------

def load_model_from_ckpt(ckpt_path: str,
                         width: int, blocks: int, kernel: int,
                         dropout: float, residual: bool, use_bn: bool,
                         separable: bool, sep2d: bool,
                         device: torch.device):
    path_lower = ckpt_path.lower()
    # TorchScript path
    if path_lower.endswith(".ts"):
        m = torch.jit.load(ckpt_path, map_location=device)
        m.eval()
        print("[load] Loaded TorchScript model.")
        return m, True  # is_ts

    # Eager model (must match architecture used in training)
    model = tcn.TCN(in_ch=2, ch=width, out_ch=2, k=kernel, blocks=blocks,
                    dropout=dropout, use_bn=use_bn, residual=residual,
                    separable=separable, sep2d=sep2d).to(device)
    model.eval()

    ck = torch.load(ckpt_path, map_location=device)
    if "model" in ck and isinstance(ck["model"], dict):
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        if missing or unexpected:
            print(f"[warn] missing={missing}, unexpected={unexpected}")
        print("[load] Loaded weights from 'model' (pure export).")
        return model, False

    if "ema_state" in ck and isinstance(ck["ema_state"], dict):
        sd = model.state_dict()
        for n, v in ck["ema_state"].items():
            if n in sd: sd[n] = v.to(sd[n].device)
        model.load_state_dict(sd, strict=True)
        print("[load] Constructed model from 'ema_state' (EMA).")
        return model, False

    if "model" in ck:
        print("[warn] No 'ema_state'; using raw 'model' weights.")
        model.load_state_dict(ck["model"], strict=True)
        return model, False

    raise ValueError(f"Unsupported checkpoint format. Keys: {list(ck.keys())[:8]}...")

# ---------- eval / metrics ----------

@torch.no_grad()
def eval_on_loader(model, loss_fn, dl, device, H: int, is_ts: bool):
    model.eval()
    tot_loss = torch.zeros((), dtype=torch.float64, device=device)
    n_loss   = torch.zeros((), dtype=torch.float64, device=device)
    err_pow  = torch.zeros((), dtype=torch.float64, device=device)
    ref_pow  = torch.zeros((), dtype=torch.float64, device=device)
    sig_pow  = torch.zeros((), dtype=torch.float64, device=device)
    nse_in   = torch.zeros((), dtype=torch.float64, device=device)
    nse_out  = torch.zeros((), dtype=torch.float64, device=device)
    kept_s   = torch.zeros((), dtype=torch.float64, device=device)
    total_s  = torch.zeros((), dtype=torch.float64, device=device)
    eps = 1e-12

    for x, y in dl:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        yhat = model(x)
        if not is_ts and isinstance(yhat, (tuple, list)):  # eager model returns (y, None)
            yhat = yhat[0]
        loss = loss_fn(yhat, y, H)
        bs = x.size(0)
        tot_loss += loss.detach() * bs; n_loss += bs

        m = (y.pow(2).sum(dim=-1, keepdim=True) > eps).float()
        valid = (m.sum(dim=(1, 2)) > 0)
        if valid.any():
            m = m[valid]; yt = y[valid]*m; yh = yhat[valid]*m; xx = x[valid]*m
            err_pow += (yh - yt).pow(2).sum(dtype=torch.float64)
            ref_pow += yt.pow(2).sum(dtype=torch.float64)
            sig_pow += yt.pow(2).sum(dtype=torch.float64)
            nse_in  += (xx - yt).pow(2).sum(dtype=torch.float64)
            nse_out += (yh - yt).pow(2).sum(dtype=torch.float64)
            kept_s  += m.sum(dtype=torch.float64)
            total_s += torch.tensor(float(m.numel()), dtype=torch.float64, device=device)

    mean_loss = (tot_loss / torch.clamp_min(n_loss, 1)).item()
    if ref_pow.item() <= eps or kept_s.item() == 0:
        return {"loss": mean_loss, "evm_pct": float("nan"), "evm_db": float("nan"),
                "snr_in": float("nan"), "snr_out": float("nan"), "ref_cov": 0.0, "n": int(n_loss.item())}

    evm_lin = math.sqrt(float(err_pow / (ref_pow + 1e-12)))
    evm_pct = 100.0 * evm_lin
    evm_db  = 20.0 * math.log10(max(1e-12, evm_lin))
    snr_in  = 10.0 * math.log10(float((sig_pow + 1e-12) / (nse_in + 1e-12)))
    snr_out = 10.0 * math.log10(float((sig_pow + 1e-12) / (nse_out + 1e-12)))
    cov     = 100.0 * float(kept_s.item() / max(1.0, total_s.item()))
    return {"loss": mean_loss, "evm_pct": evm_pct, "evm_db": evm_db,
            "snr_in": snr_in, "snr_out": snr_out, "ref_cov": cov, "n": int(n_loss.item())}

@torch.no_grad()
def benchmark_ms_per_block(model, device, W=256, iters=200, warmup=20):
    model.eval()
    x = torch.randn(1, W, 2, device=device, dtype=torch.float32)
    for _ in range(warmup): model(x)
    t0 = time.perf_counter()
    for _ in range(iters): model(x)
    t1 = time.perf_counter()
    ms = 1000.0 * (t1 - t0) / iters
    sps = W / (ms * 1e-3)
    return ms, sps

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default="tcn_denoiser_ema_model.pt")   # .pt or .ts
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--kernel", type=int, default=7)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--use_bn", action="store_true")
    ap.add_argument("--residual", dest="residual", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--separable", action="store_true")
    ap.add_argument("--sep2d", action="store_true")  # <— fast Conv2d depthwise path
    ap.add_argument("--W", type=int, default=512)
    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--prefetch", type=int, default=2)
    ap.add_argument("--device", choices=["cuda","cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit_windows", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    if device.type == "cpu":
        # sensible defaults for CPU benchmarking
        torch.set_num_threads(os.cpu_count() or 4)
        torch.set_num_interop_threads(1)

    # Data (same windowing as training)
    ds_tr, ds_va, ds_te, dl_tr, dl_va, dl_te, W_eff = tcn.load_npz_dataset(
        args.data, W=args.W, H=args.H, batch=args.batch, workers=args.workers, prefetch=args.prefetch)
    if args.limit_windows:
        from itertools import islice
        n_batches = (args.limit_windows + args.batch - 1) // args.batch
        dl_va = list(islice(dl_va, n_batches))
        print(f"[fast] limiting validation to ~{args.limit_windows} windows ({n_batches} batches)")
    print(f"[data] val windows: {len(ds_va)} | W_eff={W_eff} H={args.H}")

    # Model
    model, is_ts = load_model_from_ckpt(args.ckpt, args.width, args.blocks, args.kernel,
                                        args.dropout, args.residual, args.use_bn,
                                        args.separable, args.sep2d, device)
    print(f"[model] width={args.width} blocks={args.blocks} k={args.kernel} "
          f"type={'sep2d' if args.sep2d else ('separable' if args.separable else 'full')} "
          f"| ckpt={'TorchScript' if is_ts else 'PyTorch'}")

    # Loss + eval
    loss_fn = tcn.MaskedTimeLoss(alpha=0.05, beta_evm_norm=0.02, spec_weight=0.01)
    val = eval_on_loader(model, loss_fn, dl_va, device, H=args.H, is_ts=is_ts)
    dsnr = (val["snr_out"] - val["snr_in"]) if (not math.isnan(val["snr_in"]) and not math.isnan(val["snr_out"])) else float("nan")
    print("\n=== POC VALIDATION (EMA inference) ===")
    print(f"loss {val['loss']:.6f} | EVM% {val['evm_pct']:.2f} ({val['evm_db']:.2f} dB)")
    print(f"SNR_in {val['snr_in']:.2f} → SNR_out {val['snr_out']:.2f} | Δ {dsnr:+.2f} dB | cov {val['ref_cov']:.1f}%")

    # Throughput
    ms_per_block, sps = benchmark_ms_per_block(model, device, W=args.W, iters=200, warmup=20)
    print(f"\n=== THROUGHPUT (device={device.type}) ===")
    print(f"{ms_per_block:.2f} ms per {args.W}-sample block  →  ~{sps:,.0f} samples/sec")

if __name__ == "__main__":
    main()
