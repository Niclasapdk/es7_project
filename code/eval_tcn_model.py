#!/usr/bin/env python3
"""
Evaluation script for the TCN jammer-denoiser.

- Loads a trained checkpoint (best.pt from chat_tcn_train.py)
- Runs inference on a chosen split of an NPZ dataset
- Computes per-sample SNR_in, SNR_out, ΔSNR and RMS EVM (%)
- Prints aggregate statistics and optionally saves all per-sample metrics to an .npz file

Usage example:

  python eval_tcn_jammer.py \
      --ckpt ckpts_tcn_jammer/best.pt \
      --data gnss_l1_sweptcw_hard_v3.npz \
      --split val \
      --batch 512 \
      --out eval_metrics_hard_val.npz
"""

import argparse
import time
from pathlib import Path
from torch import nn

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Import model + helpers from the training script
from chat_tcn_train import (
    ResidualTCN,
    perseq_rms_norm,
    snr_in_out_evm_per_sample,
)


# ---------------- Dataset helper ----------------

class IQWindows(Dataset):
    """
    Minimal dataset wrapper for [N, T, 2] IQ windows.
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        assert X.shape == Y.shape and X.ndim == 3 and X.shape[-1] == 2, \
            f"Bad shapes {X.shape} vs {Y.shape}"
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i: int):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])


# ---------------- NPZ split loader ----------------

def load_npz_split(path: str, split: str):
    """
    Load a specific split from an NPZ file.

    Expected keys:
      - train: Xtr, Ytr
      - val:   Xva, Yva
      - test:  Xte, Yte

    Fallbacks:
      - If only X/Y are present, use them directly (for val: last 10%).
    """
    npz = np.load(path)
    keys = set(npz.keys())

    if split == "train":
        if {"Xtr", "Ytr"}.issubset(keys):
            X, Y = npz["Xtr"], npz["Ytr"]
        elif {"X", "Y"}.issubset(keys):
            X, Y = npz["X"], npz["Y"]
        else:
            raise ValueError(f"No train split found in {path}")
    elif split == "val":
        if {"Xva", "Yva"}.issubset(keys):
            X, Y = npz["Xva"], npz["Yva"]
        elif {"X", "Y"}.issubset(keys):
            # Use last 10% as val if only X/Y exist
            X_all, Y_all = npz["X"], npz["Y"]
            N = X_all.shape[0]
            n_va = max(1, int(0.1 * N))
            X, Y = X_all[-n_va:], Y_all[-n_va:]
        else:
            raise ValueError(f"No val split found in {path}")
    elif split == "test":
        if {"Xte", "Yte"}.issubset(keys):
            X, Y = npz["Xte"], npz["Yte"]
        else:
            raise ValueError(f"No test split found in {path}")
    else:
        raise ValueError(f"Unknown split: {split}")

    return X, Y


# ---------------- Model loader ----------------

def load_model_from_ckpt(ckpt_path: str, device: torch.device):
    """
    Reconstruct ResidualTCN from a training checkpoint produced by chat_tcn_train.py.

    Note: checkpoints saved with EMA only store *parameters* (no BN running stats),
    so we load with strict=False and then force BatchNorm layers to use batch
    statistics at eval time.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model"]
    ck_args = ckpt.get("args", {})

    width = ck_args.get("width", 32)
    blocks = ck_args.get("blocks", 4)
    kernel = ck_args.get("kernel", 3)
    dropout = ck_args.get("dropout", 0.05)

    model = ResidualTCN(in_ch=2, hid=width, blocks=blocks, k=kernel, dropout=dropout)

    # Load parameters but allow missing BN buffers (running_mean / running_var)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("[load_model_from_ckpt] Warning: incompatible keys when loading state_dict:")
        if incompatible.missing_keys:
            print("  missing:", incompatible.missing_keys)
        if incompatible.unexpected_keys:
            print("  unexpected:", incompatible.unexpected_keys)

    model.to(device)

    # Put model in "eval" but keep BatchNorm layers using batch stats
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.train()  # BN uses batch statistics; dropout etc stay in eval

    perseq_norm = bool(ck_args.get("perseq_norm", False))

    return model, perseq_norm, ck_args

# ---------------- Evaluation loop ----------------

@torch.no_grad()
def run_eval(model: torch.nn.Module,
             loader: DataLoader,
             device: torch.device,
             perseq_norm: bool):
    """
    Evaluate model on a DataLoader, returning per-sample metrics and summary stats.
    """
    all_snr_in = []
    all_snr_out = []
    all_evm = []

    t0 = time.time()

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)  # [B, T, 2]
        yb = yb.to(device, non_blocking=True)

        if perseq_norm:
            xb, yb = perseq_rms_norm(xb, yb)

        # model expects [B, 2, T] jammer input, outputs [B, 2, T] estimated jammer
        jam = xb.permute(0, 2, 1)             # [B, 2, T]
        j_hat = model(jam).permute(0, 2, 1)   # [B, T, 2]
        y_hat = xb - j_hat                    # denoised

        snr_in_b, snr_out_b, evm_b = snr_in_out_evm_per_sample(xb, yb, y_hat)

        all_snr_in.append(snr_in_b.cpu().numpy())
        all_snr_out.append(snr_out_b.cpu().numpy())
        all_evm.append(evm_b.cpu().numpy())

    dt = time.time() - t0

    snr_in = np.concatenate(all_snr_in, axis=0)
    snr_out = np.concatenate(all_snr_out, axis=0)
    evm = np.concatenate(all_evm, axis=0)
    snr_gain = snr_out - snr_in

    summary = {
        "N": snr_in.shape[0],
        "snr_in_mean": float(snr_in.mean()),
        "snr_out_mean": float(snr_out.mean()),
        "snr_gain_mean": float(snr_gain.mean()),
        "evm_mean": float(evm.mean()),
        "snr_in_p5": float(np.quantile(snr_in, 0.05)),
        "snr_in_p50": float(np.quantile(snr_in, 0.50)),
        "snr_in_p95": float(np.quantile(snr_in, 0.95)),
        "snr_gain_p5": float(np.quantile(snr_gain, 0.05)),
        "snr_gain_p50": float(np.quantile(snr_gain, 0.50)),
        "snr_gain_p95": float(np.quantile(snr_gain, 0.95)),
        "evm_p5": float(np.quantile(evm, 0.05)),
        "evm_p50": float(np.quantile(evm, 0.50)),
        "evm_p95": float(np.quantile(evm, 0.95)),
        "runtime_sec": float(dt),
    }

    # hardest sample = lowest SNR_in
    hardest_idx = int(np.argmin(snr_in))
    summary["hardest_idx"] = hardest_idx
    summary["hardest_snr_in"] = float(snr_in[hardest_idx])
    summary["hardest_snr_out"] = float(snr_out[hardest_idx])
    summary["hardest_snr_gain"] = float(snr_gain[hardest_idx])
    summary["hardest_evm"] = float(evm[hardest_idx])

    return snr_in, snr_out, snr_gain, evm, summary


# ---------------- CLI ----------------

def build_argparser():
    ap = argparse.ArgumentParser(description="Evaluate TCN jammer-denoiser checkpoint.")
    ap.add_argument("--ckpt", required=True,
                    help="Path to checkpoint .pt file (from chat_tcn_train.py).")
    ap.add_argument("--data", required=True,
                    help="Path to NPZ dataset file.")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val",
                    help="Which split of the NPZ to evaluate on (default: val).")
    ap.add_argument("--batch", type=int, default=512,
                    help="Batch size for evaluation.")
    ap.add_argument("--workers", type=int, default=4,
                    help="Number of DataLoader workers.")
    ap.add_argument("--cpu", action="store_true",
                    help="Force CPU even if CUDA is available.")
    ap.add_argument("--out", type=str, default=None,
                    help="Optional path to .npz file where per-sample metrics + summary will be saved.")
    return ap


def main():
    args = build_argparser().parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    # Load data
    X, Y = load_npz_split(args.data, args.split)
    ds = IQWindows(X, Y)
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Loaded split '{args.split}' from {args.data}: {len(ds)} windows of shape {X.shape[1:]}")

    # Load model
    model, perseq_norm, ck_args = load_model_from_ckpt(args.ckpt, device)
    print(f"Loaded checkpoint from {args.ckpt}")
    print(f"Model config from ckpt: width={ck_args.get('width')}, blocks={ck_args.get('blocks')}, "
          f"kernel={ck_args.get('kernel')}, dropout={ck_args.get('dropout')}")
    print(f"Per-sequence RMS norm used during training: {perseq_norm}")

    # Run evaluation
    snr_in, snr_out, snr_gain, evm, summary = run_eval(model, loader, device, perseq_norm)

    # Pretty-print summary
    print(
        f"Eval | N={summary['N']} | "
        f"SNR_in {summary['snr_in_mean']:+.2f} dB → "
        f"SNR_out {summary['snr_out_mean']:+.2f} dB "
        f"(Δ {summary['snr_gain_mean']:+.2f} dB) | "
        f"EVM {summary['evm_mean']:.2f}%"
    )
    print(
        f"  percentiles: ΔSNR P5={summary['snr_gain_p5']:+.2f} dB, "
        f"P50={summary['snr_gain_p50']:+.2f} dB, "
        f"P95={summary['snr_gain_p95']:+.2f} dB"
    )
    print(
        f"  hardest sample (idx={summary['hardest_idx']}): "
        f"SNR_in {summary['hardest_snr_in']:+.2f} dB → "
        f"SNR_out {summary['hardest_snr_out']:+.2f} dB "
        f"(Δ {summary['hardest_snr_gain']:+.2f} dB) | "
        f"EVM {summary['hardest_evm']:.2f}%"
    )
    print(f"  runtime: {summary['runtime_sec']:.1f} s")

    # Optional: save metrics + summary
    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            snr_in_db=snr_in.astype(np.float32),
            snr_out_db=snr_out.astype(np.float32),
            snr_gain_db=snr_gain.astype(np.float32),
            evm_rms_pct=evm.astype(np.float32),
            summary=np.array(summary, dtype=object),
            ckpt_path=args.ckpt,
            data_path=args.data,
            split=args.split,
        )
        print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
