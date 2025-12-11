#!/usr/bin/env python3
"""
TCN jammer-denoiser training script (no notch, jammer-estimation formulation).

Features:
- DDP (torchrun) support
- AMP 2.0 (torch.amp.autocast / GradScaler('cuda', ...))
- Composite loss: time-domain + spectral + optional smoothness
- Optional per-window RMS normalization
- Cosine / CAWR schedulers with warmup
- EMA weights
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


# ---------------- I/Q helpers ----------------

def ensure_iq(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is [B, T, 2] float IQ."""
    if x.ndim != 3 or x.shape[-1] != 2:
        raise RuntimeError(f"Expected [B, T, 2] IQ tensor, got shape {tuple(x.shape)}")
    return x


def complex_from_iq(x: torch.Tensor) -> torch.Tensor:
    """Convert [B, T, 2] → complex [B, T]."""
    x = ensure_iq(x)
    return torch.complex(x[..., 0], x[..., 1])


def iq_from_complex(z: torch.Tensor) -> torch.Tensor:
    """Convert complex [B, T] → [B, T, 2]."""
    return torch.stack([z.real, z.imag], dim=-1)


# ---------------- optional per-sequence normalization ----------------

def perseq_rms_norm(x_iq: torch.Tensor, y_iq: torch.Tensor, eps: float = 1e-8):
    """
    Normalize both input and target by target RMS per sequence.
    x_iq, y_iq: [B, T, 2]
    """
    y_iq = ensure_iq(y_iq)
    yr = y_iq[..., 0]
    yi = y_iq[..., 1]
    rms = torch.sqrt((yr ** 2 + yi ** 2).mean(dim=1, keepdim=True).clamp_min(eps))  # [B, 1]
    rms = rms.unsqueeze(-1)  # [B, 1, 1]
    return x_iq / rms, y_iq / rms


# ---------------- Dataset ----------------

class IQWindows(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        assert X.shape == Y.shape and X.ndim == 3 and X.shape[-1] == 2, \
            f"Bad shapes {X.shape} vs {Y.shape}"
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i: int):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])


# ---------------- NPZ loader ----------------

def load_npz(path: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    npz = np.load(path)
    k = set(npz.keys())

    if {"Xtr", "Ytr", "Xva", "Yva"}.issubset(k):
        Xtr = npz["Xtr"]
        Ytr = npz["Ytr"]
        Xva = npz["Xva"]
        Yva = npz["Yva"]
    elif {"Xtr", "Ytr"}.issubset(k):
        Xtr = npz["Xtr"]
        Ytr = npz["Ytr"]
        if {"Xva", "Yva"}.issubset(k):
            Xva = npz["Xva"]
            Yva = npz["Yva"]
        else:
            # simple 90/10 split
            N = Xtr.shape[0]
            n_va = max(1, int(0.1 * N))
            Xva, Yva = Xtr[-n_va:], Ytr[-n_va:]
            Xtr, Ytr = Xtr[:-n_va], Ytr[:-n_va]
    else:
        # fallback: assume "X","Y"
        X = npz["X"]
        Y = npz["Y"]
        N = X.shape[0]
        n_va = max(1, int(0.1 * N))
        Xtr, Ytr = X[:-n_va], Y[:-n_va]
        Xva, Yva = X[-n_va:], Y[-n_va:]

    return {"train": (Xtr, Ytr), "val": (Xva, Yva)}


# ---------------- spectral & smoothness losses ----------------

def spectral_loss(y_true_iq: torch.Tensor,
                  y_pred_iq: torch.Tensor,
                  fs: float,
                  inband: float,
                  guard: float,
                  w_in: float = 1.0,
                  w_guard: float = 1.0,
                  w_out: float = 1.0) -> torch.Tensor:
    """
    L1 loss on magnitude spectra, with weights on in-band / guard / out-of-band.
    """
    y_true_iq = ensure_iq(y_true_iq)
    y_pred_iq = ensure_iq(y_pred_iq)
    yt = complex_from_iq(y_true_iq)
    yp = complex_from_iq(y_pred_iq)
    B, T = yt.shape

    Yt = torch.fft.fft(yt, dim=1)
    Yp = torch.fft.fft(yp, dim=1)
    freqs = torch.fft.fftfreq(T, d=1.0 / fs).to(yt.device)

    mag_t = Yt.abs()
    mag_p = Yp.abs()
    diff = (mag_p - mag_t).abs()         # [B, T]

    f_in = inband / 2.0
    mask_in = (freqs >= -f_in) & (freqs <= f_in)
    if guard > 0.0:
        mask_guard = (freqs >= -guard) & (freqs <= guard)
        mask_inband = mask_in & ~mask_guard
    else:
        mask_guard = torch.zeros_like(mask_in, dtype=torch.bool)
        mask_inband = mask_in

    mask_out = ~mask_in

    loss = 0.0
    if w_in > 0.0 and mask_inband.any():
        loss = loss + w_in * diff[:, mask_inband].mean()
    if w_guard > 0.0 and mask_guard.any():
        loss = loss + w_guard * diff[:, mask_guard].mean()
    if w_out > 0.0 and mask_out.any():
        loss = loss + w_out * diff[:, mask_out].mean()

    return loss


def first_diff_loss(y_true_iq: torch.Tensor,
                    y_pred_iq: torch.Tensor) -> torch.Tensor:
    """
    First-difference (discrete derivative) L1 loss between true and predicted IQ.
    """
    y_true_iq = ensure_iq(y_true_iq)
    y_pred_iq = ensure_iq(y_pred_iq)
    dt_true = y_true_iq[:, 1:, :] - y_true_iq[:, :-1, :]
    dt_pred = y_pred_iq[:, 1:, :] - y_pred_iq[:, :-1, :]
    return (dt_pred - dt_true).abs().mean()


# ---------------- Model: Residual TCN ----------------

class CausalConv1d(nn.Conv1d):
    """1-D causal conv implemented via left padding and trimming."""
    def __init__(self, in_ch: int, out_ch: int, k: int, dilation: int = 1):
        pad = (k - 1) * dilation
        super().__init__(in_ch, out_ch, kernel_size=k, padding=pad, dilation=dilation)
        self._pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        if self._pad > 0:
            return out[..., :-self._pad]
        return out


class TCNBlock(nn.Module):
    def __init__(self, ch: int, k: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(ch, ch, k, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(ch)
        self.act1 = nn.GELU()
        self.conv2 = CausalConv1d(ch, ch, k, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(ch)
        self.act2 = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act1(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act2(y)
        return x + y


class ResidualTCN(nn.Module):
    """
    Simple residual TCN:
      input:  [B, 2, T]  (jammed IQ)
      output: [B, 2, T]  (estimated interference IQ)
    """
    def __init__(self, in_ch: int, hid: int, blocks: int, k: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Conv1d(in_ch, hid, kernel_size=1)
        layers = []
        dilation = 1
        for _ in range(blocks):
            layers.append(TCNBlock(hid, k, dilation=dilation, dropout=dropout))
            dilation *= 2
        self.tcn = nn.Sequential(*layers)
        self.out_proj = nn.Conv1d(hid, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.tcn(h)
        out = self.out_proj(h)
        return out


# ---------------- SNR / EVM metrics ----------------

@torch.no_grad()
def snr_in_out_evm_per_sample(x_iq: torch.Tensor,
                              y_true_iq: torch.Tensor,
                              y_pred_iq: torch.Tensor,
                              eps: float = 1e-12):
    """
    Per-sequence SNR_in, SNR_out and EVM metrics.

    Returns three tensors of shape [B]:
      - snr_in_db
      - snr_out_db
      - evm_rms_pct
    """
    x = complex_from_iq(x_iq)
    y = complex_from_iq(y_true_iq)
    yhat = complex_from_iq(y_pred_iq)

    sig_pow = (y.abs() ** 2).mean(dim=1).clamp_min(eps)          # [B]
    in_err_pow = (x - y).abs().pow(2).mean(dim=1).clamp_min(eps) # [B]
    out_err_pow = (yhat - y).abs().pow(2).mean(dim=1).clamp_min(eps)

    snr_in = 10.0 * torch.log10(sig_pow / in_err_pow)    # [B]
    snr_out = 10.0 * torch.log10(sig_pow / out_err_pow)  # [B]

    err_pow = (yhat - y).abs().pow(2).mean(dim=1).clamp_min(eps)
    ref_pow = sig_pow
    evm = torch.sqrt(err_pow / ref_pow) * 100.0          # [B], percent

    return snr_in, snr_out, evm


@torch.no_grad()
def snr_in_out_raw(x_iq: torch.Tensor,
                   y_true_iq: torch.Tensor,
                   y_pred_iq: torch.Tensor,
                   eps: float = 1e-12):
    """
    Backwards-compatible mean SNR_in and SNR_out in dB.
    """
    snr_in_vec, snr_out_vec, _ = snr_in_out_evm_per_sample(x_iq, y_true_iq, y_pred_iq, eps=eps)
    snr_in = snr_in_vec.mean().item()
    snr_out = snr_out_vec.mean().item()
    return snr_in, snr_out


@torch.no_grad()
def evm_rms_pct_raw(y_true_iq: torch.Tensor,
                    y_pred_iq: torch.Tensor,
                    eps: float = 1e-12) -> float:
    """
    Root-mean-square EVM (%) ignoring bandlimits / alignment.
    Backwards-compatible wrapper around per-sample metric.
    """
    _, _, evm_vec = snr_in_out_evm_per_sample(y_pred_iq, y_true_iq, y_pred_iq, eps=eps)
    # NOTE: we don't actually use this in the training loop anymore,
    # but keep it for compatibility.
    return evm_vec.mean().item()


# ---------------- Composite denoising loss ----------------

class DenoiseLoss(nn.Module):
    """
    Composite loss for denoising:
      - time-domain L1 between y_hat and y
      - optional smoothness term (first differences)
      - optional spectral magnitude loss with band weights
    """
    def __init__(self,
                 fs: float,
                 inband: float,
                 guard: float,
                 time_w: float = 1.0,
                 spec_w: float = 0.3,
                 smooth_w: float = 0.0,
                 w_in: float = 1.0,
                 w_guard: float = 1.0,
                 w_out: float = 1.0):
        super().__init__()
        self.fs = float(fs)
        self.inband = float(inband)
        self.guard = float(guard)
        self.time_w = float(time_w)
        self.spec_w = float(spec_w)
        self.smooth_w = float(smooth_w)
        self.w_in = float(w_in)
        self.w_guard = float(w_guard)
        self.w_out = float(w_out)

    def forward(self, y_hat_iq: torch.Tensor, y_true_iq: torch.Tensor) -> torch.Tensor:
        y_hat_iq = ensure_iq(y_hat_iq)
        y_true_iq = ensure_iq(y_true_iq)

        loss = 0.0

        if self.time_w > 0.0:
            loss_time = (y_hat_iq - y_true_iq).abs().mean()
            loss = loss + self.time_w * loss_time

        if self.smooth_w > 0.0:
            loss_smooth = first_diff_loss(y_true_iq, y_hat_iq)
            loss = loss + self.smooth_w * loss_smooth

        if self.spec_w > 0.0:
            loss_spec = spectral_loss(
                y_true_iq, y_hat_iq,
                fs=self.fs, inband=self.inband, guard=self.guard,
                w_in=self.w_in, w_guard=self.w_guard, w_out=self.w_out,
            )
            loss = loss + self.spec_w * loss_spec

        return loss


# ---------------- EMA ----------------

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        for n, p in (model.module if hasattr(model, "module") else model).named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.detach().clone()
        self.backup = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        m = model.module if hasattr(model, "module") else model
        for n, p in m.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply(self, model: nn.Module):
        self.backup = {}
        m = model.module if hasattr(model, "module") else model
        for n, p in m.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.data.detach().clone()
                p.data.copy_(self.shadow[n].to(p.device))

    @torch.no_grad()
    def restore(self, model: nn.Module):
        if not self.backup:
            return
        m = model.module if hasattr(model, "module") else model
        for n, p in m.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}

    def state_dict_for(self, model: nn.Module):
        return {k: v.cpu() for k, v in self.shadow.items()}


# ---------------- DDP utils ----------------

def ddp_init():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        world = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local = int(os.environ.get("LOCAL_RANK", "0"))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local)
        return world, rank, local
    return 1, 0, 0


def is_master(rank: int) -> bool:
    return rank == 0


# ---------------- Scheduler helpers ----------------

def linear_warmup(step: int, warm_steps: int) -> float:
    return min(1.0, (step + 1) / max(1, warm_steps))


def configure_scheduler(opt, args, steps_per_epoch: int, total_steps: int):
    if args.scheduler == "cosine":
        warm_steps = int(args.warmup_frac * total_steps)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, total_steps - warm_steps)
        )
        return {"type": "cosine", "obj": sched, "warm_steps": warm_steps}
    elif args.scheduler == "cawr":
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=max(1, args.cawr_T0), T_mult=max(1, args.cawr_Tmult)
        )
        warm_steps = int(args.warmup_frac * steps_per_epoch * args.cawr_T0) if args.warmup_frac > 0 else 0
        return {"type": "cawr", "obj": sched, "warm_steps": warm_steps}
    else:
        return {"type": "none", "obj": None, "warm_steps": 0}


# ---------------- Data loaders & model factory ----------------

def make_model(in_ch: int, hid: int, blocks: int, k: int, dropout: float) -> nn.Module:
    return ResidualTCN(in_ch=in_ch, hid=hid, blocks=blocks, k=k, dropout=dropout)


def make_loader(X: np.ndarray, Y: np.ndarray, batch: int, shuffle: bool,
                rank: int, world: int, num_workers: int = 4):
    ds = IQWindows(X, Y)
    if world > 1:
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank,
                                     shuffle=shuffle, drop_last=False)
        return DataLoader(ds, batch_size=batch, sampler=sampler,
                          num_workers=num_workers, pin_memory=True)
    else:
        return DataLoader(ds, batch_size=batch, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)


# ---------------- Train ----------------

def train(args):
    world, rank, local = ddp_init()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    master = is_master(rank)
    torch.backends.cudnn.benchmark = True

    data = load_npz(args.data)
    Xtr, Ytr = data["train"]
    Xva, Yva = data["val"]

    train_loader = make_loader(Xtr, Ytr, args.batch, True, rank, world, args.workers)
    val_loader = make_loader(Xva, Yva, max(1, args.batch // 2), False, rank, world, args.workers)

    # model predicts interference; input = jammed IQ (2 channels)
    model = make_model(in_ch=2, hid=args.width, blocks=args.blocks,
                       k=args.kernel, dropout=args.dropout).to(device)
    if world > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local], find_unused_parameters=False
        )

    loss_fn = DenoiseLoss(
        fs=args.fs,
        inband=args.inband,
        guard=args.guard,
        time_w=args.time_w,
        spec_w=args.spec_w,
        smooth_w=args.smooth_w,
        w_in=args.spec_w_in,
        w_guard=args.spec_w_guard,
        w_out=args.spec_w_out,
    )

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999)
    )

    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.epochs * steps_per_epoch
    sched_info = configure_scheduler(opt, args, steps_per_epoch, total_steps)

    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    ema = EMA(model, decay=args.ema) if args.ema > 0 else None

    best_val = float("inf")
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(1, args.epochs + 1):

        model.train()
        t0 = time.time()
        train_loss_sum = 0.0
        n_train_batches = 0

        if world > 1 and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        for i, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)  # [B, T, 2] jammed
            yb = yb.to(device, non_blocking=True)  # [B, T, 2] clean

            if args.perseq_norm:
                xb, yb = perseq_rms_norm(xb, yb)

            jam = xb.permute(0, 2, 1)   # [B, 2, T]

            with torch.amp.autocast('cuda', enabled=args.amp):
                j_hat = model(jam)               # [B, 2, T]
                j_hat = j_hat.permute(0, 2, 1)   # [B, T, 2]
                y_hat = xb - j_hat               # denoised signal
                loss = loss_fn(y_hat, yb)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            # LR scheduler step
            if sched_info["type"] != ["none"]:
                if sched_info["warm_steps"] > 0 and step < sched_info["warm_steps"]:
                    scale = linear_warmup(step, sched_info["warm_steps"])
                    for pg in opt.param_groups:
                        pg["lr"] = args.lr * scale
                else:
                    if sched_info["type"] == "cosine":
                        sched_info["obj"].step()
                    elif sched_info["type"] == "cawr":
                        sched_info["obj"].step(epoch - 1 + i / max(1, steps_per_epoch))

            step += 1
            train_loss_sum += loss.item()
            n_train_batches += 1

            if ema is not None:
                ema.update(model)

        if master:
            dt = time.time() - t0
            train_loss = train_loss_sum / max(1, n_train_batches)
            print(f"Epoch {epoch:03d} | train loss {train_loss:.6f} | {dt:.1f}s")

        # ---------------- Validation ----------------
        if ema is not None:
            ema.apply(model)
        model.eval()

        val_loss = 0.0
        snr_in_sum = 0.0
        snr_out_sum = 0.0
        snr_gain_sum = 0.0
        evm_sum = 0.0
        n_batches = 0

        # hardest sample on this rank (lowest input SNR)
        hardest_in = float("inf")
        hardest_out = 0.0
        hardest_evm = 0.0

        t0 = time.time()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                if args.perseq_norm:
                    xb, yb = perseq_rms_norm(xb, yb)

                jam = xb.permute(0, 2, 1)
                j_hat = model(jam).permute(0, 2, 1)
                y_hat = xb - j_hat

                # loss
                val_loss += loss_fn(y_hat, yb).item()

                # per-sample metrics
                snr_in_b, snr_out_b, evm_b = snr_in_out_evm_per_sample(xb, yb, y_hat)
                snr_gain_b = snr_out_b - snr_in_b

                snr_in_sum += snr_in_b.mean().item()
                snr_out_sum += snr_out_b.mean().item()
                snr_gain_sum += snr_gain_b.mean().item()
                evm_sum += evm_b.mean().item()
                n_batches += 1

                # track hardest sample on this rank: minimal SNR_in
                batch_hardest_in, idx_min = snr_in_b.min(dim=0)
                if batch_hardest_in.item() < hardest_in:
                    hardest_in = batch_hardest_in.item()
                    hardest_out = snr_out_b[idx_min].item()
                    hardest_evm = evm_b[idx_min].item()

        if ema is not None:
            ema.restore(model)

        # reduce across ranks (main metrics)
        if world > 1:
            t = torch.tensor(
                [val_loss, snr_in_sum, snr_out_sum, snr_gain_sum, evm_sum, n_batches],
                device=device,
            )
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            val_loss, snr_in_sum, snr_out_sum, snr_gain_sum, evm_sum, n_batches = t.tolist()

        val_loss /= max(1, n_batches)
        snr_in = snr_in_sum / max(1, n_batches)
        snr_out = snr_out_sum / max(1, n_batches)
        snr_gain = snr_gain_sum / max(1, n_batches)
        evm = evm_sum / max(1, n_batches)

        # gather hardest sample (lowest SNR_in) across ranks
        if world > 1:
            triple = torch.tensor([hardest_in, hardest_out, hardest_evm], device=device)
            gathered = [torch.empty_like(triple) for _ in range(world)]
            dist.all_gather(gathered, triple)
            if master:
                triples = torch.stack(gathered)  # [world, 3]
                best_idx = torch.argmin(triples[:, 0])  # 0 = SNR_in
                hardest_in_global = triples[best_idx, 0].item()
                hardest_out_global = triples[best_idx, 1].item()
                hardest_evm_global = triples[best_idx, 2].item()
        else:
            hardest_in_global = hardest_in
            hardest_out_global = hardest_out
            hardest_evm_global = hardest_evm

        if master:
            dt = time.time() - t0
            msg = (
                f"Epoch {epoch:03d} | val {val_loss:.6f} | "
                f"SNR_in {snr_in:+.2f} dB → SNR_out {snr_out:+.2f} dB "
                f"(Δ {snr_gain:+.2f} dB) | "
                f"EVM {evm:.2f}% | "
                f"hardest sample: SNR_in {hardest_in_global:+.2f} dB → "
                f"SNR_out {hardest_out_global:+.2f} dB | "
                f"EVM {hardest_evm_global:.2f}% | "
                f"{dt:.1f}s"
            )
            print(msg)

        # save best
        if master and (val_loss < best_val):
            best_val = val_loss
            ck = {
                "model": (
                    ema.state_dict_for(model)
                    if (ema is not None)
                    else (
                        model.module.state_dict()
                        if hasattr(model, "module")
                        else model.state_dict()
                    )
                ),
                "args": vars(args),
                "val": val_loss,
                "epoch": epoch,
            }
            Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
            out_path = Path(args.ckpt_dir) / "best.pt"
            torch.save(ck, out_path)
            if master:
                print("  ↳ saved best ->", out_path)

    if world > 1:
        dist.destroy_process_group()


# ---------------- CLI ----------------

def build_argparser():
    ap = argparse.ArgumentParser("TCN jammer-denoiser trainer (no notch)")

    # data / io
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--ckpt-dir", type=str, default="ckpts_tcn_jammer")

    # signal params (for spectral loss weighting)
    ap.add_argument("--fs", type=float, default=4.092e6)
    ap.add_argument("--inband", type=float, default=2.046e6)
    ap.add_argument("--guard", type=float, default=150e3)

    # model
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--kernel", type=int, default=7)
    ap.add_argument("--dropout", type=float, default=0.05)

    # loss weights
    ap.add_argument("--time-w", type=float, default=1.0)
    ap.add_argument("--spec-w", type=float, default=0.3)
    ap.add_argument("--spec-w-in", type=float, default=1.0)
    ap.add_argument("--spec-w-guard", type=float, default=1.0)
    ap.add_argument("--spec-w-out", type=float, default=1.0)
    ap.add_argument("--smooth-w", type=float, default=0.05)

    # normalization
    ap.add_argument("--perseq-norm", action="store_true",
                    help="Per-window RMS normalization using clean RMS.")

    # opt + scheduler
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scheduler", type=str, default="cawr",
                    choices=["cosine", "cawr", "none"])
    ap.add_argument("--cawr-T0", type=int, default=5,
                    help="Epochs between first restart for CAWR.")
    ap.add_argument("--cawr-Tmult", type=int, default=2,
                    help="Restart period multiplier for CAWR.")
    ap.add_argument("--warmup-frac", type=float, default=0.06,
                    help="Fraction of steps for LR warmup.")

    return ap


def main():
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
