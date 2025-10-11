#!/usr/bin/env python3
import argparse, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt

# ======== Model variants that match different checkpoint schemas ========
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, kernel, dilation=1, bias=True):
        super().__init__(in_ch, out_ch, kernel, stride=1, padding=0, dilation=dilation, bias=bias)
        self.left_pad = dilation * (kernel - 1)
    def forward(self, x):
        return super().forward(F.pad(x, (self.left_pad, 0)))

# --- Variant A: WeightNorm TCN (keys like: tcn.0.c1.weight_g / tcn.0.c2.weight_v) ---
class ResidualBlockWN(nn.Module):
    def __init__(self, ch, kernel, dilation, dropout=0.0):
        super().__init__()
        self.c1 = nn.utils.weight_norm(CausalConv1d(ch, ch, kernel, dilation=dilation))
        self.c2 = nn.utils.weight_norm(CausalConv1d(ch, ch, kernel, dilation=dilation))
        self.dp = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        y = self.act(self.c1(x))
        y = self.dp(y)
        y = self.act(self.c2(y))
        return x + y

class TCN_WN(nn.Module):
    """Matches checkpoints with keys like: inp.*, tcn.<idx>.c1.*, tcn.<idx>.c2.*, out.*"""
    def __init__(self, in_ch=2, out_ch=2, width=192, blocks=10, kernel=7, dilation_growth=2, dropout=0.0):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, width, 1)
        layers, d = [], 1
        for _ in range(blocks):
            layers.append(ResidualBlockWN(width, kernel, dilation=d, dropout=dropout)); d *= dilation_growth
        self.tcn = nn.Sequential(*layers)
        self.out = nn.Conv1d(width, out_ch, 1)
    def forward(self, x): return self.out(self.tcn(self.inp(x)))

# --- Variant B: Batch/GroupNorm TCN (keys like: blocks.0.conv1.*, blocks.0.norm1.*) ---
class ResidualBlockBN(nn.Module):
    def __init__(self, ch, kernel, dilation, dropout=0.0, norm_kind="bn"):
        super().__init__()
        self.conv1 = CausalConv1d(ch, ch, kernel, dilation=dilation)
        self.conv2 = CausalConv1d(ch, ch, kernel, dilation=dilation)
        if norm_kind == "bn":
            self.norm1 = nn.BatchNorm1d(ch)
            self.norm2 = nn.BatchNorm1d(ch)
        elif norm_kind == "gn":
            self.norm1 = nn.GroupNorm(8, ch)  # 8 groups as a sane default
            self.norm2 = nn.GroupNorm(8, ch)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        self.dp = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        y = self.conv1(x); y = self.norm1(y); y = self.act(y)
        y = self.dp(y)
        y = self.conv2(y); y = self.norm2(y); y = self.act(y)
        return x + y

class TCN_BN(nn.Module):
    """Matches checkpoints with keys like: inp.*, blocks.<idx>.conv1.*, blocks.<idx>.norm1.*, out.*"""
    def __init__(self, in_ch=2, out_ch=2, width=192, blocks=10, kernel=7, dilation_growth=2, dropout=0.0, norm_kind="bn"):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, width, 1)
        self.blocks = nn.ModuleList()
        d = 1
        for _ in range(blocks):
            self.blocks.append(ResidualBlockBN(width, kernel, dilation=d, dropout=dropout, norm_kind=norm_kind))
            d *= dilation_growth
        self.out = nn.Conv1d(width, out_ch, 1)
    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks: h = b(h)
        return self.out(h)

# ----- Utils -----
def to_T2(a):
    a = np.asarray(a)
    if np.iscomplexobj(a): return np.stack([a.real, a.imag], -1).astype(np.float32)
    a = np.squeeze(a)
    if a.ndim == 1: return np.stack([a.astype(np.float32), np.zeros_like(a, np.float32)], -1)
    if a.ndim == 2:
        if a.shape[-1]==2: return a.astype(np.float32)
        if a.shape[0]==2:  return a.T.astype(np.float32)
        if a.shape[-1]==1: return np.concatenate([a.astype(np.float32), np.zeros_like(a, np.float32)], -1)
        if a.shape[0]==1:
            b=a.reshape(-1).astype(np.float32); return np.stack([b, np.zeros_like(b)], -1)
    return to_T2(np.squeeze(a))

def complex_from_T2(x): return x[...,0].astype(np.float32) + 1j*x[...,1].astype(np.float32)
def evm_pct(yhat, ref):
    num = np.sum(np.abs(yhat-ref)**2); den = np.sum(np.abs(ref)**2) + 1e-12
    return float(np.sqrt(num/den)*100.0)
def snr_db(sig, err):
    ps = np.mean(np.abs(sig)**2)+1e-12; pe = np.mean(np.abs(err)**2)+1e-12
    return 10.0*np.log10(ps/pe)

def norm_none(x):        return x.astype(np.float32), {"mode":"none"}
def norm_center(x):
    mu = np.mean(x,0,keepdims=True); return (x-mu).astype(np.float32), {"mode":"center","mu":mu}
def norm_unit_rms_global(x):
    mu = np.mean(x,0,keepdims=True); xc = x-mu; rms = np.sqrt(np.mean(xc**2)+1e-12)
    return (xc/rms).astype(np.float32), {"mode":"urg","mu":mu,"rms":rms}
def norm_unit_rms_perch(x):
    mu = np.mean(x,0,keepdims=True); xc = x-mu; rms = np.sqrt(np.mean(xc**2,axis=0,keepdims=True)+1e-12)
    return (xc/rms).astype(np.float32), {"mode":"urp","mu":mu,"rms":rms}

def denorm(y, stats):
    m=stats["mode"]
    if m=="none": return y.astype(np.float32)
    if m=="center": return (y + stats["mu"]).astype(np.float32)
    if m=="urg": return (y*stats["rms"] + stats["mu"]).astype(np.float32)
    if m=="urp": return (y*stats["rms"] + stats["mu"]).astype(np.float32)
    return y.astype(np.float32)

def detect_xy_keys(nzf, split):
    keys=list(nzf.keys()); split=split.lower()
    xs=[f"X{split[:2]}", f"X_{split}", f"X{split}", f"x_{split}", f"x{split}",
        "X","input","jammed","noisy","J","Xte","Xtest","Xva","Xval","Xtr","Xtrain"]
    ys=[f"Y{split[:2]}", f"Y_{split}", f"Y{split}", f"y_{split}", f"y{split}",
        "Y","target","clean","Yte","Ytest","Yva","Yval","Ytr","Ytrain"]
    xs=[k for k in xs if k in keys]; ys=[k for k in ys if k in keys]
    for xk in xs:
        for yk in ys:
            if nzf[xk].shape[0]==nzf[yk].shape[0]: return xk, yk
    raise KeyError(f"Could not find X/Y keys in {keys}")

def build_and_load_exact(ckpt_path, device):
    import torch
    ckpt = torch.load(ckpt_path, map_location=device)

    # Pull arch args (width/blocks/kernel/dropout); fall back to sane defaults
    a = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    width  = int(a.get("width", 192))
    blocks = int(a.get("blocks", 10))
    kernel = int(a.get("kernel", 7))
    drop   = float(a.get("dropout", 0.0))

    # Extract a candidate state dict (EMA > state_dict > model_state > model > net > top-level)
    def looks_like_state(d):
        return isinstance(d, dict) and any(("weight" in k or "bias" in k) for k in d.keys())
    state, source = None, None
    if isinstance(ckpt, dict):
        use_ema = bool(a.get("eval_use_ema", False))
        if use_ema and "ema_state" in ckpt and looks_like_state(ckpt["ema_state"]):
            state, source = ckpt["ema_state"], "ema_state"
        else:
            for k in ("state_dict","model_state","model","net"):
                if k in ckpt and looks_like_state(ckpt[k]):
                    state, source = ckpt[k], k; break
            if state is None and looks_like_state(ckpt):
                state, source = ckpt, "top_level"
    else:
        state, source = ckpt, "raw"

    if state is None:
        raise RuntimeError("No weights found in checkpoint.")

    # Strip 'module.' prefixes if saved with DDP
    state = { (k[7:] if k.startswith("module.") else k): v for k,v in state.items() }
    keys = list(state.keys())

    # ---- ARCH DETECTION by key pattern ----
    if any(k.startswith("blocks.0.conv1.") for k in keys):
        arch_kind = "BN"
        model = TCN_BN(in_ch=2, out_ch=2, width=width, blocks=blocks, kernel=kernel, dropout=drop, norm_kind="bn").to(device)
    elif any(k.startswith("tcn.0.c1.") for k in keys) or any(".weight_g" in k for k in keys):
        arch_kind = "WN"
        model = TCN_WN(in_ch=2, out_ch=2, width=width, blocks=blocks, kernel=kernel, dropout=drop).to(device)
    else:
        raise RuntimeError("Unknown checkpoint layout (neither 'blocks.*.conv1.*' nor 'tcn.*.c1.*').")

    # Load strictly
    ret = model.load_state_dict(state, strict=True)
    missing = getattr(ret, "missing_keys", [])
    unexpected = getattr(ret, "unexpected_keys", [])
    print(f"[arch] {arch_kind} | [load] source={source} | missing={missing} | unexpected={unexpected}")
    print("[params]", sum(p.numel() for p in model.parameters()))
    model.eval()
    return model, a


# ----- Main -----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--x-key"); ap.add_argument("--y-key")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--eval-n", type=int, default=0)
    ap.add_argument("--save", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device=torch.device(args.device)
    model, margs = build_and_load_exact(args.ckpt, device)

    nzf = np.load(args.data, allow_pickle=True)
    try:
        xk, yk = (args.x_key, args.y_key) if (args.x_key and args.y_key) else detect_xy_keys(nzf, args.split)
        X, Y = nzf[xk], nzf[yk]
    finally:
        nzf.close()
    print(f"[info] keys: X='{xk}', Y='{yk}' | shapes: X{X.shape}, Y{Y.shape}")

    # All combos to try
    norms = [("none", norm_none), ("center", norm_center),
             ("unit_rms_global", norm_unit_rms_global), ("unit_rms_perch", norm_unit_rms_perch)]
    outmodes = ["clean","x-minus","x-plus"]

    def run_once(i):
        x=to_T2(X[i]); y=to_T2(Y[i]); T=min(len(x),len(y)); x=x[:T]; y=y[:T]
        best=None; best_evm=1e9; best_info=None; best_yhat=None
        for nname, nfn in norms:
            x_n, st = nfn(x)
            with torch.no_grad():
                xin = torch.from_numpy(x_n.T).float().unsqueeze(0).to(device)
                pred_n = model(xin).squeeze(0).cpu().numpy().T
            cand = {
                "clean":   denorm(pred_n, st),
                "x-minus": denorm(x_n - pred_n, st),
                "x-plus":  denorm(x_n + pred_n, st),
            }
            ref_c = complex_from_T2(y)
            for mode in outmodes:
                yh = cand[mode]
                evm = evm_pct(complex_from_T2(yh), ref_c)
                if evm < best_evm:
                    best_evm = evm; best = (nname, mode); best_yhat = yh
                    best_info = (x, y, yh)
        # metrics
        x, y, yhat = best_info
        jam_c = complex_from_T2(x); den_c = complex_from_T2(yhat); ref_c = complex_from_T2(y)
        evm_in = evm_pct(jam_c, ref_c); evm_out = evm_pct(den_c, ref_c)
        snr_in = snr_db(ref_c, jam_c-ref_c); snr_out = snr_db(ref_c, den_c-ref_c)
        return dict(idx=i, norm=best[0], mode=best[1], evm_in=evm_in, evm_out=evm_out,
                    snr_in=snr_in, snr_out=snr_out, dsnr=snr_out-snr_in, yhat=best_yhat, x=x, y=y)

    # optional quick eval across N samples
    if args.eval_n>0:
        import random
        idxs = random.sample(range(X.shape[0]), min(args.eval_n, X.shape[0]))
        res=[run_once(i) for i in idxs]
        print(f"[EVAL {len(res)}] mean EVM_in {np.mean([r['evm_in'] for r in res]):.2f}% | "
              f"mean EVM_out {np.mean([r['evm_out'] for r in res]):.2f}% | mean ΔSNR {np.mean([r['dsnr'] for r in res]):+.2f} dB")
        # also print the most common chosen combo
        from collections import Counter
        print("[combo] most common:", Counter((r['norm'], r['mode']) for r in res).most_common(3))

    # single index plot
    r = run_once(args.idx)
    print(f"[single idx {r['idx']}] norm={r['norm']} mode={r['mode']} | "
          f"EVM_in {r['evm_in']:.2f}% -> EVM_out {r['evm_out']:.2f}% | "
          f"SNR_in {r['snr_in']:.2f} dB -> SNR_out {r['snr_out']:.2f} dB | ΔSNR {r['dsnr']:+.2f} dB")

    t = np.arange(len(r['x']))
    plt.figure(figsize=(11,5.5))
    plt.plot(t, r['x'][:,0], label=f"Jammed (EVM {r['evm_in']:.2f}%, SNR {r['snr_in']:.2f} dB)")
    plt.plot(t, r['yhat'][:,0], label=f"Denoised [{r['mode']}, {r['norm']}] (EVM {r['evm_out']:.2f}%, SNR {r['snr_out']:.2f} dB)")
    plt.plot(t, r['y'][:,0], label="Clean (reference)", linewidth=1.4)
    plt.title(f"TCN Denoising | idx {r['idx']} | ΔSNR {r['dsnr']:+.2f} dB")
    plt.xlabel("Sample"); plt.ylabel("I component"); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    if args.save: plt.savefig(args.save, dpi=150); print("[saved]", args.save)
    plt.show()

if __name__ == "__main__":
    main()
