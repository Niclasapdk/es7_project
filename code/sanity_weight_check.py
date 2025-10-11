# ckpt_inspect.py
import torch, os, time, hashlib, numpy as np, sys
path = sys.argv[1] if len(sys.argv)>1 else "tcn_denoiser.pt"

ckpt = torch.load(path, map_location="cpu")
print("[info] file:", os.path.abspath(path))
print("[info] mtime:", time.ctime(os.path.getmtime(path)))
print("[info] type:", type(ckpt))

meta = {}
state = None
if isinstance(ckpt, dict):
    meta = {k: ckpt.get(k) for k in ["epoch","step","best_epoch","best_val","val_loss","val_evm","val_snr","args","model_cfg","cfg","hparams"]}
    for k in ("state_dict","model_state","model","ema_state","net"):
        if k in ckpt: state = ckpt[k]; break
    if state is None: state = ckpt  # maybe raw state_dict
elif hasattr(ckpt, "state_dict"):
    state = ckpt.state_dict()
else:
    state = ckpt

print("[meta]", {k:v for k,v in meta.items() if v is not None})

# hash + quick stats
h = hashlib.sha256()
nparam = 0
for k in sorted(state.keys()):
    arr = state[k].cpu().numpy()
    h.update(arr.tobytes())
    nparam += arr.size
print("[params]", nparam, "| sha256:", h.hexdigest()[:16])

# peek first conv weights if present
for k in ("inp.weight","module.inp.weight"):
    if k in state:
        w = state[k].cpu().numpy()
        print(f"[{k}] shape={w.shape} mean={w.mean():.6f} std={w.std():.6f}")
        break
