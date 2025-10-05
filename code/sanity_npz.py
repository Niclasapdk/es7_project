#!/usr/bin/env python3
"""
sanity_npz.py — quick dataset sanity checks for paired IQ NPZ files.

Supports:
  - Xtr/Ytr, Xva/Yva [, Xte/Yte]
  - X_train/Y_train, X_val/Y_val [, X_test/Y_test]
  - X/Y [+ split]
Flattens (N,L,2) or (N,2,L) to (N, 2*L). Leaves (N,2*L) as-is.
Prints:
  - keys & shapes
  - mean|Y-X| and MSE per split
  - std(X), std(Y), min/max std across features
  - NaN/Inf counts
  - rough corr between X and Y (per-sample averaged)
Flags potential issues.
"""
from pathlib import Path
import argparse, sys
import numpy as np

def _to_flat_2win(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3:
        # (N,L,2) or (N,2,L)
        if arr.shape[-1] == 2:
            I, Q = arr[...,0], arr[...,1]
        elif arr.shape[1] == 2:
            I, Q = arr[:,0,:], arr[:,1,:]
        else:
            raise ValueError(f"3D shape not recognized: {arr.shape}")
        flat = np.concatenate([I, Q], axis=1).astype(np.float32)
        L = I.shape[1]
        return flat, L
    elif arr.ndim == 2:
        F = arr.shape[1]
        if F % 2 != 0:
            raise ValueError(f"Feature dim {F} not even; cannot split I/Q.")
        return arr.astype(np.float32), F//2
    else:
        raise ValueError(f"Unsupported array shape {arr.shape}")

def _have(d, *keys): return all(k in d for k in keys)

def _pick_splits(D):
    keys = set(D.files)
    # A) explicit *_train/val/test
    if _have(D, 'X_train','Y_train','X_val','Y_val'):
        Xtr, L1 = _to_flat_2win(D['X_train'])
        Ytr, L2 = _to_flat_2win(D['Y_train'])
        Xva, L3 = _to_flat_2win(D['X_val'])
        Yva, L4 = _to_flat_2win(D['Y_val'])
        Xte = Yte = None
        if _have(D, 'X_test','Y_test'):
            Xte, L5 = _to_flat_2win(D['X_test'])
            Yte, L6 = _to_flat_2win(D['Y_test'])
            assert L5 == L6 == L1, "Inconsistent window length (test)"
        assert L1 == L2 == L3 == L4, "Inconsistent window lengths (train/val)"
        return (Xtr,Ytr),(Xva,Yva),(Xte,Yte), L1, "train/val/test"
    # B) compact Xtr/Xva/Ytr/Yva
    if _have(D, 'Xtr','Ytr','Xva','Yva'):
        Xtr, L1 = _to_flat_2win(D['Xtr'])
        Ytr, L2 = _to_flat_2win(D['Ytr'])
        Xva, L3 = _to_flat_2win(D['Xva'])
        Yva, L4 = _to_flat_2win(D['Yva'])
        Xte = Yte = None
        if _have(D, 'Xte','Yte'):
            Xte, L5 = _to_flat_2win(D['Xte'])
            Yte, L6 = _to_flat_2win(D['Yte'])
            assert L5 == L6 == L1, "Inconsistent window length (test)"
        assert L1 == L2 == L3 == L4, "Inconsistent window lengths (train/val)"
        return (Xtr,Ytr),(Xva,Yva),(Xte,Yte), L1, "Xtr/Xva"
    # C) X/Y (+ split or random check only)
    if _have(D, 'X','Y'):
        X, L1 = _to_flat_2win(D['X'])
        Y, L2 = _to_flat_2win(D['Y'])
        assert L1 == L2, "Inconsistent window length (X vs Y)"
        if 'split' in D:
            split = D['split']
            if split.dtype.kind in 'UO':
                m = {'train':0,'val':1,'validation':1,'test':2}
                S = np.array([m.get(str(s).lower(),0) for s in split], dtype=np.int32)
            else:
                S = np.array(split, dtype=np.int32)
            tr, va, te = (S==0), (S==1), (S==2)
            return (X[tr],Y[tr]), (X[va],Y[va]), (X[te],Y[te]), L1, "X/Y+split"
        else:
            # single pair; treat as 'train' only
            return (X,Y), (None,None), (None,None), L1, "X/Y only"
    raise SystemExit("Unrecognized NPZ structure. Need X/Y (±split) or Xtr/Xva/Ytr/Yva or *_train/val(/test).")

def _basic_stats(X, Y, name):
    if X is None or Y is None: 
        print(f"[{name}] (absent)"); 
        return
    assert X.shape == Y.shape, f"{name}: X and Y shape mismatch {X.shape} vs {Y.shape}"
    # NaN/Inf checks
    nan_inf_X = np.isnan(X).sum() + np.isinf(X).sum()
    nan_inf_Y = np.isnan(Y).sum() + np.isinf(Y).sum()
    # Mean |Y-X| and MSE
    diff = Y - X
    mean_abs = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff**2))
    # Featurewise std
    sdX = X.std(axis=0)
    sdY = Y.std(axis=0)
    sdX_min, sdX_max = float(sdX.min()), float(sdX.max())
    sdY_min, sdY_max = float(sdY.min()), float(sdY.max())
    # Rough per-sample correlation between X and Y (avoid div0)
    def _corr(a,b):
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        na = np.linalg.norm(a, axis=1) + 1e-12
        nb = np.linalg.norm(b, axis=1) + 1e-12
        c = np.sum(a*b, axis=1) / (na*nb)
        return float(np.mean(c))
    corr = _corr(X, Y)

    print(f"[{name}] N={len(X)}  F={X.shape[1]}  mean|Y-X|={mean_abs:.6g}  MSE={mse:.6g}  corr~={corr:.4f}")
    print(f"[{name}] std(X): min={sdX_min:.3g} max={sdX_max:.3g} | std(Y): min={sdY_min:.3g} max={sdY_max:.3g}")
    print(f"[{name}] NaN/Inf -> X:{nan_inf_X}  Y:{nan_inf_Y}")

    # Warnings
    if mean_abs < 1e-8 and mse < 1e-12:
        print(f"  ⚠️  {name}: X and Y appear identical (residual≈0).")
    if sdX_min <= 0 or sdY_min <= 0:
        print(f"  ⚠️  {name}: Some features have zero std (constant features).")
    if nan_inf_X or nan_inf_Y:
        print(f"  ⚠️  {name}: Contains NaNs/Infs.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to NPZ dataset")
    args = ap.parse_args()

    p = Path(args.data)
    if not p.exists():
        print(f"File not found: {p}")
        sys.exit(1)

    D = np.load(p, allow_pickle=True)
    print("=== NPZ KEYS ===")
    for k in sorted(D.files):
        try:
            s = D[k].shape
        except Exception:
            s = "<object>"
        print(f" - {k}: {s}")

    (Xtr,Ytr), (Xva,Yva), (Xte,Yte), win, mode = _pick_splits(D)
    print(f"\nDetected split mode: {mode} | window={win}\n")

    _basic_stats(Xtr, Ytr, "train")
    _basic_stats(Xva, Yva, "val")
    _basic_stats(Xte, Yte, "test")

    # Quick verdict
    problems = []
    if Xtr is not None and np.allclose(Xtr, Ytr):
        problems.append("TRAIN Y==X (identical pairs)")
    if Xtr is not None and (np.isclose(Xtr.std(axis=0), 0).any() or np.isclose(Ytr.std(axis=0), 0).any()):
        problems.append("Zero-std features in TRAIN")
    if problems:
        print("\nVerdict: ⚠️  Potential issues -> " + "; ".join(problems))
    else:
        print("\nVerdict: ✅  No obvious red flags found.")

if __name__ == "__main__":
    main()
