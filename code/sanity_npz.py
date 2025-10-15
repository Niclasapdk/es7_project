# save as sanity_npz.py and run: python3 sanity_npz.py --data gnss_synth_sweepcw_500k.npz
import argparse, numpy as np
p=argparse.ArgumentParser(); p.add_argument("--data", required=True); args=p.parse_args()
Z=np.load(args.data)
print("Keys:", list(Z.keys()))

def get_iq(Z, base):
    # prefer explicit I/Q siblings if present
    I = None; Q = None
    for k in [f"{base}_I", f"{base}_i", f"{base}I", f"{base}_real", f"{base}_re", f"{base}_r"]:
        if k in Z: I = Z[k]; break
    for k in [f"{base}_Q", f"{base}_q", f"{base}Q", f"{base}_imag", f"{base}_im", f"{base}_j"]:
        if k in Z: Q = Z[k]; break
    if I is not None and Q is not None:
        return I.astype(np.float32), Q.astype(np.float32)
    A = Z.get(base)
    if A is None:
        raise SystemExit(f"Missing {base} and no siblings")
    A = A.astype(np.float32)
    if np.iscomplexobj(A):   # complex [N,T]
        return A.real, A.imag
    if A.ndim==3 and A.shape[1]==2:  # [N,2,T]
        return A[:,0,:], A[:,1,:]
    if A.ndim==3 and A.shape[2]==2:  # [N,T,2]
        return A[:,:,0], A[:,:,1]
    if A.ndim==2:                    # real [N,T] (Q=0 fallback)
        return A, np.zeros_like(A)
    raise SystemExit(f"Unsupported shape for {base}: {A.shape}")

Xi,Xq = get_iq(Z,"Xtr"); Yi,Yq = get_iq(Z,"Ytr")
print("Xtr shape:", Xi.shape, "Ytr shape:", Yi.shape)
N,T = Xi.shape
# sample small subset
idx = np.random.default_rng(0).integers(0,N, size=4096)
x = Xi[idx]+1j*Xq[idx]; y = Yi[idx]+1j*Yq[idx]

def evm_unaligned(yhat, y):
    num = np.sum(np.abs(yhat-y)**2); den = np.sum(np.abs(y)**2)+1e-12
    return 100.0*np.sqrt(num/den)
def evm_aligned(yhat, y):
    # per-sample complex scalar
    a = np.sum(np.conj(yhat)*y, axis=1) / (np.sum(np.abs(yhat)**2, axis=1)+1e-12)
    yha = yhat * a[:,None]
    num = np.sum(np.abs(yha-y)**2); den = np.sum(np.abs(y)**2)+1e-12
    return 100.0*np.sqrt(num/den)

print("Baseline metrics with yhat = x (identity):")
print("  EVM% (unaligned):", f"{evm_unaligned(x,y):.2f}")
print("  EVM% (aligned)  :", f"{evm_aligned(x,y):.2f}")
print("  SNR_in (dB)     :", 10*np.log10(np.sum(np.abs(y)**2)/(np.sum(np.abs(x-y)**2)+1e-20)))
print("  ||X-Y||/||Y||    :", np.linalg.norm((x-y).ravel())/(np.linalg.norm(y.ravel())+1e-12))
same = np.allclose(Xi, Yi) and np.allclose(Xq, Yq)
print("  Are Xtr and Ytr identical? ", same)
