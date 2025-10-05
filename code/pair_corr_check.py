#!/usr/bin/env python3
import numpy as np
from pathlib import Path

def to_flat(a):
    a = np.asarray(a)
    if a.ndim==3 and a.shape[-1]==2:
        I,Q = a[...,0], a[...,1]
        return np.concatenate([I,Q],axis=1).astype(np.float32)
    if a.ndim==3 and a.shape[1]==2:
        I,Q = a[:,0,:], a[:,1,:]
        return np.concatenate([I,Q],axis=1).astype(np.float32)
    return a.astype(np.float32)

def corr_per_sample(X, Y):
    Xc = X - X.mean(1, keepdims=True)
    Yc = Y - Y.mean(1, keepdims=True)
    nX = np.linalg.norm(Xc, axis=1) + 1e-12
    nY = np.linalg.norm(Yc, axis=1) + 1e-12
    return np.sum(Xc*Yc, axis=1)/(nX*nY)

P = Path(r".\artifacts\gnss_synth_sweepcw_500k.npz")
D = np.load(P, allow_pickle=True)
X = to_flat(D['Xtr']); Y = to_flat(D['Ytr'])

c = corr_per_sample(X[:20000], Y[:20000])  # sample 20k for speed
qs = np.quantile(c, [0.01,0.1,0.5,0.9,0.99])
print("corr quantiles 1%,10%,50%,90%,99%:", qs)
