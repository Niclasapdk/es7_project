#!/usr/bin/env python3
import os
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# =========================
# CONFIG
# =========================
BASE_PATH = "real_data/"

# Granularity for labels: "binary" | "by_freq" | "by_freq_gain"
GRANULARITY = "binary"

# Windowing
WINDOW = 64            # samples per window
HOP = 16              # stride (samples)
LAYOUT = "block"       # 'block' -> [I... , Q...] ; 'interleave' -> [I0,Q0,I1,Q1,...]

# Normalization options
PER_FILE_NORM = True        # normalize each measurement before windowing
PER_WINDOW_NORM = True      # normalize each window (see function below)
ZERO_MEAN_PER_WINDOW = True # subtract mean I/Q per window before unit-power

# Split ratios (by measurement)
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SEED = 42

# Model / Training
HIDDEN = [64, 32, 16]
DROPOUT = 0.2
BATCH_SIZE = 512
EPOCHS = 60
LR = 2e-3
WEIGHT_DECAY = 1e-4
USE_WEIGHTED_SAMPLER = True
CKPT_PATH = "mlp_best.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(SEED)

# =========================
# LOADER (your teammate's)
# =========================
def load_measurement_data(base_path):
    """
    Load all measurement data from the nested folder structure
    Returns a list of dictionaries with data and metadata
    """
    data_records = []
    base_path = Path(base_path)

    # Load data WITHOUT interference
    no_int_path = base_path / "Clean_BPSK_reference"
    if no_int_path.exists():
        for csv_file in no_int_path.glob("*.csv"):
            df = pd.read_csv(csv_file)
            data_records.append({
                'data': df[['I', 'Q']].values,  # Nx2 I/Q
                'time': df['Time'].values,
                'has_interference': False,
                'interference_freq': None,
                'interference_gain': None,
                'file_path': str(csv_file),
                'measurement_id': csv_file.stem
            })

    # Load data WITH interference
    with_int_path = base_path / "BPSK_Interference"
    if with_int_path.exists():
        for freq_dir in with_int_path.iterdir():
            if freq_dir.is_dir():
                freq_name = freq_dir.name
                freq_value = extract_frequency(freq_name)
                for gain_dir in freq_dir.iterdir():
                    if gain_dir.is_dir():
                        gain_name = gain_dir.name
                        gain_value = extract_gain(gain_name)
                        for csv_file in gain_dir.glob("*.csv"):
                            df = pd.read_csv(csv_file)
                            data_records.append({
                                'data': df[['I', 'Q']].values,
                                'time': df['Time'].values,
                                'has_interference': True,
                                'interference_freq': freq_value,
                                'interference_gain': gain_value,
                                'file_path': str(csv_file),
                                'measurement_id': csv_file.stem
                            })
    return data_records

def extract_frequency(freq_dir_name):
    """Adapt to your naming convention; currently returns string after 'freq_' if present."""
    if 'freq_' in freq_dir_name:
        return freq_dir_name.replace('freq_', '')
    return freq_dir_name

def extract_gain(gain_dir_name):
    """Adapt to your naming convention; currently returns string after 'gain_' if present."""
    if 'gain_' in gain_dir_name:
        return gain_dir_name.replace('gain_', '')
    return gain_dir_name

# =========================
# PRINT DATA SUMMARY
# =========================
all_data = load_measurement_data(BASE_PATH)

print(f"Total measurements: {len(all_data)}")
has_interference = sum(1 for r in all_data if r['has_interference'])
no_interference = len(all_data) - has_interference
print(f"With interference: {has_interference}")
print(f"Without interference: {no_interference}")

freq_gain_combinations = Counter()
for r in all_data:
    if r['has_interference']:
        combo = f"freq_{r['interference_freq']}_gain_{r['interference_gain']}"
        freq_gain_combinations[combo] += 1

print("\nInterference type distribution:")
for combo, count in freq_gain_combinations.items():
    print(f"  {combo}: {count} measurements")

print(f"\nData shape for first measurement: {all_data[0]['data'].shape}")
print(f"Data type: {all_data[0]['data'].dtype}")

# =========================
# HELPERS
# =========================
def build_record_label(rec, granularity="binary"):
    if not rec["has_interference"]:
        return "clean"
    if granularity == "binary":
        return "interference"
    if granularity == "by_freq":
        return f"f:{rec.get('interference_freq')}"
    if granularity == "by_freq_gain":
        return f"f:{rec.get('interference_freq')}|g:{rec.get('interference_gain')}"
    raise ValueError("bad granularity")

def window_iq(iq: np.ndarray, win: int, hop: int, layout: str) -> np.ndarray:
    """iq: (N,2) float; returns (W, 2*win) float32"""
    iq = iq.astype(np.float32)
    N = iq.shape[0]
    if N < win:
        return np.empty((0, 2*win), dtype=np.float32)
    starts = range(0, N - win + 1, hop)
    chunks = np.stack([iq[s:s+win] for s in starts], axis=0)  # (W,win,2)
    I, Q = chunks[..., 0], chunks[..., 1]
    if layout == "interleave":
        flat = np.empty((chunks.shape[0], 2*win), dtype=np.float32)
        flat[:, 0::2] = I
        flat[:, 1::2] = Q
        return flat
    return np.concatenate([I, Q], axis=1).astype(np.float32)

def per_window_normalize(X, layout="block", zero_mean=False):
    """
    Unit-power normalization per window; optional zero-mean per window.
    """
    W, D = X.shape
    win = D // 2
    if layout == "interleave":
        I = X[:, 0::2].copy()
        Q = X[:, 1::2].copy()
    else:
        I = X[:, :win].copy()
        Q = X[:, win:].copy()

    if zero_mean:
        I -= I.mean(axis=1, keepdims=True)
        Q -= Q.mean(axis=1, keepdims=True)

    P = (I**2 + Q**2).sum(axis=1, keepdims=True) + 1e-8
    I /= np.sqrt(P)
    Q /= np.sqrt(P)

    if layout == "interleave":
        out = np.empty_like(X)
        out[:, 0::2] = I
        out[:, 1::2] = Q
        return out
    else:
        return np.concatenate([I, Q], axis=1)

def window_power(X, layout="block"):
    W, D = X.shape
    win = D // 2
    if layout == "interleave":
        I = X[:, 0::2]
        Q = X[:, 1::2]
    else:
        I = X[:, :win]
        Q = X[:, win:]
    return (I**2 + Q**2).mean(axis=1)

class NPDataset(Dataset):
    def __init__(self, X, y):  # X already float32
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, dropout=0.1, use_bn=True):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h)]
            if use_bn: layers += [nn.BatchNorm1d(h)]
            layers += [nn.ReLU(inplace=True)]
            if dropout > 0: layers += [nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, num_classes)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def f1_macro_from_logits(logits, y, C):
    yt = y.cpu().numpy(); yp = logits.argmax(1).cpu().numpy()
    f1s = []
    for c in range(C):
        tp = np.sum((yt==c)&(yp==c))
        fp = np.sum((yt!=c)&(yp==c))
        fn = np.sum((yt==c)&(yp!=c))
        prec = tp/(tp+fp+1e-12); rec = tp/(tp+fn+1e-12)
        f1s.append(2*prec*rec/(prec+rec+1e-12))
    return float(np.mean(f1s))

def confusion_matrix_from_preds(y_true, y_pred, C):
    M = np.zeros((C, C), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[t, p] += 1
    return M

# =========================
# LABELS PER MEASUREMENT
# =========================
keys = [build_record_label(r, GRANULARITY) for r in all_data]
classes = sorted(set(keys), key=lambda k: (k != "clean", k))
cls2idx = {c: i for i, c in enumerate(classes)}
y_record = np.array([cls2idx[k] for k in keys], dtype=np.int64)

# =========================
# WINDOWING (with owner_ids)
# =========================
X_list, y_list, owner_ids = [], [], []
for m_id, (rec, yr) in enumerate(zip(all_data, y_record)):
    iq = rec["data"]
    if PER_FILE_NORM:
        mu = iq.mean(axis=0, keepdims=True)
        sd = iq.std(axis=0, keepdims=True) + 1e-8
        iq = (iq - mu) / sd
    Xw = window_iq(iq, WINDOW, HOP, LAYOUT)
    if Xw.shape[0] == 0:
        continue
    X_list.append(Xw)
    y_list.append(np.full((Xw.shape[0],), yr, dtype=np.int64))
    owner_ids.append(np.full((Xw.shape[0],), m_id, dtype=np.int32))

if not X_list:
    raise SystemExit("No windows created. Lower WINDOW or check data shapes.")

X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)
owner_ids = np.concatenate(owner_ids, axis=0)
print(f"Windowed dataset: X={X.shape}, y={y.shape}, classes={classes}")

# =========================
# 3-WAY GROUPED SPLIT (by measurement): TRAIN / VAL / TEST
# =========================
rng = np.random.default_rng(SEED)
meas_by_class = {}
for m_id, yr in enumerate(y_record):
    meas_by_class.setdefault(yr, []).append(m_id)

train_meas, val_meas, test_meas = set(), set(), set()
for cls, mids in meas_by_class.items():
    mids = mids.copy()
    rng.shuffle(mids)
    n = len(mids)
    n_test = max(1, int(round(TEST_FRAC * n)))
    n_val  = max(1, int(round(VAL_FRAC  * n)))
    test_meas.update(mids[:n_test])
    val_meas.update(mids[n_test:n_test+n_val])
    train_meas.update(mids[n_test+n_val:])

assert train_meas.isdisjoint(val_meas)
assert train_meas.isdisjoint(test_meas)
assert val_meas.isdisjoint(test_meas)

train_mask = np.isin(owner_ids, list(train_meas))
val_mask   = np.isin(owner_ids, list(val_meas))
test_mask  = np.isin(owner_ids, list(test_meas))

Xtr, ytr = X[train_mask], y[train_mask]
Xva, yva = X[val_mask],   y[val_mask]
Xte, yte = X[test_mask],  y[test_mask]

print(f"Train meas: {len(train_meas)}, Val meas: {len(val_meas)}, Test meas: {len(test_meas)}")
print(f"Train win: {Xtr.shape[0]}, Val win: {Xva.shape[0]}, Test win: {Xte.shape[0]}")

# =========================
# NORMALIZATION
# =========================
# Per-window normalization nukes amplitude/DC shortcuts
if PER_WINDOW_NORM:
    Xtr = per_window_normalize(Xtr, layout=LAYOUT, zero_mean=ZERO_MEAN_PER_WINDOW)
    Xva = per_window_normalize(Xva, layout=LAYOUT, zero_mean=ZERO_MEAN_PER_WINDOW)
    Xte = per_window_normalize(Xte, layout=LAYOUT, zero_mean=ZERO_MEAN_PER_WINDOW)
    mu = sd = None
else:
    # Optional global z-score (fit on TRAIN only) if you didn't already normalize per-file
    if not PER_FILE_NORM:
        mu = Xtr.mean(axis=0, keepdims=True)
        sd = Xtr.std(axis=0, keepdims=True) + 1e-8
        Xtr = (Xtr - mu) / sd
        Xva = (Xva - mu) / sd
        Xte = (Xte - mu) / sd
    else:
        mu = None; sd = None

# =========================
# QUICK POWER BASELINE ON TEST
# =========================
p_tr = window_power(Xtr, layout=LAYOUT)
p_te = window_power(Xte, layout=LAYOUT)
m0, m1 = p_tr[ytr==0].mean(), p_tr[ytr==1].mean()
if m1 > m0:
    thr = np.median(p_tr[ytr==1])
    yp_te_baseline = (p_te > thr).astype(int)
else:
    thr = np.median(p_tr[ytr==0])
    yp_te_baseline = (p_te < thr).astype(int)
baseline_acc_test = (yp_te_baseline == yte).mean()
print(f"[Baseline power threshold] TEST ACC = {baseline_acc_test:.4f}"
      f"{' (per-window norm ON)' if PER_WINDOW_NORM else ''}")


# =========================
# DATASETS / LOADERS
# =========================
train_ds = NPDataset(Xtr, ytr)
val_ds   = NPDataset(Xva, yva)

if USE_WEIGHTED_SAMPLER:
    cls_counts = np.bincount(ytr, minlength=len(classes)).astype(np.float64)
    class_weights = (cls_counts.sum() / (cls_counts + 1e-8))
    sample_weights = class_weights[ytr]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=True)
else:
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# =========================
# MODEL / OPTIM
# =========================
in_dim = X.shape[1]
model = MLP(in_dim, HIDDEN, num_classes=len(classes), dropout=DROPOUT, use_bn=True).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
criterion = nn.CrossEntropyLoss()

best_val_f1 = -1.0
best_state = None

# =========================
# TRAIN
# =========================
for epoch in range(1, EPOCHS+1):
    # train
    model.train()
    tr_loss, tr_correct = 0.0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tr_loss += loss.item() * xb.size(0)
        tr_correct += (logits.argmax(1) == yb).float().sum().item()
    tr_loss /= len(train_ds)
    tr_acc = tr_correct / len(train_ds)

    # validate
    model.eval()
    va_loss, va_correct = 0.0, 0
    all_logits, all_y = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            va_loss += loss.item() * xb.size(0)
            va_correct += (logits.argmax(1) == yb).float().sum().item()
            all_logits.append(logits.cpu())
            all_y.append(yb.cpu())
    va_loss /= len(val_ds)
    va_acc = va_correct / len(val_ds)
    all_logits = torch.cat(all_logits, 0)
    all_y = torch.cat(all_y, 0)
    va_f1 = f1_macro_from_logits(all_logits, all_y, len(classes))
    sched.step(va_f1)

    print(f"Epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
          f"val loss {va_loss:.4f} acc {va_acc:.4f} F1 {va_f1:.4f}")

    if va_f1 > best_val_f1:
        best_val_f1 = va_f1
        best_state = {
            "model": model.state_dict(),
            "in_dim": in_dim,
            "hidden": HIDDEN,
            "num_classes": len(classes),
            "classes": classes,
            "dropout": DROPOUT,
            "window": WINDOW, "hop": HOP, "layout": LAYOUT,
            "granularity": GRANULARITY,
            "per_file_norm": PER_FILE_NORM,
            "per_window_norm": PER_WINDOW_NORM,
            "zero_mean_per_window": ZERO_MEAN_PER_WINDOW,
            "mean": (mu if 'mu' in locals() else None),
            "std": (sd if 'sd' in locals() else None),
        }
        torch.save(best_state, CKPT_PATH)

print(f"\nBest validation F1: {best_val_f1:.4f}")
print(f"Saved checkpoint -> {CKPT_PATH}")

# =========================
# FINAL TEST EVAL
# =========================
model.eval()
with torch.no_grad():
    logits_te = model(torch.from_numpy(Xte).to(device))
pred_te = logits_te.argmax(1).cpu().numpy()
test_acc = (pred_te == yte).mean()

# per-class P/R/F1 and confusion matrix
def per_class_prf(y_true, y_pred, C):
    out = []
    for c in range(C):
        tp = np.sum((y_true==c)&(y_pred==c))
        fp = np.sum((y_true!=c)&(y_pred==c))
        fn = np.sum((y_true==c)&(y_pred!=c))
        prec = tp/(tp+fp+1e-12); rec = tp/(tp+fn+1e-12)
        f1 = 2*prec*rec/(prec+rec+1e-12)
        out.append((prec, rec, f1))
    return out

cm = confusion_matrix_from_preds(yte, pred_te, len(classes))
print("\n=== FINAL TEST ===")
print(f"Accuracy: {test_acc:.4f}")
for i,(p,r,f) in enumerate(per_class_prf(yte, pred_te, len(classes))):
    print(f"  class {i} ({classes[i]}): P={p:.3f} R={r:.3f} F1={f:.3f}")

print("\nConfusion matrix (rows=true, cols=pred):")
print("      " + " ".join([f"{i:>6d}" for i in range(len(classes))]))
for i,row in enumerate(cm):
    print(f"{i:>3d} | " + " ".join([f"{v:>6d}" for v in row]))
print("\nClasses:")
for i,c in enumerate(classes):
    print(f"  {i}: {c}")
