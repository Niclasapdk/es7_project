import numpy as np
import re

PATH = "gnss_l1_sweptcw.npz"  # <-- adjust if needed

data = np.load(PATH, allow_pickle=True, mmap_mode='r')

print("Top-level NPZ keys:", data.files)
print()

# 1) Shapes
for split in ("tr", "va", "te"):
    x_key = f"X{split}"
    y_key = f"Y{split}"
    if x_key in data.files and y_key in data.files:
        X = data[x_key]
        Y = data[y_key]
        print(f"{x_key}.shape = {X.shape}")
        print(f"{y_key}.shape = {Y.shape}")
        print()

# 2) Raw meta string
if "meta" not in data.files:
    print("No 'meta' key found in NPZ.")
    raise SystemExit

meta_arr = data["meta"]
print("meta_arr.shape:", meta_arr.shape, "dtype:", meta_arr.dtype)

if meta_arr.shape == ():
    meta_raw = meta_arr.item()
else:
    meta_raw = meta_arr.ravel()[0]

if isinstance(meta_raw, bytes):
    meta_raw = meta_raw.decode("utf-8", errors="replace")

print("type(meta_raw):", type(meta_raw))

if not isinstance(meta_raw, str):
    print("meta_raw is not a string, stopping.")
    raise SystemExit

# Show only a tiny snippet so it doesn't spam your terminal
print("\n=== meta_raw head (first 400 chars) ===")
print(meta_raw[:400])
print("\n=== meta_raw tail (last 400 chars) ===")
print(meta_raw[-400:])
print()

# 3) Helper: extract scalar value by key using regex on the string
def find_scalar_number(key):
    """
    Find the FIRST occurrence of `'key': <number>` or `"key": <number>`
    and parse the number as float.
    """
    pattern = rf"['\"]{key}['\"]\s*:\s*([-+0-9.eE]+)"
    m = re.search(pattern, meta_raw)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None

def find_scalar_string(key):
    """
    Find the FIRST occurrence of `'key': 'value'` or `"key": "value"`.
    """
    pattern = rf"['\"]{key}['\"]\s*:\s*(['\"])(.*?)\1"
    m = re.search(pattern, meta_raw)
    if not m:
        return None
    return m.group(2)

# 4) Extract header-ish fields
keys_num = [
    "fs",
    "chip_rate",
    "samples_per_chip",
    "block_len",
    "block_ms",
    "prn_low",
    "prn_high",
    "doppler_max_hz",
    "snr_db",
    "jsr_db",
    "n_train",
    "n_val",
    "n_test",
]
keys_str = ["jammer"]

print("=== Approx. dataset-level scalars scraped from meta_raw ===")
for k in keys_num:
    v = find_scalar_number(k)
    print(f"{k:15s} -> {v}")

for k in keys_str:
    v = find_scalar_string(k)
    print(f"{k:15s} -> {v}")

# 5) Rough SNR / JSR ranges across ALL occurrences in meta_raw
def find_all_numbers_for_key(key):
    pattern = rf"['\"]{key}['\"]\s*:\s*([-+0-9.eE]+)"
    matches = re.findall(pattern, meta_raw)
    vals = []
    for s in matches:
        try:
            vals.append(float(s))
        except ValueError:
            pass
    return vals

snr_vals = find_all_numbers_for_key("snr_db")
jsr_vals = find_all_numbers_for_key("jsr_db")

print("\n=== snr_db occurrences ===")
if snr_vals:
    print(f"count = {len(snr_vals)}, min = {min(snr_vals)}, max = {max(snr_vals)}")
else:
    print("no numeric snr_db found")

print("\n=== jsr_db occurrences ===")
if jsr_vals:
    print(f"count = {len(jsr_vals)}, min = {min(jsr_vals)}, max = {max(jsr_vals)}")
else:
    print("no numeric jsr_db found")
