#!/usr/bin/env python3
"""
Universal benchmark script that auto-detects model architecture.

Usage:
  python benchmark_universal.py --model checkpoints/best_model.pt --data artifacts/gnss_synth_sweepcw_20k.npz
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


# ==================== Original 3-Stage Model ====================
class LightweightSignalSeparator(nn.Module):
    """Original 3-stage model"""
    def __init__(self, block_len=256, base_channels=32, dropout=0.2):
        super().__init__()
        self.block_len = block_len
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(2, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels*2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv1d(base_channels*2, base_channels*4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(base_channels*4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(base_channels*4, base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels*4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(base_channels*4, base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels*4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(base_channels*4, base_channels*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv1d(base_channels*4, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(base_channels*2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv1d(base_channels*2, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.out = nn.Conv1d(base_channels, 2, kernel_size=7, padding=3)
        
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        b = self.bottleneck(e3)
        
        d3 = self.dec3(b)
        d3 = torch.cat([d3, e2], dim=1)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        
        d1 = self.dec1(d2)
        
        out = self.out(d1)
        return out


# ==================== Fast 2-Stage Model ====================
class FastSignalSeparator(nn.Module):
    """Fast 2-stage model"""
    def __init__(self, block_len=256, base_channels=16, dropout=0.0):
        super().__init__()
        self.block_len = block_len
        
        self.enc1 = nn.Sequential(
            nn.Conv1d(2, base_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True)
        )
        if dropout > 0:
            self.enc1.add_module('dropout', nn.Dropout(dropout))
        
        self.enc2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels*2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        if dropout > 0:
            self.enc2.add_module('dropout', nn.Dropout(dropout))
        
        self.bottleneck = nn.Sequential(
            nn.Conv1d(base_channels*2, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        if dropout > 0:
            self.bottleneck.add_module('dropout', nn.Dropout(dropout))
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(base_channels*2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True)
        )
        if dropout > 0:
            self.dec2.add_module('dropout', nn.Dropout(dropout))
        
        self.dec1 = nn.Sequential(
            nn.Conv1d(base_channels*2, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True)
        )
        if dropout > 0:
            self.dec1.add_module('dropout', nn.Dropout(dropout))
        
        self.out = nn.Conv1d(base_channels, 2, kernel_size=3, padding=1)
        
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        d2 = self.dec2(b)
        d2 = torch.cat([d2, e1], dim=1)
        d1 = self.dec1(d2)
        out = self.out(d1)
        return out


# ==================== Ultra-Fast 1-Stage Model ====================
class UltraFastSeparator(nn.Module):
    """Ultra-fast 1-stage model"""
    def __init__(self, block_len=256, base_channels=24, dropout=0.0):
        super().__init__()
        self.block_len = block_len
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, base_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(base_channels*2, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.out = nn.Conv1d(base_channels, 2, kernel_size=3, padding=1)
        self.residual = nn.Conv1d(2, 2, kernel_size=1)
        
    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.out(out)
        return out + residual


# ==================== Metrics ====================
def compute_evm(pred, target):
    pred_c = pred[:, 0, :] + 1j * pred[:, 1, :]
    target_c = target[:, 0, :] + 1j * target[:, 1, :]
    error = pred_c - target_c
    error_power = torch.mean(torch.abs(error) ** 2)
    signal_power = torch.mean(torch.abs(target_c) ** 2)
    evm = torch.sqrt(error_power / (signal_power + 1e-12)) * 100
    return evm.item()


def compute_snr(pred, target):
    pred_c = pred[:, 0, :] + 1j * pred[:, 1, :]
    target_c = target[:, 0, :] + 1j * target[:, 1, :]
    error = pred_c - target_c
    error_power = torch.mean(torch.abs(error) ** 2)
    signal_power = torch.mean(torch.abs(target_c) ** 2)
    snr = 10 * torch.log10(signal_power / (error_power + 1e-12))
    return snr.item()


# ==================== Auto-detect Architecture ====================
def detect_architecture(state_dict):
    """Auto-detect model architecture from state_dict keys"""
    keys = list(state_dict.keys())
    
    # Check for enc3 (3-stage model)
    if any('enc3' in k for k in keys):
        return 'original'
    
    # Check for residual (ultrafast model)
    if any('residual' in k for k in keys):
        return 'ultrafast'
    
    # Otherwise it's the fast 2-stage model
    return 'fast'


def load_model(checkpoint_path, device):
    """Auto-detect and load the appropriate model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    args = checkpoint.get('args', {})
    
    # Auto-detect architecture
    arch = detect_architecture(state_dict)
    print(f"Auto-detected architecture: {arch}")
    
    # Get parameters
    block_len = args.get('block_len', 256)
    base_channels = args.get('base_channels', 32 if arch == 'original' else 16)
    dropout = args.get('dropout', 0.2 if arch == 'original' else 0.0)
    
    print(f"Parameters - Block: {block_len}, Channels: {base_channels}, Dropout: {dropout}\n")
    
    # Create appropriate model
    if arch == 'original':
        model = LightweightSignalSeparator(block_len=block_len, base_channels=base_channels, dropout=dropout)
    elif arch == 'ultrafast':
        model = UltraFastSeparator(block_len=block_len, base_channels=base_channels, dropout=dropout)
    else:  # fast
        model = FastSignalSeparator(block_len=block_len, base_channels=base_channels, dropout=dropout)
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model, checkpoint, arch


# ==================== Benchmark Functions ====================
def benchmark_single_inference(model, sample, device, n_warmup=100, n_runs=1000):
    model.eval()
    sample = sample.to(device)
    
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(sample)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(sample)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
    
    times = np.array(times)
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'median_ms': np.median(times),
        'p95_ms': np.percentile(times, 95),
        'p99_ms': np.percentile(times, 99)
    }


def benchmark_batch_inference(model, samples, device, batch_sizes=[1, 8, 16, 32, 64, 128], n_runs=100):
    model.eval()
    results = {}
    
    for bs in batch_sizes:
        if bs > len(samples):
            continue
        batch = samples[:bs].to(device)
        
        with torch.no_grad():
            for _ in range(10):
                _ = model(batch)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = model(batch)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)
        
        times = np.array(times)
        results[bs] = {
            'total_mean_ms': np.mean(times),
            'per_sample_ms': np.mean(times) / bs,
            'throughput_samples_per_sec': 1000.0 * bs / np.mean(times)
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Universal benchmark - auto-detects model architecture')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n_warmup', type=int, default=100)
    parser.add_argument('--n_runs', type=int, default=1000)
    parser.add_argument('--batch_benchmark', action='store_true')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    print(f"=== Universal Model Benchmark ===")
    print(f"Device: {device}")
    print(f"Model: {args.model}\n")
    
    # Load model (auto-detect architecture)
    print("Loading model...")
    model, checkpoint, arch = load_model(args.model, device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Architecture: {arch}")
    print(f"Parameters: {total_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    print(f"Validation EVM: {checkpoint.get('val_evm', 'N/A'):.2f}%")
    print(f"Validation SNR: {checkpoint.get('val_snr', 'N/A'):.2f} dB\n")
    
    # Load data
    data = np.load(args.data)
    Xva, Yva = data['Xva'], data['Yva']
    block_len = Xva.shape[1] // 2
    
    X_samples = torch.from_numpy(np.stack([Xva[:, ::2], Xva[:, 1::2]], axis=1)).float()
    Y_samples = torch.from_numpy(np.stack([Yva[:, ::2], Yva[:, 1::2]], axis=1)).float()
    
    print(f"Block length: {block_len} samples")
    print(f"Validation set: {len(X_samples)} examples\n")
    
    # Single sample benchmark
    print("=" * 60)
    print("SINGLE SAMPLE INFERENCE")
    print("=" * 60)
    
    single_sample = X_samples[0:1]
    single_target = Y_samples[0:1]
    
    timing = benchmark_single_inference(model, single_sample, device, args.n_warmup, args.n_runs)
    
    print(f"Warmup: {args.n_warmup}, Runs: {args.n_runs}\n")
    print(f"  Mean:     {timing['mean_ms']:.4f} ms")
    print(f"  Median:   {timing['median_ms']:.4f} ms")
    print(f"  Std:      {timing['std_ms']:.4f} ms")
    print(f"  Min:      {timing['min_ms']:.4f} ms")
    print(f"  Max:      {timing['max_ms']:.4f} ms")
    print(f"  95th %:   {timing['p95_ms']:.4f} ms")
    print(f"  99th %:   {timing['p99_ms']:.4f} ms")
    
    with torch.no_grad():
        pred = model(single_sample.to(device))
        evm = compute_evm(pred.cpu(), single_target)
        snr = compute_snr(pred.cpu(), single_target)
    
    print(f"\n  Output EVM: {evm:.2f}%")
    print(f"  Output SNR: {snr:.2f} dB")
    print(f"  Throughput: {1000.0 / timing['mean_ms']:.1f} samples/sec")
    
    # Real-time check
    sample_time = block_len / 2.0e6
    latency = timing['mean_ms']
    print(f"\n  Signal duration: {sample_time * 1000:.4f} ms")
    print(f"  Processing time: {latency:.4f} ms")
    print(f"  Real-time factor: {sample_time * 1000 / latency:.2f}x")
    
    if latency < sample_time * 1000:
        print(f"  ✓ REAL-TIME CAPABLE")
    else:
        print(f"  ✗ NOT REAL-TIME ({latency / (sample_time * 1000):.1f}x too slow)")
    
    # Batch benchmark
    if args.batch_benchmark:
        print("\n" + "=" * 60)
        print("BATCH INFERENCE")
        print("=" * 60)
        
        batch_results = benchmark_batch_inference(model, X_samples, device)
        
        print(f"\n{'Batch':<10} {'Total (ms)':<12} {'Per Sample (ms)':<16} {'Throughput (s/s)':<18}")
        print("-" * 60)
        for bs, res in batch_results.items():
            print(f"{bs:<10} {res['total_mean_ms']:<12.4f} {res['per_sample_ms']:<16.4f} {res['throughput_samples_per_sec']:<18.1f}")
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
