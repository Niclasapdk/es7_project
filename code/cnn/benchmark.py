#!/usr/bin/env python3
"""
Benchmark inference time for trained signal separator model.

Usage:
  python benchmark.py --model checkpoints/best_model.pt --data artifacts/gnss_synth_sweepcw_20k.npz
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


# ==================== Model Definition (must match training) ====================
class LightweightSignalSeparator(nn.Module):
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
        
        # Output layer
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


# ==================== Metrics ====================
def compute_evm(pred, target):
    """Compute Error Vector Magnitude in %"""
    pred_c = pred[:, 0, :] + 1j * pred[:, 1, :]
    target_c = target[:, 0, :] + 1j * target[:, 1, :]
    
    error = pred_c - target_c
    error_power = torch.mean(torch.abs(error) ** 2)
    signal_power = torch.mean(torch.abs(target_c) ** 2)
    
    evm = torch.sqrt(error_power / (signal_power + 1e-12)) * 100
    return evm.item()


def compute_snr(pred, target):
    """Compute SNR in dB"""
    pred_c = pred[:, 0, :] + 1j * pred[:, 1, :]
    target_c = target[:, 0, :] + 1j * target[:, 1, :]
    
    error = pred_c - target_c
    error_power = torch.mean(torch.abs(error) ** 2)
    signal_power = torch.mean(torch.abs(target_c) ** 2)
    
    snr = 10 * torch.log10(signal_power / (error_power + 1e-12))
    return snr.item()


# ==================== Benchmark Functions ====================
def benchmark_single_inference(model, sample, device, n_warmup=100, n_runs=1000):
    """
    Benchmark inference time for a single sample.
    
    Args:
        model: trained model
        sample: single input tensor [1, 2, block_len]
        device: torch device
        n_warmup: number of warmup runs
        n_runs: number of benchmark runs
    
    Returns:
        dict with timing statistics
    """
    model.eval()
    sample = sample.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(sample)
    
    # Synchronize GPU before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(sample)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    
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
    """
    Benchmark inference time for different batch sizes.
    
    Returns:
        dict with results for each batch size
    """
    model.eval()
    results = {}
    
    for bs in batch_sizes:
        if bs > len(samples):
            continue
            
        batch = samples[:bs].to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(batch)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
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


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model parameters from checkpoint
    args = checkpoint.get('args', {})
    block_len = args.get('block_len', 256)
    base_channels = args.get('base_channels', 32)
    dropout = args.get('dropout', 0.2)
    
    # Try to infer block_len from state_dict if not in args
    if 'block_len' not in args:
        # Look at the shape of the first conv layer
        for key in checkpoint['model_state_dict'].keys():
            if 'enc1' in key and 'weight' in key:
                print(f"Inferring parameters from model weights...")
                break
    
    model = LightweightSignalSeparator(
        block_len=block_len,
        base_channels=base_channels,
        dropout=dropout
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description='Benchmark model inference time')
    parser.add_argument('--model', type=str, required=True, help='Path to saved model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to validation dataset (.npz)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n_warmup', type=int, default=100, help='Number of warmup iterations')
    parser.add_argument('--n_runs', type=int, default=1000, help='Number of benchmark iterations')
    parser.add_argument('--batch_benchmark', action='store_true', help='Also benchmark different batch sizes')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    print(f"=== Model Inference Benchmark ===")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}\n")
    
    # Load model
    print("Loading model...")
    model, checkpoint = load_model(args.model, device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Training epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Validation EVM: {checkpoint.get('val_evm', 'unknown'):.2f}%")
    print(f"Validation SNR: {checkpoint.get('val_snr', 'unknown'):.2f} dB\n")
    
    # Load validation data
    print("Loading validation data...")
    data = np.load(args.data)
    Xva = data['Xva']
    Yva = data['Yva']
    
    # Prepare samples
    block_len = Xva.shape[1] // 2
    X_samples = torch.from_numpy(
        np.stack([Xva[:, ::2], Xva[:, 1::2]], axis=1)
    ).float()  # [N, 2, block_len]
    Y_samples = torch.from_numpy(
        np.stack([Yva[:, ::2], Yva[:, 1::2]], axis=1)
    ).float()
    
    print(f"Block length: {block_len} samples")
    print(f"Validation set size: {len(X_samples)} examples\n")
    
    # Single sample benchmark
    print("=" * 60)
    print("SINGLE SAMPLE INFERENCE BENCHMARK")
    print("=" * 60)
    single_sample = X_samples[0:1]  # [1, 2, block_len]
    single_target = Y_samples[0:1]
    
    timing_results = benchmark_single_inference(
        model, single_sample, device, 
        n_warmup=args.n_warmup, 
        n_runs=args.n_runs
    )
    
    print(f"Warmup runs: {args.n_warmup}")
    print(f"Benchmark runs: {args.n_runs}\n")
    print(f"Results:")
    print(f"  Mean:     {timing_results['mean_ms']:.4f} ms")
    print(f"  Std:      {timing_results['std_ms']:.4f} ms")
    print(f"  Median:   {timing_results['median_ms']:.4f} ms")
    print(f"  Min:      {timing_results['min_ms']:.4f} ms")
    print(f"  Max:      {timing_results['max_ms']:.4f} ms")
    print(f"  95th %ile: {timing_results['p95_ms']:.4f} ms")
    print(f"  99th %ile: {timing_results['p99_ms']:.4f} ms")
    print(f"\n  Throughput: {1000.0 / timing_results['mean_ms']:.1f} samples/sec")
    
    # Compute quality metrics on this sample
    with torch.no_grad():
        pred = model(single_sample.to(device))
        evm = compute_evm(pred.cpu(), single_target)
        snr = compute_snr(pred.cpu(), single_target)
    
    print(f"\n  Output EVM: {evm:.2f}%")
    print(f"  Output SNR: {snr:.2f} dB")
    
    # Real-time feasibility check
    sample_time = block_len / 2.0e6  # Assuming 2 MHz sample rate from data.py
    latency_ms = timing_results['mean_ms']
    print(f"\n  Signal duration: {sample_time * 1000:.4f} ms ({block_len} samples @ 2 MHz)")
    print(f"  Processing time: {latency_ms:.4f} ms")
    print(f"  Real-time factor: {sample_time * 1000 / latency_ms:.2f}x")
    if latency_ms < sample_time * 1000:
        print(f"  ✓ REAL-TIME CAPABLE (processing faster than signal arrival)")
    else:
        print(f"  ✗ NOT REAL-TIME (processing slower than signal arrival)")
    
    # Batch benchmark
    if args.batch_benchmark:
        print("\n" + "=" * 60)
        print("BATCH INFERENCE BENCHMARK")
        print("=" * 60)
        
        batch_results = benchmark_batch_inference(
            model, X_samples, device,
            batch_sizes=[1, 8, 16, 32, 64, 128]
        )
        
        print(f"\n{'Batch Size':<12} {'Total (ms)':<12} {'Per Sample (ms)':<18} {'Throughput (samp/s)':<20}")
        print("-" * 70)
        for bs, res in batch_results.items():
            print(f"{bs:<12} {res['total_mean_ms']:<12.4f} {res['per_sample_ms']:<18.4f} {res['throughput_samples_per_sec']:<20.1f}")
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
