#!/usr/bin/env python3
"""
Quantize trained model to INT8 for faster inference.

Usage:
  python quantize_model.py --model checkpoints/best_model.pt \
                           --data artifacts/gnss_synth_sweepcw_20k.npz \
                           --output checkpoints/best_model_int8.pt
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.quantization as quantization
from pathlib import Path


# ==================== Model Definitions (same as before) ====================
class LightweightSignalSeparator(nn.Module):
    """Original 3-stage model"""
    def __init__(self, block_len=256, base_channels=32, dropout=0.2):
        super().__init__()
        self.block_len = block_len
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
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
        x = self.quant(x)
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
        out = self.dequant(out)
        return out


class FastSignalSeparator(nn.Module):
    """Fast 2-stage model"""
    def __init__(self, block_len=256, base_channels=16, dropout=0.0):
        super().__init__()
        self.block_len = block_len
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
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
        x = self.quant(x)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        d2 = self.dec2(b)
        d2 = torch.cat([d2, e1], dim=1)
        d1 = self.dec1(d2)
        out = self.out(d1)
        out = self.dequant(out)
        return out


class UltraFastSeparator(nn.Module):
    """Ultra-fast 1-stage model"""
    def __init__(self, block_len=256, base_channels=24, dropout=0.0):
        super().__init__()
        self.block_len = block_len
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
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
        x = self.quant(x)
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.out(out)
        out = out + residual
        out = self.dequant(out)
        return out


# ==================== Helper Functions ====================
def detect_architecture(state_dict):
    keys = list(state_dict.keys())
    if any('enc3' in k for k in keys):
        return 'original'
    if any('residual' in k for k in keys):
        return 'ultrafast'
    return 'fast'


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    # Remove quant/dequant if present (for loading pretrained weights)
    state_dict = {k: v for k, v in state_dict.items() if 'quant' not in k and 'dequant' not in k}
    
    args = checkpoint.get('args', {})
    arch = detect_architecture(state_dict)
    
    block_len = args.get('block_len', 256)
    base_channels = args.get('base_channels', 32 if arch == 'original' else 16)
    dropout = args.get('dropout', 0.2 if arch == 'original' else 0.0)
    
    if arch == 'original':
        model = LightweightSignalSeparator(block_len=block_len, base_channels=base_channels, dropout=dropout)
    elif arch == 'ultrafast':
        model = UltraFastSeparator(block_len=block_len, base_channels=base_channels, dropout=dropout)
    else:
        model = FastSignalSeparator(block_len=block_len, base_channels=base_channels, dropout=dropout)
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model, checkpoint, arch


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


# ==================== Quantization ====================
def calibrate_model(model, calibration_data, num_batches=100):
    """Run calibration data through model to collect statistics"""
    print(f"Calibrating with {num_batches} batches...")
    model.eval()
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(calibration_data):
            if i >= num_batches:
                break
            _ = model(inputs)
            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{num_batches} batches")


def quantize_model_static(model, calibration_loader):
    """
    Static quantization - requires calibration data.
    Best accuracy but needs representative data.
    """
    print("\n=== Static Quantization (INT8) ===")
    
    # Fuse modules (Conv+BN+ReLU)
    model.eval()
    # Note: Fusion is automatically done by prepare
    
    # Specify quantization configuration
    model.qconfig = torch.quantization.get_default_qconfig('x86')  # or 'fbgemm' for x86
    
    # Prepare model for quantization
    model_prepared = torch.quantization.prepare(model, inplace=False)
    
    # Calibrate
    calibrate_model(model_prepared, calibration_loader, num_batches=100)
    
    # Convert to quantized model
    model_quantized = torch.quantization.convert(model_prepared, inplace=False)
    
    print("✓ Static quantization complete")
    return model_quantized


def quantize_model_dynamic(model):
    """
    Dynamic quantization - no calibration needed.
    Slightly lower accuracy but easier to use.
    """
    print("\n=== Dynamic Quantization (INT8) ===")
    
    model.eval()
    model_quantized = torch.quantization.quantize_dynamic(
        model,
        {nn.Conv1d, nn.Linear},
        dtype=torch.qint8
    )
    
    print("✓ Dynamic quantization complete")
    return model_quantized


def benchmark_model(model, sample, n_runs=1000):
    """Quick benchmark"""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(100):
            _ = model(sample)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(sample)
            end = time.perf_counter()
            times.append((end - start) * 1000)
    
    return np.mean(times), np.std(times)


# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser(description='Quantize model to INT8')
    parser.add_argument('--model', type=str, required=True, help='Path to FP32 model')
    parser.add_argument('--data', type=str, required=True, help='Path to calibration data (.npz)')
    parser.add_argument('--output', type=str, required=True, help='Path to save quantized model')
    parser.add_argument('--method', type=str, default='static', choices=['static', 'dynamic'],
                       help='Quantization method (static=best accuracy, dynamic=easier)')
    parser.add_argument('--num_calibration', type=int, default=100,
                       help='Number of batches for calibration (static only)')
    args = parser.parse_args()
    
    print("=== Model Quantization ===")
    print(f"Input model: {args.model}")
    print(f"Output model: {args.output}")
    print(f"Method: {args.method}\n")
    
    # Load original model
    print("Loading FP32 model...")
    model_fp32, checkpoint, arch = load_model(args.model)
    
    total_params = sum(p.numel() for p in model_fp32.parameters())
    print(f"Architecture: {arch}")
    print(f"Parameters: {total_params:,}")
    print(f"FP32 size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Load calibration data
    print(f"\nLoading calibration data: {args.data}")
    data = np.load(args.data)
    Xva = data['Xva'][:1000]  # Use subset for calibration
    Yva = data['Yva'][:1000]
    
    block_len = Xva.shape[1] // 2
    X_samples = torch.from_numpy(np.stack([Xva[:, ::2], Xva[:, 1::2]], axis=1)).float()
    Y_samples = torch.from_numpy(np.stack([Yva[:, ::2], Yva[:, 1::2]], axis=1)).float()
    
    # Create calibration loader
    from torch.utils.data import TensorDataset, DataLoader
    calib_dataset = TensorDataset(X_samples, Y_samples)
    calib_loader = DataLoader(calib_dataset, batch_size=16, shuffle=False)
    
    print(f"Calibration samples: {len(X_samples)}")
    
    # Test sample for benchmarking
    test_sample = X_samples[0:1]
    test_target = Y_samples[0:1]
    
    # Benchmark original model
    print("\n" + "="*60)
    print("ORIGINAL FP32 MODEL")
    print("="*60)
    
    with torch.no_grad():
        pred_fp32 = model_fp32(test_sample)
        evm_fp32 = compute_evm(pred_fp32, test_target)
        snr_fp32 = compute_snr(pred_fp32, test_target)
    
    time_fp32, std_fp32 = benchmark_model(model_fp32, test_sample, n_runs=1000)
    
    print(f"Inference time: {time_fp32:.4f} ± {std_fp32:.4f} ms")
    print(f"EVM: {evm_fp32:.2f}%")
    print(f"SNR: {snr_fp32:.2f} dB")
    
    # Quantize
    if args.method == 'static':
        model_int8 = quantize_model_static(model_fp32, calib_loader)
    else:
        model_int8 = quantize_model_dynamic(model_fp32)
    
    # Benchmark quantized model
    print("\n" + "="*60)
    print(f"QUANTIZED INT8 MODEL ({args.method.upper()})")
    print("="*60)
    
    with torch.no_grad():
        pred_int8 = model_int8(test_sample)
        evm_int8 = compute_evm(pred_int8, test_target)
        snr_int8 = compute_snr(pred_int8, test_target)
    
    time_int8, std_int8 = benchmark_model(model_int8, test_sample, n_runs=1000)
    
    print(f"Inference time: {time_int8:.4f} ± {std_int8:.4f} ms")
    print(f"EVM: {evm_int8:.2f}%")
    print(f"SNR: {snr_int8:.2f} dB")
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    speedup = time_fp32 / time_int8
    size_reduction = 4.0  # Approximate (FP32 -> INT8)
    evm_degradation = evm_int8 - evm_fp32
    snr_degradation = snr_fp32 - snr_int8
    
    print(f"Speedup: {speedup:.2f}x ({time_fp32:.4f}ms → {time_int8:.4f}ms)")
    print(f"Model size reduction: ~{size_reduction:.1f}x")
    print(f"EVM change: {evm_degradation:+.2f}% ({evm_fp32:.2f}% → {evm_int8:.2f}%)")
    print(f"SNR change: {snr_degradation:+.2f}dB ({snr_fp32:.2f}dB → {snr_int8:.2f}dB)")
    
    # Real-time check
    sample_time = block_len / 2.0e6
    print(f"\nSignal duration: {sample_time * 1000:.4f} ms")
    print(f"FP32 processing: {time_fp32:.4f} ms (RT factor: {sample_time * 1000 / time_fp32:.2f}x)")
    print(f"INT8 processing: {time_int8:.4f} ms (RT factor: {sample_time * 1000 / time_int8:.2f}x)")
    
    if time_int8 < sample_time * 1000:
        print("✓ INT8 model is REAL-TIME CAPABLE!")
    else:
        print(f"✗ INT8 model still NOT real-time ({time_int8 / (sample_time * 1000):.1f}x too slow)")
    
    # Save quantized model
    print(f"\nSaving quantized model to: {args.output}")
    
    # Create directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Save with metadata
    save_dict = {
        'model_state_dict': model_int8.state_dict(),
        'quantization_method': args.method,
        'original_evm': evm_fp32,
        'quantized_evm': evm_int8,
        'speedup': speedup,
        'architecture': arch,
        'args': checkpoint.get('args', {})
    }
    
    torch.save(save_dict, args.output)
    print("✓ Saved successfully")
    
    print("\n" + "="*60)
    print("Quantization complete!")
    print("="*60)


if __name__ == '__main__':
    main()
