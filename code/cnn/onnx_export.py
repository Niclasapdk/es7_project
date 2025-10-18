#!/usr/bin/env python3
"""
Export PyTorch model to ONNX and benchmark with ONNX Runtime.

Step 1 - Export:
  python onnx_export.py --model checkpoints/best_model.pt --output checkpoints/best_model.onnx

Step 2 - Benchmark:
  python onnx_export.py --model checkpoints/best_model.pt --output checkpoints/best_model.onnx \
                        --data artifacts/gnss_synth_sweepcw_20k.npz --benchmark
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


# ==================== Model Definitions ====================
class LightweightSignalSeparator(nn.Module):
    """Original 3-stage model"""
    def __init__(self, block_len=256, base_channels=32, dropout=0.2):
        super().__init__()
        self.block_len = block_len
        
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
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, checkpoint, arch, block_len


def compute_evm(pred, target):
    pred_c = pred[:, 0, :] + 1j * pred[:, 1, :]
    target_c = target[:, 0, :] + 1j * target[:, 1, :]
    error = pred_c - target_c
    error_power = np.mean(np.abs(error) ** 2)
    signal_power = np.mean(np.abs(target_c) ** 2)
    evm = np.sqrt(error_power / (signal_power + 1e-12)) * 100
    return float(evm)


def compute_snr(pred, target):
    pred_c = pred[:, 0, :] + 1j * pred[:, 1, :]
    target_c = target[:, 0, :] + 1j * target[:, 1, :]
    error = pred_c - target_c
    error_power = np.mean(np.abs(error) ** 2)
    signal_power = np.mean(np.abs(target_c) ** 2)
    snr = 10 * np.log10(signal_power / (error_power + 1e-12))
    return float(snr)


# ==================== ONNX Export ====================
def export_to_onnx(model, output_path, block_len, opset_version=14):
    """Export PyTorch model to ONNX format"""
    print("\n=== Exporting to ONNX ===")
    
    # Create dummy input
    dummy_input = torch.randn(1, 2, block_len)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✓ Model exported to: {output_path}")
    
    # Get file size
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  ONNX file size: {file_size_mb:.2f} MB")


# ==================== ONNX Inference ====================
def benchmark_onnx(onnx_path, sample, n_warmup=100, n_runs=1000):
    """Benchmark ONNX model inference"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("\n ERROR: onnxruntime not installed!")
        print("Install with: pip install onnxruntime")
        return None, None
    
    print("\n=== ONNX Runtime Benchmark ===")
    
    # Create inference session
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 1  # Single thread for fair comparison
    
    session = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Warmup
    for _ in range(n_warmup):
        _ = session.run([output_name], {input_name: sample})
    
    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        output = session.run([output_name], {input_name: sample})
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
    }, output[0]


def benchmark_pytorch(model, sample, n_warmup=100, n_runs=1000):
    """Benchmark PyTorch model inference"""
    model.eval()
    sample_torch = torch.from_numpy(sample).float()
    
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(sample_torch)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            output = model(sample_torch)
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
    }, output.numpy()


# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser(description='Export and benchmark ONNX model')
    parser.add_argument('--model', type=str, required=True, help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, required=True, help='Path to save ONNX model')
    parser.add_argument('--data', type=str, help='Path to test data (.npz) for benchmarking')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark comparison')
    parser.add_argument('--opset', type=int, default=14, help='ONNX opset version')
    args = parser.parse_args()
    
    print("=== ONNX Model Export and Benchmark ===")
    
    # Load PyTorch model
    print(f"\nLoading PyTorch model: {args.model}")
    model, checkpoint, arch, block_len = load_model(args.model)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Architecture: {arch}")
    print(f"Parameters: {total_params:,}")
    print(f"Block length: {block_len}")
    
    # Export to ONNX
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    export_to_onnx(model, args.output, block_len, opset_version=args.opset)
    
    # Benchmark if requested
    if args.benchmark:
        if not args.data:
            print("\nERROR: --data required for benchmarking")
            return
        
        try:
            import onnxruntime as ort
            print(f"ONNX Runtime version: {ort.__version__}")
        except ImportError:
            print("\nERROR: onnxruntime not installed!")
            print("Install with: pip install onnxruntime")
            return
        
        # Load test data
        print(f"\nLoading test data: {args.data}")
        data = np.load(args.data)
        Xva = data['Xva']
        Yva = data['Yva']
        
        # Prepare test sample
        X_samples = np.stack([Xva[:, ::2], Xva[:, 1::2]], axis=1).astype(np.float32)
        Y_samples = np.stack([Yva[:, ::2], Yva[:, 1::2]], axis=1).astype(np.float32)
        
        test_sample = X_samples[0:1]
        test_target = Y_samples[0:1]
        
        print(f"Test sample shape: {test_sample.shape}")
        
        # Benchmark PyTorch
        print("\n" + "="*60)
        print("PYTORCH BASELINE")
        print("="*60)
        
        pytorch_timing, pytorch_output = benchmark_pytorch(model, test_sample, n_runs=1000)
        pytorch_evm = compute_evm(pytorch_output, test_target)
        pytorch_snr = compute_snr(pytorch_output, test_target)
        
        print(f"Inference time: {pytorch_timing['mean_ms']:.4f} ± {pytorch_timing['std_ms']:.4f} ms")
        print(f"  Median: {pytorch_timing['median_ms']:.4f} ms")
        print(f"  Min: {pytorch_timing['min_ms']:.4f} ms")
        print(f"  95th percentile: {pytorch_timing['p95_ms']:.4f} ms")
        print(f"EVM: {pytorch_evm:.2f}%")
        print(f"SNR: {pytorch_snr:.2f} dB")
        
        # Benchmark ONNX
        print("\n" + "="*60)
        print("ONNX RUNTIME")
        print("="*60)
        
        onnx_timing, onnx_output = benchmark_onnx(args.output, test_sample, n_runs=1000)
        
        if onnx_timing is not None:
            onnx_evm = compute_evm(onnx_output, test_target)
            onnx_snr = compute_snr(onnx_output, test_target)
            
            print(f"Inference time: {onnx_timing['mean_ms']:.4f} ± {onnx_timing['std_ms']:.4f} ms")
            print(f"  Median: {onnx_timing['median_ms']:.4f} ms")
            print(f"  Min: {onnx_timing['min_ms']:.4f} ms")
            print(f"  95th percentile: {onnx_timing['p95_ms']:.4f} ms")
            print(f"EVM: {onnx_evm:.2f}%")
            print(f"SNR: {onnx_snr:.2f} dB")
            
            # Comparison
            print("\n" + "="*60)
            print("COMPARISON")
            print("="*60)
            
            speedup = pytorch_timing['mean_ms'] / onnx_timing['mean_ms']
            evm_diff = onnx_evm - pytorch_evm
            snr_diff = pytorch_snr - onnx_snr
            
            print(f"Speedup: {speedup:.2f}x ({pytorch_timing['mean_ms']:.4f}ms → {onnx_timing['mean_ms']:.4f}ms)")
            print(f"EVM change: {evm_diff:+.2f}% ({pytorch_evm:.2f}% → {onnx_evm:.2f}%)")
            print(f"SNR change: {snr_diff:+.2f}dB ({pytorch_snr:.2f}dB → {onnx_snr:.2f}dB)")
            
            # Output difference
            max_diff = np.max(np.abs(pytorch_output - onnx_output))
            mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
            print(f"Output difference - Max: {max_diff:.6f}, Mean: {mean_diff:.6f}")
            
            # Real-time check
            sample_time = block_len / 2.0e6
            print(f"\nSignal duration: {sample_time * 1000:.4f} ms")
            print(f"PyTorch: {pytorch_timing['mean_ms']:.4f} ms (RT: {sample_time * 1000 / pytorch_timing['mean_ms']:.2f}x)")
            print(f"ONNX:    {onnx_timing['mean_ms']:.4f} ms (RT: {sample_time * 1000 / onnx_timing['mean_ms']:.2f}x)")
            
            if onnx_timing['mean_ms'] < sample_time * 1000:
                print("✓ ONNX model is REAL-TIME CAPABLE!")
            else:
                factor = onnx_timing['mean_ms'] / (sample_time * 1000)
                print(f"✗ ONNX still NOT real-time ({factor:.1f}x too slow)")
    
    print("\n" + "="*60)
    print("Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
