#!/usr/bin/env python3
"""
Inference script for Dilated ResNet BPSK Signal Separation
Load a trained model and test on new data.

Example usage:
    # Test on NPZ file
    python test_model.py --model bpsk_dilated_resnet_model.pth --data test_signals.npz
    
    # Test on single numpy array
    python test_model.py --model bpsk_dilated_resnet_model.pth --signal noisy_signal.npy --plot
    
    # Batch inference
    python test_model.py --model bpsk_dilated_resnet_model.pth --data test_signals.npz --output results.npz
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
import os


# =====================================
# Copy Model Architecture (must match training)
# =====================================
class DilatedResBlock(nn.Module):
    """Residual block with dilated convolution for large receptive field."""
    
    def __init__(self, channels, kernel_size=5, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, 
                               dilation=dilation, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                               dilation=dilation, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.alpha * residual)
        return out


class ChannelAttention(nn.Module):
    """Lightweight channel attention mechanism."""
    
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        
    def forward(self, x):
        b, c, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        attention = torch.sigmoid(avg_out + max_out).view(b, c, 1)
        return x * attention


class DilatedResNet(nn.Module):
    """Dilated ResNet for signal denoising."""
    
    def __init__(self, input_size=512, base_channels=32, num_blocks=6, dropout=0.1):
        super().__init__()
        
        self.input_conv = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU()
        )
        
        dilations = [1, 2, 4, 8, 16, 16]
        self.res_blocks = nn.ModuleList()
        for i, dilation in enumerate(dilations[:num_blocks]):
            self.res_blocks.append(
                DilatedResBlock(base_channels, kernel_size=5, dilation=dilation, dropout=dropout)
            )
        
        if num_blocks > 6:
            reverse_dilations = [8, 4, 2, 1]
            for dilation in reverse_dilations[:num_blocks-6]:
                self.res_blocks.append(
                    DilatedResBlock(base_channels, kernel_size=5, dilation=dilation, dropout=dropout)
                )
        
        self.attention = ChannelAttention(base_channels, reduction=4)
        
        self.refinement = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.output_conv = nn.Conv1d(base_channels, 1, kernel_size=7, padding=3)
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        identity = x.clone()
        x = x.unsqueeze(1)
        
        x = self.input_conv(x)
        skip = x.clone()
        
        for res_block in self.res_blocks:
            x = res_block(x)
        
        x = self.attention(x)
        x = x + skip
        x = self.refinement(x)
        
        noise_estimate = self.output_conv(x).squeeze(1)
        clean_estimate = identity - self.residual_weight * noise_estimate
        
        return clean_estimate, noise_estimate


# =====================================
# Metrics Functions
# =====================================
def complex_from_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Convert interleaved [Re, Im] to complex tensor."""
    if x.dim() == 1:
        x = x.unsqueeze(0)
    
    batch_size = x.shape[0]
    n_samples = x.shape[1] // 2
    x_reshaped = x.view(batch_size, n_samples, 2)
    return torch.complex(x_reshaped[..., 0], x_reshaped[..., 1])


def calculate_snr(clean: torch.Tensor, noisy: torch.Tensor) -> float:
    """Calculate SNR in dB."""
    clean_c = complex_from_interleaved(clean)
    noisy_c = complex_from_interleaved(noisy)
    
    signal_power = torch.mean(torch.abs(clean_c)**2)
    noise_power = torch.mean(torch.abs(noisy_c - clean_c)**2)
    
    if noise_power < 1e-10:
        return 40.0
    
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()


def calculate_evm(reference: torch.Tensor, measured: torch.Tensor) -> float:
    """Calculate Error Vector Magnitude (EVM) as percentage."""
    ref_c = complex_from_interleaved(reference)
    meas_c = complex_from_interleaved(measured)
    
    error_power = torch.mean(torch.abs(ref_c - meas_c)**2)
    ref_power = torch.mean(torch.abs(ref_c)**2)
    
    if ref_power < 1e-10:
        return 100.0
    
    evm = torch.sqrt(error_power / ref_power) * 100
    return evm.item()


# =====================================
# Model Loading
# =====================================
def load_model(checkpoint_path: str, device: torch.device, 
               base_channels: int = 32, num_blocks: int = 6) -> DilatedResNet:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        device: Device to load model on
        base_channels: Number of base channels (must match training)
        num_blocks: Number of residual blocks (must match training)
    
    Returns:
        Loaded model in eval mode
    """
    # Initialize model
    model = DilatedResNet(
        input_size=512,
        base_channels=base_channels,
        num_blocks=num_blocks,
        dropout=0.1  # Dropout is disabled in eval mode anyway
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  Trained for {checkpoint['epoch']+1} epochs")
    if 'val_loss' in checkpoint:
        print(f"  Validation loss: {checkpoint['val_loss']:.6f}")
    if 'model_info' in checkpoint:
        info = checkpoint['model_info']
        print(f"  Parameters: {info['total_parameters']:,}")
        print(f"  Receptive field: {info['receptive_field']} samples")
    
    return model


# =====================================
# Inference Functions
# =====================================
def denoise_signal(model: DilatedResNet, noisy_signal: np.ndarray, 
                   device: torch.device, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Denoise signal(s) using trained model.
    
    Args:
        model: Trained DilatedResNet model
        noisy_signal: Noisy signals, shape [N, 512] or [512]
        device: Device for inference
        batch_size: Batch size for processing multiple signals
    
    Returns:
        clean_signals: Denoised signals
        noise_estimates: Estimated noise/jamming
    """
    model.eval()
    
    # Handle single signal
    if noisy_signal.ndim == 1:
        noisy_signal = noisy_signal[np.newaxis, :]
    
    # Convert to torch
    noisy_tensor = torch.from_numpy(noisy_signal).float()
    
    all_clean = []
    all_noise = []
    
    # Process in batches
    with torch.no_grad():
        for i in range(0, len(noisy_tensor), batch_size):
            batch = noisy_tensor[i:i+batch_size].to(device)
            clean, noise = model(batch)
            all_clean.append(clean.cpu().numpy())
            all_noise.append(noise.cpu().numpy())
    
    clean_signals = np.concatenate(all_clean, axis=0)
    noise_estimates = np.concatenate(all_noise, axis=0)
    
    return clean_signals, noise_estimates


def evaluate_test_set(model: DilatedResNet, noisy_signals: np.ndarray,
                      clean_signals: Optional[np.ndarray], device: torch.device) -> Dict:
    """
    Evaluate model on test set with ground truth.
    
    Args:
        model: Trained model
        noisy_signals: Noisy test signals [N, 512]
        clean_signals: Clean reference signals [N, 512] (optional)
        device: Device for inference
    
    Returns:
        Dictionary of metrics
    """
    denoised, noise_est = denoise_signal(model, noisy_signals, device)
    
    results = {
        'num_samples': len(noisy_signals),
        'denoised_signals': denoised,
        'noise_estimates': noise_est
    }
    
    # If ground truth is available, calculate metrics
    if clean_signals is not None:
        snr_improvements = []
        evms_before = []
        evms_after = []
        
        noisy_tensor = torch.from_numpy(noisy_signals).float()
        clean_tensor = torch.from_numpy(clean_signals).float()
        denoised_tensor = torch.from_numpy(denoised).float()
        
        for i in range(len(noisy_signals)):
            snr_before = calculate_snr(clean_tensor[i], noisy_tensor[i])
            snr_after = calculate_snr(clean_tensor[i], denoised_tensor[i])
            snr_improvements.append(snr_after - snr_before)
            
            evm_before = calculate_evm(clean_tensor[i], noisy_tensor[i])
            evm_after = calculate_evm(clean_tensor[i], denoised_tensor[i])
            evms_before.append(evm_before)
            evms_after.append(evm_after)
        
        results.update({
            'snr_improvement_mean': np.mean(snr_improvements),
            'snr_improvement_std': np.std(snr_improvements),
            'snr_improvement_min': np.min(snr_improvements),
            'snr_improvement_max': np.max(snr_improvements),
            'evm_before_mean': np.mean(evms_before),
            'evm_after_mean': np.mean(evms_after),
            'evm_reduction': np.mean(evms_before) - np.mean(evms_after),
            'snr_improvements': snr_improvements,
            'evms_before': evms_before,
            'evms_after': evms_after
        })
    
    return results


# =====================================
# Visualization
# =====================================
def plot_inference_results(noisy: np.ndarray, denoised: np.ndarray, 
                          clean: Optional[np.ndarray] = None,
                          save_path: Optional[str] = None):
    """
    Plot inference results for visual inspection.
    
    Args:
        noisy: Noisy signal [512]
        denoised: Denoised signal [512]
        clean: Clean reference signal [512] (optional)
        save_path: Path to save figure
    """
    # Convert to complex
    def to_complex(x):
        x_reshaped = x.reshape(-1, 2)
        return x_reshaped[:, 0] + 1j * x_reshaped[:, 1]
    
    noisy_c = to_complex(noisy)
    denoised_c = to_complex(denoised)
    clean_c = to_complex(clean) if clean is not None else None
    
    # Create figure
    n_rows = 3 if clean is not None else 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4*n_rows))
    
    # Time domain - Real
    axes[0, 0].plot(noisy_c.real[:100], 'r-', alpha=0.7, label='Noisy', linewidth=1.5)
    axes[0, 0].plot(denoised_c.real[:100], 'b-', alpha=0.8, label='Denoised', linewidth=1.5)
    if clean_c is not None:
        axes[0, 0].plot(clean_c.real[:100], 'g--', alpha=0.6, label='Clean', linewidth=1.5)
    axes[0, 0].set_title('Real Part (first 100 samples)', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Time domain - Imaginary
    axes[0, 1].plot(noisy_c.imag[:100], 'r-', alpha=0.7, label='Noisy', linewidth=1.5)
    axes[0, 1].plot(denoised_c.imag[:100], 'b-', alpha=0.8, label='Denoised', linewidth=1.5)
    if clean_c is not None:
        axes[0, 1].plot(clean_c.imag[:100], 'g--', alpha=0.6, label='Clean', linewidth=1.5)
    axes[0, 1].set_title('Imaginary Part (first 100 samples)', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Constellation
    axes[1, 0].scatter(noisy_c.real[::2], noisy_c.imag[::2], 
                      alpha=0.3, s=5, c='red', label='Noisy')
    axes[1, 0].scatter(denoised_c.real[::2], denoised_c.imag[::2],
                      alpha=0.5, s=5, c='blue', label='Denoised')
    if clean_c is not None:
        axes[1, 0].scatter(clean_c.real[::2], clean_c.imag[::2],
                          alpha=0.2, s=3, c='green', label='Clean', marker='x')
    axes[1, 0].set_title('Constellation Diagram', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('I (In-phase)')
    axes[1, 0].set_ylabel('Q (Quadrature)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axis('equal')
    
    # Power Spectrum
    from numpy.fft import fftshift, fft
    
    def compute_spectrum(signal):
        spectrum = fftshift(fft(signal))
        power_db = 20 * np.log10(np.abs(spectrum) + 1e-10)
        return power_db
    
    freqs = np.linspace(-0.5, 0.5, len(noisy_c))
    axes[1, 1].plot(freqs, compute_spectrum(noisy_c), 'r-', alpha=0.6, label='Noisy', linewidth=1.5)
    axes[1, 1].plot(freqs, compute_spectrum(denoised_c), 'b-', alpha=0.8, label='Denoised', linewidth=1.5)
    if clean_c is not None:
        axes[1, 1].plot(freqs, compute_spectrum(clean_c), 'g--', alpha=0.6, label='Clean', linewidth=1.5)
    axes[1, 1].set_title('Power Spectrum', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Normalized Frequency')
    axes[1, 1].set_ylabel('Power (dB)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Metrics (if clean signal available)
    if clean_c is not None:
        noisy_t = torch.from_numpy(noisy).float()
        clean_t = torch.from_numpy(clean).float()
        denoised_t = torch.from_numpy(denoised).float()
        
        snr_before = calculate_snr(clean_t, noisy_t)
        snr_after = calculate_snr(clean_t, denoised_t)
        evm_before = calculate_evm(clean_t, noisy_t)
        evm_after = calculate_evm(clean_t, denoised_t)
        
        # Error magnitude
        error_before = noisy_c - clean_c
        error_after = denoised_c - clean_c
        axes[2, 0].plot(np.abs(error_before)[:100], 'r-', alpha=0.7, 
                       label='Error Before', linewidth=1.5)
        axes[2, 0].plot(np.abs(error_after)[:100], 'b-', alpha=0.7,
                       label='Error After', linewidth=1.5)
        axes[2, 0].set_title('Error Magnitude', fontsize=12, fontweight='bold')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Metrics text
        axes[2, 1].axis('off')
        metrics_text = f"""Performance Metrics:

SNR Before:  {snr_before:.2f} dB
SNR After:   {snr_after:.2f} dB
Improvement: {snr_after - snr_before:.2f} dB

EVM Before:  {evm_before:.2f}%
EVM After:   {evm_after:.2f}%
Reduction:   {evm_before - evm_after:.2f}%"""
        
        axes[2, 1].text(0.1, 0.5, metrics_text, fontsize=13, 
                       verticalalignment='center', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', 
                                alpha=0.8, edgecolor='navy', linewidth=2))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


# =====================================
# Main Testing Script
# =====================================
def main():
    parser = argparse.ArgumentParser(description='Test Dilated ResNet on new data')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--data', type=str, 
                       help='Path to test data (NPZ file with X_test and optionally Y_test)')
    parser.add_argument('--signal', type=str,
                       help='Path to single signal file (.npy)')
    parser.add_argument('--output', type=str,
                       help='Path to save denoised results (.npz)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--plot_index', type=int, default=0,
                       help='Index of sample to plot (default: 0)')
    parser.add_argument('--channels', type=int, default=32,
                       help='Base channels (must match training)')
    parser.add_argument('--blocks', type=int, default=6,
                       help='Number of blocks (must match training)')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model, device, args.channels, args.blocks)
    
    # Load test data
    if args.data:
        print(f"\nLoading test data from {args.data}")
        data = np.load(args.data)
        
        # Check for different key names
        if 'Xtr' in data:
            X_test = data['Xtr']
            Y_test = data.get('Ytr', None)
        elif 'Xte' in data:
            X_test = data['Xte']
            Y_test = data.get('Yte', None)
        elif 'X' in data:
            X_test = data['X']
            Y_test = data.get('Y', None)
        else:
            raise ValueError("NPZ file must contain 'X_test', 'Xte', or 'X' key")
        
        print(f"Test samples: {X_test.shape[0]}")
        
    elif args.signal:
        print(f"\nLoading signal from {args.signal}")
        X_test = np.load(args.signal)
        Y_test = None
        
        if X_test.ndim == 1:
            X_test = X_test[np.newaxis, :]
        print(f"Signal shape: {X_test.shape}")
        
    else:
        raise ValueError("Must provide either --data or --signal")
    
    # Run inference
    print("\nRunning inference...")
    results = evaluate_test_set(model, X_test, Y_test, device)
    
    # Print results
    print(f"\nProcessed {results['num_samples']} signals")
    
    if Y_test is not None:
        print("\n" + "="*50)
        print("PERFORMANCE METRICS")
        print("="*50)
        print(f"SNR Improvement: {results['snr_improvement_mean']:.2f} ± {results['snr_improvement_std']:.2f} dB")
        print(f"  Min: {results['snr_improvement_min']:.2f} dB")
        print(f"  Max: {results['snr_improvement_max']:.2f} dB")
        print(f"\nEVM Before: {results['evm_before_mean']:.2f}%")
        print(f"EVM After:  {results['evm_after_mean']:.2f}%")
        print(f"Reduction:  {results['evm_reduction']:.2f}%")
    
    # Save results
    if args.output:
        save_dict = {
            'denoised_signals': results['denoised_signals'],
            'noise_estimates': results['noise_estimates'],
            'noisy_signals': X_test
        }
        
        if Y_test is not None:
            save_dict['clean_signals'] = Y_test
            save_dict['snr_improvements'] = results['snr_improvements']
            save_dict['evms_before'] = results['evms_before']
            save_dict['evms_after'] = results['evms_after']
        
        np.savez(args.output, **save_dict)
        print(f"\nSaved results to {args.output}")
    
    # Visualization
    if args.plot:
        idx = min(args.plot_index, len(X_test) - 1)
        print(f"\nGenerating visualization for sample {idx}...")
        
        plot_path = f"inference_result_sample_{idx}.png"
        plot_inference_results(
            X_test[idx],
            results['denoised_signals'][idx],
            Y_test[idx] if Y_test is not None else None,
            save_path=plot_path
        )


if __name__ == "__main__":
    main()
