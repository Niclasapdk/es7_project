#!/usr/bin/env python3
"""
Real-time Inference script for Dilated ResNet BPSK Signal Separation
Supports both batch processing and real-time streaming visualization.

Example usage:
    # Real-time streaming from stdin (pipe data)
    python test_model_streaming.py --model model.pth --stream --source stdin
    
    # Real-time streaming from file (simulated streaming)
    python test_model_streaming.py --model model.pth --stream --source file --data signals.npz --fps 30
    
    # Real-time streaming with UDP socket
    python test_model_streaming.py --model model.pth --stream --source udp --port 5000
    
    # Original batch mode
    python test_model_streaming.py --model model.pth --data test_signals.npz --plot
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple, Dict, Optional, Iterator
import os
import sys
import time
from collections import deque
import struct
import socket
import select


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
    """Load a trained model from checkpoint."""
    model = DilatedResNet(
        input_size=512,
        base_channels=base_channels,
        num_blocks=num_blocks,
        dropout=0.1
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  Trained for {checkpoint['epoch']+1} epochs")
    if 'val_loss' in checkpoint:
        print(f"  Validation loss: {checkpoint['val_loss']:.6f}")
    
    return model


# =====================================
# Streaming Data Sources
# =====================================
class StreamSource:
    """Base class for streaming data sources."""
    
    def __iter__(self) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Yield signal samples as (noisy, clean) tuples. Clean can be None."""
        raise NotImplementedError
    
    def close(self):
        """Clean up resources."""
        pass


class StdinStreamSource(StreamSource):
    """Stream signals from stdin (binary format)."""
    
    def __init__(self, signal_length=512):
        self.signal_length = signal_length
        self.bytes_per_signal = signal_length * 4  # float32
        
    def __iter__(self):
        print("Reading from stdin (expecting float32 binary data)...", file=sys.stderr)
        while True:
            data = sys.stdin.buffer.read(self.bytes_per_signal)
            if len(data) < self.bytes_per_signal:
                break
            signal = np.frombuffer(data, dtype=np.float32)
            yield signal, None  # No clean signal available from stdin


class FileStreamSource(StreamSource):
    """Simulate streaming from a file."""
    
    def __init__(self, filepath: str, fps: float = 30):
        data = np.load(filepath)
        
        if 'Xtr' in data:
            self.signals = data['Xtr']
            self.clean_signals = data.get('Ytr', None)
        elif 'Xte' in data:
            self.signals = data['Xte']
            self.clean_signals = data.get('Yte', None)
        elif 'X' in data:
            self.signals = data['X']
            self.clean_signals = data.get('Y', None)
        else:
            raise ValueError("NPZ file must contain 'X_test', 'Xte', or 'X' key")
        
        self.fps = fps
        self.delay = 1.0 / fps
        self.idx = 0
        
        if self.clean_signals is not None:
            print(f"Loaded clean reference signals for comparison", file=sys.stderr)
        else:
            print(f"No clean reference signals found in file", file=sys.stderr)
        
    def __iter__(self):
        print(f"Streaming {len(self.signals)} signals at {self.fps} FPS", file=sys.stderr)
        while self.idx < len(self.signals):
            clean = self.clean_signals[self.idx] if self.clean_signals is not None else None
            yield self.signals[self.idx], clean
            self.idx += 1
            time.sleep(self.delay)


class UDPStreamSource(StreamSource):
    """Stream signals via UDP socket."""
    
    def __init__(self, port: int = 5000, signal_length: int = 512, timeout: float = 1.0):
        self.port = port
        self.signal_length = signal_length
        self.timeout = timeout
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', port))
        self.sock.setblocking(False)
        print(f"Listening for UDP packets on port {port}...", file=sys.stderr)
        
    def __iter__(self):
        while True:
            ready = select.select([self.sock], [], [], self.timeout)
            if ready[0]:
                data, addr = self.sock.recvfrom(self.signal_length * 4 + 1024)
                if len(data) >= self.signal_length * 4:
                    signal = np.frombuffer(data[:self.signal_length * 4], dtype=np.float32)
                    yield signal, None  # No clean signal available from UDP
    
    def close(self):
        self.sock.close()


# =====================================
# Real-time Visualization
# =====================================
class RealtimeVisualizer:
    """Real-time matplotlib visualization for signal denoising."""
    
    def __init__(self, history_length: int = 100, show_metrics: bool = True):
        self.history_length = history_length
        self.show_metrics = show_metrics
        
        # Data buffers
        self.noisy_buffer = deque(maxlen=history_length)
        self.denoised_buffer = deque(maxlen=history_length)
        self.snr_history = deque(maxlen=50)
        self.latency_history = deque(maxlen=50)
        
        # Setup figure - simplified to only show time domain
        self.fig = plt.figure(figsize=(12, 4))
        gs = self.fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
        
        # Time domain plots
        self.ax_time_real = self.fig.add_subplot(gs[0, 0])
        self.ax_time_imag = self.fig.add_subplot(gs[0, 1])
        
        # Constellation
        # self.ax_constellation = self.fig.add_subplot(gs[0, 2])
        
        # Spectrum
        # self.ax_spectrum = self.fig.add_subplot(gs[1, :2])
        
        # Metrics
        # self.ax_metrics = self.fig.add_subplot(gs[1, 2])
        # self.ax_metrics.axis('off')
        
        # History plots
        # self.ax_snr_history = self.fig.add_subplot(gs[2, :2])
        # self.ax_latency = self.fig.add_subplot(gs[2, 2])
        
        # Initialize lines
        self.lines = {}
        self.scatters = {}
        self.texts = {}
        
        self._setup_plots()
        
        # Statistics
        self.frame_count = 0
        self.start_time = time.time()
        
    def _setup_plots(self):
        """Initialize plot elements."""
        # Time domain - Real
        self.lines['noisy_real'], = self.ax_time_real.plot([], [], 'r-', alpha=0.6, label='Noisy', lw=1.5)
        self.lines['denoised_real'], = self.ax_time_real.plot([], [], 'b-', alpha=0.8, label='Denoised', lw=1.5)
        self.lines['clean_real'], = self.ax_time_real.plot([], [], 'g--', alpha=0.6, label='Clean', lw=1.5)
        self.ax_time_real.set_title('Real Part', fontweight='bold')
        self.ax_time_real.legend(loc='upper right')
        self.ax_time_real.grid(True, alpha=0.3)
        self.ax_time_real.set_ylim(-2, 2)
        
        # Time domain - Imaginary
        self.lines['noisy_imag'], = self.ax_time_imag.plot([], [], 'r-', alpha=0.6, label='Noisy', lw=1.5)
        self.lines['denoised_imag'], = self.ax_time_imag.plot([], [], 'b-', alpha=0.8, label='Denoised', lw=1.5)
        self.lines['clean_imag'], = self.ax_time_imag.plot([], [], 'g--', alpha=0.6, label='Clean', lw=1.5)
        self.ax_time_imag.set_title('Imaginary Part', fontweight='bold')
        self.ax_time_imag.legend(loc='upper right')
        self.ax_time_imag.grid(True, alpha=0.3)
        self.ax_time_imag.set_ylim(-2, 2)
        
        # # Constellation
        # self.scatters['noisy'] = self.ax_constellation.scatter([], [], alpha=0.3, s=10, c='red', label='Noisy')
        # self.scatters['denoised'] = self.ax_constellation.scatter([], [], alpha=0.6, s=10, c='blue', label='Denoised')
        # self.scatters['clean'] = self.ax_constellation.scatter([], [], alpha=0.4, s=8, c='green', marker='x', label='Clean')
        # self.ax_constellation.set_title('Constellation Diagram', fontweight='bold')
        # self.ax_constellation.set_xlabel('I (In-phase)')
        # self.ax_constellation.set_ylabel('Q (Quadrature)')
        # self.ax_constellation.legend(loc='upper right')
        # self.ax_constellation.grid(True, alpha=0.3)
        # self.ax_constellation.set_xlim(-2, 2)
        # self.ax_constellation.set_ylim(-2, 2)
        # self.ax_constellation.set_aspect('equal')
        
        # # Spectrum
        # self.lines['spectrum_noisy'], = self.ax_spectrum.plot([], [], 'r-', alpha=0.6, label='Noisy', lw=1.5)
        # self.lines['spectrum_denoised'], = self.ax_spectrum.plot([], [], 'b-', alpha=0.8, label='Denoised', lw=1.5)
        # self.lines['spectrum_clean'], = self.ax_spectrum.plot([], [], 'g--', alpha=0.6, label='Clean', lw=1.5)
        # self.ax_spectrum.set_title('Power Spectrum', fontweight='bold')
        # self.ax_spectrum.set_xlabel('Normalized Frequency')
        # self.ax_spectrum.set_ylabel('Power (dB)')
        # self.ax_spectrum.legend(loc='upper right')
        # self.ax_spectrum.grid(True, alpha=0.3)
        # self.ax_spectrum.set_ylim(-60, 20)
        
        # # SNR History
        # self.lines['snr_history'], = self.ax_snr_history.plot([], [], 'g-', lw=2)
        # self.ax_snr_history.set_title('SNR History', fontweight='bold')
        # self.ax_snr_history.set_xlabel('Frame')
        # self.ax_snr_history.set_ylabel('SNR (dB)')
        # self.ax_snr_history.grid(True, alpha=0.3)
        
        # # Latency
        # self.lines['latency'], = self.ax_latency.plot([], [], 'purple', lw=2)
        # self.ax_latency.set_title('Processing Latency', fontweight='bold')
        # self.ax_latency.set_xlabel('Frame')
        # self.ax_latency.set_ylabel('Time (ms)')
        # self.ax_latency.grid(True, alpha=0.3)
        
        # # Metrics text
        # self.texts['metrics'] = self.ax_metrics.text(
        #     0.1, 0.5, '', transform=self.ax_metrics.transAxes,
        #     fontsize=11, verticalalignment='center', family='monospace',
        #     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        # )
        
    def update(self, noisy: np.ndarray, denoised: np.ndarray, 
               latency_ms: float, clean: Optional[np.ndarray] = None):
        """Update visualization with new data."""
        
        # Convert to complex
        def to_complex(x):
            x_reshaped = x.reshape(-1, 2)
            return x_reshaped[:, 0] + 1j * x_reshaped[:, 1]
        
        noisy_c = to_complex(noisy)
        denoised_c = to_complex(denoised)
        clean_c = to_complex(clean) if clean is not None else None
        
        # Update buffers
        self.noisy_buffer.append(noisy_c)
        self.denoised_buffer.append(denoised_c)
        self.latency_history.append(latency_ms)
        
        # Calculate SNR if clean signal is available
        snr = None
        if clean is not None:
            clean_tensor = torch.from_numpy(clean).float()
            denoised_tensor = torch.from_numpy(denoised).float()
            snr = calculate_snr(clean_tensor, denoised_tensor)
            self.snr_history.append(snr)
        
        # Time domain (last 100 samples of current signal)
        display_len = min(100, len(noisy_c))
        self.lines['noisy_real'].set_data(range(display_len), noisy_c.real[:display_len])
        self.lines['denoised_real'].set_data(range(display_len), denoised_c.real[:display_len])
        self.lines['noisy_imag'].set_data(range(display_len), noisy_c.imag[:display_len])
        self.lines['denoised_imag'].set_data(range(display_len), denoised_c.imag[:display_len])
        
        if clean_c is not None:
            self.lines['clean_real'].set_data(range(display_len), clean_c.real[:display_len])
            self.lines['clean_imag'].set_data(range(display_len), clean_c.imag[:display_len])
        else:
            self.lines['clean_real'].set_data([], [])
            self.lines['clean_imag'].set_data([], [])
        
        self.ax_time_real.set_xlim(0, display_len)
        self.ax_time_imag.set_xlim(0, display_len)
        
        # # Constellation (subsample for performance)
        # self.scatters['noisy'].set_offsets(np.c_[noisy_c.real[::2], noisy_c.imag[::2]])
        # self.scatters['denoised'].set_offsets(np.c_[denoised_c.real[::2], denoised_c.imag[::2]])
        
        # if clean_c is not None:
        #     self.scatters['clean'].set_offsets(np.c_[clean_c.real[::2], clean_c.imag[::2]])
        # else:
        #     self.scatters['clean'].set_offsets(np.empty((0, 2)))
        
        # # Spectrum
        # from numpy.fft import fftshift, fft
        
        # def compute_spectrum(signal):
        #     spectrum = fftshift(fft(signal))
        #     power_db = 20 * np.log10(np.abs(spectrum) + 1e-10)
        #     return power_db
        
        # freqs = np.linspace(-0.5, 0.5, len(noisy_c))
        # self.lines['spectrum_noisy'].set_data(freqs, compute_spectrum(noisy_c))
        # self.lines['spectrum_denoised'].set_data(freqs, compute_spectrum(denoised_c))
        
        # if clean_c is not None:
        #     self.lines['spectrum_clean'].set_data(freqs, compute_spectrum(clean_c))
        # else:
        #     self.lines['spectrum_clean'].set_data([], [])
        
        # self.ax_spectrum.set_xlim(-0.5, 0.5)
        
        # # SNR History
        # if len(self.snr_history) > 0:
        #     self.lines['snr_history'].set_data(range(len(self.snr_history)), list(self.snr_history))
        #     self.ax_snr_history.set_xlim(0, max(50, len(self.snr_history)))
        #     self.ax_snr_history.set_ylim(min(self.snr_history) - 5, max(self.snr_history) + 5)
        
        # # Latency
        # self.lines['latency'].set_data(range(len(self.latency_history)), list(self.latency_history))
        # self.ax_latency.set_xlim(0, max(50, len(self.latency_history)))
        # if len(self.latency_history) > 0:
        #     self.ax_latency.set_ylim(0, max(self.latency_history) * 1.2)
        
        # # Metrics text
        # self.frame_count += 1
        # elapsed = time.time() - self.start_time
        # fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        # avg_latency = np.mean(self.latency_history) if len(self.latency_history) > 0 else 0
        
        # metrics_text = f"Frame: {self.frame_count}\n"
        # metrics_text += f"FPS: {fps:.1f}\n"
        # metrics_text += f"Latency: {latency_ms:.2f} ms\n"
        # metrics_text += f"Avg Latency: {avg_latency:.2f} ms\n"
        
        # if snr is not None:
        #     metrics_text += f"\nCurrent SNR: {snr:.2f} dB\n"
        #     if len(self.snr_history) > 1:
        #         metrics_text += f"Avg SNR: {np.mean(self.snr_history):.2f} dB"
        
        # self.texts['metrics'].set_text(metrics_text)
        
        # Update frame count for statistics
        self.frame_count += 1
        
        return list(self.lines.values()) + list(self.scatters.values()) + list(self.texts.values())


# =====================================
# Streaming Processor
# =====================================
class StreamingProcessor:
    """Process streaming signals with real-time denoising."""
    
    def __init__(self, model: DilatedResNet, device: torch.device, 
                 visualize: bool = True, save_output: bool = False, 
                 output_path: Optional[str] = None):
        self.model = model
        self.device = device
        self.visualize = visualize
        self.save_output = save_output
        self.output_path = output_path
        
        if visualize:
            self.visualizer = RealtimeVisualizer()
        
        if save_output and output_path:
            self.output_buffer = []
        
    def process_stream(self, stream_source: StreamSource):
        """Process signals from stream source."""
        print("Starting real-time processing...")
        print("Press Ctrl+C to stop")
        
        try:
            for signal, clean in stream_source:
                t_start = time.time()
                
                # Denoise
                signal_tensor = torch.from_numpy(signal).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    denoised, _ = self.model(signal_tensor)
                denoised_np = denoised.cpu().numpy()[0]
                
                latency_ms = (time.time() - t_start) * 1000
                
                # Update visualization
                if self.visualize:
                    self.visualizer.update(signal, denoised_np, latency_ms, clean)
                    plt.pause(0.001)
                
                # Save if requested
                if self.save_output:
                    save_dict = {
                        'noisy': signal,
                        'denoised': denoised_np,
                        'latency_ms': latency_ms
                    }
                    if clean is not None:
                        save_dict['clean'] = clean
                    self.output_buffer.append(save_dict)
                
                # Print stats periodically
                if self.visualizer.frame_count % 30 == 0:
                    print(f"Frame {self.visualizer.frame_count}: "
                          f"{latency_ms:.2f} ms latency", file=sys.stderr)
                    
        except KeyboardInterrupt:
            print("\nStopping stream...", file=sys.stderr)
        finally:
            stream_source.close()
            
            if self.save_output and self.output_path:
                self._save_results()
    
    def _save_results(self):
        """Save buffered results."""
        if not self.output_buffer:
            return
        
        noisy_signals = np.array([b['noisy'] for b in self.output_buffer])
        denoised_signals = np.array([b['denoised'] for b in self.output_buffer])
        latencies = np.array([b['latency_ms'] for b in self.output_buffer])
        
        save_dict = {
            'noisy_signals': noisy_signals,
            'denoised_signals': denoised_signals,
            'latencies': latencies
        }
        
        # Include clean signals if they were available
        if 'clean' in self.output_buffer[0]:
            clean_signals = np.array([b['clean'] for b in self.output_buffer])
            save_dict['clean_signals'] = clean_signals
        
        np.savez(self.output_path, **save_dict)
        
        print(f"\nSaved {len(self.output_buffer)} processed signals to {self.output_path}")


# =====================================
# Original Batch Processing Functions
# =====================================
def denoise_signal(model: DilatedResNet, noisy_signal: np.ndarray, 
                   device: torch.device, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """Denoise signal(s) using trained model (batch mode)."""
    model.eval()
    
    if noisy_signal.ndim == 1:
        noisy_signal = noisy_signal[np.newaxis, :]
    
    noisy_tensor = torch.from_numpy(noisy_signal).float()
    
    all_clean = []
    all_noise = []
    
    with torch.no_grad():
        for i in range(0, len(noisy_tensor), batch_size):
            batch = noisy_tensor[i:i+batch_size].to(device)
            clean, noise = model(batch)
            all_clean.append(clean.cpu().numpy())
            all_noise.append(noise.cpu().numpy())
    
    clean_signals = np.concatenate(all_clean, axis=0)
    noise_estimates = np.concatenate(all_noise, axis=0)
    
    return clean_signals, noise_estimates


def plot_inference_results(noisy: np.ndarray, denoised: np.ndarray, 
                          clean: Optional[np.ndarray] = None,
                          save_path: Optional[str] = None):
    """
    Plot inference results for visual inspection (batch mode).
    
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
# Main Script
# =====================================
def main():
    parser = argparse.ArgumentParser(description='Real-time and Batch Signal Denoising')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--channels', type=int, default=32,
                       help='Base channels (must match training)')
    parser.add_argument('--blocks', type=int, default=6,
                       help='Number of blocks (must match training)')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Streaming arguments
    parser.add_argument('--stream', action='store_true',
                       help='Enable real-time streaming mode')
    parser.add_argument('--source', type=str, choices=['stdin', 'file', 'udp'],
                       default='file', help='Stream source type')
    parser.add_argument('--port', type=int, default=5000,
                       help='UDP port for network streaming')
    parser.add_argument('--fps', type=float, default=30,
                       help='Frames per second for file streaming')
    
    # Batch mode arguments
    parser.add_argument('--data', type=str,
                       help='Path to test data (NPZ file)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots (batch mode)')
    
    # Output arguments
    parser.add_argument('--output', type=str,
                       help='Path to save results')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization in streaming mode')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model, device, args.channels, args.blocks)
    
    # Streaming mode
    if args.stream:
        print("\n" + "="*50)
        print("REAL-TIME STREAMING MODE")
        print("="*50)
        
        # Create stream source
        if args.source == 'stdin':
            stream_source = StdinStreamSource()
        elif args.source == 'file':
            if not args.data:
                raise ValueError("--data required for file streaming")
            stream_source = FileStreamSource(args.data, fps=args.fps)
        elif args.source == 'udp':
            stream_source = UDPStreamSource(port=args.port)
        
        # Create processor
        processor = StreamingProcessor(
            model, device,
            visualize=not args.no_viz,
            save_output=args.output is not None,
            output_path=args.output
        )
        
        # Run streaming
        processor.process_stream(stream_source)
        
        if not args.no_viz:
            plt.show()
    
    # Batch mode (original functionality)
    else:
        print("\n" + "="*50)
        print("BATCH PROCESSING MODE")
        print("="*50)
        
        if not args.data:
            raise ValueError("--data required for batch mode")
        
        # Load and process
        print(f"\nLoading test data from {args.data}")
        data = np.load(args.data)
        
        if 'Xtr' in data:
            X_test = data['Xtr']
        elif 'Xte' in data:
            X_test = data['Xte']
        elif 'X' in data:
            X_test = data['X']
        else:
            raise ValueError("NPZ file must contain 'X_test', 'Xte', or 'X' key")
        
        print(f"Test samples: {X_test.shape[0]}")
        
        # Run inference
        print("\nRunning batch inference...")
        denoised, noise_est = denoise_signal(model, X_test, device)
        
        # Save results
        if args.output:
            np.savez(args.output, 
                    noisy_signals=X_test,
                    denoised_signals=denoised,
                    noise_estimates=noise_est)
            print(f"\nSaved results to {args.output}")
        
        # Plot if requested
        if args.plot:
            print("\nGenerating visualization...")
            plot_inference_results(X_test[0], denoised[0], 
                                 save_path="batch_result.png")


if __name__ == "__main__":
    main()
