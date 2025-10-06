import torch
import torch.nn as nn
import torch.optim as torch_optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import json

class NoiseSuppressionCNN(nn.Module):
    def __init__(self, input_len=512):
        super().__init__()
        
        # Keep spatial dimension at 256 throughout
        self.initial = nn.Sequential(
            nn.Conv1d(2, 64, 5, padding=2),  # [batch, 64, 256]
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Multi-scale with same output spatial dim
        self.branch1 = nn.Sequential(  # Small kernel
            nn.Conv1d(64, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(  # Medium kernel  
            nn.Conv1d(64, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(  # Large kernel
            nn.Conv1d(64, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv1d(64 + 96, 128, 3, padding=1),  # 64 + (32×3) = 160
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 2, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Reshape input
        x = x.view(-1, 2, 256)  # [batch, 2, 256]
        
        # Initial features
        base = self.initial(x)  # [batch, 64, 256]
        
        # Multi-scale branches
        b1 = self.branch1(base)  # [batch, 32, 256]
        b2 = self.branch2(base)  # [batch, 32, 256]  
        b3 = self.branch3(base)  # [batch, 32, 256]
        
        # Combine
        combined = torch.cat([base, b1, b2, b3], dim=1)  # [batch, 64+96=160, 256]
        output = self.fusion(combined)  # [batch, 2, 256]
        
        return output.view(-1, 512)  # [batch, 512]
def calculate_snr(clean_signal, noise_component):
    """
    Calculate SNR in dB
    clean_signal: original clean signal [batch, 512]
    noise_component: noise/jammer component [batch, 512]
    """
    signal_power = torch.mean(clean_signal**2, dim=1)  # [batch]
    noise_power = torch.mean(noise_component**2, dim=1)  # [batch]
    
    # Avoid division by zero
    snr_linear = signal_power / (noise_power + 1e-12)
    snr_db = 10 * torch.log10(snr_linear + 1e-12)  # [batch]
    return snr_db

def load_and_prepare_data(npz_path, batch_size=32, val_split=0.1):
    """Load NPZ data and prepare PyTorch datasets"""
    data = np.load(npz_path)
    
    Xtr, Ytr, Xva, Yva = data['Xtr'], data['Ytr'], data['Xva'], data['Yva']
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(Xtr)
    Y_train = torch.FloatTensor(Ytr)
    X_val = torch.FloatTensor(Xva)
    Y_val = torch.FloatTensor(Yva)
    
    print(f"Training data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")
    
    # Create datasets
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    """Train the noise suppression model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch_optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    snr_improvements = []
    
    print("Starting training...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        batch_snr_improvements = []
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: estimate noise component
            noise_estimate = model(x_batch)
            
            # Reconstruct clean signal by subtracting noise estimate
            # x_batch = clean + noise, so clean_estimate = x_batch - noise_estimate
            clean_estimate = x_batch - noise_estimate
            
            # Calculate SNR improvement
            original_noise = x_batch - y_batch  # Actual noise in input
            estimated_noise = noise_estimate    # Our model's noise estimate
            
            original_snr = calculate_snr(y_batch, original_noise)
            improved_snr = calculate_snr(y_batch, x_batch - clean_estimate)
            snr_improvement = improved_snr - original_snr
            
            batch_snr_improvements.extend(snr_improvement.cpu().detach().numpy())
            
            # Loss: how well we estimate the actual noise component
            loss = criterion(noise_estimate, original_noise)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_snr_improvements = []
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                noise_estimate = model(x_batch)
                clean_estimate = x_batch - noise_estimate
                
                original_noise = x_batch - y_batch
                loss = criterion(noise_estimate, original_noise)
                val_loss += loss.item()
                
                # Calculate SNR improvement for validation
                original_snr = calculate_snr(y_batch, original_noise)
                improved_snr = calculate_snr(y_batch, x_batch - clean_estimate)
                snr_improvement = improved_snr - original_snr
                val_snr_improvements.extend(snr_improvement.cpu().numpy())
        
        # Average losses and SNR improvements
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        avg_snr_improvement = np.mean(batch_snr_improvements)
        avg_val_snr_improvement = np.mean(val_snr_improvements)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        snr_improvements.append(avg_snr_improvement)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'  Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            print(f'  Train SNR Improvement: {avg_snr_improvement:.2f} dB')
            print(f'  Val SNR Improvement: {avg_val_snr_improvement:.2f} dB')
    
    return train_losses, val_losses, snr_improvements

def evaluate_model(model, val_loader, num_examples=3):
    """Evaluate the model and plot some examples"""
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(val_loader):
            if batch_idx >= num_examples:
                break
                
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            for i in range(min(2, x_batch.size(0))):  # Plot first 2 examples
                x_single = x_batch[i:i+1]
                y_single = y_batch[i:i+1]
                
                noise_estimate = model(x_single)
                clean_estimate = x_single - noise_estimate
                
                # Convert to complex for plotting
                x_complex = x_single[0].cpu().numpy().reshape(-1, 2) @ [1, 1j]
                y_complex = y_single[0].cpu().numpy().reshape(-1, 2) @ [1, 1j]
                clean_est_complex = clean_estimate[0].cpu().numpy().reshape(-1, 2) @ [1, 1j]
                
                # Calculate metrics
                original_noise = x_single - y_single
                original_snr = calculate_snr(y_single, original_noise).item()
                improved_snr = calculate_snr(y_single, x_single - clean_estimate).item()
                
                # Plot
                plt.figure(figsize=(15, 4))
                
                plt.subplot(1, 3, 1)
                plt.plot(np.real(x_complex), label='Jammed (Real)', alpha=0.7)
                plt.plot(np.real(y_complex), label='Clean (Real)', alpha=0.7)
                plt.title(f'Input vs Target\nOriginal SNR: {original_snr:.2f} dB')
                plt.legend()
                
                plt.subplot(1, 3, 2)
                plt.plot(np.real(clean_est_complex), label='Denoised (Real)', alpha=0.7)
                plt.plot(np.real(y_complex), label='Clean (Real)', alpha=0.7)
                plt.title(f'Denoised vs Target\nImproved SNR: {improved_snr:.2f} dB')
                plt.legend()
                
                plt.subplot(1, 3, 3)
                snr_improvement = improved_snr - original_snr
                plt.bar(['Original', 'Improved'], [original_snr, improved_snr])
                plt.title(f'SNR Improvement: {snr_improvement:.2f} dB')
                plt.ylabel('SNR (dB)')
                
                plt.tight_layout()
                plt.show()

def main():
    # Configuration
    npz_path = "artifacts/gnss_synth_sweepcw_20k.npz"  # Update this path
    batch_size = 32
    epochs = 50
    
    # Load data
    print("Loading data...")
    train_loader, val_loader = load_and_prepare_data(npz_path, batch_size)
    
    # Create model
    model = NoiseSuppressionCNN(input_len=512)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    train_losses, val_losses, snr_improvements = train_model(
        model, train_loader, val_loader, epochs=epochs
    )
    
if __name__ == "__main__":
    main()
