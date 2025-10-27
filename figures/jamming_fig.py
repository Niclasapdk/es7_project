import sys
from pathlib import Path

# Add the project root to Python path
script_dir = Path(__file__).resolve().parent  # figures/
project_root = script_dir.parent              # main_folder/
sys.path.insert(0, str(project_root))

# Import your config
from figureconfig.plot_config import set_plot_style, save_figure, get_color, COLORS

import matplotlib.pyplot as plt
import numpy as np

# Set the plotting style at the start
set_plot_style(style='whitegrid', context='paper')

# Define the frequency bands
bands = {
    'Jamming': {'start': 1175.45, 'end': 1177.45, 'center': 1176.45},
    'L1': {'start': 1166.22, 'end': 1186.68, 'center': 1176.45},
    'L2': {'start': 1217.37, 'end': 1237.83, 'center': 1227.60},
}

# Sample PSD values for each band (you can replace these with your actual data)
psd_values = {
    'L1': 21.5,  # dBm/Hz or your unit
    'L2': 21.5,
    'Jamming': 35 
}

# Create the plot
fig, ax = plt.subplots(figsize=(16, 9))

# Plot each band as a bar
colors = [get_color('danger'), get_color('secondary'), get_color('accent'), get_color('primary')]
for idx, (band_name, band_info) in enumerate(bands.items()):
    psd = psd_values[band_name]
    width = band_info['end'] - band_info['start']
    center = band_info['center']
    height = psd_values[band_name]
    
    # Draw the bar
    ax.bar(center, height, width=width, 
           color=colors[idx], alpha=0.7, 
           edgecolor='black', linewidth=1.5,
           label=f'{band_name} ({band_info["start"]:.2f}-{band_info["end"]:.2f} MHz), {(-180 + psd)} dBW', bottom = -180)
# For text positioning, add the bottom offset to your calculations
bottom_offset = -180

ax.text(bands['L1']['center'], bottom_offset + psd_values['L1']/2, 'L1',
        ha='center', va='center', fontweight='bold', color='white')

ax.text(bands['L2']['center'], bottom_offset + psd_values['L2']/2, 'L2',
        ha='center', va='center', fontweight='bold', color='white')

ax.set_xlabel('Frequency (MHz)')
ax.set_ylabel('PSD (dBW/Hz)')
ax.set_title('GPS Frequency Bands Power Spectral Density')
ax.set_xticks(np.arange(1160, 1250, 20))  # Ticks every 50 MHz
ax.set_ylim(-180, -130)  # Adjust based on your data

# Add grid
ax.grid(True, alpha=0.3, linestyle='--')

# Legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(loc='upper right', framealpha=0.7)

save_figure(fig, 'jamming_gps_bands', formats=['png'])


plt.tight_layout()
plt.show()