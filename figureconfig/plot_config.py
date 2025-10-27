# scripts/03_plotting.py
import sys
from pathlib import Path

# Get the project root directory
script_dir = Path(__file__).resolve().parent  # scripts/
project_root = script_dir.parent              # myproject/

# Add to Python's search path
sys.path.insert(0, str(project_root))
# config/plot_config.py
import matplotlib.pyplot as plt
import seaborn as sns

# Color palettes for different use cases
COLORS = {
    'first': '#2E86AB',
    'second': '#A23B72',
    'third': '#F18F01',
    'fourth': '#06A77D',
    'fifth': '#F77F00',
    'sixth': '#D62828'
}

CATEGORICAL_PALETTE = ["#3D09F8", '#A23B72', '#F18F01', '#06A77D', '#8338EC', '#FF006E']

def set_plot_style(style='whitegrid', context='notebook'):
    """
    Set consistent plotting style
    
    Args:
        style: seaborn style ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks')
        context: seaborn context ('paper', 'notebook', 'talk', 'poster')
    """
    sns.set_style(style)
    sns.set_context(context)
    
    # Custom rcParams
    plt.rcParams.update({
        'figure.figsize': (16, 9),
        'figure.dpi': 100,  # Screen display
        'savefig.dpi': 300,  # High-res saving
        'font.size': 18,
        'axes.labelsize': 20,
        'axes.titlesize': 24,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'legend.frameon': True,
        'legend.shadow': False,
        'legend.facecolor': '#FFFFFF',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.color': '#000000',
        'grid.alpha': 1.0,
        'grid.linewidth': 1.0,
        'grid.linestyle': '-',
    })
    
    # Set color palette
    sns.set_palette(CATEGORICAL_PALETTE)

def save_figure(fig, filename, formats=['png', 'pdf']):
    """
    Save figure in multiple formats with consistent settings
    
    Args:
        fig: matplotlib figure object
        filename: base filename without extension
        formats: list of formats to save
    """
    for fmt in formats:
        fig.savefig(
            f'figures/{filename}.{fmt}',
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
    print(f"✓ Saved {filename} as {', '.join(formats)}")

def get_color(name):
    """Get a color from the palette"""
    return COLORS.get(name, COLORS['first'])
