import sys
from pathlib import Path

# Add the project root to Python path
script_dir = Path(__file__).resolve().parent  # figures/
project_root = script_dir.parent              # main_folder/
sys.path.insert(0, str(project_root))

# Import your config
from figureconfig.plot_config import set_plot_style, save_figure, get_color, COLORS
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

set_plot_style()

fig, ax = plt.subplots()
colors = [get_color('first'), get_color('second'), get_color('third'), get_color('fourth')]

points = np.array([-1, 1])

ax.scatter(points, np.zeros_like(points), s=200, c=colors[3], marker='o')
circle = plt.Circle((0, 0), 1, fill=False, color=colors[2], linestyle='--', linewidth=1.5, alpha=0.5)
ax.add_artist(circle)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
plt.xlabel('In-Phase (I)')
plt.ylabel('Quadrature (Q)')
plt.title('BPSK Constellation Diagram')
plt.xlim(-2, 2)
plt.ylim(-1, 1)

save_figure(fig, 'constellation_bpsk_fig', formats=['png'])
