import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

fig, ax = plt.subplots(figsize=(4.5, 2.2))

# x-positions of layers
x_input  = 0.15
x_h1     = 0.40
x_h2     = 0.65
x_output = 0.90

# y-positions
y_input  = [0.25, 0.75]
y_h1     = np.linspace(0.2, 0.8, 4)
y_h2     = np.linspace(0.2, 0.8, 4)
y_output = [0.5]

radius = 0.035  # slightly smaller to avoid crowding

def draw_layer(x, ys, dropped_idx=None):
    nodes = []
    for i, y in enumerate(ys):
        circ = Circle((x, y), radius=radius, facecolor="white",
                      edgecolor="black", linewidth=1.0)
        ax.add_patch(circ)
        # X for dropped neurons
        if dropped_idx is not None and i in dropped_idx:
            s = radius * 0.9
            ax.plot([x-s, x+s], [y-s, y+s], color="black", linewidth=0.9)
            ax.plot([x-s, x+s], [y+s, y-s], color="black", linewidth=0.9)
            dropped = True
        else:
            dropped = False
        nodes.append((x, y, dropped))
    return nodes

inp = draw_layer(x_input,  y_input)
h1  = draw_layer(x_h1,    y_h1, dropped_idx=[1, 3])
h2  = draw_layer(x_h2,    y_h2, dropped_idx=[2])
out = draw_layer(x_output, y_output)

def connect_layers(layer_from, layer_to):
    for (x1, y1, d1) in layer_from:
        for (x2, y2, d2) in layer_to:
            alpha = 0.5
            if d1 or d2:
                alpha = 0.1  # fade connections from/to dropped nodes
            ax.plot([x1, x2], [y1, y2],
                    color="black", linewidth=0.6, alpha=alpha)

connect_layers(inp, h1)
connect_layers(h1, h2)
connect_layers(h2, out)

# labels
ax.text(x_input,  0.08, "Input\nlayer",     ha="center", va="top", fontsize=7)
ax.text(x_h1,     0.08, "Hidden\nlayer 1",  ha="center", va="top", fontsize=7)
ax.text(x_h2,     0.08, "Hidden\nlayer 2",  ha="center", va="top", fontsize=7)
ax.text(x_output, 0.08, "Output\nlayer",    ha="center", va="top", fontsize=7)

ax.set_xlim(0.05, 0.98)
ax.set_ylim(0.05, 0.95)
ax.axis("off")

# ensure circles are actually round, not squished
ax.set_aspect("equal", adjustable="box")

fig.tight_layout(pad=0.3)

png_path = "dropout_single_net_v2.png"
fig.savefig(png_path, dpi=300)