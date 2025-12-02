import numpy as np
import matplotlib.pyplot as plt

# Input range
x = np.linspace(-6, 6, 400)

# Activations
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)
relu = np.maximum(0, x)

fig, ax = plt.subplots(figsize=(7, 3.5))

# Curves
ax.plot(x, relu,    label="ReLU",    color="red")
ax.plot(x, sigmoid, label="Sigmoid", color="green")
ax.plot(x, tanh,    label="Tanh",    color="blue")

# Axes limits
ax.set_xlim(-6, 6)
ax.set_ylim(-1.0, 1.0)

# Dashed axes through origin
ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
ax.axvline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)

# Labels / title
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output")
ax.set_title("Activation Functions")

# Legend
ax.legend(loc="lower right", frameon=False)

# No background grid (just the dashed axes)
ax.grid(False)

fig.tight_layout()
fig.savefig("activation_functions_style2.png", dpi=300)