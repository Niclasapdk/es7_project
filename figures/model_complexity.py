import numpy as np
import matplotlib.pyplot as plt

# Model capacity axis
x = np.linspace(0, 1.0, 200)

# Synthetic curves
bias2 = 0.4 * (1 - x)**2 + 0.02
variance = 0.4 * x**2 + 0.02
total_error = bias2 + variance + 0.02  # shifted up a bit

fig, ax = plt.subplots(figsize=(6.5, 3.8))

ax.plot(x, bias2, label=r"Bias$^2$",   color="tab:blue",  linewidth=2.2)
ax.plot(x, variance, label="Variance",  color="tab:green", linewidth=2.2)
ax.plot(x, total_error, label="Total error", color="black", linewidth=2.2)

ax.set_ylim(0, 1.0)
ax.set_xlim(0, 1.0)

ax.set_xlabel("Model capacity (complexity)")
ax.set_ylabel("Error")
ax.set_title("Bias--variance trade-off and total error")

# Mark approximate optimal capacity
opt_x = 0.5
ax.axvline(opt_x, color="black", linestyle="--", linewidth=1.0)
ax.text(opt_x + 0.01, ax.get_ylim()[1]*0.6, "Optimal\ncapacity",
        fontsize=8, va="center", ha="left")

ax.legend(loc="upper right", frameon=False)
ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.6)

fig.tight_layout()
fig.savefig("bias_variance_tradeoff.png", dpi=300)