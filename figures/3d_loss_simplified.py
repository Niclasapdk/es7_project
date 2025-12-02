import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plots

# Bowl with two dips: quadratic + two negative Gaussians
def loss(w1, w2):
    base = 0.6 * (w1**2 + w2**2)
    sigma2 = 0.18
    a1, a2 = -0.9, -0.9
    x1, y1 = -0.6, 0.2
    x2, y2 =  0.6, -0.2
    r1 = (w1 - x1)**2 + (w2 - y1)**2
    r2 = (w1 - x2)**2 + (w2 - y2)**2
    dip1 = a1 * np.exp(-r1 / sigma2)
    dip2 = a2 * np.exp(-r2 / sigma2)
    return base + dip1 + dip2

# Grid
w1 = np.linspace(-1.5, 1.5, 220)
w2 = np.linspace(-1.5, 1.5, 220)
W1, W2 = np.meshgrid(w1, w2)
L = loss(W1, W2)

# Gradient of the loss
def grad_loss(w):
    w1, w2 = w
    sigma2 = 0.18
    a1, a2 = -0.9, -0.9
    x1, y1 = -0.6, 0.2
    x2, y2 =  0.6, -0.2

    r1 = (w1 - x1)**2 + (w2 - y1)**2
    r2 = (w1 - x2)**2 + (w2 - y2)**2
    e1 = np.exp(-r1 / sigma2)
    e2 = np.exp(-r2 / sigma2)

    dl_dw1 = 1.2*w1 + a1 * e1 * (-2*(w1 - x1)/sigma2) + a2 * e2 * (-2*(w1 - x2)/sigma2)
    dl_dw2 = 1.2*w2 + a1 * e1 * (-2*(w2 - y1)/sigma2) + a2 * e2 * (-2*(w2 - y2)/sigma2)
    return np.array([dl_dw1, dl_dw2])

# Gradient-descent-like path
steps = 30
eta = 0.12
w_path = np.zeros((steps, 2))
w = np.array([-1.2, 1.0])   # starting point
for i in range(steps):
    w_path[i] = w
    w = w - eta * grad_loss(w)

L_path = loss(w_path[:,0], w_path[:,1])

# Plot
fig = plt.figure(figsize=(7, 4.5))
ax = fig.add_subplot(111, projection='3d')

# Gradient path
ax.plot(w_path[:,0], w_path[:,1], L_path,
        marker='o', markersize=2, linewidth=1.5, color='red', label='Gradient path')

# Surface
ax.plot_surface(W1, W2, L, linewidth=0, antialiased=True, alpha=0.7)

# Contours on base plane
min_L = L.min()
ax.contour(W1, W2, L, zdir='z', offset=min_L - 0.5, linewidths=0.7)

ax.set_xlabel("Weight 1")
ax.set_ylabel("Weight 2")
ax.set_zlabel("Loss")
ax.set_title("3D loss with two weights")

ax.set_zlim(min_L - 0.5, L.max())
ax.view_init(elev=30, azim=-55)
ax.legend(loc='upper left')

fig.tight_layout()
fig.savefig("loss_surface_3d_bowl_two_dips.png", dpi=300)