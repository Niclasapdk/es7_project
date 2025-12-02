import numpy as np
import matplotlib.pyplot as plt

# Arbitrary confusion matrix
cm = np.array([
    [52,  4,  3],
    [ 5, 61,  6],
    [ 2,  7, 49]
])

classes = ["Class A", "Class B", "Class C"]

fig, ax = plt.subplots(figsize=(4.5, 3.5))

im = ax.imshow(cm, cmap="Blues")

ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("Confusion matrix")

plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

cbar = fig.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Count", rotation=-90, va="center")

max_val = cm.max()
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        val = cm[i, j]
        ax.text(
            j, i, str(val),
            ha="center", va="center", fontsize=8,
            color="white" if val > max_val / 2 else "black"
        )

fig.tight_layout()
fig.savefig("confusion_matrix_3class.png", dpi=300)