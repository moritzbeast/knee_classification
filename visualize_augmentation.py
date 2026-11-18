import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from dataset import load_xray

torch.manual_seed(0)

xray = load_xray("data/Labeled/3/labeled_KL3_image106.png")

# each transformation on its own, plus all combined (same as train_transform in main.py)
transforms = {
    "Horizontal flip": v2.RandomHorizontalFlip(p=1.0),
    "Rotation ±10°": v2.RandomAffine(degrees=10),
    "Translation 5%": v2.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    "Scale 0.9-1.1": v2.RandomAffine(degrees=0, scale=(0.9, 1.1)),
    "Brightness/contrast": v2.ColorJitter(brightness=0.2, contrast=0.2),
    "All combined": v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
    ]),
}

n = 4
fig, axes = plt.subplots(len(transforms), n + 1, figsize=(2.5 * (n + 1), 2.5 * len(transforms)))

for row, (name, t) in zip(axes, transforms.items()):
    samples = [xray] + [t(xray) for _ in range(n)]

    for ax, x in zip(row, samples):
        # fixed 0-1 range, otherwise imshow rescales and hides brightness changes
        ax.imshow(x[0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    row[0].set_title("Original", fontsize=8)
    row[1].set_title(name, fontsize=8)

os.makedirs("data/outputs", exist_ok=True)
fig.tight_layout()
fig.savefig("data/outputs/augmentations.png", dpi=150)
plt.close(fig)
