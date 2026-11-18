from args import *
import pandas as pd
import os
from dataset import knee_xray_dataset, compute_mean_std
from torch.utils.data import DataLoader
from model import MyModel
from trainer import train_model, plot_confusion_matrix
from utils import set_seed, make_run_dir
from torchvision.transforms import v2
import torch
import numpy as np

# 1. We need the arguments
args = get_args()
run_dir = make_run_dir(args)
print(f"Saving results to {run_dir}")

# parallel image loading on the GPU server; 0 on the Mac, because macOS would
# run this whole script again in every worker (no if __name__ == "__main__")
num_workers = 8 if torch.cuda.is_available() else 0

# augmentation, only for the training set
train_transform = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    v2.ColorJitter(brightness=0.2, contrast=0.2),
])

# val predictions of all folds: every train image is in exactly one val fold,
# predicted by a model that never saw it -> one confusion matrix over all 1881 images
all_val_labels, all_val_preds = [], []

# 2. Iterate over each fold
for fold in range(5):
    set_seed(args.seed)

    train_set = pd.read_csv(os.path.join(args.csv_dir, f"train_fold_{fold}.csv"))
    val_set = pd.read_csv(os.path.join(args.csv_dir, f"val_fold_{fold}.csv"))

    # 3. Preparing Dataset
    mean, std = compute_mean_std(train_set)
    print(f"Fold {fold} | Mean {mean:.4f} | Std {std:.4f}")

    train_dataset = knee_xray_dataset(train_set, mean, std, train_transform if args.augment else None)
    val_dataset = knee_xray_dataset(val_set, mean, std)

    # 4. Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, num_workers=num_workers)

    # 5. Initializing the model
    model = MyModel(backbone=args.backbone)

    # 6. Train the model
    labels, preds = train_model(model, train_loader, val_loader, fold, run_dir)
    all_val_labels.append(labels)
    all_val_preds.append(preds)

# 7. Confusion matrix over all folds
plot_confusion_matrix(np.concatenate(all_val_labels), np.concatenate(all_val_preds), "All folds - validation, best epochs",
                      os.path.join(run_dir, "confusion_matrix_all_folds.png"))