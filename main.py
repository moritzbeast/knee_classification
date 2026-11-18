from args import *
import pandas as pd
import os
from dataset import knee_xray_dataset, compute_mean_std
from torch.utils.data import DataLoader
from model import MyModel
from trainer import train_model
from utils import set_seed, make_run_dir
from torchvision.transforms import v2

# 1. We need the arguments
args = get_args()
run_dir = make_run_dir(args)
print(f"Saving results to {run_dir}")

# augmentation, only for the training set
train_transform = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    v2.ColorJitter(brightness=0.2, contrast=0.2),
])

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
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bs)

    # 5. Initializing the model
    model = MyModel(backbone=args.backbone)

    # 6. Train the model
    train_model(model, train_loader, val_loader, fold, run_dir)