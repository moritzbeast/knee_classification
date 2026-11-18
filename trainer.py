import os
import warnings
from args import *
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

args = get_args()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

METRICS = ["loss", "acc", "bal_acc", "f1"]


def compute_metrics(labels, preds):
    # sklearn warns if a class is never predicted (e.g. KL4 early in training);
    # this would flood the live output, the values are not affected
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acc = accuracy_score(labels, preds)
        bal_acc = balanced_accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)

    return acc, bal_acc, f1


def format_metrics(loss, acc, bal_acc, f1):
    return f"Loss {loss:.4f} | Acc {acc:.4f} | Bal Acc {bal_acc:.4f} | F1 {f1:.4f}"


def train_model(model, train_loader, val_loader, fold, run_dir):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_bal_acc = -1

    history = {f"{split}_{key}": [] for split in ["train", "val"] for key in METRICS}

    pbar = tqdm(range(args.epochs), desc=f"Fold {fold}")
    for epoch in pbar:
        model.train()
        running_loss = 0

        all_labels = []
        all_preds = []

        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for i, batch in enumerate(batch_bar):
            inputs = batch["xray"].to(device)
            labels = batch["target"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            all_labels.append(labels.cpu().numpy())
            all_preds.append(outputs.argmax(dim=1).cpu().numpy())

            # live values over all batches of this epoch so far;
            # after the last batch these are the train metrics of the epoch
            train_loss = running_loss/(i + 1)
            train_acc, train_bal_acc, train_f1 = compute_metrics(np.concatenate(all_labels), np.concatenate(all_preds))
            # dict instead of keyword args, otherwise tqdm sorts the keys alphabetically
            batch_bar.set_postfix({"loss": f"{train_loss:.4f}", "acc": f"{train_acc:.4f}",
                                   "bal_acc": f"{train_bal_acc:.4f}", "f1": f"{train_f1:.4f}"})

        val_loss, val_acc, val_bal_acc, val_f1 = validate_model(model, val_loader, criterion)

        for key, value in [("train_loss", train_loss), ("train_acc", train_acc),
                           ("train_bal_acc", train_bal_acc), ("train_f1", train_f1),
                           ("val_loss", val_loss), ("val_acc", val_acc),
                           ("val_bal_acc", val_bal_acc), ("val_f1", val_f1)]:
            history[key].append(value)

        tqdm.write(f"Epoch {epoch+1}/{args.epochs}")
        tqdm.write(f"  Train | {format_metrics(train_loss, train_acc, train_bal_acc, train_f1)}")
        tqdm.write(f"  Val   | {format_metrics(val_loss, val_acc, val_bal_acc, val_f1)}")

        if val_bal_acc > best_bal_acc:
            best_bal_acc = val_bal_acc
            torch.save(model.state_dict(), os.path.join(run_dir, f"best_model_fold_{fold}.pth"))
            tqdm.write(f"  New best model saved (Bal Acc: {best_bal_acc:.4f})")

        pbar.set_postfix({"val_loss": f"{val_loss:.4f}", "val_bal_acc": f"{val_bal_acc:.4f}",
                          "best": f"{best_bal_acc:.4f}"})

    plot_metrics(history, fold, run_dir)


def validate_model(model, val_loader, criterion):
    model.eval()

    running_loss = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["xray"].to(device)
            labels = batch["target"].to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            all_labels.append(labels.cpu().numpy())
            all_preds.append(outputs.argmax(dim=1).cpu().numpy())

    validation_loss = running_loss/len(val_loader)
    acc, bal_acc, f1 = compute_metrics(np.concatenate(all_labels), np.concatenate(all_preds))

    return validation_loss, acc, bal_acc, f1


def plot_metrics(history, fold, run_dir):
    metrics = [("loss", "Loss"),
               ("bal_acc", "Balanced Accuracy"),
               ("acc", "Accuracy"),
               ("f1", "F1 Score (macro)")]

    epochs = range(1, len(history["train_loss"]) + 1)
    best_epoch = int(np.argmax(history["val_bal_acc"])) + 1

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, (key, title) in zip(axes.flat, metrics):
        ax.plot(epochs, history[f"train_{key}"], color="blue", linewidth=2, label="Training")
        ax.plot(epochs, history[f"val_{key}"], color="orange", linewidth=2, linestyle="--", label="Validation")
        ax.axvline(best_epoch, color="gray", linestyle=":", linewidth=1, label=f"Best model (epoch {best_epoch})")
        ax.set_title(f"{title} over Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(f"Fold {fold}")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, f"metrics_fold_{fold}.png"), dpi=150)
    plt.close(fig)
