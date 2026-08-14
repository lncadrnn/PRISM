"""
Fine-tuning script for the CNN-ViT hybrid image detector.

Usage (run from the repo root — a plain `python training/image/train.py` fails
with ModuleNotFoundError, since the script imports via the `training` package):
    python -m training.image.train \
        --data data/image \
        --epochs 20 \
        --batch 32 \
        --lr 2e-4 \
        --output models/image_detector.pt

Resuming a run that got cut off (Kaggle/Colab session limit, crash, etc.):
    python -m training.image.train \
        --data data/image --epochs 20 --batch 32 --lr 2e-4 \
        --output models/image_detector.pt \
        --resume models/image_detector.pt.train_state.pt

Strategy (from PRISM research):
  1. Freeze both backbones for the first N warmup epochs — train only the fusion head.
  2. Unfreeze and fine-tune end-to-end with a lower LR.
  3. Save the checkpoint with the best validation F1.

Checkpointing: `--output` always holds the best-val-F1 model weights only (what
`api/modules/image/detector.py` loads for inference). A second file,
`<output>.train_state.pt`, is overwritten after *every* epoch (regardless of
whether it improved F1) with the full training state (model/optimizer/
scheduler/AMP scaler, epoch number, best F1 so far) so a cut-off run can
resume from the last completed epoch instead of restarting from scratch.
"""

import argparse
import os
import random
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, classification_report

# Allow running from repo root: python training/image/train.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../api"))

from modules.image.model import CNNViTHybrid
from training.image.dataset import ImageForensicsDataset


def parse_args():
    p = argparse.ArgumentParser(description="Train PRISM image forensics module")
    p.add_argument("--data", default="data/image", help="Root of real/fake dataset")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--warmup-epochs", type=int, default=5,
                   help="Epochs to train head-only before unfreezing backbones")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--output", default="models/image_detector.pt")
    p.add_argument("--resume", default=None,
                   help="Path to a <output>.train_state.pt to continue an interrupted run")
    p.add_argument("--device", default=None)
    return p.parse_args()


def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    n_batches = len(loader)
    t0 = time.time()
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
        if i % 10 == 0 or i == n_batches - 1:
            elapsed = time.time() - t0
            print(
                f"    batch {i+1}/{n_batches}  loss={loss.item():.4f}  "
                f"elapsed={elapsed:.1f}s  ({elapsed / (i + 1):.2f}s/batch)",
                flush=True,
            )
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    n = len(all_labels)
    f1 = f1_score(all_labels, all_preds, average="binary")
    return total_loss / n, f1, all_preds, all_labels


def make_optimizer_and_scheduler(model, args, phase):
    """phase: 'warmup' (head-only) or 'finetune' (all params, lr * 0.1).

    Always constructed fresh (last_epoch=-1) — when resuming, the caller
    restores exact state via optimizer.load_state_dict()/scheduler.load_state_dict()
    right after, which also fixes up the internal epoch counters and LR.
    """
    if phase == "warmup":
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.warmup_epochs
        )
    return optimizer, scheduler


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
    # Build two separate dataset instances so their transforms are independent.
    # Sharing a single dataset and patching .transform on the Subset would also
    # change the transform for the training split (both Subsets hold a reference
    # to the same underlying Dataset object), silently disabling augmentation.
    # This mirrors the pattern used in training/video/train.py.
    full_ds = ImageForensicsDataset(args.data, augment=True)
    counts = full_ds.class_counts()
    print(f"Dataset: {counts}")

    val_n = max(1, int(len(full_ds) * args.val_split))
    train_n = len(full_ds) - val_n

    # Determine the split indices once, then apply them to both dataset variants.
    # Shuffle before slicing: ImageForensicsDataset appends all "real" samples
    # before all "fake" samples, so an unshuffled tail slice would make the
    # validation split almost entirely one class.
    all_indices = list(range(len(full_ds)))
    random.Random(42).shuffle(all_indices)
    train_indices = all_indices[:train_n]
    val_indices = all_indices[train_n:]

    train_ds = Subset(full_ds, train_indices)

    # Validation uses a non-augmented copy so we don't touch full_ds.transform.
    val_ds_base = ImageForensicsDataset(args.data, augment=False)
    val_ds = Subset(val_ds_base, val_indices)

    # num_workers=0 on Windows to avoid DataLoader fork/pickling issues.
    num_workers = 0 if os.name == "nt" else 4
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=num_workers)

    # Class-weighted loss to handle potential imbalance
    real_n, fake_n = counts["real"], counts["fake"]
    total = real_n + fake_n
    weights = torch.tensor([total / (2 * real_n), total / (2 * fake_n)], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # --- Model ---
    model = CNNViTHybrid(freeze_backbones=True).to(device)

    best_f1, best_path = 0.0, args.output
    os.makedirs(os.path.dirname(best_path) or ".", exist_ok=True)
    train_state_path = args.resume or (best_path + ".train_state.pt")

    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
    start_epoch = 1

    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        best_f1 = ckpt["best_f1"]
        start_epoch = ckpt["epoch"] + 1
        phase = ckpt["phase"]

        if phase == "finetune":
            for p in model.parameters():
                p.requires_grad = True
        optimizer, scheduler = make_optimizer_and_scheduler(model, args, phase)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        print(f"  -> resuming at epoch {start_epoch}, phase={phase}, best_f1={best_f1:.4f}")
    else:
        optimizer, scheduler = make_optimizer_and_scheduler(model, args, "warmup")

    for epoch in range(start_epoch, args.epochs + 1):
        # Unfreeze backbones after warmup (only fires if we haven't already
        # crossed this boundary in a previous, resumed session).
        if epoch == args.warmup_epochs + 1:
            print("Unfreezing backbones for end-to-end fine-tuning")
            for p in model.parameters():
                p.requires_grad = True
            optimizer, scheduler = make_optimizer_and_scheduler(model, args, "finetune")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_f1, preds, labels = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_f1={val_f1:.4f}",
            flush=True,
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_path)
            print(f"  -> New best F1 {best_f1:.4f} -- saved to {best_path}")

        # Persist full training state every epoch so a cut-off run (Kaggle/Colab
        # session limit, crash, etc.) can resume instead of starting over.
        phase = "finetune" if epoch >= args.warmup_epochs + 1 else "warmup"
        torch.save({
            "epoch": epoch,
            "phase": phase,
            "best_f1": best_f1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
        }, best_path + ".train_state.pt")

    print(f"\nTraining complete. Best val F1: {best_f1:.4f}")
    print("\nFinal classification report:")
    _, _, preds, labels = eval_epoch(
        model, val_loader, criterion, device
    )
    print(classification_report(labels, preds, target_names=["real", "fake"]))


if __name__ == "__main__":
    main()
