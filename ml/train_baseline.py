"""
ml/train_baseline.py — SAR-only baseline U-Net training script.

Source: Notebook Cells 16 (dataset setup) and 18 (training loop).

Model: UNet(in_ch=2, out_ch=1)
Input: 2-channel SAR (VV, VH) normalized via Scheme A
Output: 1-channel flood mask (logits)
Loss: BCE + Dice (masked, ignoring NoData pixels where label == -1)

Notebook training configuration:
    optimizer: Adam, lr=1e-4
    epochs: 20
    batch_size: 8
    checkpoint saved every 5 epochs
    gradient clipping: max_norm=1.0

Usage:
    # Train from scratch
    python -m ml.train_baseline

    # Resume from checkpoint
    python -m ml.train_baseline --resume ml/checkpoints/baseline/baseline_ep15.pt

    # Custom configuration
    python -m ml.train_baseline --epochs 30 --lr 5e-5 --batch_size 4
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import cfg
from ml.models.unet import UNet
from ml.preprocessing import dice_loss, compute_iou, evaluate_iou
from ml.datasets import FloodDataset, make_train_val_ids
from ml.data_loader import collect_all_assets


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    iou: float,
    out_path: Path,
) -> None:
    """Save model state_dict plus metadata JSON alongside checkpoint."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save weights — only state_dict, exactly as the notebook did
    torch.save(model.state_dict(), str(out_path))

    # Save metadata
    meta = {
        "model_type": "baseline_unet",
        "in_ch": 2,
        "out_ch": 1,
        "epoch": epoch,
        "val_loss": round(loss, 6),
        "val_iou": round(iou, 6),
        "saved_at": datetime.utcnow().isoformat(),
        "architecture": "UNet(in_ch=2, out_ch=1)",
        "normalization": "Scheme A: (clip(s1,-30,0)+30)/30",
        "patch_size": cfg.PATCH_SIZE,
    }
    meta_path = out_path.with_suffix(".meta.json")
    with open(str(meta_path), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Checkpoint saved: {out_path}")
    print(f"  Metadata saved:   {meta_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ckpt_path: Path,
    device: str,
) -> int:
    """
    Load a checkpoint and return the starting epoch.

    Supports both:
    - Plain state_dict (saved by notebook: torch.save(model.state_dict(), ...))
    - Dict with keys {"model", "optimizer", "epoch"} (saved by train_multimodal)
    """
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    if isinstance(state, dict) and "model" in state:
        # Resumable format
        model.load_state_dict(state["model"])
        if "optimizer" in state and optimizer is not None:
            optimizer.load_state_dict(state["optimizer"])
        start_epoch = state.get("epoch", 0) + 1
    else:
        # Plain state_dict (notebook original format)
        model.load_state_dict(state)
        # Infer epoch from filename if possible
        name = ckpt_path.stem  # e.g. "baseline_ep20"
        start_epoch = 0
        if "_ep" in name:
            try:
                start_epoch = int(name.split("_ep")[-1])
            except ValueError:
                pass

    print(f"  Resumed from: {ckpt_path}  (start epoch: {start_epoch + 1})")
    return start_epoch


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = args.device or cfg.DEVICE

    # ------------------------------------------------------------------
    # Build dataset IDs from catalog (or use pre-existing ID lists)
    # ------------------------------------------------------------------
    catalog_base = args.catalog_dir or str(cfg.DATA_DIR / "catalog" / "catalog")

    hand_source = collect_all_assets(f"{catalog_base}/sen1floods11_hand_labeled_source/*/*.json")
    weak_source = collect_all_assets(f"{catalog_base}/sen1floods11_weak_labeled_source/*/*.json")

    train_ids, val_ids = make_train_val_ids(weak_source, hand_source)
    print(f"Train samples: {len(train_ids)}  |  Val samples: {len(val_ids)}")

    # ------------------------------------------------------------------
    # Datasets & loaders
    # ------------------------------------------------------------------
    train_ds = FloodDataset(
        train_ids,
        s1_suffix="S1Weak",
        label_dir="WeakLabel",
        label_suffix="S1OtsuLabelWeak",
        patch_size=cfg.PATCH_SIZE,
    )
    val_ds = FloodDataset(
        val_ids,
        s1_suffix="S1Hand",
        label_dir="LabelHand",
        label_suffix="LabelHand",
        patch_size=cfg.PATCH_SIZE,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    # ------------------------------------------------------------------
    # Model, optimizer, loss — exact notebook Cell 18 configuration
    # ------------------------------------------------------------------
    model = UNet(in_ch=2, out_ch=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    start_epoch = 0
    if args.resume:
        ckpt_path = Path(args.resume)
        start_epoch = load_checkpoint(model, optimizer, ckpt_path, device)

    # ------------------------------------------------------------------
    # Training loop — exact notebook Cell 18
    # ------------------------------------------------------------------
    print(f"\nStarting baseline training: epochs {start_epoch + 1} → {args.epochs}")
    print(f"Device: {device}\n")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        n_batches  = 0

        for s1, lbl in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            # Exactly as notebook Cell 18
            s1  = torch.nan_to_num(s1, nan=0.0, posinf=1.0, neginf=0.0)
            s1  = s1.to(device)
            lbl = lbl.to(device)

            mask   = (lbl != -1)
            if mask.sum() == 0:
                continue

            target = torch.clamp(lbl, 0, 1)

            optimizer.zero_grad()
            pred = model(s1).squeeze(1)

            loss = (
                (bce(pred, target) * mask).sum() / mask.sum().clamp(min=1)
                + dice_loss(pred, target, mask)
            )

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{args.epochs} - loss: {avg_loss:.4f}")

        # ------------------------------------------------------------------
        # Validation IoU
        # ------------------------------------------------------------------
        iou = evaluate_iou(model, val_loader, device=device)
        print(f"  Val IoU: {iou:.4f}")

        # ------------------------------------------------------------------
        # Checkpoint save — every 5 epochs (matches notebook Cell 18 behaviour)
        # Also save at the very last epoch
        # ------------------------------------------------------------------
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            ckpt_name = f"baseline_v1_ep{epoch+1}.pt"
            out_path  = cfg.CKPT_BASELINE / ckpt_name
            save_checkpoint(model, optimizer, epoch, avg_loss, iou, out_path)

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    final_iou = evaluate_iou(model, val_loader, device=device)
    print(f"\nFinal Baseline IoU: {final_iou:.4f}")
    print("(Notebook reported: ~0.4779)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FloodSense baseline SAR U-Net")
    p.add_argument("--resume",      type=str,   default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--epochs",      type=int,   default=cfg.BASELINE_EPOCHS,
                   help=f"Total epochs (default: {cfg.BASELINE_EPOCHS})")
    p.add_argument("--lr",          type=float, default=cfg.BASELINE_LR,
                   help=f"Learning rate (default: {cfg.BASELINE_LR})")
    p.add_argument("--batch_size",  type=int,   default=cfg.BATCH_SIZE_TRAIN,
                   help=f"Batch size (default: {cfg.BATCH_SIZE_TRAIN})")
    p.add_argument("--num_workers", type=int,   default=cfg.NUM_WORKERS,
                   help=f"DataLoader workers (default: {cfg.NUM_WORKERS})")
    p.add_argument("--device",      type=str,   default=None,
                   help="Device: 'cuda' or 'cpu' (default: auto-detect)")
    p.add_argument("--catalog_dir", type=str,   default=None,
                   help="Path to extracted catalog directory")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
