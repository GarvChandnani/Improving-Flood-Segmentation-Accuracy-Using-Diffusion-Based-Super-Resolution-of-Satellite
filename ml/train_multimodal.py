"""
ml/train_multimodal.py — Multimodal SAR+SynEO flood segmentation training.

Source: Notebook Cells 10 (MultiModalDataset), 26 (training loop).

Model: UNet(in_ch=5, out_ch=1)
Input: 5-channel tensor — 2-channel SAR (Scheme A) concatenated with
       3-channel pix2pix-generated SynEO (.npy files)
Output: 1-channel flood mask (logits)

Loss: masked BCE + Dice (same as baseline, ignoring NoData label=-1 pixels)

Notebook training configuration:
    optimizer: Adam, lr=1e-4
    epochs: 20
    batch_size: 8
    checkpoint saved EVERY epoch (interruption-safe)
    gradient clipping: max_norm=1.0
    checkpoint format: {"epoch": int, "model": state_dict, "optimizer": state_dict}

IMPORTANT: The interruption-safe checkpoint format differs from baseline.
    The multimodal checkpoint is a dict, not a plain state_dict.
    This matches the notebook's Cell 26 format exactly.

Usage:
    # Train from scratch (requires SynEO .npy files to exist)
    python -m ml.train_multimodal

    # Resume from checkpoint
    python -m ml.train_multimodal --resume ml/checkpoints/multimodal/multimodal_latest.pt
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import cfg
from ml.models.unet import UNet
from ml.preprocessing import dice_loss, compute_iou, evaluate_iou
from ml.datasets import MultiModalDataset, FloodDataset, make_train_val_ids
from ml.data_loader import collect_all_assets


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint_multimodal(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    avg_loss: float,
    out_path: Path,
) -> None:
    """
    Save checkpoint in the interruption-safe resumable format.

    Exact replica of notebook Cell 26:
        torch.save({
            "epoch":     epoch,
            "model":     model_mm.state_dict(),
            "optimizer": optimizer_mm.state_dict(),
        }, CHECKPOINT)

    Also maintains a "latest" checkpoint symlink-style copy.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "avg_loss":  round(avg_loss, 6),
        "saved_at":  datetime.utcnow().isoformat(),
    }
    torch.save(payload, str(out_path))

    # Keep a "latest" copy in the same directory (matches notebook path)
    latest_path = out_path.parent / "multimodal_latest.pt"
    shutil.copy2(str(out_path), str(latest_path))

    print(f"  Checkpoint saved: {out_path}")
    print(f"  Latest updated:   {latest_path}")


def load_checkpoint_multimodal(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ckpt_path: Path,
    device: str,
) -> int:
    """
    Load multimodal checkpoint.

    Supports both:
    - Resumable format: {"epoch", "model", "optimizer"} (notebook Cell 26)
    - Plain state_dict

    Returns starting epoch (epoch after the loaded one).
    """
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
        if "optimizer" in state and optimizer is not None:
            optimizer.load_state_dict(state["optimizer"])
        start_epoch = state.get("epoch", 0) + 1
        print(f"  Resumed from epoch {start_epoch} (saved epoch: {state.get('epoch', '?')})")
    else:
        # Plain state_dict fallback
        model.load_state_dict(state)
        name = ckpt_path.stem
        start_epoch = 0
        if "_ep" in name:
            try:
                start_epoch = int(name.split("_ep")[-1])
            except ValueError:
                pass

    print(f"  Starting from epoch: {start_epoch + 1}")
    return start_epoch


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = args.device or cfg.DEVICE

    # ------------------------------------------------------------------
    # Build ID lists from catalog
    # ------------------------------------------------------------------
    catalog_base = args.catalog_dir or str(cfg.DATA_DIR / "catalog" / "catalog")
    hand_source  = collect_all_assets(f"{catalog_base}/sen1floods11_hand_labeled_source/*/*.json")
    weak_source  = collect_all_assets(f"{catalog_base}/sen1floods11_weak_labeled_source/*/*.json")

    train_ids, val_ids = make_train_val_ids(weak_source, hand_source)
    print(f"Train samples: {len(train_ids)}  |  Val samples: {len(val_ids)}")

    # ------------------------------------------------------------------
    # Check SynEO files exist before starting
    # ------------------------------------------------------------------
    syneo_count = len(list(cfg.SYNEO_DIR.glob("*_SynEO.npy")))
    if syneo_count == 0:
        raise RuntimeError(
            f"No SynEO .npy files found in {cfg.SYNEO_DIR}.\n"
            "Run:  python scripts/generate_syneo.py  first."
        )
    print(f"SynEO files found: {syneo_count}")

    # ------------------------------------------------------------------
    # Datasets & loaders — exact notebook Cell 10 setup
    # ------------------------------------------------------------------
    mm_train_ds = MultiModalDataset(
        train_ids,
        s1_suffix="S1Weak",
        label_dir="WeakLabel",
        label_suffix="S1OtsuLabelWeak",
        patch_size=cfg.PATCH_SIZE,
    )
    mm_val_ds = MultiModalDataset(
        val_ids,
        s1_suffix="S1Hand",
        label_dir="LabelHand",
        label_suffix="LabelHand",
        patch_size=cfg.PATCH_SIZE,
    )
    mm_train_loader = DataLoader(
        mm_train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )
    mm_val_loader = DataLoader(
        mm_val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    # ------------------------------------------------------------------
    # Model, optimizer, loss — exact notebook Cell 10 + 26
    # ------------------------------------------------------------------
    model_mm   = UNet(in_ch=5, out_ch=1).to(device)
    optimizer  = torch.optim.Adam(model_mm.parameters(), lr=args.lr)
    bce_mm     = nn.BCEWithLogitsLoss(reduction="none")

    # ------------------------------------------------------------------
    # Resume from checkpoint (interruption-safe)
    # ------------------------------------------------------------------
    start_epoch = 0

    # Check for latest checkpoint first (notebook Cell 26 pattern)
    resume_path = args.resume
    if resume_path is None:
        latest = cfg.CKPT_MULTIMODAL / "multimodal_latest.pt"
        if latest.exists() and not args.no_auto_resume:
            resume_path = str(latest)
            print(f"Auto-resuming from: {latest}")

    if resume_path:
        start_epoch = load_checkpoint_multimodal(
            model_mm, optimizer, Path(resume_path), device
        )
    else:
        print("Starting multimodal training from epoch 1")

    # ------------------------------------------------------------------
    # Training loop — exact notebook Cell 26
    # ------------------------------------------------------------------
    print(f"\nMultimodal training: epochs {start_epoch + 1} → {args.epochs}")
    print(f"Device: {device}\n")

    for epoch in range(start_epoch, args.epochs):
        model_mm.train()
        total_loss = 0
        n_batches  = 0

        for x, lbl in tqdm(mm_train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            x   = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
            x   = x.to(device)
            lbl = lbl.to(device)

            mask = (lbl != -1)
            if mask.sum() == 0:
                continue

            target = torch.clamp(lbl, 0, 1)

            optimizer.zero_grad()
            pred = model_mm(x).squeeze(1)

            loss = (
                (bce_mm(pred, target) * mask).sum() / mask.sum().clamp(min=1)
                + dice_loss(pred, target, mask)
            )

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_mm.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{args.epochs} - loss: {avg_loss:.4f}")

        # ------------------------------------------------------------------
        # Save after EVERY epoch (interruption-safe, matches notebook Cell 26)
        # ------------------------------------------------------------------
        ckpt_name = f"multimodal_v1_ep{epoch+1}.pt"
        out_path  = cfg.CKPT_MULTIMODAL / ckpt_name
        save_checkpoint_multimodal(model_mm, optimizer, epoch, avg_loss, out_path)

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    multimodal_iou = evaluate_iou(model_mm, mm_val_loader, device=device)
    print(f"\nMultimodal IoU: {multimodal_iou:.4f}")

    # Metadata for final checkpoint
    meta = {
        "model_type": "multimodal_unet",
        "in_ch": 5,
        "out_ch": 1,
        "architecture": "UNet(in_ch=5, out_ch=1)",
        "epoch": args.epochs,
        "val_iou": round(multimodal_iou, 6),
        "normalization": "SAR: Scheme A | SynEO: pix2pix output (3ch)",
        "saved_at": datetime.utcnow().isoformat(),
    }
    meta_path = cfg.CKPT_MULTIMODAL / f"multimodal_v1_ep{args.epochs}.meta.json"
    with open(str(meta_path), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Final metadata: {meta_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FloodSense multimodal SAR+SynEO U-Net")
    p.add_argument("--resume",        type=str,   default=None,
                   help="Checkpoint to resume from (default: auto-detect latest)")
    p.add_argument("--no_auto_resume", action="store_true",
                   help="Disable automatic resume from latest checkpoint")
    p.add_argument("--epochs",        type=int,   default=cfg.MULTIMODAL_EPOCHS)
    p.add_argument("--lr",            type=float, default=cfg.MULTIMODAL_LR)
    p.add_argument("--batch_size",    type=int,   default=cfg.BATCH_SIZE_TRAIN)
    p.add_argument("--num_workers",   type=int,   default=cfg.NUM_WORKERS)
    p.add_argument("--device",        type=str,   default=None)
    p.add_argument("--catalog_dir",   type=str,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
