"""
ml/train_bbdm.py — BBDM / Latent Brownian Bridge Diffusion training script.

Source: Notebook Cells 57, 58, 60-70 (BBDM setup and forward pass exploration).

Model: LatentBrownianBridgeModel (from xuekt98/BBDM repository)
    VQGAN-F8 encodes DSE SAR (3ch, 256×256) → latent (4ch, 32×32)
    Brownian Bridge UNet operates in latent space
    VQGAN-F8 decodes sampled latent → synthetic EO (3ch, 256×256)

PREREQUISITE:
    1. BBDM repository must be cloned:
       python scripts/setup_bbdm.py
    2. VQGAN-F8 checkpoint must exist at:
       ml/checkpoints/bbdm/model.ckpt
    3. Pre-encoded latent files must exist in:
       data/processed/BBDM_latents/train/   (<id>_dse.pt, <id>_eo.pt)
       data/processed/BBDM_latents/test/

NOTE: The BBDM UNet checkpoint produced by this script goes into:
    ml/checkpoints/bbdm/bbdm_unet_ep<N>.pt

Usage:
    python -m ml.train_bbdm
    python -m ml.train_bbdm --resume ml/checkpoints/bbdm/bbdm_unet_ep40.pt
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import cfg
from ml.models.bbdm_wrapper import load_bbdm, _ensure_bbdm_on_path, BBDMConfig
from ml.datasets import PairedLatentDataset


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_bbdm_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    avg_loss: float,
    out_path: Path,
) -> None:
    """Save BBDM UNet checkpoint with training state."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "avg_loss":  round(avg_loss, 6),
        "saved_at":  datetime.utcnow().isoformat(),
    }
    torch.save(payload, str(out_path))

    meta = {
        "model_type":     "bbdm_unet",
        "epoch":          epoch + 1,
        "avg_loss":       round(avg_loss, 6),
        "architecture":   "LatentBrownianBridgeModel (LBBDM-f8)",
        "vqgan":          "VQGAN-F8 (frozen, 4×32×32 latent)",
        "unet_in_ch":     4,
        "unet_out_ch":    4,
        "latent_size":    32,
        "num_timesteps":  1000,
        "saved_at":       datetime.utcnow().isoformat(),
    }
    meta_path = out_path.with_suffix(".meta.json")
    with open(str(meta_path), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Checkpoint: {out_path}")
    print(f"  Metadata:   {meta_path}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = args.device or cfg.DEVICE

    # ------------------------------------------------------------------
    # Ensure BBDM repository is available
    # ------------------------------------------------------------------
    _ensure_bbdm_on_path()

    # ------------------------------------------------------------------
    # Load LatentBrownianBridgeModel
    # ------------------------------------------------------------------
    print("Loading LatentBrownianBridgeModel...")
    latent_bbdm = load_bbdm(
        vqgan_ckpt=str(cfg.VQGAN_CKPT),
        bbdm_ckpt=args.resume,
        device=device,
    )
    latent_bbdm.train()

    # Freeze VQGAN — only train the UNet diffusion model
    if hasattr(latent_bbdm, "vqgan"):
        for p in latent_bbdm.vqgan.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Optimizer — UNet parameters only
    # ------------------------------------------------------------------
    trainable = [p for p in latent_bbdm.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    # Recover start epoch from resume checkpoint
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt.get("epoch", 0) + 1

    # ------------------------------------------------------------------
    # Pre-encoded latent datasets (Cell 95)
    # ------------------------------------------------------------------
    train_latent_dir = str(cfg.BBDM_LATENT / "train")
    test_latent_dir  = str(cfg.BBDM_LATENT / "test")

    if not Path(train_latent_dir).exists():
        raise RuntimeError(
            f"BBDM training latent directory not found: {train_latent_dir}\n"
            "Pre-encode latents first by running the VQGAN encoder on DSE+EO pairs."
        )

    train_ds = PairedLatentDataset(train_latent_dir)
    test_ds  = PairedLatentDataset(test_latent_dir) if Path(test_latent_dir).exists() else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
    )
    print(f"Train latent pairs: {len(train_ds)}")
    if test_ds:
        print(f"Test latent pairs:  {len(test_ds)}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\nStarting BBDM training: epochs {start_epoch + 1} → {args.epochs}")
    print(f"Device: {device}\n")

    for epoch in range(start_epoch, args.epochs):
        latent_bbdm.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            dse_latent = batch["dse"].to(device)   # B × 4 × 32 × 32
            eo_latent  = batch["eo"].to(device)    # B × 4 × 32 × 32

            optimizer.zero_grad()

            # BBDM training step — calls the model's p_losses() method
            # which samples a random timestep and computes the diffusion loss
            loss_dict = latent_bbdm.p_losses(
                x_start=eo_latent,
                y=dse_latent,
            )

            if isinstance(loss_dict, dict):
                loss = loss_dict.get("loss", loss_dict.get("p_loss", next(iter(loss_dict.values()))))
            else:
                loss = loss_dict

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(latent_bbdm.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{args.epochs} - loss: {avg_loss:.6f}")

        # Save every 10 epochs
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            ckpt_name = f"bbdm_unet_ep{epoch+1}.pt"
            save_bbdm_checkpoint(
                latent_bbdm, optimizer, epoch, avg_loss,
                cfg.CKPT_BBDM / ckpt_name,
            )

    print("\nBBDM training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FloodSense BBDM (SAR → SynEO diffusion)")
    p.add_argument("--resume",      type=str,   default=None,
                   help="Path to BBDM UNet checkpoint to resume from")
    p.add_argument("--epochs",      type=int,   default=80,
                   help="Total training epochs (default: 80)")
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--batch_size",  type=int,   default=4,
                   help="Batch size for latent training (default: 4, smaller due to GPU memory)")
    p.add_argument("--num_workers", type=int,   default=cfg.NUM_WORKERS)
    p.add_argument("--device",      type=str,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
