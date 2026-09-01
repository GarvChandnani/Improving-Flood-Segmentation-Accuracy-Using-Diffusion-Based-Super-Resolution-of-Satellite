"""
ml/train_pix2pix.py — SAR → Synthetic EO Pix2Pix training script.

Source: Notebook Cells 20 (PairedDataset), 22 (training loop).

Models:
    Generator:     UNet(in_ch=2, out_ch=3)   — SAR → RGB synthetic EO
    Discriminator: PatchDiscriminator(in_ch=5) — conditioned on SAR + EO

Loss:
    D: 0.5 * (BCE(D(real), 1) + BCE(D(fake), 0))
    G: BCE(D(fake_G), 1) + LAMBDA_L1 * L1(fake_G, real_EO)
    LAMBDA_L1 = 100

Notebook training configuration:
    optimizer: Adam(lr=2e-4, betas=(0.5, 0.999)) for both G and D
    epochs: 40
    batch_size: 8
    checkpoint saved every 10 epochs
    gradient clipping: max_norm=1.0 (generator only)

Usage:
    python -m ml.train_pix2pix
    python -m ml.train_pix2pix --resume ml/checkpoints/pix2pix/pix2pix_gen_v1_ep30.pt
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
from ml.models.unet import UNet
from ml.models.discriminator import PatchDiscriminator
from ml.datasets import PairedDataset, FloodDataset, make_train_val_ids, make_paired_ids
from ml.data_loader import collect_all_assets


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint_pix2pix(
    generator: nn.Module,
    discriminator: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    epoch: int,
    g_loss: float,
    d_loss: float,
    out_dir: Path,
) -> None:
    """
    Save generator and discriminator state_dicts separately.

    Generator checkpoint is the primary artifact used for SynEO generation
    and multimodal training. Discriminator is saved for potential training
    resumption.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_path  = out_dir / f"pix2pix_gen_v1_ep{epoch+1}.pt"
    disc_path = out_dir / f"pix2pix_disc_v1_ep{epoch+1}.pt"

    torch.save(generator.state_dict(),    str(gen_path))
    torch.save(discriminator.state_dict(), str(disc_path))

    meta = {
        "model_type": "pix2pix",
        "generator": {
            "architecture": "UNet(in_ch=2, out_ch=3)",
            "in_ch": 2,
            "out_ch": 3,
        },
        "discriminator": {
            "architecture": "PatchDiscriminator(in_ch=5)",
            "in_ch": 5,
        },
        "epoch":  epoch + 1,
        "g_loss": round(g_loss, 6),
        "d_loss": round(d_loss, 6),
        "saved_at": datetime.utcnow().isoformat(),
        "normalization": "Scheme A: (clip(s1,-30,0)+30)/30 | S2: clip/3000 bands[3,2,1]",
        "lambda_l1": cfg.PIX2PIX_LAMBDA_L1,
    }
    meta_path = gen_path.with_suffix(".meta.json")
    with open(str(meta_path), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Generator checkpoint: {gen_path}")
    print(f"  Discriminator checkpoint: {disc_path}")
    print(f"  Metadata: {meta_path}")


def load_checkpoint_pix2pix(
    generator: nn.Module,
    discriminator: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    gen_ckpt_path: Path,
    device: str,
) -> int:
    """
    Load generator checkpoint (and discriminator if present alongside it).

    Returns starting epoch number.
    """
    # Load generator
    gen_state = torch.load(str(gen_ckpt_path), map_location=device, weights_only=False)
    if isinstance(gen_state, dict) and "generator" in gen_state:
        generator.load_state_dict(gen_state["generator"])
        start_epoch = gen_state.get("epoch", 0)
    else:
        generator.load_state_dict(gen_state)
        name = gen_ckpt_path.stem
        start_epoch = 0
        if "_ep" in name:
            try:
                start_epoch = int(name.split("_ep")[-1])
            except ValueError:
                pass

    # Attempt to load matching discriminator checkpoint
    disc_path = gen_ckpt_path.parent / gen_ckpt_path.name.replace("gen", "disc")
    if disc_path.exists():
        disc_state = torch.load(str(disc_path), map_location=device, weights_only=False)
        if isinstance(disc_state, dict) and "discriminator" in disc_state:
            discriminator.load_state_dict(disc_state["discriminator"])
        else:
            discriminator.load_state_dict(disc_state)
        print(f"  Discriminator checkpoint loaded: {disc_path}")

    print(f"  Resumed from epoch: {start_epoch + 1}")
    return start_epoch


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = args.device or cfg.DEVICE

    # ------------------------------------------------------------------
    # Build ID lists
    # ------------------------------------------------------------------
    catalog_base = args.catalog_dir or str(cfg.DATA_DIR / "catalog" / "catalog")
    hand_source  = collect_all_assets(f"{catalog_base}/sen1floods11_hand_labeled_source/*/*.json")
    weak_source  = collect_all_assets(f"{catalog_base}/sen1floods11_weak_labeled_source/*/*.json")

    _, val_ids   = make_train_val_ids(weak_source, hand_source)
    paired_ids   = make_paired_ids(val_ids)
    print(f"Pix2Pix pairs: {len(paired_ids)}")

    # ------------------------------------------------------------------
    # Dataset & loader
    # ------------------------------------------------------------------
    pix_ds = PairedDataset(paired_ids, patch_size=cfg.PATCH_SIZE)
    pix_loader = DataLoader(
        pix_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )

    # ------------------------------------------------------------------
    # Models — exact notebook Cell 20 instantiation
    # ------------------------------------------------------------------
    generator     = UNet(in_ch=2, out_ch=3).to(device)
    discriminator = PatchDiscriminator(in_ch=5).to(device)

    # ------------------------------------------------------------------
    # Optimizers — exact notebook Cell 22 configuration
    # ------------------------------------------------------------------
    opt_g = torch.optim.Adam(generator.parameters(),     lr=args.lr_g, betas=cfg.PIX2PIX_BETAS)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=cfg.PIX2PIX_BETAS)

    bce_logits = nn.BCEWithLogitsLoss()
    l1         = nn.L1Loss()
    LAMBDA_L1  = cfg.PIX2PIX_LAMBDA_L1

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint_pix2pix(
            generator, discriminator, opt_g, opt_d,
            Path(args.resume), device
        )

    # ------------------------------------------------------------------
    # Training loop — exact notebook Cell 22
    # ------------------------------------------------------------------
    print(f"\nStarting Pix2Pix training: epochs {start_epoch + 1} → {args.epochs}")
    print(f"Device: {device}  |  LAMBDA_L1={LAMBDA_L1}\n")

    for epoch in range(start_epoch, args.epochs):
        g_tot, d_tot, n = 0.0, 0.0, 0

        for s1, s2_real in tqdm(pix_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            s1, s2_real = s1.to(device), s2_real.to(device)

            # ----------------------------------------------------------
            # Train Discriminator
            # ----------------------------------------------------------
            s2_fake = generator(s1).detach()

            d_real = discriminator(torch.cat([s1, s2_real], dim=1))
            d_fake = discriminator(torch.cat([s1, s2_fake], dim=1))

            d_loss = 0.5 * (
                bce_logits(d_real, torch.ones_like(d_real))
                + bce_logits(d_fake, torch.zeros_like(d_fake))
            )

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # ----------------------------------------------------------
            # Train Generator
            # ----------------------------------------------------------
            s2_fake   = generator(s1)
            d_fake_g  = discriminator(torch.cat([s1, s2_fake], dim=1))

            g_loss = (
                bce_logits(d_fake_g, torch.ones_like(d_fake_g))
                + LAMBDA_L1 * l1(s2_fake, s2_real)
            )

            if torch.isnan(g_loss) or torch.isnan(d_loss):
                continue

            opt_g.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            opt_g.step()

            g_tot += g_loss.item()
            d_tot += d_loss.item()
            n     += 1

        print(
            f"Epoch {epoch+1}/{args.epochs} "
            f"- G: {g_tot/max(n,1):.4f}  D: {d_tot/max(n,1):.4f}"
        )

        # Checkpoint every 10 epochs (matches notebook Cell 22)
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            save_checkpoint_pix2pix(
                generator, discriminator,
                opt_g, opt_d,
                epoch, g_tot / max(n, 1), d_tot / max(n, 1),
                cfg.CKPT_PIX2PIX,
            )

    print("\nPix2Pix training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FloodSense Pix2Pix (SAR → SynEO)")
    p.add_argument("--resume",      type=str,   default=None,
                   help="Path to generator checkpoint to resume from")
    p.add_argument("--epochs",      type=int,   default=cfg.PIX2PIX_EPOCHS)
    p.add_argument("--lr_g",        type=float, default=cfg.PIX2PIX_LR_G)
    p.add_argument("--lr_d",        type=float, default=cfg.PIX2PIX_LR_D)
    p.add_argument("--batch_size",  type=int,   default=cfg.BATCH_SIZE_TRAIN)
    p.add_argument("--num_workers", type=int,   default=cfg.NUM_WORKERS)
    p.add_argument("--device",      type=str,   default=None)
    p.add_argument("--catalog_dir", type=str,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
