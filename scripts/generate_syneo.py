"""
scripts/generate_syneo.py — Generate SynEO .npy files for all dataset chips.

Source: Notebook Cell 24.

Uses the trained Pix2Pix generator to produce synthetic optical EO imagery
from SAR inputs for every chip. The resulting .npy files are required for
multimodal training (train_multimodal.py) and multimodal inference.

Output location: data/processed/SynEO/<country>_<chip_id>_SynEO.npy
Output shape:    (3, 256, 256) float32 — synthetic RGB EO

Usage:
    # Use default pix2pix checkpoint (latest in ml/checkpoints/pix2pix/)
    python scripts/generate_syneo.py

    # Specify a checkpoint explicitly
    python scripts/generate_syneo.py --checkpoint ml/checkpoints/pix2pix/pix2pix_gen_v1_ep40.pt

    # Regenerate even if files already exist
    python scripts/generate_syneo.py --overwrite
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    args = parse_args()

    from ml.config import cfg
    from ml.models.unet import UNet
    from ml.inference import generate_syneo_for_all
    from ml.data_loader import collect_all_assets
    from ml.datasets import make_train_val_ids
    import torch

    device = args.device or cfg.DEVICE

    # ------------------------------------------------------------------
    # Load generator
    # ------------------------------------------------------------------
    print("Loading Pix2Pix generator...")
    from ml.inference import load_model
    generator = load_model("pix2pix", args.checkpoint, device)
    print(f"Generator loaded on: {device}")

    # ------------------------------------------------------------------
    # Build ID list (all train + val chips)
    # ------------------------------------------------------------------
    catalog_base = args.catalog_dir or str(cfg.DATA_DIR / "catalog" / "catalog")

    hand_source = collect_all_assets(f"{catalog_base}/sen1floods11_hand_labeled_source/*/*.json")
    weak_source = collect_all_assets(f"{catalog_base}/sen1floods11_weak_labeled_source/*/*.json")

    train_ids, val_ids = make_train_val_ids(weak_source, hand_source)
    all_ids = train_ids + val_ids
    print(f"Total chips to process: {len(all_ids)}")

    # ------------------------------------------------------------------
    # Check existing files
    # ------------------------------------------------------------------
    existing = list(cfg.SYNEO_DIR.glob("*_SynEO.npy"))
    print(f"Existing SynEO files: {len(existing)}")

    if args.overwrite:
        print("--overwrite: regenerating all files")
    else:
        remaining = len(all_ids) - len(existing)
        print(f"Files to generate: {remaining}")

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    print(f"\nGenerating SynEO → {cfg.SYNEO_DIR}")
    generate_syneo_for_all(
        id_list=all_ids,
        generator=generator,
        s1_dir=cfg.S1_DIR,
        syneo_dir=cfg.SYNEO_DIR,
        device=device,
        patch_size=cfg.PATCH_SIZE,
    )

    total = len(list(cfg.SYNEO_DIR.glob("*_SynEO.npy")))
    print(f"\nDone. Total SynEO files: {total}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate SynEO .npy files for all dataset chips")
    p.add_argument("--checkpoint",  type=str, default=None,
                   help="Path to pix2pix generator checkpoint")
    p.add_argument("--catalog_dir", type=str, default=None)
    p.add_argument("--device",      type=str, default=None)
    p.add_argument("--overwrite",   action="store_true",
                   help="Regenerate even if .npy already exists")
    return p.parse_args()


if __name__ == "__main__":
    main()
