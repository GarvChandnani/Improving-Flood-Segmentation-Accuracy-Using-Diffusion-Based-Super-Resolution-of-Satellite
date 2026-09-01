"""
scripts/download_checkpoints.py — Dataset download and checkpoint restoration guide.

Source: Notebook Cells 8, 12 (dataset download loops).

This script handles:
    1. Downloading the Sen1Floods11 dataset from the STAC catalog
    2. Guidance on restoring checkpoints from Google Drive

CHECKPOINT RESTORATION:
    The trained model weights live in your Google Drive under:
        flood_project_v2/checkpoints/

    Copy them to the local project as follows:
        baseline_ep20.pt        → ml/checkpoints/baseline/baseline_v1_ep20.pt
        pix2pix_gen_ep40.pt     → ml/checkpoints/pix2pix/pix2pix_gen_v1_ep40.pt
        multimodal_latest.pt    → ml/checkpoints/multimodal/multimodal_latest.pt
        (any multimodal_ep*.pt) → ml/checkpoints/multimodal/
        BBDM_VQGAN/extracted/model.ckpt → ml/checkpoints/bbdm/model.ckpt
        (BBDM UNet checkpoint if trained) → ml/checkpoints/bbdm/bbdm_unet_ep*.pt

    After copying, run the hash verifier:
        python scripts/download_checkpoints.py --verify

Usage:
    # Download dataset from Sen1Floods11 catalog
    python scripts/download_checkpoints.py --download-data --catalog data/catalog

    # Verify checkpoint hashes only
    python scripts/download_checkpoints.py --verify
"""

import sys
import os
import hashlib
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Hash verification
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checkpoints() -> dict:
    """Compute and print SHA256 hashes for all checkpoint files."""
    from ml.config import cfg

    print("=" * 70)
    print("CHECKPOINT HASH VERIFICATION")
    print("=" * 70)

    extensions = {".pt", ".pth", ".ckpt", ".bin", ".safetensors"}
    results = {}

    checkpoint_dirs = [
        cfg.CKPT_BASELINE,
        cfg.CKPT_PIX2PIX,
        cfg.CKPT_MULTIMODAL,
        cfg.CKPT_BBDM,
    ]

    for ckpt_dir in checkpoint_dirs:
        if not ckpt_dir.exists():
            continue
        for f in sorted(ckpt_dir.rglob("*")):
            if f.suffix in extensions and f.is_file():
                size_mb = f.stat().st_size / (1024 ** 2)
                sha    = sha256_file(f)
                rel    = f.relative_to(cfg.CHECKPOINT_DIR)
                results[str(rel)] = {
                    "path":    str(f),
                    "size_mb": round(size_mb, 2),
                    "sha256":  sha,
                }
                print(f"\n  File:   {rel}")
                print(f"  Size:   {size_mb:.2f} MB")
                print(f"  SHA256: {sha}")

    if not results:
        print("\n  No checkpoint files found.")
        print("  Copy your .pt/.ckpt files from Google Drive first.")

    # Save hash manifest
    manifest_path = cfg.CHECKPOINT_DIR / "checkpoint_manifest.json"
    with open(str(manifest_path), "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\n  Manifest saved to: {manifest_path}")

    return results


# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------

def download_dataset(catalog_dir: str) -> None:
    """
    Download the Sen1Floods11 dataset from the STAC catalog.

    Source: Notebook Cell 12.
    """
    from ml.config import cfg
    from ml.data_loader import (
        collect_all_assets,
        download_all_s1_hand,
        download_all_labels_hand,
        download_all_s2_hand,
        download_all_s1_weak,
        download_all_weak_labels,
    )

    catalog_base = catalog_dir

    print("=" * 60)
    print("SEN1FLOODS11 DATASET DOWNLOAD")
    print("=" * 60)
    print(f"Catalog: {catalog_base}")

    # Collect asset lists
    hand_source = collect_all_assets(f"{catalog_base}/sen1floods11_hand_labeled_source/*/*.json")
    hand_label  = collect_all_assets(f"{catalog_base}/sen1floods11_hand_labeled_label/*/*.json")
    weak_source = collect_all_assets(f"{catalog_base}/sen1floods11_weak_labeled_source/*/*.json")
    weak_label  = collect_all_assets(f"{catalog_base}/sen1floods11_weak_labeled_label/*/*.json")

    print(f"Hand-labeled source: {len(hand_source)}")
    print(f"Hand-labeled label:  {len(hand_label)}")
    print(f"Weak-labeled source: {len(weak_source)}")
    print(f"Weak-labeled label:  {len(weak_label)}")

    # Create directories
    for d in [cfg.S1_DIR, cfg.LABEL_HAND, cfg.S2_HAND, cfg.WEAK_LABEL]:
        d.mkdir(parents=True, exist_ok=True)

    # Download all types (resumable)
    download_all_s1_hand(hand_source,  str(cfg.S1_DIR))
    download_all_labels_hand(hand_label, str(cfg.LABEL_HAND))
    failed_s2 = download_all_s2_hand(hand_source, str(cfg.S2_HAND))
    download_all_s1_weak(weak_source,  str(cfg.S1_DIR))
    download_all_weak_labels(weak_label, str(cfg.WEAK_LABEL))

    print(f"\nDownload complete. S2Hand failures: {len(failed_s2)}")
    if failed_s2:
        print("Failed chips:", failed_s2[:10])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download dataset and verify checkpoints")
    p.add_argument("--verify",        action="store_true",
                   help="Verify SHA256 hashes of all checkpoint files")
    p.add_argument("--download-data", action="store_true",
                   help="Download the Sen1Floods11 dataset")
    p.add_argument("--catalog",       type=str, default=None,
                   help="Path to extracted catalog directory")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verify:
        verify_checkpoints()
    if args.download_data:
        from ml.config import cfg
        catalog = args.catalog or str(cfg.DATA_DIR / "catalog" / "catalog")
        download_dataset(catalog)
    if not args.verify and not args.download_data:
        print("Usage: python scripts/download_checkpoints.py --verify")
        print("       python scripts/download_checkpoints.py --download-data --catalog <path>")
