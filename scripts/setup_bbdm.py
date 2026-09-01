"""
scripts/setup_bbdm.py — Clone and set up the BBDM external repository.

Source: Notebook Cell 35 (git clone BBDM).

This script:
    1. Clones https://github.com/xuekt98/BBDM.git to ml/models/bbdm_repo/
    2. Installs BBDM Python dependencies
    3. Verifies the repository structure

Run once before any BBDM operations:
    python scripts/setup_bbdm.py

The BBDM repository is NOT re-implemented in this project.
It is used as an external dependency via sys.path.
"""

import os
import sys
import subprocess
from pathlib import Path

# Project root is one level up from scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    from ml.config import cfg

    bbdm_dir = cfg.BBDM_REPO_DIR
    print("=" * 60)
    print("BBDM REPOSITORY SETUP")
    print("=" * 60)
    print(f"Target directory: {bbdm_dir}")

    # ------------------------------------------------------------------
    # Clone if not already present
    # ------------------------------------------------------------------
    if not bbdm_dir.exists():
        print("\nCloning BBDM repository...")
        subprocess.run(
            [
                "git", "clone",
                "https://github.com/xuekt98/BBDM.git",
                str(bbdm_dir),
            ],
            check=True,
        )
        print(f"Cloned to: {bbdm_dir}")
    else:
        print(f"\nRepository already exists at: {bbdm_dir}")
        print("To update, run:  git -C <path> pull")

    # ------------------------------------------------------------------
    # Verify required files (from notebook Cell 50)
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("REPOSITORY STRUCTURE CHECK")
    print("-" * 60)

    required = [
        "model/BrownianBridge/BrownianBridgeModel.py",
        "model/BrownianBridge/LatentBrownianBridgeModel.py",
        "datasets/custom.py",
    ]

    all_ok = True
    for rel_path in required:
        full = bbdm_dir / rel_path
        exists = full.exists()
        status = "OK" if exists else "MISSING"
        print(f"  {rel_path:55s} [{status}]")
        if not exists:
            all_ok = False

    if not all_ok:
        print("\nWARNING: Some BBDM files are missing.")
        print("The repository may have changed structure.")
        print(f"Please check: {bbdm_dir}")
    else:
        print("\n[OK] BBDM repository is complete.")

    # ------------------------------------------------------------------
    # Install BBDM Python dependencies (from notebook Cell 40)
    # ------------------------------------------------------------------
    requirements_file = bbdm_dir / "requirements.txt"
    if requirements_file.exists():
        print("\n" + "-" * 60)
        print("INSTALLING BBDM DEPENDENCIES")
        print("-" * 60)
        print(f"From: {requirements_file}")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements_file)],
            check=False,  # Don't fail on partial installs
        )
        print("Dependencies installed.")

    # ------------------------------------------------------------------
    # Check for VQGAN checkpoint
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("VQGAN CHECKPOINT CHECK")
    print("-" * 60)

    vqgan_ckpt = cfg.VQGAN_CKPT
    if vqgan_ckpt.exists():
        size_mb = vqgan_ckpt.stat().st_size / (1024 ** 2)
        print(f"  VQGAN checkpoint found: {vqgan_ckpt}")
        print(f"  Size: {size_mb:.1f} MB")
    else:
        print(f"  VQGAN checkpoint NOT FOUND: {vqgan_ckpt}")
        print()
        print("  Download the VQGAN-F8 checkpoint (model.ckpt) from your Google Drive:")
        print("  flood_project_v2/BBDM_VQGAN/extracted/model.ckpt")
        print(f"  → Copy to: {vqgan_ckpt}")

    print("\n" + "=" * 60)
    print("BBDM SETUP COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
