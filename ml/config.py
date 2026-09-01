"""
ml/config.py — Central configuration for FloodSense project.

Replaces all Colab-specific hardcoded paths (/content/, /content/drive/MyDrive/, etc.)
with project-relative paths configurable via environment variables.

Usage:
    from ml.config import cfg

    # Access paths
    checkpoint_dir = cfg.CHECKPOINT_DIR / "baseline"
    data_dir = cfg.DATA_DIR / "S1"

Environment variable overrides (optional):
    FLOODSENSE_ROOT      — project root (default: parent of this file's directory)
    FLOODSENSE_DATA      — data directory
    FLOODSENSE_CKPT      — checkpoint root
    FLOODSENSE_OUTPUTS   — outputs directory
    FLOODSENSE_BBDM_REPO — path to cloned BBDM repository
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root — the flood-project/ directory
# ---------------------------------------------------------------------------
# Default: two levels up from this file (ml/config.py → ml/ → flood-project/)
_DEFAULT_ROOT = Path(__file__).resolve().parent.parent

PROJECT_ROOT = Path(os.environ.get("FLOODSENSE_ROOT", _DEFAULT_ROOT))

# ---------------------------------------------------------------------------
# Primary directory layout
# ---------------------------------------------------------------------------
DATA_DIR       = Path(os.environ.get("FLOODSENSE_DATA",    PROJECT_ROOT / "data"))
CHECKPOINT_DIR = Path(os.environ.get("FLOODSENSE_CKPT",    PROJECT_ROOT / "ml" / "checkpoints"))
OUTPUTS_DIR    = Path(os.environ.get("FLOODSENSE_OUTPUTS", PROJECT_ROOT / "outputs"))
ML_DIR         = PROJECT_ROOT / "ml"
SCRIPTS_DIR    = PROJECT_ROOT / "scripts"

# ---------------------------------------------------------------------------
# Data subdirectories
# ---------------------------------------------------------------------------
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR   = DATA_DIR / "samples"

# Subdirectories matching notebook convention
S1_DIR        = PROCESSED_DIR / "S1"         # SAR tiles (VV/VH GeoTIFFs)
LABEL_HAND    = PROCESSED_DIR / "LabelHand"  # Hand-labeled flood masks
S2_HAND       = PROCESSED_DIR / "S2Hand"     # Sentinel-2 optical tiles
WEAK_LABEL    = PROCESSED_DIR / "WeakLabel"  # Otsu weak-label masks
SYNEO_DIR     = PROCESSED_DIR / "SynEO"      # Pix2Pix generated synthetic EO (.npy)
BBDM_LATENT   = PROCESSED_DIR / "BBDM_latents"  # Pre-encoded VQGAN latents for BBDM

# ---------------------------------------------------------------------------
# Checkpoint subdirectories (one per model track)
# ---------------------------------------------------------------------------
CKPT_BASELINE   = CHECKPOINT_DIR / "baseline"
CKPT_PIX2PIX    = CHECKPOINT_DIR / "pix2pix"
CKPT_MULTIMODAL = CHECKPOINT_DIR / "multimodal"
CKPT_BBDM       = CHECKPOINT_DIR / "bbdm"

# ---------------------------------------------------------------------------
# BBDM external repository
# ---------------------------------------------------------------------------
BBDM_REPO_DIR = Path(
    os.environ.get("FLOODSENSE_BBDM_REPO", ML_DIR / "models" / "bbdm_repo")
)

# VQGAN-F8 checkpoint (CompVis, downloaded separately)
VQGAN_CKPT = CKPT_BBDM / "model.ckpt"

# ---------------------------------------------------------------------------
# Model / training defaults — MUST match notebook training configuration
# ---------------------------------------------------------------------------

# Image size used throughout (notebook always interpolates to 256×256)
PATCH_SIZE = 256

# SAR normalization scheme A (used by baseline / pix2pix / multimodal)
SAR_CLIP_LOW  = -30.0   # dB
SAR_CLIP_HIGH =   0.0   # dB

# SAR normalization scheme B — DSE/BBDM per-channel ranges
# (from notebook Cell 34 and Cell 60)
VV_LOW  = -24.3028
VV_HIGH =  -2.4026
VH_LOW  = -34.4817
VH_HIGH =  -9.5345

# Sentinel-2 normalization
S2_CLIP_MAX = 3000.0    # used in PairedDataset (pix2pix track)
S2_DSE_SCALE = 10000.0  # used in BBDM EO preprocessing

# Inference threshold for binary flood mask
INFERENCE_THRESHOLD = 0.5

# Default device (overridden at runtime if CUDA available)
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Training defaults — mirror notebook training calls
# ---------------------------------------------------------------------------
BATCH_SIZE_TRAIN = 8   # used in all three training scripts
BATCH_SIZE_VAL   = 8
NUM_WORKERS      = 2

# Baseline (Cell 18)
BASELINE_LR     = 1e-4
BASELINE_EPOCHS = 20

# Pix2Pix (Cell 22)
PIX2PIX_LR_G    = 2e-4
PIX2PIX_LR_D    = 2e-4
PIX2PIX_BETAS   = (0.5, 0.999)
PIX2PIX_EPOCHS  = 40
PIX2PIX_LAMBDA_L1 = 100.0

# Multimodal (Cell 26)
MULTIMODAL_LR     = 1e-4
MULTIMODAL_EPOCHS = 20

# ---------------------------------------------------------------------------
# Convenience: ensure output directories exist at import time
# ---------------------------------------------------------------------------
def ensure_dirs():
    """Create all required output/checkpoint directories if they don't exist."""
    dirs = [
        DATA_DIR, RAW_DIR, PROCESSED_DIR, SAMPLES_DIR,
        S1_DIR, LABEL_HAND, S2_HAND, WEAK_LABEL, SYNEO_DIR, BBDM_LATENT,
        OUTPUTS_DIR,
        CKPT_BASELINE, CKPT_PIX2PIX, CKPT_MULTIMODAL, CKPT_BBDM,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Simple config object for dot-access convenience
# ---------------------------------------------------------------------------
class _Config:
    PROJECT_ROOT    = PROJECT_ROOT
    DATA_DIR        = DATA_DIR
    RAW_DIR         = RAW_DIR
    PROCESSED_DIR   = PROCESSED_DIR
    SAMPLES_DIR     = SAMPLES_DIR
    S1_DIR          = S1_DIR
    LABEL_HAND      = LABEL_HAND
    S2_HAND         = S2_HAND
    WEAK_LABEL      = WEAK_LABEL
    SYNEO_DIR       = SYNEO_DIR
    BBDM_LATENT     = BBDM_LATENT
    OUTPUTS_DIR     = OUTPUTS_DIR
    CHECKPOINT_DIR  = CHECKPOINT_DIR
    CKPT_BASELINE   = CKPT_BASELINE
    CKPT_PIX2PIX    = CKPT_PIX2PIX
    CKPT_MULTIMODAL = CKPT_MULTIMODAL
    CKPT_BBDM       = CKPT_BBDM
    BBDM_REPO_DIR   = BBDM_REPO_DIR
    VQGAN_CKPT      = VQGAN_CKPT
    PATCH_SIZE      = PATCH_SIZE
    SAR_CLIP_LOW    = SAR_CLIP_LOW
    SAR_CLIP_HIGH   = SAR_CLIP_HIGH
    VV_LOW          = VV_LOW
    VV_HIGH         = VV_HIGH
    VH_LOW          = VH_LOW
    VH_HIGH         = VH_HIGH
    S2_CLIP_MAX     = S2_CLIP_MAX
    S2_DSE_SCALE    = S2_DSE_SCALE
    INFERENCE_THRESHOLD = INFERENCE_THRESHOLD
    DEVICE          = DEVICE
    BATCH_SIZE_TRAIN = BATCH_SIZE_TRAIN
    BATCH_SIZE_VAL   = BATCH_SIZE_VAL
    NUM_WORKERS      = NUM_WORKERS
    BASELINE_LR      = BASELINE_LR
    BASELINE_EPOCHS  = BASELINE_EPOCHS
    PIX2PIX_LR_G     = PIX2PIX_LR_G
    PIX2PIX_LR_D     = PIX2PIX_LR_D
    PIX2PIX_BETAS    = PIX2PIX_BETAS
    PIX2PIX_EPOCHS   = PIX2PIX_EPOCHS
    PIX2PIX_LAMBDA_L1 = PIX2PIX_LAMBDA_L1
    MULTIMODAL_LR    = MULTIMODAL_LR
    MULTIMODAL_EPOCHS = MULTIMODAL_EPOCHS


cfg = _Config()
