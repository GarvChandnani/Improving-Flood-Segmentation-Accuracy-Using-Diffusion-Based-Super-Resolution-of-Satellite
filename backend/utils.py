"""
backend/utils.py — Helper functions for image rendering, file handling, and ML model caching.
"""

import os
import sys
import time
import uuid
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import cfg
from ml.preprocessing import load_tif, preprocess_sar_simple

# Caching active models in memory to avoid repeated loading
MODEL_CACHE: Dict[str, torch.nn.Module] = {}


def get_outputs_dir() -> Path:
    out_dir = cfg.OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "previews").mkdir(exist_ok=True)
    (out_dir / "predictions").mkdir(exist_ok=True)
    (out_dir / "uploads").mkdir(exist_ok=True)
    return out_dir


def sar_to_png(sar_path: str, out_png_path: str) -> str:
    """Convert 2-channel SAR GeoTIFF (VV, VH) or any image to RGB PNG for UI display."""
    try:
        sar = load_tif(sar_path).astype(np.float32)
        sar_norm = preprocess_sar_simple(sar)  # (2, H, W) in [0, 1]
        
        vv = sar_norm[0]
        vh = sar_norm[1] if sar_norm.shape[0] > 1 else vv
        mean = (vv + vh) / 2.0
        
        # Stack into RGB: R=VV, G=VH, B=Mean
        rgb = np.stack([vv, vh, mean], axis=-1)  # (H, W, 3)
        rgb_uint8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    except Exception:
        img_obj = Image.open(sar_path).convert("RGB")
        rgb_uint8 = np.array(img_obj)
    
    img = Image.fromarray(rgb_uint8)
    Path(out_png_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png_path)
    return out_png_path


def syneo_to_png(syneo_arr_or_path: Tuple[np.ndarray, str], out_png_path: str) -> str:
    """Convert 3-channel SynEO array or npy file to RGB PNG for UI display."""
    if isinstance(syneo_arr_or_path, str) or isinstance(syneo_arr_or_path, Path):
        syneo = np.load(str(syneo_arr_or_path)).astype(np.float32)
    else:
        syneo = syneo_arr_or_path.astype(np.float32)
    
    # syneo: (3, H, W)
    rgb = np.transpose(syneo, (1, 2, 0))  # (H, W, 3)
    rgb_uint8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    
    img = Image.fromarray(rgb_uint8)
    Path(out_png_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png_path)
    return out_png_path


def prob_to_png(prob_map: np.ndarray, out_png_path: str, colormap: str = "hot") -> str:
    """Convert float32 probability map [0, 1] to false-color heatmap PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.imshow(prob_map, cmap=colormap, vmin=0.0, vmax=1.0)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    Path(out_png_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out_png_path


def binary_mask_to_png(binary_mask: np.ndarray, out_png_path: str) -> str:
    """Convert binary flood mask to cyan/emerald (#10B981) on dark (#0E1511) PNG."""
    h, w = binary_mask.shape
    # Dark surface #0E1511 = (14, 21, 17), Primary emerald #10B981 = (16, 185, 129)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :] = [14, 21, 17]  # background
    
    flood_idx = (binary_mask > 0)
    rgb[flood_idx] = [16, 185, 129]  # emerald flood water
    
    img = Image.fromarray(rgb)
    Path(out_png_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png_path)
    return out_png_path


def colourised_mask_to_png(prob_map: np.ndarray, binary_mask: np.ndarray, out_png_path: str) -> str:
    """
    Generate a colour-coded RGBA mask for the Map page.
    - Blue  (#3B82F6, 200 alpha) = safe water  (binary=1 AND prob >= 0.6)
    - Red   (#EF4444, 200 alpha) = high risk    (binary=1 AND prob <  0.6, i.e. uncertain/low confidence)
    - Transparent = no flood predicted
    This PNG is overlaid directly on the SAR image in the map UI.
    """
    h, w = binary_mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)  # fully transparent background

    flood = binary_mask > 0
    high_conf = prob_map >= 0.6   # confident prediction  → blue (safe water identified)
    low_conf  = prob_map <  0.6   # uncertain prediction   → red  (risky / uncertain)

    # Safe water — blue
    safe_mask = flood & high_conf
    rgba[safe_mask] = [59, 130, 246, 200]   # #3B82F6 with alpha 200

    # Risky / uncertain — red
    risky_mask = flood & low_conf
    rgba[risky_mask] = [239, 68, 68, 200]   # #EF4444 with alpha 200

    img = Image.fromarray(rgba, mode="RGBA")
    Path(out_png_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png_path)
    return out_png_path


def difference_map_to_png(baseline_pred: np.ndarray, multimodal_pred: np.ndarray, gt: np.ndarray, out_png_path: str) -> str:
    """
    Generate error difference map for UI comparison screen.
    Green = TP (Multimodal matched GT)
    Red = False Positive / Error
    Blue = Baseline prediction
    """
    h, w = baseline_pred.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :] = [14, 21, 17]  # dark bg

    gt_mask = (gt > 0.5)
    base_mask = (baseline_pred > 0.5)
    multi_mask = (multimodal_pred > 0.5)

    # TP (emerald)
    tp = multi_mask & gt_mask
    rgb[tp] = [16, 185, 129]

    # FP in baseline (red)
    fp_base = base_mask & (~gt_mask)
    rgb[fp_base] = [255, 99, 132]

    # FN recovered by multimodal (cyan)
    recovered = multi_mask & (~base_mask) & gt_mask
    rgb[recovered] = [76, 214, 255]

    img = Image.fromarray(rgb)
    Path(out_png_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png_path)
    return out_png_path
