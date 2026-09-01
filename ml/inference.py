"""
ml/inference.py — Clean inference API for FloodSense models.

Source: Notebook Cells 6 (metrics), 24 (SynEO generation), 27 (evaluation),
        28 (visualization), 278 (baseline inference), 303-304 (multimodal inference).

All inference functions:
    - Load the correct architecture for the requested model type
    - Apply the correct preprocessing scheme for that model
    - Run a forward pass without any gradient computation
    - Return numpy arrays (for backend integration)

The backend should call these functions directly.
No retraining occurs here.

Supported model types:
    "baseline"    — UNet(in_ch=2, out_ch=1), Scheme A SAR preprocessing
    "pix2pix"     — UNet(in_ch=2, out_ch=3), Scheme A SAR preprocessing
    "multimodal"  — UNet(in_ch=5, out_ch=1), Scheme A SAR + SynEO
    "bbdm"        — LatentBrownianBridgeModel, Scheme B (DSE) SAR preprocessing
"""

import os
import sys
import json
from pathlib import Path
from collections import OrderedDict
from typing import Optional, Union, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import cfg
from ml.models.unet import UNet
from ml.models.discriminator import PatchDiscriminator
from ml.preprocessing import (
    load_tif,
    preprocess_sar_simple,
    preprocess_s2_simple,
    preprocess_sar_dse,
    preprocess_eo_dse,
    compute_iou,
    evaluate_iou,
    dice_loss,
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _strip_module_prefix(state_dict: dict) -> OrderedDict:
    """
    Remove common framework prefixes from checkpoint keys.

    From notebook Cell 277:
        for prefix in ["module.", "model.", "generator.", "netG."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
    """
    clean = OrderedDict()
    prefixes = ["module.", "model.", "generator.", "netG."]
    for k, v in state_dict.items():
        nk = k
        for prefix in prefixes:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
                break
        clean[nk] = v
    return clean


def load_model(
    model_type: str,
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
) -> nn.Module:
    """
    Load a trained FloodSense model from a checkpoint.

    Args:
        model_type:      One of "baseline", "pix2pix", "multimodal", "bbdm".
        checkpoint_path: Path to .pt checkpoint file. If None, uses default
                         checkpoint from cfg.CKPT_* directories.
        device:          Target device ("cuda" / "cpu"). Defaults to cfg.DEVICE.

    Returns:
        Model in eval mode on the specified device.

    Raises:
        FileNotFoundError if checkpoint does not exist.
        RuntimeError if state_dict keys are incompatible.
    """
    dev = device or cfg.DEVICE

    # ------------------------------------------------------------------
    # Instantiate the correct architecture
    # ------------------------------------------------------------------
    if model_type == "baseline":
        model = UNet(in_ch=2, out_ch=1)
        default_ckpt = _find_latest_checkpoint(cfg.CKPT_BASELINE)

    elif model_type == "pix2pix":
        model = UNet(in_ch=2, out_ch=3)
        default_ckpt = _find_latest_checkpoint(cfg.CKPT_PIX2PIX, pattern="*gen*.pt")

    elif model_type == "multimodal":
        model = UNet(in_ch=5, out_ch=1)
        default_ckpt = cfg.CKPT_MULTIMODAL / "multimodal_latest.pt"
        if not default_ckpt.exists():
            default_ckpt = _find_latest_checkpoint(cfg.CKPT_MULTIMODAL)

    elif model_type == "bbdm":
        from ml.models.bbdm_wrapper import load_bbdm
        return load_bbdm(
            vqgan_ckpt=checkpoint_path or str(cfg.VQGAN_CKPT),
            device=dev,
        )

    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            "Expected: 'baseline', 'pix2pix', 'multimodal', 'bbdm'."
        )

    # ------------------------------------------------------------------
    # Resolve checkpoint path
    # ------------------------------------------------------------------
    ckpt_path = Path(checkpoint_path) if checkpoint_path else default_ckpt

    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found for model_type='{model_type}'.\n"
            f"Expected at: {ckpt_path or cfg.CKPT_BASELINE}\n"
            "Please copy your checkpoint files from Google Drive."
        )

    # ------------------------------------------------------------------
    # Load state_dict
    # ------------------------------------------------------------------
    state = torch.load(str(ckpt_path), map_location=dev, weights_only=False)

    # Handle dict formats (resumable, state_dict-in-dict)
    if isinstance(state, dict):
        if "model" in state:
            state = state["model"]
        elif "state_dict" in state:
            state = state["state_dict"]

    state = _strip_module_prefix(state)

    # Strict loading — any key mismatch is a real problem
    missing, unexpected = model.load_state_dict(state, strict=True)

    if missing:
        raise RuntimeError(
            f"Missing keys when loading {model_type} checkpoint '{ckpt_path}':\n"
            + "\n".join(f"  {k}" for k in missing[:20])
        )
    if unexpected:
        print(
            f"[load_model] WARNING: Unexpected keys in checkpoint (will be ignored):\n"
            + "\n".join(f"  {k}" for k in unexpected[:10])
        )

    model = model.to(dev).eval()
    print(f"[load_model] Loaded {model_type} from {ckpt_path}")
    return model


def _find_latest_checkpoint(directory: Path, pattern: str = "*.pt") -> Optional[Path]:
    """Return the most recently modified .pt file in a directory."""
    import glob as _glob
    files = list(directory.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Preprocessing for inference
# ---------------------------------------------------------------------------

def preprocess_input(
    sar_path: str,
    scheme: str = "A",
    patch_size: int = 256,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
    """
    Load and preprocess a SAR tile for inference.

    Args:
        sar_path:   Path to a SAR GeoTIFF (2-channel VV/VH).
        scheme:     "A" — simple clip-scale (baseline/pix2pix/multimodal)
                    "B" — DSE per-channel range (BBDM)
        patch_size: Target spatial size (default: 256).
        device:     Target device.

    Returns:
        (tensor, valid_mask)
            tensor:     1 × C × patch_size × patch_size (float32, on device)
            valid_mask: (H, W) float32 array of valid pixels (None for Scheme A)
    """
    dev = device or cfg.DEVICE

    sar = load_tif(sar_path).astype(np.float32)

    if scheme == "A":
        sar_norm = preprocess_sar_simple(sar)
        valid    = None
    elif scheme == "B":
        sar_norm, valid = preprocess_sar_dse(sar)
    else:
        raise ValueError(f"Unknown preprocessing scheme: '{scheme}'. Use 'A' or 'B'.")

    # Resize to patch_size
    t = torch.from_numpy(sar_norm).unsqueeze(0)  # 1 × C × H × W
    if t.shape[-1] != patch_size or t.shape[-2] != patch_size:
        t = F.interpolate(t, size=(patch_size, patch_size), mode="bilinear", align_corners=False)

    return t.to(dev), valid


# ---------------------------------------------------------------------------
# Baseline inference
# ---------------------------------------------------------------------------

def run_baseline_inference(
    sar_path: str,
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
    threshold: float = 0.5,
    patch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run baseline SAR-only flood segmentation.

    Args:
        sar_path:        Path to SAR GeoTIFF (2-channel, VV/VH).
        checkpoint_path: Optional explicit checkpoint path.
        device:          Target device.
        threshold:       Binary mask threshold (default: 0.5).
        patch_size:      Inference resolution (default: 256).

    Returns:
        (prob_map, binary_mask)
            prob_map:    float32 (H, W) — flood probability [0, 1]
            binary_mask: uint8   (H, W) — 0/1 binary prediction
    """
    dev   = device or cfg.DEVICE
    model = load_model("baseline", checkpoint_path, dev)

    s1_tensor, _ = preprocess_input(sar_path, scheme="A", patch_size=patch_size, device=dev)
    s1_tensor    = torch.nan_to_num(s1_tensor, nan=0.0, posinf=1.0, neginf=0.0)

    with torch.no_grad():
        logits = model(s1_tensor).squeeze()  # (H, W)
        prob   = torch.sigmoid(logits).cpu().numpy().astype(np.float32)

    binary = (prob > threshold).astype(np.uint8)
    return prob, binary


# ---------------------------------------------------------------------------
# Pix2Pix: SAR → Synthetic EO generation
# ---------------------------------------------------------------------------

def generate_synthetic_eo(
    sar_path: str,
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
    patch_size: int = 256,
) -> np.ndarray:
    """
    Generate synthetic optical imagery from a SAR tile using Pix2Pix generator.

    Source: Notebook Cell 24.

    Args:
        sar_path:        Path to SAR GeoTIFF (2-channel).
        checkpoint_path: Optional explicit generator checkpoint path.
        device:          Target device.
        patch_size:      Output resolution (default: 256).

    Returns:
        syneo: float32 (3, H, W) — synthetic RGB EO image, values in [0, 1]
    """
    dev       = device or cfg.DEVICE
    generator = load_model("pix2pix", checkpoint_path, dev)

    s1_tensor, _ = preprocess_input(sar_path, scheme="A", patch_size=patch_size, device=dev)

    with torch.no_grad():
        syneo = generator(s1_tensor).squeeze(0).cpu().numpy().astype(np.float32)

    return syneo  # (3, H, W)


def generate_syneo_for_all(
    id_list: list,
    generator: nn.Module,
    s1_dir: Optional[Path] = None,
    syneo_dir: Optional[Path] = None,
    device: Optional[str] = None,
    patch_size: int = 256,
) -> None:
    """
    Generate and save SynEO .npy for all chips in id_list.

    Exact replica of notebook Cell 24.

    Args:
        id_list:    List of (country, chip_id) tuples.
        generator:  Loaded pix2pix generator model.
        s1_dir:     Directory containing S1 tiles. Defaults to cfg.S1_DIR.
        syneo_dir:  Output directory for .npy files. Defaults to cfg.SYNEO_DIR.
        device:     Target device.
        patch_size: Output resolution (default: 256).
    """
    from tqdm import tqdm

    dev  = device or cfg.DEVICE
    s1d  = Path(s1_dir)    if s1_dir    else cfg.S1_DIR
    seod = Path(syneo_dir) if syneo_dir else cfg.SYNEO_DIR
    seod.mkdir(parents=True, exist_ok=True)

    # Determine which IDs are in train vs val for suffix selection
    # (from notebook Cell 24: suffix = "S1Weak" if in train else "S1Hand")
    train_set = set()

    generator.eval()

    with torch.no_grad():
        for country, chip_id in tqdm(id_list, desc="SynEO generation"):
            out_path = seod / f"{country}_{chip_id}_SynEO.npy"
            if out_path.exists():
                continue  # Idempotent — skip existing

            # Try both suffixes
            for suffix in ["S1Weak", "S1Hand"]:
                s1_path = s1d / f"{country}_{chip_id}_{suffix}.tif"
                if s1_path.exists():
                    break
            else:
                print(f"  WARNING: No SAR file for {country}_{chip_id} — skipping")
                continue

            s1 = load_tif(str(s1_path)).astype(np.float32)
            s1 = preprocess_sar_simple(s1)

            s1_t = torch.from_numpy(s1).unsqueeze(0)
            s1_t = F.interpolate(s1_t, size=(patch_size, patch_size), mode="bilinear").to(dev)

            syneo = generator(s1_t).squeeze(0).cpu().numpy().astype(np.float32)
            np.save(str(out_path), syneo)


# ---------------------------------------------------------------------------
# Multimodal inference
# ---------------------------------------------------------------------------

def run_multimodal_inference(
    sar_path: str,
    syneo: Optional[np.ndarray] = None,
    syneo_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
    threshold: float = 0.5,
    patch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run multimodal (SAR + SynEO) flood segmentation.

    Source: Notebook Cells 303, 304.

    Checkpoint loading uses the nested format from notebook Cell 303:
        checkpoint = torch.load(CKPT, ...)
        if "model" in checkpoint:
            state_dict = checkpoint["model"]

    Args:
        sar_path:        Path to SAR GeoTIFF (2-channel).
        syneo:           Pre-generated SynEO array (3, H, W). If None, loads from syneo_path.
        syneo_path:      Path to SynEO .npy file. Used if syneo is None.
        checkpoint_path: Optional explicit checkpoint path.
        device:          Target device.
        threshold:       Binary mask threshold.
        patch_size:      Inference resolution.

    Returns:
        (prob_map, binary_mask) — same format as run_baseline_inference.
    """
    dev   = device or cfg.DEVICE
    model = load_model("multimodal", checkpoint_path, dev)

    # Load SAR
    s1_tensor, _ = preprocess_input(sar_path, scheme="A", patch_size=patch_size, device=dev)
    s1_tensor    = torch.nan_to_num(s1_tensor, nan=0.0, posinf=1.0, neginf=0.0)

    # Load SynEO
    if syneo is None:
        if syneo_path is None:
            raise ValueError("Either syneo (array) or syneo_path must be provided.")
        syneo = np.load(syneo_path)

    syneo_t = torch.from_numpy(syneo.astype(np.float32)).unsqueeze(0).to(dev)
    if syneo_t.shape[-1] != patch_size or syneo_t.shape[-2] != patch_size:
        syneo_t = F.interpolate(syneo_t, size=(patch_size, patch_size), mode="bilinear")

    # Concatenate SAR (2ch) + SynEO (3ch) → 5ch
    x = torch.cat([s1_tensor, syneo_t], dim=1)
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

    with torch.no_grad():
        logits = model(x).squeeze()
        prob   = torch.sigmoid(logits).cpu().numpy().astype(np.float32)

    binary = (prob > threshold).astype(np.uint8)
    return prob, binary


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_metrics(
    pred_mask: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.5,
    nodata_value: int = -1,
) -> Dict[str, float]:
    """
    Calculate flood segmentation metrics.

    Args:
        pred_mask:    Predicted probability map (H, W) float32 OR binary mask (H, W) uint8.
        ground_truth: Ground truth label (H, W). NoData pixels = nodata_value.
        threshold:    Binarization threshold for probability maps.
        nodata_value: Label value to ignore in evaluation.

    Returns:
        Dict with keys: iou, precision, recall, f1, accuracy
    """
    gt = ground_truth.astype(np.float32)
    valid = (gt != nodata_value)

    if pred_mask.dtype in (np.float32, np.float64):
        pred = (pred_mask > threshold).astype(np.float32)
    else:
        pred = pred_mask.astype(np.float32)

    pred = pred[valid]
    gt   = gt[valid]

    if len(pred) == 0:
        return {"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    tp = float((pred * gt).sum())
    fp = float((pred * (1 - gt)).sum())
    fn = float(((1 - pred) * gt).sum())
    tn = float(((1 - pred) * (1 - gt)).sum())

    iou       = tp / (tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    accuracy  = (tp + tn) / (tp + tn + fp + fn + 1e-6)

    return {
        "iou":       round(iou,       4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "accuracy":  round(accuracy,  4),
    }


# ---------------------------------------------------------------------------
# Save predictions
# ---------------------------------------------------------------------------

def save_prediction(
    mask: np.ndarray,
    out_path: str,
    save_tif: bool = True,
    save_npy: bool = True,
    reference_tif: Optional[str] = None,
) -> None:
    """
    Save a flood prediction mask as .npy and/or GeoTIFF.

    Args:
        mask:          Prediction array (H, W), float32 or uint8.
        out_path:      Output path without extension.
        save_tif:      Save as GeoTIFF (requires reference_tif for geo metadata).
        save_npy:      Save as .npy.
        reference_tif: Path to a reference GeoTIFF to copy CRS/transform from.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if save_npy:
        np.save(str(out_path.with_suffix(".npy")), mask)

    if save_tif:
        import rasterio
        if reference_tif is not None:
            with rasterio.open(reference_tif) as src:
                meta = src.meta.copy()
            meta.update({"count": 1, "dtype": str(mask.dtype)})
            with rasterio.open(str(out_path.with_suffix(".tif")), "w", **meta) as dst:
                dst.write(mask[np.newaxis, ...])
        else:
            # Save without geo metadata
            from PIL import Image
            img_arr = (mask * 255).astype(np.uint8) if mask.dtype == np.float32 else mask * 255
            Image.fromarray(img_arr).save(str(out_path.with_suffix(".png")))


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_predictions(
    sar: np.ndarray,
    pred_mask: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    syneo: Optional[np.ndarray] = None,
    title: str = "FloodSense Prediction",
    out_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Side-by-side visualization of SAR, SynEO (optional), prediction, and ground truth.

    Source: Notebook Cell 28.

    Args:
        sar:          (2, H, W) normalized SAR (VV channel used for display).
        pred_mask:    (H, W) float32 probability or binary mask.
        ground_truth: (H, W) float32 label. Optional.
        syneo:        (3, H, W) synthetic EO. Optional.
        title:        Plot title.
        out_path:     If set, saves figure to this path instead of / in addition to showing.
        show:         If True, calls plt.show().
    """
    import matplotlib
    matplotlib.use("Agg" if not show else "TkAgg")
    import matplotlib.pyplot as plt

    panels = [("SAR (VV)", sar[0])]
    if syneo is not None:
        panels.append(("SynEO (RGB)", np.transpose(syneo, (1, 2, 0)).clip(0, 1)))
    panels.append(("Prediction", pred_mask))
    if ground_truth is not None:
        panels.append(("Ground Truth", ground_truth))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=12)

    for ax, (label, data) in zip(axes, panels):
        if data.ndim == 3:
            ax.imshow(data)
        else:
            cmap = "gray" if "SAR" in label else ("RdBu_r" if "Truth" in label else "hot")
            ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
