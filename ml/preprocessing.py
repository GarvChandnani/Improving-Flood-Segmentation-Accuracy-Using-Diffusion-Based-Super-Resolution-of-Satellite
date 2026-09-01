"""
ml/preprocessing.py — SAR and EO preprocessing functions for FloodSense.

Source: Notebook Cells 6, 16, 20, 34, 60.

TWO NORMALIZATION SCHEMES ARE PRESERVED:

Scheme A — Baseline / Pix2Pix / Multimodal (Cells 6, 16, 20):
    SAR: (clip(sar, -30, 0) + 30) / 30  →  [0, 1] per channel
    EO:  clip(s2, 0, 3000) / 3000        →  [0, 1] per channel
    Output: 2 channels (VV, VH)

Scheme B — DSE / BBDM (Cells 34, 60):
    SAR: per-channel range normalization using dataset statistics
         VV: [−24.3028, −2.4026], VH: [−34.4817, −9.5345]
         plus a 3rd channel: (VV_norm + VH_norm) / 2
    EO:  s2 / 10000.0, clipped to [0, 1]
    Output: 3 channels (VV_norm, VH_norm, mean)

These schemes are INTENTIONALLY different — each is calibrated to its
respective model track. Do NOT unify them without retraining.

Also contains the loss functions and metric functions from Cell 6
that are shared across training and inference.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

import rasterio


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_tif(path: str) -> np.ndarray:
    """
    Load a GeoTIFF and return its data as a numpy array.

    Returns: np.ndarray of shape (bands, H, W)

    Exact replica of notebook Cell 6:
        def load_tif(path):
            with rasterio.open(path) as src:
                return src.read()
    """
    with rasterio.open(path) as src:
        return src.read()


# ---------------------------------------------------------------------------
# Scheme A — Simple clip-scale SAR (baseline / pix2pix / multimodal)
# ---------------------------------------------------------------------------

def preprocess_sar_simple(
    sar: np.ndarray,
    clip_low: float = -30.0,
    clip_high: float = 0.0,
) -> np.ndarray:
    """
    Normalize a 2-channel SAR array using the simple clip-and-scale scheme.

    Used by: FloodDataset, PairedDataset, MultiModalDataset.

    Exact logic from notebook Cells 6 / 16 / 20:
        s1 = np.nan_to_num((np.clip(s1, -30, 0) + 30) / 30)

    Args:
        sar:       np.float32 array of shape (2, H, W).  Channel 0 = VV, 1 = VH.
        clip_low:  Lower dB clip value  (notebook default: −30).
        clip_high: Upper dB clip value  (notebook default:  0).

    Returns:
        np.float32 array of shape (2, H, W), values in [0, 1],
        with NaN / ±Inf replaced by 0.
    """
    sar = sar.astype(np.float32)
    span = clip_high - clip_low          # 30.0
    sar = (np.clip(sar, clip_low, clip_high) - clip_low) / span
    sar = np.nan_to_num(sar, nan=0.0, posinf=1.0, neginf=0.0)
    return sar


def preprocess_s2_simple(
    s2: np.ndarray,
    clip_max: float = 3000.0,
) -> np.ndarray:
    """
    Normalize a Sentinel-2 tile for Pix2Pix training (Scheme A).

    Selects bands [3, 2, 1] (NIR, Red, Green) if ≥ 4 bands are present,
    otherwise uses the first 3 bands.

    Exact logic from notebook Cell 20:
        s2 = np.nan_to_num(np.clip(s2, 0, 3000) / 3000)
        s2 = s2[[3,2,1], :, :] if s2.shape[0] >= 4 else s2[:3, :, :]

    Args:
        s2:       np.float32 array of shape (C, H, W).
        clip_max: Upper reflectance clip value (notebook default: 3000).

    Returns:
        np.float32 array of shape (3, H, W), values in [0, 1].
    """
    s2 = s2.astype(np.float32)
    s2 = np.nan_to_num(np.clip(s2, 0.0, clip_max) / clip_max)
    s2 = s2[[3, 2, 1], :, :] if s2.shape[0] >= 4 else s2[:3, :, :]
    return s2


# ---------------------------------------------------------------------------
# Scheme B — DSE/BBDM per-channel range normalization
# ---------------------------------------------------------------------------

# Dataset statistics from notebook Cells 34 / 60
VV_LOW  = -24.3028
VV_HIGH =  -2.4026
VH_LOW  = -34.4817
VH_HIGH =  -9.5345


def _normalize_channel(
    x: np.ndarray,
    low: float,
    high: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize a single SAR channel to [0, 1] using fixed range values.

    Exact replica of notebook Cell 34:
        def normalize_channel(x, low, high):
            x = x.astype(np.float32)
            result = np.zeros_like(x, dtype=np.float32)
            valid = np.isfinite(x)
            result[valid] = (x[valid] - low) / (high - low)
            result = np.clip(result, 0.0, 1.0)
            return result, valid

    Returns:
        (normalized_channel, valid_mask) both shape (H, W)
    """
    x = x.astype(np.float32)
    result = np.zeros_like(x, dtype=np.float32)
    valid = np.isfinite(x)
    result[valid] = (x[valid] - low) / (high - low)
    result = np.clip(result, 0.0, 1.0)
    return result, valid


def preprocess_sar_dse(
    sar: np.ndarray,
    vv_low: float = VV_LOW,
    vv_high: float = VV_HIGH,
    vh_low: float = VH_LOW,
    vh_high: float = VH_HIGH,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 2-channel SAR array to a 3-channel DSE representation.

    Used by: BBDM training, BBDM inference pipeline.

    Exact logic from notebook Cells 34 and 60:
        vv_norm, vv_valid = normalize_channel(vv, VV_LOW, VV_HIGH)
        vh_norm, vh_valid = normalize_channel(vh, VH_LOW, VH_HIGH)
        valid = vv_valid & vh_valid
        mean_channel = (vv_norm + vh_norm) / 2.0
        vv_norm[~valid] = 0.0
        vh_norm[~valid] = 0.0
        mean_channel[~valid] = 0.0
        dse_sar = np.stack([vv_norm, vh_norm, mean_channel], axis=0)

    Args:
        sar:     np.float32 array of shape (2, H, W).  Channel 0=VV, 1=VH.
        vv_low, vv_high: VV dB normalization range.
        vh_low, vh_high: VH dB normalization range.

    Returns:
        dse_sar: np.float32 (3, H, W) — channels [VV_norm, VH_norm, (VV+VH)/2]
        valid:   np.float32 (H, W)    — 1.0 where both VV and VH are finite
    """
    sar = sar.astype(np.float32)
    vv = sar[0]
    vh = sar[1]

    vv_norm, vv_valid = _normalize_channel(vv, vv_low, vv_high)
    vh_norm, vh_valid = _normalize_channel(vh, vh_low, vh_high)

    valid = vv_valid & vh_valid

    mean_channel = (vv_norm + vh_norm) / 2.0

    # Zero out invalid (NoData) pixels
    vv_norm[~valid] = 0.0
    vh_norm[~valid] = 0.0
    mean_channel[~valid] = 0.0

    dse_sar = np.stack([vv_norm, vh_norm, mean_channel], axis=0).astype(np.float32)
    return dse_sar, valid.astype(np.float32)


def preprocess_eo_dse(
    eo: np.ndarray,
    scale: float = 10000.0,
) -> np.ndarray:
    """
    Normalize a Sentinel-2 tile for BBDM/DSE training.

    Exact logic from notebook Cell 34:
        eo = eo.astype(np.float32)
        eo = eo / 10000.0
        eo = np.clip(eo, 0.0, 1.0)

    Args:
        eo:    np.float32 array of shape (C, H, W). Sentinel-2 reflectance.
        scale: Division scale (notebook default: 10000.0).

    Returns:
        np.float32 array of shape (C, H, W), values in [0, 1].
    """
    eo = eo.astype(np.float32)
    eo = eo / scale
    eo = np.clip(eo, 0.0, 1.0)
    return eo


# ---------------------------------------------------------------------------
# Tensor resizing helper (shared by all datasets)
# ---------------------------------------------------------------------------

def resize_to(
    arr: np.ndarray,
    size: int = 256,
    mode_2d: str = "nearest",
    mode_nd: str = "bilinear",
) -> np.ndarray:
    """
    Resize a numpy array to (size × size) using F.interpolate.

    Matches notebook resize pattern:
        torch.nn.functional.interpolate(
            torch.from_numpy(arr).unsqueeze(0),
            size=(size, size),
            mode='bilinear'
        ).squeeze(0).numpy()

    Args:
        arr:     (C, H, W) float32 numpy array.
        size:    Target spatial size.
        mode_2d: Interpolation mode for 2D (label) arrays.
        mode_nd: Interpolation mode for multi-channel arrays.

    Returns:
        (C, size, size) float32 numpy array.
    """
    t = torch.from_numpy(arr.astype(np.float32))
    if t.ndim == 2:
        # Label: (H, W) → (1, 1, H, W)
        t = t.unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(size, size), mode=mode_2d)
        return t.squeeze(0).squeeze(0).numpy()
    else:
        # Multi-channel: (C, H, W) → (1, C, H, W)
        t = t.unsqueeze(0)
        t = F.interpolate(t, size=(size, size), mode=mode_nd)
        return t.squeeze(0).numpy()


# ---------------------------------------------------------------------------
# Loss functions (from notebook Cell 6)
# ---------------------------------------------------------------------------

def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Soft Dice loss computed only over valid (non-NoData) pixels.

    Exact replica of notebook Cell 6:
        pred = torch.sigmoid(pred)
        pred, target = pred[mask], target[mask]
        if pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        inter = (pred * target).sum()
        return 1 - (2*inter + eps) / (pred.sum() + target.sum() + eps)

    Args:
        pred:   Raw logits, shape (B, H, W).
        target: Binary labels, shape (B, H, W), values in {0, 1}.
        mask:   Boolean mask of valid pixels, shape (B, H, W).
        eps:    Smoothing term.

    Returns:
        Scalar Dice loss tensor.
    """
    pred = torch.sigmoid(pred)
    pred   = pred[mask]
    target = target[mask]
    if pred.numel() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    inter = (pred * target).sum()
    return 1 - (2 * inter + eps) / (pred.sum() + target.sum() + eps)


def compute_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Intersection-over-Union computed over valid pixels only.

    Exact replica of notebook Cell 6:
        pred = (torch.sigmoid(pred) > threshold).float()
        pred, target = pred[mask], target[mask]
        inter = (pred * target).sum()
        union = ((pred + target) > 0).float().sum()
        return (inter / (union + 1e-6)).item()

    Args:
        pred:      Raw logits, shape (B, H, W).
        target:    Binary labels, shape (B, H, W).
        mask:      Boolean mask of valid pixels.
        threshold: Binarization threshold (default 0.5).

    Returns:
        IoU as a Python float.
    """
    pred   = (torch.sigmoid(pred) > threshold).float()
    pred   = pred[mask]
    target = target[mask]
    inter  = (pred * target).sum()
    union  = ((pred + target) > 0).float().sum()
    return (inter / (union + 1e-6)).item()


def evaluate_iou(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: Optional[str] = None,
) -> float:
    """
    Evaluate mean IoU of a segmentation model on a DataLoader.

    Exact replica of notebook Cell 6:
        model.eval()
        ious = []
        with torch.no_grad():
            for x, lbl in loader:
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
                x, lbl = x.cuda(), lbl.cuda()
                mask = (lbl != -1)
                target = torch.clamp(lbl, 0, 1)
                pred = model(x).squeeze(1)
                ious.append(compute_iou(pred, target, mask))
        return np.mean(ious)

    Args:
        model:  Segmentation model (UNet).
        loader: Validation DataLoader yielding (inputs, labels).
        device: Target device. If None, uses 'cuda' if available else 'cpu'.

    Returns:
        Mean IoU over all batches.
    """
    import numpy as np

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    ious = []

    with torch.no_grad():
        for x, lbl in loader:
            x   = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
            x   = x.to(device)
            lbl = lbl.to(device)

            mask   = (lbl != -1)
            target = torch.clamp(lbl, 0, 1)

            pred = model(x).squeeze(1)
            ious.append(compute_iou(pred, target, mask))

    return float(np.mean(ious))
