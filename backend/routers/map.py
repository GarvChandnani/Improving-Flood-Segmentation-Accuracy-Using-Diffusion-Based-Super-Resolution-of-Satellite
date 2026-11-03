"""
backend/routers/map.py — Geospatial map and overlay router.
Returns only samples that have a colourised prediction mask (blue=safe, red=risky).
If user uploaded samples exist, returns them exclusively with the newest upload first.
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from PIL import Image

from backend.schemas import MapOverlayInfo
from backend.routers.samples import list_samples
from ml.config import cfg

router = APIRouter(prefix="/api", tags=["Geospatial Map"])


def _colormask_path(sample_id: str) -> Optional[Path]:
    """Return the absolute Path of the colourised mask if it exists, else None."""
    p = cfg.OUTPUTS_DIR / "predictions" / f"{sample_id}_colormask.png"
    return p if p.exists() else None


def _ensure_colormask(sample_id: str) -> Optional[str]:
    """
    Ensure the RGBA colourised mask (blue=safe, red=risky) exists for sample_id.
    1. If already on disk, return its URL.
    2. Otherwise, if raw prob/mask numpy arrays exist, generate it.
    3. If still not found, execute fast inference to create it.
    """
    cm_path = cfg.OUTPUTS_DIR / "predictions" / f"{sample_id}_colormask.png"
    if cm_path.exists():
        return f"/outputs/predictions/{sample_id}_colormask.png"

    prob_path = cfg.OUTPUTS_DIR / "predictions" / f"{sample_id}_latest_prob.npy"
    mask_path = cfg.OUTPUTS_DIR / "predictions" / f"{sample_id}_latest_mask.npy"
    if prob_path.exists() and mask_path.exists():
        from backend.utils import colourised_mask_to_png
        prob = np.load(str(prob_path)).astype(np.float32)
        binary = np.load(str(mask_path)).astype(np.uint8)
        colourised_mask_to_png(prob, binary, str(cm_path))
        return f"/outputs/predictions/{sample_id}_colormask.png"

    # For uploaded samples, run inference automatically to guarantee colormask exists
    try:
        from backend.routers.inference import execute_multimodal_inference, InferenceRequest
        execute_multimodal_inference(InferenceRequest(sample_id=sample_id, model_type="multimodal"))
        if cm_path.exists():
            return f"/outputs/predictions/{sample_id}_colormask.png"
    except Exception:
        try:
            from backend.routers.inference import execute_baseline_inference, InferenceRequest
            execute_baseline_inference(InferenceRequest(sample_id=sample_id, model_type="baseline"))
            if cm_path.exists():
                return f"/outputs/predictions/{sample_id}_colormask.png"
        except Exception:
            pass

    return None


@router.get("/map/samples", response_model=List[MapOverlayInfo])
def get_map_overlays():
    """
    Return map overlay info.
    If any user uploaded samples exist, return ONLY the uploaded samples (newest first)
    so the user's uploaded image is shown directly without placeholder/demo interference.
    """
    samples = list_samples()
    overlays: List[MapOverlayInfo] = []

    for s in samples:
        pred_url = _ensure_colormask(s.sample_id)
        if not pred_url:
            continue

        overlays.append(
            MapOverlayInfo(
                sample_id=s.sample_id,
                location_name=s.location_name,
                center_lat=s.center_lat,
                center_lon=s.center_lon,
                bounds=s.bounds,
                sar_tile_url=s.s1_url,
                syneo_tile_url=s.syneo_url,
                prediction_tile_url=pred_url,
                ground_truth_tile_url=s.label_url,
                metrics=None,
            )
        )

    # Prioritize user uploads: if any uploaded samples exist, return ONLY them (newest first)
    uploaded_overlays = [o for o in overlays if o.sample_id.startswith("UPLOAD_")]
    if uploaded_overlays:
        return list(reversed(uploaded_overlays))

    return overlays


@router.get("/map/{sample_id}", response_model=MapOverlayInfo)
def get_map_overlay_for_sample(sample_id: str):
    """Fetch a single overlay; 404 if no prediction mask exists."""
    for o in get_map_overlays():
        if o.sample_id.lower() == sample_id.lower():
            return o
    raise HTTPException(status_code=404, detail=f"No map overlay for '{sample_id}'. Run inference first.")
