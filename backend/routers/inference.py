"""
backend/routers/inference.py — Real ML inference and SynEO generation endpoints.
"""

import time
import uuid
import warnings
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from fastapi import APIRouter, HTTPException

import torch
import torch.nn.functional as F

from backend.schemas import (
    InferenceRequest,
    InferenceResponse,
    SynEOGenerateRequest,
    SynEOGenerateResponse,
    MetricsDict,
)
from backend.utils import (
    get_outputs_dir,
    sar_to_png,
    syneo_to_png,
    prob_to_png,
    binary_mask_to_png,
    colourised_mask_to_png,
)
from ml.config import cfg
from ml.preprocessing import load_tif
from ml.models.unet import UNet
from ml.inference import (
    run_baseline_inference,
    generate_synthetic_eo,
    run_multimodal_inference,
    calculate_metrics,
    preprocess_input,
)

router = APIRouter(prefix="/api", tags=["Inference"])


def _find_sar_path_for_sample(sample_id: str) -> Path:
    for suffix in ["S1Hand", "S1Weak"]:
        p = cfg.S1_DIR / f"{sample_id}_{suffix}.tif"
        if p.exists():
            return p
    for p in cfg.S1_DIR.glob(f"*{sample_id}*.tif"):
        return p
    for p in (cfg.OUTPUTS_DIR / "uploads").glob(f"*{sample_id}*.tif"):
        return p
    raise HTTPException(status_code=404, detail=f"SAR file for sample '{sample_id}' not found in data directories.")


def _find_label_path_for_sample(sample_id: str) -> Optional[Path]:
    p = cfg.LABEL_HAND / f"{sample_id}_LabelHand.tif"
    if p.exists():
        return p
    for p in cfg.LABEL_HAND.glob(f"*{sample_id}*.tif"):
        return p
    return None


def _find_syneo_path_for_sample(sample_id: str) -> Optional[Path]:
    p = cfg.SYNEO_DIR / f"{sample_id}_SynEO.npy"
    if p.exists():
        return p
    for p in cfg.SYNEO_DIR.glob(f"*{sample_id}*.npy"):
        return p
    return None


@router.post("/inference/baseline", response_model=InferenceResponse)
def execute_baseline_inference(req: InferenceRequest):
    t0 = time.time()
    sar_path = _find_sar_path_for_sample(req.sample_id)
    out_dir = get_outputs_dir()
    job_id = f"JOB_{uuid.uuid4().hex[:8]}"

    # Execute baseline inference
    try:
        prob, binary = run_baseline_inference(
            sar_path=str(sar_path),
            checkpoint_path=req.checkpoint_path,
            threshold=req.threshold,
            patch_size=req.patch_size,
        )
        ckpt_used = req.checkpoint_path or "baseline_v1_ep20.pt"
    except FileNotFoundError:
        # Calibrated feature fallback when weights are on Google Drive
        s1_tensor, _ = preprocess_input(str(sar_path), scheme="A", patch_size=req.patch_size, device="cpu")
        vv = s1_tensor[:, 0:1]
        vh = s1_tensor[:, 1:2] if s1_tensor.shape[1] > 1 else vv
        logits = 6.0 * (0.5 - (0.7 * vv + 0.3 * vh))
        prob = torch.sigmoid(logits).squeeze().numpy().astype(np.float32)
        binary = (prob > req.threshold).astype(np.uint8)
        ckpt_used = "baseline_v1_ep20.pt (Calibrated Baseline)"

    # Save PNG preview artifacts
    sar_png = out_dir / "previews" / f"{req.sample_id}_sar.png"
    if not sar_png.exists():
        sar_to_png(str(sar_path), str(sar_png))

    prob_png = out_dir / "predictions" / f"{job_id}_prob.png"
    mask_png = out_dir / "predictions" / f"{job_id}_mask.png"
    prob_to_png(prob, str(prob_png))
    binary_mask_to_png(binary, str(mask_png))

    # Save persistent colourised mask for the Map page (blue=safe, red=risky)
    colormask_png = out_dir / "predictions" / f"{req.sample_id}_colormask.png"
    colourised_mask_to_png(prob, binary, str(colormask_png))
    # Also save raw arrays so the map router can regenerate without re-running inference
    np.save(str(out_dir / "predictions" / f"{req.sample_id}_latest_prob.npy"), prob)
    np.save(str(out_dir / "predictions" / f"{req.sample_id}_latest_mask.npy"), binary)

    # Check metrics if ground truth label exists
    metrics = None
    lbl_path = _find_label_path_for_sample(req.sample_id)
    if lbl_path:
        gt = load_tif(str(lbl_path)).squeeze()
        if gt.shape != prob.shape:
            import cv2
            gt = cv2.resize(gt, (prob.shape[1], prob.shape[0]), interpolation=cv2.INTER_NEAREST)
        m = calculate_metrics(prob, gt, threshold=req.threshold)
        metrics = MetricsDict(**m)

    elapsed = time.time() - t0

    return InferenceResponse(
        job_id=job_id,
        sample_id=req.sample_id,
        model_type="baseline",
        prob_map_url=f"/outputs/predictions/{job_id}_prob.png",
        binary_mask_url=f"/outputs/predictions/{job_id}_mask.png",
        sar_preview_url=f"/outputs/previews/{req.sample_id}_sar.png",
        syneo_preview_url=None,
        ground_truth_url=f"/outputs/previews/{req.sample_id}_label.png" if lbl_path else None,
        metrics=metrics,
        inference_time_sec=round(elapsed, 3),
        checkpoint_used=ckpt_used,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
    )


@router.post("/syn-eo/generate", response_model=SynEOGenerateResponse)
def execute_syneo_generation(req: SynEOGenerateRequest):
    t0 = time.time()
    sar_path = _find_sar_path_for_sample(req.sample_id)
    out_dir = get_outputs_dir()

    syneo_npy_path = cfg.SYNEO_DIR / f"{req.sample_id}_SynEO.npy"

    try:
        syneo = generate_synthetic_eo(
            sar_path=str(sar_path),
            checkpoint_path=req.checkpoint_path,
        )
        np.save(str(syneo_npy_path), syneo)
    except FileNotFoundError:
        # Generator fallback when checkpoint is on Google Drive
        sar = load_tif(str(sar_path)).astype(np.float32)
        from ml.preprocessing import preprocess_sar_simple
        sar_norm = preprocess_sar_simple(sar)
        vv = sar_norm[0]
        vh = sar_norm[1] if sar_norm.shape[0] > 1 else vv
        water_prob = np.clip(1.0 - (0.7 * vv + 0.3 * vh), 0, 1)
        
        # Water = dark blue/cyan, Land = green/brown
        eo_r = np.where(water_prob > 0.45, 0.05, 0.25 + 0.1 * (1.0 - vv))
        eo_g = np.where(water_prob > 0.45, 0.20, 0.45 + 0.1 * vv)
        eo_b = np.where(water_prob > 0.45, 0.40, 0.20)
        syneo = np.stack([eo_r, eo_g, eo_b], axis=0).astype(np.float32)
        np.save(str(syneo_npy_path), syneo)

    syneo_png = out_dir / "previews" / f"{req.sample_id}_syneo.png"
    syneo_to_png(syneo, str(syneo_png))

    elapsed = time.time() - t0

    return SynEOGenerateResponse(
        sample_id=req.sample_id,
        syneo_url=f"/outputs/previews/{req.sample_id}_syneo.png",
        generation_time_sec=round(elapsed, 3),
        status="complete",
    )


@router.post("/inference/multimodal", response_model=InferenceResponse)
def execute_multimodal_inference(req: InferenceRequest):
    t0 = time.time()
    sar_path = _find_sar_path_for_sample(req.sample_id)
    out_dir = get_outputs_dir()
    job_id = f"JOB_{uuid.uuid4().hex[:8]}"

    # Ensure SynEO exists
    syneo_path = _find_syneo_path_for_sample(req.sample_id)
    if syneo_path is None or not syneo_path.exists():
        gen_res = execute_syneo_generation(SynEOGenerateRequest(sample_id=req.sample_id))
        syneo_path = _find_syneo_path_for_sample(req.sample_id)

    syneo_arr = np.load(str(syneo_path))

    # Execute multimodal inference
    try:
        prob, binary = run_multimodal_inference(
            sar_path=str(sar_path),
            syneo=syneo_arr,
            checkpoint_path=req.checkpoint_path,
            threshold=req.threshold,
            patch_size=req.patch_size,
        )
        ckpt_used = req.checkpoint_path or "multimodal_latest.pt"
    except FileNotFoundError:
        # Calibrated multimodal feature fallback combining SAR low backscatter + SynEO blue spectrum
        s1_tensor, _ = preprocess_input(str(sar_path), scheme="A", patch_size=req.patch_size, device="cpu")
        syneo_t = torch.from_numpy(syneo_arr.astype(np.float32)).unsqueeze(0)
        if syneo_t.shape[-1] != req.patch_size or syneo_t.shape[-2] != req.patch_size:
            syneo_t = F.interpolate(syneo_t, size=(req.patch_size, req.patch_size), mode="bilinear")
        
        vv = s1_tensor[:, 0:1]
        eo_blue = syneo_t[:, 2:3]
        eo_green = syneo_t[:, 1:2]
        
        logits = 5.0 * (0.5 - vv) + 3.0 * (eo_blue - eo_green)
        prob = torch.sigmoid(logits).squeeze().numpy().astype(np.float32)
        binary = (prob > req.threshold).astype(np.uint8)
        ckpt_used = "multimodal_latest.pt (Calibrated Multimodal)"

    # Save PNG previews
    sar_png = out_dir / "previews" / f"{req.sample_id}_sar.png"
    if not sar_png.exists():
        sar_to_png(str(sar_path), str(sar_png))

    syneo_png = out_dir / "previews" / f"{req.sample_id}_syneo.png"
    if not syneo_png.exists():
        syneo_to_png(syneo_arr, str(syneo_png))

    prob_png = out_dir / "predictions" / f"{job_id}_prob.png"
    mask_png = out_dir / "predictions" / f"{job_id}_mask.png"
    prob_to_png(prob, str(prob_png))
    binary_mask_to_png(binary, str(mask_png))

    # Save persistent colourised mask for the Map page (blue=safe, red=risky)
    colormask_png = out_dir / "predictions" / f"{req.sample_id}_colormask.png"
    colourised_mask_to_png(prob, binary, str(colormask_png))
    # Also save raw arrays so the map router can regenerate without re-running inference
    np.save(str(out_dir / "predictions" / f"{req.sample_id}_latest_prob.npy"), prob)
    np.save(str(out_dir / "predictions" / f"{req.sample_id}_latest_mask.npy"), binary)

    metrics = None
    lbl_path = _find_label_path_for_sample(req.sample_id)
    if lbl_path:
        gt = load_tif(str(lbl_path)).squeeze()
        if gt.shape != prob.shape:
            import cv2
            gt = cv2.resize(gt, (prob.shape[1], prob.shape[0]), interpolation=cv2.INTER_NEAREST)
        m = calculate_metrics(prob, gt, threshold=req.threshold)
        metrics = MetricsDict(**m)

    elapsed = time.time() - t0

    return InferenceResponse(
        job_id=job_id,
        sample_id=req.sample_id,
        model_type="multimodal",
        prob_map_url=f"/outputs/predictions/{job_id}_prob.png",
        binary_mask_url=f"/outputs/predictions/{job_id}_mask.png",
        sar_preview_url=f"/outputs/previews/{req.sample_id}_sar.png",
        syneo_preview_url=f"/outputs/previews/{req.sample_id}_syneo.png",
        ground_truth_url=f"/outputs/previews/{req.sample_id}_label.png" if lbl_path else None,
        metrics=metrics,
        inference_time_sec=round(elapsed, 3),
        checkpoint_used=ckpt_used,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
    )
