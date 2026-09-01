"""
backend/routers/health.py — Health check and system telemetry endpoints.
"""

import time
import torch
from fastapi import APIRouter
from backend.schemas import SystemStatusResponse
from ml.config import cfg

router = APIRouter(prefix="/api", tags=["System"])

_start_time = time.time()


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "version": "2.4-satellite-ops",
        "uptime_sec": round(time.time() - _start_time, 2),
    }


@router.get("/system/status", response_model=SystemStatusResponse)
def get_system_status():
    t0 = time.time()
    
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU (No CUDA GPU)"
    
    vram_used = 0.0
    vram_total = 0.0
    if cuda_avail:
        vram_used = torch.cuda.memory_allocated(0) / (1024 ** 3)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    # Check sample and checkpoint counts
    sample_files = list(cfg.S1_DIR.glob("*.tif"))
    ckpt_files = list(cfg.CHECKPOINT_DIR.rglob("*.pt")) + list(cfg.CHECKPOINT_DIR.rglob("*.ckpt"))
    
    latency = (time.time() - t0) * 1000.0

    return SystemStatusResponse(
        backend_status="online",
        gpu_connected=cuda_avail,
        gpu_name=gpu_name,
        cuda_available=cuda_avail,
        vram_used_gb=round(vram_used, 2),
        vram_total_gb=round(vram_total, 2),
        api_latency_ms=round(latency, 2),
        active_model="Multimodal U-Net",
        active_checkpoint="multimodal_latest.pt",
        model_loaded=True,
        samples_count=len(sample_files),
        checkpoints_count=len(ckpt_files),
    )
