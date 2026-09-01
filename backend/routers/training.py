"""
backend/routers/training.py — Training dashboard and telemetry router.
"""

from typing import List
from fastapi import APIRouter
from backend.schemas import TrainingRunInfo

router = APIRouter(prefix="/api", tags=["Training"])

MOCK_RUNS = [
    TrainingRunInfo(
        run_id="RUN_BASELINE_V1",
        model_name="Baseline SAR U-Net",
        status="Completed",
        current_epoch=20,
        total_epochs=20,
        loss=0.3412,
        iou=0.7142,
        learning_rate=0.0001,
        checkpoint_saved="baseline_v1_ep20.pt",
        message="Training completed successfully on 2-channel Sentinel-1 SAR tiles.",
    ),
    TrainingRunInfo(
        run_id="RUN_PIX2PIX_V1",
        model_name="Pix2Pix Generator",
        status="Completed",
        current_epoch=40,
        total_epochs=40,
        loss=0.1895,
        iou=0.0,
        learning_rate=0.0002,
        checkpoint_saved="pix2pix_gen_v1_ep40.pt",
        message="Generator and discriminator trained on paired SAR-optical tiles.",
    ),
    TrainingRunInfo(
        run_id="RUN_MULTIMODAL_V1",
        model_name="Multimodal U-Net (SAR+SynEO)",
        status="Completed",
        current_epoch=20,
        total_epochs=20,
        loss=0.1580,
        iou=0.8420,
        learning_rate=0.0001,
        checkpoint_saved="multimodal_latest.pt",
        message="Multimodal U-Net achieved highest validation IoU (+13.2% net gain).",
    ),
]


@router.get("/training/runs", response_model=List[TrainingRunInfo])
def list_training_runs():
    return MOCK_RUNS


@router.post("/training/start")
def start_training_job():
    return {
        "status": "info",
        "message": "Training scripts (ml/train_*.py) are configured for CLI / GPU node execution. Run via CLI: python -m ml.train_multimodal",
        "command_hint": "python -m ml.train_multimodal --epochs 20",
    }
