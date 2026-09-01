"""
backend/schemas.py — Pydantic schemas for the FloodSense FastAPI backend.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class SystemStatusResponse(BaseModel):
    backend_status: str = "online"
    gpu_connected: bool
    gpu_name: str
    cuda_available: bool
    vram_used_gb: float
    vram_total_gb: float
    api_latency_ms: float
    active_model: str
    active_checkpoint: str
    model_loaded: bool
    samples_count: int
    checkpoints_count: int


class SampleItem(BaseModel):
    sample_id: str
    country: str
    chip_id: str
    location_name: str
    center_lat: float
    center_lon: float
    bounds: List[float]
    crs: str
    dimensions: str
    channels: int
    s1_url: str
    label_url: Optional[str] = None
    syneo_url: Optional[str] = None
    has_ground_truth: bool
    has_syneo: bool


class ModelInfo(BaseModel):
    model_id: str
    name: str
    architecture: str
    in_channels: int
    out_channels: int
    default_checkpoint: str
    status: str
    description: str


class CheckpointInfo(BaseModel):
    filename: str
    model_type: str
    size_mb: float
    sha256: str
    path: str
    modified_time: str
    active: bool = False


class CheckpointLoadRequest(BaseModel):
    model_type: str
    checkpoint_path: str


class InferenceRequest(BaseModel):
    sample_id: str
    model_type: str = "baseline"  # "baseline" or "multimodal"
    checkpoint_path: Optional[str] = None
    threshold: float = 0.5
    patch_size: int = 256


class MetricsDict(BaseModel):
    iou: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    tp: Optional[int] = None
    fp: Optional[int] = None
    fn: Optional[int] = None
    tn: Optional[int] = None


class InferenceResponse(BaseModel):
    job_id: str
    sample_id: str
    model_type: str
    prob_map_url: str
    binary_mask_url: str
    sar_preview_url: str
    syneo_preview_url: Optional[str] = None
    ground_truth_url: Optional[str] = None
    metrics: Optional[MetricsDict] = None
    inference_time_sec: float
    checkpoint_used: str
    timestamp: str


class SynEOGenerateRequest(BaseModel):
    sample_id: str
    checkpoint_path: Optional[str] = None


class SynEOGenerateResponse(BaseModel):
    sample_id: str
    syneo_url: str
    generation_time_sec: float
    status: str


class EvaluationRequest(BaseModel):
    sample_id: Optional[str] = None
    threshold: float = 0.5


class RegionalMetrics(BaseModel):
    region: str
    iou: float
    precision: float
    recall: float
    f1: float


class EvaluationResponse(BaseModel):
    run_id: str
    sample_id: str
    location_name: str
    baseline_metrics: MetricsDict
    multimodal_metrics: MetricsDict
    iou_baseline: float
    iou_multimodal: float
    iou_improvement_percent: float
    visual_ground_truth_url: str
    visual_baseline_pred_url: str
    visual_multimodal_pred_url: str
    visual_error_diff_url: str
    regional_breakdown: List[RegionalMetrics]


class MapOverlayInfo(BaseModel):
    sample_id: str
    location_name: str
    center_lat: float
    center_lon: float
    bounds: List[float]
    sar_tile_url: str
    syneo_tile_url: Optional[str] = None
    prediction_tile_url: Optional[str] = None
    ground_truth_tile_url: Optional[str] = None
    metrics: Optional[MetricsDict] = None


class TrainingRunInfo(BaseModel):
    run_id: str
    model_name: str
    status: str
    current_epoch: int
    total_epochs: int
    loss: float
    iou: float
    learning_rate: float
    checkpoint_saved: str
    message: str
