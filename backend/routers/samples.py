"""
backend/routers/samples.py — Sample management and GeoTIFF upload endpoints.
"""

import json
import uuid
import warnings
import numpy as np
from pathlib import Path
from typing import List, Optional
import rasterio
from rasterio.transform import from_bounds
from PIL import Image
from fastapi import APIRouter, HTTPException, UploadFile, File

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="rasterio")

from backend.schemas import SampleItem
from backend.utils import get_outputs_dir, sar_to_png, syneo_to_png
from ml.config import cfg

router = APIRouter(prefix="/api", tags=["Samples"])


def _load_sample_manifest() -> List[dict]:
    manifest_path = cfg.DATA_DIR / "samples" / "samples_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as fp:
            return json.load(fp)
    return []


@router.get("/samples", response_model=List[SampleItem])
def list_samples():
    raw_manifest = _load_sample_manifest()
    items = []
    
    out_dir = get_outputs_dir()
    
    for s in raw_manifest:
        sample_id = s["sample_id"]
        s1_path = s["s1_path"]
        lbl_path = s.get("label_path")
        syneo_path = s.get("syneo_path")
        
        # Generate or locate preview PNGs
        s1_png = out_dir / "previews" / f"{sample_id}_sar.png"
        if not s1_png.exists() and Path(s1_path).exists():
            sar_to_png(s1_path, str(s1_png))
            
        syneo_png = out_dir / "previews" / f"{sample_id}_syneo.png"
        if not syneo_png.exists() and syneo_path and Path(syneo_path).exists():
            syneo_to_png(syneo_path, str(syneo_png))

        items.append(
            SampleItem(
                sample_id=sample_id,
                country=s.get("country", "Unknown"),
                chip_id=s.get("chip_id", sample_id),
                location_name=s.get("location_name", f"Region {sample_id}"),
                center_lat=s.get("center_lat", 0.0),
                center_lon=s.get("center_lon", 0.0),
                bounds=s.get("bounds", [0, 0, 1, 1]),
                crs=s.get("crs", "EPSG:4326"),
                dimensions=s.get("dimensions", "256x256"),
                channels=s.get("channels", 2),
                s1_url=f"/outputs/previews/{sample_id}_sar.png",
                label_url=f"/outputs/previews/{sample_id}_label.png" if lbl_path else None,
                syneo_url=f"/outputs/previews/{sample_id}_syneo.png" if syneo_path else None,
                has_ground_truth=Path(lbl_path).exists() if lbl_path else False,
                has_syneo=Path(syneo_path).exists() if syneo_path else False,
            )
        )
    return items


@router.get("/samples/{sample_id}", response_model=SampleItem)
def get_sample_details(sample_id: str):
    samples = list_samples()
    for s in samples:
        if s.sample_id.lower() == sample_id.lower():
            return s
    raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")


@router.post("/upload", response_model=SampleItem)
async def upload_sar_file(file: UploadFile = File(...)):
    """
    Upload a user SAR GeoTIFF or image file (.tif, .tiff, .png, .jpg).
    Validates file format, converts to 2-channel SAR GeoTIFF, saves tile in data/processed/S1/.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload a GeoTIFF (.tif) or image file (.png, .jpg)."
        )

    out_dir = get_outputs_dir()
    upload_path = out_dir / "uploads" / file.filename
    
    content = await file.read()
    with open(upload_path, "wb") as f:
        f.write(content)

    sample_id = f"UPLOAD_{Path(file.filename).stem}"
    target_s1_path = cfg.S1_DIR / f"{sample_id}_S1Hand.tif"
    cfg.S1_DIR.mkdir(parents=True, exist_ok=True)

    crs = "EPSG:4326"
    bounds = [68.0, 26.5, 68.25, 26.75]
    center_lat = 26.625
    center_lon = 68.125
    width, height = 256, 256

    try:
        with rasterio.open(upload_path) as src:
            width, height = src.width, src.height
            if src.crs:
                crs = str(src.crs)
            if src.bounds:
                bounds = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
                center_lat = (src.bounds.bottom + src.bounds.top) / 2.0
                center_lon = (src.bounds.left + src.bounds.right) / 2.0
            sar_data = src.read().astype(np.float32)
    except Exception:
        # Fallback to PIL for non-GeoTIFF standard images
        img = Image.open(upload_path).convert("RGB")
        width, height = img.width, img.height
        arr = np.array(img).astype(np.float32)  # (H, W, 3)
        # Convert RGB pixel intensities to simulated SAR dB backscatter [-30, 0]
        vv = -10.0 - (255.0 - arr[..., 0]) / 255.0 * 20.0
        vh = -16.0 - (255.0 - arr[..., 1]) / 255.0 * 20.0
        sar_data = np.stack([vv, vh], axis=0)

    # Ensure 2 channels (VV, VH)
    if sar_data.ndim == 2:
        vv = sar_data
        vh = sar_data - 5.0
        sar_data = np.stack([vv, vh], axis=0)
    elif sar_data.ndim == 3:
        if sar_data.shape[0] == 1:
            vv = sar_data[0]
            vh = sar_data[0] - 5.0
            sar_data = np.stack([vv, vh], axis=0)
        elif sar_data.shape[0] > 2:
            sar_data = sar_data[:2]

    # Save normalized 2-channel GeoTIFF
    transform = from_bounds(*bounds, width, height)
    with rasterio.open(
        str(target_s1_path),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=2,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(sar_data[0], 1)
        dst.write(sar_data[1], 2)

    # Generate preview PNG
    s1_png = out_dir / "previews" / f"{sample_id}_sar.png"
    sar_to_png(str(target_s1_path), str(s1_png))

    new_item = SampleItem(
        sample_id=sample_id,
        country="User Ingest",
        chip_id=Path(file.filename).stem,
        location_name=f"User Tile: {file.filename}",
        center_lat=center_lat,
        center_lon=center_lon,
        bounds=bounds,
        crs=crs,
        dimensions=f"{width}x{height}",
        channels=2,
        s1_url=f"/outputs/previews/{sample_id}_sar.png",
        label_url=None,
        syneo_url=None,
        has_ground_truth=False,
        has_syneo=False,
    )
    return new_item
