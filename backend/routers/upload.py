"""
backend/routers/upload.py — Handles SAR file uploads, stores the file, creates a preview PNG,
and registers the new sample in the samples manifest so it persists across page navigation.
"""

import os
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from backend.utils import get_outputs_dir, sar_to_png
from backend.routers.samples import _load_sample_manifest

router = APIRouter(prefix="/api", tags=["Upload"])

MANIFEST_PATH = Path(os.getenv("FLOOD_PROJECT_ROOT", ".")) / "data" / "samples" / "samples_manifest.json"

def _save_to_manifest(item: Dict) -> None:
    """Append a new sample entry to the manifest JSON.
    The manifest is a list of dicts; we load, append, and write back.
    """
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    else:
        data = []
    data.append(item)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)

@router.post("/upload")
async def upload_sar_file(file: UploadFile = File(...)):
    """Accept a SAR TIFF/GeoTIFF, store it, generate a preview PNG and register the sample.
    Returns the newly created ``SampleItem`` JSON.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    # Save raw file
    out_dir = get_outputs_dir()
    uploads_dir = out_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    raw_path = uploads_dir / file.filename
    with open(raw_path, "wb") as f:
        contents = await file.read()
        f.write(contents)
    # Generate preview PNG for quick UI display
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    sample_id = Path(file.filename).stem
    preview_png = preview_dir / f"{sample_id}_sar.png"
    try:
        sar_to_png(str(raw_path), str(preview_png))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate preview: {e}")
    # Build manifest entry
    sample_entry = {
        "sample_id": sample_id,
        "country": "UserUpload",
        "chip_id": sample_id,
        "location_name": f"Uploaded {sample_id}",
        "center_lat": 0.0,
        "center_lon": 0.0,
        "bounds": [0, 0, 1, 1],
        "crs": "EPSG:4326",
        "dimensions": "256x256",
        "channels": 2,
        "s1_path": str(raw_path),
        "s1_url": f"/outputs/previews/{sample_id}_sar.png",
        "label_path": None,
        "label_url": None,
        "syneo_path": None,
        "syneo_url": None,
    }
    _save_to_manifest(sample_entry)
    # Return the same shape as SampleItem (frontend expects those fields)
    return JSONResponse(content=sample_entry)

