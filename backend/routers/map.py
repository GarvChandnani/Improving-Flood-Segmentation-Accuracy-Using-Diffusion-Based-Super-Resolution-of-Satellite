"""
backend/routers/map.py — Geospatial map and overlay telemetry router.
"""

from typing import List
from fastapi import APIRouter, HTTPException

from backend.schemas import MapOverlayInfo
from backend.routers.samples import list_samples

router = APIRouter(prefix="/api", tags=["Geospatial Map"])


@router.get("/map/samples", response_model=List[MapOverlayInfo])
def get_map_overlays():
    samples = list_samples()
    overlays = []
    
    for s in samples:
        overlays.append(
            MapOverlayInfo(
                sample_id=s.sample_id,
                location_name=s.location_name,
                center_lat=s.center_lat,
                center_lon=s.center_lon,
                bounds=s.bounds,
                sar_tile_url=s.s1_url,
                syneo_tile_url=s.syneo_url,
                prediction_tile_url=s.s1_url,  # replaced when inference executed
                ground_truth_tile_url=s.label_url,
                metrics=None,
            )
        )
    return overlays


@router.get("/map/{sample_id}", response_model=MapOverlayInfo)
def get_map_overlay_for_sample(sample_id: str):
    overlays = get_map_overlays()
    for o in overlays:
        if o.sample_id.lower() == sample_id.lower():
            return o
    raise HTTPException(status_code=404, detail=f"Map overlay for '{sample_id}' not found.")
