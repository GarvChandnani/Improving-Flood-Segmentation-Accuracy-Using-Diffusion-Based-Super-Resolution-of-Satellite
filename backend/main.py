"""
backend/main.py — Main FastAPI application entry point for FloodSense.
Serves real ML API endpoints, generated outputs, and the Stitch React frontend UI.
"""

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import cfg
from backend.utils import get_outputs_dir

from backend.routers import (
    health,
    samples,
    models,
    inference,
    evaluation,
    map as map_router,
    training,
)

app = FastAPI(
    title="FloodSense Intelligence API & Dashboard",
    description="Satellite SAR Flood Detection & Multimodal AI Segmentation Backend API",
    version="2.4-satellite-ops",
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(health.router)
app.include_router(samples.router)
app.include_router(models.router)
app.include_router(inference.router)
app.include_router(evaluation.router)
app.include_router(map_router.router)
app.include_router(training.router)

# Serve outputs/ preview files statically
outputs_dir = get_outputs_dir()
app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")

# Mount production React frontend build if present
frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

    @app.get("/")
    def serve_frontend_root():
        return FileResponse(str(frontend_dist / "index.html"))

    @app.get("/{catchall:path}")
    def serve_frontend_catchall(catchall: str):
        # Fallback to index.html for SPA client-side routing if not an API/output route
        if catchall.startswith("api") or catchall.startswith("outputs"):
            return None
        target_file = frontend_dist / catchall
        if target_file.exists() and target_file.is_file():
            return FileResponse(str(target_file))
        return FileResponse(str(frontend_dist / "index.html"))
else:
    @app.get("/")
    def root():
        return {
            "title": "FloodSense Intelligence API",
            "version": "2.4-satellite-ops",
            "docs_url": "/docs",
            "health_url": "/api/health",
            "status": "online",
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
