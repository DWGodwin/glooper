import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from server.config import get_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/plugins/cam")


@router.get("/chips/{chip_id}/overlay")
def cam_overlay(chip_id: str):
    cfg = get_config()
    cam_path = Path(cfg["data_dir"]) / "cams" / f"{chip_id}.png"
    if not cam_path.exists():
        raise HTTPException(status_code=404, detail=f"No CAM overlay for chip '{chip_id}'")
    return Response(content=cam_path.read_bytes(), media_type="image/png")


@router.get("/chips/{chip_id}/raw")
def cam_raw(chip_id: str):
    cfg = get_config()
    cam_path = Path(cfg["data_dir"]) / "cams_raw" / f"{chip_id}.npy"
    if not cam_path.exists():
        raise HTTPException(status_code=404, detail=f"No CAM raw data for chip '{chip_id}'")
    return Response(content=cam_path.read_bytes(), media_type="application/octet-stream")


@router.post("/train")
def trigger_training():
    # Future: dispatch to worker
    return {"ok": True, "status": "not_implemented"}


@router.get("/status")
def training_status():
    return {"status": "idle"}
