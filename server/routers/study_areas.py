from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.config import get_config
from server.db import insert_chips
from server.grid import compute_grid
from server.worker_client import WorkerUnavailable, start_batch, get_job_status

router = APIRouter(prefix="/api")


class BBox(BaseModel):
    sw: list[float]
    ne: list[float]


class CreateStudyAreaRequest(BaseModel):
    bbox: BBox
    split: Literal["train", "test", "validate"]


@router.post("/study-areas")
def create_study_area(req: CreateStudyAreaRequest):
    cfg = get_config()
    try:
        chips = compute_grid(
            sw_lonlat=req.bbox.sw,
            ne_lonlat=req.bbox.ne,
            split=req.split,
            chip_size_m=cfg["chip_size_m"],
            crs=cfg["crs"],
            max_chips=cfg["max_chips_per_request"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    insert_chips(chips, crs=cfg["crs"])

    job_id = uuid4().hex[:12]
    try:
        start_batch(job_id, chips, crs=cfg["crs"])
    except WorkerUnavailable:
        return {"ok": True, "count": len(chips), "job_id": None,
                "warning": "Chip worker unavailable — chips inserted but not prefetched"}

    return {"ok": True, "count": len(chips), "job_id": job_id}


@router.get("/prefetch/{job_id}")
def prefetch_status(job_id: str):
    try:
        status = get_job_status(job_id)
    except WorkerUnavailable:
        raise HTTPException(status_code=503, detail="Chip worker unavailable")
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return status
