from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.config import get_config
from server.db import insert_chips
from server.grid import compute_grid

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

    insert_chips(chips)

    features = []
    for chip in chips:
        features.append({
            "type": "Feature",
            "properties": {
                "id": chip["id"],
                "status": "unlabeled",
                "split": chip["split"],
                "label": None,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [chip["geojson_coords"]],
            },
        })

    return {
        "type": "FeatureCollection",
        "features": features,
    }
