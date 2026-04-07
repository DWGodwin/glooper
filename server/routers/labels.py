import base64
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.config import get_config
from server.db import insert_labels, get_all_labels, delete_labels, get_chip_by_id

router = APIRouter(prefix="/api")


class LabelUpload(BaseModel):
    type: str = "FeatureCollection"
    features: list[dict]
    crs: str = "EPSG:4326"


class DeleteLabelsRequest(BaseModel):
    ids: list[str]


@router.post("/labels")
def create_labels(req: LabelUpload):
    insert_labels(req.features, crs=req.crs)
    return {"ok": True, "count": len(req.features)}


@router.get("/labels")
def list_labels():
    labels = get_all_labels()
    features = []
    for lab in labels:
        features.append({
            "type": "Feature",
            "properties": {"id": lab["id"], "class": lab["class"]},
            "geometry": lab["geojson"],
        })
    return {"type": "FeatureCollection", "features": features}


@router.delete("/labels")
def remove_labels(req: DeleteLabelsRequest):
    count = delete_labels(req.ids)
    return {"deleted": count}


class MaskUpload(BaseModel):
    chip_id: str
    mask: str  # base64-encoded PNG (512x512 single-channel)
    split: str = "train"


@router.post("/labels/mask")
def upload_mask(req: MaskUpload):
    chip = get_chip_by_id(req.chip_id)
    if not chip:
        raise HTTPException(status_code=404, detail=f"Chip {req.chip_id} not found")

    labels_dir = Path("data/labels")
    labels_dir.mkdir(parents=True, exist_ok=True)

    mask_bytes = base64.b64decode(req.mask)
    mask_path = labels_dir / f"{req.chip_id}.png"
    mask_path.write_bytes(mask_bytes)

    return {"ok": True, "chip_id": req.chip_id, "path": str(mask_path)}
