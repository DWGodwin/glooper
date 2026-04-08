import base64

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.config import get_config
from server.db import insert_labels, get_all_labels, delete_labels, get_chip_by_id, save_chip_label

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


class ChipLabelUpload(BaseModel):
    mask: str  # base64-encoded PNG (single-channel binary mask)
    label_class: str = "positive"


@router.post("/chips/{chip_id}/label")
def upload_chip_label(chip_id: str, req: ChipLabelUpload):
    chip = get_chip_by_id(chip_id)
    if not chip:
        raise HTTPException(status_code=404, detail=f"Chip '{chip_id}' not found")

    try:
        mask_bytes = base64.b64decode(req.mask)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 mask data")

    try:
        label_id = save_chip_label(chip_id, mask_bytes, req.label_class)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"ok": True, "chip_id": chip_id, "label_id": label_id}
