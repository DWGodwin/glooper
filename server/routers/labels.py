from fastapi import APIRouter
from pydantic import BaseModel

from server.db import insert_labels, get_all_labels, delete_labels

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
