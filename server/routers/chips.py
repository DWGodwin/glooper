from fastapi import APIRouter
from pydantic import BaseModel

from server.db import get_all_chips, delete_chips as db_delete_chips

router = APIRouter(prefix="/api")


class DeleteChipsRequest(BaseModel):
    ids: list[str]


@router.get("/chips")
def list_chips():
    chips = get_all_chips()
    features = []
    for chip in chips:
        features.append({
            "type": "Feature",
            "properties": {
                "id": chip["id"],
                "status": chip["status"],
                "split": chip["split"],
                "label": None,
            },
            "geometry": chip["geojson"],
        })

    return {
        "type": "FeatureCollection",
        "features": features,
    }


@router.delete("/chips")
def delete_chips(req: DeleteChipsRequest):
    count = db_delete_chips(req.ids)
    return {"deleted": count}
