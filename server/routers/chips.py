import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from server.config import get_config
from server.db import get_all_chips, get_chip_by_id, delete_chips as db_delete_chips
from server.providers import get_chip_image
from server.worker_client import WorkerUnavailable

logger = logging.getLogger(__name__)

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


@router.get("/chips/{chip_id}/image")
def chip_image(chip_id: str):
    chip = get_chip_by_id(chip_id)
    if not chip:
        raise HTTPException(status_code=404, detail=f"Chip '{chip_id}' not found")

    cfg = get_config()
    crs = cfg["crs"]

    try:
        png_bytes = get_chip_image(chip_id, chip["geometry_wkt"], crs)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No image available for chip '{chip_id}'")
    except WorkerUnavailable:
        raise HTTPException(status_code=503, detail="Chip worker unavailable — image not cached")
    except Exception:
        logger.exception("Failed to get chip image for '%s'", chip_id)
        raise HTTPException(status_code=502, detail="Imagery provider failed")

    return Response(content=png_bytes, media_type="image/png")


@router.delete("/chips")
def delete_chips(req: DeleteChipsRequest):
    count = db_delete_chips(req.ids)
    return {"deleted": count}
