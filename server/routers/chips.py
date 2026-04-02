from fastapi import APIRouter
from pydantic import BaseModel
from pyproj import Transformer

from server.config import get_config
from server.db import get_all_chips, delete_chips as db_delete_chips

router = APIRouter(prefix="/api")

_transformer = None


def _get_transformer():
    global _transformer
    if _transformer is None:
        cfg = get_config()
        _transformer = Transformer.from_crs(cfg["crs"], "EPSG:4326", always_xy=True)
    return _transformer


def _wkt_to_lonlat_ring(wkt):
    """Parse a simple POLYGON WKT in projected CRS and return a lon/lat coordinate ring."""
    # WKT format: POLYGON((x1 y1, x2 y2, ...))
    inner = wkt.split("((")[1].rstrip("))")
    pairs = inner.split(", ")
    t = _get_transformer()
    ring = []
    for pair in pairs:
        x, y = pair.split()
        lon, lat = t.transform(float(x), float(y))
        ring.append([lon, lat])
    return ring


class DeleteChipsRequest(BaseModel):
    ids: list[str]


@router.get("/chips")
def list_chips():
    chips = get_all_chips()
    features = []
    for chip in chips:
        ring = _wkt_to_lonlat_ring(chip["geometry_wkt"])
        features.append({
            "type": "Feature",
            "properties": {
                "id": chip["id"],
                "status": chip["status"],
                "split": chip["split"],
                "label": None,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [ring],
            },
        })

    return {
        "type": "FeatureCollection",
        "features": features,
    }


@router.delete("/chips")
def delete_chips(req: DeleteChipsRequest):
    count = db_delete_chips(req.ids)
    return {"deleted": count}
