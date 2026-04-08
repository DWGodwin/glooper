import base64
import json

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional

from server.config import get_config, get_vectorization_config
from server.db import insert_labels, get_all_labels, get_labels_by_bbox, delete_labels, delete_labels_by_geometry, get_chip_by_id, save_chip_label, parse_mask_for_chip
from server.vectorize import vectorize_mask

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
def list_labels(bbox: Optional[str] = Query(None)):
    if bbox:
        parts = bbox.split(",")
        if len(parts) != 4:
            raise HTTPException(status_code=400, detail="bbox must be 4 comma-separated floats: west,south,east,north")
        try:
            coords = tuple(float(p) for p in parts)
        except ValueError:
            raise HTTPException(status_code=400, detail="bbox values must be numeric")
        labels = get_labels_by_bbox(coords)
    else:
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


class SpatialDeleteRequest(BaseModel):
    point: Optional[list[float]] = None  # [lon, lat]
    bbox: Optional[list[float]] = None   # [west, south, east, north]


@router.delete("/labels/by-geometry")
def delete_labels_spatial(req: SpatialDeleteRequest):
    if not req.point and not req.bbox:
        raise HTTPException(status_code=400, detail="Provide 'point' or 'bbox'")

    from pyproj import Transformer

    cfg = get_config()
    project_crs = cfg["crs"]
    transformer = Transformer.from_crs("EPSG:4326", project_crs, always_xy=True)

    if req.point:
        lon, lat = req.point
        x, y = transformer.transform(lon, lat)
        buf = 1.5  # ~1.5m buffer in projected CRS
        wkt = (
            f"POLYGON(({x - buf} {y - buf}, {x + buf} {y - buf}, "
            f"{x + buf} {y + buf}, {x - buf} {y + buf}, {x - buf} {y - buf}))"
        )
    else:
        west, south, east, north = req.bbox
        min_x, min_y = transformer.transform(west, south)
        max_x, max_y = transformer.transform(east, north)
        wkt = (
            f"POLYGON(({min_x} {min_y}, {max_x} {min_y}, "
            f"{max_x} {max_y}, {min_x} {max_y}, {min_x} {min_y}))"
        )

    count = delete_labels_by_geometry(wkt)
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


class VectorizePreviewRequest(BaseModel):
    mask: str  # base64-encoded PNG
    label_class: str = "positive"
    vectorization: Optional[dict] = None  # per-request config overrides


@router.post("/chips/{chip_id}/vectorize-preview")
def vectorize_preview(chip_id: str, req: VectorizePreviewRequest):
    """Vectorize a mask and return GeoJSON preview without persisting."""
    chip = get_chip_by_id(chip_id)
    if not chip:
        raise HTTPException(status_code=404, detail=f"Chip '{chip_id}' not found")

    try:
        mask_bytes = base64.b64decode(req.mask)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 mask data")

    try:
        mask_array, transform = parse_mask_for_chip(chip_id, mask_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    config = get_vectorization_config(req.label_class)
    if req.vectorization:
        config.update(req.vectorization)

    try:
        geometry = vectorize_mask(mask_array, transform, config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Transform to EPSG:4326 for GeoJSON
    from pyproj import Transformer
    from shapely.ops import transform as shapely_transform

    cfg = get_config()
    project_crs = cfg["crs"]
    transformer = Transformer.from_crs(project_crs, "EPSG:4326", always_xy=True)
    geometry_4326 = shapely_transform(transformer.transform, geometry)
    # Flip to lat/lon for GeoJSON (standard is lon/lat, but our pipeline uses flipped)
    from shapely.geometry import mapping
    geojson_geom = mapping(geometry_4326)

    # Count vertices
    def count_vertices(g):
        if g.geom_type.startswith("Multi"):
            return sum(len(list(p.exterior.coords)) for p in g.geoms)
        if hasattr(g, 'exterior'):
            return len(list(g.exterior.coords))
        return 0

    return {
        "type": "Feature",
        "properties": {"vertex_count": count_vertices(geometry_4326)},
        "geometry": geojson_geom,
    }
