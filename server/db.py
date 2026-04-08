import io
import json
import re
import threading
import uuid
from pathlib import Path

import duckdb
import numpy as np
import rasterio.features
from PIL import Image
from rasterio.transform import from_bounds
from shapely.geometry import shape
from shapely.ops import unary_union

from server.config import get_config

_conn = None
_write_lock = threading.Lock()


def init_db():
    global _conn
    cfg = get_config()
    db_path = Path(cfg["db_path"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _conn = duckdb.connect(str(db_path))
    _conn.execute("INSTALL spatial")
    _conn.execute("LOAD spatial")
    _conn.execute("""
        CREATE TABLE IF NOT EXISTS chips (
            id TEXT PRIMARY KEY,
            split TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'unlabeled',
            geometry GEOMETRY NOT NULL
        )
    """)
    _conn.execute("""
        CREATE TABLE IF NOT EXISTS labels (
            id TEXT PRIMARY KEY,
            class TEXT NOT NULL DEFAULT 'positive',
            geometry GEOMETRY NOT NULL
        )
    """)


def get_db():
    if _conn is None:
        raise RuntimeError("Database not initialized — call init_db() first")
    return _conn


def _cursor():
    """Return a thread-local cursor for concurrent reads."""
    return get_db().cursor()


def insert_chips(chips, crs):
    """Batch insert chips, skipping duplicates."""
    conn = get_db()
    with _write_lock:
        for chip in chips:
            conn.execute(
                "INSERT OR IGNORE INTO chips (id, split, status, geometry) VALUES (?, ?, 'unlabeled', ST_GeomFromText(?))",
                [chip["id"], chip["split"], chip["geometry_wkt"]],
            )


def get_all_chips():
    cfg = get_config()
    crs = cfg["crs"]
    rows = _cursor().execute(
        f"SELECT id, split, status, ST_AsGeoJSON(ST_FlipCoordinates(ST_Transform(geometry, '{crs}', 'EPSG:4326'))) AS geojson FROM chips"
    ).fetchall()
    return [
        {"id": r[0], "split": r[1], "status": r[2], "geojson": json.loads(r[3])}
        for r in rows
    ]


def get_chip_by_id(chip_id):
    """Return a chip's id and geometry as WKT, or None if not found."""
    rows = _cursor().execute(
        "SELECT id, ST_AsText(geometry) AS geometry_wkt FROM chips WHERE id = ?",
        [chip_id],
    ).fetchall()
    if not rows:
        return None
    return {"id": rows[0][0], "geometry_wkt": rows[0][1]}


def delete_chips(ids):
    if not ids:
        return 0
    conn = get_db()
    placeholders = ", ".join(["?"] * len(ids))
    with _write_lock:
        result = conn.execute(f"DELETE FROM chips WHERE id IN ({placeholders})", ids)
        return result.fetchone()[0] if result.description else len(ids)


# ── Labels ──────────────────────────────────────────────────────────


def insert_labels(features: list[dict], crs: str):
    """Insert GeoJSON features as label rows, transforming to project CRS."""
    conn = get_db()
    cfg = get_config()
    project_crs = cfg["crs"]

    with _write_lock:
        for feat in features:
            geom_json = json.dumps(feat["geometry"])
            label_id = feat.get("id") or feat.get("properties", {}).get("id")
            label_class = feat.get("properties", {}).get("class", "positive")

            if crs == project_crs:
                conn.execute(
                    "INSERT OR REPLACE INTO labels (id, class, geometry) "
                    "VALUES (?, ?, ST_GeomFromGeoJSON(?))",
                    [label_id, label_class, geom_json],
                )
            else:
                conn.execute(
                    "INSERT OR REPLACE INTO labels (id, class, geometry) "
                    "VALUES (?, ?, ST_Transform(ST_GeomFromGeoJSON(?), ?, ?))",
                    [label_id, label_class, geom_json, crs, project_crs],
                )


def get_all_labels(crs: str = "EPSG:4326") -> list[dict]:
    """Return all labels as GeoJSON-ready dicts."""
    cfg = get_config()
    project_crs = cfg["crs"]
    rows = _cursor().execute(
        f"SELECT id, class, ST_AsGeoJSON(ST_FlipCoordinates(ST_Transform(geometry, '{project_crs}', '{crs}'))) "
        "FROM labels"
    ).fetchall()
    return [
        {"id": r[0], "class": r[1], "geojson": json.loads(r[2])}
        for r in rows
    ]


def delete_labels(ids: list[str]) -> int:
    if not ids:
        return 0
    conn = get_db()
    placeholders = ", ".join(["?"] * len(ids))
    with _write_lock:
        result = conn.execute(f"DELETE FROM labels WHERE id IN ({placeholders})", ids)
        return result.fetchone()[0] if result.description else len(ids)


def save_chip_label(chip_id: str, mask_png_bytes: bytes, label_class: str = "positive"):
    """Vectorize a binary mask PNG and save as a polygon label in DuckDB.

    The mask is georeferenced using the chip's projected geometry from DuckDB.
    Returns the label ID on success.
    """
    chip = get_chip_by_id(chip_id)
    if chip is None:
        raise ValueError(f"Chip {chip_id} not found")

    # Parse chip WKT to extract bounding box
    # WKT vertex order from grid.py: NE, SE, SW, NW, NE (closing)
    # Format: POLYGON((e2 n2, e2 n, e n, e n2, e2 n2))
    wkt = chip["geometry_wkt"]
    coord_text = re.search(r"\(\((.+)\)\)", wkt).group(1)
    vertices = [tuple(map(float, p.strip().split())) for p in coord_text.split(",")]
    eastings = [v[0] for v in vertices]
    northings = [v[1] for v in vertices]
    min_e, max_e = min(eastings), max(eastings)
    min_n, max_n = min(northings), max(northings)

    # Read mask PNG into a numpy array
    img = Image.open(io.BytesIO(mask_png_bytes)).convert("L")
    mask_array = np.array(img)
    # Threshold to binary (any non-zero pixel = 1)
    mask_array = (mask_array > 127).astype(np.uint8)

    height, width = mask_array.shape

    # Reject empty masks
    if mask_array.sum() == 0:
        raise ValueError("Mask is empty — nothing to save")

    # Build affine: pixel (0,0) = top-left = (min_e, max_n)
    transform = from_bounds(min_e, min_n, max_e, max_n, width, height)

    # Vectorize mask to polygons in projected CRS
    polygons = []
    for geom, value in rasterio.features.shapes(mask_array, transform=transform):
        if value == 1:
            polygons.append(shape(geom))

    if not polygons:
        raise ValueError("Mask vectorization produced no polygons")

    merged = unary_union(polygons)
    simplified = merged.simplify(transform.a * 0.5, preserve_topology=True)

    label_id = str(uuid.uuid4())
    conn = get_db()
    with _write_lock:
        conn.execute(
            "INSERT INTO labels (id, class, geometry) "
            "VALUES (?, ?, ST_GeomFromText(?))",
            [label_id, label_class, simplified.wkt],
        )

    return label_id


def get_labels_for_chips(chip_ids: list[str]) -> dict[str, list[tuple[bytes, str]]]:
    """Spatial join: return {chip_id: [(wkb_bytes, class), ...]} for label burning."""
    if not chip_ids:
        return {}
    conn = get_db()
    placeholders = ", ".join(["?"] * len(chip_ids))
    rows = _cursor().execute(
        f"""
        SELECT c.id AS chip_id,
               ST_AsBinary(ST_Intersection(l.geometry, c.geometry)) AS label_geom,
               l.class
        FROM chips c
        JOIN labels l ON ST_Intersects(c.geometry, l.geometry)
        WHERE c.id IN ({placeholders})
        """,
        chip_ids,
    ).fetchall()

    result: dict[str, list[tuple[bytes, str]]] = {}
    for chip_id, wkb, cls in rows:
        result.setdefault(chip_id, []).append((bytes(wkb), cls))
    return result
