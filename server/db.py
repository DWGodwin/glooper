import io
import json
import re
import threading
import uuid
from pathlib import Path

import duckdb
import numpy as np
from PIL import Image
from rasterio.transform import from_bounds

from server.config import get_config, get_vectorization_config
from server.vectorize import vectorize_mask

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
    _conn.execute(
        "CREATE INDEX IF NOT EXISTS labels_geom_idx ON labels USING RTREE (geometry)"
    )


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


def get_labels_by_bbox(bbox: tuple[float, float, float, float]) -> list[dict]:
    """Return labels intersecting the given EPSG:4326 bbox (west, south, east, north)."""
    from pyproj import Transformer

    cfg = get_config()
    project_crs = cfg["crs"]
    transformer = Transformer.from_crs("EPSG:4326", project_crs, always_xy=True)

    west, south, east, north = bbox
    min_x, min_y = transformer.transform(west, south)
    max_x, max_y = transformer.transform(east, north)

    envelope_wkt = (
        f"POLYGON(({min_x} {min_y}, {max_x} {min_y}, "
        f"{max_x} {max_y}, {min_x} {max_y}, {min_x} {min_y}))"
    )

    rows = _cursor().execute(
        f"SELECT id, class, ST_AsGeoJSON(ST_FlipCoordinates(ST_Transform(geometry, '{project_crs}', 'EPSG:4326'))) "
        "FROM labels WHERE ST_Intersects(geometry, ST_GeomFromText(?))",
        [envelope_wkt],
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


def delete_labels_by_geometry(geometry_wkt: str) -> int:
    """Delete all labels intersecting the given geometry (in project CRS)."""
    conn = get_db()
    with _write_lock:
        ids = conn.execute(
            "SELECT id FROM labels WHERE ST_Intersects(geometry, ST_GeomFromText(?))",
            [geometry_wkt],
        ).fetchall()
        if not ids:
            return 0
        id_list = [r[0] for r in ids]
        placeholders = ", ".join(["?"] * len(id_list))
        conn.execute(f"DELETE FROM labels WHERE id IN ({placeholders})", id_list)
        return len(id_list)


def parse_mask_for_chip(chip_id: str, mask_png_bytes: bytes):
    """Parse a binary mask PNG and georeference it using the chip's geometry.

    Returns (mask_array, transform) where mask_array is binary uint8 and
    transform is a rasterio Affine mapping pixels to projected CRS.
    """
    chip = get_chip_by_id(chip_id)
    if chip is None:
        raise ValueError(f"Chip {chip_id} not found")

    wkt = chip["geometry_wkt"]
    coord_text = re.search(r"\(\((.+)\)\)", wkt).group(1)
    vertices = [tuple(map(float, p.strip().split())) for p in coord_text.split(",")]
    eastings = [v[0] for v in vertices]
    northings = [v[1] for v in vertices]
    min_e, max_e = min(eastings), max(eastings)
    min_n, max_n = min(northings), max(northings)

    img = Image.open(io.BytesIO(mask_png_bytes)).convert("L")
    mask_array = np.array(img)
    mask_array = (mask_array > 127).astype(np.uint8)

    height, width = mask_array.shape

    if mask_array.sum() == 0:
        raise ValueError("Mask is empty — nothing to save")

    transform = from_bounds(min_e, min_n, max_e, max_n, width, height)
    return mask_array, transform


def save_chip_label(chip_id: str, mask_png_bytes: bytes, label_class: str = "positive"):
    """Vectorize a binary mask PNG and save as a polygon label in DuckDB.

    Returns the label ID on success.
    """
    mask_array, transform = parse_mask_for_chip(chip_id, mask_png_bytes)
    config = get_vectorization_config(label_class)
    geometry = vectorize_mask(mask_array, transform, config)

    label_id = str(uuid.uuid4())
    conn = get_db()
    with _write_lock:
        conn.execute(
            "INSERT INTO labels (id, class, geometry) "
            "VALUES (?, ?, ST_GeomFromText(?))",
            [label_id, label_class, geometry.wkt],
        )
        conn.execute(
            "UPDATE chips SET status = 'labeled' WHERE id = ?",
            [chip_id],
        )

    return label_id


def _delete_chip_files(chip_ids: list[str]):
    """Remove cached files (images, embeddings, features) for deleted chips."""
    cfg = get_config()
    data_dir = Path(cfg["data_dir"])
    dirs_and_exts = [
        (data_dir / "chips", ".tif"),
        (data_dir / "chips_png", ".png"),
        (data_dir / "sam_embeddings", ".npy"),
    ]
    for chip_id in chip_ids:
        for directory, ext in dirs_and_exts:
            path = directory / f"{chip_id}{ext}"
            path.unlink(missing_ok=True)


def delete_chips_by_geometry(geometry_wkt: str) -> dict:
    """Delete chips intersecting geometry, cascade-deleting their labels first.

    Returns {"chips_deleted": int, "labels_deleted": int}.
    """
    conn = get_db()
    with _write_lock:
        # Find intersecting chips
        chip_rows = conn.execute(
            "SELECT id FROM chips WHERE ST_Intersects(geometry, ST_GeomFromText(?))",
            [geometry_wkt],
        ).fetchall()
        if not chip_rows:
            return {"chips_deleted": 0, "labels_deleted": 0}

        chip_ids = [r[0] for r in chip_rows]
        placeholders = ", ".join(["?"] * len(chip_ids))

        # Delete labels that intersect these chips (must happen before chip deletion)
        label_rows = conn.execute(
            f"""
            SELECT DISTINCT l.id FROM labels l
            JOIN chips c ON ST_Intersects(c.geometry, l.geometry)
            WHERE c.id IN ({placeholders})
            """,
            chip_ids,
        ).fetchall()
        labels_deleted = 0
        if label_rows:
            label_ids = [r[0] for r in label_rows]
            lp = ", ".join(["?"] * len(label_ids))
            conn.execute(f"DELETE FROM labels WHERE id IN ({lp})", label_ids)
            labels_deleted = len(label_ids)

        # Delete chips
        conn.execute(f"DELETE FROM chips WHERE id IN ({placeholders})", chip_ids)

    # Clean up cached files outside the write lock
    _delete_chip_files(chip_ids)

    return {"chips_deleted": len(chip_ids), "labels_deleted": labels_deleted}


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
