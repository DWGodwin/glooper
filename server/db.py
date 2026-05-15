import datetime
import io
import json
import logging
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

_local = threading.local()
_write_lock = threading.Lock()
_db_path = None

logger = logging.getLogger(__name__)

ACTIVE_RUN_STATUSES = ("queued", "training", "predicting")


def _ensure_schema(conn):
    """Create tables and indexes if they don't exist."""
    conn.execute("INSTALL spatial")
    conn.execute("LOAD spatial")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chips (
            id TEXT PRIMARY KEY,
            split TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'unlabeled',
            geometry GEOMETRY NOT NULL
        )
    """)
    conn.execute(
        "ALTER TABLE chips ADD COLUMN IF NOT EXISTS complete BOOLEAN DEFAULT FALSE"
    )
    conn.execute("""
        CREATE TABLE IF NOT EXISTS labels (
            id TEXT PRIMARY KEY,
            class TEXT NOT NULL DEFAULT 'positive',
            geometry GEOMETRY NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS labels_geom_idx ON labels USING RTREE (geometry)"
    )
    conn.execute("""
        CREATE TABLE IF NOT EXISTS training_runs (
            id TEXT PRIMARY KEY,
            started_at TIMESTAMP NOT NULL,
            completed_at TIMESTAMP,
            status TEXT NOT NULL,
            hyperparams_json TEXT,
            train_loss_curve_json TEXT,
            val_loss_curve_json TEXT,
            model_path TEXT,
            error_message TEXT
        )
    """)
    # Predictions used to be vector polygons; they're now per-chip raster PNGs
    # written to disk, so drop the legacy schema if encountered.
    existing_pred_cols = {
        r[0]
        for r in conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'predictions'"
        ).fetchall()
    }
    if existing_pred_cols and "mask_path" not in existing_pred_cols:
        logger.info("Dropping legacy 'predictions' table (geometry-based schema)")
        conn.execute("DROP TABLE predictions")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            run_id TEXT NOT NULL,
            chip_id TEXT NOT NULL,
            class TEXT NOT NULL DEFAULT 'positive',
            score DOUBLE,
            mask_path TEXT NOT NULL,
            PRIMARY KEY (run_id, chip_id)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS predictions_run_idx ON predictions (run_id)"
    )


def _get_conn():
    """Return a per-thread DuckDB connection, creating one if needed."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.execute("SELECT 1")
            return conn
        except Exception:
            logger.warning("Thread-local DuckDB connection dead — reopening")
            try:
                conn.close()
            except Exception:
                pass
    conn = duckdb.connect(str(_db_path))
    conn.execute("LOAD spatial")
    _local.conn = conn
    return conn


def init_db():
    global _db_path
    cfg = get_config()
    _db_path = Path(cfg["db_path"])
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    # Bootstrap schema on the first connection
    conn = duckdb.connect(str(_db_path))
    _ensure_schema(conn)
    conn.close()


def insert_chips(chips, crs):
    """Batch insert chips, skipping duplicates."""
    conn = _get_conn()
    with _write_lock:
        for chip in chips:
            conn.execute(
                "INSERT OR IGNORE INTO chips (id, split, status, geometry) VALUES (?, ?, 'unlabeled', ST_GeomFromText(?))",
                [chip["id"], chip["split"], chip["geometry_wkt"]],
            )


def get_all_chips():
    cfg = get_config()
    crs = cfg["crs"]
    rows = _get_conn().execute(
        f"SELECT id, split, status, complete, ST_AsGeoJSON(ST_FlipCoordinates(ST_Transform(geometry, '{crs}', 'EPSG:4326'))) AS geojson FROM chips"
    ).fetchall()
    return [
        {"id": r[0], "split": r[1], "status": r[2], "complete": bool(r[3]), "geojson": json.loads(r[4])}
        for r in rows
    ]


def get_chip_by_id(chip_id):
    """Return a chip's id and geometry as WKT, or None if not found."""
    rows = _get_conn().execute(
        "SELECT id, ST_AsText(geometry) AS geometry_wkt FROM chips WHERE id = ?",
        [chip_id],
    ).fetchall()
    if not rows:
        return None
    return {"id": rows[0][0], "geometry_wkt": rows[0][1]}


def delete_chips(ids):
    if not ids:
        return 0
    conn = _get_conn()
    placeholders = ", ".join(["?"] * len(ids))
    with _write_lock:
        result = conn.execute(f"DELETE FROM chips WHERE id IN ({placeholders})", ids)
        return result.fetchone()[0] if result.description else len(ids)


# ── Labels ──────────────────────────────────────────────────────────


def insert_labels(features: list[dict], crs: str):
    """Insert GeoJSON features as label rows, transforming to project CRS."""
    conn = _get_conn()
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
                    "VALUES (?, ?, ST_Transform(ST_GeomFromGeoJSON(?), ?, ?, always_xy := true))",
                    [label_id, label_class, geom_json, crs, project_crs],
                )


def get_all_labels(crs: str = "EPSG:4326") -> list[dict]:
    """Return all labels as GeoJSON-ready dicts."""
    cfg = get_config()
    project_crs = cfg["crs"]
    rows = _get_conn().execute(
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

    rows = _get_conn().execute(
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
    conn = _get_conn()
    placeholders = ", ".join(["?"] * len(ids))
    with _write_lock:
        result = conn.execute(f"DELETE FROM labels WHERE id IN ({placeholders})", ids)
        return result.fetchone()[0] if result.description else len(ids)


def delete_labels_by_geometry(geometry_wkt: str) -> int:
    """Delete all labels intersecting the given geometry (in project CRS)."""
    conn = _get_conn()
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
    conn = _get_conn()
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


def mark_chip_complete(chip_id: str, complete: bool = True) -> bool:
    """Set the chip's 'complete' flag. Returns False if the chip does not exist."""
    conn = _get_conn()
    with _write_lock:
        if not conn.execute("SELECT 1 FROM chips WHERE id = ?", [chip_id]).fetchone():
            return False
        conn.execute(
            "UPDATE chips SET complete = ? WHERE id = ?",
            [complete, chip_id],
        )
    return True


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
    conn = _get_conn()
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


def get_training_chips() -> dict:
    """Return chip IDs eligible for training, split by partition.

    Only chips marked complete (exhaustively labeled) participate in training
    or validation — sparse/partial chips are excluded so loss is computed over
    every pixel without needing a per-pixel loss mask.
    """
    conn = _get_conn()
    train = conn.execute(
        "SELECT id FROM chips WHERE split = 'train' AND complete = TRUE"
    ).fetchall()
    validate = conn.execute(
        "SELECT id FROM chips WHERE split = 'validate' AND complete = TRUE"
    ).fetchall()
    return {
        "train": [{"id": r[0]} for r in train],
        "validate": [{"id": r[0]} for r in validate],
    }


def get_labels_for_chips(chip_ids: list[str]) -> dict[str, list[tuple[bytes, str]]]:
    """Spatial join: return {chip_id: [(wkb_bytes, class), ...]} for label burning."""
    if not chip_ids:
        return {}
    conn = _get_conn()
    placeholders = ", ".join(["?"] * len(chip_ids))
    rows = conn.execute(
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


# ── Training runs ───────────────────────────────────────────────────


_RUN_COLUMNS = (
    "id, started_at, completed_at, status, hyperparams_json, "
    "train_loss_curve_json, val_loss_curve_json, model_path, error_message"
)

_CURVE_FIELDS = ("train_loss_curve_json", "val_loss_curve_json")


def _parse_run_row(row) -> dict | None:
    if row is None:
        return None
    id_, started_at, completed_at, status, hp, train_curve, val_curve, model_path, err = row
    return {
        "id": id_,
        "started_at": started_at,
        "completed_at": completed_at,
        "status": status,
        "hyperparams": json.loads(hp) if hp else None,
        "train_loss_curve": json.loads(train_curve) if train_curve else None,
        "val_loss_curve": json.loads(val_curve) if val_curve else None,
        "model_path": model_path,
        "error_message": err,
    }


def create_training_run(run_id: str, hyperparams_json: str | None = None) -> None:
    """Insert a new training run row in 'queued' state."""
    conn = _get_conn()
    started_at = datetime.datetime.now(datetime.timezone.utc)
    with _write_lock:
        conn.execute(
            "INSERT INTO training_runs (id, started_at, status, hyperparams_json) "
            "VALUES (?, ?, 'queued', ?)",
            [run_id, started_at, hyperparams_json],
        )


def update_training_run(run_id: str, **fields) -> None:
    """Partial update; list values for *_loss_curve_json fields are JSON-serialized."""
    if not fields:
        return
    cleaned = {}
    for k, v in fields.items():
        if k in _CURVE_FIELDS and isinstance(v, list):
            cleaned[k] = json.dumps(v)
        else:
            cleaned[k] = v
    set_clause = ", ".join(f"{k} = ?" for k in cleaned)
    values = list(cleaned.values()) + [run_id]
    conn = _get_conn()
    with _write_lock:
        conn.execute(
            f"UPDATE training_runs SET {set_clause} WHERE id = ?",
            values,
        )


def list_training_runs(limit: int = 20) -> list[dict]:
    """Return most-recent-first list of training runs."""
    rows = _get_conn().execute(
        f"SELECT {_RUN_COLUMNS} FROM training_runs ORDER BY started_at DESC LIMIT ?",
        [limit],
    ).fetchall()
    return [_parse_run_row(r) for r in rows]


def get_training_run(run_id: str) -> dict | None:
    row = _get_conn().execute(
        f"SELECT {_RUN_COLUMNS} FROM training_runs WHERE id = ?",
        [run_id],
    ).fetchone()
    return _parse_run_row(row)


def get_active_training_run() -> dict | None:
    placeholders = ", ".join(["?"] * len(ACTIVE_RUN_STATUSES))
    row = _get_conn().execute(
        f"SELECT {_RUN_COLUMNS} FROM training_runs "
        f"WHERE status IN ({placeholders}) "
        "ORDER BY started_at DESC LIMIT 1",
        list(ACTIVE_RUN_STATUSES),
    ).fetchone()
    return _parse_run_row(row)


def delete_training_run(run_id: str) -> None:
    """Delete a training run, its prediction rows, and cached mask PNGs on disk."""
    import shutil

    conn = _get_conn()
    with _write_lock:
        conn.execute("DELETE FROM predictions WHERE run_id = ?", [run_id])
        conn.execute("DELETE FROM training_runs WHERE id = ?", [run_id])
    shutil.rmtree(_predictions_dir(run_id), ignore_errors=True)


# ── Predictions ─────────────────────────────────────────────────────


def _predictions_dir(run_id: str) -> Path:
    cfg = get_config()
    return Path(cfg["data_dir"]) / "predictions" / run_id


def save_prediction_mask(
    run_id: str,
    chip_id: str,
    score: float,
    png_bytes: bytes,
    label_class: str = "positive",
) -> str:
    """Write a per-chip prediction mask PNG to disk and upsert the predictions row.

    Returns the absolute file path that was written.
    """
    out_dir = _predictions_dir(run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / f"{chip_id}.png"
    mask_path.write_bytes(png_bytes)

    conn = _get_conn()
    with _write_lock:
        conn.execute(
            "INSERT OR REPLACE INTO predictions (run_id, chip_id, class, score, mask_path) "
            "VALUES (?, ?, ?, ?, ?)",
            [run_id, chip_id, label_class, score, str(mask_path)],
        )
    return str(mask_path)


def get_prediction_mask_path(run_id: str, chip_id: str) -> str | None:
    """Return the mask file path for a (run, chip) pair, or None if not predicted."""
    row = _get_conn().execute(
        "SELECT mask_path FROM predictions WHERE run_id = ? AND chip_id = ?",
        [run_id, chip_id],
    ).fetchone()
    return row[0] if row else None


def get_predictions_by_bbox(run_id: str, bbox_lonlat: tuple) -> list[dict]:
    """Return predictions in `run_id` whose chip intersects an EPSG:4326 bbox.

    Spatial filtering uses the chip footprint (a join), since predictions are
    chip-aligned rasters with no independent geometry.
    """
    from pyproj import Transformer

    cfg = get_config()
    project_crs = cfg["crs"]
    transformer = Transformer.from_crs("EPSG:4326", project_crs, always_xy=True)

    west, south, east, north = bbox_lonlat
    min_x, min_y = transformer.transform(west, south)
    max_x, max_y = transformer.transform(east, north)
    envelope_wkt = (
        f"POLYGON(({min_x} {min_y}, {max_x} {min_y}, "
        f"{max_x} {max_y}, {min_x} {max_y}, {min_x} {min_y}))"
    )

    rows = _get_conn().execute(
        "SELECT p.chip_id, p.class, p.score "
        "FROM predictions p JOIN chips c ON c.id = p.chip_id "
        "WHERE p.run_id = ? AND ST_Intersects(c.geometry, ST_GeomFromText(?))",
        [run_id, envelope_wkt],
    ).fetchall()
    return [
        {"chip_id": r[0], "class": r[1], "score": r[2]}
        for r in rows
    ]
