import json
from pathlib import Path

import duckdb

from server.config import get_config

_conn = None


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


def insert_chips(chips, crs):
    """Batch insert chips, skipping duplicates."""
    conn = get_db()
    for chip in chips:
        conn.execute(
            "INSERT OR IGNORE INTO chips (id, split, status, geometry) VALUES (?, ?, 'unlabeled', ST_GeomFromText(?))",
            [chip["id"], chip["split"], chip["geometry_wkt"]],
        )


def get_all_chips():
    conn = get_db()
    cfg = get_config()
    crs = cfg["crs"]
    rows = conn.execute(
        f"SELECT id, split, status, ST_AsGeoJSON(ST_FlipCoordinates(ST_Transform(geometry, '{crs}', 'EPSG:4326'))) AS geojson FROM chips"
    ).fetchall()
    return [
        {"id": r[0], "split": r[1], "status": r[2], "geojson": json.loads(r[3])}
        for r in rows
    ]


def get_chip_by_id(chip_id):
    """Return a chip's id and geometry as WKT, or None if not found."""
    conn = get_db()
    rows = conn.execute(
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
    result = conn.execute(f"DELETE FROM chips WHERE id IN ({placeholders})", ids)
    return result.fetchone()[0] if result.description else len(ids)


# ── Labels ──────────────────────────────────────────────────────────


def insert_labels(features: list[dict], crs: str):
    """Insert GeoJSON features as label rows, transforming to project CRS."""
    conn = get_db()
    cfg = get_config()
    project_crs = cfg["crs"]

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
    conn = get_db()
    cfg = get_config()
    project_crs = cfg["crs"]
    rows = conn.execute(
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
    result = conn.execute(f"DELETE FROM labels WHERE id IN ({placeholders})", ids)
    return result.fetchone()[0] if result.description else len(ids)


def get_labels_for_chips(chip_ids: list[str]) -> dict[str, list[tuple[bytes, str]]]:
    """Spatial join: return {chip_id: [(wkb_bytes, class), ...]} for label burning."""
    if not chip_ids:
        return {}
    conn = get_db()
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
