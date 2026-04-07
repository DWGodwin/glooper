"""Export current live data to the demo directory.

Copies pre-cached chip PNGs, SAM embeddings, SAM decoder, and the chip
catalog (from the running API or DuckDB directly) into public/demo/data/
so the static demo build can serve them.

Usage:
    python -m scripts.export_demo                 # reads from DuckDB
    python -m scripts.export_demo --from-api      # reads catalog from running server
"""

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

import duckdb
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
DEMO_DIR = ROOT / "public" / "demo" / "data"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


def export_catalog_from_db(cfg: dict) -> dict:
    """Build the chip catalog GeoJSON from DuckDB.

    Mirrors the transform in server/db.py:get_all_chips().
    """
    db_path = ROOT / cfg["db_path"]
    crs = cfg["crs"]
    conn = duckdb.connect(str(db_path), read_only=True)
    conn.execute("LOAD spatial")

    rows = conn.execute(f"""
        SELECT id, status, split,
               ST_AsGeoJSON(ST_FlipCoordinates(ST_Transform(geometry, '{crs}', 'EPSG:4326'))) AS geojson
        FROM chips
    """).fetchall()
    conn.close()

    features = []
    for chip_id, status, split, geojson_str in rows:
        features.append({
            "type": "Feature",
            "properties": {"id": chip_id, "status": status, "split": split, "label": None},
            "geometry": json.loads(geojson_str),
        })

    return {"type": "FeatureCollection", "features": features}


def export_catalog_from_api(api_base: str) -> dict:
    """Fetch the chip catalog from a running server."""
    url = f"{api_base}/api/chips"
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())


def main():
    parser = argparse.ArgumentParser(description="Export live data to demo directory")
    parser.add_argument("--from-api", action="store_true", help="Fetch catalog from running server instead of DuckDB")
    parser.add_argument("--api-base", default="http://localhost:8000", help="Server URL (with --from-api)")
    args = parser.parse_args()

    cfg = load_config()
    data_dir = ROOT / cfg.get("data_dir", "data")
    chips_dir = data_dir / "chips_png"
    embeddings_dir = data_dir / "sam_embeddings"
    decoder_path = data_dir / "sam_decoder.onnx"

    # Get chip catalog
    if args.from_api:
        print(f"Fetching chip catalog from {args.api_base}...")
        catalog = export_catalog_from_api(args.api_base)
    else:
        print("Reading chip catalog from DuckDB...")
        catalog = export_catalog_from_db(cfg)

    chip_ids = {f["properties"]["id"] for f in catalog["features"]}
    print(f"  {len(chip_ids)} chips in catalog")

    # Set up demo directories
    demo_chips = DEMO_DIR / "chips"
    demo_embeddings = DEMO_DIR / "sam_embeddings"
    demo_chips.mkdir(parents=True, exist_ok=True)
    demo_embeddings.mkdir(parents=True, exist_ok=True)

    # Write catalog
    catalog_path = DEMO_DIR / "metadata.geojson"
    with open(catalog_path, "w") as f:
        json.dump(catalog, f)
    print(f"  Wrote {catalog_path}")

    # Copy chip PNGs
    copied, missing = 0, 0
    for chip_id in sorted(chip_ids):
        src = chips_dir / f"{chip_id}.png"
        if src.exists():
            shutil.copy2(src, demo_chips / f"{chip_id}.png")
            copied += 1
        else:
            missing += 1
    print(f"  Chips: {copied} copied, {missing} missing PNGs")
    if missing:
        print("  (Run the chip worker to generate missing PNGs)")

    # Copy SAM embeddings
    copied, missing = 0, 0
    for chip_id in sorted(chip_ids):
        src = embeddings_dir / f"{chip_id}.npy"
        if src.exists():
            shutil.copy2(src, demo_embeddings / f"{chip_id}.npy")
            copied += 1
        else:
            missing += 1
    print(f"  SAM embeddings: {copied} copied, {missing} missing")

    # Copy SAM decoder
    if decoder_path.exists():
        shutil.copy2(decoder_path, DEMO_DIR / "sam_decoder.onnx")
        print(f"  Copied SAM decoder")
    else:
        print(f"  WARNING: {decoder_path} not found")

    print("Done.")


if __name__ == "__main__":
    main()
