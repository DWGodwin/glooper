"""DuckDB CRUD layer tests.

These use the `db` fixture (which writes a tmp config and runs init_db).
Each test gets a fresh DB file under tmp_path.
"""
import io
import re

import numpy as np
import pytest
from PIL import Image
from pyproj import Transformer

from server import db as db_mod
from server.grid import compute_grid


CRS = "EPSG:32619"
SW = [-72.6, 42.2]
NE = [-72.598, 42.202]
CHIP_SIZE_M = 67.2


def _make_chips(split="train"):
    return compute_grid(
        sw_lonlat=SW, ne_lonlat=NE, split=split,
        chip_size_m=CHIP_SIZE_M, crs=CRS,
    )


def _project_4326_bbox(west, south, east, north):
    t = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    min_x, min_y = t.transform(west, south)
    max_x, max_y = t.transform(east, north)
    return min_x, min_y, max_x, max_y


def _label_feature_projected(min_x, min_y, max_x, max_y, label_id="lab-1", cls="positive"):
    """Build a GeoJSON Polygon feature whose coordinates are in the project CRS."""
    return {
        "type": "Feature",
        "id": label_id,
        "properties": {"class": cls},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y],
                [min_x, min_y],
            ]],
        },
    }


def test_insert_and_get_all_chips_round_trip(db):
    chips = _make_chips()
    db_mod.insert_chips(chips, crs=CRS)

    fetched = db_mod.get_all_chips()
    assert len(fetched) == len(chips)
    assert {c["id"] for c in fetched} == {c["id"] for c in chips}
    assert all(c["split"] == "train" for c in fetched)
    assert all(c["status"] == "unlabeled" for c in fetched)

    # GeoJSON should be lon/lat order
    geom = fetched[0]["geojson"]
    assert geom["type"] == "Polygon"
    lon, lat = geom["coordinates"][0][0]
    assert -73 < lon < -72
    assert 42 < lat < 43


def test_insert_chips_is_idempotent(db):
    chips = _make_chips()
    db_mod.insert_chips(chips, crs=CRS)
    db_mod.insert_chips(chips, crs=CRS)

    fetched = db_mod.get_all_chips()
    assert len(fetched) == len(chips)


def test_delete_chips_removes_rows(db):
    chips = _make_chips()
    db_mod.insert_chips(chips, crs=CRS)

    target = chips[0]["id"]
    db_mod.delete_chips([target])

    remaining = {c["id"] for c in db_mod.get_all_chips()}
    assert target not in remaining
    assert len(remaining) == len(chips) - 1


def test_delete_chips_nonexistent_is_noop(db):
    chips = _make_chips()
    db_mod.insert_chips(chips, crs=CRS)
    before = len(db_mod.get_all_chips())

    db_mod.delete_chips(["does_not_exist"])

    assert len(db_mod.get_all_chips()) == before


def test_get_chip_by_id_roundtrip(db):
    chips = _make_chips()
    db_mod.insert_chips(chips, crs=CRS)

    target_id = chips[0]["id"]
    got = db_mod.get_chip_by_id(target_id)
    assert got is not None
    assert got["id"] == target_id
    assert "POLYGON" in got["geometry_wkt"]


def test_get_chip_by_id_missing_returns_none(db):
    assert db_mod.get_chip_by_id("missing") is None


def test_insert_and_get_labels_project_crs_round_trip(db):
    min_x, min_y, max_x, max_y = _project_4326_bbox(-72.5995, 42.2005, -72.5990, 42.2010)
    feat = _label_feature_projected(min_x, min_y, max_x, max_y)
    db_mod.insert_labels([feat], crs=CRS)

    got = db_mod.get_all_labels(crs="EPSG:4326")
    assert len(got) == 1
    assert got[0]["id"] == "lab-1"
    assert got[0]["class"] == "positive"

    coords = got[0]["geojson"]["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    # Round-trip 4326→32619→4326 leaves ~3m of drift between pyproj's and
    # DuckDB's PROJ grids; ~1e-4 deg (~11m) is still tight enough to catch
    # axis swaps without false positives.
    assert min(lons) == pytest.approx(-72.5995, abs=1e-4)
    assert max(lons) == pytest.approx(-72.5990, abs=1e-4)
    assert min(lats) == pytest.approx(42.2005, abs=1e-4)
    assert max(lats) == pytest.approx(42.2010, abs=1e-4)


def test_insert_labels_with_4326_crs_round_trip(db):
    feat = {
        "type": "Feature",
        "id": "lab-4326",
        "properties": {"class": "positive"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-72.5995, 42.2005],
                [-72.5990, 42.2005],
                [-72.5990, 42.2010],
                [-72.5995, 42.2010],
                [-72.5995, 42.2005],
            ]],
        },
    }
    db_mod.insert_labels([feat], crs="EPSG:4326")
    got = db_mod.get_all_labels(crs="EPSG:4326")
    coords = got[0]["geojson"]["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    assert min(lons) == pytest.approx(-72.5995, abs=1e-5)
    assert min(lats) == pytest.approx(42.2005, abs=1e-5)


def test_get_labels_by_bbox_filters_spatially(db):
    inside_bbox = _project_4326_bbox(-72.5995, 42.2005, -72.5990, 42.2010)
    outside_bbox = _project_4326_bbox(-72.5905, 42.2105, -72.5900, 42.2110)
    db_mod.insert_labels(
        [
            _label_feature_projected(*inside_bbox, label_id="in"),
            _label_feature_projected(*outside_bbox, label_id="out"),
        ],
        crs=CRS,
    )

    inside = db_mod.get_labels_by_bbox((-72.5996, 42.2004, -72.5989, 42.2011))
    ids = {l["id"] for l in inside}
    assert ids == {"in"}


def test_delete_labels_by_id(db):
    a = _project_4326_bbox(-72.5995, 42.2005, -72.5990, 42.2010)
    b = _project_4326_bbox(-72.5985, 42.2015, -72.5980, 42.2020)
    db_mod.insert_labels(
        [
            _label_feature_projected(*a, label_id="a"),
            _label_feature_projected(*b, label_id="b"),
        ],
        crs=CRS,
    )

    deleted = db_mod.delete_labels(["a"])
    assert deleted == 1
    remaining = {l["id"] for l in db_mod.get_all_labels()}
    assert remaining == {"b"}


def test_delete_labels_by_geometry_counts(db):
    a = _project_4326_bbox(-72.5995, 42.2005, -72.5990, 42.2010)
    b = _project_4326_bbox(-72.5905, 42.2105, -72.5900, 42.2110)
    db_mod.insert_labels(
        [
            _label_feature_projected(*a, label_id="a"),
            _label_feature_projected(*b, label_id="b"),
        ],
        crs=CRS,
    )

    min_x, min_y, max_x, max_y = _project_4326_bbox(-72.5996, 42.2004, -72.5989, 42.2011)
    wkt = (
        f"POLYGON(({min_x} {min_y}, {max_x} {min_y}, "
        f"{max_x} {max_y}, {min_x} {max_y}, {min_x} {min_y}))"
    )
    deleted = db_mod.delete_labels_by_geometry(wkt)
    assert deleted == 1

    remaining = {l["id"] for l in db_mod.get_all_labels()}
    assert remaining == {"b"}


def test_delete_labels_by_geometry_no_match_returns_zero(db):
    # Polygon out in the Atlantic — no labels intersect
    wkt = "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))"
    assert db_mod.delete_labels_by_geometry(wkt) == 0


def _white_square_png(size=64, sq=20):
    arr = np.zeros((size, size), dtype=np.uint8)
    off = (size - sq) // 2
    arr[off:off + sq, off:off + sq] = 255
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_save_chip_label_sets_status_labeled(db):
    chips = _make_chips()
    db_mod.insert_chips(chips, crs=CRS)
    chip_id = chips[0]["id"]

    png = _white_square_png()
    label_id = db_mod.save_chip_label(chip_id, png, "positive")
    assert label_id

    fetched = {c["id"]: c for c in db_mod.get_all_chips()}
    assert fetched[chip_id]["status"] == "labeled"

    labels = db_mod.get_all_labels()
    assert len(labels) == 1
    assert labels[0]["id"] == label_id


def test_save_chip_label_empty_mask_raises(db):
    chips = _make_chips()
    db_mod.insert_chips(chips, crs=CRS)
    chip_id = chips[0]["id"]

    blank = Image.fromarray(np.zeros((64, 64), dtype=np.uint8), mode="L")
    buf = io.BytesIO()
    blank.save(buf, format="PNG")
    with pytest.raises(ValueError, match="Mask is empty"):
        db_mod.save_chip_label(chip_id, buf.getvalue())


def test_save_chip_label_missing_chip_raises(db):
    png = _white_square_png()
    with pytest.raises(ValueError, match="not found"):
        db_mod.save_chip_label("nonexistent_chip", png)


def _full_extent_wkt(chips):
    eastings, northings = [], []
    for c in chips:
        coord_text = re.search(r"\(\((.+)\)\)", c["geometry_wkt"]).group(1)
        for p in coord_text.split(","):
            x, y = map(float, p.strip().split())
            eastings.append(x)
            northings.append(y)
    min_e, max_e = min(eastings), max(eastings)
    min_n, max_n = min(northings), max(northings)
    return (
        f"POLYGON(({min_e} {min_n}, {max_e} {min_n}, "
        f"{max_e} {max_n}, {min_e} {max_n}, {min_e} {min_n}))"
    )


def test_delete_chips_by_geometry_cascades_labels(db):
    chips = _make_chips()
    db_mod.insert_chips(chips, crs=CRS)
    n_chips = len(chips)

    # Place a label inside the first chip (in project CRS so insert is correct).
    coord_text = re.search(r"\(\((.+)\)\)", chips[0]["geometry_wkt"]).group(1)
    verts = [tuple(map(float, p.strip().split())) for p in coord_text.split(",")]
    eastings = [v[0] for v in verts]
    northings = [v[1] for v in verts]
    min_e, max_e = min(eastings), max(eastings)
    min_n, max_n = min(northings), max(northings)
    feat = _label_feature_projected(
        min_e + 1, min_n + 1, max_e - 1, max_n - 1, label_id="lab"
    )
    db_mod.insert_labels([feat], crs=CRS)

    wkt = _full_extent_wkt(chips)
    result = db_mod.delete_chips_by_geometry(wkt)
    assert result == {"chips_deleted": n_chips, "labels_deleted": 1}

    assert db_mod.get_all_chips() == []
    assert db_mod.get_all_labels() == []


def test_delete_chips_by_geometry_no_match(db):
    wkt = "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))"
    result = db_mod.delete_chips_by_geometry(wkt)
    assert result == {"chips_deleted": 0, "labels_deleted": 0}


def test_get_labels_for_chips_spatial_join(db):
    chips = _make_chips()
    db_mod.insert_chips(chips, crs=CRS)

    coord_text = re.search(r"\(\((.+)\)\)", chips[0]["geometry_wkt"]).group(1)
    verts = [tuple(map(float, p.strip().split())) for p in coord_text.split(",")]
    eastings = [v[0] for v in verts]
    northings = [v[1] for v in verts]
    feat = _label_feature_projected(
        min(eastings) + 1, min(northings) + 1,
        max(eastings) - 1, max(northings) - 1,
        label_id="L",
    )
    db_mod.insert_labels([feat], crs=CRS)

    out = db_mod.get_labels_for_chips([chips[0]["id"]])
    assert chips[0]["id"] in out
    rows = out[chips[0]["id"]]
    assert len(rows) == 1
    wkb, cls = rows[0]
    assert cls == "positive"
    assert isinstance(wkb, bytes) and len(wkb) > 0


def test_get_labels_for_chips_empty_input(db):
    assert db_mod.get_labels_for_chips([]) == {}
