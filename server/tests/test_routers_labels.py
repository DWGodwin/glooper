"""Endpoint tests for /api/labels and /api/chips/{id}/label."""

import base64
import io

import numpy as np
from PIL import Image
from pyproj import Transformer


CRS = "EPSG:32619"
SMALL_BBOX = {"sw": [-72.6, 42.2], "ne": [-72.598, 42.202]}


def _project_4326(lon, lat):
    t = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    return t.transform(lon, lat)


def _label_feature(min_x, min_y, max_x, max_y, label_id="lab", cls="positive"):
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


def _project_feature(lon_a, lat_a, lon_b, lat_b, **kw):
    x1, y1 = _project_4326(lon_a, lat_a)
    x2, y2 = _project_4326(lon_b, lat_b)
    return _label_feature(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), **kw)


def test_post_and_get_labels_round_trip(client):
    feat = _project_feature(-72.5995, 42.2005, -72.5990, 42.2010, label_id="a")
    resp = client.post(
        "/api/labels",
        json={"type": "FeatureCollection", "features": [feat], "crs": CRS},
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "count": 1}

    got = client.get("/api/labels").json()
    assert got["type"] == "FeatureCollection"
    assert len(got["features"]) == 1
    assert got["features"][0]["properties"]["id"] == "a"
    assert got["features"][0]["properties"]["class"] == "positive"


def test_get_labels_bbox_filter(client):
    inside = _project_feature(-72.5995, 42.2005, -72.5990, 42.2010, label_id="in")
    outside = _project_feature(-72.5905, 42.2105, -72.5900, 42.2110, label_id="out")
    client.post(
        "/api/labels",
        json={"type": "FeatureCollection", "features": [inside, outside], "crs": CRS},
    )

    resp = client.get("/api/labels?bbox=-72.5996,42.2004,-72.5989,42.2011")
    assert resp.status_code == 200
    ids = {f["properties"]["id"] for f in resp.json()["features"]}
    assert ids == {"in"}


def test_get_labels_bbox_wrong_count_is_400(client):
    resp = client.get("/api/labels?bbox=-72.6,42.2,-72.5")
    assert resp.status_code == 400


def test_get_labels_bbox_non_numeric_is_400(client):
    resp = client.get("/api/labels?bbox=west,south,east,north")
    assert resp.status_code == 400


def test_delete_labels_by_id(client):
    feat = _project_feature(-72.5995, 42.2005, -72.5990, 42.2010, label_id="x")
    client.post(
        "/api/labels",
        json={"type": "FeatureCollection", "features": [feat], "crs": CRS},
    )

    resp = client.request("DELETE", "/api/labels", json={"ids": ["x"]})
    assert resp.status_code == 200
    assert resp.json() == {"deleted": 1}
    assert client.get("/api/labels").json()["features"] == []


def test_delete_labels_by_geometry_requires_point_or_bbox(client):
    resp = client.request("DELETE", "/api/labels/by-geometry", json={})
    assert resp.status_code == 400


def test_delete_labels_by_geometry_with_point(client):
    feat = _project_feature(-72.5995, 42.2005, -72.5990, 42.2010, label_id="hit")
    client.post(
        "/api/labels",
        json={"type": "FeatureCollection", "features": [feat], "crs": CRS},
    )

    resp = client.request(
        "DELETE",
        "/api/labels/by-geometry",
        json={"point": [-72.5992, 42.2007]},
    )
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 1


def _white_square_png_b64(size=64, sq=20):
    arr = np.zeros((size, size), dtype=np.uint8)
    off = (size - sq) // 2
    arr[off:off + sq, off:off + sq] = 255
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _blank_png_b64(size=64):
    img = Image.fromarray(np.zeros((size, size), dtype=np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _first_chip_id(client):
    client.post("/api/study-areas", json={"bbox": SMALL_BBOX, "split": "train"})
    return client.get("/api/chips").json()["features"][0]["properties"]["id"]


def test_post_chip_label_success(client):
    chip_id = _first_chip_id(client)
    mask_b64 = _white_square_png_b64()

    resp = client.post(
        f"/api/chips/{chip_id}/label",
        json={"mask": mask_b64, "label_class": "positive"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["chip_id"] == chip_id
    assert body["label_id"]

    # The chip should now have status='labeled'
    chips = client.get("/api/chips").json()["features"]
    chip = next(c for c in chips if c["properties"]["id"] == chip_id)
    assert chip["properties"]["status"] == "labeled"


def test_post_chip_label_missing_chip_is_404(client):
    resp = client.post(
        "/api/chips/does_not_exist/label",
        json={"mask": _white_square_png_b64()},
    )
    assert resp.status_code == 404


def test_post_chip_label_empty_mask_is_400(client):
    chip_id = _first_chip_id(client)
    resp = client.post(
        f"/api/chips/{chip_id}/label",
        json={"mask": _blank_png_b64()},
    )
    assert resp.status_code == 400


def test_post_chip_label_bad_base64_is_400(client):
    chip_id = _first_chip_id(client)
    # Not a PNG and also not even valid base64 image data
    resp = client.post(
        f"/api/chips/{chip_id}/label",
        json={"mask": "@@@not-base64@@@"},
    )
    assert resp.status_code == 400


def test_vectorize_preview_returns_feature(client):
    chip_id = _first_chip_id(client)
    resp = client.post(
        f"/api/chips/{chip_id}/vectorize-preview",
        json={"mask": _white_square_png_b64()},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "Feature"
    assert body["geometry"]["type"] == "Polygon"
    assert "vertex_count" in body["properties"]
    assert body["properties"]["vertex_count"] >= 4


def test_vectorize_preview_missing_chip_is_404(client):
    resp = client.post(
        "/api/chips/does_not_exist/vectorize-preview",
        json={"mask": _white_square_png_b64()},
    )
    assert resp.status_code == 404
