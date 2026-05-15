"""Endpoint tests for /api/study-areas."""

SMALL_BBOX = {"sw": [-72.6, 42.2], "ne": [-72.598, 42.202]}
LARGE_BBOX = {"sw": [-72.6, 42.2], "ne": [-72.59, 42.21]}


def test_post_study_area_creates_chips(client):
    resp = client.post(
        "/api/study-areas",
        json={"bbox": SMALL_BBOX, "split": "train"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["count"] > 0
    assert body["job_id"] is not None

    chips_resp = client.get("/api/chips")
    assert chips_resp.status_code == 200
    assert len(chips_resp.json()["features"]) == body["count"]


def test_post_study_area_invalid_split_is_422(client):
    resp = client.post(
        "/api/study-areas",
        json={"bbox": SMALL_BBOX, "split": "invalid"},
    )
    assert resp.status_code == 422


def test_post_study_area_exceeds_max_chips_is_400(client):
    resp = client.post(
        "/api/study-areas",
        json={"bbox": LARGE_BBOX, "split": "train"},
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "exceeding limit" in detail
    assert "chips" in detail


def test_post_study_area_worker_unavailable_returns_warning(client, monkeypatch):
    from server.routers import study_areas as sar
    from server.worker_client import WorkerUnavailable

    def _boom(*args, **kwargs):
        raise WorkerUnavailable("worker down")

    monkeypatch.setattr(sar, "start_batch", _boom)

    resp = client.post(
        "/api/study-areas",
        json={"bbox": SMALL_BBOX, "split": "train"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["job_id"] is None
    assert "warning" in body
    # Chips were still inserted despite the worker being down
    assert body["count"] > 0
    chips = client.get("/api/chips").json()["features"]
    assert len(chips) == body["count"]


def test_delete_study_area_requires_point_or_bbox(client):
    resp = client.request("DELETE", "/api/study-areas", json={})
    assert resp.status_code == 400
    assert "point" in resp.json()["detail"] or "bbox" in resp.json()["detail"]


def test_delete_study_area_by_point_removes_overlapping_chip(client):
    client.post("/api/study-areas", json={"bbox": SMALL_BBOX, "split": "train"})
    before = client.get("/api/chips").json()["features"]
    assert len(before) > 0

    # Pick a point inside one of the returned chips
    target = before[0]["geometry"]["coordinates"][0][0]  # [lon, lat]

    resp = client.request(
        "DELETE", "/api/study-areas", json={"point": target},
    )
    assert resp.status_code == 200
    result = resp.json()
    assert result["chips_deleted"] >= 1

    after = client.get("/api/chips").json()["features"]
    assert len(after) == len(before) - result["chips_deleted"]


def test_delete_study_area_by_bbox(client):
    client.post("/api/study-areas", json={"bbox": SMALL_BBOX, "split": "train"})
    before = client.get("/api/chips").json()["features"]
    assert len(before) > 0

    resp = client.request(
        "DELETE",
        "/api/study-areas",
        json={"bbox": [-72.6, 42.2, -72.598, 42.202]},
    )
    assert resp.status_code == 200
    result = resp.json()
    assert result["chips_deleted"] == len(before)
    assert client.get("/api/chips").json()["features"] == []


def test_prefetch_status_unavailable_returns_503(client, monkeypatch):
    from server.routers import study_areas as sar
    from server.worker_client import WorkerUnavailable

    def _boom(job_id):
        raise WorkerUnavailable("worker down")

    monkeypatch.setattr(sar, "get_job_status", _boom)

    resp = client.get("/api/prefetch/abc123")
    assert resp.status_code == 503
