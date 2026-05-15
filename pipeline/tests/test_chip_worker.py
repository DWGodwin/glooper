"""Tests for pipeline.src.chip_worker — config, image conversion, dispatch.

Several tests rely on monkeypatching two module attributes:
  - chip_worker._executor (replaced by `sync_executor` fixture for inline calls)
  - chip_worker._maybe_start_embeddings (replaced by `init_worker` with a no-op
    so batch jobs never reach _ensure_sam / SAM download)
Refactoring the worker to inline either of these will silently break these tests.
"""
import io
from pathlib import Path

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_bounds
from shapely.geometry import box

import pipeline.src.chip_worker as cw


def _synthetic_tif_bytes(count=4, dtype="uint8", height=64, width=64):
    transform = from_bounds(0, 0, width, height, width, height)
    np_dtype = np.dtype(dtype)
    if np.issubdtype(np_dtype, np.integer):
        fill_max = np.iinfo(np_dtype).max
    else:
        fill_max = 1.0
    data = np.zeros((count, height, width), dtype=np_dtype)
    for i in range(count):
        if np.issubdtype(np_dtype, np.integer):
            data[i] = (fill_max // (count + 1)) * (i + 1)
        else:
            data[i] = (i + 1) / (count + 1)

    with rasterio.MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=height,
            width=width,
            count=count,
            dtype=dtype,
            crs="EPSG:32619",
            transform=transform,
        ) as dst:
            dst.write(data)
        return memfile.read()


def test_load_config_reads_from_env_var(test_config):
    loaded = cw._load_config()
    assert loaded["chip_size_px"] == 64
    assert loaded["crs"] == "EPSG:32619"
    assert loaded["imagery_provider"] == "cog"


def test_load_config_missing_file_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("GLOOPER_CONFIG", str(tmp_path / "missing.yaml"))
    assert cw._load_config() == {}


def test_tif_to_png_4_band_uint8():
    png_bytes = cw._tif_to_png(_synthetic_tif_bytes(count=4), rgb_bands=(0, 1, 2))
    img = Image.open(io.BytesIO(png_bytes))
    assert img.size == (64, 64)
    assert img.mode == "RGB"


def test_tif_to_png_single_band_replicates_channel():
    # Single-band path hits the `if src.count == 1: band_indices = [1, 1, 1]` branch.
    png_bytes = cw._tif_to_png(_synthetic_tif_bytes(count=1))
    img = Image.open(io.BytesIO(png_bytes))
    assert img.size == (64, 64)
    assert img.mode == "RGB"


def test_tif_to_png_float32_uses_max_normalization():
    # Float input takes the `data.max() or 1.0` normalization branch.
    png_bytes = cw._tif_to_png(_synthetic_tif_bytes(count=4, dtype="float32"))
    img = Image.open(io.BytesIO(png_bytes))
    assert img.size == (64, 64)
    assert img.mode == "RGB"


def test_init_provider_creates_directories(test_config, monkeypatch):
    monkeypatch.setattr(cw, "load_provider", lambda name, config: object())
    cw._init_provider(test_config)

    data_dir = Path(test_config["data_dir"])
    assert cw._chips_dir == data_dir / "chips"
    assert cw._png_dir == data_dir / "chips_png"
    assert cw._embeddings_dir == data_dir / "sam_embeddings"
    assert cw._sam_batch_size == 1
    assert (data_dir / "chips").is_dir()
    assert (data_dir / "chips_png").is_dir()
    assert (data_dir / "sam_embeddings").is_dir()


def test_handle_request_single_chip(init_worker):
    chip_id = "267120.00e_4681152.00n"
    chip_geom = box(0, 0, 64, 64)
    response = cw._handle_request({
        "chip_id": chip_id,
        "geometry_wkt": chip_geom.wkt,
        "crs": "EPSG:32619",
    })

    assert response["status"] == "ok"
    tif_path = Path(response["path"])
    assert tif_path.exists()
    assert tif_path.name == f"{chip_id}.tif"
    # PNG sidecar should have been written too.
    assert (cw._png_dir / f"{chip_id}.png").exists()


def test_handle_request_batch_registers_job(init_worker, sync_executor):
    chip_geom = box(0, 0, 64, 64).wkt
    batch = [
        {"chip_id": "a", "geometry_wkt": chip_geom, "crs": "EPSG:32619"},
        {"chip_id": "b", "geometry_wkt": chip_geom, "crs": "EPSG:32619"},
    ]
    response = cw._handle_request({"batch": batch, "job_id": "j1"})

    assert response == {"status": "accepted", "job_id": "j1", "total": 2}
    assert "j1" in cw._jobs
    job = cw._jobs["j1"]
    assert job["chips_total"] == 2
    # sync_executor ran _process_batch_chip inline for both chips.
    assert job["chips_done"] == 2
    assert job["chips_failed"] == 0
    # _maybe_start_embeddings is patched to no-op, so phase stays in "chips".
    assert job["phase"] == "chips"


def test_handle_request_query_job_after_batch(init_worker, sync_executor):
    chip_geom = box(0, 0, 64, 64).wkt
    cw._handle_request({
        "batch": [{"chip_id": "a", "geometry_wkt": chip_geom, "crs": "EPSG:32619"}],
        "job_id": "j2",
    })
    status = cw._handle_request({"query_job": "j2"})

    assert status["status"] == "progress"
    assert status["phase"] == "chips"
    assert status["chips_total"] == 1
    assert status["chips_done"] == 1
    assert status["chips_failed"] == 0


def test_handle_request_query_unknown_job(init_worker):
    response = cw._handle_request({"query_job": "does-not-exist"})
    assert response["status"] == "error"
    assert "does-not-exist" in response["message"]


def test_handle_request_unknown_format(init_worker):
    response = cw._handle_request({})
    assert response == {"status": "error", "message": "Unknown request format"}
