import io
import threading
from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_bounds


def _reset_chip_worker_globals():
    """Clear chip_worker module-level state so each test starts clean."""
    import pipeline.src.chip_worker as cw

    cw._provider = None
    cw._chips_dir = None
    cw._png_dir = None
    cw._embeddings_dir = None
    cw._sam_batch_size = 4
    cw._jobs = {}
    cw._jobs_lock = threading.Lock()
    cw._sam_model = None
    cw._sam_device = None


def _write_synthetic_tif(
    path: Path,
    height: int = 64,
    width: int = 64,
    count: int = 4,
    dtype: str = "uint8",
    crs: str = "EPSG:32619",
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 64.0, 64.0),
) -> Path:
    """Write a small multi-band GeoTIFF with deterministic per-band content."""
    transform = from_bounds(*bounds, width, height)
    np_dtype = np.dtype(dtype)
    if np.issubdtype(np_dtype, np.integer):
        fill_max = np.iinfo(np_dtype).max
    else:
        fill_max = 1.0
    data = np.zeros((count, height, width), dtype=np_dtype)
    for i in range(count):
        # Each band gets a distinct constant so tests can verify band selection.
        data[i] = (fill_max // (count + 1)) * (i + 1) if np.issubdtype(np_dtype, np.integer) else (i + 1) / (count + 1)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)
    return path


def _synthetic_tif_bytes(
    height: int = 64,
    width: int = 64,
    count: int = 4,
    dtype: str = "uint8",
    crs: str = "EPSG:32619",
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 64.0, 64.0),
) -> bytes:
    """Return a synthetic GeoTIFF as in-memory bytes via rasterio.MemoryFile."""
    transform = from_bounds(*bounds, width, height)
    np_dtype = np.dtype(dtype)
    if np.issubdtype(np_dtype, np.integer):
        fill_max = np.iinfo(np_dtype).max
    else:
        fill_max = 1.0
    data = np.zeros((count, height, width), dtype=np_dtype)
    for i in range(count):
        data[i] = (fill_max // (count + 1)) * (i + 1) if np.issubdtype(np_dtype, np.integer) else (i + 1) / (count + 1)

    with rasterio.MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=height,
            width=width,
            count=count,
            dtype=dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(data)
        return memfile.read()


@pytest.fixture
def test_config(tmp_path, monkeypatch):
    """Minimal test config + GLOOPER_CONFIG env + chip_worker global reset."""
    cfg = {
        "chip_size_px": 64,
        "crs": "EPSG:32619",
        "bands": ["R", "G", "B", "NIR"],
        "rgb_bands": ["R", "G", "B"],
        "data_dir": str(tmp_path / "data"),
        "worker_host": "127.0.0.1",
        "worker_port": 0,
        "sam_batch_size": 1,
        "imagery_provider": "cog",
        "imagery_provider_config": {},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("GLOOPER_CONFIG", str(cfg_path))

    _reset_chip_worker_globals()
    yield cfg
    _reset_chip_worker_globals()


@pytest.fixture
def synthetic_tif(tmp_path):
    """Factory: write a small multi-band GeoTIFF to disk."""
    def _make(
        path: Path | None = None,
        height: int = 64,
        width: int = 64,
        count: int = 4,
        dtype: str = "uint8",
        crs: str = "EPSG:32619",
        bounds: tuple[float, float, float, float] = (0.0, 0.0, 64.0, 64.0),
    ) -> Path:
        if path is None:
            path = tmp_path / "chip.tif"
        return _write_synthetic_tif(
            Path(path),
            height=height,
            width=width,
            count=count,
            dtype=dtype,
            crs=crs,
            bounds=bounds,
        )

    return _make


@pytest.fixture
def stub_provider():
    """An ImageryProvider whose get_chip_image returns synthetic TIF bytes."""
    class _Stub:
        def __init__(self):
            self.calls = []

        def get_chip_image(self, chip_id, geometry_wkt, crs):
            self.calls.append((chip_id, geometry_wkt, crs))
            return _synthetic_tif_bytes(crs=crs)

    return _Stub()


@pytest.fixture
def sync_executor(monkeypatch):
    """Replace chip_worker._executor with a synchronous shim.

    .submit(fn, *args, **kwargs) invokes fn inline and returns a stub future
    whose .result() returns the call's return value. This eliminates thread
    races in assertions on _jobs[...]["chips_done"].
    """
    import pipeline.src.chip_worker as cw

    class _Future:
        def __init__(self, value=None, exc=None):
            self._value = value
            self._exc = exc

        def result(self):
            if self._exc is not None:
                raise self._exc
            return self._value

    class _SyncExecutor:
        def submit(self, fn, *args, **kwargs):
            try:
                return _Future(value=fn(*args, **kwargs))
            except Exception as e:
                return _Future(exc=e)

    monkeypatch.setattr(cw, "_executor", _SyncExecutor())


@pytest.fixture
def init_worker(test_config, stub_provider, monkeypatch):
    """Initialize chip_worker with a stub provider and disable SAM path.

    Patches load_provider to return stub_provider so _init_provider doesn't
    try to open a real COG. Patches _maybe_start_embeddings to a no-op so
    batch jobs never invoke _ensure_sam (which would download SAM).
    """
    import pipeline.src.chip_worker as cw

    monkeypatch.setattr(cw, "load_provider", lambda name, config: stub_provider)
    monkeypatch.setattr(cw, "_maybe_start_embeddings", lambda job_id: None)
    cw._init_provider(test_config)
    return stub_provider
