"""Chip image access — cache lookup with worker dispatch for generation."""

import io
import logging
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from server.config import get_config
from server.worker_client import WorkerUnavailable, check_worker, request_chip

logger = logging.getLogger(__name__)

_cache_dir: Path | None = None


def init_provider():
    """Set up the cache directory and check worker connectivity at startup."""
    global _cache_dir
    cfg = get_config()
    _cache_dir = Path(cfg["data_dir"]) / "chips"
    _cache_dir.mkdir(parents=True, exist_ok=True)

    host = cfg.get("worker_host", "localhost")
    port = cfg.get("worker_port", 9100)
    if check_worker():
        logger.info("Chip worker reachable at %s:%s", host, port)
    else:
        logger.warning(
            "Chip worker not reachable at %s:%s — only cached chips will be served",
            host, port,
        )


def tif_to_png(tif_bytes: bytes, rgb_bands: tuple[int, ...] = (0, 1, 2)) -> bytes:
    """Convert GeoTIFF bytes to RGB PNG bytes.

    Args:
        tif_bytes: Raw GeoTIFF file content.
        rgb_bands: 0-based band indices to use as R, G, B.
    """
    data = tifffile.imread(io.BytesIO(tif_bytes))  # (H, W, C) or (C, H, W) or (H, W)

    # Normalize to (H, W, C)
    if data.ndim == 2:
        # Single band — replicate to 3
        data = np.stack([data, data, data], axis=-1)
    elif data.ndim == 3:
        # tifffile reads as (H, W, C) for interleaved or (C, H, W) for band-sequential
        if data.shape[0] < data.shape[2]:
            # (C, H, W) → (H, W, C)
            data = np.transpose(data, (1, 2, 0))
        # Select RGB bands
        data = data[:, :, list(rgb_bands)]

    # Normalize to uint8
    if data.dtype != np.uint8:
        if np.issubdtype(data.dtype, np.integer):
            max_val = np.iinfo(data.dtype).max
        else:
            max_val = data.max() or 1.0
        data = (data.astype(np.float32) / max_val * 255).clip(0, 255).astype(np.uint8)

    img = Image.fromarray(data)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def get_cached_chip(chip_id: str) -> bytes | None:
    """Return chip PNG bytes from cache, or None if not cached."""
    path = _cache_dir / f"{chip_id}.tif"
    if path.exists():
        return tif_to_png(path.read_bytes())
    return None


def get_chip_image(chip_id: str, geometry_wkt: str, crs: str) -> bytes:
    """Return chip image as PNG: cache hit first, then worker dispatch.

    Raises WorkerUnavailable if the chip is not cached and the worker
    cannot be reached.
    """
    cached = get_cached_chip(chip_id)
    if cached is not None:
        return cached

    # Dispatch to worker — it writes to the same cache dir
    path = request_chip(chip_id, geometry_wkt, crs)
    return tif_to_png(path.read_bytes())
