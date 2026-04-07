"""Chip image access — serves pre-cached PNGs, dispatches to worker for generation."""

import logging
from pathlib import Path

from server.config import get_config
from server.worker_client import WorkerUnavailable, check_worker, request_chip

logger = logging.getLogger(__name__)

_png_dir: Path | None = None


def init_provider():
    """Set up the cache directory and check worker connectivity at startup."""
    global _png_dir
    cfg = get_config()
    _png_dir = Path(cfg["data_dir"]) / "chips_png"
    _png_dir.mkdir(parents=True, exist_ok=True)

    host = cfg.get("worker_host", "localhost")
    port = cfg.get("worker_port", 9100)
    if check_worker():
        logger.info("Chip worker reachable at %s:%s", host, port)
    else:
        logger.warning(
            "Chip worker not reachable at %s:%s — only cached chips will be served",
            host, port,
        )


def get_chip_image(chip_id: str, geometry_wkt: str, crs: str) -> bytes:
    """Return chip image as PNG bytes.

    The chip worker pre-renders PNGs alongside TIFs during prefetch.
    If the PNG is cached, serve it directly. Otherwise dispatch to the
    worker (which will generate both TIF and PNG) and return the PNG.

    Raises WorkerUnavailable if the chip is not cached and the worker
    cannot be reached.
    """
    png_path = _png_dir / f"{chip_id}.png"
    if png_path.exists():
        return png_path.read_bytes()

    # Dispatch to worker — it writes both .tif and .png
    request_chip(chip_id, geometry_wkt, crs)

    # Worker should have created the PNG
    if png_path.exists():
        return png_path.read_bytes()

    raise FileNotFoundError(f"PNG not found for chip '{chip_id}' after worker dispatch")
