"""Chip image access — cache lookup with worker dispatch for generation."""

import logging
from pathlib import Path

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


def get_cached_chip(chip_id: str) -> bytes | None:
    """Return chip PNG bytes from cache, or None if not cached."""
    path = _cache_dir / f"{chip_id}.png"
    if path.exists():
        return path.read_bytes()
    return None


def get_chip_image(chip_id: str, geometry_wkt: str, crs: str) -> bytes:
    """Return chip image: cache hit first, then worker dispatch.

    Raises WorkerUnavailable if the chip is not cached and the worker
    cannot be reached.
    """
    cached = get_cached_chip(chip_id)
    if cached is not None:
        return cached

    # Dispatch to worker — it writes to the same cache dir
    path = request_chip(chip_id, geometry_wkt, crs)
    return path.read_bytes()
