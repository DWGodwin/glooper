"""Background chip image prefetch using a thread pool."""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from server.providers import get_provider

logger = logging.getLogger(__name__)

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=3)


def _fetch_chip(job_id: str, chip_id: str, geometry_wkt: str, crs: str):
    provider = get_provider()
    try:
        provider.get_chip_image(chip_id, geometry_wkt, crs)
        with _jobs_lock:
            _jobs[job_id]["done"] += 1
    except Exception:
        logger.exception("Prefetch failed for chip '%s'", chip_id)
        with _jobs_lock:
            _jobs[job_id]["failed"] += 1


def start_prefetch(job_id: str, chips: list[dict], crs: str):
    with _jobs_lock:
        _jobs[job_id] = {"total": len(chips), "done": 0, "failed": 0}

    for chip in chips:
        _executor.submit(_fetch_chip, job_id, chip["id"], chip["geometry_wkt"], crs)


def get_job_status(job_id: str) -> dict | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        return {"total": job["total"], "done": job["done"], "failed": job["failed"]}
