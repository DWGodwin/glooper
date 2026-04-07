"""Lightweight TCP client for the pipeline chip worker."""

import json
import logging
import socket
from pathlib import Path

from server.config import get_config

logger = logging.getLogger(__name__)

_TIMEOUT = 60  # seconds


class WorkerUnavailable(Exception):
    pass


def _send_request(request: dict) -> dict:
    """Send a JSON request to the worker, return the parsed response."""
    cfg = get_config()
    host = cfg.get("worker_host", "localhost")
    port = cfg.get("worker_port", 9100)

    try:
        sock = socket.create_connection((host, port), timeout=_TIMEOUT)
    except (ConnectionRefusedError, OSError) as e:
        raise WorkerUnavailable(f"Cannot reach chip worker at {host}:{port}: {e}") from e

    try:
        sock.sendall(json.dumps(request).encode() + b"\n")
        # Read until newline
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.strip())
    finally:
        sock.close()


def request_chip(chip_id: str, geometry_wkt: str, crs: str) -> Path:
    """Request a single chip from the worker. Returns the cached file path."""
    resp = _send_request({"chip_id": chip_id, "geometry_wkt": geometry_wkt, "crs": crs})
    if resp.get("status") != "ok":
        raise RuntimeError(f"Worker error: {resp.get('message', 'unknown')}")
    return Path(resp["path"])


def start_batch(job_id: str, chips: list[dict], crs: str) -> str:
    """Submit a batch of chips to the worker. Returns the job ID."""
    batch = [{"chip_id": c["id"], "geometry_wkt": c["geometry_wkt"], "crs": crs} for c in chips]
    resp = _send_request({"batch": batch, "job_id": job_id})
    if resp.get("status") != "accepted":
        raise RuntimeError(f"Worker error: {resp.get('message', 'unknown')}")
    return resp["job_id"]


def get_job_status(job_id: str) -> dict:
    """Query the worker for batch job progress."""
    resp = _send_request({"query_job": job_id})
    if resp.get("status") == "error":
        raise RuntimeError(f"Worker error: {resp.get('message', 'unknown')}")
    return {
        "phase": resp.get("phase", "chips"),
        "chips_total": resp.get("chips_total", resp.get("total", 0)),
        "chips_done": resp.get("chips_done", resp.get("done", 0)),
        "chips_failed": resp.get("chips_failed", resp.get("failed", 0)),
        "embed_total": resp.get("embed_total", 0),
        "embed_done": resp.get("embed_done", 0),
        "embed_failed": resp.get("embed_failed", 0),
    }


def check_worker() -> bool:
    """Return True if the worker is reachable."""
    cfg = get_config()
    host = cfg.get("worker_host", "localhost")
    port = cfg.get("worker_port", 9100)
    try:
        sock = socket.create_connection((host, port), timeout=2)
        sock.close()
        return True
    except (ConnectionRefusedError, OSError):
        return False
