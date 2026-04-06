"""TCP worker process for chip image generation.

Listens on a TCP socket, receives JSON requests, generates chip images
using the configured imagery provider, and writes results to the cache directory.

Protocol (newline-delimited JSON):
    Request:  {"chip_id": "...", "geometry_wkt": "...", "crs": "..."}
    Batch:    {"batch": [{"chip_id": ..., "geometry_wkt": ..., "crs": ...}, ...], "job_id": "..."}
    Status:   {"query_job": "job_id"}
    Response: {"status": "ok", "path": "/path/to/chip.png"}
              {"status": "error", "message": "..."}
              {"status": "progress", "total": N, "done": N, "failed": N}
"""

import argparse
import json
import logging
import os
import socketserver
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from pipeline.src.providers import load_provider

logger = logging.getLogger(__name__)

_provider = None
_cache_dir = None
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=4)


def _load_config():
    config_path = Path(os.environ.get("GLOOPER_CONFIG", "config.yaml"))
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _init_provider(config: dict):
    global _provider, _cache_dir
    name = config.get("imagery_provider", "static")
    provider_config = config.get("imagery_provider_config", {})
    _provider = load_provider(name, provider_config)

    _cache_dir = Path(config.get("data_dir", "public/data")) / "chips"
    _cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Provider '%s' loaded, cache dir: %s", name, _cache_dir)


def _generate_chip(chip_id: str, geometry_wkt: str, crs: str) -> Path:
    """Generate a chip image and write it to cache. Returns the path."""
    cached = _cache_dir / f"{chip_id}.png"
    if cached.exists():
        return cached

    data = _provider.get_chip_image(chip_id, geometry_wkt, crs)
    cached.write_bytes(data)
    return cached


def _process_batch_chip(job_id: str, chip_id: str, geometry_wkt: str, crs: str):
    try:
        _generate_chip(chip_id, geometry_wkt, crs)
        with _jobs_lock:
            _jobs[job_id]["done"] += 1
    except Exception:
        logger.exception("Failed to generate chip '%s'", chip_id)
        with _jobs_lock:
            _jobs[job_id]["failed"] += 1


def _handle_request(data: dict) -> dict:
    # Batch request
    if "batch" in data:
        job_id = data.get("job_id", "unknown")
        chips = data["batch"]
        with _jobs_lock:
            _jobs[job_id] = {"total": len(chips), "done": 0, "failed": 0}
        for chip in chips:
            _executor.submit(
                _process_batch_chip,
                job_id,
                chip["chip_id"],
                chip["geometry_wkt"],
                chip["crs"],
            )
        return {"status": "accepted", "job_id": job_id, "total": len(chips)}

    # Job status query
    if "query_job" in data:
        job_id = data["query_job"]
        with _jobs_lock:
            job = _jobs.get(job_id)
        if job is None:
            return {"status": "error", "message": f"Unknown job '{job_id}'"}
        return {"status": "progress", "total": job["total"], "done": job["done"], "failed": job["failed"]}

    # Single chip request
    if "chip_id" in data:
        try:
            path = _generate_chip(data["chip_id"], data["geometry_wkt"], data["crs"])
            return {"status": "ok", "path": str(path)}
        except FileNotFoundError as e:
            return {"status": "error", "message": str(e)}
        except Exception as e:
            logger.exception("Chip generation failed for '%s'", data["chip_id"])
            return {"status": "error", "message": str(e)}

    return {"status": "error", "message": "Unknown request format"}


class ChipRequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as e:
                response = {"status": "error", "message": f"Invalid JSON: {e}"}
            else:
                response = _handle_request(request)
            self.wfile.write(json.dumps(response).encode() + b"\n")
            self.wfile.flush()


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True


def main():
    parser = argparse.ArgumentParser(description="Glooper chip generation worker")
    parser.add_argument("--host", default=None, help="Host to listen on")
    parser.add_argument("--port", type=int, default=None, help="Port to listen on")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    config = _load_config()
    _init_provider(config)

    host = args.host or config.get("worker_host", "localhost")
    port = args.port or config.get("worker_port", 9100)

    server = ThreadedTCPServer((host, port), ChipRequestHandler)
    logger.info("Chip worker listening on %s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
