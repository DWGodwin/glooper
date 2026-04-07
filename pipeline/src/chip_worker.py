"""TCP worker process for chip image generation and SAM embedding.

Listens on a TCP socket, receives JSON requests, generates chip images
using the configured imagery provider, and writes results to the cache directory.
When a batch job completes, automatically generates SAM image embeddings.

Protocol (newline-delimited JSON):
    Request:  {"chip_id": "...", "geometry_wkt": "...", "crs": "..."}
    Batch:    {"batch": [{"chip_id": ..., "geometry_wkt": ..., "crs": ...}, ...], "job_id": "..."}
    Status:   {"query_job": "job_id"}
    Response: {"status": "ok", "path": "/path/to/chip.tif"}
              {"status": "error", "message": "..."}
              {"status": "progress", "phase": "chips"|"embeddings"|"complete",
               "chips_total": N, "chips_done": N, "chips_failed": N,
               "embed_total": N, "embed_done": N, "embed_failed": N}
"""

import argparse
import json
import logging
import os
import socketserver
import threading
import urllib.request
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

# SAM — loaded lazily on first embeddings job
_sam_model = None
_sam_device = None
_embeddings_dir: Path | None = None

SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
MODELS_DIR = Path(__file__).parent / "models"


def _load_config():
    config_path = Path(os.environ.get("GLOOPER_CONFIG", "config.yaml"))
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _init_provider(config: dict):
    global _provider, _cache_dir, _embeddings_dir
    name = config.get("imagery_provider", "static")
    provider_config = config.get("imagery_provider_config", {})
    provider_config.setdefault("crs", config.get("crs", "EPSG:4326"))
    _provider = load_provider(name, provider_config)

    data_dir = Path(config.get("data_dir", "public/data"))
    _cache_dir = data_dir / "chips"
    _cache_dir.mkdir(parents=True, exist_ok=True)
    _embeddings_dir = data_dir / "sam_embeddings"
    _embeddings_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Provider '%s' loaded, cache dir: %s", name, _cache_dir)


def _generate_chip(chip_id: str, geometry_wkt: str, crs: str) -> Path:
    """Generate a chip image and write it to cache. Returns the path."""
    cached = _cache_dir / f"{chip_id}.tif"
    if cached.exists():
        return cached

    data = _provider.get_chip_image(chip_id, geometry_wkt, crs)
    cached.write_bytes(data)
    return cached


def _process_batch_chip(job_id: str, chip_id: str, geometry_wkt: str, crs: str):
    try:
        _generate_chip(chip_id, geometry_wkt, crs)
        with _jobs_lock:
            _jobs[job_id]["chips_done"] += 1
    except Exception:
        logger.exception("Failed to generate chip '%s'", chip_id)
        with _jobs_lock:
            _jobs[job_id]["chips_failed"] += 1

    _maybe_start_embeddings(job_id)


def _maybe_start_embeddings(job_id: str):
    """Transition to embeddings phase once all chips are done."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job or job["phase"] != "chips":
            return
        if job["chips_done"] + job["chips_failed"] < job["chips_total"]:
            return
        job["phase"] = "embeddings"
        chip_ids = list(job["chip_ids"])

    logger.info("Job '%s' chips complete, starting SAM embeddings for %d chips", job_id, len(chip_ids))
    _executor.submit(_run_embeddings, job_id, chip_ids)


def _ensure_sam():
    """Lazy-load SAM model and export ONNX decoder on first use."""
    global _sam_model, _sam_device

    if _sam_model is not None:
        return

    import torch
    from segment_anything import sam_model_registry
    from segment_anything.utils.onnx import SamOnnxModel

    checkpoint = MODELS_DIR / "sam_vit_b_01ec64.pth"
    if not checkpoint.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading SAM checkpoint (~375MB)...")
        urllib.request.urlretrieve(SAM_CHECKPOINT_URL, str(checkpoint))

    _sam_device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    logger.info("Loading SAM ViT-B on %s", _sam_device)
    _sam_model = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
    _sam_model.eval()

    # Export ONNX decoder for in-browser inference
    onnx_path = _embeddings_dir.parent / "sam_decoder.onnx"
    if not onnx_path.exists():
        logger.info("Exporting SAM decoder to ONNX...")
        onnx_m = SamOnnxModel(_sam_model, return_single_mask=False)
        dummy = {
            "image_embeddings": torch.randn(1, 256, 64, 64),
            "point_coords": torch.randint(0, 1024, (1, 2, 2), dtype=torch.float),
            "point_labels": torch.randint(0, 4, (1, 2), dtype=torch.float),
            "mask_input": torch.randn(1, 1, 256, 256),
            "has_mask_input": torch.tensor([1.0]),
            "orig_im_size": torch.tensor([512.0, 512.0]),
        }
        torch.onnx.export(
            onnx_m, tuple(dummy.values()), str(onnx_path),
            input_names=list(dummy.keys()),
            output_names=["masks", "iou_predictions", "low_res_masks"],
            dynamic_axes={"point_coords": {1: "num_points"}, "point_labels": {1: "num_points"}},
            opset_version=17, dynamo=False,
        )
        logger.info("Saved ONNX decoder to %s", onnx_path)

    _sam_model = _sam_model.to(_sam_device)


def _run_embeddings(job_id: str, chip_ids: list[str]):
    """Generate SAM image embeddings for a list of chips."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from pipeline.src.dataset import ChipDataset, sam_collate

    try:
        _ensure_sam()
    except Exception:
        logger.exception("Failed to load SAM model")
        with _jobs_lock:
            _jobs[job_id]["phase"] = "complete"
        return

    # Filter to chips that actually need embeddings
    need = [cid for cid in chip_ids if not (_embeddings_dir / f"{cid}.npy").exists()]
    with _jobs_lock:
        _jobs[job_id]["embed_total"] = len(need)

    if not need:
        logger.info("Job '%s': all %d chips already have embeddings", job_id, len(chip_ids))
        with _jobs_lock:
            _jobs[job_id]["phase"] = "complete"
        return

    dataset = ChipDataset(need, _cache_dir)
    loader = DataLoader(dataset, batch_size=4, num_workers=0, collate_fn=sam_collate)

    for batch in loader:
        pixel_values = batch["pixel_values"].to(_sam_device)
        with torch.no_grad():
            embeddings = _sam_model.image_encoder(pixel_values)

        for i, cid in enumerate(batch["chip_ids"]):
            emb = embeddings[i:i + 1].cpu().numpy()
            np.save(_embeddings_dir / f"{cid}.npy", emb)

        with _jobs_lock:
            _jobs[job_id]["embed_done"] += len(batch["chip_ids"])

    logger.info("Job '%s': embeddings complete", job_id)
    with _jobs_lock:
        _jobs[job_id]["phase"] = "complete"


def _handle_request(data: dict) -> dict:
    # Batch request
    if "batch" in data:
        job_id = data.get("job_id", "unknown")
        chips = data["batch"]
        with _jobs_lock:
            _jobs[job_id] = {
                "phase": "chips",
                "chip_ids": [c["chip_id"] for c in chips],
                "chips_total": len(chips), "chips_done": 0, "chips_failed": 0,
                "embed_total": 0, "embed_done": 0, "embed_failed": 0,
            }
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
        return {
            "status": "progress",
            "phase": job["phase"],
            "chips_total": job["chips_total"], "chips_done": job["chips_done"], "chips_failed": job["chips_failed"],
            "embed_total": job["embed_total"], "embed_done": job["embed_done"], "embed_failed": job["embed_failed"],
        }

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

    log_level = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s %(name)s %(levelname)s %(message)s")

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
