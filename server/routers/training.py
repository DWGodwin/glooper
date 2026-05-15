"""Training run dispatch / monitor / cancel endpoints."""

import datetime
import logging
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.config import get_training_config, merge_hyperparams
from server.db import (
    ACTIVE_RUN_STATUSES,
    create_training_run,
    get_active_training_run,
    get_labels_for_chips,
    get_training_chips,
    get_training_run,
    save_prediction_mask,
    list_training_runs,
    update_training_run,
)
from server.training_client import (
    TrainingWorkerUnavailable,
    cancel_training,
    get_training_status,
    start_training,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/training")

TERMINAL_STATUSES = {"done", "failed", "canceled"}

# Map worker phase strings to DB status strings.
_PHASE_TO_STATUS = {
    "queued": "queued",
    "training": "training",
    "predicting": "predicting",
    "done": "done",
    "canceled": "canceled",
    "error": "failed",
}


class StartRunRequest(BaseModel):
    hyperparams: Optional[dict] = None


class LabelsRequest(BaseModel):
    chip_ids: list[str]


class PredictionsRequest(BaseModel):
    chip_id: str
    score: float
    mask_png_b64: str


def _models_dir() -> Path:
    """Resolve the training-runs output directory using the same convention as data_dir."""
    import os

    from server.config import get_config

    cfg = get_config()
    training_cfg = get_training_config()
    models_dir = Path(training_cfg.get("models_dir", "data/training_runs"))
    if not models_dir.is_absolute():
        config_path = Path(os.environ.get("GLOOPER_CONFIG", "config.yaml")).resolve()
        config_dir = config_path.parent if config_path.exists() else Path(cfg["data_dir"]).parent
        models_dir = config_dir / models_dir
    return models_dir


def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _serialize_run(run: dict) -> dict:
    """Convert datetime fields to ISO strings for JSON output."""
    out = dict(run)
    for k in ("started_at", "completed_at"):
        if isinstance(out.get(k), datetime.datetime):
            out[k] = out[k].isoformat()
    return out


@router.get("/config")
def training_config():
    """Default hyperparameters the frontend uses to populate the run form."""
    cfg = get_training_config()
    # Worker addresses & paths are server-side only.
    for k in ("worker_host", "worker_port", "models_dir"):
        cfg.pop(k, None)
    return cfg


@router.post("/runs")
def create_run(req: StartRunRequest):
    import json

    active = get_active_training_run()
    if active is not None:
        raise HTTPException(
            status_code=409,
            detail={"error": "training_in_progress", "active_run_id": active["id"]},
        )

    run_id = uuid.uuid4().hex[:12]
    hyperparams = merge_hyperparams(req.hyperparams)
    model_path = str(_models_dir() / f"{run_id}.pt")

    create_training_run(run_id, hyperparams_json=json.dumps(hyperparams))

    try:
        start_training(run_id, hyperparams, model_path_out=model_path)
    except TrainingWorkerUnavailable:
        update_training_run(
            run_id,
            status="failed",
            error_message="worker_unavailable",
            completed_at=_now_utc(),
        )
        raise HTTPException(status_code=503, detail="Training worker unavailable")
    except RuntimeError as e:
        update_training_run(
            run_id, status="failed", error_message=str(e), completed_at=_now_utc(),
        )
        raise HTTPException(status_code=500, detail=str(e))

    return {"run_id": run_id, "status": "queued"}


@router.get("/runs")
def list_runs():
    runs = list_training_runs(limit=20)
    return {"runs": [_serialize_run(r) for r in runs]}


@router.get("/runs/{run_id}")
def get_run(run_id: str):
    run = get_training_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    if run["status"] in ACTIVE_RUN_STATUSES:
        run = _poll_and_persist(run)

    out = _serialize_run(run)
    return out


def _poll_and_persist(run: dict) -> dict:
    """Ask the worker for live progress, persist deltas, return the refreshed row."""
    run_id = run["id"]
    try:
        worker_status = get_training_status(run_id)
    except TrainingWorkerUnavailable:
        logger.warning("Worker unavailable while polling run %s", run_id)
        return run

    if worker_status is None:
        # Worker doesn't know about this run (likely restarted).
        update_training_run(
            run_id,
            status="failed",
            error_message="worker_lost_run",
            completed_at=_now_utc(),
        )
        return get_training_run(run_id) or run

    new_status = _PHASE_TO_STATUS.get(worker_status.get("phase"), run["status"])
    updates: dict = {"status": new_status}

    train_curve = worker_status.get("train_loss_curve")
    val_curve = worker_status.get("val_loss_curve")
    if train_curve is not None:
        updates["train_loss_curve_json"] = train_curve
    if val_curve is not None:
        updates["val_loss_curve_json"] = val_curve

    if new_status in TERMINAL_STATUSES:
        updates["completed_at"] = _now_utc()
        if new_status == "failed":
            updates["error_message"] = worker_status.get("error") or "training_failed"

    update_training_run(run_id, **updates)
    return get_training_run(run_id) or run


@router.get("/chips")
def list_training_chips():
    """Worker-facing: chips eligible for training, split by partition."""
    return get_training_chips()


@router.post("/labels")
def fetch_labels(req: LabelsRequest):
    """Worker-facing: spatial-join labels for a set of chip IDs.

    Returns {chip_id: [{"wkb_hex": str, "class": str}, ...]} — WKB hex-encoded
    so the worker can rasterize via shapely without a DuckDB connection.
    """
    joined = get_labels_for_chips(req.chip_ids)
    return {
        cid: [{"wkb_hex": wkb.hex(), "class": cls} for wkb, cls in entries]
        for cid, entries in joined.items()
    }


@router.post("/runs/{run_id}/predictions")
def post_predictions(run_id: str, req: PredictionsRequest):
    """Worker-facing: persist a per-chip U-Net probability mask (RGBA PNG)."""
    import base64
    import binascii

    if get_training_run(run_id) is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    try:
        png_bytes = base64.b64decode(req.mask_png_b64, validate=True)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"mask_png_b64 not valid base64: {e}")

    save_prediction_mask(run_id, req.chip_id, float(req.score), png_bytes)
    return {"ok": True}


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str):
    run = get_training_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    if run["status"] in TERMINAL_STATUSES:
        return {"run_id": run_id, "status": run["status"]}

    try:
        cancel_training(run_id)
    except TrainingWorkerUnavailable:
        # Cancel is a best-effort signal; if the worker is gone, mark the row anyway.
        logger.warning("Worker unavailable while canceling run %s", run_id)

    update_training_run(
        run_id, status="canceled", completed_at=_now_utc(),
    )
    return {"run_id": run_id, "status": "canceled"}
