"""Per-chip raster prediction endpoints: list, serve mask, serve diff."""

import io
import logging
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from PIL import Image

from server.db import (
    ACTIVE_RUN_STATUSES,
    get_prediction_mask_path,
    get_predictions_by_bbox,
    get_training_run,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/predictions")

# Diff composite colors — match the legacy diff layer palette.
_GAIN_RGB = (34, 197, 94)    # #22c55e
_LOSS_RGB = (239, 68, 68)    # #ef4444
_CONST_RGB = (156, 163, 175) # #9ca3af
_DIFF_ALPHA_ACTIVE = 200
_DIFF_ALPHA_CONST = 110
_ACTIVE_THRESHOLD = 128       # alpha channel cutoff (~0.5 probability)


def _parse_bbox(bbox: str) -> tuple[float, float, float, float]:
    parts = bbox.split(",")
    if len(parts) != 4:
        raise HTTPException(
            status_code=400, detail="bbox must be 4 comma-separated floats: west,south,east,north",
        )
    try:
        return tuple(float(p) for p in parts)
    except ValueError:
        raise HTTPException(status_code=400, detail="bbox values must be numeric")


def _load_alpha(path: Path) -> np.ndarray:
    """Load an RGBA prediction PNG and return its alpha plane as uint8."""
    with Image.open(path) as img:
        return np.array(img.convert("RGBA"))[..., 3]


@router.get("")
def list_predictions(
    run_id: str = Query(...),
    bbox: str = Query(...),
):
    run = get_training_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    rows = get_predictions_by_bbox(run_id, _parse_bbox(bbox))
    return {"predictions": rows}


@router.get("/{run_id}/chips/{chip_id}/mask.png")
def prediction_mask(run_id: str, chip_id: str):
    path = get_prediction_mask_path(run_id, chip_id)
    if not path or not Path(path).exists():
        raise HTTPException(
            status_code=404, detail=f"No prediction mask for run '{run_id}' chip '{chip_id}'"
        )
    return FileResponse(path, media_type="image/png")


@router.get("/diff")
def diff_predictions(
    run_a: str = Query(...),
    run_b: str = Query(...),
    bbox: str = Query(...),
):
    """List chips that have a prediction in *either* run within the bbox.

    The diff raster itself is computed lazily by `/diff/{run_a}/{run_b}/chips/{chip_id}/mask.png`.
    """
    a_run = get_training_run(run_a)
    if a_run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_a}' not found")
    b_run = get_training_run(run_b)
    if b_run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_b}' not found")

    for r in (a_run, b_run):
        if r["status"] in ACTIVE_RUN_STATUSES:
            raise HTTPException(
                status_code=409,
                detail=f"Run '{r['id']}' is {r['status']} — wait until both runs are terminal",
            )

    bbox_tuple = _parse_bbox(bbox)
    a_rows = get_predictions_by_bbox(run_a, bbox_tuple)
    b_rows = get_predictions_by_bbox(run_b, bbox_tuple)

    a_ids = {r["chip_id"] for r in a_rows}
    b_ids = {r["chip_id"] for r in b_rows}
    chips = sorted(a_ids | b_ids)
    return {
        "chips": [
            {"chip_id": cid, "in_a": cid in a_ids, "in_b": cid in b_ids}
            for cid in chips
        ]
    }


@router.get("/diff/{run_a}/{run_b}/chips/{chip_id}/mask.png")
def prediction_diff_mask(run_a: str, run_b: str, chip_id: str):
    """Compute and return an RGBA diff composite for one chip across two runs.

    Pixels are categorized as: gain (active in B only, green), loss (active in A only,
    red), or constant (active in both, faint gray). Threshold is alpha > 128.
    """
    a_path = get_prediction_mask_path(run_a, chip_id)
    b_path = get_prediction_mask_path(run_b, chip_id)
    if not a_path and not b_path:
        raise HTTPException(
            status_code=404, detail=f"No predictions for chip '{chip_id}' in either run"
        )

    a_alpha = _load_alpha(Path(a_path)) if a_path and Path(a_path).exists() else None
    b_alpha = _load_alpha(Path(b_path)) if b_path and Path(b_path).exists() else None

    # Reconcile shapes — when only one side has a mask, mirror its dimensions.
    if a_alpha is not None and b_alpha is not None and a_alpha.shape != b_alpha.shape:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction shape mismatch for chip '{chip_id}': "
                   f"{a_alpha.shape} vs {b_alpha.shape}",
        )
    if a_alpha is None:
        a_alpha = np.zeros_like(b_alpha)
    if b_alpha is None:
        b_alpha = np.zeros_like(a_alpha)

    a_active = a_alpha > _ACTIVE_THRESHOLD
    b_active = b_alpha > _ACTIVE_THRESHOLD
    gain = b_active & ~a_active
    loss = a_active & ~b_active
    constant = a_active & b_active

    h, w = a_alpha.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[gain]     = (*_GAIN_RGB, _DIFF_ALPHA_ACTIVE)
    rgba[loss]     = (*_LOSS_RGB, _DIFF_ALPHA_ACTIVE)
    rgba[constant] = (*_CONST_RGB, _DIFF_ALPHA_CONST)

    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=False)
    return Response(content=buf.getvalue(), media_type="image/png")
