"""TCP worker process for U-Net segmentation training.

Mirrors `pipeline/src/chip_worker.py`: listens on a TCP socket, dispatches
newline-delimited JSON requests, runs training jobs in a thread-pool executor.

Protocol (newline-delimited JSON):
    Request:
        {"action": "start",  "run_id": "...", "hyperparams": {...}, "model_path_out": "..."}
        {"action": "cancel", "run_id": "..."}
        {"action": "status", "run_id": "..."}
    Response:
        {"ok": true}                                 — start / cancel ack
        {"status": "not_found"}                      — unknown run for status
        {"phase": "training"|"predicting"|"done"|"error"|"canceled",
         "epoch": N, "total_epochs": N,
         "train_loss": float|null, "val_loss": float|null,
         "train_loss_curve": [...], "val_loss_curve": [...],
         "predictions_total": N, "predictions_done": N,
         "error": str|null}                          — status response
"""

import argparse
import json
import logging
import os
import socketserver
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from pipeline.src.server_client import (
    fetch_labels,
    fetch_training_chips,
    post_prediction_mask,
)
from server.config import get_config, get_training_config

# Prediction render color — tints the probability mask on the map.
# Stays in lockstep with the frontend prediction layer color.
_PRED_TINT_RGB = (59, 130, 246)  # #3b82f6

logger = logging.getLogger(__name__)

_chips_dir: Path | None = None
_models_dir: Path | None = None
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1)  # one active training run at a time


def _init() -> None:
    """Resolve paths from the shared config (uses GLOOPER_CONFIG env var)."""
    global _chips_dir, _models_dir
    cfg = get_config()
    _chips_dir = (Path(cfg["data_dir"]) / "chips").resolve()
    training_cfg = get_training_config()
    models_dir = Path(training_cfg.get("models_dir", "data/training_runs"))
    if not models_dir.is_absolute():
        # Resolve relative to the config file's parent (same convention as data_dir / db_path).
        config_path = Path(os.environ.get("GLOOPER_CONFIG", "config.yaml")).resolve()
        config_dir = config_path.parent if config_path.exists() else Path.cwd()
        models_dir = config_dir / models_dir
    _models_dir = models_dir
    _models_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Train worker init: chips=%s models=%s", _chips_dir, _models_dir)


# ── Job state ──────────────────────────────────────────────────────


def _new_job(total_epochs: int) -> dict:
    return {
        "phase": "queued",
        "epoch": 0,
        "total_epochs": total_epochs,
        "train_loss": None,
        "val_loss": None,
        "train_loss_curve": [],
        "val_loss_curve": [],
        "predictions_total": 0,
        "predictions_done": 0,
        "cancel_event": threading.Event(),
        "error": None,
    }


def _status_payload(job: dict) -> dict:
    return {k: v for k, v in job.items() if k != "cancel_event"}


# ── Augmentation ───────────────────────────────────────────────────


def _augment(image, label, augs):
    import torch
    if "hflip" in augs and torch.rand(1).item() < 0.5:
        image = torch.flip(image, dims=[-1])
        label = torch.flip(label, dims=[-1])
    if "vflip" in augs and torch.rand(1).item() < 0.5:
        image = torch.flip(image, dims=[-2])
        label = torch.flip(label, dims=[-2])
    if "rot90" in augs:
        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            image = torch.rot90(image, k, dims=[-2, -1])
            label = torch.rot90(label, k, dims=[-2, -1])
    return image, label


def _train_collate(batch: list[dict]) -> dict:
    import torch
    return {
        "chip_ids": [b["chip_id"] for b in batch],
        "image": torch.stack([b["image"] for b in batch]),
        "label_mask": torch.stack([b["label_mask"] for b in batch]),
    }


def _resolve_pos_weight(pos_weight, train_ds, device):
    """Resolve the pos_weight hyperparam to a tensor (or None).

    - None / falsy            → no weighting (vanilla BCE).
    - "auto" (string)         → compute N_neg / N_pos across the train set,
                                clamped to [1.0, 100.0] so an empty/near-empty
                                label set doesn't blow up the loss.
    - numeric (int/float/str) → use as-is.
    """
    import torch
    if pos_weight is None or pos_weight is False:
        return None

    if isinstance(pos_weight, str) and pos_weight.lower() == "auto":
        n_pos = 0
        n_neg = 0
        for i in range(len(train_ds)):
            label = train_ds[i]["label_mask"]
            n_pos += int(label.sum().item())
            n_neg += int(label.numel() - label.sum().item())
        if n_pos == 0:
            logger.warning("pos_weight=auto: no positive pixels found; falling back to 1.0")
            return None
        weight = min(100.0, max(1.0, n_neg / n_pos))
        logger.info("pos_weight=auto resolved to %.3f (n_pos=%d, n_neg=%d)", weight, n_pos, n_neg)
        return torch.tensor(weight, device=device)

    try:
        weight = float(pos_weight)
    except (TypeError, ValueError):
        logger.warning("pos_weight=%r not parseable; ignoring", pos_weight)
        return None
    if weight <= 0:
        return None
    return torch.tensor(weight, device=device)


# ── Training entrypoint ────────────────────────────────────────────


def _run_training(run_id: str, hyperparams: dict, model_path_out: str | None) -> None:
    """Long-running training fn. Sets job state via the shared _jobs dict."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from pipeline.src.models.unet import build_unet
    from pipeline.src.training_dataset import TrainingDataset

    job = _jobs[run_id]
    cancel_event: threading.Event = job["cancel_event"]

    try:
        model_size = hyperparams.get("model_size", "small")
        epochs = int(hyperparams.get("epochs", 20))
        batch_size = int(hyperparams.get("batch_size", 4))
        lr = float(hyperparams.get("learning_rate", 0.001))
        augs = hyperparams.get("augmentations", []) or []
        pos_weight_hp = hyperparams.get("pos_weight", None)

        with _jobs_lock:
            job["phase"] = "training"
            job["total_epochs"] = epochs

        train_chips, validate_chips = fetch_training_chips()
        if not train_chips:
            raise RuntimeError(
                "No complete=True 'train' chips in database — nothing to learn from"
            )

        labels = fetch_labels(train_chips + validate_chips)

        train_ds = TrainingDataset(train_chips, _chips_dir, labels)
        val_ds = TrainingDataset(validate_chips, _chips_dir, labels) if validate_chips else None

        if len(train_ds) == 0:
            raise RuntimeError("All train chips skipped (TIFFs missing) — prefetch first")

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, collate_fn=_train_collate,
        )
        val_loader = (
            DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                       num_workers=0, collate_fn=_train_collate)
            if val_ds and len(val_ds) > 0 else None
        )

        device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        logger.info("Training run %s on %s: %d train chips, %d val chips, %d epochs",
                    run_id, device, len(train_ds), len(val_ds) if val_ds else 0, epochs)

        model = build_unet(model_size=model_size, in_channels=3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        pos_weight_tensor = _resolve_pos_weight(pos_weight_hp, train_ds, device)

        train_curve: list[float] = []
        val_curve: list[float] = []

        for epoch in range(epochs):
            if cancel_event.is_set():
                logger.info("Run %s canceled before epoch %d", run_id, epoch)
                with _jobs_lock:
                    job["phase"] = "canceled"
                return

            model.train()
            batch_losses: list[float] = []
            for batch in train_loader:
                if cancel_event.is_set():
                    break
                image = batch["image"].to(device)
                label = batch["label_mask"].to(device)
                image, label = _augment(image, label, augs)

                logits = model(image)
                loss = F.binary_cross_entropy_with_logits(
                    logits, label.unsqueeze(1), pos_weight=pos_weight_tensor,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.item()))

            train_loss_epoch = float(np.mean(batch_losses)) if batch_losses else float("nan")

            val_loss_epoch: float | None = None
            if val_loader is not None:
                model.eval()
                val_batch_losses: list[float] = []
                with torch.no_grad():
                    for batch in val_loader:
                        image = batch["image"].to(device)
                        label = batch["label_mask"].to(device).unsqueeze(1)
                        logits = model(image)
                        loss = F.binary_cross_entropy_with_logits(
                            logits, label, pos_weight=pos_weight_tensor,
                        )
                        val_batch_losses.append(float(loss.item()))
                if val_batch_losses:
                    val_loss_epoch = float(np.mean(val_batch_losses))

            train_curve.append(train_loss_epoch)
            if val_loss_epoch is not None:
                val_curve.append(val_loss_epoch)

            with _jobs_lock:
                job["epoch"] = epoch + 1
                job["train_loss"] = train_loss_epoch
                job["val_loss"] = val_loss_epoch
                job["train_loss_curve"] = list(train_curve)
                job["val_loss_curve"] = list(val_curve)

            logger.info("Run %s epoch %d/%d train=%.4f val=%s",
                        run_id, epoch + 1, epochs, train_loss_epoch,
                        f"{val_loss_epoch:.4f}" if val_loss_epoch is not None else "n/a")

        if cancel_event.is_set():
            with _jobs_lock:
                job["phase"] = "canceled"
            return

        weights_path = Path(model_path_out) if model_path_out else _models_dir / f"{run_id}.pt"
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), weights_path)
        logger.info("Run %s weights saved to %s", run_id, weights_path)

        _predict_phase(run_id, model, device, train_chips, validate_chips, cancel_event)

        if cancel_event.is_set():
            with _jobs_lock:
                job["phase"] = "canceled"
        else:
            with _jobs_lock:
                job["phase"] = "done"
    except Exception as e:
        logger.exception("Run %s failed", run_id)
        with _jobs_lock:
            job["phase"] = "error"
            job["error"] = str(e)


def _probs_to_png_bytes(probs: np.ndarray) -> bytes:
    """Encode a [0,1] probability mask as an RGBA PNG (tinted, alpha = probs * 255)."""
    import io as _io
    from PIL import Image

    h, w = probs.shape
    alpha = np.clip(probs * 255.0, 0, 255).astype(np.uint8)
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = _PRED_TINT_RGB[0]
    rgba[..., 1] = _PRED_TINT_RGB[1]
    rgba[..., 2] = _PRED_TINT_RGB[2]
    rgba[..., 3] = alpha
    buf = _io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _predict_phase(run_id, model, device, train_chips, validate_chips, cancel_event):
    """Forward U-Net over train ∪ validate chips, writing each probability mask as a PNG."""
    import rasterio
    import torch

    model.eval()

    all_chips = list({*train_chips, *validate_chips})
    with _jobs_lock:
        _jobs[run_id]["phase"] = "predicting"
        _jobs[run_id]["predictions_total"] = len(all_chips)
        _jobs[run_id]["predictions_done"] = 0

    for chip_id in all_chips:
        if cancel_event.is_set():
            return

        tif_path = _chips_dir / f"{chip_id}.tif"
        if not tif_path.exists():
            with _jobs_lock:
                _jobs[run_id]["predictions_done"] += 1
            continue

        try:
            with rasterio.open(tif_path) as src:
                data = src.read([1, 2, 3])
                dtype = src.dtypes[0]
        except Exception:
            logger.exception("Run %s: failed to read %s", run_id, tif_path)
            with _jobs_lock:
                _jobs[run_id]["predictions_done"] += 1
            continue

        if np.issubdtype(np.dtype(dtype), np.integer):
            max_val = np.iinfo(np.dtype(dtype)).max
        else:
            max_val = float(data.max()) or 1.0

        image = torch.from_numpy(data.astype(np.float32) / max_val).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(image))[0, 0].cpu().numpy()

        # Average confidence over the "active" pixels — informational, used to rank
        # predictions in the UI. Falls back to the mean over the full chip when no
        # pixel passes 0.5 so empty predictions still post a score.
        active = probs > 0.5
        avg_score = float(probs[active].mean()) if active.any() else float(probs.mean())

        png_bytes = _probs_to_png_bytes(probs)
        post_prediction_mask(run_id, chip_id, avg_score, png_bytes)

        with _jobs_lock:
            _jobs[run_id]["predictions_done"] += 1


# ── Request dispatch ───────────────────────────────────────────────


def _handle_request(data: dict) -> dict:
    action = data.get("action")
    run_id = data.get("run_id")

    if action == "start":
        if not run_id:
            return {"ok": False, "error": "run_id required"}
        hyperparams = data.get("hyperparams") or {}
        model_path_out = data.get("model_path_out")
        with _jobs_lock:
            for rid, job in _jobs.items():
                if rid != run_id and job["phase"] in ("queued", "training", "predicting"):
                    return {"ok": False, "error": f"another run is active: {rid}"}
            _jobs[run_id] = _new_job(int(hyperparams.get("epochs", 20)))
        _executor.submit(_run_training, run_id, hyperparams, model_path_out)
        return {"ok": True}

    if action == "cancel":
        if not run_id:
            return {"ok": False, "error": "run_id required"}
        with _jobs_lock:
            job = _jobs.get(run_id)
        if job is not None:
            job["cancel_event"].set()
        return {"ok": True}

    if action == "status":
        if not run_id:
            return {"status": "not_found"}
        with _jobs_lock:
            job = _jobs.get(run_id)
        if job is None:
            return {"status": "not_found"}
        return _status_payload(job)

    return {"ok": False, "error": f"Unknown action: {action!r}"}


class TrainRequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as e:
                response = {"ok": False, "error": f"Invalid JSON: {e}"}
            else:
                response = _handle_request(request)
            self.wfile.write(json.dumps(response).encode() + b"\n")
            self.wfile.flush()


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True


def main():
    parser = argparse.ArgumentParser(description="Glooper training worker")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    log_level = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    _init()
    training_cfg = get_training_config()
    host = args.host or training_cfg.get("worker_host", "localhost")
    port = args.port or training_cfg.get("worker_port", 9101)

    server = ThreadedTCPServer((host, port), TrainRequestHandler)
    logger.info("Train worker listening on %s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
