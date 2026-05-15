"""Lightweight TCP client for the pipeline training worker."""

import json
import logging
import socket

from server.config import get_training_config

logger = logging.getLogger(__name__)

_TIMEOUT = 60  # seconds


class TrainingWorkerUnavailable(Exception):
    pass


def _endpoint() -> tuple[str, int]:
    cfg = get_training_config()
    return cfg.get("worker_host", "localhost"), int(cfg.get("worker_port", 9101))


def _send_request(request: dict) -> dict:
    host, port = _endpoint()
    try:
        sock = socket.create_connection((host, port), timeout=_TIMEOUT)
    except (ConnectionRefusedError, OSError) as e:
        raise TrainingWorkerUnavailable(
            f"Cannot reach training worker at {host}:{port}: {e}"
        ) from e

    try:
        sock.sendall(json.dumps(request).encode() + b"\n")
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.strip())
    finally:
        sock.close()


def start_training(run_id: str, hyperparams: dict, model_path_out: str | None = None) -> dict:
    resp = _send_request({
        "action": "start",
        "run_id": run_id,
        "hyperparams": hyperparams,
        "model_path_out": model_path_out,
    })
    if not resp.get("ok"):
        raise RuntimeError(f"Training worker error: {resp.get('error', 'unknown')}")
    return resp


def cancel_training(run_id: str) -> dict:
    resp = _send_request({"action": "cancel", "run_id": run_id})
    if not resp.get("ok"):
        raise RuntimeError(f"Training worker error: {resp.get('error', 'unknown')}")
    return resp


def get_training_status(run_id: str) -> dict | None:
    resp = _send_request({"action": "status", "run_id": run_id})
    if resp.get("status") == "not_found":
        return None
    return resp


def check_training_worker() -> bool:
    host, port = _endpoint()
    try:
        sock = socket.create_connection((host, port), timeout=2)
        sock.close()
        return True
    except (ConnectionRefusedError, OSError):
        return False
