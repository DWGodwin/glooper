"""HTTP client used by the training worker to read chips/labels and write
predictions through the FastAPI server, so the worker never opens DuckDB
directly (DuckDB only allows one process to hold the file lock).
"""

import json
import logging
import os
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

_DEFAULT_BASE = "http://localhost:8000"
_TIMEOUT = 60.0


class ServerUnavailable(Exception):
    pass


def _base_url() -> str:
    return os.environ.get("GLOOPER_SERVER_URL", _DEFAULT_BASE).rstrip("/")


def _request(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{_base_url()}{path}"
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Server returned HTTP {e.code} for {method} {path}: {e.read().decode(errors='replace')}"
        ) from e
    except urllib.error.URLError as e:
        raise ServerUnavailable(f"Cannot reach server at {url}: {e}") from e


def fetch_training_chips() -> tuple[list[str], list[str]]:
    """Return (train_chip_ids, validate_chip_ids). The server filters to complete=True only."""
    payload = _request("GET", "/api/training/chips")
    train = [c["id"] for c in payload["train"]]
    validate = [c["id"] for c in payload["validate"]]
    return train, validate


def fetch_labels(chip_ids: list[str]) -> dict[str, list[tuple[bytes, str]]]:
    """Return {chip_id: [(wkb_bytes, class), ...]} for chip-label rasterization."""
    if not chip_ids:
        return {}
    payload = _request("POST", "/api/training/labels", body={"chip_ids": chip_ids})
    return {
        cid: [(bytes.fromhex(entry["wkb_hex"]), entry["class"]) for entry in entries]
        for cid, entries in payload.items()
    }


def post_prediction_mask(run_id: str, chip_id: str, score: float, png_bytes: bytes) -> bool:
    """POST one chip's U-Net probability mask (RGBA PNG, alpha encodes probability)."""
    import base64

    resp = _request(
        "POST",
        f"/api/training/runs/{run_id}/predictions",
        body={
            "chip_id": chip_id,
            "score": float(score),
            "mask_png_b64": base64.b64encode(png_bytes).decode("ascii"),
        },
    )
    return bool(resp.get("ok", False))
