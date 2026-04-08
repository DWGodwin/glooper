import logging
import os
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_config = None

DEFAULTS = {
    "chip_size_m": 67.2,
    "chip_size_px": 448,
    "crs": "EPSG:32619",
    "max_chips_per_request": 10000,
    "db_path": "data/glooper.duckdb",
    "data_dir": "public/data",
    "worker_host": "localhost",
    "worker_port": 9100,
    "plugins": [],
}


def get_config():
    global _config
    if _config is not None:
        return _config

    config_path = Path(os.environ.get("GLOOPER_CONFIG", "config.yaml")).resolve()
    if config_path.exists():
        with open(config_path) as f:
            _config = {**DEFAULTS, **yaml.safe_load(f)}
    else:
        _config = dict(DEFAULTS)

    # Resolve paths relative to config file's parent directory
    config_dir = config_path.parent if config_path.exists() else Path.cwd()
    for key in ("data_dir", "db_path"):
        p = Path(_config[key])
        if not p.is_absolute():
            _config[key] = str(config_dir / p)

    # Normalize plugins: bare strings → {"name": str}
    raw = _config.get("plugins", [])
    _config["plugins"] = [
        p if isinstance(p, dict) else {"name": p}
        for p in raw
    ]

    return _config


def get_enabled_plugins() -> list[str]:
    return [p["name"] for p in get_config()["plugins"]]


def get_plugin_config(name: str) -> dict:
    for p in get_config()["plugins"]:
        if p["name"] == name:
            return p
    return {}


def validate_plugin_deps():
    enabled = set(get_enabled_plugins())
    for p in get_config()["plugins"]:
        dep = p.get("requires_embeddings")
        if dep and dep not in enabled:
            logger.warning(
                "Plugin '%s' requires embeddings from '%s', but '%s' is not enabled",
                p["name"], dep, dep,
            )
