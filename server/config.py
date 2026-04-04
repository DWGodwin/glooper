import os
from pathlib import Path

import yaml

_config = None

DEFAULTS = {
    "chip_size_m": 76.8,
    "crs": "EPSG:32619",
    "max_chips_per_request": 10000,
    "db_path": "data/glooper.duckdb",
    "data_dir": "public/data",
}


def get_config():
    global _config
    if _config is not None:
        return _config

    config_path = Path(os.environ.get("GLOOPER_CONFIG", "config.yaml"))
    if config_path.exists():
        with open(config_path) as f:
            _config = {**DEFAULTS, **yaml.safe_load(f)}
    else:
        _config = dict(DEFAULTS)

    return _config
