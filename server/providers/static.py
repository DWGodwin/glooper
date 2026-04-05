"""Static file provider — reads pre-existing PNGs from disk."""

from pathlib import Path


class StaticProvider:
    def __init__(self, data_dir: str):
        self._chips_dir = Path(data_dir) / "chips"

    def get_chip_image(self, chip_id: str, geometry_wkt: str, crs: str) -> bytes:
        path = self._chips_dir / f"{chip_id}.png"
        if not path.exists():
            raise FileNotFoundError(f"No chip image at {path}")
        return path.read_bytes()


def create(config: dict) -> StaticProvider:
    from server.config import get_config
    data_dir = config.get("data_dir") or get_config()["data_dir"]
    return StaticProvider(data_dir)
