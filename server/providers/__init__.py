"""Imagery provider protocol, loader, and caching wrapper."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from server.config import get_config

logger = logging.getLogger(__name__)

_provider = None


@runtime_checkable
class ImageryProvider(Protocol):
    def get_chip_image(self, chip_id: str, geometry_wkt: str, crs: str) -> bytes: ...


class CachingProvider:
    """Wraps any provider with a filesystem cache at {data_dir}/chips/."""

    def __init__(self, inner: ImageryProvider, cache_dir: Path):
        self._inner = inner
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def get_chip_image(self, chip_id: str, geometry_wkt: str, crs: str) -> bytes:
        cached = self._cache_dir / f"{chip_id}.png"
        if cached.exists():
            return cached.read_bytes()

        data = self._inner.get_chip_image(chip_id, geometry_wkt, crs)
        cached.write_bytes(data)
        return data


def load_provider(name: str, config: dict) -> ImageryProvider:
    """Import server.providers.{name} and call its create(config) function."""
    try:
        module = importlib.import_module(f"server.providers.{name}")
    except ModuleNotFoundError as e:
        raise ValueError(f"Unknown imagery provider '{name}': {e}") from e

    if not hasattr(module, "create"):
        raise ValueError(f"Provider module 'server.providers.{name}' has no create() function")

    return module.create(config)


def init_provider():
    """Load and cache the configured provider at startup."""
    global _provider
    cfg = get_config()
    name = cfg.get("imagery_provider", "static")
    provider_config = cfg.get("imagery_provider_config", {})

    inner = load_provider(name, provider_config)

    cache_dir = Path(cfg["data_dir"]) / "chips"
    _provider = CachingProvider(inner, cache_dir)
    logger.info("Imagery provider '%s' initialized (cache: %s)", name, cache_dir)


def get_provider() -> ImageryProvider:
    if _provider is None:
        raise RuntimeError("Provider not initialized — call init_provider() first")
    return _provider
