"""Imagery provider protocol and loader for the pipeline worker."""

from __future__ import annotations

import importlib
from typing import Protocol, runtime_checkable


@runtime_checkable
class ImageryProvider(Protocol):
    def get_chip_image(self, chip_id: str, geometry_wkt: str, crs: str) -> bytes:
        """Return chip imagery as GeoTIFF bytes (all bands, native dtype, CRS/transform embedded)."""
        ...


def load_provider(name: str, config: dict) -> ImageryProvider:
    """Import a provider module and call its create(config) function.

    Short names (e.g. "cog") resolve to pipeline.src.providers.{name}.
    Dotted paths are imported directly.
    """
    module_path = name if "." in name else f"pipeline.src.providers.{name}"
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ValueError(f"Unknown imagery provider '{name}': {e}") from e

    if not hasattr(module, "create"):
        raise ValueError(f"Provider module '{module_path}' has no create() function")

    return module.create(config)
