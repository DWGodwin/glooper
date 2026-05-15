import io

import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box

from pipeline.src.providers import load_provider
from pipeline.src.providers.cog import COGProvider


def _write_synthetic_cog(path, height=256, width=256, count=4, crs="EPSG:32619"):
    """Write a multi-band GeoTIFF that COGProvider can read windowed from."""
    import numpy as np

    transform = from_bounds(0, 0, width, height, width, height)
    data = np.zeros((count, height, width), dtype="uint8")
    for i in range(count):
        data[i] = (i + 1) * 50
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype="uint8",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)
    return path


def test_load_provider_short_name(tmp_path):
    cog_path = _write_synthetic_cog(tmp_path / "src.tif")
    provider = load_provider("cog", {"cog_path": str(cog_path), "chip_size": 64})
    assert isinstance(provider, COGProvider)


def test_load_provider_dotted_path(tmp_path):
    cog_path = _write_synthetic_cog(tmp_path / "src.tif")
    provider = load_provider(
        "pipeline.src.providers.cog", {"cog_path": str(cog_path)}
    )
    assert isinstance(provider, COGProvider)


def test_load_provider_unknown_raises():
    with pytest.raises(ValueError, match="Unknown imagery provider"):
        load_provider("nope", {})


def test_load_provider_missing_create_raises():
    # The providers package itself has no create() function.
    with pytest.raises(ValueError, match="no create"):
        load_provider("pipeline.src.providers", {})


def test_cog_provider_requires_cog_path():
    from pipeline.src.providers.cog import create

    with pytest.raises(ValueError, match="cog_path"):
        create({})


def test_cog_get_chip_image_returns_valid_geotiff(tmp_path):
    cog_path = _write_synthetic_cog(tmp_path / "src.tif", height=256, width=256, count=4)
    provider = COGProvider(str(cog_path), chip_size=64)

    # Chip covering pixels (50, 50) -> (114, 114) of the source.
    chip_geom = box(50, 50, 114, 114)
    tif_bytes = provider.get_chip_image(
        "test_chip", chip_geom.wkt, "EPSG:32619"
    )

    with rasterio.MemoryFile(tif_bytes) as memfile:
        with memfile.open() as src:
            assert src.count == 4
            assert src.width == 64
            assert src.height == 64
            assert src.crs.to_string() == "EPSG:32619"
