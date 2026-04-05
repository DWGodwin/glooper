"""COG provider — windowed read from a Cloud Optimized GeoTIFF."""

import io

import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from PIL import Image
from shapely import wkt


class COGProvider:
    def __init__(self, cog_path: str, chip_size: int = 448):
        self._cog_path = cog_path
        self._chip_size = chip_size

    def get_chip_image(self, chip_id: str, geometry_wkt: str, crs: str) -> bytes:
        geom = wkt.loads(geometry_wkt)
        chip_bounds = geom.bounds  # (minx, miny, maxx, maxy) in chip CRS

        with rasterio.open(self._cog_path) as src:
            # Transform chip bounds from chip CRS to COG native CRS
            src_bounds = transform_bounds(crs, src.crs, *chip_bounds)

            window = from_bounds(*src_bounds, transform=src.transform)
            # Read RGB bands, resample to chip_size x chip_size
            data = src.read(
                indexes=(1, 2, 3),
                window=window,
                out_shape=(3, self._chip_size, self._chip_size),
            )

        # data shape: (3, H, W) -> transpose to (H, W, 3) for PIL
        img = Image.fromarray(data.transpose(1, 2, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def create(config: dict) -> COGProvider:
    cog_path = config.get("cog_path")
    if not cog_path:
        raise ValueError("COGProvider requires 'cog_path' in imagery_provider_config")
    chip_size = config.get("chip_size", 448)
    return COGProvider(cog_path, chip_size)
