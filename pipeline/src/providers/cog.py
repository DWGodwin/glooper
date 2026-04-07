"""COG provider — windowed read from a Cloud Optimized GeoTIFF."""

import io

import rasterio
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
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
            # Read all bands at native dtype
            data = src.read(
                window=window,
                out_shape=(src.count, self._chip_size, self._chip_size),
            )
            src_dtype = src.dtypes[0]
            src_count = src.count
            src_crs = src.crs

        # Build a transform for the output chip in the chip's own CRS
        chip_transform = transform_from_bounds(
            *chip_bounds, self._chip_size, self._chip_size
        )

        buf = io.BytesIO()
        with rasterio.open(
            buf,
            "w",
            driver="GTiff",
            height=self._chip_size,
            width=self._chip_size,
            count=src_count,
            dtype=src_dtype,
            crs=crs,
            transform=chip_transform,
        ) as dst:
            dst.write(data)

        return buf.getvalue()


def create(config: dict) -> COGProvider:
    cog_path = config.get("cog_path")
    if not cog_path:
        raise ValueError("COGProvider requires 'cog_path' in imagery_provider_config")
    chip_size = config.get("chip_size", 448)
    return COGProvider(cog_path, chip_size)
