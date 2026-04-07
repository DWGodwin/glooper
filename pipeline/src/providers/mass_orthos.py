"""MassGIS Orthoimagery provider.

Downloads JP2 tiles from the MassGIS ortho catalog on demand,
converts to TIF, and renders chip-sized GeoTIFFs via windowed reads.

Config keys:
    year: int          — ortho year (default 2021)
    cache_dir: str     — where to cache downloaded tiles (default "data/mass_orthos")
"""

import io
import logging
import threading
from pathlib import Path

import geopandas as gpd
import pooch
import rasterio
import rioxarray as rxr
import shapely
from pooch import Unzip
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import transform_bounds, transform_geom
from rasterio.windows import from_bounds
from shapely import wkt

logger = logging.getLogger(__name__)

_CATALOG_URL = (
    "https://s3.us-east-1.amazonaws.com/download.massgis.digital.mass.gov/"
    "images/coq{year}_15cm_jp2/COQ{year}INDEX_POLY.zip"
)
_TILE_BASE_URL = (
    "https://s3.us-east-1.amazonaws.com/download.massgis.digital.mass.gov/"
    "images/coq{year}_15cm_jp2/"
)


class MassOrthosProvider:
    def __init__(self, year: int, cache_dir: str, crs: str, chip_size: int = 448):
        self._year = year
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._crs = crs
        self._chip_size = chip_size
        self._catalog = None
        self._tile_locks: dict[str, threading.Lock] = {}
        self._tile_locks_lock = threading.Lock()

    def _load_catalog(self) -> gpd.GeoDataFrame:
        if self._catalog is not None:
            return self._catalog
        url = _CATALOG_URL.format(year=self._year)
        catalog = gpd.read_file(url)
        src_crs = str(catalog.crs)
        logger.info("Loaded catalog (%s), reprojecting to %s", src_crs, self._crs)
        # defensively use rasterio warping operations over geopandas to_crs to avoid PYPROJ issues
        # TODO: look into the environment in pixi to try and fix this if possible
        # catalog = catalog.set_geometry(
        #     [shapely.geometry.shape(transform_geom(src_crs, self._crs, shapely.geometry.mapping(g)))
        #      for g in catalog.geometry],
        #     crs=self._crs,
        # )
        catalog = catalog.to_crs(self._crs)
        self._catalog = catalog
        return self._catalog

    def _find_overlapping_tiles(self, geom_wkt: str) -> list[str]:
        """Return TILENAME values for ortho tiles overlapping the chip geometry."""
        catalog = self._load_catalog()
        chip_gdf = gpd.GeoDataFrame(geometry=[wkt.loads(geom_wkt)], crs=self._crs)
        joined = gpd.sjoin(catalog, chip_gdf)
        return joined["TILENAME"].unique().tolist()

    def _get_tile_lock(self, tilename: str) -> threading.Lock:
        with self._tile_locks_lock:
            if tilename not in self._tile_locks:
                self._tile_locks[tilename] = threading.Lock()
            return self._tile_locks[tilename]

    def _ensure_tif(self, tilename: str) -> Path:
        """Download the JP2 tile (if needed), convert to TIF, return TIF path."""
        tif_path = self._cache_dir / f"{tilename}.tif"
        if tif_path.exists():
            return tif_path

        with self._get_tile_lock(tilename):
            # Re-check after acquiring lock
            if tif_path.exists():
                return tif_path

            # Download and unzip the JP2
            downloader = pooch.create(
                path=self._cache_dir / "downloads",
                base_url=_TILE_BASE_URL.format(year=self._year),
                registry={f"{tilename}.zip": None},
            )
            extracted = downloader.fetch(f"{tilename}.zip", processor=Unzip())
            jp2_files = [f for f in extracted if f.endswith(".jp2")]
            if not jp2_files:
                raise FileNotFoundError(f"No JP2 found in {tilename}.zip")

            # Convert JP2 -> TIF
            rxr.open_rasterio(jp2_files[0]).rio.to_raster(str(tif_path))
            return tif_path

    def get_chip_image(self, chip_id: str, geometry_wkt: str, crs: str) -> bytes:
        tilenames = self._find_overlapping_tiles(geometry_wkt)
        if not tilenames:
            raise FileNotFoundError(
                f"No MassGIS tiles overlap chip {chip_id}"
            )

        # Use the first overlapping tile (most chips fit in one tile)
        tif_path = self._ensure_tif(tilenames[0])

        geom = wkt.loads(geometry_wkt)
        chip_bounds = geom.bounds

        with rasterio.open(tif_path) as src:
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


def create(config: dict) -> MassOrthosProvider:
    year = config.get("year", 2023)
    cache_dir = config.get("cache_dir", "data/mass_orthos")
    crs = config.get("crs", "EPSG:4326")
    chip_size = config.get("chip_size", 448)
    return MassOrthosProvider(year, cache_dir, crs, chip_size)
