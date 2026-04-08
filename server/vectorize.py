"""Vectorize binary masks to Shapely geometries with configurable strategies."""

import numpy as np
import rasterio.features
from rasterio.transform import Affine
from shapely.geometry import shape, MultiPolygon, Polygon
from shapely.ops import unary_union


def vectorize_mask(mask_array: np.ndarray, transform: Affine, config: dict):
    """Vectorize a binary mask array into a Shapely geometry.

    Args:
        mask_array: Binary uint8 array (1 = positive).
        transform: Rasterio affine mapping pixel coords to projected CRS.
        config: Vectorization config dict with keys: strategy, tolerance_px,
                min_area_px, max_vertices.

    Returns:
        Shapely geometry in projected CRS.
    """
    raw = _pad_and_clip_vectorize(mask_array, transform)
    if raw is None or raw.is_empty:
        raise ValueError("Mask vectorization produced no polygons")

    pixel_size = abs(transform.a)

    min_area_px = config.get("min_area_px", 0)
    if min_area_px > 0:
        raw = _filter_by_area(raw, pixel_size, min_area_px)
        if raw is None or raw.is_empty:
            raise ValueError("All polygons filtered by min_area_px")

    strategy = config.get("strategy", "simplify")
    tolerance_px = config.get("tolerance_px", 0.5)
    tolerance = pixel_size * tolerance_px

    if strategy == "simplify":
        result = _simplify(raw, tolerance)
    elif strategy == "convex_hull":
        result = _convex_hull(raw)
    elif strategy == "min_rotated_rect":
        result = _min_rotated_rect(raw)
    else:
        raise ValueError(f"Unknown vectorization strategy: {strategy}")

    max_vertices = config.get("max_vertices")
    if max_vertices:
        result = _cap_vertices(result, max_vertices, tolerance)

    result = _clip_to_chip(result, mask_array.shape, transform)

    return result


def _pad_and_clip_vectorize(mask_array: np.ndarray, transform: Affine):
    """Pad mask edges that touch the chip boundary, vectorize, then clip.

    When positive pixels touch a chip edge, rasterio.features.shapes traces
    the 0-to-1 boundary *inward* from the edge, leaving a sub-pixel gap.
    Padding those edges by one pixel pushes the traced boundary outside the
    chip, and clipping to the chip extent gives perfectly straight edges
    flush with the boundary.
    """
    h, w = mask_array.shape
    pixel_size = abs(transform.a)

    pad_t = int(mask_array[0, :].any())
    pad_b = int(mask_array[-1, :].any())
    pad_l = int(mask_array[:, 0].any())
    pad_r = int(mask_array[:, -1].any())

    if not (pad_t or pad_b or pad_l or pad_r):
        return _raw_vectorize(mask_array, transform)

    padded = np.pad(mask_array, ((pad_t, pad_b), (pad_l, pad_r)),
                    mode='edge')

    padded_transform = Affine(
        transform.a, transform.b, transform.c - pad_l * pixel_size,
        transform.d, transform.e, transform.f - pad_t * transform.e,
    )

    raw = _raw_vectorize(padded, padded_transform)
    return raw


def _clip_to_chip(geom, shape, transform):
    """Clip geometry to the chip's bounding box derived from the transform."""
    from shapely.geometry import box
    h, w = shape
    pixel_size = abs(transform.a)
    min_e = transform.c
    max_n = transform.f
    max_e = min_e + w * pixel_size
    min_n = max_n + h * transform.e  # transform.e is negative
    return geom.intersection(box(min_e, min_n, max_e, max_n))


def _raw_vectorize(mask_array: np.ndarray, transform: Affine):
    """Extract polygons from binary mask and merge into a single geometry."""
    polygons = []
    for geom, value in rasterio.features.shapes(mask_array, transform=transform):
        if value == 1:
            polygons.append(shape(geom))
    if not polygons:
        return None
    return unary_union(polygons)


def _filter_by_area(geom, pixel_size: float, min_area_px: float):
    """Drop polygon components smaller than min_area_px pixels."""
    min_area = min_area_px * pixel_size * pixel_size
    parts = geom.geoms if hasattr(geom, 'geoms') else [geom]
    kept = [p for p in parts if p.area >= min_area]
    if not kept:
        return None
    return unary_union(kept)


def _simplify(geom, tolerance: float):
    """Douglas-Peucker simplification."""
    return geom.simplify(tolerance, preserve_topology=True)


def _convex_hull(geom):
    """Per-component convex hull."""
    parts = geom.geoms if hasattr(geom, 'geoms') else [geom]
    hulls = [p.convex_hull for p in parts]
    return unary_union(hulls)


def _min_rotated_rect(geom):
    """Per-component minimum rotated rectangle."""
    parts = geom.geoms if hasattr(geom, 'geoms') else [geom]
    rects = [p.minimum_rotated_rectangle for p in parts]
    return unary_union(rects)


def _cap_vertices(geom, max_vertices: int, base_tolerance: float):
    """Iteratively simplify until vertex count is under max_vertices."""
    def _vertex_count(g):
        if hasattr(g, 'geoms'):
            return sum(_vertex_count(p) for p in g.geoms)
        coords = g.exterior.coords if hasattr(g, 'exterior') else []
        return len(list(coords))

    tolerance = base_tolerance
    result = geom
    for _ in range(20):
        if _vertex_count(result) <= max_vertices:
            break
        tolerance *= 1.5
        result = geom.simplify(tolerance, preserve_topology=True)
    return result
