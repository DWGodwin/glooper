import numpy as np
import pytest
from rasterio.transform import from_bounds
from shapely.geometry import MultiPolygon, Polygon

from server.vectorize import vectorize_mask


# 10×10 mask, 1m pixels, projected origin at (0, 0) ↔ pixel (0, 10).
def _mask_and_transform(h=10, w=10):
    mask = np.zeros((h, w), dtype=np.uint8)
    transform = from_bounds(0, 0, w, h, w, h)
    return mask, transform


def _vertex_count(geom):
    if isinstance(geom, MultiPolygon):
        return sum(len(list(p.exterior.coords)) for p in geom.geoms)
    return len(list(geom.exterior.coords))


def test_single_block_returns_polygon():
    mask, transform = _mask_and_transform()
    mask[3:7, 3:7] = 1  # 4×4 block, fully interior

    result = vectorize_mask(
        mask, transform,
        {"strategy": "simplify", "tolerance_px": 0.5, "min_area_px": 0},
    )

    assert isinstance(result, Polygon)
    assert result.area == pytest.approx(16.0, abs=0.01)
    # Simplified rectangle: 4 unique vertices + closure
    assert _vertex_count(result) == 5


def test_edge_touching_block_is_flush_with_chip_boundary():
    mask, transform = _mask_and_transform()
    # Right-edge column filled top-to-bottom 1px wide; touches right edge → padding kicks in
    mask[2:8, 9:10] = 1

    result = vectorize_mask(
        mask, transform,
        {"strategy": "simplify", "tolerance_px": 0.5, "min_area_px": 0},
    )

    # max_e of the chip is 10 (w=10, pixel_size=1, transform.c=0)
    assert result.bounds[2] == pytest.approx(10.0, abs=1e-6)
    # And the geometry must stay inside the chip
    assert result.bounds[0] >= -1e-9
    assert result.bounds[3] <= 10 + 1e-9


def test_empty_mask_raises():
    mask, transform = _mask_and_transform()
    with pytest.raises(ValueError, match="produced no polygons"):
        vectorize_mask(
            mask, transform,
            {"strategy": "simplify", "tolerance_px": 0.5, "min_area_px": 0},
        )


def test_min_area_filter_drops_speck():
    mask, transform = _mask_and_transform()
    mask[5, 5] = 1
    mask[5, 6] = 1  # 2-pixel speck = 2 m² < 4 px * 1m²

    with pytest.raises(ValueError, match="filtered by min_area_px"):
        vectorize_mask(
            mask, transform,
            {"strategy": "simplify", "tolerance_px": 0.5, "min_area_px": 4},
        )


def test_convex_hull_collapses_l_shape():
    mask, transform = _mask_and_transform()
    # L-shape: vertical bar + horizontal foot
    mask[2:8, 3:5] = 1
    mask[6:8, 5:8] = 1

    simplified = vectorize_mask(
        mask, transform,
        {"strategy": "simplify", "tolerance_px": 0.5, "min_area_px": 0},
    )
    hulled = vectorize_mask(
        mask, transform,
        {"strategy": "convex_hull", "tolerance_px": 0.5, "min_area_px": 0},
    )

    # Convex hull of an L is a quadrilateral or pentagon — strictly fewer
    # vertices than the original concave outline.
    assert _vertex_count(hulled) < _vertex_count(simplified)
    # And the hull must contain the original geometry.
    assert hulled.contains(simplified.buffer(-1e-9))


def test_max_vertices_caps_count():
    # Disc roughly centered in a 50×50 mask gives a noisy edge with many
    # polygon vertices after rasterio.features.shapes traces the boundary.
    mask = np.zeros((50, 50), dtype=np.uint8)
    yy, xx = np.ogrid[:50, :50]
    mask[((yy - 25) ** 2 + (xx - 25) ** 2) <= 20 ** 2] = 1
    transform = from_bounds(0, 0, 50, 50, 50, 50)

    uncapped = vectorize_mask(
        mask, transform,
        {"strategy": "simplify", "tolerance_px": 0.1, "min_area_px": 0},
    )
    capped = vectorize_mask(
        mask, transform,
        {"strategy": "simplify", "tolerance_px": 0.1, "min_area_px": 0,
         "max_vertices": 12},
    )

    assert _vertex_count(uncapped) > 12
    assert _vertex_count(capped) <= 12


def test_unknown_strategy_raises():
    mask, transform = _mask_and_transform()
    mask[3:7, 3:7] = 1

    with pytest.raises(ValueError, match="Unknown vectorization strategy"):
        vectorize_mask(
            mask, transform,
            {"strategy": "bogus", "tolerance_px": 0.5, "min_area_px": 0},
        )
