import re

import pytest

from server.grid import compute_grid


CHIP_SIZE = 67.2
CRS = "EPSG:32619"


def test_basic_bbox_produces_chips():
    chips = compute_grid(
        sw_lonlat=[-72.6, 42.2],
        ne_lonlat=[-72.598, 42.202],
        split="train",
        chip_size_m=CHIP_SIZE,
        crs=CRS,
    )
    assert len(chips) > 0
    for chip in chips:
        assert chip["split"] == "train"
        assert "id" in chip
        assert "geometry_wkt" in chip


def test_chip_id_format():
    chips = compute_grid(
        sw_lonlat=[-72.6, 42.2],
        ne_lonlat=[-72.598, 42.202],
        split="train",
        chip_size_m=CHIP_SIZE,
        crs=CRS,
    )
    pattern = re.compile(r"^\d+\.\d{2}e_\d+\.\d{2}n$")
    for chip in chips:
        assert pattern.match(chip["id"]), f"unexpected id: {chip['id']}"


def test_chips_snap_to_grid():
    chips = compute_grid(
        sw_lonlat=[-72.6, 42.2],
        ne_lonlat=[-72.598, 42.202],
        split="train",
        chip_size_m=CHIP_SIZE,
        crs=CRS,
    )
    for chip in chips:
        e_str, n_str = chip["id"].split("_")
        e = float(e_str.rstrip("e"))
        n = float(n_str.rstrip("n"))
        # Each chip origin should be an integer multiple of chip_size_m
        assert e / CHIP_SIZE == pytest.approx(round(e / CHIP_SIZE), abs=1e-6)
        assert n / CHIP_SIZE == pytest.approx(round(n / CHIP_SIZE), abs=1e-6)


def test_wkt_vertex_order_is_ne_se_sw_nw_ne():
    chips = compute_grid(
        sw_lonlat=[-72.6, 42.2],
        ne_lonlat=[-72.598, 42.202],
        split="train",
        chip_size_m=CHIP_SIZE,
        crs=CRS,
    )
    chip = chips[0]
    coord_text = re.search(r"\(\((.+)\)\)", chip["geometry_wkt"]).group(1)
    verts = [tuple(map(float, p.strip().split())) for p in coord_text.split(",")]
    assert len(verts) == 5, "WKT polygon should have 4 unique vertices plus closure"
    ne, se, sw, nw, close = verts
    assert close == ne, "ring should close on NE corner"
    # NE has max easting AND max northing
    assert ne[0] > sw[0] and ne[1] > sw[1]
    # SE shares easting with NE, northing with SW
    assert se[0] == ne[0] and se[1] == sw[1]
    # NW shares easting with SW, northing with NE
    assert nw[0] == sw[0] and nw[1] == ne[1]


def test_max_chips_exceeded_raises_with_count():
    # Larger bbox that should easily exceed a tiny cap
    with pytest.raises(ValueError) as exc:
        compute_grid(
            sw_lonlat=[-72.6, 42.2],
            ne_lonlat=[-72.598, 42.202],
            split="train",
            chip_size_m=CHIP_SIZE,
            crs=CRS,
            max_chips=1,
        )
    msg = str(exc.value)
    assert "exceeding limit of 1" in msg
    # The error message should contain the actual chip count
    assert re.search(r"create \d+ chips", msg)


def test_inverted_bbox_returns_empty():
    chips = compute_grid(
        sw_lonlat=[-72.598, 42.202],
        ne_lonlat=[-72.6, 42.2],
        split="train",
        chip_size_m=CHIP_SIZE,
        crs=CRS,
    )
    assert chips == []


def test_split_is_propagated():
    chips = compute_grid(
        sw_lonlat=[-72.6, 42.2],
        ne_lonlat=[-72.598, 42.202],
        split="test",
        chip_size_m=CHIP_SIZE,
        crs=CRS,
    )
    assert all(c["split"] == "test" for c in chips)
