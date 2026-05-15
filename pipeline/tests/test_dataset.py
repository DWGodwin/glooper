import numpy as np
import pytest
import torch
from shapely.geometry import Polygon
from shapely import wkb as shapely_wkb

from pipeline.src.dataset import ChipDataset, sam_collate


def test_default_bands_select_all_four(synthetic_tif, tmp_path):
    synthetic_tif(tmp_path / "a.tif")
    ds = ChipDataset(["a"], tmp_path)
    assert ds._band_indices == [1, 2, 3, 4]
    assert ds._rgb_indices == [0, 1, 2]


def test_rgb_only_band_subset(synthetic_tif, tmp_path):
    synthetic_tif(tmp_path / "a.tif")
    ds = ChipDataset(["a"], tmp_path, bands=["R", "G", "B"])
    assert ds._band_indices == [1, 2, 3]
    assert ds._rgb_indices == [0, 1, 2]


def test_nonstandard_band_subset(synthetic_tif, tmp_path):
    synthetic_tif(tmp_path / "a.tif")
    ds = ChipDataset(["a"], tmp_path, bands=["NIR", "R"])
    assert ds._band_indices == [4, 1]
    # Only R is present in the subset; G/B are dropped from rgb_indices.
    assert ds._rgb_indices == [1]


def test_from_directory_lists_sorted_ids(synthetic_tif, tmp_path):
    synthetic_tif(tmp_path / "b.tif")
    synthetic_tif(tmp_path / "a.tif")
    synthetic_tif(tmp_path / "c.tif")
    ds = ChipDataset.from_directory(tmp_path)
    assert len(ds) == 3
    assert ds.chip_ids == ["a", "b", "c"]


def test_getitem_returns_float32_normalized(synthetic_tif, tmp_path):
    synthetic_tif(tmp_path / "a.tif")
    ds = ChipDataset(["a"], tmp_path)
    item = ds[0]

    assert item["chip_id"] == "a"
    assert isinstance(item["image"], torch.Tensor)
    assert item["image"].shape == (4, 64, 64)
    assert item["image"].dtype == torch.float32
    assert 0.0 <= float(item["image"].min())
    assert float(item["image"].max()) <= 1.0
    assert item["mask"] is None


def test_getitem_with_band_subset_drops_channels(synthetic_tif, tmp_path):
    synthetic_tif(tmp_path / "a.tif")
    ds = ChipDataset(["a"], tmp_path, bands=["R", "G", "B"])
    item = ds[0]
    assert item["image"].shape == (3, 64, 64)


def test_burn_rasterizes_polygon_into_mask(synthetic_tif, tmp_path):
    # Chip bounds (0, 0, 64, 64) match the synthetic_tif default.
    synthetic_tif(tmp_path / "a.tif")
    poly = Polygon([(10, 10), (30, 10), (30, 30), (10, 30), (10, 10)])
    labels = {"a": [(shapely_wkb.dumps(poly), "positive")]}

    ds = ChipDataset(["a"], tmp_path, labels=labels)
    item = ds[0]

    assert item["mask"] is not None
    assert item["mask"].dtype == torch.int64
    assert item["mask"].shape == (64, 64)
    assert int(item["mask"].sum()) > 0
    assert bool((item["mask"] == 1).any())


def test_burn_empty_label_list_returns_none(synthetic_tif, tmp_path):
    synthetic_tif(tmp_path / "a.tif")
    ds = ChipDataset(["a"], tmp_path, labels={"a": []})
    assert ds[0]["mask"] is None


def test_to_rgb_extracts_three_channels(synthetic_tif, tmp_path):
    synthetic_tif(tmp_path / "a.tif")
    ds = ChipDataset(["a"], tmp_path)
    rgb = ds.to_rgb(ds[0]["image"])
    assert rgb.shape == (3, 64, 64)
    assert rgb.dtype == torch.float32


def test_to_rgb_uint8_returns_hw3_array(synthetic_tif, tmp_path):
    synthetic_tif(tmp_path / "a.tif")
    ds = ChipDataset(["a"], tmp_path)
    arr = ds.to_rgb_uint8(ds[0]["image"])
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (64, 64, 3)
    assert arr.dtype == np.uint8


def test_sam_collate_three_band_batch():
    batch = [
        {"chip_id": "a", "image": torch.zeros(3, 64, 64)},
        {"chip_id": "b", "image": torch.ones(3, 64, 64)},
    ]
    out = sam_collate(batch)
    assert out["chip_ids"] == ["a", "b"]
    assert out["pixel_values"].shape == (2, 3, 1024, 1024)
    assert out["pixel_values"].dtype == torch.float32
    # After SAM normalization, the all-zero input should sit well below 0 and
    # the all-one input well above 0 — sanity check that normalization ran.
    assert float(out["pixel_values"][0].mean()) < 0
    assert float(out["pixel_values"][1].mean()) > 0


def test_sam_collate_four_band_uses_first_three():
    # 4-band input (e.g. RGBA / RGB+NIR) — collate slices :3.
    img = torch.zeros(4, 64, 64)
    img[3] = 1.0  # NIR full — should be discarded
    out = sam_collate([{"chip_id": "a", "image": img}])
    assert out["pixel_values"].shape == (1, 3, 1024, 1024)


def test_sam_collate_single_band_broadcasts_to_three_channels():
    # Current behavior: when img has <3 channels, sam_collate leaves it as-is.
    # The subsequent `(rgb - mean)` broadcasts a (1, H, W) tensor against a
    # (3, 1, 1) constant, producing (3, H, W). This test pins that behavior
    # so we notice if the broadcast accidentally breaks.
    out = sam_collate([{"chip_id": "a", "image": torch.zeros(1, 64, 64)}])
    assert out["pixel_values"].shape == (1, 3, 1024, 1024)
