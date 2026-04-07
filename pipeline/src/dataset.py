"""ChipDataset — PyTorch Dataset for batched GPU processing of GeoTIFF chips."""

from pathlib import Path

import duckdb
import numpy as np
import rasterio
from rasterio.features import rasterize
import torch
from torch.utils.data import Dataset


class ChipDataset(Dataset):
    """Reads GeoTIFF chips with optional label burning.

    Class attributes define the full band list and RGB subset. Override in
    subclasses or pass ``bands`` at init to select a subset.

    Each item returns::

        {"chip_id": str,
         "image": Tensor(C, H, W) float32 [0, 1],
         "mask": Tensor(H, W) int64 | None}
    """

    all_bands: list[str] = ["R", "G", "B", "NIR"]
    rgb_bands: list[str] = ["R", "G", "B"]

    def __init__(
        self,
        chip_ids: list[str],
        chips_dir: str | Path,
        labels: dict[str, list[tuple[bytes, str]]] | None = None,
        bands: list[str] | None = None,
    ):
        self.chip_ids = chip_ids
        self.chips_dir = Path(chips_dir)
        self.labels = labels
        self._bands = bands or self.all_bands
        self._band_indices = [self.all_bands.index(b) + 1 for b in self._bands]  # 1-based for rasterio
        self._rgb_indices = [self._bands.index(b) for b in self.rgb_bands if b in self._bands]

    @classmethod
    def from_directory(cls, chips_dir: str | Path, **kwargs):
        """Construct a dataset from all .tif files in a directory."""
        chips_dir = Path(chips_dir)
        chip_ids = sorted(p.stem for p in chips_dir.glob("*.tif"))
        return cls(chip_ids, chips_dir, **kwargs)

    def __len__(self):
        return len(self.chip_ids)

    def __getitem__(self, idx):
        chip_id = self.chip_ids[idx]
        path = self.chips_dir / f"{chip_id}.tif"

        with rasterio.open(path) as src:
            data = src.read(indexes=self._band_indices)  # (C, H, W)
            dtype = src.dtypes[0]
            transform = src.transform
            crs = src.crs
            h, w = src.height, src.width

        # Normalize to float32 [0, 1]
        if np.issubdtype(np.dtype(dtype), np.integer):
            max_val = np.iinfo(np.dtype(dtype)).max
        else:
            max_val = data.max() or 1.0
        image = torch.from_numpy(data.astype(np.float32) / max_val)

        # Burn labels if available
        mask = None
        if self.labels is not None:
            mask = self._burn(chip_id, (h, w), transform, crs)

        return {
            "chip_id": chip_id,
            "image": image,
            "mask": mask,
        }

    def _burn(self, chip_id: str, shape: tuple[int, int], transform, crs) -> torch.Tensor | None:
        """Rasterize pre-clipped label polygons into a mask."""
        label_entries = self.labels.get(chip_id)
        if not label_entries:
            return None

        from shapely import wkb

        CLASS_MAP = {"positive": 1, "negative": 2}
        shapes = []
        for wkb_bytes, cls in label_entries:
            geom = wkb.loads(wkb_bytes)
            if geom.is_empty:
                continue
            val = CLASS_MAP.get(cls, 1)
            shapes.append((geom, val))

        if not shapes:
            return None

        mask = rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype=np.int64)
        return torch.from_numpy(mask)

    @property
    def rgb_indices(self) -> list[int]:
        """0-based indices into the band dimension for RGB extraction."""
        return self._rgb_indices

    def to_rgb(self, tensor: torch.Tensor) -> torch.Tensor:
        """Extract RGB bands from a (C, H, W) tensor → (3, H, W)."""
        return tensor[self._rgb_indices]

    def to_rgb_uint8(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert (C, H, W) float32 tensor → (H, W, 3) uint8 ndarray."""
        rgb = self.to_rgb(tensor)
        return (rgb * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()


def sam_collate(batch: list[dict]) -> dict:
    """Collate for SAM encoder: extract RGB, resize to 1024, SAM-normalize.

    Returns::

        {"chip_ids": [str, ...],
         "pixel_values": Tensor(B, 3, 1024, 1024)}
    """
    import torch.nn.functional as F

    # SAM pixel normalization constants
    mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1) / 255.0
    std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1) / 255.0

    chip_ids = [s["chip_id"] for s in batch]
    images = []
    for s in batch:
        img = s["image"]
        # Take first 3 bands as RGB
        rgb = img[:3] if img.shape[0] >= 3 else img
        # Resize to 1024x1024
        rgb = F.interpolate(rgb.unsqueeze(0), size=(1024, 1024), mode="bilinear", align_corners=False).squeeze(0)
        # SAM normalization
        rgb = (rgb - mean) / std
        images.append(rgb)

    return {
        "chip_ids": chip_ids,
        "pixel_values": torch.stack(images),
    }


def dino_collate(batch: list[dict]) -> dict:
    """Collate for DINOv2: extract RGB, resize to 518, ImageNet-normalize.

    Returns::

        {"chip_ids": [str, ...],
         "pixel_values": Tensor(B, 3, 518, 518)}
    """
    import torch.nn.functional as F

    # ImageNet normalization constants
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    chip_ids = [s["chip_id"] for s in batch]
    images = []
    for s in batch:
        img = s["image"]
        rgb = img[:3] if img.shape[0] >= 3 else img
        rgb = F.interpolate(rgb.unsqueeze(0), size=(518, 518), mode="bicubic", align_corners=False).squeeze(0)
        rgb = (rgb - mean) / std
        images.append(rgb)

    return {
        "chip_ids": chip_ids,
        "pixel_values": torch.stack(images),
    }


def query_labels(db_path: str | Path, chip_ids: list[str]) -> dict[str, list[tuple[bytes, str]]]:
    """Run the spatial join query against DuckDB (read-only).

    Returns {chip_id: [(wkb_bytes, class_str), ...]} for use as the
    ``labels`` argument to ChipDataset. Runs once at dataset construction
    time, before DataLoader workers fork.
    """
    if not chip_ids:
        return {}

    conn = duckdb.connect(str(db_path), read_only=True)
    conn.execute("LOAD spatial")

    placeholders = ", ".join(["?"] * len(chip_ids))
    rows = conn.execute(
        f"""
        SELECT c.id AS chip_id,
               ST_AsBinary(ST_Intersection(l.geometry, c.geometry)) AS label_geom,
               l.class
        FROM chips c
        JOIN labels l ON ST_Intersects(c.geometry, l.geometry)
        WHERE c.id IN ({placeholders})
        """,
        chip_ids,
    ).fetchall()
    conn.close()

    result: dict[str, list[tuple[bytes, str]]] = {}
    for chip_id, wkb, cls in rows:
        result.setdefault(chip_id, []).append((bytes(wkb), cls))
    return result
