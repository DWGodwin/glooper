"""TrainingDataset — wraps ChipDataset to emit (image, label_mask).

Only `complete=True` chips reach this dataset (filtered server-side), so every
pixel is supervised — full BCE over the whole chip, no per-pixel loss mask.

Chips whose TIFF is missing on disk are dropped at construction with a single
warning log line — re-prefetch is the user's responsibility.
"""

import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset

from pipeline.src.dataset import ChipDataset

logger = logging.getLogger(__name__)


class TrainingDataset(Dataset):
    def __init__(
        self,
        chip_ids: list[str],
        chips_dir: str | Path,
        labels: dict[str, list[tuple[bytes, str]]],
    ):
        chips_dir = Path(chips_dir)

        present: list[str] = []
        missing: list[str] = []
        for chip_id in chip_ids:
            if (chips_dir / f"{chip_id}.tif").exists():
                present.append(chip_id)
            else:
                missing.append(chip_id)

        if missing:
            logger.warning(
                "TrainingDataset: skipping %d chip(s) with missing TIFFs (first: %s)",
                len(missing), missing[0],
            )

        self.chip_ids = present

        positive_labels = {
            cid: [(wkb, cls) for wkb, cls in v if cls == "positive"]
            for cid, v in labels.items()
        }
        self._inner = ChipDataset(
            present,
            chips_dir,
            labels=positive_labels,
            bands=["R", "G", "B"],
        )

    def __len__(self) -> int:
        return len(self.chip_ids)

    def __getitem__(self, idx: int) -> dict:
        item = self._inner[idx]
        image = item["image"]
        h, w = image.shape[-2], image.shape[-1]

        rasterized = item["mask"]
        if rasterized is None:
            label_mask = torch.zeros((h, w), dtype=torch.float32)
        else:
            label_mask = (rasterized == 1).float()

        return {
            "chip_id": item["chip_id"],
            "image": image,
            "label_mask": label_mask,
        }
