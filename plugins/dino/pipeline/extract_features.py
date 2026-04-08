"""
Extract DINOv2 patch-level features from satellite image chips.

Reads GeoTIFF chips from data/chips/, batches through DINOv2 via
DataLoader, and saves the patch token features as .npy files
to data/embeddings/dino/.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import yaml

from pipeline.src.dataset import ChipDataset
from plugins.dino.pipeline.collate import dino_collate

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

_cfg = None
def _get_config():
    global _cfg
    if _cfg is None:
        config_path = Path(os.environ.get("GLOOPER_CONFIG", PROJECT_ROOT / "config.yaml"))
        if config_path.exists():
            with open(config_path) as f:
                _cfg = yaml.safe_load(f) or {}
        else:
            _cfg = {}
    return _cfg

def _get_plugin_config(name):
    for p in _get_config().get("plugins", []):
        if isinstance(p, dict) and p.get("name") == name:
            return p
    return {}

DATA_DIR = PROJECT_ROOT / _get_config().get("data_dir", "data")
CHIPS_DIR = DATA_DIR / "chips"
FEATURES_DIR = DATA_DIR / "embeddings" / "dino"
MODEL_NAME = _get_plugin_config("dino").get("model", "dinov2_vits14")


def main():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # Build dataset, filtering out chips that already have features
    all_ids = sorted(p.stem for p in CHIPS_DIR.glob("*.tif"))
    chip_ids = [cid for cid in all_ids if not (FEATURES_DIR / f"{cid}.npy").exists()]
    if not all_ids:
        print(f"No chips found in {CHIPS_DIR}")
        sys.exit(1)
    if not chip_ids:
        print(f"All {len(all_ids)} chips already have features")
        return

    print(f"Found {len(all_ids)} chips, {len(chip_ids)} need features")

    # Load DINOv2
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Model: {MODEL_NAME}")

    model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME, pretrained=True)
    model = model.to(device)
    model.eval()

    dataset = ChipDataset(chip_ids, CHIPS_DIR)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=dino_collate,
        pin_memory=(device == "cuda"),
    )

    done = 0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        with torch.no_grad():
            features = model.forward_features(pixel_values)
            patch_tokens = features["x_norm_patchtokens"]  # (B, n_patches, 384)

        for i, chip_id in enumerate(batch["chip_ids"]):
            np.save(FEATURES_DIR / f"{chip_id}.npy", patch_tokens[i].cpu().numpy())

        done += len(batch["chip_ids"])
        if done % 10 <= len(batch["chip_ids"]) or done == len(chip_ids):
            print(f"  [{done}/{len(chip_ids)}] {patch_tokens.shape[1]} patches × {patch_tokens.shape[2]} dims")

    print("Done.")


if __name__ == "__main__":
    main()
