"""
Step 1: Extract DINOv2 ViT-S/14 patch-level features from satellite image chips.

Reads GeoTIFF chips from public/data/chips/, batches through DINOv2 via
DataLoader, and saves the patch token features as .npy files.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pipeline.src.dataset import ChipDataset, dino_collate

CHIPS_DIR = Path(__file__).parent.parent / "public" / "data" / "chips"
FEATURES_DIR = Path(__file__).parent / "features"


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

    # Load DINOv2 ViT-S/14
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
    model = model.to(device)
    model.eval()

    dataset = ChipDataset(chip_ids, CHIPS_DIR)
    loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=4,
        collate_fn=dino_collate,
        pin_memory=(device != "cpu"),
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
