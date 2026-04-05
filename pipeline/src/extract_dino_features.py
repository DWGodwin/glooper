"""
Step 1: Extract DINOv2 ViT-S/14 patch-level features from satellite image chips.

Reads RGB PNGs from public/data/chips/, runs each through DINOv2,
and saves the patch token features as .npy files to pipeline/features/.

"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

CHIPS_DIR = Path(__file__).parent.parent / "public" / "data" / "chips"
FEATURES_DIR = Path(__file__).parent / "features"


def build_transform():
    """DINOv2 standard preprocessing: resize to match patch grid, normalize."""
    return transforms.Compose([
        # 37 * 14 = 518, ensures clean 37x37 patch grid
        transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def main():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    chip_paths = sorted(CHIPS_DIR.glob("*.png"))
    if not chip_paths:
        print(f"No chips found in {CHIPS_DIR}")
        sys.exit(1)

    print(f"Found {len(chip_paths)} chips")

    # Load DINOv2 ViT-S/14
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
    model = model.to(device)
    model.eval()

    transform = build_transform()

    for i, chip_path in enumerate(chip_paths):
        chip_id = chip_path.stem
        out_path = FEATURES_DIR / f"{chip_id}.npy"

        if out_path.exists():
            continue

        img = Image.open(chip_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            features = model.forward_features(tensor)
            patch_tokens = features["x_norm_patchtokens"]  # (1, n_patches, 384)

        np.save(out_path, patch_tokens[0].cpu().numpy())

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i + 1}/{len(chip_paths)}] {chip_id} → {patch_tokens.shape[1]} patches × {patch_tokens.shape[2]} dims")

    print("Done.")


if __name__ == "__main__":
    main()
