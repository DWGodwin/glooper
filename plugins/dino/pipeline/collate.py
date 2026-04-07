"""DINOv2 collate function — config-driven input size."""

import os
from pathlib import Path

import yaml


def _get_dino_input_size():
    config_path = Path(os.environ.get("GLOOPER_CONFIG", "config.yaml"))
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
    for p in cfg.get("plugins", []):
        if isinstance(p, dict) and p.get("name") == "dino":
            return p.get("input_size", 518)
    return 518


INPUT_SIZE = _get_dino_input_size()


def dino_collate(batch: list[dict]) -> dict:
    """Collate for DINOv2: extract RGB, resize to input_size, ImageNet-normalize.

    Returns::

        {"chip_ids": [str, ...],
         "pixel_values": Tensor(B, 3, INPUT_SIZE, INPUT_SIZE)}
    """
    import torch
    import torch.nn.functional as F

    # ImageNet normalization constants
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    chip_ids = [s["chip_id"] for s in batch]
    images = []
    for s in batch:
        img = s["image"]
        rgb = img[:3] if img.shape[0] >= 3 else img
        rgb = F.interpolate(rgb.unsqueeze(0), size=(INPUT_SIZE, INPUT_SIZE), mode="bicubic", align_corners=False).squeeze(0)
        rgb = (rgb - mean) / std
        images.append(rgb)

    return {
        "chip_ids": chip_ids,
        "pixel_values": torch.stack(images),
    }
