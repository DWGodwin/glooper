"""
Step 3: Generate SAM encoder embeddings and export the decoder to ONNX.

Runs SAM ViT-B encoder on each chip via DataLoader and saves the image
embeddings as .npy files. Also exports the SAM decoder as an ONNX model
for in-browser inference via onnxruntime-web.

Outputs:
  - public/data/sam_embeddings/{chip_id}.npy  — SAM image embedding (1, 256, 64, 64) float32
  - public/data/sam_decoder.onnx              — SAM decoder ONNX model (~16MB)
"""

import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel
from torch.utils.data import DataLoader

from pipeline.src.dataset import ChipDataset, sam_collate

CHIPS_DIR = Path(__file__).parent.parent / "public" / "data" / "chips"
DATA_DIR = Path(__file__).parent.parent / "public" / "data"
EMBEDDINGS_DIR = DATA_DIR / "sam_embeddings"
ONNX_PATH = DATA_DIR / "sam_decoder.onnx"

MODELS_DIR = Path(__file__).parent / "models"
SAM_CHECKPOINT = MODELS_DIR / "sam_vit_b_01ec64.pth"
SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"


def export_decoder_onnx(sam):
    """Export the SAM decoder to ONNX for in-browser inference."""
    if ONNX_PATH.exists():
        print(f"  ONNX decoder already exists at {ONNX_PATH}, skipping export")
        return

    print("  Exporting SAM decoder to ONNX...")
    onnx_model = SamOnnxModel(sam.cpu(), return_single_mask=False)

    dummy_inputs = {
        "image_embeddings": torch.randn(1, 256, 64, 64),
        "point_coords": torch.randint(0, 1024, (1, 2, 2), dtype=torch.float),
        "point_labels": torch.randint(0, 4, (1, 2), dtype=torch.float),
        "mask_input": torch.randn(1, 1, 256, 256),
        "has_mask_input": torch.tensor([1.0]),
        "orig_im_size": torch.tensor([512.0, 512.0]),
    }

    torch.onnx.export(
        onnx_model,
        tuple(dummy_inputs.values()),
        str(ONNX_PATH),
        input_names=list(dummy_inputs.keys()),
        output_names=["masks", "iou_predictions", "low_res_masks"],
        dynamic_axes={
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"  Saved ONNX decoder to {ONNX_PATH}")


def main():
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    if not SAM_CHECKPOINT.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading SAM checkpoint (~375MB)...")
        urllib.request.urlretrieve(SAM_CHECKPOINT_URL, SAM_CHECKPOINT)
        print(f"  Saved to {SAM_CHECKPOINT}")

    # Build dataset, filtering out chips that already have embeddings
    all_ids = sorted(p.stem for p in CHIPS_DIR.glob("*.tif"))
    chip_ids = [cid for cid in all_ids if not (EMBEDDINGS_DIR / f"{cid}.npy").exists()]
    if not all_ids:
        print(f"No chips found in {CHIPS_DIR}")
        sys.exit(1)
    if not chip_ids:
        print(f"All {len(all_ids)} chips already have embeddings")
        return

    print(f"Found {len(all_ids)} chips, {len(chip_ids)} need embeddings")

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    sam = sam_model_registry[MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
    sam = sam.to(device)
    sam.eval()

    export_decoder_onnx(sam)
    sam = sam.to(device)

    dataset = ChipDataset(chip_ids, CHIPS_DIR)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=sam_collate,
        pin_memory=(device == "cuda"),
    )

    print("Generating embeddings...")
    done = 0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        with torch.no_grad():
            embeddings = sam.image_encoder(pixel_values)  # (B, 256, 64, 64)

        for i, chip_id in enumerate(batch["chip_ids"]):
            emb = embeddings[i : i + 1].cpu().numpy()  # (1, 256, 64, 64)
            np.save(EMBEDDINGS_DIR / f"{chip_id}.npy", emb)

        done += len(batch["chip_ids"])
        if done % 10 <= len(batch["chip_ids"]) or done == len(chip_ids):
            print(f"  [{done}/{len(chip_ids)}] {embeddings.shape}")

    print("Done.")


if __name__ == "__main__":
    main()
