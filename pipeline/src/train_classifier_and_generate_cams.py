"""
Step 2: Train a linear classifier on DINOv2 patch features and generate CAMs.

Uses presence/absence labels from metadata.geojson to train a logistic regression
on mean-pooled DINOv2 features. Then generates per-patch class activation maps (CAMs)
by applying the learned weights to each patch token individually.

Outputs:
  - public/data/cams/{chip_id}.png       — RGBA heatmap overlay (transparent background)
  - public/data/cams_raw/{chip_id}.npy   — Raw float32 CAM array (37x37), used as SAM mask_input
"""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.cm as cm
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

DATA_DIR = Path(__file__).parent.parent / "public" / "data"
FEATURES_DIR = Path(__file__).parent / "features"
CAMS_DIR = DATA_DIR / "cams"
CAMS_RAW_DIR = DATA_DIR / "cams_raw"
METADATA_PATH = DATA_DIR / "metadata.geojson"

PATCH_GRID = 37  # 518px / 14px per patch


def load_dataset():
    """Load features and labels, returning mean-pooled features + labels + chip IDs."""
    with open(METADATA_PATH) as f:
        meta = json.load(f)

    chip_ids = []
    features = []
    labels = []

    for feat in meta["features"]:
        chip_id = feat["properties"]["id"]
        npy_path = FEATURES_DIR / f"{chip_id}.npy"
        if not npy_path.exists():
            print(f"  Warning: no features for {chip_id}, skipping")
            continue

        patch_features = np.load(npy_path)  # (1369, 384)
        mean_features = patch_features.mean(axis=0)  # (384,)

        chip_ids.append(chip_id)
        features.append(mean_features)
        labels.append(1 if feat["properties"]["label"] == "present" else 0)

    return np.array(features), np.array(labels), chip_ids


def train_classifier(X, y):
    """Train logistic regression and report cross-val accuracy."""
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")

    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(f"  Cross-val accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    clf.fit(X, y)
    train_acc = clf.score(X, y)
    print(f"  Train accuracy: {train_acc:.3f}")

    return clf


def generate_cam(clf, patch_features):
    """Generate CAM by applying classifier weights to each patch token.

    The logistic regression weight vector tells us how much each feature dimension
    contributes to the positive class. Dot product with each patch token gives
    a per-patch activation score.
    """
    weights = clf.coef_[0]  # (384,)
    bias = clf.intercept_[0]

    # Per-patch logit for the positive class
    cam = patch_features @ weights + bias  # (1369,)
    cam = cam.reshape(PATCH_GRID, PATCH_GRID)  # (37, 37)

    return cam


def cam_to_heatmap_png(cam, size=512):
    """Convert raw CAM to an RGBA heatmap PNG with transparent background."""
    # Normalize to [0, 1]
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam_norm = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam_norm = np.zeros_like(cam)

    # Resize to chip size
    cam_img = Image.fromarray((cam_norm * 255).astype(np.uint8), mode="L")
    cam_img = cam_img.resize((size, size), Image.BILINEAR)
    cam_resized = np.array(cam_img).astype(np.float32) / 255.0

    # Apply colormap (hot regions = solar panels likely)
    colormap = matplotlib.colormaps["inferno"]
    heatmap_rgba = colormap(cam_resized)  # (512, 512, 4) float in [0,1]
    heatmap_rgba = (heatmap_rgba * 255).astype(np.uint8)

    # Scale alpha by activation strength — low activation = more transparent
    heatmap_rgba[:, :, 3] = (cam_resized * 200).astype(np.uint8)

    return Image.fromarray(heatmap_rgba, mode="RGBA")


def main():
    CAMS_DIR.mkdir(parents=True, exist_ok=True)
    CAMS_RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    X, y, chip_ids = load_dataset()
    print(f"  {len(chip_ids)} chips, {y.sum()} positive, {(1-y).sum()} negative")

    print("Training classifier...")
    clf = train_classifier(X, y)

    print("Generating CAMs...")
    for i, chip_id in enumerate(chip_ids):
        patch_features = np.load(FEATURES_DIR / f"{chip_id}.npy")
        cam = generate_cam(clf, patch_features)

        # Save raw CAM for SAM mask_input
        np.save(CAMS_RAW_DIR / f"{chip_id}.npy", cam.astype(np.float32))

        # Save heatmap overlay PNG
        heatmap = cam_to_heatmap_png(cam)
        heatmap.save(CAMS_DIR / f"{chip_id}.png")

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i + 1}/{len(chip_ids)}] {chip_id}")

    print("Done.")


if __name__ == "__main__":
    main()
