# Labeling System Design Plan

## Problem Area Definition

**Users:** Domain experts who need to map specific features from geospatial imagery and have the knowledge to identify those features visually, but not necessarily AI expertise.

**Problem:** They are trying to turn their visual knowledge of what something looks like in satellite imagery into an accurate, complete map of where that thing exists — without needing to understand or manage the pipeline that makes it possible.

**Key framing principles:**

- The user's mental model is "I know what solar panels look like, I want a map of all of them in this area" — not "I need to generate training data for a segmentation model"
- "Feedback" means the map is visibly getting better as they work, not just that a validation metric improved
- The question they want answered is "am I done yet?" — not "did my loss converge?"
- The machine learning pipeline (embeddings, CAMs, SAM, training loops) is infrastructure that serves this goal, not something the user interacts with directly

## Core Principle: Separate Labels from Chip Size

Labels are raster canvases with three pixel values: positive, negative, and ignore. They are georeferenced (tied to geographic coordinates, not pixel coordinates of any particular chip size). Chip size is a training hyperparameter, not a property of the label. Relabeling is expensive; re-chipping is cheap.

## Data Storage

### Regional Imagery
- Source imagery stored as COG or ZARR per region
- This is the authoritative source; the labeling UI never touches it directly

### Labeling Grid (Pre-computed, for speed)
- Non-overlapping 448x448 chips (multiple of 14x32 for DINOv2 ViT-S/14 patch size) tiling each region completely
- Chips stored as pre-computed PNGs/TIFs keyed by chip ID
- DINOv2 embeddings and SAM embeddings pre-computed per chip
- GeoParquet catalog per region: one row per chip with geometry, label status, asset paths

### Label Canvas (Regional, for reusability)
- Regional raster (e.g. GeoTIFF) with three pixel values: positive, negative, ignore
- Initialized to all ignore
- Painted with chip-level labels as labeling progresses
- Exported as the shipped dataset — consumers re-chip however they want

### Conversion
- Chip-level label rasters paint onto the regional canvas using each chip's affine transform
- Run at export time, not in the loop
- Labeling grid must be non-overlapping and gap-free so painting back has no conflicts

## Dataset Splits

### Validation and Test Sets
- Select a few geographic regions for each
- Label exhaustively — every pixel is positive or negative, no ignore class
- Spatial blocking ensures no leakage between splits
- Verify representativeness post-hoc using DINOv2 embedding distributions (PCA/UMAP)
- If a split is unrepresentative, swap whole regions between splits

### Training Set
- Sparse/incremental labeling is fine — ignore pixels are excluded from loss
- Only positive and negative pixels contribute to training

## Labeling Workflow

### Bootstrap Phase
1. Select some positive examples manually in the training region
2. Use DINOv2 embedding similarity to propagate presence/absence labels quickly across the labeling grid
3. Train a linear probe on the embeddings to generate CAMs
4. Auto-label as negative: chips with absence label AND low CAM activation (use a conservative threshold — better to leave as ignore than mislabel)
5. Paint auto-negatives to the canvas

### Active Labeling Phase
1. Prioritize chips by representativeness in embedding space — target underrepresented clusters
2. User reviews chip, clicks to run SAM, produces positive mask
3. Marking a chip as reviewed sets all non-positive pixels in that chip to negative
4. Mask is painted to the canvas immediately

### In-Browser Feedback
- Lightweight decoder head (linear or small MLP) on frozen DINOv2 features
- Fine-tunes continuously as labels come in
- Provides real-time estimate of model performance
- Caveat: this will plateau below what a full model achieves — don't stop labeling based on this alone
- User can trigger inference against a representative subset of the validation set

### HPC Training Loop
- FastAPI layer triggers an event to send training/validation data to HPC
- HPC job reads from the canvas, chips at whatever size it wants, trains a heavier model
- HPC model predictions feed back into the labeling UI as priors (like CAMs) to guide where to label next
- Chips where the heavy model disagrees with the lightweight browser model are high-value review candidates

## Chip Sizes

- **Labeling UI:** fixed 448x448 grid (pre-computed for speed)
- **In-browser training/validation:** same 448x448 chips (shares pre-computed assets)
- **HPC training:** any chip size, random offsets, overlapping crops — reads from the canvas
- **HPC benchmarking/validation:** any chip size

flowchart TD
    Start([🚀 Project Start]) --> Define["Define training, test\n& validation areas"]
    
    Define --> TestVal["Label test & validation\nexhaustively with SAM"]
    Define --> Boot["Bootstrap training labels\nvia embeddings & CAMs"]

    Boot --> Active{"Smart labeling loop"}
    
    Active --> Priority["Embeddings & model performance\nreveal what to label next"]
    Priority --> Human["👤 Humans refine labels\nguided by confidence scores"]
    Human --> Retrain["Retrain & measure\nagainst validation set"]
    
    Retrain -- "Not yet" --> Active
    Retrain -- "Goal met ✅" --> Done([📍 Accurate map delivered])
