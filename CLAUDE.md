# Glooper — Human-in-the-Loop Geospatial Labeling

## Docs

Design docs and plans live in `docs/` (tracked in git — read them for context, update them as part of the work that changes them):
- `docs/backend-architecture.md` — server/pipeline architecture
- `docs/labeling-system-design.md` — labeling system design
- `docs/ma-campaign-*.md` — MA campaign decisions, methodology, roadmap
- `docs/plans/` — per-task implementation plans, named after the issue/branch

## Project Purpose

Interactive system for human-in-the-loop geospatial ML. The user's mental model is "I know what solar panels look like, I want a map of all of them in this area" — not "I need to generate training data for a segmentation model." The ML pipeline is infrastructure that serves this goal, not something the user interacts with directly.

The system is a FastAPI backend + React frontend with a heavy compute pipeline. The server is the primary development target: it orchestrates the chip catalog, asset delivery, multi-user locking, and HPC job dispatch. The frontend communicates exclusively through the server API. A static asset fallback (`public/data/`) exists for the GitHub Pages demo only.

## MVP: The Core Loop

1. **Define study areas** — draw rectangles on the map, assign each to a split (train / test / validate), generating a chip grid snapped to UTM coordinates
2. **Label chips** — SAM produces a mask, paintbrush cleans it up, confirming paints to label canvas
3. **Train** — trigger a training job against the current label canvas; monitor status
4. **Inspect** — view predictions overlaid with ground truth to diagnose quality and guide next labeling
5. **Infer** — draw an area, trigger inference, inspect detections

Test/validation areas: exhaustive labeling (every pixel positive or negative). Training areas: sparse/incremental (ignore pixels excluded from loss). The paintbrush editor is core, not a nice-to-have.

## Plugin Architecture

Everything beyond the core loop is a plugin. Core loop gaps block everything; plugins can be added in any order. Plugins are enabled in `config.yaml` under the `plugins:` key (e.g., `plugins: [cam]`).

**Plugin structure:** Each plugin lives in `plugins/<name>/` with optional subdirectories:
- `frontend/` — React hooks, components, view definitions
- `server/` — FastAPI router module (must export `router`)
- `pipeline/` — Python compute scripts

**Config-driven activation:** The server dynamically mounts plugin routers from `plugins.<name>.server.router`. The frontend fetches enabled plugins from `GET /api/config/plugins` at startup.

**Layer provider interface:** Plugins inject into the labeling view via layer-provider hooks:
```js
{
  onChipSelect(map, chipId, imageCoords) -> Promise<void>,  // add map layers
  onChipDeselect(map) -> void,                               // remove layers
  maskPrior: Float32Array | null,                            // SAM mask prior (256x256)
  controls: Array<{label, active, onToggle}>,                // BasemapPicker buttons
  syncVisibility(map) -> void,                               // react to state changes
}
```

**Plugin views:** Plugins can contribute views via `{ id, label, Component }` — rendered in ViewNav as tabs.

**Adding a new plugin:** Register it in `src/plugins.js` with its hooks/views, create the directory under `plugins/<name>/`, and add to `config.yaml`.

**Attachment points:**
1. **Mask prior injection** — layer provider's `maskPrior` fed as SAM `mask_input`
2. **Prompt generation** — auto-generate SAM point/box prompts from spatial signals
3. **Chip prioritization** — reorder labeling queue (embedding similarity, uncertainty, etc.)
4. **Post-label quality signals** — consistency checks, confidence, bootstrap filtering

## Architecture

```
src/        # React app (browser) — Vite, MapLibre, onnxruntime-web
server/     # FastAPI orchestration (no torch, but rasterio/shapely OK) — DuckDB
pipeline/   # Heavy compute (torch, SAM, DINOv2) — GPU/HPC
plugins/    # Optional plugins (frontend + server + pipeline per plugin)
```

Data access: `src/data.js` routes to FastAPI (`VITE_DATA_SOURCE=api`, default) or static `public/data/` fallback (`VITE_DATA_SOURCE=static`, demo only). The browser always receives chip catalog data as GeoJSON; DuckDB is the authoritative store.

Core pipeline scripts: `extract_dino_features.py` → `generate_sam_embeddings.py`. CAM generation is in the cam plugin: `plugins/cam/pipeline/train_classifier_and_generate_cams.py`. All use chip ID naming: `{easting}e_{northing}n`.

## Technical Decisions

- **DINOv2 ViT-S/14**: Smallest variant, good enough for linear probe on solar panels
- **SAM ViT-B**: Decoder exported to ONNX, runs in-browser in under a second
- **CAMs as SAM mask_input**: DINOv2 CAM fed as mask prior so SAM refines semantic activation into precise boundary (plugin, not core)
- **MapLibre over Leaflet**: WebGL rendering, better for multiple image layer overlays
- **Paintbrush as canvas overlay**: 2D canvas over MapLibre map, brush strokes write to pixel buffer uploaded as label mask

## Labeling System Design

Labels are vector polygons stored in DuckDB (projected CRS). The browser produces a binary raster mask (SAM + paintbrush corrections); the system vectorizes it before persisting. Vector is the highest-information format — vector-to-raster is cheap (rasterize at any resolution for training), raster-to-vector is lossy and expensive. Chip size is a training hyperparameter, not a label property — relabeling is expensive, re-chipping is cheap.

- **Labeling grid:** 448×448 chips (multiple of 14×32 for DINOv2 ViT-S/14 patch size)
- **HPC training:** any chip size, random offsets, overlapping crops rasterized from vector labels
- **Splits:** test/validate exhaustive, training sparse. Spatial blocking prevents leakage.

## Imagery

- Source: NAIP or similar high-resolution RGB as COG. Target: solar panels.
- Current chip size: 512×512 (moving to 448×448)
- Labels: vector polygons in DuckDB, rasterized to pixel masks for training

## Style and Code Conventions

- React: functional components, hooks, useState/useReducer — no class components, no state libraries
- Styling: plain CSS — no component libraries, no Tailwind
- Map: MapLibre GL JS directly, not react-map-gl wrapper
- .npy files: minimal parser returning Float32Array, no numpy-js library

## Deployment

- **Dev:** `npm run dev` + `python -m server.main` (Vite proxies to FastAPI on :8000)
- **Production:** `npm run build && python -m server.main` (single process, single port)
- **Demo:** `npm run build:demo` → GitHub Pages (static, no server)
- Environments: `src/` npm, `server/` pip/venv, `pipeline/` pixi
