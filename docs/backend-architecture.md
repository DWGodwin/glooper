# Backend Architecture

## Repo Structure

Three independent concerns, each with its own dependencies:

```
src/        # React app (browser)
server/     # FastAPI orchestration (lightweight, always-on)
pipeline/   # Heavy compute (GPU, batch, HPC)
```

### `src/` — React App
Browser-based UI. Displays map, chip imagery, overlays. Runs SAM decoder (ONNX) and lightweight fine-tuning in-browser. All data access goes through the API. In production tiers, the built frontend (`dist/`) is served by FastAPI — labelers only need a browser.

### `server/` — FastAPI Orchestration
Lightweight Python service. No torch, no GDAL, no heavy geo libraries. Coordinates everything: serves assets, manages chip state, handles multi-user locking, submits HPC jobs, stores browser model weights. Backed by DuckDB (local) or Postgres+PostGIS (multi-user/cloud).

### `pipeline/` — Heavy Compute
Torch, SAM, DINOv2, rasterio, GDAL. Runs on GPU machines / HPC. Chipping, embedding extraction, model training, inference. Reads labels exported by the server, writes outputs to a location the server can pick up.

## Deployment Tiers

Three modes from the same repo, increasing in complexity:

1. **Static demo (no server):** `npm run dev`. Frontend reads from `public/data/` directly. No Python, no API. For demos and presentations only.
2. **Local single-user:** `npm run build && python -m server.main`. FastAPI serves the built frontend and the API from a single process on one port. DuckDB file on local disk. Data directory on local disk.
3. **Networked multi-user:** Same as tier 2, but run on a shared machine. Labelers just open `http://<host>:8000` in a browser — no dev tooling, no npm, no config on their machines. The project manager runs the server; labelers just need a browser.

In tiers 2 and 3, FastAPI serves both the API and the built frontend as static files:

```python
# server/main.py
app.mount("/", StaticFiles(directory="dist", html=True))
```

Project manager workflow:
1. `npm run build` (produces `dist/`)
2. `python -m server.main` (one process, one port, serves everything)

## Catalog Store

DuckDB (local) or Postgres+PostGIS (networked). One row per chip:

| Column | Type | Description |
|---|---|---|
| id | text | Chip ID (`{easting}e_{northing}n`) |
| region | text | Region identifier |
| geometry | geometry | Chip polygon in projected CRS |
| status | text | `unlabeled`, `ready`, `locked`, `labeled`, `flagged` |
| locked_by | text | User ID holding the lock, null if unlocked |
| locked_at | timestamp | When the lock was acquired |
| label_path | text | Path to label mask blob |
| image_path | text | Path to chip image |
| cam_path | text | Path to CAM overlay |
| embedding_path | text | Path to SAM embedding |
| prediction_score | float | Latest browser model confidence |

GeoParquet is an export format, not the live store. Used for:
- Snapshot exports to HPC for training jobs
- Archival and sharing
- Initial load when bootstrapping a region

## Region Initialization

1. User draws a polygon on the map
2. Browser sends the polygon to FastAPI
3. FastAPI generates the 448x448 chip grid tiling the polygon (non-overlapping, gap-free)
4. Inserts all rows into DuckDB with status=`unlabeled`
5. Returns the grid as GeoJSON for immediate display
6. Kicks off pipeline jobs asynchronously: chip source imagery, run DINOv2, run SAM encoder
7. As each chip's assets become available, updates its row (sets asset paths, status=`ready`)

## API Surface

### Chip Catalog
```
GET  /api/chips?region={id}&status={status}   → GeoJSON FeatureCollection (chip polygons + status)
```

### Chip Assets
```
GET  /api/chips/{id}/image       → chip PNG
GET  /api/chips/{id}/cam         → CAM overlay PNG
GET  /api/chips/{id}/embedding   → SAM embedding (.npy)
GET  /api/chips/{id}/label       → label mask
```

### Labeling Workflow
```
POST   /api/chips/{id}/lock      → claim chip for labeling (409 if already locked)
POST   /api/chips/{id}/label     → save label mask, update status, release lock
DELETE /api/chips/{id}/lock      → release without saving (navigated away, timed out)
```

Stale lock timeout: 15 minutes. Checked on claim, no separate reaper process needed.

### Browser Model
```
GET  /api/models/{region}/browser-decoder   → latest browser-trained weights
POST /api/models/{region}/browser-decoder   → upload updated weights from browser
POST /api/chips/{id}/prediction             → save browser model prediction for a chip
```

### HPC Integration
```
POST /api/jobs/train         → export labels + submit training job to HPC
GET  /api/jobs/{id}/status   → check job status
```

## Multi-User Sync

Users don't edit the same chip. Coordination is chip-level locking via the API.

Browser stays in sync via polling: refetch chip status GeoJSON every ~10 seconds. Upgrade to SSE/WebSocket only if polling latency becomes a problem.

All browser-side computation (SAM decoder, lightweight fine-tuning) runs on each labeler's machine. The server is not involved in inference — it only stores and serves data.

## Data Flow

### Labeling
```
Browser → POST /api/chips/{id}/label → FastAPI writes mask blob, updates DuckDB row
```

### Browser Fine-Tuning
```
Browser fine-tunes on labels collected so far
  → POST /api/models/{region}/browser-decoder (weights, kilobytes)
  → POST /api/chips/{id}/prediction (per-chip scores)
Next user session: GET weights → resume from latest checkpoint
```

### HPC Training
```
User triggers training
  → FastAPI exports labels (GeoParquet + mask blobs) to shared storage (S3, shared FS)
  → Submits job to HPC (Slurm, SageMaker, SSH+nohup)
  → HPC reads export, trains, writes predictions back to shared storage
  → FastAPI picks up predictions, makes them available in browser
```

FastAPI does not run pipeline code. It submits jobs and shuttles data.

## Environments

| Component | Environment |
|---|---|
| `src/` | npm (build only — labelers don't need it) |
| `server/` | pip / venv |
| `pipeline/` | pixi |

No containers needed initially. Add a Dockerfile for the server when deploying to cloud. Let HPC handle its own container story (Singularity/Apptainer).

## Configuration

Infrastructure config lives in a `config.yaml` (or environment variables), not in the app UI. Covers:
- Data directory path (local disk or S3 bucket)
- DuckDB path or Postgres connection string
- HPC connection details (Slurm endpoint, shared storage path)

These are set once at setup time by the project manager. Labelers never touch config — they just open a URL.

Project-level setup (draw a region, select source imagery, name it) happens in the app UI.

## Asset Storage

Assets (chip images, CAMs, embeddings, label masks, model weights) live on disk or S3. FastAPI resolves chip ID → file path via the DuckDB row. The browser never constructs file paths — it uses chip IDs and the API resolves the rest.

`public/data/` exists only for the static demo tier. In all API-backed modes, all data flows through FastAPI endpoints.

## Label Canvas Export

The label canvas (regional raster, GeoTIFF) is an export artifact, not the live store. To produce it:
1. Query DuckDB for all labeled chips in a region
2. Read each chip's label mask
3. Paint onto a regional raster using each chip's affine transform
4. Write as COG

This runs at export time (before HPC jobs) or on demand. The live working format is individual mask blobs per chip, coordinated by DuckDB.
