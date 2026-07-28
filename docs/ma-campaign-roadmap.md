# MA Campaign — Feature Roadmap

Ticket-ready feature breakdown for the ~3-week solar-panel mapping campaign. Each item is scoped to be one GitHub issue: title, what/why, files touched, acceptance criteria, rough size. Ordering within a milestone is the suggested implementation order; milestones are dependency-ordered.

Decisions backing this roadmap: `.claude/ma-campaign-decisions.md`. Metric definitions, sample sizes, and QA protocol: `.claude/ma-campaign-methodology.md` (verified research report).

---

## Milestone 0 — GPU machine bring-up (Day 1, no code)

### 0.1 Deploy full stack on GPU machine
Run server + chip worker + train worker + built frontend on the GPU box; browser connects over LAN/SSH tunnel. No split-machine deployment (shared-filesystem assumption stays).
- **Steps:** clone, `pixi install -e cuda` in `pipeline/`, `npm run build`, `python -m server.main`, migrate `data/glooper.duckdb` + `data/` if existing labels carry over.
- **Accept:** CUDA training run and SAM embedding generation complete end-to-end on the box; labeling UI usable from laptop browser.
- **Size:** S (config/docs only)

### 0.2 Verify MassGIS ortho year availability (2019/2021/2023/2025)
Cheap check, big blast radius. Confirm the `coq{year}_15cm_jp2` URL pattern (or equivalent) exists for all four years and that band count/dtype/resolution are consistent across years.
- **Files:** `pipeline/src/providers/mass_orthos.py` (inspection only)
- **Accept:** documented tile-index URL + band layout per year; any inconsistency (3-band year, different dtype) flagged with a mitigation plan. **2025 existence is load-bearing — it is the entire temporal holdout.**
- **Size:** S

---

## Milestone 1 — Year-scoped chip identity (in progress on `feature/year-scoped-chips`)

Same-location multi-year is a requirement. Chip identity becomes `{easting}e_{northing}n_{year}` (or composite key). This milestone is already underway in a separate worktree; listed here for completeness of the dependency chain.

### 1.1 Server core: year-scoped chip IDs and schema
- **Files:** `server/db.py`, `server/grid.py`, `server/routers/study_areas.py`, `server/routers/chips.py`
- **Accept:** chips table keyed by year-scoped ID; grid generation stamps year; GeoJSON properties include year.

### 1.2 Multi-year mass_orthos provider + worker protocol
Provider holds per-year catalog/cache; `year` threaded through `server/worker_client.py` → `pipeline/src/chip_worker.py` → `get_chip_image`.
- **Files:** `pipeline/src/providers/mass_orthos.py`, `pipeline/src/providers/__init__.py`, `pipeline/src/chip_worker.py`, `server/worker_client.py`

### 1.3 Year-scoped labels + join fixes + copy-forward
Labels attach to (geometry, year), not geometry alone. Cross-year reuse only via explicit copy-forward-and-review — never implicit. Protects the 2025 holdout.
- **Files:** `server/db.py` (`get_labels_for_chips`), labels router, frontend labeling view

### 1.4 Training/prediction path with year-scoped IDs
File naming for TIFs/PNGs/embeddings/predictions follows the new ID; train worker and prediction overlays unaffected functionally.
- **Files:** `pipeline/src/train_worker.py`, `pipeline/src/dataset.py`, `server/routers/predictions.py`, `src/views/useLabelingView.js` (`parseChipId`)

### 1.5 Frontend year support + config + data reset
Year selector in DefineArea; documented one-time data reset for existing location-only data.

---

## Milestone 2 — Strata/region metadata (Days 2–3, parallel with M1 tail)

### 2.1 `study_areas` table + region/stratum columns on chips
Persist areas (currently ephemeral) with `id, name, split, region, stratum, year, geometry, created_at`; stamp chips with `region`, `stratum`, `area_id` (year comes from M1).
- **Files:** `server/db.py` (use existing `ALTER TABLE ... IF NOT EXISTS` migration pattern), `server/routers/study_areas.py`, `server/grid.py`, `server/routers/chips.py`
- **Accept:** `CreateStudyAreaRequest` takes `region: central|east|west`, `stratum: rural|suburban|urban`; chip GeoJSON carries all metadata; deleting an area cascades.
- **Size:** M

### 2.2 DefineArea UI: region/stratum/year selectors
Dropdowns next to the split buttons; payload through `data.createStudyArea`.
- **Files:** `src/DefineArea.jsx`, `src/views/useDefineAreaView.js`, `src/data.js`
- **Size:** S

### 2.3 Box-distance leakage check on area creation
Spec paper p. 45, never built. Warn (not block) when a new area is within X m of an existing area in a different split, same year-group. Research found no literature constant for 15 cm imagery — use a configurable buffer (default ≥1 km, documented as a judgment call); note this affects train-vs-test placement only, test chips need no spacing from each other (Stehman 2000).
- **Files:** `server/routers/study_areas.py` (ST_Distance query), `src/DefineArea.jsx` (warning display)
- **Size:** S

---

## Milestone 3 — Transferability evaluation (Days 6–10) — THE #1 DELIVERABLE

### 3.1 Test chips enter the predict phase
`get_training_chips()` currently returns train/validate only; test chips are never predicted. Extend `GET /api/training/chips` to include complete test chips (predict-only, never in the loss).
- **Files:** `server/db.py`, `server/routers/training.py`, `pipeline/src/train_worker.py` (`_predict_phase`)
- **Size:** S

### 3.2 Per-chip pixel confusion counts in the train worker
For each complete chip at predict time, rasterize labels (reuse `ChipDataset` rasterization — avoids train/eval mismatch) and compute tp/fp/fn/tn at threshold 0.5 (+ 2–3 extra thresholds from 8-bit alpha). POST alongside the mask.
- **Files:** `pipeline/src/train_worker.py`, `pipeline/src/training_dataset.py`, `pipeline/src/server_client.py`
- **Size:** M

### 3.3 `chip_metrics` table + metrics ingestion
- **Files:** `server/db.py` (new table `run_id, chip_id, threshold, tp, fp, fn, tn`), `server/routers/training.py` (`PredictionsRequest`)
- **Size:** S

### 3.4 Metrics endpoint: group-by split × region × stratum × year
`GET /api/training/runs/{run_id}/metrics` → precision, recall, F1, IoU, chip counts, positive-pixel fraction per cell + overall rows; `?format=csv`. Per Olofsson 2014: report as area-proportion error matrix, SE + 95% CI (±1.96 SE) on every estimate, stratum-area-weighted aggregates (combined ratio estimator for precision/recall/IoU — not weighted averages), **no kappa**. Label columns user's/producer's accuracy alongside precision/recall for the remote-sensing audience.
- **Files:** `server/routers/training.py`, `server/db.py`
- **Accept:** CSV downloads with one row per (split, region, stratum, year) cell; west-MA and 2025 rows are the headline transferability comparison.
- **Size:** M

### 3.5 Object-level (instance) metrics
Vectorize predictions (reuse `server/vectorize.py`); follow the Duke solar protocol (Hu et al. 2022): group predicted polygons into arrays via **3 m dilation**, match to label arrays at **IoU ≥ 0.5** (safely below the human boundary-agreement median of 0.86) → per-object TP/FP/FN → object precision/recall/F1 per cell, plus **false positives per unit area** (precision is unstable for rare objects), panel counts, area totals. Polygon-to-polygon in DuckDB/shapely — no raster connected-components needed. CIs: bootstrap over chips within cell (design-based object-level variance is an open literature question).
- **Files:** `server/vectorize.py`, new `server/metrics.py`, `server/routers/training.py`
- **Size:** M–L

### 3.6 Best-val-checkpoint saving
Save best-val-loss weights instead of last epoch. Cheap, materially improves reported numbers.
- **Files:** `pipeline/src/train_worker.py`
- **Size:** S

### 3.7 Minimal metrics table UI
Plain table of group-by cells in/next to `src/TrainingPanel.jsx` or a plugin-view tab. CSV is the real deliverable — no chart polish.
- **Size:** S

---

## Milestone 4 — Labeling throughput + bias hygiene (Days 10–12, interleaves with M3)

### 4.1 One-keystroke save+complete+next
Collapse the Enter/Enter/Cmd+Enter dance into one binding (preview → save → complete → advance).
- **Files:** `src/views/useLabelingView.js`
- **Size:** S — biggest per-chip win for exhaustive labeling

### 4.2 Prefetch next chip's image + SAM embedding
Warm the next incomplete chip during current-chip labeling.
- **Files:** `src/views/useLabelingView.js`, `src/data.js`
- **Size:** S

### 4.3 Per-area progress counters
Complete/total per split and per area (data already in `featureById` + M2 metadata).
- **Files:** `src/InfoPanel.jsx` or `src/DefineArea.jsx`
- **Size:** S

### 4.4 Enforce prediction-hiding on test/validate chips
Spec paper p. 24: prediction/diff overlays must be unavailable while a test or validate chip is selected — labeler must not see model output on evaluation areas.
- **Files:** `src/views/useLabelingView.js`, `src/BasemapPicker.jsx`
- **Size:** S

### 4.5 Single-annotator QA: blind relabel subsample
Support relabeling a random subsample of completed test/val chips without showing the original label; compute intra-annotator agreement. Protocol per Clark & Pacifici 2023 precedent: **≥1 week between passes**; report object-level F1 and boundary IoU of the relabel vs original (reporting formal agreement statistics exceeds published single-annotator practice). QA targets **omission** — ~30% of solar polygons were single-annotator finds in the Duke data. Pre-register a trigger (e.g., object F1 ≥ 0.9 in the subsample, else audit that cell).
- **Files:** `server/db.py` (label versions or shadow table), small labeling-view mode
- **Size:** M

### 4.6 (Stretch) Seed mask from latest prediction on training chips
Pre-seed paintbrush from prediction PNG (alpha > 128) via the existing `maskPrior` slot — **training areas only** (4.4 guards eval areas).
- **Size:** M — only if labeling is the bottleneck in week 3

---

## Milestone 5 — Bounded wide-area inference (Days 12–15)

### 5.1 `inference` split for study areas
Excluded from training/eval queries by construction.
- **Files:** `server/routers/study_areas.py`, `server/db.py`
- **Size:** S

### 5.2 Predict-only worker action
`{"action": "predict", run_id, model_path, chip_ids}` loads a saved `.pt`, reuses `_predict_phase`. New `POST /api/training/runs/{run_id}/predict` with bbox chip selection. Note: shares the single-worker executor — long inference blocks retraining.
- **Files:** `pipeline/src/train_worker.py`, `server/training_client.py`, `server/routers/training.py`
- **Size:** M

### 5.3 Ranked detection review list
Chips ranked by `predictions.score`; click flies the map to the chip. Reuses existing overlay.
- **Files:** `src/TrainingPanel.jsx` or plugin view, `server/routers/predictions.py`
- **Size:** S

### 5.4 Export: predictions → GeoJSON/GPKG
Frame 4 of the spec: union + simplify predicted polygons over an inference area, export with per-object confidence. Pairs with the vectorization post-processing note (simplify post-union, not per-chip).
- **Files:** `server/vectorize.py`, new export endpoint
- **Size:** M

---

## Explicitly out of scope (do not ticket)

- Split-machine deployment (server local / workers remote)
- Whole-state inference (~6M chips) or tiled/streaming inference engine
- Active learning / embedding-based chip prioritization (dino/cam plugins stay off)
- Full PR curves beyond 8-bit thresholds; report UI polish beyond table + CSV
- Multi-run experiment management

## Known risks

1. **2025 imagery availability** (0.2) — the temporal holdout depends on it.
2. **Labeling budget dominates the calendar**: Olofsson-standard is 50–100 chips per cell that matters; with 27+ cells that exceeds a 3-week solo budget. Prioritize headline cells (west-MA × strata, 2025 × strata) at 50–75 chips; pool in-domain cells with wider (honest) CIs. **Evaluation chips must be randomly selected within study areas** (probability sampling) for CIs to be defensible — add a small ticket for system-selected random eval chips.
3. Single training-run serialization: inference jobs (5.2) block retraining on the one GPU worker.
4. DuckDB row-at-a-time inserts under a global lock: tolerable at 10k chips/area; batch with `executemany` if painful.
