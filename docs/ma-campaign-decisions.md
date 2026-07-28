# MA Solar-Panel Mapping Campaign — Design Decisions

Real solar-panel mapping project over Massachusetts, ~3 weeks starting late July 2026. #1 deliverable: comprehensive performance statistics demonstrating transferability. GPU machine available; whole stack (server + workers + built frontend) runs on it.

## Locked decisions (2026-07-26)

- **Single-person project** — no consensus masks; label QA must use single-annotator methods (e.g., blind test-retest relabeling).
- **Object-level metrics required** alongside pixel IoU — per-object hits/misses via IoU-threshold instance matching.
- **Same-location multi-year WILL happen** — not an MVP cut. Chip identity must include year (breaks location-only `{easting}e_{northing}n` ID convention: file naming, label joins, caches all affected).
- **4 imagery years: 2019, 2021, 2023, 2025.** All of 2025 is a temporal holdout test set. Train/val on 2019/2021/2023.
- **Spatial design:** regions central/east/west MA × strata rural/suburban/urban; west MA is the spatial holdout.
- **Labels must be year-scoped** (labeled against specific imagery); cross-year label reuse only as an explicit copy-forward-and-review action, never implicit — protects the 2025 holdout's integrity.
- **Whole-state inference out of scope**; inference areas capped ~10k chips (~45 km² per area).

## Key codebase gaps (as of July 2026)

- No metrics code at all; test chips never used in the predict phase (`get_training_chips` returns train/validate only).
- No `study_areas` table — no region/stratum/year metadata on areas or chips.
- `mass_orthos` provider is single-year; chip-request protocol carries no year.
- Prediction PNGs store probability as 8-bit alpha — adequate for thresholded metrics, limits fine threshold sweeps.

## Protocol items from the Glooper spec paper (references/)

- Box-distance leakage check between study areas in different splits (spec p. 45, never built).
- Hide predictions/model performance entirely while labeling test/validate areas (p. 24).
- Evaluate on the test split once, at the end (p. 46).
- Failure-mode diagnostics worth having during the campaign: train/val gap (too few labels), prediction/label disagreement ranking (noisy labels), underrepresented subtypes (p. 42–43).
