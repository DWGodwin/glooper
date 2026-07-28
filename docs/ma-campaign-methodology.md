# MA Campaign — Accuracy Assessment Methodology (Research Report)

Synthesized from a deep-research pass (22 sources fetched, 98 claims extracted, 25 adversarially verified, 21 confirmed / 4 refuted). Every finding below survived a 3-vote refutation panel against the primary source. Companion docs: `.claude/ma-campaign-decisions.md`, `.claude/ma-campaign-roadmap.md`.

## The protocol in one paragraph

Select assessment chips by **stratified random sampling within each region×stratum cell** (a probability design is what makes per-cell estimates defensible), targeting **50–100 exhaustively labeled chips per cell that matters** (Olofsson et al. 2014's simplified allocation; their own worked example settled on 75). Report an **area-proportion error matrix** (not raw counts), **user's/producer's accuracy (= precision/recall) with SE and 95% CI on every estimate**, stratum-weighted aggregates (never naive pooling), reference-based area estimates, and **no kappa**. For objects, follow the **Duke solar protocol**: group predicted pixels into arrays via **3 m dilation**, match at **IoU ≥ 0.5**, report object precision/recall/AP plus **false positives per unit area**. QA the single-annotator labels with **time-separated repeat passes (≥1 week)** plus cross-referencing independent imagery/records; a blind-relabel subset with reported intra-annotator agreement *exceeds* published practice.

## Confirmed findings

### 1. Sampling design (Olofsson et al. 2014; Stehman 2000) — verified 3-0
- Stratified random sampling within cells is the recommended good-practice design. Chips must be **randomly selected within cells**, not hand-picked: "It is difficult to envision a circumstance in which a deviation from this condition of probability sampling would be acceptable for a scientifically rigorous assessment of accuracy."
- **Practical consequence for Glooper:** study areas can be drawn purposively, but evaluation chips inside them must be selected by the system at random (a feature worth a small ticket), and the estimates then generalize to the sampled frame — document that scope limit.

### 2. Sample size (Olofsson 2014 Eq. 13 = Cochran; Wagner & Stehman 2015; Foody 2009) — verified 3-0
- **50–100 chips per rare/priority stratum** as the starting allocation; Cochran's stratified formula targets a chosen SE of overall accuracy (Olofsson's example: SE = 0.01 → n = 641 total, 75 per rare stratum).
- Justify the total a priori with effect size / significance / power (Foody 2009, which also gives paired-sample equations for comparing model versions on the same chips — McNemar-style).
- Caveats: 50–100 is a simplified heuristic framed for land-change strata; Stehman & Wagner 2024 (RSE 300:113881) is the directly applicable rare-class successor. With 9 region×stratum cells × multiple years, full per-cell precision at 50–100 may exceed a 3-week single-person budget — prioritize cells (see budget note below).

### 3. Reporting standards (Olofsson 2014; AREA2) — verified 3-0
- Error matrix in **estimated area proportions** (p̂ij = Wi·nij/ni), not sample counts.
- **No kappa** ("strongly discouraged"; hardened by Pontius & Millones 2011, Foody 2020).
- Area estimates from the **reference classification of the sample** (stratified estimator), not map pixel counting.
- **Every** accuracy and area estimate carries SE and approximate 95% CI (± 1.96 SE).
- Aggregate across strata with **area weights**; for ratio metrics (precision/recall/IoU) the strictly correct aggregate is a **combined ratio estimator** (Stehman 2014), not a weighted average of per-stratum ratios.
- Vocabulary bridge: user's accuracy = precision = 1 − commission; producer's accuracy = recall = 1 − omission. IoU has no classical analogue — report it as a separate metric. (Pipeline pitfall: confusion-matrix orientation differs between remote-sensing convention and scikit-learn — build an orientation check.)

### 4. Object-level metrics (Hu/Bradbury/Malof, Duke group — arXiv:1902.10895 / Applied Energy 2022) — verified 3-0
- **Arrays, not individual panels**, as the assessment object: group connected predicted pixels via **3-meter dilation**.
- TP when predicted array reaches **IoU ≥ 0.5** with a true array; report precision, recall, PR curve, AP.
- Also report **false positives per unit area** — precision alone is unstable for rare objects.

### 5. Human-agreement priors (Bradbury et al. 2016; Hu et al.) — verified 3-0
- Inter-annotator on ~0.3 m imagery: **~30% of polygons found by only one of two annotators** (object omission is the dominant error mode); for co-identified polygons, boundary agreement is high — 99.4% of pairs > IoU 0.5, median 0.86.
- A trained annotator audited against installation records missed only **~3.8%** of panels.
- Implications: IoU ≥ 0.5 matching sits safely below human boundary disagreement (won't split genuine matches); **QA must target omission, not boundary quality**. Treat numbers as order-of-magnitude priors (different imagery/setup), which argues for running our own blind-relabel subset.

### 6. Single-annotator QA (Clark & Pacifici 2023, Scientific Data; Hu et al.) — verified 3-0
- Published precedent: single annotator examined **every tile at least twice, ≥1 week between passes**; verified each object against independent imagery (Google Earth), assigning per-label confidence categories (97.8% high-confidence); audited misses against installation records.
- Precedent reports confidence categories only — **reporting formal intra-annotator object-level F1 and boundary IoU from a blind-relabel subset would exceed published practice** (good headline for the report). No published threshold exists for "agreement too low, relabel" — set one ourselves and pre-register it.

### 7. Spatial transferability justification (Hu et al., Table 1) — verified 3-0
- Same model, identical training, identical nominal resolution: **pixel IoU 0.73 / array AP 0.82 in Connecticut vs 0.60 / 0.71 in San Diego**. Geographic/sensor shift alone produces large drops — this is the citable justification for the west-MA holdout. Extension to held-out-year is inference (flagged), but sound: ortho campaigns differ in sensor/radiometry the way regions do.

### 8. Spatial autocorrelation and test chips (Stehman 2000; Stehman & Foody 2019) — verified 3-0
- Autocorrelation among assessment units **does not bias** design-based accuracy estimators or their variance estimators. Test chips do NOT need to be spaced apart within assessment areas.
- Spatial blocking/separation is a **train-test leakage** concern only. Keep separation between training areas and test/holdout areas; don't waste budget spacing test chips.
- Note: a stronger formulation ("spatial separation not required for valid accuracy assessment", Wadoux et al. 2021 framing) was **refuted 0-3** in verification — state this only in the bias-of-estimators form above.

### 9. Precedent bar (DeepSolar as contrast case) — verified 3-0
- DeepSolar reported per-stratum precision/recall (93.1/88.5 residential, 93.7/90.5 non-residential) but at image level only (no IoU, no object matching), and follow-up work (Kasmi et al.) found train/test region overlap. The MA design (spatial + temporal holdout, object-level F1@IoU, per-cell CIs) **exceeds the published bar on exactly the criticized axes** — say so in the report.

## Open questions (no confirmed literature answer — decide ourselves)

1. **Train-test buffer distance**: no surviving quantitative claim for 15 cm imagery. Defensible route: derive from an autocorrelation-range analysis of the imagery/features, or pick a conservative buffer (e.g., ≥1 km) and document it as a judgment call, not a literature constant.
2. **2025 landscape change vs model error**: no published protocol for separating genuinely-new panels from false positives in a held-out year. Proposed (our own design): date each 2025 FP/FN against 2023 imagery and report **change-adjusted and raw metrics side by side**.
3. **Design-based variance for object-level F1@IoU**: Olofsson's estimators are pixel-unit; Radoux & Bogaert 2017 is the cited adaptation path (unverified). Pragmatic fallback: bootstrap CIs over chips within each cell, clearly labeled as such.
4. **Intra-annotator agreement threshold**: no published trigger. Pre-register one (e.g., object-level F1 ≥ 0.9 within the relabel subset, else audit that cell).

## Budget note (27+ cells × 50–100 chips is too many)

Full per-cell precision everywhere exceeds a 3-week single-person budget. Prioritize:
- **Headline cells** (west-MA × 3 strata; 2025 × 3 strata): full 50–75 chips each.
- **In-domain val cells** (central/east × strata × 2019–2023): can share a pooled estimate with per-cell breakdown reported at lower precision (wider CIs are honest, not wrong — every estimate carries its CI anyway).
- Use Foody 2009 power framing to state up front what difference the design can detect.

## Refuted claims (do not reuse)

- "Halving CI width requires 4× chips / SE = s/√n" as sourced to AREA2 (0-3) — the arithmetic is standard but the sourced formulation failed; derive properly if needed.
- Wadoux et al. 2021 "spatial separation not required for valid accuracy assessment" as a blanket claim (0-3) — only the conditioned, probability-sampling form survives (finding 8).

## Key sources

- Olofsson et al. 2014, "Good practices for estimating area and assessing accuracy of land change", RSE 148:42–57
- Stehman 2000, RSE 72:35–45; Stehman & Foody 2019; Stehman 2014 (combined ratio estimator); Stehman & Wagner 2024, RSE 300:113881
- Wagner & Stehman 2015, RSE (optimal single-class allocation)
- Foody 2009, IJRS 30(20):5273–5291 (sample size, power, paired comparisons)
- Hu, Bradbury, Malof et al., arXiv:1902.10895 / Applied Energy 327 (2022) — Duke solar protocol, CT/San-Diego transfer drop
- Bradbury et al. 2016, Scientific Data (Duke solar dataset, inter-annotator agreement)
- Clark & Pacifici 2023, Scientific Data (single-annotator QA precedent)
- DeepSolar: Yu et al. 2018, Joule; critique in Kasmi et al., arXiv:2207.07466
- AREA2 (Olofsson group, BU): area2.readthedocs.io
