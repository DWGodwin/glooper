# Glooper: The Geospatial Human-in-the-Loop Pipeline for Remote Sensing ML

![glooper_demo](https://github.com/user-attachments/assets/b2854e49-2db5-4a1d-b38c-9b7b0ef764a4)

High quality label data is key for making maps with machine learning. Creating high quality label data is labor-intensive, monotonous, and tricky to do well. Feedback is on the order of weeks, and projects die in the back-and-forth on data quality. What would it take to **look forward to** a labeling campaign? What would it take to be able to come back to a mapping project and make it even better?

What if you could start with the cheapest possible label — "this tile has solar panels, this one doesn't" — and bootstrap your way to pixel-level segmentation masks in a day instead of a week or a month? Instead of wading through mind-numbingly easy examples that you suspect add nothing to model performance, you could be annotating most challenging, highest-value samples first and seeing their impact in real time. Domain expertise should be spent on the hard conceptual problems that people want to work on, not on precision clicking.

## This Demo

That's the vision for where this is heading. Right now, this interactive demo for a proof-of-concept human-in-the-loop geospatial ML workflow shows you one piece of the pipeline puzzle.

The core interaction shown in this demo is this: DINOv2 generates class activation maps (CAMs) from cheap presence/absence labels, those CAMs feed as mask priors into SAM's decoder running in-browser via ONNX, and the user clicks to refine segmentation masks in real time on remote sensing imagery.

One caveat: This is ~100 chips of presence/absence data I made in 10 minutes, so the CAMs do not isolate specific classes. A real project would require more presence/absence chips for useful CAMs - luckily, similarity search within embeddings could be used to grow this type of data **fast**.

The supervision ladder:

1. **Presence/absence labels** → the cheapest form of annotation (1 second per chip)
2. **DINOv2 activation maps** → a foundation model turns tile-level labels into spatial heatmaps showing *where* objects likely are
3. **SAM boundary refinement** → a single click turns a hazy heatmap into a precise segmentation mask, in real time, in the browser
4. **Human correction** → the expert spends their time on the hard cases, not the obvious ones

Each step amplifies the one before it - Cheap labels produce useful heatmaps, Heatmaps guide precise segmentation, and human labelers refine the hardest examples and get the easy ones done quickly.

## Where This Is Going

This demo is the seed of a larger vision for human-in-the-loop geospatial ML:

- **Rapid Bootstrapping.** Label a few presence/absence chips. Embedding similarity search helps you grow and refine your seed dataset. CAMs can be used as priors to point you to what you're looking for in the interface.
- **The labeling flywheel.** Your refined masks train a domain model. The model produces better predictions. Better predictions mean less correction next time. The system gets faster as you use it.
- **Goal-directed labeling.** Set an accuracy target. Label until you reach it. The system estimates how many more labels you need and directs your attention to the samples that will help most.
- **Keyboard-driven workflows.** Hotkeys to navigate between samples, annotate, and review. Hundreds of labels per hour instead of dozens.
- **Cloud-native geospatial data.** Point the tool at COGs, PMTiles, or tile servers. Multi-user collaboration with no downloading, no preprocessing, no format conversion.
- **Embeddings as a core component.** Embeddings are used to interactively and rapidly create presence/absence data. Calculate embeddings with custom models or point to existing embedding catalogs on the cloud.

## Running the Demo

```bash
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

Pre-computed assets (chips, CAMs, SAM embeddings) are included in the repo. The `pipeline/` directory contains the offline Python scripts that generated them — you don't need to run these to try the demo.

## Tech Stack

**Browser:** React + Vite, MapLibre GL JS, ONNX Runtime Web, proj4

**Pipeline:** Python (pixi), PyTorch, DINOv2 (ViT-S/14), SAM (ViT-B), geopandas, scikit-learn

## License

MIT
