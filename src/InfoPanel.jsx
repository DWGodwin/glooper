import { useState } from 'react'

function InfoPanel() {
  const [collapsed, setCollapsed] = useState(false)

  if (collapsed) {
    return (
      <button className="info-toggle" onClick={() => setCollapsed(false)}>
        ?
      </button>
    )
  }

  return (
    <div className="info-panel">
      <button className="info-close" onClick={() => setCollapsed(true)}>
        &times;
      </button>
      <h2>Glooper Demo: Refining Labels</h2>
      <p className="info-subtitle">
        High-quality segmentation from minimal supervision, running in your browser. GitHub Repo: https://github.com/DWGodwin/glooper/
      </p>

      <div className="info-pipeline">
        <div className="info-step">
          <span className="info-step-num">1</span>
          <div>
            <strong>Cheap labels</strong>
            <span className="info-detail">
              Each image chip gets a simple present/absent label &mdash; no pixel-level annotation needed. In this case, green chips have rooftop solar, red chips have none.
            </span>
          </div>
        </div>
        <div className="info-step">
          <span className="info-step-num">2</span>
          <div>
            <strong>DINOv2 activation maps</strong>
            <span className="info-detail">
              A linear probe on DINOv2 features produces class activation maps that highlight areas of interest based on presence/absence labels.
            </span>
          </div>
        </div>
        <div className="info-step">
          <span className="info-step-num">3</span>
          <div>
            <strong>SAM refinement in-browser</strong>
            <span className="info-detail">
              The class activation map seeds SAM's decoder (running locally via ONNX). Click to refine &mdash; SAM snaps to precise boundaries.
            </span>
          </div>
        </div>
      </div>

      <div className="info-tryit">
        <strong>Try it:</strong> Click a chip on the map, then click within it to segment.
        Right-click to exclude areas. Use [ and ] to select the best mask. Toggle the class activation map overlay to see what DINOv2 sees.
      </div>
    </div>
  )
}

export default InfoPanel
