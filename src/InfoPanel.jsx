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
        High-quality segmentation from minimal supervision, running in your browser.
      </p>
      <p className="info-subtitle">
        GitHub Repo: <a href="https://github.com/DWGodwin/glooper/" target="_blank" rel="noopener">DWGodwin/glooper</a>
      </p>

      <div className="info-tryit">
        <strong>Try it:</strong>
        <ul>
          <li>Click a chip on the map, then click within it to segment.</li>
          <li>Right-click to exclude areas.</li>
          <li>Use <kbd>[</kbd> and <kbd>]</kbd> to select the best mask.</li>
          <li>Prompt with more points iteratively.</li>
          <li><kbd>b</kbd> triggers painting mode to manually correct SAM predictions as a final step.</li>
          <li><kbd>Enter</kbd> previews a vectorized SAM mask that is sent to the label database in the full version.</li>
        </ul>
      </div>
    </div>
  )
}

export default InfoPanel
