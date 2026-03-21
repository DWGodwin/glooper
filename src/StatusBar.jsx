function StatusBar({ selectedChipId, clickPoints, maskIndex, maskResults }) {
  if (!selectedChipId) {
    return (
      <div className="status-bar">
        <span className="status-bar-text">Click a chip to select it</span>
      </div>
    )
  }

  if (maskIndex >= 0 && maskResults) {
    return (
      <div className="status-bar status-bar-active">
        <span className="status-bar-mask">
          Mask {maskIndex + 1}/{maskResults.masks.length}
          {' '}&middot; IoU {maskResults.scores[maskIndex].toFixed(2)}
          {' '}&middot; {clickPoints.length} point{clickPoints.length !== 1 ? 's' : ''}
        </span>
        <span className="status-bar-hints">
          use [ & ] to cycle masks &middot; right-click exclude &middot; z undo &middot; ⌫ clear
        </span>
      </div>
    )
  }

  if (clickPoints.length === 0) {
    return (
      <div className="status-bar">
        <span className="status-bar-text">
          Click to add a point &middot; right-click to exclude &middot; ⌫ deselect
        </span>
      </div>
    )
  }

  return null
}

export default StatusBar
