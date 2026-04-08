function StatusBar({ selectedChipId, clickPoints, maskIndex, maskResults, paintMode, brushSize, deleteMode }) {
  if (deleteMode) {
    return (
      <div className="status-bar status-bar-active" style={{ borderColor: '#ef4444' }}>
        <span className="status-bar-mask" style={{ color: '#ef4444' }}>DELETE MODE</span>
        <span className="status-bar-hints">
          click to delete &middot; drag box to delete region &middot; d exit
        </span>
      </div>
    )
  }

  if (!selectedChipId) {
    return (
      <div className="status-bar">
        <span className="status-bar-text">Click a chip to select it &middot; d delete mode</span>
      </div>
    )
  }

  if (paintMode) {
    return (
      <div className="status-bar status-bar-active">
        <span className="status-bar-mask">
          Paint: {paintMode === 'add' ? 'Add' : 'Erase'} ({brushSize}px)
        </span>
        <span className="status-bar-hints">
          a add &middot; e erase &middot; +/- size &middot; Esc exit paint
        </span>
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
          b paint &middot; [ ] cycle masks &middot; right-click exclude &middot; z undo &middot; ⌫ clear
        </span>
      </div>
    )
  }

  if (clickPoints.length === 0) {
    return (
      <div className="status-bar">
        <span className="status-bar-text">
          Click to add a point &middot; right-click to exclude &middot; b paint &middot; ⌫ deselect
        </span>
      </div>
    )
  }

  return null
}

export default StatusBar
