function DefineArea({ drawMode, onToggleDraw, activeSplit, onSplitChange, prefetchJob }) {
  const splits = ['train', 'test', 'validate']

  return (
    <div className="define-area-panel">
      <div className="split-selector">
        {splits.map((split) => (
          <button
            key={split}
            className={`split-btn split-${split} ${activeSplit === split ? 'active' : ''}`}
            onClick={() => onSplitChange(split)}
          >
            {split}
          </button>
        ))}
      </div>
      <button
        className={`draw-toggle ${drawMode ? 'active' : ''}`}
        onClick={onToggleDraw}
      >
        {drawMode ? 'Cancel Drawing' : 'Draw Rectangle'}
      </button>
      {prefetchJob && prefetchJob.done + prefetchJob.failed < prefetchJob.total && (
        <div className="prefetch-status">
          Caching images... {prefetchJob.done}/{prefetchJob.total}
          {prefetchJob.failed > 0 && ` (${prefetchJob.failed} failed)`}
        </div>
      )}
    </div>
  )
}

export default DefineArea
