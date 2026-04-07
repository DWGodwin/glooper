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
      {prefetchJob && prefetchJob.phase !== 'complete' && (
        <div className="prefetch-status">
          {prefetchJob.phase === 'chips' && (
            <>Caching images... {prefetchJob.chips_done}/{prefetchJob.chips_total}
            {prefetchJob.chips_failed > 0 && ` (${prefetchJob.chips_failed} failed)`}</>
          )}
          {prefetchJob.phase === 'embeddings' && (
            <>Generating embeddings... {prefetchJob.embed_done}/{prefetchJob.embed_total}</>
          )}
        </div>
      )}
    </div>
  )
}

export default DefineArea
