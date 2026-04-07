
function ViewNav({activeView, onActiveViewChange, pluginViews = []}) {
  return (
    <div className="viewnav">
      <button
        className={activeView === 'define-area' ? 'active' : ''}
        onClick={() => onActiveViewChange('define-area')}
      >
        Define Study Area
      </button>
      <button
        className={activeView === 'labeling' ? 'active' : ''}
        onClick={() => onActiveViewChange('labeling')}
      >
        Label Area
      </button>
      {pluginViews.map((pv) => (
        <button
          key={pv.id}
          className={activeView === pv.id ? 'active' : ''}
          onClick={() => onActiveViewChange(pv.id)}
        >
          {pv.label}
        </button>
      ))}
    </div>
  )
}

export default ViewNav
