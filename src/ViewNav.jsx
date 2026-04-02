
function ViewNav({activeView, onActiveViewChange}) {
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
    </div>
  )
}

export default ViewNav
