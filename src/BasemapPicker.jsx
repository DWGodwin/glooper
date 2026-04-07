import { BASEMAP_KEYS } from './mapStyle'

function BasemapPicker({ activeBasemap, onBasemapChange, pluginControls = [] }) {
  return (
    <div className="basemap-picker">
      {BASEMAP_KEYS.map((key) => (
        <button
          key={key}
          className={activeBasemap === key ? 'active' : ''}
          onClick={() => onBasemapChange(key)}
        >
          {key === 'imagery' ? 'Imagery' : key === 'massgis' ? 'MassGIS' : 'Map'}
        </button>
      ))}
      {pluginControls.map((ctrl, i) => (
        <button
          key={ctrl.label || i}
          className={ctrl.active ? 'active' : ''}
          onClick={ctrl.onToggle}
        >
          {ctrl.label}
        </button>
      ))}
    </div>
  )
}

export default BasemapPicker
