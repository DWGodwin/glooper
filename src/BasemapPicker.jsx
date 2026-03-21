import { BASEMAP_KEYS } from './mapStyle'

function BasemapPicker({ activeBasemap, onBasemapChange, camVisible, onToggleCam }) {
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
      <button
        className={camVisible ? 'active' : ''}
        onClick={onToggleCam}
      >
        Toggle Class Activation Map
      </button>
    </div>
  )
}

export default BasemapPicker
