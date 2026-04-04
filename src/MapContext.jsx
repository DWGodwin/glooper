import { createContext, useContext, useRef, useEffect, useState, useCallback } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import { BASEMAP_KEYS, makeStyle } from './mapStyle'

const MapContext = createContext(null)

// eslint-disable-next-line react-refresh/only-export-components
export function useMap() {
  return useContext(MapContext)
}

export function MapProvider({ children }) {
  const containerRef = useRef(null)
  const [map, setMap] = useState(null)
  const [activeBasemap, setActiveBasemap] = useState('imagery')
  const initializedRef = useRef(false)

  useEffect(() => {
    if (initializedRef.current) return
    initializedRef.current = true

    const m = new maplibregl.Map({
      container: containerRef.current,
      style: makeStyle(),
      center: [-71.82, 42.25],
      zoom: 17,
    })

    m.addControl(new maplibregl.NavigationControl(), 'top-right')

    m.on('load', () => {
      setMap(m)
    })
  }, [])

  const switchBasemap = useCallback((key) => {
    setActiveBasemap(key)
    if (!map || !map.isStyleLoaded()) return
    for (const k of BASEMAP_KEYS) {
      map.setLayoutProperty(k, 'visibility', k === key ? 'visible' : 'none')
    }
  }, [map])

  const value = {
    map,
    activeBasemap,
    switchBasemap,
  }

  return (
    <MapContext.Provider value={value}>
      <div ref={containerRef} className="map-container" />
      {children}
    </MapContext.Provider>
  )
}
