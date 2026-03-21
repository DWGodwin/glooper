import { useRef, useEffect } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import './App.css'

function App() {
  const mapContainer = useRef(null)
  const map = useRef(null)

  useEffect(() => {
    if (map.current) return

    map.current = new maplibregl.Map({
      container: mapContainer.current,
      style: {
        version: 8,
        sources: {
          'esri-world-imagery': {
            type: 'raster',
            tiles: [
              'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            ],
            tileSize: 256,
            attribution:
              'Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics',
          },
        },
        layers: [
          {
            id: 'esri-world-imagery',
            type: 'raster',
            source: 'esri-world-imagery',
            minzoom: 0,
            maxzoom: 19,
          },
        ],
      },
      center: [-117.1, 33.1],
      zoom: 12,
    })

    map.current.addControl(new maplibregl.NavigationControl(), 'top-right')

    return () => {
      map.current.remove()
      map.current = null
    }
  }, [])

  return <div ref={mapContainer} className="map-container" />
}

export default App
