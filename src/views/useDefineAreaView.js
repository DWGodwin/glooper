import { useEffect, useRef, useState, useCallback } from 'react'
import proj4 from 'proj4'
import { data } from '../data.js'

const IS_DEMO = import.meta.env.VITE_DATA_SOURCE !== 'api'
const CHIP_SIZE_M = 76.8

proj4.defs('EPSG:32619', '+proj=utm +zone=19 +datum=WGS84 +units=m +no_defs')

function computeGrid(sw, ne, split) {
  const swUtm = proj4('EPSG:4326', 'EPSG:32619', sw)
  const neUtm = proj4('EPSG:4326', 'EPSG:32619', ne)

  const minE = Math.floor(swUtm[0] / CHIP_SIZE_M) * CHIP_SIZE_M
  const minN = Math.floor(swUtm[1] / CHIP_SIZE_M) * CHIP_SIZE_M
  const maxE = Math.ceil(neUtm[0] / CHIP_SIZE_M) * CHIP_SIZE_M
  const maxN = Math.ceil(neUtm[1] / CHIP_SIZE_M) * CHIP_SIZE_M

  const features = []
  for (let e = minE; e < maxE; e += CHIP_SIZE_M) {
    for (let n = minN; n < maxN; n += CHIP_SIZE_M) {
      const sw_ll = proj4('EPSG:32619', 'EPSG:4326', [e, n])
      const se_ll = proj4('EPSG:32619', 'EPSG:4326', [e + CHIP_SIZE_M, n])
      const ne_ll = proj4('EPSG:32619', 'EPSG:4326', [e + CHIP_SIZE_M, n + CHIP_SIZE_M])
      const nw_ll = proj4('EPSG:32619', 'EPSG:4326', [e, n + CHIP_SIZE_M])

      features.push({
        type: 'Feature',
        geometry: {
          type: 'Polygon',
          coordinates: [[nw_ll, ne_ll, se_ll, sw_ll, nw_ll]],
        },
        properties: {
          split,
          id: `${e.toFixed(2)}e_${n.toFixed(2)}n`,
        },
      })
    }
  }
  return features
}

export function useDefineAreaView({ active, map, chipGrid }) {
  const [drawMode, setDrawMode] = useState(false)
  const [activeSplit, setActiveSplit] = useState('train')
  const [studyAreas, setStudyAreas] = useState([])

  const drawModeRef = useRef(false)
  const activeSplitRef = useRef('train')
  const isDrawingRef = useRef(false)
  const drawStartRef = useRef(null)
  const initializedRef = useRef(false)

  // Keep refs in sync with state
  useEffect(() => { drawModeRef.current = drawMode }, [drawMode])
  useEffect(() => { activeSplitRef.current = activeSplit }, [activeSplit])

  // Cursor management for draw mode
  useEffect(() => {
    if (!map) return
    if (drawMode) {
      map.dragPan.disable()
      map.getCanvas().style.cursor = 'crosshair'
    } else {
      map.dragPan.enable()
      map.getCanvas().style.cursor = ''
    }
  }, [map, drawMode])

  // Sync study areas to map source
  useEffect(() => {
    if (!map || !map.getSource('study-areas')) return
    map.getSource('study-areas').setData({
      type: 'FeatureCollection',
      features: studyAreas,
    })
  }, [map, studyAreas])

  // Add sources/layers and event handlers
  useEffect(() => {
    if (!map || initializedRef.current) return
    initializedRef.current = true

    map.addSource('study-areas', {
      type: 'geojson',
      data: { type: 'FeatureCollection', features: [] },
    })
    map.addLayer({
      id: 'study-areas-fill',
      type: 'fill',
      source: 'study-areas',
      paint: {
        'fill-color': [
          'match', ['get', 'split'],
          'train', '#3b82f6',
          'test', '#ef4444',
          'validate', '#f59e0b',
          '#888888',
        ],
        'fill-opacity': 0.15,
      },
    })
    map.addLayer({
      id: 'study-areas-outline',
      type: 'line',
      source: 'study-areas',
      paint: {
        'line-color': [
          'match', ['get', 'split'],
          'train', '#3b82f6',
          'test', '#ef4444',
          'validate', '#f59e0b',
          '#888888',
        ],
        'line-width': 2,
      },
    })

    map.addSource('draw-rect', {
      type: 'geojson',
      data: { type: 'FeatureCollection', features: [] },
    })
    map.addLayer({
      id: 'draw-rect-fill',
      type: 'fill',
      source: 'draw-rect',
      paint: {
        'fill-color': '#3b82f6',
        'fill-opacity': 0.15,
      },
    })
    map.addLayer({
      id: 'draw-rect-outline',
      type: 'line',
      source: 'draw-rect',
      paint: {
        'line-color': '#3b82f6',
        'line-width': 2,
        'line-dasharray': [4, 2],
      },
    })
  }, [map])

  // Draw interaction handlers — active only when this view is active
  useEffect(() => {
    if (!map || !active) return

    function onMouseDown(e) {
      if (!drawModeRef.current) return
      e.preventDefault()
      isDrawingRef.current = true
      drawStartRef.current = { lng: e.lngLat.lng, lat: e.lngLat.lat }
    }

    function onMouseMove(e) {
      if (drawModeRef.current && !isDrawingRef.current) {
        map.getCanvas().style.cursor = 'crosshair'
      }

      if (!isDrawingRef.current || !drawStartRef.current) return

      const current = e.lngLat
      const start = drawStartRef.current
      const coords = [
        [start.lng, start.lat],
        [current.lng, start.lat],
        [current.lng, current.lat],
        [start.lng, current.lat],
        [start.lng, start.lat],
      ]
      map.getSource('draw-rect').setData({
        type: 'FeatureCollection',
        features: [{
          type: 'Feature',
          geometry: { type: 'Polygon', coordinates: [coords] },
          properties: {},
        }],
      })
    }

    function onMouseUp(e) {
      if (!isDrawingRef.current || !drawStartRef.current) return
      isDrawingRef.current = false

      const end = e.lngLat
      const start = drawStartRef.current
      const sw = [Math.min(start.lng, end.lng), Math.min(start.lat, end.lat)]
      const ne = [Math.max(start.lng, end.lng), Math.max(start.lat, end.lat)]

      drawStartRef.current = null

      // Clear draw preview
      map.getSource('draw-rect').setData({ type: 'FeatureCollection', features: [] })

      // Skip tiny accidental clicks
      const swPx = map.project(sw)
      const nePx = map.project(ne)
      if (Math.abs(nePx.x - swPx.x) < 5 && Math.abs(nePx.y - swPx.y) < 5) return

      const split = activeSplitRef.current

      if (IS_DEMO) {
        const gridFeatures = computeGrid(sw, ne, split)
        setStudyAreas((prev) => [...prev, ...gridFeatures])
      } else {
        data.createStudyArea({ sw, ne }, split).then((geojson) => {
          if (geojson.features) {
            setStudyAreas((prev) => [...prev, ...geojson.features])
            chipGrid.refreshChips()
          }
        })
      }

      // Auto-exit draw mode
      setDrawMode(false)
    }

    map.on('mousedown', onMouseDown)
    map.on('mousemove', onMouseMove)
    map.on('mouseup', onMouseUp)

    return () => {
      map.off('mousedown', onMouseDown)
      map.off('mousemove', onMouseMove)
      map.off('mouseup', onMouseUp)
    }
  }, [map, active, chipGrid])

  const toggleDraw = useCallback(() => {
    setDrawMode((prev) => !prev)
  }, [])

  return {
    drawMode,
    toggleDraw,
    activeSplit,
    setActiveSplit,
  }
}
