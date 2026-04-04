import { useEffect, useRef, useState, useCallback } from 'react'
import { data } from '../data.js'

export function useDefineAreaView({ active, map, chipGrid }) {
  const [drawMode, setDrawMode] = useState(false)
  const [activeSplit, setActiveSplit] = useState('train')

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

  // Add draw-rect source/layer
  useEffect(() => {
    if (!map || initializedRef.current) return
    initializedRef.current = true

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

      data.createStudyArea({ sw, ne }, split).then(() => {
        chipGrid.refreshChips()
      })

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
