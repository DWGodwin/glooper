import { useEffect, useRef, useState, useCallback } from 'react'
import { data } from '../data.js'

export function useDefineAreaView({ active, map, chipGrid }) {
  const [drawMode, setDrawMode] = useState(false)
  const [deleteMode, setDeleteMode] = useState(false)
  const [activeSplit, setActiveSplit] = useState('train')
  const [prefetchJob, setPrefetchJob] = useState(null)

  const drawModeRef = useRef(false)
  const deleteModeRef = useRef(false)
  const activeSplitRef = useRef('train')
  const isDrawingRef = useRef(false)
  const drawStartRef = useRef(null)
  const initializedRef = useRef(false)

  // Keep refs in sync with state
  useEffect(() => { drawModeRef.current = drawMode }, [drawMode])
  useEffect(() => { deleteModeRef.current = deleteMode }, [deleteMode])
  useEffect(() => { activeSplitRef.current = activeSplit }, [activeSplit])

  // Cursor management for draw/delete mode
  useEffect(() => {
    if (!map) return
    if (drawMode || deleteMode) {
      map.dragPan.disable()
      map.getCanvas().style.cursor = 'crosshair'
    } else {
      map.dragPan.enable()
      map.getCanvas().style.cursor = ''
    }
  }, [map, drawMode, deleteMode])

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

  // Update draw-rect color based on mode
  useEffect(() => {
    if (!map || !map.getLayer('draw-rect-fill')) return
    const color = deleteMode ? '#ef4444' : '#3b82f6'
    map.setPaintProperty('draw-rect-fill', 'fill-color', color)
    map.setPaintProperty('draw-rect-outline', 'line-color', color)
  }, [map, deleteMode])

  // Draw interaction handlers — active only when this view is active
  useEffect(() => {
    if (!map || !active) return

    function onMouseDown(e) {
      if (!drawModeRef.current && !deleteModeRef.current) return
      e.preventDefault()
      isDrawingRef.current = true
      drawStartRef.current = { lng: e.lngLat.lng, lat: e.lngLat.lat }
    }

    function onMouseMove(e) {
      if ((drawModeRef.current || deleteModeRef.current) && !isDrawingRef.current) {
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

      const swPx = map.project(sw)
      const nePx = map.project(ne)
      const isTinyClick = Math.abs(nePx.x - swPx.x) < 5 && Math.abs(nePx.y - swPx.y) < 5

      if (deleteModeRef.current) {
        if (isTinyClick) {
          // Point delete
          data.deleteStudyArea({ point: [start.lng, start.lat] }).then((res) => {
            console.log(`Deleted ${res.chips_deleted} chip(s), ${res.labels_deleted} label(s)`)
            chipGrid.refreshChips()
          })
        } else {
          // Box delete
          const bbox = [
            Math.min(start.lng, end.lng), Math.min(start.lat, end.lat),
            Math.max(start.lng, end.lng), Math.max(start.lat, end.lat),
          ]
          data.deleteStudyArea({ bbox }).then((res) => {
            console.log(`Deleted ${res.chips_deleted} chip(s), ${res.labels_deleted} label(s)`)
            chipGrid.refreshChips()
          })
        }
        return
      }

      // Skip tiny accidental clicks for draw mode
      if (isTinyClick) return

      const split = activeSplitRef.current

      data.createStudyArea({ sw, ne }, split).then((res) => {
        chipGrid.refreshChips()
        if (res.job_id) {
          setPrefetchJob({
            jobId: res.job_id,
            phase: 'chips',
            chips_total: res.count, chips_done: 0, chips_failed: 0,
            embed_total: 0, embed_done: 0, embed_failed: 0,
          })
        }
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

  // Poll prefetch status
  const prefetchJobId = prefetchJob?.jobId
  useEffect(() => {
    if (!prefetchJobId) return
    const interval = setInterval(() => {
      data.prefetchStatus(prefetchJobId).then((status) => {
        setPrefetchJob((prev) => prev && { ...prev, ...status })
        if (status.phase === 'complete') {
          clearInterval(interval)
        }
      })
    }, 2000)
    return () => clearInterval(interval)
  }, [prefetchJobId])

  const toggleDraw = useCallback(() => {
    setDeleteMode(false)
    setDrawMode((prev) => !prev)
  }, [])

  const toggleDelete = useCallback(() => {
    setDrawMode(false)
    setDeleteMode((prev) => !prev)
  }, [])

  return {
    drawMode,
    toggleDraw,
    deleteMode,
    toggleDelete,
    activeSplit,
    setActiveSplit,
    prefetchJob,
  }
}
