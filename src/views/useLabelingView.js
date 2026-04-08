import { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import maplibregl from 'maplibre-gl'
import { initSamDecoder, isDecoderReady, runSamDecoder, maskToDataURL, maskToPngBase64 } from '../sam'
import { loadNpy } from '../npy'
import { data } from '../data.js'
import { usePaintbrush } from '../hooks/usePaintbrush'

// Convert lon/lat click to pixel coords within the chip using inverse
// bilinear interpolation across all four corners.
// corners: [TL, TR, BR, BL] in [lon, lat]
// Pixel space: u goes right (TL→TR), v goes DOWN (TL→BL).
// Lat increases upward, so the v-axis (TL→BL) has decreasing lat.
// We set up the bilinear with pixel-space orientation:
//   P = TL + (TR-TL)*u + (BL-TL)*v + (TL-TR-BL+BR)*u*v
function lonLatToPixel(lon, lat, corners) {
  const [tl, tr, br, bl] = corners
  const ax = tl[0], ay = tl[1]
  const bx = tr[0] - tl[0], by = tr[1] - tl[1]
  const cx = bl[0] - tl[0], cy = bl[1] - tl[1]
  const dx = br[0] - tr[0] - bl[0] + tl[0], dy = br[1] - tr[1] - bl[1] + tl[1]

  const ex = lon - ax, ey = lat - ay

  const cross = (ux, uy, vx, vy) => ux * vy - uy * vx

  const A = cross(dx, dy, cx, cy)
  const B = cross(bx, by, cx, cy) + cross(ex, ey, dx, dy)
  const C = cross(ex, ey, bx, by)

  let u, v
  if (Math.abs(A) < 1e-12) {
    v = B !== 0 ? -C / B : 0
  } else {
    const disc = B * B - 4 * A * C
    if (disc < 0) return { u: -1, v: -1, x: -1, y: -1 }
    const sq = Math.sqrt(disc)
    const v1 = (-B + sq) / (2 * A)
    const v2 = (-B - sq) / (2 * A)
    v = (v1 >= -0.01 && v1 <= 1.01) ? v1 : v2
  }

  const denom = bx + dx * v
  if (Math.abs(denom) > 1e-12) {
    u = (ex - cx * v) / denom
  } else {
    const denomY = by + dy * v
    u = denomY !== 0 ? (ey - cy * v) / denomY : 0
  }

  return { u, v, x: u * 512, y: v * 512 }
}

function raiseLabels(map) {
  if (map.getLayer('labels-fill')) map.moveLayer('labels-fill')
  if (map.getLayer('labels-outline')) map.moveLayer('labels-outline')
}

function getMapBbox(map) {
  const bounds = map.getBounds()
  return `${bounds.getWest()},${bounds.getSouth()},${bounds.getEast()},${bounds.getNorth()}`
}

export function useLabelingView({ active, map, featureById, layerProviders = [] }) {
  const [selectedChipId, setSelectedChipId] = useState(null)
  const [clickPoints, setClickPoints] = useState([])
  const [maskResults, setMaskResults] = useState(null)
  const [maskIndex, setMaskIndex] = useState(-1)
  const [showLabels, setShowLabels] = useState(false)
  const [previewGeojson, setPreviewGeojson] = useState(null)
  const [deleteMode, setDeleteMode] = useState(false)
  const [chipCorners, setChipCorners] = useState(null)
  const paintbrush = usePaintbrush()
  const paintbrushRef = useRef(paintbrush)
  useEffect(() => { paintbrushRef.current = paintbrush }, [paintbrush])

  // Mutable handler state (refs)
  const chipRef = useRef({ id: null, corners: null, embedding: null })
  const pointsRef = useRef([])
  const lastLowResMaskRef = useRef(null)
  const pointMarkersRef = useRef([])
  const maskResultsRef = useRef(null)
  const maskIndexRef = useRef(-1)
  const selectingChipRef = useRef(false)
  const selectGenRef = useRef(0)
  const samInitRef = useRef(false)
  const paintModeRef = useRef(null)
  const layerProvidersRef = useRef(layerProviders)
  const showLabelsRef = useRef(false)
  const previewGeojsonRef = useRef(null)
  const pendingMaskRef = useRef(null)
  const deleteModeRef = useRef(false)
  const deleteDrawingRef = useRef(false)
  const deleteStartRef = useRef(null)
  const deleteRectInitRef = useRef(false)

  useEffect(() => { layerProvidersRef.current = layerProviders }, [layerProviders])
  useEffect(() => { showLabelsRef.current = showLabels }, [showLabels])
  useEffect(() => { previewGeojsonRef.current = previewGeojson }, [previewGeojson])
  useEffect(() => { deleteModeRef.current = deleteMode }, [deleteMode])
  useEffect(() => { maskResultsRef.current = maskResults }, [maskResults])
  useEffect(() => { maskIndexRef.current = maskIndex }, [maskIndex])
  useEffect(() => { paintModeRef.current = paintbrush.paintMode }, [paintbrush.paintMode])

  // Sync layer provider visibility (e.g. CAM toggle)
  useEffect(() => {
    if (!map) return
    for (const lp of layerProviders) {
      if (lp.syncVisibility) lp.syncVisibility(map)
    }
  }, [map, layerProviders])

  // Delete mode: cursor + dragPan management
  useEffect(() => {
    if (!map) return
    if (deleteMode) {
      map.getCanvas().style.cursor = 'crosshair'
    } else {
      map.getCanvas().style.cursor = ''
    }
  }, [map, deleteMode])

  // Add delete-rect source/layer once
  useEffect(() => {
    if (!map || deleteRectInitRef.current) return
    deleteRectInitRef.current = true
    map.addSource('delete-rect', {
      type: 'geojson',
      data: { type: 'FeatureCollection', features: [] },
    })
    map.addLayer({
      id: 'delete-rect-fill',
      type: 'fill',
      source: 'delete-rect',
      paint: { 'fill-color': '#ef4444', 'fill-opacity': 0.15 },
    })
    map.addLayer({
      id: 'delete-rect-outline',
      type: 'line',
      source: 'delete-rect',
      paint: { 'line-color': '#ef4444', 'line-width': 2, 'line-dasharray': [4, 2] },
    })
  }, [map])

  // Init SAM decoder once
  useEffect(() => {
    if (samInitRef.current) return
    samInitRef.current = true
    initSamDecoder().then(() => console.log('SAM decoder ready'))
  }, [])

  // Labels layer: auto-sync with server via bbox filtering, polling, and moveend
  useEffect(() => {
    if (!map || !active || !showLabels) {
      if (map) {
        if (map.getLayer('labels-fill')) map.removeLayer('labels-fill')
        if (map.getLayer('labels-outline')) map.removeLayer('labels-outline')
        if (map.getSource('labels')) map.removeSource('labels')
      }
      return
    }

    let cancelled = false
    let debounceTimer = null

    function fetchLabels() {
      const bbox = getMapBbox(map)
      fetch(data.labelsUrl(bbox))
        .then(r => r.json())
        .then(geojson => {
          if (cancelled) return
          if (map.getSource('labels')) {
            map.getSource('labels').setData(geojson)
          } else {
            map.addSource('labels', { type: 'geojson', data: geojson })
            map.addLayer({
              id: 'labels-fill',
              type: 'fill',
              source: 'labels',
              paint: {
                'fill-color': '#22c55e',
                'fill-opacity': 0.35,
              },
            })
            map.addLayer({
              id: 'labels-outline',
              type: 'line',
              source: 'labels',
              paint: {
                'line-color': '#16a34a',
                'line-width': 1.5,
              },
            })
          }
        })
    }

    function onMoveEnd() {
      clearTimeout(debounceTimer)
      debounceTimer = setTimeout(fetchLabels, 1000)
    }

    fetchLabels()
    map.on('moveend', onMoveEnd)

    return () => {
      cancelled = true
      clearTimeout(debounceTimer)
      map.off('moveend', onMoveEnd)
      if (map.getLayer('labels-fill')) map.removeLayer('labels-fill')
      if (map.getLayer('labels-outline')) map.removeLayer('labels-outline')
      if (map.getSource('labels')) map.removeSource('labels')
    }
  }, [map, active, showLabels])

  // Event handlers — active only when this view is active
  useEffect(() => {
    if (!map || !active) return

    const chip = chipRef.current
    const points = pointsRef.current
    const markers = pointMarkersRef.current

    function showPreviewLayer(geojson) {
      if (map.getLayer('preview-fill')) map.removeLayer('preview-fill')
      if (map.getLayer('preview-outline')) map.removeLayer('preview-outline')
      if (map.getSource('preview')) map.removeSource('preview')
      map.addSource('preview', { type: 'geojson', data: geojson })
      map.addLayer({
        id: 'preview-fill',
        type: 'fill',
        source: 'preview',
        paint: { 'fill-color': '#22c55e', 'fill-opacity': 0.4 },
      })
      map.addLayer({
        id: 'preview-outline',
        type: 'line',
        source: 'preview',
        paint: { 'line-color': '#16a34a', 'line-width': 2 },
      })
    }

    function removePreviewLayer() {
      if (map.getLayer('preview-fill')) map.removeLayer('preview-fill')
      if (map.getLayer('preview-outline')) map.removeLayer('preview-outline')
      if (map.getSource('preview')) map.removeSource('preview')
      previewGeojsonRef.current = null
      pendingMaskRef.current = null
      setPreviewGeojson(null)
    }

    function removeOverlays() {
      for (const id of ['mask-overlay', 'chip-overlay']) {
        if (map.getLayer(id)) map.removeLayer(id)
        if (map.getSource(id)) map.removeSource(id)
      }
      // Let layer providers clean up their own overlays
      for (const lp of layerProvidersRef.current) {
        lp.onChipDeselect(map)
      }
    }

    function clearPointMarkers() {
      for (const marker of markers) marker.remove()
      markers.length = 0
    }

    function clearSegmentation() {
      if (map.getLayer('mask-overlay')) map.removeLayer('mask-overlay')
      if (map.getSource('mask-overlay')) map.removeSource('mask-overlay')
      clearPointMarkers()
      points.length = 0
      lastLowResMaskRef.current = null
      setMaskResults(null)
      setMaskIndex(-1)
      setClickPoints([])
      removePreviewLayer()
    }

    function deselectChip() {
      clearSegmentation()
      removeOverlays()
      chip.id = null
      chip.corners = null
      chip.embedding = null
      setSelectedChipId(null)
      setChipCorners(null)
      if (map.getLayer('chips-outline')) {
        map.setPaintProperty('chips-outline', 'line-color', [
          'match', ['get', 'split'],
          'train', '#3b82f6',
          'test', '#ef4444',
          'validate', '#f59e0b',
          '#888888',
        ])
        map.setPaintProperty('chips-outline', 'line-width', 1.5)
      }
    }

    function addPointMarker(lon, lat, label) {
      const el = document.createElement('div')
      el.className = 'point-marker'
      el.style.cssText = `
        width: 12px; height: 12px; border-radius: 50%;
        background: ${label === 1 ? '#22c55e' : '#ef4444'};
        border: 2px solid white;
        box-shadow: 0 0 4px rgba(0,0,0,0.5);
      `
      const marker = new maplibregl.Marker({ element: el })
        .setLngLat([lon, lat])
        .addTo(map)
      markers.push(marker)
    }

    async function showMask(mask, imageCoords) {
      const dataURL = await maskToDataURL(mask)
      const src = map.getSource('mask-overlay')
      if (src) {
        src.updateImage({ url: dataURL, coordinates: imageCoords })
      } else {
        map.addSource('mask-overlay', {
          type: 'image',
          url: dataURL,
          coordinates: imageCoords,
        })
        map.addLayer({
          id: 'mask-overlay',
          type: 'raster',
          source: 'mask-overlay',
        })
      }
      raiseLabels(map)
    }

    function getMaskPrior() {
      if (lastLowResMaskRef.current) return lastLowResMaskRef.current
      for (const lp of layerProvidersRef.current) {
        if (lp.maskPrior) return lp.maskPrior
      }
      return null
    }

    async function handleSegClick(lon, lat, label) {
      if (!chip.id || !chip.embedding || !chip.corners) return
      if (!isDecoderReady()) return

      const pix = lonLatToPixel(lon, lat, chip.corners)
      if (!pix || pix.u < 0 || pix.u > 1 || pix.v < 0 || pix.v > 1) return

      points.push({ x: pix.x, y: pix.y, label, lon, lat })
      addPointMarker(lon, lat, label)
      setClickPoints([...points])

      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))

      const maskPrior = getMaskPrior()
      const { masks, scores, lowResMasks } = await runSamDecoder(chip.embedding, points, maskPrior)
      setMaskResults({ masks, scores, lowResMasks, imageCoords: chip.corners })
      setMaskIndex(0)
      showMask(masks[0], chip.corners)

      if (lowResMasks && lowResMasks.length > 0) {
        lastLowResMaskRef.current = new Float32Array(lowResMasks[0])
      }
    }

    async function selectChip(chipId) {
      if (chipId === chip.id) return

      removeOverlays()
      clearSegmentation()

      const feature = featureById.current[chipId]
      if (!feature) return

      const gen = ++selectGenRef.current

      // Vertex order from source: [NE, SE, SW, NW, NE(closing)]
      // MapLibre image coords: [TL, TR, BR, BL] = [NW, NE, SE, SW]
      const coords = feature.geometry.coordinates[0]
      const imageCoords = [coords[3], coords[0], coords[1], coords[2]]

      map.addSource('chip-overlay', {
        type: 'image',
        url: data.chipImageUrl(chipId),
        coordinates: imageCoords,
      })
      map.addLayer({
        id: 'chip-overlay',
        type: 'raster',
        source: 'chip-overlay',
        paint: { 'raster-resampling': 'nearest' },
      })
      raiseLabels(map)

      // Let layer providers add their overlays
      for (const lp of layerProvidersRef.current) {
        lp.onChipSelect(map, chipId, imageCoords)
      }

      chip.id = chipId
      chip.corners = imageCoords
      setSelectedChipId(chipId)
      setChipCorners(imageCoords)

      const lngs = imageCoords.map(c => c[0])
      const lats = imageCoords.map(c => c[1])
      map.fitBounds(
        [[Math.min(...lngs), Math.min(...lats)], [Math.max(...lngs), Math.max(...lats)]],
        { padding: 80, duration: 500 }
      )

      map.setPaintProperty('chips-outline', 'line-color', [
        'case',
        ['==', ['get', 'id'], chipId],
        '#facc15',
        ['match', ['get', 'split'],
          'train', '#3b82f6',
          'test', '#ef4444',
          'validate', '#f59e0b',
          '#888888',
        ],
      ])
      map.setPaintProperty('chips-outline', 'line-width', [
        'case',
        ['==', ['get', 'id'], chipId],
        3,
        1.5,
      ])

      try {
        const embedding = await loadNpy(data.embeddingUrl(chipId))
        if (selectGenRef.current !== gen) return // stale — user selected another chip
        chip.embedding = embedding
      } catch (e) {
        console.error(`Failed to load embedding for ${chipId}:`, e)
        if (selectGenRef.current !== gen) return
        chip.embedding = null
      }
    }

    async function rerunSam() {
      if (!chip.id || !chip.embedding || !chip.corners) return
      if (points.length === 0) {
        if (map.getLayer('mask-overlay')) map.removeLayer('mask-overlay')
        if (map.getSource('mask-overlay')) map.removeSource('mask-overlay')
        lastLowResMaskRef.current = null
        setMaskResults(null)
        setMaskIndex(-1)
        return
      }
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))

      const maskPrior = getMaskPrior()
      const { masks, scores, lowResMasks } = await runSamDecoder(chip.embedding, points, maskPrior)
      setMaskResults({ masks, scores, lowResMasks, imageCoords: chip.corners })
      setMaskIndex(0)
      showMask(masks[0], chip.corners)
      if (lowResMasks && lowResMasks.length > 0) {
        lastLowResMaskRef.current = new Float32Array(lowResMasks[0])
      }
    }

    function removePointAt(idx) {
      if (idx < 0 || idx >= points.length) return
      points.splice(idx, 1)
      clearPointMarkers()
      for (const pt of points) addPointMarker(pt.lon, pt.lat, pt.label)
      setClickPoints([...points])
      rerunSam()
    }

    // Map click handlers
    // ── Delete mode helpers ──
    function refreshLabelsLayer() {
      if (showLabelsRef.current && map.getSource('labels')) {
        fetch(data.labelsUrl(getMapBbox(map))).then(r => r.json()).then(geojson => {
          if (map.getSource('labels')) map.getSource('labels').setData(geojson)
        })
      }
    }

    function onDeleteMouseDown(e) {
      if (!deleteModeRef.current) return
      e.preventDefault()
      deleteDrawingRef.current = true
      deleteStartRef.current = { lng: e.lngLat.lng, lat: e.lngLat.lat }
      map.dragPan.disable()
    }

    function onDeleteMouseMove(e) {
      if (!deleteDrawingRef.current || !deleteStartRef.current) return
      const start = deleteStartRef.current
      const current = e.lngLat
      const coords = [
        [start.lng, start.lat],
        [current.lng, start.lat],
        [current.lng, current.lat],
        [start.lng, current.lat],
        [start.lng, start.lat],
      ]
      map.getSource('delete-rect').setData({
        type: 'FeatureCollection',
        features: [{
          type: 'Feature',
          geometry: { type: 'Polygon', coordinates: [coords] },
          properties: {},
        }],
      })
    }

    function onDeleteMouseUp(e) {
      if (!deleteDrawingRef.current || !deleteStartRef.current) return
      deleteDrawingRef.current = false
      map.dragPan.enable()

      const end = e.lngLat
      const start = deleteStartRef.current
      deleteStartRef.current = null

      map.getSource('delete-rect').setData({ type: 'FeatureCollection', features: [] })

      // Check if this was a tiny click (point delete) vs a real box drag
      const swPx = map.project([Math.min(start.lng, end.lng), Math.min(start.lat, end.lat)])
      const nePx = map.project([Math.max(start.lng, end.lng), Math.max(start.lat, end.lat)])
      if (Math.abs(nePx.x - swPx.x) < 5 && Math.abs(nePx.y - swPx.y) < 5) {
        // Point delete
        data.deleteLabelsByGeometry({ point: [start.lng, start.lat] }).then((res) => {
          console.log(`Deleted ${res.deleted} label(s)`)
          refreshLabelsLayer()
        })
      } else {
        // Box delete
        const bbox = [
          Math.min(start.lng, end.lng), Math.min(start.lat, end.lat),
          Math.max(start.lng, end.lng), Math.max(start.lat, end.lat),
        ]
        data.deleteLabelsByGeometry({ bbox }).then((res) => {
          console.log(`Deleted ${res.deleted} label(s)`)
          refreshLabelsLayer()
        })
      }
    }

    function onChipFillClick(e) {
      if (deleteModeRef.current) return
      const chipId = e.features[0].properties.id
      if (chipId !== chip.id) {
        selectingChipRef.current = true
        selectChip(chipId)
      }
    }

    function onMapClick(e) {
      if (deleteModeRef.current) return
      if (selectingChipRef.current) {
        selectingChipRef.current = false
        return
      }
      if (!chip.id) return
      if (paintModeRef.current) return
      handleSegClick(e.lngLat.lng, e.lngLat.lat, 1)
    }

    function onContextMenu(e) {
      e.preventDefault()
      if (deleteModeRef.current) return
      if (!chip.id) return
      if (paintModeRef.current) return
      const rect = map.getCanvas().getBoundingClientRect()
      const point = new maplibregl.Point(e.clientX - rect.left, e.clientY - rect.top)
      const coord = map.unproject(point)
      handleSegClick(coord.lng, coord.lat, 0)
    }

    function handleKeyDown(e) {
      if (e.key === 'Backspace') {
        e.preventDefault()
        if (paintModeRef.current) return
        if (previewGeojsonRef.current) {
          removePreviewLayer()
          return
        }
        if (points.length > 0) {
          clearSegmentation()
        } else {
          deselectChip()
        }
        return
      }
      if (e.key === 'Enter') {
        e.preventDefault()
        if (!chip.id) return

        // Step 2: preview is showing — persist the label
        if (previewGeojsonRef.current && pendingMaskRef.current) {
          const pendingBase64 = pendingMaskRef.current
          data.saveChipLabel(chip.id, pendingBase64, 'positive').then((res) => {
            if (res.ok) {
              console.log('Label saved:', res.label_id)
              clearSegmentation()
              paintbrushRef.current.clearCorrections()
              if (showLabelsRef.current && map.getSource('labels')) {
                fetch(data.labelsUrl(getMapBbox(map))).then(r => r.json()).then(geojson => {
                  if (map.getSource('labels')) map.getSource('labels').setData(geojson)
                })
              }
            } else {
              console.error('Failed to save label:', res.detail || res)
            }
          })
          return
        }

        // Step 1: generate preview
        const r = maskResultsRef.current
        if (!r || r.masks.length === 0) return
        const idx = maskIndexRef.current >= 0 ? maskIndexRef.current : 0
        const samMask = r.masks[idx]
        const finalMask = paintbrushRef.current.compositeMask(samMask)
        const base64 = maskToPngBase64(finalMask)
        pendingMaskRef.current = base64
        data.vectorizePreview(chip.id, base64, 'positive').then((feature) => {
          if (feature.geometry) {
            const geojson = { type: 'FeatureCollection', features: [feature] }
            previewGeojsonRef.current = geojson
            setPreviewGeojson(geojson)
            showPreviewLayer(geojson)
            console.log(`Preview: ${feature.properties.vertex_count} vertices — Enter to save, Backspace to dismiss`)
          } else {
            console.error('Preview failed:', feature.detail || feature)
            pendingMaskRef.current = null
          }
        })
        return
      }
      if (e.key === 'Escape') {
        e.preventDefault()
        if (paintModeRef.current) {
          paintbrushRef.current.setPaintMode(null)
          return
        }
        deselectChip()
        return
      }

      // Delete mode toggle
      if (e.key === 'd' || e.key === 'D') {
        if (deleteModeRef.current) {
          setDeleteMode(false)
          map.dragPan.enable()
        } else {
          setDeleteMode(true)
          paintbrushRef.current.setPaintMode(null)
          setShowLabels(true)
        }
        return
      }

      if (deleteModeRef.current) return

      // Paintbrush shortcuts
      if (e.key === 'b' || e.key === 'B') {
        if (!chip.id) return
        paintbrushRef.current.setPaintMode((prev) => prev ? null : 'add')
        return
      }
      if (e.key === 'a' || e.key === 'A') {
        if (!chip.id) return
        paintbrushRef.current.setPaintMode('add')
        return
      }
      if (e.key === 'e' || e.key === 'E') {
        if (!chip.id) return
        paintbrushRef.current.setPaintMode('erase')
        return
      }
      if (e.key === '=' || e.key === '+') {
        paintbrushRef.current.adjustBrushSize(2)
        return
      }
      if (e.key === '-' || e.key === '_') {
        paintbrushRef.current.adjustBrushSize(-2)
        return
      }

      if (paintModeRef.current) return

      if (e.key === 'z' || e.key === 'Z') {
        if (points.length > 0) removePointAt(points.length - 1)
        return
      }
      const r = maskResultsRef.current
      if (!r || r.masks.length <= 1) return
      if (e.key !== '[' && e.key !== ']') return
      e.preventDefault()
      setMaskIndex((prev) => {
        const idx = e.key === ']'
          ? (prev + 1) % r.masks.length
          : (prev - 1 + r.masks.length) % r.masks.length
        showMask(r.masks[idx], r.imageCoords)
        if (r.lowResMasks && r.lowResMasks[idx]) {
          lastLowResMaskRef.current = new Float32Array(r.lowResMasks[idx])
        }
        paintbrushRef.current.clearCorrections()
        return idx
      })
    }

    function onChipMouseMove(e) {
      const chipId = e.features[0].properties.id
      map.getCanvas().style.cursor = chipId === chip.id ? 'crosshair' : 'pointer'
    }

    function onChipMouseLeave() {
      map.getCanvas().style.cursor = ''
    }

    map.on('click', 'chips-fill', onChipFillClick)
    map.on('click', onMapClick)
    map.getCanvas().addEventListener('contextmenu', onContextMenu)
    document.addEventListener('keydown', handleKeyDown)
    map.on('mousemove', 'chips-fill', onChipMouseMove)
    map.on('mouseleave', 'chips-fill', onChipMouseLeave)
    map.on('mousedown', onDeleteMouseDown)
    map.on('mousemove', onDeleteMouseMove)
    map.on('mouseup', onDeleteMouseUp)

    return () => {
      map.off('click', 'chips-fill', onChipFillClick)
      map.off('click', onMapClick)
      map.getCanvas().removeEventListener('contextmenu', onContextMenu)
      document.removeEventListener('keydown', handleKeyDown)
      map.off('mousemove', 'chips-fill', onChipMouseMove)
      map.off('mouseleave', 'chips-fill', onChipMouseLeave)
      map.off('mousedown', onDeleteMouseDown)
      map.off('mousemove', onDeleteMouseMove)
      map.off('mouseup', onDeleteMouseUp)
    }
  // paintbrush methods are accessed via paintbrushRef to avoid tearing down
  // all map handlers on every paintbrush state change.
  }, [map, active, featureById])

  // Update the map mask overlay when paintbrush modifies the composited mask
  const handleMaskUpdate = useCallback(async (composited) => {
    if (!map || !chipRef.current.corners) return
    const dataURL = await maskToDataURL(composited)
    const src = map.getSource('mask-overlay')
    if (src) {
      src.updateImage({ url: dataURL, coordinates: chipRef.current.corners })
    } else {
      map.addSource('mask-overlay', {
        type: 'image',
        url: dataURL,
        coordinates: chipRef.current.corners,
      })
      map.addLayer({
        id: 'mask-overlay',
        type: 'raster',
        source: 'mask-overlay',
      })
    }
    raiseLabels(map)
  }, [map])

  // Collect controls from all layer providers
  const toggleLabels = useCallback(() => setShowLabels(prev => !prev), [])
  const pluginControls = useMemo(() => [
    { label: 'Labels', active: showLabels, onToggle: toggleLabels },
    ...layerProviders.flatMap(lp => lp.controls || []),
  ], [showLabels, toggleLabels, layerProviders])

  // Get the current SAM mask for compositing
  const currentSamMask = useMemo(
    () => maskResults && maskIndex >= 0 ? maskResults.masks[maskIndex] : null,
    [maskResults, maskIndex],
  )
  return {
    selectedChipId,
    clickPoints,
    maskResults,
    maskIndex,
    pluginControls,
    paintbrush,
    chipCorners,
    currentSamMask,
    handleMaskUpdate,
    previewGeojson,
    deleteMode,
  }
}
