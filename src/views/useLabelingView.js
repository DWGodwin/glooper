import { useEffect, useRef, useState, useCallback } from 'react'
import maplibregl from 'maplibre-gl'
import { initSamDecoder, isDecoderReady, runSamDecoder, maskToDataURL } from '../sam'
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

export function useLabelingView({ active, map, featureById, layerProviders = [] }) {
  const [selectedChipId, setSelectedChipId] = useState(null)
  const [clickPoints, setClickPoints] = useState([])
  const [maskResults, setMaskResults] = useState(null)
  const [maskIndex, setMaskIndex] = useState(-1)
  const paintbrush = usePaintbrush()

  // Mutable handler state (refs)
  const chipRef = useRef({ id: null, corners: null, embedding: null })
  const pointsRef = useRef([])
  const lastLowResMaskRef = useRef(null)
  const pointMarkersRef = useRef([])
  const maskResultsRef = useRef(null)
  const selectingChipRef = useRef(false)
  const samInitRef = useRef(false)
  const paintModeRef = useRef(null)
  const layerProvidersRef = useRef(layerProviders)

  useEffect(() => { layerProvidersRef.current = layerProviders }, [layerProviders])
  useEffect(() => { maskResultsRef.current = maskResults }, [maskResults])
  useEffect(() => { paintModeRef.current = paintbrush.paintMode }, [paintbrush.paintMode])

  // Sync layer provider visibility (e.g. CAM toggle)
  useEffect(() => {
    if (!map) return
    for (const lp of layerProviders) {
      if (lp.syncVisibility) lp.syncVisibility(map)
    }
  }, [map, layerProviders])

  // Init SAM decoder once
  useEffect(() => {
    if (samInitRef.current) return
    samInitRef.current = true
    initSamDecoder().then(() => console.log('SAM decoder ready'))
  }, [])

  // Event handlers — active only when this view is active
  useEffect(() => {
    if (!map || !active) return

    const chip = chipRef.current
    const points = pointsRef.current
    const markers = pointMarkersRef.current

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
    }

    function deselectChip() {
      clearSegmentation()
      removeOverlays()
      chip.id = null
      chip.corners = null
      chip.embedding = null
      setSelectedChipId(null)
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

    function showMask(mask, imageCoords) {
      const dataURL = maskToDataURL(mask)
      if (map.getLayer('mask-overlay')) map.removeLayer('mask-overlay')
      if (map.getSource('mask-overlay')) map.removeSource('mask-overlay')
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

      // Let layer providers add their overlays
      for (const lp of layerProvidersRef.current) {
        lp.onChipSelect(map, chipId, imageCoords)
      }

      chip.id = chipId
      chip.corners = imageCoords
      setSelectedChipId(chipId)

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

      const embedding = await loadNpy(data.embeddingUrl(chipId))
      chip.embedding = embedding
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
    function onChipFillClick(e) {
      const chipId = e.features[0].properties.id
      if (chipId !== chip.id) {
        selectingChipRef.current = true
        selectChip(chipId)
      }
    }

    function onMapClick(e) {
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
        if (points.length > 0) {
          clearSegmentation()
        } else {
          deselectChip()
        }
        return
      }
      if (e.key === 'Escape') {
        e.preventDefault()
        if (paintModeRef.current) {
          paintbrush.setPaintMode(null)
          return
        }
        deselectChip()
        return
      }

      // Paintbrush shortcuts
      if (e.key === 'b' || e.key === 'B') {
        if (!chip.id) return
        paintbrush.setPaintMode((prev) => prev ? null : 'add')
        return
      }
      if (e.key === 'a' || e.key === 'A') {
        if (!chip.id) return
        paintbrush.setPaintMode('add')
        return
      }
      if (e.key === 'e' || e.key === 'E') {
        if (!chip.id) return
        paintbrush.setPaintMode('erase')
        return
      }
      if (e.key === '=' || e.key === '+') {
        paintbrush.adjustBrushSize(2)
        return
      }
      if (e.key === '-' || e.key === '_') {
        paintbrush.adjustBrushSize(-2)
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
        paintbrush.clearCorrections()
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

    return () => {
      map.off('click', 'chips-fill', onChipFillClick)
      map.off('click', onMapClick)
      map.getCanvas().removeEventListener('contextmenu', onContextMenu)
      document.removeEventListener('keydown', handleKeyDown)
      map.off('mousemove', 'chips-fill', onChipMouseMove)
      map.off('mouseleave', 'chips-fill', onChipMouseLeave)
    }
  }, [map, active, featureById, paintbrush])

  // Update the map mask overlay when paintbrush modifies the composited mask
  const handleMaskUpdate = useCallback((composited) => {
    if (!map || !chipRef.current.corners) return
    const dataURL = maskToDataURL(composited)
    if (map.getLayer('mask-overlay')) map.removeLayer('mask-overlay')
    if (map.getSource('mask-overlay')) map.removeSource('mask-overlay')
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
  }, [map])

  // Collect controls from all layer providers
  const pluginControls = layerProviders.flatMap(lp => lp.controls || [])

  // Get the current SAM mask for compositing
  const currentSamMask = maskResults && maskIndex >= 0 ? maskResults.masks[maskIndex] : null
  const chipCorners = chipRef.current.corners

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
  }
}
