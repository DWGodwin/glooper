import { useEffect, useRef, useState, useCallback } from 'react'
import maplibregl from 'maplibre-gl'
import { initSamDecoder, isDecoderReady, runSamDecoder, maskToDataURL } from '../sam'
import { loadNpy } from '../npy'
import { data } from '../data.js'

// Convert lon/lat click to pixel coords within the chip
// corners: [TL, TR, BR, BL] in [lon, lat]
function lonLatToPixel(lon, lat, corners) {
  const [tl, tr, , bl] = corners
  const u = (lon - tl[0]) / (tr[0] - tl[0])
  const v = (lat - tl[1]) / (bl[1] - tl[1])
  return { u, v, x: u * 512, y: v * 512 }
}

// Rescale a raw CAM (arbitrary length, assumed square) to 256x256 logit-scale
function camToMaskInput(camData) {
  const srcSize = Math.round(Math.sqrt(camData.length))
  const dst = new Float32Array(256 * 256)
  const scaleX = srcSize / 256
  const scaleY = srcSize / 256
  let min = Infinity, max = -Infinity
  for (let i = 0; i < camData.length; i++) {
    if (camData[i] < min) min = camData[i]
    if (camData[i] > max) max = camData[i]
  }
  const range = max - min || 1
  for (let y = 0; y < 256; y++) {
    for (let x = 0; x < 256; x++) {
      const srcX = Math.min(Math.floor(x * scaleX), srcSize - 1)
      const srcY = Math.min(Math.floor(y * scaleY), srcSize - 1)
      const val = camData[srcY * srcSize + srcX]
      dst[y * 256 + x] = ((val - min) / range) * 8 - 4
    }
  }
  return dst
}

export function useLabelingView({ active, map, featureById }) {
  const [selectedChipId, setSelectedChipId] = useState(null)
  const [clickPoints, setClickPoints] = useState([])
  const [maskResults, setMaskResults] = useState(null)
  const [maskIndex, setMaskIndex] = useState(-1)
  const [camVisible, setCamVisible] = useState(false)

  // Mutable handler state (refs)
  const chipRef = useRef({ id: null, corners: null, embedding: null, camMask: null })
  const pointsRef = useRef([])
  const lastLowResMaskRef = useRef(null)
  const pointMarkersRef = useRef([])
  const maskResultsRef = useRef(null)
  const camVisibleRef = useRef(false)
  const selectingChipRef = useRef(false)
  const samInitRef = useRef(false)

  useEffect(() => { maskResultsRef.current = maskResults }, [maskResults])
  useEffect(() => { camVisibleRef.current = camVisible }, [camVisible])

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
      for (const id of ['mask-overlay', 'cam-overlay', 'chip-overlay']) {
        if (map.getLayer(id)) map.removeLayer(id)
        if (map.getSource(id)) map.removeSource(id)
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
      chip.camMask = null
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

    async function handleSegClick(lon, lat, label) {
      if (!chip.id || !chip.embedding || !chip.corners) return
      if (!isDecoderReady()) return

      const pix = lonLatToPixel(lon, lat, chip.corners)
      if (!pix || pix.u < 0 || pix.u > 1 || pix.v < 0 || pix.v > 1) return

      points.push({ x: pix.x, y: pix.y, label, lon, lat })
      addPointMarker(lon, lat, label)
      setClickPoints([...points])

      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))

      const maskPrior = lastLowResMaskRef.current || chip.camMask || null
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
      })

      map.addSource('cam-overlay', {
        type: 'image',
        url: data.chipCamUrl(chipId),
        coordinates: imageCoords,
      })
      map.addLayer({
        id: 'cam-overlay',
        type: 'raster',
        source: 'cam-overlay',
        paint: { 'raster-opacity': 0.7 },
        layout: { visibility: camVisibleRef.current ? 'visible' : 'none' },
      })

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

      const [embedding, camRaw] = await Promise.all([
        loadNpy(data.embeddingUrl(chipId)),
        loadNpy(data.chipCamRawUrl(chipId)).catch(() => null),
      ])
      chip.embedding = embedding
      chip.camMask = camRaw ? camToMaskInput(camRaw) : null
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

      const maskPrior = lastLowResMaskRef.current || chip.camMask || null
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
      handleSegClick(e.lngLat.lng, e.lngLat.lat, 1)
    }

    function onContextMenu(e) {
      e.preventDefault()
      if (!chip.id) return
      const rect = map.getCanvas().getBoundingClientRect()
      const point = new maplibregl.Point(e.clientX - rect.left, e.clientY - rect.top)
      const coord = map.unproject(point)
      handleSegClick(coord.lng, coord.lat, 0)
    }

    function handleKeyDown(e) {
      if (e.key === 'Backspace') {
        e.preventDefault()
        if (points.length > 0) {
          clearSegmentation()
        } else {
          deselectChip()
        }
        return
      }
      if (e.key === 'Escape') {
        e.preventDefault()
        deselectChip()
        return
      }
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
  }, [map, active, featureById])

  const toggleCam = useCallback(() => {
    setCamVisible((prev) => {
      const next = !prev
      camVisibleRef.current = next
      if (map && map.getLayer('cam-overlay')) {
        map.setLayoutProperty('cam-overlay', 'visibility', next ? 'visible' : 'none')
      }
      return next
    })
  }, [map])

  return {
    selectedChipId,
    clickPoints,
    maskResults,
    maskIndex,
    camVisible,
    toggleCam,
  }
}
