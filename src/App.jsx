import { useRef, useEffect, useState, useCallback } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import './App.css'
import { initSamDecoder, isDecoderReady, runSamDecoder, maskToDataURL } from './sam'
import { loadNpy } from './npy'
import { BASEMAP_KEYS, makeStyle } from './mapStyle'
import StatusBar from './StatusBar'
import BasemapPicker from './BasemapPicker'
import ViewNav from './ViewNav'
import InfoPanel from './InfoPanel'

function App() {
  const mapContainer = useRef(null)
  const map = useRef(null)
  const initialized = useRef(false)
  const [activeView, setActiveView] = useState('define-area')
  const [activeBasemap, setActiveBasemap] = useState('imagery')
  const [camVisible, setCamVisible] = useState(false)
  const camVisibleRef = useRef(false)
  const [maskResults, setMaskResults] = useState(null)
  const maskResultsRef = useRef(null)
  const [maskIndex, setMaskIndex] = useState(-1)
  const [clickPoints, setClickPoints] = useState([]) // { x, y, label, lon, lat }
  const [selectedChipId, setSelectedChipId] = useState(null)

  useEffect(() => { maskResultsRef.current = maskResults }, [maskResults])

  useEffect(() => {
    if (initialized.current) return
    initialized.current = true

    const m = new maplibregl.Map({
      container: mapContainer.current,
      style: makeStyle(),
      center: [-71.82, 42.25],
      zoom: 17,
    })
    map.current = m

    m.addControl(new maplibregl.NavigationControl(), 'top-right')

    let activeChipId = null
    let activeChipCorners = null   // [TL, TR, BR, BL] in [lon, lat] for image overlay
    let activeEmbedding = null
    let activeCamMask = null      // CAM rescaled to 256*256 logits for SAM mask_input
    let lastLowResMask = null     // 256*256 logit mask from previous SAM run
    let pointMarkers = []         // MapLibre Marker instances
    let points = []               // { x, y, label, lon, lat } — mutable accumulator

    function removeOverlays() {
      for (const id of ['mask-overlay', 'cam-overlay', 'chip-overlay']) {
        if (m.getLayer(id)) m.removeLayer(id)
        if (m.getSource(id)) m.removeSource(id)
      }
    }

    function clearPointMarkers() {
      for (const marker of pointMarkers) marker.remove()
      pointMarkers = []
    }

    function clearSegmentation() {
      if (m.getLayer('mask-overlay')) m.removeLayer('mask-overlay')
      if (m.getSource('mask-overlay')) m.removeSource('mask-overlay')
      clearPointMarkers()
      points = []
      lastLowResMask = null
      setMaskResults(null)
      setMaskIndex(-1)
      setClickPoints([])
    }

    function deselectChip() {
      clearSegmentation()
      removeOverlays()
      activeChipId = null
      activeChipCorners = null
      activeEmbedding = null
      activeCamMask = null
      setSelectedChipId(null)
      // Reset outline styles
      if (m.getLayer('chips-outline')) {
        m.setPaintProperty('chips-outline', 'line-color', [
          'case',
          ['==', ['get', 'label'], 'present'],
          '#22c55e',
          '#ef4444',
        ])
        m.setPaintProperty('chips-outline', 'line-width', 1.5)
      }
    }

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
      // Find min/max for normalization to logit range [-4, 4]
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
          // Normalize to [-4, 4] logit range
          dst[y * 256 + x] = ((val - min) / range) * 8 - 4
        }
      }
      return dst
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
        .addTo(m)
      pointMarkers.push(marker)
    }

    function showMask(mask, imageCoords) {
      const dataURL = maskToDataURL(mask)
      if (m.getLayer('mask-overlay')) m.removeLayer('mask-overlay')
      if (m.getSource('mask-overlay')) m.removeSource('mask-overlay')
      m.addSource('mask-overlay', {
        type: 'image',
        url: dataURL,
        coordinates: imageCoords,
      })
      m.addLayer({
        id: 'mask-overlay',
        type: 'raster',
        source: 'mask-overlay',
      })
    }

    async function handleSegClick(lon, lat, label) {
      if (!activeChipId || !activeEmbedding || !activeChipCorners) return
      if (!isDecoderReady()) return

      const pix = lonLatToPixel(lon, lat, activeChipCorners)
      if (!pix || pix.u < 0 || pix.u > 1 || pix.v < 0 || pix.v > 1) return

      points.push({ x: pix.x, y: pix.y, label, lon, lat })
      addPointMarker(lon, lat, label)
      setClickPoints([...points])

      // Double-rAF: first rAF queues us before paint, second fires after paint completes
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))

      const maskPrior = lastLowResMask || activeCamMask || null

      const { masks, scores, lowResMasks } = await runSamDecoder(activeEmbedding, points, maskPrior)
      setMaskResults({ masks, scores, lowResMasks, imageCoords: activeChipCorners })
      setMaskIndex(0)
      showMask(masks[0], activeChipCorners)

      if (lowResMasks && lowResMasks.length > 0) {
        lastLowResMask = new Float32Array(lowResMasks[0])
      }
    }

    m.on('load', async () => {
      initSamDecoder().then(() => console.log('SAM decoder ready'))

      const resp = await fetch(`${import.meta.env.BASE_URL}data/metadata.geojson`)
      const raw = await resp.json()

      // Index original features by chip ID so we can look up exact coordinates
      // instead of using tile-quantized geometry from click events
      const featureById = {}
      for (const f of raw.features) featureById[f.properties.id] = f

      m.addSource('chips', { type: 'geojson', data: raw })

      m.addLayer({
        id: 'chips-fill',
        type: 'fill',
        source: 'chips',
        paint: {
          'fill-color': [
            'case',
            ['==', ['get', 'label'], 'present'],
            '#22c55e',
            '#ef4444',
          ],
          'fill-opacity': 0.15,
        },
      })

      m.addLayer({
        id: 'chips-outline',
        type: 'line',
        source: 'chips',
        paint: {
          'line-color': [
            'case',
            ['==', ['get', 'label'], 'present'],
            '#22c55e',
            '#ef4444',
          ],
          'line-width': 1.5,
        },
      })

      // Select a chip on click, or place a point if chip is already selected
      let selectingChip = false // guard to prevent click-through to point handler

      async function selectChip(chipId) {
        if (chipId === activeChipId) return

        removeOverlays()
        clearSegmentation()

        // Use original GeoJSON coordinates, not tile-quantized from click event.
        // Vertex order from source: [NE, SE, SW, NW, NE(closing)]
        // MapLibre image coords: [TL, TR, BR, BL] = [NW, NE, SE, SW]
        const coords = featureById[chipId].geometry.coordinates[0]
        const imageCoords = [coords[3], coords[0], coords[1], coords[2]]

        m.addSource('chip-overlay', {
          type: 'image',
          url: `${import.meta.env.BASE_URL}data/chips/${chipId}.png`,
          coordinates: imageCoords,
        })
        m.addLayer({
          id: 'chip-overlay',
          type: 'raster',
          source: 'chip-overlay',
        })

        m.addSource('cam-overlay', {
          type: 'image',
          url: `${import.meta.env.BASE_URL}data/cams/${chipId}.png`,
          coordinates: imageCoords,
        })
        m.addLayer({
          id: 'cam-overlay',
          type: 'raster',
          source: 'cam-overlay',
          paint: { 'raster-opacity': 0.7 },
          layout: { visibility: camVisibleRef.current ? 'visible' : 'none' },
        })

        activeChipId = chipId
        activeChipCorners = imageCoords
        setSelectedChipId(chipId)

        // Zoom to the selected chip
        const lngs = imageCoords.map(c => c[0])
        const lats = imageCoords.map(c => c[1])
        m.fitBounds(
          [[Math.min(...lngs), Math.min(...lats)], [Math.max(...lngs), Math.max(...lats)]],
          { padding: 80, duration: 500 }
        )

        // Highlight the selected chip
        m.setPaintProperty('chips-outline', 'line-color', [
          'case',
          ['==', ['get', 'id'], chipId],
          '#facc15',
          [
            'case',
            ['==', ['get', 'label'], 'present'],
            '#22c55e',
            '#ef4444',
          ],
        ])
        m.setPaintProperty('chips-outline', 'line-width', [
          'case',
          ['==', ['get', 'id'], chipId],
          3,
          1.5,
        ])

        // Load embedding and raw CAM in parallel
        const [embedding, camRaw] = await Promise.all([
          loadNpy(`${import.meta.env.BASE_URL}data/sam_embeddings/${chipId}.npy`),
          loadNpy(`${import.meta.env.BASE_URL}data/cams_raw/${chipId}.npy`).catch(() => null),
        ])
        activeEmbedding = embedding
        activeCamMask = camRaw ? camToMaskInput(camRaw) : null
      }

      m.on('click', 'chips-fill', (e) => {
        const chipId = e.features[0].properties.id
        if (chipId !== activeChipId) {
          // Selecting a new chip — block the general click handler
          selectingChip = true
          selectChip(chipId)
        }
        // If clicking the already-selected chip, let the general handler place a point
      })

      // Left click: positive point (only if not selecting a chip)
      m.on('click', (e) => {
        if (selectingChip) {
          selectingChip = false
          return
        }
        if (!activeChipId) return
        handleSegClick(e.lngLat.lng, e.lngLat.lat, 1)
      })

      // Right click: negative point
      m.getCanvas().addEventListener('contextmenu', (e) => {
        e.preventDefault()
        if (!activeChipId) return
        const rect = m.getCanvas().getBoundingClientRect()
        const point = new maplibregl.Point(e.clientX - rect.left, e.clientY - rect.top)
        const coord = m.unproject(point)
        handleSegClick(coord.lng, coord.lat, 0)
      })

      // Re-run SAM with current points (after point removal)
      async function rerunSam() {
        if (!activeChipId || !activeEmbedding || !activeChipCorners) return
        if (points.length === 0) {
          if (m.getLayer('mask-overlay')) m.removeLayer('mask-overlay')
          if (m.getSource('mask-overlay')) m.removeSource('mask-overlay')
          lastLowResMask = null
          setMaskResults(null)
          setMaskIndex(-1)
          return
        }
        // Let the browser paint updated markers before ONNX inference blocks the thread
        await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))

        const maskPrior = lastLowResMask || activeCamMask || null
        const { masks, scores, lowResMasks } = await runSamDecoder(activeEmbedding, points, maskPrior)
        setMaskResults({ masks, scores, lowResMasks, imageCoords: activeChipCorners })
        setMaskIndex(0)
        showMask(masks[0], activeChipCorners)
        if (lowResMasks && lowResMasks.length > 0) {
          lastLowResMask = new Float32Array(lowResMasks[0])
        }
      }

      // Remove point at index, update markers and re-run SAM
      function removePointAt(idx) {
        if (idx < 0 || idx >= points.length) return
        points.splice(idx, 1)
        // Rebuild markers with updated indices
        clearPointMarkers()
        for (const pt of points) addPointMarker(pt.lon, pt.lat, pt.label)
        setClickPoints([...points])
        rerunSam()
      }

      // Keyboard shortcuts
      function handleKeyDown(e) {
        // Backspace: if points exist, clear them; otherwise deselect chip
        if (e.key === 'Backspace') {
          e.preventDefault()
          if (points.length > 0) {
            clearSegmentation()
          } else {
            deselectChip()
          }
          return
        }

        // Escape: deselect chip entirely
        if (e.key === 'Escape') {
          e.preventDefault()
          deselectChip()
          return
        }

        // z/Z: undo last point
        if (e.key === 'z' || e.key === 'Z') {
          if (points.length > 0) removePointAt(points.length - 1)
          return
        }

        // [ and ] cycle through mask candidates
        const r = maskResultsRef.current
        if (!r || r.masks.length <= 1) return
        if (e.key !== '[' && e.key !== ']') return
        e.preventDefault()
        setMaskIndex((prev) => {
          const idx = e.key === ']'
            ? (prev + 1) % r.masks.length
            : (prev - 1 + r.masks.length) % r.masks.length
          showMask(r.masks[idx], r.imageCoords)
          // Update low-res mask to match selected candidate
          if (r.lowResMasks && r.lowResMasks[idx]) {
            lastLowResMask = new Float32Array(r.lowResMasks[idx])
          }
          return idx
        })
      }
      document.addEventListener('keydown', handleKeyDown)

      m.on('mousemove', 'chips-fill', (e) => {
        const chipId = e.features[0].properties.id
        m.getCanvas().style.cursor = chipId === activeChipId ? 'crosshair' : 'pointer'
      })
      m.on('mouseleave', 'chips-fill', () => {
        m.getCanvas().style.cursor = ''
      })
    })
  }, [])

  const switchBasemap = useCallback(
    (key) => {
      setActiveBasemap(key)
      const m = map.current
      if (!m || !m.isStyleLoaded()) return
      for (const k of BASEMAP_KEYS) {
        m.setLayoutProperty(k, 'visibility', k === key ? 'visible' : 'none')
      }
    },
    []
  )
  
  const toggleCam = useCallback(() => {
    setCamVisible((prev) => {
      const next = !prev
      camVisibleRef.current = next
      const m = map.current
      if (m && m.getLayer('cam-overlay')) {
        m.setLayoutProperty('cam-overlay', 'visibility', next ? 'visible' : 'none')
      }
      return next
    })
  }, [])

  return (
    <div className="map-wrap">
      
      <ViewNav
        activeView={activeView}
        onActiveViewChange={setActiveView}
      />
      
      <div ref={mapContainer} className="map-container" />
      
      {activeView === 'define-area' && <div />}
      {activeView === 'labeling' && 
          <div>
            <StatusBar
              selectedChipId={selectedChipId}
              clickPoints={clickPoints}
              maskIndex={maskIndex}
              maskResults={maskResults}
            />
            <BasemapPicker
              activeBasemap={activeBasemap}
              onBasemapChange={switchBasemap}
              camVisible={camVisible}
              onToggleCam={toggleCam}
            />
          </div>
      }
    </div>
)
}

export default App
