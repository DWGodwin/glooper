import { useRef, useEffect, useState, useCallback } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import './App.css'
import proj4 from 'proj4'
import { initSamDecoder, isDecoderReady, runSamDecoder, maskToDataURL } from './sam'
import { loadNpy } from './npy'
import { BASEMAP_KEYS, makeStyle } from './mapStyle'
import StatusBar from './StatusBar'
import BasemapPicker from './BasemapPicker'
import ViewNav from './ViewNav'
import InfoPanel from './InfoPanel'
import DefineArea from './DefineArea'
import { data } from './data.js'

const IS_DEMO = import.meta.env.VITE_DATA_SOURCE !== 'api'

proj4.defs('EPSG:32619', '+proj=utm +zone=19 +datum=WGS84 +units=m +no_defs')

function App() {
  const mapContainer = useRef(null)
  const map = useRef(null)
  const initialized = useRef(false)
  const [activeView, setActiveView] = useState(IS_DEMO ? 'labeling' : 'define-area')
  const [activeBasemap, setActiveBasemap] = useState('imagery')
  const [camVisible, setCamVisible] = useState(false)
  const camVisibleRef = useRef(false)
  const [maskResults, setMaskResults] = useState(null)
  const maskResultsRef = useRef(null)
  const [maskIndex, setMaskIndex] = useState(-1)
  const [clickPoints, setClickPoints] = useState([]) // { x, y, label, lon, lat }
  const [selectedChipId, setSelectedChipId] = useState(null)
  const [activeSplit, setActiveSplit] = useState('train')
  const [drawMode, setDrawMode] = useState(false)
  const drawModeRef = useRef(false)
  const [studyAreas, setStudyAreas] = useState([]) // GeoJSON Features
  const activeSplitRef = useRef('train')

  useEffect(() => { maskResultsRef.current = maskResults }, [maskResults])
  useEffect(() => { drawModeRef.current = drawMode }, [drawMode])
  useEffect(() => { activeSplitRef.current = activeSplit }, [activeSplit])
  useEffect(() => {
    const m = map.current
    if (!m) return
    if (!drawMode) m.getCanvas().style.cursor = ''
  }, [drawMode])

  useEffect(() => {
    const m = map.current
    if (!m || !m.getSource('study-areas')) return
    m.getSource('study-areas').setData({
      type: 'FeatureCollection',
      features: studyAreas,
    })
  }, [studyAreas])

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
    let isDrawing = false
    let drawStart = null          // { lng, lat }

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

      const resp = await fetch(data.chipsUrl())
      const raw = await resp.json()

      // Index original features by chip ID so we can look up exact coordinates
      // instead of using tile-quantized geometry from click events
      const featureById = {}
      for (const f of raw.features) featureById[f.properties.id] = f

      m.addSource('study-areas', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      })
      m.addLayer({
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
      m.addLayer({
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

      m.addSource('draw-rect', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      })
      m.addLayer({
        id: 'draw-rect-fill',
        type: 'fill',
        source: 'draw-rect',
        paint: {
          'fill-color': '#3b82f6',
          'fill-opacity': 0.15,
        },
      })
      m.addLayer({
        id: 'draw-rect-outline',
        type: 'line',
        source: 'draw-rect',
        paint: {
          'line-color': '#3b82f6',
          'line-width': 2,
          'line-dasharray': [4, 2],
        },
      })

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

      const CHIP_SIZE_M = 76.8

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
          url: data.chipImageUrl(chipId),
          coordinates: imageCoords,
        })
        m.addLayer({
          id: 'chip-overlay',
          type: 'raster',
          source: 'chip-overlay',
        })

        m.addSource('cam-overlay', {
          type: 'image',
          url: data.chipCamUrl(chipId),
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
          loadNpy(data.embeddingUrl(chipId)),
          loadNpy(data.chipCamRawUrl(chipId)).catch(() => null),
        ])
        activeEmbedding = embedding
        activeCamMask = camRaw ? camToMaskInput(camRaw) : null
      }

      m.on('click', 'chips-fill', (e) => {
        if (drawModeRef.current) return
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
        if (drawModeRef.current) return
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

      // Draw mode: mousedown starts a rectangle
      m.on('mousedown', (e) => {
        if (!drawModeRef.current) return
        e.preventDefault()
        isDrawing = true
        drawStart = { lng: e.lngLat.lng, lat: e.lngLat.lat }
        m.dragPan.disable()
        m.getCanvas().style.cursor = 'crosshair'
      })

      // Draw mode: mousemove updates the preview rectangle
      m.on('mousemove', (e) => {
        if (drawModeRef.current && !isDrawing) {
          m.getCanvas().style.cursor = 'crosshair'
        }

        if (!isDrawing || !drawStart) return

        const current = e.lngLat
        const coords = [
          [drawStart.lng, drawStart.lat],
          [current.lng, drawStart.lat],
          [current.lng, current.lat],
          [drawStart.lng, current.lat],
          [drawStart.lng, drawStart.lat],
        ]
        m.getSource('draw-rect').setData({
          type: 'FeatureCollection',
          features: [{
            type: 'Feature',
            geometry: { type: 'Polygon', coordinates: [coords] },
            properties: {},
          }],
        })
      })

      // Draw mode: mouseup finalizes the rectangle and computes the chip grid
      m.on('mouseup', (e) => {
        if (!isDrawing || !drawStart) return
        isDrawing = false

        const end = e.lngLat
        const sw = [Math.min(drawStart.lng, end.lng), Math.min(drawStart.lat, end.lat)]
        const ne = [Math.max(drawStart.lng, end.lng), Math.max(drawStart.lat, end.lat)]

        drawStart = null
        m.dragPan.enable()

        // Clear draw preview
        m.getSource('draw-rect').setData({ type: 'FeatureCollection', features: [] })

        // Skip tiny accidental clicks (less than ~5px)
        const swPx = m.project(sw)
        const nePx = m.project(ne)
        if (Math.abs(nePx.x - swPx.x) < 5 && Math.abs(nePx.y - swPx.y) < 5) return

        const split = activeSplitRef.current

        if (IS_DEMO) {
          const gridFeatures = computeGrid(sw, ne, split)
          setStudyAreas((prev) => [...prev, ...gridFeatures])
        } else {
          data.createStudyArea({ sw, ne }, split).then((geojson) => {
            if (geojson.features) {
              setStudyAreas((prev) => [...prev, ...geojson.features])
              // Refresh chips source so new chips appear on the map
              fetch(data.chipsUrl())
                .then((r) => r.json())
                .then((chipData) => {
                  if (m.getSource('chips')) m.getSource('chips').setData(chipData)
                  // Update featureById index
                  for (const f of chipData.features) featureById[f.properties.id] = f
                })
            }
          })
        }

        // Auto-exit draw mode
        setDrawMode(false)
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
        if (drawModeRef.current) return
        const chipId = e.features[0].properties.id
        m.getCanvas().style.cursor = chipId === activeChipId ? 'crosshair' : 'pointer'
      })
      m.on('mouseleave', 'chips-fill', () => {
        if (drawModeRef.current) return
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
      
      {!IS_DEMO &&
        <ViewNav
          activeView={activeView}
          onActiveViewChange={setActiveView}
        />
      }
      
      <div ref={mapContainer} className="map-container" />
      
      {activeView === 'define-area' &&
        <DefineArea
          drawMode={drawMode}
          onToggleDraw={() => setDrawMode((prev) => !prev)}
          activeSplit={activeSplit}
          onSplitChange={setActiveSplit}
        />
      }
      {activeView === 'labeling' &&
          <div>
            {IS_DEMO && <InfoPanel />}
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
