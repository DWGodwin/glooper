import { useRef, useEffect, useCallback } from 'react'
import { SAM_MASK_SIZE, maskToDataURL } from './sam'

const PAINT_SOURCE = 'paintbrush-overlay'
const PAINT_LAYER = 'paintbrush-overlay'

export default function PaintbrushOverlay({
  map,
  chipCorners,
  paintMode,
  brushSize,
  paintAt,
  compositeMask,
  samMask,
  onMaskUpdate,
}) {
  const paintingRef = useRef(false)
  const rafRef = useRef(null)
  const dirtyRef = useRef(false)
  const inFlightRef = useRef(false)

  // Convert screen coords to SAM_MASK_SIZE pixel coords using all four
  // projected chip corners. Uses inverse bilinear interpolation so that
  // rotation / shear from the map projection is handled correctly.
  const screenToCanvas = useCallback((clientX, clientY) => {
    if (!map || !chipCorners) return null
    const rect = map.getCanvas().getBoundingClientRect()
    // chipCorners: [TL(NW), TR(NE), BR(SE), BL(SW)]
    const p0 = map.project(chipCorners[0]) // TL
    const p1 = map.project(chipCorners[1]) // TR
    const p2 = map.project(chipCorners[3]) // BL
    const p3 = map.project(chipCorners[2]) // BR

    const px = clientX - rect.left
    const py = clientY - rect.top

    // Solve for (u,v) where:
    //   P = (1-u)(1-v)*p0 + u(1-v)*p1 + (1-u)v*p2 + uv*p3
    // Rearrange: P = A + B*u + C*v + D*u*v
    const ax = p0.x, ay = p0.y
    const bx = p1.x - p0.x, by = p1.y - p0.y
    const cx = p2.x - p0.x, cy = p2.y - p0.y
    const dx = p3.x - p1.x - p2.x + p0.x, dy = p3.y - p1.y - p2.y + p0.y

    const ex = px - ax, ey = py - ay

    // Quadratic in v: (d×c)v² + (b×c + d×e - ?)v - b×e = 0
    // Using cross products (2D): a×b = ax*by - ay*bx
    const cross = (ux, uy, vx, vy) => ux * vy - uy * vx

    const A = cross(dx, dy, cx, cy)
    const B = cross(bx, by, cx, cy) + cross(ex, ey, dx, dy)
    const C = cross(ex, ey, bx, by)

    // Solve Av² + Bv + C = 0  (note sign: we want  A·v² - B·v + C = 0
    // rewritten as  A·v² + B·v + C = 0 with the signs as derived)
    // Actually: from P = A + Bu + Cv + Duv  ⟹  u = (E - Cv) / (B + Dv)
    // substituting into the other component gives a quadratic in v.
    let u, v
    if (Math.abs(A) < 1e-6) {
      // Linear case (no shear)
      v = B !== 0 ? -C / B : 0
    } else {
      const disc = B * B - 4 * A * C
      if (disc < 0) return null
      const sq = Math.sqrt(disc)
      const v1 = (-B + sq) / (2 * A)
      const v2 = (-B - sq) / (2 * A)
      v = (v1 >= -0.01 && v1 <= 1.01) ? v1 : v2
    }

    const denom = bx + dx * v
    if (Math.abs(denom) > 1e-6) {
      u = (ex - cx * v) / denom
    } else {
      const denomY = by + dy * v
      u = denomY !== 0 ? (ey - cy * v) / denomY : 0
    }

    return {
      x: u * SAM_MASK_SIZE,
      y: v * SAM_MASK_SIZE,
      inBounds: u >= -0.01 && u <= 1.01 && v >= -0.01 && v <= 1.01,
    }
  }, [map, chipCorners])

  // Update the MapLibre image source with current composited mask
  const updateOverlay = useCallback(async () => {
    if (!map || !chipCorners) return
    if (inFlightRef.current) return
    inFlightRef.current = true
    try {
      const composited = compositeMask(samMask)
      const dataURL = await maskToDataURL(composited)

      const src = map.getSource(PAINT_SOURCE)
      if (src) {
        src.updateImage({ url: dataURL, coordinates: chipCorners })
      } else {
        map.addSource(PAINT_SOURCE, {
          type: 'image',
          url: dataURL,
          coordinates: chipCorners,
        })
        map.addLayer({
          id: PAINT_LAYER,
          type: 'raster',
          source: PAINT_SOURCE,
        })
      }
    } finally {
      inFlightRef.current = false
    }
  }, [map, chipCorners, compositeMask, samMask])

  // Clean up the MapLibre source/layer when paint mode deactivates or chip changes
  useEffect(() => {
    if (!map) return
    return () => {
      if (map.getLayer(PAINT_LAYER)) map.removeLayer(PAINT_LAYER)
      if (map.getSource(PAINT_SOURCE)) map.removeSource(PAINT_SOURCE)
    }
  }, [map, chipCorners, paintMode])

  // Show overlay immediately when entering paint mode
  useEffect(() => {
    if (!map || !chipCorners || !paintMode) return
    updateOverlay()
  }, [map, chipCorners, paintMode, updateOverlay])

  // Animation frame loop: batch paint strokes into image source updates
  useEffect(() => {
    if (!paintMode) return
    const tick = () => {
      if (dirtyRef.current) {
        updateOverlay()
        dirtyRef.current = false
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [paintMode, updateOverlay])

  // Mouse handlers on the map canvas
  useEffect(() => {
    if (!map || !paintMode || !chipCorners) return

    const canvas = map.getCanvas()

    // Brush cursor
    const cursorSize = brushSize
    const cursorSvg = `url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='${cursorSize}' height='${cursorSize}'><circle cx='${cursorSize / 2}' cy='${cursorSize / 2}' r='${cursorSize / 2 - 1}' fill='none' stroke='white' stroke-width='1.5'/></svg>") ${cursorSize / 2} ${cursorSize / 2}, crosshair`
    const prevCursor = canvas.style.cursor
    canvas.style.cursor = cursorSvg

    function onMouseDown(e) {
      // Only handle left button
      if (e.button !== 0) return
      e.preventDefault()
      e.stopPropagation()
      paintingRef.current = true
      map.dragPan.disable()
      const pt = screenToCanvas(e.clientX, e.clientY)
      if (pt && pt.inBounds) {
        paintAt(pt.x, pt.y)
        dirtyRef.current = true
      }
    }

    function onMouseMove(e) {
      if (!paintingRef.current) return
      e.preventDefault()
      e.stopPropagation()
      const pt = screenToCanvas(e.clientX, e.clientY)
      if (pt && pt.inBounds) {
        paintAt(pt.x, pt.y)
        dirtyRef.current = true
      }
    }

    function onMouseUp(e) {
      if (!paintingRef.current) return
      e.preventDefault()
      e.stopPropagation()
      paintingRef.current = false
      map.dragPan.enable()
      // Flush final render and notify parent
      updateOverlay()
      if (onMaskUpdate) {
        onMaskUpdate(compositeMask(samMask))
      }
    }

    canvas.addEventListener('mousedown', onMouseDown)
    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)

    return () => {
      canvas.removeEventListener('mousedown', onMouseDown)
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
      canvas.style.cursor = prevCursor
      if (paintingRef.current) {
        paintingRef.current = false
        map.dragPan.enable()
      }
    }
  }, [map, paintMode, chipCorners, brushSize, screenToCanvas, paintAt, updateOverlay, onMaskUpdate, compositeMask, samMask])

  // No DOM element needed — everything renders through MapLibre
  return null
}
