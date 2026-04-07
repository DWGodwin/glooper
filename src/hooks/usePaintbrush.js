import { useRef, useState, useCallback } from 'react'
import { SAM_MASK_SIZE } from '../sam'

const PIXELS = SAM_MASK_SIZE * SAM_MASK_SIZE
const ADD = 1
const ERASE = 2
const DEFAULT_BRUSH_SIZE = 15
const MIN_BRUSH = 3
const MAX_BRUSH = 50

export function usePaintbrush() {
  const [paintMode, setPaintMode] = useState(null) // null | 'add' | 'erase'
  const [brushSize, setBrushSize] = useState(DEFAULT_BRUSH_SIZE)
  const correctionsRef = useRef(new Uint8Array(PIXELS))

  const paintAt = useCallback((x, y) => {
    const buf = correctionsRef.current
    const r = brushSize / 2
    const r2 = r * r
    const x0 = Math.max(0, Math.floor(x - r))
    const y0 = Math.max(0, Math.floor(y - r))
    const x1 = Math.min(SAM_MASK_SIZE - 1, Math.ceil(x + r))
    const y1 = Math.min(SAM_MASK_SIZE - 1, Math.ceil(y + r))
    const val = paintMode === 'erase' ? ERASE : ADD
    for (let py = y0; py <= y1; py++) {
      for (let px = x0; px <= x1; px++) {
        const dx = px - x
        const dy = py - y
        if (dx * dx + dy * dy <= r2) {
          buf[py * SAM_MASK_SIZE + px] = val
        }
      }
    }
  }, [brushSize, paintMode])

  const compositeMask = useCallback((samMask) => {
    const buf = correctionsRef.current
    const out = new Float32Array(PIXELS)
    for (let i = 0; i < PIXELS; i++) {
      const sam = samMask ? samMask[i] : 0
      const corr = buf[i]
      if (corr === ERASE) {
        out[i] = 0
      } else if (corr === ADD) {
        out[i] = 1
      } else {
        out[i] = sam
      }
    }
    return out
  }, [])

  const clearCorrections = useCallback(() => {
    correctionsRef.current = new Uint8Array(PIXELS)
  }, [])

  const adjustBrushSize = useCallback((delta) => {
    setBrushSize((prev) => Math.max(MIN_BRUSH, Math.min(MAX_BRUSH, prev + delta)))
  }, [])

  return {
    paintMode,
    setPaintMode,
    brushSize,
    setBrushSize,
    adjustBrushSize,
    correctionsRef,
    paintAt,
    compositeMask,
    clearCorrections,
  }
}
