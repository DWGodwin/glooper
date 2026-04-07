import { useState, useRef, useCallback } from 'react'
import { loadNpy } from '@src/npy'
import { camData } from './data'

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

export function useCamLabelingLayers() {
  const [camVisible, setCamVisible] = useState(false)
  const camVisibleRef = useRef(false)
  const camMaskRef = useRef(null)

  const onChipSelect = useCallback(async (map, chipId, imageCoords) => {
    // Add CAM overlay layer
    map.addSource('cam-overlay', {
      type: 'image',
      url: camData.camOverlayUrl(chipId),
      coordinates: imageCoords,
    })
    map.addLayer({
      id: 'cam-overlay',
      type: 'raster',
      source: 'cam-overlay',
      paint: { 'raster-opacity': 0.7 },
      layout: { visibility: camVisibleRef.current ? 'visible' : 'none' },
    })

    // Load raw CAM for mask prior
    try {
      const camRaw = await loadNpy(camData.camRawUrl(chipId))
      camMaskRef.current = camToMaskInput(camRaw)
    } catch {
      camMaskRef.current = null
    }
  }, [])

  const onChipDeselect = useCallback((map) => {
    if (map.getLayer('cam-overlay')) map.removeLayer('cam-overlay')
    if (map.getSource('cam-overlay')) map.removeSource('cam-overlay')
    camMaskRef.current = null
  }, [])

  const toggleCam = useCallback(() => {
    setCamVisible((prev) => {
      const next = !prev
      camVisibleRef.current = next
      return next
    })
  }, [])

  // Apply visibility changes to existing map layer
  const syncVisibility = useCallback((map) => {
    if (map && map.getLayer('cam-overlay')) {
      map.setLayoutProperty('cam-overlay', 'visibility', camVisible ? 'visible' : 'none')
    }
  }, [camVisible])

  return {
    onChipSelect,
    onChipDeselect,
    get maskPrior() { return camMaskRef.current },
    controls: [{ label: 'Toggle Class Activation Map', active: camVisible, onToggle: toggleCam }],
    syncVisibility,
  }
}
