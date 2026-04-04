import { useEffect, useRef, useCallback } from 'react'
import { data } from '../data.js'

export function useChipGrid(map) {
  const featureByIdRef = useRef({})
  const initializedRef = useRef(false)

  const loadChips = useCallback(async () => {
    if (!map) return
    const resp = await fetch(data.chipsUrl())
    const raw = await resp.json()

    const index = {}
    for (const f of raw.features) index[f.properties.id] = f
    featureByIdRef.current = index

    if (map.getSource('chips')) {
      map.getSource('chips').setData(raw)
    }
    return raw
  }, [map])

  useEffect(() => {
    if (!map || initializedRef.current) return
    initializedRef.current = true

    map.addSource('chips', {
      type: 'geojson',
      data: { type: 'FeatureCollection', features: [] },
    })

    map.addLayer({
      id: 'chips-fill',
      type: 'fill',
      source: 'chips',
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

    map.addLayer({
      id: 'chips-outline',
      type: 'line',
      source: 'chips',
      paint: {
        'line-color': [
          'match', ['get', 'split'],
          'train', '#3b82f6',
          'test', '#ef4444',
          'validate', '#f59e0b',
          '#888888',
        ],
        'line-width': 1.5,
      },
    })

    loadChips()
  }, [map, loadChips])

  return { featureById: featureByIdRef, refreshChips: loadChips }
}
