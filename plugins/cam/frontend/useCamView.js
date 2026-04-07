import { useState, useCallback } from 'react'
import { camData } from './data'

export function useCamView() {
  const [chipLabels, setChipLabels] = useState({}) // chipId -> 'positive' | 'negative' | null
  const [trainingStatus, setTrainingStatus] = useState('idle')

  const toggleChipLabel = useCallback((chipId) => {
    setChipLabels((prev) => {
      const current = prev[chipId] || null
      const next = current === null ? 'positive'
        : current === 'positive' ? 'negative'
        : null
      return { ...prev, [chipId]: next }
    })
  }, [])

  const triggerTraining = useCallback(async () => {
    setTrainingStatus('running')
    try {
      const result = await camData.triggerCamTraining()
      setTrainingStatus(result.status || 'complete')
    } catch {
      setTrainingStatus('error')
    }
  }, [])

  return {
    chipLabels,
    toggleChipLabel,
    trainingStatus,
    triggerTraining,
  }
}
