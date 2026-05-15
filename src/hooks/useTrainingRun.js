import { useCallback, useEffect, useRef, useState } from 'react'
import { data, HttpError } from '../data.js'

const TERMINAL = new Set(['done', 'failed', 'canceled'])
const POLL_MS = 1500

function lastOf(arr) {
  return Array.isArray(arr) && arr.length > 0 ? arr[arr.length - 1] : null
}

export function useTrainingRun() {
  const [activeRunId, setActiveRunId] = useState(null)
  const [run, setRun] = useState(null)
  const [currentRunId, setCurrentRunId] = useState(null)
  const [prevRunId, setPrevRunId] = useState(null)
  const [error, setError] = useState(null)
  const [canceling, setCanceling] = useState(false)
  const [defaults, setDefaults] = useState(null)

  const lastCompletedRef = useRef(null)
  const onCompleteCbRef = useRef(null)

  useEffect(() => {
    let cancelled = false
    data.getTrainingConfig()
      .then((cfg) => { if (!cancelled) setDefaults(cfg) })
      .catch((e) => console.error('Failed to fetch training config:', e))
    return () => { cancelled = true }
  }, [])

  const refreshCompletedRuns = useCallback(async () => {
    try {
      const { runs } = await data.listTrainingRuns()
      const done = runs.filter((r) => r.status === 'done')
      setCurrentRunId(done[0]?.id ?? null)
      setPrevRunId(done[1]?.id ?? null)
    } catch (e) {
      console.error('Failed to list training runs:', e)
    }
  }, [])

  // Adopt any in-progress run on mount.
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const { runs } = await data.listTrainingRuns()
        if (cancelled) return
        const active = runs.find((r) => !TERMINAL.has(r.status))
        if (active) {
          setActiveRunId(active.id)
          setRun(active)
        }
        const done = runs.filter((r) => r.status === 'done')
        setCurrentRunId(done[0]?.id ?? null)
        setPrevRunId(done[1]?.id ?? null)
        lastCompletedRef.current = done[0]?.id ?? null
      } catch (e) {
        console.error('Failed to list training runs:', e)
      }
    })()
    return () => { cancelled = true }
  }, [])

  // Poll the active run until it reaches a terminal state.
  useEffect(() => {
    if (!activeRunId) return
    let cancelled = false
    let timer = null

    async function tick() {
      try {
        const r = await data.getTrainingRun(activeRunId)
        if (cancelled) return
        setRun(r)
        if (TERMINAL.has(r.status)) {
          setCanceling(false)
          await refreshCompletedRuns()
          if (cancelled) return
          // Fire completion callback if a brand-new 'done' run just appeared.
          if (r.status === 'done' && r.id !== lastCompletedRef.current) {
            lastCompletedRef.current = r.id
            onCompleteCbRef.current?.(r.id)
          }
          return
        }
        timer = setTimeout(tick, POLL_MS)
      } catch (e) {
        console.error('Polling training run failed:', e)
        if (!cancelled) timer = setTimeout(tick, POLL_MS)
      }
    }

    tick()
    return () => {
      cancelled = true
      if (timer) clearTimeout(timer)
    }
  }, [activeRunId, refreshCompletedRuns])

  const startRun = useCallback(async (hyperparams = null) => {
    setError(null)
    try {
      const resp = await data.startTrainingRun(hyperparams)
      setActiveRunId(resp.run_id)
      setRun({ id: resp.run_id, status: resp.status || 'queued' })
      return { ok: true, runId: resp.run_id }
    } catch (e) {
      if (e instanceof HttpError) {
        if (e.status === 409) {
          // Server says another run is active; adopt it.
          const activeId = e.body?.detail?.active_run_id
          if (activeId) {
            setActiveRunId(activeId)
          }
          setError({ kind: 'in_progress', activeRunId: activeId ?? null })
          return { ok: false, kind: 'in_progress' }
        }
        if (e.status === 503) {
          setError({ kind: 'worker_unavailable' })
          return { ok: false, kind: 'worker_unavailable' }
        }
        setError({ kind: 'http', status: e.status, body: e.body })
        return { ok: false, kind: 'http', status: e.status }
      }
      setError({ kind: 'network', message: String(e) })
      return { ok: false, kind: 'network' }
    }
  }, [])

  const cancelRun = useCallback(async () => {
    if (!activeRunId) return
    setCanceling(true)
    try {
      await data.cancelTrainingRun(activeRunId)
    } catch (e) {
      console.error('Cancel failed:', e)
      setCanceling(false)
    }
  }, [activeRunId])

  const clearError = useCallback(() => setError(null), [])

  const onComplete = useCallback((cb) => {
    onCompleteCbRef.current = cb
  }, [])

  const phase = run?.status ?? 'idle'
  const isTerminal = TERMINAL.has(phase)
  const trainCurve = run?.train_loss_curve ?? []
  const valCurve = run?.val_loss_curve ?? []

  return {
    activeRunId,
    run,
    phase,
    isTerminal,
    epoch: trainCurve.length,
    totalEpochs: run?.hyperparams?.epochs ?? null,
    trainLoss: lastOf(trainCurve),
    valLoss: lastOf(valCurve),
    errorMessage: run?.error_message ?? null,
    error,
    canceling,
    currentRunId,
    prevRunId,
    defaults,
    startRun,
    cancelRun,
    clearError,
    onComplete,
  }
}
