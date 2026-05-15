import { useState } from 'react'

function fmtLoss(v) {
  if (v == null || Number.isNaN(v)) return '—'
  return v.toFixed(4)
}

// Parse a string from the form. Empty → undefined (server uses its default).
// "auto"/"null" are passed through as-is for fields that accept those tokens.
function parseField(raw, kind) {
  const s = String(raw ?? '').trim()
  if (s === '') return undefined
  if (kind === 'int') {
    const n = parseInt(s, 10)
    return Number.isFinite(n) ? n : undefined
  }
  if (kind === 'float') {
    const n = parseFloat(s)
    return Number.isFinite(n) ? n : undefined
  }
  if (kind === 'pos_weight') {
    if (s.toLowerCase() === 'auto') return 'auto'
    if (s.toLowerCase() === 'null' || s.toLowerCase() === 'none') return null
    const n = parseFloat(s)
    return Number.isFinite(n) ? n : undefined
  }
  return s
}

function buildHyperparams(form) {
  const out = {}
  const epochs = parseField(form.epochs, 'int')
  const batchSize = parseField(form.batch_size, 'int')
  const lr = parseField(form.learning_rate, 'float')
  const posW = parseField(form.pos_weight, 'pos_weight')
  if (epochs !== undefined) out.epochs = epochs
  if (batchSize !== undefined) out.batch_size = batchSize
  if (lr !== undefined) out.learning_rate = lr
  if (posW !== undefined) out.pos_weight = posW
  if (form.model_size) out.model_size = form.model_size
  return Object.keys(out).length === 0 ? null : out
}

const EMPTY_FORM = { epochs: '', batch_size: '', learning_rate: '', pos_weight: '', model_size: '' }

function placeholder(d, key) {
  if (!d) return ''
  const v = d[key]
  if (v == null) return 'null'
  return String(v)
}

function TrainingPanel({ training }) {
  const {
    phase,
    isTerminal,
    epoch,
    totalEpochs,
    trainLoss,
    valLoss,
    error,
    errorMessage,
    canceling,
    activeRunId,
    defaults,
    startRun,
    cancelRun,
    clearError,
  } = training

  const [form, setForm] = useState(EMPTY_FORM)
  const [showAdvanced, setShowAdvanced] = useState(false)

  const showRunning = !!activeRunId && !isTerminal
  const showIdle = !activeRunId
  const showDone = !!activeRunId && isTerminal

  const handleStart = () => {
    clearError()
    startRun(buildHyperparams(form))
  }

  const updateField = (key) => (e) => setForm((f) => ({ ...f, [key]: e.target.value }))

  const hyperparamForm = (
    <div className="training-panel-form">
      <button
        type="button"
        className="training-panel-link"
        onClick={() => setShowAdvanced((v) => !v)}
      >
        {showAdvanced ? 'Hide' : 'Show'} hyperparameters
      </button>
      {showAdvanced && (
        <div className="training-panel-fields">
          <label>
            Epochs
            <input
              type="number" min="1" step="1"
              placeholder={placeholder(defaults, 'epochs')}
              value={form.epochs} onChange={updateField('epochs')}
            />
          </label>
          <label>
            Batch size
            <input
              type="number" min="1" step="1"
              placeholder={placeholder(defaults, 'batch_size')}
              value={form.batch_size} onChange={updateField('batch_size')}
            />
          </label>
          <label>
            Learning rate
            <input
              type="number" min="0" step="0.0001"
              placeholder={placeholder(defaults, 'learning_rate')}
              value={form.learning_rate} onChange={updateField('learning_rate')}
            />
          </label>
          <label>
            pos_weight
            <input
              type="text"
              placeholder={placeholder(defaults, 'pos_weight') || 'auto | null | float'}
              value={form.pos_weight} onChange={updateField('pos_weight')}
            />
          </label>
          <label>
            Model size
            <select value={form.model_size} onChange={updateField('model_size')}>
              <option value="">{`default${defaults?.model_size ? ` (${defaults.model_size})` : ''}`}</option>
              <option value="small">small</option>
              <option value="medium">medium</option>
              <option value="large">large</option>
            </select>
          </label>
        </div>
      )}
    </div>
  )

  return (
    <div className="training-panel">
      <div className="training-panel-title">Training</div>

      {error?.kind === 'worker_unavailable' && (
        <div className="training-panel-error">
          Training worker unavailable.{' '}
          <button className="training-panel-link" onClick={handleStart}>
            Retry
          </button>
        </div>
      )}
      {error?.kind === 'in_progress' && (
        <div className="training-panel-error">Run already in progress.</div>
      )}
      {error?.kind === 'http' && (
        <div className="training-panel-error">
          Request failed ({error.status}).{' '}
          <button className="training-panel-link" onClick={clearError}>
            Dismiss
          </button>
        </div>
      )}
      {error?.kind === 'network' && (
        <div className="training-panel-error">
          Network error.{' '}
          <button className="training-panel-link" onClick={handleStart}>
            Retry
          </button>
        </div>
      )}

      {showIdle && (
        <>
          {hyperparamForm}
          <button className="training-panel-primary" onClick={handleStart}>
            Refresh Predictions
          </button>
        </>
      )}

      {showRunning && (
        <>
          <div className="training-panel-status">
            {phase === 'queued' && <>Queued…</>}
            {phase === 'training' && (
              <>
                Epoch {epoch}/{totalEpochs ?? '?'} · train {fmtLoss(trainLoss)} · val {fmtLoss(valLoss)}
              </>
            )}
            {phase === 'predicting' && <>Predicting…</>}
          </div>
          <button
            className="training-panel-secondary"
            onClick={cancelRun}
            disabled={canceling}
          >
            {canceling ? 'Cancelling…' : 'Cancel'}
          </button>
        </>
      )}

      {showDone && (
        <>
          <div className="training-panel-status">
            {phase === 'done' && (
              <>
                Done · {epoch} epoch{epoch === 1 ? '' : 's'} · train {fmtLoss(trainLoss)} · val {fmtLoss(valLoss)}
              </>
            )}
            {phase === 'canceled' && <>Canceled at epoch {epoch}</>}
            {phase === 'failed' && (
              <span className="training-panel-failed">
                Failed: {errorMessage || 'training_failed'}
              </span>
            )}
          </div>
          {hyperparamForm}
          <button className="training-panel-primary" onClick={handleStart}>
            New Run
          </button>
        </>
      )}
    </div>
  )
}

export default TrainingPanel
