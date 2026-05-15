const BASE = import.meta.env.VITE_STATIC_BASE ?? import.meta.env.BASE_URL
const API = import.meta.env.VITE_API_BASE || ''
const STATIC = import.meta.env.VITE_DATA_SOURCE !== 'api'

export class HttpError extends Error {
  constructor(status, body) {
    super(`HTTP ${status}`)
    this.status = status
    this.body = body
  }
}

async function requestJson(url, init) {
  const res = await fetch(url, init)
  let body = null
  try {
    body = await res.json()
  } catch {
    body = null
  }
  if (!res.ok) throw new HttpError(res.status, body)
  return body
}

export const data = {
  chipsUrl:      ()   => STATIC ? `${BASE}data/metadata.geojson`        : `${API}/api/chips`,
  chipImageUrl:  (id) => STATIC ? `${BASE}data/chips/${id}.png`          : `${API}/api/chips/${id}/image`,
  embeddingUrl:  (id) => STATIC ? `${BASE}data/sam_embeddings/${id}.npy` : `${API}/api/chips/${id}/sam-embedding`,
  samDecoderUrl: ()   => STATIC ? `${BASE}data/sam_decoder.onnx`         : `${API}/api/models/sam-decoder`,
  labelsUrl:     (bbox) => bbox ? `${API}/api/labels?bbox=${bbox}` : `${API}/api/labels`,
  createStudyArea: (bbox, split) => fetch(`${API}/api/study-areas`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ bbox, split }),
  }).then(r => r.json()),
  prefetchStatus: (jobId) => fetch(`${API}/api/prefetch/${jobId}`).then(r => r.json()),
  deleteChips: (ids) => fetch(`${API}/api/chips`, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ids }),
  }).then(r => r.json()),
  saveChipLabel: (chipId, maskPngBase64, labelClass = 'positive') =>
    fetch(`${API}/api/chips/${chipId}/label`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mask: maskPngBase64, label_class: labelClass }),
    }).then(r => r.json()),
  markChipComplete: (chipId, complete = true) =>
    fetch(`${API}/api/chips/${chipId}/complete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ complete }),
    }).then(r => r.json()),
  vectorizePreview: (chipId, maskPngBase64, labelClass = 'positive', vectorization = null) =>
    fetch(`${API}/api/chips/${chipId}/vectorize-preview`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mask: maskPngBase64, label_class: labelClass, ...(vectorization && { vectorization }) }),
    }).then(r => r.json()),
  getVectorizationConfig: (labelClass = null) =>
    fetch(`${API}/api/config/vectorization${labelClass ? `/${labelClass}` : ''}`).then(r => r.json()),
  deleteLabelsByGeometry: ({ point, bbox }) =>
    fetch(`${API}/api/labels/by-geometry`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...(point && { point }), ...(bbox && { bbox }) }),
    }).then(r => r.json()),
  deleteStudyArea: ({ point, bbox }) =>
    fetch(`${API}/api/study-areas`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...(point && { point }), ...(bbox && { bbox }) }),
    }).then(r => r.json()),

  // Training
  getTrainingConfig: () => requestJson(`${API}/api/training/config`),
  startTrainingRun: (hyperparams = null) =>
    requestJson(`${API}/api/training/runs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ hyperparams }),
    }),
  getTrainingRun: (runId) => requestJson(`${API}/api/training/runs/${runId}`),
  listTrainingRuns: () => requestJson(`${API}/api/training/runs`),
  cancelTrainingRun: (runId) =>
    requestJson(`${API}/api/training/runs/${runId}/cancel`, { method: 'POST' }),

  // Predictions
  getPredictions: (runId, bbox) =>
    requestJson(`${API}/api/predictions?run_id=${encodeURIComponent(runId)}&bbox=${bbox}`),
  predictionMaskUrl: (runId, chipId) =>
    `${API}/api/predictions/${encodeURIComponent(runId)}/chips/${encodeURIComponent(chipId)}/mask.png`,
  getPredictionsDiff: (runA, runB, bbox) =>
    requestJson(
      `${API}/api/predictions/diff?run_a=${encodeURIComponent(runA)}` +
      `&run_b=${encodeURIComponent(runB)}&bbox=${bbox}`,
    ),
  predictionDiffMaskUrl: (runA, runB, chipId) =>
    `${API}/api/predictions/diff/${encodeURIComponent(runA)}/${encodeURIComponent(runB)}` +
    `/chips/${encodeURIComponent(chipId)}/mask.png`,
}
