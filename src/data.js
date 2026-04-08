const BASE = import.meta.env.VITE_STATIC_BASE ?? import.meta.env.BASE_URL
const API = import.meta.env.VITE_API_BASE || ''
const STATIC = import.meta.env.VITE_DATA_SOURCE !== 'api'

export const data = {
  chipsUrl:      ()   => STATIC ? `${BASE}data/metadata.geojson`        : `${API}/api/chips`,
  chipImageUrl:  (id) => STATIC ? `${BASE}data/chips/${id}.png`          : `${API}/api/chips/${id}/image`,
  embeddingUrl:  (id) => STATIC ? `${BASE}data/sam_embeddings/${id}.npy` : `${API}/api/chips/${id}/sam-embedding`,
  samDecoderUrl: ()   => STATIC ? `${BASE}data/sam_decoder.onnx`         : `${API}/api/models/sam-decoder`,
  labelsUrl:     ()   => `${API}/api/labels`,
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
  vectorizePreview: (chipId, maskPngBase64, labelClass = 'positive', vectorization = null) =>
    fetch(`${API}/api/chips/${chipId}/vectorize-preview`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mask: maskPngBase64, label_class: labelClass, ...(vectorization && { vectorization }) }),
    }).then(r => r.json()),
  getVectorizationConfig: (labelClass = null) =>
    fetch(`${API}/api/config/vectorization${labelClass ? `/${labelClass}` : ''}`).then(r => r.json()),
}
