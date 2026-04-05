const BASE = import.meta.env.BASE_URL
const API = import.meta.env.VITE_API_BASE || ''
const STATIC = import.meta.env.VITE_DATA_SOURCE !== 'api'

export const data = {
  chipsUrl:      ()   => STATIC ? `${BASE}data/metadata.geojson`        : `${API}/api/chips`,
  chipImageUrl:  (id) => STATIC ? `${BASE}data/chips/${id}.png`          : `${API}/api/chips/${id}/image`,
  chipCamUrl:    (id) => STATIC ? `${BASE}data/cams/${id}.png`           : `${API}/api/chips/${id}/cam`,
  chipCamRawUrl: (id) => STATIC ? `${BASE}data/cams_raw/${id}.npy`       : `${API}/api/chips/${id}/cam-raw`,
  embeddingUrl:  (id) => STATIC ? `${BASE}data/sam_embeddings/${id}.npy` : `${API}/api/chips/${id}/embedding`,
  samDecoderUrl: ()   => STATIC ? `${BASE}data/sam_decoder.onnx`         : `${API}/api/models/sam-decoder`,
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
}
