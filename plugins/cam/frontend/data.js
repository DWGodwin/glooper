const BASE = import.meta.env.VITE_STATIC_BASE ?? import.meta.env.BASE_URL
const API = import.meta.env.VITE_API_BASE || ''
const STATIC = import.meta.env.VITE_DATA_SOURCE !== 'api'

export const camData = {
  camOverlayUrl: (id) => STATIC ? `${BASE}data/cams/${id}.png`     : `${API}/api/plugins/cam/chips/${id}/overlay`,
  camRawUrl:     (id) => STATIC ? `${BASE}data/cams_raw/${id}.npy` : `${API}/api/plugins/cam/chips/${id}/raw`,
  triggerCamTraining: () => fetch(`${API}/api/plugins/cam/train`, { method: 'POST' }).then(r => r.json()),
}
