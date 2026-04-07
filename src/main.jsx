import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { initPlugins } from './plugins'

const IS_DEMO = import.meta.env.VITE_DATA_SOURCE !== 'api'
const API = import.meta.env.VITE_API_BASE || ''

async function boot() {
  if (!IS_DEMO) {
    try {
      const res = await fetch(`${API}/api/config/plugins`)
      const plugins = await res.json()
      initPlugins(plugins)
    } catch (e) {
      console.warn('Failed to fetch plugin config, proceeding with no plugins', e)
      initPlugins([])
    }
  }

  createRoot(document.getElementById('root')).render(
    <StrictMode>
      <App />
    </StrictMode>,
  )
}

boot()
