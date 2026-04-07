import { useCamLabelingLayers } from '@plugins/cam/frontend/useCamLabelingLayers'
import CamView from '@plugins/cam/frontend/CamView'

const IS_DEMO = import.meta.env.VITE_DATA_SOURCE !== 'api'

// Static plugin registry — each known plugin is listed here, gated by config.
// Plugins are disabled in demo mode.
const PLUGIN_REGISTRY = {
  dino: {},
  cam: {
    useLabelingLayers: useCamLabelingLayers,
    views: [{ id: 'cam-training', label: 'CAM Training', Component: CamView }],
  },
}

let _enabledPlugins = null

export function initPlugins(pluginList) {
  _enabledPlugins = pluginList
}

function isEnabled(name) {
  if (IS_DEMO) return false
  return (_enabledPlugins || []).includes(name)
}

// Calls all plugin labeling-layer hooks. The set of hooks called is fixed after
// initPlugins() runs (before first render), so hook count is stable across renders.
// Each hook is called unconditionally but returns a no-op provider when disabled.
export function usePluginLabelingLayers() {
  const cam = useCamLabelingLayers()
  const layers = []
  if (isEnabled('cam')) layers.push(cam)
  return layers
}

export function getPluginViews() {
  const views = []
  for (const [name, reg] of Object.entries(PLUGIN_REGISTRY)) {
    if (isEnabled(name) && reg.views) {
      views.push(...reg.views)
    }
  }
  return views
}
