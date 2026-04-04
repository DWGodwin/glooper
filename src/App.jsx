import { useState } from 'react'
import './App.css'
import StatusBar from './StatusBar'
import BasemapPicker from './BasemapPicker'
import ViewNav from './ViewNav'
import InfoPanel from './InfoPanel'
import DefineArea from './DefineArea'
import { MapProvider, useMap } from './MapContext'
import { useChipGrid } from './hooks/useChipGrid'
import { useDefineAreaView } from './views/useDefineAreaView'
import { useLabelingView } from './views/useLabelingView'

const IS_DEMO = import.meta.env.VITE_DATA_SOURCE !== 'api'

function AppInner() {
  const [activeView, setActiveView] = useState(IS_DEMO ? 'labeling' : 'define-area')
  const { map, activeBasemap, switchBasemap } = useMap()
  const chipGrid = useChipGrid(map)
  const defineArea = useDefineAreaView({ active: activeView === 'define-area', map, chipGrid })
  const labeling = useLabelingView({ active: activeView === 'labeling', map, featureById: chipGrid.featureById })

  return (
    <>
      {!IS_DEMO &&
        <ViewNav
          activeView={activeView}
          onActiveViewChange={setActiveView}
        />
      }

      {activeView === 'define-area' &&
        <DefineArea
          drawMode={defineArea.drawMode}
          onToggleDraw={defineArea.toggleDraw}
          activeSplit={defineArea.activeSplit}
          onSplitChange={defineArea.setActiveSplit}
        />
      }
      {activeView === 'labeling' &&
        <div>
          {IS_DEMO && <InfoPanel />}
          <StatusBar
            selectedChipId={labeling.selectedChipId}
            clickPoints={labeling.clickPoints}
            maskIndex={labeling.maskIndex}
            maskResults={labeling.maskResults}
          />
          <BasemapPicker
            activeBasemap={activeBasemap}
            onBasemapChange={switchBasemap}
            camVisible={labeling.camVisible}
            onToggleCam={labeling.toggleCam}
          />
        </div>
      }
    </>
  )
}

function App() {
  return (
    <div className="map-wrap">
      <MapProvider>
        <AppInner />
      </MapProvider>
    </div>
  )
}

export default App
