import { useCamView } from './useCamView'

function CamView() {
  const { chipLabels, toggleChipLabel, trainingStatus, triggerTraining } = useCamView()

  const positiveCount = Object.values(chipLabels).filter(v => v === 'positive').length
  const negativeCount = Object.values(chipLabels).filter(v => v === 'negative').length

  return (
    <div className="cam-view">
      <div className="cam-view-header">
        <h3>CAM Training</h3>
        <p>Click chips on the map to toggle: positive / negative / unlabeled</p>
        <p>{positiveCount} positive, {negativeCount} negative</p>
        <button
          onClick={triggerTraining}
          disabled={trainingStatus === 'running' || (positiveCount === 0 && negativeCount === 0)}
        >
          {trainingStatus === 'running' ? 'Training...' : 'Train Classifier'}
        </button>
        {trainingStatus === 'error' && <p className="error">Training failed</p>}
        {trainingStatus === 'complete' && <p>Training complete</p>}
      </div>
    </div>
  )
}

export default CamView
