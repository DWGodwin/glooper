import * as ort from 'onnxruntime-web'

let session = null

/**
 * Initialize the SAM ONNX decoder session. Call once on app load.
 */
let sessionReady = null

export async function initSamDecoder() {
  if (sessionReady) return sessionReady
  sessionReady = ort.InferenceSession.create(`${import.meta.env.BASE_URL}data/sam_decoder.onnx`)
    .then((s) => {
      session = s
      console.log('SAM decoder inputs:', session.inputNames)
      console.log('SAM decoder outputs:', session.outputNames)
      return session
    })
  return sessionReady
}

export function isDecoderReady() {
  return session !== null
}

/**
 * Run SAM decoder with multiple click points and optional mask prior.
 * Requires the ONNX decoder to be exported with dynamic axes on point_coords/point_labels.
 *
 * @param {Float32Array} embedding - SAM image embedding, shape (1, 256, 64, 64)
 * @param {{ x: number, y: number, label: number }[]} points - Click points in 512x512 pixel coords. label: 1=foreground, 0=background
 * @param {Float32Array|null} maskInput256 - Previous low-res mask (256*256 logits) or CAM prior. Null = no mask prior.
 * @returns {{ masks: Float32Array[], scores: number[], lowResMasks: Float32Array[] }}
 */
export async function runSamDecoder(embedding, points, maskInput256 = null) {
  if (!session) throw new Error('SAM decoder not initialized')

  const imageEmbeddings = new ort.Tensor('float32', embedding, [1, 256, 64, 64])

  // SAM expects coords in its 1024x1024 input space.
  // For a 512x512 chip the scale factor is 1024/512 = 2.
  const scale = 1024 / 512

  // Build point coords and labels arrays. Always include a padding point at the end.
  const numPoints = points.length + 1
  const coordsData = new Float32Array(numPoints * 2)
  const labelsData = new Float32Array(numPoints)
  for (let i = 0; i < points.length; i++) {
    coordsData[i * 2] = points[i].x * scale
    coordsData[i * 2 + 1] = points[i].y * scale
    labelsData[i] = points[i].label
  }
  // Padding point
  coordsData[(numPoints - 1) * 2] = 0
  coordsData[(numPoints - 1) * 2 + 1] = 0
  labelsData[numPoints - 1] = -1

  const pointCoords = new ort.Tensor('float32', coordsData, [1, numPoints, 2])
  const pointLabels = new ort.Tensor('float32', labelsData, [1, numPoints])

  const maskInput = new ort.Tensor(
    'float32',
    maskInput256 || new Float32Array(256 * 256),
    [1, 1, 256, 256]
  )
  const hasMaskInput = new ort.Tensor(
    'float32',
    new Float32Array([maskInput256 ? 1 : 0]),
    [1]
  )

  const origImSize = new ort.Tensor('float32', new Float32Array([512, 512]), [2])

  const feeds = {
    image_embeddings: imageEmbeddings,
    point_coords: pointCoords,
    point_labels: pointLabels,
    mask_input: maskInput,
    has_mask_input: hasMaskInput,
    orig_im_size: origImSize,
  }

  if (session.inputNames.includes('multimask_output')) {
    feeds.multimask_output = new ort.Tensor('bool', new Uint8Array([1]), [1])
  }

  const results = await session.run(feeds)

  const maskData = results.masks.data
  const maskDims = results.masks.dims
  const numMasks = maskDims[1]
  const h = maskDims[2]
  const w = maskDims[3]
  const pixelsPerMask = h * w

  const scores = results.iou_predictions
    ? Array.from(results.iou_predictions.data).slice(0, numMasks)
    : Array.from({ length: numMasks }, () => 0)

  // Build sorted index (best IoU first)
  const indices = Array.from({ length: numMasks }, (_, i) => i)
  indices.sort((a, b) => scores[b] - scores[a])

  const masks = indices.map((idx) => {
    const offset = idx * pixelsPerMask
    const binaryMask = new Float32Array(pixelsPerMask)
    for (let i = 0; i < pixelsPerMask; i++) {
      binaryMask[i] = maskData[offset + i] > 0 ? 1 : 0
    }
    return binaryMask
  })
  const sortedScores = indices.map((idx) => scores[idx])

  // Extract low-res masks (256x256 logits) for iterative refinement
  const lowResMasks = []
  if (results.low_res_masks) {
    const lrData = results.low_res_masks.data
    const lrPixels = 256 * 256
    for (const idx of indices) {
      lowResMasks.push(new Float32Array(lrData.buffer, lrData.byteOffset + idx * lrPixels * 4, lrPixels))
    }
  }

  return { masks, scores: sortedScores, lowResMasks }
}

/**
 * Convert a binary mask (512x512) to a data URL for use as a map image overlay.
 * Mask pixels are rendered as semi-transparent blue.
 */
export function maskToDataURL(mask, width = 512, height = 512) {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  const imageData = ctx.createImageData(width, height)

  for (let i = 0; i < mask.length; i++) {
    const offset = i * 4
    if (mask[i] === 1) {
      imageData.data[offset] = 59      // R
      imageData.data[offset + 1] = 130 // G
      imageData.data[offset + 2] = 246 // B
      imageData.data[offset + 3] = 153 // A (~60%)
    }
    // else: fully transparent (default 0,0,0,0)
  }

  ctx.putImageData(imageData, 0, 0)
  return canvas.toDataURL()
}
