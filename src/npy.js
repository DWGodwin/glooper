/**
 * Minimal .npy file parser. Returns a Float32Array of the raw data.
 * Only supports float32 little-endian (dtype '<f4'), which is all
 * the pipeline produces.
 */
export async function loadNpy(url) {
  const resp = await fetch(url)
  if (!resp.ok) throw new Error(`Failed to load ${url}: ${resp.status}`)
  const buf = await resp.arrayBuffer()
  const view = new DataView(buf)

  // .npy format: 6-byte magic, 2-byte version, then header
  const headerLen = view.getUint16(8, true) // little-endian
  const headerStart = 10
  const dataStart = headerStart + headerLen

  return new Float32Array(buf, dataStart)
}
