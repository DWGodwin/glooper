export const BASEMAP_KEYS = ['imagery', 'osm', 'massgis']

export function makeStyle() {
  return {
    version: 8,
    sources: {
      imagery: {
        type: 'raster',
        tiles: [
          'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        ],
        tileSize: 256,
        attribution:
          'Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics',
      },
      massgis: {
        type: 'raster',
        tiles: [
          'https://tiles.arcgis.com/tiles/hGdibHYSPO59RG1h/arcgis/rest/services/orthos2023/MapServer/tile/{z}/{y}/{x}',
        ],
        tileSize: 256,
        attribution:
          'Tiles &copy; MASSGIS &mdash; Source: MASSGIS',
      },
      osm: {
        type: 'raster',
        tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
        tileSize: 256,
        attribution: '&copy; OpenStreetMap contributors',
      },
    },
    layers: [
      {
        id: 'imagery',
        type: 'raster',
        source: 'imagery',
        minzoom: 0,
        maxzoom: 18,
        layout: { visibility: 'visible' },
      },
      {
        id: 'massgis',
        type: 'raster',
        source: 'massgis',
        minzoom: 0,
        maxzoom: 20,
        layout: { visibility: 'none' },
      },
      {
        id: 'osm',
        type: 'raster',
        source: 'osm',
        minzoom: 0,
        maxzoom: 18,
        layout: { visibility: 'none' },
      },
    ],
  }
}
