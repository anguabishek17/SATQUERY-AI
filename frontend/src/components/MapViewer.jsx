import { MapContainer, ImageOverlay, Rectangle, useMap, useMapEvents } from 'react-leaflet'
import { CRS } from 'leaflet'
import { useEffect, useMemo, useState, useCallback } from 'react'

// Uses Leaflet's CRS.Simple (a flat pixel-space CRS, not lat/lng) to pan/zoom
// raster previews. Swap to a real geographic CRS + WMS/XYZ tile source once
// GeoTIFFs are served with their actual georeferencing (rasterio -> COG -> tile server).

function FitOnLoad({ bounds }) {
  const map = useMap()
  useEffect(() => {
    if (bounds) map.fitBounds(bounds, { padding: [10, 10] })
  }, [bounds, map])
  return null
}

// Draws an AOI rectangle via plain mouse events — no extra npm dependency
// (leaflet-draw isn't in package.json). Active only while `drawing` is true;
// disables map dragging for the duration of the drag so it doesn't pan
// instead of drawing.
function AoiDrawer({ drawing, imgHeight, onComplete }) {
  const [start, setStart] = useState(null)
  const [current, setCurrent] = useState(null)

  const map = useMapEvents({
    mousedown(e) {
      if (!drawing) return
      map.dragging.disable()
      setStart(e.latlng)
      setCurrent(e.latlng)
    },
    mousemove(e) {
      if (!drawing || !start) return
      setCurrent(e.latlng)
    },
    mouseup(e) {
      if (!drawing || !start) return
      map.dragging.enable()
      const end = e.latlng
      // Leaflet CRS.Simple: lat behaves like a "y from bottom" axis, lng like x.
      // Flip back to image pixel space (y from top) using the known image height.
      const x1 = Math.min(start.lng, end.lng)
      const x2 = Math.max(start.lng, end.lng)
      const y1 = imgHeight - Math.max(start.lat, end.lat)
      const y2 = imgHeight - Math.min(start.lat, end.lat)
      setStart(null)
      setCurrent(null)
      if (x2 - x1 > 2 && y2 - y1 > 2) {
        onComplete([x1, y1, x2, y2])
      }
    },
  })

  if (!drawing || !start || !current) return null
  const bounds = [
    [Math.min(start.lat, current.lat), Math.min(start.lng, current.lng)],
    [Math.max(start.lat, current.lat), Math.max(start.lng, current.lng)],
  ]
  return <Rectangle bounds={bounds} pathOptions={{ color: '#2FD3A8', weight: 1.5, dashArray: '4 3', fillOpacity: 0.05 }} />
}

export default function MapViewer({ previewUrl, boxes, aoiBbox, onAoiChange }) {
  const [naturalSize, setNaturalSize] = useState(null) // { w, h }
  const [drawing, setDrawing] = useState(false)

  // Load the actual image to get its real pixel dimensions so bounds/overlay
  // math is correct for whatever the user uploaded, instead of assuming a
  // fixed 512x512 canvas.
  useEffect(() => {
    setNaturalSize(null)
    if (!previewUrl) return
    const img = new Image()
    img.onload = () => setNaturalSize({ w: img.naturalWidth, h: img.naturalHeight })
    img.src = previewUrl
  }, [previewUrl])

  const imgBounds = useMemo(() => {
    if (!naturalSize) return [[0, 0], [512, 512]]
    return [[0, 0], [naturalSize.h, naturalSize.w]]
  }, [naturalSize])

  const imgHeight = naturalSize?.h ?? 512

  const rectangles = useMemo(() => {
    if (!boxes?.length) return []
    // Backend returns [x1,y1,x2,y2] in pixel space (y from top); flip y for
    // Leaflet's bottom-left-origin CRS.Simple.
    return boxes.map(([x1, y1, x2, y2]) => [
      [imgHeight - y2, x1],
      [imgHeight - y1, x2],
    ])
  }, [boxes, imgHeight])

  const aoiRectangle = useMemo(() => {
    if (!aoiBbox) return null
    const [x1, y1, x2, y2] = aoiBbox
    return [
      [imgHeight - y2, x1],
      [imgHeight - y1, x2],
    ]
  }, [aoiBbox, imgHeight])

  const handleAoiComplete = useCallback((bbox) => {
    setDrawing(false)
    onAoiChange?.(bbox)
  }, [onAoiChange])

  return (
    <div>
      <div className="mb-2 flex items-center gap-2">
        <button
          onClick={() => setDrawing((d) => !d)}
          disabled={!previewUrl}
          className={`rounded border px-2.5 py-1 text-xs transition-colors disabled:opacity-30 ${
            drawing ? 'border-teal/60 bg-teal/15 text-teal' : 'border-border text-ink-dim hover:text-ink'
          }`}
        >
          {drawing ? 'Click + drag to draw…' : 'Draw AOI'}
        </button>
        {aoiBbox && (
          <button
            onClick={() => onAoiChange?.(null)}
            className="rounded border border-border px-2.5 py-1 text-xs text-ink-dim hover:border-coral/40 hover:text-coral"
          >
            Clear AOI
          </button>
        )}
        {aoiBbox && (
          <span className="font-mono text-[11px] text-ink-dim">
            {Math.round(aoiBbox[2] - aoiBbox[0])}×{Math.round(aoiBbox[3] - aoiBbox[1])} px selected
          </span>
        )}
      </div>

      <div className="h-[420px] overflow-hidden rounded border border-border">
        <MapContainer
          crs={CRS.Simple}
          center={[imgHeight / 2, (naturalSize?.w ?? 512) / 2]}
          zoom={0}
          minZoom={-4}
          maxZoom={4}
          style={{ height: '100%', width: '100%', background: '#0F1413' }}
        >
          {previewUrl && <ImageOverlay url={previewUrl} bounds={imgBounds} />}
          {rectangles.map((bounds, i) => (
            <Rectangle key={i} bounds={bounds} pathOptions={{ color: '#D85A30', weight: 1.5, fillOpacity: 0.08 }} />
          ))}
          {aoiRectangle && (
            <Rectangle bounds={aoiRectangle} pathOptions={{ color: '#2FD3A8', weight: 1.5, fillOpacity: 0.06 }} />
          )}
          <AoiDrawer drawing={drawing} imgHeight={imgHeight} onComplete={handleAoiComplete} />
          <FitOnLoad bounds={imgBounds} />
        </MapContainer>
      </div>
    </div>
  )
}
