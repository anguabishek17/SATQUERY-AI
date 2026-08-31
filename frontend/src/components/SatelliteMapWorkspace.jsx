import { useState, useRef, useEffect, useCallback } from 'react'
import { MapContainer, TileLayer, Rectangle, useMapEvents, useMap } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import '@geoman-io/leaflet-geoman-free'
import '@geoman-io/leaflet-geoman-free/dist/leaflet-geoman.css'
import { uploadImage } from '../api/client'

function MapFlyTo({ coords }) {
  const map = useMap()
  useEffect(() => {
    if (coords && coords.lat && coords.lon) {
      map.flyTo([coords.lat, coords.lon], 14, { duration: 1.5 })
    }
  }, [coords, map])
  return null
}

function GeomanControls({ active, onAoiComplete }) {
  const map = useMap()

  useEffect(() => {
    if (!map) return

    map.pm.addControls({
      position: 'topleft',
      drawCircleMarker: false,
      drawCircle: false,
      drawPolyline: false,
      drawMarker: false,
      drawPolygon: false,
      drawText: false,
      drawRectangle: true,
      editMode: true,
      dragMode: true,
      cutPolygon: false,
      removalMode: true,
    })

    map.pm.setGlobalOptions({
      pathOptions: {
        color: '#6366F1',
        fillColor: '#6366F1',
        fillOpacity: 0.18,
        weight: 2,
      },
    })

    map.on('pm:create', (e) => {
      const layer = e.layer
      const bounds = layer.getBounds()
      const sw = bounds.getSouthWest()
      const ne = bounds.getNorthEast()

      onAoiComplete?.([sw.lng, sw.lat, ne.lng, ne.lat])
    })

    return () => {
      try {
        map.pm.removeControls()
      } catch (err) {
        // Safe cleanup
      }
    }
  }, [map, onAoiComplete])

  return null
}

function MapCapturer({ triggerCapture, onCaptured }) {
  const map = useMap()
  useEffect(() => {
    if (triggerCapture) {
      const bounds = map.getBounds()
      const sw = bounds.getSouthWest()
      const ne = bounds.getNorthEast()
      const container = map.getContainer()

      // Canvas Snapshot Compositor
      let capturedDataUrl = null
      try {
        const canvas = document.createElement('canvas')
        canvas.width = container.clientWidth || 600
        canvas.height = container.clientHeight || 440
        const ctx = canvas.getContext('2d')

        ctx.fillStyle = '#0B0F19'
        ctx.fillRect(0, 0, canvas.width, canvas.height)

        const tiles = container.querySelectorAll('img.leaflet-tile')
        const parentRect = container.getBoundingClientRect()

        tiles.forEach((img) => {
          try {
            const rect = img.getBoundingClientRect()
            const x = rect.left - parentRect.left
            const y = rect.top - parentRect.top
            ctx.drawImage(img, x, y, rect.width, rect.height)
          } catch (e) {
            // Ignore tile CORS errors
          }
        })
        capturedDataUrl = canvas.toDataURL('image/png')
      } catch (err) {
        // Fallback
      }

      onCaptured?.({
        bounds: [sw.lng, sw.lat, ne.lng, ne.lat],
        center: map.getCenter(),
        zoom: map.getZoom(),
        dataUrl: capturedDataUrl,
      })
    }
  }, [triggerCapture, map, onCaptured])
  return null
}

export default function SatelliteMapWorkspace({
  previewUrl,
  isUploadedImage = false,
  uploadedFilename = null,
  geojsonOverlay,
  boxes,
  aoiBbox,
  onAoiChange,
  physicalMetrics,
  sensorInfo,
  rawResult,
  activeLayers = { trueColor: true, ndvi: false, detection: true },
  searchedCoords,
  onCapturedImage,
}) {
  const [opacity, setOpacity] = useState(0.85)
  const [selectedBuilding, setSelectedBuilding] = useState(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [dragStart, setDragStart] = useState(null)
  const [draftAoi, setDraftAoi] = useState(null)
  const [triggerCapture, setTriggerCapture] = useState(false)
  const [capturedRaster, setCapturedRaster] = useState(null)
  const containerRef = useRef(null)

  const isGeoTiff = sensorInfo?.is_geotiff ?? false
  const features = geojsonOverlay?.features || []
  const hasFootprints = features.length > 0

  const handleCaptureView = () => {
    setTriggerCapture(true)
  }

  const handleViewCaptured = async (capturedInfo) => {
    setTriggerCapture(false)
    setCapturedRaster(capturedInfo)
    setIsDrawing(true)

    // Auto Upload Map Capture Snapshot ONLY IF user has not uploaded a dedicated satellite file
    if (!isUploadedImage && capturedInfo.dataUrl) {
      try {
        const res = await fetch(capturedInfo.dataUrl)
        const blob = await res.blob()
        const file = new File([blob], 'captured_map_extent.png', { type: 'image/png' })
        const uploadRes = await uploadImage(file, 'optical')
        if (uploadRes && uploadRes.file_id) {
          onCapturedImage?.([{ file_id: uploadRes.file_id, modality: 'optical', filename: 'captured_map_extent.png' }])
        }
      } catch (err) {
        // Safe fallback
      }
    }
  }

  // Interactive Live SVG Dragging Handlers
  const handleMouseDown = useCallback((e) => {
    if (!isDrawing || !containerRef.current) return
    const rect = containerRef.current.getBoundingClientRect()
    const normX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
    const normY = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height))
    setDragStart({ x: normX, y: normY })
    setDraftAoi({ x1: normX, y1: normY, x2: normX, y2: normY })
  }, [isDrawing])

  const handleMouseMove = useCallback((e) => {
    if (!isDrawing || !dragStart || !containerRef.current) return
    const rect = containerRef.current.getBoundingClientRect()
    const normX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
    const normY = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height))
    setDraftAoi({
      x1: Math.min(dragStart.x, normX),
      y1: Math.min(dragStart.y, normY),
      x2: Math.max(dragStart.x, normX),
      y2: Math.max(dragStart.y, normY),
    })
  }, [isDrawing, dragStart])

  const handleMouseUp = useCallback(() => {
    if (!isDrawing || !draftAoi) return
    const x1 = draftAoi.x1 * 512
    const x2 = draftAoi.x2 * 512
    const y1 = draftAoi.y1 * 512
    const y2 = draftAoi.y2 * 512

    setDragStart(null)
    setDraftAoi(null)
    setIsDrawing(false)

    if (x2 - x1 > 5 && y2 - y1 > 5) {
      onAoiChange?.([x1, y1, x2, y2])
    }
  }, [isDrawing, draftAoi, onAoiChange])

  function handleExportGeoJSON() {
    if (!geojsonOverlay) return
    const str = JSON.stringify(geojsonOverlay, null, 2)
    const blob = new Blob([str], { type: 'application/geo+json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'satquery_building_footprints.geojson'
    a.click()
    URL.revokeObjectURL(url)
  }

  const activeImageSrc = previewUrl || (!isUploadedImage ? capturedRaster?.dataUrl : null)

  return (
    <div className="space-y-3 select-none font-sans">
      {/* GIS Workspace Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-indigo-500/20 bg-[#131927]/90 px-3.5 py-2 text-xs font-mono text-slate-300 shadow-lg backdrop-blur-md">
        <div className="flex items-center gap-2">
          <span className="flex items-center gap-1.5 font-semibold text-indigo-400">
            <span className={`inline-block h-2 w-2 rounded-full ${isUploadedImage ? 'bg-emerald-400' : 'bg-indigo-500'} animate-pulse`} />
            {isUploadedImage
              ? `🛰️ ${isGeoTiff ? 'GeoTIFF Satellite Layer' : 'Uploaded Satellite Image'}: ${uploadedFilename || 'Original Scene'}`
              : capturedRaster
              ? '📸 Map Viewport Capture'
              : 'World Satellite Map (Esri)'}
          </span>
          <span className="text-[10px] text-slate-500 border-l border-slate-700 pl-2">
            {sensorInfo?.crs || 'EPSG:4326 / WGS84'}
          </span>
        </div>

        {/* Dual AOI Controls */}
        <div className="flex items-center gap-2">
          {!previewUrl && !isUploadedImage && (
            <button
              onClick={handleCaptureView}
              className="rounded-lg bg-indigo-600/20 text-indigo-300 border border-indigo-500/40 px-3 py-1 text-xs hover:bg-indigo-600/30 transition-all font-medium flex items-center gap-1.5 shadow"
            >
              📸 Capture Current View
            </button>
          )}

          <button
            onClick={() => setIsDrawing((d) => !d)}
            className={`rounded-lg px-3.5 py-1 text-xs font-medium transition-all ${
              isDrawing
                ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-500/30 ring-2 ring-indigo-400'
                : 'bg-[#1E293B] text-slate-300 hover:bg-[#283548] hover:text-white border border-slate-700'
            }`}
          >
            {isDrawing ? '✏️ Dragging Rectangle...' : '📐 Draw AOI'}
          </button>

          {aoiBbox && (
            <button
              onClick={() => onAoiChange?.(null)}
              className="rounded-lg bg-rose-500/15 text-rose-400 border border-rose-500/30 px-2.5 py-1 text-xs hover:bg-rose-500/25 transition-colors"
            >
              Clear AOI
            </button>
          )}

          {hasFootprints && (
            <button
              onClick={handleExportGeoJSON}
              className="rounded-lg bg-indigo-500/15 text-indigo-400 border border-indigo-500/30 px-2.5 py-1 text-xs hover:bg-indigo-500/25 transition-colors"
            >
              📥 Export GeoJSON
            </button>
          )}
        </div>

        {/* Vector Opacity Slider */}
        {hasFootprints && activeLayers.detection && (
          <div className="flex items-center gap-2 text-[11px]">
            <span className="text-slate-400">Footprints Opacity:</span>
            <input
              type="range"
              min="0.1"
              max="1"
              step="0.05"
              value={opacity}
              onChange={(e) => setOpacity(Number(e.target.value))}
              className="w-20 accent-indigo-500 cursor-pointer"
            />
          </div>
        )}
      </div>

      {/* Main Viewport Container */}
      <div
        ref={containerRef}
        className={`relative h-[440px] w-full overflow-hidden rounded-2xl border border-indigo-500/20 bg-[#0B0F19] shadow-2xl ${
          isDrawing ? 'cursor-crosshair' : 'cursor-default'
        }`}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
      >
        {/* Image Source Identification Banner */}
        {isUploadedImage && (
          <div className="absolute top-3 left-3 z-30 rounded-lg bg-emerald-950/90 border border-emerald-500/50 px-3 py-1.5 text-[11px] font-mono text-emerald-300 shadow-xl backdrop-blur-md flex items-center gap-2">
            <span className="inline-block h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
            🛰️ Active Analysis File: {uploadedFilename || 'Original Satellite Image'} (High-Res)
          </div>
        )}
        {!isUploadedImage && capturedRaster && (
          <div className="absolute top-3 left-3 z-30 rounded-lg bg-indigo-900/90 border border-indigo-400/50 px-3 py-1.5 text-[11px] font-mono text-indigo-200 shadow-xl backdrop-blur-md flex items-center gap-2">
            <span className="inline-block h-2 w-2 rounded-full bg-indigo-400 animate-pulse" />
            📸 Map Viewport Capture ({capturedRaster.bounds.map((b) => b.toFixed(3)).join(', ')})
          </div>
        )}

        {/* Scenario 1: User uploaded an image or captured view -> Show raster canvas viewer directly */}
        {activeImageSrc ? (
          <div className="relative h-full w-full flex items-center justify-center bg-[#070A11]">
            <img
              src={activeImageSrc}
              alt="Satellite Scene"
              className={`h-full w-full object-cover pointer-events-none transition-opacity duration-300 ${
                activeLayers.trueColor ? 'opacity-100' : 'opacity-30'
              }`}
            />

            {/* Interactive SVG Overlay for Live AOI Dragging & Finalized Box */}
            <svg className="absolute inset-0 w-full h-full pointer-events-none z-30">
              {/* Live Draft Rectangle during Mouse Drag */}
              {draftAoi && (
                <rect
                  x={`${draftAoi.x1 * 100}%`}
                  y={`${draftAoi.y1 * 100}%`}
                  width={`${(draftAoi.x2 - draftAoi.x1) * 100}%`}
                  height={`${(draftAoi.y2 - draftAoi.y1) * 100}%`}
                  fill="rgba(99, 102, 241, 0.22)"
                  stroke="#818CF8"
                  strokeWidth="2.5"
                  strokeDasharray="4 4"
                />
              )}

              {/* Finalized Active AOI Box */}
              {aoiBbox && !draftAoi && (
                <g>
                  <rect
                    x={`${(aoiBbox[0] / 512) * 100}%`}
                    y={`${(aoiBbox[1] / 512) * 100}%`}
                    width={`${((aoiBbox[2] - aoiBbox[0]) / 512) * 100}%`}
                    height={`${((aoiBbox[3] - aoiBbox[1]) / 512) * 100}%`}
                    fill="rgba(99, 102, 241, 0.15)"
                    stroke="#818CF8"
                    strokeWidth="2.5"
                    rx="4"
                  />
                  <rect
                    x={`${(aoiBbox[0] / 512) * 100}%`}
                    y={`${(aoiBbox[1] / 512) * 100 - 4}%`}
                    width="125"
                    height="20"
                    fill="#4F46E5"
                    rx="4"
                  />
                  <text
                    x={`${(aoiBbox[0] / 512) * 100 + 2}%`}
                    y={`${(aoiBbox[1] / 512) * 100 - 1}%`}
                    fill="#FFFFFF"
                    fontSize="9.5"
                    fontWeight="bold"
                    fontFamily="monospace"
                  >
                    SELECTED AOI CROP
                  </text>
                </g>
              )}
            </svg>

            {/* Polygon / BBox Overlay (Visible if activeLayers.detection is true) */}
            {activeLayers.detection &&
              features.map((feat, idx) => {
                const bbox = feat.properties?.pixel_bbox
                if (!bbox) return null
                const [bx1, by1, bx2, by2] = bbox
                const id = feat.properties?.building_id || `BLDG-${idx + 1}`
                const area = feat.properties?.area_sq_m || 0
                const conf = feat.properties?.confidence || 0

                return (
                  <div
                    key={idx}
                    onClick={(e) => {
                      e.stopPropagation()
                      setSelectedBuilding({ id, area, conf, bbox })
                    }}
                    className="absolute border-2 border-rose-500 bg-rose-500/20 hover:bg-rose-500/40 hover:border-white cursor-pointer transition-all z-20 rounded-sm shadow-[0_0_8px_rgba(239,68,68,0.4)]"
                    style={{
                      left: `${(bx1 / 512) * 100}%`,
                      top: `${(by1 / 512) * 100}%`,
                      width: `${((bx2 - bx1) / 512) * 100}%`,
                      height: `${((by2 - by1) / 512) * 100}%`,
                      opacity: opacity,
                    }}
                  />
                )
              })}
          </div>
        ) : (
          /* Scenario 2: No image uploaded -> Show Leaflet World Satellite Map with Leaflet-Geoman */
          <div className="h-full w-full z-10">
            <MapContainer
              center={[13.0827, 80.2707]} // Default Chennai coordinates
              zoom={13}
              scrollWheelZoom={true}
              style={{ height: '100%', width: '100%' }}
              zoomControl={false}
            >
              <TileLayer
                attribution="&copy; Esri World Imagery"
                url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                crossOrigin="anonymous"
              />
              <GeomanControls active={isDrawing} onAoiComplete={(bbox) => onAoiChange?.(bbox)} />
              <MapFlyTo coords={searchedCoords} />
              <MapCapturer triggerCapture={triggerCapture} onCaptured={handleViewCaptured} />
            </MapContainer>
          </div>
        )}

        {/* Selected Building Details Overlay Card */}
        {selectedBuilding && (
          <div className="absolute top-4 right-4 z-40 w-64 rounded-2xl border border-indigo-500/40 bg-[#131927]/95 p-4 text-xs shadow-2xl backdrop-blur-xl font-mono text-slate-200">
            <div className="flex items-center justify-between border-b border-slate-700/80 pb-2">
              <span className="font-bold text-indigo-400 flex items-center gap-1.5">
                🏢 {selectedBuilding.id}
              </span>
              <button
                onClick={() => setSelectedBuilding(null)}
                className="text-slate-400 hover:text-white text-sm transition-colors"
              >
                ✕
              </button>
            </div>
            <div className="mt-3 space-y-2 text-[11px] text-slate-300">
              <div className="flex justify-between">
                <span className="text-slate-400">Footprint Area:</span>
                <strong className="text-white">{selectedBuilding.area.toFixed(1)} m²</strong>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Model Confidence:</span>
                <strong className="text-emerald-400">{(selectedBuilding.conf * 100).toFixed(1)}%</strong>
              </div>
              <div className="pt-2 text-[10px] text-slate-500 border-t border-slate-800">
                Centroid: {selectedBuilding.bbox.map((v) => Math.round(v)).join(', ')}
              </div>
            </div>
          </div>
        )}
      </div>


    </div>
  )
}
