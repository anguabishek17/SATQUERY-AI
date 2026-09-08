import { useState } from 'react'

export default function OpticalSarWorkspace({
  opticalUrl,
  sarUrl,
  result,
  aoiBbox,
  onAoiChange,
}) {
  const [activeTab, setActiveTab] = useState('dual') // 'dual' | 'fusion'
  const [drawing, setDrawing] = useState(false)
  const [dragStart, setDragStart] = useState(null)

  const fusionScore = result?.fusion_agreement_score ?? result?.raw?.confidence
  const metrics = result?.physical_metrics

  function handleMouseDown(e, imgElem) {
    if (!drawing || !imgElem) return
    const rect = imgElem.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    // Normalized 0-1 coords
    const normX = x / rect.width
    const normY = y / rect.height
    setDragStart({ x: normX, y: normY, rectWidth: rect.width, rectHeight: rect.height })
  }

  function handleMouseUp(e, imgElem) {
    if (!drawing || !dragStart || !imgElem) return
    const rect = imgElem.getBoundingClientRect()
    const endX = (e.clientX - rect.left) / rect.width
    const endY = (e.clientY - rect.top) / rect.height

    const x1 = Math.min(dragStart.x, endX) * 512
    const x2 = Math.max(dragStart.x, endX) * 512
    const y1 = Math.min(dragStart.y, endY) * 512
    const y2 = Math.max(dragStart.y, endY) * 512

    setDragStart(null)
    setDrawing(false)
    if (x2 - x1 > 5 && y2 - y1 > 5) {
      onAoiChange?.([x1, y1, x2, y2])
    }
  }

  return (
    <div className="space-y-3">
      {/* Navigation & Mode Tabs */}
      <div className="flex flex-wrap items-center justify-between gap-2 rounded border border-border bg-card p-2 text-xs">
        <div className="flex items-center gap-1 font-mono">
          <button
            onClick={() => setActiveTab('dual')}
            className={`rounded px-3 py-1 transition-colors ${
              activeTab === 'dual' ? 'bg-teal/20 text-teal border border-teal/40' : 'text-ink-dim hover:text-ink'
            }`}
          >
            Synchronized Optical | SAR View
          </button>
          <button
            onClick={() => setActiveTab('fusion')}
            className={`rounded px-3 py-1 transition-colors ${
              activeTab === 'fusion' ? 'bg-teal/20 text-teal border border-teal/40' : 'text-ink-dim hover:text-ink'
            }`}
          >
            Fusion Analysis Dashboard
          </button>
        </div>

        {/* Universal Shared AOI Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setDrawing((d) => !d)}
            className={`rounded px-2.5 py-1 font-mono text-[11px] transition-colors ${
              drawing ? 'bg-teal/20 text-teal border border-teal/60' : 'bg-surface text-ink-dim hover:text-ink border border-border'
            }`}
          >
            {drawing ? 'Click + drag on image...' : 'Draw Shared AOI'}
          </button>
          {aoiBbox && (
            <button
              onClick={() => onAoiChange?.(null)}
              className="rounded bg-coral/10 text-coral px-2 py-1 font-mono text-[11px] hover:bg-coral/20"
            >
              Clear AOI
            </button>
          )}
        </div>


      </div>

      {activeTab === 'dual' ? (
        /* Dual Synchronized Raster Viewports */
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 select-none">
          {/* Optical Viewport */}
          <div className="overflow-hidden rounded border border-border bg-card">
            <div className="flex items-center justify-between border-b border-border bg-surface px-3 py-1.5 text-xs">
              <span className="font-mono text-teal font-medium">OPTICAL RASTER</span>
              <span className="font-mono text-[10px] text-ink-dim">Multispectral (RGB/NIR)</span>
            </div>
            <div
              className="relative h-[380px] w-full bg-surface cursor-crosshair overflow-hidden"
              onMouseDown={(e) => handleMouseDown(e, e.currentTarget)}
              onMouseUp={(e) => handleMouseUp(e, e.currentTarget)}
            >
              {opticalUrl ? (
                <img src={opticalUrl} alt="Optical Imagery" className="h-full w-full object-contain" />
              ) : (
                <div className="flex h-full items-center justify-center text-xs text-ink-dim font-mono">
                  Upload Optical GeoTIFF
                </div>
              )}

              {/* Shared AOI Rectangle Overlay */}
              {aoiBbox && (
                <div
                  className="absolute border-2 border-teal bg-teal/10 pointer-events-none"
                  style={{
                    left: `${(aoiBbox[0] / 512) * 100}%`,
                    top: `${(aoiBbox[1] / 512) * 100}%`,
                    width: `${((aoiBbox[2] - aoiBbox[0]) / 512) * 100}%`,
                    height: `${((aoiBbox[3] - aoiBbox[1]) / 512) * 100}%`,
                  }}
                >
                  <span className="absolute top-0 left-0 bg-teal px-1 text-[9px] text-background font-mono font-bold">
                    SHARED AOI
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* SAR Viewport */}
          <div className="overflow-hidden rounded border border-border bg-card">
            <div className="flex items-center justify-between border-b border-border bg-surface px-3 py-1.5 text-xs">
              <span className="font-mono text-teal font-medium">SAR RASTER</span>
              <span className="font-mono text-[10px] text-ink-dim">Backscatter (VV/VH dB)</span>
            </div>
            <div className="relative h-[380px] w-full bg-surface overflow-hidden">
              {sarUrl ? (
                <img src={sarUrl} alt="SAR Imagery" className="h-full w-full object-contain filter grayscale" />
              ) : (
                <div className="flex h-full items-center justify-center text-xs text-ink-dim font-mono">
                  Upload SAR GeoTIFF
                </div>
              )}

              {/* Synchronized Shared AOI Rectangle Overlay */}
              {aoiBbox && (
                <div
                  className="absolute border-2 border-teal bg-teal/10 pointer-events-none"
                  style={{
                    left: `${(aoiBbox[0] / 512) * 100}%`,
                    top: `${(aoiBbox[1] / 512) * 100}%`,
                    width: `${((aoiBbox[2] - aoiBbox[0]) / 512) * 100}%`,
                    height: `${((aoiBbox[3] - aoiBbox[1]) / 512) * 100}%`,
                  }}
                >
                  <span className="absolute top-0 left-0 bg-teal px-1 text-[9px] text-background font-mono font-bold">
                    SHARED AOI
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      ) : (
        /* Fusion Analysis Dashboard */
        <div className="rounded border border-border bg-card p-4 space-y-4">
          <h4 className="font-mono text-xs uppercase tracking-wide text-teal">Cross-Modal Joint Fusion Metrics</h4>
          
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 font-mono text-xs">
            <div className="rounded border border-border bg-surface p-3 text-center">
              <p className="text-[10px] text-ink-dim uppercase">Water Agreement</p>
              <p className="mt-1 text-lg font-bold text-teal">{metrics?.water_pct ?? 0}%</p>
            </div>
            <div className="rounded border border-border bg-surface p-3 text-center">
              <p className="text-[10px] text-ink-dim uppercase">Built-up Agreement</p>
              <p className="mt-1 text-lg font-bold text-teal">{metrics?.builtup_pct ?? 0}%</p>
            </div>
            <div className="rounded border border-border bg-surface p-3 text-center">
              <p className="text-[10px] text-ink-dim uppercase">Vegetation</p>
              <p className="mt-1 text-lg font-bold text-teal">{metrics?.vegetation_pct ?? 0}%</p>
            </div>
            <div className="rounded border border-border bg-surface p-3 text-center">
              <p className="text-[10px] text-ink-dim uppercase">Disagreement</p>
              <p className="mt-1 text-lg font-bold text-coral">{metrics?.disagreement_pct ?? 0}%</p>
            </div>
          </div>

          {result?.output_text && (
            <div className="rounded border border-teal/30 bg-teal/5 p-3 text-xs text-ink">
              <p className="font-mono text-[10px] text-teal uppercase mb-1">Evidence Summary</p>
              <p>{result.output_text}</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
