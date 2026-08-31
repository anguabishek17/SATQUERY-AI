import { useState } from 'react'

export default function BiTemporalWorkspace({
  t0Url,
  t1Url,
  result,
  aoiBbox,
  onAoiChange,
}) {
  const [sliderPos, setSliderPos] = useState(50)
  const [viewMode, setViewMode] = useState('swipe') // 'swipe' | 'side-by-side'
  const [drawing, setDrawing] = useState(false)
  const [dragStart, setDragStart] = useState(null)

  const changeStats = result?.change_stats || result?.physical_metrics

  function handleMouseDown(e, containerElem) {
    if (!drawing || !containerElem) return
    const rect = containerElem.getBoundingClientRect()
    const x = (e.clientX - rect.left) / rect.width
    const y = (e.clientY - rect.top) / rect.height
    setDragStart({ x, y })
  }

  function handleMouseUp(e, containerElem) {
    if (!drawing || !dragStart || !containerElem) return
    const rect = containerElem.getBoundingClientRect()
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
    <div className="space-y-3 select-none">
      {/* Control Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 rounded border border-border bg-card p-2 text-xs font-mono">
        <div className="flex items-center gap-1">
          <button
            onClick={() => setViewMode('swipe')}
            className={`rounded px-3 py-1 transition-colors ${
              viewMode === 'swipe' ? 'bg-teal/20 text-teal border border-teal/40' : 'text-ink-dim hover:text-ink'
            }`}
          >
            Swipe Viewer (T0 vs T1)
          </button>
          <button
            onClick={() => setViewMode('side-by-side')}
            className={`rounded px-3 py-1 transition-colors ${
              viewMode === 'side-by-side' ? 'bg-teal/20 text-teal border border-teal/40' : 'text-ink-dim hover:text-ink'
            }`}
          >
            Side-by-Side
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
            {drawing ? 'Click + drag to draw AOI...' : 'Draw Shared AOI'}
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

        {changeStats && (
          <div className="flex items-center gap-3 text-[11px] text-ink-dim">
            <span>Changed: <strong className="text-teal">{changeStats.pct_changed}%</strong></span>
            <span>Area: <strong className="text-teal">{changeStats.changed_area_ha ?? changeStats.changed_ha} ha</strong></span>
          </div>
        )}
      </div>

      {/* Swipe Comparison Viewer */}
      {viewMode === 'swipe' ? (
        <div
          className="relative h-[420px] w-full overflow-hidden rounded border border-border bg-surface cursor-crosshair"
          onMouseDown={(e) => handleMouseDown(e, e.currentTarget)}
          onMouseUp={(e) => handleMouseUp(e, e.currentTarget)}
        >
          {/* T1 (After) Image */}
          {t1Url && (
            <img src={t1Url} alt="T1 Satellite Imagery" className="absolute inset-0 h-full w-full object-contain pointer-events-none" />
          )}

          {/* T0 (Before) Clipped Overlay Image */}
          {t0Url && (
            <div
              className="absolute inset-0 overflow-hidden pointer-events-none"
              style={{ width: `${sliderPos}%` }}
            >
              <img src={t0Url} alt="T0 Satellite Imagery" className="h-full w-full max-w-none object-contain" />
            </div>
          )}

          {/* Shared AOI Box Overlay */}
          {aoiBbox && (
            <div
              className="absolute border-2 border-teal bg-teal/10 pointer-events-none z-20"
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

          {/* Vertical Swipe Divider Handle */}
          <div
            className="absolute top-0 bottom-0 z-30 w-1 bg-teal cursor-ew-resize shadow-lg"
            style={{ left: `${sliderPos}%` }}
          >
            <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 flex h-8 w-8 items-center justify-center rounded-full border border-teal bg-card text-teal font-mono text-[10px] shadow">
              ◀▶
            </div>
          </div>

          {/* Slider Position Input */}
          <input
            type="range"
            min="0"
            max="100"
            value={sliderPos}
            onChange={(e) => setSliderPos(Number(e.target.value))}
            className="absolute inset-0 z-40 opacity-0 cursor-ew-resize w-full h-full"
            disabled={drawing}
          />

          <div className="absolute bottom-3 left-3 z-10 rounded bg-card/80 px-2 py-1 font-mono text-[10px] text-teal">
            DATE T0 (BEFORE)
          </div>
          <div className="absolute bottom-3 right-3 z-10 rounded bg-card/80 px-2 py-1 font-mono text-[10px] text-teal">
            DATE T1 (AFTER)
          </div>
        </div>
      ) : (
        /* Side by Side View */
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <div className="overflow-hidden rounded border border-border bg-card">
            <div className="border-b border-border bg-surface px-3 py-1.5 font-mono text-xs text-teal font-medium">
              T0 (BEFORE)
            </div>
            <div className="relative h-[360px] bg-surface overflow-hidden">
              {t0Url && <img src={t0Url} alt="T0" className="h-full w-full object-contain" />}
              {aoiBbox && (
                <div
                  className="absolute border-2 border-teal bg-teal/10 pointer-events-none"
                  style={{
                    left: `${(aoiBbox[0] / 512) * 100}%`,
                    top: `${(aoiBbox[1] / 512) * 100}%`,
                    width: `${((aoiBbox[2] - aoiBbox[0]) / 512) * 100}%`,
                    height: `${((aoiBbox[3] - aoiBbox[1]) / 512) * 100}%`,
                  }}
                />
              )}
            </div>
          </div>
          <div className="overflow-hidden rounded border border-border bg-card">
            <div className="border-b border-border bg-surface px-3 py-1.5 font-mono text-xs text-teal font-medium">
              T1 (AFTER)
            </div>
            <div className="relative h-[360px] bg-surface overflow-hidden">
              {t1Url && <img src={t1Url} alt="T1" className="h-full w-full object-contain" />}
              {aoiBbox && (
                <div
                  className="absolute border-2 border-teal bg-teal/10 pointer-events-none"
                  style={{
                    left: `${(aoiBbox[0] / 512) * 100}%`,
                    top: `${(aoiBbox[1] / 512) * 100}%`,
                    width: `${((aoiBbox[2] - aoiBbox[0]) / 512) * 100}%`,
                    height: `${((aoiBbox[3] - aoiBbox[1]) / 512) * 100}%`,
                  }}
                />
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
