import { useState } from 'react'
import UploadPanel from '../components/UploadPanel'
import SatelliteMapWorkspace from '../components/SatelliteMapWorkspace'
import OpticalSarWorkspace from '../components/OpticalSarWorkspace'
import BiTemporalWorkspace from '../components/BiTemporalWorkspace'
import { runQuery, geocodeLocation, previewUrl, reportPdfUrl, getErrorMessage } from '../api/client'

export default function Dashboard() {
  const [uploadedImages, setUploadedImages] = useState(null)
  const [mapCapturedImages, setMapCapturedImages] = useState(null)

  // Original uploaded satellite imagery takes strict precedence over temporary viewport captures
  const activeImages = uploadedImages?.length ? uploadedImages : mapCapturedImages
  const isCustomUploaded = Boolean(uploadedImages?.length)

  const [result, setResult] = useState(null)
  const [sessionId, setSessionId] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [aoiBbox, setAoiBbox] = useState(null)
  const [queryInput, setQueryInput] = useState('')
  const [searchLocation, setSearchLocation] = useState('')
  const [searchBusy, setSearchBusy] = useState(false)
  const [searchedCoords, setSearchedCoords] = useState(null)

  const [messages, setMessages] = useState([
    {
      sender: 'assistant',
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      text: 'SatQuery AI Mission Console initialized. Search location, draw an AOI on the satellite map, or upload custom imagery for single-image, cross-modal, or bi-temporal analysis.',
    },
  ])

  // Active layers state for right sidebar
  const [activeLayers, setActiveLayers] = useState({
    trueColor: true,
    falseColor: false,
    ndvi: false,
    sar: false,
    detection: true,
  })

  // Image URLs
  const primaryImageUrl = activeImages?.length ? previewUrl(activeImages[0].file_id) : null
  const secondaryImageUrl = activeImages?.length > 1 ? previewUrl(activeImages[1].file_id) : null

  // Modality & Workflow classification
  const inputConfig = result?.input_config
  const isCrossModal = inputConfig === 'cross_modal' || (activeImages?.some((i) => i.modality === 'optical') && activeImages?.some((i) => i.modality === 'sar'))
  const isBiTemporal = inputConfig === 'bi_temporal' || (activeImages?.length >= 2 && !isCrossModal)

  // Instant AOI Area Calculation
  let instantAoiHa = null
  let instantAoiKm2 = null
  if (aoiBbox) {
    const dx = Math.abs(aoiBbox[2] - aoiBbox[0])
    const dy = Math.abs(aoiBbox[3] - aoiBbox[1])
    const areaPx = dx * dy
    const areaM2 = areaPx * 100 // 10m pixel resolution assumption
    instantAoiHa = (areaM2 / 10000).toFixed(4)
    instantAoiKm2 = (areaM2 / 1000000).toFixed(4)
  }

  const displayHa = result?.physical_metrics?.area_ha ?? instantAoiHa ?? '--'
  const displayKm2 = result?.physical_metrics?.area_sq_km ?? instantAoiKm2 ?? '--'

  const displayCount = result?.physical_metrics?.building_count !== undefined && result?.physical_metrics?.building_count !== null
    ? result.physical_metrics.building_count
    : '--'

  const displayDensity = result?.physical_metrics?.density_per_km2 !== undefined && result?.physical_metrics?.density_per_km2 !== null
    ? `${result.physical_metrics.density_per_km2} / km²`
    : '--'

  const features = result?.geojson_overlay?.features || []
  const hasFootprints = features.length > 0

  const isBuildingTask = result?.task === 'object_counting'
  const isBuildingLoaded = result?.detector_status === 'loaded' || result?.raw?.detector_status === 'loaded' || result?.execution_trace?.some(step => step.detail?.includes('(LOADED [OK])'))

  async function handleLocationSearch(e) {
    e.preventDefault()
    if (!searchLocation.trim()) return
    setSearchBusy(true)
    setError(null)
    try {
      const data = await geocodeLocation(searchLocation.trim())
      if (data.success && data.primary) {
        setSearchedCoords({ lat: data.primary.lat, lon: data.primary.lon })
        setMessages((prev) => [
          ...prev,
          {
            sender: 'assistant',
            time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            text: `📍 Navigated map to ${data.primary.display_name} (${data.primary.lat.toFixed(4)}, ${data.primary.lon.toFixed(4)}).`,
          },
        ])
      } else {
        setError('Location not found. Please try another place name.')
      }
    } catch (err) {
      const errMsg = getErrorMessage(err)
      setError(`Location search failed: ${errMsg}`)
    } finally {
      setSearchBusy(false)
    }
  }

  async function handleSubmit(customQuery) {
    const textToSubmit = customQuery || queryInput
    if (!textToSubmit.trim()) return

    const userMsg = { sender: 'user', time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }), text: textToSubmit }
    setMessages((prev) => [...prev, userMsg])
    setQueryInput('')
    setLoading(true)
    setError(null)

    try {
      const res = await runQuery(textToSubmit, activeImages || [], sessionId, aoiBbox)
      console.log('Query Response Task:', res?.task)
      console.log('Query Response Execution Trace:', res?.execution_trace)
      console.log('Query Response Raw:', res?.raw)
      setResult(res)
      setSessionId(res.session_id)

      const aiMsg = {
        sender: 'assistant',
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        text: res.answer,
        confidence: res.confidence,
        lowConfidence: res.low_confidence,
        task: res.task,
        metrics: res.physical_metrics,
      }
      setMessages((prev) => [...prev, aiMsg])
    } catch (e) {
      const errMsg = getErrorMessage(e)
      setError(`Query execution failed: ${errMsg}`)
      setMessages((prev) => [
        ...prev,
        { sender: 'assistant', time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }), text: `❌ Error: ${errMsg}` },
      ])
    } finally {
      setLoading(false)
    }
  }

  function resetSession() {
    setSessionId(null)
    setResult(null)
    setAoiBbox(null)
    setMessages([
      {
        sender: 'assistant',
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        text: 'Session reset. Ready for new satellite queries.',
      },
    ])
  }

  function handleExportResearchReport() {
    if (!result?.research_report) return
    const str = JSON.stringify(result.research_report, null, 2)
    const blob = new Blob([str], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `satquery_report_${result.report_id || 'session'}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="min-h-screen bg-[#0B0F19] text-slate-100 font-sans flex flex-col">
      {/* Top Header Bar */}
      <header className="border-b border-[#1E293B] bg-[#131927] px-6 py-3 shadow-md">
        <div className="flex items-center justify-between gap-4">
          {/* Logo & Brand */}
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-indigo-600 shadow-lg shadow-indigo-500/30 text-white font-bold text-lg">
              🛰️
            </div>
            <div>
              <h1 className="text-base font-bold tracking-tight text-white flex items-center gap-2">
                SatQuery AI
                <span className="text-[10px] font-mono font-medium text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded border border-indigo-500/20">
                  v2.0-RS
                </span>
              </h1>
              <p className="text-[11px] text-slate-400">Intelligent Satellite Analysis & Mission Console</p>
            </div>
          </div>

          {/* Location Search Bar */}
          <form onSubmit={handleLocationSearch} className="hidden md:flex flex-1 max-w-md items-center gap-2 rounded-xl border border-[#1E293B] bg-[#0B0F19] px-3.5 py-1.5 text-xs text-slate-300 focus-within:border-indigo-500 shadow-inner">
            <span className="text-slate-400">🔍</span>
            <input
              type="text"
              value={searchLocation}
              onChange={(e) => setSearchLocation(e.target.value)}
              placeholder="Search location (e.g. Chennai, Coimbatore, Berlin)..."
              className="w-full bg-transparent text-slate-200 placeholder:text-slate-500 focus:outline-none"
            />
            <button
              type="submit"
              disabled={searchBusy || !searchLocation.trim()}
              className="rounded bg-indigo-600 px-2.5 py-0.5 text-[11px] font-mono font-medium text-white hover:bg-indigo-500 disabled:opacity-40"
            >
              {searchBusy ? '...' : 'Go'}
            </button>
          </form>

          {/* Header Controls & Profile */}
          <div className="flex items-center gap-3 text-xs">
            <button className="text-slate-400 hover:text-slate-200 transition-colors flex items-center gap-1 font-medium">
              ❓ Help
            </button>
            {result?.research_report && (
              <button
                onClick={handleExportResearchReport}
                className="rounded-lg bg-indigo-600/20 text-indigo-400 border border-indigo-500/30 px-3 py-1.5 font-medium hover:bg-indigo-600/30 transition-all flex items-center gap-1.5"
              >
                📥 Export Report
              </button>
            )}
            <div className="flex items-center gap-2 border-l border-slate-700 pl-3">
              <div className="h-7 w-7 rounded-full bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center font-bold text-white text-xs shadow">
                SIH
              </div>
              <span className="font-medium text-slate-300 hidden sm:inline">Team SIH</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main 3-Column Mission Console Grid (Matching 2nd Reference Picture) */}
      <main className="flex-1 grid grid-cols-1 lg:grid-cols-[340px_1fr_300px] gap-4 p-4 max-w-[1920px] mx-auto w-full">
        {/* Left Column: Conversation Panel with Embedded Input Configuration & Chat */}
        <div className="flex flex-col gap-3 rounded-2xl border border-[#1E293B] bg-[#131927] p-4 shadow-xl h-[calc(100vh-85px)]">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h2 className="text-sm font-semibold text-white flex items-center gap-2">
              <span>💬</span> Conversation
            </h2>
            <button
              onClick={resetSession}
              className="rounded-lg bg-indigo-600/15 text-indigo-400 border border-indigo-500/30 px-2.5 py-1 text-xs font-medium hover:bg-indigo-600/25 transition-colors flex items-center gap-1"
            >
              ➕ New Chat
            </button>
          </div>

          {/* Image Input Configuration Panel (Embedded at top of Conversation) */}
          <div className="border-b border-slate-800 pb-3">
            <UploadPanel
              onImagesReady={(imgs) => {
                setUploadedImages(imgs)
                if (imgs && imgs.length > 0) {
                  setMapCapturedImages(null)
                }
                setAoiBbox(null)
              }}
            />
          </div>

          {/* Messages Scroll Area */}
          <div className="flex-1 overflow-y-auto space-y-3.5 pr-1 font-sans text-xs custom-scrollbar">
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`flex flex-col gap-1 ${
                  msg.sender === 'user' ? 'items-end' : 'items-start'
                }`}
              >
                <div className="flex items-center gap-1.5 text-[10px] text-slate-400 font-mono">
                  <span>{msg.sender === 'user' ? 'You' : 'SatQuery AI'}</span>
                  <span>•</span>
                  <span>{msg.time}</span>
                </div>
                <div
                  className={`rounded-2xl px-4 py-3 max-w-[92%] leading-relaxed ${
                    msg.sender === 'user'
                      ? 'bg-indigo-600 text-white shadow-md shadow-indigo-500/20'
                      : 'bg-[#1E293B] text-slate-200 border border-slate-700/80 shadow'
                  }`}
                >
                  <p>{msg.text}</p>

                  {/* Message Metadata Badges */}
                  {msg.confidence !== undefined && (
                    <div className="mt-2 pt-2 border-t border-slate-700/60 flex items-center justify-between text-[11px] font-mono">
                      <span className="text-slate-400">Confidence:</span>
                      <strong className={msg.confidence > 0.6 ? 'text-emerald-400' : 'text-amber-400'}>
                        {(msg.confidence * 100).toFixed(0)}%
                      </strong>
                    </div>
                  )}

                  {msg.metrics?.building_count !== undefined && msg.metrics?.building_count !== null && (
                    <div className="mt-1.5 bg-[#0B0F19]/60 rounded-lg p-2 text-[11px] font-mono space-y-1 border border-slate-700/40 text-slate-300">
                      <div>🏢 Buildings: <strong className="text-indigo-400">{msg.metrics.building_count}</strong></div>
                      <div>📐 AOI Area: <strong className="text-emerald-400">{msg.metrics.area_ha} ha ({msg.metrics.area_sq_km} km²)</strong></div>
                      <div>📊 Density: <strong className="text-indigo-400">{msg.metrics.density_per_km2} / km²</strong></div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Chat Query Box Input */}
          <div className="pt-2 border-t border-slate-800">
            {error && <div className="mb-2 rounded-lg bg-rose-500/10 border border-rose-500/30 p-2 text-[11px] text-rose-400 font-mono">{error}</div>}
            <form
              onSubmit={(e) => {
                e.preventDefault()
                handleSubmit()
              }}
              className="relative flex items-center"
            >
              <input
                type="text"
                value={queryInput}
                onChange={(e) => setQueryInput(e.target.value)}
                placeholder="Ask anything about your satellite imagery..."
                className="w-full rounded-xl border border-slate-700 bg-[#0B0F19] py-2.5 pl-3.5 pr-11 text-xs text-slate-100 placeholder:text-slate-500 focus:border-indigo-500 focus:outline-none shadow-inner"
              />
              <button
                type="submit"
                disabled={loading || !queryInput.trim()}
                className="absolute right-1.5 flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-600 text-white hover:bg-indigo-500 disabled:opacity-40 transition-colors shadow"
              >
                {loading ? '⌛' : '➔'}
              </button>
            </form>
          </div>
        </div>

        {/* Center Column: Satellite Map Workspace & 4 Metric Cards */}
        <div className="flex flex-col gap-4 overflow-y-auto custom-scrollbar">
          {/* Main Map Workspace Viewport */}
          <div className="rounded-2xl border border-[#1E293B] bg-[#131927] p-4 shadow-xl">
            <div className="mb-3 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-base">🛰️</span>
                <h3 className="text-sm font-semibold text-white font-mono">
                  {isCrossModal ? 'OPTICAL + SAR FUSION WORKSPACE' : isBiTemporal ? 'BI-TEMPORAL CHANGE DETECTOR' : 'SATELLITE MAP WORKSPACE'}
                </h3>
              </div>
              <span className="rounded-full bg-indigo-500/10 border border-indigo-500/30 px-3 py-0.5 text-[10px] font-mono font-medium text-indigo-400 uppercase tracking-wider">
                {isCrossModal ? 'FUSION MODE' : isBiTemporal ? 'BI-TEMPORAL MODE' : 'SINGLE IMAGE MODE'}
              </span>
            </div>

            {isCrossModal ? (
              <OpticalSarWorkspace
                opticalUrl={primaryImageUrl}
                sarUrl={secondaryImageUrl}
                result={result}
                aoiBbox={aoiBbox}
                onAoiChange={setAoiBbox}
              />
            ) : isBiTemporal ? (
              <BiTemporalWorkspace
                t0Url={primaryImageUrl}
                t1Url={secondaryImageUrl}
                result={result}
                aoiBbox={aoiBbox}
                onAoiChange={setAoiBbox}
              />
            ) : (
              <SatelliteMapWorkspace
                previewUrl={primaryImageUrl}
                isUploadedImage={isCustomUploaded}
                uploadedFilename={uploadedImages?.[0]?.filename}
                geojsonOverlay={result?.geojson_overlay}
                boxes={result?.bounding_boxes}
                aoiBbox={aoiBbox}
                onAoiChange={setAoiBbox}
                physicalMetrics={result?.physical_metrics}
                sensorInfo={result?.sensor_info}
                rawResult={result}
                activeLayers={activeLayers}
                searchedCoords={searchedCoords}
                onCapturedImage={(capturedImgs) => {
                  if (!uploadedImages?.length) {
                    setMapCapturedImages(capturedImgs)
                  }
                }}
              />
            )}
          </div>

          {/* 4 Metric Cards directly below map (Matching 2nd Reference Picture) */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono">
            <div className="rounded-xl border border-[#1E293B] bg-[#131927] p-3.5 text-center shadow">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-sans font-semibold">
                BUILDINGS
              </span>
              <strong className="text-indigo-400 text-xl block mt-1">
                {displayCount}
              </strong>
            </div>

            <div className="rounded-xl border border-[#1E293B] bg-[#131927] p-3.5 text-center shadow">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-sans font-semibold">
                DENSITY
              </span>
              <strong className="text-indigo-400 text-xl block mt-1">
                {displayDensity}
              </strong>
            </div>

            <div className="rounded-xl border border-[#1E293B] bg-[#131927] p-3.5 text-center shadow">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-sans font-semibold">
                AOI AREA
              </span>
              <strong className="text-emerald-400 text-xl block mt-1">
                {displayHa !== '--' ? `${displayHa} ha` : '--'}
              </strong>
            </div>

            <div className="rounded-xl border border-[#1E293B] bg-[#131927] p-3.5 text-center shadow">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-sans font-semibold">
                AREA (KM²)
              </span>
              <strong className="text-emerald-400 text-xl block mt-1">
                {displayKm2 !== '--' ? `${displayKm2} km²` : '--'}
              </strong>
            </div>
          </div>

          {/* Bottom Card: Analysis Results */}
          <div className="rounded-2xl border border-[#1E293B] bg-[#131927] p-5 shadow-xl space-y-3">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <span>📊</span> Analysis Results
              </h3>
              <span className="text-xs font-mono text-slate-400">
                {result?.task ? result.task.replace(/_/g, ' ').toUpperCase() : 'NO QUERY RUN YET'}
              </span>
            </div>

            <p className="text-xs text-slate-300 leading-relaxed font-sans">
              {result?.answer || 'Execute a satellite query above to generate remote-sensing insights and physical metric breakdown.'}
            </p>
          </div>
        </div>

        {/* Right Column: Layers/Tools & Honest Agent Activity Trace */}
        <div className="flex flex-col gap-4">
          {/* Top Card: Layers & Tools */}
          <div className="rounded-2xl border border-[#1E293B] bg-[#131927] p-4 shadow-xl space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
              <div className="flex gap-4 text-xs font-semibold">
                <button className="text-indigo-400 border-b-2 border-indigo-500 pb-1">Layers</button>
                <button className="text-slate-400 hover:text-slate-200 transition-colors pb-1">Tools</button>
              </div>
            </div>

            {/* Layer Checkboxes */}
            <div className="space-y-3 text-xs font-mono">
              <label className="flex items-center justify-between cursor-pointer group">
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={activeLayers.trueColor}
                    onChange={(e) => setActiveLayers({ ...activeLayers, trueColor: e.target.checked })}
                    className="accent-indigo-500 rounded cursor-pointer"
                  />
                  <span className="text-slate-300 group-hover:text-white">True Color (RGB)</span>
                </div>
                <span className="text-[10px] text-slate-500">100%</span>
              </label>

              <label className="flex items-center justify-between cursor-pointer group">
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={activeLayers.ndvi}
                    onChange={(e) => setActiveLayers({ ...activeLayers, ndvi: e.target.checked })}
                    className="accent-indigo-500 rounded cursor-pointer"
                  />
                  <span className="text-slate-300 group-hover:text-white">NDVI Index</span>
                </div>
                <span className="text-[10px] text-slate-500">0%</span>
              </label>

              <label className="flex items-center justify-between cursor-pointer group">
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={activeLayers.detection}
                    disabled={!hasFootprints}
                    onChange={(e) => setActiveLayers({ ...activeLayers, detection: e.target.checked })}
                    className="accent-indigo-500 rounded cursor-pointer disabled:opacity-40"
                  />
                  <span className={`group-hover:text-white ${hasFootprints ? 'text-slate-300' : 'text-slate-500'}`}>
                    {hasFootprints ? `Building Footprints (${features.length})` : 'Building Footprints (Not Available)'}
                  </span>
                </div>
                <span className={hasFootprints ? 'text-[10px] text-rose-400' : 'text-[10px] text-slate-600'}>
                  {hasFootprints ? '100%' : 'OFF'}
                </span>
              </label>
            </div>
          </div>

          {/* Bottom Card: Honest Agent Activity Trace Checklist */}
          <div className="flex-1 rounded-2xl border border-[#1E293B] bg-[#131927] p-4 shadow-xl flex flex-col justify-between">
            <div className="space-y-3">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
                <h3 className="text-xs font-semibold text-white font-mono uppercase tracking-wider">
                  AGENT ACTIVITY
                </h3>
                <span className="flex items-center gap-1.5 text-[11px] text-emerald-400 font-mono font-bold">
                  <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
                  {loading ? 'Processing...' : 'Completed'}
                </span>
              </div>

              {/* Step-by-step Honest Execution Checklist */}
              <div className="space-y-2.5 text-xs text-slate-300 font-sans">
                {result?.execution_trace && result.execution_trace.length > 0 ? (
                  result.execution_trace.map((t, i) => (
                    <div key={i} className="flex items-start gap-2">
                      <span className="text-emerald-400 mt-0.5 font-bold">✓</span>
                      <div className="flex flex-col">
                        <span className="font-medium text-slate-200 capitalize">
                          {t.step ? t.step.replace(/_/g, ' ') : 'Step'}
                        </span>
                        <span className="text-[10px] text-slate-400">{t.detail}</span>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="flex items-center gap-2 text-slate-500">
                    <span>○</span>
                    <span>Waiting for query execution...</span>
                  </div>
                )}
              </div>
            </div>

            {/* View Full Report Button */}
            {result?.report_id && (
              <a
                href={reportPdfUrl(result.report_id)}
                target="_blank"
                rel="noreferrer"
                className="mt-4 block w-full text-center rounded-xl bg-indigo-600 py-2.5 text-xs font-semibold text-white hover:bg-indigo-500 transition-colors shadow-lg shadow-indigo-500/25"
              >
                View Full Report (PDF)
              </a>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
