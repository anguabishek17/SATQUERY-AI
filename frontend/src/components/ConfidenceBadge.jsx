export default function ConfidenceBadge({ confidence, lowConfidence, confidenceCalibrated = true }) {
  const pct = Math.round(confidence * 100)
  const color = lowConfidence ? 'text-amber border-amber/40 bg-amber/10' : 'text-teal border-teal/40 bg-teal/10'

  return (
    <div className={`inline-flex items-center gap-2 rounded border px-2.5 py-1 font-mono text-xs ${color}`}>
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: 'currentColor' }} />
      {confidenceCalibrated ? (
        <>confidence {pct}%</>
      ) : (
        <span title="This model's API doesn't expose a real probability — this isn't a percentage to trust.">
          confidence not calibrated
        </span>
      )}
      {lowConfidence && <span className="ml-1 opacity-80">— flagged</span>}
    </div>
  )
}
