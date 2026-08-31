import ConfidenceBadge from './ConfidenceBadge'
import ExecutionTrace from './ExecutionTrace'
import SatQueryChain from './SatQueryChain'
import ChangeStats from './ChangeStats'
import { reportPdfUrl } from '../api/client'

export default function ResultsPanel({ result }) {
  if (!result) {
    return (
      <div className="rounded border border-dashed border-border p-8 text-center text-sm text-ink-dim">
        Run a query to see the agent's answer, confidence, and execution trace here.
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <div className="rounded border border-border bg-surface2 p-4">
        <div className="mb-2 flex items-center justify-between">
          <span className="font-mono text-[11px] uppercase tracking-wide text-coral">{result.task.replace(/_/g, ' ')}</span>
          <ConfidenceBadge
            confidence={result.confidence}
            lowConfidence={result.low_confidence}
            confidenceCalibrated={result.confidence_calibrated}
          />
        </div>
        <p className="text-sm leading-relaxed text-ink">{result.answer}</p>

        {result.object_counts?.length > 0 && (
          <div className="mt-3 grid grid-cols-2 gap-2">
            {result.object_counts.map((oc) => (
              <div key={oc.label} className="rounded border border-border bg-surface px-3 py-2 text-center">
                <div className="text-lg font-semibold text-teal">{oc.count}</div>
                <div className="text-[11px] capitalize text-ink-dim">{oc.label}{oc.count === 1 ? '' : 's'}</div>
              </div>
            ))}
          </div>
        )}

        <a
          href={reportPdfUrl(result.report_id)}
          className="mt-3 inline-block rounded border border-border px-3 py-1.5 text-xs text-ink-dim hover:border-teal/40 hover:text-teal"
          target="_blank"
          rel="noreferrer"
        >
          ↓ Download report (PDF)
        </a>
      </div>

      {result.change_stats && <ChangeStats stats={result.change_stats} />}
      {result.chain && <SatQueryChain chain={result.chain} />}
      <ExecutionTrace trace={result.execution_trace} toolsUsed={result.tools_used} />
    </div>
  )
}
