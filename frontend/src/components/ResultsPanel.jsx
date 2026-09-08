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
        <p className="text-sm leading-relaxed text-ink whitespace-pre-line">{result.answer}</p>

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

        {result.physical_metrics?.rgb_fallback && (
          <div className="mt-4 rounded border border-border bg-surface px-4 py-3">
            <div className="mb-2 text-xs font-semibold tracking-wide text-ink-dim uppercase border-b border-border pb-1">
              Method: RGB-based visual estimate
            </div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-ink-dim">Vegetation</span>
                <span className="font-mono text-teal">{result.physical_metrics.vegetation_pct !== undefined && result.physical_metrics.vegetation_pct !== null ? `${result.physical_metrics.vegetation_pct.toFixed(1)}%` : 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-ink-dim">Water</span>
                <span className="font-mono text-teal">{result.physical_metrics.water_pct !== undefined && result.physical_metrics.water_pct !== null ? `${result.physical_metrics.water_pct.toFixed(1)}%` : 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-ink-dim">Built-up</span>
                <span className="font-mono text-teal">{result.physical_metrics.builtup_pct !== undefined && result.physical_metrics.builtup_pct !== null ? `${result.physical_metrics.builtup_pct.toFixed(1)}%` : 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-ink-dim">Bare Land</span>
                <span className="font-mono text-teal">{result.physical_metrics.bare_pct !== undefined && result.physical_metrics.bare_pct !== null ? `${result.physical_metrics.bare_pct.toFixed(1)}%` : 'N/A'}</span>
              </div>
            </div>
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
