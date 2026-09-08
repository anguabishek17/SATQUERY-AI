const STEP_LABELS = {
  sensor_intelligence: 'Sensor intelligence',
  input_validation: 'Input validation',
  task_classification: 'Task classification',
  tool_selection: 'Tool selection',
  tool_execution: 'Specialist geoanalysis',
  evidence_json: 'Evidence JSON',
  ai_reasoning: 'AI reasoning layer',
  evidence_validation: 'Evidence validation',
  performance_metrics: 'Performance metrics',
  output_combination: 'Output combination',
}

export default function ExecutionTrace({ trace, toolsUsed }) {
  if (!trace?.length) return null

  return (
    <div className="rounded border border-border bg-surface2 p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-xs font-medium uppercase tracking-wide text-ink-dim">Execution trace</h3>
        <span className="font-mono text-[11px] text-ink-dim">{toolsUsed?.join(', ')}</span>
      </div>
      <ol className="space-y-2">
        {trace.map((step, i) => {
          const isVal = step.step === 'evidence_validation'
          let valBadge = null
          if (isVal) {
            if (step.detail?.includes('status=VALIDATED')) {
              valBadge = <span className="ml-2 inline-block rounded bg-emerald-500/20 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-400">VALIDATED</span>
            } else if (step.detail?.includes('status=VALIDATION_UNAVAILABLE')) {
              valBadge = <span className="ml-2 inline-block rounded bg-amber-500/20 px-1.5 py-0.5 text-[10px] font-semibold text-amber-400">VALIDATION UNAVAILABLE</span>
            } else if (step.detail?.includes('status=VALIDATION_FAILED')) {
              valBadge = <span className="ml-2 inline-block rounded bg-rose-500/20 px-1.5 py-0.5 text-[10px] font-semibold text-rose-400">VALIDATION FAILED</span>
            }
          }

          return (
            <li key={i} className="flex gap-3 font-mono text-xs">
              <span className="text-teal">{String(i + 1).padStart(2, '0')}</span>
              <div>
                <span className="text-ink">{STEP_LABELS[step.step] || step.step}</span>
                {valBadge}
                <span className="text-ink-dim"> — {step.detail}</span>
              </div>
            </li>
          )
        })}
      </ol>
    </div>
  )
}
