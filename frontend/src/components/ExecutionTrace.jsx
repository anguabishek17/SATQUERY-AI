const STEP_LABELS = {
  input_validation: 'Input validation',
  task_classification: 'Task classification',
  tool_selection: 'Tool selection',
  tool_execution: 'Tool execution',
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
        {trace.map((step, i) => (
          <li key={i} className="flex gap-3 font-mono text-xs">
            <span className="text-teal">{String(i + 1).padStart(2, '0')}</span>
            <div>
              <span className="text-ink">{STEP_LABELS[step.step] || step.step}</span>
              <span className="text-ink-dim"> — {step.detail}</span>
            </div>
          </li>
        ))}
      </ol>
    </div>
  )
}
