const ACTION_LABELS = {
  ground: 'Ground',
  buffer: 'Buffer',
  change_detect: 'Change detect',
  intersect: 'Intersect',
  sar_evidence: 'SAR evidence',
  summarize: 'Summarize',
}

export default function SatQueryChain({ chain }) {
  if (!chain?.length) return null

  return (
    <div className="rounded border border-coral/30 bg-coral/5 p-4">
      <h3 className="mb-3 text-xs font-medium uppercase tracking-wide text-coral">
        SatQuery chain — decomposed plan
      </h3>
      <ol className="space-y-1.5">
        {chain.map((step) => (
          <li key={step.step_id} className="flex items-start gap-2 font-mono text-xs">
            <span className="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full border border-coral/40 text-[10px] text-coral">
              {step.step_id}
            </span>
            <div>
              <span className="text-ink">{ACTION_LABELS[step.action] || step.action}</span>
              <span className="text-ink-dim"> — {step.description}</span>
              {step.depends_on?.length > 0 && (
                <span className="text-ink-dim/70"> (after step{step.depends_on.length > 1 ? 's' : ''} {step.depends_on.join(', ')})</span>
              )}
            </div>
          </li>
        ))}
      </ol>
    </div>
  )
}
