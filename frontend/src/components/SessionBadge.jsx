export default function SessionBadge({ sessionId, turnCount, onReset }) {
  if (!sessionId) return null

  return (
    <div className="flex items-center justify-between rounded border border-border bg-surface2 px-3 py-1.5">
      <span className="font-mono text-[11px] text-ink-dim">
        session <span className="text-teal">{sessionId.slice(0, 8)}</span> · {turnCount} turn{turnCount === 1 ? '' : 's'} remembered
      </span>
      <button onClick={onReset} className="font-mono text-[11px] text-ink-dim hover:text-coral">
        reset
      </button>
    </div>
  )
}
