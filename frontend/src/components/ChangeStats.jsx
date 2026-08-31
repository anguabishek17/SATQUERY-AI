export default function ChangeStats({ stats }) {
  if (!stats) return null

  return (
    <div className="grid grid-cols-3 gap-2">
      <div className="rounded border border-border bg-surface2 p-3 text-center">
        <div className="font-mono text-lg text-amber">{stats.pct_changed.toFixed(1)}%</div>
        <div className="mt-1 text-[10px] uppercase tracking-wide text-ink-dim">changed</div>
      </div>
      <div className="rounded border border-border bg-surface2 p-3 text-center">
        <div className="font-mono text-lg text-amber">{stats.changed_area_ha.toFixed(1)}</div>
        <div className="mt-1 text-[10px] uppercase tracking-wide text-ink-dim">hectares</div>
      </div>
      <div className="rounded border border-border bg-surface2 p-3 text-center">
        <div className="truncate font-mono text-xs text-amber" title={stats.primary_change_zone}>
          {stats.primary_change_zone}
        </div>
        <div className="mt-1 text-[10px] uppercase tracking-wide text-ink-dim">primary zone</div>
      </div>
    </div>
  )
}
