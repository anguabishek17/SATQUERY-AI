import { useState } from 'react'

const SUGGESTIONS = [
  'Describe the land-cover and major objects visible in this image.',
  'Highlight the water body referred to in the query.',
  'What changed between these two dates, and where did the change occur?',
  'Use the optical and SAR images together to identify built-up and water-covered regions.',
]

export default function QueryBox({ disabled, loading, onSubmit }) {
  const [value, setValue] = useState('')

  function submit() {
    if (!value.trim()) return
    onSubmit(value.trim())
  }

  return (
    <div className="rounded border border-border bg-surface2 p-4">
      <h3 className="mb-3 text-xs font-medium uppercase tracking-wide text-ink-dim">Query</h3>
      <textarea
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder={disabled ? 'Upload the required image(s) first…' : 'Ask a question about the loaded imagery…'}
        disabled={disabled}
        rows={3}
        className="w-full resize-none rounded border border-border bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink-dim/60 focus:border-teal/50 focus:outline-none disabled:opacity-50"
      />

      <div className="mt-2 flex flex-wrap gap-1.5">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            onClick={() => setValue(s)}
            disabled={disabled}
            className="rounded border border-border px-2 py-1 text-[11px] text-ink-dim hover:border-teal/40 hover:text-teal disabled:opacity-40"
          >
            {s.length > 38 ? s.slice(0, 38) + '…' : s}
          </button>
        ))}
      </div>

      <button
        onClick={submit}
        disabled={disabled || loading || !value.trim()}
        className="mt-3 w-full rounded bg-teal py-2 text-sm font-medium text-surface disabled:cursor-not-allowed disabled:opacity-30 hover:bg-teal/90"
      >
        {loading ? 'Running agent…' : 'Run query'}
      </button>
    </div>
  )
}
