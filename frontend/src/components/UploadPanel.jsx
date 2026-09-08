import { useState } from 'react'
import { uploadImage } from '../api/client'

const CONFIGS = [
  {
    id: 'single',
    label: 'Single image',
    slots: [
      {
        key: 'a',
        modality: 'optical',
        label: 'Image',
      },
    ],
  },
  {
    id: 'cross_modal',
    label: 'Optical + SAR pair',
    slots: [
      {
        key: 'a',
        modality: 'optical',
        label: 'Optical',
      },
      {
        key: 'b',
        modality: 'sar',
        label: 'SAR',
      },
    ],
  },
  {
    id: 'bi_temporal',
    label: 'Bi-temporal pair',
    slots: [
      {
        key: 'a',
        modality: 'optical',
        label: 'Date T0',
        needsDate: true,
      },
      {
        key: 'b',
        modality: 'optical',
        label: 'Date T1',
        needsDate: true,
      },
    ],
  },
]

export default function UploadPanel({ onImagesReady }) {
  const [configId, setConfigId] = useState('single')
  const [slotData, setSlotData] = useState({})
  const [busy, setBusy] = useState(false)

  const config = CONFIGS.find((c) => c.id === configId)

  async function handleFile(slot, file) {
    setBusy(true)

    try {
      const date = slotData[slot.key]?.acquisition_date

      const result = await uploadImage(
        file,
        slot.modality,
        date,
      )

      // Create the updated object BEFORE calling setState.
      const next = {
        ...slotData,
        [slot.key]: {
          ...result,
          file,
        },
      }

      // Update UploadPanel's own state.
      setSlotData(next)

      // Notify Dashboard OUTSIDE the state updater.
      maybeNotify(next)
    } catch (e) {
      alert(e.message)
    } finally {
      setBusy(false)
    }
  }

  function handleDate(slot, date) {
    const next = {
      ...slotData,
      [slot.key]: {
        ...(slotData[slot.key] || {}),
        acquisition_date: date,
      },
    }

    setSlotData(next)

    // If the image has already been uploaded, update the parent too.
    if (next[slot.key]?.file_id) {
      maybeNotify(next)
    }
  }

  function maybeNotify(data) {
    const ready = config.slots.every(
      (s) => data[s.key]?.file_id,
    )

    if (ready) {
      onImagesReady(
        config.slots.map((s) => ({
          file_id: data[s.key].file_id,
          modality: s.modality,
          acquisition_date:
            data[s.key].acquisition_date || null,
        })),
      )
    } else {
      onImagesReady(null)
    }
  }

  function switchConfig(id) {
    setConfigId(id)
    setSlotData({})
    onImagesReady(null)
  }

  return (
    <div className="rounded-xl border border-slate-800 bg-[#0B0F19]/80 p-3">
      <h3 className="mb-2 text-[11px] font-medium uppercase tracking-wide text-slate-400">
        Input configuration
      </h3>

      <div className="mb-2.5 flex flex-wrap gap-1.5">
        {CONFIGS.map((c) => (
          <button
            key={c.id}
            type="button"
            onClick={() => switchConfig(c.id)}
            className={`rounded-lg px-2 py-1 text-[11px] transition-colors ${
              configId === c.id
                ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/40'
                : 'border border-slate-700/60 text-slate-400 hover:text-slate-200'
            }`}
          >
            {c.label}
          </button>
        ))}
      </div>

      <div
        className="grid gap-2"
        style={{
          gridTemplateColumns: `repeat(${config.slots.length}, minmax(0, 1fr))`,
        }}
      >
        {config.slots.map((slot) => {
          const filled = slotData[slot.key]?.file_id

          return (
            <div
              key={slot.key}
              className="rounded-lg border border-dashed border-slate-700/70 p-2.5 bg-[#131927]/60"
            >
              <div className="mb-1.5 flex items-center justify-between">
                <span className="text-[11px] text-slate-300 font-medium">
                  {slot.label}
                </span>

                <span className="font-mono text-[9px] uppercase px-1 rounded bg-indigo-950/70 text-indigo-300 border border-indigo-800/40">
                  {slot.modality}
                </span>
              </div>

              <label className="block cursor-pointer rounded-lg border border-slate-700/80 bg-[#0B0F19] px-2 py-3 text-center text-xs text-slate-400 hover:border-indigo-500/60 hover:text-indigo-300 transition-colors">
                {filled
                  ? `✓ ${slotData[slot.key].filename}`
                  : 'Drop GeoTIFF / TIFF'}

                <input
                  type="file"
                  accept=".tif,.tiff,.png,.jpg,.jpeg"
                  className="hidden"
                  disabled={busy}
                  onChange={(e) => {
                    const file = e.target.files?.[0]

                    if (file) {
                      handleFile(slot, file)
                    }

                    // Allow selecting the same file again.
                    e.target.value = ''
                  }}
                />
              </label>

              {slot.needsDate && (
                <input
                  type="date"
                  className="mt-2 w-full rounded border border-border bg-surface px-2 py-1 font-mono text-xs text-ink"
                  value={
                    slotData[slot.key]?.acquisition_date || ''
                  }
                  onChange={(e) =>
                    handleDate(slot, e.target.value)
                  }
                />
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}