const BASE = '/api'

export function getErrorMessage(error) {
  if (!error) return 'Unknown error'
  if (typeof error === 'string') return error
  if (error.message) return error.message
  if (error.detail) {
    if (typeof error.detail === 'string') return error.detail
    return JSON.stringify(error.detail)
  }
  if (error.error) {
    if (typeof error.error === 'string') return error.error
    return JSON.stringify(error.error)
  }
  return JSON.stringify(error)
}

export async function uploadImage(file, modality, acquisitionDate) {
  const form = new FormData()
  form.append('file', file)
  const params = new URLSearchParams({ modality })
  if (acquisitionDate) params.append('acquisition_date', acquisitionDate)

  try {
    const res = await fetch(`${BASE}/upload?${params.toString()}`, {
      method: 'POST',
      body: form,
    })
    if (!res.ok) {
      const errJson = await res.json().catch(() => ({}))
      throw new Error(getErrorMessage(errJson) || 'Upload failed')
    }
    return res.json()
  } catch (err) {
    throw new Error(getErrorMessage(err))
  }
}

export async function runQuery(query, images, sessionId, aoiBbox) {
  try {
    const res = await fetch(`${BASE}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, images: images || [], session_id: sessionId || null, aoi_bbox: aoiBbox || null }),
    })
    if (!res.ok) {
      const errJson = await res.json().catch(() => ({}))
      throw new Error(getErrorMessage(errJson) || 'Query failed')
    }
    return res.json()
  } catch (err) {
    throw new Error(getErrorMessage(err))
  }
}

export async function geocodeLocation(query) {
  try {
    const res = await fetch(`${BASE}/geocode?query=${encodeURIComponent(query)}`)
    if (!res.ok) {
      const errJson = await res.json().catch(() => ({}))
      throw new Error(getErrorMessage(errJson) || 'Geocoding failed')
    }
    return res.json()
  } catch (err) {
    throw new Error(getErrorMessage(err))
  }
}

export function reportPdfUrl(reportId) {
  return `${BASE}/query/${reportId}/report.pdf`
}

export function previewUrl(fileId) {
  return fileId ? `${BASE}/upload/${fileId}/preview` : null
}
