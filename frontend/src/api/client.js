const BASE = '/api'

export function getErrorMessage(error) {
  if (!error) return 'Unknown error'
  if (typeof error === 'string') return error
  if (error.message) {
    if (error.message === '{}') return null // Ignore empty object stringified
    return error.message
  }
  if (error.detail) {
    if (typeof error.detail === 'string') return error.detail
    return JSON.stringify(error.detail)
  }
  if (error.error) {
    if (typeof error.error === 'string') return error.error
    return JSON.stringify(error.error)
  }
  if (typeof error === 'object' && Object.keys(error).length === 0) return null
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
      let errText = await res.text().catch(() => '')
      let errJson = null
      try { errJson = JSON.parse(errText) } catch(e) {}
      
      let msg = errJson ? getErrorMessage(errJson) : errText
      
      if (!msg) {
        if (res.status === 400) msg = 'Bad Request: Invalid image format or parameters.'
        else if (res.status === 404) msg = 'Endpoint not found. Is the backend running the correct version?'
        else if (res.status === 413) msg = 'Payload Too Large: The image is too big to process.'
        else if (res.status === 422) msg = 'Unprocessable Entity: Missing required fields.'
        else if (res.status >= 500) msg = 'Internal Server Error. Check backend logs.'
        else msg = 'Please check the backend server and image format.'
      }
      
      throw new Error(msg)
    }
    return await res.json()
  } catch (err) {
    if (err instanceof TypeError && err.message === 'Failed to fetch') {
      throw new Error('Upload failed: Backend unavailable (network/connection failure)')
    }
    const finalMsg = getErrorMessage(err) || 'Please check the backend server and image format.'
    throw new Error(`Upload failed: ${finalMsg}`)
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
    if (err instanceof TypeError && err.message === 'Failed to fetch') {
      throw new Error('Query failed: Backend unavailable (server may not be running on port 8000)')
    }
    throw new Error(`Query failed: ${getErrorMessage(err)}`)
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
    if (err instanceof TypeError && err.message === 'Failed to fetch') {
      throw new Error('Geocoding failed: Backend unavailable (server may not be running on port 8000)')
    }
    throw new Error(`Geocoding failed: ${getErrorMessage(err)}`)
  }
}

export function reportPdfUrl(reportId) {
  return `${BASE}/query/${reportId}/report.pdf`
}

export function previewUrl(fileId) {
  return fileId ? `${BASE}/upload/${fileId}/preview` : null
}
