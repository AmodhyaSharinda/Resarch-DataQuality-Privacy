// frontend/src/api.js
import axios from 'axios'

// Set this in frontend/.env as:
// VITE_API_BASE_URL=http://127.0.0.1:8000/v1
const DEFAULT_BASE = 'http://127.0.0.1:8000/v1'

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || DEFAULT_BASE).replace(/\/+$/, '')

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
})

export function getStreamConsoleUrl() {
  const base = API_BASE_URL.replace(/\/v1\/?$/, '')
  return `${base}/v1/stream/console`
}

export function fmtErr(e) {
  if (!e) return 'Unknown error'
  if (typeof e === 'string') return e
  if (e?.response?.data?.detail) return String(e.response.data.detail)
  if (e?.response?.data?.message) return String(e.response.data.message)
  if (e?.message) return String(e.message)
  return 'Unknown error'
}

// ---- NEW helpers (Risk + XAI + Fields) ----
export function getDatasetFields(dataset) {
  return api.get(`/datasets/${encodeURIComponent(dataset)}/fields`)
}

export function getRiskConfig(dataset) {
  return api.get(`/datasets/${encodeURIComponent(dataset)}/risk-config`)
}

export function setRiskConfig(dataset, payload) {
  return api.post(`/datasets/${encodeURIComponent(dataset)}/risk-config`, payload)
}

export function getEventDetails(eventId) {
  return api.get(`/events/${encodeURIComponent(eventId)}`)
}

// governance actions1
export function approveAndPromote(eventId, payload) {
  return api.post(`/events/${encodeURIComponent(eventId)}/approve`, payload)
}

export function rejectEvent(eventId, payload) {
  return api.post(`/events/${encodeURIComponent(eventId)}/reject`, payload)
}

export function rollbackEvent(eventId, payload) {
  return api.post(`/events/${encodeURIComponent(eventId)}/rollback`, payload)
}

export function listEvents(params = {}) {
  return api.get('/events', { params })
}

export function getStagingQueue(limit = 200) {
  return api.get('/staging/queue', { params: { limit } })
}