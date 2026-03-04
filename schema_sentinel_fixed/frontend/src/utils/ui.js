export const cls = (...xs) => xs.filter(Boolean).join(' ')

export function fmtTs(iso) {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleString()
  } catch {
    return String(iso)
  }
}
