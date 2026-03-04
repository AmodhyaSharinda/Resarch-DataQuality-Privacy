import React from 'react'
import { cls } from '../utils/ui'

export function RiskBadge({ risk }) {
  const r = String(risk || '').toUpperCase()
  const c = r === 'HIGH' ? 'badge-high' : r === 'MEDIUM' ? 'badge-med' : r === 'LOW' ? 'badge-low' : 'badge-neutral'
  return <span className={cls('badge', c)}>{r || '—'}</span>
}

export function RouteBadge({ route }) {
  const r = String(route || '').toUpperCase()
  const c = r === 'PRODUCTION' ? 'badge-production' : r === 'STAGING' ? 'badge-staging' : 'badge-neutral'
  return <span className={cls('badge', c)}>{r || '—'}</span>
}

export function StatusBadge({ status }) {
  const s = String(status || '').toUpperCase()
  const c =
    s === 'APPROVED' || s === 'PROMOTED'
      ? 'badge-low'
      : s === 'PENDING'
        ? 'badge-med'
        : s.includes('REJECT')
          ? 'badge-high'
          : 'badge-neutral'
  return <span className={cls('badge', c)}>{s || '—'}</span>
}
