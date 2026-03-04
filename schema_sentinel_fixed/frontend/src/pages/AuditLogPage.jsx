import React, { useEffect, useMemo, useState } from 'react'
import SectionHeader from '../components/SectionHeader'
import { api } from '../api'
import { useApp } from '../context/AppContext'
import { fmtTs } from '../utils/ui'

// ✅ readable one-line summary for table
const fmtSummary = (a) => {
  // if backend already provides summary text
  if (a.summary) return String(a.summary)

  const driftTypes = Array.isArray(a.drift_types) ? a.drift_types.join(', ') : (a.drift_types || '')
  const parts = []

  if (a.batch_id) parts.push(`BATCH: ${a.batch_id}`)
  if (driftTypes) parts.push(`DRIFT: ${driftTypes}`)

  if (a.risk_score != null) {
    parts.push(
      `RISK: ${Number(a.risk_score).toFixed(3)} (${String(a.risk_level || '').toUpperCase() || '—'})`
    )
  }

  if (a.route_applied) parts.push(`ROUTE: ${String(a.route_applied).toUpperCase()}`)
  if (a.status) parts.push(`STATUS: ${String(a.status).toUpperCase()}`)

  const diff = a.diff || {}
  const raw = a.raw_diff || {}
  const add = (diff.added || raw.added || []).length
  const rem = (diff.removed || raw.removed || []).length
  const type = (diff.type_changes || []).length
  const ren = a.renames?.mappings ? Object.keys(a.renames.mappings).length : 0

  parts.push(`CHANGES: +${add}  -${rem}  type=${type}  rename=${ren}`)

  return parts.join(' | ')
}

export default function AuditLogPage() {
  const { datasets, showToast, fmtErr } = useApp()

  const [dataset, setDataset] = useState('')
  const [auditItems, setAuditItems] = useState([])
  const [auditBusy, setAuditBusy] = useState(false)

  const normalizeAuditResponse = (data) => {
    // backend can return {events:[...]} or array; keep robust
    if (Array.isArray(data)) return data
    if (Array.isArray(data?.items)) return data.items
    if (Array.isArray(data?.events)) return data.events
    if (Array.isArray(data?.logs)) return data.logs
    return []
  }

  const refreshAudit = async () => {
    setAuditBusy(true)
    try {
      const res = await api.get('/audit/logs', { params: { limit: 500 } })
      setAuditItems(normalizeAuditResponse(res.data))
    } catch (e) {
      showToast({ type: 'error', title: 'Audit fetch failed', message: fmtErr(e) })
      setAuditItems([])
    } finally {
      setAuditBusy(false)
    }
  }

  const downloadAudit = async () => {
    try {
      const res = await api.get('/audit/logs/download', { responseType: 'blob' })
      const blob = new Blob([res.data], { type: 'text/csv' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'audit_logs.csv'
      a.click()
      URL.revokeObjectURL(url)
    } catch (e) {
      showToast({ type: 'error', title: 'Download failed', message: fmtErr(e) })
    }
  }

  // auto pick first dataset
  useEffect(() => {
    if (!dataset && (datasets || []).length > 0) {
      setDataset(datasets[0]?.name || '')
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasets])

  useEffect(() => {
    refreshAudit().catch(() => {})
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // filter rows for selected dataset
  const auditList = useMemo(() => {
    const arr = Array.isArray(auditItems) ? auditItems : []
    const ds = (dataset || '').trim().toLowerCase()
    if (!ds) return arr
    return arr.filter((a) => String(a.dataset || a.dataset_name || '').trim().toLowerCase() === ds)
  }, [auditItems, dataset])

  const fmtAction = (a) => {
    const act = String(a.action || '').trim()
    if (act) return act.toUpperCase()
    if (a.drift_types || a.schema_hash_observed || a.schema_version_candidate != null) return 'DETECTED'
    return '—'
  }

  const fmtApprover = (a) => a.approver || a.decision_by || a.user || '—'
  const fmtBatch = (a) => a.batch_id || a.batch || '—'
  const fmtEvent = (a) => {
    const id = a.event_id ?? a.id
    return id != null ? `#${id}` : '—'
  }
  const fmtTime = (a) => {
    const ts = a.created_at || a.ts || a.timestamp
    return ts ? fmtTs(ts) : '—'
  }

  const hasDebug = (a) => {
    const hasRaw = a.raw_diff && (Object.keys(a.raw_diff || {}).length > 0)
    const hasDiff = a.diff && (Object.keys(a.diff || {}).length > 0)
    const hasRen = a.renames && (Object.keys(a.renames || {}).length > 0)
    return hasRaw || hasDiff || hasRen
  }

  return (
    <section className="section">
      <SectionHeader
        title="Audit Log"
        subtitle="Filter by dataset. Shows drift detections + approvals/rejects/rollbacks."
        actions={
          <>
            <button className="btn btn-ghost" onClick={() => refreshAudit().catch(() => {})}>
              {auditBusy ? 'Loading…' : 'Refresh'}
            </button>
            <button className="btn btn-primary" onClick={() => downloadAudit().catch(() => {})}>
              Download CSV
            </button>
          </>
        }
      />

      <div className="card" style={{ marginBottom: 12 }}>
        <div className="card-body">
          <div className="form-group">
            <div className="form-label">Dataset</div>
            <select className="form-input" value={dataset} onChange={(e) => setDataset(e.target.value)}>
              <option value="">Select…</option>
              {(datasets || []).map((d) => (
                <option key={d.name} value={d.name}>
                  {d.name}
                </option>
              ))}
            </select>

            <div className="muted" style={{ marginTop: 6 }}>
              Showing {auditList.length} rows for dataset: <span className="mono">{dataset || '—'}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-body no-pad">
          <table className="table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Dataset</th>
                <th>Approver</th>
                <th>Action</th>
                <th>Event</th>
                <th>Batch</th>
                <th>Note / Summary</th>
              </tr>
            </thead>

            <tbody>
              {auditList.map((a, idx) => (
                <tr key={a.id ?? `${a.event_id ?? 'evt'}-${idx}`}>
                  <td className="muted">{fmtTime(a)}</td>
                  <td className="mono">{a.dataset || a.dataset_name || '—'}</td>
                  <td>{fmtApprover(a)}</td>

                  <td>
                    <span className="badge badge-neutral">{fmtAction(a)}</span>
                  </td>

                  <td>{fmtEvent(a)}</td>
                  <td className="mono muted">{fmtBatch(a)}</td>

                  {/* ✅ Cleaner summary + collapsible debug */}
                  <td className="muted" style={{ whiteSpace: 'pre-wrap' }}>
                    {fmtSummary(a) || a.note || a.reason || '—'}

                    {hasDebug(a) ? (
                      <details style={{ marginTop: 6 }}>
                        <summary className="muted" style={{ cursor: 'pointer' }}>
                          Raw diff / renames
                        </summary>
                        <pre style={{ marginTop: 6, maxHeight: 200, overflow: 'auto' }}>
                          {JSON.stringify({ raw_diff: a.raw_diff, diff: a.diff, renames: a.renames }, null, 2)}
                        </pre>
                      </details>
                    ) : null}
                  </td>
                </tr>
              ))}

              {auditList.length === 0 ? (
                <tr>
                  <td colSpan={7} className="muted">
                    No audit items for this dataset.
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  )
}