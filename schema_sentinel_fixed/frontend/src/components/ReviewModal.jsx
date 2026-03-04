import React from 'react'
import { RiskBadge, RouteBadge, StatusBadge } from './Badges'

export default function ReviewModal({ open, eventRow, detail, loading, error, onClose, onApprove, onReject, onRollback }) {
  if (!open) return null

  const driftTypes = detail?.drift_types || eventRow?.drift_types || []
  const diff = detail?.diff || {}

  const explainAction = () => {
    const types = (driftTypes || []).map((x) => String(x || '').toUpperCase())
    const msgs = []

    if (types.includes('ADD')) {
      msgs.push(
        'ADD: Approve = schema adds the new column. Old rows will have NULL for it. Rollback = new column will be DROPPED when storing.'
      )
    }
    if (types.includes('REMOVE') || types.includes('MISSING')) {
      msgs.push(
        'REMOVE/MISSING: Approve = schema removes (or accepts missing) field. Rollback = keep old schema; missing values become NULL (often risky). Reject = ask producer to resend.'
      )
    }
    if (types.includes('RENAME')) {
      msgs.push(
        'RENAME: Approve = schema switches to new name. Rollback = keep old name; rename mapping used if available. Reject if rename is uncertain.'
      )
    }
    if (types.includes('TYPE_CHANGE')) {
      msgs.push('TYPE_CHANGE: Approve = schema expects new type. Rollback = keep old type; unsafe casts should be rejected.')
    }
    if (types.includes('NULLABLE_CHANGE')) {
      msgs.push('NULLABLE_CHANGE: Approve = allow new nulls. Rollback = keep stricter nullability; nulls may cause reject.')
    }

    if (msgs.length === 0) {
      msgs.push(
        'This event contains drift. Approve accepts schema change. Reject blocks data and alerts producer. Rollback keeps previous schema and stores data in old shape.'
      )
    }
    return msgs
  }

  const cardStyle = {
    position: 'fixed',
    inset: 0,
    background: 'rgba(0,0,0,0.55)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 9999,
    padding: 18,
  }

  const modalStyle = {
    width: 'min(880px, 96vw)',
    background: 'rgba(10, 18, 30, 0.96)',
    border: '1px solid rgba(255,255,255,0.08)',
    borderRadius: 14,
    boxShadow: '0 20px 80px rgba(0,0,0,0.6)',
    overflow: 'hidden',
  }

  return (
    <div style={cardStyle} onClick={onClose}>
      <div style={modalStyle} onClick={(e) => e.stopPropagation()}>
        <div
          style={{
            padding: 16,
            borderBottom: '1px solid rgba(255,255,255,0.08)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <div style={{ fontWeight: 800, fontSize: 18 }}>Schema Change Review</div>
          <button className="btn btn-ghost" onClick={onClose}>
            ✕
          </button>
        </div>

        <div style={{ padding: 16 }}>
          {loading ? <div className="muted">Loading…</div> : null}
          {error ? <div className="muted">{error}</div> : null}

          {!loading && detail ? (
            <>
              <div className="card" style={{ background: 'rgba(255,255,255,0.03)', marginBottom: 12 }}>
                <div className="card-body">
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
                    <div>
                      <div style={{ fontWeight: 700 }}>
                        Batch: <span className="muted">{detail.batch_id}</span>
                      </div>
                      <div className="muted" style={{ marginTop: 6 }}>
                        Event #{detail.id} · Dataset <b>{detail.dataset}</b>
                      </div>
                    </div>
                    <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                      <RiskBadge risk={detail.risk_level} />
                      <RouteBadge route={detail.route} />
                      <StatusBadge status={detail.status} />
                    </div>
                  </div>

                  <div className="hint" style={{ marginTop: 10 }}>
                    Drift types: <b>{(detail.drift_types || []).join(', ') || '—'}</b>
                  </div>
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div className="card" style={{ background: 'rgba(255,255,255,0.03)' }}>
                  <div className="card-header">
                    <div>
                      <div className="card-title">What will happen if you click…</div>
                      <div className="card-sub">Explained per drift type</div>
                    </div>
                  </div>
                  <div className="card-body">
                    <div className="form-label">APPROVE & PROMOTE</div>
                    <div className="muted">Activate candidate schema (if any) + promote staged rows to production.</div>

                    <div className="form-label" style={{ marginTop: 10 }}>
                      REJECT
                    </div>
                    <div className="muted">Do NOT promote. Mark rows rejected + send alert to producer team.</div>

                    <div className="form-label" style={{ marginTop: 10 }}>
                      ROLLBACK
                    </div>
                    <div className="muted">
                      Revert to previous schema + promote rows under old schema shape (drop extra fields, fill missing as null).
                    </div>

                    <div style={{ marginTop: 12 }}>
                      {(explainAction() || []).map((m, idx) => (
                        <div key={idx} className="hint" style={{ marginTop: 6 }}>
                          {m}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="card" style={{ background: 'rgba(255,255,255,0.03)' }}>
                  <div className="card-header">
                    <div>
                      <div className="card-title">Drift summary</div>
                      <div className="card-sub">Stored with the event</div>
                    </div>
                  </div>
                  <div className="card-body">
                    <pre style={{ margin: 0, maxHeight: 220, overflow: 'auto', whiteSpace: 'pre-wrap' }}>
                      {detail.summary || JSON.stringify({ drift_types: driftTypes, diff }, null, 2)}
                    </pre>
                  </div>
                </div>
              </div>
            </>
          ) : null}
        </div>

        <div
          style={{
            padding: 16,
            borderTop: '1px solid rgba(255,255,255,0.08)',
            display: 'flex',
            justifyContent: 'flex-end',
            gap: 10,
          }}
        >
          <button className="btn btn-ghost" onClick={onClose}>
            Cancel
          </button>
          <button className="btn btn-ghost" style={{ borderColor: 'rgba(255,90,90,0.5)' }} onClick={onReject}>
            ✕ Reject
          </button>
          <button className="btn btn-ghost" onClick={onRollback}>
            ⏪ Rollback
          </button>
          <button className="btn btn-primary" onClick={onApprove}>
            ✓ Approve & Promote
          </button>
        </div>
      </div>
    </div>
  )
}
