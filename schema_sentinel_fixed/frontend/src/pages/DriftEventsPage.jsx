import React, { useState } from 'react'
import { api } from '../api'
import SectionHeader from '../components/SectionHeader'
import { RiskBadge, RouteBadge, StatusBadge } from '../components/Badges'
import MiniKV from '../components/MiniKV'
import { useApp } from '../context/AppContext'
import { fmtTs } from '../utils/ui'

export default function DriftEventsPage() {
  const { pendingEvents, actionedEvents, refreshEvents, fmtErr } = useApp()

  const [expandedId, setExpandedId] = useState(null)
  const [detailsById, setDetailsById] = useState({})

  const toggleDetail = async (id) => {
    if (expandedId === id) {
      setExpandedId(null)
      return
    }
    setExpandedId(id)

    if (detailsById[id]?.data || detailsById[id]?.loading) return

    setDetailsById((prev) => ({ ...prev, [id]: { loading: true, error: null, data: null } }))
    try {
      const res = await api.get(`/events/${id}`)
      setDetailsById((prev) => ({ ...prev, [id]: { loading: false, error: null, data: res.data } }))
    } catch (e) {
      setDetailsById((prev) => ({ ...prev, [id]: { loading: false, error: fmtErr(e), data: null } }))
    }
  }

  const renderList = (items, fmt) => {
    const arr = Array.isArray(items) ? items : []
    if (arr.length === 0) return <div className="muted">—</div>
    return (
      <ul style={{ margin: '6px 0 0 16px' }}>
        {arr.slice(0, 50).map((x, i) => (
          <li key={i} className="muted">
            {fmt(x)}
          </li>
        ))}
      </ul>
    )
  }

  // ✅ merged Risk Calculation + XAI into one compact panel
  const renderRiskPanel = (detail) => {
    const r = detail?.risk_details || null
    const x = detail?.xai || null

    if (!r && !x) return <div className="muted">No risk details stored for this event.</div>

    const mode = String((r?.mode || x?.mode || '—') || '—').toUpperCase()

    // Drivers (prefer XAI Mode A drivers; fallback to r.components if needed)
    let drivers = []
    if (mode === 'A' && Array.isArray(x?.drivers)) {
      drivers = x.drivers
    } else if (r?.components && typeof r.components === 'object') {
      // convert components to driver-like list (only keep T,C,S,K,U if present)
      const keys = ['T', 'C', 'S', 'K', 'U']
      drivers = keys
        .filter((k) => r.components[k] !== undefined)
        .map((k) => ({ name: k, value: r.components[k] }))
    }

    const comps = r?.components && typeof r.components === 'object' ? r.components : {}
    const counts = r?.change_counts && typeof r.change_counts === 'object' ? r.change_counts : {}

    const riskBase = comps?.risk_base
    const riskFinal = comps?.risk_final
    const floorApplied = comps?.floor_applied

    return (
      <div>
        {/* header */}
        <div className="hint" style={{ marginBottom: 10, display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
          <span>
            Mode: <span className="badge badge-neutral">{mode}</span>
          </span>
          <span>
            Risk: <b>{detail?.risk_score != null ? Number(detail.risk_score).toFixed(4) : '—'}</b>
          </span>
          <span>
            Level: <span className="badge badge-neutral">{String(detail?.risk_level || '—').toUpperCase()}</span>
          </span>
          <span>
            Route: <span className="badge badge-neutral">{String(detail?.route || '—').toUpperCase()}</span>
          </span>
        </div>

        {/* Mode A drivers */}
        {drivers.length > 0 ? (
          <>
            <div className="form-label">Top drivers</div>
            <div style={{ marginTop: 6 }}>
              {drivers.map((d, idx) => (
                <MiniKV
                  key={idx}
                  k={d.name}
                  v={typeof d.value === 'number' ? d.value.toFixed(4) : String(d.value)}
                />
              ))}
            </div>
          </>
        ) : (
          <div className="muted">No drivers available.</div>
        )}

        {/* Key computed outputs */}
        <div style={{ marginTop: 12 }}>
          <div className="form-label">Key outputs</div>
          <div style={{ marginTop: 6 }}>
            <MiniKV k="risk_base" v={typeof riskBase === 'number' ? riskBase.toFixed(4) : riskBase == null ? '—' : String(riskBase)} />
            <MiniKV k="risk_final" v={typeof riskFinal === 'number' ? riskFinal.toFixed(4) : riskFinal == null ? '—' : String(riskFinal)} />
            <MiniKV k="floor_applied" v={floorApplied == null ? 'null' : JSON.stringify(floorApplied)} />
          </div>
        </div>

        {/* Change counts */}
        {Object.keys(counts).length > 0 ? (
          <div style={{ marginTop: 12 }}>
            <div className="form-label">Change counts</div>
            <div style={{ marginTop: 6 }}>
              {Object.entries(counts).map(([k, v]) => (
                <MiniKV key={k} k={k} v={String(v)} />
              ))}
            </div>
          </div>
        ) : null}

        {/* Notes (compact) */}
        {Array.isArray(x?.notes) && x.notes.length > 0 ? (
          <div style={{ marginTop: 12 }}>
            <div className="form-label">Notes</div>
            <ul style={{ margin: '6px 0 0 16px' }}>
              {x.notes.slice(0, 8).map((n, i) => (
                <li key={i} className="muted">
                  {n}
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        {/* Raw JSON (collapsed) */}
        <div style={{ marginTop: 12 }}>
          <details>
            <summary className="form-label" style={{ cursor: 'pointer' }}>
              Raw risk JSON
            </summary>
            <pre style={{ margin: '8px 0 0 0', maxHeight: 260, overflow: 'auto' }}>
              {JSON.stringify({ risk_details: r, xai: x }, null, 2)}
            </pre>
          </details>
        </div>
      </div>
    )
  }

  const renderEventRow = (e) => {
    const isOpen = expandedId === e.id
    const st = detailsById[e.id]
    const detail = st?.data
    const ex = detail?.explanations

    return (
      <React.Fragment key={e.id}>
        <tr>
          <td>#{e.id}</td>
          <td>{e.dataset}</td>
          <td className="muted mono">{e.batch_id}</td>
          <td className="muted">{(e.drift_types || []).join(', ') || '—'}</td>
          <td>
            <RiskBadge risk={e.risk_level} />
          </td>
          <td>
            <RouteBadge route={e.route} />
          </td>
          <td>
            <StatusBadge status={e.status} />
          </td>
          <td className="muted">{fmtTs(e.detected_at)}</td>
          <td style={{ textAlign: 'right' }}>
            <button className="btn btn-ghost" onClick={() => toggleDetail(e.id)}>
              {isOpen ? 'Hide' : 'Details'}
            </button>
          </td>
        </tr>

        {isOpen ? (
          <tr>
            <td colSpan={9} style={{ background: 'rgba(255,255,255,0.03)' }}>
              <div style={{ padding: 12 }}>
                {st?.loading ? (
                  <div className="muted">Loading details…</div>
                ) : st?.error ? (
                  <div className="muted">{st.error}</div>
                ) : !detail ? (
                  <div className="muted">No details.</div>
                ) : (
                  <div className="two-col" style={{ gap: 14 }}>
                    <div>
                      <div className="card" style={{ marginBottom: 12 }}>
                        <div className="card-header">
                          <div>
                            <div className="card-title">Detailed drift summary</div>
                            <div className="card-sub"></div>
                          </div>
                        </div>
                        <div className="card-body">
                          <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{detail.summary || '(no summary stored)'}</pre>
                        </div>
                      </div>

                      <div className="card">
                        <div className="card-header">
                          <div>
                            <div className="card-title">Explanations</div>
                            <div className="card-sub">Affected columns</div>
                          </div>
                        </div>
                        <div className="card-body">
                          {!ex ? (
                            <div className="muted">No structured explanations stored for this event.</div>
                          ) : (
                            <>
                              <div style={{ marginBottom: 10 }}>
                                <div className="form-label">RENAMED</div>
                                {renderList(ex.renamed, (x) => `${x.old} → ${x.new} (conf ${x.confidence ?? '—'})`)}
                              </div>
                              <div style={{ marginBottom: 10 }}>
                                <div className="form-label">NEW COLUMNS</div>
                                {renderList(ex.new_columns, (x) => x.column)}
                              </div>
                              <div style={{ marginBottom: 10 }}>
                                <div className="form-label">REMOVED COLUMNS</div>
                                {renderList(ex.removed_columns, (x) => x.column)}
                              </div>
                              <div style={{ marginBottom: 10 }}>
                                <div className="form-label">TYPE CHANGES</div>
                                {renderList(ex.type_changes, (x) => `${x.column}: ${x.from} → ${x.to}`)}
                              </div>
                              <div>
                                <div className="form-label">NULLABILITY</div>
                                {renderList(ex.nullable_changes, (x) => `${x.column}: ${x.from} → ${x.to}`)}
                              </div>
                            </>
                          )}
                        </div>
                      </div>
                    </div>

                    <div>
                      {/* ✅ ONE combined card */}
                      <div className="card" style={{ marginBottom: 12 }}>
                        <div className="card-header">
                          <div>
                            <div className="card-title">Risk breakdown</div>
                            <div className="card-sub"></div>
                          </div>
                        </div>
                        <div className="card-body">{renderRiskPanel(detail)}</div>
                      </div>

                     
                    </div>
                  </div>
                )}
              </div>
            </td>
          </tr>
        ) : null}
      </React.Fragment>
    )
  }

  return (
    <section className="section">
      <SectionHeader
        title="Drift Events"
        subtitle="Split into Pending decisions + Action taken (click Details to expand)"
        actions={
          <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            <span className="hint">
              Pending: <b>{pendingEvents.length}</b>
            </span>
            <span className="hint">
              Action taken: <b>{actionedEvents.length}</b>
            </span>
            <button className="btn btn-ghost" onClick={() => refreshEvents().catch(() => {})}>
              Refresh
            </button>
          </div>
        }
      />

      <div className="card">
        <div className="card-body no-pad">
          <table className="table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Dataset</th>
                <th>Batch</th>
                <th>Drift types</th>
                <th>Risk</th>
                <th>Route</th>
                <th>Status</th>
                <th>Detected</th>
                <th></th>
              </tr>
            </thead>

            <tbody>
              <tr>
                <td colSpan={9} className="table-group">
                  Pending decisions ({pendingEvents.length})
                </td>
              </tr>
              {pendingEvents.map(renderEventRow)}
              {pendingEvents.length === 0 ? (
                <tr>
                  <td colSpan={9} className="muted">
                    No pending drift events.
                  </td>
                </tr>
              ) : null}

              <tr>
                <td colSpan={9} className="table-group">
                  Action taken ({actionedEvents.length})
                </td>
              </tr>
              {actionedEvents.map(renderEventRow)}
              {actionedEvents.length === 0 ? (
                <tr>
                  <td colSpan={9} className="muted">
                    No actioned drift events yet.
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