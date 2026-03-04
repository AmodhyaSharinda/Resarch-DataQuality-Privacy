import React, { useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { useApp } from '../context/AppContext'
import { RiskBadge, RouteBadge, StatusBadge } from '../components/Badges'
import { cls, fmtTs } from '../utils/ui'

function KPI({ label, value, delta, icon, tone }) {
  return (
    <div className={cls('kpi-card', tone)}>
      <div>
        <div className="kpi-label">{label}</div>
        <div className="kpi-value">{value}</div>
        <div className="kpi-delta">{delta}</div>
      </div>
      <div className="kpi-icon">{icon}</div>
    </div>
  )
}

function PipelineNode({ icon, title, status, tone, onClick }) {
  return (
    <div
      className={cls('pipe-node', tone)}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onClick={onClick}
    >
      <div className="pipe-icon">{icon}</div>
      <div className="pipe-title">{title}</div>
      <div className="pipe-status">{status}</div>
    </div>
  )
}

export default function DashboardPage() {
  const navigate = useNavigate()
  const {
    datasets,
    events,
    batches,
    kpis,
    pendingStaging,
    streamConnected,
    setStreamLines,
    refreshEvents,
    openReview,
  } = useApp()

  const promotedCount = useMemo(() => {
    return (events || []).filter((e) => {
      const s = String(e.status || '').toUpperCase()
      return s === 'APPROVED' || s === 'PROMOTED'
    }).length
  }, [events])

  const recentActioned = useMemo(() => {
    const arr = (events || []).filter((e) => String(e.status || '').toUpperCase() !== 'PENDING')
    return arr
      .slice()
      .sort((a, b) => String(b.detected_at || '').localeCompare(String(a.detected_at || '')))
      .slice(0, 6)
  }, [events])

  return (
    <section className="section">
      <div className="page-hero">
        <div>
          <div className="page-hero-title">SYSTEM OVERVIEW</div>
          <div className="page-hero-sub">Real-time schema drift monitoring · API-backed · explainable risk scoring</div>
        </div>
        <div className="page-hero-actions">
          <div className="pill">
            <span className="pill-ico">🗃️</span>
            <span>{batches.length} batches processed</span>
          </div>
          <button className="btn btn-ghost" onClick={() => refreshEvents().catch(() => {})}>
            Refresh
          </button>
        </div>
      </div>

      <div className="kpi-grid">
        <KPI label="Total drift events" value={kpis.totalEvents} delta="Last 200" icon="⚠️" tone="danger" />
        <KPI label="In staging queue" value={pendingStaging.length} delta="Awaiting review" icon="🧪" tone="warn" />
        <KPI label="To production" value={promotedCount} delta="Approved / promoted" icon="🚀" tone="good" />
        <KPI label="Datasets" value={datasets.length} delta="Registered" icon="🗂️" tone="info" />
      </div>

      <div className="two-col" style={{ marginTop: 16 }}>
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Recent Drift Events</div>
              <div className="card-sub">Latest events from backend</div>
            </div>
            <div className="card-actions">
              <button className="btn btn-ghost" onClick={() => navigate('/drift')}>
                View All →
              </button>
            </div>
          </div>
          <div className="card-body no-pad">
            <table className="table">
              <thead>
                <tr>
                  <th>Batch</th>
                  <th>Type</th>
                  <th>Risk</th>
                  <th>Route</th>
                  <th>Status</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {(events || []).slice(0, 6).map((e) => (
                  <tr key={e.id}>
                    <td className="mono">{e.batch_id}</td>
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
                    <td style={{ textAlign: 'right' }}>
                      <button className="btn btn-ghost" onClick={() => openReview(e)}>
                        Review
                      </button>
                    </td>
                  </tr>
                ))}
                {events.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="muted">
                      No drift events yet.
                    </td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Activity Timeline</div>
              <div className="card-sub">Latest decisions & actions</div>
            </div>
            <div className="card-actions">
              <span className={cls('pulse-dot', streamConnected ? 'pulse-good' : 'pulse-warn')} />
              <button className="btn btn-ghost" onClick={() => setStreamLines([])}>
                Clear console
              </button>
            </div>
          </div>
          <div className="card-body">
            {recentActioned.length === 0 ? (
              <div className="muted">No actions yet.</div>
            ) : (
              <div className="timeline">
                {recentActioned.map((e) => {
                  const st = String(e.status || '').toUpperCase()
                  const tone =
                    st.includes('REJECT') ? 'danger' : st === 'APPROVED' || st === 'PROMOTED' ? 'good' : 'info'
                  const ico = st.includes('REJECT') ? '⛔' : st === 'APPROVED' || st === 'PROMOTED' ? '✅' : 'ℹ️'
                  return (
                    <div key={e.id} className={cls('timeline-item', tone)}>
                      <div className="timeline-ico">{ico}</div>
                      <div>
                        <div className="timeline-title">
                          {st.toLowerCase()} · <span className="mono">{e.batch_id}</span>
                        </div>
                        <div className="timeline-sub">{fmtTs(e.detected_at)}</div>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Pipeline flow (XAI node removed) */}
      <div className="card" style={{ marginTop: 16 }}>
        <div className="card-header">
          <div>
            <div className="card-title">Data Pipeline Flow</div>
          </div>
        </div>
        <div className="card-body">
          <div className="pipeline">
            <PipelineNode icon="☁️" title="Data Sources" status={streamConnected ? 'LIVE' : 'OFFLINE'} tone={streamConnected ? 'good' : 'warn'} />
            <div className="pipe-arrow">→</div>
            <PipelineNode
              icon="📡"
              title="Kafka Stream"
              status={streamConnected ? 'CONSUMING' : 'DISCONNECTED'}
              tone={streamConnected ? 'good' : 'warn'}
              onClick={() => navigate('/stream')}
            />
            <div className="pipe-arrow">→</div>
            <PipelineNode icon="🔎" title="Schema Extract" status="PROFILING" tone="info" />
            <div className="pipe-arrow">→</div>
            <PipelineNode
              icon="⚠️"
              title="Drift Engine"
              status={`${kpis.totalEvents} EVENTS`}
              tone={kpis.pending > 0 ? 'warn' : 'good'}
              onClick={() => navigate('/drift')}
            />
            <div className="pipe-arrow">→</div>
            <PipelineNode icon="🎯" title="Risk Scorer" status="RUNNING" tone="info" onClick={() => navigate('/risk')} />
            <div className="pipe-arrow">→</div>
            <PipelineNode
              icon="🧪"
              title="Staging"
              status={`${pendingStaging.length} QUEUED`}
              tone={pendingStaging.length > 0 ? 'warn' : 'good'}
              onClick={() => navigate('/staging')}
            />
            <div className="pipe-arrow">→</div>
            <PipelineNode icon="✅" title="Production" status="HEALTHY" tone="good" />
          </div>
        </div>
      </div>
    </section>
  )
}