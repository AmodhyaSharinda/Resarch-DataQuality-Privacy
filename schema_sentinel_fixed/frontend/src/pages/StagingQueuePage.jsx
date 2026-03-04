import React from 'react'
import SectionHeader from '../components/SectionHeader'
import { RiskBadge } from '../components/Badges'
import { useApp } from '../context/AppContext'

export default function StagingQueuePage() {
  const { pendingStaging, refreshEvents, openReview } = useApp()

  return (
    <section className="section">
      <SectionHeader
        title="Staging Queue"
        subtitle="Events routed to STAGING and pending decision"
        actions={
          <button className="btn btn-ghost" onClick={() => refreshEvents().catch(() => {})}>
            Refresh
          </button>
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
                <th>Risk</th>
                <th>Drift types</th>
                <th style={{ width: 220 }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {pendingStaging.map((e) => (
                <tr key={e.id}>
                  <td>#{e.id}</td>
                  <td>{e.dataset}</td>
                  <td className="muted mono">{e.batch_id}</td>
                  <td>
                    <RiskBadge risk={e.risk_level} />
                  </td>
                  <td className="muted">{(e.drift_types || []).join(', ') || '—'}</td>
                  <td style={{ textAlign: 'right' }}>
                    <button className="btn btn-primary" onClick={() => openReview(e)}>
                      Review
                    </button>
                  </td>
                </tr>
              ))}
              {pendingStaging.length === 0 ? (
                <tr>
                  <td colSpan={6} className="muted">
                    No pending staging events.
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
