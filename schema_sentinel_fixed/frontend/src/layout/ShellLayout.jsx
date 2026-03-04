import React, { useMemo } from 'react'
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom'
import { useApp } from '../context/AppContext'
import { cls } from '../utils/ui'
import { NAV, getRouteLabel } from './nav'
import ReviewModal from '../components/ReviewModal'
import Toast from '../components/Toast'

export default function ShellLayout() {
  const navigate = useNavigate()
  const loc = useLocation()

  const {
    streamConnected,
    kpis,
    pendingStaging,

    // review modal
    reviewOpen,
    reviewRow,
    reviewDetail,
    reviewLoading,
    reviewError,
    closeReview,
    doApprove,
    doReject,
    doRollback,

    // toast
    toast,
    setToast,
  } = useApp()

  const pageLabel = useMemo(() => getRouteLabel(loc.pathname), [loc.pathname])

  return (
    <div className="shell">
      <aside className="sidebar">
        <div
          className="sidebar-logo"
          onClick={() => navigate('/')}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') navigate('/')
          }}
        >
          <div className="logo-icon">Σ</div>
          <div>
            <div className="logo-text">SchemaSentinel</div>
            <div className="logo-sub">drift management platform</div>
          </div>
        </div>

        {NAV.map((group) => (
          <div className="nav-section" key={group.label}>
            <div className="nav-label">{group.label}</div>
            {group.items.map((it) => (
              <NavLink
                key={it.id}
                to={it.path}
                end={it.path === '/'}
                className={({ isActive }) => cls('nav-item', isActive && 'active')}
              >
                <span className="nav-ico">{it.icon}</span>
                <span>{it.text}</span>
                {it.id === 'drift' && kpis.pending > 0 ? <span className="nav-badge">{kpis.pending}</span> : null}
                {it.id === 'staging' && pendingStaging.length > 0 ? (
                  <span className="nav-badge">{pendingStaging.length}</span>
                ) : null}
              </NavLink>
            ))}
          </div>
        ))}

        <div className="sidebar-footer">
          <div className="small">Pipeline</div>
          <div style={{ marginTop: 8, display: 'flex', gap: 10, alignItems: 'center' }}>
            <span className={cls('pulse-dot', streamConnected ? 'pulse-good' : 'pulse-warn')}></span>
            <div className="small muted">{streamConnected ? 'SSE CONNECTED' : 'SSE DISCONNECTED'}</div>
          </div>
          <div className="small muted" style={{ marginTop: 8 }}>
            API: {import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/v1'}
          </div>
        </div>
      </aside>

      <main className="main">
        <div className="topbar">
          <div className="page-breadcrumb">
            SCHEMASENTINEL / <span>{String(pageLabel || '').toUpperCase()}</span>
          </div>
          <div className="topbar-right">
            <button className="btn btn-ghost" onClick={() => navigate('/datasets')}>
              Register Schema
            </button>
            <button className="btn btn-ghost" onClick={() => navigate('/risk')}>
              Risk Config
            </button>
            <button className="btn btn-primary" onClick={() => navigate('/stream')}>
              Simulate Batch
            </button>
          </div>
        </div>

        <Outlet />
      </main>

      <ReviewModal
        open={reviewOpen}
        eventRow={reviewRow}
        detail={reviewDetail}
        loading={reviewLoading}
        error={reviewError}
        onClose={closeReview}
        onApprove={doApprove}
        onReject={doReject}
        onRollback={doRollback}
      />

      <Toast toast={toast} onClose={() => setToast(null)} />
    </div>
  )
}
