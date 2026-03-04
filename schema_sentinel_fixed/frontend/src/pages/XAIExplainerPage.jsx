import React, { useEffect, useMemo, useState } from 'react'
import SectionHeader from '../components/SectionHeader'
import { useApp } from '../context/AppContext'
import { getEventDetails } from '../api'
import { cls } from '../utils/ui'

function Tabs({ tabs, active, onChange }) {
  return (
    <div className="tabs">
      {tabs.map((t) => (
        <button
          key={t}
          className={cls('tab', active === t && 'active')}
          onClick={() => onChange(t)}
          type="button"
        >
          {t}
        </button>
      ))}
    </div>
  )
}

function DonutGauge({ score, level }) {
  const pct = Math.max(0, Math.min(100, Number(score) || 0))
  const tone = String(level || '').toUpperCase() === 'HIGH' ? 'danger' : String(level || '').toUpperCase() === 'MEDIUM' ? 'warn' : 'good'
  return (
    <div className={cls('gauge', tone)}>
      <div className="gauge-ring" style={{ '--pct': `${pct}%` }}>
        <div className="gauge-inner">
          <div className="gauge-score">{pct.toFixed(1)}</div>
          <div className="gauge-label">{String(level || '').toUpperCase() || '—'} RISK</div>
        </div>
      </div>
    </div>
  )
}

function ContributionBars({ rows, kind }) {
  const maxMag = Math.max(0.0001, ...rows.map((r) => Math.abs(Number(r.magnitude ?? r.value ?? 0))))
  return (
    <div className="bars">
      {rows.map((r, idx) => {
        const val = Number(r.magnitude ?? r.value ?? 0) || 0
        const width = Math.round((Math.abs(val) / maxMag) * 100)
        const dir = r.direction || (val >= 0 ? '+' : '-')
        const tone = dir === '-' ? 'neg' : 'pos'
        const label = r.feature || r.name || r.key || 'feature'
        const detail = r.detail || r.type || ''
        return (
          <div key={idx} className="bar-row">
            <div className="bar-left">
              <div className="bar-name">{label}</div>
              {detail ? <div className="bar-sub">{detail}</div> : null}
            </div>
            <div className="bar-track">
              <div className={cls('bar-fill', tone)} style={{ width: `${width}%` }} />
            </div>
            <div className="bar-val">{(Number(val) || 0).toFixed(kind === 'score' ? 2 : 3)}</div>
          </div>
        )
      })}
    </div>
  )
}

function RiskBreakdown({ detail }) {
  const r = detail?.risk_details
  if (!r) return <div className="muted">No risk_details returned for this event.</div>

  const components = r.components || {}
  const entries = Object.entries(components)
    .filter(([k]) => k !== 'final')
    .map(([k, v]) => ({ key: k, value: Number(v) }))

  const final = components.final
  const mode = String(r.mode || '').toUpperCase()

  return (
    <div>
      <div className="hint" style={{ marginBottom: 10 }}>
        Risk engine mode: <span className="badge badge-neutral">{mode || '—'}</span>
        {final != null ? (
          <span className="muted"> · raw r = {(Number(final) || 0).toFixed(3)}</span>
        ) : null}
      </div>

      {entries.length ? (
        <div className="kv-list">
          {entries.map((e) => (
            <div key={e.key} className="kv">
              <div className="kv-k">{e.key}</div>
              <div className="kv-v">{Number.isFinite(e.value) ? e.value.toFixed(3) : String(e.value)}</div>
            </div>
          ))}
        </div>
      ) : (
        <div className="muted">No components.</div>
      )}

      {r.change_counts ? (
        <div style={{ marginTop: 12 }}>
          <div className="form-label">Change counts</div>
          <div className="kv-list" style={{ marginTop: 6 }}>
            {Object.entries(r.change_counts).map(([k, v]) => (
              <div key={k} className="kv">
                <div className="kv-k">{k}</div>
                <div className="kv-v">{String(v)}</div>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {r.reasons && Array.isArray(r.reasons) ? (
        <div style={{ marginTop: 12 }}>
          <div className="form-label">Why this score?</div>
          <ul style={{ margin: '8px 0 0 18px' }}>
            {r.reasons.slice(0, 6).map((x, idx) => (
              <li key={idx} className="muted">
                {x}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  )
}

function LimeRenames({ detail }) {
  const ren = detail?.renames
  const cands = ren?.candidates || []
  const mapping = ren?.mapping || {}

  if (!ren) return <div className="muted">No rename analysis stored for this event.</div>

  const top = [...cands].sort((a, b) => (b.probability || 0) - (a.probability || 0)).slice(0, 6)
  const mappingPairs = Object.entries(mapping)

  return (
    <div>
      {mappingPairs.length ? (
        <div className="card" style={{ marginBottom: 12 }}>
          <div className="card-header">
            <div>
              <div className="card-title">Rename mapping (selected)</div>
              <div className="card-sub">High-confidence mapping used by drift engine</div>
            </div>
          </div>
          <div className="card-body">
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
              {mappingPairs.map(([o, n]) => (
                <span key={`${o}-${n}`} className="badge badge-neutral">
                  {o} → {n}
                </span>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <div className="hint" style={{ marginBottom: 12 }}>
          No high-confidence mapping selected (either no rename drift, or confidence was below threshold).
        </div>
      )}

      <div className="grid-3">
        {top.map((c, idx) => {
          const f = c.features || {}
          const rows = [
            { key: 'name_similarity', value: f.name_similarity ?? f.token_jaccard ?? 0 },
            { key: 'embedding_sim', value: f.embedding_sim ?? f.semantic_sim ?? 0 },
            { key: 'type_match', value: f.type_match ?? 0 },
            { key: 'value_overlap', value: f.value_overlap ?? 0 },
          ]
          const conf = Number(c.probability || 0)
          return (
            <div key={idx} className="card">
              <div className="card-header">
                <div>
                  <div className="card-title" style={{ fontSize: 14 }}>
                    Pair: <span className="mono">{c.old}</span> → <span className="mono">{c.new}</span>
                  </div>
                  <div className="card-sub">LIME-style local explanation (feature values)</div>
                </div>
              </div>
              <div className="card-body">
                {rows.map((r) => (
                  <div key={r.key} className="lime-row">
                    <div className="lime-k">{r.key}</div>
                    <div className="lime-track">
                      <div className="lime-fill" style={{ width: `${Math.max(0, Math.min(100, (Number(r.value) || 0) * 100))}%` }} />
                    </div>
                    <div className="lime-v">{(Number(r.value) || 0).toFixed(2)}</div>
                  </div>
                ))}

                <div style={{ marginTop: 10 }}>
                  <span className={cls('badge', conf >= 0.85 ? 'badge-low' : conf >= 0.65 ? 'badge-med' : 'badge-high')}>
                    {conf >= 0.85 ? 'RENAME' : 'CHECK'} conf={conf.toFixed(2)}
                  </span>
                </div>

                <div className="hint" style={{ marginTop: 10 }}>
                  Explanation: high name + embedding similarity with type match makes rename likely. Low overlap or type mismatch reduces.
                </div>
              </div>
            </div>
          )
        })}

        {top.length === 0 ? <div className="muted">No rename candidates stored.</div> : null}
      </div>
    </div>
  )
}

export default function XAIExplainerPage() {
  const { events, showToast, fmtErr } = useApp()

  const [tab, setTab] = useState('SHAP Analysis')
  const [eventId, setEventId] = useState('')
  const [loading, setLoading] = useState(false)
  const [detail, setDetail] = useState(null)

  const options = useMemo(() => {
    return (events || []).slice(0, 200)
  }, [events])

  useEffect(() => {
    if (!eventId && options.length) setEventId(String(options[0].id))
  }, [eventId, options])

  useEffect(() => {
    if (!eventId) return
    ;(async () => {
      setLoading(true)
      try {
        const res = await getEventDetails(eventId)
        setDetail(res.data)
      } catch (e) {
        showToast({ type: 'error', title: 'Load event failed', message: fmtErr(e) })
        setDetail(null)
      } finally {
        setLoading(false)
      }
    })()
  }, [eventId, showToast, fmtErr])

  const shapRows = useMemo(() => {
    const x = detail?.xai
    if (!x) return []

    const mode = String(x.mode || '').toUpperCase()
    if (mode === 'A') {
      const drivers = Array.isArray(x.drivers) ? x.drivers : []
      return drivers.map((d) => ({ name: d.name, value: Number(d.value) || 0, direction: '+', type: d.type || '' }))
    }

    const top = Array.isArray(x.top_contributors) ? x.top_contributors : []
    return top.map((t) => ({
      feature: t.feature,
      magnitude: Number(t.magnitude) || 0,
      direction: t.direction || '+',
      detail: t.detail || '',
    }))
  }, [detail])

  return (
    <section className="section">
      <div className="xai-hero">
        <div>
          <div className="xai-title">XAI EXPLAINER</div>
          <div className="xai-sub">SHAP-inspired feature importance + LIME local explanations for drift decisions</div>
        </div>
        <div className="xai-hero-actions">
          <div className="form-group" style={{ margin: 0 }}>
            <div className="form-label">Event</div>
            <select className="form-input" value={eventId} onChange={(e) => setEventId(e.target.value)}>
              {options.map((e) => (
                <option key={e.id} value={e.id}>
                  #{e.id} · {e.dataset} · {e.batch_id}
                </option>
              ))}
              {options.length === 0 ? <option value="">No events</option> : null}
            </select>
          </div>
          <div className="badge badge-neutral">SHAP + LIME</div>
        </div>
      </div>

      <Tabs tabs={['SHAP Analysis', 'LIME Local', 'Human Explanations']} active={tab} onChange={setTab} />

      {loading ? <div className="muted" style={{ marginTop: 12 }}>
        Loading event details…
      </div> : null}

      {!loading && !detail ? <div className="muted" style={{ marginTop: 12 }}>
        Select an event to view explanations.
      </div> : null}

      {!loading && detail && tab === 'SHAP Analysis' ? (
        <div className="two-col" style={{ marginTop: 14 }}>
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">SHAP Feature Contributions</div>
                <div className="card-sub">Approximate additive contributions to risk score</div>
              </div>
            </div>
            <div className="card-body">
              {shapRows.length ? (
                <ContributionBars rows={shapRows} kind="score" />
              ) : (
                <div className="muted">No xai payload for this event.</div>
              )}
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Risk Breakdown</div>
                <div className="card-sub">Model score + component breakdown</div>
              </div>
            </div>
            <div className="card-body">
              <div className="xai-two">
                <DonutGauge score={detail.risk_score} level={detail.risk_level} />
                <div>
                  <RiskBreakdown detail={detail} />
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {!loading && detail && tab === 'LIME Local' ? (
        <div style={{ marginTop: 14 }}>
          <div className="card" style={{ marginBottom: 12 }}>
            <div className="card-body">
              <div className="muted">
                LIME perturbs features around each candidate pair and measures how the rename probability changes. In this
                prototype we display the <b>local feature values</b> stored by the rename engine, which is the most
                practical explanation for users reviewing rename decisions.
              </div>
            </div>
          </div>
          <LimeRenames detail={detail} />
        </div>
      ) : null}

      {!loading && detail && tab === 'Human Explanations' ? (
        <div style={{ marginTop: 14 }}>
          <div className="two-col">
            <div className="card">
              <div className="card-header">
                <div>
                  <div className="card-title">Summary</div>
                  <div className="card-sub">Narrative summary stored with the event</div>
                </div>
              </div>
              <div className="card-body">
                <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{detail.summary || '(no summary stored)'}</pre>
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <div>
                  <div className="card-title">Event metadata</div>
                  <div className="card-sub">What happened and why it was routed</div>
                </div>
              </div>
              <div className="card-body">
                <div className="kv-list">
                  <div className="kv">
                    <div className="kv-k">Dataset</div>
                    <div className="kv-v">{detail.dataset}</div>
                  </div>
                  <div className="kv">
                    <div className="kv-k">Batch</div>
                    <div className="kv-v mono">{detail.batch_id}</div>
                  </div>
                  <div className="kv">
                    <div className="kv-k">Drift types</div>
                    <div className="kv-v">{(detail.drift_types || []).join(', ') || '—'}</div>
                  </div>
                  <div className="kv">
                    <div className="kv-k">Risk</div>
                    <div className="kv-v">{detail.risk_level} ({Number(detail.risk_score || 0).toFixed(1)})</div>
                  </div>
                  <div className="kv">
                    <div className="kv-k">Route</div>
                    <div className="kv-v">{detail.route}</div>
                  </div>
                </div>

                <div className="hint" style={{ marginTop: 12 }}>
                  Tip: Use SHAP for <b>why risk is high</b>, use LIME for <b>why rename was chosen</b>.
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </section>
  )
}
