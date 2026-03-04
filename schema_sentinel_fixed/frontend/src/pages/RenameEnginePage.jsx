import React, { useEffect, useMemo, useState } from 'react'
import SectionHeader from '../components/SectionHeader'
import { api } from '../api'
import { useApp } from '../context/AppContext'
import { fmtTs } from '../utils/ui'

export default function RiskScoringPage() {
  const { datasets, showToast, fmtErr } = useApp()

  const [dataset, setDataset] = useState('')
  const [busy, setBusy] = useState(false)

  const [mode, setMode] = useState('A')

  // Mode A fields
  const [datasetCriticalityA, setDatasetCriticalityA] = useState('Medium')
  const [sensitivityClass, setSensitivityClass] = useState('None')
  const [regStrictness, setRegStrictness] = useState('Light')
  const [keyFields, setKeyFields] = useState([])
  const [manualKeyFields, setManualKeyFields] = useState('')

  const [suggestedKeyFields, setSuggestedKeyFields] = useState([])

  // Saved configs table (all datasets)
  const [savedTable, setSavedTable] = useState([])
  const [tableBusy, setTableBusy] = useState(false)

  // Display current saved config (for selected dataset)
  const savedPreview = useMemo(() => {
    return {
      mode,
      dataset_criticality: mode === 'A' ? datasetCriticalityA : undefined,
      dataset_criticality_num: mode === 'B' ? undefined : undefined,
      sensitivity_class: sensitivityClass,
      regulation_strictness: regStrictness,
      key_fields: keyFields,
    }
  }, [mode, datasetCriticalityA, sensitivityClass, regStrictness, keyFields])

  const refreshSuggested = async (ds) => {
    if (!ds) return
    try {
      const res = await api.get(`/datasets/${ds}/fields`)
      setSuggestedKeyFields(res.data?.suggested_key_fields || [])
    } catch {
      setSuggestedKeyFields([])
    }
  }

  const refreshSavedTable = async () => {
    setTableBusy(true)
    try {
      const res = await api.get('/risk-configs')
      setSavedTable(Array.isArray(res.data) ? res.data : [])
    } catch (e) {
      showToast({ type: 'error', title: 'Failed to load saved configs', message: fmtErr(e) })
      setSavedTable([])
    } finally {
      setTableBusy(false)
    }
  }

  const loadConfigForDataset = async (ds) => {
    if (!ds) return
    setBusy(true)
    try {
      const res = await api.get(`/datasets/${ds}/risk-config`)
      const cfg = res.data?.risk_config || {}
      const m = String(res.data?.mode || cfg.mode || 'A').toUpperCase()
      setMode(m === 'B' ? 'B' : 'A')

      setSensitivityClass(cfg.sensitivity_class || 'None')
      setRegStrictness(cfg.regulation_strictness || 'Light')

      if ((m === 'A')) setDatasetCriticalityA(cfg.dataset_criticality || 'Medium')

      const kf = Array.isArray(cfg.key_fields) ? cfg.key_fields : []
      setKeyFields(kf)

      setManualKeyFields('')
    } catch (e) {
      showToast({ type: 'error', title: 'Failed to load config', message: fmtErr(e) })
    } finally {
      setBusy(false)
    }
  }

  const saveConfig = async () => {
    if (!dataset) {
      showToast({ type: 'error', title: 'Select dataset', message: 'Pick a dataset first.' })
      return
    }

    // merge manual key fields
    const extra = manualKeyFields
      .split(',')
      .map((x) => x.trim())
      .filter(Boolean)

    const mergedKeyFields = Array.from(new Set([...(keyFields || []), ...extra]))

    const payload =
      mode === 'A'
        ? {
            mode: 'A',
            dataset_criticality: datasetCriticalityA,
            sensitivity_class: sensitivityClass,
            regulation_strictness: regStrictness,
            key_fields: mergedKeyFields,
          }
        : {
            mode: 'B',
            // keep your backend’s current B support minimal; you can expand later
            dataset_criticality: 3,
            sensitivity_class: sensitivityClass,
            regulation_strictness: regStrictness,
            key_fields: mergedKeyFields,
          }

    setBusy(true)
    try {
      await api.post(`/datasets/${dataset}/risk-config`, payload)
      showToast({ type: 'success', title: 'Saved', message: `Risk config saved for ${dataset}` })
      setKeyFields(mergedKeyFields)
      setManualKeyFields('')
      await refreshSavedTable()
    } catch (e) {
      showToast({ type: 'error', title: 'Save failed', message: fmtErr(e) })
    } finally {
      setBusy(false)
    }
  }

  // auto pick first dataset
  useEffect(() => {
    if (!dataset && (datasets || []).length > 0) {
      const first = datasets[0]?.name || ''
      setDataset(first)
      loadConfigForDataset(first).catch(() => {})
      refreshSuggested(first).catch(() => {})
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasets])

  useEffect(() => {
    refreshSavedTable().catch(() => {})
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const toggleKeyField = (f) => {
    setKeyFields((prev) => {
      const set = new Set(prev || [])
      if (set.has(f)) set.delete(f)
      else set.add(f)
      return Array.from(set)
    })
  }

  return (
    <section className="section">
      <SectionHeader
        title="Risk Scoring"
        subtitle="Configure how drift risk is calculated per dataset"
        actions={
          <button className="btn btn-primary" onClick={() => saveConfig().catch(() => {})}>
            {busy ? 'Saving…' : 'Save risk config'}
          </button>
        }
      />

      {/* Dataset selector + mode */}
      <div className="two-col">
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Dataset</div>
              <div className="card-sub">Choose dataset to configure</div>
            </div>
          </div>
          <div className="card-body">
            <div className="form-group">
              <div className="form-label">Dataset</div>
              <select
                className="form-input"
                value={dataset}
                onChange={(e) => {
                  const v = e.target.value
                  setDataset(v)
                  loadConfigForDataset(v).catch(() => {})
                  refreshSuggested(v).catch(() => {})
                }}
              >
                <option value="">Select…</option>
                {(datasets || []).map((d) => (
                  <option key={d.name} value={d.name}>
                    {d.name}
                  </option>
                ))}
              </select>
              <div className="muted" style={{ marginTop: 6 }}>
                After saving, all new drift events use this config automatically.
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Risk Mode</div>
              <div className="card-sub">A = simple/fast, B = detailed/per-field</div>
            </div>
          </div>
          <div className="card-body">
            <label className="row" style={{ gap: 10, alignItems: 'center' }}>
              <input type="radio" checked={mode === 'A'} onChange={() => setMode('A')} />
              <div>
                <div style={{ fontWeight: 600 }}>Option A — Normal Risk</div>
                <div className="muted">Uses T, C, S, K, U weights.</div>
              </div>
            </label>

            <div style={{ height: 10 }} />

            <label className="row" style={{ gap: 10, alignItems: 'center' }}>
              <input type="radio" checked={mode === 'B'} onChange={() => setMode('B')} />
              <div>
                <div style={{ fontWeight: 600 }}>Option B — Accurate Risk</div>
                <div className="muted">Per-field criticality + tolerances; union-of-risks.</div>
              </div>
            </label>
          </div>
        </div>
      </div>

      {/* Explanation */}
      <div className="card" style={{ marginTop: 12 }}>
        <div className="card-header">
          <div>
            <div className="card-title">How the risk score is calculated</div>
            <div className="card-sub">So users know why inputs matter</div>
          </div>
        </div>
        <div className="card-body">
          <details open>
            <summary style={{ cursor: 'pointer', fontWeight: 700 }}>Mode A formula (used by your riskengine.py)</summary>
            <div style={{ marginTop: 10 }} className="muted">
              <div><b>T</b> (Technical impact): depends on drift type severity + number of changes</div>
              <div><b>C</b> (Dataset criticality): Low=0.3, Medium=0.6, High=0.9</div>
              <div><b>S</b> (Sensitivity): None=0, Internal=0.2, PII=0.5, Regulated=0.8</div>
              <div><b>K</b> (Key field touched): 1 if drift hits key_fields, else 0</div>
              <div><b>U</b> (Rename uncertainty): higher when rename confidence is low</div>

              <div style={{ marginTop: 10 }}>
                <span className="mono">
                  risk_base = clip(0.55*T + 0.20*C + 0.20*S + 0.05*K)
                </span>
              </div>
              <div>
                <span className="mono">risk_final = clip(risk_base + 0.10*U)</span>
              </div>
              <div style={{ marginTop: 10 }}>
                If sensitivity is <b>Regulated</b>, the engine applies floors (e.g. deletion/type-change forces a minimum risk).
              </div>
            </div>
          </details>

          <div style={{ height: 10 }} />

          <details>
            <summary style={{ cursor: 'pointer', fontWeight: 700 }}>Mode B formula (Accurate)</summary>
            <div style={{ marginTop: 10 }} className="muted">
              Mode B calculates per-change contributions <span className="mono">r_i</span> using severity, dataset/field criticality,
              semantic type weight, sensitivity multiplier, and penalties for disallowed operations.
              <div style={{ marginTop: 10 }}>
                Final score combines them using:
                <div className="mono">risk = 1 - Π(1 - r_i)</div>
              </div>
            </div>
          </details>

          <div style={{ height: 10 }} />

          <details>
            <summary style={{ cursor: 'pointer', fontWeight: 700 }}>What should I enter?</summary>
            <div style={{ marginTop: 10 }} className="muted">
              <ul style={{ marginTop: 0 }}>
                <li><b>Dataset criticality</b>: choose High for business-critical pipelines (finance, billing, compliance).</li>
                <li><b>Sensitivity class</b>: choose PII/Regulated if data contains personal or regulated attributes.</li>
                <li><b>Regulation strictness</b>: Strict only if you must enforce hard floors for regulated systems.</li>
                <li><b>Key fields</b>: add IDs, timestamps, amounts, primary business identifiers (drift touching these increases risk).</li>
              </ul>
            </div>
          </details>
        </div>
      </div>

      {/* Inputs */}
      <div className="card" style={{ marginTop: 12 }}>
        <div className="card-header">
          <div>
            <div className="card-title">Inputs (saved per dataset)</div>
            <div className="card-sub">These are the fields used by the risk engine</div>
          </div>
        </div>

        <div className="card-body">
          <div className="two-col">
            <div className="form-group">
              <div className="form-label">Sensitivity class</div>
              <select className="form-input" value={sensitivityClass} onChange={(e) => setSensitivityClass(e.target.value)}>
                <option>None</option>
                <option>Internal</option>
                <option>PII</option>
                <option>Regulated</option>
              </select>
            </div>

            <div className="form-group">
              <div className="form-label">Regulation strictness (if regulated)</div>
              <select className="form-input" value={regStrictness} onChange={(e) => setRegStrictness(e.target.value)}>
                <option>Light</option>
                <option>Strict</option>
              </select>
            </div>
          </div>

          {mode === 'A' ? (
            <div className="form-group">
              <div className="form-label">Dataset criticality (Mode A)</div>
              <select className="form-input" value={datasetCriticalityA} onChange={(e) => setDatasetCriticalityA(e.target.value)}>
                <option>Low</option>
                <option>Medium</option>
                <option>High</option>
              </select>
            </div>
          ) : (
            <div className="muted" style={{ marginTop: 6 }}>
              Mode B supports per-field controls (next step). Currently it reuses sensitivity/strictness and will still score drift.
            </div>
          )}

          <div className="card" style={{ marginTop: 12 }}>
            <div className="card-header">
              <div>
                <div className="card-title">Key fields (optional)</div>
                <div className="card-sub">If drift touches these fields, risk increases (K=1 in Mode A)</div>
              </div>
            </div>
            <div className="card-body">
              <div className="muted">Suggested key fields</div>
              <div style={{ marginTop: 8 }}>
                {(suggestedKeyFields || []).map((f) => (
                  <label key={f} style={{ display: 'block', marginBottom: 6 }}>
                    <input
                      type="checkbox"
                      checked={(keyFields || []).includes(f)}
                      onChange={() => toggleKeyField(f)}
                      style={{ marginRight: 8 }}
                    />
                    <span className="mono">{f}</span>
                  </label>
                ))}
                {(suggestedKeyFields || []).length === 0 ? <div className="muted">No suggestions available.</div> : null}
              </div>

              <div style={{ marginTop: 12 }}>
                <div className="muted">Add more key fields (comma-separated)</div>
                <input
                  className="form-input"
                  value={manualKeyFields}
                  onChange={(e) => setManualKeyFields(e.target.value)}
                  placeholder="e.g., customer_id,timestamp,amount"
                />
              </div>
            </div>
          </div>

          <div className="muted" style={{ marginTop: 12 }}>
            Current config (preview):
            <pre className="mono" style={{ marginTop: 6, whiteSpace: 'pre-wrap' }}>
              {JSON.stringify(savedPreview, null, 2)}
            </pre>
          </div>
        </div>
      </div>

      {/* Saved configs table */}
      <div className="card" style={{ marginTop: 12 }}>
        <div className="card-header">
          <div>
            <div className="card-title">Saved risk configurations</div>
            <div className="card-sub">All datasets (loaded from DB)</div>
          </div>
          <button className="btn btn-ghost" onClick={() => refreshSavedTable().catch(() => {})}>
            {tableBusy ? 'Loading…' : 'Refresh'}
          </button>
        </div>

        <div className="card-body no-pad">
          <table className="table">
            <thead>
              <tr>
                <th>Dataset</th>
                <th>Mode</th>
                <th>Criticality</th>
                <th>Sensitivity</th>
                <th>Strictness</th>
                <th>Key fields</th>
                <th>Updated</th>
              </tr>
            </thead>
            <tbody>
              {savedTable.map((r, idx) => (
                <tr key={`${r.dataset}-${idx}`}>
                  <td className="mono">{r.dataset}</td>
                  <td><span className="badge badge-neutral">{String(r.mode || 'A')}</span></td>
                  <td className="muted">{r.dataset_criticality ?? '—'}</td>
                  <td className="muted">{r.sensitivity_class ?? '—'}</td>
                  <td className="muted">{r.regulation_strictness ?? '—'}</td>
                  <td className="muted">{r.key_fields_count ?? 0}</td>
                  <td className="muted">{r.updated_at ? fmtTs(r.updated_at) : '—'}</td>
                </tr>
              ))}

              {savedTable.length === 0 ? (
                <tr>
                  <td colSpan={7} className="muted">No saved configs found.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  )
}