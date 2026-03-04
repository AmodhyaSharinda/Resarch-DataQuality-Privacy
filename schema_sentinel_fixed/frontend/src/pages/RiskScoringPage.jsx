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

  // Shared inputs
  const [sensitivityClass, setSensitivityClass] = useState('None')
  const [regStrictness, setRegStrictness] = useState('Light')
  const [keyFields, setKeyFields] = useState([])
  const [manualKeyFields, setManualKeyFields] = useState('')
  const [suggestedKeyFields, setSuggestedKeyFields] = useState([])

  // ✅ All dataset columns (for dropdown)
  const [allFields, setAllFields] = useState([])

  // Option A inputs
  const [datasetCriticalityA, setDatasetCriticalityA] = useState('Medium')

  // Option B inputs
  const [datasetCriticalityB, setDatasetCriticalityB] = useState(3) // 1..5
  const [fieldOverrides, setFieldOverrides] = useState({})
  const [addFieldName, setAddFieldName] = useState('')

  // Saved configs table
  const [savedTable, setSavedTable] = useState([])
  const [tableBusy, setTableBusy] = useState(false)

  const refreshFields = async (ds) => {
    if (!ds) {
      setSuggestedKeyFields([])
      setAllFields([])
      return
    }
    try {
      const res = await api.get(`/datasets/${ds}/fields`)
      const fields = Array.isArray(res.data?.fields) ? res.data.fields : []
      const suggested = Array.isArray(res.data?.suggested_key_fields) ? res.data.suggested_key_fields : []
      setAllFields(fields)
      setSuggestedKeyFields(suggested)
    } catch {
      setAllFields([])
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

      // A criticality
      setDatasetCriticalityA(cfg.dataset_criticality || 'Medium')

      // B criticality (in backend it becomes dataset_criticality int 1..5)
      const dcB = cfg.dataset_criticality
      const asInt = Number.isFinite(Number(dcB)) ? Number(dcB) : 3
      setDatasetCriticalityB(Math.max(1, Math.min(5, parseInt(asInt, 10) || 3)))

      // key fields
      setKeyFields(Array.isArray(cfg.key_fields) ? cfg.key_fields : [])
      setManualKeyFields('')

      // per-field overrides
      setFieldOverrides(typeof cfg.fields === 'object' && cfg.fields ? cfg.fields : {})
      setAddFieldName('')
    } catch (e) {
      showToast({ type: 'error', title: 'Failed to load config', message: fmtErr(e) })
    } finally {
      setBusy(false)
    }
  }

  const toggleKeyField = (f) => {
    setKeyFields((prev) => {
      const set = new Set(prev || [])
      if (set.has(f)) set.delete(f)
      else set.add(f)
      return Array.from(set)
    })
  }

  const addOverrideField = () => {
    const col = (addFieldName || '').trim()
    if (!col) return
    setFieldOverrides((prev) => {
      if (prev[col]) return prev
      return {
        ...prev,
        [col]: {
          field_criticality: 3,
          semantic_type: 'Dimension',
          tolerances: {
            allow_remove: false,
            allow_type_change: false,
            allow_rename: true,
            allow_nullable_change: true,
          },
        },
      }
    })
    setAddFieldName('')
  }

  const removeOverrideField = (col) => {
    setFieldOverrides((prev) => {
      const next = { ...(prev || {}) }
      delete next[col]
      return next
    })
  }

  const updateOverride = (col, patch) => {
    setFieldOverrides((prev) => ({
      ...(prev || {}),
      [col]: {
        ...(prev?.[col] || {}),
        ...patch,
      },
    }))
  }

  const updateTol = (col, k, v) => {
    setFieldOverrides((prev) => ({
      ...(prev || {}),
      [col]: {
        ...(prev?.[col] || {}),
        tolerances: {
          ...((prev?.[col] || {}).tolerances || {}),
          [k]: v,
        },
      },
    }))
  }

  const saveConfig = async () => {
    if (!dataset) {
      showToast({ type: 'error', title: 'Select dataset', message: 'Pick a dataset first.' })
      return
    }

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
            dataset_criticality_num: datasetCriticalityB,
            sensitivity_class: sensitivityClass,
            regulation_strictness: regStrictness,
            key_fields: mergedKeyFields,
            fields: fieldOverrides,
          }

    setBusy(true)
    try {
      await api.post(`/datasets/${dataset}/risk-config`, payload)
      showToast({ type: 'success', title: 'Saved', message: `Risk config saved for ${dataset}` })
      setKeyFields(mergedKeyFields)
      setManualKeyFields('')
      await refreshSavedTable()
      await loadConfigForDataset(dataset)
    } catch (e) {
      showToast({ type: 'error', title: 'Save failed', message: fmtErr(e) })
    } finally {
      setBusy(false)
    }
  }

  // auto select first dataset
  useEffect(() => {
    if (!dataset && (datasets || []).length > 0) {
      const first = datasets[0]?.name || ''
      if (first) {
        setDataset(first)
        loadConfigForDataset(first).catch(() => {})
        refreshFields(first).catch(() => {})
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasets])

  useEffect(() => {
    refreshSavedTable().catch(() => {})
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const usedOverrideFields = useMemo(() => new Set(Object.keys(fieldOverrides || {})), [fieldOverrides])

  const availableOverrideFields = useMemo(() => {
    const cols = Array.isArray(allFields) ? allFields : []
    return cols.filter((c) => !usedOverrideFields.has(c))
  }, [allFields, usedOverrideFields])

  const preview = useMemo(() => {
    if (mode === 'A') {
      return {
        mode: 'A',
        dataset_criticality: datasetCriticalityA,
        sensitivity_class: sensitivityClass,
        regulation_strictness: regStrictness,
        key_fields: keyFields,
      }
    }
    return {
      mode: 'B',
      dataset_criticality_num: datasetCriticalityB,
      sensitivity_class: sensitivityClass,
      regulation_strictness: regStrictness,
      key_fields: keyFields,
      fields: fieldOverrides,
    }
  }, [mode, datasetCriticalityA, datasetCriticalityB, sensitivityClass, regStrictness, keyFields, fieldOverrides])

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

      {/* Dataset + Mode */}
      <div className="two-col">
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Dataset</div>
              <div className="card-sub">Pick a dataset to configure risk mode + inputs</div>
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
                  refreshFields(v).catch(() => {})
                }}
              >
                <option value="">Select…</option>
                {(datasets || []).map((d) => (
                  <option key={d.name} value={d.name}>
                    {d.name}
                  </option>
                ))}
              </select>
              <div className="muted" style={{ marginTop: 8 }}>
                After saving, all new drift events use this config automatically.
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Mode selection</div>
              <div className="card-sub">Option A vs Option B</div>
            </div>
          </div>
          <div className="card-body">
            <label className="row" style={{ gap: 10, alignItems: 'center' }}>
              <input type="radio" checked={mode === 'A'} onChange={() => setMode('A')} />
              <div>
                <div style={{ fontWeight: 700 }}>Option A — Normal Risk</div>
                <div className="muted">Minimal inputs. Fast routing.</div>
              </div>
            </label>

            <div style={{ height: 10 }} />

            <label className="row" style={{ gap: 10, alignItems: 'center' }}>
              <input type="radio" checked={mode === 'B'} onChange={() => setMode('B')} />
              <div>
                <div style={{ fontWeight: 700 }}>Option B — Accurate Risk</div>
                <div className="muted">Per-field overrides + explainability.</div>
              </div>
            </label>
          </div>
        </div>
      </div>

      {/* -----------------------------
          How risk score is calculated
          ----------------------------- */}
      <div className="card" style={{ marginTop: 12 }}>
        <div className="card-header">
          <div>
            <div className="card-title">How the risk score is calculated</div>
            <div className="card-sub">So users know why inputs matter</div>
          </div>
        </div>

        <div className="card-body">
          <details open>
            <summary style={{ cursor: 'pointer', fontWeight: 700 }}>
              Mode A formula (used by your riskengine.py)
            </summary>

            <div className="muted" style={{ marginTop: 10, lineHeight: 1.6 }}>
              <div><b>T</b> (Technical impact): depends on drift type severity + number of changes</div>
              <div><b>C</b> (Dataset criticality): Low=0.3, Medium=0.6, High=0.9</div>
              <div><b>S</b> (Sensitivity): None=0, Internal=0.2, PII=0.5, Regulated=0.8</div>
              <div><b>K</b> (Key field touched): 1 if drift hits key_fields, else 0</div>
              <div><b>U</b> (Rename uncertainty): higher when rename confidence is low</div>

              <div style={{ marginTop: 10 }} className="mono">
                risk_base = clip(0.55*T + 0.20*C + 0.20*S + 0.05*K)
              </div>
              <div className="mono">
                risk_final = clip(risk_base + 0.10*U)
              </div>

              <div style={{ marginTop: 10 }}>
                If sensitivity is <b>Regulated</b>, the engine applies floors
                (e.g. deletion/type-change forces a minimum risk).
              </div>
            </div>
          </details>

          <div style={{ height: 10 }} />

          <details>
            <summary style={{ cursor: 'pointer', fontWeight: 700 }}>
              Mode B formula (Accurate)
            </summary>

            <div className="muted" style={{ marginTop: 10, lineHeight: 1.6 }}>
              Mode B calculates per-change contributions <span className="mono">r_i</span> using:
              severity, dataset/field criticality, semantic type weight, sensitivity multiplier,
              and penalties for disallowed operations.
              <div style={{ marginTop: 10 }} className="mono">
                risk = 1 - Π(1 - r_i)
              </div>
            </div>
          </details>

          <div style={{ height: 10 }} />

          <details>
            <summary style={{ cursor: 'pointer', fontWeight: 700 }}>
              What should I enter?
            </summary>

            <div className="muted" style={{ marginTop: 10, lineHeight: 1.6 }}>
              <ul style={{ marginTop: 0, paddingLeft: 18 }}>
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
            <div className="card-title">Inputs</div>
            <div className="card-sub">Saved per dataset</div>
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
              <div className="form-label">Dataset criticality (Option A)</div>
              <select className="form-input" value={datasetCriticalityA} onChange={(e) => setDatasetCriticalityA(e.target.value)}>
                <option>Low</option>
                <option>Medium</option>
                <option>High</option>
              </select>
            </div>
          ) : (
            <div className="form-group">
              <div className="form-label">Dataset criticality 1–5 (Option B)</div>
              <input
                className="form-input"
                type="number"
                min={1}
                max={5}
                value={datasetCriticalityB}
                onChange={(e) => setDatasetCriticalityB(Math.max(1, Math.min(5, parseInt(e.target.value || '3', 10))))}
              />
            </div>
          )}

          {/* Key fields */}
          <div className="card" style={{ marginTop: 12 }}>
            <div className="card-header">
              <div>
                <div className="card-title">Key fields (optional)</div>
                <div className="card-sub">If drift touches these, risk increases</div>
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
                {(suggestedKeyFields || []).length === 0 ? <div className="muted">No suggestions.</div> : null}
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

          {/* Option B overrides dropdown */}
          {mode === 'B' ? (
            <div className="card" style={{ marginTop: 12 }}>
              <div className="card-header">
                <div>
                  <div className="card-title">Field overrides (Option B)</div>
                  <div className="card-sub">Dropdown shows all dataset columns</div>
                </div>
              </div>
              <div className="card-body">
                <div className="row" style={{ gap: 10, alignItems: 'center' }}>
                  <select
                    className="form-input"
                    value={addFieldName}
                    onChange={(e) => setAddFieldName(e.target.value)}
                    style={{ flex: 1 }}
                  >
                    <option value="">
                      {availableOverrideFields.length ? 'Select field…' : 'No fields available'}
                    </option>
                    {availableOverrideFields.map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>

                  <button className="btn btn-primary" onClick={addOverrideField} disabled={!addFieldName}>
                    Add
                  </button>
                </div>

                {Object.keys(fieldOverrides || {}).length === 0 ? (
                  <div className="muted" style={{ marginTop: 10 }}>
                    No overrides yet.
                  </div>
                ) : (
                  <div style={{ marginTop: 12 }}>
                    {Object.entries(fieldOverrides).map(([col, cfg]) => (
                      <div key={col} className="card" style={{ marginBottom: 10 }}>
                        <div className="card-header">
                          <div className="mono" style={{ fontWeight: 700 }}>
                            {col}
                          </div>
                          <button className="btn btn-ghost" onClick={() => removeOverrideField(col)}>
                            Remove
                          </button>
                        </div>

                        <div className="card-body">
                          <div className="two-col">
                            <div className="form-group">
                              <div className="form-label">Field criticality (1–5)</div>
                              <input
                                className="form-input"
                                type="number"
                                min={1}
                                max={5}
                                value={cfg?.field_criticality ?? 3}
                                onChange={(e) =>
                                  updateOverride(col, {
                                    field_criticality: Math.max(1, Math.min(5, parseInt(e.target.value || '3', 10))),
                                  })
                                }
                              />
                            </div>

                            <div className="form-group">
                              <div className="form-label">Semantic type</div>
                              <select
                                className="form-input"
                                value={cfg?.semantic_type || 'Dimension'}
                                onChange={(e) => updateOverride(col, { semantic_type: e.target.value })}
                              >
                                <option value="Identifier">Identifier</option>
                                <option value="Timestamp">Timestamp</option>
                                <option value="Measure">Measure</option>
                                <option value="Dimension">Dimension</option>
                              </select>
                            </div>
                          </div>

                          <div className="muted" style={{ marginTop: 8 }}>
                            Tolerances:
                          </div>

                          <div style={{ marginTop: 8 }}>
                            <label style={{ display: 'block', marginBottom: 6 }}>
                              <input
                                type="checkbox"
                                checked={!!(cfg?.tolerances?.allow_remove)}
                                onChange={(e) => updateTol(col, 'allow_remove', e.target.checked)}
                                style={{ marginRight: 8 }}
                              />
                              Allow remove
                            </label>

                            <label style={{ display: 'block', marginBottom: 6 }}>
                              <input
                                type="checkbox"
                                checked={!!(cfg?.tolerances?.allow_type_change)}
                                onChange={(e) => updateTol(col, 'allow_type_change', e.target.checked)}
                                style={{ marginRight: 8 }}
                              />
                              Allow type change
                            </label>

                            <label style={{ display: 'block', marginBottom: 6 }}>
                              <input
                                type="checkbox"
                                checked={!!(cfg?.tolerances?.allow_rename)}
                                onChange={(e) => updateTol(col, 'allow_rename', e.target.checked)}
                                style={{ marginRight: 8 }}
                              />
                              Allow rename
                            </label>

                            <label style={{ display: 'block', marginBottom: 6 }}>
                              <input
                                type="checkbox"
                                checked={!!(cfg?.tolerances?.allow_nullable_change)}
                                onChange={(e) => updateTol(col, 'allow_nullable_change', e.target.checked)}
                                style={{ marginRight: 8 }}
                              />
                              Allow nullable change
                            </label>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                <div className="muted" style={{ marginTop: 12 }}>
                  Current config (preview):
                  <pre className="mono" style={{ marginTop: 6, whiteSpace: 'pre-wrap' }}>
                    {JSON.stringify(preview, null, 2)}
                  </pre>
                </div>
              </div>
            </div>
          ) : (
            <div className="muted" style={{ marginTop: 12 }}>
              Current config (preview):
              <pre className="mono" style={{ marginTop: 6, whiteSpace: 'pre-wrap' }}>
                {JSON.stringify(preview, null, 2)}
              </pre>
            </div>
          )}
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
                  <td>
                    <span className="badge badge-neutral">{String(r.mode || 'A')}</span>
                  </td>
                  <td className="muted">{r.dataset_criticality ?? '—'}</td>
                  <td className="muted">{r.sensitivity_class ?? '—'}</td>
                  <td className="muted">{r.regulation_strictness ?? '—'}</td>
                  <td className="muted">{r.key_fields_count ?? 0}</td>
                  <td className="muted">{r.updated_at ? fmtTs(r.updated_at) : '—'}</td>
                </tr>
              ))}
              {savedTable.length === 0 ? (
                <tr>
                  <td colSpan={7} className="muted">
                    No saved configs found.
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