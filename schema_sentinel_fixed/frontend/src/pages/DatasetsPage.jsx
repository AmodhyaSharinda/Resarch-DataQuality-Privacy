import React, { useState } from 'react'
import SectionHeader from '../components/SectionHeader'
import { api } from '../api'
import { useApp } from '../context/AppContext'
import { fmtTs } from '../utils/ui'

export default function DatasetsPage() {
  const { datasets, refreshDatasets, showToast, fmtErr } = useApp()

  const [dsName, setDsName] = useState('')
  const [canonicalFile, setCanonicalFile] = useState(null)
  const [baselineFile, setBaselineFile] = useState(null)
  const [dsFileKey, setDsFileKey] = useState(0)
  const [dsBusy, setDsBusy] = useState(false)

  async function handleRegisterDataset(e) {
    e.preventDefault()
    if (!dsName || !canonicalFile) {
      showToast({
        type: 'error',
        title: 'Missing fields',
        message: 'Dataset name + canonical schema JSON are required.',
      })
      return
    }

    setDsBusy(true)
    try {
      const fd = new FormData()
      fd.append('dataset', dsName)
      fd.append('canonical_schema', canonicalFile)
      if (baselineFile) fd.append('baseline_csv', baselineFile)

      const res = await api.post('/datasets/register', fd)
      showToast({ type: 'success', title: 'Dataset saved', message: res.data?.message || 'OK' })

      setDsName('')
      setCanonicalFile(null)
      setBaselineFile(null)
      setDsFileKey((k) => k + 1)

      await refreshDatasets()
    } catch (e2) {
      showToast({ type: 'error', title: 'Register failed', message: fmtErr(e2) })
    } finally {
      setDsBusy(false)
    }
  }

  return (
    <section className="section">
      <SectionHeader
        title="Datasets"
        subtitle="Register canonical schema (+ optional baseline)"
        actions={
          <button className="btn btn-ghost" onClick={() => refreshDatasets().catch(() => {})}>
            Refresh
          </button>
        }
      />

      <div className="two-col">
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Registered datasets</div>
              <div className="card-sub">Loaded from /v1/datasets</div>
            </div>
          </div>
          <div className="card-body no-pad">
            <table className="table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Created</th>
                  <th>Baseline</th>
                </tr>
              </thead>
              <tbody>
                {datasets.map((d) => (
                  <tr key={d.name}>
                    <td>{d.name}</td>
                    <td className="muted">{fmtTs(d.created_at)}</td>
                    <td>
                      {d.has_baseline ? (
                        <span className="badge badge-low">YES</span>
                      ) : (
                        <span className="badge badge-neutral">NO</span>
                      )}
                    </td>
                  </tr>
                ))}
                {datasets.length === 0 ? (
                  <tr>
                    <td colSpan={3} className="muted">
                      No datasets registered yet.
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
              <div className="card-title">Register / Update dataset</div>
              <div className="card-sub">Uploads stored in backend storage/datasets</div>
            </div>
          </div>
          <div className="card-body">
            <form onSubmit={handleRegisterDataset}>
              <div className="form-group">
                <div className="form-label">Dataset name</div>
                <input
                  className="form-input"
                  value={dsName}
                  onChange={(e) => setDsName(e.target.value)}
                  placeholder="e.g., ecommerce"
                />
              </div>

              <div className="form-group">
                <div className="form-label">Canonical schema (JSON) — required</div>
                <input
                  key={`canon-${dsFileKey}`}
                  className="form-input"
                  type="file"
                  accept=".json"
                  onChange={(e) => setCanonicalFile(e.target.files?.[0] || null)}
                />
                {canonicalFile ? <div className="hint">Selected: {canonicalFile.name}</div> : null}
              </div>

              <div className="form-group">
                <div className="form-label">Baseline (CSV/XLSX) — optional</div>
                <input
                  key={`base-${dsFileKey}`}
                  className="form-input"
                  type="file"
                  accept=".csv,.xlsx,.xls,.xlsm"
                  onChange={(e) => setBaselineFile(e.target.files?.[0] || null)}
                />
                {baselineFile ? <div className="hint">Selected: {baselineFile.name}</div> : null}
              </div>

              <button className="btn btn-primary" disabled={dsBusy}>
                {dsBusy ? 'Saving…' : 'Register dataset'}
              </button>

              <div className="hint" style={{ marginTop: 10 }}>
                After registering, go to <b>Risk Scoring</b> to choose Option A/B and save inputs.
              </div>
            </form>
          </div>
        </div>
      </div>
    </section>
  )
}