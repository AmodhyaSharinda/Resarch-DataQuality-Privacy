import React, { useEffect, useMemo, useState } from 'react'
import SectionHeader from '../components/SectionHeader'
import { api } from '../api'
import { useApp } from '../context/AppContext'
import { fmtTs } from '../utils/ui'

export default function SchemaRegistryPage() {
  const { datasets, showToast, fmtErr } = useApp()

  const [registryDataset, setRegistryDataset] = useState('')
  const [registry, setRegistry] = useState([]) // always keep as array
  const [busy, setBusy] = useState(false)

  const registryList = useMemo(() => (Array.isArray(registry) ? registry : []), [registry])

  const latestSchemaJson = useMemo(() => {
    const active = registryList.find((v) => v?.active) || registryList[0]
    if (!active) return '// select a dataset'

    // backend may send: schema (object) OR schema_json (string)
    let schemaObj = active.schema
    if (!schemaObj && active.schema_json) {
      try {
        schemaObj = JSON.parse(active.schema_json)
      } catch {
        schemaObj = { _raw_schema_json: String(active.schema_json) }
      }
    }
    return JSON.stringify(schemaObj || {}, null, 2)
  }, [registryList])

  const normalizeRegistryResponse = (data) => {
    // accept:
    // 1) [...]
    // 2) { versions: [...] }
    // 3) { data: [...] }
    // 4) { items: [...] }
    if (Array.isArray(data)) return data
    if (Array.isArray(data?.versions)) return data.versions
    if (Array.isArray(data?.data)) return data.data
    if (Array.isArray(data?.items)) return data.items
    return []
  }

  const refreshRegistry = async (dataset) => {
    if (!dataset) {
      setRegistry([])
      return
    }
    setBusy(true)
    try {
      const res = await api.get(`/registry/${dataset}`)
      const list = normalizeRegistryResponse(res.data)
      setRegistry(list)
    } catch (e) {
      showToast({ type: 'error', title: 'Registry fetch failed', message: fmtErr(e) })
      setRegistry([])
    } finally {
      setBusy(false)
    }
  }

  const activateRegistry = async (dataset, version) => {
    if (!dataset) return
    try {
      const res = await api.post(`/registry/${dataset}/activate/${version}`)
      showToast({ type: 'success', title: 'Activated', message: res.data?.message || 'OK' })
      await refreshRegistry(dataset)
    } catch (e) {
      showToast({ type: 'error', title: 'Activate failed', message: fmtErr(e) })
    }
  }

  useEffect(() => {
    if (!registryDataset && (datasets || []).length > 0) {
      const first = datasets[0]?.name || ''
      if (first) {
        setRegistryDataset(first)
        refreshRegistry(first).catch(() => {})
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasets])

  return (
    <section className="section">
      <SectionHeader
        title="Schema Registry"
        subtitle="Versioned schemas created on registration + drift candidates"
        actions={
          <button className="btn btn-ghost" onClick={() => refreshRegistry(registryDataset).catch(() => {})}>
            {busy ? 'Loading…' : 'Refresh'}
          </button>
        }
      />

      <div className="two-col">
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Select dataset</div>
              <div className="card-sub">Loads from /v1/registry/&lt;dataset&gt;</div>
            </div>
          </div>

          <div className="card-body">
            <div className="form-group">
              <div className="form-label">Dataset</div>
              <select
                className="form-input"
                value={registryDataset}
                onChange={(e) => {
                  const v = e.target.value
                  setRegistryDataset(v)
                  refreshRegistry(v).catch(() => {})
                }}
              >
                <option value="">Select…</option>
                {(datasets || []).map((d) => (
                  <option key={d.name} value={d.name}>
                    {d.name}
                  </option>
                ))}
              </select>
            </div>

            <div style={{ marginTop: 12 }}>
              <table className="table">
                <thead>
                  <tr>
                    <th>Version</th>
                    <th>Hash</th>
                    <th>Status</th>
                    <th>Created</th>
                    <th>Note</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {registryList.map((v, idx) => (
                    <tr key={v.id ?? `${v.version}-${idx}`}>
                      <td>v{v.version}</td>
                      <td className="muted mono">{String(v.schema_hash || '').slice(0, 12)}…</td>
                      <td>
                        {v.active ? (
                          <span className="badge badge-low">ACTIVE</span>
                        ) : (
                          <span className="badge badge-neutral">CANDIDATE</span>
                        )}
                      </td>
                      <td className="muted">{fmtTs(v.created_at)}</td>
                      <td className="muted">{v.note || '—'}</td>
                      <td style={{ textAlign: 'right' }}>
                        {!v.active ? (
                          <button className="btn btn-primary" onClick={() => activateRegistry(registryDataset, v.version)}>
                            Activate
                          </button>
                        ) : null}
                      </td>
                    </tr>
                  ))}

                  {registryDataset && registryList.length === 0 ? (
                    <tr>
                      <td colSpan={6} className="muted">
                        No versions found (or API returned empty).
                      </td>
                    </tr>
                  ) : null}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Schema JSON (latest)</div>
              <div className="card-sub">Shows ACTIVE (else newest)</div>
            </div>
          </div>
          <div className="card-body">
            <div className="stream-window" style={{ maxHeight: 460 }}>
              <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{latestSchemaJson}</pre>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}