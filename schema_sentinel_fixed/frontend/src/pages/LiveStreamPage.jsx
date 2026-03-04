import React, { useState } from 'react'
import SectionHeader from '../components/SectionHeader'
import { useApp } from '../context/AppContext'
import { api } from '../api'

const cls = (...xs) => xs.filter(Boolean).join(' ')

export default function LiveStreamPage() {
  const {
    datasets,
    batches,
    refreshBatches,
    refreshEvents,
    showToast,
    fmtErr,
    setStreamLines,
    streamLines,
    streamConnected,
  } = useApp()

  const [batchDataset, setBatchDataset] = useState('')
  const [batchId, setBatchId] = useState('')
  const [batchFiles, setBatchFiles] = useState([])
  const [batchBusy, setBatchBusy] = useState(false)

  const [simulateBusy, setSimulateBusy] = useState(false)
  const [simulatePace, setSimulatePace] = useState(0)

  async function handleUploadBatch(e) {
    e.preventDefault()
    if (!batchDataset || batchFiles.length === 0) {
      showToast({ type: 'error', title: 'Missing fields', message: 'Dataset + at least one file are required.' })
      return
    }
    setBatchBusy(true)
    try {
      let okCount = 0
      for (const f of batchFiles) {
        const baseName = (f?.name || 'batch').replace(/\.[^.]+$/, '')
        const bid = (batchId || '').trim()
        const finalBatchId = batchFiles.length > 1 ? (bid ? `${bid}__${baseName}` : baseName) : bid || baseName

        const fd = new FormData()
        fd.append('dataset', batchDataset)
        fd.append('batch_id', finalBatchId)
        fd.append('file', f)
        await api.post('/stream/batches/upload', fd)
        okCount += 1
      }

      showToast({ type: 'success', title: 'Upload complete', message: `Stored ${okCount} batch file(s).` })
      setBatchId('')
      setBatchFiles([])
      await refreshBatches()
    } catch (e2) {
      showToast({ type: 'error', title: 'Upload failed', message: fmtErr(e2) })
    } finally {
      setBatchBusy(false)
    }
  }

  async function simulateBatch(storedBatchId) {
    setSimulateBusy(true)
    try {
      const res = await api.post(`/stream/batches/${storedBatchId}/simulate`, null, {
        params: { pace_ms: simulatePace || 0 },
      })
      showToast({ type: 'info', title: 'Simulation started', message: `Kafka topic: ${res.data?.topic}` })
    } catch (e2) {
      showToast({ type: 'error', title: 'Simulate failed', message: fmtErr(e2) })
    } finally {
      setSimulateBusy(false)
    }
  }

  return (
    <section className="section">
      <SectionHeader
        title="Live Stream"
        subtitle="Upload batches → replay to Kafka"
        actions={
          <>
            <button
              className="btn btn-ghost"
              onClick={() => {
                refreshBatches().catch(() => {})
                refreshEvents().catch(() => {})
              }}
            >
              Refresh
            </button>
            <button className="btn btn-ghost" onClick={() => setStreamLines([])}>
              Clear console
            </button>
          </>
        }
      />

      {/* Top: Upload + Stored batches */}
      <div className="two-col">
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Upload batch</div>
              <div className="card-sub">Saved in backend storage/batches</div>
            </div>
          </div>
          <div className="card-body">
            <form onSubmit={handleUploadBatch}>
              <div className="form-group">
                <div className="form-label">Dataset</div>
                <select className="form-input" value={batchDataset} onChange={(e) => setBatchDataset(e.target.value)}>
                  <option value="">Select dataset…</option>
                  {datasets.map((d) => (
                    <option key={d.name} value={d.name}>
                      {d.name}
                    </option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <div className="form-label">Batch ID</div>
                <input className="form-input" value={batchId} onChange={(e) => setBatchId(e.target.value)} placeholder="e.g., v1_add_cols" />
              </div>

              <div className="form-group">
                <div className="form-label">CSV / XLSX</div>
                <input
                  className="form-input"
                  type="file"
                  accept=".csv,.xlsx,.xls,.xlsm"
                  multiple
                  onChange={(e) => setBatchFiles(Array.from(e.target.files || []))}
                />
                {batchFiles.length ? <div className="hint">Selected: {batchFiles.map((f) => f.name).join(', ')}</div> : null}
              </div>

              <button className="btn btn-primary" disabled={batchBusy}>
                {batchBusy ? 'Uploading…' : 'Upload batch'}
              </button>
            </form>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Stored batches</div>
              <div className="card-sub">Simulate sends rows into Kafka</div>
            </div>
            <div className="card-actions">
              <div className="form-group" style={{ margin: 0 }}>
                <div className="form-label" style={{ marginBottom: 4 }}>
                  Pace (ms)
                </div>
                <input
                  className="form-input"
                  style={{ width: 120 }}
                  type="number"
                  min="0"
                  value={simulatePace}
                  onChange={(e) => setSimulatePace(Number(e.target.value || 0))}
                />
              </div>
            </div>
          </div>

          <div className="card-body no-pad">
            <table className="table">
              <thead>
                <tr>
                  <th>Dataset</th>
                  <th>Batch</th>
                  <th>Rows</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {batches.map((b) => (
                  <tr key={b.id}>
                    <td>{b.dataset}</td>
                    <td className="muted mono">{b.batch_id}</td>
                    <td className="muted">{b.row_count ?? '—'}</td>
                    <td style={{ textAlign: 'right' }}>
                      <button className="btn btn-primary" disabled={simulateBusy} onClick={() => simulateBatch(b.id)}>
                        {simulateBusy ? 'Simulating…' : 'Simulate'}
                      </button>
                    </td>
                  </tr>
                ))}
                {batches.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="muted">
                      No stored batches yet.
                    </td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Bottom: Kafka Console (same style as Dashboard) */}
      <div className="card" style={{ marginTop: 16 }}>
        <div className="card-header">
          <div>
            <div className="card-title">📡 Kafka Console</div>
            <div className="card-sub">SSE stream logs from backend</div>
          </div>
          <div className="card-actions">
            <span className={cls('pulse-dot', streamConnected ? 'pulse-good' : 'pulse-warn')} />
            <button className="btn btn-ghost" onClick={() => setStreamLines([])}>
              Clear
            </button>
          </div>
        </div>

        <div className="card-body">
          <div className="stream-window" style={{ maxHeight: 320 }}>
            {(streamLines || []).slice(-60).map((l, idx) => (
              <div className="stream-line" key={idx}>
                <span className="stream-ts">{l?.type || 'log'}</span>
                <span className="stream-msg">{l?.message ? l.message : JSON.stringify(l)}</span>
              </div>
            ))}
            {(streamLines || []).length === 0 ? <div className="muted">No console messages yet.</div> : null}
          </div>
        </div>
      </div>
    </section>
  )
}