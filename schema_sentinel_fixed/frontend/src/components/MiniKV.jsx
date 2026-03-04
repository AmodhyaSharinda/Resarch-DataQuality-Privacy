import React from 'react'

export default function MiniKV({ k, v }) {
  return (
    <div
      className="mini-kv"
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        gap: 12,
        padding: '6px 0',
        borderBottom: '1px solid rgba(255,255,255,0.06)',
      }}
    >
      <div className="muted">{k}</div>
      <div style={{ textAlign: 'right', fontFamily: 'DM Mono, monospace', fontSize: 12 }}>{v}</div>
    </div>
  )
}
