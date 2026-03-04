import React from 'react'

export default function Toast({ toast, onClose }) {
  if (!toast) return null
  return (
    <div className="toast-container" role="status" aria-live="polite">
      <div className={"toast toast-" + (toast.type || 'info')}>
        <div className="toast-title">
          {toast.title || (toast.type === 'error' ? 'Error' : toast.type === 'success' ? 'Success' : 'Info')}
        </div>
        <div className="toast-desc">{toast.message}</div>
        <button className="toast-close" onClick={onClose} aria-label="Close">
          ✕
        </button>
      </div>
    </div>
  )
}
