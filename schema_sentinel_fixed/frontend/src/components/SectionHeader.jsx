import React from 'react'

export default function SectionHeader({ title, subtitle, actions }) {
  return (
    <div className="section-header">
      <div>
        <div className="section-title">{title}</div>
        {subtitle ? <div className="section-sub">{subtitle}</div> : null}
      </div>
      <div className="section-actions">{actions}</div>
    </div>
  )
}
