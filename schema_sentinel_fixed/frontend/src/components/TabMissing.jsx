import React from 'react'
import SectionHeader from './SectionHeader'

export default function TabMissing({ name }) {
  return (
    <section className="section">
      <SectionHeader title={name} subtitle="Component missing" actions={null} />
      <div className="card">
        <div className="card-body">
          <div className="muted">
            This tab is selected, but its React component is not defined in your current build. Restore the component code
            (or split into files).
          </div>
        </div>
      </div>
    </section>
  )
}
