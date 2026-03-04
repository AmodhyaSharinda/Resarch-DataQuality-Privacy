import React from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import ShellLayout from './layout/ShellLayout'

import DashboardPage from './pages/DashboardPage'
import LiveStreamPage from './pages/LiveStreamPage'
import DriftEventsPage from './pages/DriftEventsPage'
import StagingQueuePage from './pages/StagingQueuePage'
import DatasetsPage from './pages/DatasetsPage'
import SchemaRegistryPage from './pages/SchemaRegistryPage'
import AuditLogPage from './pages/AuditLogPage'
import RiskScoringPage from './pages/RiskScoringPage'
import XAIExplainerPage from './pages/XAIExplainerPage'
import RenameEnginePage from './pages/RenameEnginePage'

export default function App() {
  return (
    <Routes>
      <Route element={<ShellLayout />}>
        <Route index element={<DashboardPage />} />
        <Route path="stream" element={<LiveStreamPage />} />
        <Route path="drift" element={<DriftEventsPage />} />
        <Route path="staging" element={<StagingQueuePage />} />

        <Route path="xai" element={<XAIExplainerPage />} />
        <Route path="risk" element={<RiskScoringPage />} />
        <Route path="rename" element={<RenameEnginePage />} />

        <Route path="registry" element={<SchemaRegistryPage />} />
        <Route path="audit" element={<AuditLogPage />} />
        <Route path="datasets" element={<DatasetsPage />} />

        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
