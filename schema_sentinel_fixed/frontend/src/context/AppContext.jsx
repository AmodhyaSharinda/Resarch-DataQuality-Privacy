import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import {
  api,
  fmtErr,
  getStreamConsoleUrl,
  getEventDetails,
  approveAndPromote,
  rejectEvent,
  rollbackEvent,
} from '../api'

const AppCtx = createContext(null)

export function useApp() {
  const ctx = useContext(AppCtx)
  if (!ctx) throw new Error('useApp must be used within <AppProvider />')
  return ctx
}

export function AppProvider({ children }) {
  const [toast, setToast] = useState(null)
  const toastTimer = useRef(null)

  const showToast = useCallback((t) => {
    setToast(t)
    if (toastTimer.current) window.clearTimeout(toastTimer.current)
    toastTimer.current = window.setTimeout(() => setToast(null), 6000)
  }, [])

  const [datasets, setDatasets] = useState([])
  const [events, setEvents] = useState([])
  const [batches, setBatches] = useState([])

  const [streamConnected, setStreamConnected] = useState(false)
  const [streamLines, setStreamLines] = useState([])
  const streamRef = useRef(null)

  // review modal
  const [reviewOpen, setReviewOpen] = useState(false)
  const [reviewRow, setReviewRow] = useState(null)
  const [reviewLoading, setReviewLoading] = useState(false)
  const [reviewError, setReviewError] = useState('')
  const [reviewDetail, setReviewDetail] = useState(null)

  const refreshDatasets = useCallback(async () => {
    const res = await api.get('/datasets')
    setDatasets(res.data || [])
  }, [])

  const refreshEvents = useCallback(async () => {
    const res = await api.get('/events', { params: { limit: 200 } })
    setEvents(res.data || [])
  }, [])

  const refreshBatches = useCallback(async () => {
    const res = await api.get('/stream/batches')
    setBatches(res.data?.batches || [])
  }, [])

  // initial load
  useEffect(() => {
    ;(async () => {
      try {
        await refreshDatasets()
        await refreshEvents()
        await refreshBatches()
      } catch (e) {
        showToast({ type: 'error', title: 'Backend not reachable', message: fmtErr(e) })
      }
    })()
  }, [refreshDatasets, refreshEvents, refreshBatches, showToast])

  // SSE stream console
  useEffect(() => {
    const url = getStreamConsoleUrl()
    try {
      const es = new EventSource(url)
      streamRef.current = es

      es.onopen = () => setStreamConnected(true)
      es.onerror = () => setStreamConnected(false)
      es.onmessage = (evt) => {
        try {
          const obj = JSON.parse(evt.data)
          setStreamLines((prev) => {
            const next = [...prev, obj]
            return next.length > 400 ? next.slice(next.length - 400) : next
          })
        } catch {
          setStreamLines((prev) => {
            const next = [...prev, { type: 'log', message: evt.data }]
            return next.length > 400 ? next.slice(next.length - 400) : next
          })
        }
      }

      return () => {
        try {
          es.close()
        } catch {
          // ignore
        }
      }
    } catch {
      setStreamConnected(false)
    }
  }, [])

  const pendingStaging = useMemo(() => {
    return (events || []).filter(
      (e) =>
        String(e.route || '').toUpperCase() === 'STAGING' && String(e.status || '').toUpperCase() === 'PENDING'
    )
  }, [events])

  const pendingEvents = useMemo(() => {
    return (events || []).filter((e) => String(e.status || '').toUpperCase() === 'PENDING')
  }, [events])

  const actionedEvents = useMemo(() => {
    return (events || []).filter((e) => String(e.status || '').toUpperCase() !== 'PENDING')
  }, [events])

  const kpis = useMemo(() => {
    const totalEvents = events.length
    const high = events.filter((e) => String(e.risk_level || '').toUpperCase() === 'HIGH').length
    const pending = events.filter((e) => String(e.status || '').toUpperCase() === 'PENDING').length
    const staging = events.filter((e) => String(e.route || '').toUpperCase() === 'STAGING').length
    return { totalEvents, high, pending, staging }
  }, [events])

  const openReview = useCallback(async (eRow) => {
    setReviewRow(eRow)
    setReviewOpen(true)
    setReviewLoading(true)
    setReviewError('')
    setReviewDetail(null)
    try {
      const res = await getEventDetails(eRow.id)
      setReviewDetail(res.data)
    } catch (e) {
      setReviewError(fmtErr(e))
    } finally {
      setReviewLoading(false)
    }
  }, [])

  const closeReview = useCallback(() => setReviewOpen(false), [])

  const doApprove = useCallback(async () => {
    if (!reviewRow) return
    const approver = prompt('Approver name (for audit):', 'admin') || 'admin'
    const note = prompt('Optional note:', '') || ''
    try {
      await approveAndPromote(reviewRow.id, { approver, note })
      showToast({
        type: 'success',
        title: 'Approved & Promoted',
        message: `Event #${reviewRow.id} promoted to production.`,
      })
      setReviewOpen(false)
      await refreshEvents()
    } catch (e) {
      showToast({ type: 'error', title: 'Approve failed', message: fmtErr(e) })
    }
  }, [reviewRow, refreshEvents, showToast])

  const doReject = useCallback(async () => {
    if (!reviewRow) return
    const approver = prompt('Approver name:', 'admin') || 'admin'
    const reason = prompt('Reason for reject:', 'Schema change rejected. Please resend.') || 'Rejected'
    try {
      const notify_email = (prompt('Email to notify (optional):', '') || '').trim()
      await rejectEvent(reviewRow.id, { approver, reason, notify_email: notify_email || null })
      showToast({ type: 'success', title: 'Rejected', message: `Event #${reviewRow.id} rejected + alert triggered.` })
      setReviewOpen(false)
      await refreshEvents()
    } catch (e) {
      showToast({ type: 'error', title: 'Reject failed', message: fmtErr(e) })
    }
  }, [reviewRow, refreshEvents, showToast])

  const doRollback = useCallback(async () => {
    if (!reviewRow) return
    const approver = prompt('Approver name:', 'admin') || 'admin'
    const reason = prompt('Rollback note:', 'Rollback to previous schema; store data in old shape.') || 'Rollback'
    try {
      await rollbackEvent(reviewRow.id, { approver, reason })
      showToast({
        type: 'success',
        title: 'Rolled back',
        message: `Event #${reviewRow.id} rolled back + promoted under old schema.`,
      })
      setReviewOpen(false)
      await refreshEvents()
    } catch (e) {
      showToast({ type: 'error', title: 'Rollback failed', message: fmtErr(e) })
    }
  }, [reviewRow, refreshEvents, showToast])

  const value = useMemo(
    () => ({
      // notifications
      toast,
      setToast,
      showToast,
      fmtErr,

      // data
      datasets,
      events,
      batches,
      refreshDatasets,
      refreshEvents,
      refreshBatches,

      // stream console
      streamConnected,
      streamLines,
      setStreamLines,

      // derived
      pendingStaging,
      pendingEvents,
      actionedEvents,
      kpis,

      // review
      reviewOpen,
      reviewRow,
      reviewLoading,
      reviewError,
      reviewDetail,
      openReview,
      closeReview,
      doApprove,
      doReject,
      doRollback,
    }),
    [
      toast,
      showToast,
      datasets,
      events,
      batches,
      refreshDatasets,
      refreshEvents,
      refreshBatches,
      streamConnected,
      streamLines,
      pendingStaging,
      pendingEvents,
      actionedEvents,
      kpis,
      reviewOpen,
      reviewRow,
      reviewLoading,
      reviewError,
      reviewDetail,
      openReview,
      closeReview,
      doApprove,
      doReject,
      doRollback,
    ]
  )

  return <AppCtx.Provider value={value}>{children}</AppCtx.Provider>
}
