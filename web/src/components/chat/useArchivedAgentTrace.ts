import { useCallback, useMemo, useState } from 'react'
import type { AgentTraceAuditResponse } from '../../api/chat'
import { asTraceRecord } from './messageTraceUtils'

export type ArchivedAgentTraceLoadStatus = 'idle' | 'loading' | 'loaded' | 'empty' | 'error'
export type LoadArchivedAgentTrace = (messageId: number) => Promise<AgentTraceAuditResponse>

function withAuditSummary(trace: Record<string, unknown>, summary: Record<string, unknown>) {
  if (Object.keys(trace).length <= 0 || Object.keys(summary).length <= 0 || Object.keys(asTraceRecord(trace.summary)).length > 0) {
    return trace
  }
  return { ...trace, summary }
}

export function useArchivedAgentTrace({
  trace,
  messageId,
  canLoadTrace,
  onLoadTrace,
}: {
  trace?: Record<string, unknown> | null
  messageId?: number
  canLoadTrace?: boolean
  onLoadTrace?: LoadArchivedAgentTrace
}) {
  const initialTrace = useMemo(() => asTraceRecord(trace), [trace])
  const [loadedState, setLoadedState] = useState<{
    messageId: number
    trace: Record<string, unknown> | null
    status: ArchivedAgentTraceLoadStatus
  }>({ messageId: 0, trace: null, status: 'idle' })

  const hasInitialTrace = Object.keys(initialTrace).length > 0
  const currentMessageId = Number(messageId || 0)
  const loadedTraceRecord = loadedState.messageId === currentMessageId ? asTraceRecord(loadedState.trace) : {}
  const loadStatus = loadedState.messageId === currentMessageId ? loadedState.status : 'idle'
  const traceRecord = hasInitialTrace ? initialTrace : loadedTraceRecord
  const hasTrace = Object.keys(traceRecord).length > 0
  const canLazyLoad = Boolean(!hasInitialTrace && canLoadTrace && onLoadTrace && currentMessageId > 0)

  const loadArchivedTrace = useCallback(async () => {
    if (!canLazyLoad || loadStatus === 'loading' || loadStatus === 'loaded') return
    if (!currentMessageId || !onLoadTrace) return
    setLoadedState({ messageId: currentMessageId, trace: null, status: 'loading' })
    try {
      const res = await onLoadTrace(currentMessageId)
      const loadedTrace = asTraceRecord(res.agent_trace)
      const nextTrace = withAuditSummary(loadedTrace, asTraceRecord(res.summary))
      if (res.available !== false && Object.keys(nextTrace).length > 0) {
        setLoadedState({ messageId: currentMessageId, trace: nextTrace, status: 'loaded' })
      } else {
        setLoadedState({ messageId: currentMessageId, trace: null, status: 'empty' })
      }
    } catch {
      setLoadedState({ messageId: currentMessageId, trace: null, status: 'error' })
    }
  }, [canLazyLoad, currentMessageId, loadStatus, onLoadTrace])

  return {
    traceRecord,
    hasTrace,
    canLazyLoad,
    loadStatus,
    loadArchivedTrace,
  }
}
