import { useCallback, useMemo, useState } from 'react'
import type { AgentTraceAuditResponse } from '../../api/chat'
import {
  buildArchivedAgentTraceLoadedState,
  buildArchivedAgentTraceSnapshot,
  type ArchivedAgentTraceLoadedState,
} from './agentTraceArchiveState'

export type { ArchivedAgentTraceLoadStatus } from './agentTraceArchiveState'
export type LoadArchivedAgentTrace = (messageId: number) => Promise<AgentTraceAuditResponse>

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
  const [loadedState, setLoadedState] = useState<ArchivedAgentTraceLoadedState>({ messageId: 0, trace: null, status: 'idle' })
  const snapshot = useMemo(() => buildArchivedAgentTraceSnapshot({
    trace,
    loadedState,
    messageId,
    canLoadTrace,
    hasLoadHandler: Boolean(onLoadTrace),
  }), [canLoadTrace, loadedState, messageId, onLoadTrace, trace])

  const loadArchivedTrace = useCallback(async () => {
    if (!snapshot.canLazyLoad || snapshot.loadStatus === 'loading' || snapshot.loadStatus === 'loaded') return
    if (!snapshot.currentMessageId || !onLoadTrace) return
    setLoadedState({ messageId: snapshot.currentMessageId, trace: null, status: 'loading' })
    try {
      const res = await onLoadTrace(snapshot.currentMessageId)
      setLoadedState(buildArchivedAgentTraceLoadedState(snapshot.currentMessageId, res))
    } catch {
      setLoadedState({ messageId: snapshot.currentMessageId, trace: null, status: 'error' })
    }
  }, [onLoadTrace, snapshot.canLazyLoad, snapshot.currentMessageId, snapshot.loadStatus])

  return {
    traceRecord: snapshot.traceRecord,
    hasTrace: snapshot.hasTrace,
    canLazyLoad: snapshot.canLazyLoad,
    loadStatus: snapshot.loadStatus,
    loadArchivedTrace,
  }
}
