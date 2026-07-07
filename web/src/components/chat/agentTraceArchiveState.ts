import type { AgentTraceAuditResponse } from '../../api/chat'
import { asTraceRecord } from './messageTraceUtils'

export type ArchivedAgentTraceLoadStatus = 'idle' | 'loading' | 'loaded' | 'empty' | 'error'

export type ArchivedAgentTraceLoadedState = {
  messageId: number
  trace: Record<string, unknown> | null
  status: ArchivedAgentTraceLoadStatus
}

export type ArchivedAgentTraceSnapshot = {
  currentMessageId: number
  traceRecord: Record<string, unknown>
  hasInitialTrace: boolean
  hasTrace: boolean
  canLazyLoad: boolean
  loadStatus: ArchivedAgentTraceLoadStatus
}

export type ArchivedAgentTraceCanLoadInput = {
  hasInitialTrace: boolean
  canLoadTrace?: boolean
  hasLoadHandler: boolean
  currentMessageId: number
}

export type ArchivedAgentTraceSnapshotInput = {
  trace?: Record<string, unknown> | null
  loadedState: ArchivedAgentTraceLoadedState
  messageId?: number
  canLoadTrace?: boolean
  hasLoadHandler: boolean
}

export function mergeAgentTraceAuditSummary(
  trace: Record<string, unknown>,
  summary: Record<string, unknown>,
) {
  if (
    Object.keys(trace).length <= 0
    || Object.keys(summary).length <= 0
    || Object.keys(asTraceRecord(trace.summary)).length > 0
  ) {
    return trace
  }
  return { ...trace, summary }
}

export function canLoadArchivedAgentTrace({
  hasInitialTrace,
  canLoadTrace,
  hasLoadHandler,
  currentMessageId,
}: ArchivedAgentTraceCanLoadInput) {
  return Boolean(!hasInitialTrace && canLoadTrace && hasLoadHandler && currentMessageId > 0)
}

export function buildArchivedAgentTraceSnapshot({
  trace,
  loadedState,
  messageId,
  canLoadTrace,
  hasLoadHandler,
}: ArchivedAgentTraceSnapshotInput): ArchivedAgentTraceSnapshot {
  const initialTrace = asTraceRecord(trace)
  const hasInitialTrace = Object.keys(initialTrace).length > 0
  const currentMessageId = Number(messageId || 0)
  const loadedTraceRecord = loadedState.messageId === currentMessageId ? asTraceRecord(loadedState.trace) : {}
  const loadStatus = loadedState.messageId === currentMessageId ? loadedState.status : 'idle'
  const traceRecord = hasInitialTrace ? initialTrace : loadedTraceRecord
  const hasTrace = Object.keys(traceRecord).length > 0
  const canLazyLoad = canLoadArchivedAgentTrace({
    hasInitialTrace,
    canLoadTrace,
    hasLoadHandler,
    currentMessageId,
  })

  return {
    currentMessageId,
    traceRecord,
    hasInitialTrace,
    hasTrace,
    canLazyLoad,
    loadStatus,
  }
}

export function buildArchivedAgentTraceLoadedState(
  messageId: number,
  response: AgentTraceAuditResponse,
): ArchivedAgentTraceLoadedState {
  const loadedTrace = asTraceRecord(response.agent_trace)
  const nextTrace = mergeAgentTraceAuditSummary(loadedTrace, asTraceRecord(response.summary))
  if (response.available !== false && Object.keys(nextTrace).length > 0) {
    return { messageId, trace: nextTrace, status: 'loaded' }
  }
  return { messageId, trace: null, status: 'empty' }
}
