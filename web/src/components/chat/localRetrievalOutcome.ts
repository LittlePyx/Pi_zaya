import type { Message } from '../../api/chat'
import { asTraceRecord, getMessageResearchTrace } from './messageTraceUtils'

const NON_FALLBACK_REASONS = new Set([
  'local_semantic_scope_not_eligible',
  'local_semantic_not_promoted',
  'structured_exact_preserved',
  'structured_section_preserved',
])

export function hasAutomaticLocalSemanticFallback(trace: unknown): boolean {
  const traceRecord = asTraceRecord(trace)
  const retrieval = asTraceRecord(traceRecord.retrieval)
  if (retrieval.bypassed === true) return false
  const outcome = asTraceRecord(retrieval.local_retrieval)
  const requestedMode = String(outcome.requested_mode || '').trim().toLowerCase()
  const usedMode = String(outcome.used_mode || '').trim().toLowerCase()
  const reason = String(outcome.fallback_reason_id || '').trim().toLowerCase()
  return requestedMode === 'hybrid_local'
    && usedMode === 'bm25'
    && Boolean(reason)
    && !NON_FALLBACK_REASONS.has(reason)
    && !reason.endsWith('_preserved')
}

export function messageHasAutomaticLocalSemanticFallback(message: Message): boolean {
  return hasAutomaticLocalSemanticFallback(getMessageResearchTrace(message))
}
