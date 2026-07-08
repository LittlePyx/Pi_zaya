import type { CiteDetail } from './citationState'
import { cleanCitationDisplayText } from './citationState'
import {
  SYSTEM_B_TRACE_ENABLED,
  compact,
} from './citationPopoverUtils'

interface SystemBTraceStrings extends Record<string, string> {
  cite_evidence_chain: string
  cite_trace_complete: string
  cite_trace_review: string
}

export interface BuildSystemBTraceModelOptions {
  detail: CiteDetail
  S: SystemBTraceStrings
  isSystemB: boolean
  traceEnabled?: boolean
}

export interface SystemBTraceModel {
  showTrace: boolean
  traceStatus: { label: string; tone: string }
  traceScore: number
  traceSteps: string[]
  traceReason: string
  traceLabel: string
}

export function buildSystemBTraceModel({
  detail,
  S,
  isSystemB,
  traceEnabled = SYSTEM_B_TRACE_ENABLED,
}: BuildSystemBTraceModelOptions): SystemBTraceModel {
  const traceSteps = isSystemB && Array.isArray(detail.systemBTraceSteps)
    ? detail.systemBTraceSteps.map((item) => compact(item)).filter(Boolean)
    : []
  const traceReason = isSystemB ? cleanCitationDisplayText(detail.systemBTraceReason) : ''
  const traceScore = Number(detail.systemBTraceScore || 0)
  const showTrace = Boolean(
    traceEnabled
    && isSystemB
    && (traceSteps.length > 0 || traceReason || traceScore > 0),
  )
  const traceStatus = detail.systemBTraceComplete
    ? { label: S.cite_trace_complete, tone: 'complete' }
    : { label: S.cite_trace_review, tone: 'review' }

  return {
    showTrace,
    traceStatus,
    traceScore,
    traceSteps,
    traceReason,
    traceLabel: S.cite_evidence_chain,
  }
}
