import { traceBool } from './agentTracePanelUtils'
import type { AgentTraceRecord } from './agentTraceTypes'
import { traceNum } from './messageTraceUtils'

export type AgentTraceMetricCountInput = {
  summary: AgentTraceRecord
  verification: AgentTraceRecord
  planCount: number
  stepCount: number
  errorCount: number
  evidenceMatrixCount: number
  researchSubtaskCount: number
}

export type AgentTraceMetricCounts = {
  totalClaims: number
  supportedClaims: number
  unsupportedClaims: number
  planStepCount: number
  toolCallCount: number
  hasErrors: boolean
  evidenceMatrixRows: number
  subtaskCount: number
}

export function buildAgentTraceMetricCounts({
  summary,
  verification,
  planCount,
  stepCount,
  errorCount,
  evidenceMatrixCount,
  researchSubtaskCount,
}: AgentTraceMetricCountInput): AgentTraceMetricCounts {
  return {
    totalClaims: 'total_claims' in summary ? traceNum(summary.total_claims) : traceNum(verification.total_claims),
    supportedClaims: 'supported_claims' in summary ? traceNum(summary.supported_claims) : traceNum(verification.supported_claims),
    unsupportedClaims: 'unsupported_claims' in summary ? traceNum(summary.unsupported_claims) : traceNum(verification.unsupported_claims),
    planStepCount: 'plan_step_count' in summary ? traceNum(summary.plan_step_count) : planCount,
    toolCallCount: 'tool_call_count' in summary ? traceNum(summary.tool_call_count) : stepCount,
    hasErrors: 'has_errors' in summary ? traceBool(summary.has_errors) : errorCount > 0,
    evidenceMatrixRows: 'evidence_matrix_rows' in summary ? traceNum(summary.evidence_matrix_rows) : evidenceMatrixCount,
    subtaskCount: 'subtask_count' in summary ? traceNum(summary.subtask_count) : researchSubtaskCount,
  }
}
