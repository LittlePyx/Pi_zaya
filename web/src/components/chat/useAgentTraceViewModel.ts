import { useMemo } from 'react'
import type { StringMap } from '../../i18n'
import { buildAgentTraceHeaderSummary } from './agentTraceHeaderSummary'
import { buildAgentTraceQualityGateTitle } from './agentTraceQualityGate'
import { buildAgentTraceScopeSummary } from './agentTraceScopeSummary'
import {
  evidenceStatusLabel,
  evidenceStatusValue,
  questionTypeLabel,
  records,
  traceBool,
  traceStepReferences,
  tx,
} from './agentTracePanelUtils'
import type { AgentTraceRecord } from './agentTraceTypes'
import type { AgentTraceReferenceRecord } from './agentTraceReferenceTypes'
import { asTraceRecord, traceNum } from './messageTraceUtils'

export type AgentSourceSummaryViewModel = {
  evidenceLabel: string
  evidenceStatus: ReturnType<typeof evidenceStatusValue>
  totalClaims: number
  supportedClaims: number
  unsupportedClaims: number
  qualityGateStatus: string
  qualityGateTitle: string
  taskLabel: string
  scopeSummary: string
  hasErrors: boolean
  researchRunStatus: string
  evidenceMatrixRows: number
  sourcePolicy: string
  evidenceMatrix: AgentTraceRecord[]
  subtaskCount: number
  unsupportedClaimRows: AgentTraceRecord[]
  references: AgentTraceReferenceRecord[]
}

export type AgentTraceDiagnosticsViewModel = {
  plan: AgentTraceRecord[]
  steps: AgentTraceRecord[]
  planStepCount: number
  toolCallCount: number
}

export type AgentTraceViewModel = {
  headerEvidence: string
  headerContext: string
  sourceSummary: AgentSourceSummaryViewModel
  diagnostics: AgentTraceDiagnosticsViewModel
}

function buildAgentTraceViewModel(trace: Record<string, unknown>, labels: Partial<StringMap>): AgentTraceViewModel {
  const plan = records(trace.plan)
  const steps = records(trace.steps)
  const context = asTraceRecord(trace.context)
  const verification = asTraceRecord(trace.verification)
  const researchRun = asTraceRecord(trace.research_run)
  const summary = asTraceRecord(trace.summary)
  const errors = Array.isArray(trace.errors) ? trace.errors : []
  const evidenceMatrix = records(researchRun.evidence_matrix)
  const researchSubtasks = records(researchRun.subtasks)
  const claimRows = records(verification.claims)
  const unsupportedClaimRows = claimRows
    .filter((claim) => claim.supported === false || String(claim.unsupported_reason || '').trim())
    .slice(0, 3)
  const totalClaims = 'total_claims' in summary ? traceNum(summary.total_claims) : traceNum(verification.total_claims)
  const supportedClaims = 'supported_claims' in summary ? traceNum(summary.supported_claims) : traceNum(verification.supported_claims)
  const unsupportedClaims = 'unsupported_claims' in summary ? traceNum(summary.unsupported_claims) : traceNum(verification.unsupported_claims)
  const planStepCount = 'plan_step_count' in summary ? traceNum(summary.plan_step_count) : plan.length
  const toolCallCount = 'tool_call_count' in summary ? traceNum(summary.tool_call_count) : steps.length
  const hasErrors = 'has_errors' in summary ? traceBool(summary.has_errors) : errors.length > 0
  const researchRunStatus = String(summary.research_run_status || researchRun.status || '').trim()
  const sourcePolicy = String(summary.source_policy || researchRun.source_policy || '').trim()
  const evidenceMatrixRows = 'evidence_matrix_rows' in summary ? traceNum(summary.evidence_matrix_rows) : evidenceMatrix.length
  const subtaskCount = 'subtask_count' in summary ? traceNum(summary.subtask_count) : researchSubtasks.length
  const questionType = String(summary.question_type || trace.question_type || 'unknown').trim()
  const queryScope = String(summary.query_scope || context.query_scope || context.queryScope || '').trim()
  const requestedScope = String(summary.requested_query_scope || context.requested_query_scope || context.requestedQueryScope || '').trim()
  const evidenceStatus = evidenceStatusValue(summary.evidence_status || verification.evidence_status)
  const evidenceLabel = evidenceStatusLabel(evidenceStatus, labels)
  const qualityGateStatus = String(summary.quality_gate_status || '').trim().toLowerCase()
  const qualityGateTitle = buildAgentTraceQualityGateTitle({
    reasons: summary.quality_gate_reasons,
    warnings: summary.quality_gate_warnings,
  })
  const taskLabel = evidenceStatus === 'not_applicable' ? tx(labels, 'agent_trace_type_general', 'General') : questionTypeLabel(questionType, labels)
  const selectedCount = traceNum(context.selected_research_context_count || context.selectedResearchContextCount)
  const scopeSummary = buildAgentTraceScopeSummary({
    queryScope,
    requestedScope,
    selectedCount,
    currentSource: context.current_source_name || context.currentSourceName || context.current_source_path || context.currentSourcePath,
  })
  const {
    headerEvidence,
    headerContext,
  } = buildAgentTraceHeaderSummary(labels, {
    evidenceLabel,
    totalClaims,
    supportedClaims,
    unsupportedClaims,
    hasErrors,
    scopeSummary,
    taskLabel,
  })

  return {
    headerEvidence,
    headerContext,
    sourceSummary: {
      evidenceLabel,
      evidenceStatus,
      totalClaims,
      supportedClaims,
      unsupportedClaims,
      qualityGateStatus,
      qualityGateTitle,
      taskLabel,
      scopeSummary,
      hasErrors,
      researchRunStatus,
      evidenceMatrixRows,
      sourcePolicy,
      evidenceMatrix,
      subtaskCount,
      unsupportedClaimRows,
      references: traceStepReferences(steps),
    },
    diagnostics: {
      plan,
      steps,
      planStepCount,
      toolCallCount,
    },
  }
}

export function useAgentTraceViewModel(trace: Record<string, unknown>, labels: Partial<StringMap>): AgentTraceViewModel {
  return useMemo(() => buildAgentTraceViewModel(trace, labels), [labels, trace])
}
