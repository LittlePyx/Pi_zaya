import { useMemo } from 'react'
import type { StringMap } from '../../i18n'
import { buildAgentTraceHeaderSummary } from './agentTraceHeaderSummary'
import { buildAgentTraceMetricCounts } from './agentTraceMetricCounts'
import { buildAgentTraceScopeSummary } from './agentTraceScopeSummary'
import { buildAgentTraceSourceRows } from './agentTraceSourceRows'
import { buildAgentTraceSourceStatus } from './agentTraceSourceStatus'
import { evidenceStatusValue, records } from './agentTracePanelUtils'
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
  const { unsupportedClaimRows, references } = buildAgentTraceSourceRows({
    verification,
    steps,
  })
  const {
    totalClaims,
    supportedClaims,
    unsupportedClaims,
    planStepCount,
    toolCallCount,
    hasErrors,
    evidenceMatrixRows,
    subtaskCount,
  } = buildAgentTraceMetricCounts({
    summary,
    verification,
    planCount: plan.length,
    stepCount: steps.length,
    errorCount: errors.length,
    evidenceMatrixCount: evidenceMatrix.length,
    researchSubtaskCount: researchSubtasks.length,
  })
  const queryScope = String(summary.query_scope || context.query_scope || context.queryScope || '').trim()
  const requestedScope = String(summary.requested_query_scope || context.requested_query_scope || context.requestedQueryScope || '').trim()
  const {
    evidenceStatus,
    evidenceLabel,
    qualityGateStatus,
    qualityGateTitle,
    taskLabel,
    researchRunStatus,
    sourcePolicy,
  } = buildAgentTraceSourceStatus({
    summary,
    verification,
    researchRun,
    traceQuestionType: trace.question_type,
    labels,
  })
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
      references,
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
