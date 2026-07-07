import type { StringMap } from '../../i18n'
import { buildAgentTraceQualityGateTitle } from './agentTraceQualityGate'
import {
  evidenceStatusLabel,
  evidenceStatusValue,
  questionTypeLabel,
  tx,
} from './agentTracePanelUtils'
import type { AgentTraceRecord } from './agentTraceTypes'

export type AgentTraceSourceStatusInput = {
  summary: AgentTraceRecord
  verification: AgentTraceRecord
  researchRun: AgentTraceRecord
  traceQuestionType: unknown
  labels: Partial<StringMap>
}

export type AgentTraceSourceStatus = {
  evidenceStatus: ReturnType<typeof evidenceStatusValue>
  evidenceLabel: string
  qualityGateStatus: string
  qualityGateTitle: string
  taskLabel: string
  researchRunStatus: string
  sourcePolicy: string
}

export function buildAgentTraceSourceStatus({
  summary,
  verification,
  researchRun,
  traceQuestionType,
  labels,
}: AgentTraceSourceStatusInput): AgentTraceSourceStatus {
  const evidenceStatus = evidenceStatusValue(summary.evidence_status || verification.evidence_status)
  const questionType = String(summary.question_type || traceQuestionType || 'unknown').trim()
  const taskLabel = evidenceStatus === 'not_applicable'
    ? tx(labels, 'agent_trace_type_general', 'General')
    : questionTypeLabel(questionType, labels)

  return {
    evidenceStatus,
    evidenceLabel: evidenceStatusLabel(evidenceStatus, labels),
    qualityGateStatus: String(summary.quality_gate_status || '').trim().toLowerCase(),
    qualityGateTitle: buildAgentTraceQualityGateTitle({
      reasons: summary.quality_gate_reasons,
      warnings: summary.quality_gate_warnings,
    }),
    taskLabel,
    researchRunStatus: String(summary.research_run_status || researchRun.status || '').trim(),
    sourcePolicy: String(summary.source_policy || researchRun.source_policy || '').trim(),
  }
}
