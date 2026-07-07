import type { AgentTraceSummaryChipProps } from './AgentTraceSummaryChip'
import type { AgentTraceLabels } from './agentTraceTypes'
import type { AgentSourceSummaryViewModel } from './agentTraceViewModel'
import {
  evidenceStatusClass,
  qualityGateClass,
  qualityGateLabel,
  shortText,
  sourcePolicyLabel,
  tx,
  txFmt,
} from './agentTracePanelUtils'

export type AgentTraceSummaryChipConfig = AgentTraceSummaryChipProps & {
  id: string
  visible?: boolean
}

function researchRunSummaryLabel(
  labels: AgentTraceLabels['labels'],
  researchRunStatus: string,
  evidenceMatrixRows: number,
) {
  return [
    researchRunStatus || tx(labels, 'agent_trace_ready', 'ready'),
    evidenceMatrixRows > 0 ? txFmt(labels, 'agent_trace_rows', '{n} rows', { n: evidenceMatrixRows }) : '',
  ].filter(Boolean).join(' / ')
}

export function buildAgentTraceSummaryChips(
  labels: AgentTraceLabels['labels'],
  viewModel: AgentSourceSummaryViewModel,
): AgentTraceSummaryChipConfig[] {
  const {
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
  } = viewModel
  const claimsSummary = `${supportedClaims}/${totalClaims}`
  const qualityGateText = qualityGateLabel(qualityGateStatus, labels)
  const scopeDisplay = scopeSummary ? shortText(scopeSummary, 72) : ''
  const researchRunSummary = researchRunStatus || evidenceMatrixRows > 0
    ? researchRunSummaryLabel(labels, researchRunStatus, evidenceMatrixRows)
    : ''
  const sourcePolicyText = sourcePolicy ? sourcePolicyLabel(sourcePolicy, labels) : ''

  return [
    {
      id: 'evidence',
      visible: Boolean(evidenceLabel),
      className: `kb-agent-trace-evidence-status ${evidenceStatusClass(evidenceStatus)}`,
      label: tx(labels, 'agent_trace_label_evidence', 'Evidence'),
      value: evidenceLabel,
      testId: 'agent-trace-evidence-status',
    },
    {
      id: 'claims',
      visible: totalClaims > 0,
      label: tx(labels, 'agent_trace_label_claims', 'Claims'),
      value: claimsSummary,
    },
    {
      id: 'unsupported-claims',
      visible: unsupportedClaims > 0,
      className: 'is-warning',
      label: tx(labels, 'agent_trace_label_needs_review', 'Needs review'),
      value: unsupportedClaims,
    },
    {
      id: 'quality-gate',
      visible: Boolean(qualityGateText),
      className: qualityGateClass(qualityGateStatus),
      label: tx(labels, 'agent_trace_label_answer_quality', 'Answer quality'),
      value: qualityGateText,
      title: qualityGateTitle,
      testId: 'agent-trace-quality-gate',
    },
    {
      id: 'task',
      label: tx(labels, 'agent_trace_label_task', 'Task'),
      value: taskLabel,
    },
    {
      id: 'scope',
      visible: Boolean(scopeSummary),
      label: tx(labels, 'agent_trace_label_scope', 'Scope'),
      value: scopeDisplay,
      title: scopeSummary,
    },
    {
      id: 'run-errors',
      visible: hasErrors,
      className: 'is-warning',
      label: tx(labels, 'agent_trace_label_run', 'Run'),
      value: tx(labels, 'agent_trace_label_errors', 'errors'),
    },
    {
      id: 'research-run',
      visible: Boolean(researchRunSummary),
      label: tx(labels, 'agent_trace_label_research_run', 'Research run'),
      value: researchRunSummary,
    },
    {
      id: 'source-policy',
      visible: Boolean(sourcePolicy),
      label: tx(labels, 'agent_trace_label_source_policy', 'Source policy'),
      value: sourcePolicyText,
    },
  ]
}
