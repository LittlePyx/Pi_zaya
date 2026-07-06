import { AgentTraceSummaryChip } from './AgentTraceSummaryChip'
import type { AgentTraceLabels } from './agentTraceTypes'
import type { AgentSourceSummaryViewModel } from './useAgentTraceViewModel'
import {
  evidenceStatusClass,
  qualityGateClass,
  qualityGateLabel,
  shortText,
  sourcePolicyLabel,
  tx,
  txFmt,
} from './agentTracePanelUtils'

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

export function AgentTraceSummaryStats({
  labels,
  viewModel,
}: AgentTraceLabels & {
  viewModel: AgentSourceSummaryViewModel
}) {
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

  return (
    <div className="kb-agent-trace-summary">
      {evidenceLabel ? (
        <AgentTraceSummaryChip
          className={`kb-agent-trace-evidence-status ${evidenceStatusClass(evidenceStatus)}`}
          label={tx(labels, 'agent_trace_label_evidence', 'Evidence')}
          value={evidenceLabel}
          testId="agent-trace-evidence-status"
        />
      ) : null}
      {totalClaims > 0 ? (
        <AgentTraceSummaryChip
          label={tx(labels, 'agent_trace_label_claims', 'Claims')}
          value={claimsSummary}
        />
      ) : null}
      {unsupportedClaims > 0 ? (
        <AgentTraceSummaryChip
          className="is-warning"
          label={tx(labels, 'agent_trace_label_needs_review', 'Needs review')}
          value={unsupportedClaims}
        />
      ) : null}
      {qualityGateText ? (
        <AgentTraceSummaryChip
          className={qualityGateClass(qualityGateStatus)}
          label={tx(labels, 'agent_trace_label_answer_quality', 'Answer quality')}
          value={qualityGateText}
          title={qualityGateTitle}
          testId="agent-trace-quality-gate"
        />
      ) : null}
      <AgentTraceSummaryChip
        label={tx(labels, 'agent_trace_label_task', 'Task')}
        value={taskLabel}
      />
      {scopeSummary ? (
        <AgentTraceSummaryChip
          label={tx(labels, 'agent_trace_label_scope', 'Scope')}
          value={scopeDisplay}
          title={scopeSummary}
        />
      ) : null}
      {hasErrors ? (
        <AgentTraceSummaryChip
          className="is-warning"
          label={tx(labels, 'agent_trace_label_run', 'Run')}
          value={tx(labels, 'agent_trace_label_errors', 'errors')}
        />
      ) : null}
      {researchRunSummary ? (
        <AgentTraceSummaryChip
          label={tx(labels, 'agent_trace_label_research_run', 'Research run')}
          value={researchRunSummary}
        />
      ) : null}
      {sourcePolicy ? (
        <AgentTraceSummaryChip
          label={tx(labels, 'agent_trace_label_source_policy', 'Source policy')}
          value={sourcePolicyText}
        />
      ) : null}
    </div>
  )
}
