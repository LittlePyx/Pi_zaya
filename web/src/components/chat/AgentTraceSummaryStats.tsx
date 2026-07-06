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
          value={`${supportedClaims}/${totalClaims}`}
        />
      ) : null}
      {unsupportedClaims > 0 ? (
        <AgentTraceSummaryChip
          className="is-warning"
          label={tx(labels, 'agent_trace_label_needs_review', 'Needs review')}
          value={unsupportedClaims}
        />
      ) : null}
      {qualityGateLabel(qualityGateStatus, labels) ? (
        <AgentTraceSummaryChip
          className={qualityGateClass(qualityGateStatus)}
          label={tx(labels, 'agent_trace_label_answer_quality', 'Answer quality')}
          value={qualityGateLabel(qualityGateStatus, labels)}
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
          value={shortText(scopeSummary, 72)}
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
      {researchRunStatus || evidenceMatrixRows > 0 ? (
        <AgentTraceSummaryChip
          label={tx(labels, 'agent_trace_label_research_run', 'Research run')}
          value={[researchRunStatus || tx(labels, 'agent_trace_ready', 'ready'), evidenceMatrixRows > 0 ? txFmt(labels, 'agent_trace_rows', '{n} rows', { n: evidenceMatrixRows }) : ''].filter(Boolean).join(' / ')}
        />
      ) : null}
      {sourcePolicy ? (
        <AgentTraceSummaryChip
          label={tx(labels, 'agent_trace_label_source_policy', 'Source policy')}
          value={sourcePolicyLabel(sourcePolicy, labels)}
        />
      ) : null}
    </div>
  )
}
