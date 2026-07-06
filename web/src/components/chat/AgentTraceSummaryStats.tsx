import type { StringMap } from '../../i18n'
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
}: {
  labels: Partial<StringMap>
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
        <div className={`kb-agent-trace-evidence-status ${evidenceStatusClass(evidenceStatus)}`} data-testid="agent-trace-evidence-status">
          <span>{tx(labels, 'agent_trace_label_evidence', 'Evidence')}</span>
          <strong>{evidenceLabel}</strong>
        </div>
      ) : null}
      {totalClaims > 0 ? (
        <div>
          <span>{tx(labels, 'agent_trace_label_claims', 'Claims')}</span>
          <strong>{supportedClaims}/{totalClaims}</strong>
        </div>
      ) : null}
      {unsupportedClaims > 0 ? (
        <div className="is-warning">
          <span>{tx(labels, 'agent_trace_label_needs_review', 'Needs review')}</span>
          <strong>{unsupportedClaims}</strong>
        </div>
      ) : null}
      {qualityGateLabel(qualityGateStatus, labels) ? (
        <div className={qualityGateClass(qualityGateStatus)} data-testid="agent-trace-quality-gate">
          <span>{tx(labels, 'agent_trace_label_answer_quality', 'Answer quality')}</span>
          <strong title={qualityGateTitle}>{qualityGateLabel(qualityGateStatus, labels)}</strong>
        </div>
      ) : null}
      <div>
        <span>{tx(labels, 'agent_trace_label_task', 'Task')}</span>
        <strong>{taskLabel}</strong>
      </div>
      {scopeSummary ? (
        <div>
          <span>{tx(labels, 'agent_trace_label_scope', 'Scope')}</span>
          <strong title={scopeSummary}>{shortText(scopeSummary, 72)}</strong>
        </div>
      ) : null}
      {hasErrors ? (
        <div className="is-warning">
          <span>{tx(labels, 'agent_trace_label_run', 'Run')}</span>
          <strong>{tx(labels, 'agent_trace_label_errors', 'errors')}</strong>
        </div>
      ) : null}
      {researchRunStatus || evidenceMatrixRows > 0 ? (
        <div>
          <span>{tx(labels, 'agent_trace_label_research_run', 'Research run')}</span>
          <strong>
            {[researchRunStatus || tx(labels, 'agent_trace_ready', 'ready'), evidenceMatrixRows > 0 ? txFmt(labels, 'agent_trace_rows', '{n} rows', { n: evidenceMatrixRows }) : ''].filter(Boolean).join(' / ')}
          </strong>
        </div>
      ) : null}
      {sourcePolicy ? (
        <div>
          <span>{tx(labels, 'agent_trace_label_source_policy', 'Source policy')}</span>
          <strong>{sourcePolicyLabel(sourcePolicy, labels)}</strong>
        </div>
      ) : null}
    </div>
  )
}
