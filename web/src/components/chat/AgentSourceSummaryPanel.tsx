import type { StringMap } from '../../i18n'
import type { CiteDetail } from './citationState'
import { AgentEvidenceMatrix } from './AgentEvidenceMatrix'
import { AgentTraceReferenceList } from './AgentTraceReferenceList'
import type { AgentSourceSummaryViewModel } from './useAgentTraceViewModel'
import {
  evidenceStatusClass,
  qualityGateClass,
  qualityGateLabel,
  shortText,
  sourcePolicyLabel,
  tx,
  txFmt,
  unsupportedReasonText,
} from './agentTracePanelUtils'
import { traceNum } from './messageTraceUtils'

export function AgentSourceSummaryPanel({
  labels,
  viewModel,
  onOpenReference,
  onAddReferenceToShelf,
}: {
  labels: Partial<StringMap>
  viewModel: AgentSourceSummaryViewModel
  onOpenReference?: (detail: CiteDetail, ref: Record<string, unknown>) => void
  onAddReferenceToShelf?: (detail: CiteDetail, ref: Record<string, unknown>) => void
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
    evidenceMatrix,
    subtaskCount,
    unsupportedClaimRows,
    references,
  } = viewModel

  return (
    <>
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
      <AgentEvidenceMatrix labels={labels} rows={evidenceMatrix} subtaskCount={subtaskCount} />
      {unsupportedClaimRows.length > 0 ? (
        <div className="kb-agent-trace-section kb-agent-trace-unsupported">
          <div className="kb-agent-trace-heading">{tx(labels, 'agent_trace_label_needs_review', 'Needs review')}</div>
          {unsupportedClaimRows.map((claim, idx) => (
            <div className="kb-agent-trace-claim" key={`${String(claim.index || 'claim')}-${idx}`} data-testid="agent-trace-unsupported-claim">
              <strong>{shortText(claim.claim_text || claim.text, 240)}</strong>
              <span>
                {tx(labels, 'agent_trace_label_needs_review', 'Needs review')}: {unsupportedReasonText(claim.unsupported_reason, labels)}
                {traceNum(claim.matched_evidence_count) > 0 ? ` / ${txFmt(labels, 'agent_trace_evidence_matches', '{n} evidence match(es)', { n: traceNum(claim.matched_evidence_count) })}` : ''}
              </span>
            </div>
          ))}
        </div>
      ) : null}
      {references.length > 0 ? (
        <div className="kb-agent-trace-section kb-agent-trace-public-refs" data-testid="agent-trace-public-references">
          <div className="kb-agent-trace-heading">{tx(labels, 'agent_trace_references', 'References')}</div>
          <AgentTraceReferenceList
            references={references}
            labels={labels}
            onOpenReference={onOpenReference}
            onAddReferenceToShelf={onAddReferenceToShelf}
          />
        </div>
      ) : null}
    </>
  )
}
