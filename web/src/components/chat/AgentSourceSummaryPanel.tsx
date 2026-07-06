import type { StringMap } from '../../i18n'
import type { CiteDetail } from './citationState'
import { AgentTraceReferenceList } from './AgentTraceReferenceList'
import type { AgentSourceSummaryViewModel } from './useAgentTraceViewModel'
import {
  evidenceStatusClass,
  evidenceStatusLabel,
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
      {evidenceMatrix.length > 0 ? (
        <div className="kb-agent-trace-section kb-agent-matrix" data-testid="agent-evidence-matrix">
          <div className="kb-agent-trace-heading">
            {tx(labels, 'agent_trace_evidence_map', 'Evidence map')}
            {subtaskCount > 0 ? <span>{txFmt(labels, 'agent_trace_subtasks', '{n} subtasks', { n: subtaskCount })}</span> : null}
          </div>
          <div className="kb-agent-matrix-scroll">
            <table>
              <thead>
                <tr>
                  <th>{tx(labels, 'agent_trace_col_paper', 'Paper')}</th>
                  <th>{tx(labels, 'agent_trace_col_method', 'Method')}</th>
                  <th>{tx(labels, 'agent_trace_col_result', 'Result')}</th>
                  <th>{tx(labels, 'agent_trace_col_limitation', 'Limitation')}</th>
                  <th>{tx(labels, 'agent_trace_col_evidence', 'Evidence')}</th>
                </tr>
              </thead>
              <tbody>
                {evidenceMatrix.slice(0, 8).map((row, idx) => {
                  const supportStatus = evidenceStatusLabel(row.support_status, labels) || shortText(row.support_status, 40)
                  return (
                    <tr key={`${String(row.source_path || row.source_name || row.paper || 'row')}-${idx}`} data-testid="agent-evidence-matrix-row">
                      <td>
                        <strong>{shortText(row.paper || row.source_name || tx(labels, 'agent_trace_source_fallback', 'Source'), 90)}</strong>
                        {row.heading_path ? <span>{shortText(row.heading_path, 90)}</span> : null}
                      </td>
                      <td>{shortText(row.method, 140) || tx(labels, 'agent_trace_not_identified', 'Not identified')}</td>
                      <td>{shortText(row.key_result, 140) || tx(labels, 'agent_trace_not_identified', 'Not identified')}</td>
                      <td>{shortText(row.limitation, 140) || tx(labels, 'agent_trace_not_identified', 'Not identified')}</td>
                      <td>
                        <span>{shortText(row.evidence_quote, 160) || tx(labels, 'agent_trace_no_quote', 'No quote')}</span>
                        <em>{[row.citation, supportStatus].filter(Boolean).join(' / ')}</em>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}
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
