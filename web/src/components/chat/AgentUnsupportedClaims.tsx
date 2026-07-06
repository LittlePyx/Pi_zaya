import type { StringMap } from '../../i18n'
import {
  shortText,
  tx,
  txFmt,
  unsupportedReasonText,
} from './agentTracePanelUtils'
import { traceNum } from './messageTraceUtils'

export function AgentUnsupportedClaims({
  labels,
  claims,
}: {
  labels: Partial<StringMap>
  claims: Record<string, unknown>[]
}) {
  if (claims.length <= 0) return null
  return (
    <div className="kb-agent-trace-section kb-agent-trace-unsupported">
      <div className="kb-agent-trace-heading">{tx(labels, 'agent_trace_label_needs_review', 'Needs review')}</div>
      {claims.map((claim, idx) => (
        <div className="kb-agent-trace-claim" key={`${String(claim.index || 'claim')}-${idx}`} data-testid="agent-trace-unsupported-claim">
          <strong>{shortText(claim.claim_text || claim.text, 240)}</strong>
          <span>
            {tx(labels, 'agent_trace_label_needs_review', 'Needs review')}: {unsupportedReasonText(claim.unsupported_reason, labels)}
            {traceNum(claim.matched_evidence_count) > 0 ? ` / ${txFmt(labels, 'agent_trace_evidence_matches', '{n} evidence match(es)', { n: traceNum(claim.matched_evidence_count) })}` : ''}
          </span>
        </div>
      ))}
    </div>
  )
}
