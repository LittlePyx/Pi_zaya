import type { StringMap } from '../../i18n'
import {
  evidenceStatusLabel,
  shortText,
  tx,
  txFmt,
} from './agentTracePanelUtils'

export function AgentEvidenceMatrix({
  labels,
  rows,
  subtaskCount,
}: {
  labels: Partial<StringMap>
  rows: Record<string, unknown>[]
  subtaskCount: number
}) {
  if (rows.length <= 0) return null
  return (
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
            {rows.slice(0, 8).map((row, idx) => {
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
  )
}
