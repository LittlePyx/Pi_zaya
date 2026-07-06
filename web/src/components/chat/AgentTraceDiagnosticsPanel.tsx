import type { StringMap } from '../../i18n'
import type { CiteDetail } from './citationState'
import { asTraceRecord, traceNum } from './messageTraceUtils'
import { AgentTraceReferenceList } from './AgentTraceReferenceList'
import {
  records,
  shortText,
  statusClass,
  tx,
  txFmt,
} from './agentTracePanelUtils'

export function AgentTraceDiagnosticsPanel({
  labels,
  plan,
  steps,
  planStepCount,
  toolCallCount,
  onOpenReference,
  onAddReferenceToShelf,
}: {
  labels: Partial<StringMap>
  plan: Record<string, unknown>[]
  steps: Record<string, unknown>[]
  planStepCount: number
  toolCallCount: number
  onOpenReference?: (detail: CiteDetail, ref: Record<string, unknown>) => void
  onAddReferenceToShelf?: (detail: CiteDetail, ref: Record<string, unknown>) => void
}) {
  if (plan.length <= 0 && steps.length <= 0) return null
  return (
    <details className="kb-agent-trace-details" data-testid="agent-trace-execution-details">
      <summary>
        <span>{tx(labels, 'agent_trace_diagnostics', 'Diagnostics')}</span>
        <span>{txFmt(labels, 'agent_trace_plan_count', '{n} plan', { n: planStepCount })}</span>
        <span>{txFmt(labels, 'agent_trace_check_count', '{n} checks', { n: toolCallCount })}</span>
      </summary>
      {plan.length > 0 ? (
        <div className="kb-agent-trace-section">
          <div className="kb-agent-trace-heading">{tx(labels, 'agent_trace_plan', 'Plan')}</div>
          {plan.map((step, idx) => (
            <div className="kb-agent-trace-row" key={`${String(step.tool || 'plan')}-${idx}`}>
              <span className={`kb-agent-trace-status ${statusClass(step.status)}`}>{String(step.status || 'pending')}</span>
              <span className="kb-agent-trace-tool">{String(step.tool || '')}</span>
              <span className="kb-agent-trace-text">{shortText(step.goal)}</span>
            </div>
          ))}
        </div>
      ) : null}
      {steps.length > 0 ? (
        <div className="kb-agent-trace-section">
          <div className="kb-agent-trace-heading">{tx(labels, 'agent_trace_check_activity', 'Check activity')}</div>
          {steps.map((step, idx) => {
            const output = asTraceRecord(step.output)
            const refs = records(output.references).slice(0, 3)
            return (
              <div className="kb-agent-trace-call" key={`${String(step.tool || 'tool')}-${idx}`}>
                <div className="kb-agent-trace-call-head">
                  <span className={`kb-agent-trace-status ${statusClass(step.status)}`}>{String(step.status || '')}</span>
                  <strong>{String(step.tool || '')}</strong>
                  {traceNum(step.elapsed_ms) > 0 ? <span>{traceNum(step.elapsed_ms)}ms</span> : null}
                </div>
                {step.observation ? <div className="kb-agent-trace-observation">{shortText(step.observation, 260)}</div> : null}
                <AgentTraceReferenceList
                  references={refs}
                  labels={labels}
                  onOpenReference={onOpenReference}
                  onAddReferenceToShelf={onAddReferenceToShelf}
                />
                {step.error ? <div className="kb-agent-trace-error">{shortText(step.error, 260)}</div> : null}
              </div>
            )
          })}
        </div>
      ) : null}
    </details>
  )
}
