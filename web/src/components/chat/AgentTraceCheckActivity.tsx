import type { StringMap } from '../../i18n'
import { asTraceRecord, traceNum } from './messageTraceUtils'
import { AgentTraceReferenceList } from './AgentTraceReferenceList'
import type { AgentTraceReferenceHandlers } from './agentTraceReferenceTypes'
import {
  records,
  shortText,
  statusClass,
  tx,
} from './agentTracePanelUtils'

export function AgentTraceCheckActivity({
  labels,
  steps,
  onOpenReference,
  onAddReferenceToShelf,
}: AgentTraceReferenceHandlers & {
  labels: Partial<StringMap>
  steps: Record<string, unknown>[]
}) {
  if (steps.length <= 0) return null

  return (
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
  )
}
