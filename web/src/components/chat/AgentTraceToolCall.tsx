import { asTraceRecord, traceNum } from './messageTraceUtils'
import { AgentTraceReferenceList } from './AgentTraceReferenceList'
import type { AgentTraceReferenceHandlers } from './agentTraceReferenceTypes'
import type { AgentTraceLabels, AgentTraceRecord } from './agentTraceTypes'
import {
  records,
  shortText,
  statusClass,
} from './agentTracePanelUtils'

export function AgentTraceToolCall({
  labels,
  step,
  onOpenReference,
  onAddReferenceToShelf,
}: AgentTraceReferenceHandlers & AgentTraceLabels & {
  step: AgentTraceRecord
}) {
  const output = asTraceRecord(step.output)
  const refs = records(output.references).slice(0, 3)

  return (
    <div className="kb-agent-trace-call">
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
}
