import type { StringMap } from '../../i18n'
import { AgentTraceReferenceList } from './AgentTraceReferenceList'
import type { AgentTraceReferenceHandlers, AgentTraceReferenceRecord } from './agentTraceReferenceTypes'
import { tx } from './agentTracePanelUtils'

export function AgentTraceReferenceSection({
  labels,
  references,
  onOpenReference,
  onAddReferenceToShelf,
}: AgentTraceReferenceHandlers & {
  labels: Partial<StringMap>
  references: AgentTraceReferenceRecord[]
}) {
  if (references.length <= 0) return null

  return (
    <div className="kb-agent-trace-section kb-agent-trace-public-refs" data-testid="agent-trace-public-references">
      <div className="kb-agent-trace-heading">{tx(labels, 'agent_trace_references', 'References')}</div>
      <AgentTraceReferenceList
        references={references}
        labels={labels}
        onOpenReference={onOpenReference}
        onAddReferenceToShelf={onAddReferenceToShelf}
      />
    </div>
  )
}
