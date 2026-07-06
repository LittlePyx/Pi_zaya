import type { StringMap } from '../../i18n'
import type { CiteDetail } from './citationState'
import { AgentTraceReferenceList } from './AgentTraceReferenceList'
import { tx } from './agentTracePanelUtils'

export function AgentTraceReferenceSection({
  labels,
  references,
  onOpenReference,
  onAddReferenceToShelf,
}: {
  labels: Partial<StringMap>
  references: Record<string, unknown>[]
  onOpenReference?: (detail: CiteDetail, ref: Record<string, unknown>) => void
  onAddReferenceToShelf?: (detail: CiteDetail, ref: Record<string, unknown>) => void
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
