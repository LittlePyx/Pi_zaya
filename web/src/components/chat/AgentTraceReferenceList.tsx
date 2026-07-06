import { AgentTraceReferenceCard } from './AgentTraceReferenceCard'
import type { AgentTraceReferenceListProps } from './agentTraceReferenceTypes'

export function AgentTraceReferenceList({
  references,
  labels,
  onOpenReference,
  onAddReferenceToShelf,
}: AgentTraceReferenceListProps) {
  if (references.length <= 0) return null
  return (
    <div className="kb-agent-trace-refs">
      {references.map((ref, refIdx) => (
        <AgentTraceReferenceCard
          key={`${String(ref.ref_num || ref.title || 'ref')}-${refIdx}`}
          reference={ref}
          labels={labels}
          onOpenReference={onOpenReference}
          onAddReferenceToShelf={onAddReferenceToShelf}
        />
      ))}
    </div>
  )
}
