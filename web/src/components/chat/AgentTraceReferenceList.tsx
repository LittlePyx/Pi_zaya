import type { StringMap } from '../../i18n'
import type { CiteDetail } from './citationState'
import { AgentTraceReferenceCard } from './AgentTraceReferenceCard'

export function AgentTraceReferenceList({
  references,
  labels,
  onOpenReference,
  onAddReferenceToShelf,
}: {
  references: Record<string, unknown>[]
  labels: Partial<StringMap>
  onOpenReference?: (detail: CiteDetail, ref: Record<string, unknown>) => void
  onAddReferenceToShelf?: (detail: CiteDetail, ref: Record<string, unknown>) => void
}) {
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
