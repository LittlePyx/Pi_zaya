import type { StringMap } from '../../i18n'
import type { CiteDetail } from './citationState'
import { AgentTraceReferenceActions } from './AgentTraceReferenceActions'
import {
  referenceDetail,
  referenceMeta,
  referenceTitle,
  shortText,
} from './agentTracePanelUtils'

export function AgentTraceReferenceCard({
  labels,
  reference,
  onOpenReference,
  onAddReferenceToShelf,
}: {
  labels: Partial<StringMap>
  reference: Record<string, unknown>
  onOpenReference?: (detail: CiteDetail, ref: Record<string, unknown>) => void
  onAddReferenceToShelf?: (detail: CiteDetail, ref: Record<string, unknown>) => void
}) {
  const detail = referenceDetail(reference)
  const meta = referenceMeta(reference)

  return (
    <div className="kb-agent-trace-ref" data-testid="agent-trace-reference">
      <strong data-testid="agent-trace-ref-title">{referenceTitle(reference)}</strong>
      {meta ? <span>{meta}</span> : null}
      {reference.why_relevant ? <em>{shortText(reference.why_relevant, 180)}</em> : null}
      <AgentTraceReferenceActions
        labels={labels}
        reference={reference}
        detail={detail}
        onOpenReference={onOpenReference}
        onAddReferenceToShelf={onAddReferenceToShelf}
      />
    </div>
  )
}
