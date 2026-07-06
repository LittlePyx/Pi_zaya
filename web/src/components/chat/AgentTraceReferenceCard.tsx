import type { StringMap } from '../../i18n'
import type { CiteDetail } from './citationState'
import {
  referenceDetail,
  referenceMeta,
  referenceTitle,
  shortText,
  tx,
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
  const canOpen = Boolean(detail?.sourcePath && onOpenReference)
  const canAdd = Boolean(detail && onAddReferenceToShelf)

  return (
    <div className="kb-agent-trace-ref" data-testid="agent-trace-reference">
      <strong data-testid="agent-trace-ref-title">{referenceTitle(reference)}</strong>
      {meta ? <span>{meta}</span> : null}
      {reference.why_relevant ? <em>{shortText(reference.why_relevant, 180)}</em> : null}
      {canOpen || canAdd ? (
        <div className="kb-agent-trace-ref-actions">
          {canOpen && detail ? (
            <button type="button" onClick={() => onOpenReference?.(detail, reference)} data-testid="agent-trace-ref-open">
              {tx(labels, 'common_open', 'Open')}
            </button>
          ) : null}
          {canAdd && detail ? (
            <button type="button" onClick={() => onAddReferenceToShelf?.(detail, reference)} data-testid="agent-trace-ref-add">
              {tx(labels, 'common_add', 'Add')}
            </button>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}
