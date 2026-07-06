import type { StringMap } from '../../i18n'
import type { CiteDetail } from './citationState'
import {
  referenceDetail,
  referenceMeta,
  referenceTitle,
  shortText,
  tx,
} from './agentTracePanelUtils'

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
      {references.map((ref, refIdx) => {
        const detail = referenceDetail(ref)
        const canOpen = Boolean(detail?.sourcePath && onOpenReference)
        const canAdd = Boolean(detail && onAddReferenceToShelf)
        return (
          <div className="kb-agent-trace-ref" key={`${String(ref.ref_num || ref.title || 'ref')}-${refIdx}`} data-testid="agent-trace-reference">
            <strong data-testid="agent-trace-ref-title">{referenceTitle(ref)}</strong>
            {referenceMeta(ref) ? <span>{referenceMeta(ref)}</span> : null}
            {ref.why_relevant ? <em>{shortText(ref.why_relevant, 180)}</em> : null}
            {canOpen || canAdd ? (
              <div className="kb-agent-trace-ref-actions">
                {canOpen && detail ? (
                  <button type="button" onClick={() => onOpenReference?.(detail, ref)} data-testid="agent-trace-ref-open">
                    {tx(labels, 'common_open', 'Open')}
                  </button>
                ) : null}
                {canAdd && detail ? (
                  <button type="button" onClick={() => onAddReferenceToShelf?.(detail, ref)} data-testid="agent-trace-ref-add">
                    {tx(labels, 'common_add', 'Add')}
                  </button>
                ) : null}
              </div>
            ) : null}
          </div>
        )
      })}
    </div>
  )
}
