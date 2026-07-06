import type { StringMap } from '../../i18n'
import type { CiteDetail } from './citationState'
import { tx } from './agentTracePanelUtils'

export function AgentTraceReferenceActions({
  labels,
  reference,
  detail,
  onOpenReference,
  onAddReferenceToShelf,
}: {
  labels: Partial<StringMap>
  reference: Record<string, unknown>
  detail: CiteDetail | null
  onOpenReference?: (detail: CiteDetail, ref: Record<string, unknown>) => void
  onAddReferenceToShelf?: (detail: CiteDetail, ref: Record<string, unknown>) => void
}) {
  const canOpen = Boolean(detail?.sourcePath && onOpenReference)
  const canAdd = Boolean(detail && onAddReferenceToShelf)

  if (!detail || (!canOpen && !canAdd)) return null

  return (
    <div className="kb-agent-trace-ref-actions">
      {canOpen ? (
        <button type="button" onClick={() => onOpenReference?.(detail, reference)} data-testid="agent-trace-ref-open">
          {tx(labels, 'common_open', 'Open')}
        </button>
      ) : null}
      {canAdd ? (
        <button type="button" onClick={() => onAddReferenceToShelf?.(detail, reference)} data-testid="agent-trace-ref-add">
          {tx(labels, 'common_add', 'Add')}
        </button>
      ) : null}
    </div>
  )
}
