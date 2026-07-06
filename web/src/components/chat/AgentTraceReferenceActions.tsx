import type { CiteDetail } from './citationState'
import type { AgentTraceReferenceCardProps } from './agentTraceReferenceTypes'
import { tx } from './agentTracePanelUtils'

export function AgentTraceReferenceActions({
  labels,
  reference,
  detail,
  onOpenReference,
  onAddReferenceToShelf,
}: AgentTraceReferenceCardProps & {
  detail: CiteDetail | null
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
