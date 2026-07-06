import type { StringMap } from '../../i18n'
import type { CiteDetail } from './citationState'

export type AgentTraceReferenceRecord = Record<string, unknown>

export type AgentTraceReferenceAction = (
  detail: CiteDetail,
  reference: AgentTraceReferenceRecord,
) => void

export type AgentTraceReferenceHandlers = {
  onOpenReference?: AgentTraceReferenceAction
  onAddReferenceToShelf?: AgentTraceReferenceAction
}

export type AgentTraceReferenceLabels = {
  labels: Partial<StringMap>
}

export type AgentTraceReferenceListProps = AgentTraceReferenceLabels & AgentTraceReferenceHandlers & {
  references: AgentTraceReferenceRecord[]
}

export type AgentTraceReferenceCardProps = AgentTraceReferenceLabels & AgentTraceReferenceHandlers & {
  reference: AgentTraceReferenceRecord
}
