import type { CiteDetail } from './citationState'
import type { AgentTraceLabels, AgentTraceRecord } from './agentTraceTypes'

export type AgentTraceReferenceRecord = AgentTraceRecord

export type AgentTraceReferenceAction = (
  detail: CiteDetail,
  reference: AgentTraceReferenceRecord,
) => void

export type AgentTraceReferenceHandlers = {
  onOpenReference?: AgentTraceReferenceAction
  onAddReferenceToShelf?: AgentTraceReferenceAction
}

export type AgentTraceReferenceListProps = AgentTraceLabels & AgentTraceReferenceHandlers & {
  references: AgentTraceReferenceRecord[]
}

export type AgentTraceReferenceCardProps = AgentTraceLabels & AgentTraceReferenceHandlers & {
  reference: AgentTraceReferenceRecord
}
