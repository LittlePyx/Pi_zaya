import {
  shortText,
} from './agentTracePanelUtils'

export type AgentTraceScopeSummaryInput = {
  queryScope: string
  requestedScope: string
  selectedCount: number
  currentSource: unknown
}

export function buildAgentTraceScopeSummary({
  queryScope,
  requestedScope,
  selectedCount,
  currentSource,
}: AgentTraceScopeSummaryInput): string {
  const currentSourceLabel = shortText(currentSource, 90)
  return [
    queryScope,
    requestedScope && requestedScope !== queryScope ? `requested ${requestedScope}` : '',
    selectedCount > 0 ? `${selectedCount} selected` : '',
    queryScope === 'current_paper' && currentSourceLabel ? currentSourceLabel : '',
  ].filter(Boolean).join(' / ')
}
