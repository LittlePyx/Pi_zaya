export type AgentTracePanelState = 'hidden' | 'stored_prompt' | 'trace'

export type AgentTracePanelStateInput = {
  traceRecord: Record<string, unknown>
  hasTrace: boolean
  canLazyLoad: boolean
}

export function buildAgentTracePanelState({
  traceRecord,
  hasTrace,
  canLazyLoad,
}: AgentTracePanelStateInput): AgentTracePanelState {
  if (!hasTrace) return canLazyLoad ? 'stored_prompt' : 'hidden'

  const mode = String(traceRecord.mode || '').trim()
  if (mode && mode !== 'research_agent') return 'hidden'

  return 'trace'
}
