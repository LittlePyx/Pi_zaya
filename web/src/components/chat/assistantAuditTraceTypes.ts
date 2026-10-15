import type { AgentTraceReferenceHandlers } from './agentTraceReferenceTypes'
import type { LoadArchivedAgentTrace } from './useArchivedAgentTrace'

export interface AssistantAuditTraceInteractions extends AgentTraceReferenceHandlers {
  onLoadAgentTrace?: LoadArchivedAgentTrace
}
