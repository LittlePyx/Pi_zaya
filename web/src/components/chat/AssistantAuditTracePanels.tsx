import type { Message } from '../../api/chat'
import { AgentTracePanel } from './AgentTracePanel'
import type { AssistantAuditTraceInteractions } from './assistantAuditTraceTypes'
import {
  getMessageAgentTrace,
  getMessageResearchTrace,
  messageHasAgentTraceHint,
} from './messageTraceUtils'
import { ResearchTracePanel } from './ResearchTracePanel'

interface AssistantAuditTracePanelsProps {
  interactions: AssistantAuditTraceInteractions
  message: Message
}

export function AssistantAuditTracePanels({
  interactions,
  message,
}: AssistantAuditTracePanelsProps) {
  if (message.role !== 'assistant') return null

  return (
    <>
      <AgentTracePanel
        trace={getMessageAgentTrace(message)}
        messageId={message.id}
        canLoadTrace={messageHasAgentTraceHint(message)}
        onLoadTrace={interactions.onLoadAgentTrace}
        onOpenReference={interactions.onOpenReference}
        onAddReferenceToShelf={interactions.onAddReferenceToShelf}
      />
      <ResearchTracePanel trace={getMessageResearchTrace(message)} />
    </>
  )
}
