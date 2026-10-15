import type { Message } from '../../api/chat'
import {
  resolveLowConfidenceMeta,
  stripLeadingLowConfidenceNotice,
  type LowConfidenceMetaLite,
} from './messageLowConfidence'

export interface AssistantCompletedContentState {
  bodyContent: string
  lowConfidenceMeta: LowConfidenceMetaLite | null
}

export function buildAssistantCompletedContentState(
  message: Message,
  rawBodyContent: string,
  S: Record<string, string>,
): AssistantCompletedContentState {
  const lowConfidenceMeta = message.role === 'assistant'
    ? resolveLowConfidenceMeta(
        message.meta && typeof message.meta === 'object'
          ? message.meta as Record<string, unknown>
          : null,
        rawBodyContent,
        S,
      )
    : null
  return {
    bodyContent: lowConfidenceMeta
      ? stripLeadingLowConfidenceNotice(rawBodyContent)
      : rawBodyContent,
    lowConfidenceMeta,
  }
}
