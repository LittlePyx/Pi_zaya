import type { Message } from '../../api/chat'
import { hasRefsPanelContent } from '../refs/refsPanelDisplay'
import {
  normalizeCiteDetail,
  toShelfItem,
  type CiteDetail,
  type CiteShelfItem,
} from './citationState'
import { getMessageCiteDetailRecords } from './messageRenderPacket'
import {
  getAssistantSelectedResearchContext,
  getUserPromptResearchContext,
} from './messageTraceUtils'
import type { SelectedResearchContextPack } from './researchContextPack'

export type MessageRow =
  | { kind: 'message'; message: Message }
  | { kind: 'refs'; userMsgId: number }

export interface MessageRowsFilter {
  activeSourcePath?: string
  activeSourceName?: string
}

export interface AssistantTracePosition {
  answerOrder: number
  userMsgId: number
}

export function buildMessageRows(
  messages: Message[],
  refs: Record<string, unknown>,
  filter: MessageRowsFilter,
): MessageRow[] {
  const out: MessageRow[] = []
  let lastUserMsgId = 0
  const renderedRefs = new Set<number>()

  for (const message of messages) {
    out.push({ kind: 'message', message })
    if (message.role === 'user') {
      lastUserMsgId = message.id
      continue
    }
    if (
      lastUserMsgId > 0
      && !renderedRefs.has(lastUserMsgId)
      && hasRefsPanelContent(refs, lastUserMsgId, filter)
    ) {
      out.push({ kind: 'refs', userMsgId: lastUserMsgId })
      renderedRefs.add(lastUserMsgId)
    }
  }

  if (
    lastUserMsgId > 0
    && !renderedRefs.has(lastUserMsgId)
    && hasRefsPanelContent(refs, lastUserMsgId, filter)
  ) {
    out.push({ kind: 'refs', userMsgId: lastUserMsgId })
  }

  return out
}

export function buildAssistantTraceByMsgId(messages: Message[]): Map<number, AssistantTracePosition> {
  const out = new Map<number, AssistantTracePosition>()
  let answerOrder = 0
  let lastUserMsgId = 0

  for (const message of messages) {
    if (message.role === 'user') {
      lastUserMsgId = message.id
      continue
    }
    if (message.role !== 'assistant') continue
    answerOrder += 1
    out.set(message.id, { answerOrder, userMsgId: lastUserMsgId })
  }

  return out
}

export function buildSelectedResearchContextByAssistantId(
  messages: Message[],
): Map<number, SelectedResearchContextPack> {
  const out = new Map<number, SelectedResearchContextPack>()
  let pendingUserContext: SelectedResearchContextPack | null = null

  for (const message of messages) {
    if (message.role === 'user') {
      pendingUserContext = getUserPromptResearchContext(message)
      continue
    }
    if (message.role !== 'assistant') continue
    const assistantContext = getAssistantSelectedResearchContext(message)
    if (assistantContext) {
      out.set(message.id, assistantContext)
    } else if (pendingUserContext) {
      out.set(message.id, pendingUserContext)
    }
    pendingUserContext = null
  }

  return out
}

export function buildLiveCiteMap(
  messages: Message[],
  conversationId: string | number | null | undefined,
  assistantTraceByMsgId: Map<number, AssistantTracePosition>,
): Map<string, CiteShelfItem> {
  const map = new Map<string, CiteShelfItem>()
  const traceConvId = String(conversationId || '')

  for (const message of messages) {
    if (message.role !== 'assistant') continue
    const rawCiteDetails = getMessageCiteDetailRecords(message)
    if (rawCiteDetails.length <= 0) continue
    const trace = assistantTraceByMsgId.get(message.id)
    for (const rawDetail of rawCiteDetails) {
      const detail = normalizeCiteDetail(rawDetail)
      if (!detail) continue
      const tracedDetail: CiteDetail = {
        ...detail,
        traceConvId,
        traceAssistantMsgId: message.id,
        traceAssistantOrder: Number(trace?.answerOrder || 0),
        traceUserMsgId: Number(trace?.userMsgId || 0),
      }
      const item = toShelfItem(tracedDetail)
      map.set(item.key, item)
    }
  }

  return map
}
