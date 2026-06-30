import type {
  Message,
  MessageMeta,
  MessageCitationDetail,
  MessagePaperGuideContracts,
  MessageProvenanceLocateTarget,
  MessageProvenanceReaderOpen,
  MessageRenderPacket,
  MessageUnlinkedReferenceCandidate,
} from '../../api/chat'

export interface MessageRenderPacketLite {
  answerMarkdown: string
  notice: string
  renderedBody: string
  renderedContent: string
  copyText: string
  copyMarkdown: string
  citeDetails: MessageCitationDetail[]
  unlinkedReferenceCandidates: MessageUnlinkedReferenceCandidate[]
  locateTarget: MessageProvenanceLocateTarget | null
  readerOpen: MessageProvenanceReaderOpen | null
}

const AGENT_TRACE_HEADING_RE = /^(?:#{1,6}\s*)?(?:[*_`~\s]*)?(?:research\s+agent\s+trace|agent\s+trace)(?:[*_`~\s]*)?:?\s*$/i
const AGENT_TRACE_KEY_RE = /^\s*(?:"?agent_trace"?|agentTrace)\s*[:=]/
const AGENT_TRACE_JSON_RE = /"(?:agent_trace|agentTrace)"\s*:/i

function lineLooksLikeAgentTraceBoundary(line: string): boolean {
  const text = String(line || '').trim()
  if (!text) return false
  if (AGENT_TRACE_HEADING_RE.test(text)) return true
  if (AGENT_TRACE_KEY_RE.test(text)) return true
  return false
}

function fencedBlockLooksLikeAgentTrace(lines: string[], index: number): boolean {
  const line = String(lines[index] || '').trim()
  if (!line.startsWith('```') && !line.startsWith('~~~')) return false
  const lookahead = lines.slice(index + 1, Math.min(lines.length, index + 12)).join('\n')
  return AGENT_TRACE_JSON_RE.test(lookahead)
}

function findAgentTraceBoundary(text: string): number {
  const lines = String(text || '').replace(/\r\n?/g, '\n').split('\n')
  for (let i = 0; i < lines.length; i += 1) {
    if (lineLooksLikeAgentTraceBoundary(lines[i]) || fencedBlockLooksLikeAgentTrace(lines, i)) {
      return i
    }
  }
  return -1
}

export function cleanAssistantAnswerPresentationText(value: unknown): string {
  const text = String(value || '')
  if (!text.trim()) return text
  const normalized = text.replace(/\r\n?/g, '\n')
  const lines = normalized.split('\n')
  const boundary = findAgentTraceBoundary(normalized)
  if (boundary < 0) return text
  return lines.slice(0, boundary).join('\n').trimEnd()
}

function cleanMessagePresentationText(message: Pick<Message, 'role'>, value: unknown): string {
  const text = String(value || '')
  return message.role === 'assistant' ? cleanAssistantAnswerPresentationText(text) : text
}

export function getMessageRenderPacket(message: Pick<Message, 'meta'>): MessageRenderPacketLite | null {
  const meta = (message.meta && typeof message.meta === 'object')
    ? message.meta as MessageMeta
    : null
  const contracts = (meta?.paper_guide_contracts && typeof meta.paper_guide_contracts === 'object')
    ? meta.paper_guide_contracts as MessagePaperGuideContracts
    : null
  const raw = (contracts?.render_packet && typeof contracts.render_packet === 'object')
    ? contracts.render_packet as MessageRenderPacket
    : null
  if (!raw) return null
  return {
    answerMarkdown: String(raw.answer_markdown || '').trim(),
    notice: String(raw.notice || '').trim(),
    renderedBody: String(raw.rendered_body || '').trim(),
    renderedContent: String(raw.rendered_content || '').trim(),
    copyText: String(raw.copy_text || '').trim(),
    copyMarkdown: String(raw.copy_markdown || '').trim(),
    citeDetails: Array.isArray(raw.cite_details)
      ? raw.cite_details.filter((item): item is MessageCitationDetail => Boolean(item) && typeof item === 'object')
      : [],
    unlinkedReferenceCandidates: Array.isArray(raw.unlinked_reference_candidates)
      ? raw.unlinked_reference_candidates.filter((item): item is MessageUnlinkedReferenceCandidate => Boolean(item) && typeof item === 'object')
      : [],
    locateTarget: (raw.locate_target && typeof raw.locate_target === 'object')
      ? raw.locate_target as MessageProvenanceLocateTarget
      : null,
    readerOpen: (raw.reader_open && typeof raw.reader_open === 'object')
      ? raw.reader_open as MessageProvenanceReaderOpen
      : null,
  }
}

export function getMessageRenderedBodyContent(message: Message): string {
  const packet = getMessageRenderPacket(message)
  return cleanMessagePresentationText(message, (
    packet?.renderedBody
    || packet?.renderedContent
    || packet?.answerMarkdown
    || message.rendered_body
    || message.rendered_content
    || message.content
    || ''
  ))
}

export function getMessageCiteDetailRecords(message: Message): Array<Record<string, unknown>> {
  const packet = getMessageRenderPacket(message)
  if (packet && packet.citeDetails.length > 0) return packet.citeDetails
  return Array.isArray(message.cite_details)
    ? message.cite_details.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === 'object')
    : []
}

export function getMessageCopyTextValue(message: Message): string {
  const packet = getMessageRenderPacket(message)
  return cleanMessagePresentationText(message, (
    packet?.copyText
    || message.copy_text
    || packet?.renderedBody
    || packet?.answerMarkdown
    || message.content
    || ''
  ))
}

export function getMessageCopyMarkdownValue(message: Message): string | undefined {
  const packet = getMessageRenderPacket(message)
  const value = cleanMessagePresentationText(message, (
    packet?.copyMarkdown
    || message.copy_markdown
    || packet?.renderedContent
    || packet?.renderedBody
    || ''
  )).trim()
  return value || undefined
}

export function getMessageNoticeValue(message: Message): string | undefined {
  const packet = getMessageRenderPacket(message)
  const value = String(packet?.notice || message.notice || '').trim()
  return value || undefined
}
