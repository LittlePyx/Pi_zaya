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
const AGENT_TRACE_DEBUG_SECTION_RE = /^(?:#{1,6}\s*)?(?:[*_`~\s]*)?(?:plan|tool\s+calls?|tools?|execution\s+details|verification|claim\s+verification)(?:[*_`~\s]*)?:?\s*$/i
const AGENT_TRACE_DEBUG_LOOKAHEAD_RE = /\b(?:retrieve_evidence|retrieve_references|build_reading_guide|compare_papers|generate_grounded_answer|verify_answer_citations|agent_trace|agentTrace|supported_claims|unsupported_claims|total_claims|question_type)\b/i

export function isAssistantSourceNoticeText(value: unknown): boolean {
  const text = String(value || '').replace(/\s+/g, ' ').trim()
  if (!text) return false
  if (/^Note:\s*no matching local knowledge-base evidence was found; this is an external model answer/i.test(text)) return true
  if (/^Note:\s*local citations\s*\[n\]\s*come from the knowledge base/i.test(text)) return true
  if (/^(?:注意|注)[:：]\s*本地知识库没有命中相关证据/.test(text)) return true
  if (/^(?:注意|注)[:：]\s*带\s*\[n\]\s*的内容来自本地知识库/.test(text)) return true
  return false
}

export function splitLeadingAssistantSourceNotice(value: unknown): { notice: string; body: string } {
  const text = String(value || '')
  if (!text.trim()) return { notice: '', body: text }
  const normalized = text.replace(/\r\n?/g, '\n')
  const lines = normalized.split('\n')
  const firstContentIndex = lines.findIndex((line) => Boolean(String(line || '').trim()))
  if (firstContentIndex < 0) return { notice: '', body: text }
  const firstLine = String(lines[firstContentIndex] || '').trim()
  if (!isAssistantSourceNoticeText(firstLine)) return { notice: '', body: text }
  const nextLines = [
    ...lines.slice(0, firstContentIndex),
    ...lines.slice(firstContentIndex + 1),
  ]
  while (nextLines.length > 0 && !String(nextLines[0] || '').trim()) nextLines.shift()
  return {
    notice: firstLine,
    body: nextLines.join('\n').trimStart(),
  }
}

function stripLeadingAssistantSourceNotice(value: unknown): string {
  const split = splitLeadingAssistantSourceNotice(value)
  return split.notice ? split.body : String(value || '')
}

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

function jsonBlockLooksLikeAgentTrace(lines: string[], index: number): boolean {
  const line = String(lines[index] || '').trim()
  if (!line.startsWith('{')) return false
  const lookahead = lines.slice(index, Math.min(lines.length, index + 12)).join('\n')
  return AGENT_TRACE_JSON_RE.test(lookahead)
}

function sectionLooksLikeAgentDebug(lines: string[], index: number): boolean {
  const line = String(lines[index] || '').trim()
  if (!AGENT_TRACE_DEBUG_SECTION_RE.test(line)) return false
  const lookahead = lines.slice(index + 1, Math.min(lines.length, index + 10)).join('\n')
  return AGENT_TRACE_DEBUG_LOOKAHEAD_RE.test(lookahead)
}

function findAgentTraceBoundary(text: string): number {
  const lines = String(text || '').replace(/\r\n?/g, '\n').split('\n')
  for (let i = 0; i < lines.length; i += 1) {
    if (
      lineLooksLikeAgentTraceBoundary(lines[i])
      || fencedBlockLooksLikeAgentTrace(lines, i)
      || jsonBlockLooksLikeAgentTrace(lines, i)
      || sectionLooksLikeAgentDebug(lines, i)
    ) {
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
  const presentation = packet
    ? (
        packet.renderedBody
        || packet.renderedContent
        || packet.answerMarkdown
        || message.content
        || ''
      )
    : (
        message.rendered_body
        || message.rendered_content
        || message.content
        || ''
      )
  const clean = cleanMessagePresentationText(message, presentation)
  return message.role === 'assistant' ? stripLeadingAssistantSourceNotice(clean) : clean
}

export function getMessageCiteDetailRecords(message: Message): Array<Record<string, unknown>> {
  const packet = getMessageRenderPacket(message)
  // A render packet is an atomic presentation contract. An intentionally
  // empty packet citation list must not resurrect stale legacy details.
  if (packet) return packet.citeDetails
  return Array.isArray(message.cite_details)
    ? message.cite_details.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === 'object')
    : []
}

export function getMessageCopyTextValue(message: Message): string {
  const packet = getMessageRenderPacket(message)
  const presentation = packet
    ? (
        packet.copyText
        || packet.renderedBody
        || packet.renderedContent
        || packet.answerMarkdown
        || message.content
        || ''
      )
    : (
        message.copy_text
        || message.rendered_body
        || message.rendered_content
        || message.content
        || ''
      )
  return cleanMessagePresentationText(message, presentation)
}

export function getMessageCopyMarkdownValue(message: Message): string | undefined {
  const packet = getMessageRenderPacket(message)
  const presentation = packet
    ? (
        packet.copyMarkdown
        || packet.renderedContent
        || packet.renderedBody
        || packet.answerMarkdown
        || message.content
        || ''
      )
    : (
        message.copy_markdown
        || message.rendered_content
        || message.rendered_body
        || ''
      )
  const value = cleanMessagePresentationText(message, presentation).trim()
  return value || undefined
}

export function getMessageNoticeValue(message: Message): string | undefined {
  const packet = getMessageRenderPacket(message)
  if (packet) return packet.notice || undefined
  const value = String(message.notice || '').trim()
  if (value) return value
  if (message.role !== 'assistant') return undefined
  const rendered = cleanMessagePresentationText(message, (
    message.rendered_body
    || message.rendered_content
    || message.content
    || ''
  ))
  const split = splitLeadingAssistantSourceNotice(rendered)
  return split.notice || undefined
}
