import type {
  Message,
  MessageMeta,
  MessageCitationDetail,
  MessagePaperGuideContracts,
  MessageProvenanceLocateTarget,
  MessageProvenanceReaderOpen,
  MessageRenderPacket,
} from '../../api/chat'

export interface MessageRenderPacketLite {
  answerMarkdown: string
  notice: string
  renderedBody: string
  renderedContent: string
  copyText: string
  copyMarkdown: string
  citeDetails: MessageCitationDetail[]
  locateTarget: MessageProvenanceLocateTarget | null
  readerOpen: MessageProvenanceReaderOpen | null
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
  return String(
    packet?.renderedBody
    || packet?.renderedContent
    || message.rendered_body
    || message.rendered_content
    || message.content
    || '',
  )
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
  return String(
    packet?.copyText
    || message.copy_text
    || packet?.renderedBody
    || packet?.answerMarkdown
    || message.content
    || '',
  )
}

export function getMessageCopyMarkdownValue(message: Message): string | undefined {
  const packet = getMessageRenderPacket(message)
  const value = String(
    packet?.copyMarkdown
    || message.copy_markdown
    || packet?.renderedContent
    || packet?.renderedBody
    || '',
  ).trim()
  return value || undefined
}

export function getMessageNoticeValue(message: Message): string | undefined {
  const packet = getMessageRenderPacket(message)
  const value = String(packet?.notice || message.notice || '').trim()
  return value || undefined
}
