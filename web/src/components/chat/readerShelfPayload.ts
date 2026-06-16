import type { ReaderCitationShelfPayload, ReaderSelectionShelfPayload } from './reader/readerTypes'
import {
  buildCiteDetailFromMeta,
  cleanCitationDisplayText,
  normalizeCiteDetail,
  normalizeShelfNote,
  type CiteDetail,
} from './citationState'

export function normalizeReaderSelectionShelfPayload(raw: unknown): ReaderSelectionShelfPayload | null {
  const rec = raw && typeof raw === 'object' ? raw as Record<string, unknown> : {}
  const text = cleanCitationDisplayText(String(rec.text || '')).trim()
  const sourcePath = String(rec.sourcePath || rec.source_path || '').trim()
  if (!text || !sourcePath) return null
  const numberField = (key: string): number | undefined => {
    const value = Number(rec[key])
    return Number.isFinite(value) ? value : undefined
  }
  return {
    text,
    sourcePath,
    sourceName: String(rec.sourceName || rec.source_name || '').trim() || undefined,
    headingPath: String(rec.headingPath || rec.heading_path || '').trim() || undefined,
    blockId: String(rec.blockId || rec.block_id || '').trim() || undefined,
    anchorId: String(rec.anchorId || rec.anchor_id || '').trim() || undefined,
    anchorKind: String(rec.anchorKind || rec.anchor_kind || '').trim() || undefined,
    startOffset: numberField('startOffset'),
    endOffset: numberField('endOffset'),
    occurrence: numberField('occurrence'),
    readableIndex: numberField('readableIndex'),
    documentOccurrence: numberField('documentOccurrence'),
    startReadableIndex: numberField('startReadableIndex'),
    endReadableIndex: numberField('endReadableIndex'),
    conversationId: String(rec.conversationId || rec.conversation_id || '').trim() || undefined,
    projectId: String(rec.projectId || rec.project_id || '').trim() || undefined,
    createdAt: numberField('createdAt') || Date.now(),
  }
}

export function normalizeReaderCitationShelfPayload(raw: unknown): ReaderCitationShelfPayload | null {
  const rec = raw && typeof raw === 'object' ? raw as Record<string, unknown> : {}
  const detailRaw = (rec.detail && typeof rec.detail === 'object')
    ? rec.detail as Record<string, unknown>
    : (rec.citationDetail && typeof rec.citationDetail === 'object')
      ? rec.citationDetail as Record<string, unknown>
      : null
  if (!detailRaw) return null
  const detail = normalizeCiteDetail(detailRaw)
  if (!detail) return null
  return {
    detail: detail as unknown as Record<string, unknown>,
    conversationId: String(rec.conversationId || rec.conversation_id || '').trim() || undefined,
    projectId: String(rec.projectId || rec.project_id || '').trim() || undefined,
    createdAt: Number.isFinite(Number(rec.createdAt)) ? Number(rec.createdAt) : Date.now(),
  }
}

export function readerSelectionShelfTitle(payload: ReaderSelectionShelfPayload): string {
  const fallback = String(payload.sourcePath || '').split(/[\\/]/).pop() || 'Reader selection'
  return cleanCitationDisplayText(String(payload.sourceName || fallback || 'Reader selection'))
    .replace(/\.en\.md$/i, '')
    .replace(/\.md$/i, '')
    .replace(/\.pdf$/i, '')
    .trim() || fallback
}

export function readerSelectionAnchor(payload: ReaderSelectionShelfPayload): string {
  return [
    'reader-selection',
    payload.sourcePath,
    payload.blockId || payload.anchorId || '',
    payload.startOffset ?? '',
    payload.endOffset ?? '',
  ].join(':')
}

export function readerSelectionNote(
  payload: ReaderSelectionShelfPayload,
  labels: { shelf_reader_selection_selected?: string; shelf_reader_selection_source?: string },
): string {
  const heading = String(payload.headingPath || '').trim()
  const source = readerSelectionShelfTitle(payload)
  const selectedLabel = labels.shelf_reader_selection_selected || 'Selected text'
  const sourceLabel = labels.shelf_reader_selection_source || 'Source'
  const parts = [
    `${sourceLabel}: ${heading ? `${source} / ${heading}` : source}`,
    `${selectedLabel}: ${payload.text}`,
  ]
  return parts.filter(Boolean).join('\n')
}

export function mergeSelectionNote(existing: string, next: string): string {
  const current = normalizeShelfNote(existing)
  const incoming = normalizeShelfNote(next)
  if (!incoming) return current
  if (!current) return incoming
  if (current.includes(incoming)) return current
  return `${current}\n\n${incoming}`.trim()
}

export function citeDetailFromReaderSelection(
  payload: ReaderSelectionShelfPayload,
  activeConvId?: string | null,
): CiteDetail | null {
  const sourceName = String(payload.sourceName || '').trim()
    || String(payload.sourcePath || '').split(/[\\/]/).pop()
    || 'Reader selection'
  const title = readerSelectionShelfTitle({ ...payload, sourceName })
  const meta = {
    anchor: readerSelectionAnchor(payload),
    num: 0,
    source_name: sourceName,
    source_path: payload.sourcePath,
    trace_conv_id: String(activeConvId || payload.conversationId || ''),
    raw: payload.text,
    cite_fmt: payload.text,
    is_inpaper: false,
    title,
    heading_path: payload.headingPath || '',
    evidence_quote: payload.text,
    evidence_source: 'reader_selection',
    citation_context: payload.text,
    citation_context_source: 'reader_selection',
    shelf_item_kind: 'reader_selection',
    shelf_origin: 'reader_selection',
    shelf_excerpt: payload.text,
    location_label: payload.headingPath || '',
    block_id: payload.blockId || '',
    anchor_id: payload.anchorId || '',
    anchor_kind: payload.anchorKind || '',
    card_kind: 'reader_selection',
    card_title: title,
    card_subtitle: payload.headingPath || '',
    card_claim: payload.text,
    card_evidence: payload.text,
  }
  return buildCiteDetailFromMeta(meta, {
    sourceName,
    sourcePath: payload.sourcePath,
    anchor: readerSelectionAnchor(payload),
  })
}
