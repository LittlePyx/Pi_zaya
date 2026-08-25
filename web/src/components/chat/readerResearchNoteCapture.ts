import type { ReaderDocBlock } from '../../api/references'
import type {
  ResearchNoteRecord,
  ResearchNoteSourceLink,
} from '../../api/chat'
import type { ReaderSelectionShelfPayload } from './reader/readerTypes'

export const RESEARCH_NOTES_CHANGED_EVENT = 'kb:research-notes-changed'
export const RESEARCH_NOTES_SYNC_CHANNEL = 'kb:research-notes-sync'

export type ReaderResearchNoteCaptureKind = 'selection' | 'table' | 'equation' | 'figure'

export interface ReaderResearchNoteCapture extends ReaderSelectionShelfPayload {
  captureKind: ReaderResearchNoteCaptureKind
  captureId: string
  locationLabel?: string
  pageStart?: number
  pageEnd?: number
  lineStart?: number
  lineEnd?: number
  assetSrc?: string
}

interface BuildCaptureOptions {
  markdown: string
  payload: ReaderSelectionShelfPayload
  readerBlocks: ReaderDocBlock[]
}

function compact(value: unknown): string {
  return String(value || '').replace(/\s+/g, ' ').trim()
}

function normalizedPath(value: unknown): string {
  return String(value || '').trim().replace(/\\/g, '/').replace(/\/+$/g, '').toLowerCase()
}

function finiteNonNegative(value: unknown): number | undefined {
  if (value == null) return undefined
  const number = Number(value)
  return Number.isFinite(number) && number >= 0 ? Math.floor(number) : undefined
}

function stableHash(value: string): string {
  let hash = 2166136261
  for (let idx = 0; idx < value.length; idx += 1) {
    hash ^= value.charCodeAt(idx)
    hash = Math.imul(hash, 16777619)
  }
  return (hash >>> 0).toString(36)
}

function captureKind(payload: ReaderSelectionShelfPayload): ReaderResearchNoteCaptureKind {
  const explicit = String(payload.captureKind || '').trim().toLowerCase()
  if (explicit === 'table' || explicit === 'equation' || explicit === 'figure') return explicit
  if (explicit === 'selection') return 'selection'
  const anchorKind = String(payload.anchorKind || '').trim().toLowerCase()
  if (anchorKind === 'table' || anchorKind === 'equation' || anchorKind === 'figure') return anchorKind
  return 'selection'
}

function findCaptureBlock(payload: ReaderSelectionShelfPayload, blocks: ReaderDocBlock[]): ReaderDocBlock | null {
  const blockId = String(payload.blockId || '').trim()
  const anchorId = String(payload.anchorId || '').trim()
  return blocks.find((block) => blockId && String(block.block_id || '').trim() === blockId)
    || blocks.find((block) => anchorId && String(block.anchor_id || '').trim() === anchorId)
    || null
}

function pageAtLine(markdown: string, targetLine: number | undefined): number | undefined {
  if (!targetLine || targetLine <= 0) return undefined
  const lines = String(markdown || '').split(/\r?\n/)
  let page: number | undefined
  const limit = Math.min(lines.length, targetLine)
  for (let idx = 0; idx < limit; idx += 1) {
    const match = lines[idx].match(/<!--\s*kb_page:\s*(\d+)\s*-->/i)
    if (match) page = Number(match[1])
  }
  return page
}

function markdownLinesForBlock(markdown: string, block: ReaderDocBlock | null): string {
  const start = finiteNonNegative(block?.line_start)
  const end = finiteNonNegative(block?.line_end)
  if (!start || !end || end < start) return ''
  return String(markdown || '')
    .split(/\r?\n/)
    .slice(start - 1, end)
    .filter((line) => !/<!--\s*kb_page:\s*\d+\s*-->/i.test(line))
    .join('\n')
    .trim()
}

function structuredText(
  kind: ReaderResearchNoteCaptureKind,
  payloadText: string,
  block: ReaderDocBlock | null,
  markdown: string,
): string {
  if (kind === 'selection') return String(payloadText || '').trim()
  const raw = String(markdownLinesForBlock(markdown, block) || block?.raw_text || block?.text || payloadText || '').trim()
  return raw || String(payloadText || '').trim()
}

function markdownImageSource(value: string): string {
  const match = String(value || '').match(/!\[[^\]]*]\((?:<([^>]+)>|([^\s)]+))(?:\s+["'][^"']*["'])?\)/)
  return String(match?.[1] || match?.[2] || '').trim()
}

export function readerCaptureIdentity(input: Partial<ReaderResearchNoteCapture | ResearchNoteSourceLink>): string {
  const record = input as Record<string, unknown>
  const sourcePath = normalizedPath(record.sourcePath ?? record.source_path)
  const blockId = compact(record.blockId ?? record.block_id)
  const anchorId = compact(record.anchorId ?? record.anchor_id)
  const startOffset = finiteNonNegative(record.startOffset ?? record.start_offset)
  const endOffset = finiteNonNegative(record.endOffset ?? record.end_offset)
  const lineStart = finiteNonNegative(record.lineStart ?? record.line_start)
  const evidence = compact(record.text ?? record.evidence_quote).toLowerCase()
  const locator = [blockId, anchorId, startOffset ?? '', endOffset ?? '', lineStart ?? ''].join('|')
  return `reader:${stableHash(`${sourcePath}|${locator}|${evidence}`)}`
}

export function buildReaderResearchNoteCapture({
  markdown,
  payload,
  readerBlocks,
}: BuildCaptureOptions): ReaderResearchNoteCapture | null {
  const sourcePath = String(payload.sourcePath || '').trim()
  const kind = captureKind(payload)
  const block = findCaptureBlock(payload, readerBlocks)
  const text = structuredText(kind, payload.text, block, markdown)
  if (!sourcePath || !text) return null
  const lineStart = finiteNonNegative(block?.line_start)
  const lineEnd = finiteNonNegative(block?.line_end)
  const pageStart = pageAtLine(markdown, lineStart)
  const pageEnd = pageAtLine(markdown, lineEnd) || pageStart
  const headingPath = String(payload.headingPath || block?.heading_path || '').trim() || undefined
  const pageLabel = pageStart
    ? (pageEnd && pageEnd !== pageStart ? `pp. ${pageStart}-${pageEnd}` : `p. ${pageStart}`)
    : ''
  const locationLabel = [pageLabel, headingPath].filter(Boolean).join(' · ') || undefined
  const assetSrc = String(payload.assetSrc || (kind === 'figure' ? markdownImageSource(text) : '')).trim() || undefined
  const base: ReaderResearchNoteCapture = {
    ...payload,
    text,
    sourcePath,
    sourceName: String(payload.sourceName || '').trim() || undefined,
    headingPath,
    blockId: String(payload.blockId || block?.block_id || '').trim() || undefined,
    anchorId: String(payload.anchorId || block?.anchor_id || '').trim() || undefined,
    anchorKind: String(payload.anchorKind || block?.kind || kind).trim() || undefined,
    captureKind: kind,
    captureId: '',
    locationLabel,
    pageStart,
    pageEnd,
    lineStart,
    lineEnd,
    assetSrc,
  }
  base.captureId = readerCaptureIdentity(base)
  return base
}

export function readerCaptureSourceLink(
  capture: ReaderResearchNoteCapture,
  conversationId?: string | null,
): ResearchNoteSourceLink {
  return {
    kind: 'source',
    label: String(capture.sourceName || capture.headingPath || 'Source').trim(),
    conversation_id: String(conversationId || capture.conversationId || '').trim() || undefined,
    source_path: capture.sourcePath,
    source_name: capture.sourceName,
    heading_path: capture.headingPath,
    location_label: capture.locationLabel,
    evidence_quote: compact(capture.text).slice(0, 1800),
    page_start: capture.pageStart,
    page_end: capture.pageEnd,
    block_id: capture.blockId,
    anchor_id: capture.anchorId,
    anchor_kind: capture.anchorKind,
    capture_id: capture.captureId,
    capture_kind: capture.captureKind,
    start_offset: finiteNonNegative(capture.startOffset),
    end_offset: finiteNonNegative(capture.endOffset),
    occurrence: finiteNonNegative(capture.occurrence),
    readable_index: finiteNonNegative(capture.readableIndex),
    document_occurrence: finiteNonNegative(capture.documentOccurrence),
    start_readable_index: finiteNonNegative(capture.startReadableIndex),
    end_readable_index: finiteNonNegative(capture.endReadableIndex),
    line_start: capture.lineStart,
    line_end: capture.lineEnd,
    asset_src: capture.assetSrc,
  }
}

export function researchNoteHasCapture(note: ResearchNoteRecord, capture: ReaderResearchNoteCapture): boolean {
  const links = Array.isArray(note.source_state?.links) ? note.source_state.links : []
  return links.some((link) => {
    if (String(link?.capture_id || '').trim()) return String(link.capture_id).trim() === capture.captureId
    return readerCaptureIdentity(link) === capture.captureId
  })
}

function quoteMarkdown(value: string): string {
  return String(value || '')
    .trim()
    .split(/\r?\n/)
    .map((line) => `> ${line || ' '}`)
    .join('\n')
}

function captureHeading(kind: ReaderResearchNoteCaptureKind, labels: Record<ReaderResearchNoteCaptureKind, string>): string {
  return labels[kind] || labels.selection
}

function captureBody(capture: ReaderResearchNoteCapture): string {
  const text = String(capture.text || '').trim()
  if (capture.captureKind === 'selection') return quoteMarkdown(text)
  if (capture.captureKind === 'equation') {
    if (/^\$\$[\s\S]+\$\$$/.test(text) || /^\\\[[\s\S]+\\\]$/.test(text)) return text
    return `$$\n${text}\n$$`
  }
  if (capture.captureKind === 'figure') {
    if (/!\[[^\]]*]\([^)]+\)/.test(text)) return text
    if (capture.assetSrc) return `![${compact(text) || 'Figure'}](${capture.assetSrc})`
    return quoteMarkdown(text)
  }
  return text
}

export function buildResearchNoteCaptureSection(
  capture: ReaderResearchNoteCapture,
  comment: string,
  labels: {
    kinds: Record<ReaderResearchNoteCaptureKind, string>
    location: string
    comment: string
  },
): string {
  const heading = captureHeading(capture.captureKind, labels.kinds)
  const source = String(capture.sourceName || '').trim()
  const title = source ? `${heading}｜${source}` : heading
  const parts = [`## ${title}`]
  if (capture.locationLabel) parts.push(`**${labels.location}：** ${capture.locationLabel}`)
  parts.push(captureBody(capture))
  const note = String(comment || '').trim()
  if (note) parts.push(`**${labels.comment}**\n\n${note}`)
  return parts.filter(Boolean).join('\n\n').trim()
}

export function appendResearchNoteCapture(existing: string, section: string): string {
  const current = String(existing || '').trim()
  const addition = String(section || '').trim()
  if (!current) return addition
  if (!addition) return current
  return `${current}\n\n---\n\n${addition}`
}

export function readerCaptureDraftKey(capture: ReaderResearchNoteCapture): string {
  return `kb:reader-note-capture:${capture.captureId}`
}
