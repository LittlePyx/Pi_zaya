import type {
  ShelfReaderLocatorMaintenanceItem,
  ShelfReaderLocatorMaintenanceResult,
} from '../../api/references'
import type { CiteShelfItem } from './citationState'


function cleanText(value: unknown): string {
  return String(value || '').trim()
}


export function shelfReaderLocatorMaintenanceInput(
  item: CiteShelfItem,
): ShelfReaderLocatorMaintenanceItem | null {
  if (item.isInpaper) return null
  const sourcePath = cleanText(item.sourcePath)
  if (!sourcePath) return null
  const evidenceQuote = cleanText(item.readerEvidenceQuote || item.evidenceQuote)
  const shelfExcerpt = cleanText(item.shelfExcerpt)
  const cardEvidence = cleanText(item.cardEvidence)
  const raw = cleanText(item.raw)
  const blockId = cleanText(item.blockId)
  const anchorId = cleanText(item.anchorId)
  const headingPath = cleanText(item.headingPath)
  if (!blockId && !anchorId && !headingPath && !evidenceQuote && !shelfExcerpt && !cardEvidence && !raw) {
    return null
  }
  const fallbackSourcePath = String(item.libraryMatchStatus || '').trim().toLowerCase() === 'in_library'
    ? cleanText(item.libraryMatchPath)
    : ''
  return {
    key: cleanText(item.key),
    source_path: sourcePath,
    source_name: cleanText(item.sourceName),
    fallback_source_path: fallbackSourcePath || undefined,
    library_match_path: fallbackSourcePath || undefined,
    block_id: blockId || undefined,
    anchor_id: anchorId || undefined,
    anchor_kind: cleanText(item.anchorKind) || undefined,
    heading_path: headingPath || undefined,
    reader_evidence_quote: evidenceQuote || undefined,
    evidence_quote: evidenceQuote || undefined,
    shelf_excerpt: shelfExcerpt || undefined,
    card_evidence: cardEvidence || undefined,
    raw: raw || undefined,
    page_start: Number(item.pageStart || 0) || undefined,
    page_end: Number(item.pageEnd || 0) || undefined,
  }
}


export function shelfReaderLocatorMaintenanceFingerprint(item: CiteShelfItem): string {
  const input = shelfReaderLocatorMaintenanceInput(item)
  if (!input) return ''
  return JSON.stringify(input)
}


export function shelfReaderLocatorSourceKey(input: ShelfReaderLocatorMaintenanceItem): string {
  return cleanText(input.source_path).replace(/\\/g, '/').toLowerCase()
}


export function mergeShelfReaderLocatorMaintenanceResult(
  item: CiteShelfItem,
  result: ShelfReaderLocatorMaintenanceResult | null | undefined,
): CiteShelfItem {
  if (!result || !result.ok || result.key !== item.key || !result.locator) return item
  const locator = result.locator
  const relocatedSourcePath = result.source_relocated ? cleanText(result.source_path) : ''
  const next: CiteShelfItem = {
    ...item,
    sourcePath: relocatedSourcePath || item.sourcePath,
    blockId: cleanText(locator.block_id),
    anchorId: cleanText(locator.anchor_id),
    anchorKind: cleanText(locator.anchor_kind),
    headingPath: cleanText(locator.heading_path),
    pageStart: Number(locator.page_start || 0) || 0,
    pageEnd: Number(locator.page_end || 0) || 0,
  }
  if (
    next.sourcePath === item.sourcePath
    && next.blockId === item.blockId
    && next.anchorId === item.anchorId
    && next.anchorKind === item.anchorKind
    && next.headingPath === item.headingPath
    && next.pageStart === item.pageStart
    && next.pageEnd === item.pageEnd
  ) return item
  return next
}
