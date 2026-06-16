import type { ShelfMetadataRepairItem } from '../../api/references'
import {
  cleanCitationDisplayText,
  isLikelyWeakCitationTitle,
  looksLowValueShelfSummary,
  normalizeCiteDetail,
  normalizeShelfItemKind,
  normalizeShelfNote,
  normalizeShelfTags,
  shelfItemMetadataQualityReady,
  shelfItemNeedsMetadataRepair,
  shelfItemRepairFingerprint,
  toShelfItem,
  type CiteDetail,
  type CiteShelfItem,
} from './citationState'

export { looksLowValueShelfSummary }
export const SHELF_MAX_ITEMS = 120

export function normalizeDoiLike(value: string): string {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/^https?:\/\/(?:dx\.)?doi\.org\//i, '')
    .replace(/^[\s"'`([{<]+|[\s"'`)\]}>.,;:]+$/g, '')
    .trim()
}

export function normalizeTitleLike(value: string): string {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9\u4e00-\u9fff]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

export function shelfKind(item: Pick<CiteShelfItem, 'shelfItemKind'>): string {
  return normalizeShelfItemKind(String(item.shelfItemKind || ''))
}

export function shelfPaperIdentity(item: CiteShelfItem): string {
  if (shelfKind(item) === 'reader_selection') {
    const source = String(item.sourcePath || item.sourceName || '').trim().toLowerCase()
    const anchor = String(item.blockId || item.anchorId || item.anchor || '').trim().toLowerCase()
    const excerpt = normalizeTitleLike(item.shelfExcerpt || item.evidenceQuote || item.raw || item.main)
    const offsets = item as CiteShelfItem & { startOffset?: number; endOffset?: number }
    return `reader-selection:${source}|${anchor}|${offsets.startOffset ?? ''}|${offsets.endOffset ?? ''}|${excerpt.slice(0, 160)}`
  }
  const doi = normalizeDoiLike(item.doi || item.doiUrl)
  if (doi) return `doi:${doi}`
  const title = normalizeTitleLike(item.title || item.main)
  const year = /^\d{4}$/.test(String(item.year || '').trim()) ? String(item.year).trim() : ''
  if (title) return `title:${title}|${year}`
  return `key:${String(item.key || '').trim()}`
}

export function shelfSourceIdentity(item: Pick<CiteShelfItem, 'sourcePath' | 'sourceName'>): string {
  return String(item.sourcePath || item.sourceName || '')
    .trim()
    .toLowerCase()
}

export function shouldMergeShelfItemsBySource(existing: CiteShelfItem, incoming: CiteShelfItem, sourceIdentity: string): boolean {
  if (!sourceIdentity) return false
  if (shelfKind(existing) === 'reader_selection') return false
  if (shelfKind(incoming) === 'reader_selection') return false
  return shelfSourceIdentity(existing) === sourceIdentity
}

export function dedupeShelfItems(items: CiteShelfItem[]): CiteShelfItem[] {
  const seen = new Set<string>()
  const out: CiteShelfItem[] = []
  for (const item of items || []) {
    const key = shelfPaperIdentity(item)
    if (!key || seen.has(key)) continue
    seen.add(key)
    out.push(item)
    if (out.length >= SHELF_MAX_ITEMS) break
  }
  return out
}

export function isWeakTitle(text: string): boolean {
  const t = String(text || '').trim()
  if (!t) return true
  if (/\bIn\s+[A-Z]/.test(t)) return true
  return isLikelyWeakCitationTitle(t)
}

export function isWeakAuthors(text: string): boolean {
  const t = String(text || '').trim()
  if (!t) return true
  if (/\b(?:journal|conference|proceedings|vol\.?|pp\.?)\b/i.test(t)) return true
  const tokens = t.match(/[A-Za-z\u4e00-\u9fff]+/g) || []
  return tokens.length <= 1
}

export function isWeakVenue(text: string): boolean {
  const t = String(text || '').trim()
  if (!t) return true
  const tokens = t.match(/[A-Za-z0-9\u4e00-\u9fff]+/g) || []
  return tokens.length <= 1
}

export function preferRicherField(field: 'title' | 'authors' | 'venue' | 'year' | 'main', current: string, incoming: string): string {
  const cur = String(current || '').trim()
  const inc = String(incoming || '').trim()
  if (!cur) return inc
  if (!inc) return cur
  if (field === 'year') {
    const curOk = /^\d{4}$/.test(cur)
    const incOk = /^\d{4}$/.test(inc)
    if (curOk && !incOk) return cur
    if (!curOk && incOk) return inc
    return cur
  }
  if (field === 'title') {
    const curWeak = isWeakTitle(cur)
    const incWeak = isWeakTitle(inc)
    if (curWeak && !incWeak) return inc
    if (!curWeak && incWeak) return cur
  } else if (field === 'authors') {
    const curWeak = isWeakAuthors(cur)
    const incWeak = isWeakAuthors(inc)
    if (curWeak && !incWeak) return inc
    if (!curWeak && incWeak) return cur
  } else if (field === 'venue') {
    const curWeak = isWeakVenue(cur)
    const incWeak = isWeakVenue(inc)
    if (curWeak && !incWeak) return inc
    if (!curWeak && incWeak) return cur
  } else if (field === 'main') {
    const curWeak = isWeakTitle(cur)
    const incWeak = isWeakTitle(inc)
    if (curWeak && !incWeak) return inc
    if (!curWeak && incWeak) return cur
  }
  return inc.length > cur.length + 12 ? inc : cur
}

export function sameTags(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) return false
  }
  return true
}

export function sameShelfItem(a: CiteShelfItem, b: CiteShelfItem): boolean {
  return (
    a.key === b.key
    && a.main === b.main
    && a.traceConvId === b.traceConvId
    && a.traceAssistantMsgId === b.traceAssistantMsgId
    && a.traceAssistantOrder === b.traceAssistantOrder
    && a.traceUserMsgId === b.traceUserMsgId
    && a.sourceName === b.sourceName
    && a.sourcePath === b.sourcePath
    && a.raw === b.raw
    && a.citeFmt === b.citeFmt
    && a.title === b.title
    && a.authors === b.authors
    && a.venue === b.venue
    && a.year === b.year
    && a.volume === b.volume
    && a.issue === b.issue
    && a.pages === b.pages
    && a.doi === b.doi
    && a.doiUrl === b.doiUrl
    && a.citationCount === b.citationCount
    && a.citationSource === b.citationSource
    && a.journalIf === b.journalIf
    && a.journalQuartile === b.journalQuartile
    && a.journalIfSource === b.journalIfSource
    && a.venueKind === b.venueKind
    && a.venueVerifiedBy === b.venueVerifiedBy
    && a.openalexVenue === b.openalexVenue
    && a.conferenceTier === b.conferenceTier
    && a.conferenceRankSource === b.conferenceRankSource
    && a.conferenceCcf === b.conferenceCcf
    && a.conferenceCcfSource === b.conferenceCcfSource
    && a.conferenceName === b.conferenceName
    && a.conferenceAcronym === b.conferenceAcronym
    && a.num === b.num
    && a.anchor === b.anchor
    && a.bibliometricsChecked === b.bibliometricsChecked
    && a.libraryMatchStatus === b.libraryMatchStatus
    && a.libraryMatchConfidence === b.libraryMatchConfidence
    && a.libraryMatchMethod === b.libraryMatchMethod
    && a.libraryMatchReason === b.libraryMatchReason
    && a.libraryMatchPath === b.libraryMatchPath
    && a.libraryMatchSha1 === b.libraryMatchSha1
    && a.libraryMatchTitle === b.libraryMatchTitle
    && a.libraryMatchDoi === b.libraryMatchDoi
    && a.libraryMatchYear === b.libraryMatchYear
    && JSON.stringify(a.metadataQuality || null) === JSON.stringify(b.metadataQuality || null)
    && a.metadataRepairStatus === b.metadataRepairStatus
    && JSON.stringify(a.metadataRepairSources || []) === JSON.stringify(b.metadataRepairSources || [])
    && JSON.stringify(a.metadataChangedFields || []) === JSON.stringify(b.metadataChangedFields || [])
    && a.externalMetadataStatus === b.externalMetadataStatus
    && a.externalMetadataReason === b.externalMetadataReason
    && a.externalMatchMethod === b.externalMatchMethod
    && a.externalMatchScore === b.externalMatchScore
    && a.externalTitleSimilarity === b.externalTitleSimilarity
    && a.externalTitle === b.externalTitle
    && a.externalAuthors === b.externalAuthors
    && a.externalVenue === b.externalVenue
    && a.externalYear === b.externalYear
    && a.externalDoi === b.externalDoi
    && a.externalDoiUrl === b.externalDoiUrl
    && a.summaryLine === b.summaryLine
    && a.summarySource === b.summarySource
    && a.summaryProvider === b.summaryProvider
    && JSON.stringify(a.summaryQuality || null) === JSON.stringify(b.summaryQuality || null)
    && a.shelfItemKind === b.shelfItemKind
    && a.shelfOrigin === b.shelfOrigin
    && a.shelfExcerpt === b.shelfExcerpt
    && a.shelfExcerptLabel === b.shelfExcerptLabel
    && a.answerClaim === b.answerClaim
    && a.headingPath === b.headingPath
    && a.evidenceQuote === b.evidenceQuote
    && a.evidenceSource === b.evidenceSource
    && a.citationContext === b.citationContext
    && a.citationContextSource === b.citationContextSource
    && a.upstreamWorkRole === b.upstreamWorkRole
    && a.userQuestionRelation === b.userQuestionRelation
    && a.locationLabel === b.locationLabel
    && a.supportRelation === b.supportRelation
    && a.whyLine === b.whyLine
    && a.blockId === b.blockId
    && a.anchorId === b.anchorId
    && a.anchorKind === b.anchorKind
    && a.pageStart === b.pageStart
    && a.pageEnd === b.pageEnd
    && a.score === b.score
    && a.note === b.note
    && sameTags(a.tags || [], b.tags || [])
  )
}

export function sameShelfItems(a: CiteShelfItem[], b: CiteShelfItem[]): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i += 1) {
    if (!sameShelfItem(a[i], b[i])) return false
  }
  return true
}

export function shelfItemsForBackend(items: CiteShelfItem[]): Array<Record<string, unknown>> {
  return dedupeShelfItems(items || [])
    .slice(0, SHELF_MAX_ITEMS)
    .map((item) => ({ ...(item as unknown as Record<string, unknown>) }))
}

export function preferExistingText(current: string, incoming: string): string {
  const cur = String(current || '').trim()
  if (cur) return cur
  return String(incoming || '').trim()
}

export function preferPositiveNumber(current: number, incoming: number): number {
  const cur = Number(current || 0)
  if (Number.isFinite(cur) && cur > 0) return cur
  const inc = Number(incoming || 0)
  if (Number.isFinite(inc) && inc > 0) return inc
  return 0
}

export function preferStringArray(current: string[], incoming: string[], preferIncoming = false): string[] {
  const cur = Array.isArray(current) ? current.map((item) => String(item || '').trim()).filter(Boolean) : []
  const inc = Array.isArray(incoming) ? incoming.map((item) => String(item || '').trim()).filter(Boolean) : []
  if (preferIncoming && inc.length > 0) return inc
  return cur.length > 0 ? cur : inc
}

export function metadataQualityRecordReady(value: unknown): boolean {
  if (!value || typeof value !== 'object') return false
  const rec = value as Record<string, unknown>
  const status = String(rec.status || '').trim().toLowerCase()
  return rec.ok === true || status === 'ready'
}

export function preferMetadataQuality(
  current: Record<string, unknown> | null,
  incoming: Record<string, unknown> | null,
): Record<string, unknown> | null {
  const currentReady = metadataQualityRecordReady(current)
  const incomingReady = metadataQualityRecordReady(incoming)
  if (currentReady || !incomingReady) return current || incoming || null
  return incoming
}

export function shelfItemHasMetadataHydrationSeed(item: CiteShelfItem): boolean {
  return Boolean(
    item.raw
    || item.citeFmt
    || item.title
    || item.doi
    || item.doiUrl
    || item.sourcePath
    || item.sourceName
  )
}

export function shelfItemNeedsPersistedMetadataHydrate(item: CiteShelfItem): boolean {
  if (!shelfItemHasMetadataHydrationSeed(item)) return false
  if (shelfItemMetadataQualityReady(item)) return false
  return shelfItemNeedsMetadataRepair(item) || !item.bibliometricsChecked || !item.metadataQuality
}

export function shelfItemHasDisplayableArticleSummary(item: CiteShelfItem): boolean {
  const summaryLine = String(item.summaryLine || '').trim()
  if (!summaryLine || looksLowValueShelfSummary(summaryLine)) return false
  const summarySource = String(item.summarySource || '').trim().toLowerCase()
  const quality = item.summaryQuality || {}
  const qualityOk = quality.ok === true || String(quality.status || '').trim().toLowerCase() === 'grounded'
  const contextOnlySource = [
    'citation_context',
    'citation_card',
    'citation_card_view',
    'metadata',
    'reference_primary_evidence',
    'references_panel_hit',
  ].includes(summarySource)
  if (contextOnlySource) return false
  const articleSummarySource = [
    'abstract',
    'fulltext',
    'navigation',
    'exact_anchor',
    'section_intent_rescue',
    'doc_list_seed',
    'doc_list_prompt_aligned',
  ].includes(summarySource)
  return Boolean(articleSummarySource || (!item.isInpaper && qualityOk))
}

export function shelfItemNeedsSummaryBackfill(item: CiteShelfItem): boolean {
  if (!shelfItemHasMetadataHydrationSeed(item)) return false
  return !shelfItemHasDisplayableArticleSummary(item)
}

export function shelfMetadataHydrateAttemptKey(item: CiteShelfItem): string {
  return [
    item.key,
    shelfItemRepairFingerprint(item),
  ].join('|')
}

export function shelfSummaryBackfillAttemptKey(item: CiteShelfItem): string {
  const quality = item.summaryQuality || {}
  return [
    item.key,
    shelfItemRepairFingerprint(item),
    String(item.summaryLine || '').trim(),
    String(item.summarySource || '').trim(),
    String(quality.status || ''),
  ].join('|')
}

export function articleSummaryPatchFromMeta(meta: Record<string, unknown>): Partial<CiteShelfItem> {
  const line = cleanCitationDisplayText(String(meta.summary_line || meta.summaryLine || '')).trim()
  const source = String(meta.summary_source || meta.summarySource || '').trim()
  const provider = String(meta.summary_provider || meta.summaryProvider || '').trim()
  const sourceKey = source.toLowerCase()
  if (!line || looksLowValueShelfSummary(line)) return {}
  if ([
    'citation_context',
    'citation_card',
    'citation_card_view',
    'metadata',
    'reference_primary_evidence',
    'references_panel_hit',
  ].includes(sourceKey)) {
    return {}
  }
  const rawQuality = meta.summary_quality || meta.summaryQuality
  const summaryQuality = rawQuality && typeof rawQuality === 'object'
    ? rawQuality as Record<string, unknown>
    : {
      ok: true,
      status: 'grounded',
      source: source || 'abstract',
      provider,
      export_ready: true,
    }
  return {
    summaryLine: line,
    summarySource: source || 'abstract',
    summaryProvider: provider,
    summaryQuality,
  }
}

export function mergeShelfItemWithLive(item: CiteShelfItem, live: CiteShelfItem): CiteShelfItem {
  const incomingMetadataReady = metadataQualityRecordReady(live.metadataQuality)
  const currentMetadataReady = metadataQualityRecordReady(item.metadataQuality)
  const metadataQuality = preferMetadataQuality(item.metadataQuality, live.metadataQuality)
  const preferIncomingMetadata = incomingMetadataReady && !currentMetadataReady
  const mergedLike = {
    ...item,
    traceConvId: preferExistingText(item.traceConvId, live.traceConvId),
    traceAssistantMsgId: preferPositiveNumber(item.traceAssistantMsgId, live.traceAssistantMsgId),
    traceAssistantOrder: preferPositiveNumber(item.traceAssistantOrder, live.traceAssistantOrder),
    traceUserMsgId: preferPositiveNumber(item.traceUserMsgId, live.traceUserMsgId),
    sourceName: preferExistingText(item.sourceName, live.sourceName),
    sourcePath: preferExistingText(item.sourcePath, live.sourcePath),
    raw: preferExistingText(item.raw, live.raw),
    citeFmt: preferExistingText(item.citeFmt, live.citeFmt),
    title: preferRicherField('title', item.title, live.title),
    authors: preferRicherField('authors', item.authors, live.authors),
    venue: preferRicherField('venue', item.venue, live.venue),
    year: preferRicherField('year', item.year, live.year),
    volume: preferExistingText(item.volume, live.volume),
    issue: preferExistingText(item.issue, live.issue),
    pages: preferExistingText(item.pages, live.pages),
    doi: preferExistingText(item.doi, live.doi),
    doiUrl: preferExistingText(item.doiUrl, live.doiUrl),
    citationSource: preferExistingText(item.citationSource, live.citationSource),
    venueKind: preferExistingText(item.venueKind, live.venueKind),
    venueVerifiedBy: preferExistingText(item.venueVerifiedBy, live.venueVerifiedBy),
    openalexVenue: preferExistingText(item.openalexVenue, live.openalexVenue),
    journalIf: preferExistingText(item.journalIf, live.journalIf),
    journalQuartile: preferExistingText(item.journalQuartile, live.journalQuartile),
    journalIfSource: preferExistingText(item.journalIfSource, live.journalIfSource),
    conferenceTier: preferExistingText(item.conferenceTier, live.conferenceTier),
    conferenceRankSource: preferExistingText(item.conferenceRankSource, live.conferenceRankSource),
    conferenceCcf: preferExistingText(item.conferenceCcf, live.conferenceCcf),
    conferenceCcfSource: preferExistingText(item.conferenceCcfSource, live.conferenceCcfSource),
    conferenceName: preferExistingText(item.conferenceName, live.conferenceName),
    conferenceAcronym: preferExistingText(item.conferenceAcronym, live.conferenceAcronym),
    summaryLine: preferRicherField('title', item.summaryLine, live.summaryLine),
    summarySource: preferExistingText(item.summarySource, live.summarySource),
    summaryProvider: preferExistingText(item.summaryProvider, live.summaryProvider),
    summaryQuality: item.summaryQuality || live.summaryQuality,
    shelfItemKind: preferExistingText(item.shelfItemKind, live.shelfItemKind),
    shelfOrigin: preferExistingText(item.shelfOrigin, live.shelfOrigin),
    shelfExcerpt: preferRicherField('title', item.shelfExcerpt, live.shelfExcerpt),
    shelfExcerptLabel: preferExistingText(item.shelfExcerptLabel, live.shelfExcerptLabel),
    metadataQuality,
    metadataRepairStatus: preferIncomingMetadata
      ? preferExistingText(live.metadataRepairStatus, item.metadataRepairStatus)
      : preferExistingText(item.metadataRepairStatus, live.metadataRepairStatus),
    metadataRepairSources: preferStringArray(item.metadataRepairSources, live.metadataRepairSources, preferIncomingMetadata),
    metadataChangedFields: preferStringArray(item.metadataChangedFields, live.metadataChangedFields, preferIncomingMetadata),
    externalMetadataStatus: preferExistingText(item.externalMetadataStatus, live.externalMetadataStatus),
    externalMetadataReason: preferExistingText(item.externalMetadataReason, live.externalMetadataReason),
    externalMatchMethod: preferExistingText(item.externalMatchMethod, live.externalMatchMethod),
    externalMatchScore: preferPositiveNumber(item.externalMatchScore, live.externalMatchScore),
    externalTitleSimilarity: preferPositiveNumber(item.externalTitleSimilarity, live.externalTitleSimilarity),
    externalTitle: preferRicherField('title', item.externalTitle, live.externalTitle),
    externalAuthors: preferRicherField('authors', item.externalAuthors, live.externalAuthors),
    externalVenue: preferRicherField('venue', item.externalVenue, live.externalVenue),
    externalYear: preferRicherField('year', item.externalYear, live.externalYear),
    externalDoi: preferExistingText(item.externalDoi, live.externalDoi),
    externalDoiUrl: preferExistingText(item.externalDoiUrl, live.externalDoiUrl),
    answerClaim: preferRicherField('title', item.answerClaim, live.answerClaim),
    headingPath: preferExistingText(item.headingPath, live.headingPath),
    evidenceQuote: preferRicherField('title', item.evidenceQuote, live.evidenceQuote),
    evidenceSource: preferExistingText(item.evidenceSource, live.evidenceSource),
    citationContext: preferRicherField('title', item.citationContext, live.citationContext),
    citationContextSource: preferExistingText(item.citationContextSource, live.citationContextSource),
    upstreamWorkRole: preferExistingText(item.upstreamWorkRole, live.upstreamWorkRole),
    userQuestionRelation: preferExistingText(item.userQuestionRelation, live.userQuestionRelation),
    locationLabel: preferExistingText(item.locationLabel, live.locationLabel),
    supportRelation: preferExistingText(item.supportRelation, live.supportRelation),
    whyLine: preferExistingText(item.whyLine, live.whyLine),
    blockId: preferExistingText(item.blockId, live.blockId),
    anchorId: preferExistingText(item.anchorId, live.anchorId),
    anchorKind: preferExistingText(item.anchorKind, live.anchorKind),
    pageStart: preferPositiveNumber(item.pageStart, live.pageStart),
    pageEnd: preferPositiveNumber(item.pageEnd, live.pageEnd),
    score: preferPositiveNumber(item.score, live.score),
    citationCount: preferPositiveNumber(item.citationCount, live.citationCount),
    num: preferPositiveNumber(item.num, live.num),
    bibliometricsChecked: Boolean(item.bibliometricsChecked || live.bibliometricsChecked),
  }

  const normalized = normalizeCiteDetail(mergedLike) || item
  const autoMain = toShelfItem(normalized).main
  return {
    ...item,
    ...normalized,
    key: item.key,
    main: preferRicherField('main', item.main, preferExistingText(live.main, autoMain)),
    tags: normalizeShelfTags(item.tags),
    note: normalizeShelfNote(item.note),
  }
}

export function snapshotDiffCounts(currentItems: CiteShelfItem[], baselineItems: CiteShelfItem[]): { added: number; removed: number } {
  const current = new Set(currentItems.map((item) => shelfPaperIdentity(item)))
  const baseline = new Set(baselineItems.map((item) => shelfPaperIdentity(item)))
  let added = 0
  let removed = 0
  for (const id of current) {
    if (!baseline.has(id)) added += 1
  }
  for (const id of baseline) {
    if (!current.has(id)) removed += 1
  }
  return { added, removed }
}

export function shelfRepairPayloads(item: CiteShelfItem): Array<Record<string, unknown>> {
  const basePayload = item as unknown as Record<string, unknown>
  return [
    basePayload,
    {
      ...basePayload,
      raw: '',
      cite_fmt: '',
      citeFmt: '',
    },
  ]
}

export function shelfRepairMetaFromEntry(entry: ShelfMetadataRepairItem): Record<string, unknown> {
  return {
    ...(entry.meta || {}),
    metadata_quality: entry.after || (entry.meta || {}).metadata_quality,
    metadata_repair_status: entry.repair_status,
    metadata_changed_fields: entry.changed_fields || [],
    metadata_repair_sources: entry.repair_sources || [],
  }
}

export function shouldRequestCitationCardPolish(detail: CiteDetail): boolean {
  const polishStatus = String(detail.citationCardPolishStatus || '').trim().toLowerCase()
  if (['full', 'failed', 'disabled', 'empty'].includes(polishStatus)) return false
  return Boolean(
    detail.cardEvidence
    || detail.evidenceQuote
    || detail.citationContext
    || detail.cardTakeaway
    || detail.raw
    || detail.citeFmt,
  )
}
