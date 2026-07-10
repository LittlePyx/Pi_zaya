import type { ConversionQualitySummary, LibrarySourceQualityItem } from '../../api/library'
import type { ShelfMetadataQuality } from '../../api/references'
import { basenameFromSourcePath, normalizeSourcePathForMatch } from '../../utils/sourcePath'
import { readerSourcePathsMatch } from './reader/readerLocateGuard'
import type { ReaderLocateResult } from './reader/readerTypes'
import {
  citationCardView,
  citationDisplay,
  cleanCitationDisplayText,
  isLikelyWeakCitationTitle,
  looksLowValueShelfSummary,
  normalizeDoiLike,
  summarySourceLabel,
  type CiteShelfItem,
} from './citationState'

export const TAG_PRESETS = ['baseline', 'idea', 'related-work'] as const

export type GroupMode = 'none' | 'tag' | 'source' | 'kind'
export type ScopeFilter = 'all' | 'conversation' | 'paper'
export type SourceQualityByPath = Record<string, LibrarySourceQualityItem>
export type ShelfExportKind = 'bib' | 'csv' | 'md' | 'ris'
export type ShelfExportScope = 'selected' | 'visible' | 'all'
export type ShelfExportOptions = { skipPreflight?: boolean; onlyMetadataReady?: boolean; autoRepair?: boolean }
export type ShelfExportRequest = { kind: ShelfExportKind; scope: ShelfExportScope }
export type SourceOpenQualityTone = 'ready' | 'partial' | 'review' | 'missing'
export type ShelfCardSurface = 'reference' | 'citation' | 'figure' | 'table' | 'equation' | 'selection' | 'excerpt'

export interface SourceOpenQualityView {
  status: 'ready' | 'partial' | 'repairing' | 'missing' | 'verified' | 'degraded' | 'failed'
  precision: 'exact_anchor' | 'block' | 'phrase' | 'fuzzy' | 'page' | 'section' | 'source_only' | 'needs_repair' | 'missing' | 'failed'
  label: string
  reason: string
  tone: SourceOpenQualityTone
  canOpen: boolean
  strictLocate: boolean
  repairable: boolean
}

export interface ShelfCardPresentation {
  surface: ShelfCardSurface
  title: string
  sourceLabel: string
  excerpt: string
  excerptLabel: string
  showAuthors: boolean
  showArticleSummary: boolean
  showExcerptInDetails: boolean
}

export const GROUP_MODE_LABEL = (S: Record<string, string>): Record<GroupMode, string> => ({
  none: S.shelf_no_group,
  tag: S.shelf_by_tag,
  source: S.shelf_by_source,
  kind: S.shelf_by_type,
})

export const SCOPE_FILTER_LABEL = (S: Record<string, string>): Record<ScopeFilter, string> => ({
  all: S.shelf_scope_all_project,
  conversation: S.shelf_scope_current_conversation,
  paper: S.shelf_scope_current_paper,
})

export const normalizeSourceIdentity = (value: string | null | undefined): string =>
  normalizeSourcePathForMatch(value)

export const basenameFromPath = (value: string): string => {
  return basenameFromSourcePath(value)
}

export const normalizeTitle = (value: string): string =>
  String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9\u4e00-\u9fff]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()

export const citeVenueYearParts = (
  item: CiteShelfItem,
  display = citationDisplay(item),
): string[] => {
  const venue = cleanCitationDisplayText(display.venue || item.venue || item.openalexVenue || item.externalVenue || '')
  const rawYear = cleanCitationDisplayText(item.year || item.externalYear || item.libraryMatchYear || '')
  const year = rawYear.match(/\b(?:19|20)\d{2}\b/)?.[0] || rawYear
  return [venue, year].filter(Boolean)
}

export const metricLabel = (prefix: string, value: string): string => {
  const text = cleanCitationDisplayText(value)
  if (!text) return ''
  return text.toLowerCase().startsWith(prefix.toLowerCase()) ? text : `${prefix} ${text}`
}

export const citeImpactMetrics = (item: CiteShelfItem, S: Record<string, string>): string[] => [
  metricLabel('IF', item.journalIf),
  metricLabel('JCR', item.journalQuartile),
  item.citationCount > 0
    ? (S.shelf_citation_count || 'Cited by {n}').replace('{n}', String(item.citationCount))
    : '',
].filter(Boolean)

export const uniqueCitationMetrics = (...groups: string[][]): string[] => {
  const seen = new Set<string>()
  const out: string[] = []
  for (const group of groups) {
    for (const value of group) {
      const text = cleanCitationDisplayText(value)
      const key = text.replace(/\s+/g, '').toLowerCase()
      if (!text || seen.has(key)) continue
      seen.add(key)
      out.push(text)
    }
  }
  return out
}

export const hasCompleteCitationIdentity = (
  item: CiteShelfItem,
  display = citationDisplay(item),
): boolean => {
  const title = String(item.title || display.main || item.main || '').trim()
  const hasTitle = Boolean(title) && !isLikelyWeakCitationTitle(title)
  const hasDoi = Boolean(normalizeDoiLike(item.doi || item.doiUrl))
  const hasAuthors = Boolean(String(item.authors || '').trim())
  const hasVenue = Boolean(String(item.venue || '').trim())
  return hasTitle && hasDoi && hasAuthors && hasVenue
}

export const externalMetadataNeedsVisibleReview = (
  item: CiteShelfItem,
  display = citationDisplay(item),
): boolean => {
  const status = String(item.externalMetadataStatus || '').trim().toLowerCase()
  if (!['candidate', 'conflict'].includes(status)) return false
  const itemDoi = normalizeDoiLike(item.doi || item.doiUrl)
  const externalDoi = normalizeDoiLike(item.externalDoi || item.externalDoiUrl)
  if (itemDoi && externalDoi && itemDoi !== externalDoi) return true
  if (status === 'candidate') return false
  return !hasCompleteCitationIdentity(item, display)
}

export const impactScore = (item: CiteShelfItem): number => {
  const ifValue = Number.parseFloat(String(item.journalIf || '').replace(/[^\d.]/g, ''))
  const ifScore = Number.isFinite(ifValue) ? ifValue : 0
  const quartile = String(item.journalQuartile || '').toUpperCase().trim()
  const quartileScore = quartile === 'Q1' ? 4 : quartile === 'Q2' ? 3 : quartile === 'Q3' ? 2 : quartile === 'Q4' ? 1 : 0
  const core = String(item.conferenceTier || '').toUpperCase().trim()
  const coreScore = core === 'A*' ? 4 : core === 'A' ? 3 : core === 'B' ? 2 : core === 'C' ? 1 : 0
  const ccf = String(item.conferenceCcf || '').toUpperCase().trim()
  const ccfScore = ccf === 'A' ? 3 : ccf === 'B' ? 2 : ccf === 'C' ? 1 : 0
  return ifScore * 10 + quartileScore + coreScore + ccfScore
}

export const sourceQualityStatus = (quality?: ConversionQualitySummary | null): string =>
  String(quality?.status || '').trim().toLowerCase()

export const sourceQualityNeedsReview = (quality?: ConversionQualitySummary | null): boolean =>
  Boolean(quality?.has_review_issue) || ['warning', 'error'].includes(sourceQualityStatus(quality))

export const sourceQualityHasReaderLocateRepair = (quality?: ConversionQualitySummary | null): boolean => {
  const attempt = quality?.conversion_report?.latest_repair_attempt
  const event = String(attempt?.event || '').trim().toLowerCase()
  const source = String(attempt?.source || '').trim().toLowerCase()
  const extra = attempt?.extra || {}
  return (
    event === 'reader_locate_reindex_required'
    || source === 'reader_locate_quality'
    || Number(extra.reader_locate_problem_count || 0) > 0
  )
}

export const sourceQualityIssueReason = (quality?: ConversionQualitySummary | null): string => {
  if (!quality) return ''
  const issue = (quality.issues || []).find((item) => item?.label || item?.code)
  return String(issue?.label || issue?.code || quality.summary || quality.label || quality.status || '').trim()
}

export const sourceQualityForItem = (
  item: CiteShelfItem,
  sourceQualityByPath: SourceQualityByPath,
): ConversionQualitySummary | null => {
  const sourcePath = String(item.sourcePath || '').trim()
  if (!sourcePath) return null
  const exact = sourceQualityByPath[sourcePath]?.conversion_quality || null
  if (exact) return exact
  const sourceKey = normalizeSourceIdentity(sourcePath)
  if (!sourceKey) return null
  for (const [path, quality] of Object.entries(sourceQualityByPath)) {
    if (normalizeSourceIdentity(path) === sourceKey) return quality?.conversion_quality || null
  }
  return null
}

export const sourceOpenQualityView = (
  item: CiteShelfItem,
  sourceQuality: ConversionQualitySummary | null | undefined,
  S: Record<string, string>,
  locateResult?: ReaderLocateResult | null,
): SourceOpenQualityView => {
  const sourcePath = String(item.sourcePath || '').trim()
  if (!sourcePath) {
    return {
      status: 'missing',
      precision: 'missing',
      label: S.shelf_source_open_missing,
      reason: S.shelf_source_open_missing,
      tone: 'missing',
      canOpen: false,
      strictLocate: false,
      repairable: false,
    }
  }
  if (locateResult && readerSourcePathsMatch(locateResult.sourcePath, sourcePath)) {
    const resultStatus = String(locateResult.status || '').trim().toLowerCase()
    const resultPrecision = String(locateResult.precision || '').trim().toLowerCase()
    const reason = String(locateResult.reason || locateResult.hint || '').trim()
    const sourceRepairClosed = sourceQualityHasReaderLocateRepair(sourceQuality) && !sourceQualityNeedsReview(sourceQuality)
    if (sourceRepairClosed && ['failed', 'fuzzy', 'section', 'source_only'].includes(resultStatus)) {
      const repairedPrecision: SourceOpenQualityView['precision'] = resultPrecision === 'fuzzy'
        ? 'fuzzy'
        : resultPrecision === 'section'
          ? 'section'
          : 'source_only'
      return {
        status: 'partial',
        precision: repairedPrecision,
        label: S.shelf_source_open_repaired_reopen,
        reason: reason
          ? `${S.shelf_source_open_repaired_reopen}: ${reason}`
          : S.shelf_source_open_repaired_reopen,
        tone: 'partial',
        canOpen: true,
        strictLocate: Boolean(locateResult.strictLocate),
        repairable: false,
      }
    }
    if (resultStatus === 'failed') {
      return {
        status: 'failed',
        precision: 'failed',
        label: S.shelf_source_open_failed,
        reason: reason ? `${S.shelf_source_open_failed}: ${reason}` : S.shelf_source_open_failed,
        tone: 'review',
        canOpen: true,
        strictLocate: Boolean(locateResult.strictLocate),
        repairable: true,
      }
    }
    if (resultStatus === 'exact' || resultStatus === 'block') {
      return {
        status: 'verified',
        precision: resultPrecision === 'phrase'
          ? 'phrase'
          : resultPrecision === 'block'
            ? 'block'
            : 'exact_anchor',
        label: resultStatus === 'block' ? S.shelf_source_open_verified_block : S.shelf_source_open_verified,
        reason: reason ? `${S.shelf_source_open_verified}: ${reason}` : S.shelf_source_open_verified,
        tone: 'ready',
        canOpen: true,
        strictLocate: Boolean(locateResult.strictLocate),
        repairable: false,
      }
    }
    if (resultStatus === 'fuzzy') {
      return {
        status: 'degraded',
        precision: 'fuzzy',
        label: S.shelf_source_open_fuzzy,
        reason: reason ? `${S.shelf_source_open_fuzzy}: ${reason}` : S.shelf_source_open_fuzzy,
        tone: locateResult.repairable ? 'review' : 'partial',
        canOpen: true,
        strictLocate: Boolean(locateResult.strictLocate),
        repairable: Boolean(locateResult.repairable),
      }
    }
    if (resultStatus === 'section') {
      return {
        status: 'degraded',
        precision: 'section',
        label: S.shelf_source_open_section_verified,
        reason: reason ? `${S.shelf_source_open_section_verified}: ${reason}` : S.shelf_source_open_section_verified,
        tone: locateResult.repairable ? 'review' : 'partial',
        canOpen: true,
        strictLocate: Boolean(locateResult.strictLocate),
        repairable: Boolean(locateResult.repairable),
      }
    }
    if (resultStatus === 'source_only') {
      return {
        status: 'degraded',
        precision: 'source_only',
        label: S.shelf_source_open_file_verified,
        reason: reason ? `${S.shelf_source_open_file_verified}: ${reason}` : S.shelf_source_open_file_verified,
        tone: locateResult.repairable ? 'review' : 'partial',
        canOpen: true,
        strictLocate: Boolean(locateResult.strictLocate),
        repairable: Boolean(locateResult.repairable),
      }
    }
  }
  if (sourceQualityNeedsReview(sourceQuality)) {
    const issue = sourceQualityIssueReason(sourceQuality)
    return {
      status: 'repairing',
      precision: 'needs_repair',
      label: S.shelf_source_open_repair,
      reason: issue ? `${S.shelf_source_open_repair}: ${issue}` : S.shelf_source_open_repair,
      tone: 'review',
      canOpen: true,
      strictLocate: false,
      repairable: true,
    }
  }

  const blockId = String(item.blockId || '').trim()
  const anchorId = String(item.anchorId || '').trim()
  if (blockId || anchorId) {
    const anchor = anchorId || blockId
    return {
      status: 'ready',
      precision: 'exact_anchor',
      label: S.shelf_source_open_exact,
      reason: anchor ? `${S.shelf_source_open_exact}: ${anchor}` : S.shelf_source_open_exact,
      tone: 'ready',
      canOpen: true,
      strictLocate: true,
      repairable: false,
    }
  }

  const pageStart = Number(item.pageStart || 0)
  const pageEnd = Number(item.pageEnd || 0)
  if ((Number.isFinite(pageStart) && pageStart > 0) || (Number.isFinite(pageEnd) && pageEnd > 0)) {
    const pageLabel = pageStart > 0 && pageEnd > 0 && pageEnd !== pageStart
      ? `${pageStart}-${pageEnd}`
      : String(pageStart || pageEnd)
    return {
      status: 'partial',
      precision: 'page',
      label: S.shelf_source_open_page,
      reason: pageLabel ? `${S.shelf_source_open_page}: p.${pageLabel}` : S.shelf_source_open_page,
      tone: 'partial',
      canOpen: true,
      strictLocate: false,
      repairable: false,
    }
  }

  const section = String(item.headingPath || item.locationLabel || item.cardLocator || '').trim()
  if (section) {
    return {
      status: 'partial',
      precision: 'section',
      label: S.shelf_source_open_section,
      reason: `${S.shelf_source_open_section}: ${section}`,
      tone: 'partial',
      canOpen: true,
      strictLocate: false,
      repairable: false,
    }
  }

  return {
    status: 'partial',
    precision: 'source_only',
    label: S.shelf_source_open_file,
    reason: S.shelf_source_open_file,
    tone: 'partial',
    canOpen: true,
    strictLocate: false,
    repairable: false,
  }
}

export const sourceListKey = (sources: Array<{ source_path: string; source_name?: string }>): string =>
  sources.map((item) => `${item.source_path}\t${item.source_name || ''}`).join('\n')

export const metadataQuality = (item: CiteShelfItem): ShelfMetadataQuality | null => {
  const raw = item.metadataQuality
  if (!raw || typeof raw !== 'object') return null
  return raw as unknown as ShelfMetadataQuality
}

export const metadataExportAcceptance = (item: CiteShelfItem): Record<string, unknown> | null => {
  const raw = item.metadataExportAcceptance
  if (!raw || typeof raw !== 'object') return null
  return raw
}

const metadataExportAcceptanceReady = (item: CiteShelfItem): boolean | null => {
  const acceptance = metadataExportAcceptance(item)
  if (!acceptance) return null
  if (!('export_ready' in acceptance) && !('exportReady' in acceptance)) return null
  return acceptance.export_ready === true || acceptance.exportReady === true
}

export const metadataQualityReady = (item: CiteShelfItem): boolean => {
  const acceptanceReady = metadataExportAcceptanceReady(item)
  if (acceptanceReady !== null) return acceptanceReady
  const quality = metadataQuality(item)
  if (!quality) return false
  const status = String(quality.status || '').trim().toLowerCase()
  return quality.ok === true || status === 'ready'
}

export const metadataQualityNeedsRepair = (item: CiteShelfItem): boolean => {
  const quality = metadataQuality(item)
  if (metadataQualityReady(item)) return false
  const acceptance = metadataExportAcceptance(item)
  if (acceptance) {
    const missing = Array.isArray(acceptance.missing_fields) ? acceptance.missing_fields : acceptance.missingFields
    const issues = Array.isArray(acceptance.issue_codes) ? acceptance.issue_codes : acceptance.issueCodes
    if ((Array.isArray(missing) && missing.length > 0) || (Array.isArray(issues) && issues.length > 0)) return true
  }
  if (!quality) return false
  return Boolean(quality.repairable || quality.retryable || (quality.issues || []).length > 0)
}

export const summaryQuality = (item: CiteShelfItem): Record<string, unknown> | null => {
  const raw = item.summaryQuality
  if (!raw || typeof raw !== 'object') return null
  return raw
}

export const trustedSummarySource = (source: string): boolean => [
  'abstract',
  'fulltext',
  'navigation',
  'exact_anchor',
  'section_intent_rescue',
  'doc_list_seed',
  'doc_list_prompt_aligned',
].includes(String(source || '').trim().toLowerCase())

export const summaryQualityView = (
  item: CiteShelfItem,
  S: Record<string, string>,
): { ok: boolean; status: string; score: number; label: string; tone: 'ready' | 'fallback' | 'review' } => {
  const contract = summaryQuality(item)
  const source = String(contract?.source || item.summarySource || '').trim().toLowerCase()
  const status = String(contract?.status || '').trim().toLowerCase()
  const scoreRaw = Number(contract?.score || 0)
  const score = Number.isFinite(scoreRaw) && scoreRaw > 0
    ? Math.round(scoreRaw)
    : (trustedSummarySource(source) ? 92 : source === 'metadata' ? 68 : 78)
  const ok = contract?.ok === true || status === 'grounded' || trustedSummarySource(source)
  const fallback = status === 'fallback' || source === 'metadata' || (!ok && Boolean(item.summaryLine))
  const label = ok
    ? S.shelf_summary_quality_grounded
    : fallback
      ? S.shelf_summary_quality_fallback
      : S.shelf_summary_quality_review
  return {
    ok,
    status: status || (ok ? 'grounded' : fallback ? 'fallback' : 'review'),
    score,
    label: label.replace('{score}', String(score)),
    tone: ok ? 'ready' : fallback ? 'fallback' : 'review',
  }
}

export type ShelfSummaryDisplay = {
  line: string
  sourceLabel: string
  quality: ReturnType<typeof summaryQualityView>
  kind: 'article' | 'evidence' | 'empty'
  headingLabel: string
  showQuality: boolean
}

export const shelfSummarySourceLabels = (S: Record<string, string>) => ({
  fulltext: S.shelf_summary_source_fulltext,
  crossref: S.shelf_summary_source_crossref,
  openalex: S.shelf_summary_source_openalex,
  semanticScholar: S.shelf_summary_source_semantic_scholar,
  doiLandingPage: S.shelf_summary_source_doi_landing_page,
  abstract: S.shelf_summary_source_abstract,
  citationContext: S.shelf_summary_source_citation_context,
  citationCard: S.shelf_summary_source_citation_card,
  metadata: S.shelf_summary_source_metadata,
})

export const trustedArticleSummarySource = (source: string): boolean => [
  'abstract',
  'fulltext',
  'navigation',
  'exact_anchor',
  'section_intent_rescue',
  'doc_list_seed',
  'doc_list_prompt_aligned',
].includes(String(source || '').trim().toLowerCase())

export const contextOnlySummarySource = (source: string): boolean => [
  'citation_context',
  'citation_card',
  'citation_card_view',
  'metadata',
  'reference_primary_evidence',
  'references_panel_hit',
].includes(String(source || '').trim().toLowerCase())

export const compactShelfSummaryCandidate = (value: string, limit = 520): string => {
  const text = cleanCitationDisplayText(value)
    .replace(/\s+/g, ' ')
    .trim()
  if (!text) return ''
  if (text.length <= limit) return text
  return `${text.slice(0, Math.max(0, limit - 1)).trimEnd()}...`
}

export const looksMetadataOnlyShelfSummary = (value: string): boolean => {
  const text = compactShelfSummaryCandidate(value, 520)
  if (!text) return false
  return /仅检索到|暂无可用摘要|缺少可用摘要|建议.*DOI|metadata only|no abstract/i.test(text)
}

export const shelfSummaryDisplay = (
  item: CiteShelfItem,
  cardView: ReturnType<typeof citationCardView>,
  S: Record<string, string>,
): ShelfSummaryDisplay => {
  const quality = summaryQualityView(item, S)
  const sourceLabels = shelfSummarySourceLabels(S)
  const qualityContract = summaryQuality(item)
  const source = String(item.summarySource || qualityContract?.source || '').trim().toLowerCase()
  const existing = compactShelfSummaryCandidate(item.summaryLine)
  if (
    existing
    && !looksLowValueShelfSummary(existing)
    && !looksMetadataOnlyShelfSummary(existing)
    && !contextOnlySummarySource(source)
    && (
      trustedArticleSummarySource(source)
      || (!item.isInpaper && quality.ok)
    )
  ) {
    return {
      line: existing,
      sourceLabel: summarySourceLabel(source, item.summaryProvider, sourceLabels),
      quality,
      kind: 'article',
      headingLabel: S.shelf_summary_head,
      showQuality: true,
    }
  }

  const cardSummary = item.cardView
    ? compactShelfSummaryCandidate(cardView.summary)
    : ''
  if (cardSummary && !looksLowValueShelfSummary(cardSummary) && !looksMetadataOnlyShelfSummary(cardSummary)) {
    return {
      line: cardSummary,
      sourceLabel: sourceLabels.citationCard,
      quality,
      kind: 'evidence',
      headingLabel: S.shelf_evidence_note_head || 'Evidence note',
      showQuality: false,
    }
  }

  return {
    line: '',
    sourceLabel: '',
    quality,
    kind: 'empty',
    headingLabel: S.shelf_summary_head,
    showQuality: false,
  }
}

export const metadataIssueChip = (code: string, S: Record<string, string>): string => {
  const key = String(code || '').trim().toLowerCase()
  if (key === 'missing_doi') return S.shelf_issue_match_doi
  if (key === 'doi_not_promoted') return S.shelf_issue_write_doi
  if (key === 'missing_authors') return S.shelf_issue_missing_authors
  if (key === 'missing_venue') return S.shelf_issue_missing_venue
  if (key === 'missing_year') return S.shelf_issue_missing_year
  if (key === 'weak_or_missing_title') return S.shelf_issue_fix_title
  if (key === 'missing_source') return S.shelf_issue_missing_source
  if (key.startsWith('external_metadata_')) return S.shelf_issue_external_metadata
  return key ? key.replace(/_/g, ' ') : S.shelf_issue_auto_complete
}
