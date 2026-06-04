import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Button, Input, Select, message } from 'antd'
import { CloseOutlined, DeleteOutlined, FileSearchOutlined, SaveOutlined } from '@ant-design/icons'
import type { CiteShelfItem } from './citationState'
import type { ReaderLocateResult } from './reader/readerTypes'
import type { ConversionQualitySummary, LibrarySourceQualityItem } from '../../api/library'
import type { ShelfMetadataQuality, ShelfMetadataRepairImpact, ShelfMetadataRepairItem } from '../../api/references'
import { libraryApi } from '../../api/library'
import {
  citationCardView,
  citationDisplay,
  citationFormats,
  citeMetricSummary,
  cleanCitationDisplayText,
  isLikelyWeakCitationTitle,
  looksLowValueShelfSummary,
  normalizeShelfItemKind,
  normalizeShelfTags,
  shelfItemKindLabel,
  shelfItemHasConflictingVenueSignals,
  shelfItemNeedsMetadataRepair,
  shelfOriginLabel,
  strictRepairMerge,
  summarySourceLabel,
} from './citationState'
import { useT } from '../../i18n'
import { referencesApi } from '../../api/references'

interface Props {
  open: boolean
  visible?: boolean
  presentation?: 'floating' | 'dock'
  items: CiteShelfItem[]
  readerLocateResults?: Record<string, ReaderLocateResult>
  sourceQualityRefreshToken?: number
  snapshots: Array<{ id: string; name: string; createdAt: number }>
  selectedSnapshotId: string
  snapshotDiff: string
  focusedKey: string
  summaryLoadingKey: string
  repairLoadingKey: string
  repairImpact: ShelfMetadataRepairImpact | null
  repairingKeys?: string[]
  onToggle: () => void
  onClear: () => void
  onSelect: (item: CiteShelfItem) => void
  onOpenSource?: (item: CiteShelfItem) => void
  onRemove: (key: string) => void
  onUpdateTags: (key: string, tags: string[]) => void
  onUpdateNote: (key: string, note: string) => void
  onRepair: (item: CiteShelfItem, options?: { silent?: boolean }) => void
  onApplyRepairCandidates?: (updates: Array<{ key: string; metas: Array<Record<string, unknown>> }>) => boolean
  onSelectSnapshot: (id: string) => void
  onSaveSnapshot: () => void
  onLoadSnapshot: () => void
  onDeleteSnapshot: () => void
}

const TAG_PRESETS = ['baseline', 'idea', 'related-work'] as const

type GroupMode = 'none' | 'tag' | 'source' | 'kind'
type SourceQualityByPath = Record<string, LibrarySourceQualityItem>
type ShelfExportKind = 'bib' | 'csv' | 'ris'
type ShelfExportOptions = { skipPreflight?: boolean; onlyMetadataReady?: boolean; autoRepair?: boolean }
type SourceOpenQualityTone = 'ready' | 'partial' | 'review' | 'missing'

interface SourceOpenQualityView {
  status: 'ready' | 'partial' | 'repairing' | 'missing' | 'verified' | 'degraded' | 'failed'
  precision: 'exact_anchor' | 'block' | 'phrase' | 'fuzzy' | 'page' | 'section' | 'source_only' | 'needs_repair' | 'missing' | 'failed'
  label: string
  reason: string
  tone: SourceOpenQualityTone
  canOpen: boolean
  strictLocate: boolean
  repairable: boolean
}

const GROUP_MODE_LABEL = (S: Record<string, string>): Record<GroupMode, string> => ({
  none: S.shelf_no_group,
  tag: S.shelf_by_tag,
  source: S.shelf_by_source,
  kind: S.shelf_by_type,
})

const normalizeDoiLike = (value: string): string =>
  String(value || '')
    .trim()
    .toLowerCase()
    .replace(/^https?:\/\/(?:dx\.)?doi\.org\//i, '')
    .replace(/^[\s"'`([{<]+|[\s"'`)\]}>.,;:]+$/g, '')
    .trim()

const normalizeTitle = (value: string): string =>
  String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9\u4e00-\u9fff]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()

const paperIdentity = (item: CiteShelfItem): string => {
  const doiKey = normalizeDoiLike(item.doi || item.doiUrl)
  if (doiKey) return `doi:${doiKey}`
  const titleKey = normalizeTitle(item.title || item.main)
  const year = /^\d{4}$/.test(String(item.year || '').trim()) ? String(item.year).trim() : ''
  if (titleKey) return `title:${titleKey}|${year}`
  return `key:${item.key}`
}

const doiExportValue = (item: CiteShelfItem): string =>
  normalizeDoiLike(item.doi || item.doiUrl) || String(item.doi || item.doiUrl || '').trim()

const hasCompleteCitationIdentity = (
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

const externalMetadataNeedsVisibleReview = (
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

const impactScore = (item: CiteShelfItem): number => {
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

const sourceQualityStatus = (quality?: ConversionQualitySummary | null): string =>
  String(quality?.status || '').trim().toLowerCase()

const sourceQualityNeedsReview = (quality?: ConversionQualitySummary | null): boolean =>
  Boolean(quality?.has_review_issue) || ['warning', 'error'].includes(sourceQualityStatus(quality))

const sourceQualityHasReaderLocateRepair = (quality?: ConversionQualitySummary | null): boolean => {
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

const sourceQualityIssueReason = (quality?: ConversionQualitySummary | null): string => {
  if (!quality) return ''
  const issue = (quality.issues || []).find((item) => item?.label || item?.code)
  return String(issue?.label || issue?.code || quality.summary || quality.label || quality.status || '').trim()
}

const sourceQualityForItem = (
  item: CiteShelfItem,
  sourceQualityByPath: SourceQualityByPath,
): ConversionQualitySummary | null => {
  const sourcePath = String(item.sourcePath || '').trim()
  return sourcePath ? sourceQualityByPath[sourcePath]?.conversion_quality || null : null
}

const sourceOpenQualityView = (
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
  if (locateResult && String(locateResult.sourcePath || '').trim() === sourcePath) {
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

const sourceListKey = (sources: Array<{ source_path: string; source_name?: string }>): string =>
  sources.map((item) => `${item.source_path}\t${item.source_name || ''}`).join('\n')

const metadataQuality = (item: CiteShelfItem): ShelfMetadataQuality | null => {
  const raw = item.metadataQuality
  if (!raw || typeof raw !== 'object') return null
  return raw as unknown as ShelfMetadataQuality
}

const metadataQualityReady = (item: CiteShelfItem): boolean => {
  const quality = metadataQuality(item)
  if (!quality) return false
  const status = String(quality.status || '').trim().toLowerCase()
  return quality.ok === true || status === 'ready'
}

const metadataQualityNeedsRepair = (item: CiteShelfItem): boolean => {
  const quality = metadataQuality(item)
  if (!quality) return false
  if (metadataQualityReady(item)) return false
  return Boolean(quality.repairable || quality.retryable || (quality.issues || []).length > 0)
}

const summaryQuality = (item: CiteShelfItem): Record<string, unknown> | null => {
  const raw = item.summaryQuality
  if (!raw || typeof raw !== 'object') return null
  return raw
}

const trustedSummarySource = (source: string): boolean => [
  'abstract',
  'fulltext',
  'citation_context',
  'reference_primary_evidence',
  'navigation',
  'exact_anchor',
  'section_intent_rescue',
  'doc_list_seed',
  'doc_list_prompt_aligned',
].includes(String(source || '').trim().toLowerCase())

const summaryQualityView = (
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

type ShelfSummaryDisplay = {
  line: string
  sourceLabel: string
  quality: ReturnType<typeof summaryQualityView>
}

const shelfSummarySourceLabels = (S: Record<string, string>) => ({
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

const trustedArticleSummarySource = (source: string): boolean => [
  'abstract',
  'fulltext',
  'reference_primary_evidence',
  'navigation',
  'exact_anchor',
  'section_intent_rescue',
  'doc_list_seed',
  'doc_list_prompt_aligned',
].includes(String(source || '').trim().toLowerCase())

const compactShelfSummaryCandidate = (value: string, limit = 520): string => {
  const text = cleanCitationDisplayText(value)
    .replace(/\s+/g, ' ')
    .trim()
  if (!text) return ''
  if (text.length <= limit) return text
  return `${text.slice(0, Math.max(0, limit - 1)).trimEnd()}...`
}

const looksMetadataOnlyShelfSummary = (value: string): boolean => {
  const text = compactShelfSummaryCandidate(value, 520)
  if (!text) return false
  return /仅检索到|暂无可用摘要|缺少可用摘要|建议.*DOI|metadata only|no abstract/i.test(text)
}

const groundedDisplaySummaryQuality = (
  base: ReturnType<typeof summaryQualityView>,
  S: Record<string, string>,
): ReturnType<typeof summaryQualityView> => {
  const score = Math.max(Number(base.score || 0), 88)
  return {
    ok: true,
    status: 'grounded',
    score,
    label: S.shelf_summary_quality_grounded.replace('{score}', String(score)),
    tone: 'ready',
  }
}

const shelfSummaryDisplay = (
  item: CiteShelfItem,
  cardView: ReturnType<typeof citationCardView>,
  S: Record<string, string>,
): ShelfSummaryDisplay => {
  const quality = summaryQualityView(item, S)
  const sourceLabels = shelfSummarySourceLabels(S)
  const source = String(item.summarySource || '').trim().toLowerCase()
  const existing = compactShelfSummaryCandidate(item.summaryLine)
  if (
    existing
    && !item.isInpaper
    && source === 'metadata'
    && !looksLowValueShelfSummary(existing)
    && !looksMetadataOnlyShelfSummary(existing)
  ) {
    return {
      line: existing,
      sourceLabel: summarySourceLabel('fulltext', '', sourceLabels),
      quality: groundedDisplaySummaryQuality(quality, S),
    }
  }
  if (
    existing
    && !looksLowValueShelfSummary(existing)
    && (
      trustedArticleSummarySource(source)
      || (!item.isInpaper && quality.ok)
    )
  ) {
    return {
      line: existing,
      sourceLabel: summarySourceLabel(item.summarySource, item.summaryProvider, sourceLabels),
      quality,
    }
  }

  const viewSummary = compactShelfSummaryCandidate(cardView.summary)
  if (!item.isInpaper && viewSummary && !looksLowValueShelfSummary(viewSummary)) {
    return {
      line: viewSummary,
      sourceLabel: summarySourceLabel('citation_card_view', '', sourceLabels),
      quality,
    }
  }

  const directEvidence = compactShelfSummaryCandidate(item.evidenceQuote || item.cardEvidence, 360)
  if (!item.isInpaper && directEvidence && !looksLowValueShelfSummary(directEvidence)) {
    return {
      line: directEvidence,
      sourceLabel: summarySourceLabel('citation_card', '', sourceLabels),
      quality,
    }
  }

  return {
    line: '',
    sourceLabel: '',
    quality,
  }
}

const metadataIssueChip = (code: string, S: Record<string, string>): string => {
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

export function CiteShelf({
  open,
  visible,
  presentation = 'floating',
  items,
  readerLocateResults = {},
  sourceQualityRefreshToken = 0,
  snapshots,
  selectedSnapshotId,
  snapshotDiff,
  focusedKey,
  summaryLoadingKey,
  repairLoadingKey,
  repairImpact,
  repairingKeys = [],
  onToggle,
  onClear,
  onSelect,
  onOpenSource,
  onRemove,
  onUpdateTags,
  onUpdateNote,
  onRepair,
  onApplyRepairCandidates,
  onSelectSnapshot,
  onSaveSnapshot,
  onLoadSnapshot,
  onDeleteSnapshot,
}: Props) {
  const S = useT()
  const isDockPresentation = presentation === 'dock'
  const panelVisible = visible ?? open
  const [expandedSummaryKeys, setExpandedSummaryKeys] = useState<Record<string, boolean>>({})
  const [selectedKeys, setSelectedKeys] = useState<Record<string, boolean>>({})
  const [searchText, setSearchText] = useState('')
  const [sortKey, setSortKey] = useState<'recent' | 'cited' | 'year' | 'impact'>('recent')
  const [groupMode, setGroupMode] = useState<GroupMode>('none')
  const [tagFilter, setTagFilter] = useState<string>('all')
  const [advancedFiltersOpen, setAdvancedFiltersOpen] = useState(false)
  const [preflightExportKind, setPreflightExportKind] = useState<ShelfExportKind | ''>('')
  const [exportRepairingKind, setExportRepairingKind] = useState<ShelfExportKind | ''>('')
  const [batchTagInput, setBatchTagInput] = useState('')
  const [editingNoteKeys, setEditingNoteKeys] = useState<Record<string, boolean>>({})
  const [copyState, setCopyState] = useState<'idle' | 'gbt' | 'bibtex' | 'error'>('idle')
  const [sourceQualityByPath, setSourceQualityByPath] = useState<SourceQualityByPath>({})
  const [sourceRepairingKey, setSourceRepairingKey] = useState('')
  const copyStateTimerRef = useRef<number | null>(null)
  const sourceRepairStreamRef = useRef<AbortController | null>(null)
  const autoSourceRepairKeysRef = useRef<Record<string, boolean>>({})
  const repairLoadingKeySet = useMemo(
    () => new Set([repairLoadingKey, ...repairingKeys].map((key) => String(key || '').trim()).filter(Boolean)),
    [repairLoadingKey, repairingKeys],
  )

  const sourceQualitySources = useMemo(() => {
    const seen = new Set<string>()
    const out: Array<{ source_path: string; source_name: string }> = []
    for (const item of items) {
      const sourcePath = String(item.sourcePath || '').trim()
      if (!sourcePath || seen.has(sourcePath)) continue
      seen.add(sourcePath)
      out.push({
        source_path: sourcePath,
        source_name: String(item.sourceName || item.title || item.main || '').trim(),
      })
    }
    return out
  }, [items])

  const sourceQualityKey = useMemo(
    () => sourceQualitySources.map((item) => `${item.source_path}\t${item.source_name}`).join('\n'),
    [sourceQualitySources],
  )

  const splitSummary = (text: string): string[] => {
    const normalized = String(text || '').replace(/\s+/g, ' ').trim()
    if (!normalized) return []
    const sentences = normalized
      .split(/(?<=[\u3002\uff01\uff1f!?；;])\s*/)
      .map((item) => item.trim())
      .filter(Boolean)
    if (sentences.length >= 2) return sentences.slice(0, 4)

    if (normalized.length <= 100) return [normalized]
    const chunks: string[] = []
    let current = ''
    const parts = normalized
      .split(/(?<=[\uff0c,])\s*/)
      .map((item) => item.trim())
      .filter(Boolean)
    for (const part of parts) {
      if ((current + part).length > 56 && current) {
        chunks.push(current.trim())
        current = part
      } else {
        current += part
      }
    }
    if (current.trim()) chunks.push(current.trim())
    return (chunks.length > 0 ? chunks : [normalized]).slice(0, 4)
  }

  const sourceTraceLabel = (item: CiteShelfItem): { labels: string[]; debugTitle: string } => {
    const labels: string[] = []
    const answerOrder = Number(item.traceAssistantOrder || 0)
    if (Number.isFinite(answerOrder) && answerOrder > 0) {
      labels.push(S.shelf_answer.replace('{n}', String(answerOrder)))
    }
    const num = Number(item.num || 0)
    if (Number.isFinite(num) && num > 0) labels.push(S.shelf_ref_num.replace('{n}', String(num)))
    const anchor = String(item.anchor || '').trim()
    const debugTitle = anchor ? S.shelf_anchor.replace('{anchor}', anchor) : ''
    return { labels, debugTitle }
  }

  const libraryMatchView = (item: CiteShelfItem): { label: string; title: string; tone: 'ready' | 'missing' } | null => {
    const status = String(item.libraryMatchStatus || '').trim().toLowerCase()
    if (!status || status === 'unknown') return null
    const method = String(item.libraryMatchMethod || '').trim().toLowerCase()
    const methodLabel = method === 'doi'
      ? S.shelf_library_match_doi
      : method.includes('title')
        ? S.shelf_library_match_title
        : S.shelf_library_match_local
    if (status === 'in_library') {
      const matched = String(item.libraryMatchTitle || item.libraryMatchPath || item.libraryMatchDoi || '').trim()
      return {
        label: S.shelf_library_in_library,
        title: matched ? `${methodLabel}: ${matched}` : methodLabel,
        tone: 'ready',
      }
    }
    if (status === 'not_in_library') {
      return {
        label: S.shelf_library_not_in_library,
        title: S.shelf_library_not_in_library_tip,
        tone: 'missing',
      }
    }
    return null
  }

  const qualityHints = (
    item: CiteShelfItem,
    display: ReturnType<typeof citationDisplay>,
  ): { chips: string[]; tip: string; needsRepair: boolean } => {
    const chips: string[] = []
    const contract = metadataQuality(item)
    if (contract && metadataQualityReady(item)) {
      return { chips: [], tip: '', needsRepair: false }
    }
    if (contract && Array.isArray(contract.issues) && contract.issues.length > 0) {
      for (const issue of contract.issues) {
        const code = String(issue.code || '').trim()
        const chip = metadataIssueChip(code, S)
        if (chip && !chips.includes(chip)) chips.push(chip)
      }
      const needsRepair = metadataQualityNeedsRepair(item)
      const score = Number(contract.score || 0)
      const tip = needsRepair
        ? S.shelf_metadata_repair_tip_score.replace('{score}', String(Number.isFinite(score) ? Math.round(score) : 0))
        : S.shelf_metadata_recorded_tip
      return { chips: chips.slice(0, 3), tip, needsRepair }
    }
    const rawTitle = String(item.title || '').trim()
    const visibleTitle = String(display.main || rawTitle || item.main || '').trim()
    const hasWeakTitle = isLikelyWeakCitationTitle(visibleTitle)
    const hasWeakStoredTitle = isLikelyWeakCitationTitle(rawTitle)
    const hasDoi = Boolean(normalizeDoiLike(item.doi || item.doiUrl))
    const hasAuthors = Boolean(String(item.authors || '').trim())
    const hasVenue = Boolean(String(item.venue || '').trim())
    const hasMetaConflict = shelfItemHasConflictingVenueSignals(item)
    const externalNeedsReview = externalMetadataNeedsVisibleReview(item, display)
    const unresolved = !item.bibliometricsChecked
    const bibliographicEntry = Boolean(item.isInpaper || item.raw || item.citeFmt || hasDoi || item.externalDoi || item.externalDoiUrl)
    const needsRepair = shelfItemNeedsMetadataRepair(item, display)

    if (externalNeedsReview) chips.push(S.shelf_external_metadata_review)
    if (bibliographicEntry && !hasDoi) chips.push(S.shelf_missing_doi)
    if (bibliographicEntry && !hasAuthors) chips.push(S.shelf_missing_author)
    if (bibliographicEntry && !hasVenue) chips.push(S.shelf_missing_venue)
    if (hasWeakTitle) chips.push(S.shelf_weak_title)
    if (hasMetaConflict) chips.push(S.shelf_meta_conflict)
    if (bibliographicEntry && unresolved && chips.length <= 1) chips.push(S.shelf_pending_verify)

    if (!chips.length) return { chips: [], tip: '', needsRepair }

    let tip = S.shelf_auto_fix_tip
    if (externalNeedsReview) tip = item.externalMetadataReason || S.shelf_external_metadata_tip
    else if (bibliographicEntry && !hasDoi) tip = S.shelf_no_doi_tip
    else if (hasMetaConflict) tip = S.shelf_conflict_tip
    else if (hasWeakStoredTitle && !hasWeakTitle) tip = S.shelf_weak_stored_tip
    else if (hasWeakTitle) tip = S.shelf_weak_title_tip
    return { chips: chips.slice(0, 3), tip, needsRepair }
  }

  useEffect(() => {
    if (!open || sourceQualitySources.length <= 0) return
    let cancelled = false
    libraryApi.sourceQuality(sourceQualitySources)
      .then((res) => {
        if (cancelled) return
        const next: SourceQualityByPath = {}
        for (const item of Array.isArray(res.items) ? res.items : []) {
          const sourcePath = String(item.source_path || '').trim()
          if (!sourcePath) continue
          next[sourcePath] = item
        }
        setSourceQualityByPath((prev) => ({ ...prev, ...next }))
      })
      .catch(() => {
        if (!cancelled) setSourceQualityByPath((prev) => ({ ...prev }))
      })
    return () => {
      cancelled = true
    }
  }, [open, sourceQualityKey, sourceQualityRefreshToken, sourceQualitySources])

  useEffect(() => () => {
    sourceRepairStreamRef.current?.abort()
    sourceRepairStreamRef.current = null
  }, [])

  const duplicateCountByIdentity = useMemo(() => {
    const counter: Record<string, number> = {}
    for (const item of items) {
      const key = paperIdentity(item)
      counter[key] = (counter[key] || 0) + 1
    }
    return counter
  }, [items])

  const allTags = useMemo(() => {
    const seen = new Set<string>()
    const out: string[] = []
    for (const item of items) {
      for (const tag of normalizeShelfTags(item.tags)) {
        const key = tag.toLowerCase()
        if (seen.has(key)) continue
        seen.add(key)
        out.push(tag)
      }
    }
    return out.sort((a, b) => a.localeCompare(b, 'en'))
  }, [items])

  const shelfReadiness = useMemo(() => {
    let metadataReadyItems = 0
    let metadataReview = 0
    let duplicateItems = 0
    let summaryReady = 0
    let summaryReview = 0
    let sourceOpenable = 0
    let sourceOpenExact = 0
    let sourceOpenPartial = 0
    let sourceOpenReview = 0
    for (const item of items) {
      const display = citationDisplay(item)
      const needsMetadataReview = shelfItemNeedsMetadataRepair(item, display)
      const isDuplicate = (duplicateCountByIdentity[paperIdentity(item)] || 0) > 1
      const summaryDisplay = shelfSummaryDisplay(item, citationCardView(item), S)
      const hasSummary = Boolean(summaryDisplay.line)
      const summaryView = summaryDisplay.quality
      const sourceOpenView = sourceOpenQualityView(
        item,
        sourceQualityForItem(item, sourceQualityByPath),
        S,
        readerLocateResults[item.key],
      )

      if (needsMetadataReview) metadataReview += 1
      else metadataReadyItems += 1
      if (isDuplicate) duplicateItems += 1
      if (hasSummary && summaryView.ok) summaryReady += 1
      else summaryReview += 1
      if (sourceOpenView.status === 'repairing' || sourceOpenView.status === 'failed' || sourceOpenView.repairable) sourceOpenReview += 1
      if (sourceOpenView.canOpen && sourceOpenView.status !== 'repairing' && sourceOpenView.status !== 'failed') sourceOpenable += 1
      if (sourceOpenView.precision === 'exact_anchor') sourceOpenExact += 1
      if (sourceOpenView.status === 'partial' || sourceOpenView.status === 'degraded') sourceOpenPartial += 1
    }
    const total = items.length
    const summaryRate = total > 0 ? Math.round((summaryReady / total) * 100) : 0
    return {
      total,
      metadataReadyItems,
      metadataReview,
      duplicateItems,
      summaryRate,
      summaryReview,
      sourceOpenable,
      sourceOpenExact,
      sourceOpenPartial,
      sourceOpenReview,
      status: total <= 0 ? 'empty' : (metadataReview > 0 || sourceOpenReview > 0) ? 'review' : 'ready',
    }
  }, [S, duplicateCountByIdentity, items, readerLocateResults, sourceQualityByPath])

  const visibleItems = useMemo(() => {
    const keyword = searchText.trim().toLowerCase()
    const matched = items.filter((item) => {
      const tags = normalizeShelfTags(item.tags)
      if (tagFilter !== 'all' && !tags.some((tag) => tag.toLowerCase() === tagFilter.toLowerCase())) return false
      if (!keyword) return true
      const text = [
        item.title,
        item.main,
        item.authors,
        item.venue,
        item.doi,
        item.doiUrl,
        item.sourceName,
        item.note,
        item.shelfItemKind,
        shelfItemKindLabel(item.shelfItemKind, S),
        item.shelfOrigin,
        shelfOriginLabel(item.shelfOrigin, S),
        item.shelfExcerpt,
        item.traceAssistantOrder ? S.shelf_answer.replace('{n}', String(item.traceAssistantOrder)) : '',
        ...tags,
      ]
        .map((part) => String(part || '').toLowerCase())
        .join(' ')
      return text.includes(keyword)
    })
    const sorted = [...matched]
    if (sortKey === 'cited') {
      sorted.sort((a, b) => (b.citationCount || 0) - (a.citationCount || 0))
    } else if (sortKey === 'year') {
      sorted.sort((a, b) => Number(String(b.year || 0)) - Number(String(a.year || 0)))
    } else if (sortKey === 'impact') {
      sorted.sort((a, b) => impactScore(b) - impactScore(a))
    }
    return sorted
  }, [S, items, searchText, sortKey, tagFilter])

  const groupedVisibleItems = useMemo(() => {
    if (groupMode === 'none') {
      return [{ key: 'all', label: S.shelf_all, items: visibleItems }]
    }
    const groups = new Map<string, { label: string; items: CiteShelfItem[] }>()
    for (const item of visibleItems) {
      let groupKey = ''
      let groupLabel = ''
      if (groupMode === 'tag') {
        const tags = normalizeShelfTags(item.tags)
        const primaryTag = tags[0] || S.shelf_untagged
        groupKey = `tag:${primaryTag.toLowerCase()}`
        groupLabel = S.shelf_tag_prefix.replace('{tag}', primaryTag)
      } else if (groupMode === 'kind') {
        const kind = normalizeShelfItemKind(item.shelfItemKind)
        groupKey = `kind:${kind}`
        groupLabel = (S.shelf_type_prefix || 'Type · {type}').replace('{type}', shelfItemKindLabel(kind, S))
      } else {
        const src = String(item.sourceName || item.sourcePath || '').trim() || S.shelf_unknown_source
        groupKey = `source:${src}`
        groupLabel = S.shelf_source_prefix.replace('{src}', src)
      }
      const existing = groups.get(groupKey)
      if (existing) {
        existing.items.push(item)
      } else {
        groups.set(groupKey, { label: groupLabel, items: [item] })
      }
    }
    return Array.from(groups.entries()).map(([k, v]) => ({ key: k, label: v.label, items: v.items }))
  }, [
    groupMode,
    S,
    visibleItems,
  ])

  const selectedCount = Object.values(selectedKeys).filter(Boolean).length
  const selectedItems = useMemo(
    () => items.filter((item) => Boolean(selectedKeys[item.key])),
    [items, selectedKeys],
  )
  const selectedMetadataReviewItems = useMemo(
    () => selectedItems.filter((item) => shelfItemNeedsMetadataRepair(item, citationDisplay(item))),
    [selectedItems],
  )
  const selectedMetadataReviewCount = selectedMetadataReviewItems.length
  const selectedMetadataReviewKeySet = useMemo(
    () => new Set(selectedMetadataReviewItems.map((item) => item.key)),
    [selectedMetadataReviewItems],
  )
  const selectedReviewSources = useMemo(() => {
    const seen = new Set<string>()
    const out: Array<{ source_path: string; source_name: string }> = []
    for (const item of selectedItems) {
      const sourcePath = String(item.sourcePath || '').trim()
      if (!sourcePath || seen.has(sourcePath)) continue
      const locateView = sourceOpenQualityView(
        item,
        sourceQualityByPath[sourcePath]?.conversion_quality || null,
        S,
        readerLocateResults[item.key],
      )
      if (!sourceQualityNeedsReview(sourceQualityByPath[sourcePath]?.conversion_quality) && !locateView.repairable) continue
      seen.add(sourcePath)
      out.push({
        source_path: sourcePath,
        source_name: String(item.sourceName || item.title || item.main || '').trim(),
      })
    }
    return out
  }, [S, readerLocateResults, selectedItems, sourceQualityByPath])
  const selectedReviewSourceKey = useMemo(
    () => sourceListKey(selectedReviewSources),
    [selectedReviewSources],
  )
  const locateRepairSources = useMemo(() => {
    const seen = new Set<string>()
    const out: Array<{ source_path: string; source_name: string }> = []
    for (const item of items) {
      const sourcePath = String(item.sourcePath || '').trim()
      if (!sourcePath || seen.has(sourcePath)) continue
      const locateView = sourceOpenQualityView(
        item,
        sourceQualityByPath[sourcePath]?.conversion_quality || null,
        S,
        readerLocateResults[item.key],
      )
      if (!locateView.repairable) continue
      seen.add(sourcePath)
      out.push({
        source_path: sourcePath,
        source_name: String(item.sourceName || item.title || item.main || '').trim(),
      })
    }
    return out
  }, [S, items, readerLocateResults, sourceQualityByPath])
  const locateRepairSourceKey = useMemo(
    () => sourceListKey(locateRepairSources),
    [locateRepairSources],
  )
  const visibleKeySet = useMemo(() => new Set(visibleItems.map((item) => item.key)), [visibleItems])
  const visibleSelectedCount = useMemo(
    () => visibleItems.reduce((acc, item) => acc + (selectedKeys[item.key] ? 1 : 0), 0),
    [selectedKeys, visibleItems],
  )
  const advancedFilterActive = (groupMode !== 'none') || (tagFilter !== 'all')
  const snapshotOptions = useMemo(
    () => snapshots.map((item) => {
      const created = Number(item.createdAt || 0)
      const labelTime = Number.isFinite(created) && created > 0
        ? new Date(created).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })
        : ''
      return {
        value: item.id,
        label: labelTime ? `${item.name} · ${labelTime}` : item.name,
      }
    }),
    [snapshots],
  )
  const hasSnapshotChoices = snapshotOptions.length > 0 || Boolean(selectedSnapshotId)
  const showSnapshotTools = hasSnapshotChoices || Boolean(snapshotDiff)

  const setTransientCopyState = (next: 'gbt' | 'bibtex' | 'error') => {
    setCopyState(next)
    if (copyStateTimerRef.current !== null) {
      window.clearTimeout(copyStateTimerRef.current)
      copyStateTimerRef.current = null
    }
    copyStateTimerRef.current = window.setTimeout(() => {
      setCopyState('idle')
      copyStateTimerRef.current = null
    }, 1800)
  }

  const writeClipboard = async (text: string) => {
    const payload = String(text || '').trim()
    if (!payload) return
    if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(payload)
      return
    }
    const el = document.createElement('textarea')
    el.value = payload
    el.setAttribute('readonly', 'true')
    el.style.position = 'fixed'
    el.style.left = '-9999px'
    document.body.appendChild(el)
    el.select()
    const ok = document.execCommand('copy')
    document.body.removeChild(el)
    if (!ok) throw new Error('clipboard-copy-failed')
  }

  const copySelectedAs = async (kind: 'gbt' | 'bibtex') => {
    if (selectedItems.length <= 0) return
    const text = selectedItems.map((item) => citationFormats(item)[kind]).join('\n\n')
    try {
      await writeClipboard(text)
      setTransientCopyState(kind)
    } catch {
      setTransientCopyState('error')
    }
  }

  const repairSources = useCallback(async (
    sources: Array<{ source_path: string; source_name: string }>,
    options: { silent?: boolean; repairKey?: string } = {},
  ) => {
    const silent = Boolean(options.silent)
    if (sources.length <= 0) {
      if (!silent) message.info(S.shelf_source_quality_repair_none)
      return
    }
    const repairKey = options.repairKey || sourceListKey(sources)
    const repairSources = sources.map((item) => ({ ...item }))
    const refreshRepairSources = async () => {
      if (repairSources.length <= 0) return
      const res = await libraryApi.sourceQuality(repairSources)
      const next: SourceQualityByPath = {}
      for (const item of Array.isArray(res.items) ? res.items : []) {
        const sourcePath = String(item.source_path || '').trim()
        if (!sourcePath) continue
        next[sourcePath] = item
      }
      setSourceQualityByPath((prev) => ({ ...prev, ...next }))
    }
    const refreshRepairRunAndSources = async (runId: string, needsReindex: boolean) => {
      if (needsReindex) {
        let advanced = false
        if (runId) {
          try {
            await libraryApi.advanceQualityRepairRun(runId)
            advanced = true
          } catch {
            advanced = false
          }
        }
        try {
          if (!advanced) await libraryApi.reindex()
        } catch {
          // Source quality will still be refreshed so the UI can show the latest diagnostics.
        }
      }
      await refreshRepairSources()
    }
    const clearRepairing = () => {
      setSourceRepairingKey((cur) => (cur === repairKey ? '' : cur))
    }
    setSourceRepairingKey(repairKey)
    let watchingConversion = false
    try {
      const res = await libraryApi.repairQuality({
        sources: repairSources,
        speed_mode: 'balanced',
        replace: true,
      })
      const runId = String(res.repair_run_id || res.repair_run?.run_id || '').trim()
      const queued = Number(res.enqueued || 0)
      const repaired = Number(res.repaired || 0)
      const needsReindex = Boolean(res.needs_reindex || res.impact?.needs_reindex)
      if (queued > 0) {
        if (!silent) message.success(S.shelf_source_quality_repair_queued.replace('{n}', String(queued)))
        watchingConversion = true
        sourceRepairStreamRef.current?.abort()
        sourceRepairStreamRef.current = libraryApi.streamConvertStatus(
          () => {},
          () => {
            sourceRepairStreamRef.current = null
            void refreshRepairRunAndSources(runId, needsReindex).finally(clearRepairing)
          },
          () => {
            sourceRepairStreamRef.current = null
            void refreshRepairRunAndSources(runId, needsReindex).finally(clearRepairing)
          },
        )
      } else if (repaired > 0) {
        if (!silent) message.success(`Markdown repaired: ${repaired}`)
        await refreshRepairRunAndSources(runId, needsReindex)
      } else if (needsReindex) {
        await refreshRepairRunAndSources(runId, needsReindex)
      } else {
        if (!silent) message.info(S.shelf_source_quality_repair_none)
        await refreshRepairSources()
      }
    } catch (err) {
      if (!silent) message.error(err instanceof Error ? err.message : S.shelf_source_quality_repair_fail)
    } finally {
      if (!watchingConversion) clearRepairing()
    }
  }, [S])

  useEffect(() => {
    if (!open || selectedReviewSources.length <= 0 || sourceRepairingKey) return
    const pending = selectedReviewSources.filter((item) => {
      const key = item.source_path || item.source_name
      return key && !autoSourceRepairKeysRef.current[key]
    })
    if (pending.length <= 0) return
    const timer = window.setTimeout(() => {
      for (const item of pending) {
        const key = item.source_path || item.source_name
        if (key) autoSourceRepairKeysRef.current[key] = true
      }
      void repairSources(pending, { silent: true, repairKey: sourceListKey(pending) })
    }, 350)
    return () => window.clearTimeout(timer)
  }, [open, repairSources, selectedReviewSourceKey, selectedReviewSources, sourceRepairingKey])

  useEffect(() => {
    if (!open || locateRepairSources.length <= 0 || sourceRepairingKey) return
    const pending = locateRepairSources.filter((item) => {
      const key = `locate:${item.source_path || item.source_name}`
      return key && !autoSourceRepairKeysRef.current[key]
    })
    if (pending.length <= 0) return
    const timer = window.setTimeout(() => {
      for (const item of pending) {
        const key = `locate:${item.source_path || item.source_name}`
        if (key) autoSourceRepairKeysRef.current[key] = true
      }
      void repairSources(pending, { silent: true, repairKey: sourceListKey(pending) })
    }, 450)
    return () => window.clearTimeout(timer)
  }, [locateRepairSourceKey, locateRepairSources, open, repairSources, sourceRepairingKey])

  const nowStamp = () => {
    const d = new Date()
    const pad = (v: number) => String(v).padStart(2, '0')
    return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}`
  }

  const downloadTextFile = (filename: string, text: string, mime = 'text/plain;charset=utf-8') => {
    const blob = new Blob([text], { type: mime })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  const csvEscape = (value: unknown): string => {
    const text = String(value ?? '')
    if (!text) return ''
    if (!/[",\n]/.test(text)) return text
    return `"${text.replace(/"/g, '""')}"`
  }

  const repairPayloadsForExport = (item: CiteShelfItem): Array<Record<string, unknown>> => {
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

  const repairMetaFromEntry = (entry: ShelfMetadataRepairItem): Record<string, unknown> => ({
    ...(entry.meta || {}),
    metadata_quality: entry.after || (entry.meta || {}).metadata_quality,
    metadata_repair_status: entry.repair_status,
    metadata_changed_fields: entry.changed_fields || [],
    metadata_repair_sources: entry.repair_sources || [],
  })

  const repairMetadataBeforeExport = async (kind: ShelfExportKind, exportItems: CiteShelfItem[]): Promise<CiteShelfItem[]> => {
    const candidates = exportItems.filter((item) => shelfItemNeedsMetadataRepair(item, citationDisplay(item)))
    if (candidates.length <= 0) return exportItems

    const payloads = candidates.flatMap((item) => repairPayloadsForExport(item))
    const noticeKey = `cite-shelf-export-repair-${kind}`
    setExportRepairingKind(kind)
    message.loading({
      key: noticeKey,
      content: S.shelf_export_repairing.replace('{n}', String(candidates.length)),
      duration: 0,
    })
    try {
      const res = await referencesApi.repairShelfMetadata(payloads, payloads.length)
      const metasByKey = new Map<string, Array<Record<string, unknown>>>()
      for (const entry of Array.isArray(res.items) ? res.items : []) {
        const meta = repairMetaFromEntry(entry)
        if (!meta || Object.keys(meta).length <= 0) continue
        const key = String(entry.key || meta.key || '').trim()
        if (!key) continue
        metasByKey.set(key, [...(metasByKey.get(key) || []), meta])
      }
      const updates = Array.from(metasByKey.entries()).map(([key, metas]) => ({ key, metas }))
      onApplyRepairCandidates?.(updates)

      let repairedReadyCount = 0
      let unresolvedCount = 0
      const repairedItems = exportItems.map((item) => {
        const wasReady = !shelfItemNeedsMetadataRepair(item, citationDisplay(item))
        const metas = metasByKey.get(item.key) || []
        let next = item
        for (const meta of metas) {
          const accepted = strictRepairMerge(next, meta)
          if (accepted) next = accepted
        }
        const isReady = !shelfItemNeedsMetadataRepair(next, citationDisplay(next))
        if (!wasReady && isReady) repairedReadyCount += 1
        if (!isReady) unresolvedCount += 1
        return next
      })

      if (repairedReadyCount > 0 && unresolvedCount <= 0) {
        message.success({
          key: noticeKey,
          content: S.shelf_export_repaired.replace('{n}', String(repairedReadyCount)),
          duration: 2,
        })
      } else if (repairedReadyCount > 0) {
        message.warning({
          key: noticeKey,
          content: S.shelf_export_repaired_partial
            .replace('{n}', String(repairedReadyCount))
            .replace('{m}', String(unresolvedCount)),
          duration: 3,
        })
      } else {
        message.warning({
          key: noticeKey,
          content: S.shelf_export_repair_no_change,
          duration: 3,
        })
      }
      return repairedItems
    } catch {
      message.warning({
        key: noticeKey,
        content: S.shelf_export_repair_failed,
        duration: 3,
      })
      return exportItems
    } finally {
      setExportRepairingKind((current) => (current === kind ? '' : current))
    }
  }

  const exportSelectedAs = async (kind: ShelfExportKind, options: ShelfExportOptions = {}) => {
    if (selectedItems.length <= 0) return
    if (exportRepairingKind) return
    if (selectedMetadataReviewCount > 0 && !options.skipPreflight) {
      setPreflightExportKind(kind)
      return
    }
    let exportItems = options.onlyMetadataReady
      ? selectedItems.filter((item) => !selectedMetadataReviewKeySet.has(item.key))
      : selectedItems
    if (exportItems.length <= 0) {
      message.warning(S.shelf_export_preflight_no_healthy)
      return
    }
    if (options.autoRepair) {
      exportItems = await repairMetadataBeforeExport(kind, exportItems)
    }
    try {
      const base = `cite_shelf_selected_${nowStamp()}`
      if (kind === 'bib') {
        const bib = exportItems.map((item) => citationFormats(item).bibtex).join('\n\n').trim()
        if (!bib) return
        downloadTextFile(`${base}.bib`, bib, 'application/x-bibtex;charset=utf-8')
        message.success(S.shelf_export_bibtex.replace('{n}', String(exportItems.length)))
        return
      }
      if (kind === 'ris') {
        const ris = exportItems.map((item) => citationFormats(item).ris).join('\n\n').trim()
        if (!ris) return
        downloadTextFile(`${base}.ris`, ris, 'application/x-research-info-systems;charset=utf-8')
        message.success(S.shelf_export_ris.replace('{n}', String(exportItems.length)))
        return
      }
      const headers = [
        'title',
        'authors',
        'year',
        'venue',
        'doi',
        'source',
        'source_quality_status',
        'source_quality_issues',
        'source_open_status',
        'source_open_precision',
        'source_open_reason',
        'library_match_status',
        'library_match_method',
        'library_match_path',
        'reference_num',
        'citation_count',
        'journal_if',
        'journal_quartile',
        'conference_tier',
        'conference_ccf',
        'summary_source',
        'summary_provider',
        'summary_quality_status',
        'summary_quality_score',
        'summary',
      ]
      const rows = exportItems.map((item) => ([
        sourceQualityForItem(item, sourceQualityByPath),
        item,
      ] as const)).map(([sourceQuality, item]) => ([
        citationCardView(item).header.title || item.title || item.main,
        item.authors,
        item.year,
        item.venue,
        doiExportValue(item),
        item.sourceName || item.sourcePath,
        sourceQuality?.status || '',
        (sourceQuality?.issues || []).map((issue) => issue.label || issue.code).filter(Boolean).join('; '),
        sourceOpenQualityView(item, sourceQuality, S, readerLocateResults[item.key]).status,
        sourceOpenQualityView(item, sourceQuality, S, readerLocateResults[item.key]).precision,
        sourceOpenQualityView(item, sourceQuality, S, readerLocateResults[item.key]).reason,
        item.libraryMatchStatus,
        item.libraryMatchMethod,
        item.libraryMatchPath,
        item.num || '',
        item.citationCount || 0,
        item.journalIf,
        item.journalQuartile,
        item.conferenceTier,
        item.conferenceCcf,
        item.summarySource,
        item.summaryProvider,
        summaryQualityView(item, S).status,
        summaryQualityView(item, S).score,
        item.summaryLine || citationCardView(item).summary,
      ].map((field) => csvEscape(field)).join(',')))
      const csv = `${headers.join(',')}\n${rows.join('\n')}`
      downloadTextFile(`${base}.csv`, csv, 'text/csv;charset=utf-8')
      message.success(S.shelf_export_csv.replace('{n}', String(exportItems.length)))
    } catch {
      message.error(S.shelf_export_failed)
    }
  }

  const toggleSelect = (key: string, checked: boolean) => {
    setSelectedKeys((prev) => {
      const next = { ...prev }
      if (checked) next[key] = true
      else delete next[key]
      return next
    })
  }
  const removeSelected = () => {
    const keys = Object.keys(selectedKeys).filter((key) => selectedKeys[key])
    for (const key of keys) onRemove(key)
    setSelectedKeys({})
  }
  const clearSelected = () => setSelectedKeys({})
  const addVisibleToSelection = () => {
    if (visibleItems.length <= 0) return
    setSelectedKeys((prev) => {
      const next = { ...prev }
      for (const item of visibleItems) {
        next[item.key] = true
      }
      return next
    })
  }
  const removeVisibleFromSelection = () => {
    if (visibleItems.length <= 0) return
    setSelectedKeys((prev) => {
      const next = { ...prev }
      for (const key of visibleKeySet) {
        delete next[key]
      }
      return next
    })
  }

  const applyTagToSelected = (tagInput: string) => {
    const clean = normalizeShelfTags([tagInput])[0]
    if (!clean) return
    for (const item of selectedItems) {
      const nextTags = normalizeShelfTags([...(item.tags || []), clean])
      onUpdateTags(item.key, nextTags)
    }
    setBatchTagInput('')
  }

  const removeTagFromSelected = (tagInput: string) => {
    const clean = normalizeShelfTags([tagInput])[0]
    if (!clean) return
    const key = clean.toLowerCase()
    for (const item of selectedItems) {
      const nextTags = normalizeShelfTags((item.tags || []).filter((tag) => tag.toLowerCase() !== key))
      onUpdateTags(item.key, nextTags)
    }
  }

  useEffect(() => {
    const validKeys = new Set(items.map((item) => item.key))
    setSelectedKeys((prev) => {
      const next: Record<string, boolean> = {}
      let changed = false
      for (const [key, checked] of Object.entries(prev)) {
        if (!checked) continue
        if (validKeys.has(key)) next[key] = true
        else changed = true
      }
      if (!changed && Object.keys(next).length !== Object.keys(prev).length) changed = true
      return changed ? next : prev
    })
    setExpandedSummaryKeys((prev) => {
      const next: Record<string, boolean> = {}
      let changed = false
      for (const [key, expanded] of Object.entries(prev)) {
        if (!expanded) continue
        if (validKeys.has(key)) next[key] = true
        else changed = true
      }
      if (!changed && Object.keys(next).length !== Object.keys(prev).length) changed = true
      return changed ? next : prev
    })
    setEditingNoteKeys((prev) => {
      const next: Record<string, boolean> = {}
      let changed = false
      for (const [key, editing] of Object.entries(prev)) {
        if (!editing) continue
        if (validKeys.has(key)) next[key] = true
        else changed = true
      }
      if (!changed && Object.keys(next).length !== Object.keys(prev).length) changed = true
      return changed ? next : prev
    })
  }, [items])

  useEffect(() => {
    if (tagFilter === 'all') return
    if (allTags.some((tag) => tag.toLowerCase() === tagFilter.toLowerCase())) return
    setTagFilter('all')
  }, [allTags, tagFilter])

  useEffect(() => {
    if (selectedMetadataReviewCount > 0) return
    setPreflightExportKind('')
  }, [selectedMetadataReviewCount])

  useEffect(() => {
    return () => {
      if (copyStateTimerRef.current !== null) {
        window.clearTimeout(copyStateTimerRef.current)
        copyStateTimerRef.current = null
      }
    }
  }, [])

  return (
    <>
      {!isDockPresentation ? (
        <button
          aria-label={S.shelf_title}
          className={`kb-shelf-toggle-btn fixed right-4 top-1/2 z-30 -translate-y-1/2 transition ${open ? 'pointer-events-none opacity-0' : ''}`}
          data-testid="citation-shelf-toggle"
          onClick={onToggle}
          type="button"
        >
          {S.shelf_title}
        </button>
      ) : null}
      <aside
        className={isDockPresentation
          ? `kb-shelf-panel kb-shelf-panel-docked ${panelVisible ? 'is-visible' : 'is-hidden'}`
          : `kb-shelf-panel fixed right-0 top-0 z-40 h-full w-[360px] max-w-[92vw] transition-transform duration-300 ${open ? 'translate-x-0' : 'translate-x-full'}`}
        data-testid="citation-shelf"
      >
        <div className="flex h-full flex-col">
          <div className="kb-shelf-head border-b border-[var(--border)] px-3 py-3">
            <div className="kb-shelf-head-top">
              <div className="kb-shelf-head-meta">
                <div className="kb-shelf-title">{S.shelf_title}</div>
                <div className="kb-shelf-count">
                  {S.shelf_collect_count.replace('{n}', String(items.length))}{searchText.trim() ? ` · ${S.shelf_match_count.replace('{n}', String(visibleItems.length))}` : ''}
                </div>
              </div>
              <div className="kb-shelf-head-actions">
                <Button
                  size="small"
                  icon={<SaveOutlined />}
                  onClick={onSaveSnapshot}
                  disabled={items.length === 0}
                  aria-label={S.shelf_save_snapshot}
                  title={S.shelf_save_snapshot}
                  data-testid="citation-shelf-save-snapshot"
                />
                <Button
                  size="small"
                  icon={<DeleteOutlined />}
                  onClick={onClear}
                  disabled={items.length === 0}
                  aria-label={S.shelf_clear}
                  title={S.shelf_clear}
                  data-testid="citation-shelf-clear"
                />
                <Button
                  size="small"
                  icon={<CloseOutlined />}
                  onClick={onToggle}
                  aria-label={S.shelf_close}
                  title={S.shelf_close}
                  data-testid="citation-shelf-close"
                />
              </div>
            </div>
            {showSnapshotTools ? (
              <>
                {hasSnapshotChoices ? (
                  <div
                    className="kb-shelf-snapshot-row"
                    onClick={(event) => event.stopPropagation()}
                  >
                    <>
                      <Select
                        size="small"
                        value={selectedSnapshotId || undefined}
                        placeholder={snapshotOptions.length > 0 ? S.shelf_select_snapshot : S.shelf_no_snapshot}
                        className="kb-shelf-snapshot-select"
                        data-testid="citation-shelf-snapshot-select"
                        options={snapshotOptions}
                        onChange={(value) => onSelectSnapshot(String(value || ''))}
                      />
                      <Button size="small" onClick={onLoadSnapshot} disabled={!selectedSnapshotId} data-testid="citation-shelf-load-snapshot">
                        {S.shelf_load}
                      </Button>
                      <Button size="small" onClick={onDeleteSnapshot} disabled={!selectedSnapshotId} data-testid="citation-shelf-delete-snapshot">
                        {S.shelf_delete}
                      </Button>
                    </>
                  </div>
                ) : null}
                {snapshotDiff ? (
                  <div className="kb-shelf-snapshot-diff">{snapshotDiff}</div>
                ) : null}
              </>
            ) : null}
            {items.length > 0 ? (
              <div
                className={`kb-shelf-readiness is-${shelfReadiness.status}`}
                data-testid="citation-shelf-readiness"
              >
                <div className="kb-shelf-readiness-main">
                  <span className="kb-shelf-readiness-status">
                    {shelfReadiness.status === 'ready'
                      ? S.shelf_readiness_ready
                      : S.shelf_readiness_review}
                  </span>
                  <span className="kb-shelf-readiness-count">
                    {S.shelf_readiness_count
                      .replace('{ready}', String(shelfReadiness.metadataReadyItems))
                      .replace('{total}', String(shelfReadiness.total))}
                  </span>
                  <span className={`kb-shelf-readiness-chip ${shelfReadiness.sourceOpenReview > 0 ? 'is-review' : ''}`}>
                    {shelfReadiness.sourceOpenReview > 0
                      ? S.shelf_readiness_source_open_review.replace('{n}', String(shelfReadiness.sourceOpenReview))
                      : S.shelf_readiness_source_open_ready.replace('{n}', String(shelfReadiness.sourceOpenable))}
                  </span>
                  {shelfReadiness.summaryReview > 0 ? (
                    <span className="kb-shelf-readiness-chip is-review">
                      {S.shelf_readiness_summary_review.replace('{n}', String(shelfReadiness.summaryReview))}
                    </span>
                  ) : (
                    <span className="kb-shelf-readiness-chip">
                      {S.shelf_readiness_summary_grounded.replace('{n}', `${shelfReadiness.summaryRate}%`)}
                    </span>
                  )}
                  {shelfReadiness.metadataReview > 0 ? (
                    <span className="kb-shelf-readiness-chip is-review">
                      {S.shelf_readiness_meta.replace('{n}', String(shelfReadiness.metadataReview))}
                    </span>
                  ) : null}
                  {shelfReadiness.duplicateItems > 0 ? (
                    <span className="kb-shelf-readiness-chip is-review">
                      {S.shelf_readiness_dups.replace('{n}', String(shelfReadiness.duplicateItems))}
                    </span>
                  ) : null}
                  {shelfReadiness.sourceOpenPartial > 0 ? (
                    <span className="kb-shelf-readiness-chip is-calibrating">
                      {S.shelf_readiness_source_open_partial.replace('{n}', String(shelfReadiness.sourceOpenPartial))}
                    </span>
                  ) : null}
                </div>
                {repairImpact ? (
                  <div className="kb-shelf-repair-impact" data-testid="citation-shelf-repair-impact">
                    <span>{S.shelf_repair_impact_changed.replace('{n}', String(repairImpact.changed))}</span>
                    {repairImpact.score_delta ? <span>Q{repairImpact.before_avg_score} -&gt; Q{repairImpact.after_avg_score}</span> : null}
                    {(repairImpact.changed_fields || []).slice(0, 3).map((field) => (
                      <span key={`field-${field.name}`}>{field.name} x{field.count}</span>
                    ))}
                  </div>
                ) : null}
              </div>
            ) : null}
            {selectedCount > 0 ? (
              <>
                <div className="kb-shelf-batch-row">
                  <span className="kb-shelf-batch-count" data-testid="citation-shelf-batch-count">{S.shelf_batch_count.replace('{n}', String(selectedCount))}</span>
                  <Button size="small" onClick={removeSelected} data-testid="citation-shelf-batch-remove">
                    {S.shelf_batch_remove}
                  </Button>
                  <Button size="small" onClick={() => void copySelectedAs('gbt')} data-testid="citation-shelf-copy-gbt">
                    {copyState === 'gbt' ? S.shelf_copied_gbt : S.shelf_copy_gbt}
                  </Button>
                  <Button size="small" onClick={() => void copySelectedAs('bibtex')} data-testid="citation-shelf-copy-bibtex">
                    {copyState === 'bibtex' ? S.shelf_copied_bibtex : S.shelf_copy_bibtex}
                  </Button>
                  <Button
                    size="small"
                    onClick={() => void exportSelectedAs('bib')}
                    loading={exportRepairingKind === 'bib'}
                    disabled={Boolean(exportRepairingKind && exportRepairingKind !== 'bib')}
                    data-testid="citation-shelf-export-bib"
                  >
                    {S.shelf_export_bib_btn}
                  </Button>
                  <Button
                    size="small"
                    onClick={() => void exportSelectedAs('ris')}
                    loading={exportRepairingKind === 'ris'}
                    disabled={Boolean(exportRepairingKind && exportRepairingKind !== 'ris')}
                    data-testid="citation-shelf-export-ris"
                  >
                    {S.shelf_export_ris_btn}
                  </Button>
                  <Button
                    size="small"
                    onClick={() => void exportSelectedAs('csv')}
                    loading={exportRepairingKind === 'csv'}
                    disabled={Boolean(exportRepairingKind && exportRepairingKind !== 'csv')}
                    data-testid="citation-shelf-export-csv"
                  >
                    {S.shelf_export_csv_btn}
                  </Button>
                  <div className="flex min-w-[170px] items-center gap-1" onClick={(event) => event.stopPropagation()}>
                    <Select
                      size="small"
                      value={batchTagInput || undefined}
                      placeholder={S.shelf_batch_tag_placeholder}
                      style={{ minWidth: 124 }}
                      showSearch
                      onChange={(value) => {
                        setBatchTagInput(value)
                        applyTagToSelected(value)
                      }}
                      options={[...TAG_PRESETS, ...allTags]
                        .filter((tag, idx, arr) => arr.findIndex((x) => x.toLowerCase() === tag.toLowerCase()) === idx)
                        .map((tag) => ({ value: tag, label: tag }))}
                    />
                    <Button
                      size="small"
                      onClick={() => {
                        if (!batchTagInput.trim()) return
                        removeTagFromSelected(batchTagInput)
                        setBatchTagInput('')
                      }}
                    >
                      {S.shelf_remove_tag}
                    </Button>
                  </div>
                  <button type="button" className="kb-shelf-clear-select" onClick={clearSelected}>
                    {S.shelf_clear_selection}
                  </button>
                </div>
                {preflightExportKind && selectedMetadataReviewCount > 0 ? (
                  <div className="kb-shelf-export-preflight" data-testid="citation-shelf-export-preflight">
                    <div className="kb-shelf-export-preflight-copy">
                      <strong>{S.shelf_export_preflight_title}</strong>
                      <span>{S.shelf_export_preflight_body.replace('{n}', String(selectedMetadataReviewCount))}</span>
                    </div>
                    <div className="kb-shelf-export-preflight-actions">
                      <Button
                        size="small"
                        onClick={() => {
                          void exportSelectedAs(preflightExportKind, { skipPreflight: true, onlyMetadataReady: true })
                          setPreflightExportKind('')
                        }}
                        disabled={Boolean(exportRepairingKind)}
                        data-testid="citation-shelf-export-preflight-healthy"
                      >
                        {S.shelf_export_preflight_healthy}
                      </Button>
                      <Button
                        size="small"
                        onClick={async () => {
                          await exportSelectedAs(preflightExportKind, { skipPreflight: true, autoRepair: true })
                          setPreflightExportKind('')
                        }}
                        loading={Boolean(exportRepairingKind)}
                        data-testid="citation-shelf-export-preflight-continue"
                      >
                        {S.shelf_export_preflight_autofill}
                      </Button>
                    </div>
                  </div>
                ) : null}
              </>
            ) : null}
          </div>
          <div className="kb-shelf-scroll flex-1 overflow-y-auto px-3 py-3">
            {items.length > 0 ? (
              <div className="kb-shelf-toolbar-wrap">
                <div className="kb-shelf-toolbar">
                  <div className="kb-shelf-toolbar-main">
                  <Input
                    allowClear
                    placeholder={S.shelf_search_placeholder}
                    value={searchText}
                    onChange={(event) => setSearchText(event.target.value)}
                    className="kb-shelf-search"
                    data-testid="citation-shelf-search"
                  />
                  <Select
                    value={sortKey}
                    onChange={(value) => setSortKey(value)}
                    className="kb-shelf-sort"
                    options={[
                      { value: 'recent', label: S.shelf_sort_recent },
                      { value: 'cited', label: S.shelf_sort_cited },
                      { value: 'year', label: S.shelf_sort_year },
                      { value: 'impact', label: S.shelf_sort_impact },
                    ]}
                  />
                  <button
                    type="button"
                    className={`kb-shelf-advanced-toggle ${advancedFiltersOpen ? 'is-open' : ''} ${advancedFilterActive ? 'is-active' : ''}`}
                    onClick={() => setAdvancedFiltersOpen((prev) => !prev)}
                  >
                    {advancedFiltersOpen ? S.shelf_advanced_collapse : S.shelf_advanced_filter}
                  </button>
                  </div>
                  {advancedFiltersOpen ? (
                    <div className="kb-shelf-filters">
                  <Select
                    value={groupMode}
                    onChange={(value) => setGroupMode(value as GroupMode)}
                    className="kb-shelf-sort"
                    options={[
                      { value: 'none', label: S.shelf_group_none },
                      { value: 'tag', label: S.shelf_group_tag },
                      { value: 'source', label: S.shelf_group_source },
                      { value: 'kind', label: S.shelf_group_type },
                    ]}
                  />
                  <Select
                    allowClear
                    value={tagFilter === 'all' ? undefined : tagFilter}
                    onChange={(value) => setTagFilter(value || 'all')}
                    className="kb-shelf-sort"
                    placeholder={S.shelf_tag_filter_placeholder}
                    options={allTags.map((tag) => ({ value: tag, label: tag }))}
                  />
                  <Button size="small" onClick={addVisibleToSelection} disabled={visibleItems.length <= 0} data-testid="citation-shelf-add-visible">
                    {S.shelf_add_to_queue}
                  </Button>
                  <Button size="small" onClick={removeVisibleFromSelection} disabled={visibleSelectedCount <= 0} data-testid="citation-shelf-remove-visible">
                    {S.shelf_remove_from_queue}
                  </Button>
                </div>
                  ) : null}
                  {!advancedFiltersOpen && advancedFilterActive ? (
                    <div className="kb-shelf-filter-pills">
                      {groupMode !== 'none' ? (
                        <button
                          type="button"
                          className="kb-shelf-filter-pill"
                          onClick={() => setGroupMode('none')}
                        >
                          {S.shelf_filter_pill_group.replace('{mode}', GROUP_MODE_LABEL(S)[groupMode])}
                        </button>
                      ) : null}
                      {tagFilter !== 'all' ? (
                        <button
                          type="button"
                          className="kb-shelf-filter-pill"
                          onClick={() => setTagFilter('all')}
                        >
                          {S.shelf_filter_pill_tag.replace('{tag}', tagFilter)}
                        </button>
                      ) : null}
                    </div>
                  ) : null}
                </div>
                {copyState === 'error' ? (
                  <div className="kb-shelf-copy-hint">{S.shelf_copy_error}</div>
                ) : null}
              </div>
            ) : null}
            {items.length === 0 ? (
              <div className="kb-shelf-empty">
                {S.shelf_empty_hint}
              </div>
            ) : (
              <div className="kb-shelf-list space-y-2">
                {groupedVisibleItems.map((group) => (
                  <div key={group.key} className="space-y-2">
                    {groupMode !== 'none' ? (
                      <div className="kb-shelf-group-title">
                        {group.label} · {group.items.length}
                      </div>
                    ) : null}
                    {group.items.map((item) => {
                      const display = citationDisplay(item)
                      const cardView = citationCardView(item)
                      const shelfTitle = String(cardView.header.title || display.main || item.main || '').trim()
                      const duplicateCount = duplicateCountByIdentity[paperIdentity(item)] || 0
                      const trace = sourceTraceLabel(item)
                      const visibleTraceLabels = trace.labels.length > 1 ? trace.labels.slice(1) : trace.labels
                      const itemTags = normalizeShelfTags(item.tags)
                      const quality = qualityHints(item, display)
                      const noteText = String(item.note || '').trim()
                      const isFocused = item.key === focusedKey
                      const metrics = citeMetricSummary(item)
                      const visibleMetrics = isFocused ? metrics : metrics.slice(0, 2)
                      const shelfSummary = shelfSummaryDisplay(item, cardView, S)
                      const shelfSummaryLine = shelfSummary.line
                      const shelfSummarySource = shelfSummary.sourceLabel
                      const shelfSummaryQuality = shelfSummary.quality
                      const itemSourceQuality = sourceQualityForItem(item, sourceQualityByPath)
                      const itemSourceOpenQuality = sourceOpenQualityView(
                        item,
                        itemSourceQuality,
                        S,
                        readerLocateResults[item.key],
                      )
                      const noteEditing = Boolean(editingNoteKeys[item.key] && isFocused)
                      const visibleQualityChips = isFocused ? quality.chips.slice(0, 3) : quality.chips.slice(0, 1)
                      const showQuality = Boolean(quality.needsRepair || isFocused)
                      const libraryMatch = libraryMatchView(item)
                      const shelfKind = normalizeShelfItemKind(item.shelfItemKind)
                      const shelfKindText = shelfItemKindLabel(shelfKind, S)
                      const shelfOriginText = shelfOriginLabel(item.shelfOrigin, S)
                      const shelfExcerpt = cleanCitationDisplayText(item.shelfExcerpt || '')
                      const rawShelfExcerptLabel = String(item.shelfExcerptLabel || '').trim()
                      const shelfExcerptLabel = rawShelfExcerptLabel === 'Reference entry'
                        ? S.shelf_reference_entry
                        : rawShelfExcerptLabel === 'Selected text'
                          ? S.shelf_reader_selection_selected
                          : rawShelfExcerptLabel === 'Excerpt'
                            ? S.shelf_excerpt_head
                            : rawShelfExcerptLabel || S.shelf_excerpt_head

                      return (
                        <div
                          key={item.key}
                          className={`kb-shelf-item ${
                            isFocused
                              ? 'kb-shelf-item-active'
                              : ''
                          }`}
                          data-testid="citation-shelf-item"
                          onClick={() => onSelect(item)}
                          onKeyDown={(event) => {
                            if (event.key === 'Enter' || event.key === ' ') {
                              event.preventDefault()
                              onSelect(item)
                            }
                          }}
                          role="button"
                          tabIndex={0}
                        >
                          <div className="kb-shelf-item-head">
                            <input
                              aria-label={`select-${item.key}`}
                              type="checkbox"
                              className="kb-shelf-check"
                              checked={Boolean(selectedKeys[item.key])}
                              onChange={(event) => {
                                event.stopPropagation()
                                toggleSelect(item.key, event.target.checked)
                              }}
                              onClick={(event) => event.stopPropagation()}
                            />
                            <div className="kb-shelf-item-main">
                              <div className="kb-shelf-item-title" data-testid="citation-shelf-item-title">{shelfTitle}</div>
                              {display.authors ? (
                                <div className="kb-shelf-item-authors">{display.authors}</div>
                              ) : null}
                            </div>
                            <div className="kb-shelf-item-actions">
                              {item.sourcePath && onOpenSource ? (
                                <button
                                  type="button"
                                  className={`kb-shelf-source-open is-${itemSourceOpenQuality.tone}`}
                                  aria-label={S.locate_label}
                                  title={itemSourceOpenQuality.reason || S.locate_label}
                                  data-testid="citation-shelf-open-source"
                                  onClick={(event) => {
                                    event.stopPropagation()
                                    onOpenSource(item)
                                  }}
                                >
                                  <FileSearchOutlined />
                                </button>
                              ) : null}
                              <button
                                type="button"
                                className="kb-shelf-item-remove"
                                aria-label={S.shelf_remove_item}
                                onClick={(event) => {
                                  event.stopPropagation()
                                  onRemove(item.key)
                                }}
                              >
                                <CloseOutlined aria-hidden="true" />
                              </button>
                            </div>
                          </div>
                          {showQuality ? (
                            <div className="kb-shelf-quality">
                              <div className="kb-shelf-quality-chips">
                                {visibleQualityChips.map((chip) => (
                                  <span key={`${item.key}-q-${chip}`} className="kb-shelf-quality-chip">
                                    {chip}
                                  </span>
                                ))}
                              </div>
                              {quality.needsRepair ? (
                                <button
                                  type="button"
                                  className="kb-shelf-repair-btn"
                                  aria-live="polite"
                                  disabled={repairLoadingKeySet.has(item.key)}
                                  data-testid="citation-shelf-repair"
                                  onClick={(event) => {
                                    event.stopPropagation()
                                    onRepair(item)
                                  }}
                                >
                                  {repairLoadingKeySet.has(item.key) ? S.shelf_repairing : S.shelf_auto_repair}
                                </button>
                              ) : null}
                            </div>
                          ) : null}
                          {quality.tip && (isFocused || quality.needsRepair) ? (
                            <div className="kb-shelf-quality-tip">{quality.tip}</div>
                          ) : null}
                          <div className="kb-shelf-meta-row">
                            <div className="kb-shelf-meta-badges">
                              <span
                                className={`kb-shelf-kind is-${shelfKind}`}
                                title={shelfOriginText || undefined}
                              >
                                {shelfKindText}
                              </span>
                              {shelfOriginText ? (
                                <span className="kb-shelf-origin" title={shelfOriginText}>
                                  {shelfOriginText}
                                </span>
                              ) : null}
                              {visibleTraceLabels.map((label, idx) => (
                                <span key={`${item.key}-trace-${idx}-${label}`} className="kb-shelf-origin" title={trace.debugTitle || undefined}>
                                  {label}
                                </span>
                              ))}
                              {duplicateCount > 1 ? (
                                <span className="kb-shelf-dup">{S.shelf_dup.replace('{n}', String(duplicateCount))}</span>
                              ) : null}
                              {itemSourceOpenQuality.tone !== 'ready' ? (
                                <span
                                  className={`kb-shelf-source-open-quality is-${itemSourceOpenQuality.tone}`}
                                  data-testid="citation-shelf-source-open-quality"
                                  title={itemSourceOpenQuality.reason}
                                >
                                  {itemSourceOpenQuality.label}
                                </span>
                              ) : null}
                              {libraryMatch ? (
                                <span
                                  className={`kb-shelf-library-match is-${libraryMatch.tone}`}
                                  data-testid="citation-shelf-library-match"
                                  title={libraryMatch.title}
                                >
                                  {libraryMatch.label}
                                </span>
                              ) : null}
                              {itemTags.map((tag) => (
                                <span key={`${item.key}-tag-${tag}`} className="kb-shelf-tag">
                                  #{tag}
                                </span>
                              ))}
                            </div>
                          </div>
                          {(isFocused || noteText) ? (
                            <div
                              className={`kb-shelf-note ${noteEditing ? '' : 'kb-shelf-note-compact'}`}
                              onClick={(event) => event.stopPropagation()}
                            >
                              {noteEditing ? (
                                <>
                                  <div className="kb-shelf-note-head">{S.shelf_note_head}</div>
                                  <Input.TextArea
                                    className="kb-shelf-note-editor"
                                    autoSize={{ minRows: 2, maxRows: 4 }}
                                    maxLength={1200}
                                    placeholder={S.shelf_note_placeholder}
                                    value={item.note || ''}
                                    onChange={(event) => onUpdateNote(item.key, event.target.value)}
                                  />
                                  <div className="kb-shelf-note-actions">
                                    <button
                                      type="button"
                                      className="kb-shelf-note-link"
                                      onClick={() => setEditingNoteKeys((prev) => ({ ...prev, [item.key]: false }))}
                                    >
                                      {S.shelf_done}
                                    </button>
                                  </div>
                                </>
                              ) : (
                                <div className="kb-shelf-note-inline">
                                  {noteText ? (
                                    <div className="kb-shelf-note-preview">
                                      {noteText}
                                    </div>
                                  ) : null}
                                  {isFocused ? (
                                    <button
                                      type="button"
                                      className="kb-shelf-note-link"
                                      onClick={() => setEditingNoteKeys((prev) => ({ ...prev, [item.key]: true }))}
                                    >
                                      {noteText ? S.shelf_edit_note : S.shelf_add_note}
                                    </button>
                                  ) : null}
                                </div>
                              )}
                            </div>
                          ) : null}
                          {isFocused && shelfExcerpt ? (
                            <div className="kb-shelf-excerpt" data-testid="citation-shelf-excerpt">
                              <div className="kb-shelf-excerpt-head">
                                {shelfExcerptLabel || S.shelf_excerpt_head}
                              </div>
                              <div className="kb-shelf-excerpt-text">
                                {shelfExcerpt}
                              </div>
                            </div>
                          ) : null}
                          {visibleMetrics.length > 0 ? (
                            <div className="kb-shelf-metrics">
                              {visibleMetrics.map((metric) => (
                                <span key={metric} className="kb-shelf-metric">
                                  {metric}
                                </span>
                              ))}
                            </div>
                          ) : null}
                          {isFocused || !item.doiUrl ? (
                            <div className="kb-shelf-doi">
                              {item.doiUrl ? (
                                <a className="kb-shelf-doi-link" href={item.doiUrl} rel="noreferrer" target="_blank">
                                  {item.doi || item.doiUrl}
                                </a>
                              ) : (
                                <span className="kb-shelf-doi-empty">{S.shelf_no_doi_link}</span>
                              )}
                            </div>
                          ) : null}
                          {isFocused ? (
                            <div className="kb-shelf-summary" data-testid="citation-shelf-summary">
                              {summaryLoadingKey === item.key ? (
                                <div className="kb-shelf-summary-text">{S.shelf_summary_loading}</div>
                              ) : shelfSummaryLine ? (
                                <>
                                  <div className="kb-shelf-summary-meta">
                                    <span className="kb-shelf-summary-head">{S.shelf_summary_head}</span>
                                    <span className="kb-shelf-summary-sep" aria-hidden="true">·</span>
                                    <span className="kb-shelf-summary-source">{shelfSummarySource}</span>
                                    <span
                                      className={`kb-shelf-summary-quality is-${shelfSummaryQuality.tone}`}
                                      data-testid="citation-shelf-summary-quality"
                                    >
                                      {shelfSummaryQuality.label}
                                    </span>
                                  </div>
                                  {(() => {
                                    const lines = splitSummary(shelfSummaryLine)
                                    const expanded = Boolean(expandedSummaryKeys[item.key])
                                    const visibleLines = expanded ? lines : lines.slice(0, 2)
                                    const canExpand = lines.length > 2
                                    return (
                                      <>
                                        <ol className="kb-shelf-summary-list">
                                          {visibleLines.map((line) => (
                                            <li key={line} className="kb-shelf-summary-text">{line}</li>
                                          ))}
                                        </ol>
                                        {canExpand ? (
                                          <button
                                            type="button"
                                            className="kb-shelf-summary-toggle"
                                            onClick={(event) => {
                                              event.stopPropagation()
                                              setExpandedSummaryKeys((prev) => ({ ...prev, [item.key]: !expanded }))
                                            }}
                                          >
                                            {expanded ? S.shelf_collapse : S.shelf_expand.replace('{n}', String(lines.length - visibleLines.length))}
                                          </button>
                                        ) : null}
                                      </>
                                    )
                                  })()}
                                </>
                              ) : (
                                <div className="kb-shelf-summary-empty">{S.shelf_summary_empty}</div>
                              )}
                            </div>
                          ) : null}
                        </div>
                      )
                    })}
                  </div>
                ))}
                {visibleItems.length === 0 ? (
                  <div className="rounded-xl border border-dashed border-[var(--border)] px-3 py-4 text-xs text-black/45 dark:text-white/45">
                    {S.shelf_no_match}
                  </div>
                ) : null}
              </div>
            )}
          </div>
        </div>
      </aside>
    </>
  )
}
