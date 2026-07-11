/* eslint-disable react-hooks/set-state-in-effect */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Button, Input, Select, message } from 'antd'
import { BookOutlined, CloseOutlined, DeleteOutlined, DownOutlined, DownloadOutlined, FileSearchOutlined, LoadingOutlined, SaveOutlined, SearchOutlined, SlidersOutlined } from '@ant-design/icons'
import type { CiteShelfItem } from './citationState'
import type { ReaderLocateResult } from './reader/readerTypes'
import type { ShelfMetadataRepairImpact } from '../../api/references'
import {
  citationCardView,
  citationDisplay,
  citationFormats,
  cleanCitationDisplayText,
  isLikelyWeakCitationTitle,
  normalizeDoiLike,
  normalizeShelfItemKind,
  normalizeShelfTags,
  shelfItemKindLabel,
  shelfItemDoiExportValue,
  shelfItemHasConflictingVenueSignals,
  shelfItemNeedsMetadataRepair,
  shelfItemPaperIdentity,
  shelfOriginLabel,
} from './citationState'
import {
  GROUP_MODE_LABEL,
  SCOPE_FILTER_LABEL,
  TAG_PRESETS,
  basenameFromPath,
  citeImpactMetrics,
  citeVenueYearParts,
  externalMetadataNeedsVisibleReview,
  impactScore,
  metadataIssueChip,
  metadataQuality,
  metadataQualityNeedsRepair,
  metadataQualityReady,
  normalizeSourceIdentity,
  normalizeTitle,
  shelfSummaryDisplay,
  sourceListKey,
  sourceOpenQualityView,
  sourceQualityForItem,
  sourceQualityNeedsReview,
  uniqueCitationMetrics,
  type GroupMode,
  type ScopeFilter,
  type ShelfCardPresentation,
  type ShelfCardSurface,
  type ShelfExportKind,
  type ShelfExportOptions,
  type ShelfExportRequest,
  type ShelfExportScope,
} from './citeShelfDisplay'
import { useCiteShelfSourceQuality } from './useCiteShelfSourceQuality'
import { useCiteShelfMetadataRepair } from './useCiteShelfMetadataRepair'
import { shelfItemHasUsableLibraryFullText } from './citeShelfRuntime'
import { useT } from '../../i18n'
import { qualityDiagnosticsVisible } from '../../utils/qualityDiagnostics'

interface Props {
  open: boolean
  visible?: boolean
  presentation?: 'floating' | 'dock'
  items: CiteShelfItem[]
  activeConvId?: string | null
  activeSourcePath?: string
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
  activeContextKeys?: Record<string, boolean>
  onToggle: () => void
  onClear: () => void
  onSelect: (item: CiteShelfItem) => void
  onOpenSource?: (item: CiteShelfItem) => void
  onOpenDiscoverySource?: (item: CiteShelfItem) => void
  onOpenMessage?: (item: CiteShelfItem) => void
  onUseSelectedAsContext?: (items: CiteShelfItem[]) => void
  onRemove: (key: string) => void
  onUpdateTags: (key: string, tags: string[]) => void
  onUpdateNote: (key: string, note: string) => void
  onRepair: (item: CiteShelfItem, options?: { silent?: boolean }) => void
  onApplyRepairCandidates?: (updates: Array<{ key: string; metas: Array<Record<string, unknown>> }>) => boolean
  onSelectSnapshot: (id: string) => void
  onSaveSnapshot: () => void
  onLoadSnapshot: () => void
  onDeleteSnapshot: () => void
  onBackgroundActivityChange?: (busy: boolean) => void
}

export function CiteShelf({
  open,
  visible,
  presentation = 'floating',
  items,
  activeConvId,
  activeSourcePath,
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
  activeContextKeys = {},
  onToggle,
  onClear,
  onSelect,
  onOpenSource,
  onOpenDiscoverySource,
  onOpenMessage,
  onUseSelectedAsContext,
  onRemove,
  onUpdateTags,
  onUpdateNote,
  onRepair,
  onApplyRepairCandidates,
  onSelectSnapshot,
  onSaveSnapshot,
  onLoadSnapshot,
  onDeleteSnapshot,
  onBackgroundActivityChange,
}: Props) {
  const S = useT()
  const isDockPresentation = presentation === 'dock'
  const panelVisible = visible ?? open
  const [expandedSummaryKeys, setExpandedSummaryKeys] = useState<Record<string, boolean>>({})
  const [expandedDetailKeys, setExpandedDetailKeys] = useState<Record<string, boolean>>({})
  const [selectedKeys, setSelectedKeys] = useState<Record<string, boolean>>({})
  const [searchText, setSearchText] = useState('')
  const [sortKey, setSortKey] = useState<'recent' | 'cited' | 'year' | 'impact'>('recent')
  const [groupMode, setGroupMode] = useState<GroupMode>('none')
  const [scopeFilter, setScopeFilter] = useState<ScopeFilter>('all')
  const [tagFilter, setTagFilter] = useState<string>('all')
  const [advancedFiltersOpen, setAdvancedFiltersOpen] = useState(false)
  const [organizeOpen, setOrganizeOpen] = useState(false)
  const [batchOrganizeOpen, setBatchOrganizeOpen] = useState(false)
  const [preflightExportRequest, setPreflightExportRequest] = useState<ShelfExportRequest | null>(null)
  const [exportPanelOpen, setExportPanelOpen] = useState(false)
  const [exportScope, setExportScope] = useState<ShelfExportScope>('all')
  const [batchTagInput, setBatchTagInput] = useState('')
  const [editingNoteKeys, setEditingNoteKeys] = useState<Record<string, boolean>>({})
  const [copyState, setCopyState] = useState<'idle' | 'gbt' | 'bibtex' | 'error'>('idle')
  const [slowTaskVisible, setSlowTaskVisible] = useState(false)
  const copyStateTimerRef = useRef<number | null>(null)
  const autoSourceRepairKeysRef = useRef<Record<string, boolean>>({})
  const shelfPanelRef = useRef<HTMLElement | null>(null)
  const repairingKeySignature = useMemo(
    () => repairingKeys.map((key) => String(key || '').trim()).filter(Boolean).join('|'),
    [repairingKeys],
  )
  const repairLoadingKeySet = useMemo(
    () => new Set([repairLoadingKey, ...repairingKeys].map((key) => String(key || '').trim()).filter(Boolean)),
    [repairLoadingKey, repairingKeys],
  )
  const activeConversationKey = String(activeConvId || '').trim()
  const activeSourceKey = normalizeSourceIdentity(activeSourcePath)
  const showSourceQualityDiagnostics = qualityDiagnosticsVisible()
  const sourceQualitySources = useMemo(() => {
    if (!showSourceQualityDiagnostics) return []
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
  }, [items, showSourceQualityDiagnostics])

  const {
    sourceQualityByPath,
    sourceRepairingKey,
    repairSources,
  } = useCiteShelfSourceQuality({
    open,
    showDiagnostics: showSourceQualityDiagnostics,
    refreshToken: sourceQualityRefreshToken,
    sources: sourceQualitySources,
  })
  const {
    exportRepairingKind,
    exportRepairingKindRef,
    repairMetadataBeforeExport,
  } = useCiteShelfMetadataRepair(onApplyRepairCandidates)

  const shelfBackgroundBusy = Boolean(
    summaryLoadingKey
      || repairLoadingKey
      || repairingKeySignature
      || exportRepairingKind
      || sourceRepairingKey,
  )
  const shelfBackgroundTaskLabel = exportRepairingKind
    ? S.shelf_background_export
    : (repairLoadingKey || repairingKeySignature || sourceRepairingKey)
      ? S.shelf_background_metadata
      : S.shelf_background_summary

  useEffect(() => {
    if (!panelVisible || !shelfBackgroundBusy) {
      setSlowTaskVisible(false)
      return undefined
    }
    const timer = window.setTimeout(() => setSlowTaskVisible(true), 900)
    return () => window.clearTimeout(timer)
  }, [
    exportRepairingKind,
    panelVisible,
    repairLoadingKey,
    repairingKeySignature,
    shelfBackgroundBusy,
    sourceRepairingKey,
    summaryLoadingKey,
  ])

  useEffect(() => {
    onBackgroundActivityChange?.(shelfBackgroundBusy)
  }, [onBackgroundActivityChange, shelfBackgroundBusy])

  useEffect(() => () => {
    onBackgroundActivityChange?.(false)
  }, [onBackgroundActivityChange])

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

  const compactCardText = (value: string, limit = 220): string => {
    const text = cleanCitationDisplayText(value).replace(/\s+/g, ' ').trim()
    if (!text) return ''
    if (text.length <= limit) return text
    return `${text.slice(0, Math.max(0, limit - 1)).trimEnd()}...`
  }

  const headingTail = (value: string): string => {
    const parts = String(value || '')
      .split('/')
      .map((part) => cleanCitationDisplayText(part))
      .filter(Boolean)
    return parts[parts.length - 1] || ''
  }

  const readerSurfaceFromAnchor = (item: CiteShelfItem, shelfKind: string): ShelfCardSurface => {
    if (shelfKind === 'reference') return 'reference'
    if (shelfKind === 'excerpt') return 'excerpt'
    if (shelfKind !== 'reader_selection') return 'citation'
    const anchorKind = String(item.anchorKind || '').trim().toLowerCase()
    if (anchorKind === 'figure' || anchorKind === 'table' || anchorKind === 'equation') return anchorKind
    return 'selection'
  }

  const surfaceLabel = (surface: ShelfCardSurface, shelfKindText: string): string => {
    if (surface === 'figure') return S.cite_anchor_figure || S.locate_badge_fig || 'Figure'
    if (surface === 'table') return S.cite_anchor_table || 'Table'
    if (surface === 'equation') return S.cite_anchor_equation || S.locate_badge_eq || 'Equation'
    if (surface === 'selection') return S.shelf_type_reader_selection || shelfKindText
    if (surface === 'reference') return S.shelf_type_reference || shelfKindText
    if (surface === 'excerpt') return S.shelf_type_excerpt || shelfKindText
    return S.shelf_type_citation || shelfKindText
  }

  const readerBlockTitle = (
    surface: ShelfCardSurface,
    excerpt: string,
    item: CiteShelfItem,
    fallbackTitle: string,
    shelfKindText: string,
  ): string => {
    const label = surfaceLabel(surface, shelfKindText)
    if (surface === 'equation') {
      const heading = headingTail(item.headingPath || item.locationLabel || '')
      return heading ? `${label} / ${heading}` : label
    }
    const text = compactCardText(excerpt || item.evidenceQuote || item.raw || '', surface === 'table' ? 92 : 118)
    if (text) return text
    const heading = headingTail(item.headingPath || item.locationLabel || '')
    if (heading) return `${label} / ${heading}`
    return fallbackTitle || label
  }

  const shelfCardPresentation = (
    item: CiteShelfItem,
    opts: {
      cardView: ReturnType<typeof citationCardView>
      display: ReturnType<typeof citationDisplay>
      shelfKind: string
      shelfKindText: string
      rawSourceLabel: string
      itemLocationLabel: string
      shelfExcerpt: string
      shelfExcerptLabel: string
    },
  ): ShelfCardPresentation => {
    const surface = readerSurfaceFromAnchor(item, opts.shelfKind)
    const isReaderBlock = surface === 'figure' || surface === 'table' || surface === 'equation' || surface === 'selection'
    const fallbackTitle = cleanCitationDisplayText(String(opts.cardView.header.title || opts.display.main || item.main || item.title || '')).trim()
    const contractEvidence = item.cardView && !isReaderBlock
      ? opts.cardView.sections.find((section) => (
        ['evidence', 'reference', 'context_summary', 'support'].includes(String(section.id || '').trim().toLowerCase())
        && cleanCitationDisplayText(section.text)
      ))
      : null
    const title = isReaderBlock
      ? readerBlockTitle(surface, opts.shelfExcerpt, item, fallbackTitle, opts.shelfKindText)
      : fallbackTitle || opts.shelfKindText
    const sourceLabel = (() => {
      const source = opts.rawSourceLabel
      if (!source) return opts.itemLocationLabel
      if (isReaderBlock) {
        const locationTail = headingTail(opts.itemLocationLabel)
        return [source, locationTail && normalizeTitle(locationTail) !== normalizeTitle(source) ? locationTail : '']
          .filter(Boolean)
          .join(' / ')
      }
      return source && normalizeTitle(source) === normalizeTitle(title) && opts.itemLocationLabel
        ? opts.itemLocationLabel
        : source
    })()
    return {
      surface,
      title,
      sourceLabel,
      excerpt: cleanCitationDisplayText(contractEvidence?.text || '') || opts.shelfExcerpt,
      excerptLabel: cleanCitationDisplayText(contractEvidence?.label || '') || opts.shelfExcerptLabel,
      showAuthors: !isReaderBlock,
      showArticleSummary: !isReaderBlock,
      showExcerptInDetails: Boolean(contractEvidence?.text || opts.shelfExcerpt),
    }
  }

  const sourceTrailRows = (item: CiteShelfItem): Array<{ id: string; label: string; value: string; title?: string }> => {
    const rows: Array<{ id: string; label: string; value: string; title?: string }> = []
    const push = (id: string, label: string, rawValue: string, title = '') => {
      const value = cleanCitationDisplayText(rawValue || '')
      if (!value) return
      rows.push({ id, label, value, title: title || value })
    }
    const sourceName = String(item.sourceName || '').trim()
    const sourcePath = String(item.sourcePath || '').trim()
    const sourceLabel = sourceName || basenameFromPath(sourcePath)
    const pageStart = Number(item.pageStart || 0)
    const pageEnd = Number(item.pageEnd || 0)
    const pageLabel = Number.isFinite(pageStart) && pageStart > 0
      ? pageEnd > pageStart
        ? `p. ${pageStart}-${pageEnd}`
        : `p. ${pageStart}`
      : ''
    const rawLocation = String(item.locationLabel || '').trim()
      || [item.headingPath, pageLabel].map((part) => String(part || '').trim()).filter(Boolean).join(' / ')
    const normalizedSourceLabel = normalizeTitle(sourceLabel)
    const normalizedLocation = normalizeTitle(rawLocation)
    const location = normalizedSourceLabel && normalizedLocation.startsWith(normalizedSourceLabel)
      ? rawLocation.slice(sourceLabel.length).replace(/^(?:\s*[/\\]\s*)+/, '').trim()
      : rawLocation
    const refNum = Number(item.num || 0)
    const refLabel = Number.isFinite(refNum) && refNum > 0 ? S.shelf_ref_num.replace('{n}', String(refNum)) : ''
    const anchor = String(item.anchor || '').trim()
    const hasLibraryFullText = shelfItemHasUsableLibraryFullText(item)
    const libraryFullTextPath = hasLibraryFullText ? String(item.libraryMatchPath || '').trim() : ''
    const libraryFullTextLabel = String(item.libraryMatchTitle || '').trim() || basenameFromPath(libraryFullTextPath)

    push('fulltext', S.shelf_trace_full_text || 'Local full text', libraryFullTextLabel, libraryFullTextPath)
    push(
      'source',
      hasLibraryFullText ? S.shelf_trace_discovery_source || 'Discovered in' : S.shelf_trace_source,
      sourceLabel,
      sourcePath || sourceLabel,
    )
    push('location', S.shelf_trace_location, location)
    push('reference', S.shelf_trace_reference, refLabel, anchor ? S.shelf_anchor.replace('{anchor}', anchor) : refLabel)
    return rows
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
        label: S.shelf_library_full_text || S.shelf_library_in_library,
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

  const duplicateCountByIdentity = useMemo(() => {
    const counter: Record<string, number> = {}
    for (const item of items) {
      const key = shelfItemPaperIdentity(item)
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
      const isDuplicate = (duplicateCountByIdentity[shelfItemPaperIdentity(item)] || 0) > 1
      const summaryDisplay = shelfSummaryDisplay(item, citationCardView(item), S)
      const hasSummary = summaryDisplay.kind === 'article' && Boolean(summaryDisplay.line)
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
      if (scopeFilter === 'conversation' && activeConversationKey) {
        if (String(item.traceConvId || '').trim() !== activeConversationKey) return false
      }
      if (scopeFilter === 'paper' && activeSourceKey) {
        if (normalizeSourceIdentity(item.sourcePath) !== activeSourceKey) return false
      }
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
        item.sourcePath,
        item.headingPath,
        item.locationLabel,
        item.answerClaim,
        item.whyLine,
        item.supportRelation,
        item.note,
        item.shelfItemKind,
        shelfItemKindLabel(item.shelfItemKind, S),
        item.shelfOrigin,
        shelfOriginLabel(item.shelfOrigin, S),
        item.shelfExcerpt,
        item.citationContext,
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
  }, [S, activeConversationKey, activeSourceKey, items, scopeFilter, searchText, sortKey, tagFilter])

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
  const selectedReviewSources = useMemo(() => {
    if (!showSourceQualityDiagnostics) return []
    const seen = new Set<string>()
    const out: Array<{ source_path: string; source_name: string }> = []
    for (const item of selectedItems) {
      const sourcePath = String(item.sourcePath || '').trim()
      if (!sourcePath || seen.has(sourcePath)) continue
      const sourceQuality = sourceQualityForItem(item, sourceQualityByPath)
      const locateView = sourceOpenQualityView(
        item,
        sourceQuality,
        S,
        readerLocateResults[item.key],
      )
      if (!sourceQualityNeedsReview(sourceQuality) && !locateView.repairable) continue
      seen.add(sourcePath)
      out.push({
        source_path: sourcePath,
        source_name: String(item.sourceName || item.title || item.main || '').trim(),
      })
    }
    return out
  }, [S, readerLocateResults, selectedItems, showSourceQualityDiagnostics, sourceQualityByPath])
  const selectedReviewSourceKey = useMemo(
    () => sourceListKey(selectedReviewSources),
    [selectedReviewSources],
  )
  const locateRepairSources = useMemo(() => {
    if (!showSourceQualityDiagnostics) return []
    const seen = new Set<string>()
    const out: Array<{ source_path: string; source_name: string }> = []
    for (const item of items) {
      const sourcePath = String(item.sourcePath || '').trim()
      if (!sourcePath || seen.has(sourcePath)) continue
      const sourceQuality = sourceQualityForItem(item, sourceQualityByPath)
      const locateView = sourceOpenQualityView(
        item,
        sourceQuality,
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
  }, [S, items, readerLocateResults, showSourceQualityDiagnostics, sourceQualityByPath])
  const locateRepairSourceKey = useMemo(
    () => sourceListKey(locateRepairSources),
    [locateRepairSources],
  )
  const visibleKeySet = useMemo(() => new Set(visibleItems.map((item) => item.key)), [visibleItems])
  const visibleSelectedCount = useMemo(
    () => visibleItems.reduce((acc, item) => acc + (selectedKeys[item.key] ? 1 : 0), 0),
    [selectedKeys, visibleItems],
  )
  const advancedFilterActive = (groupMode !== 'none') || (tagFilter !== 'all') || (scopeFilter !== 'all')
  const defaultExportScope: ShelfExportScope = selectedCount > 0
    ? 'selected'
    : (advancedFilterActive || searchText.trim()) && visibleItems.length > 0
      ? 'visible'
      : 'all'
  const activeExportScope: ShelfExportScope = exportScope === 'selected' && selectedCount <= 0
    ? defaultExportScope
    : exportScope
  const exportItemsByScope = useCallback((scope: ShelfExportScope): CiteShelfItem[] => {
    if (scope === 'selected') return selectedItems
    if (scope === 'visible') return visibleItems
    return items
  }, [items, selectedItems, visibleItems])
  const exportTargetItems = useMemo(
    () => exportItemsByScope(activeExportScope),
    [activeExportScope, exportItemsByScope],
  )
  const exportTargetCount = exportTargetItems.length
  const exportScopeLabel = (scope: ShelfExportScope): string => {
    if (scope === 'selected') return S.shelf_export_scope_selected.replace('{n}', String(selectedCount))
    if (scope === 'visible') return S.shelf_export_scope_visible.replace('{n}', String(visibleItems.length))
    return S.shelf_export_scope_all.replace('{n}', String(items.length))
  }
  const preflightExportItems = preflightExportRequest ? exportItemsByScope(preflightExportRequest.scope) : []
  const preflightMetadataReviewItems = preflightExportItems.filter((item) => shelfItemNeedsMetadataRepair(item, citationDisplay(item)))
  const preflightMetadataReviewCount = preflightMetadataReviewItems.length
  const organizeReviewCount = shelfReadiness.metadataReview
    + shelfReadiness.sourceOpenReview
  const organizeStatusLabel = slowTaskVisible
    ? (S.shelf_organize_processing || 'Updating')
    : organizeReviewCount > 0
      ? (S.shelf_organize_review || '{n} to review').replace('{n}', String(organizeReviewCount))
      : (S.shelf_organize_ready || 'Ready')
  const snapshotOptions = useMemo(
    () => snapshots.map((item) => ({
      value: item.id,
      label: item.name,
    })),
    [snapshots],
  )
  const hasSnapshotChoices = snapshotOptions.length > 0 || Boolean(selectedSnapshotId)
  const showSnapshotTools = hasSnapshotChoices || Boolean(snapshotDiff)
  const showOrganizeToggle = items.length > 0 || showSnapshotTools || slowTaskVisible
  const showOrganizeDetails = showSnapshotTools || slowTaskVisible || shelfReadiness.status !== 'ready' || Boolean(repairImpact)

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

  const exportFormatLabel = (kind: ShelfExportKind | 'gbt' | 'bibtex'): string => {
    if (kind === 'gbt') return 'GB/T'
    if (kind === 'bibtex' || kind === 'bib') return 'BibTeX'
    if (kind === 'ris') return 'RIS'
    if (kind === 'md') return 'Markdown'
    return 'CSV'
  }

  const markdownForExportItems = (exportItems: CiteShelfItem[]): string => exportItems.map((item, index) => {
    const card = citationCardView(item)
    const title = cleanCitationDisplayText(card.header.title || item.title || item.main || `${S.shelf_export_reference_fallback} ${index + 1}`)
    const authors = cleanCitationDisplayText(item.authors || '')
    const year = cleanCitationDisplayText(item.year || '')
    const venue = cleanCitationDisplayText(item.venue || '')
    const doi = shelfItemDoiExportValue(item)
    const source = cleanCitationDisplayText(item.sourceName || item.sourcePath || '')
    const summaryDisplay = shelfSummaryDisplay(item, card, S)
    const summary = summaryDisplay.kind === 'article'
      ? cleanCitationDisplayText(summaryDisplay.line)
      : ''
    const excerpt = cleanCitationDisplayText(item.shelfExcerpt || item.evidenceQuote || item.cardEvidence || '')
    const note = cleanCitationDisplayText(item.note || '')
    const tags = normalizeShelfTags(item.tags)
    const lines = [
      `## ${index + 1}. ${title}`,
      '',
      citationFormats(item).gbt,
    ]
    const meta = [
      authors ? `${S.shelf_export_md_authors}: ${authors}` : '',
      year ? `${S.shelf_export_md_year}: ${year}` : '',
      venue ? `${S.shelf_export_md_venue}: ${venue}` : '',
      doi ? `DOI: ${doi}` : '',
      source ? `${S.shelf_export_md_source}: ${source}` : '',
      tags.length > 0 ? `${S.shelf_export_md_tags}: ${tags.join(', ')}` : '',
    ].filter(Boolean)
    if (meta.length > 0) {
      lines.push('', ...meta)
    }
    if (summary) {
      lines.push('', `### ${S.shelf_summary_head}`, summary)
    }
    if (excerpt) {
      lines.push('', `### ${S.shelf_export_md_excerpt}`, `> ${excerpt.replace(/\n+/g, '\n> ')}`)
    }
    if (note) {
      lines.push('', `### ${S.shelf_note_head}`, note)
    }
    return lines.join('\n').trim()
  }).join('\n\n')

  const exportKindForClipboardRepair = (kind: 'gbt' | 'bibtex' | 'md'): ShelfExportKind => {
    if (kind === 'bibtex') return 'bib'
    if (kind === 'md') return 'md'
    return 'csv'
  }

  const copyShelfItemsAs = async (scope: ShelfExportScope, kind: 'gbt' | 'bibtex' | 'md') => {
    if (exportRepairingKindRef.current) return
    let copyItems = exportItemsByScope(scope)
    if (copyItems.length <= 0) {
      message.warning(S.shelf_export_no_items)
      return
    }
    const reviewItems = copyItems.filter((item) => shelfItemNeedsMetadataRepair(item, citationDisplay(item)))
    if (reviewItems.length > 0) {
      const repairedItems = await repairMetadataBeforeExport(exportKindForClipboardRepair(kind), copyItems)
      if (!repairedItems) return
      copyItems = repairedItems.filter((item) => !shelfItemNeedsMetadataRepair(item, citationDisplay(item)))
      if (copyItems.length <= 0) {
        if (scope === 'selected' && (kind === 'gbt' || kind === 'bibtex')) setTransientCopyState('error')
        message.warning(S.shelf_export_preflight_no_healthy)
        return
      }
    }
    const text = kind === 'md'
      ? markdownForExportItems(copyItems)
      : copyItems.map((item) => citationFormats(item)[kind]).join('\n\n')
    try {
      await writeClipboard(text)
      if (scope === 'selected' && (kind === 'gbt' || kind === 'bibtex')) setTransientCopyState(kind)
      message.success(
        S.shelf_export_copied
          .replace('{format}', exportFormatLabel(kind))
          .replace('{n}', String(copyItems.length)),
      )
    } catch {
      if (scope === 'selected' && (kind === 'gbt' || kind === 'bibtex')) setTransientCopyState('error')
      message.error(S.shelf_copy_error)
    }
  }

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

  const exportShelfItemsAs = async (scope: ShelfExportScope, kind: ShelfExportKind, options: ShelfExportOptions = {}) => {
    const targetItems = exportItemsByScope(scope)
    if (targetItems.length <= 0) {
      message.warning(S.shelf_export_no_items)
      return
    }
    if (exportRepairingKindRef.current) return
    const reviewItems = targetItems.filter((item) => shelfItemNeedsMetadataRepair(item, citationDisplay(item)))
    const reviewKeySet = new Set(reviewItems.map((item) => item.key))
    if (reviewItems.length > 0 && !options.skipPreflight) {
      setPreflightExportRequest({ kind, scope })
      setExportPanelOpen(true)
      return
    }
    let exportItems = options.onlyMetadataReady
      ? targetItems.filter((item) => !reviewKeySet.has(item.key))
      : targetItems
    if (exportItems.length <= 0) {
      message.warning(S.shelf_export_preflight_no_healthy)
      return
    }
    if (options.autoRepair) {
      const repairedExportItems = await repairMetadataBeforeExport(kind, exportItems)
      if (!repairedExportItems) return
      exportItems = repairedExportItems
    }
    try {
      const base = `cite_shelf_${scope}_${nowStamp()}`
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
      if (kind === 'md') {
        const md = markdownForExportItems(exportItems).trim()
        if (!md) return
        downloadTextFile(`${base}.md`, md, 'text/markdown;charset=utf-8')
        message.success(S.shelf_export_markdown.replace('{n}', String(exportItems.length)))
        return
      }
      const includeSourceQualityColumns = showSourceQualityDiagnostics
      const headers = [
        'title',
        'authors',
        'year',
        'venue',
        'doi',
        'source',
        ...(includeSourceQualityColumns ? [
          'source_quality_status',
          'source_quality_issues',
        ] : []),
        'source_open_status',
        'source_open_precision',
        'source_open_reason',
        'source_origin',
        'source_kind',
        'source_path',
        'trace_conversation_id',
        'trace_assistant_message_id',
        'trace_assistant_order',
        'trace_user_message_id',
        'heading_path',
        'location_label',
        'page_start',
        'page_end',
        'source_anchor',
        'shelf_excerpt',
        'answer_claim',
        'why_collected',
        'note',
        'tags',
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
      const rows = exportItems.map((item) => {
        const sourceQuality = sourceQualityForItem(item, sourceQualityByPath)
        const sourceOpen = sourceOpenQualityView(item, sourceQuality, S, readerLocateResults[item.key])
        const summaryDisplay = shelfSummaryDisplay(item, citationCardView(item), S)
        const summaryQuality = summaryDisplay.quality
        const hasArticleSummary = summaryDisplay.kind === 'article' && Boolean(summaryDisplay.line)
        return [
          citationCardView(item).header.title || item.title || item.main,
          item.authors,
          item.year,
          item.venue,
          shelfItemDoiExportValue(item),
          item.sourceName || item.sourcePath,
          ...(includeSourceQualityColumns ? [
            sourceQuality?.status || '',
            (sourceQuality?.issues || []).map((issue) => issue.label || issue.code).filter(Boolean).join('; '),
          ] : []),
          sourceOpen.status,
          sourceOpen.precision,
          sourceOpen.reason,
          item.shelfOrigin,
          item.shelfItemKind,
          item.sourcePath,
          item.traceConvId,
          item.traceAssistantMsgId,
          item.traceAssistantOrder,
          item.traceUserMsgId,
          item.headingPath,
          item.locationLabel,
          item.pageStart || '',
          item.pageEnd || '',
          item.anchor,
          cleanCitationDisplayText(item.shelfExcerpt || ''),
          cleanCitationDisplayText(item.answerClaim || ''),
          cleanCitationDisplayText(item.whyLine || item.supportRelation || item.upstreamWorkRole || ''),
          item.note || '',
          normalizeShelfTags(item.tags).join('; '),
          item.libraryMatchStatus,
          item.libraryMatchMethod,
          item.libraryMatchPath,
          item.num || '',
          item.citationCount || 0,
          item.journalIf,
          item.journalQuartile,
          item.conferenceTier,
          item.conferenceCcf,
          hasArticleSummary ? item.summarySource : '',
          hasArticleSummary ? item.summaryProvider : '',
          hasArticleSummary ? summaryQuality.status : 'missing',
          hasArticleSummary ? summaryQuality.score : 0,
          hasArticleSummary ? summaryDisplay.line : '',
        ].map((field) => csvEscape(field)).join(',')
      })
      const csv = `${headers.join(',')}\n${rows.join('\n')}`
      downloadTextFile(`${base}.csv`, csv, 'text/csv;charset=utf-8')
      message.success(S.shelf_export_csv.replace('{n}', String(exportItems.length)))
    } catch {
      message.error(S.shelf_export_failed)
    }
  }

  const exportActiveScopeAs = async (kind: ShelfExportKind, options: ShelfExportOptions = {}) => {
    await exportShelfItemsAs(activeExportScope, kind, options)
  }

  const openSelectedExportPanel = () => {
    setExportScope('selected')
    setExportPanelOpen(true)
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

  const collapseItemDetails = useCallback(() => {
    setExpandedDetailKeys({})
    setExpandedSummaryKeys({})
    setEditingNoteKeys({})
  }, [])

  const toggleItemDetails = useCallback((item: CiteShelfItem) => {
    onSelect(item)
    setExpandedSummaryKeys({})
    setExpandedDetailKeys((prev) => (prev[item.key] ? {} : { [item.key]: true }))
  }, [onSelect])

  useEffect(() => {
    const hasExpanded = Object.values(expandedDetailKeys).some(Boolean)
    if (!hasExpanded) return undefined

    const onDocumentClick = (event: MouseEvent) => {
      const target = event.target
      if (!(target instanceof Element)) return
      if (target.closest('.kb-shelf-item')) return

      const panel = shelfPanelRef.current
      if (panel?.contains(target)) {
        const interactive = target.closest('button, input, textarea, select, a, .ant-select, .ant-dropdown, .ant-picker-dropdown, [role="button"], [role="listbox"], [role="menu"]')
        if (interactive) return
      }

      collapseItemDetails()
    }

    document.addEventListener('click', onDocumentClick)
    return () => document.removeEventListener('click', onDocumentClick)
  }, [collapseItemDetails, expandedDetailKeys])

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
    setExpandedDetailKeys((prev) => {
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
    if (scopeFilter === 'conversation' && !activeConversationKey) setScopeFilter('all')
    if (scopeFilter === 'paper' && !activeSourceKey) setScopeFilter('all')
  }, [activeConversationKey, activeSourceKey, scopeFilter])

  useEffect(() => {
    if (selectedCount <= 0) setBatchOrganizeOpen(false)
  }, [selectedCount])

  useEffect(() => {
    if (!preflightExportRequest) return
    const reviewCount = exportItemsByScope(preflightExportRequest.scope)
      .filter((item) => shelfItemNeedsMetadataRepair(item, citationDisplay(item)))
      .length
    if (reviewCount > 0) return
    setPreflightExportRequest(null)
  }, [exportItemsByScope, preflightExportRequest])

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
          <BookOutlined className="kb-shelf-toggle-icon" aria-hidden="true" />
          <span className="kb-shelf-toggle-label">{S.shelf_title}</span>
        </button>
      ) : null}
      <aside
        ref={shelfPanelRef}
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
            {showOrganizeToggle || items.length > 0 ? (
              <div className="kb-shelf-status-actions">
                {showOrganizeToggle ? (
                  <button
                    type="button"
                    className={`kb-shelf-organize-toggle ${organizeOpen ? 'is-open' : ''} ${organizeReviewCount > 0 || slowTaskVisible ? 'is-review' : ''}`}
                    onClick={() => {
                      setOrganizeOpen((prev) => {
                        const next = !prev
                        if (next) setExportPanelOpen(false)
                        return next
                      })
                    }}
                    aria-expanded={organizeOpen}
                    aria-label={`${S.shelf_organize_toggle || 'Status'}: ${organizeStatusLabel}`}
                    data-testid="citation-shelf-organize-toggle"
                  >
                    <span className="kb-shelf-organize-status">{organizeStatusLabel}</span>
                    {showOrganizeDetails ? <DownOutlined className="kb-shelf-organize-chevron" aria-hidden="true" /> : null}
                  </button>
                ) : null}
                {items.length > 0 ? (
                  <div className="kb-shelf-primary-actions">
                    <button
                      type="button"
                      className="kb-shelf-command"
                      onClick={() => {
                        onSaveSnapshot()
                        setOrganizeOpen(true)
                        setExportPanelOpen(false)
                      }}
                      disabled={items.length === 0}
                      aria-label={S.shelf_save_snapshot}
                      title={S.shelf_save_snapshot_tip || S.shelf_save_snapshot}
                      data-testid="citation-shelf-save-snapshot"
                    >
                      <SaveOutlined />
                      {S.shelf_save_snapshot}
                    </button>
                    <span className="kb-shelf-command-divider" aria-hidden="true" />
                    <button
                      type="button"
                      className={`kb-shelf-command ${exportPanelOpen ? 'is-active' : ''}`}
                      onClick={() => {
                        setExportPanelOpen((prev) => {
                          const next = !prev
                          if (next) setOrganizeOpen(false)
                          return next
                        })
                      }}
                      disabled={items.length === 0}
                      aria-expanded={exportPanelOpen}
                      aria-label={S.shelf_export_toggle}
                      title={S.shelf_export_toggle}
                      data-testid="citation-shelf-export-toggle"
                    >
                      <DownloadOutlined />
                      {S.shelf_export_toggle}
                    </button>
                  </div>
                ) : null}
              </div>
            ) : null}
            {exportPanelOpen && items.length > 0 ? (
              <div className="kb-shelf-export-panel" data-testid="citation-shelf-export-panel">
                <div className="kb-shelf-export-scope-row">
                  <span>{S.shelf_export_scope}</span>
                  <div className="kb-shelf-export-scope-segments">
                    {selectedCount > 0 ? (
                      <button
                        type="button"
                        className={`kb-shelf-export-scope ${activeExportScope === 'selected' ? 'is-active' : ''}`}
                        onClick={() => setExportScope('selected')}
                        data-testid="citation-shelf-export-scope-selected"
                      >
                        {exportScopeLabel('selected')}
                      </button>
                    ) : null}
                    <button
                      type="button"
                      className={`kb-shelf-export-scope ${activeExportScope === 'visible' ? 'is-active' : ''}`}
                      onClick={() => setExportScope('visible')}
                      disabled={visibleItems.length <= 0}
                      data-testid="citation-shelf-export-scope-visible"
                    >
                      {exportScopeLabel('visible')}
                    </button>
                    <button
                      type="button"
                      className={`kb-shelf-export-scope ${activeExportScope === 'all' ? 'is-active' : ''}`}
                      onClick={() => setExportScope('all')}
                      data-testid="citation-shelf-export-scope-all"
                    >
                      {exportScopeLabel('all')}
                    </button>
                  </div>
                </div>
                <div className="kb-shelf-export-command-row">
                  <span>{S.shelf_export_download_label || 'Download'}</span>
                  <button
                    type="button"
                    className="kb-shelf-export-command"
                    onClick={() => void exportActiveScopeAs('bib')}
                    disabled={exportTargetCount <= 0 || Boolean(exportRepairingKind && exportRepairingKind !== 'bib')}
                    data-testid="citation-shelf-export-main-bib"
                  >
                    {S.shelf_export_bib_btn}
                  </button>
                  <button
                    type="button"
                    className="kb-shelf-export-command"
                    onClick={() => void exportActiveScopeAs('ris')}
                    disabled={exportTargetCount <= 0 || Boolean(exportRepairingKind && exportRepairingKind !== 'ris')}
                    data-testid="citation-shelf-export-main-ris"
                  >
                    {S.shelf_export_ris_btn}
                  </button>
                  <button
                    type="button"
                    className="kb-shelf-export-command"
                    onClick={() => void exportActiveScopeAs('md')}
                    disabled={exportTargetCount <= 0 || Boolean(exportRepairingKind && exportRepairingKind !== 'md')}
                    data-testid="citation-shelf-export-main-md"
                  >
                    {S.shelf_export_markdown_btn}
                  </button>
                </div>
                <div className="kb-shelf-export-command-row">
                  <span>{S.shelf_export_copy_label || 'Copy'}</span>
                  <button type="button" className="kb-shelf-export-command" onClick={() => void copyShelfItemsAs(activeExportScope, 'bibtex')} data-testid="citation-shelf-export-copy-bibtex">
                    {S.shelf_export_copy_bibtex}
                  </button>
                  <button type="button" className="kb-shelf-export-command" onClick={() => void copyShelfItemsAs(activeExportScope, 'gbt')} data-testid="citation-shelf-export-copy-gbt">
                    {S.shelf_export_copy_gbt}
                  </button>
                  <button type="button" className="kb-shelf-export-command" onClick={() => void copyShelfItemsAs(activeExportScope, 'md')} data-testid="citation-shelf-export-copy-md">
                    {S.shelf_export_copy_markdown}
                  </button>
                  <button type="button" className="kb-shelf-export-command" onClick={() => void exportActiveScopeAs('csv')} disabled={Boolean(exportRepairingKind)} data-testid="citation-shelf-export-main-csv">
                    {S.shelf_export_csv_btn}
                  </button>
                </div>
              </div>
            ) : null}
            {preflightExportRequest && preflightMetadataReviewCount > 0 ? (
              <div className="kb-shelf-export-preflight" data-testid="citation-shelf-export-preflight">
                <div className="kb-shelf-export-preflight-copy">
                  <strong>{S.shelf_export_preflight_title}</strong>
                  <span>{S.shelf_export_preflight_body.replace('{n}', String(preflightMetadataReviewCount))}</span>
                </div>
                <div className="kb-shelf-export-preflight-actions">
                  <Button
                    size="small"
                    onClick={() => {
                      const request = preflightExportRequest
                      void exportShelfItemsAs(request.scope, request.kind, { skipPreflight: true, onlyMetadataReady: true })
                      setPreflightExportRequest((current) => (current === request ? null : current))
                    }}
                    disabled={Boolean(exportRepairingKind)}
                    data-testid="citation-shelf-export-preflight-healthy"
                  >
                    {S.shelf_export_preflight_healthy}
                  </Button>
                  <Button
                    size="small"
                    onClick={async () => {
                      const request = preflightExportRequest
                      await exportShelfItemsAs(request.scope, request.kind, { skipPreflight: true, autoRepair: true })
                      setPreflightExportRequest((current) => (current === request ? null : current))
                    }}
                    loading={Boolean(exportRepairingKind)}
                    data-testid="citation-shelf-export-preflight-continue"
                  >
                    {S.shelf_export_preflight_autofill}
                  </Button>
                </div>
              </div>
            ) : null}
            {organizeOpen && showSnapshotTools ? (
              <div className="kb-shelf-version-panel" data-testid="citation-shelf-version-panel">
                <div className="kb-shelf-version-row">
                  <span className="kb-shelf-version-label">{S.shelf_version_title}</span>
                  {hasSnapshotChoices ? (
                    <div
                      className="kb-shelf-snapshot-row"
                      onClick={(event) => event.stopPropagation()}
                    >
                      <Select
                        size="small"
                        value={selectedSnapshotId || undefined}
                        placeholder={snapshotOptions.length > 0 ? S.shelf_select_snapshot : S.shelf_no_snapshot}
                        className="kb-shelf-snapshot-select"
                        data-testid="citation-shelf-snapshot-select"
                        options={snapshotOptions}
                        onChange={(value) => onSelectSnapshot(String(value || ''))}
                      />
                      <button type="button" className="kb-shelf-version-command" onClick={onLoadSnapshot} disabled={!selectedSnapshotId} data-testid="citation-shelf-load-snapshot">
                        {S.shelf_load}
                      </button>
                      <button type="button" className="kb-shelf-version-command" onClick={onDeleteSnapshot} disabled={!selectedSnapshotId} data-testid="citation-shelf-delete-snapshot">
                        {S.shelf_delete}
                      </button>
                    </div>
                  ) : null}
                </div>
                <div className="kb-shelf-version-hint">
                  {snapshotDiff || S.shelf_version_hint}
                </div>
              </div>
            ) : null}
            {organizeOpen && slowTaskVisible ? (
              <div className="kb-shelf-background-task" data-testid="citation-shelf-background-task">
                <LoadingOutlined spin />
                <span>{shelfBackgroundTaskLabel}</span>
              </div>
            ) : null}
            {organizeOpen && items.length > 0 ? (
              <div
                className={`kb-shelf-readiness is-${shelfReadiness.status} ${showOrganizeDetails ? '' : 'is-quiet'}`}
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
                  {shelfReadiness.metadataReview > 0 ? (
                    <span className="kb-shelf-readiness-chip is-review">
                      {S.shelf_readiness_meta.replace('{n}', String(shelfReadiness.metadataReview))}
                    </span>
                  ) : null}
                  {shelfReadiness.sourceOpenReview > 0 ? (
                    <span className="kb-shelf-readiness-chip is-review">
                      {S.shelf_readiness_source_open_review.replace('{n}', String(shelfReadiness.sourceOpenReview))}
                    </span>
                  ) : null}
                  {shelfReadiness.duplicateItems > 0 ? (
                    <span className="kb-shelf-readiness-chip is-review">
                      {S.shelf_readiness_dups.replace('{n}', String(shelfReadiness.duplicateItems))}
                    </span>
                  ) : null}
                </div>
                {repairImpact ? (
                  <div className="kb-shelf-repair-impact" data-testid="citation-shelf-repair-impact">
                    <span>{S.shelf_repair_impact_changed.replace('{n}', String(repairImpact.changed))}</span>
                  </div>
                ) : null}
              </div>
            ) : null}
            {selectedCount > 0 ? (
              <div className="kb-shelf-batch-row">
                <div className="kb-shelf-batch-head">
                  <span className="kb-shelf-batch-count" data-testid="citation-shelf-batch-count">{S.shelf_batch_count.replace('{n}', String(selectedCount))}</span>
                  <div className="kb-shelf-batch-head-actions">
                    <button
                      type="button"
                      className={`kb-shelf-batch-command ${batchOrganizeOpen ? 'is-primary' : ''}`}
                      onClick={() => setBatchOrganizeOpen((prev) => !prev)}
                      aria-expanded={batchOrganizeOpen}
                      data-testid="citation-shelf-batch-organize"
                    >
                      {S.shelf_batch_organize || 'Organize'}
                    </button>
                    <button type="button" className="kb-shelf-clear-select" onClick={clearSelected} data-testid="citation-shelf-clear-selection">
                      {S.shelf_clear_selection}
                    </button>
                  </div>
                </div>
                <div className="kb-shelf-batch-actions">
                  {onUseSelectedAsContext ? (
                    <button
                      type="button"
                      className="kb-shelf-batch-command is-primary"
                      onClick={() => onUseSelectedAsContext(selectedItems)}
                      data-testid="citation-shelf-use-context"
                    >
                      {S.shelf_use_as_context || 'Use as context'}
                    </button>
                  ) : null}
                  <button
                    type="button"
                    className="kb-shelf-batch-command"
                    onClick={openSelectedExportPanel}
                    aria-expanded={exportPanelOpen && activeExportScope === 'selected'}
                    data-testid="citation-shelf-export-selected"
                  >
                    <DownloadOutlined aria-hidden="true" />
                    {S.shelf_export_toggle}
                  </button>
                </div>
                {batchOrganizeOpen ? (
                  <div className="kb-shelf-batch-organize-panel" data-testid="citation-shelf-batch-organize-panel">
                    <div className="kb-shelf-batch-tag" onClick={(event) => event.stopPropagation()}>
                      <span className="kb-shelf-batch-label">{S.shelf_batch_tag_label || 'Tags'}</span>
                      <div className="kb-shelf-batch-tag-controls">
                        <Select
                          size="small"
                          value={batchTagInput || undefined}
                          placeholder={S.shelf_batch_tag_placeholder}
                          showSearch
                          onChange={(value) => {
                            setBatchTagInput(value)
                            applyTagToSelected(value)
                          }}
                          options={[...TAG_PRESETS, ...allTags]
                            .filter((tag, idx, arr) => arr.findIndex((x) => x.toLowerCase() === tag.toLowerCase()) === idx)
                            .map((tag) => ({ value: tag, label: tag }))}
                        />
                        <button
                          type="button"
                          className="kb-shelf-batch-command"
                          onClick={() => {
                            if (!batchTagInput.trim()) return
                            removeTagFromSelected(batchTagInput)
                            setBatchTagInput('')
                          }}
                        >
                          {S.shelf_remove_tag}
                        </button>
                      </div>
                    </div>
                    <button type="button" className="kb-shelf-batch-command is-danger" onClick={removeSelected} data-testid="citation-shelf-batch-remove">
                      {S.shelf_batch_remove}
                    </button>
                  </div>
                ) : null}
              </div>
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
                      prefix={<SearchOutlined aria-hidden="true" />}
                      value={searchText}
                      onChange={(event) => setSearchText(event.target.value)}
                      className="kb-shelf-search"
                      data-testid="citation-shelf-search"
                    />
                    <div className="kb-shelf-toolbar-controls">
                      <Select
                        value={sortKey}
                        onChange={(value) => setSortKey(value)}
                        className="kb-shelf-sort"
                        popupMatchSelectWidth={false}
                        suffixIcon={<DownOutlined />}
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
                        aria-expanded={advancedFiltersOpen}
                      >
                        <SlidersOutlined aria-hidden="true" />
                        <span>{advancedFiltersOpen ? S.shelf_advanced_collapse : S.shelf_advanced_filter}</span>
                        {advancedFilterActive ? <strong aria-hidden="true" /> : null}
                      </button>
                    </div>
                  </div>
                  {advancedFiltersOpen ? (
                    <div className="kb-shelf-filters" data-testid="citation-shelf-advanced-panel">
                      <div className="kb-shelf-filter-row">
                        <span className="kb-shelf-filter-label">{S.shelf_filter_group_label || 'Group'}</span>
                        <div className="kb-shelf-filter-segments" role="group" aria-label={S.shelf_filter_group_label || 'Group'}>
                          {([
                            ['none', S.shelf_group_short_none || S.shelf_group_none],
                            ['tag', S.shelf_group_short_tag || S.shelf_group_tag],
                            ['source', S.shelf_group_short_source || S.shelf_group_source],
                            ['kind', S.shelf_group_short_type || S.shelf_group_type],
                          ] as Array<[GroupMode, string]>).map(([value, label]) => (
                            <button
                              key={value}
                              type="button"
                              className={`kb-shelf-segment ${groupMode === value ? 'is-active' : ''}`}
                              aria-pressed={groupMode === value}
                              onClick={() => setGroupMode(value)}
                            >
                              {label}
                            </button>
                          ))}
                        </div>
                      </div>
                      <div className="kb-shelf-filter-row">
                        <span className="kb-shelf-filter-label">{S.shelf_filter_scope_label || 'Scope'}</span>
                        <div
                          className="kb-shelf-filter-segments"
                          role="group"
                          aria-label={S.shelf_filter_scope_label || 'Scope'}
                          data-testid="citation-shelf-scope-filter"
                        >
                          {([
                            ['all', S.shelf_scope_short_all || S.shelf_scope_all_project, false],
                            ['conversation', S.shelf_scope_short_conversation || S.shelf_scope_current_conversation, !activeConversationKey],
                            ['paper', S.shelf_scope_short_paper || S.shelf_scope_current_paper, !activeSourceKey],
                          ] as Array<[ScopeFilter, string, boolean]>).map(([value, label, disabled]) => (
                            <button
                              key={value}
                              type="button"
                              className={`kb-shelf-segment ${scopeFilter === value ? 'is-active' : ''}`}
                              aria-pressed={scopeFilter === value}
                              disabled={disabled}
                              data-testid={`citation-shelf-scope-${value}`}
                              onClick={() => setScopeFilter(value)}
                            >
                              {label}
                            </button>
                          ))}
                        </div>
                      </div>
                      <div className="kb-shelf-filter-row is-tag">
                        <span className="kb-shelf-filter-label">{S.shelf_filter_tag_label || 'Tag'}</span>
                        <div className="kb-shelf-filter-tag-line">
                          <Select
                            allowClear
                            value={tagFilter === 'all' ? undefined : tagFilter}
                            onChange={(value) => setTagFilter(value || 'all')}
                            className="kb-shelf-tag-filter"
                            placeholder={S.shelf_tag_filter_placeholder}
                            options={allTags.map((tag) => ({ value: tag, label: tag }))}
                          />
                          <div className="kb-shelf-filter-actions">
                            <button
                              type="button"
                              className="kb-shelf-filter-action"
                              onClick={addVisibleToSelection}
                              disabled={visibleItems.length <= 0}
                              data-testid="citation-shelf-add-visible"
                            >
                              {S.shelf_add_to_queue}
                            </button>
                            <button
                              type="button"
                              className="kb-shelf-filter-action"
                              onClick={removeVisibleFromSelection}
                              disabled={visibleSelectedCount <= 0}
                              data-testid="citation-shelf-remove-visible"
                            >
                              {S.shelf_remove_from_queue}
                            </button>
                          </div>
                        </div>
                      </div>
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
                      {scopeFilter !== 'all' ? (
                        <button
                          type="button"
                          className="kb-shelf-filter-pill"
                          onClick={() => setScopeFilter('all')}
                        >
                          {S.shelf_filter_pill_scope.replace('{mode}', SCOPE_FILTER_LABEL(S)[scopeFilter])}
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
                        {group.label} ({group.items.length})
                      </div>
                    ) : null}
                    {group.items.map((item) => {
                      const display = citationDisplay(item)
                      const cardView = citationCardView(item)
                      const shelfKind = normalizeShelfItemKind(item.shelfItemKind)
                      const shelfKindText = shelfItemKindLabel(shelfKind, S)
                      const shelfOriginText = shelfOriginLabel(item.shelfOrigin, S)
                      const isFocused = item.key === focusedKey
                      const isDetailsExpanded = Boolean(expandedDetailKeys[item.key]) || (isFocused && shelfKind === 'reference')
                      const duplicateCount = duplicateCountByIdentity[shelfItemPaperIdentity(item)] || 0
                      const itemTags = normalizeShelfTags(item.tags)
                      const quality = qualityHints(item, display)
                      const noteText = String(item.note || '').trim()
                      const shelfSummary = shelfSummaryDisplay(item, cardView, S)
                      const shelfSummaryLine = shelfSummary.line
                      const shelfSummarySource = shelfSummary.sourceLabel
                      const shelfSummaryQuality = shelfSummary.quality
                      const shelfSummaryHeading = shelfSummary.headingLabel
                      const shelfSummaryLines = splitSummary(shelfSummaryLine)
                      const rawItemSourceLabel = String(item.sourceName || basenameFromPath(item.sourcePath) || '').trim()
                      const itemLocationLabel = String(item.locationLabel || item.headingPath || '').trim()
                      const shelfExcerpt = cleanCitationDisplayText(item.shelfExcerpt || '')
                      const rawShelfExcerptLabel = String(item.shelfExcerptLabel || '').trim()
                      const shelfExcerptLabel = rawShelfExcerptLabel === 'Reference entry'
                        ? S.shelf_reference_entry
                        : rawShelfExcerptLabel === 'Selected text'
                          ? S.shelf_reader_selection_selected
                          : rawShelfExcerptLabel === 'Excerpt'
                            ? S.shelf_excerpt_head
                            : rawShelfExcerptLabel || S.shelf_excerpt_head
                      const shelfCard = shelfCardPresentation(item, {
                        cardView,
                        display,
                        shelfKind,
                        shelfKindText,
                        rawSourceLabel: rawItemSourceLabel,
                        itemLocationLabel,
                        shelfExcerpt,
                        shelfExcerptLabel,
                      })
                      const shelfTitle = shelfCard.title
                      const publicationParts = shelfCard.showArticleSummary
                        ? uniqueCitationMetrics(citeVenueYearParts(item, display), citeImpactMetrics(item, S))
                        : []
                      const itemSourceLabel = shelfCard.sourceLabel
                      const itemSourceQuality = sourceQualityForItem(item, sourceQualityByPath)
                      const itemSourceOpenQuality = sourceOpenQualityView(
                        item,
                        itemSourceQuality,
                        S,
                        readerLocateResults[item.key],
                      )
                      const noteEditing = Boolean(editingNoteKeys[item.key] && isDetailsExpanded)
                      const visibleQualityChips = isFocused ? quality.chips.slice(0, 3) : quality.chips.slice(0, 1)
                      const showQuality = organizeOpen && shelfCard.showArticleSummary && Boolean(quality.needsRepair || isFocused)
                      const libraryMatch = libraryMatchView(item)
                      const hasLibraryFullText = shelfItemHasUsableLibraryFullText(item)
                      const primaryOpenLabel = hasLibraryFullText
                        ? S.shelf_open_full_text || S.shelf_open_source
                        : S.shelf_open_source
                      const sourceTrail = organizeOpen && isDetailsExpanded ? sourceTrailRows(item) : []
                      const showSourceOpenBadge = showSourceQualityDiagnostics && organizeOpen && isDetailsExpanded && (
                        itemSourceOpenQuality.tone === 'review'
                        || itemSourceOpenQuality.tone === 'missing'
                        || itemSourceOpenQuality.label === S.shelf_source_open_repaired_reopen
                      )
                      const showLibraryMatch = shelfCard.showArticleSummary && Boolean(
                        libraryMatch && (libraryMatch.tone === 'ready' || (organizeOpen && isDetailsExpanded)),
                      )
                      const messageTargetId = Number(item.traceAssistantMsgId || item.traceUserMsgId || 0)
                      const canOpenMessage = Boolean(onOpenMessage && Number.isFinite(messageTargetId) && messageTargetId > 0)
                      const isContextActive = Boolean(activeContextKeys[item.key])
                      const isItemSelected = Boolean(selectedKeys[item.key])

                      return (
                        <div
                          key={item.key}
                          className={`kb-shelf-item is-${shelfCard.surface} ${
                            isFocused
                              ? 'kb-shelf-item-active'
                              : ''
                          } ${isContextActive ? 'is-context' : ''} ${isDetailsExpanded ? 'is-expanded' : ''}`}
                          data-testid="citation-shelf-item"
                          aria-expanded={isDetailsExpanded}
                          aria-label={`${shelfTitle} ${isDetailsExpanded ? S.shelf_hide_details : S.shelf_show_details}`}
                          onClick={() => toggleItemDetails(item)}
                          onKeyDown={(event) => {
                            if (event.key === 'Enter' || event.key === ' ') {
                              event.preventDefault()
                              toggleItemDetails(item)
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
                              checked={isItemSelected}
                              readOnly
                              onChange={(event) => {
                                event.stopPropagation()
                              }}
                              onClick={(event) => {
                                event.stopPropagation()
                                toggleSelect(item.key, !isItemSelected)
                              }}
                            />
                            <div className="kb-shelf-item-main">
                              <div className="kb-shelf-item-title" data-testid="citation-shelf-item-title">{shelfTitle}</div>
                              {shelfCard.showAuthors && display.authors ? (
                                <div className="kb-shelf-item-authors">{display.authors}</div>
                              ) : null}
                              {publicationParts.length > 0 ? (
                                <div className="kb-shelf-item-venue" data-testid="citation-shelf-item-venue">
                                  {publicationParts.map((part, index) => (
                                    <span key={`${item.key}-pub-${part}`} className="kb-shelf-item-venue-part">
                                      {index > 0 ? <span className="kb-shelf-item-venue-separator">·</span> : null}
                                      {part}
                                    </span>
                                  ))}
                                </div>
                              ) : null}
                            </div>
                            <div className="kb-shelf-item-actions">
                              {(item.sourcePath || hasLibraryFullText) && onOpenSource ? (
                                <button
                                  type="button"
                                  className={`kb-shelf-source-open is-${hasLibraryFullText ? 'ready' : itemSourceOpenQuality.tone}`}
                                  aria-label={primaryOpenLabel}
                                  title={hasLibraryFullText ? primaryOpenLabel : itemSourceOpenQuality.reason || primaryOpenLabel}
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
                          {organizeOpen && shelfCard.showArticleSummary && quality.tip && (isFocused || quality.needsRepair) ? (
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
                              {duplicateCount > 1 ? (
                                <span className="kb-shelf-dup">{S.shelf_dup.replace('{n}', String(duplicateCount))}</span>
                              ) : null}
                              {showSourceOpenBadge ? (
                                <span
                                  className={`kb-shelf-source-open-quality is-${itemSourceOpenQuality.tone}`}
                                  data-testid="citation-shelf-source-open-quality"
                                  title={itemSourceOpenQuality.reason}
                                >
                                  {itemSourceOpenQuality.label}
                                </span>
                              ) : null}
                              {showLibraryMatch && libraryMatch ? (
                                <span
                                  className={`kb-shelf-library-match is-${libraryMatch.tone}`}
                                  data-testid="citation-shelf-library-match"
                                  title={libraryMatch.title}
                                >
                                  {libraryMatch.label}
                                </span>
                              ) : null}
                              {isContextActive ? (
                                <span className="kb-shelf-context-badge" data-testid="citation-shelf-context-badge">
                                  {S.shelf_context_badge || 'Context'}
                                </span>
                              ) : null}
                              {itemTags.map((tag) => (
                                <span key={`${item.key}-tag-${tag}`} className="kb-shelf-tag">
                                  #{tag}
                                </span>
                              ))}
                            </div>
                          </div>
                          {(noteText || noteEditing) ? (
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
                                  {noteText ? (
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
                          {isDetailsExpanded && shelfCard.showArticleSummary ? (
                            <div className="kb-shelf-summary" data-testid="citation-shelf-summary">
                              {summaryLoadingKey === item.key ? (
                                <div className="kb-shelf-summary-text">{S.shelf_summary_loading}</div>
                              ) : shelfSummaryLine ? (
                                <>
                                  <div className="kb-shelf-summary-meta">
                                    <span className="kb-shelf-summary-head">{shelfSummaryHeading}</span>
                                    {shelfSummarySource ? (
                                      <span className="kb-shelf-summary-source">/ {shelfSummarySource}</span>
                                    ) : null}
                                    {shelfSummary.showQuality ? (
                                      <span
                                        className={`kb-shelf-summary-quality is-${shelfSummaryQuality.tone}`}
                                        data-testid="citation-shelf-summary-quality"
                                      >
                                        {shelfSummaryQuality.label}
                                      </span>
                                    ) : null}
                                  </div>
                                  {(() => {
                                    const lines = shelfSummaryLines
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
                          {isDetailsExpanded && shelfCard.showExcerptInDetails && shelfCard.excerpt ? (
                            <div className="kb-shelf-excerpt" data-testid="citation-shelf-excerpt">
                              <div className="kb-shelf-excerpt-head">
                                {shelfCard.excerptLabel || S.shelf_excerpt_head}
                              </div>
                              <div className="kb-shelf-excerpt-text">
                                {shelfCard.excerpt}
                              </div>
                            </div>
                          ) : null}
                          {isDetailsExpanded && shelfCard.showArticleSummary && (item.doiUrl || !itemSourceLabel) ? (
                            <div className="kb-shelf-doi">
                              {item.doiUrl ? (
                                <a
                                  className="kb-shelf-doi-link"
                                  href={item.doiUrl}
                                  rel="noreferrer"
                                  target="_blank"
                                  onClick={(event) => event.stopPropagation()}
                                >
                                  {item.doi || item.doiUrl}
                                </a>
                              ) : (
                                <span className="kb-shelf-doi-empty">{S.shelf_no_doi_link}</span>
                              )}
                            </div>
                          ) : null}
                          {isDetailsExpanded && sourceTrail.length > 0 ? (
                            <div
                              className="kb-shelf-trace-detail"
                              data-testid="citation-shelf-source-trail"
                            >
                              <div className="kb-shelf-trace-head">{S.shelf_provenance_head || S.shelf_trace_head}</div>
                              <div className="kb-shelf-trace-rows">
                                {sourceTrail.map((row) => (
                                  <div
                                    key={`${item.key}-trail-${row.id}`}
                                    className="kb-shelf-trace-row"
                                    data-testid={`citation-shelf-trace-row-${row.id}`}
                                    title={row.title}
                                  >
                                    <span className="kb-shelf-trace-label">{row.label}</span>
                                    <span
                                      className="kb-shelf-trace-value"
                                      data-testid={row.id === 'source' ? 'citation-shelf-item-source' : undefined}
                                    >
                                      {row.value}
                                    </span>
                                  </div>
                                ))}
                              </div>
                              <div className="kb-shelf-trace-actions">
                                {(item.sourcePath || hasLibraryFullText) && onOpenSource ? (
                                  <button
                                    type="button"
                                    className="kb-shelf-trace-action"
                                    data-testid={hasLibraryFullText ? 'citation-shelf-trail-open-full-text' : 'citation-shelf-trail-open-source'}
                                    onClick={(event) => {
                                      event.stopPropagation()
                                      onOpenSource(item)
                                    }}
                                  >
                                    {primaryOpenLabel}
                                  </button>
                                ) : null}
                                {hasLibraryFullText && item.sourcePath && onOpenDiscoverySource ? (
                                  <button
                                    type="button"
                                    className="kb-shelf-trace-action"
                                    data-testid="citation-shelf-trail-open-source"
                                    onClick={(event) => {
                                      event.stopPropagation()
                                      onOpenDiscoverySource(item)
                                    }}
                                  >
                                    {S.shelf_open_citation_source || S.shelf_open_source}
                                  </button>
                                ) : null}
                                {canOpenMessage && onOpenMessage ? (
                                  <button
                                    type="button"
                                    className="kb-shelf-trace-action"
                                    data-testid="citation-shelf-trail-open-message"
                                    onClick={(event) => {
                                      event.stopPropagation()
                                      onOpenMessage(item)
                                    }}
                                  >
                                    {item.traceAssistantMsgId ? S.shelf_open_answer : S.shelf_open_message}
                                  </button>
                                ) : null}
                              </div>
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
