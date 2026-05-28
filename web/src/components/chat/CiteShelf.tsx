import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Button, Input, Select, message } from 'antd'
import { FileSearchOutlined } from '@ant-design/icons'
import type { CiteShelfItem } from './citationState'
import type { ConversionQualitySummary, LibrarySourceQualityItem } from '../../api/library'
import type { ShelfMetadataQuality, ShelfMetadataRepairImpact } from '../../api/references'
import { libraryApi } from '../../api/library'
import {
  citationCardView,
  citationDisplay,
  citationFormats,
  citeMetricSummary,
  isLikelyWeakCitationTitle,
  normalizeShelfTags,
  summarySourceLabel,
} from './citationState'
import { useT } from '../../i18n'

interface Props {
  open: boolean
  items: CiteShelfItem[]
  snapshots: Array<{ id: string; name: string; createdAt: number }>
  selectedSnapshotId: string
  snapshotDiff: string
  focusedKey: string
  summaryLoadingKey: string
  repairLoadingKey: string
  repairImpact: ShelfMetadataRepairImpact | null
  onToggle: () => void
  onClear: () => void
  onSelect: (item: CiteShelfItem) => void
  onOpenSource?: (item: CiteShelfItem) => void
  onRemove: (key: string) => void
  onUpdateTags: (key: string, tags: string[]) => void
  onUpdateNote: (key: string, note: string) => void
  onRepair: (item: CiteShelfItem, options?: { silent?: boolean }) => void
  onSelectSnapshot: (id: string) => void
  onSaveSnapshot: () => void
  onLoadSnapshot: () => void
  onDeleteSnapshot: () => void
}

const TAG_PRESETS = ['baseline', 'idea', 'related-work'] as const

type GroupMode = 'none' | 'tag' | 'source'
type SourceQualityByPath = Record<string, LibrarySourceQualityItem>
type ShelfExportKind = 'bib' | 'csv' | 'ris'

const GROUP_MODE_LABEL = (S: Record<string, string>): Record<GroupMode, string> => ({
  none: S.shelf_no_group,
  tag: S.shelf_by_tag,
  source: S.shelf_by_source,
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

const hasConflictingVenueSignals = (item: CiteShelfItem): boolean => {
  const hasJournalSignal = Boolean(String(item.journalIf || item.journalQuartile || item.journalIfSource || '').trim())
  const hasConfSignal = Boolean(
    String(item.conferenceTier || item.conferenceCcf || item.conferenceName || item.conferenceAcronym || '').trim(),
  )
  const venueKind = String(item.venueKind || '').trim().toLowerCase()
  return (
    (venueKind === 'conference' && hasJournalSignal)
    || (venueKind === 'journal' && hasConfSignal)
    || (hasJournalSignal && hasConfSignal)
  )
}

const shouldAutoRepairItem = (item: CiteShelfItem, display = citationDisplay(item)): boolean => {
  if (metadataQualityReady(item)) return false
  if (metadataQualityNeedsRepair(item)) return true
  const rawTitle = String(item.title || '').trim()
  const visibleTitle = String(display.main || rawTitle || item.main || '').trim()
  const hasDoi = Boolean(normalizeDoiLike(item.doi || item.doiUrl))
  const hasAuthors = Boolean(String(item.authors || '').trim())
  const hasVenue = Boolean(String(item.venue || '').trim())
  const unresolved = !item.bibliometricsChecked
  const rawTitleNeedsRepair = isLikelyWeakCitationTitle(rawTitle)
  const visibleTitleNeedsRepair = isLikelyWeakCitationTitle(visibleTitle)
  return (
    hasConflictingVenueSignals(item)
    || (hasDoi && (rawTitleNeedsRepair || unresolved))
    || (!hasDoi && unresolved && (visibleTitleNeedsRepair || !hasAuthors || !hasVenue))
  )
}

const autoRepairFingerprint = (item: CiteShelfItem, display = citationDisplay(item)): string => [
  normalizeDoiLike(item.doi || item.doiUrl),
  String(item.title || '').trim(),
  String(display.main || '').trim(),
  String(item.authors || '').trim(),
  String(item.venue || '').trim(),
  String(item.year || '').trim(),
  String(item.venueKind || '').trim(),
  String(item.citationCount || 0),
  item.bibliometricsChecked ? '1' : '0',
].join('|')

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

const metadataIssueChip = (code: string): string => {
  const key = String(code || '').trim().toLowerCase()
  if (key === 'missing_doi') return '自动匹配 DOI'
  if (key === 'doi_not_promoted') return '写入 DOI'
  if (key === 'missing_authors') return '自动补作者'
  if (key === 'missing_venue') return '自动补期刊/会议'
  if (key === 'missing_year') return '自动补年份'
  if (key === 'weak_or_missing_title') return '自动校正标题'
  if (key === 'missing_source') return '来源缺失'
  if (key.startsWith('external_metadata_')) return '外部元数据校验'
  return key ? key.replace(/_/g, ' ') : '自动补全'
}

export function CiteShelf({
  open,
  items,
  snapshots,
  selectedSnapshotId,
  snapshotDiff,
  focusedKey,
  summaryLoadingKey,
  repairLoadingKey,
  repairImpact,
  onToggle,
  onClear,
  onSelect,
  onOpenSource,
  onRemove,
  onUpdateTags,
  onUpdateNote,
  onRepair,
  onSelectSnapshot,
  onSaveSnapshot,
  onLoadSnapshot,
  onDeleteSnapshot,
}: Props) {
  const S = useT()
  const [expandedSummaryKeys, setExpandedSummaryKeys] = useState<Record<string, boolean>>({})
  const [selectedKeys, setSelectedKeys] = useState<Record<string, boolean>>({})
  const [searchText, setSearchText] = useState('')
  const [sortKey, setSortKey] = useState<'recent' | 'cited' | 'year' | 'impact'>('recent')
  const [groupMode, setGroupMode] = useState<GroupMode>('none')
  const [tagFilter, setTagFilter] = useState<string>('all')
  const [advancedFiltersOpen, setAdvancedFiltersOpen] = useState(false)
  const [preflightExportKind, setPreflightExportKind] = useState<ShelfExportKind | ''>('')
  const [batchTagInput, setBatchTagInput] = useState('')
  const [editingNoteKeys, setEditingNoteKeys] = useState<Record<string, boolean>>({})
  const [copyState, setCopyState] = useState<'idle' | 'gbt' | 'bibtex' | 'error'>('idle')
  const [sourceQualityByPath, setSourceQualityByPath] = useState<SourceQualityByPath>({})
  const [sourceRepairingKey, setSourceRepairingKey] = useState('')
  const copyStateTimerRef = useRef<number | null>(null)
  const sourceRepairStreamRef = useRef<AbortController | null>(null)
  const autoSourceRepairKeysRef = useRef<Record<string, boolean>>({})
  const autoRepairFingerprintsRef = useRef<Record<string, string>>({})

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
        const chip = metadataIssueChip(code)
        if (chip && !chips.includes(chip)) chips.push(chip)
      }
      const needsRepair = metadataQualityNeedsRepair(item)
      const score = Number(contract.score || 0)
      const tip = needsRepair
        ? `系统会用参考文献原文、索引缓存和 Crossref 自动补全；当前 Q${Number.isFinite(score) ? Math.round(score) : 0}。`
        : '系统已记录元数据质量状态。'
      return { chips: chips.slice(0, 3), tip, needsRepair }
    }
    const rawTitle = String(item.title || '').trim()
    const visibleTitle = String(display.main || rawTitle || item.main || '').trim()
    const hasWeakTitle = isLikelyWeakCitationTitle(visibleTitle)
    const hasWeakStoredTitle = isLikelyWeakCitationTitle(rawTitle)
    const hasDoi = Boolean(normalizeDoiLike(item.doi || item.doiUrl))
    const hasAuthors = Boolean(String(item.authors || '').trim())
    const hasVenue = Boolean(String(item.venue || '').trim())
    const hasMetaConflict = hasConflictingVenueSignals(item)
    const externalNeedsReview = externalMetadataNeedsVisibleReview(item, display)
    const unresolved = !item.bibliometricsChecked
    const bibliographicEntry = Boolean(item.isInpaper || item.raw || item.citeFmt || hasDoi || item.externalDoi || item.externalDoiUrl)
    const needsRepair = shouldAutoRepairItem(item, display)

    if (externalNeedsReview) chips.push('元数据待核对')
    if (bibliographicEntry && !hasDoi) chips.push(S.shelf_missing_doi)
    if (bibliographicEntry && !hasAuthors) chips.push(S.shelf_missing_author)
    if (bibliographicEntry && !hasVenue) chips.push(S.shelf_missing_venue)
    if (hasWeakTitle) chips.push(S.shelf_weak_title)
    if (hasMetaConflict) chips.push(S.shelf_meta_conflict)
    if (bibliographicEntry && unresolved && chips.length <= 1) chips.push(S.shelf_pending_verify)

    if (!chips.length) return { chips: [], tip: '', needsRepair }

    let tip = S.shelf_auto_fix_tip
    if (externalNeedsReview) tip = item.externalMetadataReason || '外部元数据与原参考条目需要核对，标题/作者以原条目为准，DOI 和指标先作为线索。'
    else if (bibliographicEntry && !hasDoi) tip = S.shelf_no_doi_tip
    else if (hasMetaConflict) tip = S.shelf_conflict_tip
    else if (hasWeakStoredTitle && !hasWeakTitle) tip = S.shelf_weak_stored_tip
    else if (hasWeakTitle) tip = S.shelf_weak_title_tip
    return { chips: chips.slice(0, 3), tip, needsRepair }
  }

  useEffect(() => {
    if (repairLoadingKey) return
    for (const item of items) {
      const display = citationDisplay(item)
      if (!shouldAutoRepairItem(item, display)) continue
      const fingerprint = autoRepairFingerprint(item, display)
      if (autoRepairFingerprintsRef.current[item.key] === fingerprint) continue
      autoRepairFingerprintsRef.current[item.key] = fingerprint
      onRepair(item, { silent: true })
      return
    }
  }, [items, onRepair, repairLoadingKey])

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
  }, [open, sourceQualityKey, sourceQualitySources])

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
    for (const item of items) {
      const display = citationDisplay(item)
      const needsMetadataReview = shouldAutoRepairItem(item, display)
      const isDuplicate = (duplicateCountByIdentity[paperIdentity(item)] || 0) > 1
      const hasSummary = Boolean(String(item.summaryLine || citationCardView(item).summary || '').trim())
      const summaryView = summaryQualityView(item, S)

      if (needsMetadataReview) metadataReview += 1
      else metadataReadyItems += 1
      if (isDuplicate) duplicateItems += 1
      if (hasSummary && summaryView.ok) summaryReady += 1
      else summaryReview += 1
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
      status: total <= 0 ? 'empty' : metadataReview > 0 ? 'review' : 'ready',
    }
  }, [S, duplicateCountByIdentity, items])

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
        item.traceAssistantOrder ? `回答 ${item.traceAssistantOrder}` : '',
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
  }, [items, searchText, sortKey, tagFilter])

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
    S.shelf_all,
    S.shelf_source_prefix,
    S.shelf_tag_prefix,
    S.shelf_unknown_source,
    S.shelf_untagged,
    visibleItems,
  ])

  const selectedCount = Object.values(selectedKeys).filter(Boolean).length
  const selectedItems = useMemo(
    () => items.filter((item) => Boolean(selectedKeys[item.key])),
    [items, selectedKeys],
  )
  const selectedMetadataReviewItems = useMemo(
    () => selectedItems.filter((item) => shouldAutoRepairItem(item, citationDisplay(item))),
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
      if (!sourceQualityNeedsReview(sourceQualityByPath[sourcePath]?.conversion_quality)) continue
      seen.add(sourcePath)
      out.push({
        source_path: sourcePath,
        source_name: String(item.sourceName || item.title || item.main || '').trim(),
      })
    }
    return out
  }, [selectedItems, sourceQualityByPath])
  const selectedReviewSourceKey = useMemo(
    () => sourceListKey(selectedReviewSources),
    [selectedReviewSources],
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
    const refreshIndexAndRepairSources = async (needsReindex: boolean) => {
      if (needsReindex) {
        try {
          await libraryApi.reindex()
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
            void refreshIndexAndRepairSources(needsReindex).finally(clearRepairing)
          },
          () => {
            sourceRepairStreamRef.current = null
            void refreshIndexAndRepairSources(needsReindex).finally(clearRepairing)
          },
        )
      } else if (repaired > 0) {
        if (!silent) message.success(`Markdown repaired: ${repaired}`)
        await refreshIndexAndRepairSources(needsReindex)
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

  const exportSelectedAs = (kind: ShelfExportKind, options: { skipPreflight?: boolean; onlyMetadataReady?: boolean } = {}) => {
    if (selectedItems.length <= 0) return
    if (selectedMetadataReviewCount > 0 && !options.skipPreflight) {
      setPreflightExportKind(kind)
      return
    }
    const exportItems = options.onlyMetadataReady
      ? selectedItems.filter((item) => !selectedMetadataReviewKeySet.has(item.key))
      : selectedItems
    if (exportItems.length <= 0) {
      message.warning(S.shelf_export_preflight_no_healthy)
      return
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
        (() => {
          const sourcePath = String(item.sourcePath || '').trim()
          return sourcePath ? sourceQualityByPath[sourcePath]?.conversion_quality : null
        })(),
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
      <button
        aria-label={S.shelf_title}
        className={`kb-shelf-toggle-btn fixed right-4 top-1/2 z-30 -translate-y-1/2 rounded-full border border-[var(--border)] bg-[var(--panel)] px-3 py-2 text-xs shadow-[0_10px_30px_rgba(15,23,42,0.12)] transition ${open ? 'pointer-events-none opacity-0' : ''}`}
        data-testid="citation-shelf-toggle"
        onClick={onToggle}
        type="button"
      >
        {S.shelf_title}
      </button>
      <aside
        className={`kb-shelf-panel fixed right-0 top-0 z-40 h-full w-[360px] max-w-[90vw] border-l border-[var(--border)] bg-[var(--panel)] shadow-[0_24px_64px_rgba(15,23,42,0.18)] transition-transform duration-300 ${open ? 'translate-x-0' : 'translate-x-full'}`}
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
                <Button size="small" onClick={onClear} disabled={items.length === 0} data-testid="citation-shelf-clear">
                  {S.shelf_clear}
                </Button>
                <Button size="small" onClick={onToggle} data-testid="citation-shelf-close">
                  {S.shelf_close}
                </Button>
              </div>
            </div>
            <div className="kb-shelf-snapshot-row" onClick={(event) => event.stopPropagation()}>
              <Button
                size="small"
                onClick={onSaveSnapshot}
                disabled={items.length === 0}
                data-testid="citation-shelf-save-snapshot"
              >
                {S.shelf_save_snapshot}
              </Button>
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
            </div>
            {snapshotDiff ? (
              <div className="kb-shelf-snapshot-diff">{snapshotDiff}</div>
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
                </div>
                <div className="kb-shelf-readiness-chips">
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
                  {shelfReadiness.summaryReview > 0 ? (
                    <span className="kb-shelf-readiness-chip is-review">
                      {S.shelf_readiness_summary_review.replace('{n}', String(shelfReadiness.summaryReview))}
                    </span>
                  ) : (
                    <span className="kb-shelf-readiness-chip">
                      {S.shelf_readiness_summary_grounded.replace('{n}', `${shelfReadiness.summaryRate}%`)}
                    </span>
                  )}
                </div>
                {repairImpact ? (
                  <div className="kb-shelf-repair-impact" data-testid="citation-shelf-repair-impact">
                    <span>已自动补全 {repairImpact.changed} 条</span>
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
                  <Button size="small" onClick={() => exportSelectedAs('bib')} data-testid="citation-shelf-export-bib">
                    {S.shelf_export_bib_btn}
                  </Button>
                  <Button size="small" onClick={() => exportSelectedAs('ris')} data-testid="citation-shelf-export-ris">
                    {S.shelf_export_ris_btn}
                  </Button>
                  <Button size="small" onClick={() => exportSelectedAs('csv')} data-testid="citation-shelf-export-csv">
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
                          exportSelectedAs(preflightExportKind, { skipPreflight: true, onlyMetadataReady: true })
                          setPreflightExportKind('')
                        }}
                        data-testid="citation-shelf-export-preflight-healthy"
                      >
                        {S.shelf_export_preflight_healthy}
                      </Button>
                      <Button
                        size="small"
                        onClick={() => {
                          exportSelectedAs(preflightExportKind, { skipPreflight: true })
                          setPreflightExportKind('')
                        }}
                        data-testid="citation-shelf-export-preflight-continue"
                      >
                        {S.shelf_export_preflight_continue}
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
                  <Button size="small" onClick={addVisibleToSelection} disabled={visibleItems.length <= 0} data-testid="citation-shelf-add-visible">
                    {S.shelf_add_to_queue}
                  </Button>
                  <Button size="small" onClick={removeVisibleFromSelection} disabled={visibleSelectedCount <= 0} data-testid="citation-shelf-remove-visible">
                    {S.shelf_remove_from_queue}
                  </Button>
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
              <div className="rounded-2xl border border-dashed border-[var(--border)] px-4 py-5 text-sm text-black/45 dark:text-white/45">
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
                      const subtitle = display.source
                      const duplicateCount = duplicateCountByIdentity[paperIdentity(item)] || 0
                      const trace = sourceTraceLabel(item)
                      const itemTags = normalizeShelfTags(item.tags)
                      const quality = qualityHints(item, display)
                      const noteText = String(item.note || '').trim()
                      const isFocused = item.key === focusedKey
                      const metrics = citeMetricSummary(item)
                      const shelfSummaryLine = String(item.summaryLine || cardView.summary || '').trim()
                      const shelfSummarySource = item.summaryLine
                        ? summarySourceLabel(item.summarySource, item.summaryProvider)
                        : summarySourceLabel('citation_card')
                      const shelfSummaryQuality = summaryQualityView(item, S)
                      const noteEditing = Boolean(editingNoteKeys[item.key] && isFocused)
                      const tagOptions = [...TAG_PRESETS, ...allTags]
                        .filter((tag, idx, arr) => arr.findIndex((x) => x.toLowerCase() === tag.toLowerCase()) === idx)
                        .map((tag) => ({ value: tag, label: tag }))

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
                                  className="kb-shelf-source-open"
                                  aria-label={S.locate_label}
                                  title={S.locate_label}
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
                                {S.shelf_remove_item}
                              </button>
                            </div>
                          </div>
                          {subtitle ? (
                            <div className="kb-shelf-item-source" data-testid="citation-shelf-item-source">{subtitle}</div>
                          ) : null}
                          {quality.chips.length > 0 ? (
                            <div className="kb-shelf-quality">
                              <div className="kb-shelf-quality-chips">
                                {quality.chips.map((chip) => (
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
                                  disabled={repairLoadingKey === item.key}
                                  data-testid="citation-shelf-repair"
                                  onClick={(event) => {
                                    event.stopPropagation()
                                    onRepair(item)
                                  }}
                                >
                                  {repairLoadingKey === item.key ? S.shelf_repairing : S.shelf_auto_repair}
                                </button>
                              ) : null}
                            </div>
                          ) : null}
                          {quality.tip ? (
                            <div className="kb-shelf-quality-tip">{quality.tip}</div>
                          ) : null}
                          <div className="kb-shelf-meta-row">
                            <div className="kb-shelf-meta-badges">
                              {trace.labels.map((label, idx) => (
                                <span key={`${item.key}-trace-${idx}-${label}`} className="kb-shelf-origin" title={trace.debugTitle || undefined}>
                                  {label}
                                </span>
                              ))}
                              {duplicateCount > 1 ? (
                                <span className="kb-shelf-dup">{S.shelf_dup.replace('{n}', String(duplicateCount))}</span>
                              ) : null}
                              {itemTags.map((tag) => (
                                <span key={`${item.key}-tag-${tag}`} className="kb-shelf-tag">
                                  #{tag}
                                </span>
                              ))}
                            </div>
                            <div className="kb-shelf-tag-editor kb-shelf-tag-editor-inline" onClick={(event) => event.stopPropagation()}>
                              <Select
                                mode="tags"
                                size="small"
                                maxTagCount={1}
                                maxTagTextLength={14}
                                className="w-full"
                                placeholder={S.shelf_add_tag}
                                value={itemTags}
                                options={tagOptions}
                                onChange={(value) => onUpdateTags(item.key, normalizeShelfTags(value))}
                              />
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
                          {metrics.length > 0 ? (
                            <div className="kb-shelf-metrics">
                              {metrics.map((metric) => (
                                <span key={metric} className="kb-shelf-metric">
                                  {metric}
                                </span>
                              ))}
                            </div>
                          ) : null}
                          <div className="kb-shelf-doi">
                            {item.doiUrl ? (
                              <a className="kb-shelf-doi-link" href={item.doiUrl} rel="noreferrer" target="_blank">
                                {item.doi || item.doiUrl}
                              </a>
                            ) : (
                              <span className="kb-shelf-doi-empty">{S.shelf_no_doi_link}</span>
                            )}
                          </div>
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
                              ) : null}
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
