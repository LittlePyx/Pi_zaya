/* eslint-disable react-hooks/set-state-in-effect */

import { useEffect, useMemo, useRef, useState } from 'react'
import { Button, Input, Select, message } from 'antd'
import type { CiteShelfItem } from './citationState'
import {
  citationDisplay,
  citationFormats,
  citeMetricSummary,
  isLikelyWeakCitationTitle,
  normalizeShelfTags,
  summarySourceLabel,
} from './citationState'

interface Props {
  open: boolean
  items: CiteShelfItem[]
  snapshots: Array<{ id: string; name: string; createdAt: number }>
  selectedSnapshotId: string
  snapshotDiff: string
  focusedKey: string
  summaryLoadingKey: string
  repairLoadingKey: string
  onToggle: () => void
  onClear: () => void
  onSelect: (item: CiteShelfItem) => void
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

const GROUP_MODE_LABEL: Record<GroupMode, string> = {
  none: '不分组',
  tag: '按标签',
  source: '按来源',
}

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

export function CiteShelf({
  open,
  items,
  snapshots,
  selectedSnapshotId,
  snapshotDiff,
  focusedKey,
  summaryLoadingKey,
  repairLoadingKey,
  onToggle,
  onClear,
  onSelect,
  onRemove,
  onUpdateTags,
  onUpdateNote,
  onRepair,
  onSelectSnapshot,
  onSaveSnapshot,
  onLoadSnapshot,
  onDeleteSnapshot,
}: Props) {
  const [expandedSummaryKeys, setExpandedSummaryKeys] = useState<Record<string, boolean>>({})
  const [selectedKeys, setSelectedKeys] = useState<Record<string, boolean>>({})
  const [searchText, setSearchText] = useState('')
  const [sortKey, setSortKey] = useState<'recent' | 'cited' | 'year' | 'impact'>('recent')
  const [groupMode, setGroupMode] = useState<GroupMode>('none')
  const [tagFilter, setTagFilter] = useState<string>('all')
  const [advancedFiltersOpen, setAdvancedFiltersOpen] = useState(false)
  const [batchTagInput, setBatchTagInput] = useState('')
  const [editingNoteKeys, setEditingNoteKeys] = useState<Record<string, boolean>>({})
  const [copyState, setCopyState] = useState<'idle' | 'gbt' | 'bibtex' | 'error'>('idle')
  const copyStateTimerRef = useRef<number | null>(null)
  const autoRepairFingerprintsRef = useRef<Record<string, string>>({})

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
      labels.push(`回答 ${answerOrder}`)
    }
    const num = Number(item.num || 0)
    if (Number.isFinite(num) && num > 0) labels.push(`引用 #${num}`)
    const anchor = String(item.anchor || '').trim()
    const debugTitle = anchor ? `锚点: ${anchor}` : ''
    return { labels, debugTitle }
  }

  const qualityHints = (
    item: CiteShelfItem,
    display: ReturnType<typeof citationDisplay>,
  ): { chips: string[]; tip: string; needsRepair: boolean } => {
    const chips: string[] = []
    const rawTitle = String(item.title || '').trim()
    const visibleTitle = String(display.main || rawTitle || item.main || '').trim()
    const hasWeakTitle = isLikelyWeakCitationTitle(visibleTitle)
    const hasWeakStoredTitle = isLikelyWeakCitationTitle(rawTitle)
    const hasDoi = Boolean(normalizeDoiLike(item.doi || item.doiUrl))
    const hasAuthors = Boolean(String(item.authors || '').trim())
    const hasVenue = Boolean(String(item.venue || '').trim())
    const hasMetaConflict = hasConflictingVenueSignals(item)
    const unresolved = !item.bibliometricsChecked
    const needsRepair = shouldAutoRepairItem(item, display)

    if (!hasDoi) chips.push('缺 DOI')
    if (!hasAuthors) chips.push('缺作者')
    if (!hasVenue) chips.push('缺期刊/会议')
    if (hasWeakTitle) chips.push('标题待校正')
    if (hasMetaConflict) chips.push('元数据冲突')
    if (unresolved && chips.length <= 1) chips.push('待校验')

    if (!chips.length) return { chips: [], tip: '', needsRepair }

    let tip = '系统会自动校正元数据，无需手动点击。'
    if (!hasDoi) tip = '当前缺少 DOI，系统会先按标题和参考文献串自动匹配。'
    else if (hasMetaConflict) tip = '检测到期刊/会议信号冲突，系统会自动重新匹配并校正。'
    else if (hasWeakStoredTitle && !hasWeakTitle) tip = '已优先展示解析出的标题，系统会继续回填规范元数据。'
    else if (hasWeakTitle) tip = '标题信息不完整，系统会自动按 DOI 或参考文献重新校正。'
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
      return [{ key: 'all', label: '全部条目', items: visibleItems }]
    }
    const groups = new Map<string, { label: string; items: CiteShelfItem[] }>()
    for (const item of visibleItems) {
      let groupKey = ''
      let groupLabel = ''
      if (groupMode === 'tag') {
        const tags = normalizeShelfTags(item.tags)
        const primaryTag = tags[0] || '未标记'
        groupKey = `tag:${primaryTag.toLowerCase()}`
        groupLabel = `标签 · ${primaryTag}`
      } else {
        const src = String(item.sourceName || item.sourcePath || '').trim() || '未知来源'
        groupKey = `source:${src}`
        groupLabel = `来源 · ${src}`
      }
      const existing = groups.get(groupKey)
      if (existing) {
        existing.items.push(item)
      } else {
        groups.set(groupKey, { label: groupLabel, items: [item] })
      }
    }
    return Array.from(groups.entries()).map(([k, v]) => ({ key: k, label: v.label, items: v.items }))
  }, [groupMode, visibleItems])

  const selectedCount = Object.values(selectedKeys).filter(Boolean).length
  const selectedItems = useMemo(
    () => items.filter((item) => Boolean(selectedKeys[item.key])),
    [items, selectedKeys],
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

  const exportSelectedAs = (kind: 'bib' | 'csv' | 'ris') => {
    if (selectedItems.length <= 0) return
    try {
      const base = `cite_shelf_selected_${nowStamp()}`
      if (kind === 'bib') {
        const bib = selectedItems.map((item) => citationFormats(item).bibtex).join('\n\n').trim()
        if (!bib) return
        downloadTextFile(`${base}.bib`, bib, 'application/x-bibtex;charset=utf-8')
        message.success(`已导出 BibTeX（${selectedItems.length} 条）`)
        return
      }
      if (kind === 'ris') {
        const ris = selectedItems.map((item) => citationFormats(item).ris).join('\n\n').trim()
        if (!ris) return
        downloadTextFile(`${base}.ris`, ris, 'application/x-research-info-systems;charset=utf-8')
        message.success(`已导出 RIS（${selectedItems.length} 条）`)
        return
      }
      const headers = [
        'title',
        'authors',
        'year',
        'venue',
        'doi',
        'source',
        'reference_num',
        'citation_count',
        'journal_if',
        'journal_quartile',
        'conference_tier',
        'conference_ccf',
        'summary',
      ]
      const rows = selectedItems.map((item) => ([
        item.title || item.main,
        item.authors,
        item.year,
        item.venue,
        item.doi || item.doiUrl,
        item.sourceName || item.sourcePath,
        item.num || '',
        item.citationCount || 0,
        item.journalIf,
        item.journalQuartile,
        item.conferenceTier,
        item.conferenceCcf,
        item.summaryLine,
      ].map((field) => csvEscape(field)).join(',')))
      const csv = `${headers.join(',')}\n${rows.join('\n')}`
      downloadTextFile(`${base}.csv`, csv, 'text/csv;charset=utf-8')
      message.success(`已导出 CSV（${selectedItems.length} 条）`)
    } catch {
      message.error('导出失败，请稍后重试')
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
        className={`fixed right-4 top-1/2 z-30 -translate-y-1/2 rounded-full border border-[var(--border)] bg-[var(--panel)] px-3 py-2 text-xs shadow-[0_10px_30px_rgba(15,23,42,0.12)] transition ${open ? 'pointer-events-none opacity-0' : ''}`}
        onClick={onToggle}
        type="button"
      >
        文献篮
      </button>
      <aside
        className={`fixed right-0 top-0 z-40 h-full w-[360px] max-w-[90vw] border-l border-[var(--border)] bg-[var(--panel)] shadow-[0_24px_64px_rgba(15,23,42,0.18)] transition-transform duration-300 ${open ? 'translate-x-0' : 'translate-x-full'}`}
      >
        <div className="flex h-full flex-col">
          <div className="kb-shelf-head border-b border-[var(--border)] px-3 py-3">
            <div className="kb-shelf-head-top">
              <div className="kb-shelf-head-meta">
                <div className="kb-shelf-title">文献篮</div>
                <div className="kb-shelf-count">
                  已收藏 {items.length} 条{searchText.trim() ? ` · 匹配 ${visibleItems.length} 条` : ''}
                </div>
              </div>
              <div className="kb-shelf-head-actions">
                <Button size="small" onClick={onClear} disabled={items.length === 0}>
                  清空
                </Button>
                <Button size="small" onClick={onToggle}>
                  关闭
                </Button>
              </div>
            </div>
            <div className="kb-shelf-snapshot-row" onClick={(event) => event.stopPropagation()}>
              <Button size="small" onClick={onSaveSnapshot} disabled={items.length === 0}>
                保存快照
              </Button>
              <Select
                size="small"
                value={selectedSnapshotId || undefined}
                placeholder={snapshotOptions.length > 0 ? '选择快照' : '暂无快照'}
                className="kb-shelf-snapshot-select"
                options={snapshotOptions}
                onChange={(value) => onSelectSnapshot(String(value || ''))}
              />
              <Button size="small" onClick={onLoadSnapshot} disabled={!selectedSnapshotId}>
                载入
              </Button>
              <Button size="small" onClick={onDeleteSnapshot} disabled={!selectedSnapshotId}>
                删除
              </Button>
            </div>
            {snapshotDiff ? (
              <div className="kb-shelf-snapshot-diff">{snapshotDiff}</div>
            ) : null}
            {selectedCount > 0 ? (
              <div className="kb-shelf-batch-row">
                <span className="kb-shelf-batch-count">导出队列 {selectedCount} 条</span>
                <Button size="small" onClick={removeSelected}>
                  批量移除
                </Button>
                <Button size="small" onClick={() => void copySelectedAs('gbt')}>
                  {copyState === 'gbt' ? '已复制 GB/T' : '复制 GB/T'}
                </Button>
                <Button size="small" onClick={() => void copySelectedAs('bibtex')}>
                  {copyState === 'bibtex' ? '已复制 BibTeX' : '复制 BibTeX'}
                </Button>
                <Button size="small" onClick={() => exportSelectedAs('bib')}>
                  导出 BibTeX
                </Button>
                <Button size="small" onClick={() => exportSelectedAs('ris')}>
                  导出 RIS
                </Button>
                <Button size="small" onClick={() => exportSelectedAs('csv')}>
                  导出 CSV
                </Button>
                <div className="flex min-w-[170px] items-center gap-1" onClick={(event) => event.stopPropagation()}>
                  <Select
                    size="small"
                    value={batchTagInput || undefined}
                    placeholder="批量加标签"
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
                    去标签
                  </Button>
                </div>
                <button type="button" className="kb-shelf-clear-select" onClick={clearSelected}>
                  清除勾选
                </button>
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
                    placeholder="搜索标题 / 作者 / DOI"
                    value={searchText}
                    onChange={(event) => setSearchText(event.target.value)}
                    className="kb-shelf-search"
                  />
                  <Select
                    value={sortKey}
                    onChange={(value) => setSortKey(value)}
                    className="kb-shelf-sort"
                    options={[
                      { value: 'recent', label: '最近加入' },
                      { value: 'cited', label: '被引次数' },
                      { value: 'year', label: '年份' },
                      { value: 'impact', label: 'IF/评级' },
                    ]}
                  />
                  <button
                    type="button"
                    className={`kb-shelf-advanced-toggle ${advancedFiltersOpen ? 'is-open' : ''} ${advancedFilterActive ? 'is-active' : ''}`}
                    onClick={() => setAdvancedFiltersOpen((prev) => !prev)}
                  >
                    {advancedFiltersOpen ? '收起筛选' : '高级筛选'}
                  </button>
                  <Button size="small" onClick={addVisibleToSelection} disabled={visibleItems.length <= 0}>
                    当前筛选入队
                  </Button>
                  <Button size="small" onClick={removeVisibleFromSelection} disabled={visibleSelectedCount <= 0}>
                    取消当前筛选
                  </Button>
                  </div>
                  {advancedFiltersOpen ? (
                    <div className="kb-shelf-filters">
                  <Select
                    value={groupMode}
                    onChange={(value) => setGroupMode(value as GroupMode)}
                    className="kb-shelf-sort"
                    options={[
                      { value: 'none', label: '不分组' },
                      { value: 'tag', label: '按标签' },
                      { value: 'source', label: '按来源' },
                    ]}
                  />
                  <Select
                    allowClear
                    value={tagFilter === 'all' ? undefined : tagFilter}
                    onChange={(value) => setTagFilter(value || 'all')}
                    className="kb-shelf-sort"
                    placeholder="标签筛选"
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
                          分组: {GROUP_MODE_LABEL[groupMode]} · 清除
                        </button>
                      ) : null}
                      {tagFilter !== 'all' ? (
                        <button
                          type="button"
                          className="kb-shelf-filter-pill"
                          onClick={() => setTagFilter('all')}
                        >
                          标签: {tagFilter} · 清除
                        </button>
                      ) : null}
                    </div>
                  ) : null}
                </div>
                {copyState === 'error' ? (
                  <div className="kb-shelf-copy-hint">复制失败，请检查剪贴板权限</div>
                ) : null}
              </div>
            ) : null}
            {items.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-[var(--border)] px-4 py-5 text-sm text-black/45 dark:text-white/45">
                从文内参考弹窗点击“加入文献篮”，这里会保存标题、作者、来源、DOI 和相关指标。
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
                      const subtitle = display.source
                      const duplicateCount = duplicateCountByIdentity[paperIdentity(item)] || 0
                      const trace = sourceTraceLabel(item)
                      const itemTags = normalizeShelfTags(item.tags)
                      const quality = qualityHints(item, display)
                      const noteText = String(item.note || '').trim()
                      const isFocused = item.key === focusedKey
                      const metrics = citeMetricSummary(item)
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
                              <div className="kb-shelf-item-title">{display.main}</div>
                              {display.authors ? (
                                <div className="kb-shelf-item-authors">{display.authors}</div>
                              ) : null}
                            </div>
                            <button
                              type="button"
                              className="kb-shelf-item-remove"
                              onClick={(event) => {
                                event.stopPropagation()
                                onRemove(item.key)
                              }}
                            >
                              移除
                            </button>
                          </div>
                          {subtitle ? (
                            <div className="kb-shelf-item-source">{subtitle}</div>
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
                                <span className="kb-shelf-repair-btn" aria-live="polite">
                                  {repairLoadingKey === item.key ? '自动修复中...' : '系统自动校正'}
                                </span>
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
                                <span className="kb-shelf-dup">可能重复 ×{duplicateCount}</span>
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
                                placeholder="+标签"
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
                                  <div className="kb-shelf-note-head">备注</div>
                                  <Input.TextArea
                                    className="kb-shelf-note-editor"
                                    autoSize={{ minRows: 2, maxRows: 4 }}
                                    maxLength={1200}
                                    placeholder="记录这篇文献对你有用的点..."
                                    value={item.note || ''}
                                    onChange={(event) => onUpdateNote(item.key, event.target.value)}
                                  />
                                  <div className="kb-shelf-note-actions">
                                    <button
                                      type="button"
                                      className="kb-shelf-note-link"
                                      onClick={() => setEditingNoteKeys((prev) => ({ ...prev, [item.key]: false }))}
                                    >
                                      完成
                                    </button>
                                  </div>
                                </>
                              ) : (
                                <div className="kb-shelf-note-inline">
                                  <div className="kb-shelf-note-preview">
                                    {noteText || '暂无备注'}
                                  </div>
                                  {isFocused ? (
                                    <button
                                      type="button"
                                      className="kb-shelf-note-link"
                                      onClick={() => setEditingNoteKeys((prev) => ({ ...prev, [item.key]: true }))}
                                    >
                                      {noteText ? '编辑备注' : '添加备注'}
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
                              <span className="kb-shelf-doi-empty">暂无 DOI 链接</span>
                            )}
                          </div>
                          {isFocused ? (
                            <div className="kb-shelf-summary">
                              {summaryLoadingKey === item.key ? (
                                <div className="kb-shelf-summary-text">正在生成学术概括...</div>
                              ) : item.summaryLine ? (
                                <>
                                  <div className="kb-shelf-summary-meta">
                                    <span className="kb-shelf-summary-head">学术概括</span>
                                    <span className="kb-shelf-summary-source">{summarySourceLabel(item.summarySource)}</span>
                                  </div>
                                  {(() => {
                                    const lines = splitSummary(item.summaryLine)
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
                                            {expanded ? '收起概括' : `展开剩余 ${lines.length - visibleLines.length} 条`}
                                          </button>
                                        ) : null}
                                      </>
                                    )
                                  })()}
                                </>
                              ) : (
                                <div className="kb-shelf-summary-empty">暂无可用学术概括</div>
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
                    未匹配到文献，请调整搜索词或排序方式。
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

