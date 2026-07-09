import { useCallback, useMemo, useState } from 'react'
import type { LibraryFileItem } from '../../api/library'
import type { CategoryCardItem, TagCardItem } from './LibraryTaxonomyViews'
import {
  hasConversionQualityIssue,
  matchesKeyword,
  toTextOptions,
  uniqueTextValues,
} from './libraryPageUtils'

export type ReadingStatusValue = LibraryFileItem['reading_status']

export type LibraryFileTabKey = 'pending' | 'converted' | 'all'

export type LibraryFileFilterState = {
  fileKeyword: string
  paperCategoryFilter: string
  paperTagFilter: string
  readingStatusFilter: ReadingStatusValue
  onlyUnread: boolean
  onlyUnclassified: boolean
  onlySuggested: boolean
  onlyQualityIssues: boolean
  qualityHistoryFocusNames: string[]
}

export type LibraryFileFilterModelInput = {
  S: Record<string, string>
  files: LibraryFileItem[]
  pendingFiles: LibraryFileItem[]
  convertedFiles: LibraryFileItem[]
  tabKey: LibraryFileTabKey
  state: Partial<LibraryFileFilterState>
}

type FilterFilesOptions = {
  ignoreCategoryFilter?: boolean
  ignoreTagFilter?: boolean
}

const PAPER_CATEGORY_PRESETS = [
  'NeRF',
  '3DGS',
  'SCI',
  'Diffusion',
  'Single-Photon Imaging',
  'Single-Pixel Imaging',
  'Inverse Imaging',
  'Survey',
  'Dataset',
  'Benchmark',
] as const

const DEFAULT_FILTER_STATE: LibraryFileFilterState = {
  fileKeyword: '',
  paperCategoryFilter: '',
  paperTagFilter: '',
  readingStatusFilter: '',
  onlyUnread: false,
  onlyUnclassified: false,
  onlySuggested: false,
  onlyQualityIssues: false,
  qualityHistoryFocusNames: [],
}

function normalizeFilterState(state: Partial<LibraryFileFilterState>): LibraryFileFilterState {
  return {
    ...DEFAULT_FILTER_STATE,
    ...state,
    fileKeyword: String(state.fileKeyword || ''),
    paperCategoryFilter: String(state.paperCategoryFilter || ''),
    paperTagFilter: String(state.paperTagFilter || ''),
    readingStatusFilter: (state.readingStatusFilter || '') as ReadingStatusValue,
    qualityHistoryFocusNames: Array.isArray(state.qualityHistoryFocusNames)
      ? state.qualityHistoryFocusNames.map((name) => String(name || '').trim()).filter(Boolean)
      : [],
  }
}

export function buildLibraryFileTextOptions(files: LibraryFileItem[]) {
  const paperCategoryFilterOptions = toTextOptions(
    uniqueTextValues(files.map((item) => item.paper_category))
      .sort((a, b) => a.localeCompare(b, 'en')),
  )
  const presetValues = Array.from(PAPER_CATEGORY_PRESETS)
  const dynamicCategoryValues = uniqueTextValues([
    ...files.map((item) => item.paper_category),
    ...files.map((item) => item.suggested_category),
  ]).sort((a, b) => a.localeCompare(b, 'en'))
  const presetKeys = new Set(presetValues.map((value) => value.toLowerCase()))
  const paperCategoryOptions = toTextOptions([
    ...presetValues,
    ...dynamicCategoryValues.filter((value) => !presetKeys.has(value.toLowerCase())),
  ])
  const paperTagFilterOptions = toTextOptions(
    uniqueTextValues(files.flatMap((item) => item.user_tags || []))
      .sort((a, b) => a.localeCompare(b, 'en')),
  )
  const paperTagOptions = toTextOptions(
    uniqueTextValues([
      ...files.flatMap((item) => item.user_tags || []),
      ...files.flatMap((item) => item.suggested_tags || []),
    ]).sort((a, b) => a.localeCompare(b, 'en')),
  )

  return {
    paperCategoryFilterOptions,
    paperCategoryOptions,
    paperTagFilterOptions,
    paperTagOptions,
  }
}

export function filterLibraryFiles(
  items: LibraryFileItem[],
  state: Partial<LibraryFileFilterState>,
  options: FilterFilesOptions = {},
) {
  const filterState = normalizeFilterState(state)
  const normalizedKeyword = filterState.fileKeyword.trim().toLowerCase()
  const paperTagFilter = filterState.paperTagFilter.toLowerCase()
  const qualityHistoryFocusSet = new Set(filterState.qualityHistoryFocusNames)

  return items.filter((item) => {
    if (qualityHistoryFocusSet.size > 0 && !qualityHistoryFocusSet.has(item.name)) return false
    const keywordText = [
      item.name,
      item.paper_category,
      item.reading_status,
      item.note,
      item.suggested_category,
      item.index_state,
      item.index_status,
      item.conversion_quality?.label,
      item.conversion_quality?.summary,
      ...(item.conversion_quality?.issues || []).flatMap((issue) => [issue.code, issue.label]),
      ...(item.user_tags || []),
      ...(item.suggested_tags || []),
    ]
      .map((part) => String(part || '').toLowerCase())
      .join(' ')
    if (!matchesKeyword(keywordText, normalizedKeyword)) return false
    if (!options.ignoreCategoryFilter && filterState.paperCategoryFilter && String(item.paper_category || '') !== filterState.paperCategoryFilter) return false
    if (!options.ignoreTagFilter && paperTagFilter && !(item.user_tags || []).some((tag) => String(tag || '').toLowerCase() === paperTagFilter)) return false
    if (filterState.readingStatusFilter && String(item.reading_status || '') !== filterState.readingStatusFilter) return false
    if (filterState.onlyUnread && String(item.reading_status || '') !== 'unread') return false
    if (filterState.onlyUnclassified && String(item.paper_category || '').trim()) return false
    if (filterState.onlySuggested && !item.has_suggestions) return false
    if (filterState.onlyQualityIssues && !hasConversionQualityIssue(item)) return false
    return true
  })
}

export function buildLibraryCategoryCards(
  items: LibraryFileItem[],
  S: Record<string, string>,
): CategoryCardItem[] {
  const groups = new Map<string, LibraryFileItem[]>()
  for (const item of items) {
    const rawLabel = String(item.paper_category || '').trim()
    const key = rawLabel ? `category:${rawLabel}` : 'category:__unclassified__'
    const list = groups.get(key)
    if (list) {
      list.push(item)
    } else {
      groups.set(key, [item])
    }
  }

  const out: CategoryCardItem[] = []
  for (const [key, groupItems] of groups.entries()) {
    const label = key === 'category:__unclassified__'
      ? S.lib_category_unclassified
      : String(groupItems[0]?.paper_category || '').trim()
    const tagCounts = new Map<string, number>()
    for (const item of groupItems) {
      for (const tag of item.user_tags || []) {
        const value = String(tag || '').trim()
        if (!value) continue
        tagCounts.set(value, (tagCounts.get(value) || 0) + 1)
      }
    }
    const commonTags = Array.from(tagCounts.entries())
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0], 'en'))
      .slice(0, 4)
      .map(([tag]) => tag)
    out.push({
      key,
      label: label || S.lib_category_unclassified,
      count: groupItems.length,
      unreadCount: groupItems.filter((item) => item.reading_status === 'unread').length,
      convertedCount: groupItems.filter((item) => item.category === 'converted').length,
      pendingCount: groupItems.filter((item) => item.category === 'pending').length,
      commonTags,
      recentPapers: groupItems.slice(0, 3).map((item) => item.name),
    })
  }

  return out.sort((a, b) => b.count - a.count || a.label.localeCompare(b.label, 'en'))
}

export function buildLibraryTagCards(
  items: LibraryFileItem[],
  S: Record<string, string>,
): TagCardItem[] {
  const groups = new Map<string, LibraryFileItem[]>()
  for (const item of items) {
    for (const rawTag of item.user_tags || []) {
      const label = String(rawTag || '').trim()
      if (!label) continue
      const key = label.toLowerCase()
      const list = groups.get(key)
      if (list) {
        list.push(item)
      } else {
        groups.set(key, [item])
      }
    }
  }

  const out: TagCardItem[] = []
  for (const [key, groupItems] of groups.entries()) {
    const label = groupItems.find((item) => (item.user_tags || []).some((tag) => String(tag || '').trim().toLowerCase() === key))
      ?.user_tags.find((tag) => String(tag || '').trim().toLowerCase() === key) || key
    const categoryCounts = new Map<string, number>()
    for (const item of groupItems) {
      const category = String(item.paper_category || '').trim() || S.lib_category_unclassified
      categoryCounts.set(category, (categoryCounts.get(category) || 0) + 1)
    }
    out.push({
      key,
      label: String(label),
      count: groupItems.length,
      unreadCount: groupItems.filter((item) => item.reading_status === 'unread').length,
      categories: Array.from(categoryCounts.entries())
        .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0], 'en'))
        .slice(0, 3)
        .map(([category]) => category),
      recentPapers: groupItems.slice(0, 3).map((item) => item.name),
    })
  }

  return out.sort((a, b) => b.count - a.count || a.label.localeCompare(b.label, 'en'))
}

export function buildLibraryFileFilterModel({
  S,
  files,
  pendingFiles,
  convertedFiles,
  tabKey,
  state,
}: LibraryFileFilterModelInput) {
  const filterState = normalizeFilterState(state)
  const visiblePending = filterLibraryFiles(pendingFiles, filterState)
  const visibleConverted = filterLibraryFiles(convertedFiles, filterState)
  const visibleAll = filterLibraryFiles(files, filterState)
  const visibleAllWithoutCategory = filterLibraryFiles(files, filterState, { ignoreCategoryFilter: true })
  const visibleAllWithoutTag = filterLibraryFiles(files, filterState, { ignoreTagFilter: true })
  const normalizedKeyword = filterState.fileKeyword.trim().toLowerCase()
  const hasActiveTaxonomyFilters = Boolean(
    normalizedKeyword
    || filterState.paperCategoryFilter
    || filterState.paperTagFilter
    || filterState.readingStatusFilter
    || filterState.onlyUnread
    || filterState.onlyUnclassified
    || filterState.onlySuggested
    || filterState.onlyQualityIssues
    || filterState.qualityHistoryFocusNames.length > 0
  )
  const activeTaxonomyFilterCount = [
    normalizedKeyword,
    filterState.paperCategoryFilter,
    filterState.paperTagFilter,
    filterState.readingStatusFilter,
    filterState.onlyUnread ? 'onlyUnread' : '',
    filterState.onlyUnclassified ? 'onlyUnclassified' : '',
    filterState.onlySuggested ? 'onlySuggested' : '',
    filterState.onlyQualityIssues ? 'onlyQualityIssues' : '',
    filterState.qualityHistoryFocusNames.length > 0 ? 'qualityHistoryFocus' : '',
  ].filter(Boolean).length
  const currentListItems = tabKey === 'pending'
    ? visiblePending
    : tabKey === 'converted'
      ? visibleConverted
      : visibleAll

  return {
    ...buildLibraryFileTextOptions(files),
    hasActiveTaxonomyFilters,
    activeTaxonomyFilterCount,
    visiblePending,
    visibleConverted,
    visibleAll,
    categoryCards: buildLibraryCategoryCards(visibleAllWithoutCategory, S),
    tagCards: buildLibraryTagCards(visibleAllWithoutTag, S),
    currentListItems,
  }
}

export function useLibraryFileFilters({
  S,
  files,
  pendingFiles,
  convertedFiles,
  tabKey,
}: Omit<LibraryFileFilterModelInput, 'state'>) {
  const [fileKeyword, setFileKeyword] = useState('')
  const [paperCategoryFilter, setPaperCategoryFilter] = useState('')
  const [paperTagFilter, setPaperTagFilter] = useState('')
  const [readingStatusFilter, setReadingStatusFilter] = useState<ReadingStatusValue>('')
  const [onlyUnread, setOnlyUnread] = useState(false)
  const [onlyUnclassified, setOnlyUnclassified] = useState(false)
  const [onlySuggested, setOnlySuggested] = useState(false)
  const [onlyQualityIssues, setOnlyQualityIssues] = useState(false)
  const [qualityHistoryFocusNames, setQualityHistoryFocusNames] = useState<string[]>([])

  const state = useMemo<LibraryFileFilterState>(() => ({
    fileKeyword,
    paperCategoryFilter,
    paperTagFilter,
    readingStatusFilter,
    onlyUnread,
    onlyUnclassified,
    onlySuggested,
    onlyQualityIssues,
    qualityHistoryFocusNames,
  }), [
    fileKeyword,
    onlyQualityIssues,
    onlySuggested,
    onlyUnclassified,
    onlyUnread,
    paperCategoryFilter,
    paperTagFilter,
    qualityHistoryFocusNames,
    readingStatusFilter,
  ])
  const model = useMemo(
    () => buildLibraryFileFilterModel({
      S,
      files,
      pendingFiles,
      convertedFiles,
      tabKey,
      state,
    }),
    [S, convertedFiles, files, pendingFiles, state, tabKey],
  )

  const applyPaperCategoryFilter = useCallback((value: string) => {
    setOnlyUnclassified(false)
    setPaperCategoryFilter(value)
  }, [])

  const applyPaperTagFilter = useCallback((value: string) => {
    setPaperTagFilter(value)
  }, [])

  const clearTaxonomyFilters = useCallback(() => {
    setFileKeyword('')
    setPaperCategoryFilter('')
    setPaperTagFilter('')
    setReadingStatusFilter('')
    setOnlyUnread(false)
    setOnlyUnclassified(false)
    setOnlySuggested(false)
    setOnlyQualityIssues(false)
    setQualityHistoryFocusNames([])
  }, [])

  const toggleOnlyUnclassified = useCallback(() => {
    setOnlyUnclassified((value) => {
      const next = !value
      if (next) setPaperCategoryFilter('')
      return next
    })
  }, [])

  const selectUnclassifiedCategory = useCallback(() => {
    setPaperCategoryFilter('')
    setOnlyUnclassified(true)
  }, [])

  return {
    ...state,
    ...model,
    setFileKeyword,
    setPaperCategoryFilter,
    setPaperTagFilter,
    setReadingStatusFilter,
    setOnlyUnread,
    setOnlyUnclassified,
    setOnlySuggested,
    setOnlyQualityIssues,
    setQualityHistoryFocusNames,
    applyPaperCategoryFilter,
    applyPaperTagFilter,
    clearTaxonomyFilters,
    toggleOnlyUnclassified,
    selectUnclassifiedCategory,
  }
}
