
import { useEffect, useMemo, useState, type ReactNode } from 'react'
import {
  Upload,
  AutoComplete,
  Button,
  List,
  Drawer,
  message,
  Progress,
  Select,
  Typography,
  Tabs,
  Tag,
  Switch,
  Space,
  Empty,
  Input,
  Card,
  Checkbox,
  Collapse,
  Alert,
  Tooltip,
  Dropdown,
  Modal,
  Segmented,
} from 'antd'
import {
  UploadOutlined,
  ReloadOutlined,
  StopOutlined,
  FolderOpenOutlined,
  DeleteOutlined,
  SaveOutlined,
  SearchOutlined,
  CheckOutlined,
  ClearOutlined,
  MoreOutlined,
  CopyOutlined,
  LockOutlined,
  ApiOutlined,
  ExclamationCircleOutlined,
} from '@ant-design/icons'
import type { LibraryFileItem, RenameSuggestionItem } from '../api/library'
import { libraryApi } from '../api/library'
import { useChatStore } from '../stores/chatStore'
import { settingsApi } from '../api/settings'
import { useLibraryStore } from '../stores/libraryStore'
import { useSettingsStore } from '../stores/settingsStore'
import VirtualList from 'rc-virtual-list'
import { useNavigate } from 'react-router-dom'

const { Text } = Typography
const { Dragger } = Upload
const FILE_VIRTUAL_THRESHOLD = 60
const FILE_VIRTUAL_HEIGHT = 620
const FILE_VIRTUAL_ROW_HEIGHT = 88

type FileTabKey = 'pending' | 'converted' | 'all'
type LibraryBrowseMode = 'list' | 'categories' | 'tags'
type DraftStatus = 'queued' | 'inspecting' | 'ready' | 'saving' | 'saved' | 'error'
type UploadDraftFilter = 'all' | 'todo' | 'error' | 'dup_error' | 'saved'
type UploadErrorReason = 'all' | 'duplicate' | 'path' | 'permission' | 'network' | 'other'

type UploadDraft = {
  key: string
  file: File
  name: string
  selected: boolean
  stem: string
  status: DraftStatus
  displayName: string
  note: string
}

const CONVERT_MODE = 'balanced'
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
const TAG_INPUT_SEPARATORS = [',', '，', ';', '；']
const READING_STATUS_OPTIONS = [
  { value: '', label: '全部阅读状态' },
  { value: 'unread', label: '未读' },
  { value: 'reading', label: '在读' },
  { value: 'done', label: '已读' },
  { value: 'revisit', label: '待回看' },
] as const

type ReadingStatusValue = '' | 'unread' | 'reading' | 'done' | 'revisit'
type LibraryMetaDraft = {
  paper_category: string
  reading_status: ReadingStatusValue
  note: string
  user_tags: string[]
}

type LibraryBatchMetaDraft = {
  apply_paper_category: boolean
  paper_category: string
  apply_reading_status: boolean
  reading_status: ReadingStatusValue
  add_tags: string[]
  remove_tags: string[]
}

type CategoryCardItem = {
  key: string
  label: string
  count: number
  unreadCount: number
  convertedCount: number
  pendingCount: number
  commonTags: string[]
  recentPapers: string[]
}

type TagCardItem = {
  key: string
  label: string
  count: number
  unreadCount: number
  categories: string[]
  recentPapers: string[]
}

type FilterFilesOptions = {
  ignoreCategoryFilter?: boolean
  ignoreTagFilter?: boolean
}

type TextOption = {
  value?: string | number
  label?: ReactNode
}

const SCOPE_OPTIONS = [
  { value: '200', label: '最近 200 篇' },
  { value: '1000', label: '最近 1000 篇' },
  { value: 'all', label: '全部' },
]

const RENAME_SCOPE_OPTIONS = [
  { value: '30', label: '最近 30 篇' },
  { value: '50', label: '最近 50 篇' },
  { value: '100', label: '最近 100 篇' },
  { value: 'all', label: '全部' },
]

const DRAFT_STATUS_TEXT: Record<DraftStatus, string> = {
  queued: '待处理',
  inspecting: '扫描中',
  ready: '待保存',
  saving: '保存中',
  saved: '已保存',
  error: '失败',
}

const FAILED_REASON_META: Record<Exclude<UploadErrorReason, 'all'>, { label: string, icon: ReactNode }> = {
  duplicate: { label: '重复文件', icon: <CopyOutlined /> },
  path: { label: '路径/目录', icon: <FolderOpenOutlined /> },
  permission: { label: '权限', icon: <LockOutlined /> },
  network: { label: '网络', icon: <ApiOutlined /> },
  other: { label: '其他', icon: <ExclamationCircleOutlined /> },
}

function fileTag(item: LibraryFileItem) {
  if (item.task_state === 'running') return { color: 'processing' as const, text: '转换中' }
  if (item.task_state === 'queued') return { color: 'warning' as const, text: `排队中${item.queue_pos > 0 ? ` #${item.queue_pos}` : ''}` }
  return item.category === 'converted'
    ? { color: 'success' as const, text: '已转换' }
    : { color: 'default' as const, text: '待转换' }
}

function matchesKeyword(name: string, keyword: string) {
  if (!keyword) return true
  return name.toLowerCase().includes(keyword)
}

function readingStatusLabel(value: string) {
  if (value === 'unread') return '未读'
  if (value === 'reading') return '在读'
  if (value === 'done') return '已读'
  if (value === 'revisit') return '待回看'
  return ''
}

function stripKnownSourceExt(name: string) {
  return String(name || '')
    .replace(/\.en\.md$/i, '')
    .replace(/\.md$/i, '')
    .replace(/\.pdf$/i, '')
    .trim()
}

function isDuplicateFailure(note: string) {
  const t = String(note || '').toLowerCase()
  return t.includes('重复') || t.includes('duplicate') || t.includes('already exists') || t.includes('已存在')
}

function classifyFailedReason(note: string) {
  const t = String(note || '').toLowerCase()
  if (isDuplicateFailure(t)) return 'duplicate'
  if (t.includes('目录') || t.includes('路径') || t.includes('path') || t.includes('dir')) return 'path'
  if (t.includes('权限') || t.includes('permission') || t.includes('denied')) return 'permission'
  if (t.includes('网络') || t.includes('timeout') || t.includes('network')) return 'network'
  return 'other'
}

function normalizeTextValue(value: unknown) {
  return String(value || '').replace(/\s+/g, ' ').trim()
}

function normalizeTextList(values: unknown[]) {
  const out: string[] = []
  const seen = new Set<string>()
  for (const value of values) {
    const clean = normalizeTextValue(value)
    if (!clean) continue
    const key = clean.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    out.push(clean)
  }
  return out
}

function uniqueTextValues(values: Iterable<unknown>) {
  const out: string[] = []
  const seen = new Set<string>()
  for (const value of values) {
    const clean = normalizeTextValue(value)
    if (!clean) continue
    const key = clean.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    out.push(clean)
  }
  return out
}

function toTextOptions(values: string[]) {
  return values.map((value) => ({ value, label: value }))
}

function optionMatchesInput(input: string, option?: TextOption) {
  const needle = normalizeTextValue(input).toLowerCase()
  if (!needle) return true
  const hay = normalizeTextValue(option?.value || option?.label || '').toLowerCase()
  return hay.includes(needle)
}

export default function LibraryPage() {
  const store = useLibraryStore()
  const createPaperGuideConversation = useChatStore((s) => s.createPaperGuideConversation)
  const nav = useNavigate()

  const settingsLoaded = useSettingsStore((s) => s.loaded)
  const settingsPdfDir = useSettingsStore((s) => s.pdfDir)
  const settingsMdDir = useSettingsStore((s) => s.mdDir)
  const updateSettings = useSettingsStore((s) => s.update)

  const [scope, setScope] = useState('200')
  const [tabKey, setTabKey] = useState<FileTabKey>('pending')
  const [browseMode, setBrowseMode] = useState<LibraryBrowseMode>('list')
  const [replaceMd, setReplaceMd] = useState(true)
  const [onlyBusyFiles, setOnlyBusyFiles] = useState(false)
  const [fileKeyword, setFileKeyword] = useState('')
  const [paperCategoryFilter, setPaperCategoryFilter] = useState('')
  const [paperTagFilter, setPaperTagFilter] = useState('')
  const [readingStatusFilter, setReadingStatusFilter] = useState<ReadingStatusValue>('')
  const [onlyUnread, setOnlyUnread] = useState(false)
  const [onlyUnclassified, setOnlyUnclassified] = useState(false)
  const [onlySuggested, setOnlySuggested] = useState(false)
  const [metaDrawerOpen, setMetaDrawerOpen] = useState(false)
  const [metaSaving, setMetaSaving] = useState(false)
  const [metaSuggestionSaving, setMetaSuggestionSaving] = useState(false)
  const [metaItem, setMetaItem] = useState<LibraryFileItem | null>(null)
  const [metaDraft, setMetaDraft] = useState<LibraryMetaDraft>({
    paper_category: '',
    reading_status: '',
    note: '',
    user_tags: [],
  })
  const [selectedLibraryNames, setSelectedLibraryNames] = useState<Record<string, boolean>>({})
  const [batchDrawerOpen, setBatchDrawerOpen] = useState(false)
  const [batchSaving, setBatchSaving] = useState(false)
  const [batchDraft, setBatchDraft] = useState<LibraryBatchMetaDraft>({
    apply_paper_category: false,
    paper_category: '',
    apply_reading_status: false,
    reading_status: '',
    add_tags: [],
    remove_tags: [],
  })

  const [pdfDirDraft, setPdfDirDraft] = useState('')
  const [mdDirDraft, setMdDirDraft] = useState('')
  const [savingDirs, setSavingDirs] = useState(false)
  const [pickingDir, setPickingDir] = useState<'pdf' | 'md' | null>(null)
  const [dirTouched, setDirTouched] = useState(false)

  const [uploadDrafts, setUploadDrafts] = useState<UploadDraft[]>([])
  const [uploadUseLlm, setUploadUseLlm] = useState(true)
  const [uploadDraftFilter, setUploadDraftFilter] = useState<UploadDraftFilter>('all')
  const [uploadErrorReason, setUploadErrorReason] = useState<UploadErrorReason>('all')
  const [uploadInspecting, setUploadInspecting] = useState(false)
  const [uploadSaving, setUploadSaving] = useState(false)

  const [renameScope, setRenameScope] = useState('30')
  const [renameUseLlm, setRenameUseLlm] = useState(false)
  const [renameOnlyDiff, setRenameOnlyDiff] = useState(true)
  const [renameLoading, setRenameLoading] = useState(false)
  const [renameApplying, setRenameApplying] = useState(false)
  const [renameItems, setRenameItems] = useState<RenameSuggestionItem[]>([])
  const [renameSelected, setRenameSelected] = useState<Record<string, boolean>>({})
  const [renameOverrides, setRenameOverrides] = useState<Record<string, string>>({})
  const [suggestionsRefreshing, setSuggestionsRefreshing] = useState(false)

  const uploadLocked = store.converting || Boolean(store.refSync?.running)
  const normalizedKeyword = fileKeyword.trim().toLowerCase()

  const dirDirty = useMemo(
    () =>
      pdfDirDraft.trim() !== String(settingsPdfDir || '').trim()
      || mdDirDraft.trim() !== String(settingsMdDir || '').trim(),
    [pdfDirDraft, mdDirDraft, settingsPdfDir, settingsMdDir],
  )

  const pendingFiles = useMemo(() => store.files.filter((x) => x.category === 'pending'), [store.files])
  const convertedFiles = useMemo(() => store.files.filter((x) => x.category === 'converted'), [store.files])
  const renameVisible = useMemo(() => (renameOnlyDiff ? renameItems.filter((x) => x.diff) : renameItems), [renameOnlyDiff, renameItems])
  const selectedUploadCount = useMemo(() => uploadDrafts.filter((x) => x.selected).length, [uploadDrafts])
  const selectedRenameCount = useMemo(() => renameItems.filter((x) => renameSelected[x.name]).length, [renameItems, renameSelected])
  const failedUploadDrafts = useMemo(() => uploadDrafts.filter((x) => x.status === 'error'), [uploadDrafts])
  const duplicateFailedDrafts = useMemo(
    () => failedUploadDrafts.filter((x) => isDuplicateFailure(x.note)),
    [failedUploadDrafts],
  )
  const failedUploadNotes = useMemo(
    () => Array.from(new Set(failedUploadDrafts.map((x) => String(x.note || '').trim()).filter(Boolean))).slice(0, 3),
    [failedUploadDrafts],
  )
  const failedReasonBuckets = useMemo(() => {
    const counter = new Map<Exclude<UploadErrorReason, 'all'>, number>()
    for (const item of failedUploadDrafts) {
      const key = classifyFailedReason(item.note) as Exclude<UploadErrorReason, 'all'>
      counter.set(key, (counter.get(key) || 0) + 1)
    }
    return Array.from(counter.entries())
      .map(([key, count]) => ({ key, count, label: FAILED_REASON_META[key].label }))
      .sort((a, b) => b.count - a.count)
  }, [failedUploadDrafts])
  const filteredUploadDrafts = useMemo(() => {
    const withReason = (items: UploadDraft[]) => (
      uploadErrorReason === 'all'
        ? items
        : items.filter((x) => classifyFailedReason(x.note) === uploadErrorReason)
    )
    if (uploadDraftFilter === 'all') return uploadDrafts
    if (uploadDraftFilter === 'error') return withReason(uploadDrafts.filter((x) => x.status === 'error'))
    if (uploadDraftFilter === 'dup_error') return withReason(uploadDrafts.filter((x) => x.status === 'error' && isDuplicateFailure(x.note)))
    if (uploadDraftFilter === 'saved') return uploadDrafts.filter((x) => x.status === 'saved')
    return uploadDrafts.filter((x) => ['queued', 'inspecting', 'ready', 'saving'].includes(x.status))
  }, [uploadDrafts, uploadDraftFilter, uploadErrorReason])
  const uploadDraftFilterOptions = useMemo(
    () => [
      { value: 'all', label: `全部 (${uploadDrafts.length})` },
      { value: 'todo', label: `待处理 (${uploadDrafts.filter((x) => ['queued', 'inspecting', 'ready', 'saving'].includes(x.status)).length})` },
      { value: 'error', label: `失败 (${uploadDrafts.filter((x) => x.status === 'error').length})` },
      { value: 'dup_error', label: `重复失败 (${uploadDrafts.filter((x) => x.status === 'error' && isDuplicateFailure(x.note)).length})` },
      { value: 'saved', label: `已保存 (${uploadDrafts.filter((x) => x.status === 'saved').length})` },
    ],
    [uploadDrafts],
  )
  const activeErrorReasonText = useMemo(() => {
    const map: Record<UploadErrorReason, string> = {
      all: '全部原因',
      duplicate: FAILED_REASON_META.duplicate.label,
      path: FAILED_REASON_META.path.label,
      permission: FAILED_REASON_META.permission.label,
      network: FAILED_REASON_META.network.label,
      other: FAILED_REASON_META.other.label,
    }
    return map[uploadErrorReason]
  }, [uploadErrorReason])
  const convertPercent = useMemo(
    () => (store.progress && store.progress.total > 0
      ? Math.round((store.progress.completed / store.progress.total) * 100)
      : 0),
    [store.progress],
  )
  const convertPageProgress = useMemo(() => {
    const done0 = Number(store.progress?.curPageDone || 0)
    const total0 = Number(store.progress?.curPageTotal || 0)
    if (total0 > 0) return { done: Math.max(0, done0), total: Math.max(0, total0) }
    const msg = String(store.progress?.curPageMsg || '')
    const m = msg.match(/\b(\d{1,4})\s*\/\s*(\d{1,4})\b/)
    if (!m) return { done: 0, total: 0 }
    const done = Number(m[1] || 0)
    const total = Number(m[2] || 0)
    if (!Number.isFinite(done) || !Number.isFinite(total) || total <= 0) return { done: 0, total: 0 }
    return { done: Math.max(0, done), total: Math.max(0, total) }
  }, [store.progress])
  const convertPagePercent = useMemo(
    () => (convertPageProgress.total > 0
      ? Math.round((convertPageProgress.done / Math.max(1, convertPageProgress.total)) * 100)
      : 0),
    [convertPageProgress],
  )
  const refSyncPercent = useMemo(
    () => (store.refSync && store.refSync.docsTotal > 0
      ? Math.round((store.refSync.docsDone / Math.max(1, store.refSync.docsTotal)) * 100)
      : 0),
    [store.refSync],
  )
  const showStickyStatus = Boolean((store.converting && store.progress) || store.refSync?.running)

  const paperCategoryFilterOptions = useMemo(() => {
    const values = uniqueTextValues(store.files.map((item) => item.paper_category))
      .sort((a, b) => a.localeCompare(b, 'en'))
    return toTextOptions(values)
  }, [store.files])

  const paperCategoryOptions = useMemo(() => {
    const presetValues = Array.from(PAPER_CATEGORY_PRESETS)
    const dynamicValues = uniqueTextValues([
      ...store.files.map((item) => item.paper_category),
      ...store.files.map((item) => item.suggested_category),
    ]).sort((a, b) => a.localeCompare(b, 'en'))
    const presetKeys = new Set(presetValues.map((value) => value.toLowerCase()))
    const merged = [
      ...presetValues,
      ...dynamicValues.filter((value) => !presetKeys.has(value.toLowerCase())),
    ]
    return toTextOptions(merged)
  }, [store.files])

  const paperTagFilterOptions = useMemo(() => {
    const values = uniqueTextValues(store.files.flatMap((item) => item.user_tags || []))
      .sort((a, b) => a.localeCompare(b, 'en'))
    return toTextOptions(values)
  }, [store.files])

  const paperTagOptions = useMemo(() => {
    const values = uniqueTextValues([
      ...store.files.flatMap((item) => item.user_tags || []),
      ...store.files.flatMap((item) => item.suggested_tags || []),
    ]).sort((a, b) => a.localeCompare(b, 'en'))
    return toTextOptions(values)
  }, [store.files])

  const applyPaperCategoryFilter = (value: string) => {
    setOnlyUnclassified(false)
    setPaperCategoryFilter(value)
  }

  const applyPaperTagFilter = (value: string) => {
    setPaperTagFilter(value)
  }

  const clearTaxonomyFilters = () => {
    setFileKeyword('')
    setPaperCategoryFilter('')
    setPaperTagFilter('')
    setReadingStatusFilter('')
    setOnlyUnread(false)
    setOnlyUnclassified(false)
    setOnlySuggested(false)
  }

  const hasActiveTaxonomyFilters = Boolean(
    normalizedKeyword
    || paperCategoryFilter
    || paperTagFilter
    || readingStatusFilter
    || onlyUnread
    || onlyUnclassified
    || onlySuggested
  )
  const activeTaxonomyFilterCount = [
    normalizedKeyword,
    paperCategoryFilter,
    paperTagFilter,
    readingStatusFilter,
    onlyUnread ? 'onlyUnread' : '',
    onlyUnclassified ? 'onlyUnclassified' : '',
    onlySuggested ? 'onlySuggested' : '',
  ].filter(Boolean).length

  const filterFiles = (items: LibraryFileItem[], options: FilterFilesOptions = {}) =>
    items.filter((item) => {
      const keywordText = [
        item.name,
        item.paper_category,
        item.reading_status,
        item.note,
        item.suggested_category,
        ...(item.user_tags || []),
        ...(item.suggested_tags || []),
      ]
        .map((part) => String(part || '').toLowerCase())
        .join(' ')
      if (!matchesKeyword(keywordText, normalizedKeyword)) return false
      if (onlyBusyFiles) return item.task_state !== 'idle'
      if (!options.ignoreCategoryFilter && paperCategoryFilter && String(item.paper_category || '') !== paperCategoryFilter) return false
      if (!options.ignoreTagFilter && paperTagFilter && !(item.user_tags || []).some((tag) => String(tag || '').toLowerCase() === paperTagFilter.toLowerCase())) return false
      if (readingStatusFilter && String(item.reading_status || '') !== readingStatusFilter) return false
      if (onlyUnread && String(item.reading_status || '') !== 'unread') return false
      if (onlyUnclassified && String(item.paper_category || '').trim()) return false
      if (onlySuggested && !item.has_suggestions) return false
      return true
    })

  const visiblePending = useMemo(
    () => filterFiles(pendingFiles),
    [pendingFiles, normalizedKeyword, onlyBusyFiles, paperCategoryFilter, paperTagFilter, readingStatusFilter, onlyUnread, onlyUnclassified, onlySuggested],
  )
  const visibleConverted = useMemo(
    () => filterFiles(convertedFiles),
    [convertedFiles, normalizedKeyword, onlyBusyFiles, paperCategoryFilter, paperTagFilter, readingStatusFilter, onlyUnread, onlyUnclassified, onlySuggested],
  )
  const visibleAll = useMemo(
    () => filterFiles(store.files),
    [store.files, normalizedKeyword, onlyBusyFiles, paperCategoryFilter, paperTagFilter, readingStatusFilter, onlyUnread, onlyUnclassified, onlySuggested],
  )
  const visibleAllWithoutCategory = useMemo(
    () => filterFiles(store.files, { ignoreCategoryFilter: true }),
    [store.files, normalizedKeyword, onlyBusyFiles, paperTagFilter, readingStatusFilter, onlyUnread, onlyUnclassified, onlySuggested],
  )
  const visibleAllWithoutTag = useMemo(
    () => filterFiles(store.files, { ignoreTagFilter: true }),
    [store.files, normalizedKeyword, onlyBusyFiles, paperCategoryFilter, readingStatusFilter, onlyUnread, onlyUnclassified, onlySuggested],
  )

  const categoryCards = useMemo<CategoryCardItem[]>(() => {
    const groups = new Map<string, LibraryFileItem[]>()
    for (const item of visibleAllWithoutCategory) {
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
    for (const [key, items] of groups.entries()) {
      const label = key === 'category:__unclassified__'
        ? '未分类'
        : String(items[0]?.paper_category || '').trim()
      const tagCounts = new Map<string, number>()
      for (const item of items) {
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
        label: label || '未分类',
        count: items.length,
        unreadCount: items.filter((item) => item.reading_status === 'unread').length,
        convertedCount: items.filter((item) => item.category === 'converted').length,
        pendingCount: items.filter((item) => item.category === 'pending').length,
        commonTags,
        recentPapers: items.slice(0, 3).map((item) => item.name),
      })
    }

    return out.sort((a, b) => b.count - a.count || a.label.localeCompare(b.label, 'en'))
  }, [visibleAllWithoutCategory])

  const tagCards = useMemo<TagCardItem[]>(() => {
    const groups = new Map<string, LibraryFileItem[]>()
    for (const item of visibleAllWithoutTag) {
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
    for (const [key, items] of groups.entries()) {
      const label = items.find((item) => (item.user_tags || []).some((tag) => String(tag || '').trim().toLowerCase() === key))
        ?.user_tags.find((tag) => String(tag || '').trim().toLowerCase() === key) || key
      const categoryCounts = new Map<string, number>()
      for (const item of items) {
        const category = String(item.paper_category || '').trim() || '未分类'
        categoryCounts.set(category, (categoryCounts.get(category) || 0) + 1)
      }
      out.push({
        key,
        label: String(label),
        count: items.length,
        unreadCount: items.filter((item) => item.reading_status === 'unread').length,
        categories: Array.from(categoryCounts.entries())
          .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0], 'en'))
          .slice(0, 3)
          .map(([category]) => category),
        recentPapers: items.slice(0, 3).map((item) => item.name),
      })
    }

    return out.sort((a, b) => b.count - a.count || a.label.localeCompare(b.label, 'en'))
  }, [visibleAllWithoutTag])

  const currentListItems = useMemo(() => {
    if (tabKey === 'pending') return visiblePending
    if (tabKey === 'converted') return visibleConverted
    return visibleAll
  }, [tabKey, visiblePending, visibleConverted, visibleAll])

  const selectedLibraryNamesList = useMemo(
    () => Object.keys(selectedLibraryNames).filter((name) => Boolean(selectedLibraryNames[name])),
    [selectedLibraryNames],
  )

  const selectedLibraryCount = selectedLibraryNamesList.length
  const metaSuggestionCount = (metaItem?.suggested_category ? 1 : 0) + (metaItem?.suggested_tags?.length || 0)
  const metaDraftCategory = normalizeTextValue(metaDraft.paper_category)
  const metaDraftTags = normalizeTextList(metaDraft.user_tags)

  useEffect(() => {
    void store.loadFiles(scope)
    if (store.converting && !store.sseController) store.startProgressStream()
    if (!store.refSyncController) store.startRefSyncStream()
    return () => {
      store.stopProgressStream()
      store.stopRefSyncStream()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!settingsLoaded || dirTouched) return
    setPdfDirDraft(String(settingsPdfDir || ''))
    setMdDirDraft(String(settingsMdDir || ''))
  }, [settingsLoaded, settingsPdfDir, settingsMdDir, dirTouched])

  useEffect(() => {
    const existing = new Set(store.files.map((item) => item.name))
    setSelectedLibraryNames((cur) => {
      let changed = false
      const next: Record<string, boolean> = {}
      for (const [name, selected] of Object.entries(cur)) {
        if (!selected) continue
        if (!existing.has(name)) {
          changed = true
          continue
        }
        next[name] = true
      }
      return changed ? next : cur
    })
  }, [store.files])

  const saveDirs = async () => {
    if (!pdfDirDraft.trim() || !mdDirDraft.trim()) {
      message.warning('PDF/MD 目录不能为空')
      return false
    }
    setSavingDirs(true)
    try {
      await updateSettings({ pdfDir: pdfDirDraft.trim(), mdDir: mdDirDraft.trim() })
      setDirTouched(false)
      message.success('目录已保存')
      await store.loadFiles(scope)
      return true
    } catch (err) {
      message.error(err instanceof Error ? err.message : '目录保存失败')
      return false
    } finally {
      setSavingDirs(false)
    }
  }

  const ensureDirsReady = async () => {
    if (!dirDirty) return true
    return saveDirs()
  }

  const openFolder = async (target: 'pdf_dir' | 'md_dir') => {
    const ready = await ensureDirsReady()
    if (!ready) return
    await store.openFile('', target)
  }

  const pickDir = async (target: 'pdf' | 'md') => {
    const initial = target === 'pdf' ? pdfDirDraft : mdDirDraft
    setPickingDir(target)
    try {
      const res = await settingsApi.pickDir(target, initial)
      if (!res.ok || !res.path) {
        message.info('未选择目录')
        return
      }
      setDirTouched(true)
      if (target === 'pdf') setPdfDirDraft(res.path)
      else setMdDirDraft(res.path)
    } catch (err) {
      message.error(err instanceof Error ? err.message : '打开目录选择器失败')
    } finally {
      setPickingDir(null)
    }
  }

  const addDrafts = (files: File[]) => {
    setUploadDrafts((cur) => {
      const seen = new Set(cur.map((x) => x.key))
      const next = [...cur]
      for (const file of files) {
        const key = `${file.name}:${file.size}:${file.lastModified}`
        if (seen.has(key)) continue
        seen.add(key)
        next.push({
          key,
          file,
          name: file.name,
          selected: true,
          stem: file.name.replace(/\.pdf$/i, ''),
          status: 'queued',
          displayName: file.name,
          note: '',
        })
      }
      return next
    })
  }

  const inspectDraft = async (key: string) => {
    const ready = await ensureDirsReady()
    if (!ready) return
    const target = uploadDrafts.find((x) => x.key === key)
    if (!target) return
    setUploadDrafts((cur) => cur.map((x) => (x.key === key ? { ...x, status: 'inspecting', note: '' } : x)))
    try {
      const res = await libraryApi.inspectUpload(target.file, uploadUseLlm)
      setUploadDrafts((cur) => cur.map((x) => {
        if (x.key !== key) return x
        return {
          ...x,
          stem: res.suggested_stem || x.stem,
          displayName: res.display_full_name || x.displayName,
          status: res.duplicate ? 'error' : 'ready',
          note: res.duplicate ? `重复：${String(res.existing || '')}` : '扫描完成',
        }
      }))
    } catch (err) {
      setUploadDrafts((cur) => cur.map((x) => (
        x.key === key
          ? { ...x, status: 'error', note: err instanceof Error ? err.message : '扫描失败' }
          : x
      )))
    }
  }

  const inspectSelectedDrafts = async () => {
    const selected = uploadDrafts.filter((x) => x.selected && x.status !== 'inspecting')
    if (!selected.length) {
      message.info('请先选择要扫描的文件')
      return
    }
    setUploadInspecting(true)
    try {
      for (const x of selected) {
        // eslint-disable-next-line no-await-in-loop
        await inspectDraft(x.key)
      }
      message.success(`已扫描 ${selected.length} 个文件`)
    } finally {
      setUploadInspecting(false)
    }
  }

  const saveDraft = async (key: string, convertNow: boolean) => {
    const ready = await ensureDirsReady()
    if (!ready) return
    const target = uploadDrafts.find((x) => x.key === key)
    if (!target) return
    setUploadDrafts((cur) => cur.map((x) => (x.key === key ? { ...x, status: 'saving', note: '' } : x)))
    try {
      const res = await libraryApi.commitUpload(target.file, {
        baseName: target.stem,
        convertNow,
        speedMode: CONVERT_MODE,
        allowDuplicate: false,
      })
      setUploadDrafts((cur) => cur.map((x) => {
        if (x.key !== key) return x
        if (res.duplicate) return { ...x, status: 'error', note: `重复：${String(res.existing || '')}` }
        return {
          ...x,
          status: 'saved',
          selected: false,
          note: convertNow && res.enqueued ? '已保存并加入转换队列' : '已保存',
        }
      }))
    } catch (err) {
      setUploadDrafts((cur) => cur.map((x) => (
        x.key === key
          ? { ...x, status: 'error', note: err instanceof Error ? err.message : '保存失败' }
          : x
      )))
    }
  }

  const saveSelectedDrafts = async (convertNow: boolean) => {
    const ready = await ensureDirsReady()
    if (!ready) return
    const selected = uploadDrafts.filter((x) => x.selected && x.status !== 'saving' && x.status !== 'saved')
    if (!selected.length) {
      message.info('请先选择要保存的文件')
      return
    }
    setUploadSaving(true)
    try {
      for (const x of selected) {
        // eslint-disable-next-line no-await-in-loop
        await saveDraft(x.key, convertNow)
      }
      await store.loadFiles(scope)
      message.success(`已处理 ${selected.length} 个文件`)
    } finally {
      setUploadSaving(false)
    }
  }

  const scanRenameSuggestions = async () => {
    setRenameLoading(true)
    try {
      const res = await libraryApi.listRenameSuggestions(renameScope, renameUseLlm)
      const items = Array.isArray(res.items) ? res.items : []
      setRenameItems(items)
      const selected: Record<string, boolean> = {}
      const overrides: Record<string, string> = {}
      for (const item of items) {
        selected[item.name] = Boolean(item.diff)
        overrides[item.name] = item.suggested_stem || item.name.replace(/\.pdf$/i, '')
      }
      setRenameSelected(selected)
      setRenameOverrides(overrides)
      message.success(`扫描完成：${res.changed}/${res.total_scanned} 需要改名`)
    } catch (err) {
      message.error(err instanceof Error ? err.message : '扫描改名建议失败')
    } finally {
      setRenameLoading(false)
    }
  }

  const selectFailedDrafts = () => {
    if (!failedUploadDrafts.length) {
      message.info('暂无失败项')
      return
    }
    setUploadDrafts((cur) => cur.map((x) => ({ ...x, selected: x.status === 'error' })))
    message.info(`已选择 ${failedUploadDrafts.length} 个失败项`)
  }

  const showDuplicateFailedDrafts = () => {
    if (!duplicateFailedDrafts.length) {
      message.info('当前没有重复文件失败项')
      return
    }
    applyUploadFilter('dup_error')
    message.info(`已切换到重复失败项（${duplicateFailedDrafts.length}）`)
  }

  const retryFailedDrafts = async (convertNow: boolean) => {
    const failed = uploadDrafts.filter((x) => x.status === 'error')
    if (!failed.length) {
      message.info('没有可重试的失败项')
      return
    }
    setUploadSaving(true)
    try {
      for (const x of failed) {
        // eslint-disable-next-line no-await-in-loop
        await saveDraft(x.key, convertNow)
      }
      await store.loadFiles(scope)
      message.success(`已重试 ${failed.length} 个失败项`)
    } finally {
      setUploadSaving(false)
    }
  }

  const applyRenameSuggestions = async () => {
    const names = renameItems.filter((x) => renameSelected[x.name]).map((x) => x.name)
    if (!names.length) {
      message.info('请先选择要改名的条目')
      return
    }
    setRenameApplying(true)
    try {
      const overrides: Record<string, string> = {}
      for (const name of names) overrides[name] = String(renameOverrides[name] || '').trim()
      const res = await libraryApi.applyRenameSuggestions(names, overrides, { useLlm: renameUseLlm, alsoMd: true })
      message[res.failed > 0 ? 'warning' : 'success'](`改名完成：成功 ${res.renamed}，跳过 ${res.skipped}，失败 ${res.failed}`)
      if (res.needs_reindex) message.info('改名后建议更新知识库')
      await store.loadFiles(scope)
      await scanRenameSuggestions()
    } catch (err) {
      message.error(err instanceof Error ? err.message : '应用改名失败')
    } finally {
      setRenameApplying(false)
    }
  }

  const handleConvertPending = async () => {
    const res = await store.convertPending(CONVERT_MODE)
    message[res.enqueued > 0 ? 'success' : 'info'](
      res.enqueued > 0
        ? `已加入队列 ${res.enqueued} 个待转换文件`
        : '没有可入队的待转换文件',
    )
    await store.loadFiles(scope)
  }

  const handleConvertOne = async (item: LibraryFileItem) => {
    if (item.task_state !== 'idle') return
    const replace = item.md_exists ? replaceMd : false
    if (item.md_exists && !replaceMd) {
      message.info('请先开启“覆盖已有 Markdown”再重新转换')
      return
    }
    await store.convert(item.name, CONVERT_MODE, replace)
  }

  const handleDeleteOne = async (item: LibraryFileItem) => {
    const res = await store.deleteFile(item.name, true)
    if (res.ok) {
      message.success(`已删除 ${item.name}`)
      if (res.needs_reindex) {
        message.info('删除/改名后建议更新知识库')
      }
      return
    }
    const warning = Array.isArray(res.warnings) && res.warnings.length > 0
      ? `（${res.warnings.join('；')}）`
      : ''
    message.warning(`删除未完全成功${warning}`)
  }

  const confirmDeleteOne = (item: LibraryFileItem) => {
    Modal.confirm({
      title: '确认删除这个文献吗？',
      content: item.name,
      okText: '删除',
      okType: 'danger',
      cancelText: '取消',
      onOk: async () => {
        await handleDeleteOne(item)
      },
    })
  }

  const handleReindex = async () => {
    const hide = message.loading('正在更新知识库...', 0)
    try {
      const res = await store.reindex()
      hide()
      if (!res.ok) {
        message.error('执行失败')
        return
      }
      message.success('执行完成')
      if (res.refsync_error) {
        message.warning(`引用同步启动失败：${res.refsync_error}`)
      } else if (res.refsync?.started) {
        message.info('已在后台启动引用同步')
      }
    } catch (err) {
      hide()
      message.error(err instanceof Error ? err.message : '执行失败')
    }
  }

  const handleStartRefSync = async () => {
    const hide = message.loading('正在启动引用同步...', 0)
    try {
      const res = await store.startReferenceSync()
      hide()
      if (res.started) {
        message.success('引用同步已启动')
      } else if (res.reason === 'running') {
        message.info('引用同步已在运行')
      } else {
        message.warning('引用同步未启动')
      }
    } catch (err) {
      hide()
      message.error(err instanceof Error ? err.message : '启动引用同步失败')
    }
  }

  const handleStartPaperGuide = async (item: LibraryFileItem) => {
    if (!item.md_exists || !item.md_path) {
      message.info('该文献尚未完成入库转换，请先转换后再进入阅读指导。')
      return
    }
    const hide = message.loading('正在创建阅读指导会话...', 0)
    try {
      let sourcePath = ''
      let sourceName = stripKnownSourceExt(item.name) || item.name
      let resolvedMdPath = ''
      try {
        const resolved = await libraryApi.resolveGuideSource(item.name)
        sourcePath = String(resolved.source_path || '').trim()
        sourceName = String(resolved.source_name || '').trim() || sourceName
        resolvedMdPath = String(resolved.md_path || '').trim()
      } catch {
        // Backward-compatible fallback when backend route is not available yet.
        sourcePath = String(item.md_path || '').trim()
        message.warning('阅读指导源解析失败，已回退到当前文献源。建议重启后端后再试。')
      }
      const convTitle = `阅读指导 · ${sourceName}`
      if (!sourcePath) throw new Error('source path not ready')
      await createPaperGuideConversation({
        sourcePath,
        sourceName,
        title: convTitle,
      })
      if (resolvedMdPath && resolvedMdPath !== String(item.md_path || '').trim()) {
        void store.loadFiles(scope)
      }
      hide()
      nav('/')
      message.success('已进入阅读指导会话')
    } catch (err) {
      hide()
      message.error(err instanceof Error ? err.message : '创建阅读指导会话失败')
    }
  }

  const openMetaEditor = (item: LibraryFileItem) => {
    setMetaItem(item)
    setMetaDraft({
      paper_category: normalizeTextValue(item.paper_category),
      reading_status: (String(item.reading_status || '') as ReadingStatusValue),
      note: String(item.note || ''),
      user_tags: normalizeTextList(Array.isArray(item.user_tags) ? item.user_tags : []),
    })
    setMetaDrawerOpen(true)
  }

  const saveMetaEditor = async () => {
    if (!metaItem) return
    const paperCategory = normalizeTextValue(metaDraft.paper_category)
    const userTags = normalizeTextList(metaDraft.user_tags)
    setMetaSaving(true)
    try {
      const updated = await store.updatePaperMeta({
        pdf_name: metaItem.name,
        paper_category: paperCategory,
        reading_status: metaDraft.reading_status,
        note: metaDraft.note,
        user_tags: userTags,
      })
      if (updated) setMetaItem(updated)
      setMetaDrawerOpen(false)
      message.success('文献元数据已保存')
    } catch (err) {
      message.error(err instanceof Error ? err.message : '保存文献元数据失败')
    } finally {
      setMetaSaving(false)
    }
  }

  const regenerateSuggestionsForVisible = async () => {
    const targets = visibleAll.map((item) => item.name).filter(Boolean)
    if (!targets.length) {
      message.info('当前筛选结果里没有可生成建议的文献')
      return
    }
    setSuggestionsRefreshing(true)
    try {
      const updated = await store.regenerateSuggestions({ pdf_names: targets })
      message.success(`已刷新 ${updated} 篇文献的分类建议`)
    } catch (err) {
      message.error(err instanceof Error ? err.message : '刷新建议失败')
    } finally {
      setSuggestionsRefreshing(false)
    }
  }

  const applyMetaSuggestionAction = async (body: {
    category_action?: '' | 'accept' | 'dismiss'
    accept_tags?: string[]
    dismiss_tags?: string[]
    accept_all_tags?: boolean
    dismiss_all_tags?: boolean
  }) => {
    if (!metaItem) return
    setMetaSuggestionSaving(true)
    try {
      const updated = await store.applySuggestionAction({
        pdf_name: metaItem.name,
        category_action: body.category_action,
        accept_tags: body.accept_tags,
        dismiss_tags: body.dismiss_tags,
        accept_all_tags: body.accept_all_tags,
        dismiss_all_tags: body.dismiss_all_tags,
      })
      if (updated) {
        setMetaItem(updated)
        setMetaDraft((cur) => ({
          ...cur,
          paper_category: normalizeTextValue(updated.paper_category),
          reading_status: String(updated.reading_status || '') as ReadingStatusValue,
          note: String(updated.note || ''),
          user_tags: normalizeTextList(Array.isArray(updated.user_tags) ? updated.user_tags : []),
        }))
      }
    } catch (err) {
      message.error(err instanceof Error ? err.message : '更新建议失败')
    } finally {
      setMetaSuggestionSaving(false)
    }
  }

  const regenerateMetaSuggestions = async () => {
    if (!metaItem) return
    setMetaSuggestionSaving(true)
    try {
      await store.regenerateSuggestions({ pdf_names: [metaItem.name] })
      const refreshed = useLibraryStore.getState().files.find((item) => item.name === metaItem.name) || null
      if (refreshed) {
        setMetaItem(refreshed)
        setMetaDraft((cur) => ({
          ...cur,
          paper_category: normalizeTextValue(refreshed.paper_category),
          reading_status: String(refreshed.reading_status || '') as ReadingStatusValue,
          note: String(refreshed.note || ''),
          user_tags: normalizeTextList(Array.isArray(refreshed.user_tags) ? refreshed.user_tags : []),
        }))
      }
      message.success('文献建议已刷新')
    } catch (err) {
      message.error(err instanceof Error ? err.message : '刷新建议失败')
    } finally {
      setMetaSuggestionSaving(false)
    }
  }

  const toggleLibrarySelection = (name: string, checked: boolean) => {
    setSelectedLibraryNames((cur) => {
      if (!checked && !cur[name]) return cur
      return {
        ...cur,
        [name]: checked,
      }
    })
  }

  const selectCurrentListItems = () => {
    if (!currentListItems.length) {
      message.info('当前列表没有可选文献')
      return
    }
    setSelectedLibraryNames((cur) => {
      const next = { ...cur }
      for (const item of currentListItems) next[item.name] = true
      return next
    })
  }

  const clearLibrarySelection = () => {
    setSelectedLibraryNames({})
  }

  const openBatchEditor = () => {
    if (!selectedLibraryCount) {
      message.info('请先选择要批量编辑的文献')
      return
    }
    setBatchDraft({
      apply_paper_category: false,
      paper_category: '',
      apply_reading_status: false,
      reading_status: '',
      add_tags: [],
      remove_tags: [],
    })
    setBatchDrawerOpen(true)
  }

  const saveBatchEditor = async () => {
    if (!selectedLibraryCount) return
    const paperCategory = normalizeTextValue(batchDraft.paper_category)
    const addTags = normalizeTextList(batchDraft.add_tags)
    const removeTags = normalizeTextList(batchDraft.remove_tags)
    if (
      !batchDraft.apply_paper_category
      && !batchDraft.apply_reading_status
      && addTags.length === 0
      && removeTags.length === 0
    ) {
      message.info('请先设置至少一项批量修改内容')
      return
    }
    setBatchSaving(true)
    try {
      const updated = await store.batchUpdatePaperMeta({
        pdf_names: selectedLibraryNamesList,
        apply_paper_category: batchDraft.apply_paper_category,
        paper_category: paperCategory,
        apply_reading_status: batchDraft.apply_reading_status,
        reading_status: batchDraft.reading_status,
        add_tags: addTags,
        remove_tags: removeTags,
      })
      setBatchDrawerOpen(false)
      setSelectedLibraryNames({})
      message.success(`已批量更新 ${updated} 篇文献`)
    } catch (err) {
      message.error(err instanceof Error ? err.message : '批量编辑失败')
    } finally {
      setBatchSaving(false)
    }
  }

  const selectAllUploadDrafts = () => {
    setUploadDrafts((cur) => cur.map((item) => ({ ...item, selected: true })))
  }

  const invertUploadDraftSelection = () => {
    setUploadDrafts((cur) => cur.map((item) => ({ ...item, selected: !item.selected })))
  }

  const selectRenameDiffItems = () => {
    setRenameSelected((cur) => {
      const next = { ...cur }
      for (const item of renameItems) {
        next[item.name] = Boolean(item.diff)
      }
      return next
    })
  }

  const clearRenameSelection = () => {
    setRenameSelected((cur) => {
      const next = { ...cur }
      for (const item of renameItems) {
        next[item.name] = false
      }
      return next
    })
  }

  const applyUploadFilter = (next: UploadDraftFilter) => {
    setUploadDraftFilter(next)
    if (next === 'dup_error') {
      setUploadErrorReason('duplicate')
      return
    }
    if (next !== 'error') {
      setUploadErrorReason('all')
    }
  }

  const renderFileRow = (item: LibraryFileItem) => {
    const tag = fileTag(item)
    const readingLabel = readingStatusLabel(item.reading_status)
    const metaTags = (item.user_tags || []).slice(0, 3)
    const suggestionCount = (item.suggested_category ? 1 : 0) + (item.suggested_tags || []).length
    const categoryActive = !onlyUnclassified && paperCategoryFilter && String(item.paper_category || '') === paperCategoryFilter
    const statusActive = readingStatusFilter && item.reading_status === readingStatusFilter
    const isSelected = Boolean(selectedLibraryNames[item.name])
    const showPrimaryConvertAction = !item.md_exists

    return (
      <div className={`kb-lib-file-row${isSelected ? ' is-selected' : ''}${suggestionCount > 0 ? ' has-suggestions' : ''}`}>
        <div className="kb-lib-file-select">
          <Checkbox
            checked={isSelected}
            onChange={(event) => toggleLibrarySelection(item.name, event.target.checked)}
          />
        </div>

        <div className="kb-lib-file-main">
          <div className="kb-lib-file-head">
            <div className="kb-lib-file-title-wrap">
              <Text className="kb-lib-file-title">{item.name}</Text>
            </div>
            <div className="kb-lib-file-submeta">
              <Tag color={tag.color}>{tag.text}</Tag>
              <span className={`kb-lib-file-meta-muted${item.md_exists ? ' is-ready' : ''}`}>
                {item.md_exists ? 'Markdown 已就绪' : '尚未生成 Markdown'}
              </span>
              {suggestionCount > 0 ? (
                <span className="kb-lib-file-submeta-chip is-suggestion">
                  {suggestionCount} 条系统建议
                </span>
              ) : null}
            </div>
          </div>

          {(item.paper_category || readingLabel || metaTags.length > 0) ? (
            <div className="kb-lib-file-taxonomy">
              {item.paper_category ? (
                <button
                  type="button"
                  className={`kb-lib-taxonomy-pill is-category${categoryActive ? ' is-active' : ''}`}
                  onClick={() => applyPaperCategoryFilter(String(item.paper_category || ''))}
                >
                  {item.paper_category}
                </button>
              ) : null}
              {readingLabel ? (
                <button
                  type="button"
                  className={`kb-lib-taxonomy-pill is-status${statusActive ? ' is-active' : ''}`}
                  onClick={() => setReadingStatusFilter(item.reading_status)}
                >
                  {readingLabel}
                </button>
              ) : null}
              {metaTags.map((tagValue) => (
                <button
                  key={`${item.name}-tag-${tagValue}`}
                  type="button"
                  className={`kb-lib-taxonomy-pill is-tag${paperTagFilter && tagValue.toLowerCase() === paperTagFilter.toLowerCase() ? ' is-active' : ''}`}
                  onClick={() => applyPaperTagFilter(tagValue)}
                >
                  #{tagValue}
                </button>
              ))}
              {(item.user_tags || []).length > metaTags.length ? (
                <span className="kb-lib-taxonomy-more">+{(item.user_tags || []).length - metaTags.length}</span>
              ) : null}
            </div>
          ) : null}

          {item.note ? <div className="kb-lib-file-note">{item.note}</div> : null}
        </div>

        <div className={`kb-lib-file-actions${showPrimaryConvertAction ? ' has-convert' : ' is-compact'}`}>
          <Button className="kb-lib-file-action-main" size="small" onClick={() => openMetaEditor(item)}>
            分类/标签
          </Button>
          <Button
            className="kb-lib-file-action-main"
            size="small"
            disabled={!item.md_exists || !item.md_path}
            onClick={() => { void handleStartPaperGuide(item) }}
          >
            阅读指导
          </Button>
          {showPrimaryConvertAction ? (
            <Button
              className="kb-lib-file-action-main"
              size="small"
              type="primary"
              ghost
              disabled={item.task_state !== 'idle'}
              onClick={() => { void handleConvertOne(item) }}
            >
              转换
            </Button>
          ) : null}
          <Button className="kb-lib-file-action-main" size="small" onClick={() => { void store.openFile(item.name, 'pdf') }}>
            打开 PDF
          </Button>
          <div className="kb-lib-file-more">
            <Dropdown
              trigger={['click']}
              menu={{
                items: [
                  ...(item.md_exists
                    ? [{ key: 'reconvert', label: '重新转换', disabled: item.task_state !== 'idle', icon: <ReloadOutlined /> }]
                    : []),
                  { key: 'open-md', label: '打开 MD', disabled: !item.md_exists },
                  { type: 'divider' },
                  { key: 'delete', label: '删除文献', danger: true, disabled: item.task_state !== 'idle', icon: <DeleteOutlined /> },
                ],
                onClick: ({ key }) => {
                  if (key === 'reconvert') {
                    void handleConvertOne(item)
                    return
                  }
                  if (key === 'open-md') {
                    void store.openFile(item.name, 'md')
                    return
                  }
                  if (key === 'delete') {
                    confirmDeleteOne(item)
                  }
                },
              }}
            >
              <Button size="small" className="kb-lib-file-more-btn" icon={<MoreOutlined />} />
            </Dropdown>
          </div>
        </div>
      </div>
    )
  }

  const renderCategoriesView = () => {
    if (!categoryCards.length) {
      return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="当前筛选下暂无分类结果" />
    }

    return (
      <div className="kb-lib-category-grid">
        {categoryCards.map((card) => {
          const isUnclassified = card.key === 'category:__unclassified__'
          const active = isUnclassified ? onlyUnclassified : (!onlyUnclassified && paperCategoryFilter === card.label)
          return (
            <button
              key={card.key}
              type="button"
              className={`kb-lib-category-card${active ? ' is-active' : ''}`}
              onClick={() => {
                if (isUnclassified) {
                  setPaperCategoryFilter('')
                  setOnlyUnclassified(true)
                } else {
                  applyPaperCategoryFilter(card.label)
                }
                setBrowseMode('list')
              }}
            >
              <div className="kb-lib-category-card-head">
                <div className="kb-lib-category-card-title">
                  <span>{card.label}</span>
                  <strong>{card.count}</strong>
                </div>
                <div className="kb-lib-category-card-meta">
                  <span>{card.unreadCount} unread</span>
                  <span>{card.convertedCount} converted</span>
                  {card.pendingCount > 0 ? <span>{card.pendingCount} pending</span> : null}
                </div>
              </div>

              {card.commonTags.length > 0 ? (
                <div className="kb-lib-category-card-tags">
                  {card.commonTags.map((tagValue) => (
                    <span key={`${card.key}-${tagValue}`} className="kb-lib-category-tag">
                      #{tagValue}
                    </span>
                  ))}
                </div>
              ) : (
                <div className="kb-lib-category-card-empty">暂时还没有明显的常用标签</div>
              )}

              <div className="kb-lib-category-card-recent">
                {card.recentPapers.map((paper) => (
                  <span key={`${card.key}-${paper}`} className="kb-lib-category-paper">
                    {paper}
                  </span>
                ))}
              </div>
            </button>
          )
        })}
      </div>
    )
  }

  const renderTagsView = () => {
    if (!tagCards.length) {
      return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="当前筛选下暂无标签结果" />
    }

    return (
      <div className="kb-lib-tag-grid">
        {tagCards.map((card) => {
          const active = paperTagFilter && card.label.toLowerCase() === paperTagFilter.toLowerCase()
          return (
            <button
              key={card.key}
              type="button"
              className={`kb-lib-tag-card${active ? ' is-active' : ''}`}
              onClick={() => {
                applyPaperTagFilter(card.label)
                setBrowseMode('list')
              }}
            >
              <div className="kb-lib-tag-card-head">
                <div className="kb-lib-tag-card-title">
                  <span>#{card.label}</span>
                  <strong>{card.count}</strong>
                </div>
                <div className="kb-lib-tag-card-meta">
                  <span>{card.unreadCount} 未读</span>
                </div>
              </div>

              {card.categories.length > 0 ? (
                <div className="kb-lib-tag-card-cats">
                  {card.categories.map((category) => (
                    <span key={`${card.key}-${category}`} className="kb-lib-tag-category">
                      {category}
                    </span>
                  ))}
                </div>
              ) : null}

              <div className="kb-lib-tag-card-recent">
                {card.recentPapers.map((paper) => (
                  <span key={`${card.key}-${paper}`} className="kb-lib-tag-paper">
                    {paper}
                  </span>
                ))}
              </div>
            </button>
          )
        })}
      </div>
    )
  }

  const renderFiles = (items: LibraryFileItem[], emptyText: string) => {
    if (!items.length) {
      return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={emptyText} />
    }

    if (items.length < FILE_VIRTUAL_THRESHOLD) {
      return (
        <List
          className="kb-lib-file-list"
          size="small"
          dataSource={items}
          renderItem={(item) => (
            <List.Item className="kb-lib-file-item">
              {renderFileRow(item)}
            </List.Item>
          )}
        />
      )
    }

    return (
      <div className="kb-lib-file-virtual-shell">
        <div className="kb-lib-file-virtual-tip">
          <Text type="secondary" className="text-xs">已启用虚拟滚动（{items.length} 条）</Text>
        </div>
        <VirtualList
          data={items}
          itemKey="name"
          height={FILE_VIRTUAL_HEIGHT}
          itemHeight={FILE_VIRTUAL_ROW_HEIGHT}
        >
          {(item: LibraryFileItem) => (
            <div className="ant-list-item kb-lib-file-item kb-lib-file-virtual-item">
              {renderFileRow(item)}
            </div>
          )}
        </VirtualList>
      </div>
    )
  }

  const counts = store.fileCounts || {
    total_view: store.files.length,
    total_all: store.files.length,
    pending: pendingFiles.length,
    converted: convertedFiles.length,
    queued: store.files.filter((x) => x.task_state === 'queued').length,
    running: store.files.filter((x) => x.task_state === 'running').length,
    reconverting: 0,
  }

  return (
    <div className="kb-library-page mx-auto w-full max-w-[1760px] space-y-4 p-5">
      <div className="kb-lib-head flex flex-wrap items-end justify-between gap-3">
        <div className="kb-lib-head-main">
          <Text className="text-2xl font-semibold">文献管理</Text>
          <div>
            <Text type="secondary" className="text-sm">先配置目录，再批量处理，最后在列表逐条检查。</Text>
          </div>
        </div>
        <Space wrap className="kb-lib-head-actions">
          <Button className="kb-lib-head-btn" icon={<ReloadOutlined />} type="primary" onClick={() => { void handleReindex() }}>更新知识库</Button>
          <Button className="kb-lib-head-btn" icon={<ReloadOutlined />} onClick={() => { void handleStartRefSync() }}>同步引用信息</Button>
        </Space>
      </div>

      <div className="kb-lib-stats-grid grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
        <Card size="small" className="kb-lib-stat"><Text type="secondary">当前视图</Text><div className="kb-lib-stat-value">{counts.total_view}</div></Card>
        <Card size="small" className="kb-lib-stat"><Text type="secondary">待转换</Text><div className="kb-lib-stat-value">{counts.pending}</div></Card>
        <Card size="small" className="kb-lib-stat"><Text type="secondary">已转换</Text><div className="kb-lib-stat-value">{counts.converted}</div></Card>
        <Card size="small" className="kb-lib-stat"><Text type="secondary">排队中</Text><div className="kb-lib-stat-value">{counts.queued}</div></Card>
        <Card size="small" className="kb-lib-stat"><Text type="secondary">运行中</Text><div className="kb-lib-stat-value">{counts.running}</div></Card>
      </div>

      <div className="kb-lib-ops-grid grid gap-4 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,0.85fr)]">
        <Card size="small" className="kb-lib-card" title="目录设置">
          <div className="space-y-3">
            <div className="grid items-center gap-2 md:grid-cols-[56px_minmax(0,1fr)_auto_auto]">
              <Text className="text-sm font-semibold">PDF</Text>
              <Input
                value={pdfDirDraft}
                placeholder="选择 PDF 文献目录"
                onChange={(e) => {
                  setDirTouched(true)
                  setPdfDirDraft(e.target.value)
                }}
              />
              <Button loading={pickingDir === 'pdf'} onClick={() => { void pickDir('pdf') }}>选择目录</Button>
              <Button icon={<FolderOpenOutlined />} onClick={() => { void openFolder('pdf_dir') }}>打开目录</Button>
            </div>

            <div className="grid items-center gap-2 md:grid-cols-[56px_minmax(0,1fr)_auto_auto]">
              <Text className="text-sm font-semibold">MD</Text>
              <Input
                value={mdDirDraft}
                placeholder="选择 Markdown 输出目录"
                onChange={(e) => {
                  setDirTouched(true)
                  setMdDirDraft(e.target.value)
                }}
              />
              <Button loading={pickingDir === 'md'} onClick={() => { void pickDir('md') }}>选择目录</Button>
              <Button icon={<FolderOpenOutlined />} onClick={() => { void openFolder('md_dir') }}>打开目录</Button>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <Button type="primary" icon={<SaveOutlined />} loading={savingDirs} disabled={!dirDirty} onClick={() => { void saveDirs() }}>
                保存目录设置
              </Button>
              <Text type="secondary" className="text-xs">支持手动输入路径，也支持点击“选择目录”弹窗挑选。</Text>
            </div>
          </div>
        </Card>

        <Card size="small" className="kb-lib-card kb-lib-upload-card" title="上传 PDF">
          <Dragger
            className="kb-lib-upload-drop"
            multiple
            accept=".pdf"
            disabled={uploadLocked}
            showUploadList={false}
            beforeUpload={(file) => {
              addDrafts([file as File])
              return false
            }}
          >
            <p className="text-lg"><UploadOutlined /></p>
            <p>上传 PDF（仅加入队列）</p>
          </Dragger>

          {uploadLocked ? (
            <Text type="secondary" className="text-xs">
              {store.converting ? '转换进行中，上传面板暂时锁定。' : '引用同步进行中，上传面板暂时锁定。'}
            </Text>
          ) : null}
        </Card>
      </div>

      {showStickyStatus ? (
        <Card size="small" className="kb-lib-card kb-lib-sticky-status">
          <div className="kb-lib-sticky-wrap">
            {store.converting && store.progress ? (
              <div className="kb-lib-sticky-item">
                <div className="kb-lib-sticky-main">
                  <Text className="kb-lib-sticky-title">转换中 {store.progress.completed}/{store.progress.total}</Text>
                  {store.progress.current ? <Text type="secondary" className="kb-lib-sticky-sub">{store.progress.current}</Text> : null}
                  {convertPageProgress.total > 0 ? (
                    <Text type="secondary" className="kb-lib-sticky-sub">
                      篇内进度 {convertPageProgress.done}/{convertPageProgress.total}
                    </Text>
                  ) : null}
                </div>
                <div className="kb-lib-sticky-progress-stack">
                  <Progress className="kb-lib-sticky-progress" percent={convertPercent} status="active" size="small" />
                  {convertPageProgress.total > 0 ? (
                    <Progress className="kb-lib-sticky-progress kb-lib-sticky-progress-inner" percent={convertPagePercent} status="active" size="small" />
                  ) : null}
                </div>
                <Button size="small" danger icon={<StopOutlined />} onClick={() => { void store.cancelConvert() }}>
                  停止
                </Button>
              </div>
            ) : null}

            {store.refSync?.running ? (
              <div className="kb-lib-sticky-item">
                <div className="kb-lib-sticky-main">
                  <Text className="kb-lib-sticky-title">引用同步中</Text>
                  <Text type="secondary" className="kb-lib-sticky-sub">
                    {store.refSync.current
                      ? `${store.refSync.stage || '运行中'} | ${store.refSync.current}`
                      : (store.refSync.message || '等待同步任务')}
                  </Text>
                </div>
                <Progress className="kb-lib-sticky-progress" percent={refSyncPercent} status="active" size="small" />
                <Tag color="processing">运行中</Tag>
              </div>
            ) : null}
          </div>
        </Card>
      ) : null}

      <Collapse
        className="kb-lib-collapse"
        size="small"
        items={[
          {
            key: 'upload-workbench',
            label: `上传工作台（${uploadDrafts.length}）`,
            children: (
              <div className="space-y-3">
                <div className="kb-lib-upload-toolbar flex flex-wrap items-center gap-2">
                  <Switch checked={uploadUseLlm} onChange={setUploadUseLlm} />
                  <Text className="text-sm text-[var(--muted)]">使用 LLM 补全信息</Text>
                  <Select
                    value={uploadDraftFilter}
                    onChange={(value) => applyUploadFilter(value as UploadDraftFilter)}
                    options={uploadDraftFilterOptions}
                    className="kb-lib-upload-filter"
                  />
                  <Tooltip title="全选草稿"><Button icon={<CheckOutlined />} onClick={selectAllUploadDrafts}>全选</Button></Tooltip>
                  <Tooltip title="反选草稿"><Button icon={<ClearOutlined />} onClick={invertUploadDraftSelection}>反选</Button></Tooltip>
                  <Button loading={uploadInspecting} disabled={uploadLocked} onClick={() => { void inspectSelectedDrafts() }}>扫描已选</Button>
                  <Button loading={uploadSaving} disabled={uploadLocked} onClick={() => { void saveSelectedDrafts(false) }}>保存已选</Button>
                  <Button type="primary" loading={uploadSaving} disabled={uploadLocked} onClick={() => { void saveSelectedDrafts(true) }}>保存并转换</Button>
                  <Button disabled={uploadLocked} onClick={selectFailedDrafts}>选择失败项</Button>
                  <Button disabled={uploadLocked || duplicateFailedDrafts.length === 0} onClick={showDuplicateFailedDrafts}>仅看重复失败</Button>
                  <Button loading={uploadSaving} disabled={uploadLocked || failedUploadDrafts.length === 0} onClick={() => { void retryFailedDrafts(false) }}>重试失败项</Button>
                  <Button type="primary" loading={uploadSaving} disabled={uploadLocked || failedUploadDrafts.length === 0} onClick={() => { void retryFailedDrafts(true) }}>重试并转换</Button>
                  <Button disabled={uploadLocked} onClick={() => setUploadDrafts((cur) => cur.filter((x) => x.status !== 'saved'))}>清理已保存</Button>
                </div>

                <div className="kb-lib-upload-meta flex flex-wrap items-center gap-3">
                  <Text type="secondary" className="text-xs">已选 {selectedUploadCount} 项</Text>
                  <Text type="secondary" className="text-xs">显示 {filteredUploadDrafts.length}/{uploadDrafts.length} 项</Text>
                  {(uploadDraftFilter === 'error' || uploadDraftFilter === 'dup_error') && uploadErrorReason !== 'all' ? (
                    <Button size="small" onClick={() => setUploadErrorReason('all')}>
                      原因筛选：{activeErrorReasonText}（清除）
                    </Button>
                  ) : null}
                </div>

                {failedUploadDrafts.length > 0 ? (
                  <Alert
                    type="warning"
                    showIcon
                    message={`失败草稿：${failedUploadDrafts.length}`}
                    description={(
                      <div className="kb-lib-failed-summary">
                        <div className="kb-lib-failed-reasons">
                          {failedReasonBuckets.map((bucket) => (
                            <Button
                              key={bucket.key}
                              size="small"
                              icon={FAILED_REASON_META[bucket.key].icon}
                              className={`kb-lib-failed-reason-btn kb-lib-reason-tone is-${bucket.key}${uploadErrorReason === bucket.key ? ' is-active' : ''}`}
                              onClick={() => {
                                applyUploadFilter('error')
                                setUploadErrorReason(bucket.key)
                              }}
                            >
                              {bucket.label} ({bucket.count})
                            </Button>
                          ))}
                        </div>
                        <Text type="secondary" className="text-xs">
                          {failedUploadNotes.length > 0 ? failedUploadNotes.join(' | ') : '请查看行内错误信息后重试。'}
                        </Text>
                      </div>
                    )}
                  />
                ) : null}

                <List
                  size="small"
                  locale={{ emptyText: <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无上传草稿" /> }}
                  dataSource={filteredUploadDrafts}
                  pagination={{ pageSize: 8, size: 'small', showSizeChanger: false }}
                  renderItem={(x) => {
                    const reasonKey = x.status === 'error'
                      ? classifyFailedReason(x.note) as Exclude<UploadErrorReason, 'all'>
                      : null
                    return (
                      <List.Item>
                        <div className="w-full space-y-2">
                          <div className="flex flex-wrap items-center gap-2">
                            <Checkbox checked={x.selected} onChange={(e) => setUploadDrafts((cur) => cur.map((t) => (t.key === x.key ? { ...t, selected: e.target.checked } : t)))} />
                            <Text className="min-w-0 flex-1 truncate text-sm">{x.name}</Text>
                            <Tag color={x.status === 'saved' ? 'success' : x.status === 'error' ? 'error' : (x.status === 'saving' || x.status === 'inspecting') ? 'processing' : 'default'}>
                              {DRAFT_STATUS_TEXT[x.status]}
                            </Tag>
                            {reasonKey ? (
                              <span className={`kb-lib-inline-reason-chip kb-lib-reason-tone is-${reasonKey}`}>
                                {FAILED_REASON_META[reasonKey].icon}
                                <span>{FAILED_REASON_META[reasonKey].label}</span>
                              </span>
                            ) : null}
                          </div>
                          <div className="flex flex-wrap items-center gap-2 pl-6">
                            <Text type="secondary" className="text-xs">建议存储名</Text>
                            <Input value={x.stem} onChange={(e) => setUploadDrafts((cur) => cur.map((t) => (t.key === x.key ? { ...t, stem: e.target.value } : t)))} className="w-[24rem] max-w-full" />
                            <Button size="small" disabled={uploadLocked || x.status === 'saving' || x.status === 'inspecting'} onClick={() => { void inspectDraft(x.key) }}>扫描</Button>
                            <Button size="small" disabled={uploadLocked || x.status === 'saving' || x.status === 'saved' || x.status === 'inspecting'} onClick={() => { void saveDraft(x.key, false) }}>保存</Button>
                            <Button size="small" type="primary" disabled={uploadLocked || x.status === 'saving' || x.status === 'saved' || x.status === 'inspecting'} onClick={() => { void saveDraft(x.key, true) }}>保存并转换</Button>
                          </div>
                          <Text type="secondary" className="block pl-6 text-xs">{x.displayName}</Text>
                          {x.note ? (
                            <Text type="secondary" className={`block pl-6 text-xs${reasonKey ? ' kb-lib-fail-note' : ''}`}>
                              {x.note}
                            </Text>
                          ) : null}
                        </div>
                      </List.Item>
                    )
                  }}
                />
              </div>
            ),
          },
        ]}
      />

      <Card size="small" className="kb-lib-card" title="转换与列表筛选">
        <div className="kb-lib-convert-shell">
          <div className="kb-lib-convert-row kb-lib-convert-row-top">
            <Select
              value={scope}
              onChange={(value) => { setScope(value); void store.loadFiles(value) }}
              className="kb-lib-convert-scope"
              options={SCOPE_OPTIONS}
            />
            <Input
              value={fileKeyword}
              onChange={(e) => setFileKeyword(e.target.value)}
              allowClear
              prefix={<SearchOutlined className="opacity-50" />}
              placeholder="筛选文件名"
              className="kb-lib-convert-search"
            />
            <Button className="kb-lib-convert-refresh" icon={<ReloadOutlined />} onClick={() => { void store.loadFiles(scope) }}>
              刷新
            </Button>
          </div>

          <div className="kb-lib-convert-row kb-lib-convert-row-filters">
            <Select
              value={paperCategoryFilter || undefined}
              allowClear
              placeholder="按分类筛选"
              className="kb-lib-convert-filter"
              options={paperCategoryFilterOptions}
              onChange={(value) => setPaperCategoryFilter(String(value || ''))}
            />
            <Select
              value={paperTagFilter || undefined}
              allowClear
              showSearch
              placeholder="按标签筛选"
              className="kb-lib-convert-filter"
              options={paperTagFilterOptions}
              optionFilterProp="label"
              onChange={(value) => setPaperTagFilter(String(value || ''))}
            />
            <Select
              value={readingStatusFilter || undefined}
              allowClear
              placeholder="按阅读状态筛选"
              className="kb-lib-convert-filter"
              options={READING_STATUS_OPTIONS.filter((item) => item.value)}
              onChange={(value) => setReadingStatusFilter(String(value || '') as ReadingStatusValue)}
            />
            <Button
              className="kb-lib-convert-refresh"
              onClick={() => {
                setPaperCategoryFilter('')
                setPaperTagFilter('')
                setReadingStatusFilter('')
              }}
            >
              清空元数据筛选
            </Button>
          </div>

          <div className="kb-lib-convert-row kb-lib-convert-row-toggle">
            <div className="kb-lib-switch-item">
              <Switch checked={onlyBusyFiles} onChange={setOnlyBusyFiles} />
              <Text className="text-sm text-[var(--muted)]">仅显示排队/运行</Text>
            </div>
            <div className="kb-lib-switch-item">
              <Switch checked={replaceMd} onChange={setReplaceMd} />
              <Text className="text-sm text-[var(--muted)]">重新转换时覆盖已有 Markdown</Text>
            </div>
          </div>

          <div className="kb-lib-convert-row kb-lib-convert-row-actions">
            <Button type="primary" onClick={() => { void handleConvertPending() }}>立即转换待处理</Button>
            {store.converting ? <Button icon={<StopOutlined />} danger onClick={() => { void store.cancelConvert() }}>停止</Button> : null}
          </div>
        </div>
      </Card>

      {store.refSync && !store.refSync.running ? (
        <Card size="small" className="kb-lib-card" title="引用同步">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Text type="secondary" className="text-xs">
                {store.refSync.current
                  ? `${store.refSync.stage || '运行中'} | ${store.refSync.current}`
                  : (store.refSync.message || '等待同步任务')}
              </Text>
              <Tag color={store.refSync.running ? 'processing' : (store.refSync.status === 'error' ? 'error' : 'default')}>
                {store.refSync.running ? '运行中' : (store.refSync.status === 'idle' ? '空闲' : store.refSync.status)}
              </Tag>
            </div>
            {store.refSync.docsTotal > 0 ? (
              <Progress
                percent={Math.round((store.refSync.docsDone / Math.max(1, store.refSync.docsTotal)) * 100)}
                status={store.refSync.running ? 'active' : (store.refSync.status === 'error' ? 'exception' : 'normal')}
              />
            ) : null}
            {store.refSync.error ? <Text type="danger" className="text-xs">{store.refSync.error}</Text> : null}
          </div>
        </Card>
      ) : null}

      <Collapse
        className="kb-lib-collapse"
        size="small"
        items={[
          {
            key: 'rename',
            label: `文件名管理（${renameVisible.length}/${renameItems.length}）`,
            children: (
              <div className="space-y-3">
                <div className="flex flex-wrap items-center gap-2">
                  <Select value={renameScope} onChange={setRenameScope} className="w-40" options={RENAME_SCOPE_OPTIONS} />
                  <Switch checked={renameUseLlm} onChange={setRenameUseLlm} />
                  <Text className="text-sm text-[var(--muted)]">使用 LLM 补全信息</Text>
                  <Switch checked={renameOnlyDiff} onChange={setRenameOnlyDiff} />
                  <Text className="text-sm text-[var(--muted)]">仅显示需改名项</Text>
                  <Button onClick={selectRenameDiffItems}>选择需改名项</Button>
                  <Button onClick={clearRenameSelection}>清空选择</Button>
                  <Button loading={renameLoading} onClick={() => { void scanRenameSuggestions() }}>扫描建议</Button>
                  <Button type="primary" loading={renameApplying} disabled={renameItems.length === 0} onClick={() => { void applyRenameSuggestions() }}>应用已选改名</Button>
                </div>

                <Text type="secondary" className="text-xs">已选 {selectedRenameCount} 项</Text>

                <List
                  size="small"
                  locale={{ emptyText: <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无扫描结果" /> }}
                  dataSource={renameVisible}
                  pagination={{ pageSize: 12, size: 'small', showSizeChanger: false }}
                  renderItem={(item) => (
                    <List.Item>
                      <div className="w-full space-y-2">
                        <div className="flex items-center gap-2">
                          <Checkbox checked={Boolean(renameSelected[item.name])} onChange={(e) => setRenameSelected((cur) => ({ ...cur, [item.name]: e.target.checked }))} />
                          <Text className="min-w-0 flex-1 truncate text-sm">{item.name}</Text>
                          <Tag color={item.diff ? 'warning' : 'default'}>{item.diff ? '建议改名' : '无需改名'}</Tag>
                        </div>
                        <div className="flex flex-wrap items-center gap-2 pl-6">
                          <Input value={renameOverrides[item.name] || ''} onChange={(e) => setRenameOverrides((cur) => ({ ...cur, [item.name]: e.target.value }))} className="w-[26rem] max-w-full" />
                        </div>
                        <Text type="secondary" className="block pl-6 text-xs">{item.display_full_name}</Text>
                      </div>
                    </List.Item>
                  )}
                />
              </div>
            ),
          },
        ]}
      />

      <Card size="small" className="kb-lib-card kb-lib-taxonomy-bar" title="文献分类与标签">
        <div className="kb-lib-taxonomy-shell">
          <div className="kb-lib-taxonomy-top">
            <div className="kb-lib-taxonomy-top-main">
              <div className="kb-lib-taxonomy-top-copy">
                <Text className="kb-lib-taxonomy-kicker">Library taxonomy</Text>
                <Text type="secondary" className="kb-lib-taxonomy-note">
                  先筛选范围，再确认建议或批量整理，列表会保留当前视图上下文。
                </Text>
              </div>
              <Segmented
                className="kb-lib-browse-switch"
                value={browseMode}
                onChange={(value) => setBrowseMode(value as LibraryBrowseMode)}
                options={[
                  { label: '列表', value: 'list' },
                  { label: '分类', value: 'categories' },
                  { label: '标签', value: 'tags' },
                ]}
              />
            </div>
            <div className="kb-lib-taxonomy-top-side">
              <Text type="secondary" className="kb-lib-taxonomy-result">
                已显示 {visibleAll.length}/{store.files.length} 篇文献
              </Text>
              <div className="kb-lib-taxonomy-stat-row">
                <span className="kb-lib-taxonomy-stat-chip">
                  {browseMode === 'list' ? '工作列表' : browseMode === 'categories' ? '分类浏览' : '标签索引'}
                </span>
                <span className={`kb-lib-taxonomy-stat-chip${hasActiveTaxonomyFilters ? ' is-active' : ''}`}>
                  {hasActiveTaxonomyFilters ? `${activeTaxonomyFilterCount} 个筛选中` : '未加筛选'}
                </span>
              </div>
            </div>
          </div>

          <div className="kb-lib-taxonomy-controls">
            <div className="kb-lib-taxonomy-filters">
              <Input
                value={fileKeyword}
                onChange={(e) => setFileKeyword(e.target.value)}
                allowClear
                prefix={<SearchOutlined className="opacity-50" />}
                placeholder="搜索标题、分类、标签或备注"
                className="kb-lib-taxonomy-search"
              />
              <Select
                value={paperCategoryFilter || undefined}
                allowClear
                placeholder="分类"
                className="kb-lib-taxonomy-select"
                options={paperCategoryFilterOptions}
                onChange={(value) => applyPaperCategoryFilter(String(value || ''))}
              />
              <Select
                value={paperTagFilter || undefined}
                allowClear
                showSearch
                placeholder="标签"
                className="kb-lib-taxonomy-select"
                options={paperTagFilterOptions}
                optionFilterProp="label"
                onChange={(value) => applyPaperTagFilter(String(value || ''))}
              />
              <Select
                value={readingStatusFilter || undefined}
                allowClear
                placeholder="阅读状态"
                className="kb-lib-taxonomy-select"
                options={READING_STATUS_OPTIONS.filter((item) => item.value)}
                onChange={(value) => setReadingStatusFilter(String(value || '') as ReadingStatusValue)}
              />
            </div>

            <div className="kb-lib-taxonomy-quick">
              <Checkbox checked={onlyUnread} onChange={(event) => setOnlyUnread(event.target.checked)}>
                只看未读
              </Checkbox>
              <Checkbox
                checked={onlyUnclassified}
                onChange={(event) => {
                  const checked = event.target.checked
                  setOnlyUnclassified(checked)
                  if (checked) setPaperCategoryFilter('')
                }}
              >
                只看未分类
              </Checkbox>
              <Checkbox checked={onlySuggested} onChange={(event) => setOnlySuggested(event.target.checked)}>
                只看有建议
              </Checkbox>
              <div className="kb-lib-taxonomy-quick-actions">
                <Button loading={suggestionsRefreshing} disabled={!visibleAll.length} onClick={() => { void regenerateSuggestionsForVisible() }}>
                  刷新建议
                </Button>
                <Button onClick={clearTaxonomyFilters} disabled={!hasActiveTaxonomyFilters}>
                  清空筛选
                </Button>
              </div>
            </div>
          </div>
        </div>
      </Card>

      {browseMode === 'list' ? (
        <Card size="small" className="kb-lib-card kb-lib-batch-card">
          <div className="kb-lib-batch-bar">
            <div className="kb-lib-batch-summary">
              <div className="kb-lib-batch-badges">
                <span className="kb-lib-batch-badge is-strong">已选 {selectedLibraryCount} 篇</span>
                <span className="kb-lib-batch-badge">{currentListItems.length} 篇在当前列表</span>
              </div>
              <Text className="kb-lib-batch-count">批量整理当前选择</Text>
              <Text type="secondary" className="kb-lib-batch-hint">批量编辑只会作用于已选文献，适合先批量设分类，再统一加减标签。</Text>
            </div>
            <div className="kb-lib-batch-actions">
              <Button onClick={selectCurrentListItems}>选中当前列表</Button>
              <Button onClick={clearLibrarySelection} disabled={!selectedLibraryCount}>清空选中</Button>
              <Button type="primary" onClick={openBatchEditor} disabled={!selectedLibraryCount}>批量编辑</Button>
            </div>
          </div>
        </Card>
      ) : null}

      {browseMode === 'list' ? (
        <Tabs
        className="kb-lib-tabs"
        activeKey={tabKey}
        onChange={(key) => setTabKey(key as FileTabKey)}
        items={[
          { key: 'pending', label: `待转换 (${visiblePending.length})`, children: renderFiles(visiblePending, '暂无待转换文件') },
          { key: 'converted', label: `已转换 (${visibleConverted.length})`, children: renderFiles(visibleConverted, '暂无已转换文件') },
          { key: 'all', label: `当前视图 (${visibleAll.length})`, children: renderFiles(visibleAll, '暂无文件') },
        ]}
      />
      ) : browseMode === 'categories' ? (
        <Card size="small" className="kb-lib-card">
          {renderCategoriesView()}
        </Card>
      ) : (
        <Card size="small" className="kb-lib-card">
          {renderTagsView()}
        </Card>
      )}

      <Drawer
        title={metaItem ? `文献元数据 · ${metaItem.name}` : '文献元数据'}
        open={metaDrawerOpen}
        width={420}
        onClose={() => setMetaDrawerOpen(false)}
        destroyOnClose={false}
      >
        <div className="kb-lib-meta-drawer">
          {metaItem ? (
            <div className="kb-lib-meta-hero">
              <div className="kb-lib-meta-hero-copy">
                <Text className="kb-lib-meta-hero-title">{stripKnownSourceExt(metaItem.name) || metaItem.name}</Text>
                <Text type="secondary" className="kb-lib-meta-hero-note">
                  分类和标签完全由你掌控。可以沿用已有词汇，也可以直接录入你自己的整理方式。
                </Text>
              </div>
              <Space wrap size={[6, 6]} className="kb-lib-meta-chip-row">
                <Tag color={metaDraftCategory ? 'blue' : 'default'}>{metaDraftCategory || '未分类'}</Tag>
                {metaDraft.reading_status ? (
                  <Tag color="gold">{readingStatusLabel(metaDraft.reading_status)}</Tag>
                ) : (
                  <Tag>阅读状态未设置</Tag>
                )}
                <Tag color={metaSuggestionCount ? 'processing' : 'default'}>
                  {metaSuggestionCount ? `${metaSuggestionCount} 条系统建议` : '暂无系统建议'}
                </Tag>
              </Space>
              {metaDraftTags.length ? (
                <div className="kb-lib-meta-chip-row">
                  {metaDraftTags.slice(0, 8).map((tagValue) => (
                    <Tag key={`meta-current-${tagValue}`}>{tagValue}</Tag>
                  ))}
                </div>
              ) : null}
            </div>
          ) : null}

          <section className="kb-lib-meta-section">
            <div className="kb-lib-meta-section-head">
              <div className="kb-lib-meta-section-copy">
                <Text className="kb-lib-meta-section-title">我的整理</Text>
                <Text type="secondary" className="kb-lib-meta-section-note">
                  主分类放稳定归属，标签放可复用的检索切面。
                </Text>
              </div>
            </div>

            <div className="kb-lib-meta-field">
              <Text type="secondary" className="kb-lib-meta-label">主分类</Text>
              <AutoComplete
                value={metaDraft.paper_category}
                allowClear
                options={paperCategoryOptions}
                placeholder="选择已有分类，或直接输入自己的分类"
                filterOption={optionMatchesInput}
                onChange={(value) => setMetaDraft((cur) => ({ ...cur, paper_category: String(value || '') }))}
                onBlur={() => setMetaDraft((cur) => ({ ...cur, paper_category: normalizeTextValue(cur.paper_category) }))}
              />
              <Text type="secondary" className="kb-lib-meta-help">
                可直接新建分类。建议保持短、稳定、能跨多篇论文复用。
              </Text>
            </div>

            <div className="kb-lib-meta-field">
              <Text type="secondary" className="kb-lib-meta-label">阅读状态</Text>
              <Select
                value={metaDraft.reading_status || undefined}
                allowClear
                placeholder="选择阅读状态"
                options={READING_STATUS_OPTIONS.filter((item) => item.value)}
                onChange={(value) => setMetaDraft((cur) => ({ ...cur, reading_status: String(value || '') as ReadingStatusValue }))}
              />
            </div>

            <div className="kb-lib-meta-field">
              <Text type="secondary" className="kb-lib-meta-label">标签</Text>
              <Select
                mode="tags"
                value={metaDraft.user_tags}
                showSearch
                maxTagCount="responsive"
                tokenSeparators={TAG_INPUT_SEPARATORS}
                placeholder="输入标签后回车，也支持逗号 / 分号分隔"
                options={paperTagOptions}
                optionFilterProp="label"
                onChange={(value) => setMetaDraft((cur) => ({ ...cur, user_tags: normalizeTextList(value as unknown[]) }))}
              />
              <Text type="secondary" className="kb-lib-meta-help">
                标签更适合放 modality、task、constraint、method property 这类可复用 facet。
              </Text>
            </div>

            <div className="kb-lib-meta-field">
              <Text type="secondary" className="kb-lib-meta-label">备注</Text>
              <Input.TextArea
                autoSize={{ minRows: 5, maxRows: 9 }}
                value={metaDraft.note}
                placeholder="记录这篇文献的用途、结论或后续阅读计划"
                onChange={(event) => setMetaDraft((cur) => ({ ...cur, note: event.target.value }))}
              />
            </div>
          </section>

          <section className="kb-lib-meta-section kb-lib-meta-section-suggest">
            <div className="kb-lib-suggest-head">
              <div className="kb-lib-meta-section-copy">
                <Text className="kb-lib-meta-section-title">系统建议</Text>
                <Text type="secondary" className="kb-lib-meta-section-note">
                  系统只建议，不会自动覆盖你已经确认的分类和标签。
                </Text>
              </div>
              <Space size={8} wrap>
                <Button size="small" loading={metaSuggestionSaving} onClick={() => { void regenerateMetaSuggestions() }}>
                  刷新建议
                </Button>
                {metaItem?.has_suggestions ? (
                  <>
                    <Button
                      size="small"
                      type="primary"
                      ghost
                      loading={metaSuggestionSaving}
                      onClick={() => {
                        void applyMetaSuggestionAction({
                          category_action: metaItem?.suggested_category ? 'accept' : '',
                          accept_all_tags: true,
                        })
                      }}
                    >
                      接受全部
                    </Button>
                    <Button
                      size="small"
                      loading={metaSuggestionSaving}
                      onClick={() => {
                        void applyMetaSuggestionAction({
                          category_action: metaItem?.suggested_category ? 'dismiss' : '',
                          dismiss_all_tags: true,
                        })
                      }}
                    >
                      忽略全部
                    </Button>
                  </>
                ) : null}
              </Space>
            </div>

            {metaItem?.has_suggestions ? (
              <div className="kb-lib-suggest-list">
                {metaItem.suggested_category ? (
                  <div className="kb-lib-suggest-item">
                    <div className="kb-lib-suggest-copy">
                      <Text className="kb-lib-suggest-title">建议分类</Text>
                      <div className="kb-lib-meta-chip-row">
                        <Tag color="blue">{metaItem.suggested_category}</Tag>
                      </div>
                    </div>
                    <Space size={8}>
                      <Button
                        size="small"
                        type="primary"
                        ghost
                        loading={metaSuggestionSaving}
                        onClick={() => { void applyMetaSuggestionAction({ category_action: 'accept' }) }}
                      >
                        接受
                      </Button>
                      <Button
                        size="small"
                        loading={metaSuggestionSaving}
                        onClick={() => { void applyMetaSuggestionAction({ category_action: 'dismiss' }) }}
                      >
                        忽略
                      </Button>
                    </Space>
                  </div>
                ) : null}

                {(metaItem?.suggested_tags || []).map((tagValue) => (
                  <div key={`meta-suggest-${tagValue}`} className="kb-lib-suggest-item">
                    <div className="kb-lib-suggest-copy">
                      <Text className="kb-lib-suggest-title">建议标签</Text>
                      <div className="kb-lib-meta-chip-row">
                        <Tag>{tagValue}</Tag>
                      </div>
                    </div>
                    <Space size={8}>
                      <Button
                        size="small"
                        type="primary"
                        ghost
                        loading={metaSuggestionSaving}
                        onClick={() => { void applyMetaSuggestionAction({ accept_tags: [tagValue] }) }}
                      >
                        接受
                      </Button>
                      <Button
                        size="small"
                        loading={metaSuggestionSaving}
                        onClick={() => { void applyMetaSuggestionAction({ dismiss_tags: [tagValue] }) }}
                      >
                        忽略
                      </Button>
                    </Space>
                  </div>
                ))}
              </div>
            ) : (
              <Alert
                type="info"
                showIcon
                className="kb-lib-suggest-empty"
                message="当前还没有分类建议"
                description="建议会结合你确认过的分类、标签和论文信号生成；你始终可以直接手动录入自己的分类与标签。"
              />
            )}
          </section>

          <div className="kb-lib-meta-actions">
            <Button onClick={() => setMetaDrawerOpen(false)}>
              取消
            </Button>
            <Button type="primary" loading={metaSaving} onClick={() => { void saveMetaEditor() }}>
              保存
            </Button>
          </div>
        </div>
      </Drawer>

      <Drawer
        title={`批量编辑 · ${selectedLibraryCount} 篇文献`}
        open={batchDrawerOpen}
        width={420}
        onClose={() => setBatchDrawerOpen(false)}
        destroyOnClose={false}
      >
        <div className="kb-lib-meta-drawer">
          <div className="kb-lib-meta-hero kb-lib-meta-hero-batch">
            <div className="kb-lib-meta-hero-copy">
              <Text className="kb-lib-meta-hero-title">批量编辑 {selectedLibraryCount} 篇文献</Text>
              <Text type="secondary" className="kb-lib-meta-hero-note">
                适合先统一主分类和阅读状态，再一次性补充或移除标签。
              </Text>
            </div>
            <Space wrap size={[6, 6]} className="kb-lib-meta-chip-row">
              <Tag color={selectedLibraryCount ? 'blue' : 'default'}>{selectedLibraryCount} 篇已选</Tag>
              {batchDraft.apply_paper_category && normalizeTextValue(batchDraft.paper_category) ? (
                <Tag color="processing">将设置分类: {normalizeTextValue(batchDraft.paper_category)}</Tag>
              ) : null}
              {batchDraft.add_tags.length ? (
                <Tag color="green">新增 {normalizeTextList(batchDraft.add_tags).length} 个标签</Tag>
              ) : null}
            </Space>
          </div>

          <section className="kb-lib-meta-section">
            <div className="kb-lib-meta-section-head">
              <div className="kb-lib-meta-section-copy">
                <Text className="kb-lib-meta-section-title">批量设置</Text>
                <Text type="secondary" className="kb-lib-meta-section-note">
                  只会影响当前选中的文献，不会改到未选中的内容。
                </Text>
              </div>
            </div>

            <div className={`kb-lib-meta-field ${batchDraft.apply_paper_category ? '' : 'is-muted'}`}>
              <Checkbox
                checked={batchDraft.apply_paper_category}
                onChange={(event) => setBatchDraft((cur) => ({ ...cur, apply_paper_category: event.target.checked }))}
              >
                批量设置主分类
              </Checkbox>
              <AutoComplete
                value={batchDraft.paper_category}
                allowClear
                disabled={!batchDraft.apply_paper_category}
                options={paperCategoryOptions}
                placeholder="选择已有分类，或直接输入自己的分类"
                filterOption={optionMatchesInput}
                onChange={(value) => setBatchDraft((cur) => ({ ...cur, paper_category: String(value || '') }))}
                onBlur={() => setBatchDraft((cur) => ({ ...cur, paper_category: normalizeTextValue(cur.paper_category) }))}
              />
              <Text type="secondary" className="kb-lib-meta-help">
                这里也支持手动录入新分类，会写入到所有已选文献。
              </Text>
            </div>

            <div className={`kb-lib-meta-field ${batchDraft.apply_reading_status ? '' : 'is-muted'}`}>
              <Checkbox
                checked={batchDraft.apply_reading_status}
                onChange={(event) => setBatchDraft((cur) => ({ ...cur, apply_reading_status: event.target.checked }))}
              >
                批量设置阅读状态
              </Checkbox>
              <Select
                value={batchDraft.reading_status || undefined}
                allowClear
                disabled={!batchDraft.apply_reading_status}
                placeholder="选择阅读状态"
                options={READING_STATUS_OPTIONS.filter((item) => item.value)}
                onChange={(value) => setBatchDraft((cur) => ({ ...cur, reading_status: String(value || '') as ReadingStatusValue }))}
              />
            </div>
          </section>

          <section className="kb-lib-meta-section">
            <div className="kb-lib-meta-section-head">
              <div className="kb-lib-meta-section-copy">
                <Text className="kb-lib-meta-section-title">标签批处理</Text>
                <Text type="secondary" className="kb-lib-meta-section-note">
                  新增标签支持自由输入；移除标签只从已存在标签里选，避免误删。
                </Text>
              </div>
            </div>

            <div className="kb-lib-meta-field">
              <Text type="secondary" className="kb-lib-meta-label">批量新增标签</Text>
              <Select
                mode="tags"
                value={batchDraft.add_tags}
                showSearch
                maxTagCount="responsive"
                tokenSeparators={TAG_INPUT_SEPARATORS}
                placeholder="输入新增标签后回车，也支持逗号 / 分号分隔"
                options={paperTagOptions}
                optionFilterProp="label"
                onChange={(value) => setBatchDraft((cur) => ({ ...cur, add_tags: normalizeTextList(value as unknown[]) }))}
              />
            </div>

            <div className="kb-lib-meta-field">
              <Text type="secondary" className="kb-lib-meta-label">批量移除标签</Text>
              <Select
                mode="multiple"
                value={batchDraft.remove_tags}
                maxTagCount="responsive"
                placeholder="选择要移除的标签"
                options={paperTagFilterOptions}
                optionFilterProp="label"
                onChange={(value) => setBatchDraft((cur) => ({ ...cur, remove_tags: normalizeTextList(value as unknown[]) }))}
              />
            </div>
          </section>

          <div className="kb-lib-meta-actions">
            <Button onClick={() => setBatchDrawerOpen(false)}>
              取消
            </Button>
            <Button type="primary" loading={batchSaving} onClick={() => { void saveBatchEditor() }}>
              应用到已选文献
            </Button>
          </div>
        </div>
      </Drawer>
    </div>
  )
}
