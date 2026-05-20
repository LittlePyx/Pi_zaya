
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
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
import { useT } from '../i18n'

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
  savedName?: string
  savedSha1?: string
  taskId?: string
  convertRequested?: boolean
  suggestionBasisLabel?: string
  suggestionBasisDetail?: string
  suggestionMatchMethod?: string
  suggestionYearSource?: string
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
function READING_STATUS_OPTIONS(S: Record<string, string>) {
  return [
    { value: '', label: S.lib_reading_status_all },
    { value: 'unread', label: S.lib_reading_status_unread },
    { value: 'reading', label: S.lib_reading_status_reading },
    { value: 'done', label: S.lib_reading_status_done },
    { value: 'revisit', label: S.lib_reading_status_revisit },
  ] as const
}

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

type SuggestionMetaInfo = {
  match_method?: string
  year_source?: string
  basis_label?: string
  basis_detail?: string
}

function SCOPE_OPTIONS(S: Record<string, string>) {
  return [
    { value: '200', label: S.lib_scope_recent_200 },
    { value: '1000', label: S.lib_scope_recent_1000 },
    { value: 'all', label: S.lib_scope_all },
  ]
}

function RENAME_SCOPE_OPTIONS(S: Record<string, string>) {
  return [
    { value: '30', label: S.lib_scope_recent_30 },
    { value: '50', label: S.lib_scope_recent_50 },
    { value: '100', label: S.lib_scope_recent_100 },
    { value: 'all', label: S.lib_scope_all },
  ]
}

function DRAFT_STATUS_TEXT(S: Record<string, string>): Record<DraftStatus, string> {
  return {
    queued: S.lib_draft_queued,
    inspecting: S.lib_draft_inspecting,
    ready: S.lib_draft_ready,
    saving: S.lib_draft_saving,
    saved: S.lib_draft_saved,
    error: S.lib_draft_error,
  }
}

function FAILED_REASON_META(S: Record<string, string>): Record<Exclude<UploadErrorReason, 'all'>, { label: string, icon: ReactNode }> {
  return {
    duplicate: { label: S.lib_fail_duplicate, icon: <CopyOutlined /> },
    path: { label: S.lib_fail_path, icon: <FolderOpenOutlined /> },
    permission: { label: S.lib_fail_permission, icon: <LockOutlined /> },
    network: { label: S.lib_fail_network, icon: <ApiOutlined /> },
    other: { label: S.lib_fail_other, icon: <ExclamationCircleOutlined /> },
  }
}

function fileTag(item: LibraryFileItem, S: Record<string, string>) {
  if (item.task_state === 'running') return { color: 'processing' as const, text: S.lib_tag_converting }
  if (item.task_state === 'queued') return { color: 'warning' as const, text: `${S.lib_tag_queued}${item.queue_pos > 0 ? ` #${item.queue_pos}` : ''}` }
  return item.category === 'converted'
    ? { color: 'success' as const, text: S.lib_tag_converted }
    : { color: 'default' as const, text: S.lib_tag_pending }
}

function derivePageProgress(done0: number, total0: number, msg0: string) {
  const done = Number(done0 || 0)
  const total = Number(total0 || 0)
  if (total > 0) return { done: Math.max(0, done), total: Math.max(0, total) }
  const msg = String(msg0 || '')
  const m = msg.match(/\b(\d{1,4})\s*\/\s*(\d{1,4})\b/)
  if (!m) return { done: 0, total: 0 }
  const parsedDone = Number(m[1] || 0)
  const parsedTotal = Number(m[2] || 0)
  if (!Number.isFinite(parsedDone) || !Number.isFinite(parsedTotal) || parsedTotal <= 0) {
    return { done: 0, total: 0 }
  }
  return { done: Math.max(0, parsedDone), total: Math.max(0, parsedTotal) }
}

function deriveConvertStageLabel(msg0: string, S_?: Record<string, string>) {
  const msg = String(msg0 || '').trim().toLowerCase()
  if (!msg) return ''
  if (msg.includes('ingesting')) return S_ ? S_.lib_convert_ingesting : '正在更新知识库索引...'
  if (msg.includes('cancel')) return S_ ? S_.lib_convert_cancelling : '正在取消转换...'
  return ''
}

function matchesKeyword(name: string, keyword: string) {
  if (!keyword) return true
  return name.toLowerCase().includes(keyword)
}

function readingStatusLabel(value: string, S_?: Record<string, string>) {
  if (value === 'unread') return S_ ? S_.lib_reading_status_unread : '未读'
  if (value === 'reading') return S_ ? S_.lib_reading_status_reading : '在读'
  if (value === 'done') return S_ ? S_.lib_reading_status_done : '已读'
  if (value === 'revisit') return S_ ? S_.lib_reading_status_revisit : '待回看'
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

function isUploadDraftConverted(draft: UploadDraft, files: LibraryFileItem[]) {
  if (!draft.convertRequested || draft.status !== 'saved') return false
  const match = files.find((item) => {
    if (draft.savedSha1 && item.sha1) return item.sha1 === draft.savedSha1
    if (draft.savedName) return item.name === draft.savedName
    return false
  })
  if (!match) return false
  return match.md_exists && match.task_state === 'idle' && match.category === 'converted'
}

function suggestionBasisTagColor(meta?: SuggestionMetaInfo) {
  const method = String(meta?.match_method || '').trim().toLowerCase()
  const yearSource = String(meta?.year_source || '').trim().toLowerCase()
  if (method === 'doi') return 'success'
  if (method === 'crossref_strong') return 'processing'
  if (yearSource === 'filename') return 'gold'
  if (method === 'crossref_weak') return 'warning'
  return 'default'
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
  const S = useT()
  const store = useLibraryStore()
  const createPaperGuideConversation = useChatStore((s) => s.createPaperGuideConversation)
  const nav = useNavigate()

  const settingsLoaded = useSettingsStore((s) => s.loaded)
  const settingsPdfDir = useSettingsStore((s) => s.pdfDir)
  const settingsMdDir = useSettingsStore((s) => s.mdDir)
  const updateSettings = useSettingsStore((s) => s.update)

  const [scope, setScope] = useState('200')
  const [tabKey, setTabKey] = useState<FileTabKey>('all')
  const [browseMode, setBrowseMode] = useState<LibraryBrowseMode>('list')
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
  const [dirEditorOpen, setDirEditorOpen] = useState(false)

  const [uploadDrafts, setUploadDrafts] = useState<UploadDraft[]>([])
  const [uploadUseLlm, setUploadUseLlm] = useState(true)
  const [uploadDraftFilter, setUploadDraftFilter] = useState<UploadDraftFilter>('all')
  const [uploadErrorReason, setUploadErrorReason] = useState<UploadErrorReason>('all')
  const [uploadInspecting, setUploadInspecting] = useState(false)
  const [uploadSaving, setUploadSaving] = useState(false)
  const [uploadWorkbenchOpen, setUploadWorkbenchOpen] = useState(false)
  const autoInspectingRef = useRef(false)

  const [renameScope, setRenameScope] = useState('30')
  const [renameLoading, setRenameLoading] = useState(false)
  const [renameApplying, setRenameApplying] = useState(false)
  const [renameItems, setRenameItems] = useState<RenameSuggestionItem[]>([])
  const [renameSelected, setRenameSelected] = useState<Record<string, boolean>>({})
  const [renameOverrides, setRenameOverrides] = useState<Record<string, string>>({})
  const [renameResultsOpen, setRenameResultsOpen] = useState(false)
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
  const renameOnlyDiff = true
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
      .map(([key, count]) => ({ key, count, label: FAILED_REASON_META(S)[key].label }))
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
      { value: 'all', label: S.lib_upload_filter_all.replace('{n}', String(uploadDrafts.length)) },
      { value: 'todo', label: S.lib_upload_filter_todo.replace('{n}', String(uploadDrafts.filter((x) => ['queued', 'inspecting', 'ready', 'saving'].includes(x.status)).length)) },
      { value: 'error', label: S.lib_upload_filter_error.replace('{n}', String(uploadDrafts.filter((x) => x.status === 'error').length)) },
      { value: 'dup_error', label: S.lib_upload_filter_dup.replace('{n}', String(uploadDrafts.filter((x) => x.status === 'error' && isDuplicateFailure(x.note)).length)) },
      { value: 'saved', label: S.lib_upload_filter_saved.replace('{n}', String(uploadDrafts.filter((x) => x.status === 'saved').length)) },
    ],
    [uploadDrafts],
  )
  const activeErrorReasonText = useMemo(() => {
    const map: Record<UploadErrorReason, string> = {
      all: S.lib_error_filter_all,
      duplicate: FAILED_REASON_META(S).duplicate.label,
      path: FAILED_REASON_META(S).path.label,
      permission: FAILED_REASON_META(S).permission.label,
      network: FAILED_REASON_META(S).network.label,
      other: FAILED_REASON_META(S).other.label,
    }
    return map[uploadErrorReason]
  }, [uploadErrorReason])
  const convertPercent = useMemo(() => {
    if (!store.progress || store.progress.total <= 0) return 0
    const tasks = Array.isArray(store.progress.activeTasks) ? store.progress.activeTasks : []
    let activeFraction = 0
    if (tasks.length > 0) {
      for (const task of tasks) {
        const taskProgress = derivePageProgress(task.cur_page_done, task.cur_page_total, task.cur_page_msg)
        if (taskProgress.total <= 0) continue
        activeFraction += Math.min(0.999, taskProgress.done / Math.max(1, taskProgress.total))
      }
    } else {
      const fallback = derivePageProgress(
        store.progress.curPageDone,
        store.progress.curPageTotal,
        store.progress.curPageMsg,
      )
      if (fallback.total > 0) {
        activeFraction = Math.min(0.999, fallback.done / Math.max(1, fallback.total))
      }
    }
    const rawPercent = Math.min(100, Math.round(((store.progress.completed + activeFraction) / Math.max(1, store.progress.total)) * 100))
    const stillRunning = store.progress.completed < store.progress.total
      && (
        Number(store.progress.activeCount || 0) > 0
        || tasks.length > 0
        || Boolean(String(store.progress.current || '').trim())
      )
    return stillRunning ? Math.min(rawPercent, 99) : rawPercent
  }, [store.progress])
  const convertPageProgress = useMemo(() => (
    derivePageProgress(
      Number(store.progress?.curPageDone || 0),
      Number(store.progress?.curPageTotal || 0),
      String(store.progress?.curPageMsg || ''),
    )
  ), [store.progress])
  const convertPagePercent = useMemo(
    () => (convertPageProgress.total > 0
      ? Math.round((convertPageProgress.done / Math.max(1, convertPageProgress.total)) * 100)
      : 0),
    [convertPageProgress],
  )
  const convertActiveSummary = useMemo(() => {
    const tasks = Array.isArray(store.progress?.activeTasks) ? store.progress.activeTasks : []
    if (!tasks.length) return ''
    const names = tasks.map((task) => String(task.name || '').trim()).filter(Boolean)
    if (!names.length) return ''
    const preview = names.slice(0, 3).join('\u3001')
    const suffix = names.length > 3 ? ' ...' : ''
    return `\u5e76\u884c\u4e2d ${tasks.length} \u7bc7\uff1a${preview}${suffix}`
  }, [store.progress])
  const convertStageLabel = useMemo(
    () => deriveConvertStageLabel(String(store.progress?.curPageMsg || ''), S),
    [store.progress],
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

  const filterFiles = useCallback(
    (items: LibraryFileItem[], options: FilterFilesOptions = {}) =>
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
        if (!options.ignoreCategoryFilter && paperCategoryFilter && String(item.paper_category || '') !== paperCategoryFilter) return false
        if (!options.ignoreTagFilter && paperTagFilter && !(item.user_tags || []).some((tag) => String(tag || '').toLowerCase() === paperTagFilter.toLowerCase())) return false
        if (readingStatusFilter && String(item.reading_status || '') !== readingStatusFilter) return false
        if (onlyUnread && String(item.reading_status || '') !== 'unread') return false
        if (onlyUnclassified && String(item.paper_category || '').trim()) return false
        if (onlySuggested && !item.has_suggestions) return false
        return true
      }),
    [normalizedKeyword, onlySuggested, onlyUnclassified, onlyUnread, paperCategoryFilter, paperTagFilter, readingStatusFilter],
  )

  const visiblePending = useMemo(
    () => filterFiles(pendingFiles),
    [filterFiles, pendingFiles],
  )
  const visibleConverted = useMemo(
    () => filterFiles(convertedFiles),
    [convertedFiles, filterFiles],
  )
  const visibleAll = useMemo(
    () => filterFiles(store.files),
    [filterFiles, store.files],
  )
  const visibleAllWithoutCategory = useMemo(
    () => filterFiles(store.files, { ignoreCategoryFilter: true }),
    [filterFiles, store.files],
  )
  const visibleAllWithoutTag = useMemo(
    () => filterFiles(store.files, { ignoreTagFilter: true }),
    [filterFiles, store.files],
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
        ? S.lib_category_unclassified
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
        label: label || S.lib_category_unclassified,
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
        const category = String(item.paper_category || '').trim() || S.lib_category_unclassified
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
    if (!settingsLoaded) return
    if (!String(settingsPdfDir || '').trim() || !String(settingsMdDir || '').trim()) {
      setDirEditorOpen(true)
    }
  }, [settingsLoaded, settingsPdfDir, settingsMdDir])

  useEffect(() => {
    if (uploadDrafts.length === 0) {
      setUploadWorkbenchOpen(false)
      return
    }
    setUploadWorkbenchOpen(true)
  }, [uploadDrafts.length])

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

  const saveDirs = useCallback(async () => {
    if (!pdfDirDraft.trim() || !mdDirDraft.trim()) {
      message.warning(S.lib_msg_dir_empty)
      return false
    }
    setSavingDirs(true)
    try {
      await updateSettings({ pdfDir: pdfDirDraft.trim(), mdDir: mdDirDraft.trim() })
      setDirTouched(false)
      setDirEditorOpen(false)
      message.success(S.lib_msg_save_dir_success)
      await store.loadFiles(scope)
      return true
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.lib_msg_save_dir_fail)
      return false
    } finally {
      setSavingDirs(false)
    }
  }, [mdDirDraft, pdfDirDraft, scope, store, updateSettings])

  const ensureDirsReady = useCallback(async () => {
    if (!dirDirty) return true
    return saveDirs()
  }, [dirDirty, saveDirs])

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
        message.info(S.lib_msg_no_dir_selected)
        return
      }
      setDirTouched(true)
      if (target === 'pdf') setPdfDirDraft(res.path)
      else setMdDirDraft(res.path)
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.lib_msg_pick_dir_fail)
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
          savedName: '',
          savedSha1: '',
          taskId: '',
          convertRequested: false,
          suggestionBasisLabel: '',
          suggestionBasisDetail: '',
          suggestionMatchMethod: '',
          suggestionYearSource: '',
        })
      }
      return next
    })
  }

  const inspectDraft = useCallback(async (key: string) => {
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
          suggestionBasisLabel: String(res.meta?.basis_label || ''),
          suggestionBasisDetail: String(res.meta?.basis_detail || ''),
          suggestionMatchMethod: String(res.meta?.match_method || ''),
          suggestionYearSource: String(res.meta?.year_source || ''),
          status: res.duplicate ? 'error' : 'ready',
          note: res.duplicate ? `${S.lib_upload_dup_prefix}${String(res.existing || '')}` : S.lib_upload_scan_done,
        }
      }))
    } catch (err) {
      setUploadDrafts((cur) => cur.map((x) => (
        x.key === key
          ? { ...x, status: 'error', note: err instanceof Error ? err.message : S.lib_upload_scan_fail }
          : x
      )))
    }
  }, [ensureDirsReady, uploadDrafts, uploadUseLlm])

  const inspectSelectedDrafts = async () => {
    const selected = uploadDrafts.filter((x) => x.selected && x.status !== 'inspecting')
    if (!selected.length) {
      message.info(S.lib_msg_select_scan)
      return
    }
    setUploadInspecting(true)
    try {
      for (const x of selected) {
        await inspectDraft(x.key)
      }
      message.success(S.lib_msg_scanned_count.replace('{n}', String(selected.length)))
    } finally {
      setUploadInspecting(false)
    }
  }

  useEffect(() => {
    if (uploadLocked || dirDirty || uploadInspecting || autoInspectingRef.current) return
    const queuedKeys = uploadDrafts
      .filter((x) => x.status === 'queued')
      .map((x) => x.key)
    if (!queuedKeys.length) return

    autoInspectingRef.current = true
    setUploadInspecting(true)

    void (async () => {
      try {
        for (const key of queuedKeys) {
          // Auto-fill suggested names for newly added upload drafts.
          await inspectDraft(key)
        }
      } finally {
        autoInspectingRef.current = false
        setUploadInspecting(false)
      }
    })()
  }, [dirDirty, inspectDraft, uploadDrafts, uploadInspecting, uploadLocked])

  useEffect(() => {
    setUploadDrafts((cur) => {
      const next = cur.filter((draft) => !isUploadDraftConverted(draft, store.files))
      return next.length === cur.length ? cur : next
    })
  }, [store.files])

  const saveDraft = async (key: string, convertNow: boolean, opts?: { syncUi?: boolean }) => {
    const syncUi = opts?.syncUi ?? true
    const ready = await ensureDirsReady()
    if (!ready) return { saved: false, enqueued: false }
    const target = uploadDrafts.find((x) => x.key === key)
    if (!target) return { saved: false, enqueued: false }
    setUploadDrafts((cur) => cur.map((x) => (
      x.key === key
        ? {
          ...x,
          status: 'saving',
          note: '',
          savedName: '',
          savedSha1: '',
          taskId: '',
          convertRequested: false,
        }
        : x
    )))
    try {
      const res = await libraryApi.commitUpload(target.file, {
        baseName: target.stem,
        convertNow,
        speedMode: CONVERT_MODE,
        allowDuplicate: false,
      })
      const savedName = String(res.name || target.file.name || '')
      const enqueued = Boolean(convertNow && res.enqueued)
      setUploadDrafts((cur) => cur.map((x) => {
        if (x.key !== key) return x
        if (res.duplicate) return { ...x, status: 'error', note: `${S.lib_upload_dup_prefix}${String(res.existing || '')}` }
        return {
          ...x,
          status: 'saved',
          selected: false,
          stem: savedName.replace(/\.pdf$/i, '') || x.stem,
          displayName: savedName || x.displayName,
          savedName,
          savedSha1: String(res.sha1 || ''),
          taskId: String(res.task_id || ''),
          convertRequested: enqueued,
          note: enqueued ? S.lib_msg_saved_enqueued.replace('{name}', savedName) : S.lib_msg_saved_only.replace('{name}', savedName),
        }
      }))
      if (res.duplicate) return { saved: false, enqueued: false }
      if (syncUi) {
        await store.loadFiles(scope)
        if (enqueued) store.startProgressStream()
      }
      return { saved: true, enqueued }
    } catch (err) {
      setUploadDrafts((cur) => cur.map((x) => (
        x.key === key
          ? { ...x, status: 'error', note: err instanceof Error ? err.message : S.lib_upload_save_fail }
          : x
      )))
      return { saved: false, enqueued: false }
    }
  }

  const saveSelectedDrafts = async (convertNow: boolean) => {
    const ready = await ensureDirsReady()
    if (!ready) return
    const selected = uploadDrafts.filter((x) => x.selected && x.status !== 'saving' && x.status !== 'saved')
    if (!selected.length) {
      message.info(S.lib_msg_select_save)
      return
    }
    setUploadSaving(true)
    try {
      let anyEnqueued = false
      for (const x of selected) {
        const result = await saveDraft(x.key, convertNow, { syncUi: false })
        anyEnqueued = anyEnqueued || Boolean(result.enqueued)
      }
      await store.loadFiles(scope)
      if (anyEnqueued) store.startProgressStream()
      message.success(S.lib_msg_processed_count.replace('{n}', String(selected.length)))
    } finally {
      setUploadSaving(false)
    }
  }

  const scanRenameSuggestions = async () => {
    setRenameLoading(true)
    try {
      const res = await libraryApi.listRenameSuggestions(renameScope, true)
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
      setRenameResultsOpen(items.some((item) => item.diff))
      message.success(S.lib_msg_scan_result.replace('{changed}', String(res.changed)).replace('{total}', String(res.total_scanned)))
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.lib_msg_scan_rename_fail)
    } finally {
      setRenameLoading(false)
    }
  }

  const selectFailedDrafts = () => {
    if (!failedUploadDrafts.length) {
      message.info(S.lib_msg_no_failed_items)
      return
    }
    setUploadDrafts((cur) => cur.map((x) => ({ ...x, selected: x.status === 'error' })))
    message.info(S.lib_msg_selected_failed.replace('{n}', String(failedUploadDrafts.length)))
  }

  const showDuplicateFailedDrafts = () => {
    if (!duplicateFailedDrafts.length) {
      message.info(S.lib_msg_no_dup_failures)
      return
    }
    applyUploadFilter('dup_error')
    message.info(S.lib_msg_switched_dup.replace('{n}', String(duplicateFailedDrafts.length)))
  }

  const retryFailedDrafts = async (convertNow: boolean) => {
    const failed = uploadDrafts.filter((x) => x.status === 'error')
    if (!failed.length) {
      message.info(S.lib_msg_no_retryable)
      return
    }
    setUploadSaving(true)
    try {
      let anyEnqueued = false
      for (const x of failed) {
        const result = await saveDraft(x.key, convertNow, { syncUi: false })
        anyEnqueued = anyEnqueued || Boolean(result.enqueued)
      }
      await store.loadFiles(scope)
      if (anyEnqueued) store.startProgressStream()
      message.success(S.lib_msg_retried_count.replace('{n}', String(failed.length)))
    } finally {
      setUploadSaving(false)
    }
  }

  const applyRenameSuggestions = async () => {
    const names = renameItems.filter((x) => renameSelected[x.name]).map((x) => x.name)
    if (!names.length) {
      message.info(S.lib_msg_select_rename)
      return
    }
    setRenameApplying(true)
    try {
      const overrides: Record<string, string> = {}
      for (const name of names) overrides[name] = String(renameOverrides[name] || '').trim()
      const res = await libraryApi.applyRenameSuggestions(names, overrides, { useLlm: true, alsoMd: true })
      message[res.failed > 0 ? 'warning' : 'success'](S.lib_msg_rename_result.replace('{ok}', String(res.renamed)).replace('{skip}', String(res.skipped)).replace('{fail}', String(res.failed)))
      if (res.needs_reindex) message.info(S.lib_msg_rename_suggest_reindex)
      await store.loadFiles(scope)
      await scanRenameSuggestions()
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.lib_msg_apply_rename_fail)
    } finally {
      setRenameApplying(false)
    }
  }

  const handleConvertPending = async () => {
    const res = await store.convertPending(CONVERT_MODE)
    message[res.enqueued > 0 ? 'success' : 'info'](
      res.enqueued > 0
        ? S.lib_msg_enqueued_count.replace('{n}', String(res.enqueued))
        : S.lib_msg_no_convertible,
    )
    await store.loadFiles(scope)
  }

  const handleConvertOne = async (item: LibraryFileItem) => {
    if (item.task_state !== 'idle') return
    await store.convert(item.name, CONVERT_MODE, true)
  }

  const handleDeleteOne = async (item: LibraryFileItem) => {
    const res = await store.deleteFile(item.name, true)
    if (res.ok) {
      message.success(S.lib_msg_deleted_name.replace('{name}', item.name))
      if (res.needs_reindex) {
        message.info(S.lib_msg_delete_suggest_reindex)
      }
      return
    }
    const warning = Array.isArray(res.warnings) && res.warnings.length > 0
      ? `（${res.warnings.join('；')}）`
      : ''
    message.warning(S.lib_msg_delete_not_complete.replace('{warning}', warning))
  }

  const confirmDeleteOne = (item: LibraryFileItem) => {
    Modal.confirm({
      title: S.lib_menu_delete_confirm_title,
      content: item.name,
      okText: S.lib_menu_delete_ok,
      okType: 'danger',
      cancelText: S.lib_menu_delete_cancel,
      onOk: async () => {
        await handleDeleteOne(item)
      },
    })
  }

  const handleReindex = async () => {
    const hide = message.loading(S.lib_msg_updating_kb, 0)
    try {
      const res = await store.reindex()
      hide()
      if (!res.ok) {
        message.error(S.lib_msg_exec_fail)
        return
      }
      message.success(S.lib_msg_exec_done)
      if (res.refsync_error) {
        message.warning(S.lib_msg_refsync_fail_detail.replace('{error}', String(res.refsync_error)))
      } else if (res.refsync?.started) {
        message.info(S.lib_msg_refsync_started_bg)
      }
    } catch (err) {
      hide()
      message.error(err instanceof Error ? err.message : S.lib_msg_exec_fail)
    }
  }

  const handleStartRefSync = async () => {
    const hide = message.loading(S.lib_msg_starting_refsync, 0)
    try {
      const res = await store.startReferenceSync()
      hide()
      if (res.started) {
        message.success(S.lib_msg_refsync_started)
      } else if (res.reason === 'running') {
        message.info(S.lib_msg_refsync_already_running)
      } else {
        message.warning(S.lib_msg_refsync_not_started)
      }
    } catch (err) {
      hide()
      message.error(err instanceof Error ? err.message : S.lib_msg_start_refsync_fail)
    }
  }

  const handleStartPaperGuide = async (item: LibraryFileItem) => {
    if (!item.md_exists || !item.md_path) {
      message.info(S.lib_msg_guide_not_converted)
      return
    }
    const hide = message.loading(S.lib_msg_creating_guide, 0)
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
        message.warning(S.lib_msg_guide_source_fallback)
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
      message.success(S.lib_msg_guide_entered)
    } catch (err) {
      hide()
      message.error(err instanceof Error ? err.message : S.lib_msg_guide_create_fail)
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
      message.success(S.lib_msg_meta_saved)
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.lib_msg_meta_save_fail)
    } finally {
      setMetaSaving(false)
    }
  }

  const regenerateSuggestionsForVisible = async () => {
    const targets = visibleAll.map((item) => item.name).filter(Boolean)
    if (!targets.length) {
      message.info(S.lib_msg_no_suggestion_candidates)
      return
    }
    setSuggestionsRefreshing(true)
    try {
      const updated = await store.regenerateSuggestions({ pdf_names: targets })
      message.success(S.lib_msg_suggestions_refreshed_count.replace('{n}', String(updated)))
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.lib_msg_refresh_suggestion_fail)
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
      message.error(err instanceof Error ? err.message : S.lib_msg_update_suggestion_fail)
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
      message.success(S.lib_msg_suggestion_refreshed)
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.lib_msg_refresh_suggestion_fail)
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
      message.info(S.lib_msg_no_selectable)
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
      message.info(S.lib_msg_select_batch_edit)
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
      message.info(S.lib_msg_set_batch_content)
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
      message.success(S.lib_msg_batch_updated_count.replace('{n}', String(updated)))
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.lib_msg_batch_edit_fail)
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
    const tag = fileTag(item, S)
    const statusTone =
      tag.color === 'success'
        ? 'is-success'
        : tag.color === 'processing'
          ? 'is-processing'
          : tag.color === 'warning'
            ? 'is-warning'
            : 'is-default'
    const readingLabel = readingStatusLabel(item.reading_status, S)
    const metaTags = item.user_tags || []
    const suggestionCount = (item.suggested_category ? 1 : 0) + (item.suggested_tags || []).length
    const categoryActive = !onlyUnclassified && paperCategoryFilter && String(item.paper_category || '') === paperCategoryFilter
    const statusActive = readingStatusFilter && item.reading_status === readingStatusFilter
    const isSelected = Boolean(selectedLibraryNames[item.name])
    const showPrimaryConvertAction = !item.md_exists
    const itemProgress = derivePageProgress(item.cur_page_done, item.cur_page_total, item.cur_page_msg)
    const itemProgressPercent = itemProgress.total > 0
      ? Math.round((itemProgress.done / Math.max(1, itemProgress.total)) * 100)
      : 0

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
              <Text className="kb-lib-file-title" title={item.name}>{item.name}</Text>
            </div>
            <div className="kb-lib-file-submeta">
              <span className={`kb-lib-file-status-chip ${statusTone}`}>{tag.text}</span>
              {!item.md_exists ? <span className="kb-lib-file-meta-muted">{S.lib_file_no_md}</span> : null}
              {suggestionCount > 0 ? (
                <span className="kb-lib-file-submeta-chip is-suggestion">
                  {S.lib_file_suggestions.replace('{n}', String(suggestionCount))}
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
            </div>
          ) : null}

          {item.note ? <div className="kb-lib-file-note">{item.note}</div> : null}
          {item.task_state === 'running' ? (
            <div style={{ marginTop: 8 }}>
              {itemProgress.total > 0 ? (
                <>
                  <Progress percent={itemProgressPercent} status="active" size="small" showInfo={false} />
                  <Text type="secondary" className="text-xs">
                    {`\u9875\u8fdb\u5ea6 ${itemProgress.done}/${itemProgress.total}`}
                  </Text>
                </>
              ) : null}
              {item.cur_page_msg ? (
                <div>
                  <Text type="secondary" className="text-xs">{item.cur_page_msg}</Text>
                </div>
              ) : null}
            </div>
          ) : null}
        </div>

        <div className={`kb-lib-file-actions${showPrimaryConvertAction ? ' has-convert' : ' is-compact'}`}>
          <Button className="kb-lib-file-action-main" size="small" onClick={() => openMetaEditor(item)}>
            {S.lib_btn_categorize}
          </Button>
          {item.md_exists ? (
            <Button
              className="kb-lib-file-action-link"
              type="text"
              size="small"
              disabled={!item.md_path}
              onClick={() => { void handleStartPaperGuide(item) }}
            >
              {S.lib_btn_read}
            </Button>
          ) : null}
          {showPrimaryConvertAction ? (
            <Button
              className="kb-lib-file-action-link is-accent"
              type="text"
              size="small"
              disabled={item.task_state !== 'idle'}
              onClick={() => { void handleConvertOne(item) }}
            >
              {S.lib_btn_convert}
            </Button>
          ) : null}
          <Button className="kb-lib-file-action-link" type="text" size="small" onClick={() => { void store.openFile(item.name, 'pdf') }}>
            PDF
          </Button>
          <div className="kb-lib-file-more">
            <Dropdown
              trigger={['click']}
              menu={{
                items: [
                  ...(item.md_exists
                    ? [{ key: 'reconvert', label: S.lib_btn_reconvert, disabled: item.task_state !== 'idle', icon: <ReloadOutlined /> }]
                    : []),
                  { key: 'open-md', label: S.lib_btn_open_md, disabled: !item.md_exists },
                  { type: 'divider' },
                  { key: 'delete', label: S.lib_btn_delete, danger: true, disabled: item.task_state !== 'idle', icon: <DeleteOutlined /> },
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
      return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={S.lib_empty_category} />
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
                <div className="kb-lib-category-card-empty">{S.lib_tag_empty_common}</div>
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
      return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={S.lib_empty_tag} />
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
                  <span>{S.lib_tag_unread_count.replace('{n}', String(card.unreadCount))}</span>
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
          <Text type="secondary" className="text-xs">{S.lib_virtual_scroll_hint.replace('{n}', String(items.length))}</Text>
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

  const directoriesConfigured = Boolean(pdfDirDraft.trim() && mdDirDraft.trim())
  const showDirEditor = dirEditorOpen || !directoriesConfigured
  const workbenchStats = [
    { key: 'view', label: S.lib_stats_view, value: counts.total_view },
    { key: 'pending', label: S.lib_stats_pending, value: counts.pending },
    { key: 'converted', label: S.lib_stats_converted, value: counts.converted },
    { key: 'queued', label: S.lib_stats_queued, value: counts.queued },
    { key: 'running', label: S.lib_stats_running, value: counts.running },
  ]

  const renameHasResults = renameItems.length > 0
  const renameHasVisibleItems = renameVisible.length > 0
  const hasRenameSelection = selectedRenameCount > 0
  const showUploadWorkbench = uploadWorkbenchOpen && uploadDrafts.length > 0
  const showTaxonomySelectAction = browseMode === 'list' && currentListItems.length > 0
  const showTaxonomyRefreshAction = browseMode === 'list' && visibleAll.length > 0
  const showTaxonomyClearAction = hasActiveTaxonomyFilters
  const showTaxonomyTopActions = showTaxonomySelectAction || showTaxonomyRefreshAction || showTaxonomyClearAction

  const renameWorkbenchSection = (
    <section className="kb-lib-workbench-section kb-lib-workbench-section-rename">
      <div className="kb-lib-section-head">
        <div className="kb-lib-section-copy">
          <Text className="kb-lib-section-title">{S.lib_section_rename}</Text>
        </div>
      </div>

      <div className="kb-lib-rename-summary">
        <div className="kb-lib-rename-summary-main">
          <Select value={renameScope} onChange={setRenameScope} className="kb-lib-rename-scope" options={RENAME_SCOPE_OPTIONS(S)} />
          <Button size="small" className="kb-lib-action-tonal" loading={renameLoading} onClick={() => { void scanRenameSuggestions() }}>
            {renameHasResults ? S.lib_rename_recheck : S.lib_btn_rename_check}
          </Button>
          {renameHasVisibleItems ? (
            <Button className="kb-lib-action-quiet" size="small" onClick={() => setRenameResultsOpen((open) => !open)}>
              {renameResultsOpen ? S.lib_rename_collapse : S.lib_rename_expand}
            </Button>
          ) : null}
          {renameHasVisibleItems ? (
            <Button className="kb-lib-action-quiet" size="small" onClick={selectRenameDiffItems}>{S.lib_btn_select_all}</Button>
          ) : null}
          {hasRenameSelection ? (
            <Button className="kb-lib-action-quiet" size="small" onClick={clearRenameSelection}>{S.lib_btn_clear}</Button>
          ) : null}
          {hasRenameSelection ? (
            <Button className="kb-lib-action-tonal" size="small" type="primary" loading={renameApplying} onClick={() => { void applyRenameSuggestions() }}>
              {S.lib_btn_apply_rename}
            </Button>
          ) : null}
        </div>
        {renameHasResults ? (
          <div className="kb-lib-rename-summary-side">
            <div className="kb-lib-rename-badges">
              <span className="kb-lib-rename-meta">{S.lib_rename_meta_format.replace('{sel}', String(selectedRenameCount)).replace('{vis}', String(renameVisible.length)).replace('{total}', String(renameItems.length))}</span>
            </div>
          </div>
        ) : null}
      </div>

      {renameHasResults && renameHasVisibleItems && renameResultsOpen ? (
        <List
          className="kb-lib-rename-list"
          size="small"
          locale={{ emptyText: <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={S.lib_empty_rename} /> }}
          dataSource={renameVisible}
          pagination={{ pageSize: 6, size: 'small', showSizeChanger: false }}
          renderItem={(item) => (
            <List.Item className="kb-lib-rename-list-item">
              <div className="kb-lib-rename-item">
                <div className="kb-lib-rename-item-head">
                  <Checkbox
                    checked={Boolean(renameSelected[item.name])}
                    onChange={(e) => setRenameSelected((cur) => ({ ...cur, [item.name]: e.target.checked }))}
                  />
                  <Text className="kb-lib-rename-item-name">{item.name}</Text>
                  <Tag color={item.diff ? 'warning' : 'default'}>{item.diff ? S.lib_rename_suggest_rename : S.lib_rename_no_rename}</Tag>
                </div>
                <Input
                  value={renameOverrides[item.name] || ''}
                  onChange={(e) => setRenameOverrides((cur) => ({ ...cur, [item.name]: e.target.value }))}
                  className="kb-lib-rename-item-input"
                />
                <div className="flex flex-wrap items-center gap-2">
                  <Text type="secondary" className="kb-lib-rename-item-source">
                    {item.display_full_name}
                  </Text>
                  {item.meta?.basis_label ? (
                    <Tag color={suggestionBasisTagColor(item.meta)}>
                      {item.meta.basis_label}
                    </Tag>
                  ) : null}
                </div>
                {item.meta?.basis_detail ? (
                  <Text type="secondary" className="kb-lib-rename-item-source">
                    {item.meta.basis_detail}
                  </Text>
                ) : null}
              </div>
            </List.Item>
          )}
        />
      ) : null}
      {renameHasResults && !renameHasVisibleItems ? (
        <Text type="secondary" className="kb-lib-section-note">
          {S.lib_rename_no_files}
        </Text>
      ) : null}
    </section>
  )

  const preparationWorkbench = (
    <Card size="small" className="kb-lib-card kb-lib-workbench-card" title={S.lib_prep_workbench}>
      <div className="kb-lib-workbench">
        <div className="kb-lib-workbench-main">
          <section className="kb-lib-workbench-section">
            <div className="kb-lib-section-head">
              <div className="kb-lib-section-copy">
                <Text className="kb-lib-section-title">{S.lib_section_dir}</Text>
              </div>
              {directoriesConfigured ? (
                <Button className="kb-lib-action-quiet" onClick={() => setDirEditorOpen((open) => !open)}>
                  {showDirEditor ? S.lib_dir_collapse : S.lib_dir_edit}
                </Button>
              ) : null}
            </div>

            <div className="kb-lib-dir-summary">
              <div className={`kb-lib-dir-summary-row${showDirEditor ? ' is-editing' : ''}`}>
                <Text className="kb-lib-dir-summary-label">PDF</Text>
                {showDirEditor ? (
                  <Input
                    value={pdfDirDraft}
                    placeholder={S.lib_dir_select_pdf}
                    onChange={(e) => {
                      setDirTouched(true)
                      setPdfDirDraft(e.target.value)
                    }}
                  />
                ) : (
                  <Text className="kb-lib-dir-summary-path" ellipsis={{ tooltip: pdfDirDraft || S.lib_dir_no_pdf }}>
                    {pdfDirDraft || S.lib_dir_no_pdf}
                  </Text>
                )}
                {showDirEditor ? (
                  <Button className="kb-lib-action-quiet" loading={pickingDir === 'pdf'} onClick={() => { void pickDir('pdf') }}>{S.lib_dir_pick}</Button>
                ) : null}
                <Button className="kb-lib-action-quiet" icon={<FolderOpenOutlined />} disabled={!pdfDirDraft.trim()} onClick={() => { void openFolder('pdf_dir') }}>{S.lib_dir_open}</Button>
              </div>
              <div className={`kb-lib-dir-summary-row${showDirEditor ? ' is-editing' : ''}`}>
                <Text className="kb-lib-dir-summary-label">MD</Text>
                {showDirEditor ? (
                  <Input
                    value={mdDirDraft}
                    placeholder={S.lib_dir_select_md}
                    onChange={(e) => {
                      setDirTouched(true)
                      setMdDirDraft(e.target.value)
                    }}
                  />
                ) : (
                  <Text className="kb-lib-dir-summary-path" ellipsis={{ tooltip: mdDirDraft || S.lib_dir_no_md }}>
                    {mdDirDraft || S.lib_dir_no_md}
                  </Text>
                )}
                {showDirEditor ? (
                  <Button className="kb-lib-action-quiet" loading={pickingDir === 'md'} onClick={() => { void pickDir('md') }}>{S.lib_dir_pick}</Button>
                ) : null}
                <Button className="kb-lib-action-quiet" icon={<FolderOpenOutlined />} disabled={!mdDirDraft.trim()} onClick={() => { void openFolder('md_dir') }}>{S.lib_dir_open}</Button>
              </div>
            </div>

            {showDirEditor ? (
              <div className="kb-lib-section-actions">
                <Button className="kb-lib-action-tonal" type="primary" icon={<SaveOutlined />} loading={savingDirs} disabled={!dirDirty} onClick={() => { void saveDirs() }}>{S.lib_dir_save}</Button>
              </div>
            ) : null}
          </section>

          {renameWorkbenchSection}
        </div>

        <div className="kb-lib-workbench-side">
          <section className="kb-lib-workbench-section kb-lib-workbench-section-upload">
            <div className="kb-lib-section-head">
              <div className="kb-lib-section-copy">
                <Text className="kb-lib-section-title">{S.lib_upload_title}</Text>
              </div>
            </div>

            <Dragger
              multiple
              accept=".pdf"
              disabled={uploadLocked}
              showUploadList={false}
              className={`kb-lib-upload-dropzone${uploadLocked ? ' is-locked' : ''}`}
              beforeUpload={(file) => {
                addDrafts([file as File])
                return false
              }}
            >
              <div className="kb-lib-upload-dropzone-copy">
                <UploadOutlined className="kb-lib-upload-dropzone-icon" />
                <Text className="kb-lib-upload-dropzone-title">{S.lib_upload_drop_hint}</Text>
                <Text type="secondary" className="kb-lib-upload-dropzone-note">{S.lib_upload_click_hint}</Text>
              </div>
            </Dragger>

            {(uploadDrafts.length > 0 || uploadLocked) ? (
              <div className="kb-lib-upload-meta">
                {uploadDrafts.length > 0 ? (
                  <div className="kb-lib-upload-meta-main">
                    <span className="kb-lib-rename-meta">{S.lib_workbench_draft_count.replace('{n}', String(uploadDrafts.length))}</span>
                    <Button className="kb-lib-action-quiet" onClick={() => setUploadWorkbenchOpen((open) => !open)}>
                      {showUploadWorkbench ? S.lib_workbench_hide_queue : S.lib_workbench_upload_queue}
                    </Button>
                  </div>
                ) : null}
                {uploadLocked ? (
                  <Text type="secondary" className="kb-lib-upload-inline-note">
                    {store.converting ? S.lib_upload_locked_converting : S.lib_upload_locked_refsync}
                  </Text>
                ) : null}
              </div>
            ) : null}
          </section>

          <section className="kb-lib-workbench-section kb-lib-workbench-section-process">
            <div className="kb-lib-section-head">
              <div className="kb-lib-section-copy">
                <Text className="kb-lib-section-title">{S.lib_section_batch}</Text>
              </div>
            </div>

            <div className="kb-lib-process-toolbar">
              <div className="kb-lib-process-toolbar-main">
                <Select
                  value={scope}
                  onChange={(value) => { setScope(value); void store.loadFiles(value) }}
                  className="kb-lib-process-scope"
                  options={SCOPE_OPTIONS(S)}
                />
                <Button className="kb-lib-action-tonal" type="primary" onClick={() => { void handleConvertPending() }}>{S.lib_btn_convert_pending_short}</Button>
              </div>
              <div className="kb-lib-process-toolbar-side">
                <Button className="kb-lib-action-quiet kb-lib-process-refresh" icon={<ReloadOutlined />} onClick={() => { void store.loadFiles(scope) }}>{S.lib_btn_refresh}</Button>
                {store.converting ? <Button icon={<StopOutlined />} danger onClick={() => { void store.cancelConvert() }}>{S.lib_btn_stop}</Button> : null}
              </div>
            </div>
          </section>
        </div>
      </div>
    </Card>
  )

  const uploadWorkbenchCard = showUploadWorkbench ? (
    <Card
      size="small"
      className="kb-lib-card kb-lib-upload-workbench-card"
      title={S.lib_section_upload_workbench}
      extra={(
        <Space size={8}>
          <Text type="secondary" className="text-xs">{S.lib_upload_selected_count.replace('{n}', String(selectedUploadCount))}</Text>
          <Text type="secondary" className="text-xs">{S.lib_upload_show_count.replace('{n}', String(filteredUploadDrafts.length)).replace('{total}', String(uploadDrafts.length))}</Text>
          <Button size="small" onClick={() => setUploadWorkbenchOpen(false)}>{S.lib_btn_collapse}</Button>
        </Space>
      )}
    >
      <div className="space-y-3">
        <div className="kb-lib-upload-toolbar flex flex-wrap items-center gap-2">
          <Switch checked={uploadUseLlm} onChange={setUploadUseLlm} />
          <Text className="text-sm text-[var(--muted)]">{S.lib_upload_use_llm}</Text>
          <Select
            value={uploadDraftFilter}
            onChange={(value) => applyUploadFilter(value as UploadDraftFilter)}
            options={uploadDraftFilterOptions}
            className="kb-lib-upload-filter"
          />
          <Tooltip title={S.lib_btn_select_all}><Button icon={<CheckOutlined />} onClick={selectAllUploadDrafts}>{S.lib_btn_select_all}</Button></Tooltip>
          <Tooltip title={S.lib_btn_invert_select}><Button icon={<ClearOutlined />} onClick={invertUploadDraftSelection}>{S.lib_btn_invert_select}</Button></Tooltip>
          <Button loading={uploadInspecting} disabled={uploadLocked} onClick={() => { void inspectSelectedDrafts() }}>{S.lib_btn_scan_selected}</Button>
          <Button loading={uploadSaving} disabled={uploadLocked} onClick={() => { void saveSelectedDrafts(false) }}>{S.lib_btn_save_selected}</Button>
          <Button type="primary" loading={uploadSaving} disabled={uploadLocked} onClick={() => { void saveSelectedDrafts(true) }}>{S.lib_btn_save_and_convert}</Button>
          <Button disabled={uploadLocked} onClick={selectFailedDrafts}>{S.lib_btn_select_failed}</Button>
          <Button disabled={uploadLocked || duplicateFailedDrafts.length === 0} onClick={showDuplicateFailedDrafts}>{S.lib_btn_view_dup_failed}</Button>
          <Button loading={uploadSaving} disabled={uploadLocked || failedUploadDrafts.length === 0} onClick={() => { void retryFailedDrafts(false) }}>{S.lib_btn_retry_failed}</Button>
          <Button type="primary" loading={uploadSaving} disabled={uploadLocked || failedUploadDrafts.length === 0} onClick={() => { void retryFailedDrafts(true) }}>{S.lib_btn_retry_and_convert}</Button>
          <Button disabled={uploadLocked} onClick={() => setUploadDrafts((cur) => cur.filter((x) => x.status !== 'saved'))}>{S.lib_btn_clear_saved}</Button>
        </div>

        {(uploadDraftFilter === 'error' || uploadDraftFilter === 'dup_error') && uploadErrorReason !== 'all' ? (
          <div className="kb-lib-upload-meta flex flex-wrap items-center gap-3">
            <Button size="small" onClick={() => setUploadErrorReason('all')}>
              {S.lib_upload_filter_reason.replace('{reason}', activeErrorReasonText)}
            </Button>
          </div>
        ) : null}

        {failedUploadDrafts.length > 0 ? (
          <Alert
            type="warning"
            showIcon
            message={S.lib_upload_failed_drafts.replace('{n}', String(failedUploadDrafts.length))}
            description={(
              <div className="kb-lib-failed-summary">
                <div className="kb-lib-failed-reasons">
                  {failedReasonBuckets.map((bucket) => (
                    <Button
                      key={bucket.key}
                      size="small"
                      icon={FAILED_REASON_META(S)[bucket.key].icon}
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
                  {failedUploadNotes.length > 0 ? failedUploadNotes.join(' | ') : S.lib_upload_error_hint}
                </Text>
              </div>
            )}
          />
        ) : null}

        <List
          size="small"
          locale={{ emptyText: <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={S.lib_upload_empty} /> }}
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
                      {DRAFT_STATUS_TEXT(S)[x.status]}
                    </Tag>
                    {reasonKey ? (
                      <span className={`kb-lib-inline-reason-chip kb-lib-reason-tone is-${reasonKey}`}>
                        {FAILED_REASON_META(S)[reasonKey].icon}
                        <span>{FAILED_REASON_META(S)[reasonKey].label}</span>
                      </span>
                    ) : null}
                  </div>
                  <div className="flex flex-wrap items-center gap-2 pl-6">
                    <Text type="secondary" className="text-xs">{S.lib_upload_suggest_name}</Text>
                    <Input value={x.stem} onChange={(e) => setUploadDrafts((cur) => cur.map((t) => (t.key === x.key ? { ...t, stem: e.target.value } : t)))} className="w-[24rem] max-w-full" />
                    <Button size="small" disabled={uploadLocked || x.status === 'saving' || x.status === 'inspecting'} onClick={() => { void inspectDraft(x.key) }}>{S.lib_btn_scan}</Button>
                    <Button size="small" disabled={uploadLocked || x.status === 'saving' || x.status === 'saved' || x.status === 'inspecting'} onClick={() => { void saveDraft(x.key, false) }}>{S.lib_btn_save}</Button>
                    <Button size="small" type="primary" disabled={uploadLocked || x.status === 'saving' || x.status === 'saved' || x.status === 'inspecting'} onClick={() => { void saveDraft(x.key, true) }}>{S.lib_btn_save_and_convert}</Button>
                  </div>
                  <div className="flex flex-wrap items-center gap-2 pl-6">
                    <Text type="secondary" className="text-xs">{x.displayName}</Text>
                    {x.suggestionBasisLabel ? (
                      <Tag color={suggestionBasisTagColor({ match_method: x.suggestionMatchMethod, year_source: x.suggestionYearSource })}>
                        {x.suggestionBasisLabel}
                      </Tag>
                    ) : null}
                  </div>
                  {x.suggestionBasisDetail ? (
                    <Text type="secondary" className="block pl-6 text-xs">{x.suggestionBasisDetail}</Text>
                  ) : null}
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
    </Card>
  ) : null

  return (
    <div className="kb-library-page mx-auto w-full max-w-[1760px] space-y-5 p-5">
      <div className="kb-lib-head flex flex-wrap items-end justify-between gap-3">
        <div className="kb-lib-head-main">
          <Text className="text-2xl font-semibold">{S.page_library}</Text>
          <div>
            <Text type="secondary" className="text-sm">{S.lib_page_subtitle}</Text>
          </div>
        </div>
        <Space wrap className="kb-lib-head-actions">
          <Button className="kb-lib-head-btn" icon={<ReloadOutlined />} type="primary" onClick={() => { void handleReindex() }}>{S.reindex_now}</Button>
          <Button className="kb-lib-head-btn" icon={<ReloadOutlined />} onClick={() => { void handleStartRefSync() }}>{S.lib_btn_sync_refs}</Button>
        </Space>
      </div>

      <div className="kb-lib-summary-strip">
        {workbenchStats.map((item) => (
          <div key={item.key} className="kb-lib-summary-chip">
            <Text type="secondary" className="kb-lib-summary-label">{item.label}</Text>
            <div className="kb-lib-summary-value">{item.value}</div>
          </div>
        ))}
      </div>

      {preparationWorkbench}
      {uploadWorkbenchCard}

      <div className="kb-lib-stats-grid grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
        <Card size="small" className="kb-lib-stat"><Text type="secondary">{S.lib_stats_view}</Text><div className="kb-lib-stat-value">{counts.total_view}</div></Card>
        <Card size="small" className="kb-lib-stat"><Text type="secondary">{S.lib_stats_pending}</Text><div className="kb-lib-stat-value">{counts.pending}</div></Card>
        <Card size="small" className="kb-lib-stat"><Text type="secondary">{S.lib_stats_converted}</Text><div className="kb-lib-stat-value">{counts.converted}</div></Card>
        <Card size="small" className="kb-lib-stat"><Text type="secondary">{S.lib_stats_queued}</Text><div className="kb-lib-stat-value">{counts.queued}</div></Card>
        <Card size="small" className="kb-lib-stat"><Text type="secondary">{S.lib_stats_running}</Text><div className="kb-lib-stat-value">{counts.running}</div></Card>
      </div>

      {showStickyStatus ? (
        <Card size="small" className="kb-lib-card kb-lib-sticky-status">
          <div className="kb-lib-sticky-wrap">
            {store.converting && store.progress ? (
              <div className="kb-lib-sticky-item">
                <div className="kb-lib-sticky-main">
                  <Text className="kb-lib-sticky-title">{S.lib_convert_progress.replace('{done}', String(store.progress.completed)).replace('{total}', String(store.progress.total))}</Text>
                  {store.progress.current ? <Text type="secondary" className="kb-lib-sticky-sub">{store.progress.current}</Text> : null}
                  {convertActiveSummary ? <Text type="secondary" className="kb-lib-sticky-sub">{convertActiveSummary}</Text> : null}
                  {convertStageLabel ? <Text type="secondary" className="kb-lib-sticky-sub">{convertStageLabel}</Text> : null}
                  {convertPageProgress.total > 0 ? (
                    <Text type="secondary" className="kb-lib-sticky-sub">
                      {S.lib_convert_page_progress} {convertPageProgress.done}/{convertPageProgress.total}
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
                  {S.lib_btn_stop}
                </Button>
              </div>
            ) : null}

            {store.refSync?.running ? (
              <div className="kb-lib-sticky-item">
                <div className="kb-lib-sticky-main">
                  <Text className="kb-lib-sticky-title">{S.lib_refsync_title}</Text>
                  <Text type="secondary" className="kb-lib-sticky-sub">
                    {store.refSync.current
                      ? `${store.refSync.stage || S.lib_refsync_running} | ${store.refSync.current}`
                      : (store.refSync.message || S.lib_refsync_waiting)}
                  </Text>
                </div>
                <Progress className="kb-lib-sticky-progress" percent={refSyncPercent} status="active" size="small" />
                <Tag color="processing">{S.lib_refsync_running}</Tag>
              </div>
            ) : null}
          </div>
        </Card>
      ) : null}

      <Card size="small" className="kb-lib-card kb-lib-legacy-convert-card" title={S.lib_convert_scope}>
        <div className="kb-lib-convert-shell">
          <div className="kb-lib-convert-row kb-lib-convert-row-top">
            <Select
              value={scope}
              onChange={(value) => { setScope(value); void store.loadFiles(value) }}
              className="kb-lib-convert-scope"
              options={SCOPE_OPTIONS(S)}
            />
            <Input
              value={fileKeyword}
              onChange={(e) => setFileKeyword(e.target.value)}
              allowClear
              prefix={<SearchOutlined className="opacity-50" />}
              placeholder={S.lib_filter_filename}
              className="kb-lib-convert-search"
            />
            <Button className="kb-lib-convert-refresh" icon={<ReloadOutlined />} onClick={() => { void store.loadFiles(scope) }}>
              {S.lib_btn_refresh}
            </Button>
          </div>

          <div className="kb-lib-convert-row kb-lib-convert-row-filters">
            <Select
              value={paperCategoryFilter || undefined}
              allowClear
              placeholder={S.lib_filter_category}
              className="kb-lib-convert-filter"
              options={paperCategoryFilterOptions}
              onChange={(value) => setPaperCategoryFilter(String(value || ''))}
            />
            <Select
              value={paperTagFilter || undefined}
              allowClear
              showSearch
              placeholder={S.lib_filter_tag}
              className="kb-lib-convert-filter"
              options={paperTagFilterOptions}
              optionFilterProp="label"
              onChange={(value) => setPaperTagFilter(String(value || ''))}
            />
            <Select
              value={readingStatusFilter || undefined}
              allowClear
              placeholder={S.lib_filter_reading}
              className="kb-lib-convert-filter"
              options={READING_STATUS_OPTIONS(S).filter((item) => item.value)}
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
              {S.lib_btn_clear_metadata_filter}
            </Button>
          </div>

          <div className="kb-lib-convert-row kb-lib-convert-row-actions">
            <Button type="primary" onClick={() => { void handleConvertPending() }}>{S.lib_btn_convert_pending}</Button>
            {store.converting ? <Button icon={<StopOutlined />} danger onClick={() => { void store.cancelConvert() }}>{S.lib_btn_stop}</Button> : null}
          </div>
        </div>
      </Card>

      {store.refSync && !store.refSync.running && (store.refSync.status === 'error' || Boolean(store.refSync.error)) ? (
        <Card size="small" className="kb-lib-card" title={S.lib_card_refsync}>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Text type="secondary" className="text-xs">
                {store.refSync.current
                  ? `${store.refSync.stage || S.lib_refsync_running} | ${store.refSync.current}`
                  : (store.refSync.message || S.lib_refsync_waiting)}
              </Text>
              <Tag color={store.refSync.running ? 'processing' : (store.refSync.status === 'error' ? 'error' : 'default')}>
                {store.refSync.running ? S.lib_refsync_running : (store.refSync.status === 'idle' ? S.lib_refsync_idle : store.refSync.status)}
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

      <Card size="small" className="kb-lib-card kb-lib-taxonomy-bar" title={S.lib_taxonomy_title}>
        <div className="kb-lib-taxonomy-shell">
          <div className="kb-lib-taxonomy-top">
            <div className="kb-lib-taxonomy-view">
              <Segmented
                className="kb-lib-browse-switch"
                value={browseMode}
                onChange={(value) => setBrowseMode(value as LibraryBrowseMode)}
                options={[
                  { label: S.lib_browse_list, value: 'list' },
                  { label: S.lib_browse_categories, value: 'categories' },
                  { label: S.lib_browse_tags, value: 'tags' },
                ]}
              />
            </div>
            <div className="kb-lib-taxonomy-meta">
              <div className="kb-lib-taxonomy-summary">
                <Text type="secondary" className="kb-lib-taxonomy-result">
                  {S.lib_taxonomy_result.replace('{n}', String(visibleAll.length)).replace('{total}', String(store.files.length))}
                </Text>
                {hasActiveTaxonomyFilters ? (
                  <span className="kb-lib-taxonomy-status-pill">
                    {S.lib_taxonomy_filtering.replace('{n}', String(activeTaxonomyFilterCount))}
                  </span>
                ) : null}
              </div>
              {showTaxonomyTopActions ? (
                <div className="kb-lib-taxonomy-top-actions">
                  {showTaxonomySelectAction ? (
                    <Button className="kb-lib-action-quiet" onClick={selectCurrentListItems}>
                      {S.lib_btn_select_current_list}
                    </Button>
                  ) : null}
                  {showTaxonomyRefreshAction ? (
                    <Button
                      className="kb-lib-action-tonal"
                      loading={suggestionsRefreshing}
                      onClick={() => { void regenerateSuggestionsForVisible() }}
                    >
                      {S.lib_btn_refresh_suggestions}
                    </Button>
                  ) : null}
                  {showTaxonomyClearAction ? (
                    <Button className="kb-lib-action-quiet" onClick={clearTaxonomyFilters}>
                      {S.lib_btn_clear_filters}
                    </Button>
                  ) : null}
                </div>
              ) : null}
            </div>
          </div>

          <div className="kb-lib-taxonomy-controls">
            <div className="kb-lib-taxonomy-filters">
              <Input
                value={fileKeyword}
                onChange={(e) => setFileKeyword(e.target.value)}
                allowClear
                prefix={<SearchOutlined className="opacity-50" />}
                placeholder={S.lib_search_placeholder}
                className="kb-lib-taxonomy-search"
              />
              <Select
                value={paperCategoryFilter || undefined}
                allowClear
                placeholder={S.lib_search_category}
                className="kb-lib-taxonomy-select"
                options={paperCategoryFilterOptions}
                onChange={(value) => applyPaperCategoryFilter(String(value || ''))}
              />
              <Select
                value={paperTagFilter || undefined}
                allowClear
                showSearch
                placeholder={S.lib_search_tag}
                className="kb-lib-taxonomy-select"
                options={paperTagFilterOptions}
                optionFilterProp="label"
                onChange={(value) => applyPaperTagFilter(String(value || ''))}
              />
              <Select
                value={readingStatusFilter || undefined}
                allowClear
                placeholder={S.lib_search_reading}
                className="kb-lib-taxonomy-select"
                options={READING_STATUS_OPTIONS(S).filter((item) => item.value)}
                onChange={(value) => setReadingStatusFilter(String(value || '') as ReadingStatusValue)}
              />
            </div>

            <div className="kb-lib-taxonomy-quick">
              <div className="kb-lib-taxonomy-toggle-row">
                <button
                  type="button"
                  className={`kb-lib-taxonomy-pill is-status${onlyUnread ? ' is-active' : ''}`}
                  onClick={() => setOnlyUnread((value) => !value)}
                >
                  {S.lib_taxonomy_unread}
                </button>
                <button
                  type="button"
                  className={`kb-lib-taxonomy-pill is-category${onlyUnclassified ? ' is-active' : ''}`}
                  onClick={() => {
                    const next = !onlyUnclassified
                    setOnlyUnclassified(next)
                    if (next) setPaperCategoryFilter('')
                  }}
                >
                  {S.lib_category_unclassified}
                </button>
                <button
                  type="button"
                  className={`kb-lib-taxonomy-pill is-suggestion${onlySuggested ? ' is-active' : ''}`}
                  onClick={() => setOnlySuggested((value) => !value)}
                >
                  {S.lib_taxonomy_has_suggestions}
                </button>
              </div>
            </div>
          </div>
        </div>
      </Card>

      {browseMode === 'list' && selectedLibraryCount > 0 ? (
        <Card size="small" className="kb-lib-card kb-lib-batch-card">
          <div className="kb-lib-batch-bar">
            <div className="kb-lib-batch-summary">
              <div className="kb-lib-batch-badges">
                <span className="kb-lib-batch-badge is-strong">{S.lib_batch_selected_count.replace('{n}', String(selectedLibraryCount))}</span>
                <span className="kb-lib-batch-badge">{S.lib_batch_current_count.replace('{n}', String(currentListItems.length))}</span>
              </div>
              <Text className="kb-lib-batch-count">{S.lib_batch_title_selected}</Text>
              <Text type="secondary" className="kb-lib-batch-hint">{S.lib_batch_hint_scope}</Text>
            </div>
            <div className="kb-lib-batch-actions">
              <Button onClick={selectCurrentListItems}>{S.lib_btn_select_current_list}</Button>
              <Button onClick={clearLibrarySelection} disabled={!selectedLibraryCount}>{S.lib_btn_clear_selection}</Button>
              <Button type="primary" onClick={openBatchEditor} disabled={!selectedLibraryCount}>{S.lib_batch_title}</Button>
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
          { key: 'pending', label: S.lib_tab_pending.replace('{n}', String(visiblePending.length)), children: renderFiles(visiblePending, S.lib_empty_pending) },
          { key: 'converted', label: S.lib_tab_converted.replace('{n}', String(visibleConverted.length)), children: renderFiles(visibleConverted, S.lib_empty_converted) },
          { key: 'all', label: S.lib_tab_all.replace('{n}', String(visibleAll.length)), children: renderFiles(visibleAll, S.lib_empty_all) },
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
        title={metaItem ? S.lib_meta_title.replace('{name}', metaItem.name) : S.lib_meta_title_fallback}
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
                  {S.lib_meta_hero_hint}
                </Text>
              </div>
              <Space wrap size={[6, 6]} className="kb-lib-meta-chip-row">
                <Tag color={metaDraftCategory ? 'blue' : 'default'}>{metaDraftCategory || S.lib_category_unclassified}</Tag>
                {metaDraft.reading_status ? (
                  <Tag color="gold">{readingStatusLabel(metaDraft.reading_status, S)}</Tag>
                ) : (
                  <Tag>{S.lib_meta_status_not_set}</Tag>
                )}
                <Tag color={metaSuggestionCount ? 'processing' : 'default'}>
                  {metaSuggestionCount ? S.lib_meta_suggestions.replace('{n}', String(metaSuggestionCount)) : S.lib_meta_no_suggestions}
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
                <Text className="kb-lib-meta-section-title">{S.lib_meta_section_my_org}</Text>
                <Text type="secondary" className="kb-lib-meta-section-note">
                  {S.lib_meta_org_hint}
                </Text>
              </div>
            </div>

            <div className="kb-lib-meta-field">
              <Text type="secondary" className="kb-lib-meta-label">{S.lib_meta_label_category}</Text>
              <AutoComplete
                value={metaDraft.paper_category}
                allowClear
                options={paperCategoryOptions}
                placeholder={S.lib_meta_category_placeholder}
                filterOption={optionMatchesInput}
                onChange={(value) => setMetaDraft((cur) => ({ ...cur, paper_category: String(value || '') }))}
                onBlur={() => setMetaDraft((cur) => ({ ...cur, paper_category: normalizeTextValue(cur.paper_category) }))}
              />
              <Text type="secondary" className="kb-lib-meta-help">
                {S.lib_meta_category_hint}
              </Text>
            </div>

            <div className="kb-lib-meta-field">
              <Text type="secondary" className="kb-lib-meta-label">{S.lib_meta_label_status}</Text>
              <Select
                value={metaDraft.reading_status || undefined}
                allowClear
                placeholder={S.lib_meta_reading_placeholder}
                options={READING_STATUS_OPTIONS(S).filter((item) => item.value)}
                onChange={(value) => setMetaDraft((cur) => ({ ...cur, reading_status: String(value || '') as ReadingStatusValue }))}
              />
            </div>

            <div className="kb-lib-meta-field">
              <Text type="secondary" className="kb-lib-meta-label">{S.lib_meta_label_tags}</Text>
              <Select
                mode="tags"
                value={metaDraft.user_tags}
                showSearch
                maxTagCount="responsive"
                tokenSeparators={TAG_INPUT_SEPARATORS}
                placeholder={S.lib_meta_tag_placeholder}
                options={paperTagOptions}
                optionFilterProp="label"
                onChange={(value) => setMetaDraft((cur) => ({ ...cur, user_tags: normalizeTextList(value as unknown[]) }))}
              />
              <Text type="secondary" className="kb-lib-meta-help">
                {S.lib_meta_tags_hint}
              </Text>
            </div>

            <div className="kb-lib-meta-field">
              <Text type="secondary" className="kb-lib-meta-label">{S.lib_meta_label_note}</Text>
              <Input.TextArea
                autoSize={{ minRows: 5, maxRows: 9 }}
                value={metaDraft.note}
                placeholder={S.lib_meta_note_placeholder}
                onChange={(event) => setMetaDraft((cur) => ({ ...cur, note: event.target.value }))}
              />
            </div>
          </section>

          <section className="kb-lib-meta-section kb-lib-meta-section-suggest">
            <div className="kb-lib-suggest-head">
              <div className="kb-lib-meta-section-copy">
                <Text className="kb-lib-meta-section-title">{S.lib_meta_section_system}</Text>
                <Text type="secondary" className="kb-lib-meta-section-note">
                  {S.lib_meta_system_hint}
                </Text>
              </div>
              <Space size={8} wrap>
                <Button size="small" loading={metaSuggestionSaving} onClick={() => { void regenerateMetaSuggestions() }}>
                  {S.lib_btn_refresh_suggestions}
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
                      {S.lib_btn_accept_all}
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
                      {S.lib_btn_dismiss_all}
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
                      <Text className="kb-lib-suggest-title">{S.lib_meta_suggest_category}</Text>
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
                        {S.lib_btn_accept}
                      </Button>
                      <Button
                        size="small"
                        loading={metaSuggestionSaving}
                        onClick={() => { void applyMetaSuggestionAction({ category_action: 'dismiss' }) }}
                      >
                        {S.lib_btn_dismiss}
                      </Button>
                    </Space>
                  </div>
                ) : null}

                {(metaItem?.suggested_tags || []).map((tagValue) => (
                  <div key={`meta-suggest-${tagValue}`} className="kb-lib-suggest-item">
                    <div className="kb-lib-suggest-copy">
                      <Text className="kb-lib-suggest-title">{S.lib_meta_suggest_tags}</Text>
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
                        {S.lib_btn_accept}
                      </Button>
                      <Button
                        size="small"
                        loading={metaSuggestionSaving}
                        onClick={() => { void applyMetaSuggestionAction({ dismiss_tags: [tagValue] }) }}
                      >
                        {S.lib_btn_dismiss}
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
                message={S.lib_meta_no_suggestions_msg}
                description={S.lib_batch_hint}
              />
            )}
          </section>

          <div className="kb-lib-meta-actions">
            <Button onClick={() => setMetaDrawerOpen(false)}>
              {S.lib_btn_cancel}
            </Button>
            <Button type="primary" loading={metaSaving} onClick={() => { void saveMetaEditor() }}>
              {S.lib_btn_save}
            </Button>
          </div>
        </div>
      </Drawer>

      <Drawer
        title={S.lib_batch_edit_count_format.replace('{n}', String(selectedLibraryCount))}
        open={batchDrawerOpen}
        width={420}
        onClose={() => setBatchDrawerOpen(false)}
        destroyOnClose={false}
      >
        <div className="kb-lib-meta-drawer">
          <div className="kb-lib-meta-hero kb-lib-meta-hero-batch">
            <div className="kb-lib-meta-hero-copy">
              <Text className="kb-lib-meta-hero-title">{S.lib_batch_edit_hero.replace('{n}', String(selectedLibraryCount))}</Text>
              <Text type="secondary" className="kb-lib-meta-hero-note">
                {S.lib_batch_notice}
              </Text>
            </div>
            <Space wrap size={[6, 6]} className="kb-lib-meta-chip-row">
              <Tag color={selectedLibraryCount ? 'blue' : 'default'}>{S.lib_batch_selected_tag.replace('{n}', String(selectedLibraryCount))}</Tag>
              {batchDraft.apply_paper_category && normalizeTextValue(batchDraft.paper_category) ? (
                <Tag color="processing">{S.lib_batch_set_category_label.replace('{category}', normalizeTextValue(batchDraft.paper_category))}</Tag>
              ) : null}
              {batchDraft.add_tags.length ? (
                <Tag color="green">{S.lib_batch_add_tag_count.replace('{n}', String(normalizeTextList(batchDraft.add_tags).length))}</Tag>
              ) : null}
            </Space>
          </div>

          <section className="kb-lib-meta-section">
            <div className="kb-lib-meta-section-head">
              <div className="kb-lib-meta-section-copy">
                <Text className="kb-lib-meta-section-title">{S.lib_batch_section_setting}</Text>
                <Text type="secondary" className="kb-lib-meta-section-note">
                  {S.lib_batch_setting_hint}
                </Text>
              </div>
            </div>

            <div className={`kb-lib-meta-field ${batchDraft.apply_paper_category ? '' : 'is-muted'}`}>
              <Checkbox
                checked={batchDraft.apply_paper_category}
                onChange={(event) => setBatchDraft((cur) => ({ ...cur, apply_paper_category: event.target.checked }))}
              >
                {S.lib_batch_set_category_cb}
              </Checkbox>
              <AutoComplete
                value={batchDraft.paper_category}
                allowClear
                disabled={!batchDraft.apply_paper_category}
                options={paperCategoryOptions}
                placeholder={S.lib_meta_category_placeholder}
                filterOption={optionMatchesInput}
                onChange={(value) => setBatchDraft((cur) => ({ ...cur, paper_category: String(value || '') }))}
                onBlur={() => setBatchDraft((cur) => ({ ...cur, paper_category: normalizeTextValue(cur.paper_category) }))}
              />
              <Text type="secondary" className="kb-lib-meta-help">
                {S.lib_batch_category_hint}
              </Text>
            </div>

            <div className={`kb-lib-meta-field ${batchDraft.apply_reading_status ? '' : 'is-muted'}`}>
              <Checkbox
                checked={batchDraft.apply_reading_status}
                onChange={(event) => setBatchDraft((cur) => ({ ...cur, apply_reading_status: event.target.checked }))}
              >
                {S.lib_batch_set_status_cb}
              </Checkbox>
              <Select
                value={batchDraft.reading_status || undefined}
                allowClear
                disabled={!batchDraft.apply_reading_status}
                placeholder={S.lib_meta_reading_placeholder}
                options={READING_STATUS_OPTIONS(S).filter((item) => item.value)}
                onChange={(value) => setBatchDraft((cur) => ({ ...cur, reading_status: String(value || '') as ReadingStatusValue }))}
              />
            </div>
          </section>

          <section className="kb-lib-meta-section">
            <div className="kb-lib-meta-section-head">
              <div className="kb-lib-meta-section-copy">
                <Text className="kb-lib-meta-section-title">{S.lib_batch_section_tags}</Text>
                <Text type="secondary" className="kb-lib-meta-section-note">
                  {S.lib_batch_tags_hint}
                </Text>
              </div>
            </div>

            <div className="kb-lib-meta-field">
              <Text type="secondary" className="kb-lib-meta-label">{S.lib_batch_label_add_tags}</Text>
              <Select
                mode="tags"
                value={batchDraft.add_tags}
                showSearch
                maxTagCount="responsive"
                tokenSeparators={TAG_INPUT_SEPARATORS}
                placeholder={S.lib_batch_add_tag_placeholder}
                options={paperTagOptions}
                optionFilterProp="label"
                onChange={(value) => setBatchDraft((cur) => ({ ...cur, add_tags: normalizeTextList(value as unknown[]) }))}
              />
            </div>

            <div className="kb-lib-meta-field">
              <Text type="secondary" className="kb-lib-meta-label">{S.lib_batch_label_remove_tags}</Text>
              <Select
                mode="multiple"
                value={batchDraft.remove_tags}
                maxTagCount="responsive"
                placeholder={S.lib_batch_remove_tag_placeholder}
                options={paperTagFilterOptions}
                optionFilterProp="label"
                onChange={(value) => setBatchDraft((cur) => ({ ...cur, remove_tags: normalizeTextList(value as unknown[]) }))}
              />
            </div>
          </section>

          <div className="kb-lib-meta-actions">
            <Button onClick={() => setBatchDrawerOpen(false)}>
              {S.lib_btn_cancel}
            </Button>
            <Button type="primary" loading={batchSaving} onClick={() => { void saveBatchEditor() }}>
              {S.lib_btn_apply_to_selected}
            </Button>
          </div>
        </div>
      </Drawer>
    </div>
  )
}
