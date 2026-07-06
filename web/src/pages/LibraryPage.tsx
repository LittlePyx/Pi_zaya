
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import {
  Upload,
  AutoComplete,
  Button,
  Drawer,
  message,
  Pagination,
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
  Modal,
} from 'antd'
import {
  UploadOutlined,
  ReloadOutlined,
  StopOutlined,
  FolderOpenOutlined,
  SearchOutlined,
  CheckOutlined,
  ClearOutlined,
  CopyOutlined,
  LockOutlined,
  ApiOutlined,
  ExclamationCircleOutlined,
} from '@ant-design/icons'
import type {
  ConversionQualitySummary,
  LibraryFileItem,
  LibraryFigureAssetRefreshResponse,
  LibraryFigureAssetScanResponse,
  LibraryQualityActionDelta,
  LibraryQualityActionHistoryItem,
  LibraryQualityActionSnapshot,
  LibraryConversionQualityBatchResponse,
  LibraryQualityFeatureHealth,
  LibraryQualityFeatureHealthItem,
  LibraryQualityFailureCase,
  LibraryQualityFullChain,
  LibraryQualityFullChainStage,
  LibraryQualityOverviewResponse,
  LibraryQualityPriorityAction,
  LibraryReaderLocateSourceRecommendation,
  LibraryQualityRepairAction,
  LibraryQualityRepairImpact,
  LibraryQualityRepairRun,
  LibraryResearchQaRerunResponse,
  RenameSuggestionItem,
} from '../api/library'
import { libraryApi } from '../api/library'
import { referencesApi, type ReferenceSyncStats, type ShelfMetadataBackfillJobState } from '../api/references'
import { useChatStore } from '../stores/chatStore'
import { settingsApi } from '../api/settings'
import { useLibraryStore } from '../stores/libraryStore'
import { useSettingsStore } from '../stores/settingsStore'
import { useNavigate } from 'react-router-dom'
import { useT } from '../i18n'
import {
  WorkbenchMetricStrip,
  WorkbenchPanel,
  WorkbenchStatusPill,
  type WorkbenchMetricItem,
  type WorkbenchTone,
} from '../components/library/WorkbenchPrimitives'
import { LibraryMetadataDrawer } from './library/LibraryMetadataDrawer'
import {
  LibraryQualityFigureAssetsPanel,
  LibraryQualityMetadataBackfillPanel,
} from './library/LibraryQualityMaintenancePanels'
import { LibraryQualityChainPanels } from './library/LibraryQualityChainPanels'
import { LibraryQualityIssuePanels } from './library/LibraryQualityIssuePanels'
import { LibraryQualityReportPanels } from './library/LibraryQualityReportPanels'
import { LibraryQualityHistoryPanel } from './library/LibraryQualityHistoryPanel'
import { LibraryQualityStatusPanels } from './library/LibraryQualityStatusPanels'
import { LibraryQualityOverviewPanels } from './library/LibraryQualityOverviewPanels'
import { LibraryQualityCenter } from './library/LibraryQualityCenter'
import { LibraryDirectorySettings } from './library/LibraryDirectorySettings'
import { LibraryFileRow } from './library/LibraryFileRow'
import { LibraryFileList } from './library/LibraryFileList'
import {
  LibraryCategoryCards,
  LibraryTagCards,
  type CategoryCardItem,
  type TagCardItem,
} from './library/LibraryTaxonomyViews'
import { LibraryTaxonomyToolbar } from './library/LibraryTaxonomyToolbar'
import { dispatchOpenSettings } from '../components/layout/settingsEvents'
import { qualityDiagnosticsVisible, qualityStatusVisible } from '../utils/qualityDiagnostics'
import {
  SCOPE_OPTIONS,
  RENAME_SCOPE_OPTIONS,
  buildQualityRepairHistoryRecord,
  conversionQualityIssueEntries,
  conversionQualityScore,
  conversionQualityStatus,
  conversionSourceReadiness,
  derivePageProgress,
  formatSeconds,
  hasConversionQualityIssue,
  isUploadDraftConverted,
  loadQualityRepairHistory,
  matchesKeyword,
  normalizeQualityRepairHistory,
  normalizeTextList,
  normalizeTextValue,
  numericStat,
  optionMatchesInput,
  qualityActionDeltaText,
  qualityBuildActionDelta,
  qualityDomainNumber,
  qualityDomainStatus,
  qualityFailureCaseMatchesStage,
  qualityOverviewStageSnapshot,
  qualityStatusText,
  qualityTopFailureText,
  qualityVerificationFromRerun,
  qualityVerificationText,
  saveQualityRepairHistory,
  saveResearchQaReplayFailureCase,
  stripKnownSourceExt,
  suggestionBasisTagColor,
  summarizeConversionQualityRepair,
  toTextOptions,
  type QualityRepairHistoryRecord,
  type UploadDraft,
  uniqueTextValues,
} from './library/libraryPageUtils'

const { Text } = Typography
const { Dragger } = Upload
const RENAME_PAGE_SIZE = 6
const UPLOAD_DRAFT_PAGE_SIZE = 8
const EMPTY_REF_SYNC_STATS: ReferenceSyncStats = {}
const INTERNAL_ROUTES_ENABLED = import.meta.env.VITE_ENABLE_INTERNAL_ROUTES === '1'
const QUALITY_DIAGNOSTICS_VISIBLE = qualityDiagnosticsVisible()
const QUALITY_STATUS_VISIBLE = qualityStatusVisible()

type FileTabKey = 'pending' | 'converted' | 'all'
type LibraryBrowseMode = 'list' | 'categories' | 'tags'
type DraftStatus = 'queued' | 'inspecting' | 'ready' | 'saving' | 'saved' | 'error'
type UploadDraftFilter = 'all' | 'todo' | 'error' | 'dup_error' | 'saved'
type UploadErrorReason = 'all' | 'duplicate' | 'path' | 'permission' | 'network' | 'other'

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

type QualityRepairBaseline = {
  quality: ConversionQualitySummary | null
  startedAt: number
}

type QualityRepairRunOptions = {
  autoReindexImmediate?: boolean
  autoReindexQueued?: boolean
  operationToken?: LibraryQualityOperationToken
}

type LibraryQualityOperationToken = {
  id: number
  key: string
  scope: string
}

type QualityIssueStat = {
  key: string
  label: string
  severity: string
  papers: number
  count: number
  repairStrategy?: string
}

type QualityReportRecommendationView = {
  name: string
  score: number
  issues: string[]
}

type QualityDomainView = {
  key: 'conversion' | 'research_qa' | 'citation_cards' | 'reader_locate'
  label: string
  available: boolean
  status: string
  statusLabel: string
  countText: string
  detailText: string
  failureText: string
}

type QualityFullChainActionResult = {
  status: 'success' | 'warning' | 'error' | 'info'
  summary: string
  detail?: string
  deltaText?: string
  verificationText?: string
  improved?: boolean | null
  updatedAt: number
}

type FilterFilesOptions = {
  ignoreCategoryFilter?: boolean
  ignoreTagFilter?: boolean
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

function deriveConvertStageLabel(msg0: string, S_?: Record<string, string>) {
  const msg = String(msg0 || '').trim().toLowerCase()
  if (!msg) return ''
  if (msg.includes('ingesting')) return S_ ? S_.lib_convert_ingesting : '正在更新知识库索引...'
  if (msg.includes('cancel')) return S_ ? S_.lib_convert_cancelling : '正在取消转换...'
  return ''
}

function readingStatusLabel(value: string, S_?: Record<string, string>) {
  if (value === 'unread') return S_ ? S_.lib_reading_status_unread : '未读'
  if (value === 'reading') return S_ ? S_.lib_reading_status_reading : '在读'
  if (value === 'done') return S_ ? S_.lib_reading_status_done : '已读'
  if (value === 'revisit') return S_ ? S_.lib_reading_status_revisit : '待回看'
  return ''
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

export default function LibraryPage() {
  const S = useT()
  const store = useLibraryStore()
  const createPaperGuideConversation = useChatStore((s) => s.createPaperGuideConversation)
  const nav = useNavigate()

  const settingsLoaded = useSettingsStore((s) => s.loaded)
  const settingsPdfDir = useSettingsStore((s) => s.pdfDir)
  const settingsMdDir = useSettingsStore((s) => s.mdDir)
  const hasTextApiKey = useSettingsStore((s) => s.hasTextApiKey)
  const llmReadiness = useSettingsStore((s) => s.llmReadiness)
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
  const [onlyQualityIssues, setOnlyQualityIssues] = useState(false)
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
  const [qualityRepairingNames, setQualityRepairingNames] = useState<Record<string, boolean>>({})
  const [qualityRepairResults, setQualityRepairResults] = useState<Record<string, string>>({})
  const [qualityRepairImpact, setQualityRepairImpact] = useState<LibraryQualityRepairImpact | null>(null)
  const [qualityBatchRunning, setQualityBatchRunning] = useState(false)
  const [qualityBatchResult, setQualityBatchResult] = useState<LibraryConversionQualityBatchResponse | null>(null)
  const [figureAssetScan, setFigureAssetScan] = useState<LibraryFigureAssetScanResponse | null>(null)
  const [figureAssetScanRunning, setFigureAssetScanRunning] = useState(false)
  const [figureAssetRefreshResult, setFigureAssetRefreshResult] = useState<LibraryFigureAssetRefreshResponse | null>(null)
  const [figureAssetRefreshRunning, setFigureAssetRefreshRunning] = useState(false)
  const [qualityRepairRun, setQualityRepairRun] = useState<LibraryQualityRepairRun | null>(null)
  const [qualityRepairAdvancing, setQualityRepairAdvancing] = useState(false)
  const [qualityRepairHistory, setQualityRepairHistory] = useState<Record<string, QualityRepairHistoryRecord>>(() => loadQualityRepairHistory())
  const [qualityHistoryFocusNames, setQualityHistoryFocusNames] = useState<string[]>([])
  const [qualityCenterOpen, setQualityCenterOpen] = useState(false)
  const [qualityArtifactOpening, setQualityArtifactOpening] = useState('')
  const [qualityCaseActionKey, setQualityCaseActionKey] = useState('')
  const [qualityFullChainActionKey, setQualityFullChainActionKey] = useState('')
  const [qualityFullChainResults, setQualityFullChainResults] = useState<Record<string, QualityFullChainActionResult>>({})
  const [shelfMetadataBackfillState, setShelfMetadataBackfillState] = useState<ShelfMetadataBackfillJobState | null>(null)
  const [shelfMetadataBackfillRefreshing, setShelfMetadataBackfillRefreshing] = useState(false)
  const [qualityCaseRerunResults, setQualityCaseRerunResults] = useState<Record<string, LibraryResearchQaRerunResponse>>({})
  const [qualityFailureFilter, setQualityFailureFilter] = useState('')
  const qualityRepairBaselinesRef = useRef<Record<string, QualityRepairBaseline>>({})
  const qualityOperationSeqRef = useRef(0)
  const activeQualityOperationRef = useRef<LibraryQualityOperationToken | null>(null)
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
  const [uploadDraftPage, setUploadDraftPage] = useState(1)
  const autoInspectingRef = useRef(false)

  const [renameScope, setRenameScope] = useState('30')
  const [renameLoading, setRenameLoading] = useState(false)
  const [renameApplying, setRenameApplying] = useState(false)
  const [renameItems, setRenameItems] = useState<RenameSuggestionItem[]>([])
  const [renameSelected, setRenameSelected] = useState<Record<string, boolean>>({})
  const [renameOverrides, setRenameOverrides] = useState<Record<string, string>>({})
  const [renameResultsOpen, setRenameResultsOpen] = useState(false)
  const [renamePage, setRenamePage] = useState(1)
  const [suggestionsRefreshing, setSuggestionsRefreshing] = useState(false)

  const uploadLocked = store.converting || Boolean(store.refSync?.running)
  const normalizedKeyword = fileKeyword.trim().toLowerCase()
  const textModelReady = !settingsLoaded
    || (hasTextApiKey && llmReadiness?.providers.text?.severity !== 'error')
  const openApiSettings = useCallback(() => {
    dispatchOpenSettings('text')
  }, [])
  const warnLlmFallback = useCallback((action: string) => {
    message.warning(S.lib_llm_unavailable_fallback.replace('{action}', action))
    openApiSettings()
  }, [S.lib_llm_unavailable_fallback, openApiSettings])
  const beginQualityOperation = useCallback((key: string): LibraryQualityOperationToken => {
    const token = {
      id: qualityOperationSeqRef.current + 1,
      key,
      scope,
    }
    qualityOperationSeqRef.current = token.id
    activeQualityOperationRef.current = token
    setQualityCaseActionKey('')
    setQualityFullChainActionKey('')
    setQualityBatchRunning(false)
    setShelfMetadataBackfillRefreshing(false)
    setQualityRepairAdvancing(false)
    return token
  }, [scope])
  const qualityOperationIsCurrent = useCallback((token?: LibraryQualityOperationToken | null): boolean => {
    if (!token) return true
    const active = activeQualityOperationRef.current
    return Boolean(active && active.id === token.id && active.key === token.key && active.scope === token.scope && scope === token.scope)
  }, [scope])
  const qualityOperationIsActive = useCallback((token?: LibraryQualityOperationToken | null): boolean => {
    if (!token) return true
    const active = activeQualityOperationRef.current
    return Boolean(active && active.id === token.id && active.key === token.key && active.scope === token.scope)
  }, [])
  const clearQualityOperation = useCallback((token?: LibraryQualityOperationToken | null) => {
    if (!token) {
      activeQualityOperationRef.current = null
      return
    }
    const active = activeQualityOperationRef.current
    if (active && active.id === token.id && active.key === token.key) {
      activeQualityOperationRef.current = null
    }
  }, [])

  const dirDirty = useMemo(
    () =>
      pdfDirDraft.trim() !== String(settingsPdfDir || '').trim()
      || mdDirDraft.trim() !== String(settingsMdDir || '').trim(),
    [pdfDirDraft, mdDirDraft, settingsPdfDir, settingsMdDir],
  )

  const pendingFiles = useMemo(() => store.files.filter((x) => x.category === 'pending'), [store.files])
  const convertedFiles = useMemo(() => store.files.filter((x) => x.category === 'converted'), [store.files])
  const qualityReviewCount = useMemo(() => store.files.filter((x) => hasConversionQualityIssue(x)).length, [store.files])
  const qualityReadyCount = useMemo(
    () => store.files.filter((x) => conversionQualityStatus(x.conversion_quality) === 'good').length,
    [store.files],
  )
  const qualitySourceReadinessStats = useMemo(() => {
    const stats = {
      ready: 0,
      autofixed: 0,
      blocked: 0,
      review: 0,
      processing: 0,
      pending: 0,
      indexStale: 0,
      unknown: 0,
    }
    for (const item of store.files) {
      const readiness = conversionSourceReadiness(item, S)
      if (readiness.qaReady) stats.ready += 1
      if (readiness.kind === 'autofixed') stats.autofixed += 1
      if (readiness.blocked) stats.blocked += 1
      if (readiness.kind === 'review') stats.review += 1
      if (readiness.kind === 'processing') stats.processing += 1
      if (readiness.kind === 'pending') stats.pending += 1
      if (readiness.kind === 'index_stale') stats.indexStale += 1
      if (readiness.kind === 'unknown') stats.unknown += 1
    }
    return stats
  }, [S, store.files])
  const backendQualityOverview = store.qualityOverview?.ok ? store.qualityOverview : null
  const latestBackendRepairRun = backendQualityOverview?.repair_runs?.[0] || null
  const refreshShelfMetadataBackfillState = useCallback(async (silent = true) => {
    if (!silent) setShelfMetadataBackfillRefreshing(true)
    try {
      const state = await referencesApi.shelfMetadataBackfillStatus()
      setShelfMetadataBackfillState(state)
      return state
    } catch (err) {
      if (!silent) message.error(err instanceof Error ? err.message : 'Metadata backfill status failed')
      return null
    } finally {
      if (!silent) setShelfMetadataBackfillRefreshing(false)
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    referencesApi.shelfMetadataBackfillStatus()
      .then((state) => {
        if (!cancelled) setShelfMetadataBackfillState(state)
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!shelfMetadataBackfillState?.running) return
    const timer = window.setInterval(() => {
      void refreshShelfMetadataBackfillState(true)
    }, 1600)
    return () => window.clearInterval(timer)
  }, [refreshShelfMetadataBackfillState, shelfMetadataBackfillState?.job_id, shelfMetadataBackfillState?.running])

  useEffect(() => {
    if (!latestBackendRepairRun) return
    setQualityRepairRun((cur) => {
      if (!cur || cur.run_id !== latestBackendRepairRun.run_id) return latestBackendRepairRun
      const curUpdated = Number(cur.updated_at || 0)
      const nextUpdated = Number(latestBackendRepairRun.updated_at || 0)
      if (nextUpdated > curUpdated) return latestBackendRepairRun
      const curState = `${normalizeTextValue(cur.status)}:${normalizeTextValue(cur.phase)}:${String(cur.reindexed)}`
      const nextState = `${normalizeTextValue(latestBackendRepairRun.status)}:${normalizeTextValue(latestBackendRepairRun.phase)}:${String(latestBackendRepairRun.reindexed)}`
      return curState === nextState ? cur : latestBackendRepairRun
    })
  }, [latestBackendRepairRun])
  const fallbackQualityReportStats = useMemo(() => {
    const assessed = store.files.filter((item) => item.conversion_quality)
    const convertedWithoutQuality = store.files.filter((item) => item.category === 'converted' && !item.conversion_quality).length
    const scores = assessed
      .map((item) => conversionQualityScore(item.conversion_quality))
      .filter((score) => Number.isFinite(score) && score > 0)
    const avgScore = scores.length > 0
      ? Math.round(scores.reduce((acc, score) => acc + score, 0) / scores.length)
      : 0
    return {
      assessed: assessed.length,
      converted: convertedFiles.length,
      review: qualityReviewCount,
      good: qualityReadyCount,
      unknown: convertedWithoutQuality,
      avgScore,
    }
  }, [convertedFiles.length, qualityReadyCount, qualityReviewCount, store.files])
  const qualityReportStats = useMemo(() => {
    const summary = backendQualityOverview?.summary
    if (!summary) return fallbackQualityReportStats
    return {
      assessed: Number(summary.assessed || 0),
      converted: Number(summary.converted || 0),
      review: Number(summary.review || 0),
      good: Number(summary.good || 0),
      unknown: Number(summary.unknown || 0),
      avgScore: Number(summary.avg_score || 0),
    }
  }, [backendQualityOverview, fallbackQualityReportStats])
  const fallbackQualityIssueStats = useMemo<QualityIssueStat[]>(() => {
    const stats = new Map<string, QualityIssueStat>()
    for (const item of store.files) {
      const seenInPaper = new Set<string>()
      for (const issue of item.conversion_quality?.issues || []) {
        const label = normalizeTextValue(issue.label || issue.code)
        const key = normalizeTextValue(issue.code || issue.label).toLowerCase()
        if (!key || !label) continue
        const existing = stats.get(key) || {
          key,
          label,
          severity: String(issue.severity || '').trim().toLowerCase(),
          papers: 0,
          count: 0,
        }
        existing.count += Math.max(1, Math.round(Number(issue.count || 0) || 1))
        if (!seenInPaper.has(key)) {
          existing.papers += 1
          seenInPaper.add(key)
        }
        if (String(issue.severity || '').trim().toLowerCase() === 'error') existing.severity = 'error'
        stats.set(key, existing)
      }
    }
    const severityWeight = (severity: string) => (severity === 'error' ? 2 : severity === 'warning' ? 1 : 0)
    return Array.from(stats.values())
      .sort((a, b) => severityWeight(b.severity) - severityWeight(a.severity)
        || b.papers - a.papers
        || b.count - a.count
        || a.label.localeCompare(b.label, 'en'))
      .slice(0, 5)
  }, [store.files])
  const qualityIssueStats = useMemo<QualityIssueStat[]>(() => {
    const issues = Array.isArray(backendQualityOverview?.top_issues) ? backendQualityOverview.top_issues : []
    if (!issues.length) return fallbackQualityIssueStats
    return issues.slice(0, 5).map((issue) => ({
      key: normalizeTextValue(issue.code || issue.label).toLowerCase(),
      label: normalizeTextValue(issue.label || issue.code),
      severity: normalizeTextValue(issue.severity || 'warning').toLowerCase(),
      papers: Number(issue.papers || 0),
      count: Number(issue.count || 0),
      repairStrategy: normalizeTextValue(issue.repair_strategy),
    })).filter((issue) => Boolean(issue.key && issue.label))
  }, [backendQualityOverview, fallbackQualityIssueStats])
  const qualityRepairHistoryList = useMemo(
    () => Object.values(qualityRepairHistory).sort((a, b) => b.updatedAt - a.updatedAt),
    [qualityRepairHistory],
  )
  const qualityRepairHistoryStats = useMemo(() => {
    const total = qualityRepairHistoryList.length
    const fixedCount = qualityRepairHistoryList.reduce((acc, item) => acc + item.fixedIssues.length, 0)
    const improved = qualityRepairHistoryList.filter((item) => item.afterScore > item.beforeScore).length
    const avgDelta = total > 0
      ? Math.round(qualityRepairHistoryList.reduce((acc, item) => acc + (item.afterScore - item.beforeScore), 0) / total)
      : 0
    return { total, fixedCount, improved, avgDelta }
  }, [qualityRepairHistoryList])
  const qualityHistoryFocusSet = useMemo(
    () => new Set(qualityHistoryFocusNames.map((name) => String(name || '').trim()).filter(Boolean)),
    [qualityHistoryFocusNames],
  )
  const qualityHistoryRemainingNames = useMemo(() => {
    const availableNames = new Set(store.files.map((item) => item.name))
    return qualityRepairHistoryList
      .filter((record) => record.remainingIssues.length > 0 && availableNames.has(record.name))
      .map((record) => record.name)
  }, [qualityRepairHistoryList, store.files])
  const localQualityRepairRecommendedItems = useMemo(() => (
    store.files
      .filter((item) => item.task_state === 'idle' && hasConversionQualityIssue(item))
      .sort((a, b) => {
        const aHistory = qualityRepairHistory[a.name]
        const bHistory = qualityRepairHistory[b.name]
        const aRemaining = aHistory?.remainingIssues.length || 0
        const bRemaining = bHistory?.remainingIssues.length || 0
        const aScore = conversionQualityScore(a.conversion_quality)
        const bScore = conversionQualityScore(b.conversion_quality)
        return bRemaining - aRemaining
          || aScore - bScore
          || String(a.name || '').localeCompare(String(b.name || ''), 'en')
      })
      .slice(0, 5)
  ), [qualityRepairHistory, store.files])
  const qualityReportRecommendations = useMemo<QualityReportRecommendationView[]>(() => {
    const overviewItems = Array.isArray(backendQualityOverview?.recommended) ? backendQualityOverview.recommended : []
    if (overviewItems.length > 0) {
      return overviewItems.slice(0, 5)
        .map((item) => ({
          name: normalizeTextValue(item.name),
          score: Math.max(0, Math.min(100, Math.round(Number(item.score || 0)))),
          issues: (Array.isArray(item.issues) ? item.issues : [])
            .map((issue) => normalizeTextValue(issue.label || issue.code))
            .filter(Boolean)
            .slice(0, 2),
        }))
        .filter((item) => Boolean(item.name))
    }
    return localQualityRepairRecommendedItems.slice(0, 5).map((item) => ({
      name: item.name,
      score: conversionQualityScore(item.conversion_quality),
      issues: conversionQualityIssueEntries(item.conversion_quality).map((issue) => issue.label).slice(0, 2),
    }))
  }, [backendQualityOverview, localQualityRepairRecommendedItems])
  const qualityDomainViews = useMemo<QualityDomainView[]>(() => {
    const domains = backendQualityOverview?.domains || {}
    const conversion = domains.conversion
    const researchQa = domains.research_qa
    const citationCards = domains.citation_cards
    const readerLocate = domains.reader_locate

    const conversionStatus = qualityDomainStatus(conversion, backendQualityOverview?.status || 'unknown')
    const conversionReview = conversion ? qualityDomainNumber(conversion, 'review') : qualityReportStats.review
    const conversionGood = conversion ? qualityDomainNumber(conversion, 'good') : qualityReportStats.good
    const conversionAvg = conversion ? qualityDomainNumber(conversion, 'avg_score') : qualityReportStats.avgScore
    const conversionUnknown = conversion ? qualityDomainNumber(conversion, 'unknown') : qualityReportStats.unknown

    const qaStatus = qualityDomainStatus(researchQa)
    const qaAvailable = researchQa?.available !== false && Boolean(researchQa)
    const qaTotal = qualityDomainNumber(researchQa, 'total')
    const qaPassed = qualityDomainNumber(researchQa, 'passed')
    const qaFailed = qualityDomainNumber(researchQa, 'failed')

    const cardStatus = qualityDomainStatus(citationCards)
    const cardsAvailable = citationCards?.available !== false && Boolean(citationCards)
    const trackedChecks = qualityDomainNumber(citationCards, 'tracked_checks')
    const failedChecks = qualityDomainNumber(citationCards, 'failed_checks')
    const shelfItems = qualityDomainNumber(citationCards, 'shelf_item_count')
    const shelfExportReady = qualityDomainNumber(citationCards, 'shelf_export_ready_count')
    const shelfSummaryExportReady = qualityDomainNumber(citationCards, 'shelf_summary_export_ready_count')
    const shelfExportDetail = shelfItems > 0 ? `; shelf export ${shelfExportReady}/${shelfItems}; summaries ${shelfSummaryExportReady}/${shelfItems}` : ''
    const readerStatus = qualityDomainStatus(readerLocate)
    const readerAvailable = readerLocate?.available !== false && Boolean(readerLocate)
    const readerTotal = qualityDomainNumber(readerLocate, 'total')
    const readerFailed = qualityDomainNumber(readerLocate, 'failed')
    const readerDegraded = qualityDomainNumber(readerLocate, 'degraded')
    const readerRepairable = qualityDomainNumber(readerLocate, 'repairable')

    return [
      {
        key: 'conversion',
        label: S.lib_quality_domain_conversion,
        available: true,
        status: conversionStatus,
        statusLabel: qualityStatusText(conversionStatus, S),
        countText: conversionReview > 0
          ? `${conversionReview} ${S.lib_quality_domain_failed}`
          : `${conversionGood} ${S.lib_quality_domain_passed}`,
        detailText: `Q${Math.round(conversionAvg)} · ${conversionUnknown} ${S.lib_quality_report_unknown}`,
        failureText: qualityTopFailureText(conversion),
      },
      {
        key: 'research_qa',
        label: S.lib_quality_domain_research_qa,
        available: qaAvailable,
        status: qaAvailable ? qaStatus : 'unknown',
        statusLabel: qaAvailable ? qualityStatusText(qaStatus, S) : S.lib_quality_domain_unavailable,
        countText: qaAvailable
          ? (qaFailed > 0 ? `${qaFailed} ${S.lib_quality_domain_failed}` : `${qaPassed}/${qaTotal} ${S.lib_quality_domain_passed}`)
          : S.lib_quality_domain_unavailable,
        detailText: qaAvailable ? S.lib_quality_domain_cases.replace('{n}', String(qaTotal)) : '',
        failureText: qualityTopFailureText(researchQa),
      },
      {
        key: 'citation_cards',
        label: S.lib_quality_domain_citation_cards,
        available: cardsAvailable,
        status: cardsAvailable ? cardStatus : 'unknown',
        statusLabel: cardsAvailable ? qualityStatusText(cardStatus, S) : S.lib_quality_domain_unavailable,
        countText: cardsAvailable
          ? (failedChecks > 0 ? `${failedChecks} ${S.lib_quality_domain_failed}` : `${trackedChecks} ${S.lib_quality_domain_passed}`)
          : S.lib_quality_domain_unavailable,
        detailText: cardsAvailable
          ? `${S.lib_quality_domain_checks.replace('{n}', String(trackedChecks))}${shelfExportDetail}`
          : '',
        failureText: qualityTopFailureText(citationCards),
      },
      {
        key: 'reader_locate',
        label: 'Reader locate',
        available: readerAvailable,
        status: readerAvailable ? readerStatus : 'unknown',
        statusLabel: readerAvailable ? qualityStatusText(readerStatus, S) : S.lib_quality_domain_unavailable,
        countText: readerAvailable
          ? (readerFailed > 0 || readerDegraded > 0 ? `${readerFailed + readerDegraded} need repair` : `${readerTotal} verified`)
          : S.lib_quality_domain_unavailable,
        detailText: readerAvailable ? `${readerRepairable} repairable source signals` : '',
        failureText: qualityTopFailureText(readerLocate),
      },
    ]
  }, [backendQualityOverview, qualityReportStats, S])
  const qualityPriorityActions = useMemo<LibraryQualityPriorityAction[]>(
    () => (Array.isArray(backendQualityOverview?.priority_actions) ? backendQualityOverview.priority_actions : [])
      .filter((item) => item && normalizeTextValue(item.domain))
      .slice(0, 4),
    [backendQualityOverview],
  )
  const actionableQualityPriorityActions = useMemo(
    () => qualityPriorityActions.filter((item) => (
      Number(item.count || 0) > 0
      || normalizeTextValue(item.severity).toLowerCase() === 'error'
    )),
    [qualityPriorityActions],
  )
  const qualityFullChain = useMemo<LibraryQualityFullChain | null>(() => {
    const fullChain = backendQualityOverview?.full_chain
    if (!fullChain || fullChain.available === false) return null
    return fullChain
  }, [backendQualityOverview])
  const qualityFullChainStages = useMemo(
    () => (Array.isArray(qualityFullChain?.stages) ? qualityFullChain.stages : [])
      .filter((stage) => stage && normalizeTextValue(stage.key))
      .slice(0, 6),
    [qualityFullChain],
  )
  const qualityFullChainRootCauses = useMemo(
    () => (Array.isArray(qualityFullChain?.root_causes) ? qualityFullChain.root_causes : [])
      .filter((cause) => cause && normalizeTextValue(cause.code || cause.label))
      .slice(0, 5),
    [qualityFullChain],
  )
  const qualityFullChainActionHistory = useMemo<LibraryQualityActionHistoryItem[]>(
    () => (Array.isArray(qualityFullChain?.action_history) ? qualityFullChain.action_history : [])
      .filter((item) => item && normalizeTextValue(item.stage_key) && normalizeTextValue(item.summary))
      .slice(0, 8),
    [qualityFullChain],
  )
  const qualityFullChainPersistedResults = useMemo<Record<string, QualityFullChainActionResult>>(() => {
    const out: Record<string, QualityFullChainActionResult> = {}
    for (const item of qualityFullChainActionHistory) {
      const key = normalizeTextValue(item.stage_key).toLowerCase()
      if (!key || out[key]) continue
      const status = normalizeTextValue(item.status).toLowerCase()
      out[key] = {
        status: status === 'success' || status === 'warning' || status === 'error' ? status : 'info',
        summary: normalizeTextValue(item.summary),
        detail: normalizeTextValue(item.detail),
        deltaText: qualityActionDeltaText(item),
        verificationText: qualityVerificationText(item.verification),
        improved: typeof item.improved === 'boolean' ? item.improved : item.delta?.improved,
        updatedAt: Number(item.created_at || 0) * 1000,
      }
    }
    return out
  }, [qualityFullChainActionHistory])
  const qualityReaderLocateRecommendedSources = useMemo<LibraryReaderLocateSourceRecommendation[]>(() => {
    const sources = backendQualityOverview?.reader_locate?.recommended_sources
    if (!Array.isArray(sources)) return []
    return sources
      .filter((item) => item && (normalizeTextValue(item.source_path) || normalizeTextValue(item.source_name)))
      .slice(0, 12)
  }, [backendQualityOverview])
  const qualityFeatureHealth = useMemo<LibraryQualityFeatureHealth | null>(() => {
    const featureHealth = backendQualityOverview?.feature_health
    if (!featureHealth || featureHealth.available === false) return null
    return featureHealth
  }, [backendQualityOverview])
  const qualityFeatureHealthItems = useMemo<LibraryQualityFeatureHealthItem[]>(
    () => (Array.isArray(qualityFeatureHealth?.items) ? qualityFeatureHealth.items : [])
      .filter((item) => item && normalizeTextValue(item.key))
      .slice(0, 8),
    [qualityFeatureHealth],
  )
  const shelfMetadataBackfillScan = useMemo(() => {
    const state = shelfMetadataBackfillState
    return state?.after_scan || state?.result?.after_scan || state?.scan || state?.result?.scan || null
  }, [shelfMetadataBackfillState])
  const shelfMetadataBackfillResult = shelfMetadataBackfillState?.result || null
  const shelfMetadataBackfillProgress = Math.max(0, Math.min(100, Math.round(Number(shelfMetadataBackfillState?.progress?.percent || 0))))
  const shelfMetadataBackfillPhase = normalizeTextValue(shelfMetadataBackfillState?.phase || shelfMetadataBackfillState?.status || 'idle').replace(/_/g, ' ')
  const shelfMetadataBackfillRunning = Boolean(shelfMetadataBackfillState?.running)
  const shelfMetadataBackfillTone = shelfMetadataBackfillRunning
    ? 'warning'
    : normalizeTextValue(shelfMetadataBackfillState?.status).toLowerCase() === 'error'
      ? 'error'
      : shelfMetadataBackfillScan
        ? (Number(shelfMetadataBackfillScan.needs_repair || 0) > 0 ? 'warning' : 'good')
        : 'unknown'
  const qualityRerunSummary = backendQualityOverview?.rerun_summary
  const qualityFailureCases = useMemo<LibraryQualityFailureCase[]>(
    () => (Array.isArray(backendQualityOverview?.failure_cases) ? backendQualityOverview.failure_cases : [])
      .filter((item) => item && normalizeTextValue(item.id))
      .slice(0, 12),
    [backendQualityOverview],
  )
  const qualityFailureFilters = useMemo(() => {
    const stats = new Map<string, number>()
    for (const item of qualityFailureCases) {
      for (const failure of item.failures || []) {
        const name = normalizeTextValue(failure.name)
        if (!name) continue
        stats.set(name, (stats.get(name) || 0) + 1)
      }
    }
    return Array.from(stats.entries())
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0], 'en'))
      .slice(0, 6)
      .map(([name, count]) => ({ name, count }))
  }, [qualityFailureCases])
  const visibleQualityFailureCases = useMemo(() => {
    const filter = normalizeTextValue(qualityFailureFilter)
    if (!filter) return qualityFailureCases
    return qualityFailureCases.filter((item) => (item.failures || []).some((failure) => normalizeTextValue(failure.name) === filter))
  }, [qualityFailureCases, qualityFailureFilter])
  const qualityRepairRecommendedNames = useMemo(
    () => qualityReportRecommendations.map((item) => item.name).filter(Boolean),
    [qualityReportRecommendations],
  )
  const qualityCenterBusy = qualityBatchRunning
    || qualityRepairAdvancing
    || shelfMetadataBackfillRunning
    || Object.values(qualityRepairingNames).some(Boolean)
  const qualityCenterDomainProblems = qualityDomainViews.filter((domain) => (
    domain.available
    && !['good', 'unknown'].includes(normalizeTextValue(domain.status).toLowerCase())
  ))
  const qualityCenterFailureCount = qualityFailureCases.length
  const qualityCenterMetadataRemaining = Number(shelfMetadataBackfillScan?.needs_repair || 0)
  const qualityCenterProblemCount = qualityReportStats.review
    + qualitySourceReadinessStats.blocked
    + qualityCenterFailureCount
    + qualityCenterDomainProblems.length
  const qualityCenterTone = qualityCenterBusy
    ? 'processing'
    : (qualitySourceReadinessStats.blocked > 0
      || qualityReportStats.review > 0
      || qualityCenterFailureCount > 0
      || qualityCenterDomainProblems.some((domain) => normalizeTextValue(domain.status).toLowerCase() === 'error'))
        ? 'error'
        : (qualityReportStats.unknown > 0 || qualityCenterMetadataRemaining > 0 || actionableQualityPriorityActions.length > 0)
          ? 'warning'
          : 'good'
  const qualityCenterStatusLabel = qualityCenterTone === 'processing'
    ? S.lib_quality_center_status_processing
    : qualityCenterTone === 'good'
      ? S.lib_quality_center_status_ready
      : qualityCenterTone === 'error'
        ? S.lib_quality_center_status_repair
        : S.lib_quality_center_status_attention
  const qualityCenterSummary = qualityCenterTone === 'good'
    ? S.lib_quality_center_summary_ready
      .replace('{ready}', String(qualityReportStats.good))
      .replace('{total}', String(qualityReportStats.assessed || qualityReportStats.converted))
    : qualityCenterTone === 'processing'
      ? S.lib_quality_center_summary_running
      : S.lib_quality_center_summary_review
        .replace('{review}', String(qualityReportStats.review))
        .replace('{blocked}', String(qualitySourceReadinessStats.blocked))
        .replace('{cases}', String(qualityCenterFailureCount))
        .replace('{domains}', String(qualityCenterDomainProblems.length))
  const qualityCenterNextAction = qualityCenterTone === 'good'
    ? S.lib_quality_center_action_none
    : qualityCenterTone === 'processing'
      ? S.lib_quality_center_action_monitor
      : qualityRepairRecommendedNames.length > 0
        ? S.lib_quality_center_action_repair.replace('{n}', String(qualityRepairRecommendedNames.length))
        : qualityCenterMetadataRemaining > 0
          ? S.lib_quality_center_action_metadata.replace('{n}', String(qualityCenterMetadataRemaining))
          : qualityReportStats.review > 0
            ? S.lib_quality_center_action_review
            : S.lib_quality_center_action_open
  const qualityCenterSignals = [
    { key: 'usable', label: S.lib_quality_center_signal_usable, value: `${qualityReportStats.good}/${qualityReportStats.assessed || qualityReportStats.converted || 0}` },
    { key: 'risk', label: S.lib_quality_center_signal_attention, value: String(qualityCenterProblemCount) },
    { key: 'locate', label: S.lib_quality_center_signal_locate, value: String(qualitySourceReadinessStats.blocked) },
    { key: 'metadata', label: S.lib_quality_center_signal_metadata, value: String(qualityCenterMetadataRemaining) },
  ]
  const renameOnlyDiff = true
  const renameVisible = useMemo(() => (renameOnlyDiff ? renameItems.filter((x) => x.diff) : renameItems), [renameOnlyDiff, renameItems])
  const selectedUploadCount = useMemo(() => uploadDrafts.filter((x) => x.selected).length, [uploadDrafts])
  const selectedRenameCount = useMemo(() => renameItems.filter((x) => renameSelected[x.name]).length, [renameItems, renameSelected])
  const failedUploadDrafts = useMemo(() => uploadDrafts.filter((x) => x.status === 'error'), [uploadDrafts])
  const duplicateFailedDrafts = useMemo(
    () => failedUploadDrafts.filter((x) => isDuplicateFailure(x.note)),
    [failedUploadDrafts],
  )
  const retryableFailedUploadDrafts = useMemo(
    () => failedUploadDrafts.filter((x) => x.failureStage !== 'duplicate' && !isDuplicateFailure(x.note)),
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
  }, [failedUploadDrafts, S])
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
    [uploadDrafts, S],
  )
  const renamePageCount = Math.max(1, Math.ceil(renameVisible.length / RENAME_PAGE_SIZE))
  const uploadDraftPageCount = Math.max(1, Math.ceil(filteredUploadDrafts.length / UPLOAD_DRAFT_PAGE_SIZE))
  const pagedRenameVisible = useMemo(
    () => renameVisible.slice((renamePage - 1) * RENAME_PAGE_SIZE, renamePage * RENAME_PAGE_SIZE),
    [renamePage, renameVisible],
  )
  const pagedUploadDrafts = useMemo(
    () => filteredUploadDrafts.slice((uploadDraftPage - 1) * UPLOAD_DRAFT_PAGE_SIZE, uploadDraftPage * UPLOAD_DRAFT_PAGE_SIZE),
    [filteredUploadDrafts, uploadDraftPage],
  )
  useEffect(() => {
    if (renamePage > renamePageCount) setRenamePage(renamePageCount)
  }, [renamePage, renamePageCount])
  useEffect(() => {
    if (uploadDraftPage > uploadDraftPageCount) setUploadDraftPage(uploadDraftPageCount)
  }, [uploadDraftPage, uploadDraftPageCount])
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
  }, [uploadErrorReason, S])
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
    [store.progress, S],
  )
  const refSyncPercent = useMemo(
    () => (store.refSync && store.refSync.docsTotal > 0
      ? Math.round((store.refSync.docsDone / Math.max(1, store.refSync.docsTotal)) * 100)
      : 0),
    [store.refSync],
  )
  const refSyncStats = useMemo(() => store.refSync?.stats || EMPTY_REF_SYNC_STATS, [store.refSync?.stats])
  const refSyncMetricItems = useMemo<WorkbenchMetricItem[]>(() => {
    const refsTotal = numericStat(refSyncStats, 'refs_total')
    const docsTotal = store.refSync?.docsTotal ?? numericStat(refSyncStats, 'docs_total')
    const docsDone = store.refSync?.docsDone ?? numericStat(refSyncStats, 'docs_indexed')
    const statusReady = numericStat(refSyncStats, 'refs_metadata_status_complete')
      + numericStat(refSyncStats, 'refs_metadata_status_crossref_enriched')
      + numericStat(refSyncStats, 'refs_metadata_status_bibliographic_ready')
      + numericStat(refSyncStats, 'refs_metadata_status_non_article_source_ok')
      + numericStat(refSyncStats, 'refs_metadata_status_no_doi_expected')
    const metadataReady = Math.max(numericStat(refSyncStats, 'refs_metadata_user_ready'), statusReady, numericStat(refSyncStats, 'refs_metadata_ready'))
    const network = numericStat(refSyncStats, 'crossref_network_attempts')
    const elapsed = numericStat(refSyncStats, 'elapsed_s')
    return [
      { key: 'docs', label: S.lib_refsync_metric_docs, value: `${docsDone}/${docsTotal || 0}`, tone: 'good' },
      { key: 'refs', label: S.lib_refsync_metric_refs, value: refsTotal, tone: 'good' },
      { key: 'ready', label: S.lib_refsync_metric_ready, value: `${metadataReady}/${refsTotal}`, tone: 'info' },
      { key: 'network_elapsed', label: S.lib_refsync_metric_network_elapsed, value: `${network} / ${formatSeconds(elapsed)}`, tone: 'info' },
    ]
  }, [S, refSyncStats, store.refSync?.docsDone, store.refSync?.docsTotal])
  const refSyncQueueItems = useMemo<WorkbenchMetricItem[]>(() => {
    const onlineReady = Math.max(numericStat(refSyncStats, 'refs_crossref_ok'), numericStat(refSyncStats, 'refs_metadata_status_crossref_enriched'))
    const webSource = numericStat(refSyncStats, 'refs_web_source_ok')
    const nonArticle = numericStat(refSyncStats, 'refs_action_non_article_ok')
      || (numericStat(refSyncStats, 'refs_metadata_status_non_article_source_ok')
        + numericStat(refSyncStats, 'refs_metadata_status_no_doi_expected'))
    const manualRepair = numericStat(refSyncStats, 'refs_action_source_repair')
      || (numericStat(refSyncStats, 'refs_missing_reason_truncated_reference')
        + numericStat(refSyncStats, 'refs_missing_reason_low_confidence_match'))
    return [
      { key: 'online_ready', label: S.lib_refsync_metric_online_ready, value: onlineReady, tone: onlineReady > 0 ? 'good' : 'info' },
      { key: 'web_source', label: S.lib_refsync_metric_web_source, value: webSource, tone: webSource > 0 ? 'good' : 'info' },
      { key: 'non_article', label: S.lib_refsync_metric_non_article, value: nonArticle, tone: nonArticle > 0 ? 'good' : 'info' },
      { key: 'manual_repair', label: S.lib_refsync_metric_manual_repair, value: manualRepair, tone: manualRepair > 0 ? 'warn' : 'good' },
    ]
  }, [S, refSyncStats])
  const refSyncDisplayMessage = useMemo(() => {
    if (!store.refSync) return S.lib_refsync_waiting
    if (store.refSync.current) return `${store.refSync.stage || S.lib_refsync_running} | ${store.refSync.current}`
    const refsTotal = numericStat(refSyncStats, 'refs_total')
    if (store.refSync.status === 'done' && refsTotal > 0) {
      const statusReady = numericStat(refSyncStats, 'refs_metadata_status_complete')
        + numericStat(refSyncStats, 'refs_metadata_status_crossref_enriched')
        + numericStat(refSyncStats, 'refs_metadata_status_bibliographic_ready')
        + numericStat(refSyncStats, 'refs_metadata_status_non_article_source_ok')
        + numericStat(refSyncStats, 'refs_metadata_status_no_doi_expected')
      const metadataReady = Math.max(numericStat(refSyncStats, 'refs_metadata_user_ready'), statusReady, numericStat(refSyncStats, 'refs_metadata_ready'))
      const onlineReady = Math.max(numericStat(refSyncStats, 'refs_crossref_ok'), numericStat(refSyncStats, 'refs_metadata_status_crossref_enriched'))
      const webSource = numericStat(refSyncStats, 'refs_web_source_ok')
      const nonArticle = numericStat(refSyncStats, 'refs_action_non_article_ok')
        || (numericStat(refSyncStats, 'refs_metadata_status_non_article_source_ok')
          + numericStat(refSyncStats, 'refs_metadata_status_no_doi_expected'))
      const manualRepair = numericStat(refSyncStats, 'refs_action_source_repair')
        || (numericStat(refSyncStats, 'refs_missing_reason_truncated_reference')
          + numericStat(refSyncStats, 'refs_missing_reason_low_confidence_match'))
      return S.lib_refsync_done_summary
        .replace('{ready}', String(metadataReady))
        .replace('{refsTotal}', String(refsTotal))
        .replace('{onlineReady}', String(onlineReady))
        .replace('{webSource}', String(webSource))
        .replace('{nonArticle}', String(nonArticle))
        .replace('{manualRepair}', String(manualRepair))
    }
    return store.refSync.message || S.lib_refsync_waiting
  }, [S, refSyncStats, store.refSync])
  const showRefSyncCard = Boolean(store.refSync && (
    store.refSync.running
    || store.refSync.status === 'done'
    || store.refSync.status === 'error'
    || Boolean(store.refSync.error)
  ))
  const refSyncStatusTone: WorkbenchTone = store.refSync?.running
    ? 'processing'
    : store.refSync?.status === 'error'
      ? 'danger'
      : store.refSync?.status === 'done'
        ? 'good'
        : 'neutral'
  const refSyncStatusLabel = store.refSync?.running
    ? S.lib_refsync_running
    : (store.refSync?.status === 'idle' ? S.lib_refsync_idle : String(store.refSync?.status || ''))
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
    setOnlyQualityIssues(false)
    setQualityHistoryFocusNames([])
  }

  const hasActiveTaxonomyFilters = Boolean(
    normalizedKeyword
    || paperCategoryFilter
    || paperTagFilter
    || readingStatusFilter
    || onlyUnread
    || onlyUnclassified
    || onlySuggested
    || onlyQualityIssues
    || qualityHistoryFocusNames.length > 0
  )
  const activeTaxonomyFilterCount = [
    normalizedKeyword,
    paperCategoryFilter,
    paperTagFilter,
    readingStatusFilter,
    onlyUnread ? 'onlyUnread' : '',
    onlyUnclassified ? 'onlyUnclassified' : '',
    onlySuggested ? 'onlySuggested' : '',
    onlyQualityIssues ? 'onlyQualityIssues' : '',
    qualityHistoryFocusNames.length > 0 ? 'qualityHistoryFocus' : '',
  ].filter(Boolean).length

  const filterFiles = useCallback(
    (items: LibraryFileItem[], options: FilterFilesOptions = {}) =>
      items.filter((item) => {
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
        if (!options.ignoreCategoryFilter && paperCategoryFilter && String(item.paper_category || '') !== paperCategoryFilter) return false
        if (!options.ignoreTagFilter && paperTagFilter && !(item.user_tags || []).some((tag) => String(tag || '').toLowerCase() === paperTagFilter.toLowerCase())) return false
        if (readingStatusFilter && String(item.reading_status || '') !== readingStatusFilter) return false
        if (onlyUnread && String(item.reading_status || '') !== 'unread') return false
        if (onlyUnclassified && String(item.paper_category || '').trim()) return false
        if (onlySuggested && !item.has_suggestions) return false
        if (onlyQualityIssues && !hasConversionQualityIssue(item)) return false
        return true
      }),
    [normalizedKeyword, onlyQualityIssues, onlySuggested, onlyUnclassified, onlyUnread, paperCategoryFilter, paperTagFilter, qualityHistoryFocusSet, readingStatusFilter],
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
  }, [visibleAllWithoutCategory, S])

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
  }, [visibleAllWithoutTag, S])

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
  const batchDraftCategory = normalizeTextValue(batchDraft.paper_category)
  const batchDraftAddTags = normalizeTextList(batchDraft.add_tags)
  const batchDraftRemoveTags = normalizeTextList(batchDraft.remove_tags)
  const batchDraftWillClearCategory = batchDraft.apply_paper_category && !batchDraftCategory
  const batchDraftWillClearStatus = batchDraft.apply_reading_status && !batchDraft.reading_status
  const batchDraftReadingLabel = batchDraft.apply_reading_status
    ? readingStatusLabel(batchDraft.reading_status, S)
    : ''
  const selectedQualityReviewNames = useMemo(
    () => store.files
      .filter((item) => Boolean(selectedLibraryNames[item.name]) && hasConversionQualityIssue(item) && item.task_state === 'idle')
      .map((item) => item.name),
    [store.files, selectedLibraryNames],
  )
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

  useEffect(() => {
    const pending = qualityRepairBaselinesRef.current
    const pendingNames = Object.keys(pending)
    if (!pendingNames.length) return
    const nextPending = { ...pending }
    const nextResults: Record<string, string> = {}
    const nextHistory: Record<string, QualityRepairHistoryRecord> = {}
    for (const item of store.files) {
      const baseline = pending[item.name]
      if (!baseline) continue
      if (item.task_state !== 'idle' || !item.conversion_quality) continue
      nextResults[item.name] = summarizeConversionQualityRepair(baseline.quality, item.conversion_quality, S)
      nextHistory[item.name] = buildQualityRepairHistoryRecord(item.name, baseline.quality, item.conversion_quality)
      delete nextPending[item.name]
    }
    if (Object.keys(nextResults).length <= 0) return
    qualityRepairBaselinesRef.current = nextPending
    setQualityRepairResults((cur) => ({ ...cur, ...nextResults }))
    setQualityRepairHistory((cur) => {
      const merged = normalizeQualityRepairHistory({ ...cur, ...nextHistory })
      saveQualityRepairHistory(merged)
      return merged
    })
  }, [store.files, S])

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
  }, [mdDirDraft, pdfDirDraft, scope, store, updateSettings, S])

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
          failureStage: '',
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

  const inspectDraft = useCallback(async (key: string, opts?: { useLlm?: boolean }): Promise<{
    ok: boolean
    duplicate: boolean
    suggestedStem: string
  }> => {
    const ready = await ensureDirsReady()
    if (!ready) return { ok: false, duplicate: false, suggestedStem: '' }
    const target = uploadDrafts.find((x) => x.key === key)
    if (!target) return { ok: false, duplicate: false, suggestedStem: '' }
    const effectiveUseLlm = Boolean(opts?.useLlm ?? uploadUseLlm)
    setUploadDrafts((cur) => cur.map((x) => (x.key === key ? { ...x, status: 'inspecting', failureStage: '', note: '' } : x)))
    try {
      const res = await libraryApi.inspectUpload(target.file, effectiveUseLlm)
      const suggestedStem = String(res.suggested_stem || target.stem || '')
      setUploadDrafts((cur) => cur.map((x) => {
        if (x.key !== key) return x
        return {
          ...x,
          stem: suggestedStem || x.stem,
          displayName: res.display_full_name || x.displayName,
          suggestionBasisLabel: String(res.meta?.basis_label || ''),
          suggestionBasisDetail: String(res.meta?.basis_detail || ''),
          suggestionMatchMethod: String(res.meta?.match_method || ''),
          suggestionYearSource: String(res.meta?.year_source || ''),
          status: res.duplicate ? 'error' : 'ready',
          failureStage: res.duplicate ? 'duplicate' : '',
          note: res.duplicate ? `${S.lib_upload_dup_prefix}${String(res.existing || '')}` : S.lib_upload_scan_done,
        }
      }))
      return { ok: !res.duplicate, duplicate: Boolean(res.duplicate), suggestedStem }
    } catch (err) {
      setUploadDrafts((cur) => cur.map((x) => (
        x.key === key
          ? { ...x, status: 'error', failureStage: 'inspect', note: err instanceof Error ? err.message : S.lib_upload_scan_fail }
          : x
      )))
      return { ok: false, duplicate: false, suggestedStem: '' }
    }
  }, [ensureDirsReady, uploadDrafts, uploadUseLlm, S])

  const inspectSelectedDrafts = async () => {
    const selected = uploadDrafts.filter((x) => x.selected && x.status !== 'inspecting')
    if (!selected.length) {
      message.info(S.lib_msg_select_scan)
      return
    }
    const effectiveUseLlm = uploadUseLlm && textModelReady
    if (uploadUseLlm && !textModelReady) {
      warnLlmFallback(S.lib_upload_use_llm)
    }
    setUploadInspecting(true)
    try {
      for (const x of selected) {
        await inspectDraft(x.key, { useLlm: effectiveUseLlm })
      }
      message.success(S.lib_msg_scanned_count.replace('{n}', String(selected.length)))
    } finally {
      setUploadInspecting(false)
    }
  }

  const inspectSingleDraft = useCallback((key: string) => {
    const effectiveUseLlm = uploadUseLlm && textModelReady
    if (uploadUseLlm && !textModelReady) {
      warnLlmFallback(S.lib_upload_use_llm)
    }
    void inspectDraft(key, { useLlm: effectiveUseLlm })
  }, [S.lib_upload_use_llm, inspectDraft, textModelReady, uploadUseLlm, warnLlmFallback])

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
        const effectiveUseLlm = uploadUseLlm && textModelReady
        if (uploadUseLlm && !textModelReady) {
          warnLlmFallback(S.lib_upload_use_llm)
        }
        for (const key of queuedKeys) {
          // Auto-fill suggested names for newly added upload drafts.
          await inspectDraft(key, { useLlm: effectiveUseLlm })
        }
      } finally {
        autoInspectingRef.current = false
        setUploadInspecting(false)
      }
    })()
  }, [S.lib_upload_use_llm, dirDirty, inspectDraft, textModelReady, uploadDrafts, uploadInspecting, uploadLocked, uploadUseLlm, warnLlmFallback])

  useEffect(() => {
    setUploadDrafts((cur) => {
      const next = cur.filter((draft) => !isUploadDraftConverted(draft, store.files))
      return next.length === cur.length ? cur : next
    })
  }, [store.files])

  const saveDraft = async (key: string, convertNow: boolean, opts?: { syncUi?: boolean; baseName?: string }) => {
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
          failureStage: '',
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
        baseName: opts?.baseName ?? target.stem,
        convertNow,
        speedMode: CONVERT_MODE,
        allowDuplicate: false,
      })
      const savedName = String(res.name || target.file.name || '')
      const enqueued = Boolean(convertNow && res.enqueued)
      setUploadDrafts((cur) => cur.map((x) => {
        if (x.key !== key) return x
        if (res.duplicate) {
          return {
            ...x,
            status: 'error',
            failureStage: 'duplicate',
            note: `${S.lib_upload_dup_prefix}${String(res.existing || '')}`,
          }
        }
        return {
          ...x,
          status: 'saved',
          failureStage: '',
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
          ? { ...x, status: 'error', failureStage: 'save', note: err instanceof Error ? err.message : S.lib_upload_save_fail }
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
      const effectiveUseLlm = textModelReady
      if (!textModelReady) {
        warnLlmFallback(S.lib_btn_rename_check)
      }
      const res = await libraryApi.listRenameSuggestions(renameScope, effectiveUseLlm)
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
    const retryable = failed.filter((x) => x.failureStage !== 'duplicate' && !isDuplicateFailure(x.note))
    if (!retryable.length) {
      message.info(S.lib_msg_no_retryable)
      return
    }
    setUploadSaving(true)
    setUploadInspecting(true)
    try {
      let anyEnqueued = false
      const effectiveUseLlm = uploadUseLlm && textModelReady
      if (uploadUseLlm && !textModelReady) {
        warnLlmFallback(S.lib_upload_use_llm)
      }
      for (const x of retryable) {
        if (x.failureStage === 'inspect') {
          const inspectResult = await inspectDraft(x.key, { useLlm: effectiveUseLlm })
          if (!inspectResult.ok || !convertNow) continue
          const result = await saveDraft(x.key, true, {
            syncUi: false,
            baseName: inspectResult.suggestedStem || x.stem,
          })
          anyEnqueued = anyEnqueued || Boolean(result.enqueued)
          continue
        }
        const result = await saveDraft(x.key, convertNow, { syncUi: false })
        anyEnqueued = anyEnqueued || Boolean(result.enqueued)
      }
      await store.loadFiles(scope)
      if (anyEnqueued) store.startProgressStream()
      message.success(S.lib_msg_retried_count.replace('{n}', String(retryable.length)))
    } finally {
      setUploadInspecting(false)
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
      const effectiveUseLlm = textModelReady
      if (!textModelReady) {
        warnLlmFallback(S.lib_btn_apply_rename)
      }
      const res = await libraryApi.applyRenameSuggestions(names, overrides, { useLlm: effectiveUseLlm, alsoMd: true })
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

  const recordQualityFullChainResult = (
    stageKey: string,
    result: Omit<QualityFullChainActionResult, 'updatedAt'>,
    meta: {
      stageLabel?: string
      action?: string
      targetIds?: string[]
      metrics?: Record<string, string | number | boolean | null | undefined>
      before?: LibraryQualityActionSnapshot
      after?: LibraryQualityActionSnapshot
      verification?: Record<string, unknown>
    } = {},
  ) => {
    const key = normalizeTextValue(stageKey).toLowerCase()
    if (!key) return
    const nowMs = Date.now()
    const hasSnapshots = Boolean(meta.before || meta.after)
    const verificationOk = meta.verification?.quality_ok === true
    const rawDelta: LibraryQualityActionDelta = hasSnapshots ? qualityBuildActionDelta(meta.before, meta.after) : {}
    const delta: LibraryQualityActionDelta = hasSnapshots && rawDelta.improved === false && verificationOk
      ? {
        ...rawDelta,
        improved: true,
        worsened: false,
        summary: 'Improved: QA rerun passed; overview unchanged',
      }
      : rawDelta
    const improved = typeof delta.improved === 'boolean' ? delta.improved : null
    const verificationText = qualityVerificationText(meta.verification)
    const resultStatus = result.status === 'success' && improved === false ? 'warning' : result.status
    setQualityFullChainResults((cur) => ({
      ...cur,
      [key]: {
        ...result,
        status: resultStatus,
        deltaText: hasSnapshots ? delta.summary : '',
        verificationText,
        improved,
        updatedAt: nowMs,
      },
    }))
    void libraryApi.recordQualityAction({
      stage_key: key,
      stage_label: normalizeTextValue(meta.stageLabel || key),
      action: normalizeTextValue(meta.action),
      status: resultStatus,
      summary: result.summary,
      detail: result.detail,
      target_ids: meta.targetIds || [],
      metrics: meta.metrics || {},
      before: meta.before,
      after: meta.after,
      delta: hasSnapshots ? delta : {},
      improved,
      verification: meta.verification || {},
      created_at: Math.floor(nowMs / 1000),
    }).catch(() => {
      // Persisting the audit trail should never block the repair flow.
    })
  }

  const repairQualityByNames = async (names: string[], opts: QualityRepairRunOptions = {}) => {
    const targets = Array.from(new Set(names.map((name) => String(name || '').trim()).filter(Boolean)))
    if (!targets.length) {
      message.info(S.lib_msg_quality_repair_none)
      return { ok: true, targetCount: 0, queued: 0, repaired: 0, needsReindex: false, reindexed: false, impact: null as LibraryQualityRepairImpact | null }
    }
    const operationToken = opts.operationToken || beginQualityOperation(`quality-repair:${targets.join('|')}`)
    const ownsOperation = !opts.operationToken
    const startedAt = Date.now()
    const baselineByName = new Map(store.files.map((item) => [item.name, item.conversion_quality || null]))
    qualityRepairBaselinesRef.current = {
      ...qualityRepairBaselinesRef.current,
      ...Object.fromEntries(targets.map((name) => [name, { quality: baselineByName.get(name) || null, startedAt }])),
    }
    setQualityRepairResults((cur) => {
      const next = { ...cur }
      for (const name of targets) delete next[name]
      return next
    })
    setQualityRepairingNames((cur) => {
      const next = { ...cur }
      for (const name of targets) next[name] = true
      return next
    })
    try {
      const res = await store.repairQuality({
        pdf_names: targets,
        speed_mode: CONVERT_MODE,
        replace: true,
      }, {
        autoReindexAfterQueued: opts.autoReindexQueued !== false,
      })
      const queued = Number(res.enqueued || 0)
      const repaired = Number(res.repaired || 0)
      const impact = res.impact || null
      const needsReindex = Boolean(res.needs_reindex || impact?.needs_reindex)
      let reindexed = false
      if (!qualityOperationIsCurrent(operationToken)) {
        return { ok: false, targetCount: targets.length, queued, repaired, needsReindex, reindexed, impact: null as LibraryQualityRepairImpact | null }
      }
      if (res.repair_run) {
        setQualityRepairRun(res.repair_run)
      }
      if (impact) {
        setQualityRepairImpact(impact)
      }
      if (queued > 0 || repaired > 0) {
        message.success(
          queued > 0
            ? S.lib_msg_quality_repair_enqueued.replace('{n}', String(queued))
            : `Markdown repaired: ${repaired}`,
        )
      } else {
        qualityRepairBaselinesRef.current = Object.fromEntries(
          Object.entries(qualityRepairBaselinesRef.current).filter(([name]) => !targets.includes(name)),
        )
        message.info(S.lib_msg_quality_repair_none)
      }
      await store.loadFiles(scope)
      if (!qualityOperationIsCurrent(operationToken)) {
        return { ok: false, targetCount: targets.length, queued, repaired, needsReindex, reindexed, impact: null as LibraryQualityRepairImpact | null }
      }
      if (needsReindex && repaired > 0 && queued <= 0 && opts.autoReindexImmediate !== false) {
        reindexed = await handleReindex(operationToken)
        if (!qualityOperationIsCurrent(operationToken)) {
          return { ok: false, targetCount: targets.length, queued, repaired, needsReindex, reindexed, impact: null as LibraryQualityRepairImpact | null }
        }
        if (impact) {
          setQualityRepairImpact({ ...impact, reindexed })
        }
        if (res.repair_run?.run_id) {
          const status = reindexed ? 'completed' : 'warning'
          const phase = reindexed ? 'reindex_complete' : 'reindex_failed'
          setQualityRepairRun({ ...res.repair_run, status, phase, reindexed })
          libraryApi.updateQualityRepairRun(res.repair_run.run_id, { status, phase, reindexed }).catch(() => {})
        }
        if (reindexed) await store.loadFiles(scope)
      }
      return { ok: true, targetCount: targets.length, queued, repaired, needsReindex, reindexed, impact }
    } catch (err) {
      qualityRepairBaselinesRef.current = Object.fromEntries(
        Object.entries(qualityRepairBaselinesRef.current).filter(([name]) => !targets.includes(name)),
      )
      if (qualityOperationIsCurrent(operationToken)) {
        message.error(err instanceof Error ? err.message : S.lib_msg_quality_repair_failed)
      }
      return { ok: false, targetCount: targets.length, queued: 0, repaired: 0, needsReindex: false, reindexed: false, impact: null as LibraryQualityRepairImpact | null }
    } finally {
      setQualityRepairingNames((cur) => {
        const next = { ...cur }
        for (const name of targets) delete next[name]
        return next
      })
      if (ownsOperation) clearQualityOperation(operationToken)
    }
  }

  const handleRepairQualityOne = async (item: LibraryFileItem) => {
    if (item.task_state !== 'idle' || !hasConversionQualityIssue(item)) return
    await repairQualityByNames([item.name])
  }

  const handleRepairSelectedQuality = async () => {
    await repairQualityByNames(selectedQualityReviewNames)
  }

  const handleFocusQualityReview = () => {
    if (qualityReviewCount <= 0) {
      message.info(S.lib_quality_report_no_issues)
      return
    }
    setQualityCenterOpen(true)
    setQualityHistoryFocusNames([])
    setOnlyQualityIssues(true)
    setBrowseMode('list')
    setTabKey('all')
  }

  const handleFocusQualityIssue = (label: string) => {
    const keyword = String(label || '').trim()
    if (!keyword) return
    setQualityCenterOpen(true)
    setFileKeyword(keyword)
    setQualityHistoryFocusNames([])
    setOnlyQualityIssues(true)
    setBrowseMode('list')
    setTabKey('all')
  }

  const focusQualityHistoryNames = (names: string[]) => {
    const availableNames = new Set(store.files.map((item) => item.name))
    const rawTargets = Array.from(new Set(names.map((name) => String(name || '').trim()).filter(Boolean)))
    const targets = rawTargets.filter((name) => availableNames.has(name))
    setQualityCenterOpen(true)
    if (!targets.length) {
      if (rawTargets.length > 0) {
        setQualityHistoryFocusNames(rawTargets)
        setBrowseMode('list')
        setTabKey('all')
        if (scope !== 'all') {
          setScope('all')
          void store.loadFiles('all')
        }
        return
      }
      message.info(S.lib_quality_history_no_remaining)
      return
    }
    setQualityHistoryFocusNames(targets)
    setBrowseMode('list')
    setTabKey('all')
  }

  const handleFocusQualityHistoryRemaining = () => {
    focusQualityHistoryNames(qualityHistoryRemainingNames)
  }

  const handleRepairRecommendedQuality = async (opts: QualityRepairRunOptions = {}) => {
    if (!qualityRepairRecommendedNames.length) {
      message.info(S.lib_quality_history_no_recommended)
      return { ok: true, targetCount: 0, queued: 0, repaired: 0, needsReindex: false, reindexed: false, impact: null as LibraryQualityRepairImpact | null }
    }
    return repairQualityByNames(qualityRepairRecommendedNames, opts)
  }

  const openQualityArtifact = async (domain: 'research_qa' | 'citation_cards', target: 'report' | 'folder' | 'raw' | 'summary' | 'runbook') => {
    const key = `${domain}:${target}`
    setQualityArtifactOpening(key)
    try {
      await libraryApi.openQualityArtifact(domain, target)
      message.success(S.lib_quality_artifact_opened)
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.lib_quality_artifact_open_failed)
    } finally {
      setQualityArtifactOpening('')
    }
  }

  const repairReaderLocateSources = async () => {
    const sources = qualityReaderLocateRecommendedSources
      .map((source) => ({
        source_path: normalizeTextValue(source.source_path || source.md_path || source.pdf_path),
        source_name: normalizeTextValue(source.source_name),
      }))
      .filter((source) => source.source_path || source.source_name)
    if (!sources.length) {
      message.info('Reader locate has no source repair targets yet')
      return
    }
    const operationToken = beginQualityOperation('reader-locate-source-repair')
    try {
      const res = await store.repairQuality({
        sources,
        speed_mode: CONVERT_MODE,
        replace: true,
        md_autofix: true,
      }, {
        autoReindexAfterQueued: true,
      })
      if (!qualityOperationIsCurrent(operationToken)) return
      if (res.repair_run) setQualityRepairRun(res.repair_run)
      if (res.impact) setQualityRepairImpact(res.impact)
      const enqueued = Number(res.enqueued || 0)
      const repaired = Number(res.repaired || 0)
      const needsReindex = Boolean(res.needs_reindex || res.impact?.needs_reindex)
      if (enqueued > 0) {
        message.success(S.lib_msg_quality_repair_enqueued.replace('{n}', String(enqueued)))
      } else if (repaired > 0) {
        message.success(`Reader locate source repair applied: ${repaired}`)
      } else if (Number(res.failed || 0) > 0) {
        message.warning('Reader locate source repair needs another pass')
      } else {
        message.info(S.lib_msg_quality_repair_none)
      }
      if (needsReindex && enqueued <= 0) {
        if (res.repair_run?.run_id) {
          const advanced = await libraryApi.advanceQualityRepairRun(res.repair_run.run_id)
          if (!qualityOperationIsCurrent(operationToken)) return
          setQualityRepairRun(advanced.item)
          if (advanced.reindex?.ok) {
            setQualityRepairImpact(res.impact ? { ...res.impact, reindexed: true } : null)
          } else if (advanced.reindex) {
            setQualityRepairImpact(res.impact ? { ...res.impact, reindexed: false } : null)
          }
        } else {
          const reindexed = await handleReindex(operationToken)
          if (!qualityOperationIsCurrent(operationToken)) return
          if (res.impact) setQualityRepairImpact({ ...res.impact, reindexed })
        }
      }
      await store.loadQualityOverview('all')
    } catch (err) {
      if (qualityOperationIsCurrent(operationToken)) {
        message.error(err instanceof Error ? err.message : 'Reader locate repair failed')
      }
    } finally {
      clearQualityOperation(operationToken)
    }
  }

  const handleQualityPriorityAction = async (action: LibraryQualityPriorityAction) => {
    const domain = normalizeTextValue(action.domain)
    if (domain === 'conversion') {
      handleFocusQualityReview()
      return
    }
    if (domain === 'reader_locate') {
      await repairReaderLocateSources()
      return
    }
    if (domain === 'research_qa' || domain === 'citation_cards') {
      const label = normalizeTextValue(action.label).toLowerCase()
      await openQualityArtifact(domain, label.includes('run') ? 'runbook' : 'report')
    }
  }

  const openResearchQaReplayCase = (item: LibraryQualityFailureCase) => {
    const caseId = normalizeTextValue(item.id)
    if (!caseId) {
      void openQualityArtifact('research_qa', 'report')
      return
    }
    saveResearchQaReplayFailureCase(item)
    if (!INTERNAL_ROUTES_ENABLED) {
      void openQualityArtifact('research_qa', 'report')
      return
    }
    nav(`/__research_qa_replay__?case=${encodeURIComponent(caseId)}&source=quality`)
  }

  const qualityCaseRepairSources = (item: LibraryQualityFailureCase) => {
    return (item.source_diagnostics || [])
      .filter((source) => Boolean(source.repairable))
      .filter((source) => Boolean(source.needs_repair) || ['error', 'warning'].includes(normalizeTextValue(source.quality_status).toLowerCase()))
      .map((source) => ({
        source_path: normalizeTextValue(source.source_path || source.md_path || source.pdf_path),
        source_name: normalizeTextValue(source.source_name || source.title),
      }))
      .filter((source) => source.source_path || source.source_name)
  }

  const waitForLibraryConversionDone = async (timeoutMs = 180000) => {
    return new Promise<boolean>((resolve) => {
      let settled = false
      let ctrl: AbortController | null = null
      let timer = 0
      const finish = (ok: boolean) => {
        if (settled) return
        settled = true
        window.clearTimeout(timer)
        ctrl?.abort()
        resolve(ok)
      }
      timer = window.setTimeout(() => {
        finish(false)
      }, timeoutMs)
      ctrl = libraryApi.streamConvertStatus(
        (data) => {
          if (data.done) finish(true)
        },
        () => finish(true),
        () => finish(false),
      )
    })
  }

  const repairQualityCaseSources = async (
    item: LibraryQualityFailureCase,
    opts: { manageActionKey?: boolean; waitForCompletion?: boolean; silent?: boolean; actionKey?: string } & QualityRepairRunOptions = {},
  ): Promise<{
    queued: number
    completed: boolean
    repaired: number
    needsReindex: boolean
    reindexed: boolean
    impact: LibraryQualityRepairImpact | null
  }> => {
    const sources = qualityCaseRepairSources(item)
    if (!sources.length) {
      if (!opts.silent) message.info(S.lib_msg_quality_repair_none)
      return { queued: 0, completed: true, repaired: 0, needsReindex: false, reindexed: false, impact: null }
    }
    const operationToken = opts.operationToken || beginQualityOperation(`case-source-repair:${normalizeTextValue(item.id)}`)
    const ownsOperation = !opts.operationToken
    const key = opts.actionKey || `${item.id}:repair_sources`
    const manageActionKey = opts.manageActionKey !== false
    if (manageActionKey) setQualityCaseActionKey(key)
    try {
      const res = await store.repairQuality({
        sources,
        speed_mode: CONVERT_MODE,
        replace: true,
      }, {
        autoReindexAfterQueued: opts.waitForCompletion ? false : opts.autoReindexQueued !== false,
      })
      const queued = Number(res.enqueued || 0)
      const repaired = Number(res.repaired || 0)
      const impact = res.impact || null
      const needsReindex = Boolean(res.needs_reindex || impact?.needs_reindex)
      let reindexed = false
      if (!qualityOperationIsCurrent(operationToken)) {
        return { queued, completed: false, repaired, needsReindex, reindexed, impact: null }
      }
      if (res.repair_run) {
        setQualityRepairRun(res.repair_run)
      }
      if (impact) {
        setQualityRepairImpact(impact)
      }
      if (queued > 0) {
        if (!opts.silent) message.success(S.lib_msg_quality_repair_enqueued.replace('{n}', String(queued)))
        const completed = opts.waitForCompletion ? await waitForLibraryConversionDone() : false
        if (!qualityOperationIsCurrent(operationToken)) {
          return { queued, completed: false, repaired, needsReindex, reindexed, impact: null }
        }
        if (completed && needsReindex && opts.autoReindexImmediate !== false) {
          reindexed = await handleReindex(operationToken)
          if (!qualityOperationIsCurrent(operationToken)) {
            return { queued, completed: false, repaired, needsReindex, reindexed, impact: null }
          }
          if (impact) setQualityRepairImpact({ ...impact, reindexed })
          if (res.repair_run?.run_id) {
            const status = reindexed ? 'completed' : 'warning'
            const phase = reindexed ? 'reindex_complete' : 'reindex_failed'
            setQualityRepairRun({ ...res.repair_run, status, phase, reindexed })
            libraryApi.updateQualityRepairRun(res.repair_run.run_id, { status, phase, reindexed }).catch(() => {})
          }
          if (reindexed) await store.loadFiles(scope)
        }
        return { queued, completed, repaired, needsReindex, reindexed, impact }
      } else if (repaired > 0) {
        if (!opts.silent) message.success(`Markdown repaired: ${repaired}`)
        if (needsReindex && opts.autoReindexImmediate !== false) {
          reindexed = await handleReindex(operationToken)
          if (!qualityOperationIsCurrent(operationToken)) {
            return { queued: 0, completed: false, repaired, needsReindex, reindexed, impact: null }
          }
          if (impact) setQualityRepairImpact({ ...impact, reindexed })
          if (res.repair_run?.run_id) {
            const status = reindexed ? 'completed' : 'warning'
            const phase = reindexed ? 'reindex_complete' : 'reindex_failed'
            setQualityRepairRun({ ...res.repair_run, status, phase, reindexed })
            libraryApi.updateQualityRepairRun(res.repair_run.run_id, { status, phase, reindexed }).catch(() => {})
          }
          if (reindexed) await store.loadFiles(scope)
        }
        return { queued: 0, completed: true, repaired, needsReindex, reindexed, impact }
      } else {
        if (!opts.silent) message.info(S.lib_msg_quality_repair_none)
        return { queued: 0, completed: true, repaired, needsReindex, reindexed, impact }
      }
    } catch (err) {
      if (qualityOperationIsCurrent(operationToken)) {
        message.error(err instanceof Error ? err.message : S.lib_msg_quality_repair_failed)
      }
      return { queued: 0, completed: false, repaired: 0, needsReindex: false, reindexed: false, impact: null }
    } finally {
      if (manageActionKey && qualityOperationIsActive(operationToken)) setQualityCaseActionKey('')
      if (ownsOperation) clearQualityOperation(operationToken)
    }
  }

  const copyQualityFailureSummary = async (item: LibraryQualityFailureCase) => {
    const lines = [
      `Case: ${normalizeTextValue(item.id)}`,
      `Question: ${normalizeTextValue(item.question)}`,
      `Failures: ${(item.failure_names || []).join(' / ') || 'none'}`,
      `Missing docs: ${(item.missing_expected_doc_ids || []).join(' / ') || 'none'}`,
      `Root causes: ${(item.root_causes || []).map((cause) => cause.label).join(' / ') || 'unknown'}`,
      `Sources: ${(item.source_diagnostics || []).map((source) => source.title || source.source_name || source.source_path).filter(Boolean).join(' / ') || 'none'}`,
    ].join('\n')
    try {
      await navigator.clipboard.writeText(lines)
      message.success(S.copied || 'Copied')
    } catch {
      message.error(S.copy_failed || 'Copy failed')
    }
  }

  const storeQualityCaseRerunResult = (caseId: string, res: LibraryResearchQaRerunResponse) => {
    setQualityCaseRerunResults((cur) => ({ ...cur, [caseId]: res }))
    if (res.status === 'passed' || res.quality_ok) {
      message.success(`QA case passed: ${caseId}`)
    } else if (res.status === 'failed') {
      message.warning(`QA case still failing: ${caseId}`)
    } else if (normalizeTextValue(res.error_kind).toLowerCase() === 'connection') {
      message.warning(`QA service is unreachable: ${caseId}`)
    } else if (normalizeTextValue(res.error_kind).toLowerCase() === 'timeout') {
      message.warning(`QA rerun timed out: ${caseId}`)
    } else {
      message.error(`QA rerun error: ${caseId}`)
    }
  }

  const runQualityFailureCaseRerun = async (item: LibraryQualityFailureCase, operationToken?: LibraryQualityOperationToken) => {
    const caseId = normalizeTextValue(item.id)
    if (!caseId) return null
    const token = operationToken || beginQualityOperation(`qa-rerun:${caseId}`)
    const ownsOperation = !operationToken
    const res = await libraryApi.rerunResearchQaCase({ case_id: caseId })
    if (!qualityOperationIsCurrent(token)) {
      if (ownsOperation) clearQualityOperation(token)
      return null
    }
    storeQualityCaseRerunResult(caseId, res)
    await store.loadQualityOverview('all')
    if (ownsOperation) clearQualityOperation(token)
    return res
  }

  const rerunQualityFailureCase = async (item: LibraryQualityFailureCase) => {
    const caseId = normalizeTextValue(item.id)
    if (!caseId) return
    const key = `${item.id}:rerun_case:`
    setQualityCaseActionKey(key)
    const operationToken = beginQualityOperation(key)
    try {
      await runQualityFailureCaseRerun(item, operationToken)
    } catch (err) {
      if (qualityOperationIsCurrent(operationToken)) {
        message.error(err instanceof Error ? err.message : 'QA rerun failed')
      }
    } finally {
      if (qualityOperationIsActive(operationToken)) setQualityCaseActionKey('')
      clearQualityOperation(operationToken)
    }
  }

  const repairQualityCaseShelfMetadata = async (item: LibraryQualityFailureCase, operationToken?: LibraryQualityOperationToken) => {
    const token = operationToken || beginQualityOperation(`case-metadata-repair:${normalizeTextValue(item.id)}`)
    const ownsOperation = !operationToken
    const backendTargets = Array.isArray(item.shelf_metadata_repair_targets) ? item.shelf_metadata_repair_targets : []
    const fallbackItems = [
      ...(Array.isArray(item.citation_diagnostics) ? item.citation_diagnostics : []),
      ...(Array.isArray(item.ref_diagnostics) ? item.ref_diagnostics : []),
    ]
    const sourceItems = backendTargets.length > 0 ? backendTargets : fallbackItems
    const candidates = sourceItems
      .map((entry, index) => ({
        record: entry as unknown as Record<string, unknown>,
        index,
      }))
      .map(({ record, index }) => ({
        ...record,
        key: normalizeTextValue(record.key) || `${item.id}:${backendTargets.length > 0 ? 'shelf-meta' : 'meta'}:${index}`,
        anchor: normalizeTextValue(record.anchor) || `${item.id}-${backendTargets.length > 0 ? 'shelf-meta' : 'meta'}-${index}`,
        title: normalizeTextValue(record.title),
        source_path: normalizeTextValue(record.source_path),
        source_name: normalizeTextValue(record.source_name || record.title),
        raw: normalizeTextValue(record.raw || record.cite_fmt || record.evidence_quote),
      }))
      .filter((entry) => entry.source_path || entry.source_name || entry.title || entry.raw)
      .slice(0, 12)
    if (!candidates.length) {
      if (ownsOperation) clearQualityOperation(token)
      return { ready: 0, exportReady: 0, changed: 0, retryable: 0, unresolved: 0, verification: {} as Record<string, unknown> }
    }
    let res: Awaited<ReturnType<typeof referencesApi.repairShelfMetadata>>
    try {
      res = await referencesApi.repairShelfMetadata(candidates as Array<Record<string, unknown>>, candidates.length)
    } catch (err) {
      if (ownsOperation) clearQualityOperation(token)
      throw err
    }
    const ready = Number(res.ready || 0)
    const exportReady = Number(res.acceptance?.export_ready_after || res.export_ready || ready)
    const changed = Number(res.changed || 0)
    const retryable = Number(res.retryable || 0)
    const unresolved = Number(res.acceptance?.unresolved_after || res.unresolved || 0)
    if (!qualityOperationIsCurrent(token)) {
      if (ownsOperation) clearQualityOperation(token)
      return { ready: 0, exportReady: 0, changed: 0, retryable: 0, unresolved: 0, verification: {} as Record<string, unknown> }
    }
    if (res.repair_run) {
      setQualityRepairRun(res.repair_run as unknown as LibraryQualityRepairRun)
    }
    if (retryable > 0) {
      message.warning(`Metadata repair queued for retry: ${retryable}`)
    } else if (changed > 0) {
      message.success(`Citation metadata repaired: ${changed}`)
    }
    if (ownsOperation) clearQualityOperation(token)
    return { ready, exportReady, changed, retryable, unresolved, verification: (res.verification || res.repair_run?.verification || {}) as Record<string, unknown> }
  }

  const applyQualityFailureRepairPlan = async (
    item: LibraryQualityFailureCase,
    action: LibraryQualityRepairAction,
    operationToken?: LibraryQualityOperationToken,
  ) => {
    const caseId = normalizeTextValue(item.id)
    const steps = Array.isArray(action.steps) ? action.steps : []
    const stepKinds = new Set(steps.map((step) => normalizeTextValue(step.kind)))
    const key = `${item.id}:apply_repair_plan:${action.target || ''}`
    const token = operationToken || beginQualityOperation(key)
    const ownsOperation = !operationToken
    setQualityCaseActionKey(key)
    try {
      let sourceRepairImpact: LibraryQualityRepairImpact | null = null
      if (stepKinds.has('repair_sources')) {
        const result = await repairQualityCaseSources(item, {
          actionKey: key,
          manageActionKey: false,
          waitForCompletion: true,
          autoReindexImmediate: !stepKinds.has('rebuild_index'),
          operationToken: token,
        })
        if (!qualityOperationIsCurrent(token)) return { ok: false, caseId, status: 'stale', rerun: null as LibraryResearchQaRerunResponse | null }
        sourceRepairImpact = result.impact
        if (result.queued > 0 && !result.completed) {
          message.warning('Source repair is still running; QA rerun will wait for the next refresh.')
          await store.loadQualityOverview('all')
          return { ok: true, caseId, status: 'source_repair_running', rerun: null as LibraryResearchQaRerunResponse | null }
        }
      }
      if (stepKinds.has('repair_shelf_metadata')) {
        await repairQualityCaseShelfMetadata(item, token)
        if (!qualityOperationIsCurrent(token)) return { ok: false, caseId, status: 'stale', rerun: null as LibraryResearchQaRerunResponse | null }
      }
      if (stepKinds.has('rebuild_index')) {
        const ok = await handleReindex(token)
        if (!qualityOperationIsCurrent(token)) return { ok: false, caseId, status: 'stale', rerun: null as LibraryResearchQaRerunResponse | null }
        if (sourceRepairImpact) setQualityRepairImpact({ ...sourceRepairImpact, reindexed: ok })
        if (!ok) return { ok: false, caseId, status: 'reindex_failed', rerun: null as LibraryResearchQaRerunResponse | null }
      }
      if (stepKinds.has('rerun_case') && caseId) {
        const rerun = await runQualityFailureCaseRerun(item, token)
        if (!qualityOperationIsCurrent(token)) return { ok: false, caseId, status: 'stale', rerun: null as LibraryResearchQaRerunResponse | null }
        return { ok: Boolean(rerun?.quality_ok || rerun?.status === 'passed'), caseId, status: String(rerun?.status || ''), rerun }
      } else {
        await store.loadQualityOverview('all')
      }
      return { ok: true, caseId, status: 'repaired', rerun: null as LibraryResearchQaRerunResponse | null }
    } catch (err) {
      if (qualityOperationIsCurrent(token)) {
        message.error(err instanceof Error ? err.message : 'Quality repair plan failed')
      }
      return { ok: false, caseId, status: 'error', rerun: null as LibraryResearchQaRerunResponse | null }
    } finally {
      if (qualityOperationIsActive(token)) setQualityCaseActionKey('')
      if (ownsOperation) clearQualityOperation(token)
    }
  }

  const handleQualityFailureAction = async (item: LibraryQualityFailureCase, actionOrKind: LibraryQualityRepairAction | string, target = '') => {
    const actionKind = typeof actionOrKind === 'string' ? actionOrKind : actionOrKind.kind
    const actionTarget = typeof actionOrKind === 'string' ? target : (actionOrKind.target || target)
    const key = `${item.id}:${actionKind}:${actionTarget}`
    if (actionKind === 'open_replay') {
      openResearchQaReplayCase(item)
      return
    }
    if (actionKind === 'apply_repair_plan' && typeof actionOrKind !== 'string') {
      await applyQualityFailureRepairPlan(item, actionOrKind)
      return
    }
    if (actionKind === 'rerun_case') {
      await rerunQualityFailureCase(item)
      return
    }
    if (actionKind === 'repair_sources') {
      await repairQualityCaseSources(item)
      return
    }
    if (actionKind === 'rebuild_index') {
      const operationToken = beginQualityOperation(key)
      setQualityCaseActionKey(key)
      try {
        await handleReindex(operationToken)
      } finally {
        if (qualityOperationIsActive(operationToken)) setQualityCaseActionKey('')
        clearQualityOperation(operationToken)
      }
      return
    }
    if (actionKind === 'open_artifact') {
      await openQualityArtifact('research_qa', actionTarget === 'raw' ? 'raw' : 'report')
    }
  }

  const firstQualityCaseForStage = (stageKey: string) => (
    qualityFailureCases.find((item) => qualityFailureCaseMatchesStage(item, stageKey)) || qualityFailureCases[0] || null
  )

  const repairQualityStageShelfMetadata = async (stageKey: string, operationToken?: LibraryQualityOperationToken) => {
    const token = operationToken || beginQualityOperation(`stage-metadata:${stageKey}`)
    const ownsOperation = !operationToken
    const targets = qualityFailureCases.filter((item) => qualityFailureCaseMatchesStage(item, stageKey)).slice(0, 3)
    if (!targets.length) {
      const state = await startShelfMetadataBackfill({ silent: true, operationToken: token })
      if (!qualityOperationIsCurrent(token)) {
        if (ownsOperation) clearQualityOperation(token)
        return { targetCount: 0, targetIds: [] as string[], ready: 0, exportReady: 0, changed: 0, retryable: 0, unresolved: 0, verification: {} as Record<string, unknown>, running: false }
      }
      const res = state?.result || null
      const scan = state?.after_scan || res?.after_scan || state?.scan || res?.scan || null
      const ready = Number(res?.ready || scan?.ready || 0)
      const exportReady = Number(res?.acceptance?.export_ready_after || res?.export_ready || scan?.export_ready || ready)
      const changed = Number(res?.changed || res?.preheated || 0)
      const retryable = Number(res?.retryable || scan?.retryable || 0)
      const unresolved = Number(res?.acceptance?.unresolved_after || res?.unresolved || res?.remaining_targets || scan?.needs_repair || 0)
      const targetCount = Number(res?.requested || scan?.target_count || state?.target_total || 0)
      const verification = (state?.verification || res?.verification || res?.repair_run?.verification || {}) as Record<string, unknown>
      if (res?.repair_run) {
        setQualityRepairRun(res.repair_run as unknown as LibraryQualityRepairRun)
      }
      if (state?.running) {
        message.success('Library metadata backfill is running')
      } else if (changed > 0) {
        message.success(`Library metadata backfilled: ${changed}`)
      } else if (retryable > 0) {
        message.warning(`Library metadata can retry: ${retryable}`)
      } else if (targetCount > 0 && exportReady > 0) {
        message.success(`Library metadata export-ready: ${exportReady}`)
      } else {
        message.info('No repairable library metadata found.')
      }
      await store.loadQualityOverview('all')
      if (ownsOperation) clearQualityOperation(token)
      return { targetCount, targetIds: [] as string[], ready, exportReady, changed, retryable, unresolved, verification, running: Boolean(state?.running) }
    }
    let changed = 0
    let ready = 0
    let exportReady = 0
    let retryable = 0
    let unresolved = 0
    let verification: Record<string, unknown> = {}
    for (const item of targets) {
      const res = await repairQualityCaseShelfMetadata(item, token)
      if (!qualityOperationIsCurrent(token)) {
        if (ownsOperation) clearQualityOperation(token)
        return { targetCount: 0, targetIds: [] as string[], ready: 0, exportReady: 0, changed: 0, retryable: 0, unresolved: 0, verification: {} as Record<string, unknown>, running: false }
      }
      changed += Number(res.changed || 0)
      ready += Number(res.ready || 0)
      exportReady += Number(res.exportReady || 0)
      retryable += Number(res.retryable || 0)
      unresolved += Number(res.unresolved || 0)
      if (!Object.keys(verification).length && res.verification && Object.keys(res.verification).length) {
        verification = res.verification
      }
    }
    if (changed <= 0 && ready <= 0) {
      message.info('No repairable citation metadata found in the current failed cases.')
    }
    await store.loadQualityOverview('all')
    if (ownsOperation) clearQualityOperation(token)
    return { targetCount: targets.length, targetIds: targets.map((item) => normalizeTextValue(item.id)).filter(Boolean), ready, exportReady, changed, retryable, unresolved, verification, running: false }
  }

  const refreshQualityOverviewSnapshot = async () => {
    await store.loadQualityOverview('all')
    const overview = useLibraryStore.getState().qualityOverview
    return overview?.ok ? overview : null
  }

  async function startShelfMetadataBackfill(options: { silent?: boolean; operationToken?: LibraryQualityOperationToken } = {}) {
    const operationToken = options.operationToken || beginQualityOperation('shelf-metadata-backfill')
    const ownsOperation = !options.operationToken
    setShelfMetadataBackfillRefreshing(true)
    try {
      const res = await referencesApi.startShelfMetadataBackfill(40, 240)
      if (!qualityOperationIsCurrent(operationToken)) return null
      setShelfMetadataBackfillState(res.state)
      const state = res.state || null
      if (!options.silent) {
        if (res.started) {
          message.success('Library metadata backfill started')
        } else if (state?.running || res.reason === 'already_running') {
          message.info('Library metadata backfill is already running')
        } else {
          message.warning('Library metadata backfill did not start')
        }
      }
      return state
    } catch (err) {
      if (qualityOperationIsCurrent(operationToken) && !options.silent) {
        message.error(err instanceof Error ? err.message : 'Metadata backfill failed to start')
      }
      return null
    } finally {
      if (qualityOperationIsActive(operationToken)) setShelfMetadataBackfillRefreshing(false)
      if (ownsOperation) clearQualityOperation(operationToken)
    }
  }

  const handleQualityFullChainStage = async (stage: LibraryQualityFullChainStage) => {
    const stageKey = normalizeTextValue(stage.key).toLowerCase()
    const action = normalizeTextValue(stage.action).toLowerCase()
    const operationToken = beginQualityOperation(`full-chain:${stageKey}:${action}`)
    const caseTarget = firstQualityCaseForStage(stageKey)
    const beforeOverview = backendQualityOverview
    const beforeSnapshot = qualityOverviewStageSnapshot(beforeOverview, stageKey)
    const recordStageResult = (
      result: Omit<QualityFullChainActionResult, 'updatedAt'>,
      meta: {
        targetIds?: string[]
        metrics?: Record<string, string | number | boolean | null | undefined>
        afterOverview?: LibraryQualityOverviewResponse | null
        verification?: Record<string, unknown>
      } = {},
    ) => {
      if (!qualityOperationIsCurrent(operationToken)) return
      const latestOverview = meta.afterOverview
        || (useLibraryStore.getState().qualityOverview?.ok ? useLibraryStore.getState().qualityOverview : null)
        || backendQualityOverview
      const afterSnapshot = qualityOverviewStageSnapshot(latestOverview, stageKey)
      recordQualityFullChainResult(stageKey, result, {
        stageLabel: stage.label,
        action: stage.action,
        before: beforeSnapshot,
        after: afterSnapshot,
        verification: meta.verification,
        targetIds: meta.targetIds,
        metrics: meta.metrics,
      })
    }
    setQualityFullChainActionKey(stageKey)
    try {
      if (stageKey === 'conversion' || action === 'repair_conversion') {
        if (qualityRepairRecommendedNames.length > 0) {
          const repair = await handleRepairRecommendedQuality({
            autoReindexImmediate: false,
            autoReindexQueued: false,
            operationToken,
          })
          if (!qualityOperationIsCurrent(operationToken)) return
          const completed = Number(repair?.queued || 0) > 0 ? await waitForLibraryConversionDone() : true
          if (!qualityOperationIsCurrent(operationToken)) return
          const repaired = Number(repair?.repaired || 0)
          const queued = Number(repair?.queued || 0)
          const needsReindex = Boolean(repair?.needsReindex || repair?.impact?.needs_reindex)
          const reindexed = completed && needsReindex ? await handleReindex(operationToken) : false
          if (!qualityOperationIsCurrent(operationToken)) return
          if (repair?.impact && needsReindex) {
            setQualityRepairImpact({ ...repair.impact, reindexed })
          }
          if (reindexed) await store.loadFiles(scope)
          if (!qualityOperationIsCurrent(operationToken)) return
          const rerun = completed && (!needsReindex || reindexed) && caseTarget ? await runQualityFailureCaseRerun(caseTarget, operationToken) : null
          if (!qualityOperationIsCurrent(operationToken)) return
          const afterOverview = await refreshQualityOverviewSnapshot()
          if (!qualityOperationIsCurrent(operationToken)) return
          const repairOk = Boolean(repair?.ok)
          const reindexFailed = Boolean(completed && needsReindex && !reindexed)
          recordStageResult({
            status: reindexFailed ? 'warning' : (queued > 0 || repaired > 0 ? 'success' : (repairOk ? 'info' : 'error')),
            summary: queued > 0
              ? (completed
                ? (reindexFailed ? `Converted ${queued} sources; index refresh failed` : `Verified ${queued} conversion repairs`)
                : `Queued ${queued} conversion repairs`)
              : (repaired > 0
                ? (reindexFailed ? `Markdown autofix repaired ${repaired}; index refresh failed` : `Markdown autofix repaired ${repaired} sources`)
                : (repairOk ? 'No conversion repair was queued' : 'Conversion repair failed')),
            detail: rerun?.case_id
              ? `Regression check: ${rerun.case_id}`
              : (reindexFailed ? 'Rebuild the retrieval index before rerunning QA.' : (repair?.targetCount ? `${repair.targetCount} recommended sources checked` : undefined)),
          }, {
            targetIds: qualityRepairRecommendedNames.slice(0, 12),
            metrics: {
              queued,
              repaired,
              target_count: Number(repair?.targetCount || 0),
              conversion_completed: Boolean(completed),
              needs_reindex: needsReindex,
              reindexed,
              qa_rerun_quality_ok: Boolean(rerun?.quality_ok || rerun?.status === 'passed'),
            },
            afterOverview,
            verification: qualityVerificationFromRerun(rerun),
          })
        } else {
          handleFocusQualityReview()
          recordStageResult({
            status: 'info',
            summary: 'Focused the conversion review list',
          })
        }
        return
      }
      if (stageKey === 'retrieval' || action === 'rebuild_index') {
        const ok = await handleReindex(operationToken)
        if (!qualityOperationIsCurrent(operationToken)) return
        const rerun = ok && caseTarget ? await runQualityFailureCaseRerun(caseTarget, operationToken) : null
        if (!qualityOperationIsCurrent(operationToken)) return
        if (ok && !rerun) await store.loadQualityOverview('all')
        const afterOverview = await refreshQualityOverviewSnapshot()
        if (!qualityOperationIsCurrent(operationToken)) return
        const rerunPassed = Boolean(rerun?.quality_ok || rerun?.status === 'passed')
        recordStageResult({
          status: ok ? (rerun && !rerunPassed ? 'warning' : 'success') : 'error',
          summary: ok
            ? (rerun ? (rerunPassed ? `Reindex verified: ${caseTarget?.id}` : `Reindex done; QA still failing: ${caseTarget?.id}`) : 'Rebuilt retrieval index')
            : 'Retrieval index rebuild failed',
          detail: rerun?.failures?.length ? `${rerun.failures.length} failures remain` : (caseTarget?.id ? `Regression check: ${caseTarget.id}` : undefined),
        }, {
          targetIds: caseTarget?.id ? [caseTarget.id] : [],
          metrics: {
            qa_rerun_quality_ok: rerun ? rerunPassed : false,
            failure_count: Number(rerun?.failures?.length || 0),
          },
          afterOverview,
          verification: qualityVerificationFromRerun(rerun),
        })
        return
      }
      if (stageKey === 'citations' || stageKey === 'shelf' || action === 'repair_citation_cards' || action === 'repair_shelf_metadata') {
        const result = await repairQualityStageShelfMetadata(stageKey === 'citations' ? 'citations' : 'shelf', operationToken)
        if (!qualityOperationIsCurrent(operationToken)) return
        const rerun = result.targetCount > 0 && caseTarget ? await runQualityFailureCaseRerun(caseTarget, operationToken) : null
        if (!qualityOperationIsCurrent(operationToken)) return
        const afterOverview = await refreshQualityOverviewSnapshot()
        if (!qualityOperationIsCurrent(operationToken)) return
        const rerunPassed = Boolean(rerun?.quality_ok || rerun?.status === 'passed')
        const shelfVerification = result.verification && Object.keys(result.verification).length ? result.verification : {}
        const qaVerification = qualityVerificationFromRerun(rerun)
        const verification = Object.keys(shelfVerification).length && Object.keys(qaVerification).length
          ? {
            type: 'combined_shelf_metadata_verification',
            quality_ok: Boolean(shelfVerification.quality_ok) && Boolean(qaVerification.quality_ok),
            shelf_metadata: shelfVerification,
            research_qa: qaVerification,
          }
          : (Object.keys(shelfVerification).length ? shelfVerification : qaVerification)
        const resultRunning = Boolean(result.running)
        recordStageResult({
          status: resultRunning ? 'success' : (result.retryable > 0 || result.unresolved > 0 || (rerun && !rerunPassed) ? 'warning' : (result.changed > 0 || result.ready > 0 ? 'success' : 'info')),
          summary: resultRunning
            ? 'Metadata backfill started'
            : rerun
            ? (rerunPassed ? `Metadata repair verified: ${caseTarget?.id}` : `Metadata checked; QA still failing: ${caseTarget?.id}`)
            : (result.changed > 0
              ? (result.unresolved > 0 ? `Metadata repaired ${result.changed}; ${result.unresolved} still missing` : `Metadata repaired: ${result.changed}`)
              : (result.exportReady > 0 ? `Metadata export ready: ${result.exportReady}` : (result.ready > 0 ? `Metadata already ready: ${result.ready}` : 'Opened citation quality report'))),
          detail: resultRunning
            ? (result.targetCount > 0 ? `${result.targetCount} metadata targets queued` : 'Scanning the reference index')
            : (rerun?.failures?.length ? `${rerun.failures.length} failures remain` : (result.targetCount > 0 ? `${result.targetCount} failed cases checked` : undefined)),
        }, {
          targetIds: result.targetIds,
          metrics: {
            changed: result.changed,
            ready: result.ready,
            export_ready: result.exportReady,
            retryable: result.retryable,
            unresolved: result.unresolved,
            target_count: result.targetCount,
            async_running: resultRunning,
            qa_rerun_quality_ok: rerun ? rerunPassed : false,
          },
          afterOverview,
          verification,
        })
        return
      }
      if (stageKey === 'repair_loop' || action === 'rerun_failed_cases') {
        if (caseTarget) {
          const rerun = await runQualityFailureCaseRerun(caseTarget, operationToken)
          if (!qualityOperationIsCurrent(operationToken)) return
          const afterOverview = await refreshQualityOverviewSnapshot()
          if (!qualityOperationIsCurrent(operationToken)) return
          recordStageResult({
            status: rerun?.quality_ok || rerun?.status === 'passed' ? 'success' : 'warning',
            summary: rerun?.quality_ok || rerun?.status === 'passed'
              ? `Rerun passed: ${caseTarget.id}`
              : `Rerun still failing: ${caseTarget.id}`,
            detail: rerun?.failures?.length ? `${rerun.failures.length} failures remain` : undefined,
          }, {
            targetIds: [caseTarget.id],
            metrics: {
              quality_ok: Boolean(rerun?.quality_ok),
              failure_count: Number(rerun?.failures?.length || 0),
            },
            afterOverview,
            verification: qualityVerificationFromRerun(rerun),
          })
        } else {
          await openQualityArtifact('research_qa', 'report')
          recordStageResult({
            status: 'info',
            summary: 'Opened QA report',
          })
        }
        return
      }
      if (stageKey === 'research_qa' || action === 'fix_failed_qa_cases') {
        const plan = caseTarget?.repair_actions?.find((item) => item.kind === 'apply_repair_plan')
        if (caseTarget && plan) {
          const result = await applyQualityFailureRepairPlan(caseTarget, plan, operationToken)
          if (!qualityOperationIsCurrent(operationToken)) return
          const afterOverview = await refreshQualityOverviewSnapshot()
          if (!qualityOperationIsCurrent(operationToken)) return
          recordStageResult({
            status: result?.ok ? 'success' : 'warning',
            summary: result?.rerun?.quality_ok || result?.rerun?.status === 'passed'
              ? `Repair plan passed: ${caseTarget.id}`
              : `Repair plan ran: ${caseTarget.id}`,
            detail: result?.status ? `Last status: ${result.status}` : undefined,
          }, {
            targetIds: [caseTarget.id],
            metrics: {
              quality_ok: Boolean(result?.rerun?.quality_ok),
              has_rerun: Boolean(result?.rerun),
            },
            afterOverview,
            verification: qualityVerificationFromRerun(result?.rerun),
          })
        } else if (caseTarget) {
          const rerun = await runQualityFailureCaseRerun(caseTarget, operationToken)
          if (!qualityOperationIsCurrent(operationToken)) return
          const afterOverview = await refreshQualityOverviewSnapshot()
          if (!qualityOperationIsCurrent(operationToken)) return
          recordStageResult({
            status: rerun?.quality_ok || rerun?.status === 'passed' ? 'success' : 'warning',
            summary: rerun?.quality_ok || rerun?.status === 'passed'
              ? `QA case passed: ${caseTarget.id}`
              : `QA case still failing: ${caseTarget.id}`,
            detail: rerun?.failures?.length ? `${rerun.failures.length} failures remain` : undefined,
          }, {
            targetIds: [caseTarget.id],
            metrics: {
              quality_ok: Boolean(rerun?.quality_ok),
              failure_count: Number(rerun?.failures?.length || 0),
            },
            afterOverview,
            verification: qualityVerificationFromRerun(rerun),
          })
        } else {
          await openQualityArtifact('research_qa', action === 'run_research_qa' ? 'runbook' : 'report')
          recordStageResult({
            status: 'info',
            summary: action === 'run_research_qa' ? 'Opened QA runbook' : 'Opened QA report',
          })
        }
        return
      }
      if (stageKey === 'citation_cards') {
        await openQualityArtifact('citation_cards', 'report')
        recordStageResult({
          status: 'info',
          summary: 'Opened citation-card quality report',
        })
        return
      }
      message.info('No direct action is available for this quality stage yet.')
      recordStageResult({
        status: 'info',
        summary: 'No direct action available for this stage',
      })
    } catch (err) {
      if (qualityOperationIsCurrent(operationToken)) {
        message.error(err instanceof Error ? err.message : 'Quality stage action failed')
      }
    } finally {
      if (qualityOperationIsActive(operationToken)) setQualityFullChainActionKey('')
      clearQualityOperation(operationToken)
    }
  }

  const handleQualityFeatureHealthAction = async (item: LibraryQualityFeatureHealthItem) => {
    const targetStage = normalizeTextValue(item.target_stage).toLowerCase()
    const stage = qualityFullChainStages.find((stageItem) => normalizeTextValue(stageItem.key).toLowerCase() === targetStage)
    if (stage) {
      await handleQualityFullChainStage(stage)
      return
    }
    const featureKey = normalizeTextValue(item.key).toLowerCase()
    if (featureKey === 'pdf_conversion') {
      handleFocusQualityReview()
      return
    }
    if (featureKey === 'reader_locate') {
      await repairReaderLocateSources()
      return
    }
    if (featureKey === 'citation_cards' || featureKey === 'literature_basket') {
      await openQualityArtifact('citation_cards', 'report')
      return
    }
    await openQualityArtifact('research_qa', featureKey === 'general_qa' ? 'report' : 'runbook')
  }

  const handleQualityActionHistoryOpen = async (item: LibraryQualityActionHistoryItem) => {
    const stageKey = normalizeTextValue(item.stage_key).toLowerCase()
    const targetIds = normalizeTextList(item.target_ids || [])
    const firstTarget = targetIds[0] || ''
    if (stageKey === 'conversion') {
      if (targetIds.length > 0) {
        focusQualityHistoryNames(targetIds)
      } else {
        handleFocusQualityReview()
      }
      return
    }
    if (['research_qa', 'retrieval', 'repair_loop', 'citations', 'shelf'].includes(stageKey)) {
      if (firstTarget) {
        const failureCase = qualityFailureCases.find((row) => normalizeTextValue(row.id) === firstTarget)
        if (failureCase) saveResearchQaReplayFailureCase(failureCase)
        if (!INTERNAL_ROUTES_ENABLED) {
          await openQualityArtifact(stageKey === 'citations' || stageKey === 'shelf' ? 'citation_cards' : 'research_qa', 'report')
          return
        }
        nav(`/__research_qa_replay__?case=${encodeURIComponent(firstTarget)}&source=quality-history`)
        return
      }
      await openQualityArtifact(stageKey === 'citations' || stageKey === 'shelf' ? 'citation_cards' : 'research_qa', 'report')
      return
    }
    if (stageKey === 'citation_cards') {
      await openQualityArtifact('citation_cards', 'report')
      return
    }
    await openQualityArtifact('research_qa', 'report')
  }

  const handleDeleteOne = async (item: LibraryFileItem) => {
    if (item.task_state !== 'idle') {
      message.warning(S.lib_menu_delete_busy)
      return
    }
    const res = await store.deleteFile(item.name, true)
    if (res.ok) {
      message.success(S.lib_msg_deleted_name.replace('{name}', item.name))
      if (Number(res.removed_queued || 0) > 0) {
        message.info(S.lib_msg_delete_removed_queued.replace('{n}', String(res.removed_queued)))
      }
      if (res.needs_reindex) {
        message.info(S.lib_msg_delete_suggest_reindex)
      }
      return
    }
    const warning = Array.isArray(res.warnings) && res.warnings.length > 0
      ? `: ${res.warnings.join('; ')}`
      : ''
    message.warning(S.lib_msg_delete_not_complete.replace('{warning}', warning))
  }

  const confirmDeleteOne = (item: LibraryFileItem) => {
    Modal.confirm({
      title: S.lib_menu_delete_confirm_title,
      content: (
        <div className="kb-lib-delete-confirm">
          <Text strong>{item.name}</Text>
          <Text type="secondary">{S.lib_menu_delete_confirm_detail}</Text>
          <Text type="secondary">{S.lib_menu_delete_confirm_index}</Text>
        </div>
      ),
      okText: S.lib_menu_delete_ok,
      okType: 'danger',
      cancelText: S.lib_menu_delete_cancel,
      onOk: async () => {
        await handleDeleteOne(item)
      },
    })
  }

  const handleReindex = async (operationToken?: LibraryQualityOperationToken): Promise<boolean> => {
    const token = operationToken || beginQualityOperation('reindex')
    const ownsOperation = !operationToken
    const hide = message.loading(S.lib_msg_updating_kb, 0)
    try {
      const res = await store.reindex()
      hide()
      if (!qualityOperationIsCurrent(token)) return false
      if (!res.ok) {
        const detail = [
          res.structured_indices_error,
          res.stderr,
          res.refsync_error,
        ].map((item) => String(item || '').trim()).find(Boolean)
        message.error(detail ? `${S.lib_msg_exec_fail}: ${detail}` : S.lib_msg_exec_fail)
        return false
      }
      message.success(S.lib_msg_exec_done)
      if (res.refsync_error) {
        message.warning(S.lib_msg_refsync_fail_detail.replace('{error}', String(res.refsync_error)))
      } else if (res.refsync?.started) {
        message.info(S.lib_msg_refsync_started_bg)
      }
      return true
    } catch (err) {
      hide()
      if (qualityOperationIsCurrent(token)) {
        message.error(err instanceof Error ? err.message : S.lib_msg_exec_fail)
      }
      return false
    } finally {
      if (ownsOperation) clearQualityOperation(token)
    }
  }

  const runConversionQualityBatch = async (repair: boolean) => {
    const operationToken = beginQualityOperation(repair ? 'quality-batch-repair' : 'quality-batch-scan')
    const hide = message.loading(repair ? 'Repairing conversion quality...' : 'Scanning conversion source quality...', 0)
    setQualityBatchRunning(true)
    try {
      const res = await libraryApi.conversionQualityBatch({
        repair,
        rebuild_indices: true,
        limit: 1000,
      })
      hide()
      if (!qualityOperationIsCurrent(operationToken)) return
      setQualityBatchResult(res)
      if (!res.ok) {
        message.error(S.lib_msg_exec_fail)
        return
      }
      if (repair && res.needs_reindex) {
        const reindexed = await handleReindex(operationToken)
        if (!qualityOperationIsCurrent(operationToken)) return
        if (!reindexed) {
          message.warning('Conversion repair finished, but index refresh needs retry.')
        } else {
          message.success(`Safe repair finished: ${res.changed} changed, index refreshed`)
        }
      } else {
        message.success(repair
          ? `Safe repair finished: ${res.changed} changed, ${res.ready} ready`
          : `Source scan finished: ${res.scanned} checked, ${res.ready} ready`)
      }
      await store.loadFiles(scope)
      if (!qualityOperationIsCurrent(operationToken)) return
      await store.loadQualityOverview('all')
    } catch (err) {
      hide()
      if (qualityOperationIsCurrent(operationToken)) {
        message.error(err instanceof Error ? err.message : S.lib_msg_exec_fail)
      }
    } finally {
      if (qualityOperationIsActive(operationToken)) setQualityBatchRunning(false)
      clearQualityOperation(operationToken)
    }
  }

  const runFigureAssetQualityScan = async (includeAll = false) => {
    setFigureAssetScanRunning(true)
    try {
      const res = await libraryApi.figureAssetQualityScan({
        limit: 1000,
        include_all: includeAll,
      })
      setFigureAssetScan(res)
      setFigureAssetRefreshResult(null)
      if (Number(res.refresh_recommended || 0) > 0) {
        message.warning(`Figure asset scan found ${res.refresh_recommended} sources to refresh`)
      } else {
        message.success(`Figure asset scan checked ${res.scanned} sources`)
      }
      return res
    } catch (err) {
      message.error(err instanceof Error ? err.message : 'Figure asset scan failed')
      return null
    } finally {
      setFigureAssetScanRunning(false)
    }
  }

  const refreshFigureAssets = async () => {
    if (!figureAssetScan) {
      message.info('Run a figure asset scan before refreshing flagged sources')
      return
    }
    const sourceItems = (figureAssetScan.items || []).filter((item) => Boolean(item.refresh_recommended))
    const sources = sourceItems.map((item) => ({
      source_path: item.md_path,
      source_name: item.source_name || item.pdf_name,
    })).filter((item) => item.source_path || item.source_name)
    if (sources.length <= 0) {
      message.info('No figure assets need refresh right now')
      return
    }
    setFigureAssetRefreshRunning(true)
    try {
      const res = await libraryApi.refreshFigureAssets({
        sources,
        limit: Math.max(1, sources.length),
        speed_mode: CONVERT_MODE,
        replace: true,
        target_dpi: figureAssetScan?.target_dpi,
      })
      setFigureAssetRefreshResult(res)
      if (Number(res.enqueued || 0) > 0) {
        message.success(`Figure asset refresh queued: ${res.enqueued}`)
        store.startProgressStream()
      } else if (Number(res.skipped_busy || 0) > 0) {
        message.warning(`Figure asset refresh skipped busy sources: ${res.skipped_busy}`)
      } else if (Number(res.failed || 0) > 0) {
        message.error(`Figure asset refresh failed: ${res.failed}`)
      } else {
        message.info('No figure assets need refresh right now')
      }
      await store.loadFiles(scope)
      await store.loadQualityOverview('all')
      return res
    } catch (err) {
      message.error(err instanceof Error ? err.message : 'Figure asset refresh failed')
      return null
    } finally {
      setFigureAssetRefreshRunning(false)
    }
  }

  const handleAdvanceQualityRepairRun = async () => {
    const runId = normalizeTextValue(qualityRepairRun?.run_id)
    if (!runId) return
    const operationToken = beginQualityOperation(`advance-repair-run:${runId}`)
    const hide = message.loading('Continuing repair run...', 0)
    setQualityRepairAdvancing(true)
    try {
      const res = await libraryApi.advanceQualityRepairRun(runId)
      hide()
      if (!qualityOperationIsCurrent(operationToken)) return
      if (res.item) {
        setQualityRepairRun(res.item)
        const impact = res.item.impact
        if (impact && typeof impact === 'object' && typeof (impact as LibraryQualityRepairImpact).needs_reindex === 'boolean') {
          setQualityRepairImpact(impact as LibraryQualityRepairImpact)
        }
      }
      if (res.waiting) {
        message.info('Conversion is still running; continue again after it finishes.')
      } else if (res.ok) {
        message.success(res.advanced ? 'Repair run advanced' : 'Repair run is already up to date')
      } else {
        message.error(res.detail || S.lib_msg_exec_fail)
      }
      if (res.reindex?.refsync_error) {
        message.warning(S.lib_msg_refsync_fail_detail.replace('{error}', String(res.reindex.refsync_error)))
      } else if (res.reindex?.refsync?.started) {
        message.info(S.lib_msg_refsync_started_bg)
      }
      await store.loadFiles(scope)
      if (!qualityOperationIsCurrent(operationToken)) return
      await store.loadQualityOverview('all')
    } catch (err) {
      hide()
      if (qualityOperationIsCurrent(operationToken)) {
        message.error(err instanceof Error ? err.message : S.lib_msg_exec_fail)
      }
    } finally {
      if (qualityOperationIsActive(operationToken)) setQualityRepairAdvancing(false)
      clearQualityOperation(operationToken)
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
      const convTitle = S.default_guide_title.replace('{name}', sourceName)
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
    const confirmedCategory = normalizeTextValue(item.paper_category)
    const suggestedCategory = normalizeTextValue(item.suggested_category)
    const confirmedTags = normalizeTextList(Array.isArray(item.user_tags) ? item.user_tags : [])
    const suggestedTags = normalizeTextList(Array.isArray(item.suggested_tags) ? item.suggested_tags : [])
    setMetaItem(item)
    setMetaDraft({
      paper_category: confirmedCategory || suggestedCategory,
      reading_status: (String(item.reading_status || '') as ReadingStatusValue),
      note: String(item.note || ''),
      user_tags: confirmedTags.length > 0 ? confirmedTags : normalizeTextList([...confirmedTags, ...suggestedTags]),
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
      const updated = await store.regenerateSuggestions({ pdf_names: targets, auto_apply_empty: true })
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

  const confirmBatchEditorRisk = async (params: { paperCategory: string, removeTags: string[] }): Promise<boolean> => {
    const willClearCategory = batchDraft.apply_paper_category && !params.paperCategory
    const willClearStatus = batchDraft.apply_reading_status && !batchDraft.reading_status
    const willRemoveTags = params.removeTags.length > 0
    if (!willClearCategory && !willClearStatus && !willRemoveTags) return true

    const previewTags = params.removeTags.slice(0, 8).join(', ')
    const removeTagsText = params.removeTags.length > 8 ? `${previewTags}...` : previewTags
    return new Promise<boolean>((resolve) => {
      let settled = false
      const done = (value: boolean) => {
        if (settled) return
        settled = true
        resolve(value)
      }
      Modal.confirm({
        title: S.lib_batch_confirm_title,
        icon: <ExclamationCircleOutlined />,
        content: (
          <div className="kb-lib-batch-confirm">
            <Text>{S.lib_batch_confirm_detail.replace('{n}', String(selectedLibraryCount))}</Text>
            {willClearCategory ? <Text type="warning">{S.lib_batch_confirm_clear_category}</Text> : null}
            {willClearStatus ? <Text type="warning">{S.lib_batch_confirm_clear_status}</Text> : null}
            {willRemoveTags ? (
              <Text type="danger">{S.lib_batch_confirm_remove_tags.replace('{tags}', removeTagsText)}</Text>
            ) : null}
          </div>
        ),
        okText: S.lib_batch_confirm_ok,
        cancelText: S.lib_batch_confirm_cancel,
        okButtonProps: { danger: true },
        onOk: () => done(true),
        onCancel: () => done(false),
        afterClose: () => done(false),
      })
    })
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
    const confirmed = await confirmBatchEditorRisk({ paperCategory, removeTags })
    if (!confirmed) return
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
    return (
      <LibraryFileRow
        S={S}
        item={item}
        selected={Boolean(selectedLibraryNames[item.name])}
        readingLabel={readingStatusLabel(item.reading_status, S)}
        onlyUnclassified={onlyUnclassified}
        paperCategoryFilter={paperCategoryFilter}
        readingStatusFilter={readingStatusFilter}
        paperTagFilter={paperTagFilter}
        qualityStatusVisible={QUALITY_STATUS_VISIBLE}
        qualityDiagnosticsVisible={QUALITY_DIAGNOSTICS_VISIBLE}
        qualityRepairing={Boolean(qualityRepairingNames[item.name])}
        qualityRepairResult={qualityRepairResults[item.name]}
        qualityRepairRecord={qualityRepairHistory[item.name]}
        onSelectionChange={toggleLibrarySelection}
        onApplyPaperCategoryFilter={applyPaperCategoryFilter}
        onSetReadingStatusFilter={setReadingStatusFilter}
        onApplyPaperTagFilter={applyPaperTagFilter}
        onRepairQuality={(rowItem) => { void handleRepairQualityOne(rowItem) }}
        onReindex={() => { void handleReindex() }}
        onOpenMeta={openMetaEditor}
        onStartPaperGuide={(rowItem) => { void handleStartPaperGuide(rowItem) }}
        onConvert={(rowItem) => { void handleConvertOne(rowItem) }}
        onOpenPdf={(name) => { void store.openFile(name, 'pdf') }}
        onOpenMarkdown={(name) => { void store.openFile(name, 'md') }}
        onDelete={confirmDeleteOne}
      />
    )
  }

  const renderFiles = (items: LibraryFileItem[], emptyText: string) => {
    return (
      <LibraryFileList
        items={items}
        emptyText={emptyText}
        virtualScrollHint={S.lib_virtual_scroll_hint}
        renderRow={renderFileRow}
      />
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
    quality_review: qualityReviewCount,
    quality_ready: qualityReadyCount,
    index_ready: qualitySourceReadinessStats.ready,
    index_quality_blocked: qualitySourceReadinessStats.blocked,
    index_stale: qualitySourceReadinessStats.indexStale,
  }

  const directoriesConfigured = Boolean(pdfDirDraft.trim() && mdDirDraft.trim())
  const showDirEditor = dirEditorOpen || !directoriesConfigured
  const workbenchStats: WorkbenchMetricItem[] = [
    { key: 'view', label: S.lib_stats_view, value: counts.total_view, tone: 'neutral' },
    { key: 'pending', label: S.lib_stats_pending, value: counts.pending, tone: counts.pending > 0 ? 'info' : 'neutral' },
    { key: 'converted', label: S.lib_stats_converted, value: counts.converted, tone: 'good' },
    { key: 'queued', label: S.lib_stats_queued, value: counts.queued, tone: counts.queued > 0 ? 'processing' : 'neutral' },
    { key: 'running', label: S.lib_stats_running, value: counts.running, tone: counts.running > 0 ? 'processing' : 'neutral' },
    { key: 'source_ready', label: S.lib_stats_source_ready, value: qualitySourceReadinessStats.ready, tone: 'good' },
    { key: 'source_blocked', label: S.lib_stats_quality_blocked, value: qualitySourceReadinessStats.blocked, tone: qualitySourceReadinessStats.blocked > 0 ? 'warn' : 'neutral' },
    { key: 'quality', label: S.lib_quality_report_review, value: counts.quality_review, tone: counts.quality_review > 0 ? 'warn' : 'neutral' },
  ]

  const renameHasResults = renameItems.length > 0
  const renameHasVisibleItems = renameVisible.length > 0
  const hasRenameSelection = selectedRenameCount > 0
  const showUploadWorkbench = uploadWorkbenchOpen && uploadDrafts.length > 0
  const showTaxonomySelectAction = browseMode === 'list' && currentListItems.length > 0
  const showTaxonomyRefreshAction = browseMode === 'list' && visibleAll.length > 0
  const showTaxonomyClearAction = hasActiveTaxonomyFilters
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
        <div className="kb-lib-rename-list">
          <div className="kb-lib-rename-list-body" role="list">
            {pagedRenameVisible.map((item) => (
              <div key={item.name} className="kb-lib-rename-list-item" role="listitem">
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
              </div>
            ))}
          </div>
          {renameVisible.length > RENAME_PAGE_SIZE ? (
            <Pagination
              className="kb-lib-list-pagination"
              size="small"
              current={renamePage}
              pageSize={RENAME_PAGE_SIZE}
              total={renameVisible.length}
              showSizeChanger={false}
              onChange={setRenamePage}
            />
          ) : null}
        </div>
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
          <LibraryDirectorySettings
            S={S}
            directoriesConfigured={directoriesConfigured}
            showDirEditor={showDirEditor}
            pdfDirDraft={pdfDirDraft}
            mdDirDraft={mdDirDraft}
            pickingDir={pickingDir}
            savingDirs={savingDirs}
            dirDirty={dirDirty}
            onToggleEditor={() => setDirEditorOpen((open) => !open)}
            onPdfDirChange={(value) => {
              setDirTouched(true)
              setPdfDirDraft(value)
            }}
            onMdDirChange={(value) => {
              setDirTouched(true)
              setMdDirDraft(value)
            }}
            onPickDir={pickDir}
            onOpenFolder={openFolder}
            onSaveDirs={saveDirs}
          />

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
                  data-testid="library-process-scope"
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
          <Button loading={uploadSaving || uploadInspecting} disabled={uploadLocked || retryableFailedUploadDrafts.length === 0} onClick={() => { void retryFailedDrafts(false) }}>{S.lib_btn_retry_failed}</Button>
          <Button type="primary" loading={uploadSaving || uploadInspecting} disabled={uploadLocked || retryableFailedUploadDrafts.length === 0} onClick={() => { void retryFailedDrafts(true) }}>{S.lib_btn_retry_and_convert}</Button>
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

        {filteredUploadDrafts.length > 0 ? (
          <div className="kb-lib-upload-draft-list">
            <div className="kb-lib-upload-draft-list-body" role="list">
              {pagedUploadDrafts.map((x) => {
                const reasonKey = x.status === 'error'
                  ? classifyFailedReason(x.note) as Exclude<UploadErrorReason, 'all'>
                  : null
                return (
                  <div key={x.key} className="kb-lib-upload-draft-item" role="listitem">
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
                        <Button size="small" disabled={uploadLocked || x.status === 'saving' || x.status === 'inspecting'} onClick={() => { inspectSingleDraft(x.key) }}>{S.lib_btn_scan}</Button>
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
                  </div>
                )
              })}
            </div>
            {filteredUploadDrafts.length > UPLOAD_DRAFT_PAGE_SIZE ? (
              <Pagination
                className="kb-lib-list-pagination"
                size="small"
                current={uploadDraftPage}
                pageSize={UPLOAD_DRAFT_PAGE_SIZE}
                total={filteredUploadDrafts.length}
                showSizeChanger={false}
                onChange={setUploadDraftPage}
              />
            ) : null}
          </div>
        ) : (
          <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={S.lib_upload_empty} />
        )}
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

      <WorkbenchMetricStrip items={workbenchStats} className="kb-lib-summary-strip" />

      {preparationWorkbench}
      {uploadWorkbenchCard}

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
                    {refSyncDisplayMessage}
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
              data-testid="library-convert-scope"
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

      {showRefSyncCard && store.refSync ? (
        <WorkbenchPanel className="kb-lib-refsync-card">
          <div className="kb-lib-refsync-shell">
            <div className="kb-lib-refsync-head">
              <div className="kb-lib-refsync-copy">
                <Text className="kb-lib-refsync-title">{S.lib_card_refsync}</Text>
                <Text type="secondary" className="kb-lib-refsync-hint">
                  {refSyncDisplayMessage}
                </Text>
                <Text type="secondary" className="kb-lib-refsync-meta">
                  {S.lib_refsync_hint
                    .replace('{docsDone}', String(store.refSync.docsDone))
                    .replace('{docsTotal}', String(store.refSync.docsTotal ?? numericStat(refSyncStats, 'docs_total')))
                    .replace('{refsTotal}', String(numericStat(refSyncStats, 'refs_total')))}
                </Text>
              </div>
              <WorkbenchStatusPill tone={refSyncStatusTone}>{refSyncStatusLabel}</WorkbenchStatusPill>
            </div>
            {store.refSync.docsTotal > 0 ? (
              <Progress
                percent={refSyncPercent}
                status={store.refSync.running ? 'active' : (store.refSync.status === 'error' ? 'exception' : 'normal')}
              />
            ) : null}
            <WorkbenchMetricStrip items={refSyncMetricItems} className="kb-lib-refsync-metrics" />
            <WorkbenchMetricStrip items={refSyncQueueItems} className="kb-lib-refsync-queues" />
            {store.refSync.error ? <Text type="danger" className="text-xs">{store.refSync.error}</Text> : null}
          </div>
        </WorkbenchPanel>
      ) : null}

      {QUALITY_DIAGNOSTICS_VISIBLE && (qualityReportStats.converted > 0 || qualityReportStats.assessed > 0) ? (
        <LibraryQualityCenter
          open={qualityCenterOpen}
          tone={qualityCenterTone}
          S={S}
          stats={qualityReportStats}
          statusLabel={qualityCenterStatusLabel}
          nextAction={qualityCenterNextAction}
          summary={qualityCenterSummary}
          signals={qualityCenterSignals}
          recommendedRepairCount={qualityRepairRecommendedNames.length}
          recommendedRepairBusy={qualityRepairRecommendedNames.some((name) => Boolean(qualityRepairingNames[name]))}
          onToggleOpen={() => setQualityCenterOpen((value) => !value)}
          onFocusReview={handleFocusQualityReview}
          onRepairRecommended={() => {
            setQualityCenterOpen(true)
            void handleRepairRecommendedQuality()
          }}
        >
          {qualityCenterOpen ? (
            <div className="kb-lib-quality-center-details" data-testid="library-quality-center-details">
              <LibraryQualityOverviewPanels
                S={S}
                scanRunning={qualityBatchRunning}
                repairDisabled={qualityReportStats.review <= 0 && qualitySourceReadinessStats.blocked <= 0}
                reportStats={qualityReportStats}
                sourceReadinessStats={qualitySourceReadinessStats}
                onScanSource={() => { void runConversionQualityBatch(false) }}
                onSafeRepairAll={() => { void runConversionQualityBatch(true) }}
                onFocusReview={handleFocusQualityReview}
              />
              <LibraryQualityStatusPanels
                S={S}
                batchResult={qualityBatchResult}
                repairImpact={qualityRepairImpact}
                repairRun={qualityRepairRun}
                repairAdvancing={qualityRepairAdvancing}
                domains={qualityDomainViews}
                reviewCount={qualityReportStats.review}
                readerLocateRepairCount={qualityReaderLocateRecommendedSources.length}
                artifactOpening={qualityArtifactOpening}
                onFocusReview={handleFocusQualityReview}
                onRepairReaderLocateSources={() => { void repairReaderLocateSources() }}
                onAdvanceRepairRun={() => { void handleAdvanceQualityRepairRun() }}
                onOpenArtifact={(domain, target) => { void openQualityArtifact(domain, target) }}
              />
              <LibraryQualityMetadataBackfillPanel
                S={S}
                state={shelfMetadataBackfillState}
                scan={shelfMetadataBackfillScan}
                result={shelfMetadataBackfillResult}
                tone={shelfMetadataBackfillTone}
                running={shelfMetadataBackfillRunning}
                phase={shelfMetadataBackfillPhase}
                progress={shelfMetadataBackfillProgress}
                refreshing={shelfMetadataBackfillRefreshing}
                onStart={() => { void startShelfMetadataBackfill({ silent: false }) }}
                onRefresh={() => { void refreshShelfMetadataBackfillState(false) }}
              />
              <LibraryQualityFigureAssetsPanel
                S={S}
                scan={figureAssetScan}
                scanRunning={figureAssetScanRunning}
                refreshResult={figureAssetRefreshResult}
                refreshRunning={figureAssetRefreshRunning}
                onScan={(includeAll) => { void runFigureAssetQualityScan(includeAll) }}
                onRefresh={() => { void refreshFigureAssets() }}
              />
              <LibraryQualityChainPanels
                S={S}
                featureHealth={qualityFeatureHealth}
                featureItems={qualityFeatureHealthItems}
                fullChain={qualityFullChain}
                fullChainStages={qualityFullChainStages}
                fullChainRootCauses={qualityFullChainRootCauses}
                fullChainActionHistory={qualityFullChainActionHistory}
                actionKey={qualityFullChainActionKey}
                liveResults={qualityFullChainResults}
                persistedResults={qualityFullChainPersistedResults}
                onFeatureAction={(item) => { void handleQualityFeatureHealthAction(item) }}
                onStageAction={(stage) => { void handleQualityFullChainStage(stage) }}
                onHistoryOpen={(item) => { void handleQualityActionHistoryOpen(item) }}
              />
              <LibraryQualityIssuePanels
                S={S}
                priorityActions={qualityPriorityActions}
                rerunSummary={qualityRerunSummary}
                failureCases={qualityFailureCases}
                failureFilters={qualityFailureFilters}
                failureFilter={qualityFailureFilter}
                visibleFailureCases={visibleQualityFailureCases}
                artifactOpening={qualityArtifactOpening}
                caseActionKey={qualityCaseActionKey}
                caseRerunResults={qualityCaseRerunResults}
                onPriorityAction={(action) => { void handleQualityPriorityAction(action) }}
                onOpenFailureReport={() => { void openQualityArtifact('research_qa', 'report') }}
                onFailureFilterChange={setQualityFailureFilter}
                onOpenReplayCase={openResearchQaReplayCase}
                onFailureAction={(item, action) => { void handleQualityFailureAction(item, action) }}
                onCopyFailureSummary={(item) => { void copyQualityFailureSummary(item) }}
              />
              <LibraryQualityReportPanels
                S={S}
                issues={qualityIssueStats}
                recommendations={qualityReportRecommendations}
                onFocusIssue={handleFocusQualityIssue}
                onFocusRecommendation={(name) => focusQualityHistoryNames([name])}
              />
            </div>
          ) : null}
        </LibraryQualityCenter>
      ) : null}

      <LibraryQualityHistoryPanel
        visible={QUALITY_DIAGNOSTICS_VISIBLE && qualityCenterOpen}
        S={S}
        records={qualityRepairHistoryList}
        stats={qualityRepairHistoryStats}
        remainingNames={qualityHistoryRemainingNames}
        recommendedNames={qualityRepairRecommendedNames}
        repairingNames={qualityRepairingNames}
        focusNames={qualityHistoryFocusNames}
        onFocusRemaining={handleFocusQualityHistoryRemaining}
        onRepairRecommended={() => { void handleRepairRecommendedQuality() }}
        onClearFocus={() => setQualityHistoryFocusNames([])}
        onOpenRecord={(name) => focusQualityHistoryNames([name])}
      />

      <LibraryTaxonomyToolbar
        S={S}
        browseMode={browseMode}
        visibleCount={visibleAll.length}
        totalCount={store.files.length}
        hasActiveFilters={hasActiveTaxonomyFilters}
        activeFilterCount={activeTaxonomyFilterCount}
        canSelectCurrent={showTaxonomySelectAction}
        canRefreshSuggestions={showTaxonomyRefreshAction}
        canClearFilters={showTaxonomyClearAction}
        suggestionsRefreshing={suggestionsRefreshing}
        fileKeyword={fileKeyword}
        paperCategoryFilter={paperCategoryFilter}
        paperCategoryOptions={paperCategoryFilterOptions}
        paperTagFilter={paperTagFilter}
        paperTagOptions={paperTagFilterOptions}
        readingStatusFilter={readingStatusFilter}
        readingStatusOptions={READING_STATUS_OPTIONS(S).filter((item) => item.value)}
        onlyUnread={onlyUnread}
        onlyUnclassified={onlyUnclassified}
        onlySuggested={onlySuggested}
        diagnosticsVisible={QUALITY_DIAGNOSTICS_VISIBLE}
        onlyQualityIssues={onlyQualityIssues}
        qualityReviewCount={qualityReviewCount}
        qualityHistoryFocusCount={qualityHistoryFocusNames.length}
        onBrowseModeChange={setBrowseMode}
        onSelectCurrentList={selectCurrentListItems}
        onRefreshSuggestions={() => { void regenerateSuggestionsForVisible() }}
        onClearFilters={clearTaxonomyFilters}
        onFileKeywordChange={setFileKeyword}
        onPaperCategoryFilterChange={applyPaperCategoryFilter}
        onPaperTagFilterChange={applyPaperTagFilter}
        onReadingStatusFilterChange={(value) => setReadingStatusFilter(value as ReadingStatusValue)}
        onToggleOnlyUnread={() => setOnlyUnread((value) => !value)}
        onToggleOnlyUnclassified={() => {
          const next = !onlyUnclassified
          setOnlyUnclassified(next)
          if (next) setPaperCategoryFilter('')
        }}
        onToggleOnlySuggested={() => setOnlySuggested((value) => !value)}
        onToggleOnlyQualityIssues={() => setOnlyQualityIssues((value) => !value)}
        onClearQualityHistoryFocus={() => setQualityHistoryFocusNames([])}
      />

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
              {QUALITY_DIAGNOSTICS_VISIBLE && selectedQualityReviewNames.length > 0 ? (
                <Button
                  icon={<ReloadOutlined />}
                  loading={selectedQualityReviewNames.some((name) => Boolean(qualityRepairingNames[name]))}
                  onClick={() => { void handleRepairSelectedQuality() }}
                  data-testid="library-quality-repair-selected"
                >
                  {S.lib_btn_repair_quality_selected}
                </Button>
              ) : null}
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
          <LibraryCategoryCards
            S={S}
            cards={categoryCards}
            onlyUnclassified={onlyUnclassified}
            paperCategoryFilter={paperCategoryFilter}
            onSelectCategory={(card) => {
              if (card.key === 'category:__unclassified__') {
                setPaperCategoryFilter('')
                setOnlyUnclassified(true)
              } else {
                applyPaperCategoryFilter(card.label)
              }
              setBrowseMode('list')
            }}
          />
        </Card>
      ) : (
        <Card size="small" className="kb-lib-card">
          <LibraryTagCards
            S={S}
            cards={tagCards}
            paperTagFilter={paperTagFilter}
            onSelectTag={(card) => {
              applyPaperTagFilter(card.label)
              setBrowseMode('list')
            }}
          />
        </Card>
      )}

      <LibraryMetadataDrawer
        open={metaDrawerOpen}
        item={metaItem}
        draft={metaDraft}
        draftCategory={metaDraftCategory}
        draftTags={metaDraftTags}
        suggestionCount={metaSuggestionCount}
        saving={metaSaving}
        suggestionSaving={metaSuggestionSaving}
        S={S}
        paperCategoryOptions={paperCategoryOptions}
        paperTagOptions={paperTagOptions}
        readingStatusOptions={READING_STATUS_OPTIONS(S).filter((item) => item.value)}
        tagInputSeparators={TAG_INPUT_SEPARATORS}
        onClose={() => setMetaDrawerOpen(false)}
        onDraftChange={setMetaDraft}
        onSave={() => { void saveMetaEditor() }}
        onRegenerateSuggestions={() => { void regenerateMetaSuggestions() }}
        onApplySuggestionAction={(body) => { void applyMetaSuggestionAction(body) }}
        readingStatusLabel={readingStatusLabel}
      />

      <Drawer
        title={S.lib_batch_edit_count_format.replace('{n}', String(selectedLibraryCount))}
        open={batchDrawerOpen}
        size={420}
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
              {batchDraft.apply_paper_category ? (
                batchDraftWillClearCategory ? (
                  <Tag color="warning">{S.lib_batch_clear_category_label}</Tag>
                ) : (
                  <Tag color="processing">{S.lib_batch_set_category_label.replace('{category}', batchDraftCategory)}</Tag>
                )
              ) : null}
              {batchDraft.apply_reading_status ? (
                batchDraftWillClearStatus ? (
                  <Tag color="warning">{S.lib_batch_clear_status_label}</Tag>
                ) : (
                  <Tag color="gold">{S.lib_batch_set_status_label.replace('{status}', batchDraftReadingLabel)}</Tag>
                )
              ) : null}
              {batchDraftAddTags.length ? (
                <Tag color="green">{S.lib_batch_add_tag_count.replace('{n}', String(batchDraftAddTags.length))}</Tag>
              ) : null}
              {batchDraftRemoveTags.length ? (
                <Tag color="red">{S.lib_batch_remove_tag_count.replace('{n}', String(batchDraftRemoveTags.length))}</Tag>
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
