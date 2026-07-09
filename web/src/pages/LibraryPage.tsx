
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Button,
  message,
  Typography,
  Tabs,
  Space,
  Card,
  Modal,
} from 'antd'
import {
  ReloadOutlined,
} from '@ant-design/icons'
import type {
  ConversionQualitySummary,
  LibraryFileItem,
  LibraryFigureAssetRefreshResponse,
  LibraryFigureAssetScanResponse,
  LibraryQualityActionHistoryItem,
  LibraryConversionQualityBatchResponse,
  LibraryQualityFeatureHealthItem,
  LibraryQualityFailureCase,
  LibraryQualityFullChainStage,
  LibraryQualityOverviewResponse,
  LibraryQualityPriorityAction,
  LibraryQualityRepairAction,
  LibraryQualityRepairImpact,
  LibraryQualityRepairRun,
  LibraryResearchQaRerunResponse,
} from '../api/library'
import { libraryApi } from '../api/library'
import { referencesApi, type ReferenceSyncStats, type ShelfMetadataBackfillJobState } from '../api/references'
import { useChatStore } from '../stores/chatStore'
import { useLibraryStore } from '../stores/libraryStore'
import { useSettingsStore } from '../stores/settingsStore'
import { useNavigate } from 'react-router-dom'
import { useT } from '../i18n'
import {
  WorkbenchMetricStrip,
  type WorkbenchMetricItem,
  type WorkbenchTone,
} from '../components/library/WorkbenchPrimitives'
import { LibraryBatchMetadataDrawer } from './library/LibraryBatchMetadataDrawer'
import { LibraryBatchSelectionBar } from './library/LibraryBatchSelectionBar'
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
import { LibraryUploadIntake } from './library/LibraryUploadIntake'
import { LibraryUploadDraftWorkbench } from './library/LibraryUploadDraftWorkbench'
import { LibraryProcessControls } from './library/LibraryProcessControls'
import { LibraryRenameWorkbench } from './library/LibraryRenameWorkbench'
import { LibraryLegacyConvertCard } from './library/LibraryLegacyConvertCard'
import { LibraryRefSyncCard } from './library/LibraryRefSyncCard'
import { LibraryStickyStatus } from './library/LibraryStickyStatus'
import { LibraryFileRow } from './library/LibraryFileRow'
import { LibraryFileList } from './library/LibraryFileList'
import {
  LibraryCategoryCards,
  LibraryTagCards,
} from './library/LibraryTaxonomyViews'
import { LibraryTaxonomyToolbar } from './library/LibraryTaxonomyToolbar'
import { useLibraryQualityCenterViewModel } from './library/useLibraryQualityCenterViewModel'
import {
  useLibraryQualityChainViewModel,
  type QualityFullChainActionResult,
} from './library/useLibraryQualityChainViewModel'
import { useLibraryQualityActionRecorder } from './library/useLibraryQualityActionRecorder'
import { useLibraryQualityDomainViews } from './library/useLibraryQualityDomainViews'
import { useLibraryQualityFailureCases } from './library/useLibraryQualityFailureCases'
import {
  useLibraryQualityOperationGuard,
  type LibraryQualityOperationToken,
} from './library/useLibraryQualityOperationGuard'
import { useLibraryQualityReportMetrics } from './library/useLibraryQualityReportMetrics'
import { useShelfMetadataBackfillViewModel } from './library/useShelfMetadataBackfillViewModel'
import { useLibraryDirectoryActions } from './library/useLibraryDirectoryActions'
import { useLibraryUploadDraftActions } from './library/useLibraryUploadDraftActions'
import { useLibraryRenameActions } from './library/useLibraryRenameActions'
import { useLibraryBatchMetadataActions } from './library/useLibraryBatchMetadataActions'
import { useLibraryMetadataActions } from './library/useLibraryMetadataActions'
import { useLibrarySuggestionRefreshActions } from './library/useLibrarySuggestionRefreshActions'
import {
  useLibraryFileFilters,
  type ReadingStatusValue,
} from './library/useLibraryFileFilters'
import { dispatchOpenSettings } from '../components/layout/settingsEvents'
import { qualityDiagnosticsVisible, qualityStatusVisible } from '../utils/qualityDiagnostics'
import {
  buildQualityRepairHistoryRecord,
  conversionQualityStatus,
  conversionSourceReadiness,
  derivePageProgress,
  formatSeconds,
  hasConversionQualityIssue,
  loadQualityRepairHistory,
  normalizeQualityRepairHistory,
  normalizeTextList,
  normalizeTextValue,
  numericStat,
  qualityFailureCaseMatchesStage,
  qualityOverviewStageSnapshot,
  qualityVerificationFromRerun,
  saveQualityRepairHistory,
  saveResearchQaReplayFailureCase,
  stripKnownSourceExt,
  summarizeConversionQualityRepair,
  type QualityRepairHistoryRecord,
} from './library/libraryPageUtils'

const { Text } = Typography
const RENAME_PAGE_SIZE = 6
const UPLOAD_DRAFT_PAGE_SIZE = 8
const EMPTY_REF_SYNC_STATS: ReferenceSyncStats = {}
const INTERNAL_ROUTES_ENABLED = import.meta.env.VITE_ENABLE_INTERNAL_ROUTES === '1'
const QUALITY_DIAGNOSTICS_VISIBLE = qualityDiagnosticsVisible()
const QUALITY_STATUS_VISIBLE = qualityStatusVisible()

type FileTabKey = 'pending' | 'converted' | 'all'
type LibraryBrowseMode = 'list' | 'categories' | 'tags'

const CONVERT_MODE = 'balanced'
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

type QualityRepairBaseline = {
  quality: ConversionQualitySummary | null
  startedAt: number
}

type QualityRepairRunOptions = {
  autoReindexImmediate?: boolean
  autoReindexQueued?: boolean
  operationToken?: LibraryQualityOperationToken
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


export default function LibraryPage() {
  const S = useT()
  const store = useLibraryStore()
  const createPaperGuideConversation = useChatStore((s) => s.createPaperGuideConversation)
  const nav = useNavigate()

  const settingsLoaded = useSettingsStore((s) => s.loaded)
  const hasTextApiKey = useSettingsStore((s) => s.hasTextApiKey)
  const llmReadiness = useSettingsStore((s) => s.llmReadiness)

  const [scope, setScope] = useState('200')
  const [tabKey, setTabKey] = useState<FileTabKey>('all')
  const [browseMode, setBrowseMode] = useState<LibraryBrowseMode>('list')
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
  const {
    directoriesConfigured,
    dirDirty,
    dirEditorOpen,
    ensureDirsReady,
    mdDirDraft,
    openFolder,
    pdfDirDraft,
    pickDir,
    pickingDir,
    saveDirs,
    savingDirs,
    toggleDirEditor,
    updateMdDirDraft,
    updatePdfDirDraft,
  } = useLibraryDirectoryActions({ S, scope })

  const uploadLocked = store.converting || Boolean(store.refSync?.running)
  const textModelReady = !settingsLoaded
    || (hasTextApiKey && llmReadiness?.providers.text?.severity !== 'error')
  const openApiSettings = useCallback(() => {
    dispatchOpenSettings('text')
  }, [])
  const warnLlmFallback = useCallback((action: string) => {
    message.warning(S.lib_llm_unavailable_fallback.replace('{action}', action))
    openApiSettings()
  }, [S.lib_llm_unavailable_fallback, openApiSettings])
  const {
    addDrafts,
    applyUploadFilter,
    clearSavedDrafts,
    duplicateFailedDrafts,
    failedReasonBuckets,
    failedUploadDrafts,
    failedUploadNotes,
    filteredUploadDrafts,
    inspectSelectedDrafts,
    inspectSingleDraft,
    invertUploadDraftSelection,
    pagedUploadDrafts,
    retryFailedDrafts,
    retryableFailedUploadDrafts,
    saveDraft,
    saveSelectedDrafts,
    selectAllUploadDrafts,
    selectFailedDrafts,
    selectFailedReason,
    selectedUploadCount,
    setDraftSelected,
    setDraftStem,
    setUploadDraftPage,
    setUploadErrorReason,
    setUploadUseLlm,
    setUploadWorkbenchOpen,
    showDuplicateFailedDrafts,
    uploadDraftFilter,
    uploadDraftFilterOptions,
    uploadDraftPage,
    uploadDrafts,
    uploadErrorReason,
    uploadInspecting,
    uploadSaving,
    uploadUseLlm,
    uploadWorkbenchOpen,
  } = useLibraryUploadDraftActions({
    S,
    convertMode: CONVERT_MODE,
    dirDirty,
    ensureDirsReady,
    pageSize: UPLOAD_DRAFT_PAGE_SIZE,
    scope,
    textModelReady,
    uploadLocked,
    warnLlmFallback,
  })
  const {
    applyRenameSuggestions,
    clearRenameSelection,
    pagedRenameVisible,
    renameApplying,
    renameItems,
    renameLoading,
    renameOverrides,
    renamePage,
    renameResultsOpen,
    renameScope,
    renameSelected,
    renameVisible,
    scanRenameSuggestions,
    selectRenameDiffItems,
    selectedRenameCount,
    setRenameItemSelected,
    setRenameOverride,
    setRenamePage,
    setRenameScope,
    toggleRenameResultsOpen,
  } = useLibraryRenameActions({
    S,
    pageSize: RENAME_PAGE_SIZE,
    scope,
    textModelReady,
    warnLlmFallback,
  })
  const resetQualityOperationUi = useCallback(() => {
    setQualityCaseActionKey('')
    setQualityFullChainActionKey('')
    setQualityBatchRunning(false)
    setShelfMetadataBackfillRefreshing(false)
    setQualityRepairAdvancing(false)
  }, [])
  const {
    beginQualityOperation,
    qualityOperationIsCurrent,
    qualityOperationIsActive,
    clearQualityOperation,
  } = useLibraryQualityOperationGuard({
    scope,
    onBegin: resetQualityOperationUi,
  })

  const pendingFiles = useMemo(() => store.files.filter((x) => x.category === 'pending'), [store.files])
  const convertedFiles = useMemo(() => store.files.filter((x) => x.category === 'converted'), [store.files])
  const {
    activeTaxonomyFilterCount,
    applyPaperCategoryFilter,
    applyPaperTagFilter,
    categoryCards,
    clearTaxonomyFilters,
    currentListItems,
    fileKeyword,
    hasActiveTaxonomyFilters,
    onlyQualityIssues,
    onlySuggested,
    onlyUnclassified,
    onlyUnread,
    paperCategoryFilter,
    paperCategoryFilterOptions,
    paperCategoryOptions,
    paperTagFilter,
    paperTagFilterOptions,
    paperTagOptions,
    qualityHistoryFocusNames,
    readingStatusFilter,
    selectUnclassifiedCategory,
    setFileKeyword,
    setOnlyQualityIssues,
    setOnlySuggested,
    setOnlyUnread,
    setPaperCategoryFilter,
    setPaperTagFilter,
    setQualityHistoryFocusNames,
    setReadingStatusFilter,
    tagCards,
    toggleOnlyUnclassified,
    visibleAll,
    visibleConverted,
    visiblePending,
  } = useLibraryFileFilters({
    S,
    files: store.files,
    pendingFiles,
    convertedFiles,
    tabKey,
  })
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
  const {
    qualityReportStats,
    qualityIssueStats,
    qualityReportRecommendations,
  } = useLibraryQualityReportMetrics({
    files: store.files,
    backendQualityOverview,
    qualityRepairHistory,
  })
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
  const qualityHistoryRemainingNames = useMemo(() => {
    const availableNames = new Set(store.files.map((item) => item.name))
    return qualityRepairHistoryList
      .filter((record) => record.remainingIssues.length > 0 && availableNames.has(record.name))
      .map((record) => record.name)
  }, [qualityRepairHistoryList, store.files])
  const qualityDomainViews = useLibraryQualityDomainViews({
    S,
    backendQualityOverview,
    qualityReportStats,
  })
  const {
    qualityPriorityActions,
    actionableQualityPriorityActions,
    qualityFullChain,
    qualityFullChainStages,
    qualityFullChainRootCauses,
    qualityFullChainActionHistory,
    qualityFullChainPersistedResults,
    qualityReaderLocateRecommendedSources,
    qualityFeatureHealth,
    qualityFeatureHealthItems,
  } = useLibraryQualityChainViewModel({
    backendQualityOverview,
  })
  const {
    shelfMetadataBackfillScan,
    shelfMetadataBackfillResult,
    shelfMetadataBackfillProgress,
    shelfMetadataBackfillPhase,
    shelfMetadataBackfillRunning,
    shelfMetadataBackfillTone,
  } = useShelfMetadataBackfillViewModel({
    shelfMetadataBackfillState,
  })
  const qualityRerunSummary = backendQualityOverview?.rerun_summary
  const {
    qualityFailureCases,
    qualityFailureFilters,
    visibleQualityFailureCases,
  } = useLibraryQualityFailureCases({
    backendQualityOverview,
    qualityFailureFilter,
  })
  const qualityRepairRecommendedNames = useMemo(
    () => qualityReportRecommendations.map((item) => item.name).filter(Boolean),
    [qualityReportRecommendations],
  )
  const qualityCenterView = useLibraryQualityCenterViewModel({
    S,
    reportStats: qualityReportStats,
    sourceReadinessStats: qualitySourceReadinessStats,
    domains: qualityDomainViews,
    failureCount: qualityFailureCases.length,
    metadataRemaining: Number(shelfMetadataBackfillScan?.needs_repair || 0),
    priorityActionCount: actionableQualityPriorityActions.length,
    recommendedRepairCount: qualityRepairRecommendedNames.length,
    batchRunning: qualityBatchRunning,
    repairAdvancing: qualityRepairAdvancing,
    metadataBackfillRunning: shelfMetadataBackfillRunning,
    repairingNames: qualityRepairingNames,
  })
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
  const refSyncMetaText = useMemo(() => {
    if (!store.refSync) return ''
    return S.lib_refsync_hint
      .replace('{docsDone}', String(store.refSync.docsDone))
      .replace('{docsTotal}', String(store.refSync.docsTotal ?? numericStat(refSyncStats, 'docs_total')))
      .replace('{refsTotal}', String(numericStat(refSyncStats, 'refs_total')))
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

  const {
    batchDraft,
    batchDrawerOpen,
    batchSaving,
    clearLibrarySelection,
    closeBatchEditor,
    openBatchEditor,
    saveBatchEditor,
    selectCurrentListItems,
    selectedLibraryCount,
    selectedLibraryNames,
    selectedQualityReviewNames,
    setBatchDraft,
    toggleLibrarySelection,
  } = useLibraryBatchMetadataActions({
    S,
    currentListItems,
  })
  const {
    applyMetaSuggestionAction,
    closeMetaEditor,
    metaDraft,
    metaDraftCategory,
    metaDraftTags,
    metaDrawerOpen,
    metaItem,
    metaSaving,
    metaSuggestionCount,
    metaSuggestionSaving,
    openMetaEditor,
    regenerateMetaSuggestions,
    saveMetaEditor,
    setMetaDraft,
  } = useLibraryMetadataActions({ S })
  const {
    refreshSuggestionsForVisible,
    suggestionsRefreshing,
  } = useLibrarySuggestionRefreshActions({
    S,
    items: visibleAll,
  })

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

  const { recordQualityFullChainResult } = useLibraryQualityActionRecorder({
    setQualityFullChainResults,
  })

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

  const showUploadWorkbench = uploadWorkbenchOpen && uploadDrafts.length > 0
  const showTaxonomySelectAction = browseMode === 'list' && currentListItems.length > 0
  const showTaxonomyRefreshAction = browseMode === 'list' && visibleAll.length > 0
  const showTaxonomyClearAction = hasActiveTaxonomyFilters
  const renameWorkbenchSection = (
    <LibraryRenameWorkbench
      S={S}
      renameScope={renameScope}
      renameItems={renameItems}
      renameVisible={renameVisible}
      pagedRenameVisible={pagedRenameVisible}
      renameSelected={renameSelected}
      renameOverrides={renameOverrides}
      renameResultsOpen={renameResultsOpen}
      renameLoading={renameLoading}
      renameApplying={renameApplying}
      renamePage={renamePage}
      renamePageSize={RENAME_PAGE_SIZE}
      selectedRenameCount={selectedRenameCount}
      onRenameScopeChange={setRenameScope}
      onScanRenameSuggestions={scanRenameSuggestions}
      onToggleResultsOpen={toggleRenameResultsOpen}
      onSelectDiffItems={selectRenameDiffItems}
      onClearSelection={clearRenameSelection}
      onApplyRenameSuggestions={applyRenameSuggestions}
      onSelectedChange={setRenameItemSelected}
      onOverrideChange={setRenameOverride}
      onPageChange={setRenamePage}
    />
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
            onToggleEditor={toggleDirEditor}
            onPdfDirChange={updatePdfDirDraft}
            onMdDirChange={updateMdDirDraft}
            onPickDir={pickDir}
            onOpenFolder={openFolder}
            onSaveDirs={saveDirs}
          />

          {renameWorkbenchSection}
        </div>

        <div className="kb-lib-workbench-side">
          <LibraryUploadIntake
            S={S}
            uploadLocked={uploadLocked}
            uploadDraftCount={uploadDrafts.length}
            showUploadWorkbench={showUploadWorkbench}
            lockedMessage={store.converting ? S.lib_upload_locked_converting : S.lib_upload_locked_refsync}
            onAddDrafts={addDrafts}
            onToggleWorkbench={() => setUploadWorkbenchOpen((open) => !open)}
          />

          <LibraryProcessControls
            S={S}
            scope={scope}
            converting={store.converting}
            onScopeChange={(value) => {
              setScope(value)
              void store.loadFiles(value)
            }}
            onConvertPending={handleConvertPending}
            onRefresh={() => store.loadFiles(scope)}
            onStopConvert={store.cancelConvert}
          />
        </div>
      </div>
    </Card>
  )

  const uploadWorkbenchCard = showUploadWorkbench ? (
    <LibraryUploadDraftWorkbench
      S={S}
      uploadDrafts={uploadDrafts}
      filteredUploadDrafts={filteredUploadDrafts}
      pagedUploadDrafts={pagedUploadDrafts}
      selectedUploadCount={selectedUploadCount}
      uploadDraftFilter={uploadDraftFilter}
      uploadErrorReason={uploadErrorReason}
      uploadDraftFilterOptions={uploadDraftFilterOptions}
      uploadUseLlm={uploadUseLlm}
      uploadInspecting={uploadInspecting}
      uploadSaving={uploadSaving}
      uploadLocked={uploadLocked}
      failedUploadDraftCount={failedUploadDrafts.length}
      failedUploadNotes={failedUploadNotes}
      failedReasonBuckets={failedReasonBuckets}
      duplicateFailedDraftCount={duplicateFailedDrafts.length}
      retryableFailedUploadDraftCount={retryableFailedUploadDrafts.length}
      uploadDraftPage={uploadDraftPage}
      uploadDraftPageSize={UPLOAD_DRAFT_PAGE_SIZE}
      onCollapse={() => setUploadWorkbenchOpen(false)}
      onUploadUseLlmChange={setUploadUseLlm}
      onFilterChange={applyUploadFilter}
      onClearErrorReason={() => setUploadErrorReason('all')}
      onSelectAllDrafts={selectAllUploadDrafts}
      onInvertDraftSelection={invertUploadDraftSelection}
      onInspectSelectedDrafts={inspectSelectedDrafts}
      onSaveSelectedDrafts={saveSelectedDrafts}
      onSelectFailedDrafts={selectFailedDrafts}
      onShowDuplicateFailedDrafts={showDuplicateFailedDrafts}
      onRetryFailedDrafts={retryFailedDrafts}
      onClearSavedDrafts={clearSavedDrafts}
      onSelectFailedReason={selectFailedReason}
      onDraftSelectedChange={setDraftSelected}
      onDraftStemChange={setDraftStem}
      onInspectDraft={inspectSingleDraft}
      onSaveDraft={saveDraft}
      onPageChange={setUploadDraftPage}
    />
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

      <LibraryStickyStatus
        convertRunning={store.converting}
        convertProgress={store.progress}
        convertTitle={store.progress ? S.lib_convert_progress.replace('{done}', String(store.progress.completed)).replace('{total}', String(store.progress.total)) : ''}
        convertActiveSummary={convertActiveSummary}
        convertStageLabel={convertStageLabel}
        convertPageLabel={S.lib_convert_page_progress}
        convertPageProgress={convertPageProgress}
        convertPercent={convertPercent}
        convertPagePercent={convertPagePercent}
        stopLabel={S.lib_btn_stop}
        refSyncRunning={Boolean(store.refSync?.running)}
        refSyncTitle={S.lib_refsync_title}
        refSyncMessage={refSyncDisplayMessage}
        refSyncPercent={refSyncPercent}
        refSyncRunningLabel={S.lib_refsync_running}
        onStopConvert={store.cancelConvert}
      />

      <LibraryLegacyConvertCard
        S={S}
        scope={scope}
        fileKeyword={fileKeyword}
        paperCategoryFilter={paperCategoryFilter}
        paperCategoryFilterOptions={paperCategoryFilterOptions}
        paperTagFilter={paperTagFilter}
        paperTagFilterOptions={paperTagFilterOptions}
        readingStatusFilter={readingStatusFilter}
        readingStatusOptions={READING_STATUS_OPTIONS(S).filter((item) => item.value)}
        converting={store.converting}
        onScopeChange={(value) => {
          setScope(value)
          void store.loadFiles(value)
        }}
        onFileKeywordChange={setFileKeyword}
        onPaperCategoryFilterChange={setPaperCategoryFilter}
        onPaperTagFilterChange={setPaperTagFilter}
        onReadingStatusFilterChange={(value) => setReadingStatusFilter(value as ReadingStatusValue)}
        onClearMetadataFilters={() => {
          setPaperCategoryFilter('')
          setPaperTagFilter('')
          setReadingStatusFilter('')
        }}
        onRefresh={() => store.loadFiles(scope)}
        onConvertPending={handleConvertPending}
        onStopConvert={store.cancelConvert}
      />

      {showRefSyncCard && store.refSync ? (
        <LibraryRefSyncCard
          title={S.lib_card_refsync}
          message={refSyncDisplayMessage}
          metaText={refSyncMetaText}
          statusLabel={refSyncStatusLabel}
          statusTone={refSyncStatusTone}
          percent={refSyncPercent}
          docsTotal={store.refSync.docsTotal}
          running={store.refSync.running}
          status={store.refSync.status}
          metricItems={refSyncMetricItems}
          queueItems={refSyncQueueItems}
          error={store.refSync.error}
        />
      ) : null}

      {QUALITY_DIAGNOSTICS_VISIBLE && (qualityReportStats.converted > 0 || qualityReportStats.assessed > 0) ? (
        <LibraryQualityCenter
          open={qualityCenterOpen}
          tone={qualityCenterView.tone}
          S={S}
          stats={qualityReportStats}
          statusLabel={qualityCenterView.statusLabel}
          nextAction={qualityCenterView.nextAction}
          summary={qualityCenterView.summary}
          signals={qualityCenterView.signals}
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
        onRefreshSuggestions={() => { void refreshSuggestionsForVisible() }}
        onClearFilters={clearTaxonomyFilters}
        onFileKeywordChange={setFileKeyword}
        onPaperCategoryFilterChange={applyPaperCategoryFilter}
        onPaperTagFilterChange={applyPaperTagFilter}
        onReadingStatusFilterChange={(value) => setReadingStatusFilter(value as ReadingStatusValue)}
        onToggleOnlyUnread={() => setOnlyUnread((value) => !value)}
        onToggleOnlyUnclassified={toggleOnlyUnclassified}
        onToggleOnlySuggested={() => setOnlySuggested((value) => !value)}
        onToggleOnlyQualityIssues={() => setOnlyQualityIssues((value) => !value)}
        onClearQualityHistoryFocus={() => setQualityHistoryFocusNames([])}
      />

      <LibraryBatchSelectionBar
        visible={browseMode === 'list'}
        S={S}
        selectedCount={selectedLibraryCount}
        currentCount={currentListItems.length}
        qualityDiagnosticsVisible={QUALITY_DIAGNOSTICS_VISIBLE}
        repairableQualityCount={selectedQualityReviewNames.length}
        repairingSelectedQuality={selectedQualityReviewNames.some((name) => Boolean(qualityRepairingNames[name]))}
        onSelectCurrentList={selectCurrentListItems}
        onClearSelection={clearLibrarySelection}
        onRepairSelectedQuality={() => { void handleRepairSelectedQuality() }}
        onOpenBatchEditor={openBatchEditor}
      />

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
                selectUnclassifiedCategory()
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
        onClose={closeMetaEditor}
        onDraftChange={setMetaDraft}
        onSave={() => { void saveMetaEditor() }}
        onRegenerateSuggestions={() => { void regenerateMetaSuggestions() }}
        onApplySuggestionAction={(body) => { void applyMetaSuggestionAction(body) }}
        readingStatusLabel={readingStatusLabel}
      />

      <LibraryBatchMetadataDrawer
        open={batchDrawerOpen}
        selectedCount={selectedLibraryCount}
        draft={batchDraft}
        saving={batchSaving}
        S={S}
        paperCategoryOptions={paperCategoryOptions}
        paperTagOptions={paperTagOptions}
        paperTagFilterOptions={paperTagFilterOptions}
        readingStatusOptions={READING_STATUS_OPTIONS(S).filter((item) => item.value)}
        tagInputSeparators={TAG_INPUT_SEPARATORS}
        onClose={closeBatchEditor}
        onDraftChange={setBatchDraft}
        onSave={() => { void saveBatchEditor() }}
        readingStatusLabel={readingStatusLabel}
      />
    </div>
  )
}
