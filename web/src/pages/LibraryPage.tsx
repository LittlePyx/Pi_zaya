
import { useCallback, useEffect, useMemo, useState } from 'react'
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
  LibraryFileItem,
  LibraryQualityActionHistoryItem,
  LibraryQualityFeatureHealthItem,
  LibraryQualityPriorityAction,
  LibraryQualityRepairImpact,
  LibraryQualityRepairRun,
  LibraryResearchQaRerunResponse,
} from '../api/library'
import { libraryApi } from '../api/library'
import type { ReferenceSyncStats } from '../api/references'
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
import { useLibraryQualityFocusActions } from './library/useLibraryQualityFocusActions'
import {
  useLibraryQualityRepairActions,
} from './library/useLibraryQualityRepairActions'
import { useLibraryQualityMaintenanceActions } from './library/useLibraryQualityMaintenanceActions'
import {
  useShelfMetadataBackfillActions,
} from './library/useShelfMetadataBackfillActions'
import {
  useLibraryQualityFullChainStageRecorder,
} from './library/useLibraryQualityFullChainStageRecorder'
import { useLibraryQualityFullChainActions } from './library/useLibraryQualityFullChainActions'
import { useLibraryQualityFailureCaseActions } from './library/useLibraryQualityFailureCaseActions'
import { dispatchOpenSettings } from '../components/layout/settingsEvents'
import { qualityDiagnosticsVisible, qualityStatusVisible } from '../utils/qualityDiagnostics'
import {
  conversionQualityStatus,
  conversionSourceReadiness,
  conversionStageLabel,
  conversionTaskFraction,
  derivePageProgress,
  formatSeconds,
  hasConversionQualityIssue,
  loadQualityRepairHistory,
  normalizeTextList,
  normalizeTextValue,
  numericStat,
  runningPagesLabel,
  saveResearchQaReplayFailureCase,
  stripKnownSourceExt,
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
  const [qualityRepairImpact, setQualityRepairImpact] = useState<LibraryQualityRepairImpact | null>(null)
  const [qualityBatchRunning, setQualityBatchRunning] = useState(false)
  const [qualityRepairRun, setQualityRepairRun] = useState<LibraryQualityRepairRun | null>(null)
  const [qualityRepairAdvancing, setQualityRepairAdvancing] = useState(false)
  const [qualityRepairHistory, setQualityRepairHistory] = useState<Record<string, QualityRepairHistoryRecord>>(() => loadQualityRepairHistory())
  const [qualityCenterOpen, setQualityCenterOpen] = useState(false)
  const [qualityArtifactOpening, setQualityArtifactOpening] = useState('')
  const [qualityCaseActionKey, setQualityCaseActionKey] = useState('')
  const [qualityFullChainActionKey, setQualityFullChainActionKey] = useState('')
  const [qualityFullChainResults, setQualityFullChainResults] = useState<Record<string, QualityFullChainActionResult>>({})
  const [shelfMetadataBackfillRefreshing, setShelfMetadataBackfillRefreshing] = useState(false)
  const [qualityCaseRerunResults, setQualityCaseRerunResults] = useState<Record<string, LibraryResearchQaRerunResponse>>({})
  const [qualityFailureFilter, setQualityFailureFilter] = useState('')
  const [conversionRetryingName, setConversionRetryingName] = useState('')
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
  const {
    figureAssetRefreshResult,
    figureAssetRefreshRunning,
    figureAssetScan,
    figureAssetScanRunning,
    handleReindex,
    qualityBatchResult,
    refreshFigureAssets,
    runConversionQualityBatch,
    runFigureAssetQualityScan,
  } = useLibraryQualityMaintenanceActions({
    S,
    scope,
    speedMode: CONVERT_MODE,
    beginQualityOperation,
    clearQualityOperation,
    loadFiles: store.loadFiles,
    loadQualityOverview: store.loadQualityOverview,
    qualityOperationIsActive,
    qualityOperationIsCurrent,
    reindex: store.reindex,
    setQualityBatchRunning,
    startProgressStream: store.startProgressStream,
  })
  const {
    refreshShelfMetadataBackfillState,
    shelfMetadataBackfillState,
    startShelfMetadataBackfill,
  } = useShelfMetadataBackfillActions({
    beginQualityOperation,
    clearQualityOperation,
    qualityOperationIsActive,
    qualityOperationIsCurrent,
    setShelfMetadataBackfillRefreshing,
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
  const {
    focusQualityHistoryNames,
    handleFocusQualityHistoryRemaining,
    handleFocusQualityIssue,
    handleFocusQualityReview,
  } = useLibraryQualityFocusActions({
    S,
    files: store.files,
    scope,
    qualityReviewCount,
    qualityHistoryRemainingNames,
    loadFiles: store.loadFiles,
    setBrowseMode,
    setFileKeyword,
    setOnlyQualityIssues,
    setQualityCenterOpen,
    setQualityHistoryFocusNames,
    setScope,
    setTabKey,
  })
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
  const convertPercent = useMemo(() => {
    if (!store.progress || store.progress.total <= 0) return 0
    const tasks = Array.isArray(store.progress.activeTasks) ? store.progress.activeTasks : []
    let activeFraction = 0
    if (tasks.length > 0) {
      for (const task of tasks) {
        const taskProgress = derivePageProgress(task.cur_page_done, task.cur_page_total, task.cur_page_msg)
        activeFraction += conversionTaskFraction(
          task.conversion_stage,
          taskProgress.done,
          taskProgress.total,
        )
      }
    } else {
      const fallback = derivePageProgress(
        store.progress.curPageDone,
        store.progress.curPageTotal,
        store.progress.curPageMsg,
      )
      if (fallback.total > 0) {
        activeFraction = conversionTaskFraction(
          store.progress.conversionStage,
          fallback.done,
          fallback.total,
        )
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
    () => conversionStageLabel(String(store.progress?.conversionStage || ''), S),
    [store.progress, S],
  )
  const convertRunningPagesLabel = useMemo(
    () => runningPagesLabel(
      String(store.progress?.conversionStage || ''),
      store.progress?.runningPages,
      Number(store.progress?.runningPageCount || 0),
      Number(store.progress?.curPageTotal || 0),
      S,
    ),
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
    if (store.refSync.current) {
      return S.lib_refsync_current.replace('{current}', store.refSync.current)
    }
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
    : store.refSync?.status === 'done'
      ? S.run_ok
      : store.refSync?.status === 'error'
        ? S.lib_draft_error
        : S.lib_refsync_idle

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
    handleRepairQualityOne,
    handleRepairRecommendedQuality,
    handleRepairSelectedQuality,
    qualityRepairingNames,
    qualityRepairResults,
  } = useLibraryQualityRepairActions({
    S,
    files: store.files,
    scope,
    speedMode: CONVERT_MODE,
    qualityRepairRecommendedNames,
    selectedQualityReviewNames,
    beginQualityOperation,
    clearQualityOperation,
    handleReindex,
    loadFiles: store.loadFiles,
    qualityOperationIsCurrent,
    repairQuality: store.repairQuality,
    setQualityRepairHistory,
    setQualityRepairImpact,
    setQualityRepairRun,
  })
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
  const {
    applyMetaSuggestionAction,
    closeMetaEditor,
    continueMetaEditing,
    discardMetaEditor,
    metaDraft,
    metaDraftCategory,
    metaDraftTags,
    metaDrawerOpen,
    metaBusy,
    metaCloseConfirmOpen,
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

  const handleCancelConversion = async (item: LibraryFileItem) => {
    const taskId = String(item.task_id || '').trim()
    if (!taskId) {
      message.info(S.lib_msg_conversion_task_not_found)
      await store.loadFiles(scope)
      return
    }
    try {
      const result = await store.cancelConversionTask(taskId)
      if (!result.matched) {
        message.info(S.lib_msg_conversion_task_not_found)
      } else if (result.state === 'queued_removed') {
        message.success(S.lib_msg_conversion_queue_removed.replace('{name}', item.name))
      } else {
        message.success(S.lib_msg_conversion_cancel_requested.replace('{name}', item.name))
      }
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.lib_msg_conversion_cancel_failed)
    }
  }

  const handleRetryConversion = async (item: LibraryFileItem) => {
    if (item.task_state !== 'idle' || conversionRetryingName) return
    const retryAction = String(item.last_conversion?.retry_action || '')
    if (!retryAction) return
    setConversionRetryingName(item.name)
    try {
      if (retryAction === 'reindex') {
        const result = await libraryApi.reindexFile(item.name)
        await store.loadFiles(scope)
        if (result.ok) {
          message.success(S.lib_msg_index_retry_complete.replace('{name}', item.name))
        } else {
          message.error(result.detail || S.lib_msg_index_retry_failed)
        }
        return
      }
      const speedMode = String(item.last_conversion?.speed_mode || CONVERT_MODE)
      await store.convert(item.name, speedMode, true)
      message.success(S.lib_msg_conversion_retry_enqueued.replace('{name}', item.name))
    } catch (err) {
      message.error(
        err instanceof Error
          ? err.message
          : (retryAction === 'reindex' ? S.lib_msg_index_retry_failed : S.lib_msg_conversion_retry_failed),
      )
      await store.loadFiles(scope)
    } finally {
      setConversionRetryingName('')
    }
  }

  const { recordQualityFullChainResult } = useLibraryQualityActionRecorder({
    setQualityFullChainResults,
  })
  const { recordQualityFullChainStageResult } = useLibraryQualityFullChainStageRecorder({
    backendQualityOverview,
    qualityOperationIsCurrent,
    recordQualityFullChainResult,
  })

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

  const {
    applyQualityFailureRepairPlan,
    copyQualityFailureSummary,
    firstQualityCaseForStage,
    handleQualityFailureAction,
    openResearchQaReplayCase,
    repairQualityStageShelfMetadata,
    runQualityFailureCaseRerun,
  } = useLibraryQualityFailureCaseActions({
    S,
    scope,
    speedMode: CONVERT_MODE,
    internalRoutesEnabled: INTERNAL_ROUTES_ENABLED,
    qualityFailureCases,
    nav,
    beginQualityOperation,
    clearQualityOperation,
    qualityOperationIsActive,
    qualityOperationIsCurrent,
    setQualityCaseActionKey,
    setQualityCaseRerunResults,
    setQualityRepairImpact,
    setQualityRepairRun,
    handleReindex,
    waitForLibraryConversionDone,
    repairQuality: store.repairQuality,
    loadFiles: store.loadFiles,
    loadQualityOverview: store.loadQualityOverview,
    openQualityArtifact,
    startShelfMetadataBackfill,
  })

  const { handleQualityFullChainStage } = useLibraryQualityFullChainActions({
    scope,
    qualityRepairRecommendedNames,
    beginQualityOperation,
    clearQualityOperation,
    qualityOperationIsActive,
    qualityOperationIsCurrent,
    setQualityFullChainActionKey,
    setQualityRepairImpact,
    firstQualityCaseForStage,
    handleFocusQualityReview,
    handleReindex,
    handleRepairRecommendedQuality,
    waitForLibraryConversionDone,
    runQualityFailureCaseRerun,
    repairQualityStageShelfMetadata,
    applyQualityFailureRepairPlan,
    openQualityArtifact,
    loadFiles: store.loadFiles,
    loadQualityOverview: store.loadQualityOverview,
    recordQualityFullChainStageResult,
  })

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
        onCancel={(rowItem) => { void handleCancelConversion(rowItem) }}
        onRetry={(rowItem) => { void handleRetryConversion(rowItem) }}
        retrying={conversionRetryingName === item.name}
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
        convertRunningPagesLabel={convertRunningPagesLabel}
        convertPageProgress={convertPageProgress}
        convertPercent={convertPercent}
        convertPagePercent={convertPagePercent}
        stopLabel={S.lib_btn_stop_all}
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
        busy={metaBusy}
        saving={metaSaving}
        suggestionSaving={metaSuggestionSaving}
        closeConfirmOpen={metaCloseConfirmOpen}
        S={S}
        paperCategoryOptions={paperCategoryOptions}
        paperTagOptions={paperTagOptions}
        readingStatusOptions={READING_STATUS_OPTIONS(S).filter((item) => item.value)}
        tagInputSeparators={TAG_INPUT_SEPARATORS}
        onClose={closeMetaEditor}
        onContinueEditing={continueMetaEditing}
        onDiscardAndClose={discardMetaEditor}
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
