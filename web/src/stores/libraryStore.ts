import { create } from 'zustand'
import {
  libraryApi,
  type CancelConversionResponse,
  type ConvertActiveTask,
  type LibraryFilesResponse,
  type LibraryFileItem,
  type LibraryQualityOverviewResponse,
  type LibrarySuggestionActionBody,
  type LibraryQualityRepairBody,
  type LibraryQualityRepairResponse,
  type LibraryMetaBatchUpdateBody,
  type LibraryMetaUpdateBody,
  type LibrarySuggestionRegenerateBody,
  type ResumeAllConversionsResponse,
  type ResumeConversionResponse,
} from '../api/library'
import { referencesApi, type ReferenceSyncStatusEvent, type ReferenceSyncStats } from '../api/references'
import { qualityDiagnosticsVisible } from '../utils/qualityDiagnostics'

interface ConvertProgressState {
  total: number
  completed: number
  current: string
  activeCount: number
  activeTasks: ConvertActiveTask[]
  curPageDone: number
  curPageTotal: number
  curPageMsg: string
  conversionStage: string
  runningPages: number[]
  runningPageCount: number
  last: string
}

interface RefSyncState {
  running: boolean
  status: string
  stage: string
  message: string
  error: string
  current: string
  docsDone: number
  docsTotal: number
  runId: number
  stats: ReferenceSyncStats
}

interface LibraryState {
  pdfs: { name: string; path: string }[]
  files: LibraryFileItem[]
  viewScope: string
  fileCounts: {
    total_view: number
    total_all: number
    pending: number
    converted: number
    queued: number
    running: number
    recoverable?: number
    reconverting: number
    quality_review: number
    quality_ready: number
    index_ready?: number
    index_quality_blocked?: number
    index_stale?: number
  } | null
  qualityOverview: LibraryQualityOverviewResponse | null
  qualityOverviewLoading: boolean
  qualityOverviewError: string
  converting: boolean
  recoverableCount: number
  conversionPersistenceError: string
  pendingRepairReindex: boolean
  pendingRepairRunIds: string[]
  progress: ConvertProgressState | null
  sseController: AbortController | null
  refSync: RefSyncState | null
  refSyncController: AbortController | null
  loadPdfs: () => Promise<void>
  loadFiles: (scope?: string) => Promise<void>
  loadQualityOverview: (scope?: string) => Promise<void>
  upload: (file: File, baseName?: string) => Promise<{ name: string; duplicate?: boolean; existing?: string }>
  convert: (name: string, mode?: string, replace?: boolean) => Promise<void>
  convertPending: (mode?: string, limit?: number) => Promise<{ ok: boolean; enqueued: number; skipped_busy: number; pending_total: number }>
  repairQuality: (
    body: LibraryQualityRepairBody,
    options?: { autoReindexAfterQueued?: boolean },
  ) => Promise<LibraryQualityRepairResponse>
  openFile: (pdfName: string, target?: 'pdf' | 'md' | 'pdf_dir' | 'md_dir') => Promise<void>
  deleteFile: (pdfName: string, alsoMd?: boolean) => Promise<{ ok: boolean; pdf_deleted: boolean; md_deleted: boolean; removed_queued: number; warnings: string[]; needs_reindex: boolean }>
  updatePaperMeta: (body: LibraryMetaUpdateBody) => Promise<LibraryFileItem | null>
  batchUpdatePaperMeta: (body: LibraryMetaBatchUpdateBody) => Promise<number>
  regenerateSuggestions: (body?: LibrarySuggestionRegenerateBody) => Promise<number>
  applySuggestionAction: (body: LibrarySuggestionActionBody) => Promise<LibraryFileItem | null>
  cancelConvert: () => Promise<void>
  cancelConversionTask: (taskId: string) => Promise<CancelConversionResponse>
  resumeConversionTask: (taskId: string) => Promise<ResumeConversionResponse>
  resumeAllConversions: () => Promise<ResumeAllConversionsResponse>
  reindex: () => Promise<{
    ok: boolean
    stdout: string
    stderr: string
    structured_indices: {
      version: number
      scanned: number
      rebuilt: number
      skipped: number
      failed: number
      citation_mention_count: number
      errors: Array<{ path: string; error: string }>
    } | null
    structured_indices_error: string
    refsync: { started?: boolean; reason?: string; run_id?: number } | null
    refsync_error: string
  }>
  startReferenceSync: () => Promise<{ started: boolean; reason?: string; run_id?: number }>
  startProgressStream: () => void
  stopProgressStream: () => void
  startRefSyncStream: () => void
  stopRefSyncStream: () => void
}

function normalizeRefSyncStats(value: unknown): ReferenceSyncStats {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {}
  const out: ReferenceSyncStats = {}
  for (const [key, raw] of Object.entries(value as Record<string, unknown>)) {
    if (
      raw === null
      || raw === undefined
      || typeof raw === 'number'
      || typeof raw === 'string'
      || typeof raw === 'boolean'
    ) {
      out[key] = raw
    }
  }
  return out
}

const REF_SYNC_TOP_LEVEL_STAT_KEYS = [
  'docs_total',
  'docs_indexed',
  'refs_total',
  'refs_with_doi',
  'refs_with_title',
  'refs_with_authors',
  'refs_with_venue',
  'refs_metadata_ready',
  'refs_metadata_user_ready',
  'refs_missing_doi',
  'refs_missing_title',
  'refs_missing_authors',
  'refs_missing_venue',
  'refs_unresolved',
  'refs_crossref_ok',
  'refs_source_map_ok',
] as const

function normalizeRefSyncEventStats(data: ReferenceSyncStatusEvent): ReferenceSyncStats {
  const out = normalizeRefSyncStats(data.stats)
  for (const key of REF_SYNC_TOP_LEVEL_STAT_KEYS) {
    const raw = data[key]
    if (
      raw === null
      || raw === undefined
      || typeof raw === 'number'
      || typeof raw === 'string'
      || typeof raw === 'boolean'
    ) {
      out[key] = raw
    }
  }
  if (out.docs_indexed === undefined && data.docs_done !== undefined) out.docs_indexed = data.docs_done
  return out
}

function numberValue(value: unknown): number {
  const n = Number(value || 0)
  return Number.isFinite(n) ? n : 0
}

function normalizedRunningPages(value: unknown): number[] {
  if (!Array.isArray(value)) return []
  return Array.from(new Set(
    value
      .map((item) => Number(item))
      .filter((item) => Number.isInteger(item) && item > 0),
  )).sort((a, b) => a - b)
}

function sanitizedActiveTasks(value: ConvertActiveTask[] | null | undefined): ConvertActiveTask[] {
  if (!Array.isArray(value)) return []
  return value.map((task) => {
    const runningPages = normalizedRunningPages(task.running_pages)
    return {
      ...task,
      cur_page_msg: '',
      running_pages: runningPages,
      running_page_count: Math.max(runningPages.length, numberValue(task.running_page_count)),
    }
  })
}

function sanitizedLibraryFiles(value: LibraryFileItem[] | null | undefined): LibraryFileItem[] {
  if (!Array.isArray(value)) return []
  return value.map((item) => {
    const runningPages = normalizedRunningPages(item.running_pages)
    return {
      ...item,
      cur_page_msg: '',
      running_pages: runningPages,
      running_page_count: Math.max(runningPages.length, numberValue(item.running_page_count)),
    }
  })
}

function queueSnapshotRunning(queue: LibraryFilesResponse['queue'] | null | undefined): boolean {
  if (!queue) return false
  const activeTasks = Array.isArray(queue.active_tasks) ? queue.active_tasks : []
  return Boolean(
    queue.running
    || activeTasks.length > 0
    || numberValue(queue.active_count) > 0
    || numberValue(queue.queued_count) > 0
  )
}

function progressFromQueueSnapshot(queue: LibraryFilesResponse['queue'] | null | undefined): ConvertProgressState | null {
  if (!queue) return null
  const activeTasks = sanitizedActiveTasks(queue.active_tasks)
  const primary = activeTasks[0] || null
  const primaryRunningPages = normalizedRunningPages(primary?.running_pages)
  return {
    total: numberValue(queue.total),
    completed: numberValue(queue.done),
    current: String(queue.current || primary?.name || ''),
    activeCount: numberValue(queue.active_count) || activeTasks.length,
    activeTasks,
    curPageDone: numberValue(primary?.cur_page_done),
    curPageTotal: numberValue(primary?.cur_page_total),
    // Converter stdout is private diagnostics. Public UI state uses only
    // structured stage/page fields and never retains the raw message.
    curPageMsg: '',
    conversionStage: String(primary?.conversion_stage || ''),
    runningPages: primaryRunningPages,
    runningPageCount: Math.max(
      primaryRunningPages.length,
      numberValue(primary?.running_page_count),
    ),
    last: '',
  }
}

function conversionPathKey(value: unknown) {
  return String(value || '').trim().replace(/\\/g, '/').toLowerCase()
}

function mergeActiveConversionProgress(files: LibraryFileItem[], activeTasks: ConvertActiveTask[]) {
  if (!activeTasks.length) return files
  const byPath = new Map(activeTasks.map((task) => [conversionPathKey(task.pdf), task]))
  const byName = new Map(activeTasks.map((task) => [String(task.name || '').trim(), task]))
  return files.map((item) => {
    const task = byPath.get(conversionPathKey(item.path)) || byName.get(String(item.name || '').trim())
    if (!task) return item
    const runningPages = normalizedRunningPages(task.running_pages)
    return {
      ...item,
      task_state: 'running' as const,
      task_id: task.task_id,
      status: task.replace ? 'running_reconvert' : 'running',
      replace_task: Boolean(task.replace),
      category: 'pending' as const,
      queue_pos: 0,
      cur_page_done: Number(task.cur_page_done || 0),
      cur_page_total: Number(task.cur_page_total || 0),
      cur_page_msg: '',
      conversion_stage: task.conversion_stage || 'converting',
      running_pages: runningPages,
      running_page_count: Math.max(
        runningPages.length,
        numberValue(task.running_page_count),
      ),
    }
  })
}

let filesLoadRequestSeq = 0
let qualityOverviewRequestSeq = 0
let convertProgressStreamSeq = 0
let refSyncStreamSeq = 0
let latestRequestedFileScope = '200'

export const useLibraryStore = create<LibraryState>((set, get) => ({
  pdfs: [],
  files: [],
  viewScope: '200',
  fileCounts: null,
  qualityOverview: null,
  qualityOverviewLoading: false,
  qualityOverviewError: '',
  converting: false,
  recoverableCount: 0,
  conversionPersistenceError: '',
  pendingRepairReindex: false,
  pendingRepairRunIds: [],
  progress: null,
  sseController: null,
  refSync: null,
  refSyncController: null,

  loadPdfs: async () => {
    await get().loadFiles(get().viewScope || '200')
  },

  loadFiles: async (scope = '200') => {
    const filesRequestId = filesLoadRequestSeq + 1
    const overviewRequestId = qualityOverviewRequestSeq + 1
    const shouldLoadQualityOverview = qualityDiagnosticsVisible()
    filesLoadRequestSeq = filesRequestId
    qualityOverviewRequestSeq = overviewRequestId
    latestRequestedFileScope = scope
    set({
      qualityOverview: shouldLoadQualityOverview ? get().qualityOverview : null,
      qualityOverviewLoading: shouldLoadQualityOverview,
      qualityOverviewError: '',
    })
    const [viewResult, overviewResult] = await Promise.allSettled([
      libraryApi.listFiles(scope),
      shouldLoadQualityOverview ? libraryApi.qualityOverview('all') : Promise.resolve(null),
    ])
    if (viewResult.status === 'rejected') {
      if (filesRequestId !== filesLoadRequestSeq) return
      if (overviewRequestId === qualityOverviewRequestSeq) {
        set({ qualityOverviewLoading: false })
      }
      throw viewResult.reason
    }
    if (filesRequestId !== filesLoadRequestSeq) return
    const view = viewResult.value
    const files = sanitizedLibraryFiles(view.items)
    const queueRunning = queueSnapshotRunning(view.queue)
    const queueProgress = progressFromQueueSnapshot(view.queue)
    let overviewPatch: Partial<LibraryState>
    if (!shouldLoadQualityOverview) {
      overviewPatch = {
        qualityOverview: null,
        qualityOverviewLoading: false,
        qualityOverviewError: '',
      }
    } else if (overviewResult.status === 'fulfilled') {
      overviewPatch = {
        qualityOverview: overviewResult.value,
        qualityOverviewLoading: false,
        qualityOverviewError: '',
      }
    } else {
      overviewPatch = {
        qualityOverview: null,
        qualityOverviewLoading: false,
        qualityOverviewError: overviewResult.reason instanceof Error ? overviewResult.reason.message : String(overviewResult.reason || 'quality overview failed'),
      }
    }
    const patch: Partial<LibraryState> = {
      viewScope: scope,
      files,
      fileCounts: view.counts || null,
      pdfs: files.map((item) => ({ name: item.name, path: item.path })),
      converting: queueRunning,
      recoverableCount: numberValue(view.queue?.recoverable_count),
      conversionPersistenceError: String(view.queue?.persistence_error || ''),
      progress: queueRunning ? queueProgress : null,
    }
    if (overviewRequestId === qualityOverviewRequestSeq) {
      Object.assign(patch, overviewPatch)
    }
    set(patch)
    if (queueRunning && !get().sseController) {
      get().startProgressStream()
    }
  },

  loadQualityOverview: async (scope = 'all') => {
    const requestId = qualityOverviewRequestSeq + 1
    qualityOverviewRequestSeq = requestId
    if (!qualityDiagnosticsVisible()) {
      set({ qualityOverview: null, qualityOverviewLoading: false, qualityOverviewError: '' })
      return
    }
    set({ qualityOverviewLoading: true, qualityOverviewError: '' })
    try {
      const overview = await libraryApi.qualityOverview(scope)
      if (requestId !== qualityOverviewRequestSeq) return
      set({ qualityOverview: overview, qualityOverviewLoading: false, qualityOverviewError: '' })
    } catch (err) {
      if (requestId !== qualityOverviewRequestSeq) return
      set({
        qualityOverview: null,
        qualityOverviewLoading: false,
        qualityOverviewError: err instanceof Error ? err.message : String(err || 'quality overview failed'),
      })
    }
  },

  upload: async (file, baseName) => {
    return libraryApi.upload(file, baseName)
  },

  convert: async (name, mode = 'balanced', replace = true) => {
    set({ converting: true, progress: null })
    await libraryApi.convert(name, mode, { replace })
    await get().loadFiles(get().viewScope || '200')
    get().startProgressStream()
  },

  convertPending: async (mode = 'balanced', limit = 0) => {
    const res = await libraryApi.convertPending(mode, limit)
    if (res.enqueued > 0) {
      set({ converting: true, progress: null })
      await get().loadFiles(get().viewScope || '200')
      get().startProgressStream()
    } else {
      await get().loadFiles(get().viewScope || '200')
    }
    return res
  },

  repairQuality: async (body, options) => {
    const res = await libraryApi.repairQuality(body)
    const needsReindex = Boolean(res.needs_reindex || res.impact?.needs_reindex)
    const autoReindexAfterQueued = options?.autoReindexAfterQueued !== false
    const shouldAutoReindex = Number(res.enqueued || 0) > 0 && needsReindex && autoReindexAfterQueued
    const repairRunId = String(res.repair_run_id || res.repair_run?.run_id || '').trim()
    if (Number(res.enqueued || 0) > 0) {
      set((state) => ({
        converting: true,
        progress: null,
        pendingRepairReindex: state.pendingRepairReindex || shouldAutoReindex,
        pendingRepairRunIds: shouldAutoReindex && repairRunId
          ? Array.from(new Set([...state.pendingRepairRunIds, repairRunId]))
          : state.pendingRepairRunIds,
      }))
      await get().loadFiles(get().viewScope || '200')
      get().startProgressStream()
    } else {
      await get().loadFiles(get().viewScope || '200')
    }
    return res
  },

  openFile: async (pdfName, target = 'pdf') => {
    await libraryApi.openFile(pdfName, target)
  },

  deleteFile: async (pdfName, alsoMd = true) => {
    const res = await libraryApi.deleteFile(pdfName, alsoMd)
    await get().loadFiles(get().viewScope || '200')
    return res
  },

  updatePaperMeta: async (body) => {
    const res = await libraryApi.updateMeta(body)
    let updated: LibraryFileItem | null = null
    set((state) => {
      const files = state.files.map((item) => {
        const match =
          (body.pdf_name && item.name === body.pdf_name)
          || (res.sha1 && item.sha1 === res.sha1)
          || (res.path && item.path === res.path)
        if (!match) return item
        updated = {
          ...item,
          sha1: res.sha1 || item.sha1,
          path: res.path || item.path,
          paper_category: res.paper_category,
          reading_status: res.reading_status,
          note: res.note,
          user_tags: Array.isArray(res.user_tags) ? res.user_tags : [],
          has_suggestions: Boolean(res.has_suggestions),
          suggested_category: String(res.suggested_category || ''),
          suggested_tags: Array.isArray(res.suggested_tags) ? res.suggested_tags : [],
        }
        return updated
      })
      return {
        files,
        pdfs: files.map((item) => ({ name: item.name, path: item.path })),
      }
    })
    return updated
  },

  batchUpdatePaperMeta: async (body) => {
    const res = await libraryApi.batchUpdateMeta(body)
    await get().loadFiles(get().viewScope || '200')
    return Number(res.updated || 0)
  },

  regenerateSuggestions: async (body = {}) => {
    const res = await libraryApi.regenerateSuggestions(body)
    await get().loadFiles(get().viewScope || '200')
    return Number(res.updated || 0)
  },

  applySuggestionAction: async (body) => {
    const res = await libraryApi.applySuggestionAction(body)
    let updated: LibraryFileItem | null = null
    set((state) => {
      const files = state.files.map((item) => {
        const match =
          (body.pdf_name && item.name === body.pdf_name)
          || (res.sha1 && item.sha1 === res.sha1)
          || (res.path && item.path === res.path)
        if (!match) return item
        updated = {
          ...item,
          sha1: res.sha1 || item.sha1,
          path: res.path || item.path,
          paper_category: res.paper_category,
          reading_status: res.reading_status,
          note: res.note,
          user_tags: Array.isArray(res.user_tags) ? res.user_tags : [],
          has_suggestions: Boolean(res.has_suggestions),
          suggested_category: String(res.suggested_category || ''),
          suggested_tags: Array.isArray(res.suggested_tags) ? res.suggested_tags : [],
        }
        return updated
      })
      return {
        files,
        pdfs: files.map((item) => ({ name: item.name, path: item.path })),
      }
    })
    return updated
  },

  cancelConvert: async () => {
    await libraryApi.cancelConvert()
    get().startProgressStream()
    await get().loadFiles(get().viewScope || '200')
  },

  cancelConversionTask: async (taskId) => {
    const res = await libraryApi.cancelConversionTask(taskId)
    get().startProgressStream()
    await get().loadFiles(get().viewScope || '200')
    return res
  },

  resumeConversionTask: async (taskId) => {
    const res = await libraryApi.resumeConversionTask(taskId)
    await get().loadFiles(get().viewScope || '200')
    if (res.enqueued) {
      set({ converting: true })
      get().startProgressStream()
    }
    return res
  },

  resumeAllConversions: async () => {
    const res = await libraryApi.resumeAllConversions()
    await get().loadFiles(get().viewScope || '200')
    if (res.enqueued > 0) {
      set({ converting: true })
      get().startProgressStream()
    }
    return res
  },

  reindex: async () => {
    const res = await libraryApi.reindex()
    if (res.ok) {
      get().startRefSyncStream()
    }
    return res
  },

  startReferenceSync: async () => {
    const res = await referencesApi.startSync()
    get().startRefSyncStream()
    return res
  },

  startProgressStream: () => {
    get().stopProgressStream()
    const streamId = convertProgressStreamSeq + 1
    convertProgressStreamSeq = streamId
    const streamIsCurrent = () => streamId === convertProgressStreamSeq

    const ctrl = libraryApi.streamConvertStatus(
      (data) => {
        if (!streamIsCurrent()) return
        const activeTasks = sanitizedActiveTasks(data.active_tasks)
        const runningPages = normalizedRunningPages(
          data.running_pages ?? activeTasks[0]?.running_pages,
        )
        set((state) => ({
          converting: data.running,
          recoverableCount: numberValue(data.recoverable_count),
          conversionPersistenceError: String(data.persistence_error || ''),
          files: mergeActiveConversionProgress(state.files, activeTasks),
          progress: {
            total: data.total,
            completed: data.completed,
            current: data.current,
            activeCount: Number(data.active_count || 0),
            activeTasks,
            curPageDone: data.cur_page_done,
            curPageTotal: data.cur_page_total,
            curPageMsg: '',
            conversionStage: String(data.conversion_stage || ''),
            runningPages,
            runningPageCount: Math.max(
              runningPages.length,
              numberValue(data.running_page_count ?? activeTasks[0]?.running_page_count),
            ),
            last: data.last,
          },
        }))
      },
      () => {
        if (!streamIsCurrent()) return
        const shouldReindex = get().pendingRepairReindex
        const repairRunIds = get().pendingRepairRunIds
        set({ converting: false, progress: null, sseController: null, pendingRepairReindex: false, pendingRepairRunIds: [] })
        if (shouldReindex) {
          void (async () => {
            try {
              if (repairRunIds.length > 0) {
                let advanced = false
                for (const runId of repairRunIds) {
                  try {
                    const res = await libraryApi.advanceQualityRepairRun(runId)
                    if (!streamIsCurrent()) return
                    advanced = true
                    if (res.reindex?.ok) get().startRefSyncStream()
                  } catch {
                    // Fall back to a plain index refresh below if no repair run could advance.
                  }
                }
                if (!advanced) {
                  await get().reindex()
                  if (!streamIsCurrent()) return
                }
              } else {
                await get().reindex()
                if (!streamIsCurrent()) return
              }
            } catch {
              // A failed automatic refresh should not leave the conversion stream stuck.
            }
            if (!streamIsCurrent()) return
            await get().loadFiles(latestRequestedFileScope || get().viewScope || '200')
          })()
        } else {
          void get().loadFiles(latestRequestedFileScope || get().viewScope || '200')
        }
      },
      () => {
        if (!streamIsCurrent()) return
        set({ sseController: null })
        void (async () => {
          try {
            await get().loadFiles(latestRequestedFileScope || get().viewScope || '200')
            if (!streamIsCurrent()) return
            if (get().pendingRepairReindex && !get().converting) {
              const repairRunIds = get().pendingRepairRunIds
              set({ pendingRepairReindex: false, pendingRepairRunIds: [] })
              let advanced = false
              for (const runId of repairRunIds) {
                try {
                  const res = await libraryApi.advanceQualityRepairRun(runId)
                  if (!streamIsCurrent()) return
                  advanced = true
                  if (res.reindex?.ok) get().startRefSyncStream()
                } catch {
                  // Fall back to a plain index refresh below if no repair run could advance.
                }
              }
              if (!advanced) await get().reindex()
              if (!streamIsCurrent()) return
              await get().loadFiles(latestRequestedFileScope || get().viewScope || '200')
            }
          } catch {
            if (!streamIsCurrent()) return
            set({ converting: false, progress: null, sseController: null })
          }
        })()
      },
    )
    set({ sseController: ctrl })
  },

  stopProgressStream: () => {
    convertProgressStreamSeq += 1
    get().sseController?.abort()
    set({ sseController: null })
  },

  startRefSyncStream: () => {
    get().stopRefSyncStream()
    const streamId = refSyncStreamSeq + 1
    refSyncStreamSeq = streamId
    const streamIsCurrent = () => streamId === refSyncStreamSeq

    const ctrl = referencesApi.streamSyncStatus(
      (data) => {
        if (!streamIsCurrent()) return
        const status = String(data.status || '')
        const running = Boolean(data.running)
        set({
          refSync: {
            running,
            status,
            stage: String(data.stage || ''),
            message: String(data.message || ''),
            error: String(data.error || ''),
            current: String(data.current || ''),
            docsDone: Number(data.docs_done || 0),
            docsTotal: Number(data.docs_total || 0),
            runId: Number(data.run_id || 0),
            stats: normalizeRefSyncEventStats(data),
          },
        })
      },
      () => {
        if (!streamIsCurrent()) return
        set((state) => ({
          refSyncController: null,
          refSync: state.refSync
            ? { ...state.refSync, running: false }
            : state.refSync,
        }))
      },
      (err) => {
        if (!streamIsCurrent()) return
        const error = err instanceof Error ? err.message : 'Reference sync stream failed'
        set((state) => ({
          refSyncController: null,
          refSync: state.refSync
            ? { ...state.refSync, running: false, status: 'error', error }
            : {
              running: false,
              status: 'error',
              stage: '',
              message: '',
              error,
              current: '',
              docsDone: 0,
              docsTotal: 0,
              runId: 0,
              stats: {},
            },
        }))
      },
    )
    set({ refSyncController: ctrl })
  },

  stopRefSyncStream: () => {
    refSyncStreamSeq += 1
    get().refSyncController?.abort()
    set({ refSyncController: null })
  },
}))
