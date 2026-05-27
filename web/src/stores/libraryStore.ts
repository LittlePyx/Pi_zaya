import { create } from 'zustand'
import {
  libraryApi,
  type ConvertActiveTask,
  type LibraryFileItem,
  type LibraryQualityOverviewResponse,
  type LibrarySuggestionActionBody,
  type LibraryQualityRepairBody,
  type LibraryQualityRepairResponse,
  type LibraryMetaBatchUpdateBody,
  type LibraryMetaUpdateBody,
  type LibrarySuggestionRegenerateBody,
} from '../api/library'
import { referencesApi } from '../api/references'

interface ConvertProgressState {
  total: number
  completed: number
  current: string
  activeCount: number
  activeTasks: ConvertActiveTask[]
  curPageDone: number
  curPageTotal: number
  curPageMsg: string
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
    reconverting: number
    quality_review: number
    quality_ready: number
  } | null
  qualityOverview: LibraryQualityOverviewResponse | null
  qualityOverviewLoading: boolean
  qualityOverviewError: string
  converting: boolean
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
  repairQuality: (body: LibraryQualityRepairBody) => Promise<LibraryQualityRepairResponse>
  openFile: (pdfName: string, target?: 'pdf' | 'md' | 'pdf_dir' | 'md_dir') => Promise<void>
  deleteFile: (pdfName: string, alsoMd?: boolean) => Promise<{ ok: boolean; pdf_deleted: boolean; md_deleted: boolean; removed_queued: number; warnings: string[]; needs_reindex: boolean }>
  updatePaperMeta: (body: LibraryMetaUpdateBody) => Promise<LibraryFileItem | null>
  batchUpdatePaperMeta: (body: LibraryMetaBatchUpdateBody) => Promise<number>
  regenerateSuggestions: (body?: LibrarySuggestionRegenerateBody) => Promise<number>
  applySuggestionAction: (body: LibrarySuggestionActionBody) => Promise<LibraryFileItem | null>
  cancelConvert: () => Promise<void>
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

export const useLibraryStore = create<LibraryState>((set, get) => ({
  pdfs: [],
  files: [],
  viewScope: '200',
  fileCounts: null,
  qualityOverview: null,
  qualityOverviewLoading: false,
  qualityOverviewError: '',
  converting: false,
  progress: null,
  sseController: null,
  refSync: null,
  refSyncController: null,

  loadPdfs: async () => {
    await get().loadFiles(get().viewScope || '200')
  },

  loadFiles: async (scope = '200') => {
    set({ qualityOverviewLoading: true, qualityOverviewError: '' })
    const [viewResult, overviewResult] = await Promise.allSettled([
      libraryApi.listFiles(scope),
      libraryApi.qualityOverview('all'),
    ])
    if (viewResult.status === 'rejected') {
      set({ qualityOverviewLoading: false })
      throw viewResult.reason
    }
    const view = viewResult.value
    const files = Array.isArray(view.items) ? view.items : []
    const overviewPatch = overviewResult.status === 'fulfilled'
      ? {
        qualityOverview: overviewResult.value,
        qualityOverviewLoading: false,
        qualityOverviewError: '',
      }
      : {
        qualityOverview: null,
        qualityOverviewLoading: false,
        qualityOverviewError: overviewResult.reason instanceof Error ? overviewResult.reason.message : String(overviewResult.reason || 'quality overview failed'),
      }
    set({
      viewScope: scope,
      files,
      fileCounts: view.counts || null,
      pdfs: files.map((item) => ({ name: item.name, path: item.path })),
      ...overviewPatch,
    })
  },

  loadQualityOverview: async (scope = 'all') => {
    set({ qualityOverviewLoading: true, qualityOverviewError: '' })
    try {
      const overview = await libraryApi.qualityOverview(scope)
      set({ qualityOverview: overview, qualityOverviewLoading: false, qualityOverviewError: '' })
    } catch (err) {
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

  repairQuality: async (body) => {
    const res = await libraryApi.repairQuality(body)
    if (Number(res.enqueued || 0) > 0) {
      set({ converting: true, progress: null })
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
    get().stopProgressStream()
    await libraryApi.cancelConvert()
    set({ converting: false, progress: null })
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

    const ctrl = libraryApi.streamConvertStatus(
      (data) => {
        set({
          converting: data.running,
          progress: {
            total: data.total,
            completed: data.completed,
            current: data.current,
            activeCount: Number(data.active_count || 0),
            activeTasks: Array.isArray(data.active_tasks) ? data.active_tasks : [],
            curPageDone: data.cur_page_done,
            curPageTotal: data.cur_page_total,
            curPageMsg: data.cur_page_msg,
            last: data.last,
          },
        })
      },
      () => {
        set({ converting: false, progress: null, sseController: null })
        get().loadFiles(get().viewScope || '200')
      },
      () => {
        set({ converting: false, progress: null, sseController: null })
      },
    )
    set({ sseController: ctrl })
  },

  stopProgressStream: () => {
    get().sseController?.abort()
    set({ sseController: null })
  },

  startRefSyncStream: () => {
    get().stopRefSyncStream()
    const ctrl = referencesApi.streamSyncStatus(
      (data) => {
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
          },
        })
      },
      () => {
        set((state) => ({
          refSyncController: null,
          refSync: state.refSync
            ? { ...state.refSync, running: false }
            : state.refSync,
        }))
      },
      () => {
        set({ refSyncController: null })
      },
    )
    set({ refSyncController: ctrl })
  },

  stopRefSyncStream: () => {
    get().refSyncController?.abort()
    set({ refSyncController: null })
  },
}))
