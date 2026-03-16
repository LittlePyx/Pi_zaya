import { create } from 'zustand'
import {
  libraryApi,
  type LibraryFileItem,
  type LibrarySuggestionActionBody,
  type LibraryMetaBatchUpdateBody,
  type LibraryMetaUpdateBody,
  type LibrarySuggestionRegenerateBody,
} from '../api/library'
import { referencesApi } from '../api/references'

interface ConvertProgressState {
  total: number
  completed: number
  current: string
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
  } | null
  converting: boolean
  progress: ConvertProgressState | null
  sseController: AbortController | null
  refSync: RefSyncState | null
  refSyncController: AbortController | null
  loadPdfs: () => Promise<void>
  loadFiles: (scope?: string) => Promise<void>
  upload: (file: File, baseName?: string) => Promise<{ name: string; duplicate?: boolean; existing?: string }>
  convert: (name: string, mode?: string, replace?: boolean) => Promise<void>
  convertPending: (mode?: string, limit?: number) => Promise<{ ok: boolean; enqueued: number; skipped_busy: number; pending_total: number }>
  openFile: (pdfName: string, target?: 'pdf' | 'md' | 'pdf_dir' | 'md_dir') => Promise<void>
  deleteFile: (pdfName: string, alsoMd?: boolean) => Promise<{ ok: boolean; pdf_deleted: boolean; md_deleted: boolean; removed_queued: number; warnings: string[]; needs_reindex: boolean }>
  updatePaperMeta: (body: LibraryMetaUpdateBody) => Promise<LibraryFileItem | null>
  batchUpdatePaperMeta: (body: LibraryMetaBatchUpdateBody) => Promise<number>
  regenerateSuggestions: (body?: LibrarySuggestionRegenerateBody) => Promise<number>
  applySuggestionAction: (body: LibrarySuggestionActionBody) => Promise<LibraryFileItem | null>
  cancelConvert: () => Promise<void>
  reindex: () => Promise<{ ok: boolean; stdout: string; stderr: string; refsync: { started?: boolean; reason?: string; run_id?: number } | null; refsync_error: string }>
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
  converting: false,
  progress: null,
  sseController: null,
  refSync: null,
  refSyncController: null,

  loadPdfs: async () => {
    await get().loadFiles(get().viewScope || '200')
  },

  loadFiles: async (scope = '200') => {
    const view = await libraryApi.listFiles(scope)
    const files = Array.isArray(view.items) ? view.items : []
    set({
      viewScope: scope,
      files,
      fileCounts: view.counts || null,
      pdfs: files.map((item) => ({ name: item.name, path: item.path })),
    })
  },

  upload: async (file, baseName) => {
    return libraryApi.upload(file, baseName)
  },

  convert: async (name, mode = 'balanced', replace = false) => {
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
