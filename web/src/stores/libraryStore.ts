import { create } from 'zustand'
import { libraryApi, type ConvertProgress } from '../api/library'

interface ConvertProgressState {
  total: number
  completed: number
  current: string
  curPageDone: number
  curPageTotal: number
  curPageMsg: string
  last: string
}

interface LibraryState {
  pdfs: { name: string; path: string }[]
  converting: boolean
  progress: ConvertProgressState | null
  sseController: AbortController | null
  loadPdfs: () => Promise<void>
  upload: (file: File, baseName?: string) => Promise<{ name: string; duplicate?: boolean; existing?: string }>
  convert: (name: string, mode?: string) => Promise<void>
  cancelConvert: () => Promise<void>
  reindex: () => Promise<{ ok: boolean }>
  startProgressStream: () => void
  stopProgressStream: () => void
}

export const useLibraryStore = create<LibraryState>((set, get) => ({
  pdfs: [],
  converting: false,
  progress: null,
  sseController: null,

  loadPdfs: async () => {
    const list = await libraryApi.listPdfs()
    set({ pdfs: list })
  },

  upload: async (file, baseName) => {
    return libraryApi.upload(file, baseName)
  },

  convert: async (name, mode = 'balanced') => {
    set({ converting: true, progress: null })
    await libraryApi.convert(name, mode)
    get().startProgressStream()
  },

  cancelConvert: async () => {
    get().stopProgressStream()
    await libraryApi.cancelConvert()
    set({ converting: false, progress: null })
  },

  reindex: async () => {
    return libraryApi.reindex()
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
        get().loadPdfs()
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
}))
