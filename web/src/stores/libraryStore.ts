import { create } from 'zustand'
import { libraryApi } from '../api/library'

interface LibraryState {
  pdfs: { name: string; path: string }[]
  converting: boolean
  convProgress: string
  loadPdfs: () => Promise<void>
  upload: (file: File, baseName?: string) => Promise<{ name: string; duplicate?: boolean; existing?: string }>
  convert: (name: string, mode?: string) => Promise<void>
  cancelConvert: () => Promise<void>
  reindex: () => Promise<{ ok: boolean }>
}

export const useLibraryStore = create<LibraryState>((set) => ({
  pdfs: [],
  converting: false,
  convProgress: '',

  loadPdfs: async () => {
    const list = await libraryApi.listPdfs()
    set({ pdfs: list })
  },

  upload: async (file, baseName) => {
    return libraryApi.upload(file, baseName)
  },

  convert: async (name, mode = 'balanced') => {
    set({ converting: true, convProgress: '排队中...' })
    await libraryApi.convert(name, mode)
  },

  cancelConvert: async () => {
    await libraryApi.cancelConvert()
    set({ converting: false, convProgress: '' })
  },

  reindex: async () => {
    const res = await libraryApi.reindex()
    return res
  },
}))
