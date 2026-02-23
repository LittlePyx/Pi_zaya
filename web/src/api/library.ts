import { api } from './client'

export const libraryApi = {
  listPdfs: () => api.get<{ name: string; path: string }[]>('/api/library/pdfs'),
  upload: async (file: File, baseName?: string) => {
    const fd = new FormData()
    fd.append('file', file)
    if (baseName) fd.append('base_name', baseName)
    const res = await fetch('/api/library/upload', { method: 'POST', body: fd })
    return res.json()
  },
  convert: (pdfName: string, speedMode = 'balanced') =>
    api.post('/api/library/convert', { pdf_name: pdfName, speed_mode: speedMode }),
  cancelConvert: () => api.post('/api/library/convert/cancel'),
  reindex: () => api.post<{ ok: boolean; stdout: string; stderr: string }>('/api/library/reindex'),
}
