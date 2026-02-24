import { api } from './client'

export interface ConvertProgress {
  running: boolean
  done: boolean
  total: number
  completed: number
  current: string
  cur_page_done: number
  cur_page_total: number
  cur_page_msg: string
  last: string
}

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

  streamConvertStatus: (
    onData: (data: ConvertProgress) => void,
    onDone: () => void,
    onError?: (err: unknown) => void,
  ): AbortController => {
    const ctrl = new AbortController()
    ;(async () => {
      try {
        const res = await fetch('/api/library/convert/status', { signal: ctrl.signal })
        const reader = res.body!.getReader()
        const decoder = new TextDecoder()
        let buf = ''
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buf += decoder.decode(value, { stream: true })
          const lines = buf.split('\n')
          buf = lines.pop() || ''
          for (const line of lines) {
            if (!line.startsWith('data: ')) continue
            try {
              const data = JSON.parse(line.slice(6)) as ConvertProgress
              onData(data)
              if (data.done) { onDone(); return }
            } catch { /* skip bad JSON */ }
          }
        }
      } catch (err) {
        if (!ctrl.signal.aborted) onError?.(err)
      }
    })()
    return ctrl
  },
}
