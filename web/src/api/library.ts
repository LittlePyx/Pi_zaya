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

export interface LibraryFileItem {
  name: string
  path: string
  sha1: string
  md_exists: boolean
  md_path: string
  md_folder: string
  category: 'pending' | 'converted'
  task_state: 'idle' | 'queued' | 'running'
  status: string
  replace_task: boolean
  queue_pos: number
  paper_category: string
  reading_status: '' | 'unread' | 'reading' | 'done' | 'revisit'
  note: string
  user_tags: string[]
  has_suggestions: boolean
  suggested_category: string
  suggested_tags: string[]
}

export interface LibraryMetaUpdateBody {
  pdf_name?: string
  sha1?: string
  path?: string
  paper_category: string
  reading_status: '' | 'unread' | 'reading' | 'done' | 'revisit'
  note: string
  user_tags: string[]
}

export interface LibraryMetaUpdateResponse {
  ok: boolean
  sha1: string
  path: string
  paper_category: string
  reading_status: '' | 'unread' | 'reading' | 'done' | 'revisit'
  note: string
  user_tags: string[]
  has_suggestions: boolean
  suggested_category: string
  suggested_tags: string[]
}

export interface LibraryMetaBatchUpdateBody {
  pdf_names?: string[]
  sha1s?: string[]
  apply_paper_category: boolean
  paper_category: string
  apply_reading_status: boolean
  reading_status: '' | 'unread' | 'reading' | 'done' | 'revisit'
  add_tags: string[]
  remove_tags: string[]
}

export interface LibraryMetaBatchUpdateItem {
  name: string
  sha1: string
  path: string
  paper_category: string
  reading_status: '' | 'unread' | 'reading' | 'done' | 'revisit'
  note: string
  user_tags: string[]
}

export interface LibraryMetaBatchUpdateResponse {
  ok: boolean
  requested: number
  updated: number
  items: LibraryMetaBatchUpdateItem[]
}

export interface LibrarySuggestionRegenerateBody {
  pdf_names?: string[]
  sha1s?: string[]
}

export interface LibrarySuggestionActionBody {
  pdf_name?: string
  sha1?: string
  path?: string
  category_action?: '' | 'accept' | 'dismiss'
  accept_tags?: string[]
  dismiss_tags?: string[]
  accept_all_tags?: boolean
  dismiss_all_tags?: boolean
}

export interface LibrarySuggestionResponseItem {
  name: string
  sha1: string
  path: string
  paper_category: string
  reading_status: '' | 'unread' | 'reading' | 'done' | 'revisit'
  note: string
  user_tags: string[]
  has_suggestions: boolean
  suggested_category: string
  suggested_tags: string[]
}

export interface LibrarySuggestionRegenerateResponse {
  ok: boolean
  updated: number
  items: LibrarySuggestionResponseItem[]
}

export interface LibrarySuggestionActionResponse {
  ok: boolean
  sha1: string
  path: string
  paper_category: string
  reading_status: '' | 'unread' | 'reading' | 'done' | 'revisit'
  note: string
  user_tags: string[]
  has_suggestions: boolean
  suggested_category: string
  suggested_tags: string[]
}

export interface LibraryFilesResponse {
  items: LibraryFileItem[]
  counts: {
    total_view: number
    total_all: number
    pending: number
    converted: number
    queued: number
    running: number
    reconverting: number
  }
  truncated: boolean
  scope: string
  queue: {
    running: boolean
    current: string
    done: number
    total: number
  }
}

export interface RenameSuggestionItem {
  name: string
  path: string
  suggested_name: string
  suggested_stem: string
  display_full_name: string
  diff: boolean
  meta: {
    venue: string
    year: string
    title: string
  }
  md_exists: boolean
  md_path: string
  md_folder: string
}

export interface RenameSuggestionsResponse {
  items: RenameSuggestionItem[]
  scope: string
  use_llm: boolean
  total_scanned: number
  changed: number
}

export interface RenameApplyResponse {
  ok: boolean
  renamed: number
  skipped: number
  failed: number
  needs_reindex: boolean
  items: Array<Record<string, unknown>>
}

export interface UploadInspectResponse {
  name: string
  sha1: string
  duplicate: boolean
  existing: string
  existing_path: string
  suggested_name: string
  suggested_stem: string
  display_full_name: string
  meta: {
    venue: string
    year: string
    title: string
  }
}

export interface UploadCommitResponse {
  duplicate?: boolean
  existing?: string
  path?: string
  name?: string
  sha1?: string
  citation_meta?: Record<string, unknown>
  enqueued: boolean
  task_id: string
}

export interface GuideSourceResponse {
  ok: boolean
  pdf_name: string
  pdf_path: string
  md_path: string
  md_exists: boolean
  source_path: string
  source_name: string
}

export interface LibraryReindexResponse {
  ok: boolean
  stdout: string
  stderr: string
  refsync: {
    started?: boolean
    reason?: string
    run_id?: number
  } | null
  refsync_error: string
}

export const libraryApi = {
  listPdfs: () => api.get<{ name: string; path: string }[]>('/api/library/pdfs'),
  listFiles: (scope = '200') =>
    api.get<LibraryFilesResponse>(`/api/library/files?scope=${encodeURIComponent(scope)}`),
  upload: async (file: File, baseName?: string) => {
    const fd = new FormData()
    fd.append('file', file)
    if (baseName) fd.append('base_name', baseName)
    const res = await fetch('/api/library/upload', { method: 'POST', body: fd })
    return res.json()
  },
  inspectUpload: async (file: File, useLlm = true) => {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('use_llm', String(useLlm))
    const res = await fetch('/api/library/upload/inspect', { method: 'POST', body: fd })
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
    return res.json() as Promise<UploadInspectResponse>
  },
  commitUpload: async (
    file: File,
    opts?: { baseName?: string; convertNow?: boolean; speedMode?: string; allowDuplicate?: boolean },
  ) => {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('base_name', String(opts?.baseName || ''))
    fd.append('convert_now', String(Boolean(opts?.convertNow)))
    fd.append('speed_mode', String(opts?.speedMode || 'balanced'))
    fd.append('allow_duplicate', String(Boolean(opts?.allowDuplicate)))
    const res = await fetch('/api/library/upload/commit', { method: 'POST', body: fd })
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
    return res.json() as Promise<UploadCommitResponse>
  },
  convert: (pdfName: string, speedMode = 'balanced', opts?: { replace?: boolean }) =>
    api.post('/api/library/convert', {
      pdf_name: pdfName,
      speed_mode: speedMode,
      no_llm: speedMode === 'no_llm',
      replace: Boolean(opts?.replace),
    }),
  convertPending: (speedMode = 'balanced', limit = 0) =>
    api.post<{ ok: boolean; enqueued: number; skipped_busy: number; pending_total: number }>(
      '/api/library/convert/pending',
      { speed_mode: speedMode, limit },
    ),
  cancelConvert: () => api.post('/api/library/convert/cancel'),
  openFile: (pdfName: string, target: 'pdf' | 'md' | 'pdf_dir' | 'md_dir' = 'pdf') =>
    api.post<{ ok: boolean; target: string; path: string }>('/api/library/file/open', {
      pdf_name: pdfName,
      target,
    }),
  deleteFile: (pdfName: string, alsoMd = true) =>
    api.post<{ ok: boolean; pdf_deleted: boolean; md_deleted: boolean; removed_queued: number; warnings: string[]; needs_reindex: boolean }>(
      '/api/library/file/delete',
      {
        pdf_name: pdfName,
        also_md: alsoMd,
      },
    ),
  resolveGuideSource: (pdfName: string) =>
    api.post<GuideSourceResponse>('/api/library/file/guide_source', { pdf_name: pdfName }),
  listRenameSuggestions: (scope = '30', useLlm = true) =>
    api.get<RenameSuggestionsResponse>(
      `/api/library/rename/suggestions?scope=${encodeURIComponent(scope)}&use_llm=${String(useLlm)}`,
    ),
  applyRenameSuggestions: (
    pdfNames: string[],
    baseOverrides?: Record<string, string>,
    opts?: { useLlm?: boolean; alsoMd?: boolean },
  ) =>
    api.post<RenameApplyResponse>('/api/library/rename/apply', {
      pdf_names: pdfNames,
      base_overrides: baseOverrides || {},
      use_llm: Boolean(opts?.useLlm ?? true),
      also_md: Boolean(opts?.alsoMd ?? true),
    }),
  reindex: () => api.post<LibraryReindexResponse>('/api/library/reindex'),
  updateMeta: (body: LibraryMetaUpdateBody) =>
    api.post<LibraryMetaUpdateResponse>('/api/library/meta/update', body),
  batchUpdateMeta: (body: LibraryMetaBatchUpdateBody) =>
    api.post<LibraryMetaBatchUpdateResponse>('/api/library/meta/batch_update', body),
  regenerateSuggestions: (body: LibrarySuggestionRegenerateBody) =>
    api.post<LibrarySuggestionRegenerateResponse>('/api/library/meta/suggestions/regenerate', body),
  applySuggestionAction: (body: LibrarySuggestionActionBody) =>
    api.post<LibrarySuggestionActionResponse>('/api/library/meta/suggestions/apply', body),

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
