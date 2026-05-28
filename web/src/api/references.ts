import { api } from './client'

const citationMetaCache = new Map<string, Promise<Record<string, unknown>>>()
const bibliometricsCache = new Map<string, Promise<Record<string, unknown>>>()
const citationCardPolishCache = new Map<string, Promise<Record<string, unknown>>>()

export interface ReaderDocAnchor {
  anchor_id: string
  block_id?: string
  kind: string
  heading_path?: string
  text?: string
  line_start?: number
  line_end?: number
  number?: number
}

export interface ReaderDocBlock {
  doc_id: string
  block_id: string
  anchor_id: string
  kind: string
  heading_path?: string
  order_index?: number
  line_start?: number
  line_end?: number
  text?: string
  raw_text?: string
  number?: number
}

export interface ShelfMetadataQualityIssue {
  code: string
  label: string
  field: string
  severity: string
  detail?: string
}

export interface ShelfMetadataQuality {
  contract_version: number
  ok: boolean
  status: 'ready' | 'warning' | 'error' | string
  score: number
  missing_fields: string[]
  issues: ShelfMetadataQualityIssue[]
  repairable: boolean
  retryable: boolean
  doi?: string
}

export interface ShelfMetadataRepairItem {
  key: string
  ok: boolean
  changed: boolean
  changed_fields: string[]
  repair_status: 'ready' | 'repaired' | 'partial' | 'retryable' | 'unchanged' | 'error' | string
  retryable: boolean
  fixed_issue_codes?: string[]
  remaining_issue_codes?: string[]
  repair_sources?: string[]
  error_kind?: string
  error_detail?: string
  before: ShelfMetadataQuality
  after: ShelfMetadataQuality
  before_export_acceptance?: Record<string, unknown>
  export_acceptance?: Record<string, unknown>
  meta: Record<string, unknown>
  persisted?: boolean
  persisted_targets?: string[]
}

export interface ShelfMetadataRepairAcceptance {
  contract_version: number
  requested: number
  quality_ok: boolean
  metadata_ready_before: number
  metadata_ready_after: number
  metadata_ready_delta: number
  export_ready_before: number
  export_ready_after: number
  export_ready_delta: number
  summary_export_ready_after: number
  retryable: number
  failed: number
  unresolved_after: number
  remaining_fields?: Array<{ name: string, count: number }>
  remaining_issue_codes?: Array<{ name: string, count: number }>
  retryable_keys?: string[]
  unresolved_keys?: string[]
  failed_keys?: string[]
}

export interface ShelfMetadataRepairVerification {
  type: 'shelf_metadata_repair' | string
  status: 'passed' | 'retryable' | 'failed' | 'partial' | 'skipped' | string
  quality_ok: boolean
  target_count: number
  metadata_ready_after: number
  export_ready_after: number
  changed: number
  retryable: number
  failed: number
  unresolved_after: number
  remaining_fields?: Array<{ name: string, count: number }>
  remaining_issue_codes?: Array<{ name: string, count: number }>
  summary_export_ready_after?: number
  detail?: string
}

export interface ShelfMetadataRepairImpact {
  requested: number
  ready_before: number
  ready_after: number
  ready_delta: number
  export_ready_before?: number
  export_ready_after?: number
  export_ready_delta?: number
  unresolved_after?: number
  summary_export_ready_after?: number
  changed: number
  persisted: number
  before_avg_score: number
  after_avg_score: number
  score_delta: number
  fixed_issue_codes?: Array<{ name: string, count: number }>
  remaining_issue_codes?: Array<{ name: string, count: number }>
  changed_fields?: Array<{ name: string, count: number }>
  repair_sources?: Array<{ name: string, count: number }>
}

export interface ShelfMetadataRepairResponse {
  ok: boolean
  requested: number
  ready: number
  export_ready?: number
  partial: number
  retryable: number
  failed: number
  unresolved?: number
  changed: number
  persisted?: number
  acceptance?: ShelfMetadataRepairAcceptance
  verification?: ShelfMetadataRepairVerification
  repair_run_id?: string
  repair_run?: Record<string, unknown>
  impact?: ShelfMetadataRepairImpact
  items: ShelfMetadataRepairItem[]
}

function stableStringify(value: unknown): string {
  if (value === null || value === undefined) return ''
  if (typeof value !== 'object') return JSON.stringify(value)
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(',')}]`
  const rec = value as Record<string, unknown>
  return `{${Object.keys(rec).sort().map((key) => `${JSON.stringify(key)}:${stableStringify(rec[key])}`).join(',')}}`
}

function withCache(
  cache: Map<string, Promise<Record<string, unknown>>>,
  key: string,
  loader: () => Promise<Record<string, unknown>>,
): Promise<Record<string, unknown>> {
  const cached = cache.get(key)
  if (cached) return cached
  const pending = loader()
    .then((result) => {
      cache.set(key, Promise.resolve(result))
      return result
    })
    .catch((err) => {
      cache.delete(key)
      throw err
    })
  cache.set(key, pending)
  return pending
}

export const referencesApi = {
  startSync: () =>
    api.post<{ started: boolean; reason?: string; run_id?: number }>('/api/references/sync'),
  streamSyncStatus: (
    onData: (data: Record<string, unknown>) => void,
    onDone: () => void,
    onError?: (err: unknown) => void,
  ): AbortController => {
    const ctrl = new AbortController()
    ;(async () => {
      try {
        const res = await fetch('/api/references/sync/status', { signal: ctrl.signal })
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
              const data = JSON.parse(line.slice(6)) as Record<string, unknown>
              onData(data)
              if (data.done === true) { onDone(); return }
            } catch { /* skip bad JSON */ }
          }
        }
      } catch (err) {
        if (!ctrl.signal.aborted) onError?.(err)
      }
    })()
    return ctrl
  },
  open: (sourcePath: string, page?: number | null) =>
    api.post<{ ok: boolean; message: string }>('/api/references/open', {
      source_path: sourcePath,
      page: page ?? null,
    }),
  citationMeta: (sourcePath: string) =>
    api.post<Record<string, unknown>>('/api/references/citation-meta', {
      source_path: sourcePath,
    }),
  citationMetaCached: (sourcePath: string) =>
    withCache(
      citationMetaCache,
      String(sourcePath || '').trim(),
      () => api.post<Record<string, unknown>>('/api/references/citation-meta', {
        source_path: sourcePath,
      }),
    ),
  bibliometrics: (meta: Record<string, unknown>) =>
    api.post<Record<string, unknown>>('/api/references/bibliometrics', {
      meta,
    }),
  bibliometricsCached: (meta: Record<string, unknown>) =>
    withCache(
      bibliometricsCache,
      stableStringify(meta),
      () => api.post<Record<string, unknown>>('/api/references/bibliometrics', {
        meta,
      }),
    ),
  repairShelfMetadata: (items: Array<Record<string, unknown>>, limit?: number) =>
    api.post<ShelfMetadataRepairResponse>('/api/references/shelf/metadata/repair', {
      items,
      limit: limit ?? items.length,
    }),
  citationCardPolishCached: (meta: Record<string, unknown>, waitSeconds = 4) => {
    const key = stableStringify({
      polish_client_version: 3,
      anchor: meta.anchor,
      num: meta.num,
      is_inpaper: meta.is_inpaper ?? meta.isInpaper,
      source_path: meta.source_path ?? meta.sourcePath,
      source_name: meta.source_name ?? meta.sourceName,
      title: meta.title ?? meta.card_title ?? meta.cardTitle,
      answer_claim: meta.answer_claim ?? meta.answerClaim ?? meta.card_claim ?? meta.cardClaim,
      evidence_quote: meta.evidence_quote ?? meta.evidenceQuote ?? meta.card_evidence ?? meta.cardEvidence,
      citation_context: meta.citation_context ?? meta.citationContext,
      card_takeaway: meta.card_takeaway ?? meta.cardTakeaway,
      card_context_summary: meta.card_context_summary ?? meta.cardContextSummary,
      card_reference_entry: meta.card_reference_entry ?? meta.cardReferenceEntry ?? meta.raw ?? meta.cite_fmt ?? meta.citeFmt,
      card_locator: meta.card_locator ?? meta.cardLocator ?? meta.location_label ?? meta.locationLabel,
    })
    const cached = citationCardPolishCache.get(key)
    if (cached) return cached
    const pending = api.post<Record<string, unknown>>('/api/references/citation-card-polish', {
      meta,
      wait_s: waitSeconds,
    }).then((result) => {
      const status = String(result?.citation_card_polish_status || result?.citationCardPolishStatus || '').trim().toLowerCase()
      if (status === 'pending') citationCardPolishCache.delete(key)
      else citationCardPolishCache.set(key, Promise.resolve(result))
      return result
    }).catch((err) => {
      citationCardPolishCache.delete(key)
      throw err
    })
    citationCardPolishCache.set(key, pending)
    return pending
  },
  readerDoc: (sourcePath: string) =>
    api.post<{
      ok: boolean
      source_path: string
      source_name: string
      md_path: string
      markdown: string
      anchors?: ReaderDocAnchor[]
      blocks?: ReaderDocBlock[]
    }>('/api/references/reader/doc', {
      source_path: sourcePath,
    }),
}
