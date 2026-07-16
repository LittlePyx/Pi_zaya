import { api, authFetch, responseError } from './client'
import { normalizeSourcePathForMatch } from '../utils/sourcePath'

const citationMetaCache = new Map<string, Promise<Record<string, unknown>>>()
type TimedReferenceCacheEntry = {
  expiresAt: number
  promise: Promise<Record<string, unknown>>
}

const bibliometricsCache = new Map<string, TimedReferenceCacheEntry>()
const citationCardPolishCache = new Map<string, Promise<Record<string, unknown>>>()
const BIBLIOMETRICS_READY_CACHE_MS = 5 * 60 * 1000
const BIBLIOMETRICS_EMPTY_CACHE_MS = 15 * 1000

export interface ReaderDocAnchor {
  anchor_id: string
  block_id?: string
  kind: string
  heading_path?: string
  heading_level?: number
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
  heading_level?: number
  order_index?: number
  line_start?: number
  line_end?: number
  text?: string
  raw_text?: string
  number?: number
}

export interface ReaderDocOutlineQuality {
  contract_version?: number
  ok?: boolean
  status?: string
  heading_count?: number
  has_document_title?: boolean
  max_heading_level?: number
  missing_heading_level_count?: number
  caption_heading_count?: number
  publisher_heading_count?: number
  issues?: string[]
}

export interface ReaderDocResponse {
  ok: boolean
  source_path: string
  source_name: string
  md_path: string
  doc_hash?: string
  outline_quality?: ReaderDocOutlineQuality
  markdown: string
  anchors?: ReaderDocAnchor[]
  blocks?: ReaderDocBlock[]
  cite_details?: Array<Record<string, unknown>>
  reference_cite_details?: Array<Record<string, unknown>>
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

export interface ShelfMetadataBackfillScanResponse {
  ok: boolean
  docs: number
  scanned: number
  ready: number
  export_ready: number
  needs_repair: number
  repairable: number
  retryable: number
  target_count: number
  returned_count?: number
  target_limit: number
  truncated: boolean
  missing_fields?: Array<{ name: string, count: number }>
  issue_codes?: Array<{ name: string, count: number }>
  sources?: Array<{ name: string, count: number }>
  targets: Array<Record<string, unknown>>
}

export interface ShelfMetadataBackfillResponse extends ShelfMetadataRepairResponse {
  scan?: ShelfMetadataBackfillScanResponse
  after_scan?: ShelfMetadataBackfillScanResponse
  preheated?: number
  remaining_targets?: number
}

export interface ShelfMetadataBackfillJobState {
  ok?: boolean
  job_id?: string
  status: 'idle' | 'running' | 'completed' | 'error' | string
  phase: string
  running: boolean
  limit?: number
  scan_limit?: number
  started_at?: number
  updated_at?: number
  finished_at?: number
  target_total?: number
  progress?: {
    percent?: number
    processed?: number
    total?: number
  }
  scan?: ShelfMetadataBackfillScanResponse
  after_scan?: ShelfMetadataBackfillScanResponse
  result?: ShelfMetadataBackfillResponse
  verification?: ShelfMetadataRepairVerification | Record<string, unknown>
  repair_run_id?: string
  repair_run?: Record<string, unknown>
  error_kind?: string
  error_detail?: string
}

export interface ShelfMetadataBackfillStartResponse {
  started: boolean
  reason?: string
  job_id?: string
  state: ShelfMetadataBackfillJobState
}

export type ReferenceSyncStatKey =
  | 'docs_total'
  | 'docs_indexed'
  | 'refs_total'
  | 'refs_with_doi'
  | 'refs_with_title'
  | 'refs_with_authors'
  | 'refs_with_venue'
  | 'refs_metadata_ready'
  | 'refs_metadata_user_ready'
  | 'refs_missing_doi'
  | 'refs_missing_title'
  | 'refs_missing_authors'
  | 'refs_missing_venue'
  | 'refs_unresolved'
  | 'refs_crossref_ok'
  | 'refs_source_map_ok'
  | 'refs_metadata_status_complete'
  | 'refs_metadata_status_crossref_enriched'
  | 'refs_metadata_status_bibliographic_ready'
  | 'refs_metadata_status_doi_sparse_refreshable'
  | 'refs_metadata_status_title_lookup_retryable'
  | 'refs_metadata_status_non_article_source_ok'
  | 'refs_metadata_status_no_doi_expected'
  | 'refs_metadata_status_truncated_reference'
  | 'refs_metadata_status_low_confidence_match'
  | 'refs_missing_reason_doi_sparse_refreshable'
  | 'refs_missing_reason_title_lookup_retryable'
  | 'refs_missing_reason_truncated_reference'
  | 'refs_missing_reason_low_confidence_match'
  | 'refs_action_auto_backfill'
  | 'refs_action_retry'
  | 'refs_action_source_repair'
  | 'refs_action_non_article_ok'
  | 'refs_action_retry_or_source_repair'
  | 'refs_web_source_ok'
  | 'crossref_network_attempts'
  | 'elapsed_s'

export type ReferenceSyncStats = Partial<Record<ReferenceSyncStatKey, number | string | boolean | null>> & {
  [key: string]: number | string | boolean | null | undefined
}

export interface ReferenceSyncStatusEvent {
  running?: boolean
  status?: string
  stage?: string
  message?: string
  error?: string
  current?: string
  docs_done?: number | string
  docs_total?: number | string
  run_id?: number | string
  stats?: ReferenceSyncStats | Record<string, unknown>
  done?: boolean
  [key: string]: unknown
}

function stableStringify(value: unknown): string {
  if (value === null || value === undefined) return ''
  if (typeof value !== 'object') return JSON.stringify(value)
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(',')}]`
  const rec = value as Record<string, unknown>
  return `{${Object.keys(rec).sort().map((key) => `${JSON.stringify(key)}:${stableStringify(rec[key])}`).join(',')}}`
}

function normalizedBibliometricsDoi(meta: Record<string, unknown>): string {
  return String(meta.doi || meta.doi_url || meta.doiUrl || '')
    .trim()
    .replace(/^https?:\/\/(?:dx\.)?doi\.org\//i, '')
    .replace(/^doi\s*:\s*/i, '')
    .replace(/[?#].*$/, '')
    .replace(/[\s.,;]+$/, '')
    .toLowerCase()
}

export function bibliometricsCacheKey(meta: Record<string, unknown>): string {
  const locale = String(
    meta.refs_card_locale || meta.target_locale || meta.ui_locale || meta.summary_locale || '',
  ).trim().toLowerCase()
  const doi = normalizedBibliometricsDoi(meta)
  if (doi) return stableStringify({ bibliometrics_client_version: 4, identity: `doi:${doi}`, locale })

  const title = String(meta.title || meta.card_title || meta.cardTitle || '')
    .trim()
    .toLowerCase()
    .replace(/\s+/g, ' ')
  const year = String(meta.year || '').trim()
  const source = String(meta.source_path || meta.sourcePath || '').trim().toLowerCase()
  return stableStringify({
    bibliometrics_client_version: 4,
    identity: title ? `title:${title}|year:${year}` : `source:${source}`,
    locale,
  })
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

function bibliometricsHasSummary(result: Record<string, unknown>): boolean {
  return Boolean(String(result.summary_line || result.summaryLine || '').trim())
}

function withTimedBibliometricsCache(
  key: string,
  loader: () => Promise<Record<string, unknown>>,
): Promise<Record<string, unknown>> {
  const now = Date.now()
  const cached = bibliometricsCache.get(key)
  if (cached && cached.expiresAt > now) return cached.promise
  if (cached) bibliometricsCache.delete(key)

  const entry: TimedReferenceCacheEntry = {
    expiresAt: Number.POSITIVE_INFINITY,
    promise: Promise.resolve({}),
  }
  entry.promise = loader()
    .then((result) => {
      if (bibliometricsCache.get(key) === entry) {
        entry.expiresAt = Date.now() + (
          bibliometricsHasSummary(result)
            ? BIBLIOMETRICS_READY_CACHE_MS
            : BIBLIOMETRICS_EMPTY_CACHE_MS
        )
      }
      return result
    })
    .catch((err) => {
      if (bibliometricsCache.get(key) === entry) bibliometricsCache.delete(key)
      throw err
    })
  bibliometricsCache.set(key, entry)
  return entry.promise
}

export function referenceSourcePathCacheKey(sourcePath: unknown): string {
  return normalizeSourcePathForMatch(sourcePath) || String(sourcePath || '').trim()
}

export const referencesApi = {
  startSync: () =>
    api.post<{ started: boolean; reason?: string; run_id?: number }>('/api/references/sync'),
  streamSyncStatus: (
    onData: (data: ReferenceSyncStatusEvent) => void,
    onDone: () => void,
    onError?: (err: unknown) => void,
  ): AbortController => {
    const ctrl = new AbortController()
    ;(async () => {
      try {
        const res = await authFetch('/api/references/sync/status', { signal: ctrl.signal })
        if (!res.ok) throw await responseError(res, 'reference sync status failed')
        if (!res.body) throw new Error('reference sync status stream is empty')
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
              const data = JSON.parse(line.slice(6)) as ReferenceSyncStatusEvent
              onData(data)
              if (data.done === true) { onDone(); return }
            } catch { /* skip bad JSON */ }
          }
        }
        throw new Error('reference sync status stream ended before completion')
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
      referenceSourcePathCacheKey(sourcePath),
      () => api.post<Record<string, unknown>>('/api/references/citation-meta', {
        source_path: sourcePath,
      }),
    ),
  bibliometrics: (meta: Record<string, unknown>) =>
    api.post<Record<string, unknown>>('/api/references/bibliometrics', {
      meta,
    }),
  bibliometricsCached: (meta: Record<string, unknown>) =>
    withTimedBibliometricsCache(
      bibliometricsCacheKey(meta),
      () => api.post<Record<string, unknown>>('/api/references/bibliometrics', {
        meta,
      }),
    ),
  repairShelfMetadata: (items: Array<Record<string, unknown>>, limit?: number) =>
    api.post<ShelfMetadataRepairResponse>('/api/references/shelf/metadata/repair', {
      items,
      limit: limit ?? items.length,
    }),
  scanShelfMetadataBackfill: (limit = 120) =>
    api.get<ShelfMetadataBackfillScanResponse>(`/api/references/shelf/metadata/backfill/scan?limit=${encodeURIComponent(String(limit))}`),
  backfillShelfMetadata: (limit = 40, scanLimit = 240) =>
    api.post<ShelfMetadataBackfillResponse>('/api/references/shelf/metadata/backfill', {
      limit,
      scan_limit: scanLimit,
    }),
  shelfMetadataBackfillStatus: () =>
    api.get<ShelfMetadataBackfillJobState>('/api/references/shelf/metadata/backfill/status'),
  startShelfMetadataBackfill: (limit = 40, scanLimit = 240) =>
    api.post<ShelfMetadataBackfillStartResponse>('/api/references/shelf/metadata/backfill/start', {
      limit,
      scan_limit: scanLimit,
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
  readerDoc: (sourcePath: string, init?: RequestInit) =>
    api.post<ReaderDocResponse>('/api/references/reader/doc', {
      source_path: sourcePath,
    }, init),
}
