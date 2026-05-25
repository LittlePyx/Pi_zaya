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
  citationCardPolishCached: (meta: Record<string, unknown>, waitSeconds = 4) => {
    const key = stableStringify({
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
