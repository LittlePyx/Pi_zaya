import {
  citationDisplay,
  cleanCitationDisplayText,
  normalizeShelfItemKind,
  type CiteShelfItem,
} from './citationState'

const MAX_CONTEXT_ITEMS = 8
const MAX_TEXT_CHARS = 900
const MAX_TOTAL_CHARS = 4200

export interface SelectedResearchContextItem {
  key: string
  kind: string
  title: string
  sourceName: string
  sourcePath: string
  locationLabel: string
  refNum: number | null
  doi: string
  authors: string
  year: string
  summary: string
  excerpt: string
  note: string
}

export interface SelectedResearchContextPack {
  version: 1
  id: string
  source: 'citation_shelf'
  createdAt: number
  conversationId: string
  guideSourcePath: string
  guideSourceName: string
  itemCount: number
  tokenEstimate: number
  items: SelectedResearchContextItem[]
}

function asObject(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' ? value as Record<string, unknown> : null
}

function asText(value: unknown): string {
  if (typeof value === 'string') return value.trim()
  if (typeof value === 'number' && Number.isFinite(value)) return String(value)
  return ''
}

function asNumber(value: unknown): number {
  const num = Number(value)
  return Number.isFinite(num) ? num : 0
}

function firstText(...values: unknown[]): string {
  for (const value of values) {
    const text = asText(value)
    if (text) return text
  }
  return ''
}

function clipText(value: unknown, maxChars = MAX_TEXT_CHARS): string {
  const text = cleanCitationDisplayText(asText(value))
  if (!text) return ''
  if (text.length <= maxChars) return text
  if (maxChars <= 3) return text.slice(0, maxChars)
  return `${text.slice(0, Math.max(0, maxChars - 3)).trimEnd()}...`
}

function normalizeDoi(value: unknown): string {
  return asText(value)
    .replace(/^https?:\/\/(?:dx\.)?doi\.org\//i, '')
    .replace(/^[\s"'`([{<]+|[\s"'`)\]}>.,;:]+$/g, '')
    .trim()
}

function stablePackId(items: SelectedResearchContextItem[]): string {
  const seed = items.map((item) => item.key).join('|') || String(Date.now())
  let hash = 0
  for (let idx = 0; idx < seed.length; idx += 1) {
    hash = Math.imul(31, hash) + seed.charCodeAt(idx)
    hash |= 0
  }
  return `shelf_ctx_${Date.now().toString(36)}_${Math.abs(hash).toString(36)}`
}

function buildContextItem(item: CiteShelfItem): SelectedResearchContextItem | null {
  const display = citationDisplay(item)
  const title = clipText(firstText(item.title, item.cardTitle, display.main, item.main, item.raw), 240)
  const summary = clipText(firstText(
    item.summaryLine,
    item.cardTakeaway,
    item.cardContextSummary,
    item.whyLine,
    item.cardSupportExplanation,
  ))
  const excerpt = clipText(firstText(
    item.shelfExcerpt,
    item.evidenceQuote,
    item.cardEvidence,
    item.citationContext,
    item.cardReferenceEntry,
    item.raw,
    item.main,
  ))
  const note = clipText(item.note, 520)
  if (!title && !summary && !excerpt && !note) return null
  return {
    key: item.key,
    kind: normalizeShelfItemKind(item.shelfItemKind),
    title,
    sourceName: clipText(firstText(item.sourceName, display.source), 240),
    sourcePath: asText(item.sourcePath),
    locationLabel: clipText(firstText(item.locationLabel, item.cardLocator, item.headingPath), 240),
    refNum: Number.isFinite(Number(item.displayNum || item.num)) ? Number(item.displayNum || item.num) : null,
    doi: normalizeDoi(firstText(item.doi, item.doiUrl)),
    authors: clipText(firstText(item.authors, display.authors), 240),
    year: clipText(item.year, 24),
    summary,
    excerpt,
    note,
  }
}

export function normalizeSelectedResearchContextPack(raw: unknown): SelectedResearchContextPack | null {
  const obj = asObject(raw)
  if (!obj) return null
  const rawItems = Array.isArray(obj.items) ? obj.items : []
  const items = rawItems
    .map((rawItem) => {
      const itemObj = asObject(rawItem)
      if (!itemObj) return null
      const item: SelectedResearchContextItem = {
        key: clipText(itemObj.key, 160),
        kind: clipText(itemObj.kind, 40),
        title: clipText(itemObj.title, 240),
        sourceName: clipText(itemObj.sourceName, 240),
        sourcePath: asText(itemObj.sourcePath),
        locationLabel: clipText(itemObj.locationLabel, 240),
        refNum: asNumber(itemObj.refNum) > 0 ? Math.floor(asNumber(itemObj.refNum)) : null,
        doi: normalizeDoi(itemObj.doi),
        authors: clipText(itemObj.authors, 240),
        year: clipText(itemObj.year, 24),
        summary: clipText(itemObj.summary, 900),
        excerpt: clipText(itemObj.excerpt, 900),
        note: clipText(itemObj.note, 520),
      }
      if (!item.title && !item.summary && !item.excerpt && !item.note) return null
      return item
    })
    .filter((item): item is SelectedResearchContextItem => Boolean(item))
    .slice(0, MAX_CONTEXT_ITEMS)
  if (items.length <= 0) return null
  const totalChars = items.reduce((acc, item) => (
    acc
    + item.title.length
    + item.sourceName.length
    + item.locationLabel.length
    + item.summary.length
    + item.excerpt.length
    + item.note.length
  ), 0)
  return {
    version: 1,
    id: clipText(obj.id, 120) || stablePackId(items),
    source: 'citation_shelf',
    createdAt: asNumber(obj.createdAt) || Date.now(),
    conversationId: clipText(obj.conversationId, 120),
    guideSourcePath: asText(obj.guideSourcePath),
    guideSourceName: clipText(obj.guideSourceName, 240),
    itemCount: items.length,
    tokenEstimate: Math.max(1, Math.ceil(Math.max(totalChars, asNumber(obj.tokenEstimate) * 4, 1) / 4)),
    items,
  }
}

export function buildSelectedResearchContextPack(
  items: CiteShelfItem[],
  opts?: {
    conversationId?: string | null
    guideSourcePath?: string
    guideSourceName?: string
  },
): SelectedResearchContextPack | null {
  const seen = new Set<string>()
  const out: SelectedResearchContextItem[] = []
  for (const item of items) {
    const key = asText(item.key)
    if (!key || seen.has(key)) continue
    seen.add(key)
    const built = buildContextItem(item)
    if (!built) continue
    out.push(built)
    if (out.length >= MAX_CONTEXT_ITEMS) break
  }
  if (out.length <= 0) return null

  let totalChars = 0
  const clipped = out.map((item) => {
    const next = { ...item }
    for (const key of ['summary', 'excerpt', 'note'] as const) {
      const current = next[key]
      const remaining = MAX_TOTAL_CHARS - totalChars
      if (remaining <= 0) {
        next[key] = ''
        continue
      }
      next[key] = current.length > remaining
        ? (remaining > 3 ? `${current.slice(0, Math.max(0, remaining - 3)).trimEnd()}...` : current.slice(0, remaining))
        : current
      totalChars += next[key].length
    }
    totalChars += next.title.length + next.sourceName.length + next.locationLabel.length
    return next
  })

  return {
    version: 1,
    id: stablePackId(clipped),
    source: 'citation_shelf',
    createdAt: Date.now(),
    conversationId: asText(opts?.conversationId),
    guideSourcePath: asText(opts?.guideSourcePath),
    guideSourceName: asText(opts?.guideSourceName),
    itemCount: clipped.length,
    tokenEstimate: Math.max(1, Math.ceil(Math.max(1, totalChars) / 4)),
    items: clipped,
  }
}

export function buildSelectedResearchContextPackFromItems(
  items: SelectedResearchContextItem[],
  opts?: {
    conversationId?: string | null
    guideSourcePath?: string
    guideSourceName?: string
  },
): SelectedResearchContextPack | null {
  const normalized = normalizeSelectedResearchContextPack({
    id: '',
    source: 'citation_shelf',
    conversationId: opts?.conversationId || '',
    guideSourcePath: opts?.guideSourcePath || '',
    guideSourceName: opts?.guideSourceName || '',
    items,
  })
  if (!normalized) return null
  return {
    ...normalized,
    id: stablePackId(normalized.items),
    createdAt: Date.now(),
  }
}
