import { useEffect, useLayoutEffect, useMemo, useRef, useState, type MouseEvent } from 'react'
import { Typography, message } from 'antd'
import { UserOutlined } from '@ant-design/icons'
import { MarkdownRenderer } from './MarkdownRenderer'
import { CopyBar } from './CopyBar'
import { CitationPopover } from './CitationPopover'
import { CiteShelf } from './CiteShelf'
import {
  mergeCiteMeta,
  normalizeCiteDetail,
  normalizeShelfNote,
  normalizeShelfTags,
  shelfStorageKey,
  toShelfItem,
  type CiteDetail,
  type CiteShelfItem,
} from './citationState'
import { RefsPanel } from '../refs/RefsPanel'
import type { ChatImageAttachment, Message } from '../../api/chat'
import { referencesApi } from '../../api/references'

const { Text } = Typography
const SHELF_MAX_ITEMS = 120
const SHELF_SCHEMA_VERSION = 4
const SHELF_SAVED_SCHEMA_VERSION = 1
const SHELF_SAVED_MAX_ITEMS = 16
const SHELF_SAVED_SUFFIX = ':saved_snapshots'

interface Props {
  activeConvId?: string | null
  messages: Message[]
  refs: Record<string, unknown>
  generationPartial?: string
  generationStage?: string
  jumpTarget?: { messageId: number; token: number } | null
}

function hasRenderableRefs(refs: Record<string, unknown>, msgId: number) {
  const entry = refs[String(msgId)] as { hits?: Array<{ meta?: Record<string, unknown> }> } | undefined
  if (!entry) return false
  const hits = Array.isArray(entry.hits) ? entry.hits : []
  if (hits.length > 0) return true
  return hits.some((hit) => String(hit?.meta?.ref_pack_state || '').trim().toLowerCase() === 'pending')
}

function isImageOnlyPlaceholder(content: string) {
  return /^\[Image attachment x\d+\]$/i.test(String(content || '').trim())
}

function imageAttachmentsOf(message: Message): ChatImageAttachment[] {
  return Array.isArray(message.attachments)
    ? message.attachments.filter((item): item is ChatImageAttachment => Boolean(item && item.path))
    : []
}

function AssistantAvatar() {
  return (
    <div className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center overflow-hidden rounded-full border border-[var(--border)] bg-white/90 dark:bg-white/5">
      <img src="/pi_logo.png" alt="Pi assistant" className="h-5 w-5 object-contain" loading="lazy" />
    </div>
  )
}

const shelfStorageFallback = new Map<string, string>()
interface ShelfSnapshot {
  version: number
  revision: number
  updatedAt: number
  open: boolean
  items: CiteShelfItem[]
}
interface ShelfSavedSnapshot {
  id: string
  name: string
  createdAt: number
  items: CiteShelfItem[]
}
interface ShelfSavedSnapshotPayload {
  version: number
  updatedAt: number
  snapshots: ShelfSavedSnapshot[]
}
const shelfSnapshotMemory = new Map<string, ShelfSnapshot>()
const shelfSavedSnapshotMemory = new Map<string, ShelfSavedSnapshotPayload>()

function cloneShelfSnapshot(snapshot: ShelfSnapshot): ShelfSnapshot {
  return {
    version: Number(snapshot.version || 0),
    revision: Number(snapshot.revision || 0),
    updatedAt: Number(snapshot.updatedAt || 0),
    open: Boolean(snapshot.open),
    items: (snapshot.items || []).map((item) => ({ ...item })),
  }
}

function cloneSavedSnapshot(snapshot: ShelfSavedSnapshot): ShelfSavedSnapshot {
  return {
    id: String(snapshot.id || ''),
    name: String(snapshot.name || ''),
    createdAt: Number(snapshot.createdAt || 0),
    items: (snapshot.items || []).map((item) => ({ ...item })),
  }
}

function cloneSavedSnapshotPayload(payload: ShelfSavedSnapshotPayload): ShelfSavedSnapshotPayload {
  return {
    version: Number(payload.version || 0),
    updatedAt: Number(payload.updatedAt || 0),
    snapshots: (payload.snapshots || []).map((entry) => cloneSavedSnapshot(entry)),
  }
}

function listShelfStorages(): Storage[] {
  const out: Storage[] = []
  try {
    out.push(window.localStorage)
  } catch {
    // ignore
  }
  try {
    if (!out.includes(window.sessionStorage)) out.push(window.sessionStorage)
  } catch {
    // ignore
  }
  return out
}

function readShelfStorage(key: string): string {
  for (const storage of listShelfStorages()) {
    try {
      const raw = storage.getItem(key)
      if (typeof raw === 'string') {
        shelfStorageFallback.set(key, raw)
        return raw
      }
    } catch {
      // ignore
    }
  }
  return shelfStorageFallback.get(key) || ''
}

function writeShelfStorage(key: string, payload: string) {
  // Always keep in-memory raw snapshot first to survive temporary storage failures.
  shelfStorageFallback.set(key, payload)
  let wrote = false
  for (const storage of listShelfStorages()) {
    try {
      storage.setItem(key, payload)
      wrote = true
    } catch {
      // ignore
    }
  }
  if (!wrote) return
}

function removeShelfStorage(key: string) {
  shelfSnapshotMemory.delete(key)
  for (const storage of listShelfStorages()) {
    try {
      storage.removeItem(key)
    } catch {
      // ignore
    }
  }
  shelfStorageFallback.delete(key)
}

function shelfSavedStorageKey(convId?: string | null): string {
  return `${shelfStorageKey(convId)}${SHELF_SAVED_SUFFIX}`
}

function normalizeDoiLike(value: string): string {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/^https?:\/\/(?:dx\.)?doi\.org\//i, '')
    .replace(/^[\s"'`([{<]+|[\s"'`)\]}>.,;:]+$/g, '')
    .trim()
}

function normalizeTitleLike(value: string): string {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9\u4e00-\u9fff]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function shelfPaperIdentity(item: CiteShelfItem): string {
  const doi = normalizeDoiLike(item.doi || item.doiUrl)
  if (doi) return `doi:${doi}`
  const title = normalizeTitleLike(item.title || item.main)
  const year = /^\d{4}$/.test(String(item.year || '').trim()) ? String(item.year).trim() : ''
  if (title) return `title:${title}|${year}`
  return `key:${String(item.key || '').trim()}`
}

function dedupeShelfItems(items: CiteShelfItem[]): CiteShelfItem[] {
  const seen = new Set<string>()
  const out: CiteShelfItem[] = []
  for (const item of items || []) {
    const key = shelfPaperIdentity(item)
    if (!key || seen.has(key)) continue
    seen.add(key)
    out.push(item)
    if (out.length >= SHELF_MAX_ITEMS) break
  }
  return out
}

function restoreShelfItems(rawItems: unknown[]): CiteShelfItem[] {
  const seen = new Set<string>()
  const seenIdentity = new Set<string>()
  const out: CiteShelfItem[] = []
  for (const raw of rawItems) {
    if (!raw || typeof raw !== 'object') continue
    const rec = raw as Record<string, unknown>
    const detail = normalizeCiteDetail(rec)
    if (!detail) continue
    const base = toShelfItem(detail)
    const key = String(rec.key || '').trim() || base.key
    const main = String(rec.main || '').trim() || base.main
    if (!key || seen.has(key)) continue
    const identity = shelfPaperIdentity({ ...base, key, main })
    if (seenIdentity.has(identity)) continue
    seen.add(key)
    seenIdentity.add(identity)
    out.push({
      ...base,
      key,
      main,
      tags: normalizeShelfTags(rec.tags),
      note: normalizeShelfNote(rec.note),
    })
    if (out.length >= SHELF_MAX_ITEMS) break
  }
  return out
}

function readShelfSnapshot(key: string, rawOverride?: string): ShelfSnapshot | null {
  if (typeof rawOverride !== 'string') {
    const mem = shelfSnapshotMemory.get(key)
    if (mem) return cloneShelfSnapshot(mem)
  }
  const raw = typeof rawOverride === 'string' ? rawOverride : readShelfStorage(key)
  if (!raw) return null
  try {
    const parsed = JSON.parse(raw)
    const itemsRaw: unknown[] = Array.isArray(parsed?.items) ? parsed.items : []
    const revision0 = Number(parsed?.revision || 0)
    const updatedAt0 = Number(parsed?.updatedAt || 0)
    const snapshot: ShelfSnapshot = {
      version: Number(parsed?.version || 0) || 0,
      revision: Number.isFinite(revision0) && revision0 > 0 ? Math.floor(revision0) : 0,
      updatedAt: Number.isFinite(updatedAt0) && updatedAt0 > 0 ? Math.floor(updatedAt0) : 0,
      open: Boolean(parsed?.open),
      items: restoreShelfItems(itemsRaw),
    }
    shelfSnapshotMemory.set(key, snapshot)
    return cloneShelfSnapshot(snapshot)
  } catch {
    // Corrupted payload: keep running and clear stale bad data.
    shelfSnapshotMemory.delete(key)
    removeShelfStorage(key)
    return null
  }
}

function removeSavedShelfStorage(key: string) {
  shelfSavedSnapshotMemory.delete(key)
  for (const storage of listShelfStorages()) {
    try {
      storage.removeItem(key)
    } catch {
      // ignore
    }
  }
  shelfStorageFallback.delete(key)
}

function readSavedShelfSnapshots(storageKey: string, rawOverride?: string): ShelfSavedSnapshot[] {
  if (typeof rawOverride !== 'string') {
    const mem = shelfSavedSnapshotMemory.get(storageKey)
    if (mem) return cloneSavedSnapshotPayload(mem).snapshots
  }
  const raw = typeof rawOverride === 'string' ? rawOverride : readShelfStorage(storageKey)
  if (!raw) return []
  try {
    const parsed = JSON.parse(raw)
    const snapshotsRaw: unknown[] = Array.isArray(parsed?.snapshots) ? parsed.snapshots : []
    const seen = new Set<string>()
    const snapshots: ShelfSavedSnapshot[] = []
    for (const rawItem of snapshotsRaw) {
      if (!rawItem || typeof rawItem !== 'object') continue
      const rec = rawItem as Record<string, unknown>
      const id = String(rec.id || '').trim()
      if (!id || seen.has(id)) continue
      const createdAt0 = Number(rec.createdAt || 0)
      const createdAt = Number.isFinite(createdAt0) && createdAt0 > 0 ? Math.floor(createdAt0) : Date.now()
      const name = String(rec.name || '').trim() || '未命名快照'
      const itemsRaw: unknown[] = Array.isArray(rec.items) ? rec.items : []
      snapshots.push({
        id,
        name,
        createdAt,
        items: restoreShelfItems(itemsRaw),
      })
      seen.add(id)
      if (snapshots.length >= SHELF_SAVED_MAX_ITEMS) break
    }
    const payload: ShelfSavedSnapshotPayload = {
      version: Number(parsed?.version || 0) || 0,
      updatedAt: Number(parsed?.updatedAt || 0) || 0,
      snapshots,
    }
    shelfSavedSnapshotMemory.set(storageKey, payload)
    return cloneSavedSnapshotPayload(payload).snapshots
  } catch {
    removeSavedShelfStorage(storageKey)
    return []
  }
}

function persistSavedShelfSnapshots(storageKey: string, snapshots: ShelfSavedSnapshot[]) {
  if (!Array.isArray(snapshots) || snapshots.length <= 0) {
    removeSavedShelfStorage(storageKey)
    return
  }
  const normalized = snapshots
    .slice(0, SHELF_SAVED_MAX_ITEMS)
    .map((entry) => ({
      id: String(entry.id || '').trim(),
      name: String(entry.name || '').trim() || '未命名快照',
      createdAt: Number(entry.createdAt || 0) > 0 ? Number(entry.createdAt) : Date.now(),
      items: dedupeShelfItems(entry.items || []).slice(0, SHELF_MAX_ITEMS).map((item) => ({ ...item })),
    }))
    .filter((entry) => Boolean(entry.id))
  if (normalized.length <= 0) {
    removeSavedShelfStorage(storageKey)
    return
  }
  const payload: ShelfSavedSnapshotPayload = {
    version: SHELF_SAVED_SCHEMA_VERSION,
    updatedAt: Date.now(),
    snapshots: normalized,
  }
  shelfSavedSnapshotMemory.set(storageKey, cloneSavedSnapshotPayload(payload))
  writeShelfStorage(storageKey, JSON.stringify(payload))
}

function isWeakTitle(text: string): boolean {
  const t = String(text || '').trim()
  if (!t) return true
  if (/(?:\bet\s+al\b|doi[:\s]|^https?:\/\/)/i.test(t)) return true
  if (/\bIn\s+[A-Z]/.test(t)) return true
  const tokens = t.match(/[A-Za-z0-9\u4e00-\u9fff]+/g) || []
  return tokens.length <= 2
}

function isWeakAuthors(text: string): boolean {
  const t = String(text || '').trim()
  if (!t) return true
  if (/\b(?:journal|conference|proceedings|vol\.?|pp\.?)\b/i.test(t)) return true
  const tokens = t.match(/[A-Za-z\u4e00-\u9fff]+/g) || []
  return tokens.length <= 1
}

function isWeakVenue(text: string): boolean {
  const t = String(text || '').trim()
  if (!t) return true
  const tokens = t.match(/[A-Za-z0-9\u4e00-\u9fff]+/g) || []
  return tokens.length <= 1
}

function preferRicherField(field: 'title' | 'authors' | 'venue' | 'year' | 'main', current: string, incoming: string): string {
  const cur = String(current || '').trim()
  const inc = String(incoming || '').trim()
  if (!cur) return inc
  if (!inc) return cur
  if (field === 'year') {
    const curOk = /^\d{4}$/.test(cur)
    const incOk = /^\d{4}$/.test(inc)
    if (curOk && !incOk) return cur
    if (!curOk && incOk) return inc
    return cur
  }
  if (field === 'title') {
    const curWeak = isWeakTitle(cur)
    const incWeak = isWeakTitle(inc)
    if (curWeak && !incWeak) return inc
    if (!curWeak && incWeak) return cur
  } else if (field === 'authors') {
    const curWeak = isWeakAuthors(cur)
    const incWeak = isWeakAuthors(inc)
    if (curWeak && !incWeak) return inc
    if (!curWeak && incWeak) return cur
  } else if (field === 'venue') {
    const curWeak = isWeakVenue(cur)
    const incWeak = isWeakVenue(inc)
    if (curWeak && !incWeak) return inc
    if (!curWeak && incWeak) return cur
  } else if (field === 'main') {
    const curWeak = isWeakTitle(cur)
    const incWeak = isWeakTitle(inc)
    if (curWeak && !incWeak) return inc
    if (!curWeak && incWeak) return cur
  }
  return inc.length > cur.length + 12 ? inc : cur
}

function sameTags(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) return false
  }
  return true
}

function sameShelfItem(a: CiteShelfItem, b: CiteShelfItem): boolean {
  return (
    a.key === b.key
    && a.main === b.main
    && a.traceConvId === b.traceConvId
    && a.traceAssistantMsgId === b.traceAssistantMsgId
    && a.traceAssistantOrder === b.traceAssistantOrder
    && a.traceUserMsgId === b.traceUserMsgId
    && a.sourceName === b.sourceName
    && a.sourcePath === b.sourcePath
    && a.raw === b.raw
    && a.citeFmt === b.citeFmt
    && a.title === b.title
    && a.authors === b.authors
    && a.venue === b.venue
    && a.year === b.year
    && a.volume === b.volume
    && a.issue === b.issue
    && a.pages === b.pages
    && a.doi === b.doi
    && a.doiUrl === b.doiUrl
    && a.citationCount === b.citationCount
    && a.citationSource === b.citationSource
    && a.journalIf === b.journalIf
    && a.journalQuartile === b.journalQuartile
    && a.journalIfSource === b.journalIfSource
    && a.venueKind === b.venueKind
    && a.venueVerifiedBy === b.venueVerifiedBy
    && a.openalexVenue === b.openalexVenue
    && a.conferenceTier === b.conferenceTier
    && a.conferenceRankSource === b.conferenceRankSource
    && a.conferenceCcf === b.conferenceCcf
    && a.conferenceCcfSource === b.conferenceCcfSource
    && a.conferenceName === b.conferenceName
    && a.conferenceAcronym === b.conferenceAcronym
    && a.num === b.num
    && a.anchor === b.anchor
    && a.bibliometricsChecked === b.bibliometricsChecked
    && a.summaryLine === b.summaryLine
    && a.summarySource === b.summarySource
    && a.note === b.note
    && sameTags(a.tags || [], b.tags || [])
  )
}

function sameShelfItems(a: CiteShelfItem[], b: CiteShelfItem[]): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i += 1) {
    if (!sameShelfItem(a[i], b[i])) return false
  }
  return true
}

function persistShelfSnapshot(
  storageKey: string,
  payload: { open: boolean; items: CiteShelfItem[] },
  currentRevision: number,
): number {
  const normalizedItems = payload.items.slice(0, SHELF_MAX_ITEMS)
  const existing = readShelfSnapshot(storageKey)
  if (existing && existing.open === payload.open && sameShelfItems(existing.items, normalizedItems)) {
    return Math.max(currentRevision, existing.revision)
  }
  const nextRevision = Math.max(currentRevision, existing?.revision || 0) + 1
  const snapshot: ShelfSnapshot = {
    version: SHELF_SCHEMA_VERSION,
    revision: nextRevision,
    updatedAt: Date.now(),
    open: payload.open,
    items: normalizedItems,
  }
  shelfSnapshotMemory.set(storageKey, cloneShelfSnapshot(snapshot))
  const raw = JSON.stringify(snapshot)
  writeShelfStorage(storageKey, raw)
  return nextRevision
}

function preferExistingText(current: string, incoming: string): string {
  const cur = String(current || '').trim()
  if (cur) return cur
  return String(incoming || '').trim()
}

function preferPositiveNumber(current: number, incoming: number): number {
  const cur = Number(current || 0)
  if (Number.isFinite(cur) && cur > 0) return cur
  const inc = Number(incoming || 0)
  if (Number.isFinite(inc) && inc > 0) return inc
  return 0
}

function mergeShelfItemWithLive(item: CiteShelfItem, live: CiteShelfItem): CiteShelfItem {
  const mergedLike = {
    ...item,
    traceConvId: preferExistingText(item.traceConvId, live.traceConvId),
    traceAssistantMsgId: preferPositiveNumber(item.traceAssistantMsgId, live.traceAssistantMsgId),
    traceAssistantOrder: preferPositiveNumber(item.traceAssistantOrder, live.traceAssistantOrder),
    traceUserMsgId: preferPositiveNumber(item.traceUserMsgId, live.traceUserMsgId),
    sourceName: preferExistingText(item.sourceName, live.sourceName),
    sourcePath: preferExistingText(item.sourcePath, live.sourcePath),
    raw: preferExistingText(item.raw, live.raw),
    citeFmt: preferExistingText(item.citeFmt, live.citeFmt),
    title: preferRicherField('title', item.title, live.title),
    authors: preferRicherField('authors', item.authors, live.authors),
    venue: preferRicherField('venue', item.venue, live.venue),
    year: preferRicherField('year', item.year, live.year),
    volume: preferExistingText(item.volume, live.volume),
    issue: preferExistingText(item.issue, live.issue),
    pages: preferExistingText(item.pages, live.pages),
    doi: preferExistingText(item.doi, live.doi),
    doiUrl: preferExistingText(item.doiUrl, live.doiUrl),
    citationSource: preferExistingText(item.citationSource, live.citationSource),
    venueKind: preferExistingText(item.venueKind, live.venueKind),
    venueVerifiedBy: preferExistingText(item.venueVerifiedBy, live.venueVerifiedBy),
    openalexVenue: preferExistingText(item.openalexVenue, live.openalexVenue),
    journalIf: preferExistingText(item.journalIf, live.journalIf),
    journalQuartile: preferExistingText(item.journalQuartile, live.journalQuartile),
    journalIfSource: preferExistingText(item.journalIfSource, live.journalIfSource),
    conferenceTier: preferExistingText(item.conferenceTier, live.conferenceTier),
    conferenceRankSource: preferExistingText(item.conferenceRankSource, live.conferenceRankSource),
    conferenceCcf: preferExistingText(item.conferenceCcf, live.conferenceCcf),
    conferenceCcfSource: preferExistingText(item.conferenceCcfSource, live.conferenceCcfSource),
    conferenceName: preferExistingText(item.conferenceName, live.conferenceName),
    conferenceAcronym: preferExistingText(item.conferenceAcronym, live.conferenceAcronym),
    summaryLine: preferRicherField('title', item.summaryLine, live.summaryLine),
    summarySource: preferExistingText(item.summarySource, live.summarySource),
    citationCount: preferPositiveNumber(item.citationCount, live.citationCount),
    num: preferPositiveNumber(item.num, live.num),
    bibliometricsChecked: Boolean(item.bibliometricsChecked || live.bibliometricsChecked),
  }

  const normalized = normalizeCiteDetail(mergedLike) || item
  const autoMain = toShelfItem(normalized).main
  return {
    ...item,
    ...normalized,
    key: item.key,
    main: preferRicherField('main', item.main, preferExistingText(live.main, autoMain)),
    tags: normalizeShelfTags(item.tags),
    note: normalizeShelfNote(item.note),
  }
}

function normalizeTextLite(value: string): string {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9\u4e00-\u9fff]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function firstAuthorToken(value: string): string {
  const firstChunk = String(value || '')
    .split(/[;,，；]/)[0]
    ?.trim()
    || ''
  const tokens = firstChunk.match(/[A-Za-z\u4e00-\u9fff]+/g) || []
  if (tokens.length <= 0) return ''
  const first = tokens.at(0)
  return first ? first.toLowerCase() : ''
}

function year4(value: string): string {
  const match = String(value || '').match(/\b(19|20)\d{2}\b/)
  return match ? match[0] : ''
}

function jaccardTokens(a: string, b: string): number {
  const aSet = new Set(normalizeTextLite(a).split(' ').filter(Boolean))
  const bSet = new Set(normalizeTextLite(b).split(' ').filter(Boolean))
  if (aSet.size <= 0 || bSet.size <= 0) return 0
  let inter = 0
  for (const t of aSet) {
    if (bSet.has(t)) inter += 1
  }
  const union = aSet.size + bSet.size - inter
  return union > 0 ? inter / union : 0
}

function strictRepairMerge(base: CiteShelfItem, candidateMeta: Record<string, unknown>): CiteShelfItem | null {
  if (!candidateMeta || Object.keys(candidateMeta).length <= 0) return null
  const merged = mergeCiteMeta(base, candidateMeta)
  const mergedItem = {
    ...toShelfItem(merged),
    key: base.key,
    tags: normalizeShelfTags(base.tags),
    note: normalizeShelfNote(base.note),
  }

  const baseDoi = normalizeDoiLike(base.doi || base.doiUrl)
  const mergedDoi = normalizeDoiLike(mergedItem.doi || mergedItem.doiUrl)
  if (baseDoi && mergedDoi && baseDoi !== mergedDoi) return null

  const titleSignal = jaccardTokens(base.title || base.main, mergedItem.title || mergedItem.main) >= 0.55
  const authorSignal = (
    Boolean(firstAuthorToken(base.authors))
    && firstAuthorToken(base.authors) === firstAuthorToken(mergedItem.authors)
  )
  const yearSignal = Boolean(year4(base.year) && year4(base.year) === year4(mergedItem.year))
  const venueSignal = jaccardTokens(base.venue, mergedItem.venue) >= 0.5
  const newDoiSignal = !baseDoi && Boolean(mergedDoi)

  let signalCount = 0
  if (titleSignal) signalCount += 1
  if (authorSignal) signalCount += 1
  if (yearSignal) signalCount += 1
  if (venueSignal) signalCount += 1

  const accepted = newDoiSignal ? signalCount >= 1 : signalCount >= 2
  if (!accepted) return null
  return mergedItem
}

function snapshotDiffCounts(currentItems: CiteShelfItem[], baselineItems: CiteShelfItem[]): { added: number; removed: number } {
  const current = new Set(currentItems.map((item) => shelfPaperIdentity(item)))
  const baseline = new Set(baselineItems.map((item) => shelfPaperIdentity(item)))
  let added = 0
  let removed = 0
  for (const id of current) {
    if (!baseline.has(id)) added += 1
  }
  for (const id of baseline) {
    if (!current.has(id)) removed += 1
  }
  return { added, removed }
}

export function MessageList({ activeConvId, messages, refs, generationPartial, generationStage, jumpTarget }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const [popoverDetail, setPopoverDetail] = useState<CiteDetail | null>(null)
  const [popoverPos, setPopoverPos] = useState<{ x: number; y: number } | null>(null)
  const [popoverLoading, setPopoverLoading] = useState(false)
  const [shelfOpen, setShelfOpen] = useState(false)
  const [shelfItems, setShelfItems] = useState<CiteShelfItem[]>([])
  const [focusedShelfKey, setFocusedShelfKey] = useState('')
  const [shelfSummaryLoadingKey, setShelfSummaryLoadingKey] = useState('')
  const [shelfRepairLoadingKey, setShelfRepairLoadingKey] = useState('')
  const [savedShelfSnapshots, setSavedShelfSnapshots] = useState<ShelfSavedSnapshot[]>([])
  const [selectedSavedSnapshotId, setSelectedSavedSnapshotId] = useState('')
  const skipShelfPersistOnceRef = useRef(false)
  const persistShelfTimerRef = useRef<number | null>(null)
  const activeStorageKeyRef = useRef(shelfStorageKey(activeConvId))
  const shelfRevisionByKeyRef = useRef<Record<string, number>>({})
  const latestShelfStateRef = useRef<{ convId?: string | null; open: boolean; items: CiteShelfItem[] }>({
    convId: activeConvId,
    open: false,
    items: [],
  })

  useLayoutEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const timer = window.requestAnimationFrame(() => {
      el.scrollTop = el.scrollHeight
    })
    return () => window.cancelAnimationFrame(timer)
  }, [activeConvId, generationPartial])

  useEffect(() => {
    if (!jumpTarget || !Number.isFinite(jumpTarget.messageId)) return
    const el = scrollRef.current
    if (!el) return
    const target = el.querySelector<HTMLElement>(`[data-msg-id="${jumpTarget.messageId}"]`)
    if (!target) return
    const targetRect = target.getBoundingClientRect()
    const containerRect = el.getBoundingClientRect()
    const top = targetRect.top - containerRect.top + el.scrollTop - 12
    el.scrollTo({ top: Math.max(0, top), behavior: 'smooth' })
    try {
      target.animate(
        [
          { boxShadow: '0 0 0 0 rgba(24,144,255,0.0)', backgroundColor: 'rgba(24,144,255,0.0)' },
          { boxShadow: '0 0 0 3px rgba(24,144,255,0.24)', backgroundColor: 'rgba(24,144,255,0.10)' },
          { boxShadow: '0 0 0 0 rgba(24,144,255,0.0)', backgroundColor: 'rgba(24,144,255,0.0)' },
        ],
        { duration: 900, easing: 'ease-out' },
      )
    } catch {
      // no-op: Web Animations may not be available in all envs.
    }
  }, [jumpTarget, messages])

  useEffect(() => {
    const nextStorageKey = shelfStorageKey(activeConvId)
    const nextSavedStorageKey = shelfSavedStorageKey(activeConvId)
    const prevStorageKey = activeStorageKeyRef.current
    if (persistShelfTimerRef.current !== null) {
      window.clearTimeout(persistShelfTimerRef.current)
      persistShelfTimerRef.current = null
    }
    if (prevStorageKey !== nextStorageKey) {
      const prevRevision = Number(shelfRevisionByKeyRef.current[prevStorageKey] || 0)
      const flushedRevision = persistShelfSnapshot(
        prevStorageKey,
        { open: shelfOpen, items: shelfItems },
        prevRevision,
      )
      shelfRevisionByKeyRef.current[prevStorageKey] = flushedRevision
    }

    // Switching conversation changes storage key; skip one persist cycle to avoid
    // writing previous conversation shelf state into the new key before hydration.
    skipShelfPersistOnceRef.current = true
    const savedSnapshots = readSavedShelfSnapshots(nextSavedStorageKey)
    setSavedShelfSnapshots(savedSnapshots)
    setSelectedSavedSnapshotId((current) => {
      if (current && savedSnapshots.some((item) => item.id === current)) return current
      return savedSnapshots[0]?.id || ''
    })
    const snapshot = readShelfSnapshot(nextStorageKey)
    if (!snapshot) {
      shelfRevisionByKeyRef.current[nextStorageKey] = 0
      setShelfItems([])
      setShelfOpen(false)
      setFocusedShelfKey('')
      activeStorageKeyRef.current = nextStorageKey
      return
    }
    shelfRevisionByKeyRef.current[nextStorageKey] = Math.max(0, snapshot.revision || 0)
    setShelfItems(snapshot.items)
    setShelfOpen(snapshot.open)
    setFocusedShelfKey('')
    activeStorageKeyRef.current = nextStorageKey
  }, [activeConvId])

  useEffect(() => {
    const storageKey = shelfStorageKey(activeConvId)
    const savedStorageKey = shelfSavedStorageKey(activeConvId)
    const onStorage = (event: StorageEvent) => {
      if (event.key === savedStorageKey) {
        if (event.newValue === null) {
          shelfSavedSnapshotMemory.delete(savedStorageKey)
          setSavedShelfSnapshots([])
          setSelectedSavedSnapshotId('')
          return
        }
        const snapshots = readSavedShelfSnapshots(savedStorageKey, event.newValue)
        setSavedShelfSnapshots(snapshots)
        setSelectedSavedSnapshotId((current) => {
          if (current && snapshots.some((item) => item.id === current)) return current
          return snapshots[0]?.id || ''
        })
        return
      }
      if (event.key !== storageKey) return
      if (event.newValue === null) {
        shelfSnapshotMemory.delete(storageKey)
        skipShelfPersistOnceRef.current = true
        shelfRevisionByKeyRef.current[storageKey] = 0
        setShelfItems([])
        setShelfOpen(false)
        setFocusedShelfKey('')
        return
      }
      const snapshot = readShelfSnapshot(storageKey, event.newValue)
      if (!snapshot) return
      const currentRevision = Number(shelfRevisionByKeyRef.current[storageKey] || 0)
      if (snapshot.revision <= currentRevision) return
      skipShelfPersistOnceRef.current = true
      shelfRevisionByKeyRef.current[storageKey] = snapshot.revision
      setShelfItems(snapshot.items)
      setShelfOpen(snapshot.open)
      setFocusedShelfKey('')
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [activeConvId])

  useEffect(() => {
    latestShelfStateRef.current = { convId: activeConvId, open: shelfOpen, items: shelfItems }
  }, [activeConvId, shelfItems, shelfOpen])

  useEffect(() => {
    setSelectedSavedSnapshotId((current) => {
      if (current && savedShelfSnapshots.some((item) => item.id === current)) return current
      return savedShelfSnapshots[0]?.id || ''
    })
  }, [savedShelfSnapshots])

  useEffect(() => {
    return () => {
      if (persistShelfTimerRef.current !== null) {
        window.clearTimeout(persistShelfTimerRef.current)
        persistShelfTimerRef.current = null
      }
    }
  }, [])

  useEffect(() => {
    if (skipShelfPersistOnceRef.current) {
      skipShelfPersistOnceRef.current = false
      return
    }
    const storageKey = shelfStorageKey(activeConvId)
    if (persistShelfTimerRef.current !== null) {
      window.clearTimeout(persistShelfTimerRef.current)
      persistShelfTimerRef.current = null
    }
    persistShelfTimerRef.current = window.setTimeout(() => {
      const latest = latestShelfStateRef.current
      const latestStorageKey = shelfStorageKey(latest.convId)
      if (latestStorageKey !== storageKey) {
        persistShelfTimerRef.current = null
        return
      }
      const currentRevision = Number(shelfRevisionByKeyRef.current[storageKey] || 0)
      const nextRevision = persistShelfSnapshot(
        storageKey,
        { open: latest.open, items: latest.items },
        currentRevision,
      )
      shelfRevisionByKeyRef.current[storageKey] = nextRevision
      persistShelfTimerRef.current = null
    }, 80)
    return () => {
      if (persistShelfTimerRef.current !== null) {
        window.clearTimeout(persistShelfTimerRef.current)
        persistShelfTimerRef.current = null
      }
    }
  }, [activeConvId, shelfItems, shelfOpen])

  const rows = useMemo(() => {
    const out: Array<
      | { kind: 'message'; message: Message }
      | { kind: 'refs'; userMsgId: number }
    > = []
    let lastUserMsgId = 0
    const renderedRefs = new Set<number>()

    for (const message of messages) {
      out.push({ kind: 'message', message })
      if (message.role === 'user') {
        lastUserMsgId = message.id
        continue
      }
      if (lastUserMsgId > 0 && !renderedRefs.has(lastUserMsgId) && hasRenderableRefs(refs, lastUserMsgId)) {
        out.push({ kind: 'refs', userMsgId: lastUserMsgId })
        renderedRefs.add(lastUserMsgId)
      }
    }

    return out
  }, [messages, refs])

  const assistantTraceByMsgId = useMemo(() => {
    const out = new Map<number, { answerOrder: number; userMsgId: number }>()
    let answerOrder = 0
    let lastUserMsgId = 0
    for (const message of messages) {
      if (message.role === 'user') {
        lastUserMsgId = message.id
        continue
      }
      if (message.role !== 'assistant') continue
      answerOrder += 1
      out.set(message.id, { answerOrder, userMsgId: lastUserMsgId })
    }
    return out
  }, [messages])

  const liveCiteMap = useMemo(() => {
    const map = new Map<string, CiteShelfItem>()
    const convTraceId = String(activeConvId || '')
    for (const message of messages) {
      if (message.role !== 'assistant' || !Array.isArray(message.cite_details)) continue
      const trace = assistantTraceByMsgId.get(message.id)
      for (const rawDetail of message.cite_details) {
        const detail = normalizeCiteDetail(rawDetail)
        if (!detail) continue
        const tracedDetail: CiteDetail = {
          ...detail,
          traceConvId: convTraceId,
          traceAssistantMsgId: message.id,
          traceAssistantOrder: Number(trace?.answerOrder || 0),
          traceUserMsgId: Number(trace?.userMsgId || 0),
        }
        const item = toShelfItem(tracedDetail)
        map.set(item.key, item)
      }
    }
    return map
  }, [activeConvId, assistantTraceByMsgId, messages])

  useEffect(() => {
    setShelfItems((current) => {
      let changed = false
      const next = current.map((item) => {
        const live = liveCiteMap.get(item.key)
        if (!live) return item
        const merged = mergeShelfItemWithLive(item, live)
        if (!sameShelfItem(merged, item)) {
          changed = true
          return merged
        }
        return item
      })
      const deduped = dedupeShelfItems(next)
      if (deduped.length !== current.length) changed = true
      return changed ? deduped : current
    })
  }, [liveCiteMap])

  const fetchShelfSummaryForItem = (item: CiteShelfItem) => {
    const summaryLine = String(item.summaryLine || '').trim()
    const summarySource = String(item.summarySource || '').trim().toLowerCase()
    if (summaryLine && (summarySource === 'abstract' || summarySource === 'fulltext')) return
    setShelfSummaryLoadingKey(item.key)
    referencesApi.bibliometricsCached(item as unknown as Record<string, unknown>)
      .then((meta) => {
        if (!meta || Object.keys(meta).length === 0) return
        setShelfItems((current) => current.map((entry) => {
          if (entry.key !== item.key) return entry
          const merged = mergeCiteMeta(entry, meta)
          return {
            ...toShelfItem(merged),
            key: entry.key,
            tags: normalizeShelfTags(entry.tags),
            note: normalizeShelfNote(entry.note),
          }
        }))
      })
      .finally(() => {
        setShelfSummaryLoadingKey((current) => (current === item.key ? '' : current))
      })
  }

  const repairShelfItemMeta = (item: CiteShelfItem) => {
    if (shelfRepairLoadingKey === item.key) return
    setShelfRepairLoadingKey(item.key)
    const basePayload = item as unknown as Record<string, unknown>
    const strictTitlePayload: Record<string, unknown> = {
      ...basePayload,
      raw: '',
      cite_fmt: '',
      citeFmt: '',
    }
    Promise.all([
      referencesApi.bibliometrics(basePayload).catch(() => ({})),
      referencesApi.bibliometrics(strictTitlePayload).catch(() => ({})),
    ])
      .then((metas) => {
        const candidates = metas.filter((meta) => meta && Object.keys(meta).length > 0)
        let didUpdate = false
        setShelfItems((current) => current.map((entry) => {
          if (entry.key !== item.key) return entry
          for (const meta of candidates) {
            const accepted = strictRepairMerge(entry, meta)
            if (!accepted) continue
            if (!sameShelfItem(accepted, entry)) {
              didUpdate = true
              return accepted
            }
            return entry
          }
          return entry
        }))
        if (didUpdate) message.success('已按严格规则修复元数据')
        else message.info('未通过严格匹配校验，已保留原信息')
      })
      .catch(() => {
        message.error('修复失败，请稍后重试')
      })
      .finally(() => {
        setShelfRepairLoadingKey((current) => (current === item.key ? '' : current))
      })
  }

  const openCitation = (detail: CiteDetail, event: MouseEvent<HTMLElement>) => {
    setPopoverDetail(detail)
    setPopoverPos({ x: event.clientX, y: event.clientY })
    const sourcePath = String(detail.sourcePath || '').trim()
    const isInPaperReference = Number(detail.num || 0) > 0
    const shouldFetchCitationMeta = Boolean(sourcePath) && !isInPaperReference
    const hasDoi = Boolean(String(detail.doi || '').trim())
    const shouldFetchBibliometrics = !detail.bibliometricsChecked && (
      isInPaperReference
        ? hasDoi
        : (detail.doi || detail.title || detail.venue || detail.raw || detail.citeFmt)
    )
    if (!shouldFetchCitationMeta && !shouldFetchBibliometrics) {
      setPopoverLoading(false)
      return
    }

    const reqs: Array<Promise<Record<string, unknown>>> = []
    if (shouldFetchCitationMeta && sourcePath) {
      reqs.push(referencesApi.citationMetaCached(sourcePath).catch(() => ({})))
    }
    if (shouldFetchBibliometrics) {
      reqs.push(referencesApi.bibliometricsCached(detail as unknown as Record<string, unknown>).catch(() => ({})))
    }

    const itemKey = toShelfItem(detail).key
    setPopoverLoading(true)
    Promise.all(reqs)
      .then((metas) => {
        setPopoverDetail((current) => {
          if (!current) return current
          let merged = current
          for (const meta of metas) {
            if (meta && Object.keys(meta).length > 0) {
              merged = mergeCiteMeta(merged, meta)
            }
          }
          return merged
        })
        setShelfItems((current) => current.map((item) => {
          if (item.key !== itemKey) return item
          let merged: CiteDetail = item
          for (const meta of metas) {
            if (meta && Object.keys(meta).length > 0) {
              merged = mergeCiteMeta(merged, meta)
            }
          }
          return {
            ...toShelfItem(merged),
            tags: normalizeShelfTags(item.tags),
            note: normalizeShelfNote(item.note),
          }
        }))
      })
      .finally(() => setPopoverLoading(false))
  }

  const addToShelf = (detail: CiteDetail) => {
    const item = toShelfItem(detail)
    setShelfItems((current) => {
      const identity = shelfPaperIdentity(item)
      const existing = current.find((entry) => entry.key === item.key || shelfPaperIdentity(entry) === identity)
      const mergedIncoming = existing ? mergeShelfItemWithLive(existing, item) : item
      const next = [
        {
          ...mergedIncoming,
          tags: normalizeShelfTags(existing?.tags || mergedIncoming.tags),
          note: normalizeShelfNote(existing?.note || mergedIncoming.note),
        },
        ...current.filter((entry) => entry.key !== item.key && shelfPaperIdentity(entry) !== identity),
      ]
      const deduped = dedupeShelfItems(next)
      if (deduped.length > SHELF_MAX_ITEMS) return deduped.slice(0, SHELF_MAX_ITEMS)
      if (deduped.length === next.length) return deduped.slice(0, SHELF_MAX_ITEMS)
      return deduped
    })
    setFocusedShelfKey(item.key)
    setShelfOpen(true)
    fetchShelfSummaryForItem(item)
  }

  const selectedSavedSnapshot = useMemo(
    () => savedShelfSnapshots.find((item) => item.id === selectedSavedSnapshotId) || null,
    [savedShelfSnapshots, selectedSavedSnapshotId],
  )

  const selectedSnapshotDiff = useMemo(() => {
    if (!selectedSavedSnapshot) return ''
    const diff = snapshotDiffCounts(shelfItems, selectedSavedSnapshot.items)
    if (diff.added <= 0 && diff.removed <= 0) return '与当前文献篮一致'
    return `对比当前：新增 ${diff.added} 条 · 移除 ${diff.removed} 条`
  }, [selectedSavedSnapshot, shelfItems])

  const saveShelfSnapshot = () => {
    const currentItems = dedupeShelfItems(shelfItems).slice(0, SHELF_MAX_ITEMS)
    if (currentItems.length <= 0) {
      message.info('文献篮为空，暂不能保存快照')
      return
    }
    const now = Date.now()
    const d = new Date(now)
    const pad = (value: number) => String(value).padStart(2, '0')
    const entry: ShelfSavedSnapshot = {
      id: `s_${now.toString(36)}_${Math.random().toString(36).slice(2, 7)}`,
      name: `快照 ${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`,
      createdAt: now,
      items: currentItems.map((item) => ({ ...item })),
    }
    setSavedShelfSnapshots((current) => {
      const next = [entry, ...current].slice(0, SHELF_SAVED_MAX_ITEMS)
      persistSavedShelfSnapshots(shelfSavedStorageKey(activeConvId), next)
      return next
    })
    setSelectedSavedSnapshotId(entry.id)
    message.success('已保存文献篮快照')
  }

  const loadShelfSnapshot = () => {
    if (!selectedSavedSnapshot) return
    const restored = dedupeShelfItems(selectedSavedSnapshot.items).slice(0, SHELF_MAX_ITEMS).map((item) => ({ ...item }))
    setShelfItems(restored)
    setFocusedShelfKey('')
    setShelfSummaryLoadingKey('')
    setShelfRepairLoadingKey('')
    message.success(`已载入快照：${selectedSavedSnapshot.name}`)
  }

  const deleteShelfSnapshot = () => {
    if (!selectedSavedSnapshot) return
    const removedName = selectedSavedSnapshot.name
    setSavedShelfSnapshots((current) => {
      const next = current.filter((item) => item.id !== selectedSavedSnapshot.id)
      persistSavedShelfSnapshots(shelfSavedStorageKey(activeConvId), next)
      return next
    })
    setSelectedSavedSnapshotId((current) => (current === selectedSavedSnapshot.id ? '' : current))
    message.success(`已删除快照：${removedName}`)
  }

  return (
    <>
      <div ref={scrollRef} className="h-full min-h-0 overflow-y-auto px-4 py-6 kb-main-scroll">
        <div className="mx-auto flex max-w-7xl flex-col gap-4">
          {rows.map((row, index) => {
            if (row.kind === 'refs') {
              return (
                <div key={`refs-${row.userMsgId}-${index}`} className="flex gap-3">
                  <div className="mt-1 h-7 w-7 shrink-0" />
                  <div className="max-w-[88%] flex-1">
                    <RefsPanel refs={refs} msgId={row.userMsgId} />
                  </div>
                </div>
              )
            }

            const message = row.message
            const isUser = message.role === 'user'
            const trace = assistantTraceByMsgId.get(message.id)
            const citeDetails = Array.isArray(message.cite_details)
              ? message.cite_details
                .map(normalizeCiteDetail)
                .filter((detail): detail is CiteDetail => Boolean(detail))
                .map((detail) => ({
                  ...detail,
                  traceConvId: String(activeConvId || ''),
                  traceAssistantMsgId: message.id,
                  traceAssistantOrder: Number(trace?.answerOrder || 0),
                  traceUserMsgId: Number(trace?.userMsgId || 0),
                }))
              : []
            const imageAttachments = imageAttachmentsOf(message)
            const showUserText = !(isUser && imageAttachments.length > 0 && isImageOnlyPlaceholder(message.content))
            const isImageOnlyUserMessage = isUser && imageAttachments.length > 0 && !showUserText
            const bodyContent = message.rendered_body || message.rendered_content || message.content
            const bubbleClass = isUser
              ? (isImageOnlyUserMessage ? 'w-fit max-w-[22rem]' : 'w-fit max-w-[88%]')
              : 'w-full max-w-[88%]'
            const bubblePadClass = isImageOnlyUserMessage ? 'px-4 py-4' : 'px-6 py-4'

            return (
              <div
                key={message.id}
                data-msg-id={message.id}
                className={`flex gap-3 ${isUser ? 'justify-end' : ''}`}
              >
                {!isUser ? <AssistantAvatar /> : null}
                <div
                  className={`${bubbleClass} ${bubblePadClass} rounded-[28px] ${
                    isUser ? 'bg-[var(--msg-user-bg)]' : 'border border-[var(--border)] bg-[var(--msg-ai-bg)] shadow-[0_8px_24px_rgba(15,23,42,0.03)]'
                  }`}
                >
                  {isUser ? (
                    <>
                      {imageAttachments.length > 0 ? (
                        <div
                          className={`${
                            showUserText ? 'mb-3' : ''
                          } grid ${
                            imageAttachments.length === 1 ? 'max-w-[18rem] grid-cols-1' : 'max-w-[38rem] grid-cols-2 sm:grid-cols-3'
                          } gap-2`}
                        >
                          {imageAttachments.map((item) => {
                            const src = String(item.url || '').trim()
                            const key = `${item.sha1 || item.path}-${item.name}`
                            const frameClass = 'block overflow-hidden rounded-2xl border border-[var(--border)] bg-white/70'
                            if (src) {
                              return (
                                <a
                                  key={key}
                                  href={src}
                                  target="_blank"
                                  rel="noreferrer"
                                  className={frameClass}
                                >
                                  <img
                                    src={src}
                                    alt={item.name}
                                    className="block h-32 w-full object-cover"
                                    loading="lazy"
                                  />
                                </a>
                              )
                            }
                            return (
                              <div key={key} className={frameClass}>
                                <div className="flex h-32 items-center justify-center px-3 text-center text-xs text-black/45">
                                  {item.name}
                                </div>
                              </div>
                            )
                          })}
                        </div>
                      ) : null}
                      {showUserText ? (
                        <Text className="whitespace-pre-wrap">{message.content}</Text>
                      ) : null}
                    </>
                  ) : (
                    <>
                      {message.notice ? (
                        <div className="mb-4 rounded-2xl border border-[var(--border)] bg-black/[0.03] px-4 py-3 text-sm text-black/70 dark:bg-white/[0.04] dark:text-white/70">
                          {message.notice}
                        </div>
                      ) : null}
                      <MarkdownRenderer
                        content={bodyContent}
                        citeDetails={citeDetails}
                        onCitationClick={openCitation}
                      />
                      <CopyBar
                        text={message.copy_text || message.content}
                        markdown={message.copy_markdown}
                      />
                    </>
                  )}
                </div>
                {isUser ? (
                  <div className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-[var(--border)] bg-[var(--msg-user-bg)]">
                    <UserOutlined className="text-xs" />
                  </div>
                ) : null}
              </div>
            )
          })}

          {generationPartial !== undefined && generationPartial !== null ? (
            <div className="flex gap-3">
              <AssistantAvatar />
              <div className="w-full max-w-[88%] rounded-[28px] border border-[var(--border)] bg-[var(--msg-ai-bg)] px-6 py-4 shadow-[0_8px_24px_rgba(15,23,42,0.03)]">
                {generationStage ? (
                  <div className="mb-2 flex items-center gap-2">
                    <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-[var(--accent)]" />
                    <Text type="secondary" className="text-xs">
                      {generationStage}
                    </Text>
                  </div>
                ) : null}
                {generationPartial ? (
                  <MarkdownRenderer content={generationPartial} />
                ) : (
                  <div className="flex items-center gap-1 py-1">
                    <span className="typing-dot" />
                    <span className="typing-dot" style={{ animationDelay: '0.15s' }} />
                    <span className="typing-dot" style={{ animationDelay: '0.3s' }} />
                  </div>
                )}
              </div>
            </div>
          ) : null}
        </div>
      </div>
      <CitationPopover
        detail={popoverDetail}
        position={popoverPos}
        loading={popoverLoading}
        inShelf={Boolean(popoverDetail && shelfItems.some((item) => item.key === toShelfItem(popoverDetail).key))}
        onClose={() => {
          setPopoverDetail(null)
          setPopoverPos(null)
          setPopoverLoading(false)
        }}
        onAddToShelf={addToShelf}
        onOpenShelf={() => setShelfOpen(true)}
      />
      <CiteShelf
        open={shelfOpen}
        items={shelfItems}
        focusedKey={focusedShelfKey}
        summaryLoadingKey={shelfSummaryLoadingKey}
        repairLoadingKey={shelfRepairLoadingKey}
        snapshots={savedShelfSnapshots}
        selectedSnapshotId={selectedSavedSnapshotId}
        snapshotDiff={selectedSnapshotDiff}
        onToggle={() => setShelfOpen((value) => !value)}
        onSelect={(item) => {
          setFocusedShelfKey(item.key)
          fetchShelfSummaryForItem(item)
        }}
        onRemove={(key) => {
          setShelfItems((current) => current.filter((item) => item.key !== key))
          if (focusedShelfKey === key) setFocusedShelfKey('')
          if (shelfSummaryLoadingKey === key) setShelfSummaryLoadingKey('')
          if (shelfRepairLoadingKey === key) setShelfRepairLoadingKey('')
        }}
        onClear={() => {
          setShelfItems([])
          setFocusedShelfKey('')
          setShelfSummaryLoadingKey('')
          setShelfRepairLoadingKey('')
        }}
        onUpdateTags={(key, tags) => {
          const nextTags = normalizeShelfTags(tags)
          setShelfItems((current) => current.map((item) => (
            item.key === key ? { ...item, tags: nextTags } : item
          )))
        }}
        onUpdateNote={(key, note) => {
          const nextNote = normalizeShelfNote(note)
          setShelfItems((current) => current.map((item) => (
            item.key === key ? { ...item, note: nextNote } : item
          )))
        }}
        onRepair={(item) => {
          repairShelfItemMeta(item)
        }}
        onSelectSnapshot={setSelectedSavedSnapshotId}
        onSaveSnapshot={saveShelfSnapshot}
        onLoadSnapshot={loadShelfSnapshot}
        onDeleteSnapshot={deleteShelfSnapshot}
      />
    </>
  )
}
