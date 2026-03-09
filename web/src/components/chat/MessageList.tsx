import { useEffect, useLayoutEffect, useMemo, useRef, useState, type MouseEvent } from 'react'
import { Typography, message } from 'antd'
import { UserOutlined } from '@ant-design/icons'
import { MarkdownRenderer } from './MarkdownRenderer'
import { CopyBar } from './CopyBar'
import { CitationPopover } from './CitationPopover'
import { CiteShelf } from './CiteShelf'
import type { ReaderOpenPayload } from './PaperGuideReaderDrawer'
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
import { useChatStore } from '../../stores/chatStore'

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
  onOpenReader?: (payload: ReaderOpenPayload) => void
  paperGuideSourcePath?: string
  paperGuideSourceName?: string
}

interface RefUiMetaLite {
  display_name?: string
  heading_path?: string
  section_label?: string
  subsection_label?: string
  summary_line?: string
  why_line?: string
  source_path?: string
  anchor_target_kind?: string
  anchor_target_number?: number
}

interface RefMetaLite {
  source_path?: string
  heading_path?: string
  ref_best_heading_path?: string
  ref_headings?: unknown
  ref_locs?: unknown
  ref_show_snippets?: unknown
  ref_overview_snippets?: unknown
  ref_snippets?: unknown
}

interface RefHitLite {
  ui_meta?: RefUiMetaLite
  meta?: RefMetaLite
}

interface RefEntryLite {
  hits?: RefHitLite[]
}

interface LocateCandidate {
  sourcePath: string
  sourceName: string
  headingPath: string
  focusSnippet: string
  matchText: string
  sourceType: 'guide' | 'refs'
}

const GUIDE_LOCATE_CANDIDATE_LIMIT = 1600
const REF_LOCATE_CANDIDATE_LIMIT = 900

function stripMarkdownInline(input: string): string {
  return String(input || '')
    .replace(/!\[[^\]]*\]\([^)]+\)/g, ' ')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/~~([^~]+)~~/g, '$1')
    .replace(/<[^>]+>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function buildGuideLocateCandidates(
  markdown: string,
  sourcePath: string,
  sourceName: string,
  sourceType: 'guide' | 'refs' = 'guide',
): LocateCandidate[] {
  const out: LocateCandidate[] = []
  const seen = new Set<string>()
  const lines = String(markdown || '').replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n')
  const headingStack: Array<{ level: number; text: string }> = []
  let bucket: string[] = []

  const pushCandidate = (headingPathRaw: string, snippetRaw: string) => {
    const headingPath = stripMarkdownInline(headingPathRaw)
    const text = stripMarkdownInline(snippetRaw)
    if (text.length < 24) return
    const key = `${normalizeLocateText(sourcePath)}::${normalizeLocateText(headingPath)}::${normalizeLocateText(text).slice(0, 260)}`
    if (seen.has(key)) return
    seen.add(key)
    out.push({
      sourcePath,
      sourceName,
      headingPath,
      focusSnippet: text,
      matchText: [headingPath, text].filter(Boolean).join('\n'),
      sourceType,
    })
  }

  const pushSentenceCandidates = (headingPath: string, text: string) => {
    const src = stripMarkdownInline(text)
    if (src.length < 24) return
    const sentenceList = src
      .split(/(?<=[\u3002\uff01\uff1f.!;:\uff1b\uff1a])\s+/)
      .map((item) => item.trim())
      .filter((item) => item.length >= 16)
      .slice(0, 14)
    for (let i = 0; i < sentenceList.length; i += 1) {
      const sentence = sentenceList[i]
      if (sentence.length >= 24) pushCandidate(headingPath, sentence)
      const pair = [sentence, sentenceList[i + 1] || ''].filter(Boolean).join(' ').trim()
      if (pair.length >= 30 && pair.length <= 260) {
        pushCandidate(headingPath, pair)
      }
    }
  }

  const flush = () => {
    if (bucket.length <= 0) return
    const text = stripMarkdownInline(bucket.join(' ').trim())
    bucket = []
    if (text.length < 24) return
    const headingPath = headingStack.map((item) => item.text).filter(Boolean).join(' / ')
    pushCandidate(headingPath, text)
    pushSentenceCandidates(headingPath, text)
  }

  for (const raw of lines) {
    const line = String(raw || '')
    const heading = line.match(/^\s{0,3}(#{1,6})\s+(.*)$/)
    if (heading) {
      flush()
      const level = heading[1].length
      const text = stripMarkdownInline(heading[2] || '')
      if (text) {
        while (headingStack.length > 0 && headingStack[headingStack.length - 1].level >= level) {
          headingStack.pop()
        }
        headingStack.push({ level, text })
      }
      continue
    }
    if (/^\s*([-*_]\s*){3,}\s*$/.test(line) || /^\s*```/.test(line) || /^\s*~~~/.test(line)) {
      flush()
      continue
    }
    if (!line.trim()) {
      flush()
      continue
    }
    if (/^\s*\|/.test(line) || /^\s*>/.test(line)) {
      flush()
      const text = stripMarkdownInline(line.replace(/^\s*[>|]+\s*/, ''))
      if (text.length >= 24) {
        const headingPath = headingStack.map((item) => item.text).filter(Boolean).join(' / ')
        pushCandidate(headingPath, text)
      }
      continue
    }
    bucket.push(line)
  }
  flush()
  if (out.length <= 0) return out
  // Keep a practical upper bound for runtime matching cost.
  return out.slice(0, GUIDE_LOCATE_CANDIDATE_LIMIT)
}

function hasRenderableRefs(refs: Record<string, unknown>, msgId: number) {
  const entry = refs[String(msgId)] as { hits?: Array<{ meta?: Record<string, unknown> }> } | undefined
  if (!entry) return false
  const hits = Array.isArray(entry.hits) ? entry.hits : []
  if (hits.length > 0) return true
  return hits.some((hit) => String(hit?.meta?.ref_pack_state || '').trim().toLowerCase() === 'pending')
}

function normalizeLocateText(input: string): string {
  return String(input || '')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase()
}

function tokenizeLocateText(input: string): string[] {
  const src = normalizeLocateText(input)
  if (!src) return []
  const out: string[] = []
  const latin = src.match(/[a-z0-9]{2,}/g) || []
  out.push(...latin)
  const cjkSeq = src.match(/[\u4e00-\u9fff]{1,}/g) || []
  for (const seq of cjkSeq) {
    if (seq.length <= 2) {
      out.push(seq)
      continue
    }
    for (let i = 0; i < seq.length - 1; i += 1) {
      out.push(seq.slice(i, i + 2))
    }
  }
  return out
}

function overlapScore(a: string, b: string): number {
  const ta = tokenizeLocateText(a)
  const tb = tokenizeLocateText(b)
  if (ta.length === 0 || tb.length === 0) return 0
  const sa = new Set(ta)
  const sb = new Set(tb)
  let overlap = 0
  for (const token of sa) {
    if (sb.has(token)) overlap += 1
  }
  const denom = Math.sqrt(Math.max(1, sa.size) * Math.max(1, sb.size))
  return overlap / denom
}

function coerceStringArray(input: unknown, maxItems = 8, maxChars = 2200): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  const push = (value: unknown) => {
    if (out.length >= maxItems) return
    const text = String(value || '').replace(/\s+/g, ' ').trim()
    if (!text) return
    const normalized = normalizeLocateText(text)
    if (!normalized || seen.has(normalized)) return
    seen.add(normalized)
    out.push(text.length > maxChars ? `${text.slice(0, maxChars).trimEnd()}...` : text)
  }
  if (Array.isArray(input)) {
    for (const value of input) {
      if (out.length >= maxItems) break
      push(value)
    }
    return out
  }
  push(input)
  return out
}

function pickFirstRefText(loc: Record<string, unknown>): string {
  const keys = ['snippet', 'text', 'quote', 'content', 'summary', 'why']
  for (const key of keys) {
    const value = String(loc[key] || '').trim()
    if (value) return value
  }
  return ''
}

function formulaTokens(text: string): string[] {
  const src = String(text || '')
  if (!src) return []
  const out: string[] = []
  const texCmds = src.match(/\\[a-zA-Z]{2,}/g) || []
  out.push(...texCmds.map((item) => item.toLowerCase()))
  const symbols = src.match(/[A-Za-z](?:_[A-Za-z0-9]+)?(?:\^[A-Za-z0-9]+)?/g) || []
  out.push(...symbols.map((item) => item.toLowerCase()))
  const numbers = src.match(/\b\d{1,4}\b/g) || []
  out.push(...numbers)
  return out
}

function hasFormulaSignal(text: string): boolean {
  const src = String(text || '')
  if (!src) return false
  if (/[=^_]/.test(src)) return true
  if (/\\[a-zA-Z]{2,}/.test(src)) return true
  if (/\$[^$]{1,80}\$/.test(src) || /\$\$[^]{1,200}\$\$/.test(src)) return true
  return false
}

function formulaOverlapScore(a: string, b: string): number {
  const ta = new Set(formulaTokens(a))
  const tb = new Set(formulaTokens(b))
  if (ta.size <= 0 || tb.size <= 0) return 0
  let overlap = 0
  for (const token of ta) {
    if (tb.has(token)) overlap += 1
  }
  return overlap / Math.sqrt(ta.size * tb.size)
}

function scoreLocateCandidate(snippet: string, cand: LocateCandidate): number {
  const query = String(snippet || '').trim()
  if (!query) return 0
  const qNorm = normalizeLocateText(query)
  const focusText = String(cand.focusSnippet || '').trim()
  const matchText = String(cand.matchText || focusText).trim()
  if (!matchText) return 0
  const mNorm = normalizeLocateText(matchText)
  const fNorm = normalizeLocateText(focusText)

  let score = Math.max(
    overlapScore(query, matchText),
    overlapScore(query, focusText) * 1.08,
  )

  if (qNorm && mNorm) {
    if (mNorm.includes(qNorm)) score += 0.7
    const qHead = qNorm.slice(0, Math.min(64, qNorm.length))
    if (qHead.length >= 18 && mNorm.includes(qHead)) score += 0.26
    const mHead = mNorm.slice(0, Math.min(64, mNorm.length))
    if (mHead.length >= 18 && qNorm.includes(mHead)) score += 0.18
  }
  if (qNorm && fNorm) {
    if (fNorm.includes(qNorm)) score += 0.2
    const qHead = qNorm.slice(0, Math.min(48, qNorm.length))
    if (qHead.length >= 16 && fNorm.includes(qHead)) score += 0.14
  }

  const tokenSet = new Set(tokenizeLocateText(matchText))
  const keyTokens = Array.from(new Set(tokenizeLocateText(query))).filter((token) => token.length >= 3)
  let hitCount = 0
  for (const token of keyTokens) {
    if (tokenSet.has(token)) hitCount += 1
  }
  if (hitCount > 0) {
    score += Math.min(0.36, 0.03 * hitCount)
  }
  if (query.length >= 80 && focusText.length >= 80) {
    score += 0.05
  }
  if (hasFormulaSignal(query) || hasFormulaSignal(matchText)) {
    score += 0.72 * formulaOverlapScore(query, matchText)
  }
  if (cand.sourceType === 'guide') {
    score += 0.07
  }
  return score
}

function shortHeadingLabel(input: string, maxLen = 18): string {
  const text = String(input || '').replace(/\s+/g, ' ').trim()
  if (!text) return ''
  if (text.length <= maxLen) return text
  return `${text.slice(0, Math.max(6, maxLen - 1)).trimEnd()}...`
}

function pickLocateCandidate(snippet: string, candidates: LocateCandidate[]): LocateCandidate | null {
  const query = String(snippet || '').trim()
  if (!query || candidates.length <= 0) return null
  let best: LocateCandidate | null = null
  let bestScore = 0
  for (const cand of candidates) {
    const score = scoreLocateCandidate(query, cand)
    if (score > bestScore) {
      bestScore = score
      best = cand
    }
  }
  return bestScore >= 0.11 ? best : null
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
      const name = String(rec.name || '').trim() || 'Untitled snapshot'
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
      name: String(entry.name || '').trim() || 'Untitled snapshot',
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
    .split(/[;,锛岋紱]/)[0]
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

export function MessageList({
  activeConvId,
  messages,
  refs,
  generationPartial,
  generationStage,
  jumpTarget,
  onOpenReader,
  paperGuideSourcePath,
  paperGuideSourceName,
}: Props) {
  const createPaperGuideConversation = useChatStore((s) => s.createPaperGuideConversation)
  const scrollRef = useRef<HTMLDivElement>(null)
  const [popoverDetail, setPopoverDetail] = useState<CiteDetail | null>(null)
  const [popoverPos, setPopoverPos] = useState<{ x: number; y: number } | null>(null)
  const [popoverLoading, setPopoverLoading] = useState(false)
  const [popoverGuideLoading, setPopoverGuideLoading] = useState(false)
  const [shelfOpen, setShelfOpen] = useState(false)
  const [shelfItems, setShelfItems] = useState<CiteShelfItem[]>([])
  const [focusedShelfKey, setFocusedShelfKey] = useState('')
  const [shelfSummaryLoadingKey, setShelfSummaryLoadingKey] = useState('')
  const [shelfRepairLoadingKey, setShelfRepairLoadingKey] = useState('')
  const [savedShelfSnapshots, setSavedShelfSnapshots] = useState<ShelfSavedSnapshot[]>([])
  const [selectedSavedSnapshotId, setSelectedSavedSnapshotId] = useState('')
  const [guideDocCandidates, setGuideDocCandidates] = useState<LocateCandidate[]>([])
  const skipShelfPersistOnceRef = useRef(false)
  const persistShelfTimerRef = useRef<number | null>(null)
  const activeStorageKeyRef = useRef(shelfStorageKey(activeConvId))
  const shelfRevisionByKeyRef = useRef<Record<string, number>>({})
  const latestShelfStateRef = useRef<{ convId?: string | null; open: boolean; items: CiteShelfItem[] }>({
    convId: activeConvId,
    open: false,
    items: [],
  })

  useEffect(() => {
    const sourcePath = String(paperGuideSourcePath || '').trim()
    const sourceName = String(paperGuideSourceName || '').trim()
    if (!sourcePath) {
      setGuideDocCandidates([])
      return
    }
    let cancelled = false
    referencesApi.readerDoc(sourcePath)
      .then((res) => {
        if (cancelled) return
        const markdown = String(res.markdown || '')
        if (!markdown.trim()) {
          setGuideDocCandidates([])
          return
        }
        const resolvedName = String(res.source_name || sourceName || '').trim()
        setGuideDocCandidates(buildGuideLocateCandidates(markdown, sourcePath, resolvedName || sourceName || sourcePath))
      })
      .catch(() => {
        if (!cancelled) setGuideDocCandidates([])
      })
    return () => {
      cancelled = true
    }
  }, [paperGuideSourcePath, paperGuideSourceName])

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
        if (didUpdate) message.success('Metadata repaired with strict rules')
        else message.info('Strict match did not pass; original metadata kept')
      })
      .catch(() => {
        message.error('淇澶辫触锛岃绋嶅悗閲嶈瘯')
      })
      .finally(() => {
        setShelfRepairLoadingKey((current) => (current === item.key ? '' : current))
      })
  }

  const openCitation = (detail: CiteDetail, event: MouseEvent<HTMLElement>) => {
    setPopoverDetail(detail)
    setPopoverPos({ x: event.clientX, y: event.clientY })
    setPopoverGuideLoading(false)
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

  const startPaperGuideFromDetail = async (detail: CiteDetail) => {
    const isInPaperReference = Number(detail.num || 0) > 0
    if (isInPaperReference) {
      message.info('This item is an in-answer citation. Upload the PDF in Library first, then start paper guide.')
      return
    }
    const sourcePath = String(detail.sourcePath || '').trim()
    if (!sourcePath) {
      message.info('褰撳墠寮曠敤缂哄皯鍙粦瀹氱殑鏂囩尞璺緞')
      return
    }
    const sourceName = String(detail.sourceName || detail.title || '').trim() || sourcePath.split(/[\\/]/).pop() || '鏂囩尞'
    setPopoverGuideLoading(true)
    try {
      await createPaperGuideConversation({
        sourcePath,
        sourceName,
        title: `闃呰鎸囧 路 ${sourceName}`,
      })
      message.success('Entered paper guide conversation')
      setPopoverDetail(null)
      setPopoverPos(null)
    } catch (err) {
      message.error(err instanceof Error ? err.message : '鍒涘缓闃呰鎸囧浼氳瘽澶辫触')
    } finally {
      setPopoverGuideLoading(false)
    }
  }

  const openReaderFromDetail = (detail: CiteDetail) => {
    if (!onOpenReader) return
    const sourcePath = String(detail.sourcePath || '').trim()
    if (!sourcePath) {
      message.info('褰撳墠寮曠敤缂哄皯鍙粦瀹氱殑鏂囩尞璺緞')
      return
    }
    const sourceName = String(detail.sourceName || detail.title || '').trim() || sourcePath.split(/[\\/]/).pop() || '鏂囩尞'
    const snippet = String(detail.summaryLine || detail.title || detail.raw || '').trim()
    onOpenReader({
      sourcePath,
      sourceName,
      snippet,
    })
  }

  const selectedSavedSnapshot = useMemo(
    () => savedShelfSnapshots.find((item) => item.id === selectedSavedSnapshotId) || null,
    [savedShelfSnapshots, selectedSavedSnapshotId],
  )

  const selectedSnapshotDiff = useMemo(() => {
    if (!selectedSavedSnapshot) return ''
    const diff = snapshotDiffCounts(shelfItems, selectedSavedSnapshot.items)
    if (diff.added <= 0 && diff.removed <= 0) return 'No diff from current shelf'
    return `Compared with current: +${diff.added} / -${diff.removed}`
  }, [selectedSavedSnapshot, shelfItems])

  const saveShelfSnapshot = () => {
    const currentItems = dedupeShelfItems(shelfItems).slice(0, SHELF_MAX_ITEMS)
    if (currentItems.length <= 0) {
      message.info('Shelf is empty; cannot save snapshot')
      return
    }
    const now = Date.now()
    const d = new Date(now)
    const pad = (value: number) => String(value).padStart(2, '0')
    const entry: ShelfSavedSnapshot = {
      id: `s_${now.toString(36)}_${Math.random().toString(36).slice(2, 7)}`,
      name: `蹇収 ${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`,
      createdAt: now,
      items: currentItems.map((item) => ({ ...item })),
    }
    setSavedShelfSnapshots((current) => {
      const next = [entry, ...current].slice(0, SHELF_SAVED_MAX_ITEMS)
      persistSavedShelfSnapshots(shelfSavedStorageKey(activeConvId), next)
      return next
    })
    setSelectedSavedSnapshotId(entry.id)
    message.success('宸蹭繚瀛樻枃鐚蹇収')
  }

  const loadShelfSnapshot = () => {
    if (!selectedSavedSnapshot) return
    const restored = dedupeShelfItems(selectedSavedSnapshot.items).slice(0, SHELF_MAX_ITEMS).map((item) => ({ ...item }))
    setShelfItems(restored)
    setFocusedShelfKey('')
    setShelfSummaryLoadingKey('')
    setShelfRepairLoadingKey('')
    message.success(`宸茶浇鍏ュ揩鐓э細${selectedSavedSnapshot.name}`)
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
    message.success(`宸插垹闄ゅ揩鐓э細${removedName}`)
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
                    <RefsPanel refs={refs} msgId={row.userMsgId} onOpenReader={onOpenReader} />
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
            const refsUserMsgId = Number(message.refs_user_msg_id || trace?.userMsgId || 0)
            const refEntry = refsUserMsgId > 0 ? (refs[String(refsUserMsgId)] as RefEntryLite | undefined) : undefined
            const refHits = Array.isArray(refEntry?.hits) ? refEntry.hits : []
            const uniqueSourcePaths = Array.from(
              new Set(
                citeDetails
                  .map((detail) => String(detail.sourcePath || '').trim())
                  .filter(Boolean),
              ),
            )
            const guideSourcePath = String(paperGuideSourcePath || '').trim()
            const locateSourcePath = guideSourcePath || (
              uniqueSourcePaths.length === 1 ? uniqueSourcePaths[0] : ''
            )
            const locateSourceName = String(paperGuideSourceName || '').trim() || (
              citeDetails.find((detail) => String(detail.sourcePath || '').trim() === locateSourcePath)?.sourceName || ''
            )
            const refsLocateCandidatesAll: LocateCandidate[] = (() => {
              const out: LocateCandidate[] = []
              const seen = new Set<string>()
              const push = (candidate: LocateCandidate | null) => {
                if (!candidate) return
                if (out.length >= REF_LOCATE_CANDIDATE_LIMIT) return
                const sourcePath = String(candidate.sourcePath || '').trim()
                const matchText = String(candidate.matchText || '').trim()
                if (!sourcePath || !matchText) return
                const key = `${normalizeLocateText(sourcePath)}::${normalizeLocateText(candidate.headingPath || '')}::${normalizeLocateText(matchText).slice(0, 220)}`
                if (seen.has(key)) return
                seen.add(key)
                out.push(candidate)
              }

              for (const hit of refHits) {
                const ui = hit?.ui_meta || {}
                const meta = hit?.meta || {}
                const sourcePath = String(ui.source_path || meta.source_path || '').trim()
                if (!sourcePath) continue
                const sourceName = String(ui.display_name || '').trim() || sourcePath.split(/[\\/]/).pop() || '文献'

                const headingCandidates = new Set<string>([
                  String(ui.heading_path || '').trim(),
                  String(ui.section_label || '').trim(),
                  String(ui.subsection_label || '').trim(),
                  String(meta.ref_best_heading_path || '').trim(),
                  String(meta.heading_path || '').trim(),
                ].filter(Boolean))
                const anchorKind = String(ui.anchor_target_kind || '').trim().toLowerCase()
                const anchorNum = Number(ui.anchor_target_number || 0)
                for (const heading of coerceStringArray(meta.ref_headings, 8, 160)) {
                  headingCandidates.add(String(heading || '').trim())
                }

                const refLocs = Array.isArray(meta.ref_locs) ? meta.ref_locs : []
                for (const loc0 of refLocs.slice(0, 10)) {
                  const loc = (loc0 || {}) as Record<string, unknown>
                  const headingPath = String(loc.heading_path || loc.heading || '').trim()
                  if (headingPath) headingCandidates.add(headingPath)
                  const locText = pickFirstRefText(loc)
                  if (locText) {
                    push({
                      sourcePath,
                      sourceName,
                      headingPath: headingPath || String(ui.heading_path || '').trim(),
                      focusSnippet: locText,
                      matchText: [headingPath, locText].filter(Boolean).join('\n'),
                      sourceType: 'refs',
                    })
                  }
                }

                const snippetSeeds = [
                  ...coerceStringArray(ui.summary_line, 1, 360),
                  ...coerceStringArray(ui.why_line, 1, 360),
                  ...coerceStringArray(meta.ref_show_snippets, 4, 2600),
                  ...coerceStringArray(meta.ref_snippets, 4, 2600),
                  ...coerceStringArray(meta.ref_overview_snippets, 2, 2600),
                ]
                if (anchorKind === 'equation' && Number.isFinite(anchorNum) && anchorNum > 0) {
                  snippetSeeds.push(
                    `equation ${anchorNum}`,
                    `eq ${anchorNum}`,
                    `公式${anchorNum}`,
                    `(${anchorNum})`,
                  )
                }
                const headingFallback = Array.from(headingCandidates).find(Boolean) || ''
                for (const seed of snippetSeeds) {
                  const pieces = buildGuideLocateCandidates(seed, sourcePath, sourceName, 'refs')
                  if (pieces.length > 0) {
                    for (const piece of pieces.slice(0, 40)) push(piece)
                    continue
                  }
                  push({
                    sourcePath,
                    sourceName,
                    headingPath: headingFallback,
                    focusSnippet: seed,
                    matchText: [headingFallback, seed].filter(Boolean).join('\n'),
                    sourceType: 'refs',
                  })
                }

                for (const headingPath of headingCandidates) {
                  push({
                    sourcePath,
                    sourceName,
                    headingPath,
                    focusSnippet: headingPath,
                    matchText: headingPath,
                    sourceType: 'refs',
                  })
                }
              }
              return out
            })()
            const guideLocateCandidates = guideSourcePath
              ? guideDocCandidates.filter((item) => item.sourcePath === guideSourcePath)
              : []
            const refsScopedCandidates = guideSourcePath
              ? refsLocateCandidatesAll.filter((item) => item.sourcePath === guideSourcePath)
              : refsLocateCandidatesAll
            const locateCandidates = (() => {
              if (guideLocateCandidates.length > 0) return [...guideLocateCandidates, ...refsScopedCandidates]
              if (refsScopedCandidates.length > 0) return refsScopedCandidates
              if (guideSourcePath) return guideDocCandidates
              return refsLocateCandidatesAll
            })()
            const resolveCache = new Map<string, LocateCandidate | null>()
            const usedCount = new Map<string, number>()
            const resolveLocateCandidate = (snippet: string) => {
              const key = String(snippet || '').trim()
              if (!key) return null
              if (resolveCache.has(key)) return resolveCache.get(key) || null

              const rankIn = (cands: LocateCandidate[]) => {
                let best: LocateCandidate | null = null
                let bestScore = 0
                for (const cand of cands) {
                  const base = scoreLocateCandidate(key, cand)
                  const candKey = `${cand.sourcePath}::${cand.headingPath}::${cand.focusSnippet.slice(0, 96)}`
                  const penalty = 0.03 * Number(usedCount.get(candKey) || 0)
                  const score = base - penalty
                  if (score > bestScore) {
                    best = cand
                    bestScore = score
                  }
                }
                return { best, bestScore }
              }

              let picked: LocateCandidate | null = null
              const guideOnly = locateCandidates.filter((item) => item.sourceType === 'guide')
              if (guideOnly.length > 0) {
                const guideRank = rankIn(guideOnly)
                if (guideRank.best && guideRank.bestScore >= 0.11) {
                  picked = guideRank.best
                }
              }
              if (!picked) {
                const rankAll = rankIn(locateCandidates)
                if (rankAll.best && rankAll.bestScore >= 0.12) {
                  picked = rankAll.best
                }
              }
              if (!picked) {
                picked = pickLocateCandidate(key, locateCandidates)
              }
              const finalPick = picked || null
              if (finalPick) {
                const pickKey = `${finalPick.sourcePath}::${finalPick.headingPath}::${finalPick.focusSnippet.slice(0, 96)}`
                usedCount.set(pickKey, Number(usedCount.get(pickKey) || 0) + 1)
              }
              resolveCache.set(key, finalPick || null)
              return finalPick
            }
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
                        onLocateSnippet={onOpenReader
                          ? (snippet) => {
                            const picked = resolveLocateCandidate(snippet)
                            if (!picked) {
                              return
                            }
                            onOpenReader({
                              sourcePath: picked.sourcePath,
                              sourceName: picked.sourceName,
                              headingPath: picked.headingPath,
                              snippet: picked.focusSnippet || snippet,
                            })
                          }
                          : undefined}
                        locateLabelResolver={(snippet) => {
                          const picked = resolveLocateCandidate(snippet)
                          if (!picked) {
                            const fallbackName = shortHeadingLabel(locateSourceName || locateSourcePath, 16)
                            return fallbackName ? `依据 · ${fallbackName}` : '定位原文'
                          }
                          const hint = shortHeadingLabel(picked.headingPath || picked.sourceName, 16)
                          return hint ? `依据 · ${hint}` : '定位原文'
                        }}
                        locateTitleResolver={(snippet) => {
                          const picked = resolveLocateCandidate(snippet)
                          if (!picked) return ''
                          const heading = String(picked.headingPath || '').trim()
                          if (heading) return `依据段落：${heading}`
                          const name = String(picked.sourceName || '').trim()
                          return name ? `依据来源：${name}` : '定位原文'
                        }}
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
        guideLoading={popoverGuideLoading}
        inShelf={Boolean(popoverDetail && shelfItems.some((item) => item.key === toShelfItem(popoverDetail).key))}
        onClose={() => {
          setPopoverDetail(null)
          setPopoverPos(null)
          setPopoverLoading(false)
          setPopoverGuideLoading(false)
        }}
        onAddToShelf={addToShelf}
        onOpenShelf={() => setShelfOpen(true)}
        onOpenReader={openReaderFromDetail}
        onStartGuide={startPaperGuideFromDetail}
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




