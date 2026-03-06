import { useEffect, useLayoutEffect, useMemo, useRef, useState, type MouseEvent } from 'react'
import { Typography } from 'antd'
import { UserOutlined } from '@ant-design/icons'
import { MarkdownRenderer } from './MarkdownRenderer'
import { CopyBar } from './CopyBar'
import { CitationPopover } from './CitationPopover'
import { CiteShelf } from './CiteShelf'
import { mergeCiteMeta, normalizeCiteDetail, shelfStorageKey, toShelfItem, type CiteDetail, type CiteShelfItem } from './citationState'
import { RefsPanel } from '../refs/RefsPanel'
import type { ChatImageAttachment, Message } from '../../api/chat'
import { referencesApi } from '../../api/references'

const { Text } = Typography
const SHELF_MAX_ITEMS = 120
const SHELF_SCHEMA_VERSION = 2

interface Props {
  activeConvId?: string | null
  messages: Message[]
  refs: Record<string, unknown>
  generationPartial?: string
  generationStage?: string
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
  let wrote = false
  for (const storage of listShelfStorages()) {
    try {
      storage.setItem(key, payload)
      wrote = true
    } catch {
      // ignore
    }
  }
  if (!wrote) {
    shelfStorageFallback.set(key, payload)
    return
  }
  shelfStorageFallback.set(key, payload)
}

function removeShelfStorage(key: string) {
  for (const storage of listShelfStorages()) {
    try {
      storage.removeItem(key)
    } catch {
      // ignore
    }
  }
  shelfStorageFallback.delete(key)
}

function restoreShelfItems(rawItems: unknown[]): CiteShelfItem[] {
  const seen = new Set<string>()
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
    seen.add(key)
    out.push({ ...base, key, main })
    if (out.length >= SHELF_MAX_ITEMS) break
  }
  return out
}

function readShelfSnapshot(key: string, rawOverride?: string): ShelfSnapshot | null {
  const raw = typeof rawOverride === 'string' ? rawOverride : readShelfStorage(key)
  if (!raw) return null
  try {
    const parsed = JSON.parse(raw)
    const itemsRaw: unknown[] = Array.isArray(parsed?.items) ? parsed.items : []
    const revision0 = Number(parsed?.revision || 0)
    const updatedAt0 = Number(parsed?.updatedAt || 0)
    return {
      version: Number(parsed?.version || 0) || 0,
      revision: Number.isFinite(revision0) && revision0 > 0 ? Math.floor(revision0) : 0,
      updatedAt: Number.isFinite(updatedAt0) && updatedAt0 > 0 ? Math.floor(updatedAt0) : 0,
      open: Boolean(parsed?.open),
      items: restoreShelfItems(itemsRaw),
    }
  } catch {
    // Corrupted payload: keep running and clear stale bad data.
    removeShelfStorage(key)
    return null
  }
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

function sameShelfItem(a: CiteShelfItem, b: CiteShelfItem): boolean {
  return (
    a.key === b.key
    && a.main === b.main
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
  const raw = JSON.stringify({
    version: SHELF_SCHEMA_VERSION,
    revision: nextRevision,
    updatedAt: Date.now(),
    open: payload.open,
    items: normalizedItems,
  })
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
  }
}

export function MessageList({ activeConvId, messages, refs, generationPartial, generationStage }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const [popoverDetail, setPopoverDetail] = useState<CiteDetail | null>(null)
  const [popoverPos, setPopoverPos] = useState<{ x: number; y: number } | null>(null)
  const [popoverLoading, setPopoverLoading] = useState(false)
  const [shelfOpen, setShelfOpen] = useState(false)
  const [shelfItems, setShelfItems] = useState<CiteShelfItem[]>([])
  const [focusedShelfKey, setFocusedShelfKey] = useState('')
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
    const nextStorageKey = shelfStorageKey(activeConvId)
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
    const onStorage = (event: StorageEvent) => {
      if (event.key !== storageKey) return
      if (event.newValue === null) {
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
    return () => {
      if (persistShelfTimerRef.current !== null) {
        window.clearTimeout(persistShelfTimerRef.current)
        persistShelfTimerRef.current = null
      }
      const latest = latestShelfStateRef.current
      const storageKey = shelfStorageKey(latest.convId)
      const currentRevision = Number(shelfRevisionByKeyRef.current[storageKey] || 0)
      const nextRevision = persistShelfSnapshot(
        storageKey,
        { open: latest.open, items: latest.items },
        currentRevision,
      )
      shelfRevisionByKeyRef.current[storageKey] = nextRevision
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

  const liveCiteMap = useMemo(() => {
    const map = new Map<string, CiteShelfItem>()
    for (const message of messages) {
      if (message.role !== 'assistant' || !Array.isArray(message.cite_details)) continue
      for (const rawDetail of message.cite_details) {
        const detail = normalizeCiteDetail(rawDetail)
        if (!detail) continue
        const item = toShelfItem(detail)
        map.set(item.key, item)
      }
    }
    return map
  }, [messages])

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
      return changed ? next : current
    })
  }, [liveCiteMap])

  const openCitation = (detail: CiteDetail, event: MouseEvent<HTMLElement>) => {
    setPopoverDetail(detail)
    setPopoverPos({ x: event.clientX, y: event.clientY })
    if (!detail.bibliometricsChecked && (detail.doi || detail.title || detail.venue || detail.raw || detail.citeFmt)) {
      setPopoverLoading(true)
      referencesApi.bibliometricsCached(detail as unknown as Record<string, unknown>)
        .then((meta) => {
          setPopoverDetail((current) => (current ? mergeCiteMeta(current, meta) : current))
          setShelfItems((current) => current.map((item) => (
            item.key === toShelfItem(detail).key ? toShelfItem(mergeCiteMeta(item, meta)) : item
          )))
        })
        .catch(() => {})
        .finally(() => setPopoverLoading(false))
    } else {
      setPopoverLoading(false)
    }
  }

  const addToShelf = (detail: CiteDetail) => {
    const item = toShelfItem(detail)
    setShelfItems((current) => {
      const next = [item, ...current.filter((entry) => entry.key !== item.key)].slice(0, SHELF_MAX_ITEMS)
      return next
    })
    setFocusedShelfKey(item.key)
    setShelfOpen(true)
  }

  return (
    <>
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto px-4 py-6 kb-main-scroll">
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
            const citeDetails = Array.isArray(message.cite_details)
              ? message.cite_details.map(normalizeCiteDetail).filter((detail): detail is CiteDetail => Boolean(detail))
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
        onToggle={() => setShelfOpen((value) => !value)}
        onRemove={(key) => {
          setShelfItems((current) => current.filter((item) => item.key !== key))
          if (focusedShelfKey === key) setFocusedShelfKey('')
        }}
        onClear={() => {
          setShelfItems([])
          setFocusedShelfKey('')
        }}
      />
    </>
  )
}
