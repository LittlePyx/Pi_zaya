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

export function MessageList({ activeConvId, messages, refs, generationPartial, generationStage }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const [popoverDetail, setPopoverDetail] = useState<CiteDetail | null>(null)
  const [popoverPos, setPopoverPos] = useState<{ x: number; y: number } | null>(null)
  const [popoverLoading, setPopoverLoading] = useState(false)
  const [shelfOpen, setShelfOpen] = useState(false)
  const [shelfItems, setShelfItems] = useState<CiteShelfItem[]>([])
  const [focusedShelfKey, setFocusedShelfKey] = useState('')
  const skipShelfPersistOnceRef = useRef(false)

  useLayoutEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const timer = window.requestAnimationFrame(() => {
      el.scrollTop = el.scrollHeight
    })
    return () => window.cancelAnimationFrame(timer)
  }, [activeConvId, generationPartial])

  useEffect(() => {
    // Switching conversation changes storage key; skip one persist cycle to avoid
    // writing previous conversation shelf state into the new key before hydration.
    skipShelfPersistOnceRef.current = true
    const storageKey = shelfStorageKey(activeConvId)
    try {
      const raw = window.localStorage.getItem(storageKey)
      if (!raw) {
        setShelfItems([])
        setShelfOpen(false)
        setFocusedShelfKey('')
        return
      }
      const parsed = JSON.parse(raw)
      const items: unknown[] = Array.isArray(parsed?.items) ? parsed.items : []
      setShelfItems(items.filter((item): item is CiteShelfItem => Boolean(item && typeof item === 'object')))
      setShelfOpen(Boolean(parsed?.open))
      setFocusedShelfKey('')
    } catch {
      setShelfItems([])
      setShelfOpen(false)
      setFocusedShelfKey('')
    }
  }, [activeConvId])

  useEffect(() => {
    if (skipShelfPersistOnceRef.current) {
      skipShelfPersistOnceRef.current = false
      return
    }
    const storageKey = shelfStorageKey(activeConvId)
    window.localStorage.setItem(storageKey, JSON.stringify({ open: shelfOpen, items: shelfItems }))
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
        const merged = { ...item, ...live, main: live.main || item.main }
        if (JSON.stringify(merged) !== JSON.stringify(item)) {
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
      const next = [item, ...current.filter((entry) => entry.key !== item.key)].slice(0, 120)
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
