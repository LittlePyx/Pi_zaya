/* eslint-disable react-hooks/set-state-in-effect */

import { startTransition, useDeferredValue, useEffect, useLayoutEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import { Button, message, Typography } from 'antd'
import { useChatStore } from '../stores/chatStore'
import { useSettingsStore } from '../stores/settingsStore'
import { MessageList } from '../components/chat/MessageList'
import { ChatInput } from '../components/chat/ChatInput'
import { PaperGuideReaderDrawer } from '../components/chat/PaperGuideReaderDrawer'
import { sameHighlightTarget } from '../components/chat/reader/readerDomUtils'
import type { ReaderOpenPayload, ReaderSessionHighlight } from '../components/chat/reader/readerTypes'
import type { ChatUploadItem, Message } from '../api/chat'
import { S } from '../i18n/zh'

const { Text } = Typography

const HISTORY_PAGE_SIZE = 24
const LIVE_WINDOW = 16
const READY_DISMISS_MS = 2600
const DUPLICATE_DISMISS_MS = 3600
const DESKTOP_READER_BREAKPOINT = 1280
const DESKTOP_READER_DEFAULT_WIDTH = 560
const DESKTOP_READER_MIN_WIDTH = 420
const DESKTOP_READER_MAX_WIDTH = 760
const DESKTOP_READER_WIDTH_TRANSITION = 'width 160ms cubic-bezier(0.2, 0, 0, 1)'
const READER_WIDTH_STORAGE_KEY = 'kb:paper-guide-reader-width'
const READER_COLLAPSED_STORAGE_KEY = 'kb:paper-guide-reader-collapsed'
const TIMELINE_RAIL_LABEL = '\u65f6\u95f4\u7ebf'
const showLegacyUiBlocks = false

function uploadItemKey(item: ChatUploadItem) {
  if (item.kind === 'pdf' && item.ingest_job_id) {
    return `pdf-job:${item.ingest_job_id}`
  }
  return [item.kind, item.sha1 || '', item.path || '', item.name].join(':')
}

function compactTimelineText(content: string, maxLen = 68) {
  const raw = String(content || '').replace(/\s+/g, ' ').trim()
  if (!raw) return '空白提问'
  const imgOnly = raw.match(/^\[Image attachment x(\d+)\]$/i)
  if (imgOnly) {
    return `图片提问 x${imgOnly[1] || '1'}`
  }
  if (raw.length <= maxLen) return raw
  return `${raw.slice(0, Math.max(8, maxLen - 1)).trimEnd()}...`
}

function stripSourceExt(name: string) {
  return String(name || '')
    .replace(/\.en\.md$/i, '')
    .replace(/\.md$/i, '')
    .replace(/\.pdf$/i, '')
    .trim()
}

function clampReaderWidth(value: number) {
  if (!Number.isFinite(value)) return DESKTOP_READER_DEFAULT_WIDTH
  return Math.max(DESKTOP_READER_MIN_WIDTH, Math.min(DESKTOP_READER_MAX_WIDTH, Math.round(value)))
}

function loadStoredReaderWidth() {
  if (typeof window === 'undefined') return DESKTOP_READER_DEFAULT_WIDTH
  const raw = Number(window.localStorage.getItem(READER_WIDTH_STORAGE_KEY) || 0)
  return clampReaderWidth(raw || DESKTOP_READER_DEFAULT_WIDTH)
}

function loadStoredReaderCollapsed() {
  if (typeof window === 'undefined') return false
  return window.localStorage.getItem(READER_COLLAPSED_STORAGE_KEY) === '1'
}

function readerHighlightScopeKey(convId: string | null | undefined, sourcePath: string) {
  const path = String(sourcePath || '').trim().toLowerCase()
  if (!path) return ''
  const conv = String(convId || '__detached__').trim().toLowerCase()
  return `${conv}::${path}`
}

function sameReaderSessionHighlight(
  left: Pick<ReaderSessionHighlight, 'text' | 'startOffset' | 'endOffset' | 'blockId' | 'anchorId' | 'occurrence' | 'readableIndex' | 'documentOccurrence' | 'startReadableIndex' | 'endReadableIndex'>,
  right: Pick<ReaderSessionHighlight, 'text' | 'startOffset' | 'endOffset' | 'blockId' | 'anchorId' | 'occurrence' | 'readableIndex' | 'documentOccurrence' | 'startReadableIndex' | 'endReadableIndex'>,
) {
  return sameHighlightTarget(left, right)
}

interface TimelineItem {
  order: number
  userMsgId: number
  targetMsgId: number
  questionPreview: string
  hasAnswer: boolean
}

export default function ChatPage() {
  const messages = useChatStore((s) => s.messages)
  const conversationLoading = useChatStore((s) => s.conversationLoading)
  const messagesLoadingMore = useChatStore((s) => s.messagesLoadingMore)
  const messagesHasMoreBefore = useChatStore((s) => s.messagesHasMoreBefore)
  const loadOlderMessages = useChatStore((s) => s.loadOlderMessages)
  const generation = useChatStore((s) => s.generation)
  const refs = useChatStore((s) => s.refs)
  const activeConvId = useChatStore((s) => s.activeConvId)
  const activeConversation = useChatStore((s) => s.activeConversation)
  const guideBindings = useChatStore((s) => s.guideBindings)
  const uploadItems = useChatStore((s) => s.uploadItems)
  const pendingImages = useChatStore((s) => s.pendingImages)
  const uploading = useChatStore((s) => s.uploading)
  const uploadFiles = useChatStore((s) => s.uploadFiles)
  const retryUploadItem = useChatStore((s) => s.retryUploadItem)
  const cancelUploadItem = useChatStore((s) => s.cancelUploadItem)
  const removePendingImage = useChatStore((s) => s.removePendingImage)
  const dismissUploadItem = useChatStore((s) => s.dismissUploadItem)
  const sendMessage = useChatStore((s) => s.sendMessage)
  const createPaperGuideConversation = useChatStore((s) => s.createPaperGuideConversation)
  const cancelGen = useChatStore((s) => s.cancelGeneration)
  const settings = useSettingsStore()
  const [timelineOpen, setTimelineOpen] = useState(true)
  const [timelineJump, setTimelineJump] = useState<{ messageId: number; token: number } | null>(null)
  const [activeTimelineUserMsgId, setActiveTimelineUserMsgId] = useState<number | null>(null)
  const [readerOpen, setReaderOpen] = useState(false)
  const [readerPayload, setReaderPayload] = useState<ReaderOpenPayload | null>(null)
  const [readerCollapsed, setReaderCollapsed] = useState(loadStoredReaderCollapsed)
  const [readerWidth, setReaderWidth] = useState(loadStoredReaderWidth)
  const [readerSessionHighlights, setReaderSessionHighlights] = useState<Record<string, ReaderSessionHighlight[]>>({})
  const [desktopReaderEligible, setDesktopReaderEligible] = useState(
    () => (typeof window !== 'undefined' ? window.innerWidth >= DESKTOP_READER_BREAKPOINT : false),
  )
  const [readerResizing, setReaderResizing] = useState(false)
  const [appendSignal, setAppendSignal] = useState<{ token: number; text: string } | null>(null)
  const uploadNoticeRef = useRef<Record<string, string>>({})
  const dismissTimerRef = useRef<Record<string, number>>({})
  const timelineJumpTokenRef = useRef(1)
  const readerLocateRequestRef = useRef(1)
  const splitLayoutRef = useRef<HTMLDivElement | null>(null)
  const readerResizeGuideRef = useRef<HTMLDivElement | null>(null)
  const readerResizeRef = useRef<{ startX: number; startWidth: number } | null>(null)
  const readerActivePointerIdRef = useRef<number | null>(null)
  const readerResizeFocusRestoreRef = useRef<HTMLElement | null>(null)
  const readerWidthLiveRef = useRef(readerWidth)
  const readerResizePreviewWidthRef = useRef(readerWidth)
  const timelineScrollRestoreTopRef = useRef<number | null>(null)

  const nextEventToken = () => {
    timelineJumpTokenRef.current += 1
    return timelineJumpTokenRef.current
  }

  const nextReaderLocateRequestId = () => {
    readerLocateRequestRef.current += 1
    return readerLocateRequestRef.current
  }

  const clearTimelineSelection = () => {
    setTimelineJump(null)
    setActiveTimelineUserMsgId(null)
  }

  const captureTimelineScrollTop = () => {
    const scrollHost = splitLayoutRef.current?.querySelector<HTMLElement>('.kb-main-scroll')
    timelineScrollRestoreTopRef.current = scrollHost ? scrollHost.scrollTop : null
  }

  const toggleTimelineOpen = () => {
    captureTimelineScrollTop()
    clearTimelineSelection()
    setTimelineOpen((value) => !value)
  }

  useEffect(() => {
    setTimelineOpen(true)
    clearTimelineSelection()
    setReaderOpen(false)
    setReaderPayload(null)
    setReaderCollapsed(false)
    setAppendSignal(null)
  }, [activeConvId])

  useEffect(() => () => {
    Object.values(dismissTimerRef.current).forEach((timer) => window.clearTimeout(timer))
    dismissTimerRef.current = {}
  }, [])

  useEffect(() => {
    const syncLayout = () => {
      setDesktopReaderEligible(window.innerWidth >= DESKTOP_READER_BREAKPOINT)
    }
    syncLayout()
    window.addEventListener('resize', syncLayout)
    return () => {
      window.removeEventListener('resize', syncLayout)
    }
  }, [])

  useEffect(() => {
    window.localStorage.setItem(READER_WIDTH_STORAGE_KEY, String(clampReaderWidth(readerWidth)))
  }, [readerWidth])

  useEffect(() => {
    window.localStorage.setItem(READER_COLLAPSED_STORAGE_KEY, readerCollapsed ? '1' : '0')
  }, [readerCollapsed])

  useEffect(() => {
    readerWidthLiveRef.current = readerWidth
    if (!readerResizing) {
      readerResizePreviewWidthRef.current = readerWidth
    }
  }, [readerResizing, readerWidth])

  useLayoutEffect(() => {
    const targetTop = timelineScrollRestoreTopRef.current
    if (targetTop == null) return
    timelineScrollRestoreTopRef.current = null
    const scrollHost = splitLayoutRef.current?.querySelector<HTMLElement>('.kb-main-scroll')
    if (!scrollHost) return
    let cancelled = false
    const restore = () => {
      if (cancelled) return
      scrollHost.scrollTop = targetTop
    }
    const frameA = window.requestAnimationFrame(restore)
    const frameB = window.requestAnimationFrame(() => {
      window.requestAnimationFrame(restore)
    })
    return () => {
      cancelled = true
      window.cancelAnimationFrame(frameA)
      window.cancelAnimationFrame(frameB)
    }
  }, [timelineOpen, desktopReaderEligible, readerOpen, readerCollapsed])

  const restoreReaderResizeFocus = () => {
    const target = readerResizeFocusRestoreRef.current
    readerResizeFocusRestoreRef.current = null
    if (!target || !target.isConnected) return
    try {
      target.focus({ preventScroll: true })
    } catch {
      target.focus()
    }
  }

  const clearReaderResizeSession = () => {
    readerResizeRef.current = null
    readerActivePointerIdRef.current = null
    readerResizePreviewWidthRef.current = readerWidthLiveRef.current
    document.body.classList.remove('kb-reader-resizing')
    document.body.style.removeProperty('cursor')
    document.body.style.removeProperty('user-select')
    const guide = readerResizeGuideRef.current
    if (guide) {
      guide.style.removeProperty('left')
    }
  }

  useEffect(() => () => {
    clearReaderResizeSession()
  }, [])

  const updateReaderResizeGuide = (nextWidth: number) => {
    const guide = readerResizeGuideRef.current
    const layout = splitLayoutRef.current
    const clampedWidth = clampReaderWidth(nextWidth)
    readerResizePreviewWidthRef.current = clampedWidth
    if (!guide || !layout || readerCollapsed) return
    const nextLeft = Math.max(0, layout.clientWidth - clampedWidth)
    guide.style.left = `${Math.round(nextLeft)}px`
  }

  const finishReaderResize = (commit: boolean) => {
    const finalWidth = clampReaderWidth(
      commit ? readerResizePreviewWidthRef.current || readerWidthLiveRef.current : readerWidthLiveRef.current,
    )
    clearReaderResizeSession()
    setReaderResizing(false)
    if (commit && !readerCollapsed) {
      setReaderWidth(finalWidth)
    }
    window.requestAnimationFrame(restoreReaderResizeFocus)
  }

  useEffect(() => {
    const liveKeys = new Set<string>()
    for (const item of uploadItems) {
      if (item.kind !== 'pdf') continue
      const key = uploadItemKey(item)
      liveKeys.add(key)
      const terminalState =
        item.status === 'duplicate'
          ? 'duplicate'
          : item.ingest_status === 'cancelled'
            ? 'cancelled'
            : (item.status === 'error' || item.ingest_status === 'error')
            ? 'error'
            : (item.ready || item.ingest_status === 'ready')
              ? 'ready'
              : ''
      if (!terminalState || uploadNoticeRef.current[key] === terminalState) {
        continue
      }
      uploadNoticeRef.current[key] = terminalState
      if (terminalState === 'ready') {
        message.success(`PDF 已入库: ${item.name}`)
        if (dismissTimerRef.current[key] == null) {
          dismissTimerRef.current[key] = window.setTimeout(() => {
            dismissUploadItem(key)
            delete dismissTimerRef.current[key]
          }, READY_DISMISS_MS)
        }
      } else if (terminalState === 'duplicate') {
        message.info(`PDF 已在库中: ${item.name}`)
        if (dismissTimerRef.current[key] == null) {
          dismissTimerRef.current[key] = window.setTimeout(() => {
            dismissUploadItem(key)
            delete dismissTimerRef.current[key]
          }, DUPLICATE_DISMISS_MS)
        }
      } else if (terminalState === 'cancelled') {
        message.info(`PDF 已取消: ${item.name}`)
      } else if (terminalState === 'error') {
        message.error(`PDF 入库失败: ${item.name}`)
      }
    }

    for (const key of Object.keys(uploadNoticeRef.current)) {
      if (liveKeys.has(key)) continue
      delete uploadNoticeRef.current[key]
      const timer = dismissTimerRef.current[key]
      if (timer != null) {
        window.clearTimeout(timer)
        delete dismissTimerRef.current[key]
      }
    }
  }, [dismissUploadItem, uploadItems])

  const onSend = (text: string) => {
    sendMessage(text, {
      topK: settings.topK,
      temperature: settings.temperature,
      maxTokens: settings.maxTokens,
      deepRead: true,
    })
  }

  const onUpload = async (files: File[]) => {
    try {
      await uploadFiles(files, { quickIngest: true, speedMode: 'balanced' })
    } catch {
      message.error('上传失败')
    }
  }

  const onRetryUpload = async (key: string) => {
    try {
      await retryUploadItem(key)
    } catch (err) {
      message.error(err instanceof Error ? err.message : '重试入库失败')
    }
  }

  const onCancelUpload = async (key: string) => {
    try {
      await cancelUploadItem(key)
    } catch (err) {
      message.error(err instanceof Error ? err.message : '取消入库失败')
    }
  }

  const onStartGuideFromUpload = async (item: ChatUploadItem) => {
    const sourcePath = String(item.md_path || '').trim()
    if (!sourcePath) {
      message.info('PDF 尚未完成转换，请等待入库完成后再开始阅读指导。')
      return
    }
    const sourceName = stripSourceExt(item.name) || item.name
    const hide = message.loading('正在创建阅读指导会话...', 0)
    try {
      await createPaperGuideConversation({
        sourcePath,
        sourceName,
        title: `阅读指导 · ${sourceName}`,
      })
      hide()
      message.success('已进入阅读指导会话')
    } catch (err) {
      hide()
      message.error(err instanceof Error ? err.message : '创建阅读指导会话失败')
    }
  }

  const liveRunning = Boolean(generation)
  const visibleMessages = liveRunning
    ? messages.slice(-Math.min(messages.length, LIVE_WINDOW))
    : messages
  const deferredTimelineMessages = useDeferredValue(messages)
  const hiddenCount = liveRunning
    ? Math.max(0, messages.length - visibleMessages.length)
    : 0
  const messageIndexById = useMemo(() => {
    const map = new Map<number, number>()
    messages.forEach((msg, idx) => {
      map.set(msg.id, idx)
    })
    return map
  }, [messages])
  const timelineItems = useMemo(() => {
    const out: TimelineItem[] = []
    let pendingUser: Message | null = null
    let order = 0
    for (const msg of deferredTimelineMessages) {
      if (msg.role === 'user') {
        pendingUser = msg
        continue
      }
      if (msg.role !== 'assistant' || !pendingUser) continue
      order += 1
      out.push({
        order,
        userMsgId: pendingUser.id,
        targetMsgId: msg.id,
        questionPreview: compactTimelineText(pendingUser.content),
        hasAnswer: true,
      })
      pendingUser = null
    }
    if (pendingUser) {
      order += 1
      out.push({
        order,
        userMsgId: pendingUser.id,
        targetMsgId: pendingUser.id,
        questionPreview: compactTimelineText(pendingUser.content),
        hasAnswer: false,
      })
    }
    return out
  }, [deferredTimelineMessages])
  const timelineTrackedMessageIds = useMemo(
    () => timelineItems.map((item) => item.targetMsgId),
    [timelineItems],
  )
  const timelineUserMsgIdByTargetMsgId = useMemo(() => {
    const map = new Map<number, number>()
    timelineItems.forEach((item) => {
      map.set(item.targetMsgId, item.userMsgId)
    })
    return map
  }, [timelineItems])
  const effectiveGuide = useMemo(() => {
    const convId = String(activeConvId || '').trim()
    const localGuide = convId ? guideBindings?.[convId] : undefined
    const sourcePath = String(activeConversation?.bound_source_path || localGuide?.sourcePath || '').trim()
    const sourceName = String(activeConversation?.bound_source_name || localGuide?.sourceName || '').trim()
    return { sourcePath, sourceName }
  }, [activeConvId, activeConversation?.bound_source_name, activeConversation?.bound_source_path, guideBindings])

  const jumpToTimelineItem = (item: TimelineItem) => {
    if (liveRunning) {
      message.info('当前正在生成回答，完成后再使用时间线跳转。')
      return
    }
    const idx = messageIndexById.get(item.targetMsgId)
    if (idx == null) return
    setActiveTimelineUserMsgId(null)
    const token = nextEventToken()
    window.setTimeout(() => {
      setTimelineJump({ messageId: item.targetMsgId, token })
    }, 0)
  }

  const openReader = (payload: ReaderOpenPayload) => {
    const sourcePath = String(payload?.sourcePath || '').trim()
    if (!sourcePath) {
      message.info('当前引用缺少可绑定的文献路径')
      return
    }
    const locateRequestId = nextReaderLocateRequestId()
    const locateTarget = (payload.locateTarget && typeof payload.locateTarget === 'object')
      ? {
        segmentId: String(payload.locateTarget.segmentId || '').trim() || undefined,
        sourceSegmentId: String(payload.locateTarget.sourceSegmentId || '').trim() || undefined,
        headingPath: String(payload.locateTarget.headingPath || '').trim() || undefined,
        snippet: String(payload.locateTarget.snippet || '').trim() || undefined,
        highlightSnippet: String(payload.locateTarget.highlightSnippet || '').trim() || undefined,
        evidenceQuote: String(payload.locateTarget.evidenceQuote || '').trim() || undefined,
        anchorText: String(payload.locateTarget.anchorText || '').trim() || undefined,
        blockId: String(payload.locateTarget.blockId || '').trim() || undefined,
        anchorId: String(payload.locateTarget.anchorId || '').trim() || undefined,
        anchorKind: String(payload.locateTarget.anchorKind || '').trim() || undefined,
        anchorNumber: Number.isFinite(Number(payload.locateTarget.anchorNumber))
          ? Number(payload.locateTarget.anchorNumber)
          : undefined,
        claimType: String(payload.locateTarget.claimType || '').trim() || undefined,
        locatePolicy: String(payload.locateTarget.locatePolicy || '').trim() || undefined,
        locateSurfacePolicy: String(payload.locateTarget.locateSurfacePolicy || '').trim() || undefined,
        snippetAliases: Array.isArray(payload.locateTarget.snippetAliases)
          ? payload.locateTarget.snippetAliases.map((item) => String(item || '').trim()).filter(Boolean)
          : undefined,
        relatedBlockIds: Array.isArray(payload.locateTarget.relatedBlockIds)
          ? payload.locateTarget.relatedBlockIds.map((item) => String(item || '').trim()).filter(Boolean)
          : undefined,
      }
      : undefined
    const claimGroup = (payload.claimGroup && typeof payload.claimGroup === 'object')
      ? {
        id: String(payload.claimGroup.id || '').trim() || undefined,
        kind: String(payload.claimGroup.kind || '').trim() || undefined,
        leadText: String(payload.claimGroup.leadText || '').trim() || undefined,
        distance: Number.isFinite(Number(payload.claimGroup.distance))
          ? Number(payload.claimGroup.distance)
          : undefined,
      }
      : undefined
    setReaderPayload({
      sourcePath,
      sourceName: String(payload.sourceName || '').trim(),
      headingPath: String(payload.headingPath || '').trim(),
      snippet: String(payload.snippet || '').trim(),
      highlightSnippet: String(payload.highlightSnippet || '').trim(),
      blockId: String(payload.blockId || '').trim() || undefined,
      anchorId: String(payload.anchorId || '').trim() || undefined,
      relatedBlockIds: Array.isArray(payload.relatedBlockIds)
        ? payload.relatedBlockIds.map((item) => String(item || '').trim()).filter(Boolean)
        : undefined,
      anchorKind: String(payload.anchorKind || '').trim() || undefined,
      anchorNumber: Number.isFinite(Number(payload.anchorNumber))
        ? Number(payload.anchorNumber)
        : undefined,
      strictLocate: Boolean(payload.strictLocate),
      locateMode: payload.locateMode === 'heuristic' ? 'heuristic' : undefined,
      locateTarget,
      claimGroup,
      locateRequestId,
      alternatives: Array.isArray(payload.alternatives)
        ? payload.alternatives.map((item) => ({
          headingPath: String(item?.headingPath || '').trim(),
          snippet: String(item?.snippet || '').trim(),
          highlightSnippet: String(item?.highlightSnippet || '').trim(),
          blockId: String(item?.blockId || '').trim() || undefined,
          anchorId: String(item?.anchorId || '').trim() || undefined,
          anchorKind: String(item?.anchorKind || '').trim() || undefined,
          anchorNumber: Number.isFinite(Number(item?.anchorNumber))
            ? Number(item?.anchorNumber)
            : undefined,
        }))
        : undefined,
      visibleAlternatives: Array.isArray(payload.visibleAlternatives)
        ? payload.visibleAlternatives.map((item) => ({
          headingPath: String(item?.headingPath || '').trim(),
          snippet: String(item?.snippet || '').trim(),
          highlightSnippet: String(item?.highlightSnippet || '').trim(),
          blockId: String(item?.blockId || '').trim() || undefined,
          anchorId: String(item?.anchorId || '').trim() || undefined,
          anchorKind: String(item?.anchorKind || '').trim() || undefined,
          anchorNumber: Number.isFinite(Number(item?.anchorNumber))
            ? Number(item?.anchorNumber)
            : undefined,
        }))
        : undefined,
      evidenceAlternatives: Array.isArray(payload.evidenceAlternatives)
        ? payload.evidenceAlternatives.map((item) => ({
          headingPath: String(item?.headingPath || '').trim(),
          snippet: String(item?.snippet || '').trim(),
          highlightSnippet: String(item?.highlightSnippet || '').trim(),
          blockId: String(item?.blockId || '').trim() || undefined,
          anchorId: String(item?.anchorId || '').trim() || undefined,
          anchorKind: String(item?.anchorKind || '').trim() || undefined,
          anchorNumber: Number.isFinite(Number(item?.anchorNumber))
            ? Number(item?.anchorNumber)
            : undefined,
        }))
        : undefined,
      initialAltIndex: Number.isFinite(Number(payload.initialAltIndex))
        ? Number(payload.initialAltIndex)
        : undefined,
    })
    setReaderCollapsed(false)
    setReaderOpen(true)
  }

  const appendReaderSelection = (text: string) => {
    const raw = String(text || '')
    if (!raw.trim()) return
    setAppendSignal({
      token: nextEventToken(),
      text: raw,
    })
  }

  const activeReaderHighlightScope = useMemo(
    () => readerHighlightScopeKey(activeConvId, String(readerPayload?.sourcePath || '')),
    [activeConvId, readerPayload?.sourcePath],
  )
  const activeReaderSessionHighlights = useMemo(
    () => (activeReaderHighlightScope ? readerSessionHighlights[activeReaderHighlightScope] || [] : []),
    [activeReaderHighlightScope, readerSessionHighlights],
  )
  const addReaderSessionHighlight = (highlight: ReaderSessionHighlight) => {
    const scopeKey = activeReaderHighlightScope
    if (!scopeKey) return
    setReaderSessionHighlights((current) => {
      const list = Array.isArray(current[scopeKey]) ? current[scopeKey] : []
      if (list.some((item) => sameReaderSessionHighlight(item, highlight))) {
        return current
      }
      return {
        ...current,
        [scopeKey]: [...list, highlight],
      }
    })
  }
  const removeReaderSessionHighlight = (highlightId: string) => {
    const scopeKey = activeReaderHighlightScope
    const targetId = String(highlightId || '').trim()
    if (!scopeKey || !targetId) return
    setReaderSessionHighlights((current) => {
      const list = Array.isArray(current[scopeKey]) ? current[scopeKey] : []
      const next = list.filter((item) => String(item.id || '').trim() !== targetId)
      if (next.length === list.length) return current
      return {
        ...current,
        [scopeKey]: next,
      }
    })
  }

  const desktopReaderVisible = readerOpen && desktopReaderEligible
  const desktopReaderExpanded = desktopReaderVisible && !readerCollapsed
  const timelineUiReady = !conversationLoading && timelineItems.length > 0
  const showDesktopTimeline = timelineUiReady && timelineOpen && !desktopReaderExpanded
  const showInlineTimelineToggle = timelineUiReady && (!desktopReaderEligible || desktopReaderExpanded)
  const showDesktopTimelineToggleRail = timelineUiReady && desktopReaderEligible && !desktopReaderExpanded && !timelineOpen
  const showConversationMeta = !conversationLoading && (timelineUiReady || activeConversation?.mode === 'paper_guide')
  const guideSourceLabel = String(activeConversation?.bound_source_name || '').trim()
    || String(activeConversation?.bound_source_path || '').trim()
    || '未绑定文献'
  const guideSourceReady = Boolean(activeConversation?.bound_source_ready)
  const guideStatusLabel = guideSourceReady ? '已入库可检索' : '待入库'
  const beginReaderResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!desktopReaderExpanded || !event.isPrimary) return
    const currentWidth = clampReaderWidth(readerWidthLiveRef.current)
    readerResizeRef.current = {
      startX: event.clientX,
      startWidth: currentWidth,
    }
    readerActivePointerIdRef.current = event.pointerId
    readerResizePreviewWidthRef.current = currentWidth
    updateReaderResizeGuide(currentWidth)
    const activeElement = document.activeElement
    if (
      activeElement instanceof HTMLElement
      && (activeElement.tagName === 'TEXTAREA' || activeElement.tagName === 'INPUT')
    ) {
      readerResizeFocusRestoreRef.current = activeElement
      activeElement.blur()
    } else {
      readerResizeFocusRestoreRef.current = null
    }
    setReaderResizing(true)
    document.body.classList.add('kb-reader-resizing')
    document.body.style.setProperty('cursor', 'col-resize')
    document.body.style.setProperty('user-select', 'none')
    event.currentTarget.setPointerCapture(event.pointerId)
    event.preventDefault()
  }

  const handleReaderResizeMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (readerActivePointerIdRef.current !== event.pointerId) return
    const state = readerResizeRef.current
    if (!state) return
    updateReaderResizeGuide(state.startWidth + (state.startX - event.clientX))
    event.preventDefault()
  }

  const commitReaderResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (readerActivePointerIdRef.current !== event.pointerId) return
    finishReaderResize(true)
    event.preventDefault()
  }

  const cancelReaderResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (readerActivePointerIdRef.current !== event.pointerId) return
    finishReaderResize(false)
    event.preventDefault()
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      {!activeConvId && messages.length === 0 ? (
        <div className="kb-empty-state flex flex-1 flex-col items-center justify-center gap-4 px-4">
          <div className="kb-empty-brand">
            <div className="kb-empty-logo-wrap flex h-14 w-14 items-center justify-center overflow-hidden rounded-full">
              <img src="/pi_logo.png" alt="Pi assistant" className="kb-empty-logo h-9 w-9 object-contain" loading="lazy" />
            </div>
            <div className="kb-empty-typewriter" aria-label="π-zaya · 你的知识库助理">
              π-zaya · 你的知识库助理
            </div>
          </div>
          <Text type="secondary" className="max-w-xs text-center">
            {S.no_msgs}
          </Text>
        </div>
      ) : (
        <>
          {showLegacyUiBlocks ? (
            <div className="border-b border-[var(--border)] bg-[var(--panel)]/60 px-4 py-3">
              <div className="mx-auto flex max-w-5xl items-center gap-3">
                <Button size="small" loading={messagesLoadingMore} onClick={() => { void loadOlderMessages() }}>
                  显示更早 {Math.min(HISTORY_PAGE_SIZE, hiddenCount)} 条
                </Button>
                <Button size="small" onClick={() => {}}>
                  展开全部
                </Button>
                <Text type="secondary" className="text-xs">
                  为了打开更快，当前先显示最近 {visibleMessages.length} 条消息，较早消息 {hiddenCount} 条已折叠。
                </Text>
              </div>
            </div>
          ) : null}

          {showLegacyUiBlocks ? (
            <div className="border-b border-[var(--border)] bg-[var(--panel)]/40 px-4 py-2">
              <div className="mx-auto flex max-w-5xl items-center gap-3">
                <Button size="small" onClick={() => {}}>
                  收起较早消息
                </Button>
                <Text type="secondary" className="text-xs">
                  当前已展开 {visibleMessages.length} 条消息。
                </Text>
              </div>
            </div>
          ) : null}

          {!liveRunning && messagesHasMoreBefore ? (
            <div className="border-b border-[var(--border)] bg-[var(--panel)]/60 px-4 py-3">
              <div className="mx-auto flex max-w-5xl items-center gap-3">
                <Button size="small" loading={messagesLoadingMore} onClick={() => { void loadOlderMessages() }}>
                  显示更早 {HISTORY_PAGE_SIZE} 条
                </Button>
                <Text type="secondary" className="text-xs">
                  当前先加载最近一页消息；较早消息按需分页加载。
                </Text>
              </div>
            </div>
          ) : null}

          {conversationLoading ? (
            <div className="border-b border-[var(--border)] bg-[var(--panel)]/40 px-4 py-2">
              <div className="mx-auto max-w-5xl">
                <Text type="secondary" className="text-xs">
                  正在打开会话并加载最近消息...
                </Text>
              </div>
            </div>
          ) : null}

          {liveRunning && hiddenCount > 0 ? (
            <div className="border-b border-[var(--border)] bg-[var(--panel)]/40 px-4 py-2">
              <div className="mx-auto max-w-5xl">
                <Text type="secondary" className="text-xs">
                  流式生成时仅保留最近 {visibleMessages.length} 条消息可见，以保持界面流畅。
                </Text>
              </div>
            </div>
          ) : null}

          {showConversationMeta ? (
            <div className="px-4 pb-2 pt-3">
              <div className="mx-auto max-w-7xl">
                <section className="kb-chat-meta-shell">
                  <div className="kb-chat-meta-strip">
                    {timelineItems.length > 0 ? (
                      <div className="kb-chat-meta-inline-block">
                        <span className="kb-chat-meta-label">会话时间线</span>
                        <span className="kb-chat-meta-badge">{timelineItems.length} 个节点</span>
                        {showInlineTimelineToggle ? (
                          <Button
                            size="small"
                            type="text"
                            className="kb-chat-meta-action"
                            onClick={toggleTimelineOpen}
                          >
                          {timelineOpen ? '收起' : '展开'}
                          </Button>
                        ) : null}
                      </div>
                    ) : null}
                    {activeConversation?.mode === 'paper_guide' ? (
                      <div className="kb-chat-meta-inline-block kb-chat-meta-inline-guide">
                        <span className="kb-chat-meta-label">阅读指导</span>
                        <span className="kb-chat-meta-source" title={guideSourceLabel}>{guideSourceLabel}</span>
                        <span className={`kb-chat-meta-state ${guideSourceReady ? 'is-ready' : 'is-pending'}`}>
                          {guideStatusLabel}
                        </span>
                      </div>
                    ) : null}
                  </div>
                  {timelineOpen && timelineItems.length > 0 ? (
                    <div className="kb-chat-meta-mobile-rail lg:hidden">
                      <div className="flex gap-2 overflow-x-auto">
                        {timelineItems.map((item) => (
                          <button
                            key={`m-timeline-mobile-${item.userMsgId}-${item.order}`}
                            type="button"
                            className={`kb-chat-meta-rail-pill ${activeTimelineUserMsgId === item.userMsgId ? 'is-active' : ''}`}
                            onClick={() => jumpToTimelineItem(item)}
                          >
                            Q{item.order}
                          </button>
                        ))}
                      </div>
                    </div>
                  ) : null}
                </section>
              </div>
            </div>
          ) : null}

          {showLegacyUiBlocks ? (
            <div className="border-b border-[var(--border)] bg-[var(--panel)]/30 px-4 py-2">
              <div className="mx-auto flex max-w-7xl items-center justify-between gap-3">
                <Text type="secondary" className="text-xs">
                  会话时间线：{timelineItems.length} 个提问节点
                </Text>
                <Button size="small" onClick={() => setTimelineOpen((v) => !v)}>
                  {timelineOpen ? '收起时间线' : '打开时间线'}
                </Button>
              </div>
            </div>
          ) : null}

          {showLegacyUiBlocks ? (
            <div className="border-b border-[var(--border)] bg-[var(--panel)]/20 px-3 py-2 lg:hidden">
              <div className="flex gap-2 overflow-x-auto">
                {timelineItems.map((item) => (
                  <button
                    key={`m-timeline-mobile-${item.userMsgId}-${item.order}`}
                    type="button"
                    className={`shrink-0 rounded-full border px-3 py-1 text-xs ${
                      activeTimelineUserMsgId === item.userMsgId
                        ? 'border-[var(--accent)] bg-[var(--accent)]/10 text-[var(--accent)]'
                        : 'border-[var(--border)] bg-[var(--panel)] text-black/70 dark:text-white/70'
                    }`}
                    onClick={() => jumpToTimelineItem(item)}
                  >
                    Q{item.order}
                  </button>
                ))}
              </div>
            </div>
          ) : null}

          {showLegacyUiBlocks ? (
            <div className="border-b border-[var(--border)] bg-[var(--panel)]/40 px-4 py-2">
              <div className="mx-auto flex max-w-7xl items-center justify-between gap-3">
                <Text className="text-xs">
                  阅读指导模式：
                  <span className="ml-1 font-medium">
                    {guideSourceLabel}
                  </span>
                </Text>
                <Text type="secondary" className="text-xs">
                  {guideStatusLabel}
                </Text>
              </div>
            </div>
          ) : null}

          <div ref={splitLayoutRef} className="relative flex min-h-0 flex-1">
            <div className="flex min-h-0 min-w-0 flex-1 flex-col">
              <div className="flex min-h-0 min-w-0 flex-1 flex-col">
                {conversationLoading ? (
                  <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-hidden px-6 py-6">
                    <div className="h-5 w-40 animate-pulse rounded-full bg-black/[0.06] dark:bg-white/[0.08]" />
                    <div className="ml-auto h-24 w-[68%] animate-pulse rounded-[28px] bg-black/[0.05] dark:bg-white/[0.06]" />
                    <div className="h-32 w-[82%] animate-pulse rounded-[28px] bg-black/[0.04] dark:bg-white/[0.05]" />
                    <div className="ml-auto h-20 w-[58%] animate-pulse rounded-[28px] bg-black/[0.05] dark:bg-white/[0.06]" />
                  </div>
                ) : (
                  <MessageList
                    activeConvId={activeConvId}
                    messages={visibleMessages}
                    refs={refs}
                    generationPartial={generation?.partial}
                    generationStage={generation?.stage}
                    jumpTarget={timelineJump}
                    onJumpHandled={(handled) => {
                      setTimelineJump((current) => (
                        current?.token === handled.token && current?.messageId === handled.messageId
                          ? null
                          : current
                      ))
                      setActiveTimelineUserMsgId(null)
                    }}
                    trackedMessageIds={timelineTrackedMessageIds}
                    onTrackedMessageActive={(messageId) => {
                      const nextUserMsgId = messageId != null
                        ? (timelineUserMsgIdByTargetMsgId.get(messageId) ?? null)
                        : null
                      startTransition(() => {
                        setActiveTimelineUserMsgId((current) => (
                          current === nextUserMsgId ? current : nextUserMsgId
                        ))
                      })
                    }}
                    onOpenReader={openReader}
                    paperGuideSourcePath={effectiveGuide.sourcePath}
                    paperGuideSourceName={effectiveGuide.sourceName}
                  />
                )}
              </div>
              <ChatInput
                onSend={onSend}
                onStop={cancelGen}
                onUpload={onUpload}
                onRetryUploadItem={onRetryUpload}
                onCancelUploadItem={onCancelUpload}
                onRemoveImage={removePendingImage}
                onDismissUploadItem={dismissUploadItem}
                onStartGuideFromUpload={onStartGuideFromUpload}
                uploadItems={uploadItems}
                pendingImages={pendingImages}
                uploading={uploading}
                generating={!!generation}
                appendSignal={appendSignal}
              />
            </div>
            {desktopReaderExpanded ? (
              <div
                ref={readerResizeGuideRef}
                className={`pointer-events-none absolute inset-y-0 z-20 hidden w-0 xl:block ${
                  readerResizing ? 'opacity-100' : 'opacity-0'
                }`}
                aria-hidden="true"
              >
                <div className="absolute inset-y-0 -translate-x-1/2 border-l-2 border-[var(--accent)]/75 shadow-[0_0_0_1px_rgba(22,119,255,0.15)]" />
              </div>
            ) : null}
            {showDesktopTimelineToggleRail ? (
              <div className="pointer-events-none absolute inset-y-0 right-0 z-20 hidden items-start lg:flex">
                <button
                  type="button"
                  className="kb-chat-timeline-rail-toggle pointer-events-auto"
                  aria-label={TIMELINE_RAIL_LABEL}
                  onClick={toggleTimelineOpen}
                >
                  鏃堕棿绾?
                </button>
              </div>
            ) : null}
            {showDesktopTimeline ? (
              <aside className="kb-chat-timeline hidden h-full shrink-0 border-l border-[var(--border)] lg:flex">
                <div className="kb-chat-timeline-shell">
                  <div className="kb-chat-timeline-head">
                    <div className="kb-chat-timeline-title-row">
                      <div className="min-w-0">
                        <div className="kb-chat-timeline-title">会话时间线</div>
                        <div className="kb-chat-timeline-hint">点击节点跳转到对应问答</div>
                      </div>
                      <div className="kb-chat-timeline-head-actions">
                        <span className="kb-chat-timeline-count">{timelineItems.length}</span>
                        <button
                          type="button"
                          className="kb-chat-timeline-toggle"
                          onClick={toggleTimelineOpen}
                        >
                          收起
                        </button>
                      </div>
                    </div>
                    <div className="text-sm font-medium">会话时间线</div>
                    <div className="mt-1 text-xs text-black/50 dark:text-white/50">点击可跳到对应问答</div>
                  </div>
                  <div className="kb-chat-timeline-list">
                    {timelineItems.map((item) => (
                      <button
                        key={`m-timeline-${item.userMsgId}-${item.order}`}
                        type="button"
                        className={`kb-chat-timeline-item ${
                          activeTimelineUserMsgId === item.userMsgId
                            ? 'is-active'
                            : ''
                        }`}
                        onClick={() => jumpToTimelineItem(item)}
                      >
                        <div className="kb-chat-timeline-item-meta">
                          <span className="kb-chat-timeline-item-order">Q{item.order}</span>
                          <span className={`kb-chat-timeline-item-status ${item.hasAnswer ? 'is-ready' : 'is-pending'}`}>
                            {item.hasAnswer ? '已回答' : '待回答'}
                          </span>
                        </div>
                        <div className="kb-chat-timeline-item-text">
                          {item.questionPreview}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              </aside>
            ) : null}
            {desktopReaderVisible ? (
              <aside className="hidden h-full shrink-0 xl:flex">
                {!readerCollapsed ? (
                  <div
                    className={`w-2 shrink-0 cursor-col-resize transition ${
                      readerResizing
                        ? 'bg-[var(--accent)]/30'
                        : 'bg-transparent hover:bg-black/[0.05] dark:hover:bg-white/[0.05]'
                    }`}
                    onPointerDown={beginReaderResize}
                    onPointerMove={handleReaderResizeMove}
                    onPointerUp={commitReaderResize}
                    onPointerCancel={cancelReaderResize}
                  />
                ) : null}
                <div
                  className={`flex h-full shrink-0 border-l border-[var(--border)] ${
                    readerResizing ? 'bg-[var(--panel)]' : 'bg-[var(--panel)]/70 backdrop-blur-sm'
                  } ${
                    readerCollapsed ? 'w-12' : ''
                  }`}
                  style={readerCollapsed ? undefined : {
                    width: `${readerWidth}px`,
                    transition: readerResizing ? 'none' : DESKTOP_READER_WIDTH_TRANSITION,
                  }}
                >
                  {readerCollapsed ? (
                    <div className="flex h-full w-12 flex-col items-center justify-between py-3">
                      <button
                        type="button"
                        className="rounded-full border border-[var(--border)] bg-[var(--panel)] px-2 py-1 text-xs text-black/65 transition hover:border-[var(--accent)] hover:text-[var(--accent)] dark:text-white/65"
                        onClick={() => setReaderCollapsed(false)}
                        title="Expand reader"
                      >
                        {'<'}
                      </button>
                      <div className="flex flex-1 items-center justify-center px-1">
                        <div className="[writing-mode:vertical-rl] rotate-180 text-xs text-black/45 dark:text-white/45">
                          {stripSourceExt(String(readerPayload?.sourceName || '').trim()) || 'Reader'}
                        </div>
                      </div>
                      <button
                        type="button"
                        className="rounded-full border border-[var(--border)] bg-[var(--panel)] px-2 py-1 text-xs text-black/65 transition hover:border-[var(--accent)] hover:text-[var(--accent)] dark:text-white/65"
                        onClick={() => setReaderOpen(false)}
                        title="Close reader"
                      >
                        x
                      </button>
                    </div>
                  ) : (
                    <PaperGuideReaderDrawer
                      open={readerOpen}
                      payload={readerPayload}
                      onClose={() => setReaderOpen(false)}
                      onAppendSelection={appendReaderSelection}
                      presentation="inline"
                      onCollapse={() => setReaderCollapsed(true)}
                      sessionHighlights={activeReaderSessionHighlights}
                      onAddSessionHighlight={addReaderSessionHighlight}
                      onRemoveSessionHighlight={removeReaderSessionHighlight}
                    />
                  )}
                </div>
              </aside>
            ) : null}
          </div>
        </>
      )}
      {!desktopReaderVisible ? (
        <PaperGuideReaderDrawer
          open={readerOpen}
          payload={readerPayload}
          onClose={() => setReaderOpen(false)}
          onAppendSelection={appendReaderSelection}
          sessionHighlights={activeReaderSessionHighlights}
          onAddSessionHighlight={addReaderSessionHighlight}
          onRemoveSessionHighlight={removeReaderSessionHighlight}
        />
      ) : null}
    </div>
  )
}
