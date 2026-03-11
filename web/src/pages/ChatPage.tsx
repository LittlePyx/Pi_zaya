import { useEffect, useMemo, useRef, useState } from 'react'
import { Button, message, Typography } from 'antd'
import { useChatStore } from '../stores/chatStore'
import { useSettingsStore } from '../stores/settingsStore'
import { MessageList } from '../components/chat/MessageList'
import { ChatInput } from '../components/chat/ChatInput'
import { PaperGuideReaderDrawer, type ReaderOpenPayload } from '../components/chat/PaperGuideReaderDrawer'
import type { ChatUploadItem, Message } from '../api/chat'
import { S } from '../i18n/zh'

const { Text } = Typography

const HISTORY_PAGE_SIZE = 24
const LIVE_WINDOW = 16
const READY_DISMISS_MS = 2600
const DUPLICATE_DISMISS_MS = 3600

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

interface TimelineItem {
  order: number
  userMsgId: number
  targetMsgId: number
  questionPreview: string
  hasAnswer: boolean
}

export default function ChatPage() {
  const messages = useChatStore((s) => s.messages)
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
  const [visibleCount, setVisibleCount] = useState(HISTORY_PAGE_SIZE)
  const [timelineOpen, setTimelineOpen] = useState(true)
  const [timelineJump, setTimelineJump] = useState<{ messageId: number; token: number } | null>(null)
  const [activeTimelineUserMsgId, setActiveTimelineUserMsgId] = useState<number | null>(null)
  const [readerOpen, setReaderOpen] = useState(false)
  const [readerPayload, setReaderPayload] = useState<ReaderOpenPayload | null>(null)
  const [appendSignal, setAppendSignal] = useState<{ token: number; text: string } | null>(null)
  const uploadNoticeRef = useRef<Record<string, string>>({})
  const dismissTimerRef = useRef<Record<string, number>>({})
  const timelineJumpTokenRef = useRef(1)

  const nextEventToken = () => {
    timelineJumpTokenRef.current += 1
    return timelineJumpTokenRef.current
  }

  useEffect(() => {
    setVisibleCount(HISTORY_PAGE_SIZE)
    setTimelineOpen(true)
    setTimelineJump(null)
    setActiveTimelineUserMsgId(null)
    setReaderOpen(false)
    setReaderPayload(null)
    setAppendSignal(null)
  }, [activeConvId])

  useEffect(() => () => {
    Object.values(dismissTimerRef.current).forEach((timer) => window.clearTimeout(timer))
    dismissTimerRef.current = {}
  }, [])

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
  const effectiveVisible = liveRunning
    ? Math.min(messages.length, LIVE_WINDOW)
    : Math.min(messages.length, visibleCount)
  const visibleMessages = effectiveVisible > 0 ? messages.slice(-effectiveVisible) : messages
  const hiddenCount = Math.max(0, messages.length - visibleMessages.length)
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
    for (const msg of messages) {
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
  }, [messages])
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
    const requiredVisible = messages.length - idx
    if (requiredVisible > visibleCount) {
      setVisibleCount(requiredVisible)
    }
    setActiveTimelineUserMsgId(item.userMsgId)
    const token = nextEventToken()
    const delayMs = requiredVisible > effectiveVisible ? 120 : 0
    window.setTimeout(() => {
      setTimelineJump({ messageId: item.targetMsgId, token })
    }, delayMs)
  }

  const openReader = (payload: ReaderOpenPayload) => {
    const sourcePath = String(payload?.sourcePath || '').trim()
    if (!sourcePath) {
      message.info('当前引用缺少可绑定的文献路径')
      return
    }
    setReaderPayload({
      sourcePath,
      sourceName: String(payload.sourceName || '').trim(),
      headingPath: String(payload.headingPath || '').trim(),
      snippet: String(payload.snippet || '').trim(),
      blockId: String(payload.blockId || '').trim() || undefined,
      anchorId: String(payload.anchorId || '').trim() || undefined,
      alternatives: Array.isArray(payload.alternatives)
        ? payload.alternatives.map((item) => ({
          headingPath: String(item?.headingPath || '').trim(),
          snippet: String(item?.snippet || '').trim(),
          blockId: String(item?.blockId || '').trim() || undefined,
          anchorId: String(item?.anchorId || '').trim() || undefined,
        }))
        : undefined,
      initialAltIndex: Number.isFinite(Number(payload.initialAltIndex))
        ? Number(payload.initialAltIndex)
        : undefined,
    })
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
          {!liveRunning && hiddenCount > 0 ? (
            <div className="border-b border-[var(--border)] bg-[var(--panel)]/60 px-4 py-3">
              <div className="mx-auto flex max-w-5xl items-center gap-3">
                <Button size="small" onClick={() => setVisibleCount((v) => v + HISTORY_PAGE_SIZE)}>
                  显示更早 {Math.min(HISTORY_PAGE_SIZE, hiddenCount)} 条
                </Button>
                <Button size="small" onClick={() => setVisibleCount(messages.length)}>
                  展开全部
                </Button>
                <Text type="secondary" className="text-xs">
                  为了打开更快，当前先显示最近 {visibleMessages.length} 条消息，较早消息 {hiddenCount} 条已折叠。
                </Text>
              </div>
            </div>
          ) : null}

          {!liveRunning && messages.length > HISTORY_PAGE_SIZE && visibleMessages.length > HISTORY_PAGE_SIZE ? (
            <div className="border-b border-[var(--border)] bg-[var(--panel)]/40 px-4 py-2">
              <div className="mx-auto flex max-w-5xl items-center gap-3">
                <Button size="small" onClick={() => setVisibleCount(HISTORY_PAGE_SIZE)}>
                  收起较早消息
                </Button>
                <Text type="secondary" className="text-xs">
                  当前已展开 {visibleMessages.length} 条消息。
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

          {timelineItems.length > 0 ? (
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

          {timelineOpen && timelineItems.length > 0 ? (
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

          {activeConversation?.mode === 'paper_guide' ? (
            <div className="border-b border-[var(--border)] bg-[var(--panel)]/40 px-4 py-2">
              <div className="mx-auto flex max-w-7xl items-center justify-between gap-3">
                <Text className="text-xs">
                  阅读指导模式：
                  <span className="ml-1 font-medium">
                    {String(activeConversation.bound_source_name || '').trim() || String(activeConversation.bound_source_path || '').trim() || '未绑定文献'}
                  </span>
                </Text>
                <Text type="secondary" className="text-xs">
                  {activeConversation.bound_source_ready ? '已入库可检索' : '待入库'}
                </Text>
              </div>
            </div>
          ) : null}

          <div className="flex min-h-0 flex-1">
            <div className="flex min-h-0 min-w-0 flex-1 flex-col">
              <MessageList
                activeConvId={activeConvId}
                messages={visibleMessages}
                refs={refs}
                generationPartial={generation?.partial}
                generationStage={generation?.stage}
                jumpTarget={timelineJump}
                onOpenReader={openReader}
                paperGuideSourcePath={effectiveGuide.sourcePath}
                paperGuideSourceName={effectiveGuide.sourceName}
              />
            </div>
            {timelineOpen && timelineItems.length > 0 ? (
              <aside className="hidden h-full w-[300px] shrink-0 border-l border-[var(--border)] bg-[var(--panel)]/55 lg:flex">
                <div className="flex h-full min-h-0 w-full flex-col">
                  <div className="border-b border-[var(--border)] px-3 py-3">
                    <div className="text-sm font-medium">会话时间线</div>
                    <div className="mt-1 text-xs text-black/50 dark:text-white/50">点击可跳到对应问答</div>
                  </div>
                  <div className="flex-1 space-y-2 overflow-y-auto px-2 py-2">
                    {timelineItems.map((item) => (
                      <button
                        key={`m-timeline-${item.userMsgId}-${item.order}`}
                        type="button"
                        className={`w-full rounded-xl border px-3 py-2 text-left transition ${
                          activeTimelineUserMsgId === item.userMsgId
                            ? 'border-[var(--accent)] bg-[var(--accent)]/10'
                            : 'border-[var(--border)] bg-[var(--panel)] hover:bg-black/[0.03] dark:hover:bg-white/[0.04]'
                        }`}
                        onClick={() => jumpToTimelineItem(item)}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-xs font-medium text-[var(--accent)]">Q{item.order}</span>
                          <span className="text-[11px] text-black/45 dark:text-white/45">
                            {item.hasAnswer ? '已回答' : '待回答'}
                          </span>
                        </div>
                        <div
                          className="mt-1 text-xs text-black/75 dark:text-white/75"
                          style={{
                            display: '-webkit-box',
                            WebkitLineClamp: 2,
                            WebkitBoxOrient: 'vertical',
                            overflow: 'hidden',
                          }}
                        >
                          {item.questionPreview}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              </aside>
            ) : null}
          </div>
        </>
      )}
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
      <PaperGuideReaderDrawer
        open={readerOpen}
        payload={readerPayload}
        onClose={() => setReaderOpen(false)}
        onAppendSelection={appendReaderSelection}
      />
    </div>
  )
}
