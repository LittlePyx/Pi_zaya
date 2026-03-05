import { useEffect, useRef, useState } from 'react'
import { Button, message, Typography } from 'antd'
import { MessageOutlined } from '@ant-design/icons'
import { useChatStore } from '../stores/chatStore'
import { useSettingsStore } from '../stores/settingsStore'
import { MessageList } from '../components/chat/MessageList'
import { ChatInput } from '../components/chat/ChatInput'
import type { ChatUploadItem } from '../api/chat'
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

export default function ChatPage() {
  const messages = useChatStore((s) => s.messages)
  const generation = useChatStore((s) => s.generation)
  const refs = useChatStore((s) => s.refs)
  const activeConvId = useChatStore((s) => s.activeConvId)
  const uploadItems = useChatStore((s) => s.uploadItems)
  const pendingImages = useChatStore((s) => s.pendingImages)
  const uploading = useChatStore((s) => s.uploading)
  const uploadFiles = useChatStore((s) => s.uploadFiles)
  const retryUploadItem = useChatStore((s) => s.retryUploadItem)
  const cancelUploadItem = useChatStore((s) => s.cancelUploadItem)
  const removePendingImage = useChatStore((s) => s.removePendingImage)
  const dismissUploadItem = useChatStore((s) => s.dismissUploadItem)
  const sendMessage = useChatStore((s) => s.sendMessage)
  const cancelGen = useChatStore((s) => s.cancelGeneration)
  const settings = useSettingsStore()
  const [visibleCount, setVisibleCount] = useState(HISTORY_PAGE_SIZE)
  const uploadNoticeRef = useRef<Record<string, string>>({})
  const dismissTimerRef = useRef<Record<string, number>>({})

  useEffect(() => {
    setVisibleCount(HISTORY_PAGE_SIZE)
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
      deepRead: settings.deepRead,
    })
  }

  const onUpload = async (files: File[]) => {
    try {
      await uploadFiles(files, { quickIngest: true, speedMode: 'ultra_fast' })
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

  const liveRunning = Boolean(generation)
  const effectiveVisible = liveRunning
    ? Math.min(messages.length, LIVE_WINDOW)
    : Math.min(messages.length, visibleCount)
  const visibleMessages = effectiveVisible > 0 ? messages.slice(-effectiveVisible) : messages
  const hiddenCount = Math.max(0, messages.length - visibleMessages.length)

  return (
    <div className="flex h-full min-h-0 flex-col">
      {!activeConvId && messages.length === 0 ? (
        <div className="flex flex-1 flex-col items-center justify-center gap-3 px-4">
          <div className="flex h-14 w-14 items-center justify-center rounded-full bg-[var(--accent)] opacity-20">
            <MessageOutlined className="text-2xl text-white" />
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

          <MessageList
            activeConvId={activeConvId}
            messages={visibleMessages}
            refs={refs}
            generationPartial={generation?.partial}
            generationStage={generation?.stage}
          />
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
        uploadItems={uploadItems}
        pendingImages={pendingImages}
        uploading={uploading}
        generating={!!generation}
      />
    </div>
  )
}
