import { useCallback, useEffect, useRef } from 'react'
import { message } from 'antd'
import type { ChatUploadItem } from '../../api/chat'
import { useChatStore } from '../../stores/chatStore'

const READY_DISMISS_MS = 2600
const DUPLICATE_DISMISS_MS = 3600

function uploadItemKey(item: ChatUploadItem) {
  if (item.kind === 'pdf' && item.ingest_job_id) {
    return `pdf-job:${item.ingest_job_id}`
  }
  return [item.kind, item.sha1 || '', item.path || '', item.name].join(':')
}

function stripSourceExt(name: string) {
  return String(name || '')
    .replace(/\.en\.md$/i, '')
    .replace(/\.md$/i, '')
    .replace(/\.pdf$/i, '')
    .trim()
}

function uploadTerminalState(item: ChatUploadItem) {
  if (item.status === 'duplicate') return 'duplicate'
  if (item.ingest_status === 'cancelled') return 'cancelled'
  if (item.status === 'error' || item.ingest_status === 'error') return 'error'
  if (item.ready || item.ingest_status === 'ready') return 'ready'
  return ''
}

export function useChatUploadFlow(labels: Record<string, string>) {
  const uploadItems = useChatStore((s) => s.uploadItems)
  const uploadFiles = useChatStore((s) => s.uploadFiles)
  const retryUploadItem = useChatStore((s) => s.retryUploadItem)
  const cancelUploadItem = useChatStore((s) => s.cancelUploadItem)
  const dismissUploadItem = useChatStore((s) => s.dismissUploadItem)
  const createPaperGuideConversation = useChatStore((s) => s.createPaperGuideConversation)
  const uploadNoticeRef = useRef<Record<string, string>>({})
  const dismissTimerRef = useRef<Record<string, number>>({})

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
      const terminalState = uploadTerminalState(item)
      if (!terminalState || uploadNoticeRef.current[key] === terminalState) {
        continue
      }
      uploadNoticeRef.current[key] = terminalState
      if (terminalState === 'ready') {
        message.success(`${labels.upload_pdf_ready}: ${item.name}`)
        if (dismissTimerRef.current[key] == null) {
          dismissTimerRef.current[key] = window.setTimeout(() => {
            dismissUploadItem(key)
            delete dismissTimerRef.current[key]
          }, READY_DISMISS_MS)
        }
      } else if (terminalState === 'duplicate') {
        message.info(`${labels.upload_pdf_duplicate}: ${item.name}`)
        if (dismissTimerRef.current[key] == null) {
          dismissTimerRef.current[key] = window.setTimeout(() => {
            dismissUploadItem(key)
            delete dismissTimerRef.current[key]
          }, DUPLICATE_DISMISS_MS)
        }
      } else if (terminalState === 'cancelled') {
        message.info(`${labels.upload_pdf_cancelled}: ${item.name}`)
      } else if (terminalState === 'error') {
        message.error(`${labels.upload_pdf_error}: ${item.name}`)
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
  }, [
    dismissUploadItem,
    labels.upload_pdf_cancelled,
    labels.upload_pdf_duplicate,
    labels.upload_pdf_error,
    labels.upload_pdf_ready,
    uploadItems,
  ])

  const onUpload = useCallback(async (files: File[]) => {
    try {
      await uploadFiles(files, { quickIngest: true, speedMode: 'balanced' })
    } catch {
      message.error(labels.upload_failed_generic)
    }
  }, [labels.upload_failed_generic, uploadFiles])

  const onRetryUpload = useCallback(async (key: string) => {
    try {
      await retryUploadItem(key)
    } catch (err) {
      message.error(err instanceof Error ? err.message : labels.retry_ingest_failed)
    }
  }, [labels.retry_ingest_failed, retryUploadItem])

  const onCancelUpload = useCallback(async (key: string) => {
    try {
      await cancelUploadItem(key)
    } catch (err) {
      message.error(err instanceof Error ? err.message : labels.cancel_ingest_failed)
    }
  }, [cancelUploadItem, labels.cancel_ingest_failed])

  const onStartGuideFromUpload = useCallback(async (item: ChatUploadItem) => {
    const sourcePath = String(item.md_path || '').trim()
    if (!sourcePath) {
      message.info(labels.reader_pdf_not_ready)
      return
    }
    const sourceName = stripSourceExt(item.name) || item.name
    const hide = message.loading(labels.reader_creating_guide, 0)
    try {
      await createPaperGuideConversation({
        sourcePath,
        sourceName,
        title: labels.default_guide_title.replace('{name}', sourceName),
      })
      hide()
      message.success(labels.reader_entered_guide)
    } catch (err) {
      hide()
      message.error(err instanceof Error ? err.message : labels.reader_create_guide_failed)
    }
  }, [
    createPaperGuideConversation,
    labels.default_guide_title,
    labels.reader_create_guide_failed,
    labels.reader_creating_guide,
    labels.reader_entered_guide,
    labels.reader_pdf_not_ready,
  ])

  return {
    onUpload,
    onRetryUpload,
    onCancelUpload,
    onDismissUploadItem: dismissUploadItem,
    onStartGuideFromUpload,
  }
}
