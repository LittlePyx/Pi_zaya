import {
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type ClipboardEvent,
  type DragEvent,
  type KeyboardEvent,
} from 'react'
import { Button, Input, Tag, message } from 'antd'
import { BookOutlined, CloseOutlined, PaperClipOutlined, PauseOutlined, RedoOutlined, RobotOutlined, SendOutlined, StopOutlined } from '@ant-design/icons'
import type { ChatImageAttachment, ChatUploadItem, QueryScope } from '../../api/chat'
import { useT } from '../../i18n'
import { ChatScopeControl } from './ChatScopeControl'

const { TextArea } = Input

interface Props {
  onSend: (text: string) => void
  onStop: () => void
  onUpload: (files: File[]) => Promise<void>
  onRetryUploadItem: (key: string) => Promise<void>
  onCancelUploadItem: (key: string) => Promise<void>
  onRemoveImage: (key: string) => void
  onDismissUploadItem: (key: string) => void
  onStartGuideFromUpload: (item: ChatUploadItem) => void
  uploadItems: ChatUploadItem[]
  pendingImages: ChatImageAttachment[]
  uploading: boolean
  generating: boolean
  appendSignal?: { token: number; text: string } | null
  queryScope: QueryScope
  queryScopeOptions: Array<{ value: QueryScope; disabled?: boolean }>
  onQueryScopeChange: (scope: QueryScope) => void
  agentMode: boolean
  onAgentModeChange: (enabled: boolean) => void
}

const IMAGE_EXTS = new Set(['png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp'])
const PDF_EXTS = new Set(['pdf'])
const MAX_IMAGE_BYTES = 8 * 1024 * 1024
const MAX_PDF_BYTES = 80 * 1024 * 1024

function uploadItemKey(item: ChatUploadItem) {
  if (item.kind === 'pdf' && item.ingest_job_id) {
    return `pdf-job:${item.ingest_job_id}`
  }
  return [item.kind, item.sha1 || '', item.path || '', item.name].join(':')
}

function imageKey(item: ChatImageAttachment) {
  return item.sha1 || item.path
}

function fileExt(name: string) {
  const text = String(name || '').trim().toLowerCase()
  const idx = text.lastIndexOf('.')
  return idx >= 0 ? text.slice(idx + 1) : ''
}

function isImageAttachment(item: ChatImageAttachment) {
  return String(item.mime || '').toLowerCase().startsWith('image/')
}

function isPdfIngestRunning(item: ChatUploadItem) {
  return item.kind === 'pdf' && ['processing', 'renaming', 'converting', 'ingesting'].includes(String(item.ingest_status || ''))
}

function isPdfQualityRunning(item: ChatUploadItem) {
  return item.kind === 'pdf' && ['pending', 'running'].includes(String(item.quality_status || ''))
}

function uploadTone(item: ChatUploadItem): 'default' | 'success' | 'error' | 'warning' {
  if (item.status === 'error') return 'error'
  if (item.status === 'unsupported') return 'warning'
  if (item.status === 'duplicate') return 'default'
  if (item.kind === 'pdf' && item.ingest_status === 'cancelled') return 'default'
  if (item.kind === 'pdf' && item.ready === false) return 'warning'
  if (item.kind === 'pdf' && item.quality_status === 'error') return 'warning'
  if (item.kind === 'pdf' && item.quality_status === 'cancelled') return 'default'
  return 'success'
}

function uploadLabel(item: ChatUploadItem, S: Record<string, string>) {
  if (item.kind === 'image') {
    return item.status === 'duplicate' ? S.upload_image_attached : S.upload_image_pending
  }
  if (item.kind === 'pdf') {
    if (item.status === 'duplicate') return S.upload_pdf_duplicate
    if (item.status === 'error') return S.upload_pdf_error
    if (item.ingest_status === 'cancelled') return S.upload_pdf_cancelled
    if (item.ingest_status === 'renaming') return S.upload_pdf_renaming
    if (item.ingest_status === 'converting') return S.upload_pdf_converting
    if (item.ingest_status === 'ingesting') return S.upload_pdf_ingesting
    if (item.ingest_status === 'processing') return S.upload_pdf_processing
    if (item.ready && isPdfQualityRunning(item)) return S.upload_pdf_ready_quality_running
    if (item.ready && item.quality_status === 'ready') return S.upload_pdf_ready_quality_done
    if (item.ready && item.quality_status === 'error') return S.upload_pdf_ready_quality_error
    if (item.ready && item.quality_status === 'cancelled') return S.upload_pdf_ready_quality_cancelled
    if (item.ready) return S.upload_pdf_ready
    return S.upload_pdf_pending
  }
  if (item.status === 'unsupported') return S.upload_unsupported
  return S.upload_failed
}

function uploadDetail(item: ChatUploadItem) {
  if (item.kind === 'pdf' && item.quality_status === 'error' && item.quality_error) {
    return item.quality_error
  }
  if (item.kind === 'pdf' && item.status === 'saved' && item.ready === false && item.error) {
    return item.error
  }
  if (item.kind === 'pdf' && item.status === 'duplicate' && item.existing) {
    return item.existing
  }
  return item.error || ''
}

function collectAcceptedFiles(files: File[]) {
  const accepted: File[] = []
  const rejected: string[] = []

  for (const file of files) {
    const ext = fileExt(file.name)
    const mime = String(file.type || '').toLowerCase()
    const isImage = mime.startsWith('image/') || IMAGE_EXTS.has(ext)
    const isPdf = mime === 'application/pdf' || PDF_EXTS.has(ext)

    if (!isImage && !isPdf) {
      rejected.push(`${file.name}: unsupported`)
      continue
    }
    if (isImage && file.size > MAX_IMAGE_BYTES) {
      rejected.push(`${file.name}: image > 8MB`)
      continue
    }
    if (isPdf && file.size > MAX_PDF_BYTES) {
      rejected.push(`${file.name}: pdf > 80MB`)
      continue
    }
    accepted.push(file)
  }

  return { accepted, rejected }
}

export function ChatInput({
  onSend,
  onStop,
  onUpload,
  onRetryUploadItem,
  onCancelUploadItem,
  onRemoveImage,
  onDismissUploadItem,
  onStartGuideFromUpload,
  uploadItems,
  pendingImages,
  uploading,
  generating,
  appendSignal,
  queryScope,
  queryScopeOptions,
  onQueryScopeChange,
  agentMode,
  onAgentModeChange,
}: Props) {
  const S = useT()
  const [text, setText] = useState('')
  const [dragActive, setDragActive] = useState(false)
  const ref = useRef<HTMLTextAreaElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const composingRef = useRef(false)
  const dragDepthRef = useRef(0)

  useEffect(() => {
    if (!generating) ref.current?.focus()
  }, [generating])

  useEffect(() => {
    if (!appendSignal) return
    const incoming = String(appendSignal.text || '')
    if (!incoming.trim()) return
    setText((current) => {
      const cur = String(current || '')
      if (!cur.trim()) return incoming
      const needsBreak = !cur.endsWith('\n')
      return `${cur}${needsBreak ? '\n\n' : ''}${incoming}`
    })
    window.setTimeout(() => {
      ref.current?.focus()
    }, 0)
  }, [appendSignal])

  const uploadSelectedFiles = async (files: File[]) => {
    const { accepted, rejected } = collectAcceptedFiles(files)
    if (rejected.length > 0) {
      message.warning(S.upload_skip_warning)
    }
    if (accepted.length === 0) return
    await onUpload(accepted)
  }

  const submit = () => {
    const t = text.trim()
    if ((!t && pendingImages.length === 0) || generating || uploading) return
    onSend(text)
    setText('')
  }

  const onKey = (e: KeyboardEvent) => {
    if (composingRef.current) return
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const onPickFiles = async (e: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (!files.length) return
    try {
      await uploadSelectedFiles(files)
    } finally {
      if (fileRef.current) {
        fileRef.current.value = ''
      }
    }
  }

  const onPaste = async (e: ClipboardEvent) => {
    const files = Array.from(e.clipboardData?.files || [])
    if (!files.length || generating || uploading) return
    e.preventDefault()
    await uploadSelectedFiles(files)
  }

  const onDragEnter = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    dragDepthRef.current += 1
    setDragActive(true)
  }

  const onDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    if (!dragActive) setDragActive(true)
  }

  const onDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1)
    if (dragDepthRef.current === 0) {
      setDragActive(false)
    }
  }

  const onDrop = async (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    dragDepthRef.current = 0
    setDragActive(false)
    if (generating || uploading) return
    const files = Array.from(e.dataTransfer?.files || [])
    if (!files.length) return
    await uploadSelectedFiles(files)
  }

  const statusItems = uploadItems.filter((item) => item.kind !== 'image')

  return (
    <div className="kb-chat-input-wrap">
      <div className="kb-chat-input-inner">
        {statusItems.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {statusItems.map((item) => {
              const ingestRunning = isPdfIngestRunning(item)
              const qualityRunning = isPdfQualityRunning(item)
              const canRetry = (
                item.kind === 'pdf'
                && (
                  item.status === 'error'
                  || item.ingest_status === 'error'
                  || item.ingest_status === 'cancelled'
                  || item.quality_status === 'error'
                )
              )
              const canStartGuide = (
                item.kind === 'pdf'
                && item.ready === true
                && String(item.ingest_status || '') === 'ready'
                && Boolean(String(item.md_path || '').trim())
              )
              return (
                <Tag
                  key={uploadItemKey(item)}
                  color={uploadTone(item)}
                  className="m-0 inline-flex max-w-full items-center gap-2 rounded-full border px-3 py-1 text-xs"
                >
                  <span className="truncate max-w-[16rem]">{item.name}</span>
                  <span className="opacity-75">{uploadLabel(item, S)}</span>
                  {uploadDetail(item) ? (
                    <span className="truncate max-w-[14rem] opacity-60">{uploadDetail(item)}</span>
                  ) : null}
                  {item.kind === 'pdf' && (ingestRunning || qualityRunning) ? (
                    <button
                      type="button"
                      className="inline-flex h-5 min-w-5 items-center justify-center rounded-full border-0 bg-transparent px-1 opacity-75 transition hover:opacity-100"
                      onClick={() => { void onCancelUploadItem(uploadItemKey(item)) }}
                      title={ingestRunning ? S.upload_cancel_ingest : S.upload_cancel_quality}
                    >
                      <StopOutlined />
                    </button>
                  ) : null}
                  {canRetry ? (
                    <button
                      type="button"
                      className="inline-flex h-5 min-w-5 items-center justify-center rounded-full border-0 bg-transparent px-1 opacity-75 transition hover:opacity-100"
                      onClick={() => { void onRetryUploadItem(uploadItemKey(item)) }}
                      title={item.quality_status === 'error' ? S.upload_retry_quality : S.upload_retry_ingest}
                    >
                      <RedoOutlined />
                    </button>
                  ) : null}
                  {canStartGuide ? (
                    <button
                      type="button"
                      className="inline-flex h-5 min-w-5 items-center justify-center rounded-full border-0 bg-transparent px-1 opacity-75 transition hover:opacity-100"
                      onClick={() => onStartGuideFromUpload(item)}
                      title={S.upload_start_guide}
                    >
                      <BookOutlined />
                    </button>
                  ) : null}
                  <button
                    type="button"
                    className="inline-flex h-4 w-4 items-center justify-center rounded-full border-0 bg-transparent p-0 opacity-60"
                    onClick={() => onDismissUploadItem(uploadItemKey(item))}
                  >
                    <CloseOutlined />
                  </button>
                </Tag>
              )
            })}
          </div>
        ) : null}

        {pendingImages.length > 0 ? (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4">
            {pendingImages.map((item) => {
              const src = String(item.url || '').trim()
              return (
                <div
                  key={imageKey(item)}
                  className="group relative overflow-hidden rounded-lg border border-[var(--border-subtle)] bg-[var(--surface-subtle)]"
                >
                  {src && isImageAttachment(item) ? (
                    <img
                      src={src}
                      alt={item.name}
                      className="block h-28 w-full object-cover"
                      loading="lazy"
                    />
                  ) : (
                    <div className="flex h-28 items-center justify-center px-3 text-center text-xs text-[var(--muted-text)]">
                      {item.name}
                    </div>
                  )}
                  <div className="flex items-center justify-between gap-2 border-t border-[var(--border)]/70 bg-[var(--panel)]/92 px-3 py-2">
                    <div className="min-w-0">
                      <div className="truncate text-xs font-medium">{item.name}</div>
                      <div className="text-[11px] text-[var(--muted-text)]">{S.image_send_with}</div>
                    </div>
                    <button
                      type="button"
                      className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-full border-0 bg-black/6 p-0 text-[var(--muted-text)] transition hover:bg-black/10"
                      onClick={() => onRemoveImage(imageKey(item))}
                    >
                      <CloseOutlined />
                    </button>
                  </div>
                </div>
              )
            })}
          </div>
        ) : null}

        <div
          className={`kb-chat-input-shell relative flex items-end gap-2 transition ${dragActive ? 'is-drag-active' : ''}`}
          onDragEnter={onDragEnter}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={(e) => { void onDrop(e) }}
        >
          {dragActive ? (
            <div className="pointer-events-none absolute inset-x-3 top-3 rounded-2xl border border-[var(--accent)]/30 bg-[var(--panel)]/92 px-4 py-3 text-sm text-[var(--text-main)] shadow-sm">
              {S.upload_drop_hint}
            </div>
          ) : null}

          <div className="kb-chat-composer-main flex flex-1 flex-col gap-2">
            <TextArea
              ref={ref as never}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onPaste={(e) => { void onPaste(e) }}
              onKeyDown={onKey}
              onCompositionStart={() => { composingRef.current = true }}
              onCompositionEnd={() => { composingRef.current = false }}
              placeholder={S.prompt_label}
              autoSize={{ minRows: 1, maxRows: 5 }}
              className="kb-chat-textarea flex-1"
              autoFocus
            />
            <div className="kb-chat-toolbar">
              <div className="kb-chat-toolbar-left">
                <input
                  ref={fileRef}
                  type="file"
                  accept=".pdf,.png,.jpg,.jpeg,.webp,.gif,.bmp"
                  multiple
                  className="hidden"
                  onChange={onPickFiles}
                />
                <Button
                  className="kb-attach-btn"
                  icon={<PaperClipOutlined />}
                  onClick={() => fileRef.current?.click()}
                  loading={uploading}
                  disabled={generating}
                >
                  {S.upload_file_btn}
                </Button>
                <ChatScopeControl value={queryScope} options={queryScopeOptions} onChange={onQueryScopeChange} />
                <Button
                  className={`kb-agent-mode-btn ${agentMode ? 'is-active' : ''}`}
                  icon={<RobotOutlined />}
                  type={agentMode ? 'primary' : 'default'}
                  aria-pressed={agentMode}
                  aria-label={agentMode ? S.chat_agent_mode_on_title : S.chat_agent_mode_off_title}
                  title={agentMode ? S.chat_agent_mode_on_title : S.chat_agent_mode_off_title}
                  disabled={generating}
                  onClick={() => onAgentModeChange(!agentMode)}
                >
                  {agentMode ? S.chat_agent_mode_on : S.chat_agent_mode_off}
                </Button>
              </div>
              <div className="kb-chat-toolbar-right">
                {generating ? (
                  <Button className="kb-stop-btn" icon={<PauseOutlined />} onClick={onStop} danger>
                    {S.stop}
                  </Button>
                ) : (
                  <Button
                    className="kb-send-btn"
                    type="primary"
                    icon={<SendOutlined />}
                    onClick={submit}
                    disabled={(!text.trim() && pendingImages.length === 0) || uploading}
                  >
                    {S.send}
                  </Button>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// Collect status label for a given upload item — used outside the component where S is available
