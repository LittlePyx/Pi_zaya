import type { ChatImageAttachment, ChatUploadItem } from '../api/chat'

interface UploadConversationCacheLike {
  uploadItems?: ChatUploadItem[]
  pendingImages?: ChatImageAttachment[]
}

interface UploadPollingStateLike {
  uploadItems: ChatUploadItem[]
  conversationCacheById: Record<string, UploadConversationCacheLike | undefined>
}

export function uploadItemKey(item: ChatUploadItem) {
  if (item.kind === 'pdf' && item.ingest_job_id) {
    return `pdf-job:${item.ingest_job_id}`
  }
  return [item.kind, item.sha1 || '', item.path || '', item.name].join(':')
}

export function attachmentKey(item: ChatImageAttachment) {
  return item.sha1 || item.path
}

export function mergeUploadItems(current: ChatUploadItem[], incoming: ChatUploadItem[]) {
  const next = [...current]
  const positions = new Map(next.map((item, index) => [uploadItemKey(item), index]))
  for (const item of incoming) {
    const key = uploadItemKey(item)
    const index = positions.get(key)
    if (index === undefined) {
      positions.set(key, next.length)
      next.push(item)
    } else {
      next[index] = item
    }
  }
  return next
}

export function replaceUploadItemsFromStatus(current: ChatUploadItem[], incoming: ChatUploadItem[]) {
  const statuses = new Map(incoming.map((item) => [uploadItemKey(item), item]))
  let changed = false
  const next = current.map((item) => {
    const replacement = statuses.get(uploadItemKey(item))
    if (!replacement) return item
    changed = true
    return replacement
  })
  return { items: changed ? next : current, changed }
}

export function isPdfUploadJobRunning(item: ChatUploadItem) {
  if (item.kind !== 'pdf') return false
  const ingestRunning = ['processing', 'renaming', 'converting', 'ingesting'].includes(String(item.ingest_status || ''))
  const qualityRunning = ['pending', 'running'].includes(String(item.quality_status || ''))
  return ingestRunning || qualityRunning
}

export function cachedUploadItems(view: UploadConversationCacheLike | undefined): ChatUploadItem[] {
  return Array.isArray(view?.uploadItems) ? view.uploadItems : []
}

export function cachedPendingImages(view: UploadConversationCacheLike | undefined): ChatImageAttachment[] {
  return Array.isArray(view?.pendingImages) ? view.pendingImages : []
}

export function collectUploadStatusJobIds(state: UploadPollingStateLike) {
  const seen = new Set<string>()
  const collect = (items: ChatUploadItem[]) => {
    for (const item of items || []) {
      if (!isPdfUploadJobRunning(item) || !item.ingest_job_id) continue
      const jobId = String(item.ingest_job_id || '').trim()
      if (jobId) seen.add(jobId)
    }
  }
  collect(state.uploadItems)
  for (const view of Object.values(state.conversationCacheById || {})) {
    collect(cachedUploadItems(view))
  }
  return [...seen]
}

export function needsAnyUploadStatusPolling(state: UploadPollingStateLike) {
  return collectUploadStatusJobIds(state).length > 0
}

export function mergeImageAttachments(current: ChatImageAttachment[], incoming: ChatImageAttachment[]) {
  const next = [...current]
  const seen = new Set(next.map(attachmentKey))
  for (const item of incoming) {
    const key = attachmentKey(item)
    if (!key || seen.has(key)) continue
    seen.add(key)
    next.push(item)
  }
  return next
}
