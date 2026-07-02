import { chatApi, type ChatImageAttachment, type ChatUploadItem } from '../api/chat'
import {
  cachedUploadItems,
  collectUploadStatusJobIds,
  needsAnyUploadStatusPolling,
  replaceUploadItemsFromStatus,
} from './chatStoreUploads'

export interface UploadPollingCacheView {
  uploadItems?: ChatUploadItem[]
  pendingImages?: ChatImageAttachment[]
  cachedAt?: number
}

export interface UploadPollingState<View extends UploadPollingCacheView = UploadPollingCacheView> {
  activeConvId: string | null
  uploadItems: ChatUploadItem[]
  pendingImages: ChatImageAttachment[]
  conversationCacheById: Record<string, View>
}

type UploadPollingSet<State extends UploadPollingState> = (
  patch: Partial<State> | ((state: State) => Partial<State>)
) => void

type UpsertUploadConversationCache<View extends UploadPollingCacheView> = (
  current: Record<string, View>,
  convId: string,
  patch: Partial<View>,
) => Record<string, View>

let uploadPollToken = 0
let uploadPollTimer: number | null = null

export function stopUploadPolling() {
  uploadPollToken += 1
  if (uploadPollTimer !== null && typeof window !== 'undefined') {
    window.clearTimeout(uploadPollTimer)
  }
  uploadPollTimer = null
}

export async function startUploadPolling<
  View extends UploadPollingCacheView,
  State extends UploadPollingState<View>,
>(
  set: UploadPollingSet<State>,
  getState: () => State,
  upsertConversationCache: UpsertUploadConversationCache<View>,
) {
  stopUploadPolling()
  const token = ++uploadPollToken
  let tries = 0
  const maxTries = 240
  const nextDelay = () => {
    if (tries <= 10) return 500
    if (tries <= 40) return 1000
    return 1800
  }

  const tick = async () => {
    if (token !== uploadPollToken) return
    tries += 1
    const state = getState()
    const jobIds = collectUploadStatusJobIds(state)
    if (jobIds.length === 0) {
      uploadPollTimer = null
      return
    }
    try {
      const res = await chatApi.getUploadStatuses(jobIds)
      if (token !== uploadPollToken) return
      const items = Array.isArray(res.items) ? res.items : []
      set((cur) => {
        const activeUpdate = replaceUploadItemsFromStatus(cur.uploadItems, items)
        let cacheChanged = false
        let nextCache = cur.conversationCacheById
        for (const [convId, view] of Object.entries(cur.conversationCacheById || {})) {
          const cachedUpdate = replaceUploadItemsFromStatus(cachedUploadItems(view), items)
          if (!cachedUpdate.changed) continue
          if (!cacheChanged) {
            nextCache = { ...nextCache }
            cacheChanged = true
          }
          nextCache[convId] = {
            ...view,
            uploadItems: cachedUpdate.items,
            cachedAt: Date.now(),
          } as View
        }
        if (activeUpdate.changed && cur.activeConvId) {
          nextCache = upsertConversationCache(nextCache, cur.activeConvId, {
            uploadItems: activeUpdate.items,
            pendingImages: cur.pendingImages,
            cachedAt: Date.now(),
          } as Partial<View>)
          cacheChanged = true
        }
        if (!activeUpdate.changed && !cacheChanged) return {} as Partial<State>
        return {
          ...(activeUpdate.changed ? { uploadItems: activeUpdate.items } : {}),
          ...(cacheChanged ? { conversationCacheById: nextCache } : {}),
        } as Partial<State>
      })
      const nextState = getState()
      if (!needsAnyUploadStatusPolling(nextState) || tries >= maxTries) {
        uploadPollTimer = null
        return
      }
    } catch {
      if (tries >= maxTries) {
        uploadPollTimer = null
        return
      }
    }
    if (typeof window === 'undefined') return
    uploadPollTimer = window.setTimeout(tick, nextDelay())
  }

  void tick()
}
