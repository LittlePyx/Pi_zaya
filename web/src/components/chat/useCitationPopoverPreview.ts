import { useCallback, useEffect, useRef } from 'react'
import type { MutableRefObject } from 'react'

import { referencesApi } from '../../api/references'
import type { CiteDetail } from './citationState'

export type CitationPopoverPreviewCallback = () => void

export type CitationPopoverPolishFetcher = (
  detail: CiteDetail,
  waitSeconds: number,
) => Promise<Record<string, unknown>>

export interface CitationPopoverPolishRequest {
  activeRequestKeyRef: MutableRefObject<string>
  attempt?: number
  detail: CiteDetail
  fetcher?: CitationPopoverPolishFetcher
  itemKey: string
  maxAttempts?: number
  onMeta: (itemKey: string, metas: Array<Record<string, unknown>>) => void
  retryDelayMs?: (attempt: number) => number
}

export interface CitationPopoverPreviewController {
  clearPolishRetry: () => void
  clearTimers: () => void
  keepPreviewOpen: () => void
  requestPolish: (request: CitationPopoverPolishRequest) => void
  schedulePreviewClose: (onClose: CitationPopoverPreviewCallback, delayMs?: number) => void
  schedulePreviewOpen: (onOpen: CitationPopoverPreviewCallback, delayMs?: number) => void
}

const DEFAULT_PREVIEW_OPEN_DELAY_MS = 180
const DEFAULT_PREVIEW_CLOSE_DELAY_MS = 260
const DEFAULT_POLISH_MAX_ATTEMPTS = 8

export function citationPopoverPolishWaitSeconds(attempt: number): number {
  return attempt <= 1 ? 4 : 2
}

export function citationPopoverPolishRetryDelayMs(attempt: number): number {
  return 900 + attempt * 700
}

function defaultPolishFetcher(detail: CiteDetail, waitSeconds: number): Promise<Record<string, unknown>> {
  return referencesApi.citationCardPolishCached(detail as unknown as Record<string, unknown>, waitSeconds)
}

function isPendingPolishMeta(meta: Record<string, unknown>): boolean {
  return String(meta?.citation_card_polish_status || meta?.citationCardPolishStatus || '').trim().toLowerCase() === 'pending'
}

export function useCitationPopoverPreview(): CitationPopoverPreviewController {
  const openTimerRef = useRef<number | null>(null)
  const closeTimerRef = useRef<number | null>(null)
  const polishRetryTimerRef = useRef<number | null>(null)

  const clearOpenTimer = useCallback(() => {
    if (openTimerRef.current === null) return
    window.clearTimeout(openTimerRef.current)
    openTimerRef.current = null
  }, [])

  const clearCloseTimer = useCallback(() => {
    if (closeTimerRef.current === null) return
    window.clearTimeout(closeTimerRef.current)
    closeTimerRef.current = null
  }, [])

  const clearPolishRetry = useCallback(() => {
    if (polishRetryTimerRef.current === null) return
    window.clearTimeout(polishRetryTimerRef.current)
    polishRetryTimerRef.current = null
  }, [])

  const clearTimers = useCallback(() => {
    clearOpenTimer()
    clearCloseTimer()
    clearPolishRetry()
  }, [clearCloseTimer, clearOpenTimer, clearPolishRetry])

  const schedulePreviewOpen = useCallback((
    onOpen: CitationPopoverPreviewCallback,
    delayMs = DEFAULT_PREVIEW_OPEN_DELAY_MS,
  ) => {
    clearOpenTimer()
    clearCloseTimer()
    openTimerRef.current = window.setTimeout(() => {
      openTimerRef.current = null
      onOpen()
    }, delayMs)
  }, [clearCloseTimer, clearOpenTimer])

  const schedulePreviewClose = useCallback((
    onClose: CitationPopoverPreviewCallback,
    delayMs = DEFAULT_PREVIEW_CLOSE_DELAY_MS,
  ) => {
    clearOpenTimer()
    clearCloseTimer()
    closeTimerRef.current = window.setTimeout(() => {
      closeTimerRef.current = null
      clearPolishRetry()
      onClose()
    }, delayMs)
  }, [clearCloseTimer, clearOpenTimer, clearPolishRetry])

  const requestPolish = useCallback((request: CitationPopoverPolishRequest) => {
    const runPolishRequest = (nextRequest: CitationPopoverPolishRequest) => {
      const attempt = nextRequest.attempt ?? 0
      const fetcher = nextRequest.fetcher ?? defaultPolishFetcher
      const maxAttempts = nextRequest.maxAttempts ?? DEFAULT_POLISH_MAX_ATTEMPTS
      const retryDelayMs = nextRequest.retryDelayMs ?? citationPopoverPolishRetryDelayMs

      fetcher(nextRequest.detail, citationPopoverPolishWaitSeconds(attempt))
        .then((meta) => {
          if (nextRequest.activeRequestKeyRef.current !== nextRequest.itemKey) return
          if (isPendingPolishMeta(meta)) {
            if (attempt >= maxAttempts) return
            clearPolishRetry()
            polishRetryTimerRef.current = window.setTimeout(() => {
              polishRetryTimerRef.current = null
              runPolishRequest({
                ...nextRequest,
                attempt: attempt + 1,
              })
            }, retryDelayMs(attempt))
            return
          }
          nextRequest.onMeta(nextRequest.itemKey, [meta])
        })
        .catch(() => {
          // The card already has deterministic fallback text; LLM polish is a best-effort enhancement.
        })
    }

    runPolishRequest(request)
  }, [clearPolishRetry])

  useEffect(() => clearTimers, [clearTimers])

  return {
    clearPolishRetry,
    clearTimers,
    keepPreviewOpen: clearCloseTimer,
    requestPolish,
    schedulePreviewClose,
    schedulePreviewOpen,
  }
}
