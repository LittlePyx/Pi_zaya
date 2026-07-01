import { useCallback, useEffect, useRef, useState, type RefObject } from 'react'
import { libraryApi } from '../../../api/library'
import { qualityDiagnosticsVisible } from '../../../utils/qualityDiagnostics'
import {
  normalizeReaderLocateRequestId,
  readerLocateRepairRunMatchesActiveRequest,
  readerLocateResultMatchesActiveRequest,
  readerSourcePathsMatch,
  type ReaderLocateRequestGuard,
} from './readerLocateGuard'
import type { ReaderLocateResult, ReaderOpenPayload } from './readerTypes'

const READER_LOCATE_AUTO_REPAIR_RETRY_MS = 60_000

interface RegisterReaderLocateRequestArgs {
  feedbackKey?: string
  locateRequestId: number
  sourcePath: string
  payload: ReaderOpenPayload
}

interface UseReaderLocateRepairArgs {
  activeConversationId?: string | null
  readerOpenRef: RefObject<boolean>
  readerPayloadRef: RefObject<ReaderOpenPayload | null>
  openReaderDock: (payload: ReaderOpenPayload) => void
}

export function useReaderLocateRepair({
  activeConversationId,
  readerOpenRef,
  readerPayloadRef,
  openReaderDock,
}: UseReaderLocateRepairArgs) {
  const [readerLocateResults, setReaderLocateResults] = useState<Record<string, ReaderLocateResult>>({})
  const [sourceQualityRefreshToken, setSourceQualityRefreshToken] = useState(0)
  const [qualityDiagnosticsEnabled] = useState(qualityDiagnosticsVisible)
  const readerLocateRequestRef = useRef(1)
  const readerLocateQualitySubmittedRef = useRef<Set<string>>(new Set())
  const readerLocateSourceRepairAtRef = useRef<Record<string, number>>({})
  const readerPayloadByFeedbackKeyRef = useRef<Record<string, ReaderOpenPayload>>({})
  const readerLocateGuardByFeedbackKeyRef = useRef<Record<string, ReaderLocateRequestGuard>>({})
  const readerLocateSourceRepairStreamRef = useRef<AbortController | null>(null)
  const readerLocateSourceRepairRunTokenRef = useRef(0)
  const activeConversationIdRef = useRef(String(activeConversationId || '').trim())

  useEffect(() => {
    activeConversationIdRef.current = String(activeConversationId || '').trim()
  }, [activeConversationId])

  const nextReaderLocateRequestId = useCallback(() => {
    readerLocateRequestRef.current += 1
    return readerLocateRequestRef.current
  }, [])

  const registerReaderLocateRequest = useCallback(({
    feedbackKey,
    locateRequestId,
    sourcePath,
    payload,
  }: RegisterReaderLocateRequestArgs) => {
    const key = String(feedbackKey || payload.locateFeedbackKey || '').trim()
    if (!key) return
    readerPayloadByFeedbackKeyRef.current[key] = payload
    readerLocateGuardByFeedbackKeyRef.current[key] = {
      locateRequestId,
      sourcePath: String(sourcePath || payload.sourcePath || '').trim(),
      conversationId: String(activeConversationId || '').trim(),
    }
  }, [activeConversationId])

  const resetReaderLocateRepair = useCallback(() => {
    readerPayloadByFeedbackKeyRef.current = {}
    readerLocateGuardByFeedbackKeyRef.current = {}
    readerLocateSourceRepairRunTokenRef.current += 1
    readerLocateSourceRepairStreamRef.current?.abort()
    readerLocateSourceRepairStreamRef.current = null
  }, [])

  useEffect(() => () => {
    resetReaderLocateRepair()
  }, [resetReaderLocateRepair])

  const refreshShelfSourceQuality = useCallback(() => {
    setSourceQualityRefreshToken((value) => value + 1)
  }, [])

  const retryReaderLocateAfterRepair = useCallback((feedbackKey: string, sourcePath: string) => {
    const key = String(feedbackKey || '').trim()
    const path = String(sourcePath || '').trim()
    if (!key || !readerOpenRef.current) return
    const currentPayload = readerPayloadRef.current
    if (!currentPayload) return
    if (String(currentPayload.locateFeedbackKey || '').trim() !== key) return
    if (path && !readerSourcePathsMatch(currentPayload.sourcePath, path)) return
    const locateRequestId = nextReaderLocateRequestId()
    const nextPayload: ReaderOpenPayload = {
      ...currentPayload,
      locateRequestId,
    }
    readerPayloadByFeedbackKeyRef.current[key] = nextPayload
    readerLocateGuardByFeedbackKeyRef.current[key] = {
      locateRequestId,
      sourcePath: String(nextPayload.sourcePath || '').trim(),
      conversationId: String(activeConversationIdRef.current || '').trim(),
    }
    openReaderDock(nextPayload)
  }, [nextReaderLocateRequestId, openReaderDock, readerOpenRef, readerPayloadRef])

  const completeReaderLocateSourceRepair = useCallback(async (
    runId: string,
    options: {
      needsReindex: boolean
      shouldRetryLocate: boolean
      feedbackKey: string
      sourcePath: string
      isCurrentRepair?: () => boolean
    },
  ) => {
    if (options.isCurrentRepair && !options.isCurrentRepair()) return
    let waiting = false
    if (runId && options.needsReindex) {
      try {
        const advanced = await libraryApi.advanceQualityRepairRun(runId)
        waiting = Boolean(advanced.waiting)
      } catch {
        waiting = false
      }
    }
    if (options.isCurrentRepair && !options.isCurrentRepair()) return
    refreshShelfSourceQuality()
    if (!waiting && options.shouldRetryLocate) {
      retryReaderLocateAfterRepair(options.feedbackKey, options.sourcePath)
    }
  }, [refreshShelfSourceQuality, retryReaderLocateAfterRepair])

  const handleReaderLocateResult = useCallback((result: ReaderLocateResult) => {
    const feedbackKey = String(result.locateFeedbackKey || '').trim()
    if (!feedbackKey) return
    const sourcePath = String(result.sourcePath || '').trim()
    const sourceName = String(result.sourceName || '').trim()
    const locateRequestId = normalizeReaderLocateRequestId(result.locateRequestId)
    const guard = readerLocateGuardByFeedbackKeyRef.current[feedbackKey]
    const currentPayload = readerPayloadRef.current
    const currentConversationId = String(activeConversationIdRef.current || '').trim()
    if (!readerLocateResultMatchesActiveRequest({
      result: { ...result, locateRequestId },
      guard,
      currentPayload,
      currentConversationId,
      readerOpen: readerOpenRef.current,
    })) {
      return
    }
    const submitKey = [
      feedbackKey,
      locateRequestId,
      result.status,
      result.precision,
      result.hint,
      result.reason,
    ].join('|')
    if (qualityDiagnosticsEnabled && !readerLocateQualitySubmittedRef.current.has(submitKey)) {
      readerLocateQualitySubmittedRef.current.add(submitKey)
      libraryApi.recordReaderLocateQuality({
        source_path: sourcePath,
        source_name: sourceName,
        locate_feedback_key: feedbackKey,
        locate_request_id: locateRequestId,
        status: result.status,
        precision: result.precision,
        ok: result.ok,
        repairable: result.repairable,
        strict_locate: result.strictLocate,
        hint: result.hint,
        reason: result.reason,
        active_alt_index: result.activeAltIndex,
        block_id: result.blockId,
        anchor_id: result.anchorId,
        anchor_kind: result.anchorKind,
        heading_path: result.headingPath,
      }).catch(() => {})
    }
    const locateStatus = String(result.status || '').trim().toLowerCase()
    const needsSourceRepair = Boolean(
      sourcePath
      && (
        result.repairable
        || locateStatus === 'failed'
        || (result.strictLocate && !['exact', 'block'].includes(locateStatus))
      ),
    )
    if (qualityDiagnosticsEnabled && needsSourceRepair) {
      const repairKey = sourcePath || sourceName
      const now = Date.now()
      const last = Number(readerLocateSourceRepairAtRef.current[repairKey] || 0)
      if (repairKey && now - last >= READER_LOCATE_AUTO_REPAIR_RETRY_MS) {
        readerLocateSourceRepairAtRef.current[repairKey] = now
        const repairToken = readerLocateSourceRepairRunTokenRef.current + 1
        readerLocateSourceRepairRunTokenRef.current = repairToken
        const repairResult: ReaderLocateResult = { ...result, locateRequestId }
        const isCurrentSourceRepair = () => (
          readerLocateRepairRunMatchesActiveRequest({
            expectedRunToken: repairToken,
            currentRunToken: readerLocateSourceRepairRunTokenRef.current,
            result: repairResult,
            guard: readerLocateGuardByFeedbackKeyRef.current[feedbackKey],
            currentPayload: readerPayloadRef.current,
            currentConversationId: activeConversationIdRef.current,
            readerOpen: readerOpenRef.current,
          })
        )
        libraryApi.repairQuality({
          sources: [{ source_path: sourcePath, source_name: sourceName }],
          speed_mode: 'balanced',
          replace: true,
          md_autofix: true,
        })
          .then((res) => {
            if (!isCurrentSourceRepair()) return undefined
            const runId = String(res.repair_run_id || res.repair_run?.run_id || '').trim()
            const queued = Number(res.enqueued || 0)
            const needsReindex = Boolean(res.needs_reindex || res.impact?.needs_reindex)
            const repaired = Number(res.repaired || res.impact?.repaired || 0)
            const readerLocateReindex = Number(res.impact?.reader_locate_reindex || 0)
            const shouldRetryLocate = Boolean(
              needsReindex
              || repaired > 0
              || readerLocateReindex > 0
              || (res.items || []).some((item) => Boolean(item.reader_locate_reindex_required)),
            )
            if (!runId) {
              if (!isCurrentSourceRepair()) return undefined
              refreshShelfSourceQuality()
              if (shouldRetryLocate) retryReaderLocateAfterRepair(feedbackKey, sourcePath)
              return undefined
            }
            if (queued > 0) {
              readerLocateSourceRepairStreamRef.current?.abort()
              let streamCtrl: AbortController | null = null
              const clearStreamIfCurrent = () => {
                if (!isCurrentSourceRepair() || readerLocateSourceRepairStreamRef.current !== streamCtrl) return false
                readerLocateSourceRepairStreamRef.current = null
                return true
              }
              streamCtrl = libraryApi.streamConvertStatus(
                () => {},
                () => {
                  if (!clearStreamIfCurrent()) return
                  void completeReaderLocateSourceRepair(runId, {
                    needsReindex,
                    shouldRetryLocate,
                    feedbackKey,
                    sourcePath,
                    isCurrentRepair: isCurrentSourceRepair,
                  })
                },
                () => {
                  if (!clearStreamIfCurrent()) return
                  refreshShelfSourceQuality()
                },
              )
              readerLocateSourceRepairStreamRef.current = streamCtrl
              return undefined
            }
            return completeReaderLocateSourceRepair(runId, {
              needsReindex,
              shouldRetryLocate,
              feedbackKey,
              sourcePath,
              isCurrentRepair: isCurrentSourceRepair,
            })
          })
          .catch(() => {
            if (isCurrentSourceRepair()) delete readerLocateSourceRepairAtRef.current[repairKey]
          })
      }
    }
    setReaderLocateResults((current) => {
      const prev = current[feedbackKey]
      if (
        prev
        && prev.locateRequestId === locateRequestId
        && prev.status === result.status
        && prev.precision === result.precision
        && prev.hint === result.hint
      ) {
        return current
      }
      return { ...current, [feedbackKey]: { ...result, locateRequestId } }
    })
  }, [
    completeReaderLocateSourceRepair,
    qualityDiagnosticsEnabled,
    readerOpenRef,
    readerPayloadRef,
    refreshShelfSourceQuality,
    retryReaderLocateAfterRepair,
  ])

  return {
    readerLocateResults,
    sourceQualityRefreshToken,
    nextReaderLocateRequestId,
    registerReaderLocateRequest,
    resetReaderLocateRepair,
    handleReaderLocateResult,
  }
}
