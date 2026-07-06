import { useCallback, type Dispatch, type SetStateAction } from 'react'
import type {
  LibraryQualityActionDelta,
  LibraryQualityActionSnapshot,
} from '../../api/library'
import { libraryApi } from '../../api/library'
import {
  normalizeTextValue,
  qualityBuildActionDelta,
  qualityVerificationText,
} from './libraryPageUtils'
import type { QualityFullChainActionResult } from './useLibraryQualityChainViewModel'

type QualityFullChainRecordMeta = {
  stageLabel?: string
  action?: string
  targetIds?: string[]
  metrics?: Record<string, string | number | boolean | null | undefined>
  before?: LibraryQualityActionSnapshot
  after?: LibraryQualityActionSnapshot
  verification?: Record<string, unknown>
}

type UseLibraryQualityActionRecorderParams = {
  setQualityFullChainResults: Dispatch<SetStateAction<Record<string, QualityFullChainActionResult>>>
}

export function useLibraryQualityActionRecorder({
  setQualityFullChainResults,
}: UseLibraryQualityActionRecorderParams) {
  const recordQualityFullChainResult = useCallback((
    stageKey: string,
    result: Omit<QualityFullChainActionResult, 'updatedAt'>,
    meta: QualityFullChainRecordMeta = {},
  ) => {
    const key = normalizeTextValue(stageKey).toLowerCase()
    if (!key) return
    const nowMs = Date.now()
    const hasSnapshots = Boolean(meta.before || meta.after)
    const verificationOk = meta.verification?.quality_ok === true
    const rawDelta: LibraryQualityActionDelta = hasSnapshots ? qualityBuildActionDelta(meta.before, meta.after) : {}
    const delta: LibraryQualityActionDelta = hasSnapshots && rawDelta.improved === false && verificationOk
      ? {
        ...rawDelta,
        improved: true,
        worsened: false,
        summary: 'Improved: QA rerun passed; overview unchanged',
      }
      : rawDelta
    const improved = typeof delta.improved === 'boolean' ? delta.improved : null
    const verificationText = qualityVerificationText(meta.verification)
    const resultStatus = result.status === 'success' && improved === false ? 'warning' : result.status
    setQualityFullChainResults((cur) => ({
      ...cur,
      [key]: {
        ...result,
        status: resultStatus,
        deltaText: hasSnapshots ? delta.summary : '',
        verificationText,
        improved,
        updatedAt: nowMs,
      },
    }))
    void libraryApi.recordQualityAction({
      stage_key: key,
      stage_label: normalizeTextValue(meta.stageLabel || key),
      action: normalizeTextValue(meta.action),
      status: resultStatus,
      summary: result.summary,
      detail: result.detail,
      target_ids: meta.targetIds || [],
      metrics: meta.metrics || {},
      before: meta.before,
      after: meta.after,
      delta: hasSnapshots ? delta : {},
      improved,
      verification: meta.verification || {},
      created_at: Math.floor(nowMs / 1000),
    }).catch(() => {
      // Persisting the audit trail should never block the repair flow.
    })
  }, [setQualityFullChainResults])

  return {
    recordQualityFullChainResult,
  }
}
