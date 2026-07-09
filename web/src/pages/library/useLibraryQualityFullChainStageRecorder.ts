import { useCallback } from 'react'
import type {
  LibraryQualityActionSnapshot,
  LibraryQualityFullChainStage,
  LibraryQualityOverviewResponse,
} from '../../api/library'
import { useLibraryStore } from '../../stores/libraryStore'
import {
  normalizeTextValue,
  qualityOverviewStageSnapshot,
} from './libraryPageUtils'
import type { QualityFullChainActionResult } from './useLibraryQualityChainViewModel'
import type { LibraryQualityOperationToken } from './useLibraryQualityOperationGuard'

type QualityFullChainRecordMeta = {
  stageLabel?: string
  action?: string
  targetIds?: string[]
  metrics?: Record<string, string | number | boolean | null | undefined>
  before?: LibraryQualityActionSnapshot
  after?: LibraryQualityActionSnapshot
  verification?: Record<string, unknown>
}

export type LibraryQualityFullChainStageRecordMeta = {
  targetIds?: string[]
  metrics?: Record<string, string | number | boolean | null | undefined>
  afterOverview?: LibraryQualityOverviewResponse | null
  verification?: Record<string, unknown>
}

export type LibraryQualityFullChainStageRecordInput = {
  stage: Pick<LibraryQualityFullChainStage, 'action' | 'key' | 'label'>
  stageKey?: string
  result: Omit<QualityFullChainActionResult, 'updatedAt'>
  meta?: LibraryQualityFullChainStageRecordMeta
  beforeOverview?: LibraryQualityOverviewResponse | null
  currentOverview?: LibraryQualityOverviewResponse | null
  fallbackOverview?: LibraryQualityOverviewResponse | null
}

export function buildLibraryQualityFullChainStageRecord({
  stage,
  stageKey: rawStageKey,
  result,
  meta = {},
  beforeOverview = null,
  currentOverview = null,
  fallbackOverview = null,
}: LibraryQualityFullChainStageRecordInput) {
  const stageKey = normalizeTextValue(rawStageKey || stage.key).toLowerCase()
  const latestOverview = meta.afterOverview || currentOverview || fallbackOverview || null
  const recordMeta: QualityFullChainRecordMeta = {
    stageLabel: stage.label,
    action: stage.action,
    before: qualityOverviewStageSnapshot(beforeOverview, stageKey),
    after: qualityOverviewStageSnapshot(latestOverview, stageKey),
    verification: meta.verification,
    targetIds: meta.targetIds,
    metrics: meta.metrics,
  }
  return {
    stageKey,
    result,
    meta: recordMeta,
  }
}

type UseLibraryQualityFullChainStageRecorderInput = {
  backendQualityOverview: LibraryQualityOverviewResponse | null
  qualityOperationIsCurrent: (token?: LibraryQualityOperationToken | null) => boolean
  recordQualityFullChainResult: (
    stageKey: string,
    result: Omit<QualityFullChainActionResult, 'updatedAt'>,
    meta?: QualityFullChainRecordMeta,
  ) => void
}

export function useLibraryQualityFullChainStageRecorder({
  backendQualityOverview,
  qualityOperationIsCurrent,
  recordQualityFullChainResult,
}: UseLibraryQualityFullChainStageRecorderInput) {
  const recordQualityFullChainStageResult = useCallback((
    stage: LibraryQualityFullChainStage,
    stageKey: string,
    operationToken: LibraryQualityOperationToken,
    result: Omit<QualityFullChainActionResult, 'updatedAt'>,
    meta: LibraryQualityFullChainStageRecordMeta = {},
  ) => {
    if (!qualityOperationIsCurrent(operationToken)) return
    const currentOverview = useLibraryStore.getState().qualityOverview
    const record = buildLibraryQualityFullChainStageRecord({
      stage,
      stageKey,
      result,
      meta,
      beforeOverview: backendQualityOverview,
      currentOverview: currentOverview?.ok ? currentOverview : null,
      fallbackOverview: backendQualityOverview,
    })
    recordQualityFullChainResult(record.stageKey, record.result, record.meta)
  }, [
    backendQualityOverview,
    qualityOperationIsCurrent,
    recordQualityFullChainResult,
  ])

  return {
    recordQualityFullChainStageResult,
  }
}
