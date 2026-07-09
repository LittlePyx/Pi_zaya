import type {
  LibraryResearchQaRerunResponse,
  LibraryQualityOverviewResponse,
} from '../../api/library'
import { normalizeTextValue, qualityVerificationFromRerun } from './libraryPageUtils'
import type { QualityFullChainActionResult } from './useLibraryQualityChainViewModel'
import type { LibraryQualityFullChainStageRecordMeta } from './useLibraryQualityFullChainStageRecorder'

export type LibraryQualityFullChainStageKind =
  | 'conversion'
  | 'retrieval'
  | 'metadata'
  | 'repair_loop'
  | 'research_qa'
  | 'citation_cards'
  | 'unknown'

export type LibraryQualityFullChainStageRecordModel = {
  result: Omit<QualityFullChainActionResult, 'updatedAt'>
  meta?: LibraryQualityFullChainStageRecordMeta
}

export type LibraryQualityStageMetadataResult = {
  targetCount: number
  targetIds: string[]
  ready: number
  exportReady: number
  changed: number
  retryable: number
  unresolved: number
  verification: Record<string, unknown>
  running: boolean
}

export function getLibraryQualityFullChainStageKind(
  stageKey: string,
  action: string,
): LibraryQualityFullChainStageKind {
  const key = normalizeTextValue(stageKey).toLowerCase()
  const normalizedAction = normalizeTextValue(action).toLowerCase()
  if (key === 'conversion' || normalizedAction === 'repair_conversion') return 'conversion'
  if (key === 'retrieval' || normalizedAction === 'rebuild_index') return 'retrieval'
  if (
    key === 'citations'
    || key === 'shelf'
    || normalizedAction === 'repair_citation_cards'
    || normalizedAction === 'repair_shelf_metadata'
  ) {
    return 'metadata'
  }
  if (key === 'repair_loop' || normalizedAction === 'rerun_failed_cases') return 'repair_loop'
  if (key === 'research_qa' || normalizedAction === 'fix_failed_qa_cases') return 'research_qa'
  if (key === 'citation_cards') return 'citation_cards'
  return 'unknown'
}

export function qualityFullChainRerunPassed(rerun: LibraryResearchQaRerunResponse | null | undefined) {
  return Boolean(rerun?.quality_ok || rerun?.status === 'passed')
}

export function buildLibraryQualityRetrievalStageRecord({
  ok,
  caseId,
  rerun,
  afterOverview,
}: {
  ok: boolean
  caseId?: string
  rerun: LibraryResearchQaRerunResponse | null | undefined
  afterOverview?: LibraryQualityOverviewResponse | null
}): LibraryQualityFullChainStageRecordModel {
  const rerunPassed = qualityFullChainRerunPassed(rerun)
  const failureCount = Number(rerun?.failures?.length || 0)
  return {
    result: {
      status: ok ? (rerun && !rerunPassed ? 'warning' : 'success') : 'error',
      summary: ok
        ? (rerun ? (rerunPassed ? `Reindex verified: ${caseId}` : `Reindex done; QA still failing: ${caseId}`) : 'Rebuilt retrieval index')
        : 'Retrieval index rebuild failed',
      detail: failureCount > 0 ? `${failureCount} failures remain` : (caseId ? `Regression check: ${caseId}` : undefined),
    },
    meta: {
      targetIds: caseId ? [caseId] : [],
      metrics: {
        qa_rerun_quality_ok: rerun ? rerunPassed : false,
        failure_count: failureCount,
      },
      afterOverview,
      verification: qualityVerificationFromRerun(rerun),
    },
  }
}

export function combineLibraryQualityMetadataVerification(
  shelfVerification: Record<string, unknown> | null | undefined,
  qaVerification: Record<string, unknown> | null | undefined,
) {
  const shelf = shelfVerification && Object.keys(shelfVerification).length ? shelfVerification : {}
  const qa = qaVerification && Object.keys(qaVerification).length ? qaVerification : {}
  if (Object.keys(shelf).length && Object.keys(qa).length) {
    return {
      type: 'combined_shelf_metadata_verification',
      quality_ok: Boolean(shelf.quality_ok) && Boolean(qa.quality_ok),
      shelf_metadata: shelf,
      research_qa: qa,
    }
  }
  return Object.keys(shelf).length ? shelf : qa
}

export function buildLibraryQualityMetadataStageRecord({
  result,
  caseId,
  rerun,
  afterOverview,
}: {
  result: LibraryQualityStageMetadataResult
  caseId?: string
  rerun: LibraryResearchQaRerunResponse | null | undefined
  afterOverview?: LibraryQualityOverviewResponse | null
}): LibraryQualityFullChainStageRecordModel {
  const rerunPassed = qualityFullChainRerunPassed(rerun)
  const failureCount = Number(rerun?.failures?.length || 0)
  const resultRunning = Boolean(result.running)
  const qaVerification = qualityVerificationFromRerun(rerun)
  const verification = combineLibraryQualityMetadataVerification(result.verification, qaVerification)
  return {
    result: {
      status: resultRunning
        ? 'success'
        : (result.retryable > 0 || result.unresolved > 0 || (rerun && !rerunPassed)
          ? 'warning'
          : (result.changed > 0 || result.ready > 0 ? 'success' : 'info')),
      summary: resultRunning
        ? 'Metadata backfill started'
        : rerun
        ? (rerunPassed ? `Metadata repair verified: ${caseId}` : `Metadata checked; QA still failing: ${caseId}`)
        : (result.changed > 0
          ? (result.unresolved > 0 ? `Metadata repaired ${result.changed}; ${result.unresolved} still missing` : `Metadata repaired: ${result.changed}`)
          : (result.exportReady > 0 ? `Metadata export ready: ${result.exportReady}` : (result.ready > 0 ? `Metadata already ready: ${result.ready}` : 'Opened citation quality report'))),
      detail: resultRunning
        ? (result.targetCount > 0 ? `${result.targetCount} metadata targets queued` : 'Scanning the reference index')
        : (failureCount > 0 ? `${failureCount} failures remain` : (result.targetCount > 0 ? `${result.targetCount} failed cases checked` : undefined)),
    },
    meta: {
      targetIds: result.targetIds,
      metrics: {
        changed: result.changed,
        ready: result.ready,
        export_ready: result.exportReady,
        retryable: result.retryable,
        unresolved: result.unresolved,
        target_count: result.targetCount,
        async_running: resultRunning,
        qa_rerun_quality_ok: rerun ? rerunPassed : false,
      },
      afterOverview,
      verification,
    },
  }
}

export function buildLibraryQualityRepairLoopStageRecord({
  caseId,
  rerun,
  afterOverview,
}: {
  caseId: string
  rerun: LibraryResearchQaRerunResponse | null | undefined
  afterOverview?: LibraryQualityOverviewResponse | null
}): LibraryQualityFullChainStageRecordModel {
  const rerunPassed = qualityFullChainRerunPassed(rerun)
  const failureCount = Number(rerun?.failures?.length || 0)
  return {
    result: {
      status: rerunPassed ? 'success' : 'warning',
      summary: rerunPassed
        ? `Rerun passed: ${caseId}`
        : `Rerun still failing: ${caseId}`,
      detail: failureCount > 0 ? `${failureCount} failures remain` : undefined,
    },
    meta: {
      targetIds: [caseId],
      metrics: {
        quality_ok: Boolean(rerun?.quality_ok),
        failure_count: failureCount,
      },
      afterOverview,
      verification: qualityVerificationFromRerun(rerun),
    },
  }
}
