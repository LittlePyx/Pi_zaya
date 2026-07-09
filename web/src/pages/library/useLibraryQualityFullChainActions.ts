import { useCallback } from 'react'
import { message } from 'antd'
import type {
  LibraryQualityFailureCase,
  LibraryQualityFullChainStage,
  LibraryQualityRepairAction,
  LibraryQualityRepairImpact,
  LibraryResearchQaRerunResponse,
} from '../../api/library'
import { useLibraryStore } from '../../stores/libraryStore'
import { normalizeTextValue } from './libraryPageUtils'
import {
  buildLibraryQualityConversionReviewStageRecord,
  buildLibraryQualityConversionStageRecord,
  buildLibraryQualityMetadataStageRecord,
  buildLibraryQualityRepairLoopStageRecord,
  buildLibraryQualityResearchQaOpenStageRecord,
  buildLibraryQualityResearchQaRepairPlanStageRecord,
  buildLibraryQualityResearchQaRerunStageRecord,
  buildLibraryQualityRetrievalStageRecord,
  getLibraryQualityFullChainStageKind,
  type LibraryQualityConversionRepairStageResult,
  type LibraryQualityResearchQaRepairPlanStageResult,
  type LibraryQualityStageMetadataResult,
} from './libraryQualityFullChainStageResults'
import type { QualityFullChainActionResult } from './useLibraryQualityChainViewModel'
import type {
  LibraryQualityFullChainStageRecordMeta,
} from './useLibraryQualityFullChainStageRecorder'
import type { LibraryQualityOperationToken } from './useLibraryQualityOperationGuard'
import type { LibraryQualityRepairRunOptions } from './useLibraryQualityRepairActions'

type QualityArtifactDomain = 'research_qa' | 'citation_cards'
type QualityArtifactTarget = 'report' | 'folder' | 'raw' | 'summary' | 'runbook'

type UseLibraryQualityFullChainActionsInput = {
  scope: string
  qualityRepairRecommendedNames: string[]
  beginQualityOperation: (key: string) => LibraryQualityOperationToken
  clearQualityOperation: (token?: LibraryQualityOperationToken | null) => void
  qualityOperationIsActive: (token?: LibraryQualityOperationToken | null) => boolean
  qualityOperationIsCurrent: (token?: LibraryQualityOperationToken | null) => boolean
  setQualityFullChainActionKey: (key: string) => void
  setQualityRepairImpact: (impact: LibraryQualityRepairImpact | null) => void
  firstQualityCaseForStage: (stageKey: string) => LibraryQualityFailureCase | null
  handleFocusQualityReview: () => void
  handleReindex: (operationToken?: LibraryQualityOperationToken) => Promise<boolean>
  handleRepairRecommendedQuality: (
    opts?: LibraryQualityRepairRunOptions,
  ) => Promise<LibraryQualityConversionRepairStageResult | null | undefined>
  waitForLibraryConversionDone: () => Promise<boolean>
  runQualityFailureCaseRerun: (
    item: LibraryQualityFailureCase,
    operationToken?: LibraryQualityOperationToken,
  ) => Promise<LibraryResearchQaRerunResponse | null>
  repairQualityStageShelfMetadata: (
    stageKey: string,
    operationToken?: LibraryQualityOperationToken,
  ) => Promise<LibraryQualityStageMetadataResult>
  applyQualityFailureRepairPlan: (
    item: LibraryQualityFailureCase,
    action: LibraryQualityRepairAction,
    operationToken?: LibraryQualityOperationToken,
  ) => Promise<LibraryQualityResearchQaRepairPlanStageResult>
  openQualityArtifact: (domain: QualityArtifactDomain, target: QualityArtifactTarget) => Promise<void>
  loadFiles: (scope?: string) => Promise<unknown> | unknown
  loadQualityOverview: (scope?: string) => Promise<unknown> | unknown
  recordQualityFullChainStageResult: (
    stage: LibraryQualityFullChainStage,
    stageKey: string,
    operationToken: LibraryQualityOperationToken,
    result: Omit<QualityFullChainActionResult, 'updatedAt'>,
    meta?: LibraryQualityFullChainStageRecordMeta,
  ) => void
}

export function useLibraryQualityFullChainActions({
  scope,
  qualityRepairRecommendedNames,
  beginQualityOperation,
  clearQualityOperation,
  qualityOperationIsActive,
  qualityOperationIsCurrent,
  setQualityFullChainActionKey,
  setQualityRepairImpact,
  firstQualityCaseForStage,
  handleFocusQualityReview,
  handleReindex,
  handleRepairRecommendedQuality,
  waitForLibraryConversionDone,
  runQualityFailureCaseRerun,
  repairQualityStageShelfMetadata,
  applyQualityFailureRepairPlan,
  openQualityArtifact,
  loadFiles,
  loadQualityOverview,
  recordQualityFullChainStageResult,
}: UseLibraryQualityFullChainActionsInput) {
  const refreshQualityOverviewSnapshot = useCallback(async () => {
    await loadQualityOverview('all')
    const overview = useLibraryStore.getState().qualityOverview
    return overview?.ok ? overview : null
  }, [loadQualityOverview])

  const handleQualityFullChainStage = useCallback(async (stage: LibraryQualityFullChainStage) => {
    const stageKey = normalizeTextValue(stage.key).toLowerCase()
    const action = normalizeTextValue(stage.action).toLowerCase()
    const stageKind = getLibraryQualityFullChainStageKind(stageKey, action)
    const operationToken = beginQualityOperation(`full-chain:${stageKey}:${action}`)
    const caseTarget = firstQualityCaseForStage(stageKey)
    const recordStageResult = (
      result: Omit<QualityFullChainActionResult, 'updatedAt'>,
      meta: LibraryQualityFullChainStageRecordMeta = {},
    ) => {
      recordQualityFullChainStageResult(stage, stageKey, operationToken, result, meta)
    }
    setQualityFullChainActionKey(stageKey)
    try {
      if (stageKind === 'conversion') {
        if (qualityRepairRecommendedNames.length > 0) {
          const repair = await handleRepairRecommendedQuality({
            autoReindexImmediate: false,
            autoReindexQueued: false,
            operationToken,
          })
          if (!qualityOperationIsCurrent(operationToken)) return
          const completed = Number(repair?.queued || 0) > 0 ? await waitForLibraryConversionDone() : true
          if (!qualityOperationIsCurrent(operationToken)) return
          const needsReindex = Boolean(repair?.needsReindex || repair?.impact?.needs_reindex)
          const reindexed = completed && needsReindex ? await handleReindex(operationToken) : false
          if (!qualityOperationIsCurrent(operationToken)) return
          if (repair?.impact && needsReindex) {
            setQualityRepairImpact({ ...repair.impact, reindexed })
          }
          if (reindexed) await loadFiles(scope)
          if (!qualityOperationIsCurrent(operationToken)) return
          const rerun = completed && (!needsReindex || reindexed) && caseTarget ? await runQualityFailureCaseRerun(caseTarget, operationToken) : null
          if (!qualityOperationIsCurrent(operationToken)) return
          const afterOverview = await refreshQualityOverviewSnapshot()
          if (!qualityOperationIsCurrent(operationToken)) return
          const record = buildLibraryQualityConversionStageRecord({
            repair,
            completed,
            needsReindex,
            reindexed,
            targetIds: qualityRepairRecommendedNames.slice(0, 12),
            rerun,
            afterOverview,
          })
          recordStageResult(record.result, record.meta)
        } else {
          handleFocusQualityReview()
          const record = buildLibraryQualityConversionReviewStageRecord()
          recordStageResult(record.result, record.meta)
        }
        return
      }
      if (stageKind === 'retrieval') {
        const ok = await handleReindex(operationToken)
        if (!qualityOperationIsCurrent(operationToken)) return
        const rerun = ok && caseTarget ? await runQualityFailureCaseRerun(caseTarget, operationToken) : null
        if (!qualityOperationIsCurrent(operationToken)) return
        if (ok && !rerun) await loadQualityOverview('all')
        const afterOverview = await refreshQualityOverviewSnapshot()
        if (!qualityOperationIsCurrent(operationToken)) return
        const record = buildLibraryQualityRetrievalStageRecord({
          ok,
          caseId: caseTarget?.id,
          rerun,
          afterOverview,
        })
        recordStageResult(record.result, record.meta)
        return
      }
      if (stageKind === 'metadata') {
        const result = await repairQualityStageShelfMetadata(stageKey === 'citations' ? 'citations' : 'shelf', operationToken)
        if (!qualityOperationIsCurrent(operationToken)) return
        const rerun = result.targetCount > 0 && caseTarget ? await runQualityFailureCaseRerun(caseTarget, operationToken) : null
        if (!qualityOperationIsCurrent(operationToken)) return
        const afterOverview = await refreshQualityOverviewSnapshot()
        if (!qualityOperationIsCurrent(operationToken)) return
        const record = buildLibraryQualityMetadataStageRecord({
          result,
          caseId: caseTarget?.id,
          rerun,
          afterOverview,
        })
        recordStageResult(record.result, record.meta)
        return
      }
      if (stageKind === 'repair_loop') {
        if (caseTarget) {
          const rerun = await runQualityFailureCaseRerun(caseTarget, operationToken)
          if (!qualityOperationIsCurrent(operationToken)) return
          const afterOverview = await refreshQualityOverviewSnapshot()
          if (!qualityOperationIsCurrent(operationToken)) return
          const record = buildLibraryQualityRepairLoopStageRecord({
            caseId: caseTarget.id,
            rerun,
            afterOverview,
          })
          recordStageResult(record.result, record.meta)
        } else {
          await openQualityArtifact('research_qa', 'report')
          recordStageResult({
            status: 'info',
            summary: 'Opened QA report',
          })
        }
        return
      }
      if (stageKind === 'research_qa') {
        const plan = caseTarget?.repair_actions?.find((item) => item.kind === 'apply_repair_plan')
        if (caseTarget && plan) {
          const result = await applyQualityFailureRepairPlan(caseTarget, plan, operationToken)
          if (!qualityOperationIsCurrent(operationToken)) return
          const afterOverview = await refreshQualityOverviewSnapshot()
          if (!qualityOperationIsCurrent(operationToken)) return
          const record = buildLibraryQualityResearchQaRepairPlanStageRecord({
            caseId: caseTarget.id,
            result,
            afterOverview,
          })
          recordStageResult(record.result, record.meta)
        } else if (caseTarget) {
          const rerun = await runQualityFailureCaseRerun(caseTarget, operationToken)
          if (!qualityOperationIsCurrent(operationToken)) return
          const afterOverview = await refreshQualityOverviewSnapshot()
          if (!qualityOperationIsCurrent(operationToken)) return
          const record = buildLibraryQualityResearchQaRerunStageRecord({
            caseId: caseTarget.id,
            rerun,
            afterOverview,
          })
          recordStageResult(record.result, record.meta)
        } else {
          await openQualityArtifact('research_qa', action === 'run_research_qa' ? 'runbook' : 'report')
          const record = buildLibraryQualityResearchQaOpenStageRecord(action)
          recordStageResult(record.result, record.meta)
        }
        return
      }
      if (stageKind === 'citation_cards') {
        await openQualityArtifact('citation_cards', 'report')
        recordStageResult({
          status: 'info',
          summary: 'Opened citation-card quality report',
        })
        return
      }
      message.info('No direct action is available for this quality stage yet.')
      recordStageResult({
        status: 'info',
        summary: 'No direct action available for this stage',
      })
    } catch (err) {
      if (qualityOperationIsCurrent(operationToken)) {
        message.error(err instanceof Error ? err.message : 'Quality stage action failed')
      }
    } finally {
      if (qualityOperationIsActive(operationToken)) setQualityFullChainActionKey('')
      clearQualityOperation(operationToken)
    }
  }, [
    applyQualityFailureRepairPlan,
    beginQualityOperation,
    clearQualityOperation,
    firstQualityCaseForStage,
    handleFocusQualityReview,
    handleReindex,
    handleRepairRecommendedQuality,
    loadFiles,
    loadQualityOverview,
    openQualityArtifact,
    qualityOperationIsActive,
    qualityOperationIsCurrent,
    qualityRepairRecommendedNames,
    recordQualityFullChainStageResult,
    refreshQualityOverviewSnapshot,
    repairQualityStageShelfMetadata,
    runQualityFailureCaseRerun,
    scope,
    setQualityFullChainActionKey,
    setQualityRepairImpact,
    waitForLibraryConversionDone,
  ])

  return {
    handleQualityFullChainStage,
  }
}
