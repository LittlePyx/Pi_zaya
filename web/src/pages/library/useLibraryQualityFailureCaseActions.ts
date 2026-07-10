import { useCallback } from 'react'
import type { Dispatch, SetStateAction } from 'react'
import { message } from 'antd'
import type { NavigateFunction } from 'react-router-dom'
import {
  libraryApi,
  type LibraryQualityFailureCase,
  type LibraryQualityRepairAction,
  type LibraryQualityRepairBody,
  type LibraryQualityRepairImpact,
  type LibraryQualityRepairResponse,
  type LibraryQualityRepairRun,
  type LibraryResearchQaRerunResponse,
} from '../../api/library'
import {
  referencesApi,
  type ShelfMetadataBackfillJobState,
} from '../../api/references'
import {
  normalizeTextValue,
  qualityFailureCaseMatchesStage,
  saveResearchQaReplayFailureCase,
} from './libraryPageUtils'
import {
  buildShelfMetadataBackfillStageSummary,
} from './useShelfMetadataBackfillActions'
import type { LibraryQualityOperationToken } from './useLibraryQualityOperationGuard'
import type { LibraryQualityRepairRunOptions } from './useLibraryQualityRepairActions'

type QualityArtifactDomain = 'research_qa' | 'citation_cards'
type QualityArtifactTarget = 'report' | 'folder' | 'raw' | 'summary' | 'runbook'

type QualityCaseSourceRepairResult = {
  queued: number
  completed: boolean
  repaired: number
  needsReindex: boolean
  reindexed: boolean
  impact: LibraryQualityRepairImpact | null
}

type QualityCaseMetadataRepairResult = {
  ready: number
  exportReady: number
  changed: number
  retryable: number
  unresolved: number
  verification: Record<string, unknown>
}

type QualityStageMetadataRepairResult = QualityCaseMetadataRepairResult & {
  targetCount: number
  targetIds: string[]
  running: boolean
}

type UseLibraryQualityFailureCaseActionsInput = {
  S: Record<string, string>
  scope: string
  speedMode: string
  internalRoutesEnabled: boolean
  qualityFailureCases: LibraryQualityFailureCase[]
  nav: NavigateFunction
  beginQualityOperation: (key: string) => LibraryQualityOperationToken
  clearQualityOperation: (token?: LibraryQualityOperationToken | null) => void
  qualityOperationIsActive: (token?: LibraryQualityOperationToken | null) => boolean
  qualityOperationIsCurrent: (token?: LibraryQualityOperationToken | null) => boolean
  setQualityCaseActionKey: (key: string) => void
  setQualityCaseRerunResults: Dispatch<SetStateAction<Record<string, LibraryResearchQaRerunResponse>>>
  setQualityRepairImpact: Dispatch<SetStateAction<LibraryQualityRepairImpact | null>>
  setQualityRepairRun: Dispatch<SetStateAction<LibraryQualityRepairRun | null>>
  handleReindex: (operationToken?: LibraryQualityOperationToken) => Promise<boolean>
  waitForLibraryConversionDone: () => Promise<boolean>
  repairQuality: (
    body: LibraryQualityRepairBody,
    options?: { autoReindexAfterQueued?: boolean },
  ) => Promise<LibraryQualityRepairResponse>
  loadFiles: (scope?: string) => Promise<unknown> | unknown
  loadQualityOverview: (scope?: string) => Promise<unknown> | unknown
  openQualityArtifact: (domain: QualityArtifactDomain, target: QualityArtifactTarget) => Promise<void>
  startShelfMetadataBackfill: (
    options?: { silent?: boolean; operationToken?: LibraryQualityOperationToken },
  ) => Promise<ShelfMetadataBackfillJobState | null>
}

export function useLibraryQualityFailureCaseActions({
  S,
  scope,
  speedMode,
  internalRoutesEnabled,
  qualityFailureCases,
  nav,
  beginQualityOperation,
  clearQualityOperation,
  qualityOperationIsActive,
  qualityOperationIsCurrent,
  setQualityCaseActionKey,
  setQualityCaseRerunResults,
  setQualityRepairImpact,
  setQualityRepairRun,
  handleReindex,
  waitForLibraryConversionDone,
  repairQuality,
  loadFiles,
  loadQualityOverview,
  openQualityArtifact,
  startShelfMetadataBackfill,
}: UseLibraryQualityFailureCaseActionsInput) {
  const openResearchQaReplayCase = useCallback((item: LibraryQualityFailureCase) => {
    const caseId = normalizeTextValue(item.id)
    if (!caseId) {
      void openQualityArtifact('research_qa', 'report')
      return
    }
    saveResearchQaReplayFailureCase(item)
    if (!internalRoutesEnabled) {
      void openQualityArtifact('research_qa', 'report')
      return
    }
    nav(`/__research_qa_replay__?case=${encodeURIComponent(caseId)}&source=quality`)
  }, [
    internalRoutesEnabled,
    nav,
    openQualityArtifact,
  ])

  const qualityCaseRepairSources = useCallback((item: LibraryQualityFailureCase) => (
    (item.source_diagnostics || [])
      .filter((source) => Boolean(source.repairable))
      .filter((source) => Boolean(source.needs_repair) || ['error', 'warning'].includes(normalizeTextValue(source.quality_status).toLowerCase()))
      .map((source) => ({
        source_path: normalizeTextValue(source.source_path || source.md_path || source.pdf_path),
        source_name: normalizeTextValue(source.source_name || source.title),
      }))
      .filter((source) => source.source_path || source.source_name)
  ), [])

  const repairQualityCaseSources = useCallback(async (
    item: LibraryQualityFailureCase,
    opts: { manageActionKey?: boolean; waitForCompletion?: boolean; silent?: boolean; actionKey?: string } & LibraryQualityRepairRunOptions = {},
  ): Promise<QualityCaseSourceRepairResult> => {
    const sources = qualityCaseRepairSources(item)
    if (!sources.length) {
      if (!opts.silent) message.info(S.lib_msg_quality_repair_none)
      return { queued: 0, completed: true, repaired: 0, needsReindex: false, reindexed: false, impact: null }
    }
    const operationToken = opts.operationToken || beginQualityOperation(`case-source-repair:${normalizeTextValue(item.id)}`)
    const ownsOperation = !opts.operationToken
    const key = opts.actionKey || `${item.id}:repair_sources`
    const manageActionKey = opts.manageActionKey !== false
    if (manageActionKey) setQualityCaseActionKey(key)
    try {
      const res = await repairQuality({
        sources,
        speed_mode: speedMode,
        replace: true,
      }, {
        autoReindexAfterQueued: opts.waitForCompletion ? false : opts.autoReindexQueued !== false,
      })
      const queued = Number(res.enqueued || 0)
      const repaired = Number(res.repaired || 0)
      const impact = res.impact || null
      const needsReindex = Boolean(res.needs_reindex || impact?.needs_reindex)
      let reindexed = false
      if (!qualityOperationIsCurrent(operationToken)) {
        return { queued, completed: false, repaired, needsReindex, reindexed, impact: null }
      }
      if (res.repair_run) {
        setQualityRepairRun(res.repair_run)
      }
      if (impact) {
        setQualityRepairImpact(impact)
      }
      if (queued > 0) {
        if (!opts.silent) message.success(S.lib_msg_quality_repair_enqueued.replace('{n}', String(queued)))
        const completed = opts.waitForCompletion ? await waitForLibraryConversionDone() : false
        if (!qualityOperationIsCurrent(operationToken)) {
          return { queued, completed: false, repaired, needsReindex, reindexed, impact: null }
        }
        if (completed && needsReindex && opts.autoReindexImmediate !== false) {
          reindexed = await handleReindex(operationToken)
          if (!qualityOperationIsCurrent(operationToken)) {
            return { queued, completed: false, repaired, needsReindex, reindexed, impact: null }
          }
          if (impact) setQualityRepairImpact({ ...impact, reindexed })
          if (res.repair_run?.run_id) {
            const status = reindexed ? 'completed' : 'warning'
            const phase = reindexed ? 'reindex_complete' : 'reindex_failed'
            setQualityRepairRun({ ...res.repair_run, status, phase, reindexed })
            libraryApi.updateQualityRepairRun(res.repair_run.run_id, { status, phase, reindexed }).catch(() => {})
          }
          if (reindexed) await loadFiles(scope)
        }
        return { queued, completed, repaired, needsReindex, reindexed, impact }
      } else if (repaired > 0) {
        if (!opts.silent) message.success(`Markdown repaired: ${repaired}`)
        if (needsReindex && opts.autoReindexImmediate !== false) {
          reindexed = await handleReindex(operationToken)
          if (!qualityOperationIsCurrent(operationToken)) {
            return { queued: 0, completed: false, repaired, needsReindex, reindexed, impact: null }
          }
          if (impact) setQualityRepairImpact({ ...impact, reindexed })
          if (res.repair_run?.run_id) {
            const status = reindexed ? 'completed' : 'warning'
            const phase = reindexed ? 'reindex_complete' : 'reindex_failed'
            setQualityRepairRun({ ...res.repair_run, status, phase, reindexed })
            libraryApi.updateQualityRepairRun(res.repair_run.run_id, { status, phase, reindexed }).catch(() => {})
          }
          if (reindexed) await loadFiles(scope)
        }
        return { queued: 0, completed: true, repaired, needsReindex, reindexed, impact }
      } else {
        if (!opts.silent) message.info(S.lib_msg_quality_repair_none)
        return { queued: 0, completed: true, repaired, needsReindex, reindexed, impact }
      }
    } catch (err) {
      if (qualityOperationIsCurrent(operationToken)) {
        message.error(err instanceof Error ? err.message : S.lib_msg_quality_repair_failed)
      }
      return { queued: 0, completed: false, repaired: 0, needsReindex: false, reindexed: false, impact: null }
    } finally {
      if (manageActionKey && qualityOperationIsActive(operationToken)) setQualityCaseActionKey('')
      if (ownsOperation) clearQualityOperation(operationToken)
    }
  }, [
    S,
    beginQualityOperation,
    clearQualityOperation,
    handleReindex,
    loadFiles,
    qualityCaseRepairSources,
    qualityOperationIsActive,
    qualityOperationIsCurrent,
    repairQuality,
    scope,
    setQualityCaseActionKey,
    setQualityRepairImpact,
    setQualityRepairRun,
    speedMode,
    waitForLibraryConversionDone,
  ])

  const copyQualityFailureSummary = useCallback(async (item: LibraryQualityFailureCase) => {
    const lines = [
      `Case: ${normalizeTextValue(item.id)}`,
      `Question: ${normalizeTextValue(item.question)}`,
      `Failures: ${(item.failure_names || []).join(' / ') || 'none'}`,
      `Missing docs: ${(item.missing_expected_doc_ids || []).join(' / ') || 'none'}`,
      `Root causes: ${(item.root_causes || []).map((cause) => cause.label).join(' / ') || 'unknown'}`,
      `Sources: ${(item.source_diagnostics || []).map((source) => source.title || source.source_name || source.source_path).filter(Boolean).join(' / ') || 'none'}`,
    ].join('\n')
    try {
      await navigator.clipboard.writeText(lines)
      message.success(S.copied || 'Copied')
    } catch {
      message.error(S.copy_failed || 'Copy failed')
    }
  }, [S])

  const storeQualityCaseRerunResult = useCallback((caseId: string, res: LibraryResearchQaRerunResponse) => {
    setQualityCaseRerunResults((cur) => ({ ...cur, [caseId]: res }))
    if (res.status === 'passed' || res.quality_ok) {
      message.success(`QA case passed: ${caseId}`)
    } else if (res.status === 'failed') {
      message.warning(`QA case still failing: ${caseId}`)
    } else if (normalizeTextValue(res.error_kind).toLowerCase() === 'connection') {
      message.warning(`QA service is unreachable: ${caseId}`)
    } else if (normalizeTextValue(res.error_kind).toLowerCase() === 'timeout') {
      message.warning(`QA rerun timed out: ${caseId}`)
    } else {
      message.error(`QA rerun error: ${caseId}`)
    }
  }, [setQualityCaseRerunResults])

  const runQualityFailureCaseRerun = useCallback(async (
    item: LibraryQualityFailureCase,
    operationToken?: LibraryQualityOperationToken,
  ) => {
    const caseId = normalizeTextValue(item.id)
    if (!caseId) return null
    const token = operationToken || beginQualityOperation(`qa-rerun:${caseId}`)
    const ownsOperation = !operationToken
    const res = await libraryApi.rerunResearchQaCase({ case_id: caseId })
    if (!qualityOperationIsCurrent(token)) {
      if (ownsOperation) clearQualityOperation(token)
      return null
    }
    storeQualityCaseRerunResult(caseId, res)
    await loadQualityOverview('all')
    if (ownsOperation) clearQualityOperation(token)
    return res
  }, [
    beginQualityOperation,
    clearQualityOperation,
    loadQualityOverview,
    qualityOperationIsCurrent,
    storeQualityCaseRerunResult,
  ])

  const rerunQualityFailureCase = useCallback(async (item: LibraryQualityFailureCase) => {
    const caseId = normalizeTextValue(item.id)
    if (!caseId) return
    const key = `${item.id}:rerun_case:`
    setQualityCaseActionKey(key)
    const operationToken = beginQualityOperation(key)
    try {
      await runQualityFailureCaseRerun(item, operationToken)
    } catch (err) {
      if (qualityOperationIsCurrent(operationToken)) {
        message.error(err instanceof Error ? err.message : 'QA rerun failed')
      }
    } finally {
      if (qualityOperationIsActive(operationToken)) setQualityCaseActionKey('')
      clearQualityOperation(operationToken)
    }
  }, [
    beginQualityOperation,
    clearQualityOperation,
    qualityOperationIsActive,
    qualityOperationIsCurrent,
    runQualityFailureCaseRerun,
    setQualityCaseActionKey,
  ])

  const repairQualityCaseShelfMetadata = useCallback(async (
    item: LibraryQualityFailureCase,
    operationToken?: LibraryQualityOperationToken,
  ): Promise<QualityCaseMetadataRepairResult> => {
    const token = operationToken || beginQualityOperation(`case-metadata-repair:${normalizeTextValue(item.id)}`)
    const ownsOperation = !operationToken
    const backendTargets = Array.isArray(item.shelf_metadata_repair_targets) ? item.shelf_metadata_repair_targets : []
    const fallbackItems = [
      ...(Array.isArray(item.citation_diagnostics) ? item.citation_diagnostics : []),
      ...(Array.isArray(item.ref_diagnostics) ? item.ref_diagnostics : []),
    ]
    const sourceItems = backendTargets.length > 0 ? backendTargets : fallbackItems
    const candidates = sourceItems
      .map((entry, index) => ({
        record: entry as unknown as Record<string, unknown>,
        index,
      }))
      .map(({ record, index }) => ({
        ...record,
        key: normalizeTextValue(record.key) || `${item.id}:${backendTargets.length > 0 ? 'shelf-meta' : 'meta'}:${index}`,
        anchor: normalizeTextValue(record.anchor) || `${item.id}-${backendTargets.length > 0 ? 'shelf-meta' : 'meta'}-${index}`,
        title: normalizeTextValue(record.title),
        source_path: normalizeTextValue(record.source_path),
        source_name: normalizeTextValue(record.source_name || record.title),
        raw: normalizeTextValue(record.raw || record.cite_fmt || record.evidence_quote),
      }))
      .filter((entry) => entry.source_path || entry.source_name || entry.title || entry.raw)
      .slice(0, 12)
    if (!candidates.length) {
      if (ownsOperation) clearQualityOperation(token)
      return { ready: 0, exportReady: 0, changed: 0, retryable: 0, unresolved: 0, verification: {} }
    }
    let res: Awaited<ReturnType<typeof referencesApi.repairShelfMetadata>>
    try {
      res = await referencesApi.repairShelfMetadata(candidates as Array<Record<string, unknown>>, candidates.length)
    } catch (err) {
      if (ownsOperation) clearQualityOperation(token)
      throw err
    }
    const ready = Number(res.ready || 0)
    const exportReady = Number(res.acceptance?.export_ready_after || res.export_ready || ready)
    const changed = Number(res.changed || 0)
    const retryable = Number(res.retryable || 0)
    const unresolved = Number(res.acceptance?.unresolved_after || res.unresolved || 0)
    if (!qualityOperationIsCurrent(token)) {
      if (ownsOperation) clearQualityOperation(token)
      return { ready: 0, exportReady: 0, changed: 0, retryable: 0, unresolved: 0, verification: {} }
    }
    if (res.repair_run) {
      setQualityRepairRun(res.repair_run as unknown as LibraryQualityRepairRun)
    }
    if (retryable > 0) {
      message.warning(`Metadata repair queued for retry: ${retryable}`)
    } else if (changed > 0) {
      message.success(`Citation metadata repaired: ${changed}`)
    }
    if (ownsOperation) clearQualityOperation(token)
    return { ready, exportReady, changed, retryable, unresolved, verification: (res.verification || res.repair_run?.verification || {}) as Record<string, unknown> }
  }, [
    beginQualityOperation,
    clearQualityOperation,
    qualityOperationIsCurrent,
    setQualityRepairRun,
  ])

  const applyQualityFailureRepairPlan = useCallback(async (
    item: LibraryQualityFailureCase,
    action: LibraryQualityRepairAction,
    operationToken?: LibraryQualityOperationToken,
  ) => {
    const caseId = normalizeTextValue(item.id)
    const steps = Array.isArray(action.steps) ? action.steps : []
    const stepKinds = new Set(steps.map((step) => normalizeTextValue(step.kind)))
    const key = `${item.id}:apply_repair_plan:${action.target || ''}`
    const token = operationToken || beginQualityOperation(key)
    const ownsOperation = !operationToken
    setQualityCaseActionKey(key)
    try {
      let sourceRepairImpact: LibraryQualityRepairImpact | null = null
      if (stepKinds.has('repair_sources')) {
        const result = await repairQualityCaseSources(item, {
          actionKey: key,
          manageActionKey: false,
          waitForCompletion: true,
          autoReindexImmediate: !stepKinds.has('rebuild_index'),
          operationToken: token,
        })
        if (!qualityOperationIsCurrent(token)) return { ok: false, caseId, status: 'stale', rerun: null as LibraryResearchQaRerunResponse | null }
        sourceRepairImpact = result.impact
        if (result.queued > 0 && !result.completed) {
          message.warning('Source repair is still running; QA rerun will wait for the next refresh.')
          await loadQualityOverview('all')
          return { ok: true, caseId, status: 'source_repair_running', rerun: null as LibraryResearchQaRerunResponse | null }
        }
      }
      if (stepKinds.has('repair_shelf_metadata')) {
        await repairQualityCaseShelfMetadata(item, token)
        if (!qualityOperationIsCurrent(token)) return { ok: false, caseId, status: 'stale', rerun: null as LibraryResearchQaRerunResponse | null }
      }
      if (stepKinds.has('rebuild_index')) {
        const ok = await handleReindex(token)
        if (!qualityOperationIsCurrent(token)) return { ok: false, caseId, status: 'stale', rerun: null as LibraryResearchQaRerunResponse | null }
        if (sourceRepairImpact) setQualityRepairImpact({ ...sourceRepairImpact, reindexed: ok })
        if (!ok) return { ok: false, caseId, status: 'reindex_failed', rerun: null as LibraryResearchQaRerunResponse | null }
      }
      if (stepKinds.has('rerun_case') && caseId) {
        const rerun = await runQualityFailureCaseRerun(item, token)
        if (!qualityOperationIsCurrent(token)) return { ok: false, caseId, status: 'stale', rerun: null as LibraryResearchQaRerunResponse | null }
        return { ok: Boolean(rerun?.quality_ok || rerun?.status === 'passed'), caseId, status: String(rerun?.status || ''), rerun }
      } else {
        await loadQualityOverview('all')
      }
      return { ok: true, caseId, status: 'repaired', rerun: null as LibraryResearchQaRerunResponse | null }
    } catch (err) {
      if (qualityOperationIsCurrent(token)) {
        message.error(err instanceof Error ? err.message : 'Quality repair plan failed')
      }
      return { ok: false, caseId, status: 'error', rerun: null as LibraryResearchQaRerunResponse | null }
    } finally {
      if (qualityOperationIsActive(token)) setQualityCaseActionKey('')
      if (ownsOperation) clearQualityOperation(token)
    }
  }, [
    beginQualityOperation,
    clearQualityOperation,
    handleReindex,
    loadQualityOverview,
    qualityOperationIsActive,
    qualityOperationIsCurrent,
    repairQualityCaseShelfMetadata,
    repairQualityCaseSources,
    runQualityFailureCaseRerun,
    setQualityCaseActionKey,
    setQualityRepairImpact,
  ])

  const handleQualityFailureAction = useCallback(async (
    item: LibraryQualityFailureCase,
    actionOrKind: LibraryQualityRepairAction | string,
    target = '',
  ) => {
    const actionKind = typeof actionOrKind === 'string' ? actionOrKind : actionOrKind.kind
    const actionTarget = typeof actionOrKind === 'string' ? target : (actionOrKind.target || target)
    const key = `${item.id}:${actionKind}:${actionTarget}`
    if (actionKind === 'open_replay') {
      openResearchQaReplayCase(item)
      return
    }
    if (actionKind === 'apply_repair_plan' && typeof actionOrKind !== 'string') {
      await applyQualityFailureRepairPlan(item, actionOrKind)
      return
    }
    if (actionKind === 'rerun_case') {
      await rerunQualityFailureCase(item)
      return
    }
    if (actionKind === 'repair_sources') {
      await repairQualityCaseSources(item)
      return
    }
    if (actionKind === 'rebuild_index') {
      const operationToken = beginQualityOperation(key)
      setQualityCaseActionKey(key)
      try {
        await handleReindex(operationToken)
      } finally {
        if (qualityOperationIsActive(operationToken)) setQualityCaseActionKey('')
        clearQualityOperation(operationToken)
      }
      return
    }
    if (actionKind === 'open_artifact') {
      await openQualityArtifact('research_qa', actionTarget === 'raw' ? 'raw' : 'report')
    }
  }, [
    applyQualityFailureRepairPlan,
    beginQualityOperation,
    clearQualityOperation,
    handleReindex,
    openQualityArtifact,
    openResearchQaReplayCase,
    qualityOperationIsActive,
    repairQualityCaseSources,
    rerunQualityFailureCase,
    setQualityCaseActionKey,
  ])

  const firstQualityCaseForStage = useCallback((stageKey: string) => (
    qualityFailureCases.find((item) => qualityFailureCaseMatchesStage(item, stageKey)) || qualityFailureCases[0] || null
  ), [qualityFailureCases])

  const repairQualityStageShelfMetadata = useCallback(async (
    stageKey: string,
    operationToken?: LibraryQualityOperationToken,
  ): Promise<QualityStageMetadataRepairResult> => {
    const token = operationToken || beginQualityOperation(`stage-metadata:${stageKey}`)
    const ownsOperation = !operationToken
    const targets = qualityFailureCases.filter((item) => qualityFailureCaseMatchesStage(item, stageKey)).slice(0, 3)
    if (!targets.length) {
      const state = await startShelfMetadataBackfill({ silent: true, operationToken: token })
      if (!qualityOperationIsCurrent(token)) {
        if (ownsOperation) clearQualityOperation(token)
        return { targetCount: 0, targetIds: [], ready: 0, exportReady: 0, changed: 0, retryable: 0, unresolved: 0, verification: {}, running: false }
      }
      const summary = buildShelfMetadataBackfillStageSummary(state)
      if (summary.repairRun) {
        setQualityRepairRun(summary.repairRun as unknown as LibraryQualityRepairRun)
      }
      if (summary.running) {
        message.success('Library metadata backfill is running')
      } else if (summary.changed > 0) {
        message.success(`Library metadata backfilled: ${summary.changed}`)
      } else if (summary.retryable > 0) {
        message.warning(`Library metadata can retry: ${summary.retryable}`)
      } else if (summary.targetCount > 0 && summary.exportReady > 0) {
        message.success(`Library metadata export-ready: ${summary.exportReady}`)
      } else {
        message.info('No repairable library metadata found.')
      }
      await loadQualityOverview('all')
      if (ownsOperation) clearQualityOperation(token)
      return summary
    }
    let changed = 0
    let ready = 0
    let exportReady = 0
    let retryable = 0
    let unresolved = 0
    let verification: Record<string, unknown> = {}
    for (const item of targets) {
      const res = await repairQualityCaseShelfMetadata(item, token)
      if (!qualityOperationIsCurrent(token)) {
        if (ownsOperation) clearQualityOperation(token)
        return { targetCount: 0, targetIds: [], ready: 0, exportReady: 0, changed: 0, retryable: 0, unresolved: 0, verification: {}, running: false }
      }
      changed += Number(res.changed || 0)
      ready += Number(res.ready || 0)
      exportReady += Number(res.exportReady || 0)
      retryable += Number(res.retryable || 0)
      unresolved += Number(res.unresolved || 0)
      if (!Object.keys(verification).length && res.verification && Object.keys(res.verification).length) {
        verification = res.verification
      }
    }
    if (changed <= 0 && ready <= 0) {
      message.info('No repairable citation metadata found in the current failed cases.')
    }
    await loadQualityOverview('all')
    if (ownsOperation) clearQualityOperation(token)
    return {
      targetCount: targets.length,
      targetIds: targets.map((item) => normalizeTextValue(item.id)).filter(Boolean),
      ready,
      exportReady,
      changed,
      retryable,
      unresolved,
      verification,
      running: false,
    }
  }, [
    beginQualityOperation,
    clearQualityOperation,
    loadQualityOverview,
    qualityFailureCases,
    qualityOperationIsCurrent,
    repairQualityCaseShelfMetadata,
    setQualityRepairRun,
    startShelfMetadataBackfill,
  ])

  return {
    applyQualityFailureRepairPlan,
    copyQualityFailureSummary,
    firstQualityCaseForStage,
    handleQualityFailureAction,
    openResearchQaReplayCase,
    repairQualityStageShelfMetadata,
    runQualityFailureCaseRerun,
  }
}
