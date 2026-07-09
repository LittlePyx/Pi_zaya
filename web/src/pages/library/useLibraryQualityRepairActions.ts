import { useCallback, useEffect, useRef, useState } from 'react'
import type { Dispatch, SetStateAction } from 'react'
import { message } from 'antd'
import {
  libraryApi,
  type ConversionQualitySummary,
  type LibraryFileItem,
  type LibraryQualityRepairBody,
  type LibraryQualityRepairImpact,
  type LibraryQualityRepairResponse,
  type LibraryQualityRepairRun,
} from '../../api/library'
import type { LibraryQualityOperationToken } from './useLibraryQualityOperationGuard'
import {
  buildQualityRepairHistoryRecord,
  hasConversionQualityIssue,
  normalizeQualityRepairHistory,
  saveQualityRepairHistory,
  summarizeConversionQualityRepair,
  type QualityRepairHistoryRecord,
} from './libraryPageUtils'

export type LibraryQualityRepairRunOptions = {
  autoReindexImmediate?: boolean
  autoReindexQueued?: boolean
  operationToken?: LibraryQualityOperationToken
}

type QualityRepairBaseline = {
  quality: ConversionQualitySummary | null
  startedAt: number
}

export function normalizeLibraryQualityRepairTargets(names: string[]) {
  return Array.from(new Set(names.map((name) => String(name || '').trim()).filter(Boolean)))
}

export function buildLibraryQualityRepairRunCompletionPatch(
  repairRun: LibraryQualityRepairRun,
  reindexed: boolean,
): LibraryQualityRepairRun {
  const status = reindexed ? 'completed' : 'warning'
  const phase = reindexed ? 'reindex_complete' : 'reindex_failed'
  return { ...repairRun, status, phase, reindexed }
}

type LibraryQualityRepairActionsInput = {
  S: Record<string, string>
  files: LibraryFileItem[]
  scope: string
  speedMode: string
  qualityRepairRecommendedNames: string[]
  selectedQualityReviewNames: string[]
  beginQualityOperation: (key: string) => LibraryQualityOperationToken
  clearQualityOperation: (token?: LibraryQualityOperationToken) => void
  handleReindex: (operationToken?: LibraryQualityOperationToken) => Promise<boolean>
  loadFiles: (scope?: string) => Promise<unknown> | unknown
  qualityOperationIsCurrent: (token?: LibraryQualityOperationToken) => boolean
  repairQuality: (
    body: LibraryQualityRepairBody,
    options?: { autoReindexAfterQueued?: boolean },
  ) => Promise<LibraryQualityRepairResponse>
  setQualityRepairHistory: Dispatch<SetStateAction<Record<string, QualityRepairHistoryRecord>>>
  setQualityRepairImpact: Dispatch<SetStateAction<LibraryQualityRepairImpact | null>>
  setQualityRepairRun: Dispatch<SetStateAction<LibraryQualityRepairRun | null>>
}

export function useLibraryQualityRepairActions({
  S,
  files,
  scope,
  speedMode,
  qualityRepairRecommendedNames,
  selectedQualityReviewNames,
  beginQualityOperation,
  clearQualityOperation,
  handleReindex,
  loadFiles,
  qualityOperationIsCurrent,
  repairQuality,
  setQualityRepairHistory,
  setQualityRepairImpact,
  setQualityRepairRun,
}: LibraryQualityRepairActionsInput) {
  const qualityRepairBaselinesRef = useRef<Record<string, QualityRepairBaseline>>({})
  const [qualityRepairingNames, setQualityRepairingNames] = useState<Record<string, boolean>>({})
  const [qualityRepairResults, setQualityRepairResults] = useState<Record<string, string>>({})

  useEffect(() => {
    const pending = qualityRepairBaselinesRef.current
    const pendingNames = Object.keys(pending)
    if (!pendingNames.length) return
    const nextPending = { ...pending }
    const nextResults: Record<string, string> = {}
    const nextHistory: Record<string, QualityRepairHistoryRecord> = {}
    for (const item of files) {
      const baseline = pending[item.name]
      if (!baseline) continue
      if (item.task_state !== 'idle' || !item.conversion_quality) continue
      nextResults[item.name] = summarizeConversionQualityRepair(baseline.quality, item.conversion_quality, S)
      nextHistory[item.name] = buildQualityRepairHistoryRecord(item.name, baseline.quality, item.conversion_quality)
      delete nextPending[item.name]
    }
    if (Object.keys(nextResults).length <= 0) return
    qualityRepairBaselinesRef.current = nextPending
    setQualityRepairResults((cur) => ({ ...cur, ...nextResults }))
    setQualityRepairHistory((cur) => {
      const merged = normalizeQualityRepairHistory({ ...cur, ...nextHistory })
      saveQualityRepairHistory(merged)
      return merged
    })
  }, [S, files, setQualityRepairHistory])

  const clearRepairBaselines = useCallback((targets: string[]) => {
    qualityRepairBaselinesRef.current = Object.fromEntries(
      Object.entries(qualityRepairBaselinesRef.current).filter(([name]) => !targets.includes(name)),
    )
  }, [])

  const repairQualityByNames = useCallback(async (
    names: string[],
    opts: LibraryQualityRepairRunOptions = {},
  ) => {
    const targets = normalizeLibraryQualityRepairTargets(names)
    if (!targets.length) {
      message.info(S.lib_msg_quality_repair_none)
      return { ok: true, targetCount: 0, queued: 0, repaired: 0, needsReindex: false, reindexed: false, impact: null as LibraryQualityRepairImpact | null }
    }
    const operationToken = opts.operationToken || beginQualityOperation(`quality-repair:${targets.join('|')}`)
    const ownsOperation = !opts.operationToken
    const startedAt = Date.now()
    const baselineByName = new Map(files.map((item) => [item.name, item.conversion_quality || null]))
    qualityRepairBaselinesRef.current = {
      ...qualityRepairBaselinesRef.current,
      ...Object.fromEntries(targets.map((name) => [name, { quality: baselineByName.get(name) || null, startedAt }])),
    }
    setQualityRepairResults((cur) => {
      const next = { ...cur }
      for (const name of targets) delete next[name]
      return next
    })
    setQualityRepairingNames((cur) => {
      const next = { ...cur }
      for (const name of targets) next[name] = true
      return next
    })
    try {
      const res = await repairQuality({
        pdf_names: targets,
        speed_mode: speedMode,
        replace: true,
      }, {
        autoReindexAfterQueued: opts.autoReindexQueued !== false,
      })
      const queued = Number(res.enqueued || 0)
      const repaired = Number(res.repaired || 0)
      const impact = res.impact || null
      const needsReindex = Boolean(res.needs_reindex || impact?.needs_reindex)
      let reindexed = false
      if (!qualityOperationIsCurrent(operationToken)) {
        return { ok: false, targetCount: targets.length, queued, repaired, needsReindex, reindexed, impact: null as LibraryQualityRepairImpact | null }
      }
      if (res.repair_run) {
        setQualityRepairRun(res.repair_run)
      }
      if (impact) {
        setQualityRepairImpact(impact)
      }
      if (queued > 0 || repaired > 0) {
        message.success(
          queued > 0
            ? S.lib_msg_quality_repair_enqueued.replace('{n}', String(queued))
            : `Markdown repaired: ${repaired}`,
        )
      } else {
        clearRepairBaselines(targets)
        message.info(S.lib_msg_quality_repair_none)
      }
      await loadFiles(scope)
      if (!qualityOperationIsCurrent(operationToken)) {
        return { ok: false, targetCount: targets.length, queued, repaired, needsReindex, reindexed, impact: null as LibraryQualityRepairImpact | null }
      }
      if (needsReindex && repaired > 0 && queued <= 0 && opts.autoReindexImmediate !== false) {
        reindexed = await handleReindex(operationToken)
        if (!qualityOperationIsCurrent(operationToken)) {
          return { ok: false, targetCount: targets.length, queued, repaired, needsReindex, reindexed, impact: null as LibraryQualityRepairImpact | null }
        }
        if (impact) {
          setQualityRepairImpact({ ...impact, reindexed })
        }
        if (res.repair_run?.run_id) {
          const nextRun = buildLibraryQualityRepairRunCompletionPatch(res.repair_run, reindexed)
          setQualityRepairRun(nextRun)
          libraryApi.updateQualityRepairRun(res.repair_run.run_id, {
            status: nextRun.status,
            phase: nextRun.phase,
            reindexed,
          }).catch(() => {})
        }
        if (reindexed) await loadFiles(scope)
      }
      return { ok: true, targetCount: targets.length, queued, repaired, needsReindex, reindexed, impact }
    } catch (err) {
      clearRepairBaselines(targets)
      if (qualityOperationIsCurrent(operationToken)) {
        message.error(err instanceof Error ? err.message : S.lib_msg_quality_repair_failed)
      }
      return { ok: false, targetCount: targets.length, queued: 0, repaired: 0, needsReindex: false, reindexed: false, impact: null as LibraryQualityRepairImpact | null }
    } finally {
      setQualityRepairingNames((cur) => {
        const next = { ...cur }
        for (const name of targets) delete next[name]
        return next
      })
      if (ownsOperation) clearQualityOperation(operationToken)
    }
  }, [
    S,
    beginQualityOperation,
    clearQualityOperation,
    clearRepairBaselines,
    files,
    handleReindex,
    loadFiles,
    qualityOperationIsCurrent,
    repairQuality,
    scope,
    setQualityRepairImpact,
    setQualityRepairRun,
    speedMode,
  ])

  const handleRepairQualityOne = useCallback(async (item: LibraryFileItem) => {
    if (item.task_state !== 'idle' || !hasConversionQualityIssue(item)) return
    await repairQualityByNames([item.name])
  }, [repairQualityByNames])

  const handleRepairSelectedQuality = useCallback(async () => {
    await repairQualityByNames(selectedQualityReviewNames)
  }, [repairQualityByNames, selectedQualityReviewNames])

  const handleRepairRecommendedQuality = useCallback(async (opts: LibraryQualityRepairRunOptions = {}) => {
    if (!qualityRepairRecommendedNames.length) {
      message.info(S.lib_quality_history_no_recommended)
      return { ok: true, targetCount: 0, queued: 0, repaired: 0, needsReindex: false, reindexed: false, impact: null as LibraryQualityRepairImpact | null }
    }
    return repairQualityByNames(qualityRepairRecommendedNames, opts)
  }, [S.lib_quality_history_no_recommended, qualityRepairRecommendedNames, repairQualityByNames])

  return {
    handleRepairQualityOne,
    handleRepairRecommendedQuality,
    handleRepairSelectedQuality,
    qualityRepairingNames,
    qualityRepairResults,
    repairQualityByNames,
  }
}
