import { useCallback, useEffect, useState } from 'react'
import type { Dispatch, SetStateAction } from 'react'
import { message } from 'antd'
import {
  referencesApi,
  type ShelfMetadataBackfillJobState,
} from '../../api/references'
import type { LibraryQualityOperationToken } from './useLibraryQualityOperationGuard'

export type ShelfMetadataBackfillStageSummary = {
  targetCount: number
  targetIds: string[]
  ready: number
  exportReady: number
  changed: number
  retryable: number
  unresolved: number
  verification: Record<string, unknown>
  running: boolean
  repairRun: Record<string, unknown> | null
}

export function buildShelfMetadataBackfillStageSummary(
  state: ShelfMetadataBackfillJobState | null | undefined,
): ShelfMetadataBackfillStageSummary {
  const res = state?.result || null
  const scan = state?.after_scan || res?.after_scan || state?.scan || res?.scan || null
  const ready = Number(res?.ready || scan?.ready || 0)
  const exportReady = Number(res?.acceptance?.export_ready_after || res?.export_ready || scan?.export_ready || ready)
  const changed = Number(res?.changed || res?.preheated || 0)
  const retryable = Number(res?.retryable || scan?.retryable || 0)
  const unresolved = Number(
    res?.acceptance?.unresolved_after
    || res?.unresolved
    || res?.remaining_targets
    || scan?.needs_repair
    || 0,
  )
  const targetCount = Number(res?.requested || scan?.target_count || state?.target_total || 0)
  const verification = (
    state?.verification
    || res?.verification
    || res?.repair_run?.verification
    || {}
  ) as Record<string, unknown>

  return {
    targetCount,
    targetIds: [],
    ready,
    exportReady,
    changed,
    retryable,
    unresolved,
    verification,
    running: Boolean(state?.running),
    repairRun: (res?.repair_run || null) as Record<string, unknown> | null,
  }
}

type UseShelfMetadataBackfillActionsInput = {
  beginQualityOperation: (key: string) => LibraryQualityOperationToken
  clearQualityOperation: (token?: LibraryQualityOperationToken | null) => void
  qualityOperationIsActive: (token?: LibraryQualityOperationToken | null) => boolean
  qualityOperationIsCurrent: (token?: LibraryQualityOperationToken | null) => boolean
  setShelfMetadataBackfillRefreshing: Dispatch<SetStateAction<boolean>>
}

export function useShelfMetadataBackfillActions({
  beginQualityOperation,
  clearQualityOperation,
  qualityOperationIsActive,
  qualityOperationIsCurrent,
  setShelfMetadataBackfillRefreshing,
}: UseShelfMetadataBackfillActionsInput) {
  const [shelfMetadataBackfillState, setShelfMetadataBackfillState] = useState<ShelfMetadataBackfillJobState | null>(null)

  const refreshShelfMetadataBackfillState = useCallback(async (silent = true) => {
    if (!silent) setShelfMetadataBackfillRefreshing(true)
    try {
      const state = await referencesApi.shelfMetadataBackfillStatus()
      setShelfMetadataBackfillState(state)
      return state
    } catch (err) {
      if (!silent) message.error(err instanceof Error ? err.message : 'Metadata backfill status failed')
      return null
    } finally {
      if (!silent) setShelfMetadataBackfillRefreshing(false)
    }
  }, [setShelfMetadataBackfillRefreshing])

  useEffect(() => {
    let cancelled = false
    referencesApi.shelfMetadataBackfillStatus()
      .then((state) => {
        if (!cancelled) setShelfMetadataBackfillState(state)
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!shelfMetadataBackfillState?.running) return
    const timer = window.setInterval(() => {
      void refreshShelfMetadataBackfillState(true)
    }, 1600)
    return () => window.clearInterval(timer)
  }, [
    refreshShelfMetadataBackfillState,
    shelfMetadataBackfillState?.job_id,
    shelfMetadataBackfillState?.running,
  ])

  const startShelfMetadataBackfill = useCallback(async (
    options: { silent?: boolean; operationToken?: LibraryQualityOperationToken } = {},
  ) => {
    const operationToken = options.operationToken || beginQualityOperation('shelf-metadata-backfill')
    const ownsOperation = !options.operationToken
    setShelfMetadataBackfillRefreshing(true)
    try {
      const res = await referencesApi.startShelfMetadataBackfill(40, 240)
      if (!qualityOperationIsCurrent(operationToken)) return null
      setShelfMetadataBackfillState(res.state)
      const state = res.state || null
      if (!options.silent) {
        if (res.started) {
          message.success('Library metadata backfill started')
        } else if (state?.running || res.reason === 'already_running') {
          message.info('Library metadata backfill is already running')
        } else {
          message.warning('Library metadata backfill did not start')
        }
      }
      return state
    } catch (err) {
      if (qualityOperationIsCurrent(operationToken) && !options.silent) {
        message.error(err instanceof Error ? err.message : 'Metadata backfill failed to start')
      }
      return null
    } finally {
      if (qualityOperationIsActive(operationToken)) setShelfMetadataBackfillRefreshing(false)
      if (ownsOperation) clearQualityOperation(operationToken)
    }
  }, [
    beginQualityOperation,
    clearQualityOperation,
    qualityOperationIsActive,
    qualityOperationIsCurrent,
    setShelfMetadataBackfillRefreshing,
  ])

  return {
    refreshShelfMetadataBackfillState,
    shelfMetadataBackfillState,
    startShelfMetadataBackfill,
  }
}
