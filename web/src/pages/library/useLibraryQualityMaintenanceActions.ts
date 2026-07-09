import { useCallback, useState } from 'react'
import type { Dispatch, SetStateAction } from 'react'
import { message } from 'antd'
import {
  libraryApi,
  type LibraryConversionQualityBatchResponse,
  type LibraryFigureAssetRefreshResponse,
  type LibraryFigureAssetScanResponse,
  type LibraryReindexResponse,
} from '../../api/library'
import type { LibraryQualityOperationToken } from './useLibraryQualityOperationGuard'

export type LibraryFigureAssetRefreshSource = {
  source_path: string
  source_name?: string
}

type LibraryFigureAssetRefreshScanItem = Pick<
  LibraryFigureAssetScanResponse['items'][number],
  'md_path' | 'pdf_name' | 'refresh_recommended' | 'source_name'
>

export function buildLibraryReindexFailureDetail(
  res: Pick<LibraryReindexResponse, 'structured_indices_error' | 'stderr' | 'refsync_error'> | null | undefined,
) {
  return [
    res?.structured_indices_error,
    res?.stderr,
    res?.refsync_error,
  ].map((item) => String(item || '').trim()).find(Boolean) || ''
}

export function buildLibraryFigureAssetRefreshSources(
  scan: { items?: LibraryFigureAssetRefreshScanItem[] } | null | undefined,
): LibraryFigureAssetRefreshSource[] {
  return (scan?.items || [])
    .filter((item) => Boolean(item.refresh_recommended))
    .map((item) => {
      const sourcePath = String(item.md_path || '').trim()
      const sourceName = String(item.source_name || item.pdf_name || '').trim()
      if (!sourcePath && !sourceName) return null
      return {
        source_path: sourcePath,
        ...(sourceName ? { source_name: sourceName } : {}),
      }
    })
    .filter((item): item is LibraryFigureAssetRefreshSource => Boolean(item))
}

type LibraryQualityMaintenanceActionsInput = {
  S: Record<string, string>
  scope: string
  speedMode: string
  beginQualityOperation: (key: string) => LibraryQualityOperationToken
  clearQualityOperation: (token?: LibraryQualityOperationToken | null) => void
  loadFiles: (scope: string) => Promise<unknown> | unknown
  loadQualityOverview: (scope: string) => Promise<unknown> | unknown
  qualityOperationIsActive: (token?: LibraryQualityOperationToken | null) => boolean
  qualityOperationIsCurrent: (token?: LibraryQualityOperationToken | null) => boolean
  reindex: () => Promise<LibraryReindexResponse>
  setQualityBatchRunning: Dispatch<SetStateAction<boolean>>
  startProgressStream: () => void
}

export function useLibraryQualityMaintenanceActions({
  S,
  scope,
  speedMode,
  beginQualityOperation,
  clearQualityOperation,
  loadFiles,
  loadQualityOverview,
  qualityOperationIsActive,
  qualityOperationIsCurrent,
  reindex,
  setQualityBatchRunning,
  startProgressStream,
}: LibraryQualityMaintenanceActionsInput) {
  const [qualityBatchResult, setQualityBatchResult] = useState<LibraryConversionQualityBatchResponse | null>(null)
  const [figureAssetScan, setFigureAssetScan] = useState<LibraryFigureAssetScanResponse | null>(null)
  const [figureAssetScanRunning, setFigureAssetScanRunning] = useState(false)
  const [figureAssetRefreshResult, setFigureAssetRefreshResult] = useState<LibraryFigureAssetRefreshResponse | null>(null)
  const [figureAssetRefreshRunning, setFigureAssetRefreshRunning] = useState(false)

  const handleReindex = useCallback(async (operationToken?: LibraryQualityOperationToken): Promise<boolean> => {
    const token = operationToken || beginQualityOperation('reindex')
    const ownsOperation = !operationToken
    const hide = message.loading(S.lib_msg_updating_kb, 0)
    try {
      const res = await reindex()
      hide()
      if (!qualityOperationIsCurrent(token)) return false
      if (!res.ok) {
        const detail = buildLibraryReindexFailureDetail(res)
        message.error(detail ? `${S.lib_msg_exec_fail}: ${detail}` : S.lib_msg_exec_fail)
        return false
      }
      message.success(S.lib_msg_exec_done)
      if (res.refsync_error) {
        message.warning(S.lib_msg_refsync_fail_detail.replace('{error}', String(res.refsync_error)))
      } else if (res.refsync?.started) {
        message.info(S.lib_msg_refsync_started_bg)
      }
      return true
    } catch (err) {
      hide()
      if (qualityOperationIsCurrent(token)) {
        message.error(err instanceof Error ? err.message : S.lib_msg_exec_fail)
      }
      return false
    } finally {
      if (ownsOperation) clearQualityOperation(token)
    }
  }, [
    S.lib_msg_exec_done,
    S.lib_msg_exec_fail,
    S.lib_msg_refsync_fail_detail,
    S.lib_msg_refsync_started_bg,
    S.lib_msg_updating_kb,
    beginQualityOperation,
    clearQualityOperation,
    qualityOperationIsCurrent,
    reindex,
  ])

  const runConversionQualityBatch = useCallback(async (repair: boolean) => {
    const operationToken = beginQualityOperation(repair ? 'quality-batch-repair' : 'quality-batch-scan')
    const hide = message.loading(repair ? 'Repairing conversion quality...' : 'Scanning conversion source quality...', 0)
    setQualityBatchRunning(true)
    try {
      const res = await libraryApi.conversionQualityBatch({
        repair,
        rebuild_indices: true,
        limit: 1000,
      })
      hide()
      if (!qualityOperationIsCurrent(operationToken)) return
      setQualityBatchResult(res)
      if (!res.ok) {
        message.error(S.lib_msg_exec_fail)
        return
      }
      if (repair && res.needs_reindex) {
        const reindexed = await handleReindex(operationToken)
        if (!qualityOperationIsCurrent(operationToken)) return
        if (!reindexed) {
          message.warning('Conversion repair finished, but index refresh needs retry.')
        } else {
          message.success(`Safe repair finished: ${res.changed} changed, index refreshed`)
        }
      } else {
        message.success(repair
          ? `Safe repair finished: ${res.changed} changed, ${res.ready} ready`
          : `Source scan finished: ${res.scanned} checked, ${res.ready} ready`)
      }
      await loadFiles(scope)
      if (!qualityOperationIsCurrent(operationToken)) return
      await loadQualityOverview('all')
    } catch (err) {
      hide()
      if (qualityOperationIsCurrent(operationToken)) {
        message.error(err instanceof Error ? err.message : S.lib_msg_exec_fail)
      }
    } finally {
      if (qualityOperationIsActive(operationToken)) setQualityBatchRunning(false)
      clearQualityOperation(operationToken)
    }
  }, [
    S.lib_msg_exec_fail,
    beginQualityOperation,
    clearQualityOperation,
    handleReindex,
    loadFiles,
    loadQualityOverview,
    qualityOperationIsActive,
    qualityOperationIsCurrent,
    scope,
    setQualityBatchRunning,
  ])

  const runFigureAssetQualityScan = useCallback(async (includeAll = false) => {
    setFigureAssetScanRunning(true)
    try {
      const res = await libraryApi.figureAssetQualityScan({
        limit: 1000,
        include_all: includeAll,
      })
      setFigureAssetScan(res)
      setFigureAssetRefreshResult(null)
      if (Number(res.refresh_recommended || 0) > 0) {
        message.warning(`Figure asset scan found ${res.refresh_recommended} sources to refresh`)
      } else {
        message.success(`Figure asset scan checked ${res.scanned} sources`)
      }
      return res
    } catch (err) {
      message.error(err instanceof Error ? err.message : 'Figure asset scan failed')
      return null
    } finally {
      setFigureAssetScanRunning(false)
    }
  }, [])

  const refreshFigureAssets = useCallback(async () => {
    if (!figureAssetScan) {
      message.info('Run a figure asset scan before refreshing flagged sources')
      return null
    }
    const sources = buildLibraryFigureAssetRefreshSources(figureAssetScan)
    if (sources.length <= 0) {
      message.info('No figure assets need refresh right now')
      return null
    }
    setFigureAssetRefreshRunning(true)
    try {
      const res = await libraryApi.refreshFigureAssets({
        sources,
        limit: Math.max(1, sources.length),
        speed_mode: speedMode,
        replace: true,
        target_dpi: figureAssetScan?.target_dpi,
      })
      setFigureAssetRefreshResult(res)
      if (Number(res.enqueued || 0) > 0) {
        message.success(`Figure asset refresh queued: ${res.enqueued}`)
        startProgressStream()
      } else if (Number(res.skipped_busy || 0) > 0) {
        message.warning(`Figure asset refresh skipped busy sources: ${res.skipped_busy}`)
      } else if (Number(res.failed || 0) > 0) {
        message.error(`Figure asset refresh failed: ${res.failed}`)
      } else {
        message.info('No figure assets need refresh right now')
      }
      await loadFiles(scope)
      await loadQualityOverview('all')
      return res
    } catch (err) {
      message.error(err instanceof Error ? err.message : 'Figure asset refresh failed')
      return null
    } finally {
      setFigureAssetRefreshRunning(false)
    }
  }, [
    figureAssetScan,
    loadFiles,
    loadQualityOverview,
    scope,
    speedMode,
    startProgressStream,
  ])

  return {
    figureAssetRefreshResult,
    figureAssetRefreshRunning,
    figureAssetScan,
    figureAssetScanRunning,
    handleReindex,
    qualityBatchResult,
    refreshFigureAssets,
    runConversionQualityBatch,
    runFigureAssetQualityScan,
  }
}
