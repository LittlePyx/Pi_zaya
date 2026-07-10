import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { message } from 'antd'
import { libraryApi } from '../../api/library'
import { useT } from '../../i18n'
import { sourceListKey, type SourceQualityByPath } from './citeShelfDisplay'

export interface CiteShelfSourceRef {
  source_path: string
  source_name: string
}

interface Options {
  open: boolean
  showDiagnostics: boolean
  refreshToken: number
  sources: CiteShelfSourceRef[]
}

interface RepairOptions {
  silent?: boolean
  repairKey?: string
}

interface Result {
  sourceQualityByPath: SourceQualityByPath
  sourceRepairingKey: string
  repairSources: (sources: CiteShelfSourceRef[], options?: RepairOptions) => Promise<void>
}

export function useCiteShelfSourceQuality({
  open,
  showDiagnostics,
  refreshToken,
  sources,
}: Options): Result {
  const S = useT()
  const [sourceQualityByPath, setSourceQualityByPath] = useState<SourceQualityByPath>({})
  const [sourceRepairingKey, setSourceRepairingKey] = useState('')
  const sourceRepairStreamRef = useRef<AbortController | null>(null)
  const sourceRepairRunTokenRef = useRef(0)
  const sourceQualityKey = useMemo(
    () => sources.map((item) => `${item.source_path}\t${item.source_name}`).join('\n'),
    [sources],
  )

  useEffect(() => {
    if (!showDiagnostics) {
      setSourceQualityByPath({})
      return
    }
    if (!open || sources.length <= 0) return
    let cancelled = false
    libraryApi.sourceQuality(sources)
      .then((res) => {
        if (cancelled) return
        const next: SourceQualityByPath = {}
        for (const item of Array.isArray(res.items) ? res.items : []) {
          const sourcePath = String(item.source_path || '').trim()
          if (!sourcePath) continue
          next[sourcePath] = item
        }
        setSourceQualityByPath((prev) => ({ ...prev, ...next }))
      })
      .catch(() => {
        if (!cancelled) setSourceQualityByPath((prev) => ({ ...prev }))
      })
    return () => {
      cancelled = true
    }
  }, [open, refreshToken, showDiagnostics, sourceQualityKey, sources])

  useEffect(() => () => {
    sourceRepairRunTokenRef.current += 1
    sourceRepairStreamRef.current?.abort()
    sourceRepairStreamRef.current = null
  }, [])

  const repairSources = useCallback(async (
    repairTargets: CiteShelfSourceRef[],
    options: RepairOptions = {},
  ) => {
    const silent = Boolean(options.silent)
    if (!showDiagnostics) return
    if (repairTargets.length <= 0) {
      if (!silent) message.info(S.shelf_source_quality_repair_none)
      return
    }
    const repairKey = options.repairKey || sourceListKey(repairTargets)
    const repairSourcesSnapshot = repairTargets.map((item) => ({ ...item }))
    const repairToken = sourceRepairRunTokenRef.current + 1
    sourceRepairRunTokenRef.current = repairToken
    const isCurrentRepair = () => sourceRepairRunTokenRef.current === repairToken
    const refreshRepairSources = async () => {
      if (!isCurrentRepair() || repairSourcesSnapshot.length <= 0) return
      const res = await libraryApi.sourceQuality(repairSourcesSnapshot)
      if (!isCurrentRepair()) return
      const next: SourceQualityByPath = {}
      for (const item of Array.isArray(res.items) ? res.items : []) {
        const sourcePath = String(item.source_path || '').trim()
        if (!sourcePath) continue
        next[sourcePath] = item
      }
      setSourceQualityByPath((prev) => ({ ...prev, ...next }))
    }
    const refreshRepairRunAndSources = async (runId: string, needsReindex: boolean) => {
      if (!isCurrentRepair()) return
      if (needsReindex) {
        let advanced = false
        if (runId) {
          try {
            await libraryApi.advanceQualityRepairRun(runId)
            advanced = true
          } catch {
            advanced = false
          }
        }
        try {
          if (!isCurrentRepair()) return
          if (!advanced) await libraryApi.reindex()
        } catch {
          // Source quality will still be refreshed so the UI can show the latest diagnostics.
        }
      }
      if (!isCurrentRepair()) return
      await refreshRepairSources()
    }
    const clearRepairing = () => {
      if (!isCurrentRepair()) return
      setSourceRepairingKey((current) => (current === repairKey ? '' : current))
    }
    setSourceRepairingKey(repairKey)
    let watchingConversion = false
    try {
      const res = await libraryApi.repairQuality({
        sources: repairSourcesSnapshot,
        speed_mode: 'balanced',
        replace: true,
      })
      if (!isCurrentRepair()) return
      const runId = String(res.repair_run_id || res.repair_run?.run_id || '').trim()
      const queued = Number(res.enqueued || 0)
      const repaired = Number(res.repaired || 0)
      const needsReindex = Boolean(res.needs_reindex || res.impact?.needs_reindex)
      if (queued > 0) {
        if (!silent) message.success(S.shelf_source_quality_repair_queued.replace('{n}', String(queued)))
        watchingConversion = true
        sourceRepairStreamRef.current?.abort()
        let streamCtrl: AbortController | null = null
        const clearStreamIfCurrent = () => {
          if (!isCurrentRepair() || sourceRepairStreamRef.current !== streamCtrl) return false
          sourceRepairStreamRef.current = null
          return true
        }
        streamCtrl = libraryApi.streamConvertStatus(
          () => {},
          () => {
            if (!clearStreamIfCurrent()) return
            void refreshRepairRunAndSources(runId, needsReindex).finally(clearRepairing)
          },
          () => {
            if (!clearStreamIfCurrent()) return
            void refreshRepairRunAndSources(runId, needsReindex).finally(clearRepairing)
          },
        )
        sourceRepairStreamRef.current = streamCtrl
      } else if (repaired > 0) {
        if (!silent) message.success(`Markdown repaired: ${repaired}`)
        await refreshRepairRunAndSources(runId, needsReindex)
      } else if (needsReindex) {
        await refreshRepairRunAndSources(runId, needsReindex)
      } else {
        if (!silent) message.info(S.shelf_source_quality_repair_none)
        await refreshRepairSources()
      }
    } catch (err) {
      if (isCurrentRepair() && !silent) message.error(err instanceof Error ? err.message : S.shelf_source_quality_repair_fail)
    } finally {
      if (!watchingConversion && isCurrentRepair()) clearRepairing()
    }
  }, [S, showDiagnostics])

  return { sourceQualityByPath, sourceRepairingKey, repairSources }
}
