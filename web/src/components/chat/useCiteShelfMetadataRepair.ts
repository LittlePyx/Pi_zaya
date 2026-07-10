import { useCallback, useEffect, useRef, useState } from 'react'
import { message } from 'antd'
import type { ShelfMetadataRepairItem } from '../../api/references'
import { referencesApi } from '../../api/references'
import { useT } from '../../i18n'
import {
  citationDisplay,
  shelfItemNeedsMetadataRepair,
  strictRepairMerge,
  type CiteShelfItem,
} from './citationState'
import type { ShelfExportKind } from './citeShelfDisplay'

type ApplyRepairCandidates = (
  updates: Array<{ key: string; metas: Array<Record<string, unknown>> }>,
) => boolean

function repairPayloadsForExport(item: CiteShelfItem): Array<Record<string, unknown>> {
  const basePayload = item as unknown as Record<string, unknown>
  return [
    basePayload,
    {
      ...basePayload,
      raw: '',
      cite_fmt: '',
      citeFmt: '',
    },
  ]
}

function repairMetaFromEntry(entry: ShelfMetadataRepairItem): Record<string, unknown> {
  return {
    ...(entry.meta || {}),
    metadata_quality: entry.after || (entry.meta || {}).metadata_quality,
    metadata_export_acceptance: entry.export_acceptance || (entry.meta || {}).metadata_export_acceptance,
    metadata_repair_status: entry.repair_status,
    metadata_changed_fields: entry.changed_fields || [],
    metadata_repair_sources: entry.repair_sources || [],
  }
}

export function useCiteShelfMetadataRepair(onApplyRepairCandidates?: ApplyRepairCandidates) {
  const S = useT()
  const [exportRepairingKind, setExportRepairingKind] = useState<ShelfExportKind | ''>('')
  const exportRepairRunTokenRef = useRef(0)
  const exportRepairingKindRef = useRef<ShelfExportKind | ''>('')

  useEffect(() => () => {
    exportRepairRunTokenRef.current += 1
    exportRepairingKindRef.current = ''
  }, [])

  const repairMetadataBeforeExport = useCallback(async (
    kind: ShelfExportKind,
    exportItems: CiteShelfItem[],
  ): Promise<CiteShelfItem[] | null> => {
    const candidates = exportItems.filter((item) => shelfItemNeedsMetadataRepair(item, citationDisplay(item)))
    if (candidates.length <= 0) return exportItems

    const payloads = candidates.flatMap((item) => repairPayloadsForExport(item))
    const noticeKey = `cite-shelf-export-repair-${kind}`
    const repairToken = exportRepairRunTokenRef.current + 1
    exportRepairRunTokenRef.current = repairToken
    exportRepairingKindRef.current = kind
    const isCurrentExportRepair = () => exportRepairRunTokenRef.current === repairToken
    setExportRepairingKind(kind)
    message.loading({
      key: noticeKey,
      content: S.shelf_export_repairing.replace('{n}', String(candidates.length)),
      duration: 0,
    })
    try {
      const res = await referencesApi.repairShelfMetadata(payloads, payloads.length)
      if (!isCurrentExportRepair()) return null
      const metasByKey = new Map<string, Array<Record<string, unknown>>>()
      for (const entry of Array.isArray(res.items) ? res.items : []) {
        const meta = repairMetaFromEntry(entry)
        if (!meta || Object.keys(meta).length <= 0) continue
        const key = String(entry.key || meta.key || '').trim()
        if (!key) continue
        metasByKey.set(key, [...(metasByKey.get(key) || []), meta])
      }
      const updates = Array.from(metasByKey.entries()).map(([key, metas]) => ({ key, metas }))
      onApplyRepairCandidates?.(updates)

      let repairedReadyCount = 0
      let unresolvedCount = 0
      const repairedItems = exportItems.map((item) => {
        const wasReady = !shelfItemNeedsMetadataRepair(item, citationDisplay(item))
        const metas = metasByKey.get(item.key) || []
        let next = item
        for (const meta of metas) {
          const accepted = strictRepairMerge(next, meta)
          if (accepted) next = accepted
        }
        const isReady = !shelfItemNeedsMetadataRepair(next, citationDisplay(next))
        if (!wasReady && isReady) repairedReadyCount += 1
        if (!isReady) unresolvedCount += 1
        return next
      })

      if (repairedReadyCount > 0 && unresolvedCount <= 0) {
        message.success({
          key: noticeKey,
          content: S.shelf_export_repaired.replace('{n}', String(repairedReadyCount)),
          duration: 2,
        })
      } else if (repairedReadyCount > 0) {
        message.warning({
          key: noticeKey,
          content: S.shelf_export_repaired_partial
            .replace('{n}', String(repairedReadyCount))
            .replace('{m}', String(unresolvedCount)),
          duration: 3,
        })
      } else {
        message.warning({
          key: noticeKey,
          content: S.shelf_export_repair_no_change,
          duration: 3,
        })
      }
      return repairedItems
    } catch {
      if (isCurrentExportRepair()) {
        message.warning({
          key: noticeKey,
          content: S.shelf_export_repair_failed,
          duration: 3,
        })
      }
      return isCurrentExportRepair() ? exportItems : null
    } finally {
      if (isCurrentExportRepair()) {
        exportRepairingKindRef.current = ''
        setExportRepairingKind((current) => (current === kind ? '' : current))
      }
    }
  }, [S, onApplyRepairCandidates])

  return {
    exportRepairingKind,
    exportRepairingKindRef,
    repairMetadataBeforeExport,
  }
}
