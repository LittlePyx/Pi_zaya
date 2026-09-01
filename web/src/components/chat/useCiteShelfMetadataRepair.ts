import { useCallback, useEffect, useRef, useState } from 'react'
import { message } from 'antd'
import type { ShelfMetadataRepairItem } from '../../api/references'
import { referencesApi } from '../../api/references'
import { useT } from '../../i18n'
import {
  shelfItemNeedsExportMetadataPass,
  shelfItemRepairFingerprint,
  strictRepairMerge,
  type CiteShelfItem,
} from './citationState'
import type { ShelfExportKind } from './citeShelfDisplay'

export type CiteShelfRepairedItemUpdate = {
  item: CiteShelfItem
  requestedFingerprint: string
}

type ApplyRepairedItems = (updates: CiteShelfRepairedItemUpdate[]) => void

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

export function useCiteShelfMetadataRepair(onApplyRepairedItems?: ApplyRepairedItems) {
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
    const candidates = exportItems.filter(shelfItemNeedsExportMetadataPass)
    if (candidates.length <= 0) return exportItems

    const requestedFingerprints = new Map(candidates.map(item => [
      item.key,
      shelfItemRepairFingerprint(item),
    ]))
    const payloads = candidates as unknown as Array<Record<string, unknown>>
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
      const repairedItems = exportItems.map((item) => {
        const metas = metasByKey.get(item.key) || []
        let next = item
        for (const meta of metas) {
          const accepted = strictRepairMerge(next, meta)
          if (accepted) next = accepted
        }
        return next
      })
      onApplyRepairedItems?.(repairedItems.flatMap((item) => {
        const requestedFingerprint = requestedFingerprints.get(item.key)
        return requestedFingerprint ? [{ item, requestedFingerprint }] : []
      }))
      return repairedItems
    } catch {
      return isCurrentExportRepair() ? exportItems : null
    } finally {
      if (isCurrentExportRepair()) {
        message.destroy(noticeKey)
        exportRepairingKindRef.current = ''
        setExportRepairingKind((current) => (current === kind ? '' : current))
      }
    }
  }, [S, onApplyRepairedItems])

  return {
    exportRepairingKind,
    exportRepairingKindRef,
    repairMetadataBeforeExport,
  }
}
