import { useMemo } from 'react'
import type {
  ShelfMetadataBackfillJobState,
  ShelfMetadataBackfillResponse,
  ShelfMetadataBackfillScanResponse,
} from '../../api/references'
import { normalizeTextValue } from './libraryPageUtils'

type ShelfMetadataBackfillTone = 'warning' | 'error' | 'good' | 'unknown'

type UseShelfMetadataBackfillViewModelParams = {
  shelfMetadataBackfillState: ShelfMetadataBackfillJobState | null
}

export function useShelfMetadataBackfillViewModel({
  shelfMetadataBackfillState,
}: UseShelfMetadataBackfillViewModelParams) {
  const shelfMetadataBackfillScan = useMemo<ShelfMetadataBackfillScanResponse | null>(() => {
    const state = shelfMetadataBackfillState
    return state?.after_scan || state?.result?.after_scan || state?.scan || state?.result?.scan || null
  }, [shelfMetadataBackfillState])

  const shelfMetadataBackfillResult: ShelfMetadataBackfillResponse | null = shelfMetadataBackfillState?.result || null
  const shelfMetadataBackfillProgress = Math.max(
    0,
    Math.min(100, Math.round(Number(shelfMetadataBackfillState?.progress?.percent || 0))),
  )
  const shelfMetadataBackfillPhase = normalizeTextValue(
    shelfMetadataBackfillState?.phase || shelfMetadataBackfillState?.status || 'idle',
  ).replace(/_/g, ' ')
  const shelfMetadataBackfillRunning = Boolean(shelfMetadataBackfillState?.running)
  const shelfMetadataBackfillTone: ShelfMetadataBackfillTone = shelfMetadataBackfillRunning
    ? 'warning'
    : normalizeTextValue(shelfMetadataBackfillState?.status).toLowerCase() === 'error'
      ? 'error'
      : shelfMetadataBackfillScan
        ? (Number(shelfMetadataBackfillScan.needs_repair || 0) > 0 ? 'warning' : 'good')
        : 'unknown'

  return {
    shelfMetadataBackfillScan,
    shelfMetadataBackfillResult,
    shelfMetadataBackfillProgress,
    shelfMetadataBackfillPhase,
    shelfMetadataBackfillRunning,
    shelfMetadataBackfillTone,
  }
}
