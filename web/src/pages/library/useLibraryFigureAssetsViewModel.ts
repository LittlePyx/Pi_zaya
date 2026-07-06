import { useMemo } from 'react'
import type {
  LibraryFigureAssetScanItem,
  LibraryFigureAssetScanResponse,
} from '../../api/library'
import { normalizeTextValue } from './libraryPageUtils'

type FigureAssetTone = 'warning' | 'error' | 'good' | 'unknown'

type FigureAssetIssueStat = {
  name: string
  count: number
}

type UseLibraryFigureAssetsViewModelParams = {
  scan: LibraryFigureAssetScanResponse | null
  scanRunning: boolean
  refreshRunning: boolean
}

export function useLibraryFigureAssetsViewModel({
  scan,
  scanRunning,
  refreshRunning,
}: UseLibraryFigureAssetsViewModelParams) {
  const tone: FigureAssetTone = scanRunning || refreshRunning
    ? 'warning'
    : scan
      ? (
          normalizeTextValue(scan.status).toLowerCase() === 'error'
            ? 'error'
            : Number(scan.docs_with_issues || 0) > 0 || Number(scan.refresh_recommended || 0) > 0
              ? 'warning'
              : 'good'
        )
      : 'unknown'

  const issueStats = useMemo<FigureAssetIssueStat[]>(
    () => Object.entries(scan?.issue_counts || {})
      .map(([name, count]) => ({ name, count: Number(count || 0) }))
      .filter((item) => item.name && item.count > 0)
      .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name))
      .slice(0, 6),
    [scan],
  )

  const previewItems = useMemo<LibraryFigureAssetScanItem[]>(
    () => (Array.isArray(scan?.items) ? scan.items : [])
      .filter((item) => item && (Number(item.issue_count || 0) > 0 || Boolean(item.refresh_recommended)))
      .slice(0, 5),
    [scan],
  )

  const refreshableCount = Number(scan?.refresh_recommended || 0)

  return {
    tone,
    issueStats,
    previewItems,
    refreshableCount,
  }
}
