import { useMemo } from 'react'
import type {
  LibraryQualityFailureCase,
  LibraryQualityOverviewResponse,
} from '../../api/library'
import { normalizeTextValue } from './libraryPageUtils'

export type QualityFailureFilter = {
  name: string
  count: number
}

type UseLibraryQualityFailureCasesParams = {
  backendQualityOverview: LibraryQualityOverviewResponse | null
  qualityFailureFilter: string
}

export function useLibraryQualityFailureCases({
  backendQualityOverview,
  qualityFailureFilter,
}: UseLibraryQualityFailureCasesParams) {
  const qualityFailureCases = useMemo<LibraryQualityFailureCase[]>(
    () => (Array.isArray(backendQualityOverview?.failure_cases) ? backendQualityOverview.failure_cases : [])
      .filter((item) => item && normalizeTextValue(item.id))
      .slice(0, 12),
    [backendQualityOverview],
  )

  const qualityFailureFilters = useMemo<QualityFailureFilter[]>(() => {
    const stats = new Map<string, number>()
    for (const item of qualityFailureCases) {
      for (const failure of item.failures || []) {
        const name = normalizeTextValue(failure.name)
        if (!name) continue
        stats.set(name, (stats.get(name) || 0) + 1)
      }
    }
    return Array.from(stats.entries())
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0], 'en'))
      .slice(0, 6)
      .map(([name, count]) => ({ name, count }))
  }, [qualityFailureCases])

  const visibleQualityFailureCases = useMemo(() => {
    const filter = normalizeTextValue(qualityFailureFilter)
    if (!filter) return qualityFailureCases
    return qualityFailureCases.filter((item) => (
      (item.failures || []).some((failure) => normalizeTextValue(failure.name) === filter)
    ))
  }, [qualityFailureCases, qualityFailureFilter])

  return {
    qualityFailureCases,
    qualityFailureFilters,
    visibleQualityFailureCases,
  }
}
