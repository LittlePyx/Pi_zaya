import { useMemo } from 'react'
import { normalizeTextValue } from './libraryPageUtils'

type QualityCenterReportStats = {
  assessed: number
  converted: number
  review: number
  good: number
  unknown: number
}

type QualityCenterSourceReadinessStats = {
  blocked: number
}

type QualityCenterDomainView = {
  available: boolean
  status: string
}

type UseLibraryQualityCenterViewModelParams = {
  S: Record<string, string>
  reportStats: QualityCenterReportStats
  sourceReadinessStats: QualityCenterSourceReadinessStats
  domains: ReadonlyArray<QualityCenterDomainView>
  failureCount: number
  metadataRemaining: number
  priorityActionCount: number
  recommendedRepairCount: number
  batchRunning: boolean
  repairAdvancing: boolean
  metadataBackfillRunning: boolean
  repairingNames: Record<string, boolean>
}

export function useLibraryQualityCenterViewModel({
  S,
  reportStats,
  sourceReadinessStats,
  domains,
  failureCount,
  metadataRemaining,
  priorityActionCount,
  recommendedRepairCount,
  batchRunning,
  repairAdvancing,
  metadataBackfillRunning,
  repairingNames,
}: UseLibraryQualityCenterViewModelParams) {
  return useMemo(() => {
    const busy = batchRunning
      || repairAdvancing
      || metadataBackfillRunning
      || Object.values(repairingNames).some(Boolean)
    const domainProblems = domains.filter((domain) => (
      domain.available
      && !['good', 'unknown'].includes(normalizeTextValue(domain.status).toLowerCase())
    ))
    const problemCount = reportStats.review
      + sourceReadinessStats.blocked
      + failureCount
      + domainProblems.length
    const tone = busy
      ? 'processing'
      : (sourceReadinessStats.blocked > 0
        || reportStats.review > 0
        || failureCount > 0
        || domainProblems.some((domain) => normalizeTextValue(domain.status).toLowerCase() === 'error'))
          ? 'error'
          : (reportStats.unknown > 0 || metadataRemaining > 0 || priorityActionCount > 0)
            ? 'warning'
            : 'good'
    const statusLabel = tone === 'processing'
      ? S.lib_quality_center_status_processing
      : tone === 'good'
        ? S.lib_quality_center_status_ready
        : tone === 'error'
          ? S.lib_quality_center_status_repair
          : S.lib_quality_center_status_attention
    const summary = tone === 'good'
      ? S.lib_quality_center_summary_ready
        .replace('{ready}', String(reportStats.good))
        .replace('{total}', String(reportStats.assessed || reportStats.converted))
      : tone === 'processing'
        ? S.lib_quality_center_summary_running
        : S.lib_quality_center_summary_review
          .replace('{review}', String(reportStats.review))
          .replace('{blocked}', String(sourceReadinessStats.blocked))
          .replace('{cases}', String(failureCount))
          .replace('{domains}', String(domainProblems.length))
    const nextAction = tone === 'good'
      ? S.lib_quality_center_action_none
      : tone === 'processing'
        ? S.lib_quality_center_action_monitor
        : recommendedRepairCount > 0
          ? S.lib_quality_center_action_repair.replace('{n}', String(recommendedRepairCount))
          : metadataRemaining > 0
            ? S.lib_quality_center_action_metadata.replace('{n}', String(metadataRemaining))
            : reportStats.review > 0
              ? S.lib_quality_center_action_review
              : S.lib_quality_center_action_open

    return {
      busy,
      domainProblemCount: domainProblems.length,
      metadataRemaining,
      problemCount,
      tone,
      statusLabel,
      summary,
      nextAction,
      signals: [
        { key: 'usable', label: S.lib_quality_center_signal_usable, value: `${reportStats.good}/${reportStats.assessed || reportStats.converted || 0}` },
        { key: 'risk', label: S.lib_quality_center_signal_attention, value: String(problemCount) },
        { key: 'locate', label: S.lib_quality_center_signal_locate, value: String(sourceReadinessStats.blocked) },
        { key: 'metadata', label: S.lib_quality_center_signal_metadata, value: String(metadataRemaining) },
      ],
    }
  }, [
    S,
    batchRunning,
    domains,
    failureCount,
    metadataBackfillRunning,
    metadataRemaining,
    priorityActionCount,
    recommendedRepairCount,
    repairAdvancing,
    repairingNames,
    reportStats,
    sourceReadinessStats,
  ])
}
