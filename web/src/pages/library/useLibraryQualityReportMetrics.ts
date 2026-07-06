import { useMemo } from 'react'
import type { LibraryFileItem, LibraryQualityOverviewResponse } from '../../api/library'
import {
  conversionQualityIssueEntries,
  conversionQualityScore,
  conversionQualityStatus,
  hasConversionQualityIssue,
  normalizeTextValue,
  type QualityRepairHistoryRecord,
} from './libraryPageUtils'

export type QualityIssueStat = {
  key: string
  label: string
  severity: string
  papers: number
  count: number
  repairStrategy?: string
}

export type QualityReportRecommendationView = {
  name: string
  score: number
  issues: string[]
}

type UseLibraryQualityReportMetricsParams = {
  files: LibraryFileItem[]
  backendQualityOverview: LibraryQualityOverviewResponse | null
  qualityRepairHistory: Record<string, QualityRepairHistoryRecord>
}

function severityWeight(severity: string) {
  if (severity === 'error') return 2
  if (severity === 'warning') return 1
  return 0
}

export function useLibraryQualityReportMetrics({
  files,
  backendQualityOverview,
  qualityRepairHistory,
}: UseLibraryQualityReportMetricsParams) {
  const qualityReportStats = useMemo(() => {
    const summary = backendQualityOverview?.summary
    if (summary) {
      return {
        assessed: Number(summary.assessed || 0),
        converted: Number(summary.converted || 0),
        review: Number(summary.review || 0),
        good: Number(summary.good || 0),
        unknown: Number(summary.unknown || 0),
        avgScore: Number(summary.avg_score || 0),
      }
    }

    const assessed = files.filter((item) => item.conversion_quality)
    const converted = files.filter((item) => item.category === 'converted')
    const convertedWithoutQuality = converted.filter((item) => !item.conversion_quality).length
    const scores = assessed
      .map((item) => conversionQualityScore(item.conversion_quality))
      .filter((score) => Number.isFinite(score) && score > 0)
    const avgScore = scores.length > 0
      ? Math.round(scores.reduce((acc, score) => acc + score, 0) / scores.length)
      : 0
    return {
      assessed: assessed.length,
      converted: converted.length,
      review: files.filter((item) => hasConversionQualityIssue(item)).length,
      good: files.filter((item) => conversionQualityStatus(item.conversion_quality) === 'good').length,
      unknown: convertedWithoutQuality,
      avgScore,
    }
  }, [backendQualityOverview, files])

  const fallbackQualityIssueStats = useMemo<QualityIssueStat[]>(() => {
    const stats = new Map<string, QualityIssueStat>()
    for (const item of files) {
      const seenInPaper = new Set<string>()
      for (const issue of item.conversion_quality?.issues || []) {
        const label = normalizeTextValue(issue.label || issue.code)
        const key = normalizeTextValue(issue.code || issue.label).toLowerCase()
        if (!key || !label) continue
        const existing = stats.get(key) || {
          key,
          label,
          severity: String(issue.severity || '').trim().toLowerCase(),
          papers: 0,
          count: 0,
        }
        existing.count += Math.max(1, Math.round(Number(issue.count || 0) || 1))
        if (!seenInPaper.has(key)) {
          existing.papers += 1
          seenInPaper.add(key)
        }
        if (String(issue.severity || '').trim().toLowerCase() === 'error') existing.severity = 'error'
        stats.set(key, existing)
      }
    }
    return Array.from(stats.values())
      .sort((a, b) => severityWeight(b.severity) - severityWeight(a.severity)
        || b.papers - a.papers
        || b.count - a.count
        || a.label.localeCompare(b.label, 'en'))
      .slice(0, 5)
  }, [files])

  const qualityIssueStats = useMemo<QualityIssueStat[]>(() => {
    const issues = Array.isArray(backendQualityOverview?.top_issues) ? backendQualityOverview.top_issues : []
    if (!issues.length) return fallbackQualityIssueStats
    return issues.slice(0, 5).map((issue) => ({
      key: normalizeTextValue(issue.code || issue.label).toLowerCase(),
      label: normalizeTextValue(issue.label || issue.code),
      severity: normalizeTextValue(issue.severity || 'warning').toLowerCase(),
      papers: Number(issue.papers || 0),
      count: Number(issue.count || 0),
      repairStrategy: normalizeTextValue(issue.repair_strategy),
    })).filter((issue) => Boolean(issue.key && issue.label))
  }, [backendQualityOverview, fallbackQualityIssueStats])

  const localQualityRepairRecommendedItems = useMemo(() => (
    files
      .filter((item) => item.task_state === 'idle' && hasConversionQualityIssue(item))
      .sort((a, b) => {
        const aHistory = qualityRepairHistory[a.name]
        const bHistory = qualityRepairHistory[b.name]
        const aRemaining = aHistory?.remainingIssues.length || 0
        const bRemaining = bHistory?.remainingIssues.length || 0
        const aScore = conversionQualityScore(a.conversion_quality)
        const bScore = conversionQualityScore(b.conversion_quality)
        return bRemaining - aRemaining
          || aScore - bScore
          || String(a.name || '').localeCompare(String(b.name || ''), 'en')
      })
      .slice(0, 5)
  ), [files, qualityRepairHistory])

  const qualityReportRecommendations = useMemo<QualityReportRecommendationView[]>(() => {
    const overviewItems = Array.isArray(backendQualityOverview?.recommended) ? backendQualityOverview.recommended : []
    if (overviewItems.length > 0) {
      return overviewItems.slice(0, 5)
        .map((item) => ({
          name: normalizeTextValue(item.name),
          score: Math.max(0, Math.min(100, Math.round(Number(item.score || 0)))),
          issues: (Array.isArray(item.issues) ? item.issues : [])
            .map((issue) => normalizeTextValue(issue.label || issue.code))
            .filter(Boolean)
            .slice(0, 2),
        }))
        .filter((item) => Boolean(item.name))
    }
    return localQualityRepairRecommendedItems.slice(0, 5).map((item) => ({
      name: item.name,
      score: conversionQualityScore(item.conversion_quality),
      issues: conversionQualityIssueEntries(item.conversion_quality).map((issue) => issue.label).slice(0, 2),
    }))
  }, [backendQualityOverview, localQualityRepairRecommendedItems])

  return {
    qualityReportStats,
    qualityIssueStats,
    qualityReportRecommendations,
  }
}
