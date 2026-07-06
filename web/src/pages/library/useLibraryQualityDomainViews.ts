import { useMemo } from 'react'
import type { LibraryQualityOverviewResponse } from '../../api/library'
import {
  qualityDomainNumber,
  qualityDomainStatus,
  qualityStatusText,
  qualityTopFailureText,
} from './libraryPageUtils'

type QualityReportStats = {
  review: number
  good: number
  unknown: number
  avgScore: number
}

export type QualityDomainView = {
  key: 'conversion' | 'research_qa' | 'citation_cards' | 'reader_locate'
  label: string
  available: boolean
  status: string
  statusLabel: string
  countText: string
  detailText: string
  failureText: string
}

type UseLibraryQualityDomainViewsParams = {
  S: Record<string, string>
  backendQualityOverview: LibraryQualityOverviewResponse | null
  qualityReportStats: QualityReportStats
}

export function useLibraryQualityDomainViews({
  S,
  backendQualityOverview,
  qualityReportStats,
}: UseLibraryQualityDomainViewsParams) {
  return useMemo<QualityDomainView[]>(() => {
    const domains = backendQualityOverview?.domains || {}
    const conversion = domains.conversion
    const researchQa = domains.research_qa
    const citationCards = domains.citation_cards
    const readerLocate = domains.reader_locate

    const conversionStatus = qualityDomainStatus(conversion, backendQualityOverview?.status || 'unknown')
    const conversionReview = conversion ? qualityDomainNumber(conversion, 'review') : qualityReportStats.review
    const conversionGood = conversion ? qualityDomainNumber(conversion, 'good') : qualityReportStats.good
    const conversionAvg = conversion ? qualityDomainNumber(conversion, 'avg_score') : qualityReportStats.avgScore
    const conversionUnknown = conversion ? qualityDomainNumber(conversion, 'unknown') : qualityReportStats.unknown

    const qaStatus = qualityDomainStatus(researchQa)
    const qaAvailable = researchQa?.available !== false && Boolean(researchQa)
    const qaTotal = qualityDomainNumber(researchQa, 'total')
    const qaPassed = qualityDomainNumber(researchQa, 'passed')
    const qaFailed = qualityDomainNumber(researchQa, 'failed')

    const cardStatus = qualityDomainStatus(citationCards)
    const cardsAvailable = citationCards?.available !== false && Boolean(citationCards)
    const trackedChecks = qualityDomainNumber(citationCards, 'tracked_checks')
    const failedChecks = qualityDomainNumber(citationCards, 'failed_checks')
    const shelfItems = qualityDomainNumber(citationCards, 'shelf_item_count')
    const shelfExportReady = qualityDomainNumber(citationCards, 'shelf_export_ready_count')
    const shelfSummaryExportReady = qualityDomainNumber(citationCards, 'shelf_summary_export_ready_count')
    const shelfExportDetail = shelfItems > 0 ? `; shelf export ${shelfExportReady}/${shelfItems}; summaries ${shelfSummaryExportReady}/${shelfItems}` : ''
    const readerStatus = qualityDomainStatus(readerLocate)
    const readerAvailable = readerLocate?.available !== false && Boolean(readerLocate)
    const readerTotal = qualityDomainNumber(readerLocate, 'total')
    const readerFailed = qualityDomainNumber(readerLocate, 'failed')
    const readerDegraded = qualityDomainNumber(readerLocate, 'degraded')
    const readerRepairable = qualityDomainNumber(readerLocate, 'repairable')

    return [
      {
        key: 'conversion',
        label: S.lib_quality_domain_conversion,
        available: true,
        status: conversionStatus,
        statusLabel: qualityStatusText(conversionStatus, S),
        countText: conversionReview > 0
          ? `${conversionReview} ${S.lib_quality_domain_failed}`
          : `${conversionGood} ${S.lib_quality_domain_passed}`,
        detailText: `Q${Math.round(conversionAvg)} · ${conversionUnknown} ${S.lib_quality_report_unknown}`,
        failureText: qualityTopFailureText(conversion),
      },
      {
        key: 'research_qa',
        label: S.lib_quality_domain_research_qa,
        available: qaAvailable,
        status: qaAvailable ? qaStatus : 'unknown',
        statusLabel: qaAvailable ? qualityStatusText(qaStatus, S) : S.lib_quality_domain_unavailable,
        countText: qaAvailable
          ? (qaFailed > 0 ? `${qaFailed} ${S.lib_quality_domain_failed}` : `${qaPassed}/${qaTotal} ${S.lib_quality_domain_passed}`)
          : S.lib_quality_domain_unavailable,
        detailText: qaAvailable ? S.lib_quality_domain_cases.replace('{n}', String(qaTotal)) : '',
        failureText: qualityTopFailureText(researchQa),
      },
      {
        key: 'citation_cards',
        label: S.lib_quality_domain_citation_cards,
        available: cardsAvailable,
        status: cardsAvailable ? cardStatus : 'unknown',
        statusLabel: cardsAvailable ? qualityStatusText(cardStatus, S) : S.lib_quality_domain_unavailable,
        countText: cardsAvailable
          ? (failedChecks > 0 ? `${failedChecks} ${S.lib_quality_domain_failed}` : `${trackedChecks} ${S.lib_quality_domain_passed}`)
          : S.lib_quality_domain_unavailable,
        detailText: cardsAvailable
          ? `${S.lib_quality_domain_checks.replace('{n}', String(trackedChecks))}${shelfExportDetail}`
          : '',
        failureText: qualityTopFailureText(citationCards),
      },
      {
        key: 'reader_locate',
        label: 'Reader locate',
        available: readerAvailable,
        status: readerAvailable ? readerStatus : 'unknown',
        statusLabel: readerAvailable ? qualityStatusText(readerStatus, S) : S.lib_quality_domain_unavailable,
        countText: readerAvailable
          ? (readerFailed > 0 || readerDegraded > 0 ? `${readerFailed + readerDegraded} need repair` : `${readerTotal} verified`)
          : S.lib_quality_domain_unavailable,
        detailText: readerAvailable ? `${readerRepairable} repairable source signals` : '',
        failureText: qualityTopFailureText(readerLocate),
      },
    ]
  }, [S, backendQualityOverview, qualityReportStats])
}
