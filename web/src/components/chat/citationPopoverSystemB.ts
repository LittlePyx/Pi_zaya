import type { CiteDetail, CitationCardViewSection } from './citationState'
import { cleanCitationDisplayText } from './citationState'
import {
  SYSTEM_B_TRACE_ENABLED,
  compact,
} from './citationPopoverUtils'
import { buildSystemBSourcePanelsModel } from './citationPopoverSystemBSourcePanels'
import { buildSystemBTextPanelsModel } from './citationPopoverSystemBTextPanels'

interface SystemBStrings extends Record<string, string> {
  cite_context: string
  cite_context_summary: string
  cite_current_paper_usage: string
  cite_evidence_chain: string
  cite_loading: string
  cite_loading_summary: string
  cite_location_current: string
  cite_note: string
  cite_original_reference_entry: string
  cite_paper_overview: string
  cite_reference_entry: string
  cite_summary_unavailable: string
  cite_system_b_support_default: string
  cite_trace_complete: string
  cite_trace_review: string
  cite_upstream_reference: string
  cite_upstream_role: string
}

interface BuildSystemBLiteratureCardModelOptions {
  detail: CiteDetail
  S: SystemBStrings
  isSystemB: boolean
  loading: boolean
  locatorSection?: CitationCardViewSection
  contextSummarySection?: CitationCardViewSection
  referenceSection?: CitationCardViewSection
  cardTakeaway: string
  cardEvidenceLabel: string
  cardReferenceLabel: string
  cardSupportLabel: string
  cardQualityFlags: string[]
  sourcePaperText: string
  headingPath: string
  pageLabel: string
  badgeLabel: string
  doiLabel: string
  systemBTitle: string
  systemBTitleMissing: boolean
  headerSubtitle: string
  metrics: string[]
  explicitSupportText: string
  displaySource: string
  localizeKnownBody: (value: string) => string
  localizeKnownLabel: (value: string) => string
}

export interface SystemBLiteratureCardModel {
  showTrace: boolean
  traceStatus: { label: string; tone: string }
  traceScore: number
  traceSteps: string[]
  traceReason: string
  traceLabel: string
  paperOverviewText: string
  paperOverviewLabel: string
  paperOverviewPreview: string
  showOverviewLoading: boolean
  overviewLoadingLabel: string
  showOverviewUnavailable: boolean
  overviewUnavailableLabel: string
  takeawayText: string
  takeawayLabel: string
  showLocation: boolean
  locationLabel: string
  locationText: string
  locationHint: string
  contextSummaryText: string
  contextSummaryLabel: string
  citationContextText: string
  citationContextPreview: string
  citationContextLabel: string
  showReference: boolean
  referenceLabel: string
  referencePreview: string
  showSupport: boolean
  supportLabel: string
  supportText: string
}

export function buildSystemBLiteratureCardModel({
  detail,
  S,
  isSystemB,
  loading,
  locatorSection,
  contextSummarySection,
  referenceSection,
  cardTakeaway,
  cardEvidenceLabel,
  cardReferenceLabel,
  cardSupportLabel,
  cardQualityFlags,
  sourcePaperText,
  headingPath,
  pageLabel,
  badgeLabel,
  doiLabel,
  systemBTitle,
  systemBTitleMissing,
  headerSubtitle,
  metrics,
  explicitSupportText,
  displaySource,
  localizeKnownBody,
  localizeKnownLabel,
}: BuildSystemBLiteratureCardModelOptions): SystemBLiteratureCardModel {
  const textPanels = buildSystemBTextPanelsModel({
    detail,
    S,
    isSystemB,
    contextSummarySection,
    referenceSection,
    cardTakeaway,
    cardEvidenceLabel,
    localizeKnownBody,
    localizeKnownLabel,
  })
  const traceSteps = isSystemB && Array.isArray(detail.systemBTraceSteps)
    ? detail.systemBTraceSteps.map((item) => compact(item)).filter(Boolean)
    : []
  const traceReason = isSystemB ? cleanCitationDisplayText(detail.systemBTraceReason) : ''
  const traceScore = Number(detail.systemBTraceScore || 0)
  const showTrace = Boolean(
    SYSTEM_B_TRACE_ENABLED
    && isSystemB
    && (traceSteps.length > 0 || traceReason || traceScore > 0),
  )
  const traceStatus = detail.systemBTraceComplete
    ? { label: S.cite_trace_complete, tone: 'complete' }
    : { label: S.cite_trace_review, tone: 'review' }
  const sourcePanels = buildSystemBSourcePanelsModel({
    detail,
    S,
    isSystemB,
    locatorSection,
    cardReferenceLabel,
    cardSupportLabel,
    cardQualityFlags,
    sourcePaperText,
    headingPath,
    pageLabel,
    badgeLabel,
    doiLabel,
    systemBTitle,
    systemBTitleMissing,
    headerSubtitle,
    metrics,
    explicitSupportText,
    displaySource,
    systemBContextSource: textPanels.systemBContextSource,
    systemBReferenceText: textPanels.systemBReferenceText,
    systemBExplicitReferenceText: textPanels.systemBExplicitReferenceText,
    paperOverviewText: textPanels.paperOverviewText,
    citationContextText: textPanels.citationContextText,
  })
  const showOverviewLoading = Boolean(isSystemB && loading && !textPanels.paperOverviewText)
  const showOverviewUnavailable = Boolean(
    isSystemB
    && !loading
    && detail.bibliometricsChecked
    && !textPanels.paperOverviewText
    && !sourcePanels.showReference
    && (doiLabel || systemBTitle),
  )

  return {
    showTrace,
    traceStatus,
    traceScore,
    traceSteps,
    traceReason,
    traceLabel: S.cite_evidence_chain,
    paperOverviewText: textPanels.paperOverviewText,
    paperOverviewLabel: textPanels.paperOverviewLabel,
    paperOverviewPreview: textPanels.paperOverviewPreview,
    showOverviewLoading,
    overviewLoadingLabel: S.cite_loading_summary || S.cite_loading,
    showOverviewUnavailable,
    overviewUnavailableLabel: S.cite_summary_unavailable,
    takeawayText: textPanels.takeawayText,
    takeawayLabel: textPanels.takeawayLabel,
    showLocation: sourcePanels.showLocation,
    locationLabel: sourcePanels.locationLabel,
    locationText: sourcePanels.locationText,
    locationHint: sourcePanels.locationHint,
    contextSummaryText: textPanels.contextSummaryText,
    contextSummaryLabel: textPanels.contextSummaryLabel,
    citationContextText: textPanels.citationContextText,
    citationContextPreview: textPanels.citationContextPreview,
    citationContextLabel: textPanels.citationContextLabel,
    showReference: sourcePanels.showReference,
    referenceLabel: sourcePanels.referenceLabel,
    referencePreview: sourcePanels.referencePreview,
    showSupport: sourcePanels.showSupport,
    supportLabel: sourcePanels.supportLabel,
    supportText: sourcePanels.supportText,
  }
}
