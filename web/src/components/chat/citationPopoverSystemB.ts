import type { CiteDetail, CitationCardViewSection } from './citationState'
import { cleanCitationDisplayText, looksLowValueCitationContext } from './citationState'
import {
  SYSTEM_B_ARTICLE_OVERVIEW_SOURCES,
  SYSTEM_B_TRACE_ENABLED,
  compact,
  evidencePreview,
  isOnlyPaperLabel,
  isReferenceEntryLikeText,
  looksGenericSystemBTakeawayText,
  looksNarrativeMetadataText,
  stripLocationIdentityPrefix,
  substantiallySame,
} from './citationPopoverUtils'

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
  const systemBExplicitReferenceText = localizeKnownBody(cleanCitationDisplayText(referenceSection?.text || detail.cardReferenceEntry))
  const systemBReferenceText = systemBExplicitReferenceText || cleanCitationDisplayText(compact(detail.raw) || compact(detail.citeFmt))
  const systemBOverviewSource = compact(detail.summarySource).toLowerCase()
  const systemBContextSource = compact(detail.citationContextSource).toLowerCase()
  const systemBReferenceIdentityText = cleanCitationDisplayText(systemBReferenceText)
  const normalizeSystemBTextCandidate = (value: string, opts: { allowCitationContext?: boolean } = {}): string => {
    const text = cleanCitationDisplayText(value).replace(/\s+/g, ' ').trim()
    if (!text) return ''
    if (looksGenericSystemBTakeawayText(text)) return ''
    if (looksNarrativeMetadataText(text, detail)) return ''
    if (substantiallySame(text, systemBReferenceIdentityText) || isReferenceEntryLikeText(text)) return ''
    if (!opts.allowCitationContext && looksLowValueCitationContext(text)) return ''
    return text
  }
  const firstSystemBText = (values: string[], opts?: { allowCitationContext?: boolean }): string => {
    for (const value of values) {
      const text = normalizeSystemBTextCandidate(value, opts)
      if (text) return text
    }
    return ''
  }
  const systemBOverviewSourceIsContext = [
    'answer_context',
    'citation_context',
    'reader_occurrence',
    'reader_reference_link',
    'reader_references',
  ].includes(systemBOverviewSource)
  const systemBOverviewSourceIsArticle = SYSTEM_B_ARTICLE_OVERVIEW_SOURCES.has(systemBOverviewSource)
  const paperOverviewText = isSystemB
    ? firstSystemBText([
      systemBOverviewSourceIsContext ? '' : detail.summaryLine,
    ], { allowCitationContext: systemBOverviewSourceIsArticle })
    : ''
  const paperOverviewPreview = evidencePreview(paperOverviewText, 360)
  const paperOverviewLabel = S.cite_paper_overview || 'Article overview'
  const citationContextText = ''
  const citationContextLabel = cardEvidenceLabel || S.cite_context
  const rawTakeawayText = isSystemB
    ? firstSystemBText([
      detail.cardContextSummary,
      contextSummarySection?.text || '',
      cardTakeaway,
      detail.upstreamWorkRole,
      detail.cardSupportExplanation,
      detail.supportRelation,
      detail.whyLine,
      detail.systemBTraceContext,
      systemBContextSource === 'reader_references' ? '' : detail.citationContext,
    ], { allowCitationContext: true })
    : ''
  const localizedTakeawayText = localizeKnownBody(rawTakeawayText)
  const takeawayText = localizedTakeawayText && !substantiallySame(localizedTakeawayText, systemBReferenceText)
    ? localizedTakeawayText
    : ''
  const takeawayLabel = S.cite_current_paper_usage || S.cite_upstream_role
  const contextSummaryText = ''
  const contextSummaryLabel = localizeKnownLabel(contextSummarySection?.label || '') || S.cite_context_summary
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
  const citationContextPreview = evidencePreview(citationContextText, 330)
  const rawLocationText = compact(locatorSection?.text || '') || compact(detail.cardLocator) || compact(detail.locationLabel) || [sourcePaperText, headingPath, pageLabel].filter(Boolean).join(' / ')
  const cleanedLocationText = stripLocationIdentityPrefix(rawLocationText, [
    sourcePaperText,
    detail.sourceName,
    displaySource,
  ])
  const locationIsPaperOnly = isOnlyPaperLabel(rawLocationText, [
    sourcePaperText,
    detail.sourceName,
    displaySource,
  ])
  const referenceRowLocation = (
    systemBContextSource === 'reader_references'
    || compact(detail.shelfOrigin).toLowerCase() === 'reader_references'
  ) && badgeLabel
    ? badgeLabel
    : ''
  const meaningfulLocation = locationIsPaperOnly
    ? ''
    : (cleanedLocationText || rawLocationText)
  const locationLabel = referenceRowLocation ? S.cite_reference_entry : S.cite_location_current
  const locationText = referenceRowLocation || meaningfulLocation
  const locationHint = ''
  const locationSourceIsWeak = [
    'answer_context',
    'answer_reference_mention',
    'reader_references',
  ].includes(systemBContextSource) || cardQualityFlags.some((flag) => [
    'answer_context_only',
    'reference_entry_only',
    'weak_citation_context',
    'missing_citation_context',
  ].includes(flag))
  const showLocation = Boolean(
    isSystemB
    && meaningfulLocation
    && !referenceRowLocation
    && !locationSourceIsWeak,
  )
  const supportText = isSystemB
    && explicitSupportText
    && !substantiallySame(explicitSupportText, citationContextText)
    && !substantiallySame(explicitSupportText, systemBReferenceText)
    ? explicitSupportText
    : ''
  const showSupport = false
  const hasHeaderIdentity = Boolean(
    (systemBTitle && systemBTitle !== S.cite_upstream_reference)
    || headerSubtitle
    || doiLabel
    || metrics.length > 0
  )
  const referenceHasBibliographicContext = Boolean(
    systemBReferenceText
    && /\b(?:18|19|20)\d{2}\b/.test(systemBReferenceText)
    && (
      isReferenceEntryLikeText(systemBReferenceText)
      || !systemBTitle
      || systemBReferenceText.length > systemBTitle.length + 18
    )
  )
  const referenceIsUsefulEntry = Boolean(
    systemBReferenceText
    && (
      referenceHasBibliographicContext
      || (
        (!systemBTitle || !substantiallySame(systemBReferenceText, systemBTitle))
        && (!headerSubtitle || !substantiallySame(systemBReferenceText, headerSubtitle))
      )
    )
  )
  const referenceEntryOnly = cardQualityFlags.includes('reference_entry_only')
  const referenceTitleMissing = systemBTitleMissing || cardQualityFlags.includes('missing_reference_title')
  const suppressReferenceEntry = [
    'reader_occurrence',
    'reader_reference_link',
    'reader_references',
  ].includes(systemBContextSource)
    && !referenceEntryOnly
    && !referenceTitleMissing
  const showReference = Boolean(
    systemBReferenceText
    && !suppressReferenceEntry
    && (
      (isSystemB && referenceIsUsefulEntry && (showLocation || referenceEntryOnly || referenceTitleMissing || !hasHeaderIdentity))
      || (systemBExplicitReferenceText && (referenceEntryOnly || referenceTitleMissing || !hasHeaderIdentity))
      || referenceTitleMissing
      || referenceEntryOnly
      || (!hasHeaderIdentity && !paperOverviewText)
    ),
  )
  const referencePreview = evidencePreview(systemBReferenceText, 260)
  const referenceLabel = cardReferenceLabel || S.cite_original_reference_entry || S.cite_reference_entry
  const showOverviewLoading = Boolean(isSystemB && loading && !paperOverviewText)
  const showOverviewUnavailable = Boolean(
    isSystemB
    && !loading
    && detail.bibliometricsChecked
    && !paperOverviewText
    && !showReference
    && (doiLabel || systemBTitle),
  )

  return {
    showTrace,
    traceStatus,
    traceScore,
    traceSteps,
    traceReason,
    traceLabel: S.cite_evidence_chain,
    paperOverviewText,
    paperOverviewLabel,
    paperOverviewPreview,
    showOverviewLoading,
    overviewLoadingLabel: S.cite_loading_summary || S.cite_loading,
    showOverviewUnavailable,
    overviewUnavailableLabel: S.cite_summary_unavailable,
    takeawayText,
    takeawayLabel,
    showLocation,
    locationLabel,
    locationText,
    locationHint,
    contextSummaryText,
    contextSummaryLabel,
    citationContextText,
    citationContextPreview,
    citationContextLabel,
    showReference,
    referenceLabel,
    referencePreview,
    showSupport,
    supportLabel: cardSupportLabel || S.cite_note,
    supportText,
  }
}
