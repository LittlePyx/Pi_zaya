import type { CiteDetail, CitationCardViewSection } from './citationState'
import { cleanCitationDisplayText, looksLowValueCitationContext } from './citationState'
import {
  compact,
  evidencePreview,
  isReferenceEntryLikeText,
  looksGenericSystemBTakeawayText,
  looksNarrativeMetadataText,
  substantiallySame,
} from './citationPopoverUtils'
import { resolveSystemBArticleSummary } from './systemBArticleSummary'

interface SystemBTextPanelStrings extends Record<string, string> {
  cite_context: string
  cite_context_summary: string
  cite_current_paper_usage: string
  cite_paper_overview: string
  cite_upstream_role: string
}

export interface BuildSystemBTextPanelsModelOptions {
  detail: CiteDetail
  S: SystemBTextPanelStrings
  isSystemB: boolean
  contextSummarySection?: CitationCardViewSection
  referenceSection?: CitationCardViewSection
  cardTakeaway: string
  cardEvidenceLabel: string
  localizeKnownBody: (value: string) => string
  localizeKnownLabel: (value: string) => string
}

export interface SystemBTextPanelsModel {
  systemBExplicitReferenceText: string
  systemBReferenceText: string
  systemBContextSource: string
  paperOverviewText: string
  paperOverviewLabel: string
  paperOverviewPreview: string
  takeawayText: string
  takeawayLabel: string
  contextSummaryText: string
  contextSummaryLabel: string
  citationContextText: string
  citationContextPreview: string
  citationContextLabel: string
}

export function buildSystemBTextPanelsModel({
  detail,
  S,
  isSystemB,
  contextSummarySection,
  referenceSection,
  cardTakeaway,
  cardEvidenceLabel,
  localizeKnownBody,
  localizeKnownLabel,
}: BuildSystemBTextPanelsModelOptions): SystemBTextPanelsModel {
  const systemBExplicitReferenceText = localizeKnownBody(cleanCitationDisplayText(referenceSection?.text || detail.cardReferenceEntry))
  const systemBReferenceText = systemBExplicitReferenceText || cleanCitationDisplayText(compact(detail.raw) || compact(detail.citeFmt))
  const systemBOverview = resolveSystemBArticleSummary(detail, detail, { forceSystemB: isSystemB })
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
  const paperOverviewText = isSystemB && systemBOverview.visible
    ? firstSystemBText([
      systemBOverview.line,
    ], { allowCitationContext: true })
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
  const citationContextPreview = evidencePreview(citationContextText, 330)

  return {
    systemBExplicitReferenceText,
    systemBReferenceText,
    systemBContextSource,
    paperOverviewText,
    paperOverviewLabel,
    paperOverviewPreview,
    takeawayText,
    takeawayLabel,
    contextSummaryText,
    contextSummaryLabel,
    citationContextText,
    citationContextPreview,
    citationContextLabel,
  }
}
