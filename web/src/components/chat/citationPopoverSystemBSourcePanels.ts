import type { CiteDetail, CitationCardViewSection } from './citationState'
import {
  compact,
  evidencePreview,
  isOnlyPaperLabel,
  isReferenceEntryLikeText,
  stripLocationIdentityPrefix,
  substantiallySame,
} from './citationPopoverUtils'

interface SystemBSourcePanelStrings extends Record<string, string> {
  cite_location_current: string
  cite_note: string
  cite_original_reference_entry: string
  cite_reference_entry: string
  cite_upstream_reference: string
}

export interface BuildSystemBSourcePanelsModelOptions {
  detail: CiteDetail
  S: SystemBSourcePanelStrings
  isSystemB: boolean
  locatorSection?: CitationCardViewSection
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
  systemBContextSource: string
  systemBReferenceText: string
  systemBExplicitReferenceText: string
  paperOverviewText: string
  citationContextText: string
}

export interface SystemBSourcePanelsModel {
  showLocation: boolean
  locationLabel: string
  locationText: string
  locationHint: string
  showReference: boolean
  referenceLabel: string
  referencePreview: string
  showSupport: boolean
  supportLabel: string
  supportText: string
}

export function buildSystemBSourcePanelsModel({
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
  systemBContextSource,
  systemBReferenceText,
  systemBExplicitReferenceText,
  paperOverviewText,
  citationContextText,
}: BuildSystemBSourcePanelsModelOptions): SystemBSourcePanelsModel {
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

  return {
    showLocation,
    locationLabel,
    locationText,
    locationHint,
    showReference,
    referenceLabel,
    referencePreview,
    showSupport,
    supportLabel: cardSupportLabel || S.cite_note,
    supportText,
  }
}
