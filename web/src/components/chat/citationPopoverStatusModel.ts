import type { CiteDetail, CitationCardViewSection } from './citationState'
import {
  compact,
  looksNarrativeMetadataText,
  substantiallySame,
} from './citationPopoverUtils'

interface StatusStrings extends Record<string, string> {
  cite_binding_candidate: string
  cite_binding_mismatch: string
  cite_candidate_support_default: string
  cite_external_metadata_warning: string
  cite_external_title: string
  cite_system_b_support_default: string
}

interface BuildCitationPopoverStatusModelOptions {
  detail: CiteDetail
  S: StatusStrings
  isSystemB: boolean
  supportSection?: CitationCardViewSection
  warningSection?: CitationCardViewSection
  displayMain: string
  localizeKnownBody: (value: string) => string
  localizeKnownLabel: (value: string) => string
}

export interface CitationPopoverBindingState {
  label: string
  tone: string
}

export interface CitationPopoverStatusModel {
  bindingOverlapText: string
  bindingReason: string
  bindingState: CitationPopoverBindingState | null
  bindingStatus: string
  cardQualityFlags: string[]
  cardQualityLabel: string
  cardQualityScore: number
  cardWarning: string
  explicitSupportText: string
  externalMetadataTitleHint: string
  externalMetadataWarningText: string
  showBindingReason: boolean
  showCardQuality: boolean
  showCardWarning: boolean
  showExternalMetadataWarning: boolean
  supportText: string
}

export function buildCitationPopoverStatusModel({
  detail,
  S,
  isSystemB,
  supportSection,
  warningSection,
  displayMain,
  localizeKnownBody,
  localizeKnownLabel,
}: BuildCitationPopoverStatusModelOptions): CitationPopoverStatusModel {
  const cardWarning = compact(warningSection?.text || detail.cardWarning)
  const cardQualityLabel = localizeKnownLabel(detail.cardQualityLabel)
  const cardQualityScore = Number(detail.cardQualityScore || 0)
  const cardQualityFlags = Array.isArray(detail.cardQualityFlags)
    ? detail.cardQualityFlags.map((item) => compact(item)).filter(Boolean)
    : []
  const whyText = compact(detail.whyLine)
  const bindingStatus = compact(detail.bindingStatus).toLowerCase()
  const bindingReason = localizeKnownBody(detail.bindingReason)
  const bindingOverlapText = Array.isArray(detail.bindingOverlapTerms)
    ? detail.bindingOverlapTerms.map((item) => compact(item)).filter(Boolean).join(' / ')
    : ''
  const bindingState = !isSystemB && bindingStatus && bindingStatus !== 'grounded'
    ? (
        bindingStatus === 'mismatch'
            ? { label: S.cite_binding_mismatch, tone: 'mismatch' }
            : { label: S.cite_binding_candidate, tone: 'candidate' }
      )
    : null
  const rawExplicitSupportText = localizeKnownBody(supportSection?.text || '')
    || localizeKnownBody(detail.cardSupportExplanation)
    || localizeKnownBody(detail.supportRelation)
    || whyText
    || bindingReason
  const explicitSupportText = looksNarrativeMetadataText(rawExplicitSupportText, detail) ? '' : rawExplicitSupportText
  const supportText = isSystemB
    ? (explicitSupportText || S.cite_system_b_support_default)
    : (explicitSupportText || (bindingStatus === 'candidate'
      ? S.cite_candidate_support_default
      : ''))
  const showBindingReason = Boolean(bindingReason && !substantiallySame(bindingReason, supportText))
  const showCardQuality = false
  const showCardWarning = Boolean(cardWarning && cardQualityFlags.includes('missing_reference_entry'))
  const externalMetadataStatus = compact(detail.externalMetadataStatus).toLowerCase()
  const externalMetadataReason = compact(detail.externalMetadataReason)
  const externalTitle = compact(detail.externalTitle)
  const showExternalMetadataWarning = externalMetadataStatus === 'conflict'
  const externalMetadataWarningText = showExternalMetadataWarning
    ? (externalMetadataReason || S.cite_external_metadata_warning)
    : ''
  const externalMetadataTitleHint = externalTitle && !substantiallySame(externalTitle, displayMain)
    ? S.cite_external_title.replace('{title}', externalTitle)
    : ''

  return {
    bindingOverlapText,
    bindingReason,
    bindingState,
    bindingStatus,
    cardQualityFlags,
    cardQualityLabel,
    cardQualityScore,
    cardWarning,
    explicitSupportText,
    externalMetadataTitleHint,
    externalMetadataWarningText,
    showBindingReason,
    showCardQuality,
    showCardWarning,
    showExternalMetadataWarning,
    supportText,
  }
}
