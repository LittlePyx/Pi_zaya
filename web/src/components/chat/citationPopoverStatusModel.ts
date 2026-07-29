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

function looksTemplateBindingReason(value: string): boolean {
  const text = compact(value).replace(/\s+/g, ' ')
  if (!text) return false
  return (
    /^this citation (?:reuses|uses|is only)/i.test(text)
    || /^this (?:answer sentence|claim) is supported by/i.test(text)
    || /^the (?:same )?source (?:passage )?(?:directly )?(?:contains|reports|provides)/i.test(text)
    || /^the answer and source align/i.test(text)
    || /^\u8be5?\u5f15\u7528\u590d\u7528\u751f\u6210\u56de\u7b54\u65f6/.test(text)
    || /^\u8be5?\u5f15\u7528\u4f7f\u7528\u4e86\u5df2\u6838\u5bf9/.test(text)
    || /^\u539f\u6587\u5728\u8be5\u5b9a\u4f4d\u5904\u7ed9\u51fa\u7684\u5177\u4f53\u9648\u8ff0/.test(text)
    || /^\u7b54\u6848\u4e0e\u82f1\u6587\u539f\u6587\u5728.*\u591a\u4e2a\u5177\u4f53\u52a8\u4f5c/.test(text)
  )
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
  const rawBindingReason = localizeKnownBody(detail.bindingReason)
  const bindingReason = looksTemplateBindingReason(rawBindingReason) ? '' : rawBindingReason
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
    || (isSystemB ? bindingReason : '')
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
