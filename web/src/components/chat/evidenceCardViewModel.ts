import {
  citationCardView,
  citationInlineLabel,
  cleanCitationDisplayText,
  type CiteDetail,
} from './citationState'

export interface EvidenceCardViewModel {
  label: string
  source: string
  claim: string
  claimPreview: string
  claimLabel: string
  evidence: string
  evidencePreview: string
  evidenceLabel: string
  location: string
  locationLabel: string
  support: string
  supportPreview: string
  supportLabel: string
  isEvidenceExcerpt: boolean
}

interface EvidenceCardViewModelOptions {
  S?: Record<string, string>
  evidenceLimit?: number
  claimLimit?: number
  supportLimit?: number
  includeCitationFallback?: boolean
  includeRawFallback?: boolean
  fallbackLabel?: string
  evidenceOverride?: string
  evidenceLabelOverride?: string
  claimOverride?: string
  claimLabelOverride?: string
  locationOverride?: string
  locationLabelOverride?: string
  supportOverride?: string
  supportLabelOverride?: string
}

function cleanFirst(...values: unknown[]): string {
  for (const value of values) {
    const cleaned = cleanCitationDisplayText(String(value || ''))
    if (cleaned) return cleaned
  }
  return ''
}

export function previewEvidenceText(value: string, maxLen = 260): string {
  const text = cleanCitationDisplayText(value).replace(/\s+/g, ' ').trim()
  if (!text || text.length <= maxLen) return text
  const head = text.slice(0, maxLen).replace(/[\uff0c,\uff1b;:\uff1a]\s*$/g, '').trim()
  return `${head}...`
}

export function previewClaimText(value: string, maxLen = 140): string {
  const text = cleanCitationDisplayText(value)
    .replace(/\s*\[[Rr]?\d{1,4}]\s*/g, ' ')
    .replace(/\s+/g, ' ')
    .replace(/^\s*(?:\d{1,3}[.)\u3001\uff0e]|[-*\u2022])\s*/, '')
    .trim()
  if (!text || text.length <= maxLen) return text
  const head = text.slice(0, maxLen)
  const cut = Math.max(
    head.lastIndexOf('\u3002'),
    head.lastIndexOf('\uff01'),
    head.lastIndexOf('\uff1f'),
    head.lastIndexOf('\uff1b'),
    head.lastIndexOf(';'),
    head.lastIndexOf('\uff0c'),
    head.lastIndexOf(','),
  )
  if (cut >= 40) return `${head.slice(0, cut).trim()}...`
  return `${head.slice(0, maxLen - 1).trim()}...`
}

export function buildEvidenceCardViewModel(
  detail: CiteDetail,
  options: EvidenceCardViewModelOptions = {},
): EvidenceCardViewModel {
  const S = options.S || {}
  const view = citationCardView(detail)
  const viewSection = (id: string) => view.sections.find((item) => item.id === id)
  const claimSection = viewSection('claim')
  const evidenceSection = viewSection('evidence')
  const locatorSection = viewSection('locator')
  const supportSection = viewSection('support')
  const includeCitationFallback = options.includeCitationFallback ?? true

  const label = citationInlineLabel(detail) || options.fallbackLabel || ''
  const source = cleanFirst(detail.cardTitle, detail.title, detail.sourceName, detail.sourcePath)
  const claim = cleanFirst(
    options.claimOverride,
    claimSection?.text,
    detail.cardClaim,
    detail.answerClaim,
    detail.cardTakeaway,
  )
  const evidence = cleanFirst(
    options.evidenceOverride,
    evidenceSection?.text,
    detail.cardEvidence,
    includeCitationFallback ? detail.evidenceQuote : '',
    includeCitationFallback ? detail.citationContext : '',
    includeCitationFallback ? detail.summaryLine : '',
    options.includeRawFallback ? detail.raw : '',
  )
  const location = cleanFirst(
    options.locationOverride,
    locatorSection?.text,
    detail.headingPath,
    detail.cardLocator,
    detail.locationLabel,
  )
  const support = cleanFirst(
    options.supportOverride,
    supportSection?.text,
    detail.cardSupportExplanation,
    detail.supportRelation,
    detail.whyLine,
    detail.bindingReason,
  )
  const claimLabel = cleanFirst(
    options.claimLabelOverride,
    S.cite_answer_point,
    claimSection?.label,
    detail.cardClaimLabel,
    'Answer claim',
  )
  const evidenceLabel = cleanFirst(
    options.evidenceLabelOverride,
    S.cite_original_evidence,
    evidenceSection?.label,
    detail.cardEvidenceLabel,
    'Evidence',
  )
  const locationLabel = cleanFirst(
    options.locationLabelOverride,
    S.cite_position,
    locatorSection?.label,
    detail.cardLocatorLabel,
    'Location',
  )
  const supportLabel = cleanFirst(
    options.supportLabelOverride,
    S.cite_reliability,
    S.cite_note,
    supportSection?.label,
    detail.cardSupportLabel,
    'Support',
  )
  const evidencePreview = previewEvidenceText(evidence, options.evidenceLimit ?? 260)

  return {
    label,
    source,
    claim,
    claimPreview: previewClaimText(claim, options.claimLimit ?? 140),
    claimLabel,
    evidence,
    evidencePreview,
    evidenceLabel,
    location,
    locationLabel,
    support,
    supportPreview: previewEvidenceText(support, options.supportLimit ?? 160),
    supportLabel,
    isEvidenceExcerpt: Boolean(evidence && evidencePreview !== evidence),
  }
}
