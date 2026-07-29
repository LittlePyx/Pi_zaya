import type { CiteDetail, CitationCardViewSection } from './citationState'
import {
  answerPointPreview,
  evidencePreview,
  isLowValueSystemAClaim,
  looksNarrativeMetadataText,
  substantiallySame,
} from './citationPopoverUtils'
import { buildEvidenceCardViewModel } from './evidenceCardViewModel'
import type { EvidenceCardViewModel } from './evidenceCardViewModel'

interface SystemAStrings extends Record<string, string> {
  cite_answer_point: string
  cite_evidence_focus: string
  cite_original_evidence: string
  cite_reliability: string
}

interface BuildSystemAEvidenceCardModelOptions {
  detail: CiteDetail
  S: SystemAStrings
  isSystemB: boolean
  claimSection?: CitationCardViewSection
  evidenceSection?: CitationCardViewSection
  supportSection?: CitationCardViewSection
  cardTakeaway: string
  cardTakeawayLabel: string
  cardClaimLabel: string
  cardEvidenceLabel: string
  cardSupportLabel: string
  cardQualityFlags: string[]
  cardWarning: string
  hasBindingState: boolean
  supportText: string
}

export interface SystemAEvidenceCardModel {
  showTakeaway: boolean
  takeawayLabel: string
  takeawayText: string
  contentCard: EvidenceCardViewModel
  showClaim: boolean
  showSupport: boolean
}

const GENERIC_SYSTEM_A_CLAIM_LABELS = new Set([
  '\u7b54\u6848\u4e2d\u7684\u8bdd',
  '\u5bf9\u5e94\u56de\u7b54',
])

// The backend card contract already caps System-A evidence at 520 characters.
// Re-truncating it to 250/330 here can remove a later step of the same cited
// mechanism and make the popover contradict the answer. Keep the complete
// contract-owned quote; the popover itself provides the visual scroll bound.
const SYSTEM_A_EVIDENCE_PREVIEW_LIMIT = 520

export function buildSystemAEvidenceCardModel({
  detail,
  S,
  isSystemB,
  claimSection,
  evidenceSection,
  supportSection,
  cardTakeaway,
  cardTakeawayLabel,
  cardClaimLabel,
  cardEvidenceLabel,
  cardSupportLabel,
  cardQualityFlags,
  cardWarning,
  hasBindingState,
  supportText,
}: BuildSystemAEvidenceCardModelOptions): SystemAEvidenceCardModel {
  const suppressRawEvidenceFallback = !isSystemB
    && (
      cardQualityFlags.includes('evidence_quote_filtered')
      || cardQualityFlags.includes('missing_evidence_quote')
    )
  const evidenceCard = buildEvidenceCardViewModel(detail, {
    S,
    evidenceOverride: evidenceSection?.text || detail.cardEvidence,
    evidenceLabelOverride: cardEvidenceLabel,
    claimOverride: claimSection?.text || detail.cardClaim || detail.answerClaim,
    claimLabelOverride: cardClaimLabel,
    supportOverride: supportSection?.text || detail.cardSupportExplanation || detail.supportRelation || detail.whyLine,
    supportLabelOverride: cardSupportLabel,
    includeCitationFallback: !suppressRawEvidenceFallback,
    includeRawFallback: false,
  })
  const rawClaimText = evidenceCard.claim
  const claimText = looksNarrativeMetadataText(rawClaimText, detail) ? '' : rawClaimText
  const claimPreview = answerPointPreview(claimText)
  const claimLabel = cardClaimLabel && !GENERIC_SYSTEM_A_CLAIM_LABELS.has(cardClaimLabel)
    ? cardClaimLabel
    : S.cite_answer_point
  const evidenceText = evidenceCard.evidence
  const takeawayText = !isSystemB && cardTakeaway && !substantiallySame(cardTakeaway, evidenceText)
    ? cardTakeaway
    : ''
  const evidencePreviewText = evidencePreview(evidenceText, SYSTEM_A_EVIDENCE_PREVIEW_LIMIT)
  const hasReviewRisk = Boolean(
    hasBindingState
    || cardWarning
    || cardQualityFlags.includes('candidate_binding')
    || cardQualityFlags.includes('binding_mismatch'),
  )
  const hasOccurrenceClaim = cardQualityFlags.includes('occurrence_specific_claim')
  const claimLooksUseful = !isLowValueSystemAClaim(claimText)
  const showClaim = Boolean(
    claimPreview
    && claimLooksUseful
    && (!evidenceText || ((hasReviewRisk || hasOccurrenceClaim) && !substantiallySame(claimText, evidenceText))),
  )
  const showTakeaway = Boolean(
    takeawayText
    && !(showClaim && substantiallySame(takeawayText, claimText)),
  )
  const showSupport = Boolean(
    (hasReviewRisk || supportSection?.text)
    && supportText
    && !substantiallySame(supportText, evidenceText)
    && !substantiallySame(supportText, claimText)
    && !substantiallySame(supportText, takeawayText),
  )
  const contentCard: EvidenceCardViewModel = {
    ...evidenceCard,
    claim: claimText,
    claimPreview,
    claimLabel,
    evidence: evidenceText,
    evidencePreview: evidencePreviewText,
    evidenceLabel: cardEvidenceLabel || S.cite_original_evidence,
    support: supportText,
    supportPreview: supportText,
    supportLabel: cardSupportLabel || S.cite_reliability,
    isEvidenceExcerpt: Boolean(evidenceText && evidencePreviewText !== evidenceText),
  }

  return {
    showTakeaway,
    takeawayLabel: cardTakeawayLabel || S.cite_evidence_focus,
    takeawayText,
    contentCard,
    showClaim,
    showSupport,
  }
}
