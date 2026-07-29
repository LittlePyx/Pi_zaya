import type { CiteDetail, CitationCardViewSection } from './citationState'
import { citationCardView } from './citationState'
import { buildCitationPopoverFrameModel } from './citationPopoverFrameModel'
import type { CitationPopoverFrameModel } from './citationPopoverFrameModel'
import { buildCitationPopoverLocalizers } from './citationPopoverLocalization'
import { buildCitationPopoverStatusModel } from './citationPopoverStatusModel'
import type { CitationPopoverStatusModel } from './citationPopoverStatusModel'
import { buildSystemAEvidenceCardModel } from './citationPopoverSystemA'
import type { SystemAEvidenceCardModel } from './citationPopoverSystemA'
import { buildSystemBLiteratureCardModel } from './citationPopoverSystemB'
import type { SystemBLiteratureCardModel } from './citationPopoverSystemB'
import {
  compact,
  looksNarrativeMetadataText,
} from './citationPopoverUtils'

interface CitationPopoverViewModelStrings extends Record<string, string> {
  cite_anchor_equation: string
  cite_anchor_figure: string
  cite_anchor_label: string
  cite_anchor_paragraph: string
  cite_anchor_sentence: string
  cite_anchor_table: string
  cite_answer_point: string
  cite_binding_candidate: string
  cite_binding_mismatch: string
  cite_candidate_support_default: string
  cite_context: string
  cite_context_summary: string
  cite_current_paper_usage: string
  cite_evidence_chain: string
  cite_evidence_focus: string
  cite_external_metadata_warning: string
  cite_external_title: string
  cite_frontend_candidate_reason: string
  cite_kind_evidence: string
  cite_kind_upstream: string
  cite_loading: string
  cite_loading_summary: string
  cite_location_current: string
  cite_location_paper: string
  cite_meta_author: string
  cite_meta_published: string
  cite_meta_source: string
  cite_missing_reference_entry: string
  cite_missing_reference_entry_body: string
  cite_note: string
  cite_open_evidence: string
  cite_original_evidence: string
  cite_original_reference_entry: string
  cite_paper_overview: string
  cite_position: string
  cite_read_locate: string
  cite_reference_entry: string
  cite_reliability: string
  cite_summary_unavailable: string
  cite_system_b_support_default: string
  cite_trace_complete: string
  cite_trace_review: string
  cite_upstream_reference: string
  cite_upstream_role: string
}

export interface BuildCitationPopoverViewModelOptions {
  detail: CiteDetail
  S: CitationPopoverViewModelStrings
  loading: boolean
}

export interface CitationPopoverViewModel {
  explainText: string
  frame: CitationPopoverFrameModel
  isSystemB: boolean
  status: CitationPopoverStatusModel
  systemA: SystemAEvidenceCardModel
  systemB: SystemBLiteratureCardModel
}

function findSection(sections: CitationCardViewSection[], id: string): CitationCardViewSection | undefined {
  return sections.find((item) => item.id === id)
}

export function buildCitationPopoverViewModel({
  detail,
  S,
  loading,
}: BuildCitationPopoverViewModelOptions): CitationPopoverViewModel {
  const { localizeKnownBody, localizeKnownLabel } = buildCitationPopoverLocalizers(S)
  const explicitRoute = compact(detail.citationRoute).toLowerCase()
  const isSystemB = explicitRoute === 'system_b'
    || (explicitRoute !== 'system_a' && Boolean(detail.isInpaper))
  const view = citationCardView(detail)
  const takeawaySection = findSection(view.sections, 'takeaway')
  const claimSection = findSection(view.sections, 'claim')
  const locatorSection = findSection(view.sections, 'locator')
  const contextSummarySection = findSection(view.sections, 'context_summary')
  const evidenceSection = findSection(view.sections, 'evidence')
  const referenceSection = findSection(view.sections, 'reference')
  const supportSection = findSection(view.sections, 'support')
  const warningSection = findSection(view.sections, 'warning')
  const cardTakeawayLabel = localizeKnownLabel(takeawaySection?.label || detail.cardTakeawayLabel)
  const rawCardTakeaway = compact(takeawaySection?.text || detail.cardTakeaway)
  const cardTakeaway = looksNarrativeMetadataText(rawCardTakeaway, detail) ? '' : rawCardTakeaway
  const cardClaimLabel = localizeKnownLabel(claimSection?.label || detail.cardClaimLabel)
  const cardEvidenceLabel = localizeKnownLabel(evidenceSection?.label || detail.cardEvidenceLabel)
  const cardLocatorLabel = localizeKnownLabel(locatorSection?.label || detail.cardLocatorLabel)
  const frame = buildCitationPopoverFrameModel({
    detail,
    S,
    isSystemB,
    viewHeader: view.header,
    locatorSection,
    cardLocatorLabel,
    localizeKnownLabel,
  })
  const cardReferenceLabel = localizeKnownLabel(referenceSection?.label || detail.cardReferenceLabel)
  const cardSupportLabel = localizeKnownLabel(supportSection?.label || detail.cardSupportLabel)
  const status = buildCitationPopoverStatusModel({
    detail,
    S,
    isSystemB,
    supportSection,
    warningSection,
    displayMain: frame.displayMain,
    localizeKnownBody,
    localizeKnownLabel,
  })
  const systemA = buildSystemAEvidenceCardModel({
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
    cardQualityFlags: status.cardQualityFlags,
    cardWarning: status.cardWarning,
    hasBindingState: Boolean(status.bindingState),
    supportText: status.supportText,
  })
  const systemB = buildSystemBLiteratureCardModel({
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
    cardQualityFlags: status.cardQualityFlags,
    sourcePaperText: frame.sourcePaperText,
    headingPath: frame.headingPath,
    pageLabel: frame.pageLabel,
    badgeLabel: frame.badgeLabel,
    doiLabel: frame.doiLabel,
    systemBTitle: frame.systemBTitle,
    systemBTitleMissing: frame.systemBTitleMissing,
    headerSubtitle: frame.headerSubtitle,
    metrics: frame.metrics,
    explicitSupportText: status.explicitSupportText,
    displaySource: frame.displaySource,
    localizeKnownBody,
    localizeKnownLabel,
  })

  return {
    explainText: '',
    frame,
    isSystemB,
    status,
    systemA,
    systemB,
  }
}
