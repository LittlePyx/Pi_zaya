import type { EvidenceCardViewModel } from './evidenceCardViewModel'
import { EvidenceCardContent } from './EvidenceCardContent'
import {
  SystemBContextPanels,
  SystemBLocationPanel,
  SystemBOverviewPanels,
  SystemBReferencePanel,
  SystemBSupportPanel,
  SystemBTracePanel,
} from './CitationPopoverSystemBSections'

interface SystemAEvidenceCardProps {
  showTakeaway: boolean
  takeawayLabel: string
  takeawayText: string
  card: EvidenceCardViewModel
  showClaim: boolean
  excerptLabel: string
  showSupport: boolean
}

interface SystemBLiteratureCardProps {
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
  excerptLabel: string
  showReference: boolean
  referenceLabel: string
  referencePreview: string
  showSupport: boolean
  supportLabel: string
  supportText: string
}

export function SystemAEvidenceCard({
  showTakeaway,
  takeawayLabel,
  takeawayText,
  card,
  showClaim,
  excerptLabel,
  showSupport,
}: SystemAEvidenceCardProps) {
  return (
    <div className="kb-cite-pop-evidence-map">
      {showTakeaway ? (
        <div className="kb-cite-pop-insight kb-cite-pop-takeaway" data-testid="citation-popover-system-a-takeaway">
          <span className="kb-cite-pop-section-title">{takeawayLabel}</span>
          <div className="kb-cite-pop-main">{takeawayText}</div>
        </div>
      ) : null}
      <EvidenceCardContent
        card={card}
        variant="citation-system-a"
        showHeader={false}
        showClaim={showClaim}
        showEvidence={Boolean(card.evidence)}
        showSupport={showSupport}
        excerptLabel={excerptLabel}
        claimTestId="citation-popover-system-a-claim"
        evidenceTestId="citation-popover-system-a-evidence"
        supportTestId="citation-popover-system-a-support"
      />
    </div>
  )
}

export function SystemBLiteratureCard({
  showTrace,
  traceStatus,
  traceScore,
  traceSteps,
  traceReason,
  traceLabel,
  paperOverviewText,
  paperOverviewLabel,
  paperOverviewPreview,
  showOverviewLoading,
  overviewLoadingLabel,
  showOverviewUnavailable,
  overviewUnavailableLabel,
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
  excerptLabel,
  showReference,
  referenceLabel,
  referencePreview,
  showSupport,
  supportLabel,
  supportText,
}: SystemBLiteratureCardProps) {
  return (
    <div className="kb-cite-pop-evidence-map kb-cite-pop-literature-card" data-testid="citation-popover-system-b-card">
      <SystemBTracePanel
        showTrace={showTrace}
        traceStatus={traceStatus}
        traceScore={traceScore}
        traceSteps={traceSteps}
        traceReason={traceReason}
        traceLabel={traceLabel}
      />
      <SystemBOverviewPanels
        paperOverviewText={paperOverviewText}
        paperOverviewLabel={paperOverviewLabel}
        paperOverviewPreview={paperOverviewPreview}
        showOverviewLoading={showOverviewLoading}
        overviewLoadingLabel={overviewLoadingLabel}
        showOverviewUnavailable={showOverviewUnavailable}
        overviewUnavailableLabel={overviewUnavailableLabel}
        takeawayText={takeawayText}
        takeawayLabel={takeawayLabel}
      />
      <SystemBLocationPanel
        showLocation={showLocation}
        locationLabel={locationLabel}
        locationText={locationText}
        locationHint={locationHint}
      />
      <SystemBContextPanels
        contextSummaryText={contextSummaryText}
        contextSummaryLabel={contextSummaryLabel}
        citationContextText={citationContextText}
        citationContextPreview={citationContextPreview}
        citationContextLabel={citationContextLabel}
        excerptLabel={excerptLabel}
      />
      <SystemBReferencePanel
        showReference={showReference}
        referenceLabel={referenceLabel}
        referencePreview={referencePreview}
      />
      <SystemBSupportPanel
        showSupport={showSupport}
        supportLabel={supportLabel}
        supportText={supportText}
      />
    </div>
  )
}
