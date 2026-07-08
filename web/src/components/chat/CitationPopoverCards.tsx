import type { EvidenceCardViewModel } from './evidenceCardViewModel'
import { EvidenceCardContent } from './EvidenceCardContent'

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
      {showTrace ? (
        <div
          className={`kb-cite-pop-trace kb-cite-pop-trace-${traceStatus.tone}`}
          data-testid="citation-popover-system-b-trace"
        >
          <div className="kb-cite-pop-trace-head">
            <span className="kb-cite-pop-section-title">{traceLabel}</span>
            <span className="kb-cite-pop-trace-status">{traceStatus.label}</span>
            {traceScore > 0 ? (
              <span className="kb-cite-pop-trace-score">{Math.round(traceScore * 100)}%</span>
            ) : null}
          </div>
          {traceSteps.length > 0 ? (
            <div className="kb-cite-pop-trace-steps" aria-label="System B evidence chain">
              {traceSteps.map((step, index) => (
                <span className="kb-cite-pop-trace-step-wrap" key={`${step}-${index}`}>
                  <span className="kb-cite-pop-trace-step">{step}</span>
                  {index < traceSteps.length - 1 ? <span className="kb-cite-pop-trace-arrow">→</span> : null}
                </span>
              ))}
            </div>
          ) : null}
          {traceReason ? <div className="kb-cite-pop-trace-reason">{traceReason}</div> : null}
        </div>
      ) : null}
      {paperOverviewText ? (
        <div className="kb-cite-pop-insight kb-cite-pop-paper-overview" data-testid="citation-popover-system-b-overview">
          <span className="kb-cite-pop-section-title">{paperOverviewLabel}</span>
          <div className="kb-cite-pop-main">{paperOverviewPreview}</div>
        </div>
      ) : null}
      {showOverviewLoading ? (
        <div className="kb-cite-pop-sub kb-cite-pop-system-b-loading" data-testid="citation-popover-system-b-overview-loading">
          {overviewLoadingLabel}
        </div>
      ) : null}
      {showOverviewUnavailable ? (
        <div className="kb-cite-pop-sub kb-cite-pop-system-b-empty" data-testid="citation-popover-system-b-overview-empty">
          {overviewUnavailableLabel}
        </div>
      ) : null}
      {takeawayText ? (
        <div className="kb-cite-pop-insight kb-cite-pop-takeaway" data-testid="citation-popover-system-b-takeaway">
          <span className="kb-cite-pop-section-title">{takeawayLabel}</span>
          <div className="kb-cite-pop-main">{takeawayText}</div>
        </div>
      ) : null}
      {showLocation ? (
        <div className="kb-cite-pop-locator" data-testid="citation-popover-system-b-location">
          <span className="kb-cite-pop-section-title">{locationLabel}</span>
          <span className="kb-cite-pop-locator-text">{locationText}</span>
          {locationHint ? <span className="kb-cite-pop-anchor-meta">{locationHint}</span> : null}
        </div>
      ) : null}
      {contextSummaryText ? (
        <div className="kb-cite-pop-context-summary" data-testid="citation-popover-system-b-context-summary">
          <span className="kb-cite-pop-section-title">{contextSummaryLabel}</span>
          <div className="kb-cite-pop-main">{contextSummaryText}</div>
        </div>
      ) : null}
      {citationContextText ? (
        <div className="kb-cite-pop-quote" data-testid="citation-popover-system-b-context">
          <div className="kb-cite-pop-section-line">
            <span className="kb-cite-pop-section-title">{citationContextLabel}</span>
            {citationContextPreview !== citationContextText ? <span className="kb-cite-pop-section-hint">{excerptLabel}</span> : null}
          </div>
          <blockquote>{citationContextPreview}</blockquote>
        </div>
      ) : null}
      {showReference ? (
        <div className="kb-cite-pop-evidence" data-testid="citation-popover-system-b-reference">
          <div className="kb-cite-pop-section-title">{referenceLabel}</div>
          <div className="kb-cite-pop-main">{referencePreview}</div>
        </div>
      ) : null}
      {showSupport ? (
        <div className="kb-cite-pop-why" data-testid="citation-popover-system-b-support">
          <span className="kb-cite-pop-section-title">{supportLabel}</span>
          <div className="kb-cite-pop-main">{supportText}</div>
        </div>
      ) : null}
    </div>
  )
}
