export interface SystemBTracePanelProps {
  showTrace: boolean
  traceStatus: { label: string; tone: string }
  traceScore: number
  traceSteps: string[]
  traceReason: string
  traceLabel: string
}

export interface SystemBOverviewPanelsProps {
  paperOverviewText: string
  paperOverviewLabel: string
  paperOverviewPreview: string
  showOverviewLoading: boolean
  overviewLoadingLabel: string
  showOverviewUnavailable: boolean
  overviewUnavailableLabel: string
  takeawayText: string
  takeawayLabel: string
}

export interface SystemBLocationPanelProps {
  showLocation: boolean
  locationLabel: string
  locationText: string
  locationHint: string
}

export interface SystemBContextPanelsProps {
  contextSummaryText: string
  contextSummaryLabel: string
  citationContextText: string
  citationContextPreview: string
  citationContextLabel: string
  excerptLabel: string
}

export interface SystemBReferencePanelProps {
  showReference: boolean
  referenceLabel: string
  referencePreview: string
}

export interface SystemBSupportPanelProps {
  showSupport: boolean
  supportLabel: string
  supportText: string
}

export function SystemBTracePanel({
  showTrace,
  traceStatus,
  traceScore,
  traceSteps,
  traceReason,
  traceLabel,
}: SystemBTracePanelProps) {
  if (!showTrace) return null

  return (
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
              {index < traceSteps.length - 1 ? <span className="kb-cite-pop-trace-arrow">&rarr;</span> : null}
            </span>
          ))}
        </div>
      ) : null}
      {traceReason ? <div className="kb-cite-pop-trace-reason">{traceReason}</div> : null}
    </div>
  )
}

export function SystemBOverviewPanels({
  paperOverviewText,
  paperOverviewLabel,
  paperOverviewPreview,
  showOverviewLoading,
  overviewLoadingLabel,
  showOverviewUnavailable,
  overviewUnavailableLabel,
  takeawayText,
  takeawayLabel,
}: SystemBOverviewPanelsProps) {
  return (
    <>
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
    </>
  )
}

export function SystemBLocationPanel({
  showLocation,
  locationLabel,
  locationText,
  locationHint,
}: SystemBLocationPanelProps) {
  if (!showLocation) return null

  return (
    <div className="kb-cite-pop-locator" data-testid="citation-popover-system-b-location">
      <span className="kb-cite-pop-section-title">{locationLabel}</span>
      <span className="kb-cite-pop-locator-text">{locationText}</span>
      {locationHint ? <span className="kb-cite-pop-anchor-meta">{locationHint}</span> : null}
    </div>
  )
}

export function SystemBContextPanels({
  contextSummaryText,
  contextSummaryLabel,
  citationContextText,
  citationContextPreview,
  citationContextLabel,
  excerptLabel,
}: SystemBContextPanelsProps) {
  return (
    <>
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
    </>
  )
}

export function SystemBReferencePanel({
  showReference,
  referenceLabel,
  referencePreview,
}: SystemBReferencePanelProps) {
  if (!showReference) return null

  return (
    <div className="kb-cite-pop-evidence" data-testid="citation-popover-system-b-reference">
      <div className="kb-cite-pop-section-title">{referenceLabel}</div>
      <div className="kb-cite-pop-main">{referencePreview}</div>
    </div>
  )
}

export function SystemBSupportPanel({
  showSupport,
  supportLabel,
  supportText,
}: SystemBSupportPanelProps) {
  if (!showSupport) return null

  return (
    <div className="kb-cite-pop-why" data-testid="citation-popover-system-b-support">
      <span className="kb-cite-pop-section-title">{supportLabel}</span>
      <div className="kb-cite-pop-main">{supportText}</div>
    </div>
  )
}
