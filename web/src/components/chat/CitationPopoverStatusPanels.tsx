interface CitationPopoverBindingState {
  label: string
  tone: string
}

interface CitationPopoverStatusPanelsProps {
  bindingState: CitationPopoverBindingState | null
  bindingOverlapText: string
  showBindingReason: boolean
  bindingReason: string
  showCardQuality: boolean
  cardQualityFlags: string[]
  cardQualityLabel: string
  cardQualityScore: number
  showCardWarning: boolean
  cardWarning: string
  showExternalMetadataWarning: boolean
  externalMetadataWarningText: string
  externalMetadataTitleHint: string
}

export function CitationPopoverStatusPanels({
  bindingState,
  bindingOverlapText,
  showBindingReason,
  bindingReason,
  showCardQuality,
  cardQualityFlags,
  cardQualityLabel,
  cardQualityScore,
  showCardWarning,
  cardWarning,
  showExternalMetadataWarning,
  externalMetadataWarningText,
  externalMetadataTitleHint,
}: CitationPopoverStatusPanelsProps) {
  return (
    <>
      {bindingState ? (
        <div
          className={`kb-cite-pop-binding kb-cite-pop-binding-${bindingState.tone}`}
          data-testid="citation-popover-binding-status"
        >
          <span className="kb-cite-pop-binding-label">{bindingState.label}</span>
          {bindingOverlapText ? <span className="kb-cite-pop-binding-terms">{bindingOverlapText}</span> : null}
          {showBindingReason ? <span className="kb-cite-pop-binding-reason">{bindingReason}</span> : null}
        </div>
      ) : null}
      {showCardQuality ? (
        <div
          className="kb-cite-pop-quality"
          data-testid="citation-popover-card-quality"
          title={cardQualityFlags.join(' / ')}
        >
          <span className="kb-cite-pop-quality-label">{cardQualityLabel}</span>
          {cardQualityScore > 0 ? <span className="kb-cite-pop-quality-score">{Math.round(cardQualityScore * 100)}%</span> : null}
        </div>
      ) : null}
      {showCardWarning ? (
        <div className="kb-cite-pop-warning" data-testid="citation-popover-card-warning">
          {cardWarning}
        </div>
      ) : null}
      {showExternalMetadataWarning ? (
        <div className="kb-cite-pop-warning" data-testid="citation-popover-external-metadata-warning">
          {externalMetadataWarningText}
          {externalMetadataTitleHint ? <span className="kb-cite-pop-warning-sub">{externalMetadataTitleHint}</span> : null}
        </div>
      ) : null}
    </>
  )
}
