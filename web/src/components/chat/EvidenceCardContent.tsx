import type { EvidenceCardViewModel } from './evidenceCardViewModel'

type EvidenceCardContentVariant = 'drawer' | 'citation-system-a'

export function EvidenceCardContent({
  card,
  variant,
  sourceFallback = 'Source',
  showHeader = variant === 'drawer',
  showClaim = Boolean(card.claim),
  showEvidence = Boolean(card.evidence),
  showLocation = Boolean(card.location),
  showSupport = Boolean(card.support),
  excerptLabel = '',
  claimTestId,
  evidenceTestId,
  supportTestId,
}: {
  card: EvidenceCardViewModel
  variant: EvidenceCardContentVariant
  sourceFallback?: string
  showHeader?: boolean
  showClaim?: boolean
  showEvidence?: boolean
  showLocation?: boolean
  showSupport?: boolean
  excerptLabel?: string
  claimTestId?: string
  evidenceTestId?: string
  supportTestId?: string
}) {
  if (variant === 'citation-system-a') {
    return (
      <>
        {showClaim ? (
          <div className="kb-cite-pop-claim" data-testid={claimTestId}>
            <span className="kb-cite-pop-section-title">{card.claimLabel}</span>
            <div className="kb-cite-pop-main">{card.claimPreview}</div>
          </div>
        ) : null}
        {showEvidence ? (
          <div className="kb-cite-pop-quote" data-testid={evidenceTestId}>
            <div className="kb-cite-pop-section-line">
              <span className="kb-cite-pop-section-title">{card.evidenceLabel}</span>
              {card.isEvidenceExcerpt ? <span className="kb-cite-pop-section-hint">{excerptLabel}</span> : null}
            </div>
            <blockquote>{card.evidencePreview}</blockquote>
          </div>
        ) : null}
        {showSupport ? (
          <div className="kb-cite-pop-why" data-testid={supportTestId}>
            <span className="kb-cite-pop-section-title">{card.supportLabel}</span>
            <div className="kb-cite-pop-main">{card.supportPreview}</div>
          </div>
        ) : null}
      </>
    )
  }

  return (
    <>
      {showHeader ? (
        <div className="kb-evidence-item-head">
          <span className="kb-evidence-cite-label">{card.label}</span>
          <span className="kb-evidence-source-name">{card.source || sourceFallback}</span>
        </div>
      ) : null}
      {showClaim ? (
        <div className="kb-evidence-block">
          <div className="kb-evidence-block-label">{card.claimLabel}</div>
          <div className="kb-evidence-block-text">{card.claimPreview}</div>
        </div>
      ) : null}
      {showEvidence ? (
        <div className="kb-evidence-block">
          <div className="kb-evidence-block-label">{card.evidenceLabel}</div>
          <blockquote>{card.evidencePreview}</blockquote>
        </div>
      ) : null}
      {showLocation || showSupport ? (
        <div className="kb-evidence-meta">
          {showLocation ? <span>{card.location}</span> : null}
          {showSupport ? <span>{card.supportPreview}</span> : null}
        </div>
      ) : null}
    </>
  )
}
