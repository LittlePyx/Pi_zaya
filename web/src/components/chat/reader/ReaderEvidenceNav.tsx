interface ReaderEvidenceNavProps {
  activeLabel: string
  positionLabel: string
  canGoPrev: boolean
  canGoNext: boolean
  onGoPrev: () => void
  onGoNext: () => void
}

export function ReaderEvidenceNav({
  activeLabel,
  positionLabel,
  canGoPrev,
  canGoNext,
  onGoPrev,
  onGoNext,
}: ReaderEvidenceNavProps) {
  return (
    <div
      className="kb-reader-evidence-nav"
      title={activeLabel || 'Evidence navigator'}
      data-testid="reader-evidence-nav"
    >
      <button
        type="button"
        className="kb-reader-evidence-btn"
        onClick={onGoPrev}
        disabled={!canGoPrev}
        data-testid="reader-evidence-prev"
      >
        Prev
      </button>
      <span className="kb-reader-evidence-position" data-testid="reader-evidence-position">
        {positionLabel}
      </span>
      <button
        type="button"
        className="kb-reader-evidence-btn"
        onClick={onGoNext}
        disabled={!canGoNext}
        data-testid="reader-evidence-next"
      >
        Next
      </button>
    </div>
  )
}
