import { LeftOutlined, RightOutlined } from '@ant-design/icons'

interface ReaderEvidenceNavProps {
  activeLabel: string
  positionLabel: string
  canGoPrev: boolean
  canGoNext: boolean
  onGoPrev: () => void
  onGoNext: () => void
  prevLabel: string
  nextLabel: string
}

export function ReaderEvidenceNav({
  activeLabel,
  positionLabel,
  canGoPrev,
  canGoNext,
  onGoPrev,
  onGoNext,
  prevLabel,
  nextLabel,
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
        title={prevLabel}
        aria-label={prevLabel}
        data-testid="reader-evidence-prev"
      >
        <LeftOutlined />
      </button>
      <span className="kb-reader-evidence-position" data-testid="reader-evidence-position">
        {positionLabel}
      </span>
      <button
        type="button"
        className="kb-reader-evidence-btn"
        onClick={onGoNext}
        disabled={!canGoNext}
        title={nextLabel}
        aria-label={nextLabel}
        data-testid="reader-evidence-next"
      >
        <RightOutlined />
      </button>
    </div>
  )
}
