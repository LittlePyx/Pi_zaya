import type { CiteDetail } from './citationState'

interface CitationPopoverActionsProps {
  detail: CiteDetail
  showOpenReaderAction: boolean
  canOpenReader: boolean
  openReaderLabel: string
  onOpenReader: (detail: CiteDetail) => void
  showStartGuideAction: boolean
  guideLoading: boolean
  startGuideLabel: string
  startingGuideLabel: string
  onStartGuide: (detail: CiteDetail) => void
  openShelfLabel: string
  onOpenShelf: () => void
  inShelf: boolean
  addToShelfLabel: string
  inShelfLabel: string
  onAddToShelf: (detail: CiteDetail) => void
}

export function CitationPopoverActions({
  detail,
  showOpenReaderAction,
  canOpenReader,
  openReaderLabel,
  onOpenReader,
  showStartGuideAction,
  guideLoading,
  startGuideLabel,
  startingGuideLabel,
  onStartGuide,
  openShelfLabel,
  onOpenShelf,
  inShelf,
  addToShelfLabel,
  inShelfLabel,
  onAddToShelf,
}: CitationPopoverActionsProps) {
  return (
    <div className="kb-cite-pop-actions">
      {showOpenReaderAction ? (
        <button
          className="kb-cite-pop-open-shelf kb-cite-pop-action-primary"
          type="button"
          disabled={!canOpenReader}
          onClick={() => onOpenReader(detail)}
        >
          {openReaderLabel}
        </button>
      ) : null}
      {showStartGuideAction ? (
        <button
          className="kb-cite-pop-open-shelf"
          type="button"
          onClick={() => onStartGuide(detail)}
          disabled={guideLoading}
        >
          {guideLoading ? startingGuideLabel : startGuideLabel}
        </button>
      ) : null}
      <button className="kb-cite-pop-open-shelf" type="button" onClick={onOpenShelf}>
        {openShelfLabel}
      </button>
      <button
        className={`kb-cite-pop-add ${inShelf ? 'kb-added' : ''}`}
        type="button"
        onClick={() => onAddToShelf(detail)}
      >
        {inShelf ? inShelfLabel : addToShelfLabel}
      </button>
    </div>
  )
}
