import { citationDisplay, type CiteDetail } from './citationState'
import type { UnlinkedReferenceView } from './messageCitationViews'

interface MessageReferenceCandidatesProps {
  views: UnlinkedReferenceView[]
  messageId: number
  canOpenReader: boolean
  onOpenReader: (detail: CiteDetail) => void
  onAddToShelf: (detail: CiteDetail) => void
  S: Record<string, string>
}

export function MessageReferenceCandidates({
  views,
  messageId,
  canOpenReader,
  onOpenReader,
  onAddToShelf,
  S,
}: MessageReferenceCandidatesProps) {
  if (views.length <= 0) return null

  return (
    <div className="kb-unlinked-ref-strip" data-testid={`unlinked-reference-candidates-${messageId}`}>
      <div className="kb-unlinked-ref-head">
        <span>{S.msg_reference_candidates_title || 'Possible cited papers'}</span>
        <span>{S.msg_reference_candidates_note || 'Found in this paper bibliography'}</span>
      </div>
      <div className="kb-unlinked-ref-list">
        {views.map((view) => {
          const display = citationDisplay(view.detail)
          const title = display.main || view.detail.title || view.detail.raw || S.default_source_fallback
          const metaText = [
            display.authors,
            display.venueYear || display.venue,
          ].filter(Boolean).join(' \u00b7 ')
          const key = String((view.candidate as Record<string, unknown>).id || view.detail.anchor || title)
          return (
            <div className="kb-unlinked-ref-row" key={key}>
              <div className="kb-unlinked-ref-main">
                <div className="kb-unlinked-ref-title">{title}</div>
                {metaText ? <div className="kb-unlinked-ref-meta">{metaText}</div> : null}
              </div>
              <span className="kb-unlinked-ref-reason">{view.label}</span>
              <div className="kb-unlinked-ref-actions">
                {canOpenReader && view.detail.sourcePath ? (
                  <button
                    type="button"
                    className="kb-unlinked-ref-action"
                    onClick={() => onOpenReader(view.detail)}
                  >
                    {S.msg_reference_candidate_open || 'Open'}
                  </button>
                ) : null}
                <button
                  type="button"
                  className="kb-unlinked-ref-action is-primary"
                  onClick={() => onAddToShelf(view.detail)}
                >
                  {S.msg_reference_candidate_add || 'Add'}
                </button>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
