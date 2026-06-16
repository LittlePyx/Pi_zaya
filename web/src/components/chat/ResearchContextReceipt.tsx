import { useState } from 'react'
import type { ReaderOpenPayload } from './reader/readerTypes'
import type { SelectedResearchContextItem, SelectedResearchContextPack } from './researchContextPack'
import { contextItemMeta, contextItemTitle } from './messageTraceUtils'

export function ResearchContextReceipt({
  pack,
  onOpenReader,
  onFollowUp,
  S,
}: {
  pack?: SelectedResearchContextPack | null
  onOpenReader?: (payload: ReaderOpenPayload) => void
  onFollowUp?: (pack: SelectedResearchContextPack, item: SelectedResearchContextItem) => void
  S: Record<string, string>
}) {
  const [expanded, setExpanded] = useState(false)
  if (!pack || pack.items.length <= 0) return null
  const preview = pack.items
    .slice(0, 2)
    .map((item) => contextItemTitle(item, S.default_source_fallback || 'Untitled'))
    .filter(Boolean)
    .join(' / ')
  return (
    <div className={`kb-research-context-receipt ${expanded ? 'is-expanded' : ''}`} data-testid="research-context-receipt">
      <button
        type="button"
        className="kb-research-context-receipt-head"
        aria-expanded={expanded}
        onClick={() => setExpanded((value) => !value)}
        data-testid="research-context-receipt-toggle"
      >
        <span className="kb-research-context-receipt-title">
          {(S.research_context_receipt_title || 'Used research context').replace('{n}', String(pack.items.length))}
        </span>
        <span className="kb-research-context-receipt-preview">
          {preview || (S.research_context_receipt_preview || 'Selected excerpts from the research basket')}
        </span>
        <span className="kb-research-context-receipt-toggle-text">
          {expanded ? (S.research_context_receipt_collapse || 'Collapse') : (S.research_context_receipt_expand || 'Details')}
        </span>
      </button>
      {expanded ? (
        <div className="kb-research-context-receipt-list">
          {pack.items.map((item, idx) => {
            const title = contextItemTitle(item, S.default_source_fallback || 'Untitled')
            const meta = contextItemMeta(item)
            const location = String(item.locationLabel || '').trim()
            const body = String(item.summary || item.excerpt || '').trim()
            const secondary = item.summary && item.excerpt && item.summary !== item.excerpt ? item.excerpt : ''
            const canOpen = Boolean(onOpenReader && item.sourcePath)
            const canFollowUp = Boolean(onFollowUp)
            return (
              <div className="kb-research-context-receipt-item" key={`${item.key || title}-${idx}`} data-testid="research-context-receipt-item">
                <div className="kb-research-context-receipt-item-main">
                  <div className="kb-research-context-receipt-item-title">{title}</div>
                  {meta ? <div className="kb-research-context-receipt-item-meta">{meta}</div> : null}
                  {location ? (
                    <div className="kb-research-context-receipt-location">
                      <span>{S.research_context_receipt_location || 'Location'}</span>
                      <strong>{location}</strong>
                    </div>
                  ) : null}
                  {body ? (
                    <div className="kb-research-context-receipt-body">
                      {body}
                    </div>
                  ) : null}
                  {secondary ? (
                    <div className="kb-research-context-receipt-excerpt">
                      {secondary}
                    </div>
                  ) : null}
                  {item.note ? (
                    <div className="kb-research-context-receipt-note">
                      <span>{S.research_context_receipt_note || 'Note'}</span>
                      {item.note}
                    </div>
                  ) : null}
                  {item.doi ? <div className="kb-research-context-receipt-doi">DOI {item.doi}</div> : null}
                </div>
                {canOpen || canFollowUp ? (
                  <div className="kb-research-context-receipt-actions">
                    {canFollowUp ? (
                      <button
                        type="button"
                        className="kb-research-context-receipt-follow"
                        data-testid="research-context-receipt-followup"
                        onClick={() => onFollowUp?.(pack, item)}
                      >
                        {S.research_context_receipt_followup || 'Ask follow-up'}
                      </button>
                    ) : null}
                    {canOpen ? (
                      <button
                        type="button"
                        className="kb-research-context-receipt-open"
                        onClick={() => {
                          onOpenReader?.({
                            sourcePath: item.sourcePath,
                            sourceName: item.sourceName || pack.guideSourceName || item.sourcePath.split(/[\\/]/).pop() || '',
                            headingPath: location,
                            snippet: item.excerpt || item.summary || title,
                            highlightSnippet: item.excerpt || item.summary || title,
                          })
                        }}
                      >
                        {S.research_context_receipt_open || 'Open'}
                      </button>
                    ) : null}
                  </div>
                ) : null}
              </div>
            )
          })}
        </div>
      ) : null}
    </div>
  )
}
