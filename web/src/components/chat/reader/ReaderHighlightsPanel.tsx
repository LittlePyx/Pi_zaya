import { useEffect, useRef } from 'react'
import type { ReaderSessionHighlight } from './readerTypes'

interface ReaderHighlightsPanelProps {
  items: ReaderSessionHighlight[]
  activeItemId: string
  onSelectItem: (item: ReaderSessionHighlight) => void
  onRemoveItem: (highlightId: string) => void
}

function highlightExcerpt(text: string, maxLen = 120): string {
  const raw = String(text || '').replace(/\s+/g, ' ').trim()
  if (!raw) return 'Untitled highlight'
  if (raw.length <= maxLen) return raw
  return `${raw.slice(0, Math.max(36, maxLen - 1)).trimEnd()}…`
}

export function ReaderHighlightsPanel({
  items,
  activeItemId,
  onSelectItem,
  onRemoveItem,
}: ReaderHighlightsPanelProps) {
  const activeButtonRef = useRef<HTMLButtonElement | null>(null)

  useEffect(() => {
    activeButtonRef.current?.scrollIntoView({ block: 'nearest' })
  }, [activeItemId])

  return (
    <div className="kb-reader-highlights-panel" data-testid="reader-highlights-panel">
      <div className="kb-reader-highlights-head">
        <div className="kb-reader-highlights-title">Highlights</div>
        <div className="kb-reader-highlights-count">{items.length}</div>
      </div>
      <div className="kb-reader-highlights-list">
        {items.map((item, index) => {
          const isActive = item.id === activeItemId
          return (
            <div
              key={item.id}
              className={`kb-reader-highlight-row ${isActive ? 'is-active' : ''}`}
            >
              <button
                ref={isActive ? activeButtonRef : null}
                type="button"
                className={`kb-reader-highlight-item ${isActive ? 'is-active' : ''}`}
                title={item.text}
                onClick={() => onSelectItem(item)}
                data-testid={`reader-highlight-item-${index}`}
              >
                <span className="kb-reader-highlight-item-label">{highlightExcerpt(item.text)}</span>
              </button>
              <button
                type="button"
                className="kb-reader-highlight-remove"
                title="Remove highlight"
                onClick={() => onRemoveItem(item.id)}
                data-testid={`reader-highlight-remove-${index}`}
              >
                Remove
              </button>
            </div>
          )
        })}
      </div>
    </div>
  )
}
