import { useEffect, useRef } from 'react'
import type { ReaderSessionHighlight } from './readerTypes'

interface ReaderHighlightsPanelProps {
  items: ReaderSessionHighlight[]
  activeItemId: string
  onSelectItem: (item: ReaderSessionHighlight) => void
  onRemoveItem: (highlightId: string) => void
  titleLabel: string
  removeLabel: string
  usefulLabel: string
  checkLabel: string
}

function highlightExcerpt(text: string, maxLen = 120): string {
  const raw = String(text || '').replace(/\s+/g, ' ').trim()
  if (!raw) return 'Untitled highlight'
  if (raw.length <= maxLen) return raw
  return `${raw.slice(0, Math.max(36, maxLen - 3)).trimEnd()}...`
}

function highlightLocation(item: ReaderSessionHighlight): string {
  const heading = String(item.headingPath || '').replace(/\s+/g, ' ').trim()
  if (heading) {
    return heading.split('/').map((part) => part.trim()).filter(Boolean).slice(-2).join(' / ')
  }
  return String(item.blockId || item.anchorId || '').replace(/\s+/g, ' ').trim()
}

export function ReaderHighlightsPanel({
  items,
  activeItemId,
  onSelectItem,
  onRemoveItem,
  titleLabel,
  removeLabel,
  usefulLabel,
  checkLabel,
}: ReaderHighlightsPanelProps) {
  const activeButtonRef = useRef<HTMLButtonElement | null>(null)

  useEffect(() => {
    activeButtonRef.current?.scrollIntoView({ block: 'nearest' })
  }, [activeItemId])

  return (
    <div className="kb-reader-highlights-panel" data-testid="reader-highlights-panel">
      <div className="kb-reader-highlights-head">
        <div className="kb-reader-highlights-title">{titleLabel}</div>
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
                {highlightLocation(item) ? (
                  <span className="kb-reader-highlight-item-meta">{highlightLocation(item)}</span>
                ) : null}
                {item.feedback === 'useful' || item.feedback === 'needs_check' ? (
                  <span className={`kb-reader-highlight-feedback is-${item.feedback}`}>
                    {item.feedback === 'useful' ? usefulLabel : checkLabel}
                  </span>
                ) : null}
              </button>
              <button
                type="button"
                className="kb-reader-highlight-remove"
                title={removeLabel}
                onClick={() => onRemoveItem(item.id)}
                aria-label={removeLabel}
                data-testid={`reader-highlight-remove-${index}`}
              >
                ×
              </button>
            </div>
          )
        })}
      </div>
    </div>
  )
}
