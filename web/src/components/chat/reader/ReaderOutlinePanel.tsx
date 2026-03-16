import { useEffect, useRef } from 'react'
import type { ReaderOutlineItem } from './useReaderOutline'

interface ReaderOutlinePanelProps {
  items: ReaderOutlineItem[]
  activeItemId: string
  onSelectItem: (item: ReaderOutlineItem) => void
}

export function ReaderOutlinePanel({
  items,
  activeItemId,
  onSelectItem,
}: ReaderOutlinePanelProps) {
  const activeButtonRef = useRef<HTMLButtonElement | null>(null)

  useEffect(() => {
    activeButtonRef.current?.scrollIntoView({ block: 'nearest' })
  }, [activeItemId])

  return (
    <div className="kb-reader-outline-panel" data-testid="reader-outline-panel">
      <div className="kb-reader-outline-head">
        <div className="kb-reader-outline-title">Sections</div>
        <div className="kb-reader-outline-count">{items.length}</div>
      </div>
      <div className="kb-reader-outline-list">
        {items.map((item, index) => {
          const isActive = item.id === activeItemId
          return (
            <button
              key={item.id}
              ref={isActive ? activeButtonRef : null}
              type="button"
              className={`kb-reader-outline-item ${isActive ? 'is-active' : ''}`}
              style={{ ['--kb-outline-depth' as string]: item.depth }}
              title={item.headingPath}
              onClick={() => onSelectItem(item)}
              data-testid={`reader-outline-item-${index}`}
            >
              <span className="kb-reader-outline-item-label">{item.label}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}
