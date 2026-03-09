import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import type { CiteDetail } from './citationState'
import { citationDisplay, citationInlineLabel, citeMetricSummary } from './citationState'

interface Props {
  detail: CiteDetail | null
  position: { x: number; y: number } | null
  loading: boolean
  guideLoading: boolean
  inShelf: boolean
  onClose: () => void
  onAddToShelf: (detail: CiteDetail) => void
  onOpenShelf: () => void
  onOpenReader: (detail: CiteDetail) => void
  onStartGuide: (detail: CiteDetail) => void
}

function compact(value: string) {
  return String(value || '').trim()
}

export function CitationPopover({
  detail,
  position,
  loading,
  guideLoading,
  inShelf,
  onClose,
  onAddToShelf,
  onOpenShelf,
  onOpenReader,
  onStartGuide,
}: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const [style, setStyle] = useState<{ left: number; top: number } | null>(null)

  useEffect(() => {
    if (!detail) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    const onPointerDown = (event: MouseEvent) => {
      const el = ref.current
      if (!el) return
      if (event.target instanceof Node && !el.contains(event.target)) onClose()
    }
    document.addEventListener('keydown', onKeyDown)
    document.addEventListener('mousedown', onPointerDown)
    return () => {
      document.removeEventListener('keydown', onKeyDown)
      document.removeEventListener('mousedown', onPointerDown)
    }
  }, [detail, onClose])

  useLayoutEffect(() => {
    if (!detail || !position || !ref.current) {
      setStyle(null)
      return
    }
    const rect = ref.current.getBoundingClientRect()
    const margin = 12
    const maxLeft = Math.max(margin, window.innerWidth - rect.width - margin)
    const maxTop = Math.max(margin, window.innerHeight - rect.height - margin)
    setStyle({
      left: Math.min(Math.max(margin, position.x + 10), maxLeft),
      top: Math.min(Math.max(margin, position.y + 10), maxTop),
    })
  }, [detail, position])

  if (!detail || !position) return null

  const display = citationDisplay(detail)
  const sourceLine = display.source ? `source: ${display.source}` : ''
  const doiLabel = compact(detail.doi) || compact(detail.doiUrl)
  const metrics = citeMetricSummary(detail)
  const inlineLabel = citationInlineLabel(detail)
  const canOpenReader = Boolean(compact(detail.sourcePath))

  return (
    <div
      ref={ref}
      className="kb-cite-pop fixed z-50 w-[440px] max-w-[calc(100vw-20px)]"
      style={style ?? { left: position.x + 10, top: position.y + 10, visibility: 'hidden' }}
    >
      <div className="kb-cite-pop-head">
        <div className="kb-cite-pop-title">[{inlineLabel}]</div>
        <button className="kb-cite-pop-close" onClick={onClose} type="button" aria-label="Close">
          ×
        </button>
      </div>

      <div className="kb-cite-pop-main">{display.main}</div>
      {display.authors ? <div className="kb-cite-pop-sub">{display.authors}</div> : null}
      {sourceLine ? <div className="kb-cite-pop-sub">{sourceLine}</div> : null}
      {loading ? <div className="kb-cite-pop-sub">正在拉取文献指标...</div> : null}
      {!loading && metrics.length > 0 ? (
        <div className="kb-cite-pop-metrics">
          {metrics.map((item) => (
            <span key={item} className="kb-cite-pop-metric">{item}</span>
          ))}
        </div>
      ) : null}
      {detail.doiUrl ? (
        <div className="kb-cite-pop-doi">
          DOI: <a href={detail.doiUrl} rel="noreferrer" target="_blank">{doiLabel}</a>
        </div>
      ) : null}

      <div className="kb-cite-pop-actions">
        <button className="kb-cite-pop-open-shelf" type="button" onClick={onOpenShelf}>
          打开文献篮
        </button>
        <button
          className="kb-cite-pop-open-shelf"
          type="button"
          disabled={!canOpenReader}
          onClick={() => onOpenReader(detail)}
        >
          阅读定位
        </button>
        <button
          className="kb-cite-pop-open-shelf"
          type="button"
          onClick={() => onStartGuide(detail)}
          disabled={guideLoading}
        >
          {guideLoading ? '进入中...' : '围绕此文阅读'}
        </button>
        <button
          className={`kb-cite-pop-add ${inShelf ? 'kb-added' : ''}`}
          type="button"
          onClick={() => onAddToShelf(detail)}
        >
          {inShelf ? '已加入文献篮' : '加入文献篮'}
        </button>
      </div>
    </div>
  )
}
