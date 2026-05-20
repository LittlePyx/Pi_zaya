import { useEffect, useLayoutEffect, useRef, useState } from 'react'
/* eslint-disable react-hooks/set-state-in-effect */

import type { CiteDetail } from './citationState'
import { citationDisplay, citationInlineLabel, citeMetricSummary } from './citationState'

import { useT } from '../../i18n'

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
  onMouseEnter?: () => void
  onMouseLeave?: () => void
}

function compact(value: string) {
  return String(value || '').trim()
}

function sameCompact(left: string, right: string) {
  const a = compact(left).replace(/\s+/g, ' ').toLowerCase()
  const b = compact(right).replace(/\s+/g, ' ').toLowerCase()
  return Boolean(a && b && a === b)
}

function pageRangeLabel(start: number, end: number): string {
  const p0 = Number(start || 0)
  const p1 = Number(end || 0)
  if (!Number.isFinite(p0) || p0 <= 0) return ''
  if (!Number.isFinite(p1) || p1 <= 0 || p1 === p0) return `p. ${Math.floor(p0)}`
  return `pp. ${Math.floor(Math.min(p0, p1))}-${Math.floor(Math.max(p0, p1))}`
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
  onMouseEnter,
  onMouseLeave,
}: Props) {
  const S = useT()
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
  const doiLabel = compact(detail.doi) || compact(detail.doiUrl)
  const doiHref = compact(detail.doiUrl) || (doiLabel ? `https://doi.org/${doiLabel}` : '')
  const metrics = citeMetricSummary(detail)
  const inlineLabel = citationInlineLabel(detail)
  const canOpenReader = Boolean(compact(detail.sourcePath))
  const isSystemB = Boolean(detail.isInpaper)
  const kindLabel = isSystemB ? '文内参考' : '原文证据'
  const badgeLabel = detail.num > 0 ? (isSystemB ? `[R${detail.num}]` : `#${detail.num}`) : inlineLabel
  const headingPath = compact(detail.headingPath) || (!isSystemB ? compact(detail.title) : '')
  const pageLabel = pageRangeLabel(detail.pageStart, detail.pageEnd)
  const systemAClaimText = compact(detail.answerClaim)
  const systemAEvidenceText = compact(detail.evidenceQuote) || compact(detail.summaryLine) || compact(detail.raw) || compact(detail.citeFmt)
  const systemBReferenceText = compact(detail.raw) || compact(detail.citeFmt)
  const systemBClaimText = compact(detail.answerClaim)
  const systemBCitationContextText = compact(detail.citationContext) || compact(detail.evidenceQuote) || compact(detail.summaryLine)
  const systemBRoleText = compact(detail.upstreamWorkRole) || compact(detail.whyLine)
  const systemBRelationText = compact(detail.userQuestionRelation) || compact(detail.supportRelation)
  const whyText = compact(detail.whyLine)
  const supportText = compact(detail.supportRelation) || whyText || '这条编号对应回答中使用的检索命中；打开后可以核对原文语境、章节位置和具体句子。'
  const displayMain = compact(display.main)
  const systemATitle = (displayMain && displayMain !== headingPath)
    ? displayMain
    : (compact(detail.sourceName) || compact(display.source) || displayMain)
  const systemASub = [headingPath, pageLabel].filter(Boolean).join(' · ')
  const systemALocationText = compact(detail.locationLabel) || systemASub || systemATitle
  const systemAAnchorText = [compact(detail.anchorKind), compact(detail.blockId)].filter(Boolean).join(' · ')
  const showSystemAClaim = Boolean(systemAClaimText && !sameCompact(systemAClaimText, systemAEvidenceText))
  const primaryActionLabel = isSystemB ? '打开参考所在位置' : '打开原文证据'
  const metaRows = [
    display.source ? { label: '来源', value: display.source } : null,
    display.venueYear ? { label: '发表', value: display.venueYear } : null,
  ].filter(Boolean) as Array<{ label: string; value: string }>

  return (
    <div
      ref={ref}
      className={`kb-cite-pop ${isSystemB ? 'kb-cite-pop-system-b' : 'kb-cite-pop-system-a'} fixed z-50 w-[460px] max-w-[calc(100vw-20px)]`}
      data-testid="citation-popover"
      style={style ?? { left: position.x + 10, top: position.y + 10, visibility: 'hidden' }}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      <div className="kb-cite-pop-head">
        <div className="kb-cite-pop-head-copy">
          <div className="kb-cite-pop-kicker">
            <span className="kb-cite-pop-kind">{kindLabel}</span>
            <span className="kb-cite-pop-badge">{badgeLabel}</span>
          </div>
          <div className="kb-cite-pop-title">{isSystemB ? display.main : systemATitle}</div>
        </div>
        <button className="kb-cite-pop-close" onClick={onClose} type="button" aria-label="Close">
          ×
        </button>
      </div>

      {isSystemB && display.authors ? <div className="kb-cite-pop-sub">{display.authors}</div> : null}
      {!isSystemB ? (
        <div className="kb-cite-pop-evidence-map">
          {showSystemAClaim ? (
            <div className="kb-cite-pop-claim" data-testid="citation-popover-system-a-claim">
              <span className="kb-cite-pop-section-title">回答中的判断</span>
              <div className="kb-cite-pop-main">{systemAClaimText}</div>
            </div>
          ) : null}
          <div className="kb-cite-pop-locator" data-testid="citation-popover-system-a-location">
            <span className="kb-cite-pop-section-title">原文位置</span>
            <span className="kb-cite-pop-locator-text">{systemALocationText}</span>
            {systemAAnchorText ? <span className="kb-cite-pop-anchor-meta">{systemAAnchorText}</span> : null}
          </div>
          {systemAEvidenceText ? (
            <div className="kb-cite-pop-quote" data-testid="citation-popover-system-a-evidence">
              <span className="kb-cite-pop-section-title">命中原文证据</span>
              <blockquote>{systemAEvidenceText}</blockquote>
            </div>
          ) : null}
          <div className="kb-cite-pop-why" data-testid="citation-popover-system-a-support">
            <span className="kb-cite-pop-section-title">为什么链接到这里 / 为什么能支撑</span>
            <div className="kb-cite-pop-main">{supportText}</div>
          </div>
        </div>
      ) : (
        <div className="kb-cite-pop-evidence-map">
          {systemBClaimText ? (
            <div className="kb-cite-pop-claim" data-testid="citation-popover-system-b-claim">
              <span className="kb-cite-pop-section-title">回答中的判断</span>
              <div className="kb-cite-pop-main">{systemBClaimText}</div>
            </div>
          ) : null}
          {systemBCitationContextText ? (
            <div className="kb-cite-pop-quote" data-testid="citation-popover-system-b-context">
              <span className="kb-cite-pop-section-title">当前论文引用语境</span>
              <blockquote>{systemBCitationContextText}</blockquote>
            </div>
          ) : null}
          {systemBRoleText ? (
            <div className="kb-cite-pop-why" data-testid="citation-popover-system-b-role">
              <span className="kb-cite-pop-section-title">上游文献角色</span>
              <div className="kb-cite-pop-main">{systemBRoleText}</div>
            </div>
          ) : null}
          {systemBRelationText && !sameCompact(systemBRelationText, systemBRoleText) ? (
            <div className="kb-cite-pop-why" data-testid="citation-popover-system-b-relation">
              <span className="kb-cite-pop-section-title">为什么与这个问题有关</span>
              <div className="kb-cite-pop-main">{systemBRelationText}</div>
            </div>
          ) : null}
          {systemBReferenceText ? (
            <div className="kb-cite-pop-evidence" data-testid="citation-popover-system-b-reference">
              <div className="kb-cite-pop-section-title">这篇参考文献是什么</div>
              <div className="kb-cite-pop-main">{systemBReferenceText}</div>
            </div>
          ) : null}
        </div>
      )}
      {isSystemB && (metaRows.length > 0 || doiLabel) ? (
        <div className="kb-cite-pop-meta-grid">
          {metaRows.map((item) => (
            <div key={item.label} className="kb-cite-pop-meta-item">
              <span className="kb-cite-pop-meta-label">{item.label}</span>
              <span className="kb-cite-pop-meta-value">{item.value}</span>
            </div>
          ))}
          {doiLabel ? (
            <div className="kb-cite-pop-meta-item">
              <span className="kb-cite-pop-meta-label">DOI</span>
              {doiHref ? (
                <a className="kb-cite-pop-meta-value kb-cite-pop-link" href={doiHref} rel="noreferrer" target="_blank">
                  {doiLabel}
                </a>
              ) : (
                <span className="kb-cite-pop-meta-value">{doiLabel}</span>
              )}
            </div>
          ) : null}
        </div>
      ) : null}
      {loading ? <div className="kb-cite-pop-sub">{S.cite_loading}</div> : null}
      {!loading && metrics.length > 0 ? (
        <div className="kb-cite-pop-metrics">
          {metrics.map((item) => (
            <span key={item} className="kb-cite-pop-metric">{item}</span>
          ))}
        </div>
      ) : null}

      <div className="kb-cite-pop-actions">
        <button
          className="kb-cite-pop-open-shelf kb-cite-pop-action-primary"
          type="button"
          disabled={!canOpenReader}
          onClick={() => onOpenReader(detail)}
        >
          {primaryActionLabel}
        </button>
        <button
          className="kb-cite-pop-open-shelf"
          type="button"
          onClick={() => onStartGuide(detail)}
          disabled={guideLoading}
        >
          {guideLoading ? S.cite_starting_guide : S.cite_start_guide}
        </button>
        <button className="kb-cite-pop-open-shelf" type="button" onClick={onOpenShelf}>
          {S.cite_open_shelf}
        </button>
        <button
          className={`kb-cite-pop-add ${inShelf ? 'kb-added' : ''}`}
          type="button"
          onClick={() => onAddToShelf(detail)}
        >
          {inShelf ? S.cite_in_shelf : S.cite_add_to_shelf}
        </button>
      </div>
    </div>
  )
}
