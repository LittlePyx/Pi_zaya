import { useEffect, useLayoutEffect, useRef, useState } from 'react'
/* eslint-disable react-hooks/set-state-in-effect */

import type { CiteDetail } from './citationState'
import {
  citationDisplay,
  citationInlineLabel,
  citeMetricSummary,
  cleanCitationDisplayText,
  looksLowValueCitationContext,
} from './citationState'

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

function substantiallySame(left: string, right: string) {
  const a = compact(left).replace(/\s+/g, ' ').toLowerCase()
  const b = compact(right).replace(/\s+/g, ' ').toLowerCase()
  if (!a || !b) return false
  if (a === b) return true
  if (a.length >= 36 && b.includes(a)) return true
  if (b.length >= 36 && a.includes(b)) return true
  const aTokens = new Set(a.match(/[a-z0-9\u4e00-\u9fff]{2,}/g) || [])
  const bTokens = new Set(b.match(/[a-z0-9\u4e00-\u9fff]{2,}/g) || [])
  if (aTokens.size < 6 || bTokens.size < 6) return false
  let overlap = 0
  for (const token of aTokens) {
    if (bTokens.has(token)) overlap += 1
  }
  return overlap / Math.min(aTokens.size, bTokens.size) >= 0.82
}

function comparablePaperLabel(value: string): string {
  const raw = compact(value)
  if (!raw) return ''
  const leaf = raw.replace(/\\/g, '/').split('/').pop() || raw
  return leaf
    .replace(/\.pdf$/i, '')
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase()
}

function isOnlyPaperLabel(value: string, candidates: string[]): boolean {
  const text = compact(value)
  const normalized = comparablePaperLabel(text)
  if (!text || !normalized) return false
  for (const candidate of candidates) {
    const candidateText = compact(candidate)
    const candidateNormalized = comparablePaperLabel(candidateText)
    if (!candidateText || !candidateNormalized) continue
    if (normalized === candidateNormalized) return true
    if (substantiallySame(text, candidateText)) return true
  }
  return false
}

function isLowValueSystemAClaim(value: string): boolean {
  const text = compact(value).replace(/\[[Rr]?\d{1,4}]/g, '').replace(/\s+/g, ' ')
  if (!text || text.length < 18) return true
  const tokens = text.match(/[A-Za-z0-9\u4e00-\u9fff]+/g) || []
  const hasCjk = /[\u4e00-\u9fff]/.test(text)
  if (!hasCjk && tokens.length <= 4) return true
  if (/^[A-Za-z][A-Za-z\s-]{2,48}\s+\d{1,3}$/.test(text)) return true
  const hasSentenceCue = /[：:，,。.!?；;]/.test(text)
  if (hasCjk && text.length < 24 && !hasSentenceCue) return true
  if (!hasCjk && tokens.length <= 6 && !hasSentenceCue) return true
  return false
}

function pageRangeLabel(start: number, end: number): string {
  const p0 = Number(start || 0)
  const p1 = Number(end || 0)
  if (!Number.isFinite(p0) || p0 <= 0) return ''
  if (!Number.isFinite(p1) || p1 <= 0 || p1 === p0) return `p. ${Math.floor(p0)}`
  return `pp. ${Math.floor(Math.min(p0, p1))}-${Math.floor(Math.max(p0, p1))}`
}

function anchorKindLabel(value: string): string {
  const key = compact(value).toLowerCase()
  if (key === 'sentence') return '句子'
  if (key === 'paragraph') return '段落'
  if (key === 'equation') return '公式'
  if (key === 'figure') return '图'
  if (key === 'table') return '表'
  return compact(value)
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
  const kindLabel = isSystemB ? '上游引用' : '答案依据'
  const displayNums = Array.from(new Set([
    ...(Array.isArray(detail.linkedNums) ? detail.linkedNums : []),
    detail.num,
  ].map((num) => Number(num || 0)).filter((num) => Number.isFinite(num) && num > 0))).sort((a, b) => a - b)
  const badgeNumText = displayNums.length > 1 ? displayNums.join('/') : String(displayNums[0] || '')
  const badgeLabel = badgeNumText ? (isSystemB ? `[R${badgeNumText}]` : `#${badgeNumText}`) : inlineLabel
  const headingPath = compact(detail.headingPath) || (!isSystemB ? compact(detail.title) : '')
  const pageLabel = pageRangeLabel(detail.pageStart, detail.pageEnd)
  const sourcePaperText = compact(detail.sourceName) || compact(display.source)
  const cardTitle = compact(detail.cardTitle)
  const cardSubtitle = compact(detail.cardSubtitle)
  const headerSubtitle = isSystemB ? cardSubtitle : ''
  const cardTakeawayLabel = compact(detail.cardTakeawayLabel)
  const cardTakeaway = compact(detail.cardTakeaway)
  const cardClaimLabel = compact(detail.cardClaimLabel)
  const cardEvidenceLabel = compact(detail.cardEvidenceLabel)
  const cardLocatorLabel = compact(detail.cardLocatorLabel)
  const cardReferenceLabel = compact(detail.cardReferenceLabel)
  const cardSupportLabel = compact(detail.cardSupportLabel)
  const cardWarning = compact(detail.cardWarning)
  const externalMetadataStatus = compact(detail.externalMetadataStatus).toLowerCase()
  const externalMetadataReason = compact(detail.externalMetadataReason)
  const externalTitle = compact(detail.externalTitle)
  const cardQualityLabel = compact(detail.cardQualityLabel)
  const cardQualityScore = Number(detail.cardQualityScore || 0)
  const cardQualityFlags = Array.isArray(detail.cardQualityFlags)
    ? detail.cardQualityFlags.map((item) => compact(item)).filter(Boolean)
    : []
  const cardFlow = Array.isArray(detail.cardFlow)
    ? detail.cardFlow.map((item) => compact(item)).filter(Boolean)
    : []
  const systemAClaimText = cleanCitationDisplayText(compact(detail.cardClaim) || compact(detail.answerClaim))
  const suppressRawSystemAEvidenceFallback = !isSystemB
    && (
      cardQualityFlags.includes('evidence_quote_filtered')
      || cardQualityFlags.includes('missing_evidence_quote')
    )
  const systemAEvidenceText = cleanCitationDisplayText(detail.cardEvidence)
    || (!suppressRawSystemAEvidenceFallback ? cleanCitationDisplayText(detail.evidenceQuote) : '')
    || (!suppressRawSystemAEvidenceFallback ? cleanCitationDisplayText(detail.summaryLine) : '')
    || (!suppressRawSystemAEvidenceFallback ? cleanCitationDisplayText(detail.raw) : '')
    || (!suppressRawSystemAEvidenceFallback ? cleanCitationDisplayText(detail.citeFmt) : '')
  const systemATakeawayText = !isSystemB && cardTakeaway && !substantiallySame(cardTakeaway, systemAEvidenceText)
    ? cardTakeaway
    : ''
  const systemBExplicitReferenceText = cleanCitationDisplayText(detail.cardReferenceEntry)
  const systemBReferenceText = systemBExplicitReferenceText || cleanCitationDisplayText(compact(detail.raw) || compact(detail.citeFmt))
  const systemBCardEvidenceText = cleanCitationDisplayText(detail.cardEvidence)
  const systemBRawContextCandidate = cleanCitationDisplayText(
    compact(detail.citationContext) || compact(detail.evidenceQuote) || compact(detail.summaryLine),
  )
  const suppressRawSystemBContextFallback = isSystemB
    && (
      cardQualityFlags.includes('weak_citation_context')
      || cardQualityFlags.includes('missing_citation_context')
    )
  const systemBRawContextIsLowValue = Boolean(
    systemBRawContextCandidate && looksLowValueCitationContext(systemBRawContextCandidate),
  )
  const systemBCitationContextText = systemBCardEvidenceText
    || ((!suppressRawSystemBContextFallback && !systemBRawContextIsLowValue) ? systemBRawContextCandidate : '')
  const systemBCitationContextLabel = cardEvidenceLabel || '引用语境'
  const systemBTakeawayText = isSystemB && cardTakeaway && !substantiallySame(cardTakeaway, systemBCitationContextText)
    ? cardTakeaway
    : ''
  const whyText = compact(detail.whyLine)
  const bindingStatus = compact(detail.bindingStatus).toLowerCase()
  const bindingReason = compact(detail.bindingReason)
  const bindingOverlapText = Array.isArray(detail.bindingOverlapTerms)
    ? detail.bindingOverlapTerms.map((item) => compact(item)).filter(Boolean).join(' / ')
    : ''
  const bindingState = !isSystemB && bindingStatus && bindingStatus !== 'grounded'
    ? (
        bindingStatus === 'mismatch'
            ? { label: '疑似错配', tone: 'mismatch' }
            : { label: '候选依据', tone: 'candidate' }
      )
    : null
  const explicitSupportText = compact(detail.cardSupportExplanation)
    || compact(detail.supportRelation)
    || whyText
    || bindingReason
  const supportBaseText = isSystemB
    ? (explicitSupportText || '这条链接把回答中的说法追溯到当前论文引用的上游文献。')
    : (explicitSupportText || (bindingStatus === 'candidate'
      ? '这条引用只能作为候选依据；请打开原文核对答案句和命中片段是否真正对应。'
      : ''))
  const supportText = supportBaseText
  const showBindingReason = Boolean(bindingReason && !substantiallySame(bindingReason, supportText))
  const displayMain = compact(display.main)
  const systemATitle = cardTitle || ((displayMain && displayMain !== headingPath)
    ? displayMain
    : (compact(detail.sourceName) || compact(display.source) || displayMain))
  const systemBTitleMissing = !cardTitle && !compact(detail.title)
  const systemBTitle = cardTitle || compact(detail.title) || '上游参考文献'
  const systemASub = [headingPath, pageLabel].filter(Boolean).join(' · ')
  const systemALocationText = compact(detail.cardLocator) || compact(detail.locationLabel) || systemASub || systemATitle
  const systemAAnchorText = anchorKindLabel(detail.anchorKind)
  const systemAHasReviewRisk = Boolean(bindingState || cardWarning || cardQualityFlags.includes('candidate_binding') || cardQualityFlags.includes('binding_mismatch'))
  const systemAHasOccurrenceClaim = cardQualityFlags.includes('occurrence_specific_claim')
  const systemAClaimLooksUseful = !isLowValueSystemAClaim(systemAClaimText)
  const showSystemAClaim = Boolean(
    systemAClaimText
    && systemAClaimLooksUseful
    && (!systemAEvidenceText || ((systemAHasReviewRisk || systemAHasOccurrenceClaim) && !substantiallySame(systemAClaimText, systemAEvidenceText))),
  )
  const showSystemATakeaway = Boolean(
    systemATakeawayText
    && !(showSystemAClaim && substantiallySame(systemATakeawayText, systemAClaimText)),
  )
  const showSystemASupport = Boolean(
    systemAHasReviewRisk
    &&
    supportText
    && !substantiallySame(supportText, systemAEvidenceText)
    && !substantiallySame(supportText, systemAClaimText),
  )
  const primaryActionLabel = isSystemB ? '打开引用语境' : '打开答案依据'
  const explainText = ''
  const flowSteps = isSystemB ? [] : cardFlow
  const rawSystemBLocationText = compact(detail.cardLocator) || compact(detail.locationLabel) || [sourcePaperText, headingPath, pageLabel].filter(Boolean).join(' / ')
  const systemBLocationIsPaperOnly = isOnlyPaperLabel(rawSystemBLocationText, [
    sourcePaperText,
    detail.sourceName,
    display.source,
  ])
  const systemBLocationLabel = systemBLocationIsPaperOnly ? '引用所在论文' : '引用出现位置'
  const systemBLocationText = systemBLocationIsPaperOnly
    ? (sourcePaperText || rawSystemBLocationText)
    : rawSystemBLocationText
  const systemBLocationHint = systemBLocationIsPaperOnly
    ? '只定位到引用出现的论文，尚未定位到具体章节或页码；可打开引用语境核对。'
    : ''
  const showSystemBLocation = Boolean(systemBLocationText)
  const systemBSupportText = isSystemB
    && explicitSupportText
    && !substantiallySame(explicitSupportText, systemBCitationContextText)
    && !substantiallySame(explicitSupportText, systemBReferenceText)
    ? explicitSupportText
    : ''
  const showSystemBSupport = Boolean(
    systemBSupportText
    && (
      cardQualityFlags.includes('reference_entry_only')
      || !systemBCitationContextText
      || cardWarning
    ),
  )
  const hasSystemBHeaderIdentity = Boolean(
    (systemBTitle && systemBTitle !== '上游参考文献')
    || headerSubtitle
    || doiLabel
    || metrics.length > 0
  )
  const showSystemBReference = Boolean(
    systemBReferenceText
    && (
      systemBTitleMissing
      || cardQualityFlags.includes('missing_reference_title')
      || (cardQualityFlags.includes('reference_entry_only') && !hasSystemBHeaderIdentity)
    ),
  )
  const metaRows = [
    display.source ? { label: '来源', value: display.source } : null,
    display.venueYear ? { label: '发表', value: display.venueYear } : null,
  ].filter(Boolean) as Array<{ label: string; value: string }>
  const showMetaGrid = Boolean(!isSystemB && (metaRows.length > 0 || doiLabel))
  const showMetrics = Boolean(metrics.length > 0 || (isSystemB && doiLabel))
  const showCardQuality = Boolean(
    cardQualityLabel
    && (cardWarning || systemAHasReviewRisk || cardQualityScore < 0.62),
  )
  const showExternalMetadataWarning = externalMetadataStatus === 'candidate' || externalMetadataStatus === 'conflict'
  const externalMetadataWarningText = showExternalMetadataWarning
    ? (externalMetadataReason || '外部元数据与原参考条目仍需核对，已优先保留原参考条目；DOI、被引和期刊指标仅作线索。')
    : ''
  const externalMetadataTitleHint = externalTitle && !substantiallySame(externalTitle, displayMain)
    ? `候选外部标题：${externalTitle}`
    : ''

  return (
    <div
      ref={ref}
      className={`kb-cite-pop ${isSystemB ? 'kb-cite-pop-system-b' : 'kb-cite-pop-system-a'} fixed z-50 w-[520px] max-w-[calc(100vw-20px)]`}
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
          <div className="kb-cite-pop-title">{isSystemB ? systemBTitle : systemATitle}</div>
          {headerSubtitle ? <div className="kb-cite-pop-title-sub">{headerSubtitle}</div> : null}
        </div>
        <button className="kb-cite-pop-close" onClick={onClose} type="button" aria-label="Close">
          ×
        </button>
      </div>

      {explainText ? <div className="kb-cite-pop-explain" data-testid="citation-popover-explain">{explainText}</div> : null}
      {flowSteps.length > 0 ? (
        <div className="kb-cite-pop-flow" data-testid="citation-popover-flow" aria-label="引用定位路径">
          {flowSteps.map((step, index) => (
            <div className="kb-cite-pop-flow-piece" key={step}>
              <span className="kb-cite-pop-flow-step">{step}</span>
              {index < flowSteps.length - 1 ? <span className="kb-cite-pop-flow-arrow">→</span> : null}
            </div>
          ))}
        </div>
      ) : null}
      {bindingState ? (
        <div
          className={`kb-cite-pop-binding kb-cite-pop-binding-${bindingState.tone}`}
          data-testid="citation-popover-binding-status"
        >
          <span className="kb-cite-pop-binding-label">{bindingState.label}</span>
          {bindingOverlapText ? <span className="kb-cite-pop-binding-terms">{bindingOverlapText}</span> : null}
          {showBindingReason ? <span className="kb-cite-pop-binding-reason">{bindingReason}</span> : null}
        </div>
      ) : null}
      {showCardQuality ? (
        <div
          className="kb-cite-pop-quality"
          data-testid="citation-popover-card-quality"
          title={cardQualityFlags.join(' / ')}
        >
          <span className="kb-cite-pop-quality-label">{cardQualityLabel}</span>
          {cardQualityScore > 0 ? <span className="kb-cite-pop-quality-score">{Math.round(cardQualityScore * 100)}%</span> : null}
        </div>
      ) : null}
      {cardWarning ? (
        <div className="kb-cite-pop-warning" data-testid="citation-popover-card-warning">
          {cardWarning}
        </div>
      ) : null}
      {showExternalMetadataWarning ? (
        <div className="kb-cite-pop-warning" data-testid="citation-popover-external-metadata-warning">
          {externalMetadataWarningText}
          {externalMetadataTitleHint ? <span className="kb-cite-pop-warning-sub">{externalMetadataTitleHint}</span> : null}
        </div>
      ) : null}
      {!isSystemB ? (
        <div className="kb-cite-pop-evidence-map">
          {showSystemATakeaway ? (
            <div className="kb-cite-pop-takeaway" data-testid="citation-popover-system-a-takeaway">
              <span className="kb-cite-pop-section-title">{cardTakeawayLabel || '证据重点'}</span>
              <div className="kb-cite-pop-main">{systemATakeawayText}</div>
            </div>
          ) : null}
          {systemAEvidenceText ? (
            <div className="kb-cite-pop-quote" data-testid="citation-popover-system-a-evidence">
              <span className="kb-cite-pop-section-title">{cardEvidenceLabel || '原文证据'}</span>
              <blockquote>{systemAEvidenceText}</blockquote>
            </div>
          ) : null}
          {showSystemAClaim ? (
            <div className="kb-cite-pop-claim" data-testid="citation-popover-system-a-claim">
              <span className="kb-cite-pop-section-title">{cardClaimLabel || '对应回答'}</span>
              <div className="kb-cite-pop-main">{systemAClaimText}</div>
            </div>
          ) : null}
          <div className="kb-cite-pop-locator" data-testid="citation-popover-system-a-location">
            <span className="kb-cite-pop-section-title">{cardLocatorLabel || '位置'}</span>
            <span className="kb-cite-pop-locator-text">{systemALocationText}</span>
            {systemAAnchorText ? <span className="kb-cite-pop-anchor-meta">{systemAAnchorText}</span> : null}
          </div>
          {showSystemASupport ? (
            <div className="kb-cite-pop-why" data-testid="citation-popover-system-a-support">
              <span className="kb-cite-pop-section-title">{cardSupportLabel || '可靠度'}</span>
              <div className="kb-cite-pop-main">{supportText}</div>
            </div>
          ) : null}
        </div>
      ) : (
        <div className="kb-cite-pop-evidence-map">
          {systemBTakeawayText ? (
            <div className="kb-cite-pop-takeaway" data-testid="citation-popover-system-b-takeaway">
              <span className="kb-cite-pop-section-title">{cardTakeawayLabel || '上游作用'}</span>
              <div className="kb-cite-pop-main">{systemBTakeawayText}</div>
            </div>
          ) : null}
          {systemBCitationContextText ? (
            <div className="kb-cite-pop-quote" data-testid="citation-popover-system-b-context">
              <span className="kb-cite-pop-section-title">{systemBCitationContextLabel}</span>
              <blockquote>{systemBCitationContextText}</blockquote>
            </div>
          ) : null}
          {showSystemBLocation ? (
            <div className="kb-cite-pop-locator" data-testid="citation-popover-system-b-location">
              <span className="kb-cite-pop-section-title">{systemBLocationLabel}</span>
              <span className="kb-cite-pop-locator-text">{systemBLocationText}</span>
              {systemBLocationHint ? <span className="kb-cite-pop-anchor-meta">{systemBLocationHint}</span> : null}
            </div>
          ) : null}
          {showSystemBReference ? (
            <div className="kb-cite-pop-evidence" data-testid="citation-popover-system-b-reference">
              <div className="kb-cite-pop-section-title">{cardReferenceLabel || '上游文献条目'}</div>
              <div className="kb-cite-pop-main">{systemBReferenceText}</div>
            </div>
          ) : null}
          {showSystemBSupport ? (
            <div className="kb-cite-pop-why" data-testid="citation-popover-system-b-support">
              <span className="kb-cite-pop-section-title">{cardSupportLabel || '说明'}</span>
              <div className="kb-cite-pop-main">{systemBSupportText}</div>
            </div>
          ) : null}
        </div>
      )}
      {showMetaGrid ? (
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
      {loading && !isSystemB ? <div className="kb-cite-pop-sub">{S.cite_loading}</div> : null}
      {!loading && showMetrics ? (
        <div className="kb-cite-pop-metrics">
          {isSystemB && doiLabel ? (
            doiHref ? (
              <a className="kb-cite-pop-metric kb-cite-pop-link" href={doiHref} rel="noreferrer" target="_blank">
                DOI {doiLabel}
              </a>
            ) : (
              <span className="kb-cite-pop-metric">DOI {doiLabel}</span>
            )
          ) : null}
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
