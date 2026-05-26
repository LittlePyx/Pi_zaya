import { useEffect, useLayoutEffect, useRef, useState } from 'react'
/* eslint-disable react-hooks/set-state-in-effect */

import type { CiteDetail } from './citationState'
import {
  citationCardView,
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
    .replace(/\.en\.md$/i, '')
    .replace(/\.md$/i, '')
    .replace(/\.pdf$/i, '')
    .replace(/^[A-Za-z]{2,12}-\d{4}-/, '')
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

function stripLocationIdentityPrefix(value: string, candidates: string[]): string {
  let text = compact(value)
  if (!text) return ''
  const identities = candidates.map(comparablePaperLabel).filter(Boolean)
  if (!identities.length) return text
  const sameIdentity = (left: string, right: string) => {
    const a = comparablePaperLabel(left)
    const b = comparablePaperLabel(right)
    if (!a || !b) return false
    if (a === b) return true
    if (a.length >= 16 && b.includes(a)) return true
    if (b.length >= 16 && a.includes(b)) return true
    const at = new Set(a.match(/[a-z0-9\u4e00-\u9fff]{2,}/g) || [])
    const bt = new Set(b.match(/[a-z0-9\u4e00-\u9fff]{2,}/g) || [])
    if (at.size < 3 || bt.size < 3) return false
    let overlap = 0
    for (const token of at) {
      if (bt.has(token)) overlap += 1
    }
    return overlap / Math.min(at.size, bt.size) >= 0.82
  }
  const parts = text.split(/\s*\/\s*/).map((part) => compact(part)).filter(Boolean)
  while (parts.length > 1 && identities.some((candidate) => sameIdentity(parts[0], candidate))) {
    parts.shift()
  }
  if (parts.length > 0 && parts.join(' / ') !== text) return parts.join(' / ')
  if (identities.some((candidate) => sameIdentity(text, candidate))) return ''
  for (const raw of candidates) {
    const candidate = compact(raw)
    if (candidate.length < 10) continue
    const escaped = candidate.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    const next = text.replace(new RegExp(`^\\s*${escaped}\\s*(?:/|·|-|—|:|：)\\s*`, 'i'), '').trim()
    if (next !== text) return next
  }
  return text
}

function compactIdentity(value: string): string {
  return comparablePaperLabel(value).replace(/[^a-z0-9\u4e00-\u9fff]+/g, ' ').trim()
}

function containsIdentityText(value: string, candidate: string, minLen = 22): boolean {
  const body = compactIdentity(value)
  const ident = compactIdentity(candidate)
  return Boolean(body && ident.length >= minLen && body.includes(ident))
}

function looksNarrativeMetadataText(value: string, detail: CiteDetail): boolean {
  const text = compact(value)
  if (!text) return false
  if (/\b10\.\d{4,9}\/[^\s，。；;,)）]+/i.test(text)) return true
  if (/\b(?:doi|jcr|impact\s*factor|if\s*[:：]?\s*\d|published\s+(?:in|by)|journal|conference|venue|citation\s+count|cited\s+by)\b/i.test(text)) return true
  if (/(?:发表于|发表在|期刊|会议|年份|被引|影响因子|分区|出处|来源论文|论文标题|标题是|作者是)/.test(text)) return true
  if (containsIdentityText(text, detail.title) || containsIdentityText(text, detail.cardTitle) || containsIdentityText(text, detail.sourceName) || containsIdentityText(text, detail.sourcePath)) return true
  const venue = compact(detail.venue)
  if (venue && containsIdentityText(text, venue, 7)) return true
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

function evidencePreview(value: string, maxLen = 260): string {
  const text = compact(value).replace(/\s+/g, ' ')
  if (!text || text.length <= maxLen) return text
  const head = text.slice(0, maxLen).replace(/[，,；;:：]\s*$/g, '').trim()
  return `${head}...`
}

function answerPointPreview(value: string, maxLen = 140): string {
  const text = compact(value)
    .replace(/\s*\[[Rr]?\d{1,4}]\s*/g, ' ')
    .replace(/\s+/g, ' ')
    .replace(/^\s*(?:\d{1,3}[.)、．]|[-*•])\s*/, '')
    .trim()
  if (!text || text.length <= maxLen) return text
  const head = text.slice(0, maxLen)
  const cut = Math.max(
    head.lastIndexOf('。'),
    head.lastIndexOf('！'),
    head.lastIndexOf('？'),
    head.lastIndexOf('；'),
    head.lastIndexOf(';'),
    head.lastIndexOf('，'),
    head.lastIndexOf(','),
  )
  if (cut >= 40) return `${head.slice(0, cut).trim()}...`
  return `${head.slice(0, maxLen - 1).trim()}...`
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
  const view = citationCardView(detail)
  const viewSection = (id: string) => view.sections.find((item) => item.id === id)
  const takeawaySection = viewSection('takeaway')
  const claimSection = viewSection('claim')
  const locatorSection = viewSection('locator')
  const contextSummarySection = viewSection('context_summary')
  const evidenceSection = viewSection('evidence')
  const referenceSection = viewSection('reference')
  const supportSection = viewSection('support')
  const warningSection = viewSection('warning')
  const kindLabel = compact(view.header.kicker) || (isSystemB ? '上游引用' : '答案依据')
  const displayNums = Array.from(new Set([
    ...(Array.isArray(detail.linkedNums) ? detail.linkedNums : []),
    detail.num,
  ].map((num) => Number(num || 0)).filter((num) => Number.isFinite(num) && num > 0))).sort((a, b) => a - b)
  const badgeNumText = displayNums.length > 1 ? displayNums.join('/') : String(displayNums[0] || '')
  const badgeLabel = badgeNumText ? (isSystemB ? `[R${badgeNumText}]` : `#${badgeNumText}`) : inlineLabel
  const headingPath = compact(detail.headingPath) || (!isSystemB ? compact(detail.title) : '')
  const pageLabel = pageRangeLabel(detail.pageStart, detail.pageEnd)
  const sourcePaperText = compact(detail.sourceName) || compact(display.source)
  const cardTitle = compact(view.header.title) || compact(detail.cardTitle)
  const cardSubtitle = compact(view.header.subtitle) || compact(detail.cardSubtitle)
  const rawHeaderSubtitle = isSystemB ? cardSubtitle : ''
  const cardTakeawayLabel = compact(takeawaySection?.label || detail.cardTakeawayLabel)
  const rawCardTakeaway = compact(takeawaySection?.text || detail.cardTakeaway)
  const cardTakeaway = looksNarrativeMetadataText(rawCardTakeaway, detail) ? '' : rawCardTakeaway
  const cardClaimLabel = compact(claimSection?.label || detail.cardClaimLabel)
  const cardEvidenceLabel = compact(evidenceSection?.label || detail.cardEvidenceLabel)
  const cardLocatorLabel = compact(locatorSection?.label || detail.cardLocatorLabel)
  const cardReferenceLabel = compact(referenceSection?.label || detail.cardReferenceLabel)
  const cardSupportLabel = compact(supportSection?.label || detail.cardSupportLabel)
  const cardWarning = compact(warningSection?.text || detail.cardWarning)
  const externalMetadataStatus = compact(detail.externalMetadataStatus).toLowerCase()
  const externalMetadataReason = compact(detail.externalMetadataReason)
  const externalTitle = compact(detail.externalTitle)
  const cardQualityLabel = compact(detail.cardQualityLabel)
  const cardQualityScore = Number(detail.cardQualityScore || 0)
  const cardQualityFlags = Array.isArray(detail.cardQualityFlags)
    ? detail.cardQualityFlags.map((item) => compact(item)).filter(Boolean)
    : []
  const answerContextOnly = isSystemB && (
    cardQualityFlags.includes('answer_context_only')
    || compact(detail.citationContextSource).toLowerCase() === 'answer_context'
    || compact(detail.systemBTraceSource).toLowerCase() === 'answer_context'
  )
  const cardFlow = Array.isArray(detail.cardFlow)
    ? detail.cardFlow.map((item) => compact(item)).filter(Boolean)
    : []
  const rawSystemAClaimText = cleanCitationDisplayText(
    compact(claimSection?.text || '') || compact(detail.cardClaim) || compact(detail.answerClaim),
  )
  const systemAClaimText = looksNarrativeMetadataText(rawSystemAClaimText, detail) ? '' : rawSystemAClaimText
  const systemAClaimPreview = answerPointPreview(systemAClaimText)
  const systemAClaimLabel = cardClaimLabel && !/^(?:答案中的话|对应回答)$/.test(cardClaimLabel)
    ? cardClaimLabel
    : '答案要点'
  const suppressRawSystemAEvidenceFallback = !isSystemB
    && (
      cardQualityFlags.includes('evidence_quote_filtered')
      || cardQualityFlags.includes('missing_evidence_quote')
    )
  const systemAEvidenceText = cleanCitationDisplayText(evidenceSection?.text || detail.cardEvidence)
    || (!suppressRawSystemAEvidenceFallback ? cleanCitationDisplayText(detail.evidenceQuote) : '')
    || (!suppressRawSystemAEvidenceFallback ? cleanCitationDisplayText(detail.summaryLine) : '')
  const systemATakeawayText = !isSystemB && cardTakeaway && !substantiallySame(cardTakeaway, systemAEvidenceText)
    ? cardTakeaway
    : ''
  const systemAEvidencePreview = evidencePreview(systemAEvidenceText, systemATakeawayText ? 250 : 330)
  const systemBExplicitReferenceText = cleanCitationDisplayText(referenceSection?.text || detail.cardReferenceEntry)
  const systemBReferenceText = systemBExplicitReferenceText || cleanCitationDisplayText(compact(detail.raw) || compact(detail.citeFmt))
  const systemBCardEvidenceText = answerContextOnly ? '' : cleanCitationDisplayText(evidenceSection?.text || detail.cardEvidence)
  const systemBRawContextCandidate = cleanCitationDisplayText(
    compact(detail.citationContext) || compact(detail.evidenceQuote) || compact(detail.summaryLine),
  )
  const suppressRawSystemBContextFallback = isSystemB
    && (
      cardQualityFlags.includes('weak_citation_context')
      || cardQualityFlags.includes('missing_citation_context')
      || answerContextOnly
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
  const rawSystemBContextSummary = cleanCitationDisplayText(contextSummarySection?.text || detail.cardContextSummary)
  const systemBContextSummaryText = isSystemB
    && rawSystemBContextSummary
    && !looksNarrativeMetadataText(rawSystemBContextSummary, detail)
    && !substantiallySame(rawSystemBContextSummary, systemBCitationContextText)
    && !substantiallySame(rawSystemBContextSummary, systemBReferenceText)
    && !substantiallySame(rawSystemBContextSummary, systemBTakeawayText)
    ? rawSystemBContextSummary
    : ''
  const systemBContextSummaryLabel = compact(contextSummarySection?.label || '') || '语境摘要'
  const systemBTraceSteps = isSystemB && Array.isArray(detail.systemBTraceSteps)
    ? detail.systemBTraceSteps.map((item) => compact(item)).filter(Boolean)
    : []
  const systemBTraceReason = isSystemB ? cleanCitationDisplayText(detail.systemBTraceReason) : ''
  const systemBTraceScore = Number(detail.systemBTraceScore || 0)
  const showSystemBTrace = Boolean(
    false
    && isSystemB
    && (systemBTraceSteps.length > 0 || systemBTraceReason || systemBTraceScore > 0),
  )
  const systemBTraceStatus = detail.systemBTraceComplete
    ? { label: '链路已闭合', tone: 'complete' }
    : { label: '链路需核对', tone: 'review' }
  const systemBCitationContextPreview = evidencePreview(systemBCitationContextText, systemBTakeawayText ? 250 : 330)
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
  const rawExplicitSupportText = compact(supportSection?.text || '')
    || compact(detail.cardSupportExplanation)
    || compact(detail.supportRelation)
    || whyText
    || bindingReason
  const explicitSupportText = looksNarrativeMetadataText(rawExplicitSupportText, detail) ? '' : rawExplicitSupportText
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
  const headerSubtitle = rawHeaderSubtitle && !substantiallySame(rawHeaderSubtitle, systemBTitle)
    ? rawHeaderSubtitle
    : ''
  const systemASub = [headingPath, pageLabel].filter(Boolean).join(' · ')
  const rawSystemALocationText = compact(locatorSection?.text || '') || compact(detail.cardLocator) || compact(detail.locationLabel) || systemASub || systemATitle
  const systemALocationText = stripLocationIdentityPrefix(rawSystemALocationText, [
    systemATitle,
    sourcePaperText,
    detail.sourceName,
    display.source,
  ]) || rawSystemALocationText
  const systemAAnchorText = anchorKindLabel(detail.anchorKind)
  const systemAHasReviewRisk = Boolean(bindingState || cardWarning || cardQualityFlags.includes('candidate_binding') || cardQualityFlags.includes('binding_mismatch'))
  const systemAHasOccurrenceClaim = cardQualityFlags.includes('occurrence_specific_claim')
  const systemAClaimLooksUseful = !isLowValueSystemAClaim(systemAClaimText)
  const showSystemAClaim = Boolean(
    systemAClaimPreview
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
  const rawSystemBLocationText = compact(locatorSection?.text || '') || compact(detail.cardLocator) || compact(detail.locationLabel) || [sourcePaperText, headingPath, pageLabel].filter(Boolean).join(' / ')
  const cleanedSystemBLocationText = stripLocationIdentityPrefix(rawSystemBLocationText, [
    sourcePaperText,
    detail.sourceName,
    display.source,
  ])
  const systemBLocationIsPaperOnly = isOnlyPaperLabel(rawSystemBLocationText, [
    sourcePaperText,
    detail.sourceName,
    display.source,
  ])
  const systemBLocationLabel = systemBLocationIsPaperOnly ? '引用所在论文' : '当前论文引用处'
  const systemBLocationText = systemBLocationIsPaperOnly
    ? '仅定位到当前论文'
    : (cleanedSystemBLocationText || rawSystemBLocationText)
  const systemBLocationHint = systemBLocationIsPaperOnly
    ? '只定位到哪篇论文引用了它，尚未定位到具体章节或页码。'
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
    && !cardWarning
    && (
      cardQualityFlags.includes('reference_entry_only')
      || !systemBCitationContextText
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
  const systemBReferencePreview = evidencePreview(systemBReferenceText, 260)
  const systemAMetaSource = display.source && !isOnlyPaperLabel(display.source, [systemATitle, sourcePaperText])
    ? display.source
    : ''
  const metaRows = [
    systemAMetaSource ? { label: '来源', value: systemAMetaSource } : null,
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
            <div className="kb-cite-pop-insight kb-cite-pop-takeaway" data-testid="citation-popover-system-a-takeaway">
              <span className="kb-cite-pop-section-title">{cardTakeawayLabel || '证据重点'}</span>
              <div className="kb-cite-pop-main">{systemATakeawayText}</div>
            </div>
          ) : null}
          {showSystemAClaim ? (
            <div className="kb-cite-pop-claim" data-testid="citation-popover-system-a-claim">
              <span className="kb-cite-pop-section-title">{systemAClaimLabel}</span>
              <div className="kb-cite-pop-main">{systemAClaimPreview}</div>
            </div>
          ) : null}
          {systemAEvidenceText ? (
            <div className="kb-cite-pop-quote" data-testid="citation-popover-system-a-evidence">
              <div className="kb-cite-pop-section-line">
                <span className="kb-cite-pop-section-title">{cardEvidenceLabel || '原文证据'}</span>
                {systemAEvidencePreview !== systemAEvidenceText ? <span className="kb-cite-pop-section-hint">节选</span> : null}
              </div>
              <blockquote>{systemAEvidencePreview}</blockquote>
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
          {showSystemBTrace ? (
            <div
              className={`kb-cite-pop-trace kb-cite-pop-trace-${systemBTraceStatus.tone}`}
              data-testid="citation-popover-system-b-trace"
            >
              <div className="kb-cite-pop-trace-head">
                <span className="kb-cite-pop-section-title">证据链</span>
                <span className="kb-cite-pop-trace-status">{systemBTraceStatus.label}</span>
                {systemBTraceScore > 0 ? (
                  <span className="kb-cite-pop-trace-score">{Math.round(systemBTraceScore * 100)}%</span>
                ) : null}
              </div>
              {systemBTraceSteps.length > 0 ? (
                <div className="kb-cite-pop-trace-steps" aria-label="System B evidence chain">
                  {systemBTraceSteps.map((step, index) => (
                    <span className="kb-cite-pop-trace-step-wrap" key={`${step}-${index}`}>
                      <span className="kb-cite-pop-trace-step">{step}</span>
                      {index < systemBTraceSteps.length - 1 ? <span className="kb-cite-pop-trace-arrow">→</span> : null}
                    </span>
                  ))}
                </div>
              ) : null}
              {systemBTraceReason ? <div className="kb-cite-pop-trace-reason">{systemBTraceReason}</div> : null}
            </div>
          ) : null}
          {systemBTakeawayText ? (
            <div className="kb-cite-pop-insight kb-cite-pop-takeaway" data-testid="citation-popover-system-b-takeaway">
              <span className="kb-cite-pop-section-title">{cardTakeawayLabel || '上游作用'}</span>
              <div className="kb-cite-pop-main">{systemBTakeawayText}</div>
            </div>
          ) : null}
          {showSystemBLocation ? (
            <div className="kb-cite-pop-locator" data-testid="citation-popover-system-b-location">
              <span className="kb-cite-pop-section-title">{systemBLocationLabel}</span>
              <span className="kb-cite-pop-locator-text">{systemBLocationText}</span>
              {systemBLocationHint ? <span className="kb-cite-pop-anchor-meta">{systemBLocationHint}</span> : null}
            </div>
          ) : null}
          {systemBContextSummaryText ? (
            <div className="kb-cite-pop-context-summary" data-testid="citation-popover-system-b-context-summary">
              <span className="kb-cite-pop-section-title">{systemBContextSummaryLabel}</span>
              <div className="kb-cite-pop-main">{systemBContextSummaryText}</div>
            </div>
          ) : null}
          {systemBCitationContextText ? (
            <div className="kb-cite-pop-quote" data-testid="citation-popover-system-b-context">
              <div className="kb-cite-pop-section-line">
                <span className="kb-cite-pop-section-title">{systemBCitationContextLabel}</span>
                {systemBCitationContextPreview !== systemBCitationContextText ? <span className="kb-cite-pop-section-hint">节选</span> : null}
              </div>
              <blockquote>{systemBCitationContextPreview}</blockquote>
            </div>
          ) : null}
          {showSystemBReference ? (
            <div className="kb-cite-pop-evidence" data-testid="citation-popover-system-b-reference">
              <div className="kb-cite-pop-section-title">{cardReferenceLabel || '上游文献条目'}</div>
              <div className="kb-cite-pop-main">{systemBReferencePreview}</div>
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
