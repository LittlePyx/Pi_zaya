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
import { SystemAEvidenceCard, SystemBLiteratureCard } from './CitationPopoverCards'
import { CitationPopoverHeader, type CompactMetaItem } from './CitationPopoverHeader'
import { buildEvidenceCardViewModel, previewClaimText, previewEvidenceText } from './evidenceCardViewModel'

import { useT } from '../../i18n'

const SYSTEM_B_TRACE_ENABLED = false
const SYSTEM_B_ARTICLE_OVERVIEW_SOURCES = new Set([
  'abstract',
  'fulltext',
  'reference_primary_evidence',
  'navigation',
  'exact_anchor',
  'section_intent_rescue',
  'doc_list_seed',
  'doc_list_prompt_aligned',
])

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
  showOpenReaderAction?: boolean
  showStartGuideAction?: boolean
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
  const text = compact(value)
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

function looksGenericSystemBTakeawayText(value: string): boolean {
  const text = compact(value).replace(/\s+/g, ' ')
  if (!text) return false
  return (
    /作为当前论文引用的方法背景或实现依据/.test(text)
    || /帮助核对该方法线索从哪里来/.test(text)
    || /把回答中的说法追溯到当前论文引用的上游文献/.test(text)
    || /参考文献条目.*(?:当前|本文).*Reader|本文引用.*参考文献条目|当前论文.*引用.*参考文献条目/.test(text)
    || /(?:打开|当前|本文).*论文.*引用.*(?:上游|参考).*文献/.test(text)
    || /links? the answer back to an upstream reference/i.test(text)
    || /upstream reference cited by the current paper/i.test(text)
    || /opened paper cites this upstream work as reference/i.test(text)
    || /bibliography entry is linked from the opened Reader document/i.test(text)
  )
}

function isReferenceEntryLikeText(value: string): boolean {
  const text = cleanCitationDisplayText(value).replace(/\s+/g, ' ').trim()
  if (!text || text.length < 32) return false
  if (!/\b(?:18|19|20)\d{2}\b/.test(text)) return false
  if (/\b(?:doi|arxiv|isbn|issn)\b|10\.\d{4,9}\//i.test(text)) return true
  const venueLike = /\b(?:IEEE|ACM|Springer|Elsevier|Nature|Science|Nat\.?|Opt\.?|Phys\.?|Journal|Proceedings|Trans\.?|Conf\.?|CVPR|ICCV|ICML|NeurIPS|arXiv)\b/i.test(text)
  const authorLead = /^(?:\[\s*\d{1,4}\s*\]\s*)?(?:[A-Z][A-Za-z'`-]+,\s*(?:[A-Z]\.?\s*){1,4}|[A-Z][a-zA-Z'`-]+\s+[A-Z](?:\.|\b)|[A-Z][a-zA-Z'`-]+\s+et\s+al\.?)/.test(text)
  return venueLike && authorLead
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

function anchorKindLabel(
  value: string,
  labels: {
    sentence: string
    paragraph: string
    equation: string
    figure: string
    table: string
  },
): string {
  const key = compact(value).toLowerCase()
  if (key === 'sentence') return labels.sentence
  if (key === 'paragraph') return labels.paragraph
  if (key === 'equation') return labels.equation
  if (key === 'figure') return labels.figure
  if (key === 'table') return labels.table
  return compact(value)
}

function evidencePreview(value: string, maxLen = 260): string {
  return previewEvidenceText(value, maxLen)
}

function answerPointPreview(value: string, maxLen = 140): string {
  return previewClaimText(value, maxLen)
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
  showOpenReaderAction = true,
  showStartGuideAction = true,
}: Props) {
  const S = useT()
  const ref = useRef<HTMLDivElement>(null)
  const [style, setStyle] = useState<{ left: number; top: number } | null>(null)
  const localizeKnownLabel = (value: string): string => {
    const text = compact(value)
    if (!text) return ''
    const labels: Record<string, string> = {
      上游引用: S.cite_kind_upstream,
      答案依据: S.cite_kind_evidence,
      答案中的话: S.cite_answer_point,
      对应回答: S.cite_answer_point,
      答案要点: S.cite_answer_point,
      引用语境: S.cite_context,
      语境摘要: S.cite_context_summary,
      链路已闭合: S.cite_trace_complete,
      链路需核对: S.cite_trace_review,
      疑似错配: S.cite_binding_mismatch,
      候选依据: S.cite_binding_candidate,
      上游参考文献: S.cite_upstream_reference,
      引用所在论文: S.cite_location_paper,
      当前论文引用处: S.cite_location_current,
      来源: S.cite_meta_source,
      发表: S.cite_meta_published,
      作者: S.cite_meta_author,
      位置: S.cite_position,
      锚点: S.cite_anchor_label,
      证据重点: S.cite_evidence_focus,
      原文证据: S.cite_original_evidence,
      可靠度: S.cite_reliability,
      证据链: S.cite_evidence_chain,
      上游作用: S.cite_upstream_role,
      上游文献条目: S.cite_reference_entry,
      说明: S.cite_note,
      'Missing reference entry': S.cite_missing_reference_entry,
    }
    return labels[text] || text
  }
  const localizeKnownBody = (value: string): string => {
    const text = compact(value)
    if (!text) return ''
    const missingReferenceMatch = text.match(/^Reference \[(\d{1,4})]\s+is cited in the opened Reader document, but the converted References section does not contain a matching bibliography entry\.?$/i)
    if (missingReferenceMatch) {
      return S.cite_missing_reference_entry_body.replace('{n}', missingReferenceMatch[1])
    }
    if (/前端缺少后端 cite_details/.test(text)) return S.cite_frontend_candidate_reason
    if (/前端根据本轮 References 临时补齐/.test(text)) return S.cite_candidate_support_default
    if (/这条引用只能作为候选依据/.test(text)) return S.cite_candidate_support_default
    return text
  }

  useEffect(() => {
    if (!detail) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    const onPointerDown = (event: MouseEvent) => {
      const el = ref.current
      if (!el) return
      const targetEl = event.target instanceof Element ? event.target : null
      if (targetEl?.closest('.kb-md-locate-inline-btn, .kb-prov-locate-chip, [data-kb-locate-block-id]')) return
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
      top: Math.min(Math.max(margin, position.y + 28), maxTop),
    })
  }, [detail, position])

  if (!detail || !position) return null

  const display = citationDisplay(detail)
  const doiLabel = compact(detail.doi) || compact(detail.doiUrl)
  const doiHref = compact(detail.doiUrl) || (doiLabel ? `https://doi.org/${doiLabel}` : '')
  const metrics = citeMetricSummary(detail)
  const inlineLabel = citationInlineLabel(detail, { includeSource: false })
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
  const kindLabel = localizeKnownLabel(view.header.kicker) || (isSystemB ? S.cite_kind_upstream : S.cite_kind_evidence)
  const systemADisplayNumSeeds = [
    ...(Array.isArray(detail.displayNums) ? detail.displayNums : []),
    detail.displayNum,
  ].map((num) => Number(num || 0)).filter((num) => Number.isFinite(num) && num > 0)
  const displayNums = Array.from(new Set(
    (isSystemB
      ? [
          ...(Array.isArray(detail.linkedNums) ? detail.linkedNums : []),
          detail.num,
        ]
      : (systemADisplayNumSeeds.length > 0 ? systemADisplayNumSeeds : [detail.num]))
      .map((num) => Number(num || 0))
      .filter((num) => Number.isFinite(num) && num > 0),
  )).sort((a, b) => a - b)
  const badgeNumText = displayNums.length > 1 ? displayNums.join('/') : String(displayNums[0] || '')
  const badgeLabel = badgeNumText ? (isSystemB ? `[R${badgeNumText}]` : `#${badgeNumText}`) : inlineLabel
  const headingPath = compact(detail.headingPath) || (!isSystemB ? compact(detail.title) : '')
  const pageLabel = pageRangeLabel(detail.pageStart, detail.pageEnd)
  const sourcePaperText = compact(detail.sourceName) || compact(display.source)
  const cardTitle = compact(view.header.title) || compact(detail.cardTitle)
  const cardSubtitle = compact(view.header.subtitle) || compact(detail.cardSubtitle)
  const rawHeaderSubtitle = isSystemB ? cardSubtitle : ''
  const cardTakeawayLabel = localizeKnownLabel(takeawaySection?.label || detail.cardTakeawayLabel)
  const rawCardTakeaway = compact(takeawaySection?.text || detail.cardTakeaway)
  const cardTakeaway = looksNarrativeMetadataText(rawCardTakeaway, detail) ? '' : rawCardTakeaway
  const cardClaimLabel = localizeKnownLabel(claimSection?.label || detail.cardClaimLabel)
  const cardEvidenceLabel = localizeKnownLabel(evidenceSection?.label || detail.cardEvidenceLabel)
  const cardLocatorLabel = localizeKnownLabel(locatorSection?.label || detail.cardLocatorLabel)
  const cardReferenceLabel = localizeKnownLabel(referenceSection?.label || detail.cardReferenceLabel)
  const cardSupportLabel = localizeKnownLabel(supportSection?.label || detail.cardSupportLabel)
  const cardWarning = compact(warningSection?.text || detail.cardWarning)
  const externalMetadataStatus = compact(detail.externalMetadataStatus).toLowerCase()
  const externalMetadataReason = compact(detail.externalMetadataReason)
  const externalTitle = compact(detail.externalTitle)
  const cardQualityLabel = localizeKnownLabel(detail.cardQualityLabel)
  const cardQualityScore = Number(detail.cardQualityScore || 0)
  const cardQualityFlags = Array.isArray(detail.cardQualityFlags)
    ? detail.cardQualityFlags.map((item) => compact(item)).filter(Boolean)
    : []
  const cardFlow = Array.isArray(detail.cardFlow)
    ? detail.cardFlow.map((item) => compact(item)).filter(Boolean)
    : []
  const suppressRawSystemAEvidenceFallback = !isSystemB
    && (
      cardQualityFlags.includes('evidence_quote_filtered')
      || cardQualityFlags.includes('missing_evidence_quote')
    )
  const systemAEvidenceCard = buildEvidenceCardViewModel(detail, {
    S,
    evidenceOverride: evidenceSection?.text || detail.cardEvidence,
    evidenceLabelOverride: cardEvidenceLabel,
    claimOverride: claimSection?.text || detail.cardClaim || detail.answerClaim,
    claimLabelOverride: cardClaimLabel,
    supportOverride: supportSection?.text || detail.cardSupportExplanation || detail.supportRelation || detail.whyLine || detail.bindingReason,
    supportLabelOverride: cardSupportLabel,
    includeCitationFallback: !suppressRawSystemAEvidenceFallback,
    includeRawFallback: false,
  })
  const rawSystemAClaimText = systemAEvidenceCard.claim
  const systemAClaimText = looksNarrativeMetadataText(rawSystemAClaimText, detail) ? '' : rawSystemAClaimText
  const systemAClaimPreview = answerPointPreview(systemAClaimText)
  const systemAClaimLabel = cardClaimLabel && !/^(?:答案中的话|对应回答)$/.test(cardClaimLabel)
    ? cardClaimLabel
    : S.cite_answer_point
  const systemAEvidenceText = systemAEvidenceCard.evidence
  const systemATakeawayText = !isSystemB && cardTakeaway && !substantiallySame(cardTakeaway, systemAEvidenceText)
    ? cardTakeaway
    : ''
  const systemAEvidencePreview = evidencePreview(systemAEvidenceText, systemATakeawayText ? 250 : 330)
  const systemBExplicitReferenceText = localizeKnownBody(cleanCitationDisplayText(referenceSection?.text || detail.cardReferenceEntry))
  const systemBReferenceText = systemBExplicitReferenceText || cleanCitationDisplayText(compact(detail.raw) || compact(detail.citeFmt))
  const systemBOverviewSource = compact(detail.summarySource).toLowerCase()
  const systemBContextSource = compact(detail.citationContextSource).toLowerCase()
  const systemBReferenceIdentityText = cleanCitationDisplayText(systemBReferenceText)
  const normalizeSystemBTextCandidate = (value: string, opts: { allowCitationContext?: boolean } = {}): string => {
    const text = cleanCitationDisplayText(value).replace(/\s+/g, ' ').trim()
    if (!text) return ''
    if (looksGenericSystemBTakeawayText(text)) return ''
    if (looksNarrativeMetadataText(text, detail)) return ''
    if (substantiallySame(text, systemBReferenceIdentityText) || isReferenceEntryLikeText(text)) return ''
    if (!opts.allowCitationContext && looksLowValueCitationContext(text)) return ''
    return text
  }
  const firstSystemBText = (values: string[], opts?: { allowCitationContext?: boolean }): string => {
    for (const value of values) {
      const text = normalizeSystemBTextCandidate(value, opts)
      if (text) return text
    }
    return ''
  }
  const systemBOverviewSourceIsContext = [
    'answer_context',
    'citation_context',
    'reader_occurrence',
    'reader_reference_link',
    'reader_references',
  ].includes(systemBOverviewSource)
  const systemBOverviewSourceIsArticle = SYSTEM_B_ARTICLE_OVERVIEW_SOURCES.has(systemBOverviewSource)
  const systemBPaperOverviewText = isSystemB
    ? firstSystemBText([
      systemBOverviewSourceIsContext ? '' : detail.summaryLine,
    ], { allowCitationContext: systemBOverviewSourceIsArticle })
    : ''
  const systemBPaperOverviewPreview = evidencePreview(systemBPaperOverviewText, 360)
  const systemBPaperOverviewLabel = ((S as unknown as Record<string, string>).cite_paper_overview || 'Article overview')
  const systemBCitationContextText = ''
  const systemBCitationContextLabel = cardEvidenceLabel || S.cite_context
  const rawSystemBTakeawayText = isSystemB
    ? firstSystemBText([
      detail.cardContextSummary,
      contextSummarySection?.text || '',
      cardTakeaway,
      detail.upstreamWorkRole,
      detail.cardSupportExplanation,
      detail.supportRelation,
      detail.whyLine,
      detail.systemBTraceContext,
      systemBContextSource === 'reader_references' ? '' : detail.citationContext,
    ], { allowCitationContext: true })
    : ''
  const localizedSystemBTakeawayText = localizeKnownBody(rawSystemBTakeawayText)
  const systemBTakeawayText = localizedSystemBTakeawayText && !substantiallySame(localizedSystemBTakeawayText, systemBReferenceText)
    ? localizedSystemBTakeawayText
    : ''
  const systemBTakeawayLabel = ((S as unknown as Record<string, string>).cite_current_paper_usage || S.cite_upstream_role)
  const systemBContextSummaryText = ''
  const systemBContextSummaryLabel = localizeKnownLabel(contextSummarySection?.label || '') || S.cite_context_summary
  const systemBTraceSteps = isSystemB && Array.isArray(detail.systemBTraceSteps)
    ? detail.systemBTraceSteps.map((item) => compact(item)).filter(Boolean)
    : []
  const systemBTraceReason = isSystemB ? cleanCitationDisplayText(detail.systemBTraceReason) : ''
  const systemBTraceScore = Number(detail.systemBTraceScore || 0)
  const showSystemBTrace = Boolean(
    SYSTEM_B_TRACE_ENABLED
    && isSystemB
    && (systemBTraceSteps.length > 0 || systemBTraceReason || systemBTraceScore > 0),
  )
  const systemBTraceStatus = detail.systemBTraceComplete
    ? { label: S.cite_trace_complete, tone: 'complete' }
    : { label: S.cite_trace_review, tone: 'review' }
  const systemBCitationContextPreview = evidencePreview(systemBCitationContextText, 330)
  const whyText = compact(detail.whyLine)
  const bindingStatus = compact(detail.bindingStatus).toLowerCase()
  const bindingReason = localizeKnownBody(detail.bindingReason)
  const bindingOverlapText = Array.isArray(detail.bindingOverlapTerms)
    ? detail.bindingOverlapTerms.map((item) => compact(item)).filter(Boolean).join(' / ')
    : ''
  const bindingState = !isSystemB && bindingStatus && bindingStatus !== 'grounded'
    ? (
        bindingStatus === 'mismatch'
            ? { label: S.cite_binding_mismatch, tone: 'mismatch' }
            : { label: S.cite_binding_candidate, tone: 'candidate' }
      )
    : null
  const rawExplicitSupportText = localizeKnownBody(supportSection?.text || '')
    || localizeKnownBody(detail.cardSupportExplanation)
    || localizeKnownBody(detail.supportRelation)
    || whyText
    || bindingReason
  const explicitSupportText = looksNarrativeMetadataText(rawExplicitSupportText, detail) ? '' : rawExplicitSupportText
  const supportBaseText = isSystemB
    ? (explicitSupportText || S.cite_system_b_support_default)
    : (explicitSupportText || (bindingStatus === 'candidate'
      ? S.cite_candidate_support_default
      : ''))
  const supportText = supportBaseText
  const showBindingReason = Boolean(bindingReason && !substantiallySame(bindingReason, supportText))
  const displayMain = compact(display.main)
  const systemATitle = cardTitle || ((displayMain && displayMain !== headingPath)
    ? displayMain
    : (compact(detail.sourceName) || compact(display.source) || displayMain))
  const systemBTitleMissing = !cardTitle && !compact(detail.title)
  const systemBTitle = cardTitle || compact(detail.title) || S.cite_upstream_reference
  const headerSubtitle = !isSystemB && rawHeaderSubtitle && !substantiallySame(rawHeaderSubtitle, systemBTitle)
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
  const systemAAnchorText = anchorKindLabel(detail.anchorKind, {
    sentence: S.cite_anchor_sentence,
    paragraph: S.cite_anchor_paragraph,
    equation: S.cite_anchor_equation,
    figure: S.cite_anchor_figure,
    table: S.cite_anchor_table,
  })
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
  const primaryActionLabel = isSystemB ? S.cite_read_locate : S.cite_open_evidence
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
  const systemBReferenceRowLocation = (
    systemBContextSource === 'reader_references'
    || compact(detail.shelfOrigin).toLowerCase() === 'reader_references'
  ) && badgeLabel
    ? badgeLabel
    : ''
  const systemBMeaningfulLocation = systemBLocationIsPaperOnly
    ? ''
    : (cleanedSystemBLocationText || rawSystemBLocationText)
  const systemBLocationLabel = systemBReferenceRowLocation ? S.cite_reference_entry : S.cite_location_current
  const systemBLocationText = systemBReferenceRowLocation || systemBMeaningfulLocation
  const systemBLocationHint = ''
  const systemBLocationSourceIsWeak = [
    'answer_context',
    'answer_reference_mention',
    'reader_references',
  ].includes(systemBContextSource) || cardQualityFlags.some((flag) => [
    'answer_context_only',
    'reference_entry_only',
    'weak_citation_context',
    'missing_citation_context',
  ].includes(flag))
  const showSystemBLocation = Boolean(
    isSystemB
    && systemBMeaningfulLocation
    && !systemBReferenceRowLocation
    && !systemBLocationSourceIsWeak,
  )
  const systemBSupportText = isSystemB
    && explicitSupportText
    && !substantiallySame(explicitSupportText, systemBCitationContextText)
    && !substantiallySame(explicitSupportText, systemBReferenceText)
    ? explicitSupportText
    : ''
  const showSystemBSupport = false
  const hasSystemBHeaderIdentity = Boolean(
    (systemBTitle && systemBTitle !== S.cite_upstream_reference)
    || headerSubtitle
    || doiLabel
    || metrics.length > 0
  )
  const systemBReferenceHasBibliographicContext = Boolean(
    systemBReferenceText
    && /\b(?:18|19|20)\d{2}\b/.test(systemBReferenceText)
    && (
      isReferenceEntryLikeText(systemBReferenceText)
      || !systemBTitle
      || systemBReferenceText.length > systemBTitle.length + 18
    )
  )
  const systemBReferenceIsUsefulEntry = Boolean(
    systemBReferenceText
    && (
      systemBReferenceHasBibliographicContext
      || (
        (!systemBTitle || !substantiallySame(systemBReferenceText, systemBTitle))
        && (!headerSubtitle || !substantiallySame(systemBReferenceText, headerSubtitle))
      )
    )
  )
  const systemBReferenceEntryOnly = cardQualityFlags.includes('reference_entry_only')
  const systemBReferenceTitleMissing = systemBTitleMissing || cardQualityFlags.includes('missing_reference_title')
  const suppressSystemBReferenceEntry = [
    'reader_occurrence',
    'reader_reference_link',
    'reader_references',
  ].includes(systemBContextSource)
    && !systemBReferenceEntryOnly
    && !systemBReferenceTitleMissing
  const showSystemBReference = Boolean(
    systemBReferenceText
    && !suppressSystemBReferenceEntry
    && (
      (isSystemB && systemBReferenceIsUsefulEntry && (showSystemBLocation || systemBReferenceEntryOnly || systemBReferenceTitleMissing || !hasSystemBHeaderIdentity))
      || (systemBExplicitReferenceText && (systemBReferenceEntryOnly || systemBReferenceTitleMissing || !hasSystemBHeaderIdentity))
      || systemBReferenceTitleMissing
      || systemBReferenceEntryOnly
      || (!hasSystemBHeaderIdentity && !systemBPaperOverviewText)
    ),
  )
  const systemBReferencePreview = evidencePreview(systemBReferenceText, 260)
  const systemBReferenceLabel = ((S as unknown as Record<string, string>).cite_original_reference_entry || S.cite_reference_entry)
  const showSystemBOverviewLoading = Boolean(isSystemB && loading && !systemBPaperOverviewText)
  const showSystemBOverviewUnavailable = Boolean(
    isSystemB
    && !loading
    && detail.bibliometricsChecked
    && !systemBPaperOverviewText
    && !showSystemBReference
    && (doiLabel || systemBTitle),
  )
  const systemAMetaSource = display.source && !isOnlyPaperLabel(display.source, [systemATitle, sourcePaperText])
    ? display.source
    : ''
  const metaRows = [
    systemAMetaSource ? { label: S.cite_meta_source, value: systemAMetaSource } : null,
    display.venueYear ? { label: S.cite_meta_published, value: display.venueYear } : null,
  ].filter(Boolean) as Array<{ label: string; value: string }>
  const showMetaGrid = false
  const showMetrics = false
  const showCardQuality = false
  const showCardWarning = Boolean(cardWarning && cardQualityFlags.includes('missing_reference_entry'))
  const showExternalMetadataWarning = externalMetadataStatus === 'conflict'
  const externalMetadataWarningText = showExternalMetadataWarning
    ? (externalMetadataReason || S.cite_external_metadata_warning)
    : ''
  const externalMetadataTitleHint = externalTitle && !substantiallySame(externalTitle, displayMain)
    ? S.cite_external_title.replace('{title}', externalTitle)
    : ''
  const systemACompactMetaItems = !isSystemB
    ? ([
        systemALocationText ? {
          key: 'location',
          label: cardLocatorLabel || S.cite_position,
          value: systemALocationText,
          tone: 'location',
        } : null,
        systemAAnchorText ? {
          key: 'anchor',
          label: S.cite_anchor_label,
          value: systemAAnchorText,
          tone: 'muted',
        } : null,
        ...metaRows.map((item) => ({
          key: `meta-${item.label}`,
          label: item.label,
          value: item.value,
          tone: 'muted',
        })),
        doiLabel ? {
          key: 'doi',
          label: 'DOI',
          value: doiLabel,
          href: doiHref,
          tone: 'doi',
        } : null,
        ...metrics.map((item) => ({
          key: `metric-${item}`,
          label: '',
          value: item,
          tone: 'metric',
        })),
      ].filter(Boolean) as CompactMetaItem[])
    : []
  const systemBCompactMetaItems = isSystemB
    ? ([
        display.authors ? {
          key: 'authors',
          label: '',
          value: display.authors,
          tone: 'muted',
        } : null,
        display.venueYear ? {
          key: 'published',
          label: '',
          value: display.venueYear,
          tone: 'muted',
        } : null,
        doiLabel ? {
          key: 'doi',
          label: 'DOI',
          value: doiLabel,
          href: doiHref,
          tone: 'doi',
        } : null,
        ...metrics.map((item) => ({
          key: `metric-${item}`,
          label: '',
          value: item,
          tone: 'metric',
        })),
      ].filter(Boolean) as CompactMetaItem[])
    : []
  const compactMetaItems = isSystemB ? systemBCompactMetaItems : systemACompactMetaItems

  return (
    <div
      ref={ref}
      className={`kb-cite-pop ${isSystemB ? 'kb-cite-pop-system-b w-[480px]' : 'kb-cite-pop-system-a w-[460px]'} fixed z-50 max-w-[calc(100vw-20px)]`}
      data-testid="citation-popover"
      style={style ?? { left: position.x + 10, top: position.y + 10, visibility: 'hidden' }}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      <CitationPopoverHeader
        isSystemB={isSystemB}
        kindLabel={kindLabel}
        badgeLabel={badgeLabel}
        title={isSystemB ? systemBTitle : systemATitle}
        subtitle={headerSubtitle}
        compactMetaItems={compactMetaItems}
        onClose={onClose}
      />

      {explainText ? <div className="kb-cite-pop-explain" data-testid="citation-popover-explain">{explainText}</div> : null}
      {flowSteps.length > 0 ? (
        <div className="kb-cite-pop-flow" data-testid="citation-popover-flow" aria-label={S.cite_flow_aria}>
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
      {showCardWarning ? (
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
        <SystemAEvidenceCard
          showTakeaway={showSystemATakeaway}
          takeawayLabel={cardTakeawayLabel || S.cite_evidence_focus}
          takeawayText={systemATakeawayText}
          showClaim={showSystemAClaim}
          claimLabel={systemAClaimLabel}
          claimPreview={systemAClaimPreview}
          evidenceText={systemAEvidenceText}
          evidencePreview={systemAEvidencePreview}
          evidenceLabel={cardEvidenceLabel || S.cite_original_evidence}
          excerptLabel={S.cite_excerpt}
          showSupport={showSystemASupport}
          supportLabel={cardSupportLabel || S.cite_reliability}
          supportText={supportText}
        />
      ) : (
        <SystemBLiteratureCard
          showTrace={showSystemBTrace}
          traceStatus={systemBTraceStatus}
          traceScore={systemBTraceScore}
          traceSteps={systemBTraceSteps}
          traceReason={systemBTraceReason}
          traceLabel={S.cite_evidence_chain}
          paperOverviewText={systemBPaperOverviewText}
          paperOverviewLabel={systemBPaperOverviewLabel}
          paperOverviewPreview={systemBPaperOverviewPreview}
          showOverviewLoading={showSystemBOverviewLoading}
          overviewLoadingLabel={S.cite_loading_summary || S.cite_loading}
          showOverviewUnavailable={showSystemBOverviewUnavailable}
          overviewUnavailableLabel={S.cite_summary_unavailable}
          takeawayText={systemBTakeawayText}
          takeawayLabel={systemBTakeawayLabel}
          showLocation={showSystemBLocation}
          locationLabel={systemBLocationLabel}
          locationText={systemBLocationText}
          locationHint={systemBLocationHint}
          contextSummaryText={systemBContextSummaryText}
          contextSummaryLabel={systemBContextSummaryLabel}
          citationContextText={systemBCitationContextText}
          citationContextPreview={systemBCitationContextPreview}
          citationContextLabel={systemBCitationContextLabel}
          excerptLabel={S.cite_excerpt}
          showReference={showSystemBReference}
          referenceLabel={cardReferenceLabel || systemBReferenceLabel}
          referencePreview={systemBReferencePreview}
          showSupport={showSystemBSupport}
          supportLabel={cardSupportLabel || S.cite_note}
          supportText={systemBSupportText}
        />
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
        {showOpenReaderAction ? (
          <button
            className="kb-cite-pop-open-shelf kb-cite-pop-action-primary"
            type="button"
            disabled={!canOpenReader}
            onClick={() => onOpenReader(detail)}
          >
            {primaryActionLabel}
          </button>
        ) : null}
        {showStartGuideAction ? (
          <button
            className="kb-cite-pop-open-shelf"
            type="button"
            onClick={() => onStartGuide(detail)}
            disabled={guideLoading}
          >
            {guideLoading ? S.cite_starting_guide : S.cite_start_guide}
          </button>
        ) : null}
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
