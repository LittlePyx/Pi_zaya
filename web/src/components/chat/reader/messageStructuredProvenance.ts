import type { ReaderLocateTarget, ReaderOpenPayload } from './readerTypes'
import { basenameFromSourcePath, normalizeSourcePathForMatch as normalizeSourcePathForMatchShared } from '../../../utils/sourcePath'
import { coerceStringArray, hasFormulaSignal, normalizeLocateText, stripMarkdownInline, stripProvenanceNoise, type LocateCandidate } from './messageLocateCandidates'
import { coerceReaderLocateTarget, coerceReaderOpenPayload } from './messageReaderLocatePayload'
import { isHeadingLikeQuotedAnchor, scoreStructuredPrimaryCandidate } from './messageStructuredLocateScoring'

export interface ProvenanceLocateEntry {
  segmentId: string
  segmentKind?: string
  label: string
  segmentText: string
  evidenceQuote: string
  locateTarget?: ReaderLocateTarget
  readerOpen?: ReaderOpenPayload
  hitLevel?: string
  claimType?: string
  mustLocate?: boolean
  locatePolicy?: string
  locateSurfacePolicy?: string
  claimGroupId?: string
  claimGroupKind?: string
  formulaOrigin?: string
  anchorKind?: string
  anchorText?: string
  equationNumber?: number
  supportFigureNumber?: number
  supportPanelLetters?: string[]
  snippetKey: string
  snippetAliases: string[]
  primary: LocateCandidate
  alternatives: LocateCandidate[]
  relatedBlockIds?: string[]
  sourceSegmentId?: string
  groupLeadText?: string
  groupDistance?: number
}

export interface StructuredProvenanceSegment {
  index: number
  segmentId: string
  kind: string
  segmentType: string
  evidenceMode: string
  hitLevel: string
  claimType: string
  mustLocate: boolean
  locatePolicy: string
  locateSurfacePolicy: string
  claimGroupId: string
  claimGroupKind: string
  claimGroupTargetSegmentId: string
  claimGroupTargetDistance: number
  claimGroupLeadText: string
  formulaOrigin: string
  anchorKind: string
  anchorText: string
  equationNumber: number
  text: string
  snippetKey: string
  snippetAliases: string[]
}

function normalizeSourcePathForMatch(input: string): string {
  return normalizeSourcePathForMatchShared(input)
}

function sourceDocumentIdentityKey(input: string): string {
  const normalized = normalizeSourcePathForMatch(input)
  if (!normalized) return ''
  const parts = normalized.split('/').map((item) => item.trim()).filter(Boolean)
  const file = parts[parts.length - 1] || normalized
  const stem = file
    .replace(/\.en\.md$/i, '')
    .replace(/\.md$/i, '')
    .replace(/\.pdf$/i, '')
    .replace(/\s+/g, ' ')
    .trim()
  return stem || file
}

function sourcePathsReferToSameDocument(left: string, right: string): boolean {
  const leftNorm = normalizeSourcePathForMatch(left)
  const rightNorm = normalizeSourcePathForMatch(right)
  if (!leftNorm || !rightNorm) return false
  if (leftNorm === rightNorm) return true
  const leftId = sourceDocumentIdentityKey(leftNorm)
  const rightId = sourceDocumentIdentityKey(rightNorm)
  return Boolean(leftId && rightId && leftId === rightId)
}

export function normalizeStrictAnchorText(input: string): string {
  return stripMarkdownInline(String(input || ''))
    .replace(/\s+/g, ' ')
    .trim()
}

function inferStrictAnchorKind(
  rawKind: string,
  claimType: string,
): string {
  const kind = String(rawKind || '').trim().toLowerCase()
  if (kind) return kind
  const claim = String(claimType || '').trim().toLowerCase()
  if (claim === 'formula_claim' || claim === 'inline_formula_claim' || claim === 'equation_explanation_claim') {
    return 'equation'
  }
  if (claim === 'figure_claim' || claim === 'figure_panel') return 'figure'
  if (claim === 'quote_claim') return 'quote'
  if (claim === 'blockquote_claim') return 'blockquote'
  if (
    claim === 'shell_sentence'
    || claim === 'critical_fact_claim'
    || claim === 'method_detail'
    || claim === 'prior_work'
    || claim === 'doc_map'
  ) {
    return 'paragraph'
  }
  return ''
}

export function hasSegmentStrictLocateIdentity(
  segment: Record<string, unknown> | null | undefined,
  currentSegment?: StructuredProvenanceSegment | null,
): boolean {
  const readerOpen = coerceReaderOpenPayload(segment?.reader_open)
  const locateTarget = coerceReaderLocateTarget(segment?.locate_target) || readerOpen?.locateTarget || null
  const primaryBlockId = String(segment?.primary_block_id || locateTarget?.blockId || '').trim()
  const evidenceBlockIds = Array.isArray(segment?.evidence_block_ids)
    ? segment?.evidence_block_ids.map((item) => String(item || '').trim()).filter(Boolean)
    : []
  const effectiveEvidenceBlockIds = evidenceBlockIds.length > 0
    ? evidenceBlockIds
    : coerceStringArray(locateTarget?.blockId, 1, 120)
  const anchorKindRaw = String(segment?.anchor_kind || locateTarget?.anchorKind || currentSegment?.anchorKind || '').trim().toLowerCase()
  const claimType = String(segment?.claim_type || currentSegment?.claimType || '').trim().toLowerCase()
  const anchorKind = inferStrictAnchorKind(anchorKindRaw, claimType)
  const anchorText = normalizeStrictAnchorText(String(segment?.anchor_text || locateTarget?.anchorText || currentSegment?.anchorText || ''))
  const evidenceQuote = normalizeStrictAnchorText(String(segment?.evidence_quote || locateTarget?.evidenceQuote || ''))
  if (!(primaryBlockId && effectiveEvidenceBlockIds.length > 0)) return false
  if (anchorKindRaw && (anchorText || evidenceQuote)) return true
  const locatePolicy = String(segment?.locate_policy || currentSegment?.locatePolicy || '').trim().toLowerCase()
  const mustLocate = Boolean(segment?.must_locate ?? currentSegment?.mustLocate)
  if (!(mustLocate || locatePolicy === 'required')) return false
  const segmentText = normalizeStrictAnchorText(String(segment?.text || locateTarget?.snippet || currentSegment?.text || ''))
  return Boolean(anchorKind && (anchorText || evidenceQuote || segmentText))
}

function isFormulaBundleLocateEntry(entry: Pick<ProvenanceLocateEntry, 'claimGroupKind' | 'claimGroupId' | 'anchorKind' | 'claimType' | 'primary'>): boolean {
  const groupKind = String(entry.claimGroupKind || '').trim().toLowerCase()
  if (groupKind !== 'formula_bundle') return false
  const anchorKind = String(entry.anchorKind || '').trim().toLowerCase()
  const claimType = String(entry.claimType || '').trim().toLowerCase()
  if (anchorKind === 'equation') return true
  return claimType === 'formula_claim' || claimType === 'equation_explanation_claim'
}

function formulaBundleLocateGroupKey(entry: Pick<ProvenanceLocateEntry, 'claimGroupKind' | 'claimGroupId' | 'anchorKind' | 'claimType' | 'primary'>): string {
  if (!isFormulaBundleLocateEntry(entry)) return ''
  const claimGroupId = String(entry.claimGroupId || '').trim()
  if (claimGroupId) return claimGroupId
  const sourcePath = String(entry.primary?.sourcePath || '').trim()
  const targetId = String(entry.primary?.blockId || entry.primary?.anchorId || '').trim()
  return (sourcePath && targetId) ? `${sourcePath}::${targetId}` : ''
}

function formulaBundleRepresentativeScore(entry: ProvenanceLocateEntry): number {
  let score = 0
  const claimType = String(entry.claimType || '').trim().toLowerCase()
  const locateSurfacePolicy = String(entry.locateSurfacePolicy || '').trim().toLowerCase()
  const formulaOrigin = String(entry.formulaOrigin || '').trim().toLowerCase()
  if (locateSurfacePolicy === 'primary') score += 4.5
  else if (locateSurfacePolicy === 'secondary') score += 1.25
  if (formulaOrigin === 'source') score += 1.8
  else if (formulaOrigin === 'explanation') score += 0.6
  else if (formulaOrigin === 'derived') score -= 2.4
  if (claimType === 'formula_claim') score += 4
  else if (claimType === 'inline_formula_claim') score += 1.2
  else if (claimType === 'equation_explanation_claim') score += 2
  const anchorRaw = String(entry.anchorText || entry.evidenceQuote || entry.segmentText || '').trim()
  if (/\$\$[\s\S]{8,}\$\$/.test(anchorRaw)) score += 1.4
  else if (hasFormulaSignal(anchorRaw)) score += 0.7
  if (String(entry.primary?.anchorId || '').trim()) score += 0.4
  if (String(entry.primary?.blockId || '').trim()) score += 0.3
  score -= Math.min(1, Math.max(0, Number(entry.groupDistance || 0)) * 0.08)
  return score
}

function isNegativeLocateSurfaceText(input: string): boolean {
  const raw = String(input || '').trim()
  if (!raw) return false
  return /\b(?:not stated|not mentioned|does not mention|doesn't mention|does not specify|doesn't specify|cannot be determined|not found|no external paper matched|no other papers matched|does not include)\b/i.test(raw)
}

export function shouldSuppressNegativeLocateSurface(input: {
  claimType?: string
  anchorKind?: string
  segmentText?: string
  evidenceQuote?: string
  anchorText?: string
  snippet?: string
  highlightSnippet?: string
}): boolean {
  const anchorKind = String(input.anchorKind || '').trim().toLowerCase()
  if (anchorKind === 'equation' || anchorKind === 'figure' || anchorKind === 'quote' || anchorKind === 'blockquote' || anchorKind === 'inline_formula') {
    return false
  }
  const claimType = String(input.claimType || '').trim().toLowerCase()
  const texts = [
    String(input.snippet || '').trim(),
    String(input.highlightSnippet || '').trim(),
    String(input.evidenceQuote || '').trim(),
    String(input.anchorText || '').trim(),
    String(input.segmentText || '').trim(),
  ].filter(Boolean)
  const hasNegativeSurface = texts.some((text) => isNegativeLocateSurfaceText(text))
  if (!hasNegativeSurface) return false
  return (
    !claimType
    || claimType === 'evidence_note_claim'
    || claimType === 'shell_sentence'
    || claimType === 'critical_fact_claim'
  )
}

export function isLikelyRhetoricalLocateShell(input: string): boolean {
  const raw = stripProvenanceNoise(stripMarkdownInline(String(input || '')))
    .replace(/^\s{0,3}#{1,6}\s+/g, ' ')
    .replace(/^[^\u4e00-\u9fffA-Za-z0-9]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
  if (!raw) return true
  const normalized = normalizeLocateText(raw)
  if (!normalized) return true
  if (/^(?:直接证据|间接证据|唯一且明确|延伸思考题|高价值问题|研究问题|讨论题|思考题)(?:\s*[(:：\uff08][^)\uff09:：]{0,48}[)\uff09]?)?\s*[:\uFF1A]?$/.test(raw)) {
    return true
  }
  if (/^(?:\u8bf4\u660e|\u8868\u660e|\u53ef\u89c1|\u56e0\u6b64|\u6240\u4ee5|\u603b\u4e4b|\u7efc\u4e0a|\u7531\u6b64\u53ef\u89c1|\u8fdb\u4e00\u6b65\u8bf4\u660e|\u8fdb\u4e00\u6b65\u8868\u660e|\u8fdb\u4e00\u6b65\u8bc1\u5b9e|\u63d0\u793a)\s*[:\uFF1A]?$/.test(raw)) {
    return true
  }
  if (/^(?:\u6587\u4e2d\u63d0\u5230|\u6587\u4e2d\u6307\u51fa|\u4f5c\u8005\u6307\u51fa|\u5b9e\u9a8c\u7ed3\u679c\u663e\u793a|\u7ed3\u679c\u663e\u793a|\u8868\u683c\u6807\u9898\u4e0e\u65b9\u6cd5\u547d\u540d\u660e\u786e\u4e3a).{0,160}(?:\u8bf4\u660e|\u8868\u660e|\u610f\u5473\u7740|\u63d0\u793a|\u53ef\u89c1|\u8bc1\u5b9e)\s*[:\uFF1A]$/.test(raw)) {
    return true
  }
  if (/[:\uFF1A]$/.test(raw)) {
    const informativeTail = raw
      .replace(/[:\uFF1A]\s*$/, '')
      .replace(/["'\u201c\u201d\u2018\u2019\u300a\u300b\u3008\u3009\u300c\u300d\u300e\u300f]/g, ' ')
      .replace(/\b(?:we|our|method|paper|table|figure|fig)\b/gi, ' ')
      .replace(/\s+/g, ' ')
      .trim()
    if (informativeTail.length <= 32) return true
    if (/(?:\u8bf4\u660e|\u8868\u660e|\u610f\u5473\u7740|\u63d0\u793a|\u53ef\u89c1|\u8bc1\u5b9e)\s*$/.test(informativeTail)) return true
  }
  return false
}

function scoreLocateContentCore(
  input: string,
  opts?: {
    kind?: string
    segmentType?: string
    evidenceMode?: string
  },
): number {
  const raw = stripProvenanceNoise(stripMarkdownInline(String(input || '')))
    .replace(/\s+/g, ' ')
    .trim()
  if (!raw) return 0
  if (isLikelyRhetoricalLocateShell(raw)) return 0.04
  let score = 0.18
  const length = raw.length
  if (length >= 18) score += 0.12
  if (length >= 28) score += 0.12
  if (length >= 46) score += 0.08
  if (/[“"'`]/.test(raw)) score += 0.12
  if (/[()（）=]/.test(raw)) score += 0.08
  if (/\d/.test(raw)) score += 0.08
  if (/[A-Z][A-Za-z0-9+-]{1,}/.test(raw)) score += 0.08
  if (/\b(?:gt|ground[ -]?truth|pose|camera|train|training|input|output|pipeline|rendering|volume)\b/i.test(raw)) score += 0.08
  if (/(?:使用|采用|输入|输出|恢复|重建|估计|固定|训练|生成|表征|渲染|约束|对比|利用|先用|再将|来自|对应)/.test(raw)) score += 0.12

  const kind = String(opts?.kind || '').trim().toLowerCase()
  const segmentType = String(opts?.segmentType || '').trim().toLowerCase()
  const evidenceMode = String(opts?.evidenceMode || '').trim().toLowerCase()
  if (kind === 'list_item') score += 0.08
  if (segmentType === 'bullet' || segmentType === 'evidence' || segmentType === 'equation_explanation') score += 0.08
  if (evidenceMode === 'direct') score += 0.06
  if (/[:\uFF1A]$/.test(raw)) score -= 0.18
  return Math.max(0, Math.min(1.2, score))
}

function isLikelyClaimGroupLead(
  input: string,
  opts?: {
    kind?: string
    segmentType?: string
  },
): boolean {
  const raw = stripProvenanceNoise(stripMarkdownInline(String(input || '')))
    .replace(/\s+/g, ' ')
    .trim()
  if (!raw) return false
  if (isLikelyRhetoricalLocateShell(raw)) return true
  if (/[:\uFF1A]$/.test(raw)) return true
  const kind = String(opts?.kind || '').trim().toLowerCase()
  const segmentType = String(opts?.segmentType || '').trim().toLowerCase()
  if ((kind === 'list_item' || segmentType === 'bullet') && /(?:如下|包括|分为|步骤|流程|表明|说明|可见|因此)$/.test(raw)) {
    return true
  }
  return false
}

function isLikelySectionBoundarySegment(segment: StructuredProvenanceSegment | null): boolean {
  if (!segment) return true
  const segmentType = String(segment.segmentType || '').trim().toLowerCase()
  if (segmentType === 'claim' || segmentType === 'next_step') return true
  const text = stripProvenanceNoise(stripMarkdownInline(String(segment.text || '')))
    .replace(/\s+/g, ' ')
    .trim()
  if (!text) return true
  return /^(?:\u7ed3\u8bba|\u6838\u5fc3\u7ed3\u8bba|\u4f9d\u636e|\u8bc1\u636e|\u539f\u6587|\u4e0b\u4e00\u6b65|\u5efa\u8bae|\u98ce\u9669|\u9650\u5236)\s*[:\uFF1A]?/.test(text)
}

void isLikelyClaimGroupLead
void isLikelySectionBoundarySegment

export function mergeStructuredSnippetAliases(...groups: Array<string[] | null | undefined>): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  for (const group of groups) {
    if (!Array.isArray(group)) continue
    for (const item of group) {
      const norm = normalizeStructuredLocateSnippet(String(item || ''))
      if (!norm || seen.has(norm)) continue
      seen.add(norm)
      out.push(norm)
      if (out.length >= 10) return out
    }
  }
  return out
}

export function shortSegmentLabel(input: string, maxLen = 84): string {
  const text = stripMarkdownInline(String(input || '')).replace(/\s+/g, ' ').trim()
  if (!text) return ''
  if (text.length <= maxLen) return text
  return `${text.slice(0, Math.max(18, maxLen - 3)).trimEnd()}...`
}

export function buildStructuredProvenanceLocateEntries(
  messageProvenance: Record<string, unknown> | null,
  opts: {
    guideSourcePath: string
    fallbackSourceName: string
    maxEntries?: number
    minConfidence?: number
  },
): ProvenanceLocateEntry[] {
  const guideSourcePath = String(opts?.guideSourcePath || '').trim()
  const fallbackSourceName = String(opts?.fallbackSourceName || '').trim()
  const maxEntriesRaw = Number(opts?.maxEntries || 3)
  const maxEntries = Number.isFinite(maxEntriesRaw) && maxEntriesRaw > 0 ? Math.floor(maxEntriesRaw) : 3
  const minConfidenceRaw = Number(
    opts?.minConfidence === undefined
      ? 0.62
      : opts?.minConfidence,
  )
  const minConfidence = Number.isFinite(minConfidenceRaw) ? Math.max(0, minConfidenceRaw) : 0.62
  if (!messageProvenance || typeof messageProvenance !== 'object') return []
  const strictIdentityReady = Boolean(messageProvenance.strict_identity_ready)
  const sourcePath = String(messageProvenance.source_path || '').trim()
  if (!sourcePath) return []
  if (guideSourcePath) {
    const guideNorm = normalizeSourcePathForMatch(guideSourcePath)
    const sourceNorm = normalizeSourcePathForMatch(sourcePath)
    if (guideNorm && sourceNorm && !sourcePathsReferToSameDocument(guideNorm, sourceNorm)) return []
  }
  const sourceName = String(messageProvenance.source_name || '').trim()
    || String(fallbackSourceName || '').trim()
    || basenameFromSourcePath(sourcePath)
    || 'paper'
  const blockMap = (messageProvenance.block_map && typeof messageProvenance.block_map === 'object')
    ? messageProvenance.block_map as Record<string, Record<string, unknown>>
    : {}
  const segmentsRaw = Array.isArray(messageProvenance.segments) ? messageProvenance.segments : []
  const segmentsAll = listStructuredProvenanceSegments(messageProvenance)
  if (segmentsRaw.length <= 0 || segmentsAll.length <= 0) return []
  const provenanceById = new Map(segmentsAll.map((segment) => [segment.segmentId, segment]))

  const scoredEntries: Array<{
    entry: ProvenanceLocateEntry
    score: number
    idx: number
  }> = []
  const seenSegment = new Set<string>()
  const seenContent = new Set<string>()
  for (let idx = 0; idx < segmentsAll.length; idx += 1) {
    const segment = segmentsRaw[idx] as Record<string, unknown> | null
    if (!segment || typeof segment !== 'object') continue
    const currentSegment = segmentsAll[idx] || null
    if (!currentSegment) continue
    const rawReaderOpen = coerceReaderOpenPayload((segment as Record<string, unknown>).reader_open)
    const rawLocateTarget = coerceReaderLocateTarget((segment as Record<string, unknown>).locate_target)
      || rawReaderOpen?.locateTarget
      || null
    const evidenceMode = String(segment.evidence_mode || '').trim().toLowerCase()
    const primaryBlockId = String(segment.primary_block_id || rawLocateTarget?.blockId || '').trim()
    const primaryAnchorId = String(segment.primary_anchor_id || rawLocateTarget?.anchorId || '').trim()
    const supportBlockIdsRaw = Array.isArray(segment.support_block_ids) ? segment.support_block_ids : []
    const evidenceBlockIdsRaw = Array.isArray(segment.evidence_block_ids) ? segment.evidence_block_ids : []
    const claimType = String(segment.claim_type || currentSegment.claimType || '').trim().toLowerCase()
    const mustLocate = Boolean(segment.must_locate ?? currentSegment.mustLocate)
    const locatePolicy = String(segment.locate_policy || currentSegment.locatePolicy || '').trim().toLowerCase()
    const locateSurfacePolicy = String(segment.locate_surface_policy || currentSegment.locateSurfacePolicy || '').trim().toLowerCase()
    const claimGroupId = String(segment.claim_group_id || rawReaderOpen?.claimGroup?.id || currentSegment.claimGroupId || '').trim()
    const claimGroupKind = String(segment.claim_group_kind || rawReaderOpen?.claimGroup?.kind || currentSegment.claimGroupKind || '').trim().toLowerCase()
    const formulaOrigin = String(segment.formula_origin || currentSegment.formulaOrigin || '').trim().toLowerCase()
    const segmentAnchorKind = String(segment.anchor_kind || rawLocateTarget?.anchorKind || currentSegment.anchorKind || '').trim().toLowerCase()
    const segmentAnchorText = normalizeStrictAnchorText(
      String(segment.anchor_text || rawLocateTarget?.anchorText || currentSegment.anchorText || ''),
    )
    const segmentEquationNumber = Number.isFinite(Number(
      segment.equation_number
      || (segmentAnchorKind === 'equation' ? rawLocateTarget?.anchorNumber : 0)
      || currentSegment.equationNumber
      || 0,
    ))
      ? Math.max(0, Math.floor(Number(
        segment.equation_number
        || (segmentAnchorKind === 'equation' ? rawLocateTarget?.anchorNumber : 0)
        || currentSegment.equationNumber
        || 0,
      )))
      : 0
    const supportFigureNumber = Number.isFinite(Number(
      segment.support_slot_figure_number
      || (segmentAnchorKind === 'figure' ? rawLocateTarget?.anchorNumber : 0)
      || 0,
    ))
      ? Math.max(0, Math.floor(Number(
        segment.support_slot_figure_number
        || (segmentAnchorKind === 'figure' ? rawLocateTarget?.anchorNumber : 0)
        || 0,
      )))
      : 0
    const supportPanelLetters = Array.isArray(segment.support_slot_panel_letters)
      ? Array.from(
        new Set(
          segment.support_slot_panel_letters
            .map((item) => String(item || '').trim().toLowerCase())
            .filter((item) => /^[a-z]$/.test(item)),
        ),
      )
      : []
    const blockIdsRaw = [
      ...[primaryBlockId].filter(Boolean),
      ...supportBlockIdsRaw.map((item) => String(item || '').trim()).filter(Boolean),
      ...evidenceBlockIdsRaw.map((item) => String(item || '').trim()).filter(Boolean),
      ...coerceStringArray(rawLocateTarget?.blockId, 1, 120),
    ]
    const allowHiddenRequiredEvidence = Boolean(
      locatePolicy === 'hidden'
      && strictIdentityReady
      && mustLocate
      && evidenceMode === 'direct'
      && blockIdsRaw.length > 0
    )
    if (locatePolicy === 'hidden' && !allowHiddenRequiredEvidence) continue
    if (evidenceMode !== 'direct' || blockIdsRaw.length <= 0) continue
    if (claimType === 'shell_sentence' && !mustLocate) continue

    const sourceSegmentId = String(segment.segment_id || '').trim() || `seg_${idx + 1}`
    const evidenceQuote = normalizeStrictAnchorText(String(segment.evidence_quote || rawLocateTarget?.evidenceQuote || segmentAnchorText || ''))
    const headingLikeQuote = claimType === 'quote_claim' && isHeadingLikeQuotedAnchor(segmentAnchorText || evidenceQuote || currentSegment.text)
    if (headingLikeQuote) continue
    const hasStrictIdentity = hasSegmentStrictLocateIdentity(segment, currentSegment)
    if (!hasStrictIdentity) continue
    const isRequiredPolicy = locatePolicy === 'required'
    const effectiveMustLocate = strictIdentityReady && (mustLocate || isRequiredPolicy) && !headingLikeQuote
    if (claimGroupKind === 'formula_bundle' && (locateSurfacePolicy === 'hidden' || formulaOrigin === 'derived')) {
      continue
    }
    const keepSelfTarget = effectiveMustLocate || ['quote_claim', 'blockquote_claim', 'formula_claim', 'inline_formula_claim', 'equation_explanation_claim', 'figure_claim', 'figure_panel'].includes(claimType)
    const targetSegmentId = String(
      segment.claim_group_target_segment_id
      || currentSegment.claimGroupTargetSegmentId
      || currentSegment.segmentId
      || sourceSegmentId,
    ).trim() || sourceSegmentId
    const targetDistanceRaw = Number(
      segment.claim_group_target_distance
      ?? rawReaderOpen?.claimGroup?.distance
      ?? currentSegment.claimGroupTargetDistance
      ?? 0,
    )
    const targetDistance = Number.isFinite(targetDistanceRaw) && targetDistanceRaw > 0
      ? Math.max(0, Math.floor(targetDistanceRaw))
      : 0
    const targetSegment = provenanceById.get(targetSegmentId) || currentSegment
    const segmentId = String(targetSegment.segmentId || sourceSegmentId).trim() || sourceSegmentId
    if (seenSegment.has(segmentId)) continue

    const sourceSegmentText = stripMarkdownInline(String(segment.text || '')).trim()
    const segmentText = stripMarkdownInline(
      String(
        (keepSelfTarget && segmentAnchorText)
        || rawLocateTarget?.snippet
        || targetSegment.anchorText
        || targetSegment.text
        || sourceSegmentText
        || '',
      ),
    ).trim()
    if (!segmentText) continue
    const targetSnippetAliases = Array.isArray(targetSegment.snippetAliases) ? targetSegment.snippetAliases : []
    const sourceSnippetAliases = Array.isArray(segment.snippet_aliases)
      ? segment.snippet_aliases.map((item) => String(item || ''))
      : []
    const snippetKey = normalizeStructuredLocateSnippet(
      String(targetSegment.snippetKey || segment.snippet_key || rawLocateTarget?.snippet || segmentText).trim(),
    )
    const snippetAliases = mergeStructuredSnippetAliases(
      targetSnippetAliases,
      [...sourceSnippetAliases, ...coerceStringArray(rawLocateTarget?.snippetAliases, 8, 360)],
      [segmentAnchorText],
      [segmentText, String(rawLocateTarget?.snippet || '')],
    )
    const candidates: LocateCandidate[] = []
    const seenBlock = new Set<string>()
    for (const blockIdRaw of blockIdsRaw.slice(0, 5)) {
      const blockId = String(blockIdRaw || '').trim()
      if (!blockId || seenBlock.has(blockId)) continue
      const block = blockMap[blockId]
      if (!block || typeof block !== 'object') continue
      seenBlock.add(blockId)
      const blockText = stripMarkdownInline(String(block.text || '')).trim()
      const headingPath = String(block.heading_path || '').trim()
      const anchorId = String(block.anchor_id || '').trim()
      const blockKind = String(block.kind || '').trim().toLowerCase()
      let anchorKind = String(segmentAnchorKind || blockKind || '').trim().toLowerCase()
      if (blockKind === 'equation') anchorKind = 'equation'
      if (blockKind === 'figure') anchorKind = 'figure'
      const anchorNumberRaw = Number(
        segmentEquationNumber
        || supportFigureNumber
        || rawLocateTarget?.anchorNumber
        || block.number
        || 0,
      )
      const focusSnippet = segmentAnchorText || evidenceQuote || rawLocateTarget?.highlightSnippet || blockText || segmentText || headingPath
      if (!focusSnippet) continue
      candidates.push({
        sourcePath,
        sourceName,
        headingPath,
        focusSnippet,
        matchText: [headingPath, segmentAnchorText || evidenceQuote || '', blockText || segmentText].filter(Boolean).join('\n'),
        sourceType: 'guide',
        blockId,
        anchorId: anchorId || undefined,
        anchorKind: anchorKind || undefined,
        anchorNumber: Number.isFinite(anchorNumberRaw) && anchorNumberRaw > 0
          ? Math.floor(anchorNumberRaw)
          : undefined,
      })
    }
    if (candidates.length <= 0) continue

    const rankedCandidates = [...candidates].sort((a, b) => {
      const scoreB = scoreStructuredPrimaryCandidate(b, {
        claimType,
        anchorKind: segmentAnchorKind,
        anchorText: segmentAnchorText,
        evidenceQuote,
        segmentText,
        equationNumber: segmentEquationNumber,
        supportFigureNumber,
        primaryBlockId,
        primaryAnchorId,
      })
      const scoreA = scoreStructuredPrimaryCandidate(a, {
        claimType,
        anchorKind: segmentAnchorKind,
        anchorText: segmentAnchorText,
        evidenceQuote,
        segmentText,
        equationNumber: segmentEquationNumber,
        supportFigureNumber,
        primaryBlockId,
        primaryAnchorId,
      })
      if (scoreB !== scoreA) return scoreB - scoreA
      return candidates.indexOf(a) - candidates.indexOf(b)
    })

    const primary = rankedCandidates[0]
    const alternatives = [
      primary,
      ...rankedCandidates.filter((cand) => cand !== primary),
    ]
    const entry: ProvenanceLocateEntry = {
      segmentId,
      segmentKind: String(targetSegment.kind || currentSegment.kind || '').trim().toLowerCase(),
      label: shortSegmentLabel(segmentAnchorText || evidenceQuote || segmentText || primary.focusSnippet),
      segmentText,
      evidenceQuote,
      readerOpen: rawReaderOpen || undefined,
      locateTarget: rawLocateTarget || undefined,
      hitLevel: String(segment.hit_level || rawLocateTarget?.hitLevel || currentSegment.hitLevel || '').trim().toLowerCase(),
      claimType,
      mustLocate: effectiveMustLocate,
      locatePolicy: locatePolicy || (effectiveMustLocate ? 'required' : ''),
      locateSurfacePolicy,
      claimGroupId,
      claimGroupKind,
      formulaOrigin,
      anchorKind: segmentAnchorKind || primary.anchorKind || '',
      anchorText: segmentAnchorText || evidenceQuote || '',
      equationNumber: segmentEquationNumber || primary.anchorNumber || 0,
      supportFigureNumber,
      supportPanelLetters,
      snippetKey,
      snippetAliases,
      primary,
      alternatives,
      relatedBlockIds: Array.from(new Set([
        ...coerceStringArray((segment as Record<string, unknown>).related_block_ids, 8, 180),
        ...coerceStringArray(rawLocateTarget?.relatedBlockIds, 8, 180),
      ])),
      sourceSegmentId: String(rawLocateTarget?.sourceSegmentId || sourceSegmentId).trim() || sourceSegmentId,
      groupLeadText: targetDistance > 0
        ? String(segment.claim_group_lead_text || rawReaderOpen?.claimGroup?.leadText || currentSegment.claimGroupLeadText || sourceSegmentText || '').trim() || undefined
        : undefined,
      groupDistance: targetDistance,
    }
    if (shouldSuppressNegativeLocateSurface({
      claimType,
      anchorKind: segmentAnchorKind || primary.anchorKind || '',
      segmentText,
      evidenceQuote,
      anchorText: segmentAnchorText || '',
      snippet: primary.focusSnippet || segmentText,
      highlightSnippet: primary.focusSnippet || evidenceQuote || segmentText,
    })) {
      continue
    }
    const contentKey = normalizeLocateText(segmentAnchorText || evidenceQuote || segmentText || primary.focusSnippet).slice(0, 220)
    if (contentKey && seenContent.has(contentKey)) continue
    if (contentKey) seenContent.add(contentKey)
    const evidenceConfidenceRaw = Number(segment.evidence_confidence || 0)
    const evidenceConfidence = Number.isFinite(evidenceConfidenceRaw) ? evidenceConfidenceRaw : 0
    const segmentFormula = hasFormulaSignal(segmentText)
    const contentCoreScore = scoreLocateContentCore(segmentText, {
      kind: targetSegment.kind,
      segmentType: targetSegment.segmentType,
      evidenceMode: targetSegment.evidenceMode,
    })
    const score = evidenceConfidence
      + (segmentFormula ? 0.03 : 0)
      + Math.min(0.18, contentCoreScore * 0.16)
      - Math.min(0.16, targetDistance * 0.06)
      + (effectiveMustLocate ? 0.42 : 0)
      + (claimType === 'formula_claim' ? 0.18 : 0)
      + (claimType === 'inline_formula_claim' ? 0.17 : 0)
      + (claimType === 'equation_explanation_claim' ? 0.16 : 0)
      + (claimType === 'figure_claim' ? 0.16 : 0)
      + (claimType === 'figure_panel' ? 0.2 : 0)
      + ((claimType === 'quote_claim' || claimType === 'blockquote_claim') ? 0.14 : 0)
      + (locateSurfacePolicy === 'primary' ? 0.18 : 0)
      + (locateSurfacePolicy === 'secondary' ? 0.08 : 0)
      + (formulaOrigin === 'source' ? 0.12 : 0)
      - (formulaOrigin === 'derived' ? 0.32 : 0)
      + ((segmentAnchorText || evidenceQuote).length >= 18 ? 0.05 : 0)
    scoredEntries.push({ entry, score, idx })
    seenSegment.add(segmentId)
  }
  if (scoredEntries.length <= 0) return []
  let ranked = scoredEntries
    .filter((item) => {
      if (item.score >= minConfidence) return true
      return Boolean(item.entry.mustLocate || item.entry.locatePolicy === 'required')
    })
    .sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score
      return a.idx - b.idx
    })
  if (ranked.length <= 0) {
    ranked = [...scoredEntries].sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score
      return a.idx - b.idx
    })
  }
  const bestFormulaBundleByKey = new Map<string, (typeof ranked)[number]>()
  for (const item of ranked) {
    const groupKey = formulaBundleLocateGroupKey(item.entry)
    if (!groupKey) continue
    const prev = bestFormulaBundleByKey.get(groupKey)
    if (!prev) {
      bestFormulaBundleByKey.set(groupKey, item)
      continue
    }
    const itemRepScore = formulaBundleRepresentativeScore(item.entry)
    const prevRepScore = formulaBundleRepresentativeScore(prev.entry)
    if (itemRepScore !== prevRepScore) {
      if (itemRepScore > prevRepScore) bestFormulaBundleByKey.set(groupKey, item)
      continue
    }
    if (item.score !== prev.score) {
      if (item.score > prev.score) bestFormulaBundleByKey.set(groupKey, item)
      continue
    }
    if (item.idx < prev.idx) bestFormulaBundleByKey.set(groupKey, item)
  }
  if (bestFormulaBundleByKey.size > 0) {
    ranked = ranked.filter((item) => {
      const groupKey = formulaBundleLocateGroupKey(item.entry)
      if (!groupKey) return true
      return bestFormulaBundleByKey.get(groupKey) === item
    })
  }
  const mustLocateEntries = ranked.filter((item) => item.entry.mustLocate || item.entry.locatePolicy === 'required')
  const optionalEntries = ranked.filter((item) => !(item.entry.mustLocate || item.entry.locatePolicy === 'required'))
  const limited = [
    ...mustLocateEntries,
    ...optionalEntries.slice(0, Math.max(0, maxEntries - mustLocateEntries.length)),
  ]

  // Deduplicate by primary evidence block id to avoid repeating the same source block across multiple segments.
  // Prefer required/must-locate entries; otherwise prefer higher score.
  const bestByBlock = new Map<string, (typeof limited)[number]>()
  for (const item of limited) {
    const blockId = String(item.entry?.primary?.blockId || '').trim()
    const anchorId = String(item.entry?.primary?.anchorId || '').trim()
    const key = blockId || (anchorId ? `${item.entry.primary.sourcePath}::${anchorId}` : '')
    if (!key) continue
    const prev = bestByBlock.get(key)
    if (!prev) {
      bestByBlock.set(key, item)
      continue
    }
    const prevRequired = Boolean(prev.entry.mustLocate || prev.entry.locatePolicy === 'required')
    const curRequired = Boolean(item.entry.mustLocate || item.entry.locatePolicy === 'required')
    if (curRequired && !prevRequired) {
      bestByBlock.set(key, item)
      continue
    }
    if (curRequired === prevRequired) {
      if (item.score > prev.score + 1e-6) {
        bestByBlock.set(key, item)
        continue
      }
      if (Math.abs(item.score - prev.score) <= 1e-6 && item.idx < prev.idx) {
        bestByBlock.set(key, item)
        continue
      }
    }
  }
  const deduped = Array.from(bestByBlock.values())
  return deduped
    .sort((a, b) => a.idx - b.idx)
    .map((item) => item.entry)
}

export function listStructuredProvenanceSegments(
  messageProvenance: Record<string, unknown> | null,
): StructuredProvenanceSegment[] {
  if (!messageProvenance || typeof messageProvenance !== 'object') return []
  const segmentsRaw = Array.isArray(messageProvenance.segments) ? messageProvenance.segments : []
  const out: StructuredProvenanceSegment[] = []
  for (let idx = 0; idx < segmentsRaw.length; idx += 1) {
    const segment = segmentsRaw[idx] as Record<string, unknown> | null
    if (!segment || typeof segment !== 'object') continue
    const segmentId = String(segment.segment_id || '').trim() || `seg_${idx + 1}`
    const text = stripMarkdownInline(String(segment.text || '')).replace(/\s+/g, ' ').trim()
    const snippetKeyRaw = String(segment.snippet_key || '').trim()
    const snippetAliases = Array.isArray(segment.snippet_aliases)
      ? segment.snippet_aliases
        .map((item) => normalizeStructuredLocateSnippet(String(item || '').trim()))
        .filter(Boolean)
        .slice(0, 8)
      : []
    out.push({
      index: idx,
      segmentId,
      kind: String(segment.kind || '').trim().toLowerCase(),
      segmentType: String(segment.segment_type || '').trim().toLowerCase(),
      evidenceMode: String(segment.evidence_mode || '').trim().toLowerCase(),
      hitLevel: String(segment.hit_level || '').trim().toLowerCase(),
      claimType: String(segment.claim_type || '').trim().toLowerCase(),
      mustLocate: Boolean(segment.must_locate),
      locatePolicy: String(segment.locate_policy || '').trim().toLowerCase(),
      locateSurfacePolicy: String(segment.locate_surface_policy || '').trim().toLowerCase(),
      claimGroupId: String(segment.claim_group_id || '').trim(),
      claimGroupKind: String(segment.claim_group_kind || '').trim().toLowerCase(),
      claimGroupTargetSegmentId: String(segment.claim_group_target_segment_id || '').trim(),
      claimGroupTargetDistance: Number.isFinite(Number(segment.claim_group_target_distance || 0))
        ? Math.max(0, Math.floor(Number(segment.claim_group_target_distance || 0)))
        : 0,
      claimGroupLeadText: stripMarkdownInline(String(segment.claim_group_lead_text || '')).replace(/\s+/g, ' ').trim(),
      formulaOrigin: String(segment.formula_origin || '').trim().toLowerCase(),
      anchorKind: String(segment.anchor_kind || '').trim().toLowerCase(),
      anchorText: stripMarkdownInline(String(segment.anchor_text || '')).replace(/\s+/g, ' ').trim(),
      equationNumber: Number.isFinite(Number(segment.equation_number || 0))
        ? Math.max(0, Math.floor(Number(segment.equation_number || 0)))
        : 0,
      text,
      snippetKey: normalizeStructuredLocateSnippet(snippetKeyRaw || text.slice(0, 360)),
      snippetAliases,
    })
  }
  return out
}

export function normalizeStructuredLocateSnippet(input: string): string {
  const raw = stripProvenanceNoise(stripMarkdownInline(String(input || '')))
    .replace(/\s+/g, ' ')
    .trim()
  if (!raw) return ''
  const trimmed = raw
    .replace(/\.{3,}\s*$/, '')
    .replace(/\u2026+\s*$/, '')
    .trim()
  return normalizeLocateText(trimmed)
}
