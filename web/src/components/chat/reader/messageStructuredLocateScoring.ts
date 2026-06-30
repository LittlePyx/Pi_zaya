import {
  formulaOverlapScore,
  hasFormulaSignal,
  normalizeLocateText,
  overlapScore,
  stripMarkdownInline,
  stripProvenanceNoise,
  tokenizeLocateText,
  type LocateCandidate,
} from './messageLocateCandidates'

export type StructuredLocateKind = 'paragraph' | 'list_item' | 'quote' | 'blockquote' | 'equation' | 'figure'

export interface StructuredLocateEntryLike {
  anchorKind?: string
  claimType?: string
  mustLocate?: boolean
}

export function normalizeStructuredLocateKind(input: string): StructuredLocateKind | '' {
  const raw = String(input || '').trim().toLowerCase()
  if (!raw) return ''
  if (raw === 'equation' || raw === 'math') return 'equation'
  if (raw === 'figure' || raw === 'fig' || raw === 'image' || raw === 'img') return 'figure'
  if (raw === 'list_item' || raw === 'list-item' || raw === 'li') return 'list_item'
  if (raw === 'quote' || raw === 'quoted_text') return 'quote'
  if (raw === 'blockquote' || raw === 'bq') return 'blockquote'
  if (raw === 'paragraph' || raw === 'p') return 'paragraph'
  return ''
}

export function isPreferredStrictFigureRefSnippet(input: string): boolean {
  const raw = stripProvenanceNoise(stripMarkdownInline(String(input || '')))
    .replace(/\s+/g, ' ')
    .trim()
  if (!raw) return false
  if (/^(?:figure|fig\.?|[\u56fe\u5716])\s*#?\s*\d{1,4}$/i.test(raw)) return true
  return false
}

export function isHeadingLikeQuotedAnchor(text: string): boolean {
  const raw = stripProvenanceNoise(stripMarkdownInline(String(text || '')))
    .replace(/\s+/g, ' ')
    .trim()
  if (!raw) return true
  if (/[。！？.!?;；:：]/.test(raw)) return false
  const latin = raw.match(/[A-Za-z]{3,}/g) || []
  if (latin.length > 0 && latin.length <= 8 && raw.length <= 80) {
    const verbLike = /\b(?:is|are|was|were|be|been|being|can|cannot|could|should|would|will|use|used|using|estimate|estimated|show|shown|train|training|feed|feeding|make|making|exploit|provide|compare)\b/i
    if (!verbLike.test(raw)) return true
  }
  return raw.length <= 28 && !/\d/.test(raw)
}

export function scoreStructuredAnchorCompatibility(
  renderKindInput: string,
  entry: StructuredLocateEntryLike | null | undefined,
): number {
  const renderKind = normalizeStructuredLocateKind(renderKindInput)
  const anchorKind = String(entry?.anchorKind || '').trim().toLowerCase()
  const claimType = String(entry?.claimType || '').trim().toLowerCase()
  if (!renderKind) return 0
  if (claimType === 'equation_explanation_claim') {
    if (renderKind === 'paragraph' || renderKind === 'list_item') return 0.66
    if (renderKind === 'blockquote') return 0.18
    if (renderKind === 'equation') return 0.12
    return -0.4
  }
  if (claimType === 'inline_formula_claim' || anchorKind === 'inline_formula') {
    if (renderKind === 'equation') return 0.74
    if (renderKind === 'paragraph' || renderKind === 'list_item') return 0.58
    if (renderKind === 'blockquote') return 0.2
    return -0.28
  }
  if (anchorKind === 'blockquote' || claimType === 'blockquote_claim') {
    return renderKind === 'blockquote' ? 0.72 : -1.2
  }
  if (anchorKind === 'equation' || claimType === 'formula_claim') {
    if (renderKind === 'equation') return 0.86
    return -1.05
  }
  if (claimType === 'figure_panel') {
    if (renderKind === 'figure') return 0.86
    if (renderKind === 'blockquote') return 0.64
    if (renderKind === 'paragraph' || renderKind === 'list_item') return 0.52
    return -0.54
  }
  if (anchorKind === 'figure' || claimType === 'figure_claim') {
    if (renderKind === 'figure') return 0.8
    if (renderKind === 'paragraph' || renderKind === 'list_item') return 0.08
    return -0.92
  }
  if (anchorKind === 'quote' || claimType === 'quote_claim') {
    if (renderKind === 'quote') return 0.88
    if (renderKind === 'blockquote') return 0.58
    if (renderKind === 'paragraph' || renderKind === 'list_item') return 0.28
    return -0.5
  }
  return 0
}

export function extractFigureNumbersFromText(text: string): number[] {
  const src = String(text || '')
  if (!src) return []
  const out: number[] = []
  const seen = new Set<number>()
  const push = (raw: string) => {
    const n = Number(raw)
    if (!Number.isFinite(n) || n <= 0) return
    const k = Math.floor(n)
    if (seen.has(k)) return
    seen.add(k)
    out.push(k)
  }
  for (const m of src.matchAll(/\b(?:fig(?:ure)?\.?\s*#?\s*(\d{1,4})|[\u56fe\u5716]\s*(\d{1,4}))\b/gi)) {
    push(String(m[1] || m[2] || ''))
  }
  return out
}

export function extractPanelLettersFromText(text: string): string[] {
  const src = String(text || '')
  if (!src) return []
  const out: string[] = []
  const seen = new Set<string>()
  const push = (raw: string) => {
    const ch = String(raw || '').trim().toLowerCase()
    if (!/^[a-z]$/.test(ch)) return
    if (seen.has(ch)) return
    seen.add(ch)
    out.push(ch)
  }
  for (const m of src.matchAll(/\bpanel\s*[([]?\s*([a-z])\s*[\])]?/gi)) {
    push(String(m[1] || ''))
  }
  for (const m of src.matchAll(/(?:^|[\s,;:])\(\s*([a-z])\s*\)(?=[\s,;:.]|$)/gi)) {
    push(String(m[1] || ''))
  }
  const lead = src.match(/^\s*([a-z])\s+(?:the|an|a)\b/i)
  if (lead) push(String(lead[1] || ''))
  return out
}

export function panelLetterMatchScore(text: string, letters: string[]): number {
  const target = Array.from(new Set((letters || []).map((item) => String(item || '').trim().toLowerCase()).filter((item) => /^[a-z]$/.test(item))))
  if (target.length <= 0) return 0
  const candidate = new Set(extractPanelLettersFromText(text))
  if (candidate.size <= 0) return 0
  let overlap = 0
  for (const item of target) {
    if (candidate.has(item)) overlap += 1
  }
  if (overlap <= 0) return 0
  return overlap / Math.max(1, target.length)
}

export function figureNumberMatchScore(text: string, numbers: number[]): number {
  const src = String(text || '')
  if (!src || numbers.length <= 0) return 0
  let best = 0
  for (const num of numbers) {
    if (new RegExp(`\\bfig(?:ure)?\\.?\\s*#?\\s*${num}\\b`, 'i').test(src)) best = Math.max(best, 1.0)
    if (new RegExp(`[\\u56fe\\u5716]\\s*${num}\\b`).test(src)) best = Math.max(best, 1.0)
  }
  return best
}

export function scoreProvenanceSegment(snippet: string, segmentText: string, segmentKey: string): number {
  const raw = stripProvenanceNoise(stripMarkdownInline(String(snippet || '')))
  const query = normalizeLocateText(raw)
  const segNorm = normalizeLocateText(stripProvenanceNoise(String(segmentText || '')))
  const keyNorm = normalizeLocateText(stripProvenanceNoise(String(segmentKey || '')))
  if (!query || (!segNorm && !keyNorm)) return 0

  let score = 0
  if (keyNorm) {
    if (keyNorm === query) score += 1.2
    if (keyNorm.includes(query)) score += 0.92
    if (query.includes(keyNorm) && keyNorm.length >= 20) score += 0.78
  }
  if (segNorm) {
    if (segNorm === query) score += 1.15
    if (segNorm.includes(query)) score += 0.86
    const segHead = segNorm.slice(0, Math.min(220, segNorm.length))
    if (query.includes(segHead) && segHead.length >= 20) score += 0.72
  }

  score += 0.82 * Math.max(
    overlapScore(raw, segmentText),
    overlapScore(query, segNorm),
    overlapScore(query, keyNorm),
  )

  if (hasFormulaSignal(raw) || hasFormulaSignal(segmentText)) {
    score += 0.72 * formulaOverlapScore(raw, segmentText)
  }
  return score
}

export function extractEquationNumbersFromText(text: string): number[] {
  const src = String(text || '')
  if (!src) return []
  const out: number[] = []
  const seen = new Set<number>()
  const push = (raw: string) => {
    const n = Number(raw)
    if (!Number.isFinite(n) || n <= 0) return
    const k = Math.floor(n)
    if (seen.has(k)) return
    seen.add(k)
    out.push(k)
  }
  for (const m of src.matchAll(/\b(?:eq|equation|\u516C\u5F0F)\s*[#(\uFF08]?\s*(\d{1,4})\s*[)\uFF09]?/gi)) {
    push(String(m[1] || ''))
  }
  for (const m of src.matchAll(/\((\d{1,4})\)/g)) {
    push(String(m[1] || ''))
  }
  return out
}

export function scoreLocateCandidate(snippet: string, cand: LocateCandidate): number {
  const query = String(snippet || '').trim()
  if (!query) return 0
  const qNorm = normalizeLocateText(query)
  const focusText = String(cand.focusSnippet || '').trim()
  const matchText = String(cand.matchText || focusText).trim()
  if (!matchText) return 0
  const mNorm = normalizeLocateText(matchText)
  const fNorm = normalizeLocateText(focusText)

  let score = Math.max(
    overlapScore(query, matchText),
    overlapScore(query, focusText) * 1.08,
  )

  if (qNorm && mNorm) {
    if (mNorm.includes(qNorm)) score += 0.7
    const qHead = qNorm.slice(0, Math.min(64, qNorm.length))
    if (qHead.length >= 18 && mNorm.includes(qHead)) score += 0.26
    const mHead = mNorm.slice(0, Math.min(64, mNorm.length))
    if (mHead.length >= 18 && qNorm.includes(mHead)) score += 0.18
  }
  if (qNorm && fNorm) {
    if (fNorm.includes(qNorm)) score += 0.2
    const qHead = qNorm.slice(0, Math.min(48, qNorm.length))
    if (qHead.length >= 16 && fNorm.includes(qHead)) score += 0.14
  }

  const tokenSet = new Set(tokenizeLocateText(matchText))
  const keyTokens = Array.from(new Set(tokenizeLocateText(query))).filter((token) => token.length >= 3)
  let hitCount = 0
  for (const token of keyTokens) {
    if (tokenSet.has(token)) hitCount += 1
  }
  if (hitCount > 0) {
    score += Math.min(0.36, 0.03 * hitCount)
  }
  if (query.length >= 80 && focusText.length >= 80) {
    score += 0.05
  }
  if (hasFormulaSignal(query) || hasFormulaSignal(matchText)) {
    score += 0.72 * formulaOverlapScore(query, matchText)
  }
  const qEqNumbers = extractEquationNumbersFromText(query)
  const candEqNo = Number(cand.anchorNumber || 0)
  const candKind = String(cand.anchorKind || '').trim().toLowerCase()
  if (qEqNumbers.length > 0) {
    if (candEqNo > 0 && qEqNumbers.includes(candEqNo)) score += 1.05
    if (candKind === 'equation') score += 0.18
  }
  if (hasFormulaSignal(query) && candKind === 'equation') {
    score += 0.22
  }
  if (cand.anchorId) {
    score += 0.04
  }
  if (cand.sourceType === 'guide') {
    score += 0.07
  }
  return score
}

export function scoreStructuredPrimaryCandidate(
  cand: LocateCandidate,
  opts: {
    claimType?: string
    anchorKind?: string
    anchorText?: string
    evidenceQuote?: string
    segmentText?: string
    equationNumber?: number
    supportFigureNumber?: number
    primaryBlockId?: string
    primaryAnchorId?: string
  },
): number {
  const claimType = String(opts.claimType || '').trim().toLowerCase()
  const anchorKind = String(opts.anchorKind || '').trim().toLowerCase()
  const anchorText = String(opts.anchorText || '').trim()
  const evidenceQuote = String(opts.evidenceQuote || '').trim()
  const segmentText = String(opts.segmentText || '').trim()
  const seed = anchorText || evidenceQuote || segmentText || String(cand.focusSnippet || '').trim()
  let score = scoreLocateCandidate(seed, cand)

  const candKind = String(cand.anchorKind || '').trim().toLowerCase()
  const candHeading = String(cand.headingPath || '').trim().toLowerCase()
  const candNumber = Number.isFinite(Number(cand.anchorNumber || 0))
    ? Math.max(0, Math.floor(Number(cand.anchorNumber || 0)))
    : 0
  const equationNumber = Number.isFinite(Number(opts.equationNumber || 0))
    ? Math.max(0, Math.floor(Number(opts.equationNumber || 0)))
    : 0
  const figureNumber = Number.isFinite(Number(opts.supportFigureNumber || 0))
    ? Math.max(0, Math.floor(Number(opts.supportFigureNumber || 0)))
    : 0

  if (opts.primaryBlockId && String(cand.blockId || '').trim() === String(opts.primaryBlockId || '').trim()) {
    score += 0.12
  }
  if (opts.primaryAnchorId && String(cand.anchorId || '').trim() === String(opts.primaryAnchorId || '').trim()) {
    score += 0.08
  }
  if (anchorKind && candKind === anchorKind) {
    score += 0.42
  }

  if (claimType === 'formula_claim') {
    if (candKind === 'equation') score += 1.55
    else if (candKind) score -= 0.72
    if (equationNumber > 0 && candNumber === equationNumber) score += 0.95
    if (candHeading.includes('figure')) score -= 0.26
  } else if (claimType === 'inline_formula_claim') {
    if (candKind === 'equation') score += 1.1
    else if (candKind === 'paragraph' || candKind === 'list_item' || candKind === 'blockquote') score += 0.58
  } else if (claimType === 'equation_explanation_claim') {
    const equationScoped = anchorKind === 'equation' || equationNumber > 0
    if (equationScoped) {
      if (candKind === 'equation') score += 0.96
      else if (candKind === 'paragraph' || candKind === 'list_item' || candKind === 'blockquote') score += 0.26
      else if (candKind) score -= 0.24
    } else {
      if (candKind === 'equation') score -= 0.62
      if (candKind === 'paragraph' || candKind === 'list_item' || candKind === 'blockquote') score += 0.74
    }
    if (equationNumber > 0 && candNumber === equationNumber) score += 0.18
  } else if (claimType === 'figure_claim') {
    if (candKind === 'figure') score += 1.18
    else if (candKind) score -= 0.34
    if (figureNumber > 0 && candNumber === figureNumber) score += 0.88
    else if (figureNumber > 0 && candNumber > 0) score -= 0.22
  } else if (claimType === 'figure_panel') {
    if (candKind === 'figure') score += 1.36
    else if (candKind === 'paragraph' || candKind === 'list_item' || candKind === 'blockquote') score += 0.34
    else if (candKind) score -= 0.24
    if (candHeading.includes('figure')) score += 0.22
    if (figureNumber > 0 && candNumber === figureNumber) score += 1.04
    else if (figureNumber > 0 && candNumber > 0) score -= 0.3
  } else if (claimType === 'quote_claim') {
    if (candKind === 'quote') score += 1.02
    else if (candKind === 'blockquote') score += 0.48
    else if (candKind) score -= 0.2
  } else if (claimType === 'blockquote_claim') {
    if (candKind === 'blockquote') score += 1.0
    else if (candKind === 'quote') score += 0.42
    else if (candKind) score -= 0.18
  } else if (claimType === 'method_detail' || claimType === 'prior_work' || claimType === 'doc_map') {
    if (candKind === 'paragraph' || candKind === 'list_item' || candKind === 'blockquote') score += 0.34
    if (candKind === 'equation' || candKind === 'figure') score -= 0.28
  }

  return score
}
