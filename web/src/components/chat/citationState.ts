export interface CitationCardViewSection {
  id: string
  label: string
  text: string
  kind: string
  hint: string
  tone: string
}

export interface CitationCardView {
  version: number
  route: string
  kind: string
  header: {
    kicker: string
    title: string
    subtitle: string
  }
  sections: CitationCardViewSection[]
  summary: string
  quality: {
    label: string
    score: number
    flags: string[]
    warning: string
  }
}

export interface CiteDetail {
  num: number
  displayNum?: number
  displayNums?: number[]
  anchor: string
  sourceName: string
  sourcePath: string
  traceConvId: string
  traceAssistantMsgId: number
  traceAssistantOrder: number
  traceUserMsgId: number
  raw: string
  citeFmt: string
  isInpaper: boolean
  title: string
  authors: string
  venue: string
  year: string
  volume: string
  issue: string
  pages: string
  doi: string
  doiUrl: string
  linkedNums: number[]
  evidenceFingerprint: string
  renderLocale: string
  citationRoute: string
  routingReason: string
  routingConfidence: number
  citationCount: number
  citationSource: string
  venueKind: string
  venueVerifiedBy: string
  openalexVenue: string
  journalIf: string
  journalQuartile: string
  journalIfSource: string
  conferenceTier: string
  conferenceRankSource: string
  conferenceCcf: string
  conferenceCcfSource: string
  conferenceName: string
  conferenceAcronym: string
  bibliometricsChecked: boolean
  libraryMatchStatus: string
  libraryMatchConfidence: number
  libraryMatchMethod: string
  libraryMatchReason: string
  libraryMatchPath: string
  libraryMatchSha1: string
  libraryMatchTitle: string
  libraryMatchDoi: string
  libraryMatchYear: string
  metadataQuality: Record<string, unknown> | null
  metadataRepairStatus: string
  metadataRepairSources: string[]
  metadataChangedFields: string[]
  externalMetadataStatus: string
  externalMetadataReason: string
  externalMatchMethod: string
  externalMatchScore: number
  externalTitleSimilarity: number
  externalTitle: string
  externalAuthors: string
  externalVenue: string
  externalYear: string
  externalDoi: string
  externalDoiUrl: string
  summaryLine: string
  summarySource: string
  summaryProvider: string
  summaryQuality: Record<string, unknown> | null
  shelfItemKind: string
  shelfOrigin: string
  shelfExcerpt: string
  shelfExcerptLabel: string
  answerClaim: string
  headingPath: string
  evidenceQuote: string
  evidenceSource: string
  citationContext: string
  citationContextSource: string
  upstreamWorkRole: string
  userQuestionRelation: string
  locationLabel: string
  supportRelation: string
  whyLine: string
  blockId: string
  anchorId: string
  anchorKind: string
  pageStart: number
  pageEnd: number
  score: number
  bindingStatus: string
  bindingConfidence: number
  bindingReason: string
  bindingOverlapTerms: string[]
  cardKind: string
  cardTitle: string
  cardSubtitle: string
  cardTakeawayLabel: string
  cardTakeaway: string
  cardClaimLabel: string
  cardClaim: string
  cardLocatorLabel: string
  cardLocator: string
  cardEvidenceLabel: string
  cardEvidence: string
  cardContextSummary: string
  cardReferenceLabel: string
  cardReferenceEntry: string
  cardSupportLabel: string
  cardSupportExplanation: string
  cardQualityLabel: string
  cardQualityScore: number
  cardQualityFlags: string[]
  cardWarning: string
  cardFlow: string[]
  cardDisplayContractVersion: number
  cardVisibleSections: string[]
  cardView: CitationCardView | null
  systemBTraceComplete: boolean
  systemBTraceScore: number
  systemBTraceReason: string
  systemBTraceFlags: string[]
  systemBTraceSteps: string[]
  systemBTraceAnswer: string
  systemBTraceContext: string
  systemBTraceReference: string
  systemBTraceLocator: string
  systemBTraceSource: string
  citationCardPolishStatus: string
  citationCardPolishSource: string
  citationCardPolishChecked: boolean
  citationCardPolishKey: string
  citationCardPolishRoute: string
  citationCardPolishFields: string[]
  citationCardPolishRejected: string[]
  citationCardPolishQualityScore: number
}

export interface CiteShelfItem extends CiteDetail {
  key: string
  main: string
  tags: string[]
  note: string
}

export type ShelfItemKind = 'citation' | 'reference' | 'reader_selection' | 'excerpt'

function asText(value: unknown): string {
  if (typeof value === 'string') return value.trim()
  if (typeof value === 'number' && Number.isFinite(value)) return String(value)
  return ''
}

export function cleanCitationDisplayText(value: string): string {
  return String(value || '')
    .replace(/<!--[\s\S]*?-->/g, ' ')
    .replace(/(?:\$\s*)?\^\{\s*\[[\d,\-\s;]+\]\s*\}(?:\s*\$)?/g, ' ')
    .replace(/\\textsuperscript\{\s*\[[^\]\n]{1,80}\]\s*\}/gi, ' ')
    .replace(/\\(?:cite|citep|citet|citealp|upcite)\s*\{[^}\n]{1,200}\}/gi, ' ')
    .replace(/\[\[?\s*CITE\s*:[^\]\n]{1,160}\]?\]?/gi, ' ')
    .replace(/^\s{0,3}#{1,6}\s+/gm, '')
    .replace(/^\s{0,3}>\s?/gm, '')
    .replace(/^\s{0,3}[-*+]\s+/gm, '')
    .replace(/^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?\s*$/gm, ' ')
    .replace(/^\s*\|/gm, '')
    .replace(/\|\s*$/gm, '')
    .replace(/\s*\|\s*/g, ' ')
    .replace(/\$([^$\n]{1,160})\$/g, '$1')
    .replace(/!\[[^\]]*]\([^)]+\)/g, ' ')
    .replace(/\[([^\]]+)]\([^)]+\)/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/~~([^~]+)~~/g, '$1')
    .replace(/\\(?=\s|[,;])/g, ' ')
    .replace(/(^|\s)#{1,6}\s+/g, ' ')
    .replace(/\s*\|\s*/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/^(?:\.{2,}|…)+\s*/, '')
}

export function normalizeShelfItemKind(value: string | null | undefined): ShelfItemKind {
  const key = String(value || '')
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, '_')
  if (key === 'reference' || key === 'inpaper' || key === 'reader_reference' || key === 'reader_references') return 'reference'
  if (key === 'reader_selection' || key === 'selection' || key === 'reader_excerpt') return 'reader_selection'
  if (key === 'excerpt' || key === 'note') return 'excerpt'
  return 'citation'
}

export function inferShelfItemKind(detail: Partial<CiteDetail>): ShelfItemKind {
  const explicit = String(detail.shelfItemKind || '').trim()
  if (explicit) return normalizeShelfItemKind(explicit)
  const cardKind = String(detail.cardKind || '').trim().toLowerCase()
  const evidenceSource = String(detail.evidenceSource || '').trim().toLowerCase()
  const citationContextSource = String(detail.citationContextSource || '').trim().toLowerCase()
  if (cardKind === 'reader_selection' || evidenceSource === 'reader_selection' || citationContextSource === 'reader_selection') {
    return 'reader_selection'
  }
  if (detail.isInpaper) return 'reference'
  return 'citation'
}

export function shelfItemKindLabel(kind: string | null | undefined, S: Record<string, string>): string {
  const normalized = normalizeShelfItemKind(kind)
  if (normalized === 'reference') return S.shelf_type_reference || 'Reference'
  if (normalized === 'reader_selection') return S.shelf_type_reader_selection || 'Reader excerpt'
  if (normalized === 'excerpt') return S.shelf_type_excerpt || 'Excerpt'
  return S.shelf_type_citation || 'Citation'
}

function normalizeShelfOriginKey(value: string | null | undefined): string {
  const key = String(value || '')
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, '_')
  if (key === 'reader_reference' || key === 'reader_reference_entry') return 'reader_references'
  if (key === 'selection') return 'reader_selection'
  if (key === 'answer') return 'chat_answer'
  return key
}

export function shelfOriginLabel(origin: string | null | undefined, S: Record<string, string>): string {
  const key = normalizeShelfOriginKey(origin)
  if (!key) return ''
  const label = S[`shelf_origin_${key}`]
  if (label) return label
  return key
    .split('_')
    .filter(Boolean)
    .map((part) => part.slice(0, 1).toUpperCase() + part.slice(1))
    .join(' ')
}

function inferShelfOrigin(detail: Partial<CiteDetail>, kind: ShelfItemKind): string {
  const explicit = normalizeShelfOriginKey(detail.shelfOrigin)
  if (explicit) return explicit
  const evidenceSource = normalizeShelfOriginKey(detail.evidenceSource)
  const contextSource = normalizeShelfOriginKey(detail.citationContextSource)
  if (kind === 'reader_selection') return 'reader_selection'
  if (contextSource.startsWith('reader_')) return contextSource
  if (evidenceSource.startsWith('reader_')) return evidenceSource
  if (detail.isInpaper) return 'reader_cross_reference'
  if (Number(detail.traceAssistantOrder || 0) > 0 || Number(detail.traceAssistantMsgId || 0) > 0) return 'chat_answer'
  if (kind === 'reference') return 'reference'
  return 'citation'
}

function defaultShelfExcerptLabel(kind: ShelfItemKind): string {
  if (kind === 'reference') return 'Reference entry'
  if (kind === 'reader_selection') return 'Selected text'
  return 'Excerpt'
}

function inferShelfExcerpt(detail: Partial<CiteDetail>, kind: ShelfItemKind): string {
  const explicit = cleanCitationDisplayText(String(detail.shelfExcerpt || ''))
  if (explicit) return explicit
  const candidates = kind === 'reference'
    ? [detail.citationContext, detail.cardReferenceEntry, detail.raw, detail.citeFmt]
    : kind === 'reader_selection'
      ? [detail.evidenceQuote, detail.citationContext, detail.raw, detail.cardEvidence]
      : [detail.evidenceQuote, detail.cardEvidence, detail.citationContext, detail.answerClaim, detail.raw]
  for (const candidate of candidates) {
    const text = cleanCitationDisplayText(String(candidate || ''))
    if (text) return text.slice(0, 1600)
  }
  return ''
}

function looseTokens(value: string): string[] {
  return Array.from(String(value || '').matchAll(/[A-Za-z0-9]+|[\u4e00-\u9fff]+/g)).map((match) => match[0].toLowerCase())
}

function substantiallySameText(left: string, right: string): boolean {
  const a = cleanCitationDisplayText(left).replace(/\s+/g, ' ').toLowerCase()
  const b = cleanCitationDisplayText(right).replace(/\s+/g, ' ').toLowerCase()
  if (!a || !b) return false
  if (a === b) return true
  if (a.length >= 36 && b.includes(a)) return true
  if (b.length >= 36 && a.includes(b)) return true
  const at = new Set(looseTokens(a).filter((token) => token.length >= 2))
  const bt = new Set(looseTokens(b).filter((token) => token.length >= 2))
  if (at.size < 5 || bt.size < 5) return false
  let overlap = 0
  for (const token of at) {
    if (bt.has(token)) overlap += 1
  }
  return overlap / Math.min(at.size, bt.size) >= 0.82
}

function sourceTitleCandidate(value: string): string {
  const name = String(value || '')
    .trim()
    .split(/[\\/]/)
    .pop() || ''
  return cleanCitationDisplayText(name)
    .replace(/\.(?:pdf|md)$/i, '')
    .replace(/\.en$/i, '')
    .replace(/^[A-Za-z]{2,12}-\d{4}-/, '')
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function stripTokenPrefix(text: string, candidate: string): string {
  const candidateTokens = looseTokens(candidate)
  if (candidateTokens.length < 4) return text
  const matches = Array.from(String(text || '').matchAll(/[A-Za-z0-9]+|[\u4e00-\u9fff]+/g))
  if (matches.length < candidateTokens.length) return text
  let matched = 0
  const limit = Math.min(candidateTokens.length, matches.length)
  for (let idx = 0; idx < limit; idx += 1) {
    if (matches[idx][0].toLowerCase() !== candidateTokens[idx]) break
    matched += 1
  }
  if (matched < Math.min(8, candidateTokens.length)) return text
  const end = (matches[matched - 1].index || 0) + matches[matched - 1][0].length
  return text.slice(end).replace(/^[\s,.;:，。；：-]+/, '')
}

function looksAuthorMetadataPrefix(value: string): boolean {
  const text = String(value || '').trim()
  if (text.length < 16) return false
  const commaCount = (text.match(/[,，]/g) || []).length
  const namePairs = (text.match(/\b[A-Z][a-zA-Z'`-]+\s+[A-Z][a-zA-Z'`-]+\b/g) || []).length
  const tokens = looseTokens(text)
  if (commaCount >= 2 || namePairs >= 2) return true
  return tokens.length >= 8 && /[*\\]/.test(text)
}

const BRACKET_REFERENCE_MARKER_RE = /\[\s*\d{1,4}(?:\s*[-,;]\s*\d{1,4})*\s*\]/g
const CONTENT_VERB_RE = /\b(?:is|are|was|were|be|been|being|can|could|may|might|will|would|uses?|used|shows?|shown|proposes?|proposed|demonstrates?|develops?|developed|introduces?|introduced|improves?|improved|captures?|captured|reconstructs?|reconstructed|enables?|enabled|adopts?|adopted|adopting|offers?|offering|collects?|collecting|employs?|employed|employing|解决|提出|说明|表明|用于|能够|可以|实现|采用|提升|降低)\b/i

function looksAuthorListContext(value: string): boolean {
  const text = cleanCitationDisplayText(value)
  if (text.length < 24) return false
  const markerCount = (text.match(BRACKET_REFERENCE_MARKER_RE) || []).length
  const commaCount = (text.match(/[,，]/g) || []).length
  const namePairs = (text.match(/\b[A-Z][a-zA-Z'`-]+\s+[A-Z][a-zA-Z'`-]+\b/g) || []).length
  if (markerCount >= 3 && (namePairs >= 3 || commaCount >= 4)) return true
  if (namePairs >= 4 && commaCount >= 3 && !CONTENT_VERB_RE.test(text)) return true
  return false
}

function looksBibliographyEntryContext(value: string): boolean {
  const text = cleanCitationDisplayText(value)
    .replace(/^\s*(?:\[\s*\d{1,4}\s*\]|\d{1,4}\s*[.)])\s*/, '')
  if (text.length < 30) return false
  if (!/\b(?:18|19|20)\d{2}\b/.test(text)) return false
  const startsLikeAuthors = /^(?:[A-Z][A-Za-z'`-]+,\s*(?:[A-Z]\.?\s*){1,4}|[A-Z][a-zA-Z'`-]+\s+[A-Z](?:\.|\b))/.test(text)
  const venueLike = /\b(?:IEEE|ACM|Springer|Elsevier|Nature|Science|Nat\.?|Opt\.?|Phys\.?|Journal|Proceedings|Trans\.?|Conf\.?|CVPR|ICCV|ICML|NeurIPS|arXiv)\b/i.test(text)
  const volumePages = /\b\d{1,4}\s*,\s*\d{1,6}(?:[-–]\d{1,6})?\.?$/.test(text)
  return startsLikeAuthors && venueLike && (volumePages || (text.match(/,/g) || []).length >= 3)
}

export function looksLowValueCitationContext(value: string): boolean {
  const text = cleanCitationDisplayText(value)
  if (!text) return true
  if (looksAuthorListContext(text) || looksBibliographyEntryContext(text)) return true
  const tokens = looseTokens(text)
  if (tokens.length < 5) return true
  const firstChunk = text.slice(0, 320)
  if (looksAuthorMetadataPrefix(firstChunk) && !CONTENT_VERB_RE.test(firstChunk)) return true
  const markerCount = (text.match(BRACKET_REFERENCE_MARKER_RE) || []).length
  if (markerCount >= 4 && markerCount >= Math.max(2, Math.floor(tokens.length / 8)) && !CONTENT_VERB_RE.test(text)) return true
  return false
}

const CONTENT_SENTENCE_START_RE = /\b(?:single[-\s]?pixel imaging\s+(?:is|can|uses?|technology|systems?)|deep learning\s+(?:models?|methods?|can|is|has|enables?)|snapshot compressive imaging\s+(?:is|can|uses?|recovers?)|compressive imaging\s+(?:is|can|uses?|recovers?)|neural radiance\s+(?:field|fields|representation)|a\s+DMD\s+can|this paper|this work|this study|in this (?:paper|work|study)|we\s+|however,?|recent(?:ly)?|the proposed|our\s+)\b/i
const FRAGMENT_LEAD_OK_RE = /^(?:a|an|the|this|these|those|most|many|some|several|existing|previous|prior|traditional|we|our|in|on|for|by|with|when|where|while|because|however|therefore|thus|as|if|to)\b/i

function splitEvidenceSentences(value: string): string[] {
  return String(value || '')
    .trim()
    .split(/(?<=[。！？!?.])\s+/)
    .map((item) => item.trim())
    .filter(Boolean)
}

function looksFragmentaryEvidenceSentence(value: string): boolean {
  const text = String(value || '').trim()
  if (!text) return true
  if (/^[a-z]{2,}\b/.test(text) && !FRAGMENT_LEAD_OK_RE.test(text)) return true
  if (/^(?:and|or|of|that|which|from|into|onto|within|without|using|used|measured|allowing)\b/i.test(text)) return true
  if (text.length > 80 && /\b(?:and|or|of|to|with|by|from|into|onto)$/i.test(text)) return true
  if (text.length > 120 && !/[。！？!?.]$/.test(text)) return true
  return false
}

function looksCaptionHeadingSentence(value: string): boolean {
  const text = String(value || '').trim()
  if (/^(?:fig(?:ure)?|table)\s*\d+[.:]?\s*$/i.test(text)) return true
  const tokens = looseTokens(text)
  if (/^[a-z]\s*,\s*/i.test(text)) return true
  return tokens.length <= 5 && /\b(?:configuration|configurations|overview|pipeline|results?|figure)\b/i.test(text)
}

function usableEvidenceSentence(value: string): boolean {
  const text = String(value || '').trim()
  if (looksLowValueCitationContext(text)) return false
  if (looksFragmentaryEvidenceSentence(text) || looksCaptionHeadingSentence(text)) return false
  return looseTokens(text).length >= 5
}

function evidenceSentenceQuality(value: string, detail: Pick<CiteDetail, 'answerClaim' | 'cardClaim' | 'headingPath' | 'title'>): number {
  const text = String(value || '').trim()
  if (!text) return -10
  const tokens = looseTokens(text)
  let score = 0
  if (looksFragmentaryEvidenceSentence(text)) score -= 5
  if (looksCaptionHeadingSentence(text)) score -= 2
  if (looksLowValueCitationContext(text)) score -= 6
  if (tokens.length >= 8 && tokens.length <= 90) score += 2
  else if (tokens.length < 5) score -= 2
  if (looksAuthorMetadataPrefix(text.slice(0, 180))) score -= 3
  if (/\b(?:is|are|can|uses?|proposes?|shows?|demonstrates?|improves?|captures?|reconstructs?)\b/i.test(text)) score += 1
  const contextTokens = new Set(looseTokens(`${detail.answerClaim || ''} ${detail.cardClaim || ''} ${detail.headingPath || ''} ${detail.title || ''}`))
  if (contextTokens.size > 0) {
    let overlap = 0
    for (const token of new Set(tokens)) {
      if (contextTokens.has(token)) overlap += 1
    }
    score += Math.min(2, overlap * 0.3)
  }
  if (/\b(?:single[-\s]?pixel|imaging|deep learning|compressive|neural|reconstruction|sampling|dmd)\b/i.test(text)) score += 1
  return score
}

function joinEvidenceWindow(
  sentences: string[],
  centerIndex: number,
  detail: Pick<CiteDetail, 'answerClaim' | 'cardClaim' | 'headingPath' | 'title'>,
  maxLen = 460,
): string {
  if (!sentences.length || !usableEvidenceSentence(sentences[centerIndex] || '')) return ''
  const chosen: number[] = [centerIndex]
  const centerScore = evidenceSentenceQuality(sentences[centerIndex], detail)

  const previousIndex = centerIndex - 1
  if (previousIndex >= 0 && usableEvidenceSentence(sentences[previousIndex])) {
    const previousScore = evidenceSentenceQuality(sentences[previousIndex], detail)
    if (previousScore >= 1 || centerScore < 2.5) chosen.unshift(previousIndex)
  }

  for (let nextIndex = centerIndex + 1; nextIndex < Math.min(sentences.length, centerIndex + 3); nextIndex += 1) {
    if (chosen.length >= 3) break
    if (!usableEvidenceSentence(sentences[nextIndex])) continue
    const nextScore = evidenceSentenceQuality(sentences[nextIndex], detail)
    if (nextScore < 0.5 && chosen.length > 1) continue
    chosen.push(nextIndex)
  }

  const output: string[] = []
  for (const index of Array.from(new Set(chosen)).sort((a, b) => a - b)) {
    const candidate = [...output, sentences[index]].join(' ').trim()
    if (output.length > 0 && candidate.length > maxLen) continue
    output.push(sentences[index])
  }
  return output.join(' ').trim()
}

function pickReadableEvidenceText(value: string, detail: Pick<CiteDetail, 'answerClaim' | 'cardClaim' | 'headingPath' | 'title'>): string {
  if (looksLowValueCitationContext(value)) return ''
  const sentences = splitEvidenceSentences(value)
  while (sentences.length > 0 && !usableEvidenceSentence(sentences[0])) {
    sentences.shift()
  }
  if (sentences.length <= 0) return ''
  const usable = sentences
    .slice(0, 8)
    .map((sentence, index) => ({ sentence, index }))
    .filter((item) => usableEvidenceSentence(item.sentence))
  if (!usable.length) return sentences[0]
  const first = usable[0]
  const scored = usable.map((item) => ({
    index: item.index,
    score: evidenceSentenceQuality(item.sentence, detail),
  }))
  scored.sort((a, b) => (b.score - a.score) || (a.index - b.index))
  const best = scored[0]
  const firstScore = evidenceSentenceQuality(first.sentence, detail)
  const centerIndex = best.index > first.index && best.score >= firstScore + 1 ? best.index : first.index
  return joinEvidenceWindow(sentences, centerIndex, detail) || sentences[0]
}

function stripEvidenceMetadataPrefix(
  value: string,
  detail: Pick<CiteDetail, 'sourceName' | 'title' | 'cardTitle' | 'answerClaim' | 'cardClaim' | 'headingPath'>,
): string {
  let text = cleanCitationDisplayText(value)
  if (!text) return ''
  for (const candidate of [
    sourceTitleCandidate(detail.sourceName),
    sourceTitleCandidate(detail.title),
    sourceTitleCandidate(detail.cardTitle),
  ]) {
    if (candidate.length < 18) continue
    const stripped = stripTokenPrefix(text, candidate)
    if (stripped !== text) {
      text = stripped
      break
    }
  }

  const match = text.match(CONTENT_SENTENCE_START_RE)
  if (match?.index && match.index > 0 && match.index <= 320) {
    const prefix = text.slice(0, match.index)
    if (looksAuthorMetadataPrefix(prefix)) {
      text = text.slice(match.index).replace(/^[\s,.;:，。；：-]+/, '')
    }
  }
  return pickReadableEvidenceText(text.replace(/\s+/g, ' ').trim(), detail)
}

function hasCjkText(value: string): boolean {
  return /[\u4e00-\u9fff]/.test(String(value || ''))
}

function trimTakeaway(value: string, maxLen = 110): string {
  let text = cleanCitationDisplayText(value)
    .replace(/^\s*(?:这条证据说明|证据说明|它说明|说明)[:：]\s*/, '')
    .trim()
  text = text.replace(/[。；;]\s*$/g, '')
  if (text.length > maxLen) text = `${text.slice(0, Math.max(0, maxLen - 1)).replace(/[，,；;:：]\s*$/g, '')}...`
  if (text && hasCjkText(text) && !/[。！？?]$|\.\.\.$/.test(text)) text = `${text}。`
  return text
}

function looksLowValueTakeaway(value: string): boolean {
  const text = cleanCitationDisplayText(value)
  if (!text) return true
  if (/^[A-Za-z][A-Za-z\s-]{2,48}\s+\d{1,3}$/.test(text)) return true
  if (/(?:这条证据|该证据|this evidence|the evidence).{0,12}(?:支持|支撑|supports?)/i.test(text)) return true
  const tokens = looseTokens(text)
  if (hasCjkText(text)) return text.length < 12 && !/[：:，,。；;]/.test(text)
  return tokens.length <= 6
}

function takeawayFromEnglishEvidence(evidence: string): string {
  const text = String(evidence || '')
  const low = text.toLowerCase()
  if (low.includes('dmd') && (low.includes('spatially filter') || low.includes('single-pixel camera configuration'))) {
    return 'DMD 可以作为单像素相机中的空间调制器，通过选择性重定向光束来完成采样和成像配置。'
  }
  if (low.includes('single-pixel imaging technology can capture images at wavelengths outside')) {
    return '单像素成像可以覆盖传统焦平面阵列探测器难以触达的波段，但实用性仍受图像质量和计算时间限制。'
  }
  if (low.includes('structured detection') && low.includes('optical sectioning')) {
    return '结构化检测用于在激光扫描显微中同时改善层切、分辨率和信噪比。'
  }
  if (low.includes('deep learning') && low.includes('single-pixel') && /\b(?:quality|speed|reconstruction)\b/i.test(text)) {
    return '深度学习方法主要用于提升单像素成像的重建质量、速度或采样效率。'
  }
  if (low.includes('snapshot compressive imaging') && /\b(?:recover|reconstruct)\b/i.test(text)) {
    return '快照压缩成像通过一次压缩观测恢复场景信息，是该回答所说成像任务的直接背景。'
  }
  return ''
}

function deriveSystemATakeaway(
  detail: Pick<CiteDetail, 'answerClaim' | 'cardClaim' | 'cardEvidence' | 'evidenceQuote' | 'summaryLine' | 'headingPath'>,
): string {
  const evidence = detail.cardEvidence || detail.evidenceQuote || detail.summaryLine || ''
  const evidenceTakeaway = trimTakeaway(takeawayFromEnglishEvidence(evidence))
  const claim = trimTakeaway(detail.cardClaim || detail.answerClaim || '')
  if (
    evidenceTakeaway
    && !looksLowValueTakeaway(evidenceTakeaway)
    && !substantiallySameText(evidenceTakeaway, claim)
  ) {
    return evidenceTakeaway
  }

  const heading = trimTakeaway(detail.headingPath || '', 70)
  if (heading && hasCjkText(heading) && evidence) {
    const candidate = `这条证据对应“${heading.replace(/[。！？?]$/g, '')}”这一部分的关键表述。`
    if (!looksLowValueTakeaway(candidate)) return candidate
  }
  if (claim && hasCjkText(claim) && !looksLowValueTakeaway(claim)) return claim
  return ''
}

function looksGenericSystemBTakeaway(value: string): boolean {
  const text = cleanCitationDisplayText(value).toLowerCase()
  if (!text) return true
  const genericPatterns = [
    /这条链接把回答中的说法追溯到/,
    /这条参考是当前论文给出的上游来源/,
    /这篇上游文献条目/,
    /the user is asking about the evidence/,
    /upstream paper to open next/,
    /cited prior work or background source/,
    /trace the upstream origin/,
    /this reference is the cited prior work/,
  ]
  if (genericPatterns.some((pattern) => pattern.test(text))) return true
  return looseTokens(text).length <= 5
}

function explicitSystemBTakeaway(detail: Pick<CiteDetail, 'upstreamWorkRole' | 'userQuestionRelation' | 'supportRelation' | 'whyLine'>): string {
  for (const raw of [detail.upstreamWorkRole, detail.userQuestionRelation, detail.supportRelation, detail.whyLine]) {
    let text = trimTakeaway(raw || '', 118)
    if (!text || !hasCjkText(text) || looksGenericSystemBTakeaway(text)) continue
    text = text
      .replace(/^用户问[“"].+?[”"，,；;]\s*/, '')
      .replace(/^这条参考(?:正好)?说明/, '这篇上游文献说明')
      .replace(/^它说明/, '这篇上游文献说明')
    return trimTakeaway(text, 118)
  }
  return ''
}

function deriveSystemBTakeaway(
  detail: Pick<CiteDetail, 'title' | 'answerClaim' | 'cardClaim' | 'cardEvidence' | 'citationContext' | 'evidenceQuote' | 'summaryLine' | 'upstreamWorkRole' | 'userQuestionRelation' | 'supportRelation' | 'whyLine'>,
): string {
  const explicit = explicitSystemBTakeaway(detail)
  if (explicit) return explicit

  const combined = [
    detail.title,
    detail.answerClaim,
    detail.cardClaim,
    detail.cardEvidence,
    detail.citationContext,
    detail.evidenceQuote,
    detail.summaryLine,
    detail.upstreamWorkRole,
    detail.userQuestionRelation,
    detail.supportRelation,
    detail.whyLine,
  ].join(' ').toLowerCase()
  if (combined.includes('admm-net') || /\b(?:unfold|unrolled)\b/.test(combined)) {
    return '这篇上游文献提供把迭代优化思想展开成可训练网络的前人线索。'
  }
  if (combined.includes('admm') || combined.includes('alternating direction method')) {
    return '这篇上游文献提供 ADMM 优化框架背景，用来判断当前论文是在借鉴既有方法。'
  }
  if (combined.includes('single-shot compressive spectral imaging')) {
    return '这篇上游文献提供单次压缩光谱成像的前人背景，是回答中相关概念的来源线索。'
  }
  if (/\b(?:baseline|compare|compared|comparison|against)\b/.test(combined)) {
    return '这篇上游文献在当前论文中主要作为对比基线或相关方法参照。'
  }
  if (/\b(?:dataset|benchmark|evaluation|experiment)\b/.test(combined)) {
    return '这篇上游文献提供实验数据、评测场景或 benchmark 线索。'
  }
  if (/\b(?:architecture|network|model|module)\b/.test(combined)) {
    return '这篇上游文献提供模型结构或方法设计上的前人参考。'
  }
  if (/\b(?:background|prior work|related work|origin|source)\b/.test(combined)) {
    return '这篇上游文献提供当前说法的相关工作背景和来源线索。'
  }
  return ''
}

function asNumber(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

export function normalizeShelfTags(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  const out: string[] = []
  const seen = new Set<string>()
  for (const raw of value) {
    const txt = String(raw || '').trim().replace(/\s+/g, ' ')
    if (!txt) continue
    const key = txt.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    out.push(txt.slice(0, 24))
    if (out.length >= 8) break
  }
  return out
}

export function normalizeShelfNote(value: unknown): string {
  const text = String(value || '')
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
  if (!text) return ''
  return text.slice(0, 1200)
}

function normalizeDoiLike(value: unknown): string {
  const raw = String(value || '').trim().toLowerCase()
  if (!raw) return ''
  return raw
    .replace(/^https?:\/\/(?:dx\.)?doi\.org\//i, '')
    .replace(/^[\s"'`([{<]+|[\s"'`)\]}>.,;:]+$/g, '')
    .trim()
}

function extractDoiLike(value: unknown): string {
  const text = String(value || '').trim()
  const direct = normalizeDoiLike(text)
  if (/^10\.\d{4,9}\//i.test(direct)) return direct
  const match = text.match(/\b10\.\d{4,9}\/[-._;()/:A-Z0-9]+/i)
  return match ? normalizeDoiLike(match[0]) : ''
}

function doiUrlFrom(value: unknown): string {
  const doi = normalizeDoiLike(value)
  return doi ? `https://doi.org/${doi}` : ''
}

function metadataQualityOk(value: unknown): boolean {
  if (!value || typeof value !== 'object') return false
  const rec = value as Record<string, unknown>
  const status = String(rec.status || '').trim().toLowerCase()
  return rec.ok === true || status === 'ready'
}

function metadataRepairMetaTrusted(meta: Record<string, unknown>): boolean {
  const qualityOk = metadataQualityOk(meta.metadata_quality || meta.metadataQuality)
  const repairStatus = String(meta.metadata_repair_status || meta.metadataRepairStatus || '').trim().toLowerCase()
  const changedFields = Array.isArray(meta.metadata_changed_fields)
    ? meta.metadata_changed_fields
    : meta.metadataChangedFields
  const changedCount = Array.isArray(changedFields) ? changedFields.length : 0
  return qualityOk && (changedCount > 0 || ['ready', 'repaired'].includes(repairStatus))
}

export function shelfItemMetadataQualityReady(item: CiteShelfItem): boolean {
  const quality = item.metadataQuality
  if (!quality || typeof quality !== 'object') return false
  return metadataQualityOk(quality)
}

export function shelfItemMetadataQualityNeedsRepair(item: CiteShelfItem): boolean {
  const quality = item.metadataQuality
  if (!quality || typeof quality !== 'object') return false
  if (shelfItemMetadataQualityReady(item)) return false
  const rec = quality as Record<string, unknown>
  const issues = Array.isArray(rec.issues) ? rec.issues : []
  return Boolean(rec.repairable || rec.retryable || issues.length > 0)
}

export function shelfItemHasConflictingVenueSignals(item: CiteShelfItem): boolean {
  const hasJournalSignal = Boolean(String(item.journalIf || item.journalQuartile || item.journalIfSource || '').trim())
  const hasConfSignal = Boolean(
    String(item.conferenceTier || item.conferenceCcf || item.conferenceName || item.conferenceAcronym || '').trim(),
  )
  const venueKind = String(item.venueKind || '').trim().toLowerCase()
  return (
    (venueKind === 'conference' && hasJournalSignal)
    || (venueKind === 'journal' && hasConfSignal)
    || (hasJournalSignal && hasConfSignal)
  )
}

export function shelfItemNeedsMetadataRepair(item: CiteShelfItem, display = citationDisplay(item)): boolean {
  if (shelfItemMetadataQualityReady(item)) return false
  if (shelfItemMetadataQualityNeedsRepair(item)) return true
  const rawTitle = String(item.title || '').trim()
  const visibleTitle = String(display.main || rawTitle || item.main || '').trim()
  const hasDoi = Boolean(normalizeDoiLike(item.doi || item.doiUrl))
  const hasAuthors = Boolean(String(item.authors || '').trim())
  const hasVenue = Boolean(String(item.venue || '').trim())
  const unresolved = !item.bibliometricsChecked
  const rawTitleNeedsRepair = isLikelyWeakCitationTitle(rawTitle)
  const visibleTitleNeedsRepair = isLikelyWeakCitationTitle(visibleTitle)
  return (
    shelfItemHasConflictingVenueSignals(item)
    || (hasDoi && (rawTitleNeedsRepair || unresolved))
    || (!hasDoi && unresolved && (visibleTitleNeedsRepair || !hasAuthors || !hasVenue))
  )
}

export function shelfItemRepairFingerprint(item: CiteShelfItem, display = citationDisplay(item)): string {
  return [
    normalizeDoiLike(item.doi || item.doiUrl),
    String(item.title || '').trim(),
    String(display.main || '').trim(),
    String(item.authors || '').trim(),
    String(item.venue || '').trim(),
    String(item.year || '').trim(),
    String(item.venueKind || '').trim(),
    String(item.citationCount || 0),
    item.bibliometricsChecked ? '1' : '0',
  ].join('|')
}

function pickText(rec: Record<string, unknown>, ...keys: string[]): string {
  for (const key of keys) {
    const value = asText(rec[key])
    if (value) return value
  }
  return ''
}

function pickNumber(rec: Record<string, unknown>, ...keys: string[]): number {
  for (const key of keys) {
    const value = asNumber(rec[key])
    if (value) return value
  }
  return 0
}

function pickStringArray(rec: Record<string, unknown>, ...keys: string[]): string[] {
  for (const key of keys) {
    const value = rec[key]
    if (!Array.isArray(value)) continue
    const out = value
      .map((item) => String(item || '').trim())
      .filter(Boolean)
    if (out.length > 0) return out
  }
  return []
}

function pickNumberArray(rec: Record<string, unknown>, ...keys: string[]): number[] {
  for (const key of keys) {
    const value = rec[key]
    if (!Array.isArray(value)) continue
    const out: number[] = []
    for (const item of value) {
      const num = typeof item === 'number' ? item : Number.parseInt(String(item || ''), 10)
      if (Number.isFinite(num) && num > 0) out.push(num)
    }
    const deduped = Array.from(new Set(out)).sort((a, b) => a - b)
    if (deduped.length > 0) return deduped
  }
  return []
}

function pickRecord(rec: Record<string, unknown>, ...keys: string[]): Record<string, unknown> | null {
  for (const key of keys) {
    const value = rec[key]
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      return { ...(value as Record<string, unknown>) }
    }
  }
  return null
}

function normalizeCitationCardView(value: unknown): CitationCardView | null {
  if (!value || typeof value !== 'object') return null
  const rec = value as Record<string, unknown>
  const headerRec = rec.header && typeof rec.header === 'object'
    ? rec.header as Record<string, unknown>
    : {}
  const qualityRec = rec.quality && typeof rec.quality === 'object'
    ? rec.quality as Record<string, unknown>
    : {}
  const sectionsRaw = Array.isArray(rec.sections) ? rec.sections : []
  const sections: CitationCardViewSection[] = []
  const seen = new Set<string>()
  for (const item of sectionsRaw) {
    if (!item || typeof item !== 'object') continue
    const section = item as Record<string, unknown>
    const id = asText(section.id)
    const text = cleanCitationDisplayText(asText(section.text))
    if (!id || !text || seen.has(id)) continue
    seen.add(id)
    sections.push({
      id,
      label: cleanCitationDisplayText(asText(section.label)),
      text,
      kind: asText(section.kind),
      hint: cleanCitationDisplayText(asText(section.hint)),
      tone: asText(section.tone),
    })
  }
  if (sections.length <= 0 && !asText(headerRec.title)) return null
  return {
    version: pickNumber(rec, 'version'),
    route: asText(rec.route),
    kind: asText(rec.kind),
    header: {
      kicker: cleanCitationDisplayText(asText(headerRec.kicker)),
      title: cleanCitationDisplayText(asText(headerRec.title)),
      subtitle: cleanCitationDisplayText(asText(headerRec.subtitle)),
    },
    sections,
    summary: cleanCitationDisplayText(asText(rec.summary)),
    quality: {
      label: cleanCitationDisplayText(asText(qualityRec.label)),
      score: pickNumber(qualityRec, 'score'),
      flags: pickStringArray(qualityRec, 'flags'),
      warning: cleanCitationDisplayText(asText(qualityRec.warning)),
    },
  }
}

function stripLeadCitationLabel(value: string): string {
  return String(value || '')
    .replace(/^\s*(?:\[\s*\d{1,4}\s*\]\s*){1,3}/, '')
    .replace(/^\s*\d{1,4}\s*[.)]\s*/, '')
    .trim()
}

function looksCitationLine(text: string): boolean {
  const s = stripLeadCitationLabel(String(text || '').replace(/\*+/g, '').replace(/\s+/g, ' ').trim())
  if (s.length < 24) return false
  const hasYear = /\b(?:19|20)\d{2}\b/.test(s)
  const hasVolumePagesTail = /,\s*\d{1,4}\s*,\s*\d{1,6}\.?$/.test(s)
  const hasVenueToken = /\b(?:Nat\.?|IEEE|ACM|Opt\.?|Phys\.?|Commun\.?|Journal|Proceedings|CVPR|ICCV|ICML|NeurIPS)\b/i.test(s)
  const startsLikeAuthors = /^(?:[A-Z][A-Za-z'`-]+,\s*(?:[A-Z]\.\s*){1,3})(?:,\s*[A-Z][A-Za-z'`-]+,\s*(?:[A-Z]\.\s*){1,3})*/.test(s)
  if (hasYear && hasVolumePagesTail) return true
  if (startsLikeAuthors && hasYear && hasVenueToken) return true
  return false
}

export function isLikelyWeakCitationTitle(value: string): boolean {
  const s = stripLeadCitationLabel(String(value || '').replace(/\*+/g, '').replace(/\s+/g, ' ').trim())
  if (!s) return true
  if (looksCitationLine(s)) return true
  if (/^(?:doi[:\s]|https?:\/\/|arxiv:)/i.test(s)) return true
  if (/^[A-Z][A-Za-z'`-]+,\s*(?:[A-Z]\.?)(?:\s*[A-Z]\.?)?$/i.test(s)) return true
  if (/^[A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`-]+){0,2},\s*(?:[A-Z]\.?\s*){1,3}$/i.test(s)) return true
  const tokens = s.match(/[A-Za-z0-9\u4e00-\u9fff]+/g) || []
  if (/\bet\s+al\.?$/i.test(s) && tokens.length <= 4) return true
  if (tokens.length <= 0) return true
  if (tokens.length === 1) {
    const token = tokens[0]
    if (token.length <= 2) return true
    if (/^(?:vol|no|pp|doi|arxiv|misc)$/i.test(token)) return true
  }
  return false
}

function isGenericSystemBCardTitle(value: string): boolean {
  const s = cleanCitationDisplayText(value).replace(/\s+/g, ' ').trim().toLowerCase()
  return (
    !s
    || s === '\u4e0a\u6e38\u53c2\u8003\u6587\u732e'
    || s === '\u4e0a\u6e38\u5f15\u7528'
    || s === 'upstream reference'
    || s === 'upstream citation'
  )
}

function isWeakField(key: string, value: string): boolean {
  const s = String(value || '').trim()
  if (!s) return true
  if (key === 'title') return isLikelyWeakCitationTitle(s)
  if (key === 'authors') return s.length <= 3 || (s.match(/[A-Za-z\u4e00-\u9fff]+/g)?.length || 0) <= 1
  if (key === 'venue') return s.length <= 1
  return false
}

export function normalizeCiteDetail(value: unknown): CiteDetail | null {
  if (!value || typeof value !== 'object') return null
  const rec = value as Record<string, unknown>
  const anchor = pickText(rec, 'anchor')
  if (!anchor) return null
  const libraryMatch = pickRecord(rec, 'library_match', 'libraryMatch') || {}
  const detail: CiteDetail = {
    num: pickNumber(rec, 'num'),
    displayNum: pickNumber(rec, 'display_num', 'displayNum', 'visible_num', 'visibleNum'),
    displayNums: pickNumberArray(rec, 'display_nums', 'displayNums', 'visible_nums', 'visibleNums'),
    anchor,
    sourceName: pickText(rec, 'source_name', 'sourceName'),
    sourcePath: pickText(rec, 'source_path', 'sourcePath'),
    traceConvId: pickText(rec, 'trace_conv_id', 'traceConvId'),
    traceAssistantMsgId: pickNumber(rec, 'trace_assistant_msg_id', 'traceAssistantMsgId'),
    traceAssistantOrder: pickNumber(rec, 'trace_assistant_order', 'traceAssistantOrder'),
    traceUserMsgId: pickNumber(rec, 'trace_user_msg_id', 'traceUserMsgId'),
    raw: pickText(rec, 'raw'),
    citeFmt: pickText(rec, 'cite_fmt', 'citeFmt'),
    isInpaper: rec.is_inpaper === true || rec.isInpaper === true,
    title: pickText(rec, 'title'),
    authors: pickText(rec, 'authors'),
    venue: pickText(rec, 'venue'),
    year: pickText(rec, 'year'),
    volume: pickText(rec, 'volume'),
    issue: pickText(rec, 'issue'),
    pages: pickText(rec, 'pages'),
    doi: pickText(rec, 'doi'),
    doiUrl: pickText(rec, 'doi_url', 'doiUrl'),
    linkedNums: pickNumberArray(rec, 'linked_nums', 'linkedNums'),
    evidenceFingerprint: pickText(rec, 'evidence_fingerprint', 'evidenceFingerprint'),
    renderLocale: pickText(rec, 'render_locale', 'renderLocale', 'locale'),
    citationRoute: pickText(rec, 'citation_route', 'citationRoute'),
    routingReason: pickText(rec, 'routing_reason', 'routingReason'),
    routingConfidence: pickNumber(rec, 'routing_confidence', 'routingConfidence'),
    citationCount: pickNumber(rec, 'citation_count', 'citationCount'),
    citationSource: pickText(rec, 'citation_source', 'citationSource'),
    venueKind: pickText(rec, 'venue_kind', 'venueKind'),
    venueVerifiedBy: pickText(rec, 'venue_verified_by', 'venueVerifiedBy'),
    openalexVenue: pickText(rec, 'openalex_venue', 'openalexVenue'),
    journalIf: pickText(rec, 'journal_if', 'journalIf'),
    journalQuartile: pickText(rec, 'journal_quartile', 'journalQuartile'),
    journalIfSource: pickText(rec, 'journal_if_source', 'journalIfSource'),
    conferenceTier: pickText(rec, 'conference_tier', 'conferenceTier'),
    conferenceRankSource: pickText(rec, 'conference_rank_source', 'conferenceRankSource'),
    conferenceCcf: pickText(rec, 'conference_ccf', 'conferenceCcf'),
    conferenceCcfSource: pickText(rec, 'conference_ccf_source', 'conferenceCcfSource'),
    conferenceName: pickText(rec, 'conference_name', 'conferenceName'),
    conferenceAcronym: pickText(rec, 'conference_acronym', 'conferenceAcronym'),
    bibliometricsChecked: Boolean(rec.bibliometrics_checked ?? rec.bibliometricsChecked),
    libraryMatchStatus: pickText(rec, 'library_match_status', 'libraryMatchStatus') || pickText(libraryMatch, 'status'),
    libraryMatchConfidence: pickNumber(rec, 'library_match_confidence', 'libraryMatchConfidence') || pickNumber(libraryMatch, 'confidence'),
    libraryMatchMethod: pickText(rec, 'library_match_method', 'libraryMatchMethod') || pickText(libraryMatch, 'method'),
    libraryMatchReason: pickText(rec, 'library_match_reason', 'libraryMatchReason') || pickText(libraryMatch, 'reason'),
    libraryMatchPath: pickText(rec, 'library_match_path', 'libraryMatchPath') || pickText(libraryMatch, 'path'),
    libraryMatchSha1: pickText(rec, 'library_match_sha1', 'libraryMatchSha1') || pickText(libraryMatch, 'sha1'),
    libraryMatchTitle: pickText(rec, 'library_match_title', 'libraryMatchTitle') || pickText(libraryMatch, 'title'),
    libraryMatchDoi: pickText(rec, 'library_match_doi', 'libraryMatchDoi') || pickText(libraryMatch, 'doi'),
    libraryMatchYear: pickText(rec, 'library_match_year', 'libraryMatchYear') || pickText(libraryMatch, 'year'),
    metadataQuality: pickRecord(rec, 'metadata_quality', 'metadataQuality'),
    metadataRepairStatus: pickText(rec, 'metadata_repair_status', 'metadataRepairStatus'),
    metadataRepairSources: pickStringArray(rec, 'metadata_repair_sources', 'metadataRepairSources'),
    metadataChangedFields: pickStringArray(rec, 'metadata_changed_fields', 'metadataChangedFields'),
    externalMetadataStatus: pickText(rec, 'external_metadata_status', 'externalMetadataStatus'),
    externalMetadataReason: pickText(rec, 'external_metadata_reason', 'externalMetadataReason'),
    externalMatchMethod: pickText(rec, 'external_match_method', 'externalMatchMethod'),
    externalMatchScore: pickNumber(rec, 'external_match_score', 'externalMatchScore'),
    externalTitleSimilarity: pickNumber(rec, 'external_title_similarity', 'externalTitleSimilarity'),
    externalTitle: pickText(rec, 'external_title', 'externalTitle'),
    externalAuthors: pickText(rec, 'external_authors', 'externalAuthors'),
    externalVenue: pickText(rec, 'external_venue', 'externalVenue'),
    externalYear: pickText(rec, 'external_year', 'externalYear'),
    externalDoi: pickText(rec, 'external_doi', 'externalDoi'),
    externalDoiUrl: pickText(rec, 'external_doi_url', 'externalDoiUrl'),
    summaryLine: pickText(rec, 'summary_line', 'summaryLine'),
    summarySource: pickText(rec, 'summary_source', 'summarySource'),
    summaryProvider: pickText(rec, 'summary_provider', 'summaryProvider'),
    summaryQuality: pickRecord(rec, 'summary_quality', 'summaryQuality'),
    shelfItemKind: pickText(rec, 'shelf_item_kind', 'shelfItemKind'),
    shelfOrigin: pickText(rec, 'shelf_origin', 'shelfOrigin'),
    shelfExcerpt: pickText(rec, 'shelf_excerpt', 'shelfExcerpt'),
    shelfExcerptLabel: pickText(rec, 'shelf_excerpt_label', 'shelfExcerptLabel'),
    answerClaim: pickText(rec, 'answer_claim', 'answerClaim'),
    headingPath: pickText(rec, 'heading_path', 'headingPath'),
    evidenceQuote: pickText(rec, 'evidence_quote', 'evidenceQuote'),
    evidenceSource: pickText(rec, 'evidence_source', 'evidenceSource'),
    citationContext: pickText(rec, 'citation_context', 'citationContext'),
    citationContextSource: pickText(rec, 'citation_context_source', 'citationContextSource'),
    upstreamWorkRole: pickText(rec, 'upstream_work_role', 'upstreamWorkRole'),
    userQuestionRelation: pickText(rec, 'user_question_relation', 'userQuestionRelation'),
    locationLabel: pickText(rec, 'location_label', 'locationLabel'),
    supportRelation: pickText(rec, 'support_relation', 'supportRelation'),
    whyLine: pickText(rec, 'why_line', 'whyLine'),
    blockId: pickText(rec, 'block_id', 'blockId'),
    anchorId: pickText(rec, 'anchor_id', 'anchorId'),
    anchorKind: pickText(rec, 'anchor_kind', 'anchorKind'),
    pageStart: pickNumber(rec, 'page_start', 'pageStart'),
    pageEnd: pickNumber(rec, 'page_end', 'pageEnd'),
    score: pickNumber(rec, 'score'),
    bindingStatus: pickText(rec, 'binding_status', 'bindingStatus'),
    bindingConfidence: pickNumber(rec, 'binding_confidence', 'bindingConfidence'),
    bindingReason: pickText(rec, 'binding_reason', 'bindingReason'),
    bindingOverlapTerms: pickStringArray(rec, 'binding_overlap_terms', 'bindingOverlapTerms'),
    cardKind: pickText(rec, 'card_kind', 'cardKind'),
    cardTitle: pickText(rec, 'card_title', 'cardTitle'),
    cardSubtitle: pickText(rec, 'card_subtitle', 'cardSubtitle'),
    cardTakeawayLabel: pickText(rec, 'card_takeaway_label', 'cardTakeawayLabel'),
    cardTakeaway: pickText(rec, 'card_takeaway', 'cardTakeaway'),
    cardClaimLabel: pickText(rec, 'card_claim_label', 'cardClaimLabel'),
    cardClaim: pickText(rec, 'card_claim', 'cardClaim'),
    cardLocatorLabel: pickText(rec, 'card_locator_label', 'cardLocatorLabel'),
    cardLocator: pickText(rec, 'card_locator', 'cardLocator'),
    cardEvidenceLabel: pickText(rec, 'card_evidence_label', 'cardEvidenceLabel'),
    cardEvidence: pickText(rec, 'card_evidence', 'cardEvidence'),
    cardContextSummary: pickText(rec, 'card_context_summary', 'cardContextSummary'),
    cardReferenceLabel: pickText(rec, 'card_reference_label', 'cardReferenceLabel'),
    cardReferenceEntry: pickText(rec, 'card_reference_entry', 'cardReferenceEntry'),
    cardSupportLabel: pickText(rec, 'card_support_label', 'cardSupportLabel'),
    cardSupportExplanation: pickText(rec, 'card_support_explanation', 'cardSupportExplanation'),
    cardQualityLabel: pickText(rec, 'card_quality_label', 'cardQualityLabel'),
    cardQualityScore: pickNumber(rec, 'card_quality_score', 'cardQualityScore'),
    cardQualityFlags: pickStringArray(rec, 'card_quality_flags', 'cardQualityFlags'),
    cardWarning: pickText(rec, 'card_warning', 'cardWarning'),
    cardFlow: pickStringArray(rec, 'card_flow', 'cardFlow'),
    cardDisplayContractVersion: pickNumber(rec, 'card_display_contract_version', 'cardDisplayContractVersion'),
    cardVisibleSections: pickStringArray(rec, 'card_visible_sections', 'cardVisibleSections'),
    cardView: normalizeCitationCardView(rec.card_view ?? rec.cardView),
    systemBTraceComplete: Boolean(rec.system_b_trace_complete ?? rec.systemBTraceComplete),
    systemBTraceScore: pickNumber(rec, 'system_b_trace_score', 'systemBTraceScore'),
    systemBTraceReason: pickText(rec, 'system_b_trace_reason', 'systemBTraceReason'),
    systemBTraceFlags: pickStringArray(rec, 'system_b_trace_flags', 'systemBTraceFlags'),
    systemBTraceSteps: pickStringArray(rec, 'system_b_trace_steps', 'systemBTraceSteps'),
    systemBTraceAnswer: pickText(rec, 'system_b_trace_answer', 'systemBTraceAnswer'),
    systemBTraceContext: pickText(rec, 'system_b_trace_context', 'systemBTraceContext'),
    systemBTraceReference: pickText(rec, 'system_b_trace_reference', 'systemBTraceReference'),
    systemBTraceLocator: pickText(rec, 'system_b_trace_locator', 'systemBTraceLocator'),
    systemBTraceSource: pickText(rec, 'system_b_trace_source', 'systemBTraceSource'),
    citationCardPolishStatus: pickText(rec, 'citation_card_polish_status', 'citationCardPolishStatus'),
    citationCardPolishSource: pickText(rec, 'citation_card_polish_source', 'citationCardPolishSource'),
    citationCardPolishChecked: Boolean(rec.citation_card_polish_checked ?? rec.citationCardPolishChecked),
    citationCardPolishKey: pickText(rec, 'citation_card_polish_key', 'citationCardPolishKey'),
    citationCardPolishRoute: pickText(rec, 'citation_card_polish_route', 'citationCardPolishRoute'),
    citationCardPolishFields: pickStringArray(rec, 'citation_card_polish_fields', 'citationCardPolishFields'),
    citationCardPolishRejected: pickStringArray(rec, 'citation_card_polish_rejected', 'citationCardPolishRejected'),
    citationCardPolishQualityScore: pickNumber(rec, 'citation_card_polish_quality_score', 'citationCardPolishQualityScore'),
  }
  for (const key of [
    'raw',
    'citeFmt',
    'title',
    'summaryLine',
    'shelfOrigin',
    'shelfExcerpt',
    'shelfExcerptLabel',
    'answerClaim',
    'headingPath',
    'evidenceQuote',
    'citationContext',
    'upstreamWorkRole',
    'userQuestionRelation',
    'locationLabel',
    'supportRelation',
    'whyLine',
    'bindingReason',
    'externalMetadataReason',
    'externalTitle',
    'externalAuthors',
    'externalVenue',
    'cardTitle',
    'cardSubtitle',
    'cardTakeawayLabel',
    'cardTakeaway',
    'cardClaim',
    'cardLocator',
    'cardEvidence',
    'cardContextSummary',
    'cardReferenceLabel',
    'cardReferenceEntry',
    'cardSupportExplanation',
    'cardWarning',
    'systemBTraceReason',
    'systemBTraceAnswer',
    'systemBTraceContext',
    'systemBTraceReference',
    'systemBTraceLocator',
    'systemBTraceSource',
    'citationCardPolishStatus',
    'citationCardPolishSource',
    'citationCardPolishKey',
    'citationCardPolishRoute',
    'libraryMatchReason',
    'libraryMatchPath',
    'libraryMatchTitle',
    'libraryMatchDoi',
  ] as const) {
    detail[key] = cleanCitationDisplayText(detail[key])
  }
  for (const key of [
    'summaryLine',
    'evidenceQuote',
    'citationContext',
    'cardEvidence',
    'systemBTraceContext',
  ] as const) {
    if (key === 'summaryLine' && isArticleSummaryTextSource(detail.summarySource)) {
      detail[key] = cleanCitationDisplayText(detail[key])
    } else {
      detail[key] = stripEvidenceMetadataPrefix(detail[key], detail)
    }
  }
  if (!detail.doiUrl && detail.doi) {
    detail.doiUrl = doiUrlFrom(detail.doi)
  }
  if (Number(detail.displayNum || 0) > 0 && (detail.displayNums || []).length <= 0) {
    detail.displayNums = [Number(detail.displayNum || 0)]
  }
  if (!detail.externalDoiUrl && detail.externalDoi) {
    detail.externalDoiUrl = doiUrlFrom(detail.externalDoi)
  }
  if (!detail.isInpaper) {
    detail.raw = stripEvidenceMetadataPrefix(detail.raw, detail)
    if (!detail.cardTakeaway || looksLowValueTakeaway(detail.cardTakeaway)) {
      detail.cardTakeaway = deriveSystemATakeaway(detail)
    }
    if (detail.cardTakeaway && !detail.cardTakeawayLabel) {
      detail.cardTakeawayLabel = '证据重点'
    }
  } else {
    if (!detail.cardTakeaway || looksGenericSystemBTakeaway(detail.cardTakeaway)) {
      detail.cardTakeaway = deriveSystemBTakeaway(detail)
    }
    if (detail.cardTakeaway && !detail.cardTakeawayLabel) {
      detail.cardTakeawayLabel = '上游作用'
    }
  }
  return detail
}

export function citationMain(detail: CiteDetail): string {
  if (detail.citeFmt) return stripLeadCitationLabel(detail.citeFmt)
  const parts = [detail.authors, detail.title, detail.venue, detail.year].filter(Boolean)
  if (parts.length > 0) return parts.join('. ')
  return stripLeadCitationLabel(detail.raw) || `[${detail.num || '?'}]`
}

function makeCardViewSection(
  id: string,
  label: string,
  text: string,
  kind: string,
  opts?: { hint?: string; tone?: string },
): CitationCardViewSection | null {
  const cleanText = cleanCitationDisplayText(text)
  if (!id || !cleanText) return null
  return {
    id,
    label: cleanCitationDisplayText(label),
    text: cleanText,
    kind,
    hint: cleanCitationDisplayText(opts?.hint || ''),
    tone: String(opts?.tone || '').trim(),
  }
}

function appendCardViewSection(sections: CitationCardViewSection[], section: CitationCardViewSection | null): void {
  if (!section) return
  if (sections.some((item) => item.id === section.id)) return
  const key = looseTokens(section.text).join(' ')
  if (key && sections.some((item) => {
    const existing = looseTokens(item.text).join(' ')
    return existing === key || (key.length > 24 && existing.includes(key)) || (existing.length > 24 && key.includes(existing))
  })) return
  sections.push(section)
}

export function citationCardView(detail: CiteDetail): CitationCardView {
  const stored = detail.cardView
  const isSystemB = Boolean(detail.isInpaper)
  const route = isSystemB ? 'system_b' : 'system_a'
  const storedMatchesRoute = Boolean(stored && (!stored.route || stored.route === route))
  const storedSection = (id: string): CitationCardViewSection | null => {
    if (!storedMatchesRoute) return null
    return stored?.sections?.find((item) => item.id === id) || null
  }
  const sectionLabel = (id: string, fieldValue: string, fallback: string): string => {
    return cleanCitationDisplayText(storedSection(id)?.label || fieldValue || fallback)
  }
  const sectionText = (id: string, fieldValue: string): string => {
    return cleanCitationDisplayText(storedSection(id)?.text || fieldValue || '')
  }
  const storedTitle = cleanCitationDisplayText((storedMatchesRoute ? stored?.header?.title : '') || '')
  const detailTitle = cleanCitationDisplayText(detail.title || '')
  const repairedSystemBTitle = (
    isSystemB
    && detailTitle
    && !isLikelyWeakCitationTitle(detailTitle)
    && (isGenericSystemBCardTitle(storedTitle) || isGenericSystemBCardTitle(detail.cardTitle))
  ) ? detailTitle : ''
  const title = cleanCitationDisplayText(
    repairedSystemBTitle
    || storedTitle
    || detail.cardTitle
    || (isSystemB ? detail.title : detail.sourceName)
    || detail.title
    || detail.sourcePath,
  )
  const subtitle = cleanCitationDisplayText((storedMatchesRoute ? stored?.header?.subtitle : '') || detail.cardSubtitle || '')
  const qualityFlags = detail.cardQualityFlags.length ? detail.cardQualityFlags : (storedMatchesRoute ? (stored?.quality?.flags || []) : [])
  const systemAHasReviewRisk = !isSystemB && Boolean(
    detail.cardWarning
    || qualityFlags.includes('candidate_binding')
    || qualityFlags.includes('binding_mismatch')
    || qualityFlags.includes('missing_evidence_quote')
  )
  const sections: CitationCardViewSection[] = []

  appendCardViewSection(sections, makeCardViewSection('warning', sectionLabel('warning', '', '提醒'), sectionText('warning', detail.cardWarning), 'warning', { tone: 'warning' }))
  appendCardViewSection(
    sections,
    makeCardViewSection(
      'takeaway',
      sectionLabel('takeaway', detail.cardTakeawayLabel, isSystemB ? '上游作用' : '证据重点'),
      sectionText('takeaway', detail.cardTakeaway),
      'insight',
      { tone: 'primary' },
    ),
  )
  if (!isSystemB) {
    appendCardViewSection(
      sections,
      makeCardViewSection(
        'evidence',
        sectionLabel('evidence', detail.cardEvidenceLabel, '原文证据'),
        sectionText('evidence', detail.cardEvidence),
        'quote',
      ),
    )
    appendCardViewSection(
      sections,
      makeCardViewSection(
        'locator',
        sectionLabel('locator', detail.cardLocatorLabel, '原文位置'),
        sectionText('locator', detail.cardLocator || detail.locationLabel),
        'locator',
      ),
    )
  } else {
    appendCardViewSection(
      sections,
      makeCardViewSection(
        'locator',
        sectionLabel('locator', detail.cardLocatorLabel, '当前论文引用处'),
        sectionText('locator', detail.cardLocator || detail.locationLabel),
        'locator',
      ),
    )
    appendCardViewSection(sections, makeCardViewSection('context_summary', sectionLabel('context_summary', '', '语境摘要'), sectionText('context_summary', detail.cardContextSummary), 'summary'))
    appendCardViewSection(
      sections,
      makeCardViewSection(
        'evidence',
        sectionLabel('evidence', detail.cardEvidenceLabel, '引用语境'),
        sectionText('evidence', detail.cardEvidence),
        'quote',
      ),
    )
    if (
      qualityFlags.includes('missing_reference_title')
      || qualityFlags.includes('reference_entry_only')
      || !title
      || Boolean(storedSection('reference'))
    ) {
      appendCardViewSection(
        sections,
        makeCardViewSection('reference', sectionLabel('reference', detail.cardReferenceLabel, '上游文献条目'), sectionText('reference', detail.cardReferenceEntry), 'reference'),
      )
    }
  }
  if (isSystemB || systemAHasReviewRisk) {
    appendCardViewSection(
      sections,
      makeCardViewSection('support', sectionLabel('support', detail.cardSupportLabel, isSystemB ? '说明' : '可靠度'), sectionText('support', detail.cardSupportExplanation), 'support'),
    )
  }

  const summary = trimShelfSummary(
    sections.find((item) => item.id === 'takeaway')?.text
    || sections.find((item) => item.id === 'context_summary')?.text
    || sections.find((item) => item.id === 'evidence')?.text
    || (storedMatchesRoute ? stored?.summary : '')
    || '',
    260,
  )
  return {
    version: (storedMatchesRoute ? stored?.version : 0) || 1,
    route,
    kind: (storedMatchesRoute ? stored?.kind : '') || detail.cardKind || (isSystemB ? 'upstream_reference' : 'answer_evidence'),
    header: {
      kicker: stored?.header?.kicker || (isSystemB ? '上游引用' : '答案依据'),
      title,
      subtitle,
    },
    sections,
    summary,
    quality: {
      label: detail.cardQualityLabel || (storedMatchesRoute ? stored?.quality?.label : '') || '',
      score: Number(detail.cardQualityScore || (storedMatchesRoute ? stored?.quality?.score : 0) || 0),
      flags: qualityFlags,
      warning: detail.cardWarning || (storedMatchesRoute ? stored?.quality?.warning : '') || '',
    },
  }
}

function trimShelfSummary(value: string, maxLen = 220): string {
  let text = cleanCitationDisplayText(value)
    .replace(/\s+/g, ' ')
    .trim()
  const low = text.toLowerCase()
  if (
    low === 'no summary available'
    || low === 'no summary'
    || low === 'summary pending'
    || low === 'no notes'
    || low === 'none'
    || low === 'n/a'
    || low === 'na'
    || low === 'unknown'
  ) {
    return ''
  }
  if (text.length > maxLen) {
    text = `${text.slice(0, Math.max(0, maxLen - 1)).replace(/[，,；;:：]\s*$/g, '')}...`
  }
  return text
}

function appendUniqueSummaryLine(lines: string[], value: string): void {
  const text = trimShelfSummary(value)
  if (!text) return
  const key = looseTokens(text).join(' ')
  if (!key) return
  for (const line of lines) {
    const existingKey = looseTokens(line).join(' ')
    if (existingKey === key || existingKey.includes(key) || key.includes(existingKey)) return
  }
  lines.push(text)
}

function trustedShelfSummarySource(source: string): boolean {
  const s = String(source || '').trim().toLowerCase()
  return [
    'abstract',
    'fulltext',
    'citation_context',
    'reference_primary_evidence',
    'navigation',
    'exact_anchor',
    'section_intent_rescue',
    'doc_list_seed',
    'doc_list_prompt_aligned',
  ].includes(s)
}

export function looksLowValueShelfSummary(value: string): boolean {
  const text = trimShelfSummary(value, 420)
  if (!text) return false
  const low = text.toLowerCase()
  const genericPatterns = [
    /\u5e2e\u52a9\u6838\u5bf9/,
    /\u7ebf\u7d22\u4ece\u54ea\u91cc\u6765/,
    /\u65b9\u6cd5\u80cc\u666f|\u5b9e\u73b0\u4f9d\u636e/,
    /\u4f5c\u4e3a\u5f53\u524d\u8bba\u6587\u5f15\u7528/,
    /\u5f53\u524d\u8bba\u6587\u5f15\u7528\u7684\u65b9\u6cd5/,
    /\u5f15\u7528\u7684\u65b9\u6cd5\u80cc\u666f/,
    /\u6765\u6e90\u7ebf\u7d22/,
    /\bhelps?\s+(?:verify|check|trace)\b/,
    /\bmethod\s+background\b/,
    /\bcited\s+(?:prior\s+)?work\b/,
  ]
  return genericPatterns.some((pattern) => pattern.test(text) || pattern.test(low))
}

function looksMetadataOnlyShelfSummary(value: string): boolean {
  const text = trimShelfSummary(value, 520)
  if (!text) return false
  return /仅检索到|暂无可用摘要|缺少可用摘要|建议.*DOI|metadata only|no abstract/i.test(text)
}

function isArticleSummaryTextSource(source: string): boolean {
  return [
    'abstract',
    'fulltext',
    'reference_primary_evidence',
    'navigation',
    'exact_anchor',
    'section_intent_rescue',
    'doc_list_seed',
    'doc_list_prompt_aligned',
  ].includes(String(source || '').trim().toLowerCase())
}

function deriveShelfSummary(detail: CiteDetail): { line: string; source: string } {
  const existing = trimShelfSummary(detail.summaryLine, 420)
  const existingSource = String(detail.summarySource || '').trim().toLowerCase()
  const summaryQuality = detail.summaryQuality || {}
  const qualityOk = summaryQuality.ok === true || String(summaryQuality.status || '').trim().toLowerCase() === 'grounded'
  const inpaperContextSummary = detail.isInpaper && existingSource === 'citation_context'
  const metadataOnlyExisting = existingSource === 'metadata' && looksMetadataOnlyShelfSummary(existing)
  const inpaperLowValueContext = detail.isInpaper
    && !isArticleSummaryTextSource(existingSource)
    && looksLowValueCitationContext(existing)
  if (
    existing
    && !inpaperLowValueContext
    && !looksLowValueShelfSummary(existing)
    && !inpaperContextSummary
    && !metadataOnlyExisting
    && (trustedShelfSummarySource(existingSource) || qualityOk)
  ) {
    return { line: existing, source: detail.summarySource || 'fulltext' }
  }

  const viewSummary = trimShelfSummary(citationCardView(detail).summary, 420)
  if (!detail.isInpaper && viewSummary && !looksLowValueShelfSummary(viewSummary)) {
    return { line: viewSummary, source: 'citation_card_view' }
  }
  if (existing && !inpaperContextSummary && !metadataOnlyExisting && !inpaperLowValueContext && !looksLowValueShelfSummary(existing)) {
    return { line: existing, source: existingSource === 'metadata' ? 'fulltext' : (detail.summarySource || 'fulltext') }
  }

  const lines: string[] = []
  if (detail.isInpaper) {
    return { line: '', source: '' }
  }

  appendUniqueSummaryLine(lines, detail.cardTakeaway)
  appendUniqueSummaryLine(lines, detail.answerClaim || detail.cardClaim)
  appendUniqueSummaryLine(lines, detail.evidenceQuote || detail.cardEvidence)
  return { line: lines.slice(0, 3).join(' '), source: lines.length > 0 ? 'citation_card' : '' }
}

export function toShelfItem(detail: CiteDetail): CiteShelfItem {
  const main = citationMain(detail)
  const baseKey = `${detail.anchor}|${detail.sourceName || detail.sourcePath}|${detail.num}`
  const summary = deriveShelfSummary(detail)
  const shelfItemKind = inferShelfItemKind(detail)
  const shelfOrigin = cleanCitationDisplayText(inferShelfOrigin(detail, shelfItemKind))
  const shelfExcerpt = inferShelfExcerpt(detail, shelfItemKind)
  const shelfExcerptLabel = cleanCitationDisplayText(detail.shelfExcerptLabel || defaultShelfExcerptLabel(shelfItemKind))
  return {
    ...detail,
    summaryLine: summary.line,
    summarySource: summary.line ? summary.source : detail.summarySource,
    summaryProvider: detail.summaryProvider,
    shelfItemKind,
    shelfOrigin,
    shelfExcerpt,
    shelfExcerptLabel,
    key: baseKey,
    main,
    tags: [],
    note: '',
  }
}

export function mergeCiteMeta(detail: CiteDetail, meta: Record<string, unknown>): CiteDetail {
  const merged: Record<string, unknown> = { ...detail }
  const currentDoi = normalizeDoiLike(detail.doi || detail.doiUrl)
  const currentRawDoi = extractDoiLike(detail.raw || detail.citeFmt)
  const incomingDoi = normalizeDoiLike(
    asText(meta?.doi) || asText(meta?.doi_url) || asText(meta?.doiUrl),
  )
  const trustedSystemBRepair = Boolean(
    detail.isInpaper
    && !currentRawDoi
    && metadataRepairMetaTrusted(meta),
  )
  const hasDoiConflict = Boolean(currentDoi && incomingDoi && currentDoi !== incomingDoi && !trustedSystemBRepair)
  const overwriteKeys = new Set([
    'doi',
    'doi_url',
    'citation_count',
    'citation_source',
    'journal_if',
    'journal_quartile',
    'journal_if_source',
    'conference_tier',
    'conference_rank_source',
    'conference_ccf',
    'conference_ccf_source',
    'bibliometrics_checked',
    'venue_kind',
    'venue_verified_by',
    'openalex_venue',
    'conference_name',
    'conference_acronym',
    'summary_line',
    'summary_source',
    'summary_provider',
    'summary_quality',
    'library_match',
    'library_match_status',
    'library_match_confidence',
    'library_match_method',
    'library_match_reason',
    'library_match_path',
    'library_match_sha1',
    'library_match_title',
    'library_match_doi',
    'library_match_year',
    'metadata_quality',
    'metadata_repair_status',
    'metadata_repair_sources',
    'metadata_changed_fields',
    'external_metadata_status',
    'external_metadata_reason',
    'external_match_method',
    'external_match_score',
    'external_title_similarity',
    'external_title',
    'external_authors',
    'external_venue',
    'external_year',
    'external_doi',
    'external_doi_url',
    'card_takeaway',
    'card_claim',
    'card_evidence',
    'card_context_summary',
    'card_reference_label',
    'card_reference_entry',
    'card_support_explanation',
    'card_warning',
    'system_b_trace_complete',
    'card_display_contract_version',
    'card_visible_sections',
    'card_view',
    'system_b_trace_score',
    'system_b_trace_reason',
    'system_b_trace_flags',
    'system_b_trace_steps',
    'system_b_trace_answer',
    'system_b_trace_context',
    'system_b_trace_reference',
    'system_b_trace_locator',
    'system_b_trace_source',
    'citation_card_polish_status',
    'citation_card_polish_source',
    'citation_card_polish_checked',
    'citation_card_polish_key',
    'citation_card_polish_route',
    'citation_card_polish_fields',
    'citation_card_polish_rejected',
    'citation_card_polish_quality_score',
  ])
  const conflictSensitiveKeys = new Set([
    'title',
    'authors',
    'venue',
    'year',
    'volume',
    'issue',
    'pages',
    ...overwriteKeys,
  ])
  for (const [key, rawValue] of Object.entries(meta || {})) {
    if (rawValue === null || rawValue === undefined || rawValue === '' || (Array.isArray(rawValue) && rawValue.length === 0)) {
      continue
    }
    if (hasDoiConflict && conflictSensitiveKeys.has(key)) {
      continue
    }
    if (overwriteKeys.has(key)) {
      merged[key] = rawValue
      continue
    }
    if (typeof rawValue !== 'string') {
      merged[key] = rawValue
      continue
    }
    const current = String(merged[key] || '').trim()
    const incoming = rawValue.trim()
    if (!current) {
      merged[key] = incoming
      continue
    }
    const currentWeak = isWeakField(key, current)
    const incomingWeak = isWeakField(key, incoming)
    if (currentWeak && !incomingWeak) {
      merged[key] = incoming
      continue
    }
    if (!currentWeak && incomingWeak) continue
    if (incoming.length > current.length + 12) {
      merged[key] = incoming
    }
  }
  return normalizeCiteDetail(merged) || detail
}

function normalizeTextLite(value: string): string {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9\u4e00-\u9fff]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function firstAuthorToken(value: string): string {
  const firstChunk = String(value || '')
    .split(/[;,\uFF0C\uFF1B]/)[0]
    ?.trim()
    || ''
  const tokens = firstChunk.match(/[A-Za-z\u4e00-\u9fff]+/g) || []
  if (tokens.length <= 0) return ''
  const first = tokens.at(0)
  return first ? first.toLowerCase() : ''
}

function year4(value: string): string {
  const match = String(value || '').match(/\b(19|20)\d{2}\b/)
  return match ? match[0] : ''
}

function jaccardTokens(a: string, b: string): number {
  const aSet = new Set(normalizeTextLite(a).split(' ').filter(Boolean))
  const bSet = new Set(normalizeTextLite(b).split(' ').filter(Boolean))
  if (aSet.size <= 0 || bSet.size <= 0) return 0
  let inter = 0
  for (const t of aSet) {
    if (bSet.has(t)) inter += 1
  }
  const union = aSet.size + bSet.size - inter
  return union > 0 ? inter / union : 0
}

export function strictRepairMerge(base: CiteShelfItem, candidateMeta: Record<string, unknown>): CiteShelfItem | null {
  if (!candidateMeta || Object.keys(candidateMeta).length <= 0) return null
  const merged = mergeCiteMeta(base, candidateMeta)
  const mergedItem = {
    ...toShelfItem(merged),
    key: base.key,
    tags: normalizeShelfTags(base.tags),
    note: normalizeShelfNote(base.note),
  }

  const baseDoi = normalizeDoiLike(base.doi || base.doiUrl)
  const baseRawDoi = extractDoiLike(base.raw || base.citeFmt)
  const mergedDoi = normalizeDoiLike(mergedItem.doi || mergedItem.doiUrl)
  const trustedRepair = metadataRepairMetaTrusted(candidateMeta)
  if (baseDoi && mergedDoi && baseDoi !== mergedDoi) {
    if (!(trustedRepair && base.isInpaper && !baseRawDoi)) return null
  }
  if (baseRawDoi && mergedDoi && baseRawDoi === mergedDoi) return mergedItem
  if (baseRawDoi && mergedDoi && baseRawDoi !== mergedDoi) return null
  if (trustedRepair) return mergedItem

  const titleSignal = jaccardTokens(base.title || base.main, mergedItem.title || mergedItem.main) >= 0.55
  const authorSignal = (
    Boolean(firstAuthorToken(base.authors))
    && firstAuthorToken(base.authors) === firstAuthorToken(mergedItem.authors)
  )
  const yearSignal = Boolean(year4(base.year) && year4(base.year) === year4(mergedItem.year))
  const venueSignal = jaccardTokens(base.venue, mergedItem.venue) >= 0.5
  const newDoiSignal = !baseDoi && Boolean(mergedDoi)

  let signalCount = 0
  if (titleSignal) signalCount += 1
  if (authorSignal) signalCount += 1
  if (yearSignal) signalCount += 1
  if (venueSignal) signalCount += 1

  const accepted = newDoiSignal ? signalCount >= 1 : signalCount >= 2
  if (!accepted) return null
  return mergedItem
}

export function citeMetricSummary(detail: CiteDetail): string[] {
  const items: string[] = []
  if (detail.citationCount > 0) {
    items.push(`被引 ${detail.citationCount}${detail.citationSource ? ` (${detail.citationSource})` : ''}`)
  }
  if (detail.venueKind === 'conference') {
    if (detail.conferenceTier) {
      items.push(`CORE ${detail.conferenceTier}${detail.conferenceRankSource ? ` (${detail.conferenceRankSource})` : ''}`)
    }
    if (detail.conferenceCcf) {
      items.push(`CCF ${detail.conferenceCcf}${detail.conferenceCcfSource ? ` (${detail.conferenceCcfSource})` : ''}`)
    }
  }
  if (detail.journalIf) items.push(`IF ${detail.journalIf}`)
  if (detail.journalQuartile) items.push(`JCR ${detail.journalQuartile}`)
  return items
}

export function shelfProjectScopeId(projectId?: string | null): string {
  return String(projectId || '').trim() || '__default__'
}

export function shelfStorageKey(projectId?: string | null): string {
  return `kb_cite_shelf:project:${shelfProjectScopeId(projectId)}`
}

export function legacyConversationShelfStorageKey(convId?: string | null): string {
  return `kb_cite_shelf:${String(convId || 'default')}`
}

function baseName(path: string): string {
  const text = String(path || '').trim()
  if (!text) return ''
  const parts = text.split(/[\\/]/)
  return String(parts[parts.length - 1] || '').trim()
}

function stripKnownExt(name: string): string {
  return String(name || '')
    .replace(/\.en\.md$/i, '')
    .replace(/\.md$/i, '')
    .replace(/\.pdf$/i, '')
    .trim()
}

function titleFromSourceName(sourceName: string, sourcePath: string): string {
  const raw = stripKnownExt(sourceName || baseName(sourcePath))
  if (!raw) return ''
  let candidate = raw.replace(/_/g, ' ').replace(/\s+/g, ' ').trim()
  const m = candidate.match(/^[A-Za-z]{2,20}-\d{4}-(.+)$/)
  if (m && m[1]) candidate = String(m[1]).trim()
  const m2 = candidate.match(/^\d{4}[-_ ]+(.+)$/)
  if (m2 && m2[1]) candidate = String(m2[1]).trim()
  return isWeakField('title', candidate) ? '' : candidate
}

function looksLikeAuthorSegment(value: string): boolean {
  const s = String(value || '').replace(/\s+/g, ' ').trim()
  if (!s) return false
  if (/\bet\s+al\.?$/i.test(s)) return true
  return /^(?:[A-Z][A-Za-z'`-]+,\s*(?:[A-Z]\.\s*){1,3})(?:,\s*[A-Z][A-Za-z'`-]+,\s*(?:[A-Z]\.\s*){1,3})*/.test(s)
}

function extractTitleFromCitationText(value: string): string {
  const cleaned = stripLeadCitationLabel(String(value || '').replace(/\*+/g, '').replace(/\s+/g, ' ').trim())
    .replace(/\s+(?:doi:\s*|https?:\/\/(?:dx\.)?doi\.org\/)\S+.*$/i, '')
    .trim()
  if (!cleaned) return ''
  const parts = cleaned
    .split(/\.\s+/)
    .map((part) => part.trim().replace(/[.]+$/g, '').trim())
    .filter(Boolean)
  if (parts.length >= 2 && looksLikeAuthorSegment(parts[0])) {
    const candidate = parts[1]
    if (!isLikelyWeakCitationTitle(candidate)) return candidate
  }
  return ''
}

export function citationSourceLabel(detail: CiteDetail): string {
  return detail.sourceName || baseName(detail.sourcePath)
}

function trimLabel(value: string, maxLen = 18): string {
  const s = String(value || '').trim()
  if (!s || s.length <= maxLen) return s
  return `${s.slice(0, Math.max(1, maxLen - 3)).trimEnd()}...`
}

interface InlineCitationLabelOptions {
  includeSource?: boolean
  includeYear?: boolean
  sourceMaxLen?: number
}

function compactSourceChipLabel(
  sourceName: string,
  sourcePath: string,
  options?: Pick<InlineCitationLabelOptions, 'includeYear' | 'sourceMaxLen'>,
): string {
  const includeYear = Boolean(options?.includeYear)
  const maxLen = Number(options?.sourceMaxLen || 18)
  const raw = stripKnownExt(sourceName || baseName(sourcePath))
  if (!raw) return ''
  const normalized = raw.replace(/_/g, ' ').replace(/\s+/g, ' ').trim()
  const byYear = normalized.match(/^(.+?)[-_ ]((?:19|20)\d{2})(?:[-_ ].*)?$/)
  if (byYear) {
    const venue = trimLabel(String(byYear[1] || '').replace(/[-_]+/g, ' ').replace(/\s+/g, ' ').trim(), maxLen)
    const year = String(byYear[2] || '').trim()
    if (!venue) return includeYear ? year : ''
    return includeYear ? [venue, year].filter(Boolean).join(' ') : venue
  }
  const short = trimLabel(
    normalized.replace(/(?:^|[\s\-_])((?:19|20)\d{2})(?=$|[\s\-_])/g, '').replace(/\s+/g, ' ').trim(),
    maxLen,
  )
  return short
}

export function citationInlineLabel(detail: CiteDetail, options?: InlineCitationLabelOptions): string {
  const includeSource = options?.includeSource ?? true
  const visibleNum = !detail.isInpaper && Number(detail.displayNum || 0) > 0 ? Number(detail.displayNum || 0) : detail.num
  const n = visibleNum > 0 ? String(visibleNum) : '?'
  if (!includeSource) return detail.isInpaper ? `[R${n}]` : n
  const sourceTag = compactSourceChipLabel(detail.sourceName, detail.sourcePath, options)
  if (!sourceTag) return n
  return `${sourceTag}#${n}`
}

export function citationDisplay(detail: CiteDetail) {
  const main = (() => {
    const title = String(detail.title || '').trim()
    if (!isWeakField('title', title)) return title
    const parsedTitle = extractTitleFromCitationText(detail.citeFmt || detail.raw || title)
    if (!isLikelyWeakCitationTitle(parsedTitle)) return parsedTitle
    const sourceDerived = titleFromSourceName(detail.sourceName, detail.sourcePath)
    const fallbackMain = citationMain(detail)
    if (sourceDerived && (isWeakField('title', fallbackMain) || looksCitationLine(fallbackMain))) {
      return sourceDerived
    }
    return fallbackMain
  })()
  const authors = isWeakField('authors', detail.authors) ? '' : String(detail.authors || '').trim()
  const venue = isWeakField('venue', detail.venue) ? '' : String(detail.venue || '').trim()
  const source = citationSourceLabel(detail)
  const venueYear = [venue, String(detail.year || '').trim()].filter(Boolean).join(' | ')
  return {
    main,
    authors,
    source,
    venue,
    venueYear,
  }
}

export function buildCiteDetailFromMeta(
  meta: Record<string, unknown> | null | undefined,
  fallback: {
    sourceName?: string
    sourcePath?: string
    num?: number
    anchor?: string
  } = {},
): CiteDetail | null {
  const rec: Record<string, unknown> = { ...(meta || {}) }
  if (!pickText(rec, 'anchor')) {
    rec.anchor = fallback.anchor || `source:${fallback.sourcePath || fallback.sourceName || 'unknown'}`
  }
  if (!pickNumber(rec, 'num') && fallback.num) {
    rec.num = fallback.num
  }
  if (!pickText(rec, 'source_name', 'sourceName') && fallback.sourceName) {
    rec.source_name = fallback.sourceName
  }
  if (!pickText(rec, 'source_path', 'sourcePath') && fallback.sourcePath) {
    rec.source_path = fallback.sourcePath
  }
  return normalizeCiteDetail(rec)
}

function splitCitationAuthors(value: string): string[] {
  const raw = String(value || '').trim()
  if (!raw) return []

  const cleanParts = (parts: string[]): string[] => parts
    .map((part) => part.trim().replace(/\s+/g, ' '))
    .filter(Boolean)

  const semicolonParts = cleanParts(raw.split(/\s*[;；]\s*/g))
  if (semicolonParts.length > 1) return semicolonParts

  const andParts = cleanParts(raw.split(/\s+(?:and|&)\s+/gi))
  if (andParts.length > 1) return andParts

  const commaParts = cleanParts(raw.split(/\s*,\s*/g))
  const looksInitials = (part: string): boolean => /^[A-Z](?:\.|\s|$)(?:\s*[A-Z]\.?)*$/i.test(part.trim())
  if (
    commaParts.length >= 4
    && commaParts.length % 2 === 0
    && commaParts.every((part, idx) => (idx % 2 === 0 ? !looksInitials(part) : looksInitials(part)))
  ) {
    const paired: string[] = []
    for (let idx = 0; idx < commaParts.length; idx += 2) {
      paired.push(`${commaParts[idx]} ${commaParts[idx + 1].replace(/\./g, '').trim()}`.trim())
    }
    return paired
  }
  if (
    commaParts.length > 1
    && commaParts.every((part) => (part.match(/[A-Za-z\u4e00-\u9fff]+/g) || []).length >= 2)
    && commaParts.every((part) => (part.match(/[A-Za-z\u4e00-\u9fff]+/g) || []).length <= 5)
  ) {
    return commaParts
  }

  return [raw]
}

export function citationFormats(detail: CiteDetail): { gbt: string; bibtex: string; ris: string } {
  const title = isWeakField('title', asText(detail.title)) ? citationDisplay(detail).main : asText(detail.title)
  const authors = asText(detail.authors) || '[Unknown Authors]'
  const authorList = splitCitationAuthors(authors)
  const bibtexAuthors = authorList.length > 0 ? authorList.join(' and ') : authors
  const venue =
    asText(detail.conferenceName) ||
    asText(detail.conferenceAcronym) ||
    asText(detail.venue) ||
    'Unknown Venue'
  const year = asText(detail.year) || '20xx'
  const volume = asText(detail.volume)
  const issue = asText(detail.issue)
  const pages = asText(detail.pages)
  const doiUrl = asText(detail.doiUrl)
  const doi = extractDoiLike(detail.doi) || extractDoiLike(doiUrl)
  const canonicalDoiUrl = doiUrl || (doi ? `https://doi.org/${doi}` : '')
  const entryType = detail.venueKind === 'conference' ? 'inproceedings' : 'article'
  const gbtKind = detail.venueKind === 'conference' ? '[C]' : '[J]'

  let suffix = `, ${year}`
  if (volume) suffix += `, ${volume}`
  if (issue) suffix += `(${issue})`
  if (pages) suffix += `: ${pages}`
  const gbt = `${authors}. ${title} ${gbtKind}. ${venue}${suffix}.`

  const keyBase = title.toLowerCase().replace(/[^a-z0-9]+/g, '_').slice(0, 24) || 'reference'
  const venueField = detail.venueKind === 'conference' ? 'booktitle' : 'journal'
  const bibtex = `@${entryType}{ref_${year}_${keyBase},
  title={${title}},
  author={${bibtexAuthors}},
  ${venueField}={${venue}},
  year={${year}},${volume ? `\n  volume={${volume}},` : ''}${issue ? `\n  number={${issue}},` : ''}${pages ? `\n  pages={${pages}},` : ''}${doi ? `\n  doi={${doi}},` : ''}
}`

  const risType = detail.venueKind === 'conference' ? 'CPAPER' : 'JOUR'
  const risAuthors = (() => {
    const raw = authors.trim()
    if (!raw) return ['Unknown Authors']
    const bySep = raw
      .split(/[；;]+/g)
      .map((part) => part.trim())
      .filter(Boolean)
    if (bySep.length > 0) return bySep
    const byAnd = raw
      .split(/\s+(?:and|&)\s+/i)
      .map((part) => part.trim())
      .filter(Boolean)
    return byAnd.length > 0 ? byAnd : [raw]
  })()
  const risLines: string[] = [
    `TY  - ${risType}`,
    `TI  - ${title}`,
  ]
  for (const author of (authorList.length > 0 ? authorList : risAuthors)) {
    risLines.push(`AU  - ${author}`)
  }
  risLines.push(`${detail.venueKind === 'conference' ? 'T2' : 'JO'}  - ${venue}`)
  if (/^\d{4}$/.test(year)) {
    risLines.push(`PY  - ${year}`)
  }
  if (volume) risLines.push(`VL  - ${volume}`)
  if (issue) risLines.push(`IS  - ${issue}`)
  if (pages) {
    const pageMatch = pages.match(/^\s*([A-Za-z0-9]+)\s*[-–]\s*([A-Za-z0-9]+)\s*$/)
    if (pageMatch) {
      risLines.push(`SP  - ${pageMatch[1]}`)
      risLines.push(`EP  - ${pageMatch[2]}`)
    } else {
      risLines.push(`SP  - ${pages}`)
    }
  }
  if (doi) risLines.push(`DO  - ${doi}`)
  if (canonicalDoiUrl || doi) risLines.push(`UR  - ${canonicalDoiUrl || `https://doi.org/${doi}`}`)
  risLines.push('ER  -')
  const ris = risLines.join('\n')

  return { gbt, bibtex, ris }
}

export function summarySourceLabel(
  source: string,
  provider = '',
  labels?: {
    fulltext: string
    crossref: string
    openalex: string
    semanticScholar: string
    doiLandingPage: string
    abstract: string
    citationContext: string
    citationCard: string
    metadata: string
  },
): string {
  const s = String(source || '').trim().toLowerCase()
  const p = String(provider || '').trim().toLowerCase()
  const text = labels || {
    fulltext: '全文',
    crossref: 'Crossref 摘要',
    openalex: 'OpenAlex 摘要',
    semanticScholar: 'Semantic Scholar 摘要',
    doiLandingPage: '出版商页面',
    abstract: '摘要',
    citationContext: '引用语境',
    citationCard: '证据卡片',
    metadata: '元数据',
  }
  if (s === 'fulltext') return text.fulltext
  if (s === 'abstract') {
    if (p === 'crossref') return text.crossref
    if (p === 'openalex') return text.openalex
    if (p === 'semantic_scholar') return text.semanticScholar
    if (p === 'doi_landing_page') return text.doiLandingPage
    return text.abstract
  }
  if (s === 'citation_context') return text.citationContext
  if (s === 'citation_card' || s === 'citation_card_view' || s === 'card_view') return text.citationCard
  if (s === 'metadata') return text.metadata
  return text.metadata
}
