export interface CiteDetail {
  num: number
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
  cardReferenceLabel: string
  cardReferenceEntry: string
  cardSupportLabel: string
  cardSupportExplanation: string
  cardQualityLabel: string
  cardQualityScore: number
  cardQualityFlags: string[]
  cardWarning: string
  cardFlow: string[]
  citationCardPolishStatus: string
  citationCardPolishSource: string
  citationCardPolishChecked: boolean
  citationCardPolishKey: string
}

export interface CiteShelfItem extends CiteDetail {
  key: string
  main: string
  tags: string[]
  note: string
}

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

function looseTokens(value: string): string[] {
  return Array.from(String(value || '').matchAll(/[A-Za-z0-9]+|[\u4e00-\u9fff]+/g)).map((match) => match[0].toLowerCase())
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
    .split(/(?<=[。！？!?\.])\s+/)
    .map((item) => item.trim())
    .filter(Boolean)
}

function looksFragmentaryEvidenceSentence(value: string): boolean {
  const text = String(value || '').trim()
  if (!text) return true
  if (/^[a-z]{2,}\b/.test(text) && !FRAGMENT_LEAD_OK_RE.test(text)) return true
  if (/^(?:and|or|of|that|which|from|into|onto|within|without|using|used|measured|allowing)\b/i.test(text)) return true
  if (text.length > 80 && /\b(?:and|or|of|to|with|by|from|into|onto)$/i.test(text)) return true
  if (text.length > 120 && !/[。！？!?\.]$/.test(text)) return true
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
  const claim = trimTakeaway(detail.cardClaim || detail.answerClaim || '')
  if (claim && hasCjkText(claim) && !looksLowValueTakeaway(claim)) return claim

  const evidence = detail.cardEvidence || detail.evidenceQuote || detail.summaryLine || ''
  const evidenceTakeaway = trimTakeaway(takeawayFromEnglishEvidence(evidence))
  if (evidenceTakeaway && !looksLowValueTakeaway(evidenceTakeaway)) return evidenceTakeaway

  const heading = trimTakeaway(detail.headingPath || '', 70)
  if (heading && hasCjkText(heading) && evidence) {
    const candidate = `这条证据对应“${heading.replace(/[。！？?]$/g, '')}”这一部分的关键表述。`
    if (!looksLowValueTakeaway(candidate)) return candidate
  }
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
  const detail: CiteDetail = {
    num: pickNumber(rec, 'num'),
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
    cardReferenceLabel: pickText(rec, 'card_reference_label', 'cardReferenceLabel'),
    cardReferenceEntry: pickText(rec, 'card_reference_entry', 'cardReferenceEntry'),
    cardSupportLabel: pickText(rec, 'card_support_label', 'cardSupportLabel'),
    cardSupportExplanation: pickText(rec, 'card_support_explanation', 'cardSupportExplanation'),
    cardQualityLabel: pickText(rec, 'card_quality_label', 'cardQualityLabel'),
    cardQualityScore: pickNumber(rec, 'card_quality_score', 'cardQualityScore'),
    cardQualityFlags: pickStringArray(rec, 'card_quality_flags', 'cardQualityFlags'),
    cardWarning: pickText(rec, 'card_warning', 'cardWarning'),
    cardFlow: pickStringArray(rec, 'card_flow', 'cardFlow'),
    citationCardPolishStatus: pickText(rec, 'citation_card_polish_status', 'citationCardPolishStatus'),
    citationCardPolishSource: pickText(rec, 'citation_card_polish_source', 'citationCardPolishSource'),
    citationCardPolishChecked: Boolean(rec.citation_card_polish_checked ?? rec.citationCardPolishChecked),
    citationCardPolishKey: pickText(rec, 'citation_card_polish_key', 'citationCardPolishKey'),
  }
  for (const key of [
    'raw',
    'citeFmt',
    'title',
    'summaryLine',
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
    'cardReferenceLabel',
    'cardReferenceEntry',
    'cardSupportExplanation',
    'cardWarning',
    'citationCardPolishStatus',
    'citationCardPolishSource',
    'citationCardPolishKey',
  ] as const) {
    detail[key] = cleanCitationDisplayText(detail[key])
  }
  for (const key of [
    'summaryLine',
    'evidenceQuote',
    'citationContext',
    'cardEvidence',
  ] as const) {
    detail[key] = stripEvidenceMetadataPrefix(detail[key], detail)
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

function trimShelfSummary(value: string, maxLen = 220): string {
  let text = cleanCitationDisplayText(value)
    .replace(/\s+/g, ' ')
    .trim()
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

function titleBasedShelfSummary(detail: CiteDetail): string {
  const title = trimShelfSummary(detail.title || '', 180)
  if (!title) return ''
  const lower = title.toLowerCase()
  if (lower.includes('missing cone') || lower.includes('low-pass distortion')) {
    return '题名显示这篇文献讨论三维显微图像中的缺失锥频率与低通失真问题，可作为理解成像失真或分辨率限制的上游参考。'
  }
  if (lower.includes('interferometric') || lower.includes('iscat')) {
    return '题名显示这篇文献关注干涉或散射显微成像，可作为理解无标记检测与显微分辨率提升的上游参考。'
  }
  if (lower.includes('single-pixel') || lower.includes('compressive')) {
    return '题名显示这篇文献关注单像素或压缩成像，可作为理解相关成像机制、采样策略或重建方法的上游参考。'
  }
  return `题名显示这篇文献关注“${title}”，可先作为当前回答追溯引用来源的候选读物；摘要缺失时建议打开引用语境核对。`
}

function deriveShelfSummary(detail: CiteDetail): { line: string; source: string } {
  const existing = trimShelfSummary(detail.summaryLine, 420)
  const suppressRawSystemBContext = detail.isInpaper
    && (
      detail.cardQualityFlags.includes('weak_citation_context')
      || detail.cardQualityFlags.includes('missing_citation_context')
    )
  if (existing && !(detail.isInpaper && looksLowValueCitationContext(existing))) {
    return { line: existing, source: detail.summarySource || 'metadata' }
  }

  const lines: string[] = []
  if (detail.isInpaper) {
    const takeaway = trimShelfSummary(detail.cardTakeaway || deriveSystemBTakeaway(detail), 220)
    if (takeaway && !looksGenericSystemBTakeaway(takeaway)) appendUniqueSummaryLine(lines, `上游作用：${takeaway}`)

    const rawContext = suppressRawSystemBContext ? '' : (detail.citationContext || detail.evidenceQuote)
    const context = trimShelfSummary(detail.cardEvidence || rawContext, 240)
    if (context && !looksLowValueCitationContext(context)) appendUniqueSummaryLine(lines, `引用语境：${context}`)

    const relation = trimShelfSummary(detail.userQuestionRelation || detail.upstreamWorkRole || detail.supportRelation || detail.whyLine, 220)
    if (relation && !looksGenericSystemBTakeaway(relation)) appendUniqueSummaryLine(lines, relation)

    if (lines.length <= 0) appendUniqueSummaryLine(lines, titleBasedShelfSummary(detail))
    return { line: lines.slice(0, 3).join(' '), source: 'citation_context' }
  }

  appendUniqueSummaryLine(lines, detail.cardTakeaway)
  appendUniqueSummaryLine(lines, detail.answerClaim || detail.cardClaim)
  appendUniqueSummaryLine(lines, detail.evidenceQuote || detail.cardEvidence)
  if (lines.length <= 0) appendUniqueSummaryLine(lines, titleBasedShelfSummary(detail))
  return { line: lines.slice(0, 3).join(' '), source: 'citation_card' }
}

export function toShelfItem(detail: CiteDetail): CiteShelfItem {
  const main = citationMain(detail)
  const baseKey = `${detail.anchor}|${detail.sourceName || detail.sourcePath}|${detail.num}`
  const summary = deriveShelfSummary(detail)
  return {
    ...detail,
    summaryLine: summary.line,
    summarySource: summary.line ? summary.source : detail.summarySource,
    summaryProvider: detail.summaryProvider,
    key: baseKey,
    main,
    tags: [],
    note: '',
  }
}

export function mergeCiteMeta(detail: CiteDetail, meta: Record<string, unknown>): CiteDetail {
  const merged: Record<string, unknown> = { ...detail }
  const currentDoi = normalizeDoiLike(detail.doi || detail.doiUrl)
  const incomingDoi = normalizeDoiLike(
    asText(meta?.doi) || asText(meta?.doi_url) || asText(meta?.doiUrl),
  )
  const hasDoiConflict = Boolean(currentDoi && incomingDoi && currentDoi !== incomingDoi)
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
    'card_reference_label',
    'card_reference_entry',
    'card_support_explanation',
    'card_warning',
    'citation_card_polish_status',
    'citation_card_polish_source',
    'citation_card_polish_checked',
    'citation_card_polish_key',
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

export function citeMetricSummary(detail: CiteDetail): string[] {
  const items: string[] = []
  if (detail.citationCount > 0) {
    items.push(`被引 ${detail.citationCount}${detail.citationSource ? ` (${detail.citationSource})` : ''}`)
  }
  if (detail.venueKind === 'conference') {
    const confLabel = detail.conferenceAcronym || detail.conferenceName || detail.venue
    if (confLabel) items.push(`会议 ${confLabel}`)
    if (detail.year) items.push(`年份 ${detail.year}`)
    if (detail.conferenceTier) {
      items.push(`CORE ${detail.conferenceTier}${detail.conferenceRankSource ? ` (${detail.conferenceRankSource})` : ''}`)
    }
    if (detail.conferenceCcf) {
      items.push(`CCF ${detail.conferenceCcf}${detail.conferenceCcfSource ? ` (${detail.conferenceCcfSource})` : ''}`)
    }
  } else {
    if (detail.venue) items.push(`期刊 ${detail.venue}`)
    if (detail.year) items.push(`年份 ${detail.year}`)
  }
  if (detail.journalIf) items.push(`IF ${detail.journalIf}`)
  if (detail.journalQuartile) items.push(`JCR ${detail.journalQuartile}`)
  return items
}

export function shelfStorageKey(convId?: string | null): string {
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
  const n = detail.num > 0 ? String(detail.num) : '?'
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

export function citationFormats(detail: CiteDetail): { gbt: string; bibtex: string; ris: string } {
  const title = isWeakField('title', asText(detail.title)) ? citationDisplay(detail).main : asText(detail.title)
  const authors = asText(detail.authors) || '[Unknown Authors]'
  const venue =
    asText(detail.conferenceName) ||
    asText(detail.conferenceAcronym) ||
    asText(detail.venue) ||
    'Unknown Venue'
  const year = asText(detail.year) || '20xx'
  const volume = asText(detail.volume)
  const issue = asText(detail.issue)
  const pages = asText(detail.pages)
  const doi = asText(detail.doi)
  const doiUrl = asText(detail.doiUrl)
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
  author={${authors}},
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
  for (const author of risAuthors) {
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
  if (doiUrl || doi) risLines.push(`UR  - ${doiUrl || `https://doi.org/${doi}`}`)
  risLines.push('ER  -')
  const ris = risLines.join('\n')

  return { gbt, bibtex, ris }
}

export function summarySourceLabel(source: string, provider = ''): string {
  const s = String(source || '').trim().toLowerCase()
  const p = String(provider || '').trim().toLowerCase()
  if (s === 'fulltext') return 'fulltext'
  if (s === 'abstract') {
    if (p === 'crossref') return 'Crossref abstract'
    if (p === 'openalex') return 'OpenAlex abstract'
    if (p === 'semantic_scholar') return 'Semantic Scholar abstract'
    if (p === 'doi_landing_page') return 'publisher page'
    return 'abstract'
  }
  if (s === 'citation_context') return 'citation context'
  if (s === 'citation_card') return 'citation card'
  if (s === 'metadata') return 'metadata'
  return 'metadata'
}
