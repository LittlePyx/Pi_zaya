import type { CiteDetail } from './citationState'
import { cleanCitationDisplayText } from './citationState'
import { previewClaimText, previewEvidenceText } from './evidenceCardViewModel'

export const SYSTEM_B_TRACE_ENABLED = false

export function compact(value: string | null | undefined) {
  return String(value || '').trim()
}

export function substantiallySame(left: string | null | undefined, right: string | null | undefined) {
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

function comparablePaperLabel(value: string | null | undefined): string {
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

export function isOnlyPaperLabel(value: string | null | undefined, candidates: string[]): boolean {
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

export function stripLocationIdentityPrefix(value: string | null | undefined, candidates: string[]): string {
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
    const next = text.replace(new RegExp(`^\\s*${escaped}\\s*(?:/|\\u00b7|-|\\u2014|:|\\uff1a)\\s*`, 'i'), '').trim()
    if (next !== text) return next
  }
  return text
}

function compactIdentity(value: string | null | undefined): string {
  return comparablePaperLabel(value).replace(/[^a-z0-9\u4e00-\u9fff]+/g, ' ').trim()
}

function containsIdentityText(value: string | null | undefined, candidate: string | null | undefined, minLen = 22): boolean {
  const body = compactIdentity(value)
  const ident = compactIdentity(candidate)
  return Boolean(body && ident.length >= minLen && body.includes(ident))
}

export function looksNarrativeMetadataText(value: string | null | undefined, detail: CiteDetail): boolean {
  const text = compact(value)
  if (!text) return false
  if (/\b10\.\d{4,9}\/[^\s\uff0c\u3002\uff1b;,)\uff09]+/i.test(text)) return true
  if (/\b(?:doi|jcr|impact\s*factor|if\s*[:\uff1a]?\s*\d|published\s+(?:in|by)|journal|conference|venue|citation\s+count|cited\s+by)\b/i.test(text)) return true
  if (/(?:\u53d1\u8868\u4e8e|\u53d1\u8868\u5728|\u671f\u520a|\u4f1a\u8bae|\u5e74\u4efd|\u88ab\u5f15|\u5f71\u54cd\u56e0\u5b50|\u5206\u533a|\u51fa\u5904|\u6765\u6e90\u8bba\u6587|\u8bba\u6587\u6807\u9898|\u6807\u9898\u662f|\u4f5c\u8005\u662f)/.test(text)) return true
  if (containsIdentityText(text, detail.title) || containsIdentityText(text, detail.cardTitle) || containsIdentityText(text, detail.sourceName) || containsIdentityText(text, detail.sourcePath)) return true
  const venue = compact(detail.venue)
  if (venue && containsIdentityText(text, venue, 7)) return true
  return false
}

export function looksGenericSystemBTakeawayText(value: string | null | undefined): boolean {
  const text = compact(value).replace(/\s+/g, ' ')
  if (!text) return false
  return (
    /\u4f5c\u4e3a\u5f53\u524d\u8bba\u6587\u5f15\u7528\u7684\u65b9\u6cd5\u80cc\u666f\u6216\u5b9e\u73b0\u4f9d\u636e/.test(text)
    || /\u5e2e\u52a9\u6838\u5bf9\u8be5\u65b9\u6cd5\u7ebf\u7d22\u4ece\u54ea\u91cc\u6765/.test(text)
    || /\u628a\u56de\u7b54\u4e2d\u7684\u8bf4\u6cd5\u8ffd\u6eaf\u5230\u5f53\u524d\u8bba\u6587\u5f15\u7528\u7684\u4e0a\u6e38\u6587\u732e/.test(text)
    || /\u53c2\u8003\u6587\u732e\u6761\u76ee.*(?:\u5f53\u524d|\u672c\u6587).*Reader|\u672c\u6587\u5f15\u7528.*\u53c2\u8003\u6587\u732e\u6761\u76ee|\u5f53\u524d\u8bba\u6587.*\u5f15\u7528.*\u53c2\u8003\u6587\u732e\u6761\u76ee/.test(text)
    || /(?:\u6253\u5f00|\u5f53\u524d|\u672c\u6587).*\u8bba\u6587.*\u5f15\u7528.*(?:\u4e0a\u6e38|\u53c2\u8003).*\u6587\u732e/.test(text)
    || /links? the answer back to an upstream reference/i.test(text)
    || /upstream reference cited by the current paper/i.test(text)
    || /opened paper cites this upstream work as reference/i.test(text)
    || /bibliography entry is linked from the opened Reader document/i.test(text)
  )
}

export function isReferenceEntryLikeText(value: string | null | undefined): boolean {
  const text = cleanCitationDisplayText(value || '').replace(/\s+/g, ' ').trim()
  if (!text || text.length < 32) return false
  if (!/\b(?:18|19|20)\d{2}\b/.test(text)) return false
  if (/\b(?:doi|arxiv|isbn|issn)\b|10\.\d{4,9}\//i.test(text)) return true
  const venueLike = /\b(?:IEEE|ACM|Springer|Elsevier|Nature|Science|Nat\.?|Opt\.?|Phys\.?|Journal|Proceedings|Trans\.?|Conf\.?|CVPR|ICCV|ICML|NeurIPS|arXiv)\b/i.test(text)
  const authorLead = /^(?:\[\s*\d{1,4}\s*\]\s*)?(?:[A-Z][A-Za-z'`-]+,\s*(?:[A-Z]\.?\s*){1,4}|[A-Z][a-zA-Z'`-]+\s+[A-Z](?:\.|\b)|[A-Z][a-zA-Z'`-]+\s+et\s+al\.?)/.test(text)
  return venueLike && authorLead
}

export function isLowValueSystemAClaim(value: string | null | undefined): boolean {
  const text = compact(value).replace(/\[[Rr]?\d{1,4}]/g, '').replace(/\s+/g, ' ')
  if (!text || text.length < 18) return true
  const tokens = text.match(/[A-Za-z0-9\u4e00-\u9fff]+/g) || []
  const hasCjk = /[\u4e00-\u9fff]/.test(text)
  if (!hasCjk && tokens.length <= 4) return true
  if (/^[A-Za-z][A-Za-z\s-]{2,48}\s+\d{1,3}$/.test(text)) return true
  const hasSentenceCue = /[\uff1a:\uff0c,\u3002.!?\uff1b;]/.test(text)
  if (hasCjk && text.length < 24 && !hasSentenceCue) return true
  if (!hasCjk && tokens.length <= 6 && !hasSentenceCue) return true
  return false
}

export function pageRangeLabel(start: number, end: number): string {
  const p0 = Number(start || 0)
  const p1 = Number(end || 0)
  if (!Number.isFinite(p0) || p0 <= 0) return ''
  if (!Number.isFinite(p1) || p1 <= 0 || p1 === p0) return `p. ${Math.floor(p0)}`
  return `pp. ${Math.floor(Math.min(p0, p1))}-${Math.floor(Math.max(p0, p1))}`
}

export function anchorKindLabel(
  value: string | null | undefined,
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

export function evidencePreview(value: string, maxLen = 260): string {
  return previewEvidenceText(value, maxLen)
}

export function answerPointPreview(value: string, maxLen = 140): string {
  return previewClaimText(value, maxLen)
}
