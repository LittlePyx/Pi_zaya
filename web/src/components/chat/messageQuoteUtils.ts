import {
  normalizeLocateText,
  overlapScore,
  stripMarkdownInline,
  stripProvenanceNoise,
} from './reader/messageLocateCandidates'

export function extractQuotedSpans(input: string, minLen = 10): string[] {
  const src = stripProvenanceNoise(stripMarkdownInline(String(input || '')))
  if (!src) return []
  const out: string[] = []
  const seen = new Set<string>()
  const push = (raw: string) => {
    const text = String(raw || '').replace(/\s+/g, ' ').trim()
    if (!text || text.length < minLen) return
    const key = normalizeLocateText(text)
    if (!key || seen.has(key)) return
    seen.add(key)
    out.push(text)
  }
  const patterns = [
    /["\u201C\u201D]\s*([^"\u201C\u201D]{6,260}?)\s*["\u201C\u201D]/g,
    /[\u2018\u2019']\s*([^\u2018\u2019']{6,220}?)\s*[\u2018\u2019']/g,
    /[\u300C\u300D\u300E\u300F\u300A\u300B]\s*([^\u300C\u300D\u300E\u300F\u300A\u300B]{6,260}?)\s*[\u300D\u300F\u300B]/g,
  ]
  for (const re of patterns) {
    for (const m of src.matchAll(re)) {
      push(String(m[1] || ''))
      if (out.length >= 6) return out
    }
  }
  return out
}

export function quoteMatchStats(quoteSpans: string[], ...texts: string[]): { hits: number; score: number } {
  if (!Array.isArray(quoteSpans) || quoteSpans.length <= 0) return { hits: 0, score: 0 }
  const normTexts = texts
    .map((text) => normalizeLocateText(stripProvenanceNoise(String(text || ''))))
    .filter(Boolean)
  if (normTexts.length <= 0) return { hits: 0, score: 0 }
  let hits = 0
  let score = 0
  for (const q of quoteSpans) {
    const qNorm = normalizeLocateText(q)
    if (!qNorm) continue
    let exact = false
    let bestOverlap = 0
    for (const t of normTexts) {
      if (t.includes(qNorm) || (qNorm.length >= 16 && qNorm.includes(t))) {
        exact = true
        break
      }
      bestOverlap = Math.max(bestOverlap, overlapScore(qNorm, t))
    }
    if (exact) {
      hits += 1
      score += 1.0
      continue
    }
    if (bestOverlap >= 0.66) {
      hits += 1
      score += 0.72
      continue
    }
    score += 0.45 * bestOverlap
  }
  return { hits, score }
}

export function compactHeadingPath(input: string, maxLen = 56): string {
  const raw = String(input || '').replace(/\s+/g, ' ').trim()
  if (!raw) return ''
  const parts = raw.split('/').map((p) => p.trim()).filter(Boolean)
  const leaf = (parts[parts.length - 1] || raw).trim()
  const tail = parts.length >= 2 ? `${parts[parts.length - 2]} / ${leaf}` : leaf
  const pick = tail.length >= 12 ? tail : leaf
  if (pick.length <= maxLen) return pick
  return `${pick.slice(0, Math.max(18, maxLen - 3)).trimEnd()}...`
}
