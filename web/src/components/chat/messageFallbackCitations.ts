import { basenameFromSourcePath } from '../../utils/sourcePath'
import {
  normalizeCiteDetail,
  type CiteDetail,
} from './citationState'
import {
  coerceStringArray,
  stripMarkdownInline,
  type RefHitLite,
} from './reader/messageLocateCandidates'

function citationNumbersInMarkdown(content: string): number[] {
  const out: number[] = []
  const seen = new Set<number>()
  const text = String(content || '')
  if (!text || !/\[[Rr]?\d/.test(text)) return out
  const re = /\[([Rr]?\d{1,4}(?:\s*[,，、]\s*[Rr]?\d{1,4})*)\](?!\()/g
  for (const match of text.matchAll(re)) {
    const prev = (match.index || 0) > 0 ? text[(match.index || 0) - 1] : ''
    if (prev === '!' || prev === '[' || prev === '\\') continue
    const body = String(match[1] || '')
    for (const token of body.matchAll(/[Rr]?(\d{1,4})/g)) {
      const num = Number(token[1] || 0)
      if (!Number.isFinite(num) || num <= 0 || seen.has(num)) continue
      seen.add(num)
      out.push(num)
    }
  }
  return out
}

const PLAIN_CITATION_BRACKET_RE = /\[([Rr]?\d{1,4}(?:\s*[,，、;；\-–—−]\s*[Rr]?\d{1,4})*)\](?!\()/gi

function plainCitationBracketContainsNum(body: string, num: number): boolean {
  const target = Number(num || 0)
  if (!Number.isFinite(target) || target <= 0) return false
  for (const token of String(body || '').matchAll(/[Rr]?(\d{1,4})/g)) {
    if (Number(token[1] || 0) === target) return true
  }
  return false
}

function plainCitationMarkerIndex(text: string, num: number): number {
  const exact = new RegExp(`\\[(?:R)?${num}\\]`, 'i')
  const exactIndex = text.search(exact)
  if (exactIndex >= 0) return exactIndex
  for (const match of text.matchAll(PLAIN_CITATION_BRACKET_RE)) {
    if (plainCitationBracketContainsNum(String(match[1] || ''), num)) {
      return match.index ?? -1
    }
  }
  return -1
}

function stripPlainCitationBrackets(text: string): string {
  return String(text || '').replace(PLAIN_CITATION_BRACKET_RE, ' ')
}

function answerClaimAroundCitation(content: string, num: number): string {
  const text = stripMarkdownInline(String(content || '')).replace(/\s+/g, ' ').trim()
  if (!text) return ''
  const idx = plainCitationMarkerIndex(text, num)
  if (idx < 0) return text.slice(0, 240)
  const before = text.slice(0, idx)
  const after = text.slice(idx)
  const start = Math.max(
    before.lastIndexOf('。'),
    before.lastIndexOf('！'),
    before.lastIndexOf('？'),
    before.lastIndexOf('. '),
    before.lastIndexOf('; '),
  )
  const tail = after.search(/[。！？]|\. |; /)
  const end = tail >= 0 ? idx + tail + 1 : Math.min(text.length, idx + 220)
  const sentenceStart = start >= 0 ? start + 1 : (idx > 260 ? Math.max(0, idx - 180) : 0)
  const sentence = text.slice(sentenceStart, end).trim()
  const markerInSentence = Math.max(0, idx - sentenceStart)
  let snippet = stripPlainCitationBrackets(sentence).replace(/\s+/g, ' ').trim()
  snippet = snippet.replace(/^\s*(?:\d{1,3}[.)、．]|[-*•])\s*/, '').trim()
  const maxLen = 180
  if (snippet.length <= maxLen) return snippet
  const focus = Math.min(Math.max(0, markerInSentence), snippet.length)
  const snippetBefore = snippet.slice(0, focus)
  const clauseStart = Math.max(
    snippetBefore.lastIndexOf('。'),
    snippetBefore.lastIndexOf('！'),
    snippetBefore.lastIndexOf('？'),
    snippetBefore.lastIndexOf('；'),
    snippetBefore.lastIndexOf(';'),
    snippetBefore.lastIndexOf('，'),
    snippetBefore.lastIndexOf(','),
    snippetBefore.lastIndexOf('：'),
    snippetBefore.lastIndexOf(':'),
  )
  const startAt = clauseStart >= 0 && focus - clauseStart >= 18 && focus - clauseStart <= maxLen
    ? clauseStart + 1
    : (snippet.length > maxLen * 1.35 ? Math.max(0, focus - 130) : 0)
  snippet = snippet.slice(startAt, Math.min(snippet.length, Math.max(startAt + maxLen, focus + 36))).trim()
  if (snippet.length > maxLen) {
    const soft = Math.max(
      snippet.slice(0, maxLen).lastIndexOf('。'),
      snippet.slice(0, maxLen).lastIndexOf('！'),
      snippet.slice(0, maxLen).lastIndexOf('？'),
      snippet.slice(0, maxLen).lastIndexOf('；'),
      snippet.slice(0, maxLen).lastIndexOf(';'),
      snippet.slice(0, maxLen).lastIndexOf('，'),
      snippet.slice(0, maxLen).lastIndexOf(','),
    )
    snippet = soft >= 40 ? snippet.slice(0, soft) : snippet.slice(0, maxLen - 1)
    snippet = `${snippet.trim()}...`
  }
  return snippet.replace(/^[，,；;:：]\s*/, '').trim()
}

export function buildFallbackCiteDetailsFromRefHits(opts: {
  bodyContent: string
  refHits: RefHitLite[]
  messageId: number
  traceConvId: string
  traceAssistantOrder: number
  traceUserMsgId: number
  S?: Record<string, string>
}): CiteDetail[] {
  const nums = citationNumbersInMarkdown(opts.bodyContent)
  if (nums.length <= 0 || opts.refHits.length <= 0) return []
  const out: CiteDetail[] = []
  for (const num of nums) {
    const hit = opts.refHits[num - 1]
    if (!hit) continue
    const ui = hit.ui_meta || {}
    const meta = hit.meta || {}
    const sourcePath = String(ui.source_path || meta.source_path || '').trim()
    if (!sourcePath) continue
    const sourceName = String(ui.display_name || '').trim() || basenameFromSourcePath(sourcePath) || 'paper'
    const headingPath = String(
      ui.heading_path
      || ui.section_label
      || ui.subsection_label
      || meta.ref_best_heading_path
      || meta.heading_path
      || '',
    ).trim()
    const evidenceQuote = String(
      ui.summary_line
      || coerceStringArray(meta.ref_show_snippets, 1, 900)[0]
      || coerceStringArray(meta.ref_snippets, 1, 900)[0]
      || hit.text
      || '',
    ).trim()
    const whyLine = String(ui.why_line || '').trim()
    const detail = normalizeCiteDetail({
      num,
      anchor: `kb-cite-refhit-${opts.messageId}-${num}`,
      linked_nums: [num],
      evidence_fingerprint: `frontend-refhit-${opts.messageId}-${num}`,
      source_name: sourceName,
      source_path: sourcePath,
      raw: evidenceQuote || sourceName,
      title: headingPath || sourceName,
      heading_path: headingPath,
      answer_claim: answerClaimAroundCitation(opts.bodyContent, num),
      evidence_quote: evidenceQuote,
      evidence_source: 'references_panel_hit',
      summary_line: String(ui.summary_line || evidenceQuote || '').trim(),
      summary_source: 'references_panel_hit',
      location_label: headingPath,
      support_relation: whyLine || opts.S?.cite_candidate_support_default || 'This citation is only candidate evidence. Open the source to confirm the answer sentence and matched passage actually correspond.',
      why_line: whyLine,
      binding_status: 'candidate',
      binding_confidence: 0.35,
      binding_reason: opts.S?.cite_frontend_candidate_reason || 'The backend did not return citation details, so the frontend matched this as a candidate from the References order.',
      score: Number(hit.score || 0),
    })
    if (!detail) continue
    out.push({
      ...detail,
      traceConvId: opts.traceConvId,
      traceAssistantMsgId: opts.messageId,
      traceAssistantOrder: opts.traceAssistantOrder,
      traceUserMsgId: opts.traceUserMsgId,
    })
  }
  return out
}
