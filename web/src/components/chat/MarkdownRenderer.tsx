import { Children, cloneElement, isValidElement, useMemo, type CSSProperties, type MouseEvent, type ReactNode } from 'react'
import { createContext } from 'react'
import { Fragment } from 'react'
import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import type { ComponentPropsWithoutRef } from 'react'
import { message } from 'antd'
import { useT } from '../../i18n'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import rehypeHighlight from 'rehype-highlight'
import { citationInlineLabel, type CiteDetail } from './citationState'
import type { ReaderDocAnchor, ReaderDocBlock } from '../../api/references'

const TABLE_SEPARATOR_RE = /^\s*\|?(?:\s*:?-{2,}:?\s*\|)+\s*:?-{2,}:?\s*\|?\s*$/
const TABLE_ROW_RE = /^\s*\|?.+\|.+\|?\s*$/
const REFERENCES_HEADING_RE = /^#{1,6}\s+(references|bibliography|参考文献)\b/i
const PLAIN_REFERENCES_HEADING_RE = /^(references|bibliography|参考文献)\s*$/i
const REFERENCE_ENTRY_START_RE = /^\s*(?:\[\s*\d{1,4}\s*\](?:\([^)]+\))?|\d{1,4}\.)\s+/
const REFERENCE_ENTRY_LINKED_START_RE = /^\s*\d{1,4}\s+[A-Z\u4e00-\u9fff]/
const PAGE_MARKER_LINE_RE = /^\s*(?:<!--\s*kb_page\s*:\s*(\d{1,5})\s*-->|&lt;!--\s*kb_page\s*:\s*(\d{1,5})\s*--&gt;)\s*$/i
const PAGE_MARKER_HREF_RE = /^kb-page-(\d{1,5})$/i
const INTERNAL_CONVERSION_RETRY_MARKER_RE = /<!--\s*kb:conversion_retry\b(?:(?!-->)[\s\S])*?-->/gi

type ImagePreviewMode = 'fit' | 'actual'
type ImagePreviewSize = { width: number; height: number }

function normalizeTableSegment(segment: string): string {
  let s = String(segment || '').trim()
  if (!s) return ''
  if (!s.startsWith('|')) s = `| ${s}`
  if (!s.endsWith('|')) s = `${s} |`
  return s
}

function repairCollapsedGfmTables(text: string): string {
  if (!text || !text.includes('||')) return text
  const out: string[] = []
  for (const rawLine of text.split('\n')) {
    const line = String(rawLine || '')
    if (!(line.includes('||') && line.includes('|'))) {
      out.push(line)
      continue
    }
    const segments = line
      .split(/\s*\|\|\s*/g)
      .map((part) => normalizeTableSegment(part))
      .filter(Boolean)
    if (segments.length < 2) {
      out.push(line)
      continue
    }
    const hasSeparator = segments.some((seg) => TABLE_SEPARATOR_RE.test(seg))
    const rowLikeCount = segments.filter((seg) => TABLE_ROW_RE.test(seg)).length
    if (hasSeparator && rowLikeCount >= 2) {
      out.push(...segments)
      continue
    }
    out.push(line)
  }
  return out.join('\n')
}

function isReferencesHeadingLine(text: string): boolean {
  const line = String(text || '').trim()
  if (!line) return false
  return REFERENCES_HEADING_RE.test(line) || PLAIN_REFERENCES_HEADING_RE.test(line)
}

function pageMarkerNumber(text: string): string {
  const match = String(text || '').trim().match(PAGE_MARKER_LINE_RE)
  return String(match?.[1] || match?.[2] || '').trim()
}

function isPageMarkerLine(text: string): boolean {
  return Boolean(pageMarkerNumber(text))
}

function normalizeReaderPageMarkers(text: string): string {
  const out: string[] = []
  for (const line of String(text || '').split('\n')) {
    const pageNo = pageMarkerNumber(line)
    if (!pageNo) {
      out.push(line)
      continue
    }
    while (out.length > 0 && !String(out[out.length - 1] || '').trim()) out.pop()
    if (out.length > 0) out.push('')
    out.push(`[Page ${pageNo}](#kb-page-${pageNo})`, '')
  }
  return out.join('\n')
}

function stripInternalConversionRetryMarkers(text: string): string {
  // Conversion retry comments are internal diagnostics. Replace only this exact
  // marker family, leaving all surrounding reader content and unrelated comments
  // untouched. A space prevents words on either side of an inline marker merging.
  return String(text || '').replace(INTERNAL_CONVERSION_RETRY_MARKER_RE, ' ')
}

function splitCollapsedReferenceEntries(line: string): string[] {
  const raw = String(line || '').replace(/\s+/g, ' ').trim()
  if (!raw || !REFERENCE_ENTRY_START_RE.test(raw)) return [line]
  const parts = raw
    .split(/\s+(?=\[\s*\d{1,4}\s*\](?:\([^)]+\))?\s+)/g)
    .map((item) => item.trim())
    .filter(Boolean)
  if (parts.length <= 1) return [line]
  if (!parts.every((item) => REFERENCE_ENTRY_START_RE.test(item))) return [line]
  return parts
}

function normalizeReferenceSectionSpacing(text: string): string {
  if (!text || !/(references|bibliography|参考文献)/i.test(text)) return text
  const lines = text.split('\n')
  const out: string[] = []
  let inReferences = false
  let currentEntryIndex = -1

  for (const rawLine of lines) {
    const line = String(rawLine || '')
    const trimmed = line.trim()

    if (isReferencesHeadingLine(trimmed)) {
      inReferences = true
      currentEntryIndex = -1
      if (out.length > 0 && out[out.length - 1] !== '') {
        out.push('')
      }
      out.push(line)
      out.push('')
      continue
    }

    if (inReferences) {
      if (isPageMarkerLine(trimmed)) {
        if (out.length > 0 && out[out.length - 1] !== '') {
          out.push('')
        }
        out.push(line)
        out.push('')
        currentEntryIndex = -1
        continue
      }
      if (/^#{1,6}\s+/.test(trimmed) && !isReferencesHeadingLine(trimmed)) {
        while (out.length > 0 && out[out.length - 1] === '') {
          out.pop()
        }
        out.push('')
        out.push(line)
        inReferences = false
        currentEntryIndex = -1
        continue
      }
      if (!trimmed) continue
      if (REFERENCE_ENTRY_START_RE.test(trimmed)) {
        for (const entry of splitCollapsedReferenceEntries(trimmed)) {
          if (out.length > 0 && out[out.length - 1] !== '') {
            out.push('')
          }
          out.push(entry)
          currentEntryIndex = out.length - 1
        }
        continue
      }
      if (currentEntryIndex >= 0) {
        out[currentEntryIndex] = `${out[currentEntryIndex]} ${trimmed}`.trim()
        continue
      }
      if (out.length > 0 && out[out.length - 1] !== '') {
        out.push('')
      }
      out.push(line)
      currentEntryIndex = out.length - 1
      continue
    }

    out.push(line)
  }

  return out.join('\n').replace(/\n{3,}/g, '\n\n')
}

const SUPERSCRIPT_CHAR_MAP: Record<string, string> = {
  '⁰': '0',
  '¹': '1',
  '²': '2',
  '³': '3',
  '⁴': '4',
  '⁵': '5',
  '⁶': '6',
  '⁷': '7',
  '⁸': '8',
  '⁹': '9',
}

const SUBSCRIPT_CHAR_MAP: Record<string, string> = {
  '₀': '0',
  '₁': '1',
  '₂': '2',
  '₃': '3',
  '₄': '4',
  '₅': '5',
  '₆': '6',
  '₇': '7',
  '₈': '8',
  '₉': '9',
}

const MICRO_UNIT_LETTERS = 'mWlLsSA'
const GLUED_MICRO_UNIT_RE = new RegExp(`\\\\mu([${MICRO_UNIT_LETTERS}])\\b`, 'g')
const SPACED_MICRO_UNIT_RE = new RegExp(`\\\\mu\\s+(?!\\\\mathrm\\{)([${MICRO_UNIT_LETTERS}])\\b`, 'g')

function normalizeMicroUnitLatex(text: string): string {
  const lines = String(text || '').split('\n')
  let inFence = false
  return lines.map((line) => {
    if (/^\s*```/.test(line)) {
      inFence = !inFence
      return line
    }
    if (inFence) return line
    return line
      .replace(GLUED_MICRO_UNIT_RE, (_match, unit: string) => `\\mu\\mathrm{${unit}}`)
      .replace(SPACED_MICRO_UNIT_RE, (_match, unit: string) => `\\mu\\mathrm{${unit}}`)
  }).join('\n')
}

function normalizePlainMathExpression(value: string): string {
  let s = String(value || '').trim()
  s = s.replace(/[⁰¹²³⁴⁵⁶⁷⁸⁹]+/g, (match) => `^${Array.from(match).map((ch) => SUPERSCRIPT_CHAR_MAP[ch] || ch).join('')}`)
  s = s.replace(/[₀₁₂₃₄₅₆₇₈₉]+/g, (match) => `_${Array.from(match).map((ch) => SUBSCRIPT_CHAR_MAP[ch] || ch).join('')}`)
  s = s.replace(/([A-Za-z])_([A-Za-z][A-Za-z0-9]{1,})/g, '$1_{\\mathrm{$2}}')
  return s
    .replace(/→/g, '\\to')
    .replace(/←/g, '\\leftarrow')
    .replace(/⇒/g, '\\Rightarrow')
    .replace(/↔/g, '\\leftrightarrow')
    .replace(/·/g, '\\cdot')
    .replace(/×/g, '\\times')
    .replace(/≥/g, '\\ge')
    .replace(/≤/g, '\\le')
    .replace(/≠/g, '\\ne')
    .replace(/≈/g, '\\approx')
    .replace(/\s{2,}/g, ' ')
}

function isLikelyPlainMathExpression(value: string): boolean {
  const s = String(value || '').trim()
  if (s.length < 6 || s.length > 420) return false
  if (/[$`[\]{}]/.test(s)) return false
  if (/https?:\/\//i.test(s)) return false
  if (!/[=<>≤≥≈≠∝→←↔⇒]|\\[A-Za-z]+/.test(s)) return false
  const cjkCount = (s.match(/[\u4e00-\u9fff]/g) || []).length
  if (cjkCount > 0 && cjkCount / Math.max(s.length, 1) > 0.12) return false

  let score = 0
  const relationCount = (s.match(/[=<>≤≥≈≠∝→←↔⇒]/g) || []).length
  if (relationCount > 0) score += 2
  if (relationCount >= 2) score += 1
  if (/[_^₀-₉⁰¹²³⁴⁵⁶⁷⁸⁹]/.test(s)) score += 1
  if (/[+\-*/]|·|×|\|\|/.test(s)) score += 1
  if (/[A-Za-z][A-Za-z0-9]*\s*[_₀-₉]/.test(s)) score += 1
  if (/[A-Za-z][A-Za-z0-9_]*\([^)]{1,80}\)/.test(s)) score += 1
  if (/\\[A-Za-z]+/.test(s)) score += 2
  return score >= 4
}

function wrapPlainMathParentheticalsInLine(line: string): string {
  if (!/[=<>≤≥≈≠∝→←↔⇒]/.test(line)) return line
  let out = line.replace(/（([^（）\n]{4,420})）/g, (match, inner: string) => {
    if (!isLikelyPlainMathExpression(inner)) return match
    return `（$${normalizePlainMathExpression(inner)}$）`
  })
  out = out.replace(/\(([^()\n]{4,260})\)/g, (match, inner: string, offset: number, source: string) => {
    // Markdown link and image destinations use the same parenthesis syntax.
    // Never reinterpret their URLs as inline math: relative asset routes often
    // contain both `=` and `_`, which otherwise look formula-like here.
    if (offset > 0 && source[offset - 1] === ']') return match
    if (!isLikelyPlainMathExpression(inner)) return match
    return `($${normalizePlainMathExpression(inner)}$)`
  })
  return out
}

function normalizePlainMathParentheticals(text: string): string {
  const lines = String(text || '').split('\n')
  let inFence = false
  return lines.map((line) => {
    if (/^\s*```/.test(line)) {
      inFence = !inFence
      return line
    }
    if (inFence) return line
    return wrapPlainMathParentheticalsInLine(line)
  }).join('\n')
}

function normalize(text: string) {
  return normalizePlainMathParentheticals(normalizeMicroUnitLatex(normalizeReferenceSectionSpacing(repairCollapsedGfmTables(text))
    .replace(/\\\(/g, '$').replace(/\\\)/g, '$')
    .replace(/\\\[/g, '$$').replace(/\\\]/g, '$$')))
}

function resolvePlainCitationDetail(
  marker: string,
  byNum: Map<number, CiteDetail[]>,
): CiteDetail | null {
  const raw = String(marker || '').trim()
  const wantsInpaper = /^r/i.test(raw)
  const num = Number(raw.replace(/^r/i, ''))
  if (!Number.isFinite(num) || num <= 0) return null
  const candidates = byNum.get(num) || []
  if (candidates.length <= 0) return null
  if (wantsInpaper) {
    return candidates.find((detail) => detail.isInpaper) || (candidates.length === 1 ? candidates[0] : null)
  }
  if (candidates.length === 1) return candidates[0]
  const direct = candidates.filter((detail) => !detail.isInpaper)
  if (direct.length === 1) return direct[0]
  const inpaper = candidates.filter((detail) => detail.isInpaper)
  if (direct.length <= 0 && inpaper.length === 1) return inpaper[0]
  return null
}

function cleanCitationOccurrenceText(value: string): string {
  return String(value || '')
    .replace(/\[[Rr]?\d{1,4}]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function isLowValueCitationOccurrenceContext(value: string): boolean {
  const text = cleanCitationOccurrenceText(value)
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

function nearestBoundaryIndex(value: string, direction: 'left' | 'right'): number {
  const text = String(value || '')
  const boundaries = ['\n', '。', '！', '？', '；', '.', '!', '?', ';']
  if (direction === 'left') {
    let best = -1
    for (const token of boundaries) best = Math.max(best, text.lastIndexOf(token))
    return best
  }
  let best = -1
  for (const token of boundaries) {
    const idx = text.indexOf(token)
    if (idx >= 0 && (best < 0 || idx < best)) best = idx
  }
  return best
}

function citationOccurrenceContextFromLink(link: HTMLElement): string {
  const parent = link.closest('p, li, td, th, blockquote, h1, h2, h3, h4, h5, h6, div') as HTMLElement | null
  if (!parent || typeof document === 'undefined') return ''
  try {
    const beforeRange = document.createRange()
    beforeRange.selectNodeContents(parent)
    beforeRange.setEndBefore(link)
    const beforeText = beforeRange.toString()
    beforeRange.detach()

    const afterRange = document.createRange()
    afterRange.selectNodeContents(parent)
    afterRange.setStartAfter(link)
    const afterText = afterRange.toString()
    afterRange.detach()

    const leftBoundary = nearestBoundaryIndex(beforeText, 'left')
    const left = beforeText.slice(leftBoundary >= 0 ? leftBoundary + 1 : 0)
    const rightBoundary = nearestBoundaryIndex(afterText, 'right')
    const right = rightBoundary >= 0 ? afterText.slice(0, rightBoundary + 1) : afterText.slice(0, 160)
    return cleanCitationOccurrenceText(`${left} ${link.textContent || ''} ${right}`).slice(0, 360)
  } catch {
    return cleanCitationOccurrenceText(parent.innerText || parent.textContent || '').slice(0, 360)
  }
}

function escapeRegExp(value: string): string {
  return String(value || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function withCitationOccurrenceContext(detail: CiteDetail, link: HTMLElement, enabled: boolean): CiteDetail {
  if (!enabled) return detail
  if (detail.isInpaper && link.closest('.kb-md-reference-entry')) return detail
  const context = citationOccurrenceContextFromLink(link)
  if (!context || isLowValueCitationOccurrenceContext(context)) return detail
  const existing = String(detail.cardClaim || detail.answerClaim || '').trim()
  if (existing && cleanCitationOccurrenceText(existing).toLowerCase() === context.toLowerCase()) return detail
  const flags = Array.isArray(detail.cardQualityFlags) ? detail.cardQualityFlags.slice() : []
  if (!flags.includes('occurrence_specific_claim')) flags.push('occurrence_specific_claim')
  if (detail.isInpaper) {
    return {
      ...detail,
      answerClaim: context,
      cardClaim: context,
      cardClaimLabel: detail.cardClaimLabel || 'Citation context',
      citationContext: context,
      citationContextSource: 'reader_occurrence',
      evidenceQuote: context,
      evidenceSource: 'reader_occurrence',
      cardQualityFlags: flags,
    }
  }
  return {
    ...detail,
    answerClaim: context,
    cardClaim: context,
    cardClaimLabel: detail.cardClaimLabel || '对应回答',
    cardQualityFlags: flags,
  }
}

function readerShelfOriginForLink(detail: CiteDetail, link: HTMLElement): string {
  if (link.closest('.kb-md-reference-entry')) return 'reader_references'
  if (detail.isInpaper) return 'reader_cross_reference'
  return 'reader_citation'
}

function withReaderShelfContext(
  detail: CiteDetail,
  opts: {
    origin?: string
    kind?: string
    excerpt?: string
    excerptLabel?: string
    contextSource?: string
  } = {},
): CiteDetail {
  const kind = opts.kind || (detail.isInpaper ? 'reference' : 'citation')
  const origin = opts.origin || detail.shelfOrigin || 'reader_citation'
  const excerpt = String(opts.excerpt || detail.shelfExcerpt || detail.citationContext || detail.evidenceQuote || detail.cardEvidence || '').trim()
  const excerptLabel = String(opts.excerptLabel || detail.shelfExcerptLabel || '').trim()
  return {
    ...detail,
    shelfItemKind: kind,
    shelfOrigin: origin,
    shelfExcerpt: excerpt,
    shelfExcerptLabel: excerptLabel,
    citationContext: excerpt || detail.citationContext,
    citationContextSource: opts.contextSource || detail.citationContextSource,
  }
}

function firstCitationDetailInNode(node: ReactNode, byAnchor: Map<string, CiteDetail>): CiteDetail | null {
  if (node === null || node === undefined || typeof node === 'boolean') return null
  if (typeof node === 'string' || typeof node === 'number') return null
  if (Array.isArray(node)) {
    for (const item of node) {
      const detail = firstCitationDetailInNode(item, byAnchor)
      if (detail) return detail
    }
    return null
  }
  if (!isValidElement(node)) return null
  const props = node.props as { href?: string; children?: ReactNode }
  const href = String(props.href || '').trim()
  const key = href.startsWith('#') ? href.slice(1) : ''
  if (key) {
    const detail = byAnchor.get(key)
    if (detail) return detail
  }
  return firstCitationDetailInNode(props.children, byAnchor)
}

function buildCitationNumberMap(citeDetails: CiteDetail[]): Map<number, CiteDetail[]> {
  const byNum = new Map<number, CiteDetail[]>()
  for (const detail of citeDetails) {
    const anchor = String(detail.anchor || '').trim()
    if (!anchor) continue
    const nums = new Set<number>()
    const primary = Number(detail.num || 0)
    if (Number.isFinite(primary) && primary > 0) nums.add(primary)
    if (Array.isArray(detail.linkedNums)) {
      for (const raw of detail.linkedNums) {
        const num = Number(raw || 0)
        if (Number.isFinite(num) && num > 0) nums.add(num)
      }
    }
    for (const num of nums) {
      const list = byNum.get(num) || []
      list.push(detail)
      byNum.set(num, list)
    }
  }
  return byNum
}

function firstReferenceEntryDetail(text: string, byNum: Map<number, CiteDetail[]>): CiteDetail | null {
  if (byNum.size <= 0) return null
  const match = String(text || '').trim().match(/^\[?\s*(\d{1,4})\s*\]?/)
  const num = Number(match?.[1] || 0)
  if (!Number.isFinite(num) || num <= 0) return null
  const candidates = byNum.get(num) || []
  return candidates.find((detail) => detail.isInpaper) || candidates[0] || null
}

function looksLikeAuthorAffiliationCitationLine(text: string): boolean {
  const trimmed = String(text || '').trim()
  if (!trimmed || !/\[\s*\d/.test(trimmed)) return false
  if (/[.!?。！？]\s*$/.test(trimmed)) return false
  const markerCount = Array.from(trimmed.matchAll(/\[\s*\d{1,3}(?:\s*[,，、;；]\s*\d{1,3})*\s*\]/g)).length
  if (markerCount <= 0) return false
  const words = trimmed.match(/[A-Z][A-Za-z.'-]+|[\u4e00-\u9fff]{2,}/g) || []
  const separatorHints = /(?:,|&|\band\b|；|;)/i.test(trimmed)
  return separatorHints && words.length >= Math.max(4, markerCount + 2)
}

function looksLikeBibliographyEntryText(text: string): boolean {
  const trimmed = String(text || '').replace(/\s+/g, ' ').trim()
  if (!trimmed) return false
  if (!(REFERENCE_ENTRY_START_RE.test(trimmed) || REFERENCE_ENTRY_LINKED_START_RE.test(trimmed))) return false
  const body = trimmed
    .replace(/^\s*(?:\[\s*\d{1,4}\s*\](?:\([^)]+\))?|\d{1,4}\.)\s+/, '')
    .replace(/^\s*\d{1,4}\s+/, '')
    .trim()
  if (!body) return false
  if (/^(?:department|school|college|faculty|institute|state\s+key|key\s+laboratory|laboratory|centre|center|university|academy)\b/i.test(body)) {
    return false
  }
  if (/\b(?:19|20)\d{2}\b/.test(body)) return true
  if (/\b(?:doi|arxiv|isbn|issn)\b|10\.\d{4,9}\//i.test(body)) return true
  if (/\b(?:nat\.|nature|science|cell|ieee|acm|optica|optic|photon|appl\.|phys\.|journal|proceedings|conference|trans\.|lett\.|express|commun\.)\b/i.test(body)) return true
  const sentenceDots = (body.match(/\.\s+/g) || []).length
  return /\bet\s+al\./i.test(body) && sentenceDots >= 1
}

function citationLinkMarkdownForToken(token: string, byNum: Map<number, CiteDetail[]>): { text: string; linked: boolean } {
  const raw = String(token || '').trim()
  if (!raw) return { text: '', linked: false }
  const detail = resolvePlainCitationDetail(raw, byNum)
  if (!detail?.anchor) return { text: `[${raw}]`, linked: false }
  return { text: `[${raw}](#${detail.anchor})`, linked: true }
}

function citationTokenNumber(token: string): number {
  const num = Number(String(token || '').trim().replace(/^r/i, ''))
  return Number.isFinite(num) ? num : 0
}

function citationTokenPrefix(token: string): string {
  return /^r/i.test(String(token || '').trim()) ? 'R' : ''
}

function expandPlainCitationBody(body: string, byNum: Map<number, CiteDetail[]>): { text: string; changed: boolean } {
  const rawBody = String(body || '')
  const tokens = Array.from(rawBody.matchAll(/[Rr]?\d{1,4}/g)).map((match) => ({
    raw: match[0],
    start: match.index ?? 0,
    end: (match.index ?? 0) + match[0].length,
  }))
  if (!tokens.length) return { text: rawBody, changed: false }

  const pieces: Array<{ text: string; linked: boolean }> = []
  for (let idx = 0; idx < tokens.length; idx += 1) {
    const token = tokens[idx]
    const next = tokens[idx + 1]

    const sep = next ? rawBody.slice(token.end, next.start) : ''
    const rangeSep = Boolean(next && /^\s*[-–—−]\s*$/.test(sep))
    const startNum = citationTokenNumber(token.raw)
    const endNum = next ? citationTokenNumber(next.raw) : 0
    const rangePrefix = citationTokenPrefix(token.raw) || (next ? citationTokenPrefix(next.raw) : '')
    const compatiblePrefix = !next || citationTokenPrefix(token.raw) === citationTokenPrefix(next.raw)
    const rangeLen = endNum - startNum + 1

    if (
      rangeSep
      && compatiblePrefix
      && startNum > 0
      && endNum >= startNum
      && rangeLen >= 2
      && rangeLen <= 64
    ) {
      for (let num = startNum; num <= endNum; num += 1) {
        const rendered = citationLinkMarkdownForToken(`${rangePrefix}${num}`, byNum)
        pieces.push(rendered)
      }
      idx += 1
      continue
    }

    const rendered = citationLinkMarkdownForToken(token.raw, byNum)
    pieces.push(rendered)
  }
  const linkedPieces = pieces.filter((piece) => piece.linked)
  if (!linkedPieces.length) return { text: rawBody, changed: false }
  return { text: linkedPieces.map((piece) => piece.text).join(''), changed: true }
}

function linkifyPlainCitationSegment(segment: string, byNum: Map<number, CiteDetail[]>): string {
  if (!segment || byNum.size <= 0 || !/\[[Rr]?\d/.test(segment)) return segment
  return segment.replace(
    /(!?)\[([Rr]?\d{1,4}(?:\s*[,，、;；\-–—−]\s*[Rr]?\d{1,4})*)\](?!\()/g,
    (match, imageBang: string, body: string, offset: number, full: string) => {
      if (imageBang) return match
      const prev = offset > 0 ? full[offset - 1] : ''
      if (prev === '[' || prev === '\\') return match
      const expanded = expandPlainCitationBody(body, byNum)
      return expanded.changed ? expanded.text : match
    },
  )
}

function linkifyPlainCitationTextSegment(segment: string, byNum: Map<number, CiteDetail[]>): string {
  let inReferences = false
  let seenDocumentTitle = false
  let beforeFirstBodyHeading = false
  return String(segment || '').split('\n').map((line) => {
    const trimmed = line.trim()
    if (/^#\s+/.test(trimmed)) {
      seenDocumentTitle = true
      beforeFirstBodyHeading = true
      return line
    }
    if (/^#{2,6}\s+/.test(trimmed)) {
      beforeFirstBodyHeading = false
    }
    if (isReferencesHeadingLine(trimmed)) {
      inReferences = true
      return line
    }
    if (inReferences) {
      if (/^#{1,6}\s+/.test(trimmed) && !isReferencesHeadingLine(trimmed)) {
        inReferences = false
      } else {
        return line
      }
    }
    if (seenDocumentTitle && beforeFirstBodyHeading && looksLikeAuthorAffiliationCitationLine(trimmed)) return line
    return linkifyPlainCitationSegment(line, byNum)
  }).join('\n')
}

function linkifyPlainCitationMarkers(
  text: string,
  citeDetails: CiteDetail[],
  byNum = buildCitationNumberMap(citeDetails),
): string {
  if (!text || citeDetails.length <= 0 || !/\[[Rr]?\d/.test(text)) return text
  if (byNum.size <= 0) return text

  // Citation-shaped text is valid LaTeX (for example, `[66]`). Protect math
  // before rewriting plain prose citations, otherwise the injected Markdown
  // destination is parsed by KaTeX and turns an otherwise valid formula red.
  const protectedRe = /(```[\s\S]*?```|~~~[\s\S]*?~~~|\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|`[^`\n]*`|\$(?:\\.|[^$\\\n])+\$|\\\((?:\\.|[^\\\n])*?\\\))/g
  let out = ''
  let last = 0
  for (const match of text.matchAll(protectedRe)) {
    const index = match.index ?? 0
    out += linkifyPlainCitationTextSegment(text.slice(last, index), byNum)
    out += match[0]
    last = index + match[0].length
  }
  out += linkifyPlainCitationTextSegment(text.slice(last), byNum)
  return out
}

function dedupeRepeatedReaderImageMarkdown(text: string): string {
  const srcRe = /^\s*!\[[^\]]*\]\((.+?)\)\s*$/
  const seen = new Set<string>()
  const out: string[] = []
  for (const rawLine of String(text || '').split('\n')) {
    const line = String(rawLine || '')
    const m = line.match(srcRe)
    if (!m) {
      out.push(line)
      continue
    }
    const src = String(m[1] || '').trim()
    if (!src) {
      out.push(line)
      continue
    }
    if (seen.has(src)) continue
    seen.add(src)
    out.push(line)
  }
  return out.join('\n')
}

function normalizeReaderMarkdown(text: string): string {
  const withoutInternalMarkers = stripInternalConversionRetryMarkers(text)
  return normalizeReaderPageMarkers(normalizeMicroUnitLatex(normalizeReferenceSectionSpacing(dedupeRepeatedReaderImageMarkdown(withoutInternalMarkers))))
}

interface Props {
  content: string
  citeDetails?: CiteDetail[]
  onCitationClick?: (detail: CiteDetail, event: MouseEvent<HTMLElement>) => void
  onCitationAddToShelf?: (detail: CiteDetail) => void
  onCitationHover?: (detail: CiteDetail, event: MouseEvent<HTMLElement>) => void
  onCitationLeave?: (detail: CiteDetail, event: MouseEvent<HTMLElement>) => void
  onLocateSnippet?: (snippet: string, meta?: LocateRenderMeta) => void
  canLocateSnippet?: (snippet: string, meta?: LocateRenderMeta) => boolean
  locateTitleResolver?: (snippet: string) => string
  locateButtonAttrsResolver?: (snippet: string, meta?: LocateRenderMeta) => LocateButtonAttrs | null | undefined
  inlineLocateTokenPolicy?: Partial<Record<InlineLocateTokenKind, boolean>>
  inlineTextLocateEnabled?: boolean
  inlineTextTailLocateEnabled?: boolean
  locateSurfacePolicy?: Partial<Record<LocateSurfaceKind, boolean>>
  onReaderBlockAddToShelf?: (payload: ReaderBlockShelfPayload) => void
  variant?: 'chat' | 'reader'
  readerAnchors?: ReaderDocAnchor[]
  readerBlocks?: ReaderDocBlock[]
}

export interface ReaderBlockShelfPayload {
  text: string
  headingPath?: string
  blockId?: string
  anchorId?: string
  anchorKind?: string
}

type LocateSurfaceKind = 'paragraph' | 'list_item' | 'quote' | 'blockquote' | 'equation' | 'figure' | 'table'

interface LocateRenderMeta {
  kind: LocateSurfaceKind
  order: number
}
interface LocateButtonAttrs {
  className?: string
  focus?: string
  blockId?: string
  anchorId?: string
  anchorKind?: string
  heading?: string
}
type InlineLocateTokenKind = 'quote' | 'figure_ref' | 'equation_ref' | 'table_ref'
interface InlineLocateToken {
  start: number
  end: number
  text: string
  kind: InlineLocateTokenKind
}
interface ReaderAnchorToken {
  anchorId: string
  blockId?: string
  kind: string
  headingPath?: string
  text?: string
  number?: number
}

interface ReaderAnchorAllocator {
  take: (kinds: string[]) => ReaderAnchorToken | null
}

interface ReaderBlockResolver {
  pick: (node: unknown, kinds: string[]) => ReaderAnchorToken | null
}

const BlockquoteLocateContext = createContext(false)

function isCiteLikeElement(node: ReactNode): boolean {
  if (!isValidElement(node)) return false
  const props = node.props as { className?: string; href?: string }
  const className = String(props.className || '')
  const href = String(props.href || '').trim()
  if (/\bkb-cite-chip\b/.test(className)) return true
  if (/^#kb-cite-/i.test(href)) return true
  return false
}

function isEmptyReactNode(node: ReactNode): boolean {
  if (node === null || node === undefined || typeof node === 'boolean') return true
  if (typeof node === 'string') return node.trim().length <= 0
  if (Array.isArray(node)) return node.every((item) => isEmptyReactNode(item))
  return false
}

function isTailBoundaryElement(node: ReactNode): boolean {
  if (!isValidElement(node)) return false
  const nodeType = typeof node.type === 'string' ? node.type.toLowerCase() : ''
  const props = node.props as { className?: string }
  const className = String(props.className || '')
  if (['a', 'button', 'img', 'code', 'pre'].includes(nodeType)) return true
  if (isCiteLikeElement(node) || /\bkb-md-locate-inline-btn\b/.test(className)) return true
  return false
}

function appendTailButtonToContent(children: ReactNode, btn: ReactNode, keyBase = 'tail'): ReactNode {
  const makeTail = (keyPath: string) => (
    <span key={`${keyPath}:btn`} className="kb-md-loc-tail">
      {btn}
    </span>
  )
  const withTail = (content: ReactNode, keyPath: string) => ([
    <Fragment key={`${keyPath}:content`}>{content}</Fragment>,
    makeTail(keyPath),
  ])

  const append = (node: ReactNode, keyPath: string): ReactNode => {
    if (node === null || node === undefined || typeof node === 'boolean') return node
    if (typeof node === 'string' || typeof node === 'number') {
      return withTail(node, keyPath)
    }
    if (Array.isArray(node)) {
      const items = Children.toArray(node)
      for (let idx = items.length - 1; idx >= 0; idx -= 1) {
        if (isEmptyReactNode(items[idx])) continue
        items[idx] = append(items[idx], `${keyPath}:${idx}`) as (typeof items)[number]
        return items
      }
      return [...items, makeTail(`${keyPath}:append`)]
    }
    if (!isValidElement(node)) {
      return withTail(node, keyPath)
    }
    if (isTailBoundaryElement(node)) {
      return withTail(node, keyPath)
    }

    const props = node.props as { children?: ReactNode }
    if (props.children !== undefined) {
      return cloneElement(node, undefined, append(props.children, `${keyPath}:child`))
    }
    return withTail(node, keyPath)
  }

  return append(children, keyBase)
}

function normalizeInlineLocateTokenPolicy(
  policy?: Partial<Record<InlineLocateTokenKind, boolean>>,
): Record<InlineLocateTokenKind, boolean> {
  return {
    quote: policy?.quote !== false,
    figure_ref: policy?.figure_ref !== false,
    equation_ref: policy?.equation_ref !== false,
    table_ref: policy?.table_ref !== false,
  }
}

function normalizeLocateSurfacePolicy(
  policy?: Partial<Record<LocateSurfaceKind, boolean>>,
): Record<LocateSurfaceKind, boolean> {
  return {
    paragraph: policy?.paragraph !== false,
    list_item: policy?.list_item !== false,
    quote: policy?.quote !== false,
    blockquote: policy?.blockquote !== false,
    equation: policy?.equation !== false,
    figure: policy?.figure !== false,
    table: policy?.table !== false,
  }
}

type CiteChipTone = {
  fg: string
  fgHover: string
}

function sourceKey(detail: CiteDetail): string {
  const key = String(detail.sourcePath || detail.sourceName || '').trim().toLowerCase()
  return key || String(detail.anchor || '').trim().toLowerCase()
}

function toneFromIndex(index: number): CiteChipTone {
  const palette: CiteChipTone[] = [
    { fg: '#1f63c6', fgHover: '#134c9d' },
    { fg: '#0f7d6f', fgHover: '#0b6258' },
    { fg: '#8654d6', fgHover: '#6b40b7' },
    { fg: '#bd5b00', fgHover: '#9a4a00' },
    { fg: '#bf3c79', fgHover: '#9f305f' },
    { fg: '#4f6cda', fgHover: '#3f57ba' },
    { fg: '#00799f', fgHover: '#006281' },
    { fg: '#8a6121', fgHover: '#6f4d1a' },
    { fg: '#1a72b1', fgHover: '#135b8d' },
    { fg: '#7a56bf', fgHover: '#62469d' },
    { fg: '#0c857f', fgHover: '#086763' },
    { fg: '#9a4ec2', fgHover: '#7c3ea0' },
    { fg: '#3d77d9', fgHover: '#2f60b5' },
    { fg: '#a95a12', fgHover: '#87480e' },
    { fg: '#b4436e', fgHover: '#943657' },
    { fg: '#1276a3', fgHover: '#0e5e82' },
    { fg: '#3d66c8', fgHover: '#3152a4' },
  ]
  if (index < palette.length) return palette[index]
  const hue = Math.round((index * 137.508) % 360)
  return {
    fg: `hsl(${hue} 72% 44%)`,
    fgHover: `hsl(${hue} 78% 34%)`,
  }
}

function buildToneMap(citeDetails: CiteDetail[]): Map<string, CiteChipTone> {
  const out = new Map<string, CiteChipTone>()
  let next = 0
  for (const detail of citeDetails) {
    const key = sourceKey(detail)
    if (!key || out.has(key)) continue
    out.set(key, toneFromIndex(next))
    next += 1
  }
  return out
}

type AnswerSectionKey = 'conclusion' | 'evidence' | 'limits' | 'next_steps'

const ANSWER_SECTION_LABEL: Record<AnswerSectionKey, string> = {
  conclusion: '结论',
  evidence: '依据',
  limits: '限制',
  next_steps: '下一步',
}

const ANSWER_SECTION_HEAD_RE =
  /^\s*(?:#{1,6}\s*)?(Conclusion|Evidence|Limits|Next\s*Steps|结论|依据|证据|限制|边界|建议|下一步建议|下一步)(?:\s*[:：]\s*(.*))?$/i

interface ParsedAnswerSection {
  key: AnswerSectionKey
  label: string
  body: string
}

function toSectionKey(raw: string): AnswerSectionKey | '' {
  const t = String(raw || '').replace(/\s+/g, '').toLowerCase()
  if (t === 'conclusion' || t === '结论') return 'conclusion'
  if (t === 'evidence' || t === '依据' || t === '证据') return 'evidence'
  if (t === 'limits' || t === '限制' || t === '边界') return 'limits'
  if (t === 'nextsteps' || t === '下一步' || t === '下一步建议') return 'next_steps'
  return ''
}

function extractCode(node: ReactNode): { text: string; language: string } {
  const child = Children.toArray(node)[0]
  if (isValidElement(child)) {
    const props = child.props as { className?: string; children?: ReactNode }
    const classes = String(props.className || '')
      .split(/\s+/)
      .map((item) => item.trim())
      .filter(Boolean)
    let language = ''
    for (const cls of classes) {
      if (cls === 'hljs') continue
      if (cls.startsWith('language-')) {
        language = cls.slice('language-'.length)
        break
      }
      if (cls.startsWith('lang-')) {
        language = cls.slice('lang-'.length)
        break
      }
    }
    if (!language) {
      language = classes.find((cls) => cls !== 'hljs') || ''
    }
    const text = String(Array.isArray(props.children) ? props.children.join('') : props.children || '').replace(/\n$/, '')
    return { text, language }
  }
  return { text: String(node || ''), language: '' }
}

function findElementTextByType(node: ReactNode, targetType: string): string {
  if (node === null || node === undefined || typeof node === 'boolean') return ''
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (Array.isArray(node)) {
    for (const item of node) {
      const text = findElementTextByType(item, targetType)
      if (text) return text
    }
    return ''
  }
  if (!isValidElement(node)) return ''
  const nodeType = typeof node.type === 'string' ? node.type.toLowerCase() : ''
  const props = node.props as { children?: ReactNode }
  if (nodeType === targetType) return plainText(props.children)
  return findElementTextByType(props.children, targetType)
}

function plainText(node: ReactNode): string {
  if (node === null || node === undefined || typeof node === 'boolean') return ''
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map((item) => plainText(item)).join(' ')
  if (isValidElement(node)) {
    const props = node.props as { className?: string; children?: ReactNode }
    const className = String(props.className || '')
    if (isCiteLikeElement(node)) return ''
    if (/\bkb-md-locate-inline-btn\b/.test(className)) return ''
    if (/\bkatex-html\b/.test(className)) return ''
    if (/\bkatex-mathml\b/.test(className)) {
      const annotation = findElementTextByType(props.children, 'annotation')
      return annotation || plainText(props.children)
    }
    if (/\bkatex\b/.test(className)) {
      const annotation = findElementTextByType(props.children, 'annotation')
      return annotation || plainText(props.children)
    }
    return plainText(props.children)
  }
  return ''
}

function rawNodeText(node: ReactNode): string {
  if (node === null || node === undefined || typeof node === 'boolean') return ''
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map((item) => rawNodeText(item)).join(' ')
  if (!isValidElement(node)) return ''
  const props = node.props as { children?: ReactNode }
  return rawNodeText(props.children)
}

function hasMathSignalInline(text: string): boolean {
  const src = String(text || '')
  if (!src) return false
  if (/[=^_]/.test(src)) return true
  if (/\\[a-zA-Z]{2,}/.test(src)) return true
  if (/\$[^$]{1,120}\$/.test(src) || /\$\$[^]{1,260}\$\$/.test(src)) return true
  return false
}

function isDisplayMathClass(className: string): boolean {
  const cls = String(className || '').trim()
  if (!cls) return false
  if (/\bkatex-display\b/.test(cls)) return true
  if (/\bmath-display\b/.test(cls)) return true
  if (/\bmath\b/.test(cls) && /\bdisplay\b/.test(cls)) return true
  return false
}

function isInlineMathClass(className: string): boolean {
  const cls = String(className || '').trim()
  if (!cls || isDisplayMathClass(cls)) return false
  if (/\bkatex-html\b/.test(cls)) return false
  if (/\bkatex-mathml\b/.test(cls)) return false
  if (/\bkatex\b/.test(cls)) return true
  if (/\bmath-inline\b/.test(cls)) return true
  if (/\bmath\b/.test(cls) && /\binline\b/.test(cls)) return true
  return false
}

function toLocateSnippet(node: ReactNode): string {
  let text = plainText(node).replace(/\s+/g, ' ').trim()
  if (!text) {
    text = rawNodeText(node).replace(/\s+/g, ' ').trim()
  }
  if (!text) return ''
  if (hasMathSignalInline(text)) {
    return text.length <= 320 ? text : `${text.slice(0, 320).trimEnd()}...`
  }
  if (text.length <= 220) return text
  const sentences = text
    .split(/(?<=[\u3002\uff01\uff1f.!;:\uff1b\uff1a])\s+/)
    .map((item) => String(item || '').trim())
    .filter(Boolean)
  if (sentences.length > 0) {
    const first = sentences[0] || ''
    if (first.length >= 18) {
      return first.length <= 260 ? first : `${first.slice(0, 260).trimEnd()}...`
    }
    const pair = sentences.slice(0, 2).join(' ').trim()
    if (pair.length >= 20) {
      return pair.length <= 260 ? pair : `${pair.slice(0, 260).trimEnd()}...`
    }
  }
  return `${text.slice(0, 260).trimEnd()}...`
}

function preferredBlockquoteLocateSnippet(node: ReactNode): string {
  const raw = plainText(node).replace(/\s+/g, ' ').trim() || rawNodeText(node).replace(/\s+/g, ' ').trim()
  if (!raw) return ''
  const quoteTokens = collectInlineLocateTokens(raw, { quote: true, figure_ref: false, equation_ref: false, table_ref: false })
    .filter((token) => token.kind === 'quote')
    .sort((a, b) => b.text.length - a.text.length)
  const preferred = String(quoteTokens[0]?.text || '').trim()
  if (preferred.length >= 18) return preferred
  return toLocateSnippet(node)
}

function isFigureShellElement(node: ReactNode): boolean {
  if (!isValidElement(node)) return false
  const nodeType = typeof node.type === 'string' ? node.type.toLowerCase() : ''
  const props = node.props as { className?: string; children?: ReactNode }
  const className = String(props.className || '')
  if (nodeType === 'img') return true
  if (nodeType === 'a') {
    return countFigureShells(props.children) > 0
  }
  return /\bkb-md-figure-shell\b/.test(className)
}

function countFigureShells(node: ReactNode): number {
  if (node === null || node === undefined || typeof node === 'boolean') return 0
  if (typeof node === 'string' || typeof node === 'number') return 0
  if (Array.isArray(node)) return node.reduce((acc, item) => acc + countFigureShells(item), 0)
  if (!isValidElement(node)) return 0
  if (isFigureShellElement(node)) return 1
  const props = node.props as { children?: ReactNode }
  return countFigureShells(props.children)
}

function isFigureHostParagraph(node: ReactNode): boolean {
  const figureCount = countFigureShells(node)
  if (figureCount !== 1) return false
  const text = plainText(node).replace(/\s+/g, ' ').trim()
  return text.length <= 0
}

function preferredFigureCaptionSnippet(node: ReactNode): string {
  const raw = plainText(node).replace(/\s+/g, ' ').trim() || rawNodeText(node).replace(/\s+/g, ' ').trim()
  if (!raw) return ''
  const tokens = collectInlineLocateTokens(raw, { quote: false, figure_ref: true, equation_ref: false, table_ref: false })
    .filter((token) => token.kind === 'figure_ref')
    .sort((a, b) => {
      if (b.text.length !== a.text.length) return b.text.length - a.text.length
      return a.start - b.start
    })
  const preferred = String(tokens[0]?.text || '').trim()
  return preferred
}

function looksLikeDirectQuoteToken(text: string): boolean {
  const inner = String(text || '')
    .replace(/^["'\u2018\u2019\u201C\u201D\u300C\u300D\u300E\u300F\u300A\u300B]+|["'\u2018\u2019\u201C\u201D\u300C\u300D\u300E\u300F\u300A\u300B]+$/g, '')
    .replace(/\s+/g, ' ')
    .trim()
  if (!inner) return false
  if (hasMathSignalInline(inner)) return true
  if (/[。！？.!?；;：:]/.test(inner)) return true
  const cjkCount = (inner.match(/[\u4e00-\u9fff]/g) || []).length
  if (cjkCount >= 24) return true
  const latinWords = inner.match(/[A-Za-z]{2,}/g) || []
  if (latinWords.length >= 8) return true
  return inner.length >= 48
}

function collectInlineLocateTokens(
  text: string,
  policy?: Partial<Record<InlineLocateTokenKind, boolean>>,
): InlineLocateToken[] {
  const src = String(text || '')
  if (!src) return []
  const effectivePolicy = normalizeInlineLocateTokenPolicy(policy)
  const raw: InlineLocateToken[] = []
  const isHeadingLikeQuotedToken = (start: number, text0: string): boolean => {
    const inner = String(text0 || '').replace(/^["'\u2018\u2019\u201C\u201D\u300C\u300D\u300E\u300F\u300A\u300B]+|["'\u2018\u2019\u201C\u201D\u300C\u300D\u300E\u300F\u300A\u300B]+$/g, '').trim()
    if (!inner || inner.length > 64) return false
    const prefix = src.slice(Math.max(0, start - 24), start)
    if (/(?:第\s*\d+\s*节|section\s*\d+|chapter\s*\d+|appendix|附录|章节)/i.test(prefix)) return true
    if (/^(?:introduction|method|methods|background|results?|discussion|conclusion|experiments?|experimental setup|implementation details?)$/i.test(inner)) {
      return true
    }
    return false
  }
  const looksLikeQuotedPaperTitleToken = (start: number, text0: string): boolean => {
    const inner = String(text0 || '')
      .replace(/^["'\u2018\u2019\u201C\u201D\u300C\u300D\u300E\u300F\u300A\u300B]+|["'\u2018\u2019\u201C\u201D\u300C\u300D\u300E\u300F\u300A\u300B]+$/g, '')
      .replace(/\s+/g, ' ')
      .trim()
    if (!inner) return false
    if (hasMathSignalInline(inner)) return false
    if (/[。！？.!?；;]/.test(inner)) return false
    if ((inner.match(/[\u4e00-\u9fff]/g) || []).length > 0) return false
    const latinWords = inner.match(/[A-Za-z][A-Za-z0-9'/-]*/g) || []
    if (latinWords.length < 4) return false
    const prefix = src.slice(Math.max(0, start - 90), start).toLowerCase()
    const suffix = src.slice(start + String(text0 || '').length, start + String(text0 || '').length + 100).toLowerCase()
    const titleContextBefore = /(?:论文|文献|工作|研究|参考|来源|线索|综述|作者|引用|提出|发表|paper|work|study|source|reference|title|titled|entitled|called|cited|proposed|introduced|reported|reviewed)\s*[:：]?\s*$/i
    const titleContextAfter = /^\s*(?:(?:中|里|这篇|这个|提出|发表|总结|综述|系统总结|作为|是)(?=$|[\s，,。；;：:]|\p{Script=Han})|(?:which|that|paper|work|study|source|reference|proposed|introduced|summarized|reviewed|reported|cited)\b)/iu
    if (titleContextBefore.test(prefix) || titleContextAfter.test(suffix)) return true
    return false
  }
  const push = (start: number, end: number, text0: string, kind: InlineLocateTokenKind) => {
    const text = String(text0 || '').replace(/\s+/g, ' ').trim()
    if (!text) return
    if (kind === 'quote') {
      if (text.length < 18) return
      if (isHeadingLikeQuotedToken(start, text)) return
      if (looksLikeQuotedPaperTitleToken(start, text)) return
      if (!looksLikeDirectQuoteToken(text)) return
    }
    raw.push({ start, end, text, kind })
  }
  if (effectivePolicy.quote) {
    for (const pattern of [
      /["\u201C\u201D]\s*([^"\u201C\u201D]{8,360}?)\s*["\u201C\u201D]/g,
      /[\u2018\u2019']\s*([^\u2018\u2019']{8,320}?)\s*[\u2018\u2019']/g,
      /[\u300C\u300D\u300E\u300F\u300A\u300B]([^\u300C\u300D\u300E\u300F\u300A\u300B]{8,360}?)[\u300D\u300F\u300B]/g,
    ]) {
      for (const m of src.matchAll(pattern)) {
        const full = String(m[0] || '')
        const start0 = Number(m.index || 0)
        push(start0, start0 + full.length, full, 'quote')
      }
    }
  }
  if (effectivePolicy.figure_ref) {
    for (const m of src.matchAll(/\b(?:fig(?:ure)?\.?\s*#?\s*\d{1,4}|图\s*\d{1,4})\b/gi)) {
      const full = String(m[0] || '').trim()
      if (!full) continue
      const start0 = Number(m.index || 0)
      push(start0, start0 + full.length, full, 'figure_ref')
    }
  }
  if (effectivePolicy.equation_ref) {
    for (const pattern of [
      /\b(?:eq(?:uation)?s?\.?)\s*(?:[（(]\s*)?#?\d{1,4}(?:\s*[)）])?/gi,
      /(?:公式|方程|式)\s*[（(]?\s*\d{1,4}\s*[)）]?/g,
    ]) {
      for (const m of src.matchAll(pattern)) {
        const full = String(m[0] || '').trim()
        if (!full) continue
        const start0 = Number(m.index || 0)
        push(start0, start0 + full.length, full, 'equation_ref')
      }
    }
  }
  if (effectivePolicy.table_ref) {
    for (const pattern of [
      /\b(?:table|tab\.?)\s*#?\d{1,4}\b/gi,
      /表\s*\d{1,4}/g,
    ]) {
      for (const m of src.matchAll(pattern)) {
        const full = String(m[0] || '').trim()
        if (!full) continue
        const start0 = Number(m.index || 0)
        push(start0, start0 + full.length, full, 'table_ref')
      }
    }
  }
  raw.sort((a, b) => {
    if (a.start !== b.start) return a.start - b.start
    return (b.end - b.start) - (a.end - a.start)
  })
  const out: InlineLocateToken[] = []
  let cursor = -1
  for (const item of raw) {
    if (item.start < cursor) continue
    out.push(item)
    cursor = item.end
  }
  return out
}

function normalizeReaderAnchorKind(input: string): string {
  const raw = String(input || '').trim().toLowerCase()
  if (!raw) return 'paragraph'
  if (raw === 'equation') return 'equation'
  if (raw === 'list_item' || raw === 'list-item' || raw === 'li') return 'list_item'
  if (raw === 'blockquote' || raw === 'quote') return 'blockquote'
  if (raw === 'code' || raw === 'pre') return 'code'
  if (raw === 'table') return 'table'
  if (raw === 'heading' || /^h[1-6]$/.test(raw)) return 'heading'
  if (raw === 'paragraph' || raw === 'p') return 'paragraph'
  return raw
}

function createReaderAnchorAllocator(
  readerAnchors: ReaderDocAnchor[] | undefined,
  readerBlocks: ReaderDocBlock[] | undefined,
): ReaderAnchorAllocator | null {
  const blockList = Array.isArray(readerBlocks) ? readerBlocks : []
  const anchorList = Array.isArray(readerAnchors) ? readerAnchors : []
  const list = blockList.length > 0
    ? blockList.map((item) => ({
      anchor_id: item.anchor_id,
      block_id: item.block_id,
      kind: item.kind,
      heading_path: item.heading_path,
      text: item.text || item.raw_text,
      number: item.number,
    }))
    : anchorList
  if (list.length <= 0) return null
  const all: ReaderAnchorToken[] = []
  const buckets = new Map<string, ReaderAnchorToken[]>()
  const seen = new Set<string>()
  for (const item of list) {
    const anchorId = String(item?.anchor_id || '').trim()
    const blockId = String((item as { block_id?: string } | null)?.block_id || '').trim()
    const dedupeId = blockId || anchorId
    if (!dedupeId || seen.has(dedupeId)) continue
    seen.add(dedupeId)
    const kind = normalizeReaderAnchorKind(String(item?.kind || 'paragraph'))
    const token: ReaderAnchorToken = {
      anchorId,
      blockId: blockId || undefined,
      kind,
      headingPath: String((item as { heading_path?: string } | null)?.heading_path || '').trim() || undefined,
      text: String((item as { text?: string } | null)?.text || '').replace(/\s+/g, ' ').trim() || undefined,
      number: Number.isFinite(Number((item as { number?: number } | null)?.number || 0))
        ? Math.floor(Number((item as { number?: number } | null)?.number || 0))
        : undefined,
    }
    all.push(token)
    const arr = buckets.get(kind) || []
    arr.push(token)
    buckets.set(kind, arr)
  }
  if (all.length <= 0) return null

  const used = new Set<string>()
  const kindCursor = new Map<string, number>()
  let allCursor = 0

  const takeFromKind = (kindRaw: string): ReaderAnchorToken | null => {
    const kind = normalizeReaderAnchorKind(kindRaw)
    const arr = buckets.get(kind) || []
    if (arr.length <= 0) return null
    let cursor = Number(kindCursor.get(kind) || 0)
    while (cursor < arr.length) {
      const token = arr[cursor]
      cursor += 1
      if (used.has(token.anchorId)) continue
      kindCursor.set(kind, cursor)
      used.add(token.anchorId)
      return token
    }
    kindCursor.set(kind, cursor)
    return null
  }

  const takeAny = (): ReaderAnchorToken | null => {
    while (allCursor < all.length) {
      const token = all[allCursor]
      allCursor += 1
      if (used.has(token.anchorId)) continue
      used.add(token.anchorId)
      return token
    }
    return null
  }

  return {
    take: (kinds: string[]) => {
      for (const kind of kinds || []) {
        const token = takeFromKind(kind)
        if (token) return token
      }
      return takeAny()
    },
  }
}

function _nodeLineRange(node: unknown): { start: number; end: number } | null {
  const rec = (node || {}) as {
    position?: {
      start?: { line?: number }
      end?: { line?: number }
    }
  }
  const start = Number(rec.position?.start?.line || 0)
  const endRaw = Number(rec.position?.end?.line || 0)
  if (!Number.isFinite(start) || start <= 0) return null
  const end = Number.isFinite(endRaw) && endRaw > 0 ? Math.max(start, endRaw) : start
  return { start: Math.floor(start), end: Math.floor(end) }
}

function _readerNodeText(node: unknown): string {
  if (!node || typeof node !== 'object') return ''
  const rec = node as { value?: unknown; children?: unknown[] }
  const own = typeof rec.value === 'string' ? rec.value : ''
  const childText = Array.isArray(rec.children)
    ? rec.children.map((child) => _readerNodeText(child)).filter(Boolean).join(' ')
    : ''
  return `${own} ${childText}`.replace(/\s+/g, ' ').trim()
}

function _readerTextTokenCounts(text: string): Map<string, number> {
  const counts = new Map<string, number>()
  const tokens = String(text || '').toLowerCase().match(/[a-z][a-z0-9+_-]{1,}|[+-]?(?:\d+(?:\.\d+)?|\.\d+)/g) || []
  for (const token of tokens) counts.set(token, Number(counts.get(token) || 0) + 1)
  return counts
}

function _readerTextSimilarity(left: string, right: string): number {
  const leftCounts = _readerTextTokenCounts(left)
  const rightCounts = _readerTextTokenCounts(right)
  const leftTotal = [...leftCounts.values()].reduce((sum, count) => sum + count, 0)
  const rightTotal = [...rightCounts.values()].reduce((sum, count) => sum + count, 0)
  if (Math.min(leftTotal, rightTotal) < 4) return 0
  let shared = 0
  for (const [token, count] of leftCounts.entries()) {
    shared += Math.min(count, Number(rightCounts.get(token) || 0))
  }
  return (2 * shared) / Math.max(1, leftTotal + rightTotal)
}

function createReaderBlockResolver(readerBlocks: ReaderDocBlock[] | undefined): ReaderBlockResolver | null {
  const rows = Array.isArray(readerBlocks) ? readerBlocks : []
  if (rows.length <= 0) return null
  const list = rows
    .map((row) => {
      const anchorId = String(row?.anchor_id || '').trim()
      const blockId = String(row?.block_id || '').trim()
      const kind = normalizeReaderAnchorKind(String(row?.kind || 'paragraph'))
      const lineStart = Number(row?.line_start || 0)
      const lineEndRaw = Number(row?.line_end || 0)
      const lineEnd = Number.isFinite(lineEndRaw) && lineEndRaw > 0 ? Math.max(lineStart, lineEndRaw) : lineStart
      if ((!anchorId && !blockId) || !Number.isFinite(lineStart) || lineStart <= 0) return null
      return {
        token: {
          anchorId,
          blockId: blockId || undefined,
          kind,
          headingPath: String(row?.heading_path || '').trim() || undefined,
          text: String(row?.text || row?.raw_text || '').replace(/\s+/g, ' ').trim() || undefined,
          number: Number.isFinite(Number(row?.number || 0)) ? Math.floor(Number(row?.number || 0)) : undefined,
        },
        kind,
        lineStart: Math.floor(lineStart),
        lineEnd: Math.floor(lineEnd),
        span: Math.max(1, Math.floor(lineEnd - lineStart + 1)),
      }
    })
    .filter((item): item is NonNullable<typeof item> => Boolean(item))
  if (list.length <= 0) return null

  return {
    pick: (node: unknown, kinds: string[]) => {
      const range = _nodeLineRange(node)
      const preferred = new Set((kinds || []).map((k) => normalizeReaderAnchorKind(k)))
      if (preferred.has('table')) {
        const nodeText = _readerNodeText(node)
        const rankedByText = list
          .filter((item) => item.kind === 'table' && item.token.text)
          .map((item) => ({ item, score: _readerTextSimilarity(nodeText, String(item.token.text || '')) }))
          .sort((left, right) => right.score - left.score)
        const bestText = rankedByText[0]
        const runnerUpText = rankedByText[1]
        if (
          bestText
          && bestText.score >= 0.58
          && (!runnerUpText || bestText.score - runnerUpText.score >= 0.08)
        ) {
          return bestText.item.token
        }
      }
      if (!range) return null
      let best: (typeof list)[number] | null = null
      let bestScore = Number.NEGATIVE_INFINITY

      for (const item of list) {
        const overlap = Math.max(
          0,
          Math.min(range.end, item.lineEnd) - Math.max(range.start, item.lineStart) + 1,
        )
        if (overlap <= 0) continue
        let score = (3.2 * overlap) - (0.02 * item.span)
        if (preferred.has(item.kind)) score += 2.8
        if (item.kind === 'equation' && preferred.has('equation')) score += 0.6
        if (score > bestScore) {
          best = item
          bestScore = score
        }
      }

      if (best) return best.token

      for (const item of list) {
        const dist = Math.min(
          Math.abs(range.start - item.lineStart),
          Math.abs(range.end - item.lineEnd),
        )
        if (dist > 2) continue
        let score = 1.0 - (0.22 * dist)
        if (preferred.has(item.kind)) score += 0.8
        if (score > bestScore) {
          best = item
          bestScore = score
        }
      }
      return best ? best.token : null
    },
  }
}

function readerAnchorAttrs(anchor: ReaderAnchorToken | null): Record<string, string> | undefined {
  if (!anchor) return undefined
  const attrs: Record<string, string> = {
    'data-kb-anchor-id': anchor.anchorId,
    'data-kb-anchor-kind': anchor.kind,
  }
  if (anchor.blockId) attrs['data-kb-block-id'] = anchor.blockId
  if (Number.isFinite(Number(anchor.number || 0)) && Number(anchor.number || 0) > 0) {
    attrs['data-kb-anchor-number'] = String(Math.floor(Number(anchor.number || 0)))
  }
  return attrs
}

function parseAnswerContract(text: string): { preamble: string; sections: ParsedAnswerSection[] } | null {
  const src = String(text || '')
  if (!src) return null
  const lines = src.split('\n')
  const sections: Array<{ key: AnswerSectionKey; lines: string[] }> = []
  const preamble: string[] = []
  let current: { key: AnswerSectionKey; lines: string[] } | null = null

  for (const rawLine of lines) {
    const line = String(rawLine || '')
    const m = line.match(ANSWER_SECTION_HEAD_RE)
    const key = m ? toSectionKey(String(m[1] || '')) : ''
    if (m && key) {
      if (current) sections.push(current)
      current = { key, lines: [] }
      const tail = String(m[2] || '').trim()
      if (tail) current.lines.push(tail)
      continue
    }
    if (current) current.lines.push(line)
    else preamble.push(line)
  }
  if (current) sections.push(current)

  const normalized = sections
    .map((section) => ({
      key: section.key,
      label: ANSWER_SECTION_LABEL[section.key],
      body: section.lines.join('\n').replace(/^\n+|\n+$/g, '').trim(),
    }))
    .filter((section) => section.body.length > 0)

  if (normalized.length < 2) return null
  const keys = new Set(normalized.map((section) => section.key))
  if (!keys.has('conclusion')) return null
  return {
    preamble: preamble.join('\n').replace(/^\n+|\n+$/g, '').trim(),
    sections: normalized,
  }
}

function buildMarkdownComponents(
  byAnchor: Map<string, CiteDetail>,
  citationByNum: Map<number, CiteDetail[]>,
  duplicateCitationAnchors: Set<string>,
  onCitationClick?: (detail: CiteDetail, event: MouseEvent<HTMLElement>) => void,
  onCitationAddToShelf?: (detail: CiteDetail) => void,
  onCitationHover?: (detail: CiteDetail, event: MouseEvent<HTMLElement>) => void,
  onCitationLeave?: (detail: CiteDetail, event: MouseEvent<HTMLElement>) => void,
  toneBySource?: Map<string, CiteChipTone>,
  onLocateSnippet?: (snippet: string, meta?: LocateRenderMeta) => void,
  canLocateSnippet?: (snippet: string, meta?: LocateRenderMeta) => boolean,
  locateTitleResolver?: (snippet: string) => string,
  locateButtonAttrsResolver?: (snippet: string, meta?: LocateRenderMeta) => LocateButtonAttrs | null | undefined,
  inlineLocateTokenPolicy?: Partial<Record<InlineLocateTokenKind, boolean>>,
  inlineTextLocateEnabled: boolean = true,
  inlineTextTailLocateEnabled: boolean = false,
  locateSurfacePolicy?: Partial<Record<LocateSurfaceKind, boolean>>,
  variant: 'chat' | 'reader' = 'chat',
  readerAnchorAllocator?: ReaderAnchorAllocator | null,
  readerBlockResolver?: ReaderBlockResolver | null,
  S?: Record<string, string>,
  onImagePreview?: (src: string, alt: string) => void,
  onReaderBlockAddToShelf?: (payload: ReaderBlockShelfPayload) => void,
) {
  const effectiveInlineLocateTokenPolicy = normalizeInlineLocateTokenPolicy(inlineLocateTokenPolicy)
  const effectiveLocateSurfacePolicy = normalizeLocateSurfacePolicy(locateSurfacePolicy)
  let locateRenderOrder = 0
  const pickReaderAnchor = (node: unknown, kinds: string[]) => {
    if (variant !== 'reader') return null
    const byBlock = readerBlockResolver?.pick(node, kinds)
    if (byBlock) return byBlock
    return readerAnchorAllocator?.take(kinds) || null
  }

  const nextLocateRenderOrder = (): number => {
    locateRenderOrder += 1
    return locateRenderOrder
  }

  const renderLocateButton = (
    children: ReactNode | string,
    opts?: { force?: boolean; meta?: LocateRenderMeta; snippetOverride?: string },
  ) => {
    if (!onLocateSnippet) return null
    const force = Boolean(opts?.force)
    const meta = opts?.meta
    if (meta && !effectiveLocateSurfacePolicy[meta.kind]) {
      return null
    }
    let snippet = String(opts?.snippetOverride || '').trim()
    if (!snippet) {
      snippet = typeof children === 'string'
        ? String(children || '').replace(/\s+/g, ' ').trim()
        : toLocateSnippet(children)
    }
    if (!snippet && force && typeof children !== 'string') {
      const raw = rawNodeText(children).replace(/\s+/g, ' ').trim()
      if (raw) {
        snippet = raw.length <= 480 ? raw : `${raw.slice(0, 480).trimEnd()}...`
      }
    }
    if (!snippet) return null
    if (canLocateSnippet && !canLocateSnippet(snippet, meta)) {
      return null
    }
    if (!force && !canLocateSnippet) {
      const raw = String(snippet || '').trim()
      if (!(hasMathSignalInline(raw) || raw.length >= 18)) return null
    }
    const label = S?.locate_label || '定位到原文证据'
    const title = String(locateTitleResolver?.(snippet) || '').trim() || label
    const kind = meta?.kind || 'paragraph'
    const locateAttrs = locateButtonAttrsResolver?.(snippet, meta) || {}
    const extraClassName = String(locateAttrs.className || '').trim()
    const badgeText = kind === 'equation'
      ? (S?.locate_badge_eq || '式')
      : kind === 'quote'
        ? (S?.locate_badge_quote || '引')
      : kind === 'figure'
        ? (S?.locate_badge_fig || '图')
      : kind === 'table'
        ? (S?.locate_badge_table || '表')
        : (S?.locate_badge_source || '原文')
    return (
      <button
        type="button"
        className={`kb-md-locate-inline-btn kb-md-locate-inline-btn-${kind}${extraClassName ? ` ${extraClassName}` : ''}`}
        aria-label={label}
        title={title || label}
        data-locate-kind={kind}
        data-kb-locate-focus={locateAttrs.focus || undefined}
        data-kb-locate-block-id={locateAttrs.blockId || undefined}
        data-kb-locate-anchor-id={locateAttrs.anchorId || undefined}
        data-kb-locate-anchor-kind={locateAttrs.anchorKind || undefined}
        data-kb-locate-heading={locateAttrs.heading || undefined}
        onClick={(event) => {
          event.preventDefault()
          event.stopPropagation()
          onLocateSnippet(snippet, meta)
        }}
      >
        <span className="kb-md-locate-inline-label" aria-hidden="true">{badgeText}</span>
      </button>
    )
  }

  const renderReaderBlockShelfButton = (
    anchor: ReaderAnchorToken | null,
    fallback: ReactNode | string,
    fallbackKind?: 'figure' | 'equation' | 'table',
  ) => {
    if (variant !== 'reader' || !onReaderBlockAddToShelf) return null
    const kind = normalizeReaderAnchorKind(anchor?.kind || fallbackKind || '')
    if (!['figure', 'equation', 'table'].includes(kind)) return null
    let text = String(anchor?.text || '').replace(/\s+/g, ' ').trim()
    if (!text) {
      text = typeof fallback === 'string'
        ? String(fallback || '').replace(/\s+/g, ' ').trim()
        : toLocateSnippet(fallback)
    }
    if (!text) return null
    const excerpt = text.length <= 1400 ? text : `${text.slice(0, 1400).trimEnd()}...`
    const label = S?.reader_add_to_shelf || 'Shelf'
    const kindLabel = kind === 'equation'
      ? (S?.locate_badge_eq || 'Eq')
      : kind === 'figure'
        ? (S?.locate_badge_fig || 'Fig')
        : (S?.locate_badge_table || 'Tbl')
    const title = S?.reader_add_to_shelf_title || 'Add to research basket'
    return (
      <button
        type="button"
        className={`kb-md-reader-block-shelf kb-md-reader-block-shelf-${kind}`}
        title={title}
        aria-label={title}
        data-testid="reader-block-shelf"
        data-kb-reader-block-kind={kind}
        onClick={(event) => {
          event.preventDefault()
          event.stopPropagation()
          onReaderBlockAddToShelf({
            text: excerpt,
            headingPath: anchor?.headingPath,
            blockId: anchor?.blockId,
            anchorId: anchor?.anchorId,
            anchorKind: kind,
          })
        }}
      >
        <span className="kb-md-reader-block-shelf-kind" aria-hidden="true">{kindLabel}</span>
        <span className="kb-md-reader-block-shelf-text">{label}</span>
      </button>
    )
  }

  const decorateInlineLocateAnchors = (
    children: ReactNode,
    meta: LocateRenderMeta,
  ): { content: ReactNode; count: number; figureRefCount: number } => {
    type InlineDecorateResult = { content: ReactNode; count: number; figureRefCount: number }
    const metaForToken = (kind: InlineLocateTokenKind): LocateRenderMeta => {
      if (kind === 'quote') return { ...meta, kind: 'quote' }
      if (kind === 'figure_ref') return { ...meta, kind: 'figure' }
      if (kind === 'equation_ref') return { ...meta, kind: 'equation' }
      if (kind === 'table_ref') return { ...meta, kind: 'table' }
      return meta
    }
    const renderStringNode = (text0: string, keyBase: string): InlineDecorateResult => {
      const tokens = collectInlineLocateTokens(text0, effectiveInlineLocateTokenPolicy)
      if (tokens.length <= 0) return { content: text0, count: 0, figureRefCount: 0 }
      const parts: ReactNode[] = []
      let last = 0
      let count = 0
      let figureRefCount = 0
      tokens.forEach((token, idx) => {
        if (token.start > last) {
          parts.push(text0.slice(last, token.start))
        }
        const raw = text0.slice(token.start, token.end)
        const btn = renderLocateButton(raw, {
          force: true,
          meta: metaForToken(token.kind),
          snippetOverride: token.text,
        })
        if (btn) {
          parts.push(
            <span key={`${keyBase}:${idx}:${token.start}`} className="kb-md-loc-inline">
              {raw}
              {btn}
            </span>,
          )
          count += 1
          if (token.kind === 'figure_ref') {
            figureRefCount += 1
          }
        } else {
          parts.push(raw)
        }
        last = token.end
      })
      if (last < text0.length) {
        parts.push(text0.slice(last))
      }
      return { content: parts, count, figureRefCount }
    }

    const visit = (node: ReactNode, keyBase: string): InlineDecorateResult => {
      if (node === null || node === undefined || typeof node === 'boolean') {
        return { content: node, count: 0, figureRefCount: 0 }
      }
      if (typeof node === 'string' || typeof node === 'number') {
        return renderStringNode(String(node), keyBase)
      }
      if (Array.isArray(node)) {
        const items = Children.toArray(node)
        let count = 0
        let figureRefCount = 0
        const content = items.map((item, idx) => {
          const rendered = visit(item, `${keyBase}:${idx}`)
          count += rendered.count
          figureRefCount += rendered.figureRefCount
          return rendered.content
        })
        return { content, count, figureRefCount }
      }
      if (!isValidElement(node)) {
        return { content: node, count: 0, figureRefCount: 0 }
      }
      const nodeType = typeof node.type === 'string' ? node.type.toLowerCase() : ''
      const props = node.props as { children?: ReactNode; className?: string }
      const className = String(props.className || '')
      if (isInlineMathClass(className)) {
        // Inline KaTeX variables create noisy duplicate entrances; keep entrances
        // only on numbered equation refs / block formulas.
        return { content: node, count: 0, figureRefCount: 0 }
      }
      if (['a', 'button', 'img', 'code', 'pre', 'script', 'style'].includes(nodeType)) {
        return { content: node, count: 0, figureRefCount: 0 }
      }
      if (/\bkb-cite-chip\b/.test(className) || /\bkb-md-locate-inline-btn\b/.test(className)) {
        return { content: node, count: 0, figureRefCount: 0 }
      }
      const rendered = visit(props.children, `${keyBase}:child`)
      if (rendered.count <= 0) return { content: node, count: 0, figureRefCount: 0 }
      return {
        content: cloneElement(node, undefined, rendered.content),
        count: rendered.count,
        figureRefCount: rendered.figureRefCount,
      }
    }

    return visit(children, `loc-${meta.order}-${meta.kind}`)
  }
  return {
    pre: ({ node, children }: { node?: unknown; children?: ReactNode }) => {
      const { text, language } = extractCode(children)
      if (variant === 'reader') {
        const attrs = readerAnchorAttrs(pickReaderAnchor(node, ['code']))
        return (
          <pre {...attrs}>
            <code>{text}</code>
          </pre>
        )
      }
      return (
        <div className="kb-code-block">
          <div className="kb-code-head">
            <span className="kb-code-lang">{language || 'text'}</span>
            <button
              type="button"
              className="kb-code-copy"
              onClick={() => {
                navigator.clipboard.writeText(text).then(() => message.success(S?.code_copied || '代码已复制'))
              }}
            >
              {S?.code_copy || '复制代码'}
            </button>
          </div>
          <pre>{children}</pre>
        </div>
      )
    },
    table: ({ node, children }: { node?: unknown; children?: ReactNode }) => {
      const anchor = variant === 'reader' ? pickReaderAnchor(node, ['table']) : null
      const attrs = variant === 'reader'
        ? readerAnchorAttrs(anchor)
        : undefined
      const renderOrder = nextLocateRenderOrder()
      const locateBtn = variant === 'chat'
        ? renderLocateButton(children, {
          meta: { kind: 'table', order: renderOrder },
        })
        : null
      const shelfBtn = renderReaderBlockShelfButton(anchor, children)
      const tableClass = [
        'kb-table-wrap',
        locateBtn || shelfBtn ? 'kb-md-table-action-host' : '',
        shelfBtn ? 'kb-md-reader-block-action-host' : '',
      ].filter(Boolean).join(' ')
      return (
        <div className={tableClass}>
          <table {...attrs}>{children}</table>
          {locateBtn ? <span className="kb-md-table-tail">{locateBtn}</span> : null}
          {shelfBtn ? <span className="kb-md-reader-block-shelf-tail">{shelfBtn}</span> : null}
        </div>
      )
    },
    blockquote: ({ node, children }: { node?: unknown; children?: ReactNode }) => {
      const attrs = variant === 'reader'
        ? readerAnchorAttrs(pickReaderAnchor(node, ['blockquote']))
        : undefined
      if (variant === 'reader') return <blockquote {...attrs}>{children}</blockquote>
      const btn = renderLocateButton(children, {
        meta: { kind: 'blockquote', order: nextLocateRenderOrder() },
        snippetOverride: preferredBlockquoteLocateSnippet(children),
      })
      if (!btn) {
        return (
          <BlockquoteLocateContext.Provider value>
            <blockquote {...attrs}>{children}</blockquote>
          </BlockquoteLocateContext.Provider>
        )
      }
      const tailedChildren = appendTailButtonToContent(children, btn, `blockquote-${locateRenderOrder}`)
      return (
        <BlockquoteLocateContext.Provider value>
          <blockquote {...attrs} className="kb-md-blockquote-tail">{tailedChildren}</blockquote>
        </BlockquoteLocateContext.Provider>
      )
    },
    h1: ({ node, children }: { node?: unknown; children?: ReactNode }) => {
      const attrs = variant === 'reader'
        ? readerAnchorAttrs(pickReaderAnchor(node, ['heading']))
        : undefined
      return <h1 {...attrs}>{children}</h1>
    },
    h2: ({ node, children }: { node?: unknown; children?: ReactNode }) => {
      const attrs = variant === 'reader'
        ? readerAnchorAttrs(pickReaderAnchor(node, ['heading']))
        : undefined
      return <h2 {...attrs}>{children}</h2>
    },
    h3: ({ node, children }: { node?: unknown; children?: ReactNode }) => {
      const attrs = variant === 'reader'
        ? readerAnchorAttrs(pickReaderAnchor(node, ['heading']))
        : undefined
      return <h3 {...attrs}>{children}</h3>
    },
    h4: ({ node, children }: { node?: unknown; children?: ReactNode }) => {
      const attrs = variant === 'reader'
        ? readerAnchorAttrs(pickReaderAnchor(node, ['heading']))
        : undefined
      return <h4 {...attrs}>{children}</h4>
    },
    h5: ({ node, children }: { node?: unknown; children?: ReactNode }) => {
      const attrs = variant === 'reader'
        ? readerAnchorAttrs(pickReaderAnchor(node, ['heading']))
        : undefined
      return <h5 {...attrs}>{children}</h5>
    },
    h6: ({ node, children }: { node?: unknown; children?: ReactNode }) => {
      const attrs = variant === 'reader'
        ? readerAnchorAttrs(pickReaderAnchor(node, ['heading']))
        : undefined
      return <h6 {...attrs}>{children}</h6>
    },
    a: ({ href, children }: { href?: string; children?: ReactNode }) => {
      const key = typeof href === 'string' && href.startsWith('#') ? href.slice(1) : ''
      const pageMatch = variant === 'reader' ? key.match(PAGE_MARKER_HREF_RE) : null
      if (pageMatch) {
        const pageNo = pageMatch[1] || ''
        const pageLabel = (S?.reader_page_marker || 'Page {n}').replace('{n}', pageNo)
        return (
          <span id={`kb-page-${pageNo}`} className="kb-md-page-marker" data-kb-page-marker={pageNo}>
            {pageLabel}
          </span>
        )
      }
      const detail = key ? byAnchor.get(key) : undefined
      if (detail) {
        const useOccurrenceContext = duplicateCitationAnchors.has(detail.anchor)
        const tone = toneBySource?.get(sourceKey(detail))
        const toneStyle: CSSProperties | undefined = tone
          ? ({
              ['--kb-cite-fg' as string]: tone.fg,
              ['--kb-cite-fg-hover' as string]: tone.fgHover,
            } as CSSProperties)
          : undefined
        const readerMarkerText = variant === 'reader'
          ? plainText(children).replace(/\s+/g, '').trim()
          : ''
        const inlineLabel = variant === 'reader' && detail.isInpaper && readerMarkerText
          ? `[${readerMarkerText}]`
          : citationInlineLabel(detail, { includeSource: false })
        return (
          <a
            href={`#${detail.anchor}`}
            className={detail.isInpaper ? 'kb-cite-chip-sysb' : 'kb-cite-chip'}
            style={toneStyle}
            aria-label={citationInlineLabel(detail, { includeSource: false })}
            onClick={(event) => {
              event.preventDefault()
              const occurrenceDetail = withCitationOccurrenceContext(detail, event.currentTarget, useOccurrenceContext)
              const readerOrigin = readerShelfOriginForLink(detail, event.currentTarget)
              const readerDetail = variant === 'reader'
                ? withReaderShelfContext(occurrenceDetail, {
                  origin: readerOrigin,
                  kind: detail.isInpaper ? 'reference' : 'citation',
                  excerpt: citationOccurrenceContextFromLink(event.currentTarget) || occurrenceDetail.citationContext,
                  excerptLabel: S?.shelf_excerpt_head || 'Excerpt',
                  contextSource: readerOrigin === 'reader_references' ? 'reader_references' : 'reader_occurrence',
                })
                : occurrenceDetail
              onCitationClick?.(readerDetail, event)
            }}
            onMouseEnter={(event) => {
              const occurrenceDetail = withCitationOccurrenceContext(detail, event.currentTarget, useOccurrenceContext)
              const readerOrigin = readerShelfOriginForLink(detail, event.currentTarget)
              const readerDetail = variant === 'reader'
                ? withReaderShelfContext(occurrenceDetail, {
                  origin: readerOrigin,
                  kind: detail.isInpaper ? 'reference' : 'citation',
                  excerpt: citationOccurrenceContextFromLink(event.currentTarget) || occurrenceDetail.citationContext,
                  excerptLabel: S?.shelf_excerpt_head || 'Excerpt',
                  contextSource: readerOrigin === 'reader_references' ? 'reader_references' : 'reader_occurrence',
                })
                : occurrenceDetail
              onCitationHover?.(readerDetail, event)
            }}
            onMouseLeave={(event) => {
              onCitationLeave?.(detail, event)
            }}
          >
            {inlineLabel}
          </a>
        )
      }
      return (
        <a href={href} rel="noreferrer" target="_blank">
          {children}
        </a>
      )
    },
    img: ({ node, src, alt }: { node?: unknown; src?: string; alt?: string }) => {
      const resolvedSrc = String(src || '').trim()
      if (!resolvedSrc) return null
      const figureSnippet = String(alt || resolvedSrc.split('/').pop() || 'figure').trim()
      const btn = variant === 'chat'
        ? renderLocateButton(figureSnippet, {
          force: true,
          meta: { kind: 'figure', order: nextLocateRenderOrder() },
        })
        : null
      const pickedAnchor = variant === 'reader' ? pickReaderAnchor(node, ['figure']) : null
      const anchor = normalizeReaderAnchorKind(String(pickedAnchor?.kind || '')) === 'figure'
        ? pickedAnchor
        : null
      const attrs = variant === 'reader'
        ? readerAnchorAttrs(anchor)
        : undefined
      const shelfBtn = renderReaderBlockShelfButton(anchor, figureSnippet, 'figure')
      const imageAlt = String(alt || 'figure')
      const imageTitle = S?.reader_expand_image || 'Expand image'
      const imageNode = (
        <img
          src={resolvedSrc}
          alt={imageAlt}
          className="kb-md-image"
          loading="lazy"
        />
      )
      const shellClass = [
        btn || shelfBtn ? 'kb-md-figure-shell' : '',
        shelfBtn ? 'kb-md-reader-block-action-host' : '',
      ].filter(Boolean).join(' ') || undefined
      return (
        <span className={shellClass} {...attrs}>
          {onImagePreview ? (
            <button
              type="button"
              className="kb-md-image-link kb-md-image-button"
              title={imageTitle}
              onClick={() => onImagePreview(resolvedSrc, imageAlt)}
            >
              {imageNode}
            </button>
          ) : (
            <a href={resolvedSrc} target="_blank" rel="noreferrer" className="kb-md-image-link">
              {imageNode}
            </a>
          )}
          {btn ? <span className="kb-md-figure-tail">{btn}</span> : null}
          {shelfBtn ? <span className="kb-md-reader-block-shelf-tail">{shelfBtn}</span> : null}
        </span>
      )
    },
    p: ({ node, children }: { node?: unknown; children?: ReactNode }) => (
      <BlockquoteLocateContext.Consumer>
        {(insideBlockquote) => {
          const pickedReaderAnchor = variant === 'reader' ? pickReaderAnchor(node, ['paragraph']) : null
          const readerAnchor = normalizeReaderAnchorKind(String(pickedReaderAnchor?.kind || '')) === 'paragraph'
            ? pickedReaderAnchor
            : null
          const attrs = variant === 'reader'
            ? readerAnchorAttrs(readerAnchor)
            : undefined
          const renderOrder = nextLocateRenderOrder()
          const meta = { kind: 'paragraph' as const, order: renderOrder }
          const inline = (variant === 'chat' && inlineTextLocateEnabled && !insideBlockquote)
            ? decorateInlineLocateAnchors(children, meta)
            : { content: children, count: 0, figureRefCount: 0 }
          const content = inline.count > 0 ? inline.content : children
          if (variant !== 'chat') {
            const text = plainText(content).replace(/\s+/g, ' ').trim()
            const isReferenceEntry = looksLikeBibliographyEntryText(text)
            if (!isReferenceEntry) return <p {...attrs}>{content}</p>
            const refDetail = firstCitationDetailInNode(content, byAnchor) || firstReferenceEntryDetail(text, citationByNum)
            const refDetailForRow = refDetail
              ? withReaderShelfContext(refDetail, {
                origin: 'reader_references',
                kind: 'reference',
                excerpt: text,
                excerptLabel: S?.shelf_reference_entry || 'Reference entry',
                contextSource: 'reader_references',
              })
              : null
            const canOpenRef = Boolean(refDetailForRow && onCitationClick)
            const canAddRef = Boolean(refDetailForRow && onCitationAddToShelf)
            const actionTitle = S?.cite_open_context || 'Open reference card'
            const addTitle = S?.reader_add_to_shelf_title || 'Add to research basket'
            const className = canOpenRef
              ? 'kb-md-reference-entry kb-md-reference-entry-clickable'
              : 'kb-md-reference-entry'
            return (
              <p
                {...attrs}
                className={className}
                title={canOpenRef ? actionTitle : undefined}
                role={canOpenRef ? 'button' : undefined}
                tabIndex={canOpenRef ? 0 : undefined}
                onClick={canOpenRef ? (event) => {
                  const target = event.target as HTMLElement | null
                  if (target?.closest('a,button')) return
                  onCitationClick?.(refDetailForRow as CiteDetail, event as unknown as MouseEvent<HTMLElement>)
                } : undefined}
                onKeyDown={canOpenRef ? (event) => {
                  if (event.key !== 'Enter' && event.key !== ' ') return
                  const target = event.target as HTMLElement | null
                  if (target?.closest('a,button')) return
                  event.preventDefault()
                  onCitationClick?.(refDetailForRow as CiteDetail, event as unknown as MouseEvent<HTMLElement>)
                } : undefined}
              >
                <span className="kb-md-reference-entry-body">{content}</span>
                {canAddRef ? (
                  <button
                    type="button"
                    className="kb-md-reader-block-shelf kb-md-reference-entry-action"
                    aria-label={addTitle}
                    title={addTitle}
                    onClick={(event) => {
                      event.preventDefault()
                      event.stopPropagation()
                      onCitationAddToShelf?.(refDetailForRow as CiteDetail)
                    }}
                  >
                    {S?.reader_add_to_shelf || 'Shelf'}
                  </button>
                ) : null}
              </p>
            )
          }
          if (isFigureHostParagraph(content)) {
            return <p {...attrs} className="kb-md-figure-host">{content}</p>
          }
          const figureCaptionSnippet = inline.figureRefCount > 0 ? '' : preferredFigureCaptionSnippet(content)
          if (figureCaptionSnippet) {
            const figureBtn = renderLocateButton(figureCaptionSnippet, {
              force: true,
              meta: { kind: 'figure', order: renderOrder },
              snippetOverride: figureCaptionSnippet,
            })
            if (figureBtn) {
              const tailed = appendTailButtonToContent(content, figureBtn, `figure-caption-${renderOrder}`)
              return <p {...attrs} className="kb-md-figure-caption">{tailed}</p>
            }
          }
          if (inlineTextTailLocateEnabled && !insideBlockquote && inline.count <= 0) {
            const tailBtn = renderLocateButton(content, {
              meta,
            })
            if (tailBtn) {
              const tailed = appendTailButtonToContent(content, tailBtn, `paragraph-${renderOrder}`)
              return <p {...attrs}>{tailed}</p>
            }
          }
          return <p {...attrs}>{content}</p>
        }}
      </BlockquoteLocateContext.Consumer>
    ),
    li: ({ node, children }: { node?: unknown; children?: ReactNode }) => (
      <BlockquoteLocateContext.Consumer>
        {(insideBlockquote) => {
          const attrs = variant === 'reader'
            ? readerAnchorAttrs(pickReaderAnchor(node, ['list_item']))
            : undefined
          const renderOrder = nextLocateRenderOrder()
          const meta = { kind: 'list_item' as const, order: renderOrder }
          const inline = (variant === 'chat' && inlineTextLocateEnabled && !insideBlockquote)
            ? decorateInlineLocateAnchors(children, meta)
            : { content: children, count: 0, figureRefCount: 0 }
          const content = inline.count > 0 ? inline.content : children
          if (variant === 'chat' && inlineTextTailLocateEnabled && !insideBlockquote && inline.count <= 0) {
            const tailBtn = renderLocateButton(content, {
              meta,
            })
            if (tailBtn) {
              const tailed = appendTailButtonToContent(content, tailBtn, `list-item-${renderOrder}`)
              return <li {...attrs}>{tailed}</li>
            }
          }
          return <li {...attrs}>{content}</li>
        }}
      </BlockquoteLocateContext.Consumer>
    ),
    div: (props: ComponentPropsWithoutRef<'div'> & { node?: unknown }) => {
      const { className, children, ...rest } = props || {}
      const cls = String(className || '').trim()
      const displayMath = isDisplayMathClass(cls)
      if (!displayMath) return <div className={cls || undefined} {...(rest as Record<string, unknown>)}>{children}</div>
      if (variant === 'reader') {
        // Display equations are bound at runtime to visible .katex-display nodes.
        // Static line-based binding is too unstable here and can mis-assign them
        // to neighboring paragraph blocks in the browser render path.
        return (
          <div
            className={`${cls || ''} kb-md-equation-block`.trim()}
            data-kb-display-equation="1"
            {...(rest as Record<string, unknown>)}
          >
            <span className="kb-md-equation-inline">{children}</span>
          </div>
        )
      }
      const btn = renderLocateButton(children, {
        force: true,
        meta: { kind: 'equation', order: nextLocateRenderOrder() },
      })
      if (!btn) return <div className={cls || undefined} {...(rest as Record<string, unknown>)}>{children}</div>
      return (
        <div className={`${cls || ''} kb-md-equation-block`.trim()} {...(rest as Record<string, unknown>)}>
          <span className="kb-md-equation-inline">
            {children}
            <span className="kb-md-equation-tail">{btn}</span>
          </span>
        </div>
      )
    },
    span: (props: ComponentPropsWithoutRef<'span'> & { node?: unknown }) => {
      const { className, children, ...rest } = props || {}
      const cls = String(className || '').trim()
      const displayMath = isDisplayMathClass(cls)
      if (!displayMath) return <span className={cls || undefined} {...(rest as Record<string, unknown>)}>{children}</span>
      if (variant === 'reader') {
        return (
          <span
            className={`${cls || ''} kb-md-equation-block`.trim()}
            data-kb-display-equation="1"
            {...(rest as Record<string, unknown>)}
          >
            <span className="kb-md-equation-inline">{children}</span>
          </span>
        )
      }
      const btn = renderLocateButton(children, {
        force: true,
        meta: { kind: 'equation', order: nextLocateRenderOrder() },
      })
      if (!btn) return <span className={cls || undefined} {...(rest as Record<string, unknown>)}>{children}</span>
      return (
        <span className={`${cls || ''} kb-md-equation-block`.trim()} {...(rest as Record<string, unknown>)}>
          <span className="kb-md-equation-inline">
            {children}
            <span className="kb-md-equation-tail">{btn}</span>
          </span>
        </span>
      )
    },
  }
}

export function MarkdownRenderer({
  content,
  citeDetails = [],
  onCitationClick,
  onCitationAddToShelf,
  onCitationHover,
  onCitationLeave,
  onLocateSnippet,
  canLocateSnippet,
  locateTitleResolver,
  locateButtonAttrsResolver,
  inlineLocateTokenPolicy,
  inlineTextLocateEnabled = true,
  inlineTextTailLocateEnabled = false,
  locateSurfacePolicy,
  onReaderBlockAddToShelf,
  variant = 'chat',
  readerAnchors,
  readerBlocks,
}: Props) {
  const S = useT()
  const [previewImage, setPreviewImage] = useState<{ src: string; alt: string } | null>(null)
  const [imagePreviewMode, setImagePreviewMode] = useState<ImagePreviewMode>('fit')
  const [imagePreviewSize, setImagePreviewSize] = useState<ImagePreviewSize | null>(null)
  const rawContent = String(content || '')
  const normalizedContent = variant === 'reader'
    ? normalizeReaderMarkdown(rawContent)
    : normalize(rawContent)
  const citationByNum = buildCitationNumberMap(citeDetails)
  const renderContent = linkifyPlainCitationMarkers(normalizedContent, citeDetails, citationByNum)
  const byAnchor = new Map(citeDetails.map((detail) => [detail.anchor, detail]))
  const duplicateCitationAnchors = new Set<string>()
  if (byAnchor.size > 0) {
    for (const anchor of byAnchor.keys()) {
      const pattern = new RegExp(`#${escapeRegExp(anchor)}(?=[\\s)"']|$)`, 'g')
      const count = renderContent.match(pattern)?.length || 0
      if (count > 1) duplicateCitationAnchors.add(anchor)
    }
  }
  const toneBySource = buildToneMap(citeDetails)
  const readerBlockResolver = useMemo(
    () => (variant === 'reader' ? createReaderBlockResolver(readerBlocks) : null),
    [variant, readerBlocks],
  )
  const readerAnchorAllocator = useMemo(
    () => (variant === 'reader' ? createReaderAnchorAllocator(readerAnchors, readerBlocks) : null),
    [variant, readerAnchors, readerBlocks],
  )
  const components = buildMarkdownComponents(
    byAnchor,
    citationByNum,
    duplicateCitationAnchors,
    onCitationClick,
    onCitationAddToShelf,
    onCitationHover,
    onCitationLeave,
    toneBySource,
    onLocateSnippet,
    canLocateSnippet,
    locateTitleResolver,
    locateButtonAttrsResolver,
    inlineLocateTokenPolicy,
    inlineTextLocateEnabled,
    inlineTextTailLocateEnabled,
    locateSurfacePolicy,
    variant,
    readerAnchorAllocator,
    readerBlockResolver,
    S,
    (src, alt) => {
      setPreviewImage({ src, alt })
      setImagePreviewMode('fit')
      setImagePreviewSize(null)
    },
    onReaderBlockAddToShelf,
  )
  const parsedContract = variant === 'chat' ? parseAnswerContract(renderContent) : null
  const sectionLabelMap: Record<string, string> = {
    conclusion: S.section_conclusion,
    evidence: S.section_evidence,
    limits: S.section_limits,
    next_steps: S.section_next_steps,
  }

  useEffect(() => {
    if (!previewImage) return undefined
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      setPreviewImage(null)
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [previewImage])

  const previewDevicePixelRatio = typeof window !== 'undefined' ? Math.max(1, window.devicePixelRatio || 1) : 1
  const actualPreviewStyle: CSSProperties | undefined = imagePreviewMode === 'actual' && imagePreviewSize
    ? {
      width: `${Math.max(1, Math.round(imagePreviewSize.width / previewDevicePixelRatio))}px`,
      height: 'auto',
      maxWidth: 'none',
      maxHeight: 'none',
    }
    : undefined
  const previewSizeLabel = imagePreviewSize
    ? `${imagePreviewSize.width} x ${imagePreviewSize.height}`
    : ''

  const imagePreviewNode = previewImage && typeof document !== 'undefined'
    ? createPortal(
      <div
        className="kb-md-image-preview-backdrop"
        role="dialog"
        aria-modal="true"
        aria-label={previewImage.alt || (S.reader_expand_image || 'Image preview')}
        onClick={() => setPreviewImage(null)}
      >
        <div className="kb-md-image-preview-shell" onClick={(event) => event.stopPropagation()}>
          <div className="kb-md-image-preview-bar">
            <span className="kb-md-image-preview-size">{previewSizeLabel}</span>
            <div className="kb-md-image-preview-toggle" role="group" aria-label={S.reader_image_zoom_mode || 'Image zoom mode'}>
              <button
                type="button"
                className={imagePreviewMode === 'fit' ? 'active' : undefined}
                onClick={() => setImagePreviewMode('fit')}
              >
                {S.reader_image_fit || 'Fit'}
              </button>
              <button
                type="button"
                className={imagePreviewMode === 'actual' ? 'active' : undefined}
                onClick={() => setImagePreviewMode('actual')}
              >
                {S.reader_image_actual || 'Actual'}
              </button>
            </div>
          </div>
          <button
            type="button"
            className="kb-md-image-preview-close"
            onClick={() => setPreviewImage(null)}
            aria-label={S.shelf_close || 'Close'}
          >
            &times;
          </button>
          <div className={`kb-md-image-preview-stage kb-md-image-preview-stage-${imagePreviewMode}`}>
            <img
              src={previewImage.src}
              alt={previewImage.alt || 'figure'}
              className="kb-md-image-preview-img"
              style={actualPreviewStyle}
              onLoad={(event) => {
                const img = event.currentTarget
                setImagePreviewSize({ width: img.naturalWidth, height: img.naturalHeight })
              }}
            />
          </div>
          {previewImage.alt ? <div className="kb-md-image-preview-caption">{previewImage.alt}</div> : null}
        </div>
      </div>,
      document.body,
    )
    : null

  return (
    <>
      <div className={`kb-markdown prose dark:prose-invert max-w-none min-w-0 text-sm ${variant === 'reader' ? 'kb-markdown-reader' : 'kb-markdown-chat'}`}>
      {parsedContract ? (
        <div className="kb-answer-contract">
          {parsedContract.preamble ? (
            <ReactMarkdown
              remarkPlugins={[remarkGfm, remarkMath]}
              rehypePlugins={[rehypeKatex, rehypeHighlight]}
              components={components}
            >
              {parsedContract.preamble}
            </ReactMarkdown>
          ) : null}
          {parsedContract.sections.map((section) => (
            <section key={section.key} className={`kb-answer-section kb-answer-${section.key}`}>
              <div className="kb-answer-title">{sectionLabelMap[section.key] || section.label}</div>
              <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkMath]}
                rehypePlugins={[rehypeKatex, rehypeHighlight]}
                components={components}
              >
                {section.body}
              </ReactMarkdown>
            </section>
          ))}
        </div>
      ) : (
        <ReactMarkdown
          remarkPlugins={[remarkGfm, remarkMath]}
          rehypePlugins={[rehypeKatex, rehypeHighlight]}
          components={components}
        >
          {renderContent}
        </ReactMarkdown>
      )}
      </div>
      {imagePreviewNode}
    </>
  )
}
