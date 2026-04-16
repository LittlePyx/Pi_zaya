import type { ReaderDocBlock } from '../../../api/references'
import type { ReaderLocateCandidate } from './readerTypes'

export interface ReaderSelectionState {
  text: string
  x: number
  y: number
  startOffset: number
  endOffset: number
  blockId: string
  anchorId: string
  occurrence: number
  readableIndex: number
  documentOccurrence: number
  startReadableIndex: number
  endReadableIndex: number
  canHighlight: boolean
  highlightId: string
}

export interface ReaderHighlightTargetLike {
  text?: string
  startOffset?: number
  endOffset?: number
  blockId?: string
  anchorId?: string
  occurrence?: number
  readableIndex?: number
  documentOccurrence?: number
  startReadableIndex?: number
  endReadableIndex?: number
}

export interface TextNodeCorpusSegment {
  node: Text
  start: number
  end: number
}

export interface DirectTargetResult {
  target: HTMLElement | null
  hintBlock: ReaderDocBlock | null
}

export const READABLE_BLOCK_SELECTOR =
  'p,li,blockquote,pre,code,figcaption,td,th,.katex-display,.katex,[data-kb-anchor-kind="figure"],h1,h2,h3,h4,h5,h6'

export const USER_HIGHLIGHT_NAME = 'kb-reader-user-highlight'
const USER_HIGHLIGHT_STYLE_ID = 'kb-reader-user-highlight-style'

export function normalizeText(input: string) {
  return String(input || '').replace(/\s+/g, ' ').trim().toLowerCase()
}

export function tokenizeText(input: string): string[] {
  const src = normalizeText(input)
  if (!src) return []
  const out: string[] = []
  const latin = src.match(/[a-z0-9]{2,}/g) || []
  out.push(...latin)
  const cjkSeq = src.match(/[\u4e00-\u9fff]{1,}/g) || []
  for (const seq of cjkSeq) {
    if (seq.length <= 2) {
      out.push(seq)
      continue
    }
    for (let i = 0; i < seq.length - 1; i += 1) {
      out.push(seq.slice(i, i + 2))
    }
  }
  return out
}

export function snippetMatchScore(snippet: string, block: string): number {
  const sNorm = normalizeText(snippet)
  const bNorm = normalizeText(block)
  if (!sNorm || !bNorm) return 0
  let bonus = 0
  if (bNorm.includes(sNorm)) bonus += 0.55
  const head = sNorm.slice(0, Math.min(72, sNorm.length))
  if (head && bNorm.includes(head)) bonus += 0.28

  const st = new Set(tokenizeText(sNorm))
  const bt = new Set(tokenizeText(bNorm))
  if (st.size <= 0 || bt.size <= 0) return bonus
  let overlap = 0
  for (const token of st) {
    if (bt.has(token)) overlap += 1
  }
  const sim = overlap / Math.sqrt(st.size * bt.size)
  return sim + bonus
}

export function headingMatchScore(needle: string, heading: string): number {
  const n = normalizeText(needle)
  const h = normalizeText(heading)
  if (!n || !h) return 0
  let score = 0
  if (h.includes(n) || n.includes(h)) score += 0.66
  const nt = new Set(tokenizeText(n))
  const ht = new Set(tokenizeText(h))
  if (nt.size > 0 && ht.size > 0) {
    let overlap = 0
    for (const token of nt) {
      if (ht.has(token)) overlap += 1
    }
    score += overlap / Math.sqrt(nt.size * ht.size)
  }
  return score
}

export function snippetProbeText(text: string): string {
  const src = String(text || '').replace(/\s+/g, ' ').trim()
  if (!src) return ''
  const pieces = src
    .split(/(?<=[\u3002\uff01\uff1f.!;:\uff1b\uff1a])\s+/)
    .map((item) => item.trim())
    .filter(Boolean)
  if (pieces.length <= 0) return src.slice(0, 260)
  return pieces.slice(0, 2).join(' ').slice(0, 320)
}

export function hasFormulaSignal(text: string): boolean {
  const src = String(text || '')
  if (!src) return false
  if (/[=^_]/.test(src)) return true
  if (/\\[a-zA-Z]{2,}/.test(src)) return true
  if (/\$[^$]{1,120}\$/.test(src) || /\$\$[^]{1,260}\$\$/.test(src)) return true
  return false
}

function formulaTokens(text: string): string[] {
  const src = String(text || '')
  if (!src) return []
  const out: string[] = []
  const texCmds = src.match(/\\[a-zA-Z]{2,}/g) || []
  out.push(...texCmds.map((item) => item.toLowerCase()))
  const symbols = src.match(/[A-Za-z](?:_[A-Za-z0-9]+)?(?:\^[A-Za-z0-9]+)?/g) || []
  out.push(...symbols.map((item) => item.toLowerCase()))
  const numbers = src.match(/\b\d{1,4}\b/g) || []
  out.push(...numbers)
  return out
}

export function formulaOverlapScore(a: string, b: string): number {
  const ta = new Set(formulaTokens(a))
  const tb = new Set(formulaTokens(b))
  if (ta.size <= 0 || tb.size <= 0) return 0
  let overlap = 0
  for (const token of ta) {
    if (tb.has(token)) overlap += 1
  }
  return overlap / Math.sqrt(ta.size * tb.size)
}

export function closestReadableBlock(node: Element | null): HTMLElement | null {
  if (!node) return null
  const displayEquation = node.closest('.katex-display')
  if (displayEquation) return displayEquation as HTMLElement
  return node.closest(READABLE_BLOCK_SELECTOR) as HTMLElement | null
}

export function readableBlocks(root: ParentNode | null): HTMLElement[] {
  if (!root) return []
  return Array.from(root.querySelectorAll<HTMLElement>(READABLE_BLOCK_SELECTOR))
}

export function headingCandidates(path: string) {
  const parts = String(path || '')
    .split(' / ')
    .map((item) => item.trim())
    .filter(Boolean)
  if (parts.length <= 1) return parts
  return [...parts.slice().reverse(), parts.join(' / ')]
}

function shortCandidateLabel(input: string, maxLen = 22): string {
  const text = String(input || '').replace(/\s+/g, ' ').trim()
  if (!text) return ''
  if (text.length <= maxLen) return text
  return `${text.slice(0, Math.max(8, maxLen - 3)).trimEnd()}...`
}

function candidateHeadingTail(path: string, sourceTitle = ''): string {
  const parts = String(path || '')
    .split(' / ')
    .map((item) => item.trim())
    .filter(Boolean)
  if (parts.length <= 0) return ''
  const titleNorm = normalizeText(sourceTitle)
  const filtered = parts.filter((part, idx) => !(idx === 0 && titleNorm && normalizeText(part) === titleNorm))
  const picked = (filtered.length > 0 ? filtered : parts).slice(-2)
  return picked.join(' / ')
}

function candidateAnchorLabel(anchorKind: string, anchorNumber: number): string {
  const kind = String(anchorKind || '').trim().toLowerCase()
  const num = Number.isFinite(Number(anchorNumber)) ? Math.floor(Number(anchorNumber)) : 0
  if (kind === 'equation') return num > 0 ? `Eq. (${num})` : 'Equation'
  if (kind === 'inline_formula') return num > 0 ? `Formula (${num})` : 'Formula'
  if (kind === 'figure') return num > 0 ? `Figure ${num}` : 'Figure'
  if (kind === 'quote') return 'Quote'
  if (kind === 'blockquote') return 'Quoted block'
  if (kind === 'heading') return 'Heading'
  return ''
}

export function candidateDisplayLabel(
  item: ReaderLocateCandidate,
  sourceTitle = '',
): string {
  const heading = candidateHeadingTail(String(item.headingPath || ''), sourceTitle)
  const anchor = candidateAnchorLabel(String(item.anchorKind || ''), Number(item.anchorNumber || 0))
  const snippet = shortCandidateLabel(String(item.highlightSnippet || item.snippet || ''), 42)
  if (heading && anchor) return `${heading} / ${anchor}`
  if (heading) return heading
  if (anchor && snippet && normalizeText(anchor) !== normalizeText(snippet)) return `${anchor} / ${snippet}`
  return anchor || snippet || ''
}

export function candidateVisibilityKey(
  item: ReaderLocateCandidate,
  sourceTitle = '',
): string {
  const heading = candidateHeadingTail(String(item.headingPath || ''), sourceTitle)
  const anchor = candidateAnchorLabel(String(item.anchorKind || ''), Number(item.anchorNumber || 0))
  const snippet = shortCandidateLabel(String(item.highlightSnippet || item.snippet || ''), 42)
  const primary = [normalizeText(heading), normalizeText(anchor)].filter(Boolean)
  if (primary.length > 0) return primary.join('::')
  return normalizeText(snippet)
}

export function candidateIdentityKey(item: Partial<ReaderLocateCandidate> | null | undefined): string {
  return [
    String(item?.blockId || '').trim().toLowerCase(),
    String(item?.anchorId || '').trim().toLowerCase(),
    String(item?.anchorKind || '').trim().toLowerCase(),
    Number.isFinite(Number(item?.anchorNumber || 0)) ? Math.floor(Number(item?.anchorNumber || 0)) : 0,
    String(item?.headingPath || '').trim().toLowerCase(),
    String(item?.highlightSnippet || '').trim().toLowerCase().slice(0, 180),
    String(item?.snippet || '').trim().toLowerCase().slice(0, 180),
  ].join('::')
}

export function compactLocateHintLabel(input: string): string {
  const text = String(input || '').trim()
  if (!text) return ''
  const rules: Array<[RegExp, string]> = [
    [/^Exact source phrase match\.?$/i, 'Exact phrase'],
    [/^Exact figure block match\.?$/i, 'Figure exact'],
    [/^Figure block matched\.?$/i, 'Figure block'],
    [/^Equation block matched\.?$/i, 'Equation block'],
    [/^Exact evidence block not found\. Strict locate stopped before heading fallback\.?$/i, 'Strict stopped'],
    [/^Exact evidence block not found\. Strict locate stopped before fuzzy fallback\.?$/i, 'Strict stopped'],
    [/^Inline formula match\.?$/i, 'Inline formula'],
    [/^Neighbor inline formula match\.?$/i, 'Neighbor formula'],
    [/^Explanation block matched, but inline formula was not found\.?$/i, 'Formula missing'],
    [/^Neighbor evidence block matched, but exact inline phrase was not found\.?$/i, 'Neighbor block'],
    [/^Evidence block matched, but exact inline phrase was not found\.?$/i, 'Block only'],
    [/^Evidence block matched\.?$/i, 'Evidence block'],
    [/^Heading-level fallback matched\.?$/i, 'Heading match'],
    [/^Located by heading\.?$/i, 'Heading match'],
  ]
  for (const [pattern, label] of rules) {
    if (pattern.test(text)) return label
  }
  return shortCandidateLabel(text.replace(/\.$/, ''), 28)
}

export function extractEquationNumbers(text: string): number[] {
  const src = String(text || '')
  if (!src) return []
  const out: number[] = []
  const seen = new Set<number>()
  const push = (raw: string) => {
    const num = Number(raw)
    if (!Number.isFinite(num) || num <= 0) return
    const n = Math.floor(num)
    if (seen.has(n)) return
    seen.add(n)
    out.push(n)
  }
  for (const m of src.matchAll(/\b(?:eq|equation)\s*[#(]?\s*(\d{1,4})\s*[)]?/gi)) {
    push(String(m[1] || ''))
  }
  for (const m of src.matchAll(/\((\d{1,4})\)/g)) {
    push(String(m[1] || ''))
  }
  return out
}

export function equationNumberMatchScore(blockText: string, numbers: number[]): number {
  if (numbers.length <= 0) return 0
  const text = normalizeText(blockText)
  if (!text) return 0
  let best = 0
  for (const num of numbers) {
    if (new RegExp(`\\(\\s*${num}\\s*\\)`).test(text)) best = Math.max(best, 1.0)
    if (new RegExp(`\\[\\s*${num}\\s*\\]`).test(text)) best = Math.max(best, 0.92)
    if (new RegExp(`\\beq(?:uation)?\\s*\\.?\\s*${num}\\b`, 'i').test(text)) best = Math.max(best, 0.9)
    if (new RegExp(`\\b閸忣剙绱\\s*${num}\\b`).test(text)) best = Math.max(best, 0.92)
  }
  return best
}

export function extractFigureNumbers(text: string): number[] {
  const src = String(text || '')
  if (!src) return []
  const out: number[] = []
  const seen = new Set<number>()
  const push = (raw: string) => {
    const num = Number(raw)
    if (!Number.isFinite(num) || num <= 0) return
    const n = Math.floor(num)
    if (seen.has(n)) return
    seen.add(n)
    out.push(n)
  }
  for (const m of src.matchAll(/\bfig(?:ure)?\.?\s*#?\s*(\d{1,4})\b/gi)) {
    push(String(m[1] || ''))
  }
  for (const m of src.matchAll(/閸ョ斗s*(\d{1,4})\b/g)) {
    push(String(m[1] || ''))
  }
  return out
}

export function extractQuotedSpans(text: string, minLen = 12): string[] {
  const src = String(text || '').replace(/\s+/g, ' ').trim()
  if (!src) return []
  const out: string[] = []
  const seen = new Set<string>()
  const push = (raw: string) => {
    const item = String(raw || '').replace(/\s+/g, ' ').trim()
    if (!item || item.length < minLen) return
    const key = normalizeText(item)
    if (!key || seen.has(key)) return
    seen.add(key)
    out.push(item)
  }
  for (const pattern of [
    /["\u201C\u201D]\s*([^"\u201C\u201D]{6,360}?)\s*["\u201C\u201D]/g,
    /[\u2018\u2019']\s*([^\u2018\u2019']{6,320}?)\s*[\u2018\u2019']/g,
    /[\u300C\u300D\u300E\u300F\u300A\u300B]\s*([^\u300C\u300D\u300E\u300F\u300A\u300B]{6,360}?)\s*[\u300D\u300F\u300B]/g,
  ]) {
    for (const m of src.matchAll(pattern)) {
      push(String(m[1] || ''))
      if (out.length >= 6) return out
    }
  }
  return out
}

function normalizeWithMap(text: string): { norm: string; map: number[] } {
  const src = String(text || '')
  if (!src) return { norm: '', map: [] }
  const chars: string[] = []
  const map: number[] = []
  let prevSpace = true
  for (let idx = 0; idx < src.length; idx += 1) {
    const ch = src[idx]
    if (/\s/.test(ch)) {
      if (!prevSpace) {
        chars.push(' ')
        map.push(idx)
      }
      prevSpace = true
      continue
    }
    chars.push(ch.toLowerCase())
    map.push(idx)
    prevSpace = false
  }
  while (chars.length > 0 && chars[0] === ' ') {
    chars.shift()
    map.shift()
  }
  while (chars.length > 0 && chars[chars.length - 1] === ' ') {
    chars.pop()
    map.pop()
  }
  return { norm: chars.join(''), map }
}

export function clearReaderInlineHits(root: HTMLElement | null) {
  if (!root) return
  const hits = Array.from(root.querySelectorAll<HTMLElement>('.kb-reader-inline-hit'))
  for (const hit of hits) {
    const parent = hit.parentNode
    if (!parent) continue
    while (hit.firstChild) parent.insertBefore(hit.firstChild, hit)
    parent.removeChild(hit)
    parent.normalize()
  }
}

export function clearReaderUserHighlights(root: HTMLElement | null) {
  const cssRegistry = (globalThis as { CSS?: { highlights?: { delete?: (name: string) => void } } }).CSS?.highlights
  cssRegistry?.delete?.(USER_HIGHLIGHT_NAME)
  if (!root) return
  const hits = Array.from(root.querySelectorAll<HTMLElement>('.kb-reader-user-highlight'))
  for (const hit of hits) {
    const parent = hit.parentNode
    if (!parent) continue
    while (hit.firstChild) parent.insertBefore(hit.firstChild, hit)
    parent.removeChild(hit)
    parent.normalize()
  }
}

export function clearReaderFocusClasses(root: HTMLElement | null) {
  if (!root) return
  root.querySelectorAll<HTMLElement>('.kb-reader-focus, .kb-reader-focus-secondary')
    .forEach((node) => {
      node.classList.remove('kb-reader-focus')
      node.classList.remove('kb-reader-focus-secondary')
    })
}

export function buildTextNodeCorpus(container: HTMLElement): { raw: string; nodes: TextNodeCorpusSegment[] } {
  const nodes: TextNodeCorpusSegment[] = []
  let raw = ''
  const walker = document.createTreeWalker(
    container,
    NodeFilter.SHOW_TEXT,
    {
      acceptNode(node) {
        const parent = node.parentElement
        if (!parent) return NodeFilter.FILTER_REJECT
        if (parent.closest('.kb-md-locate-inline-btn, .kb-cite-chip, script, style')) {
          return NodeFilter.FILTER_REJECT
        }
        const text = String(node.textContent || '')
        if (!text.trim()) return NodeFilter.FILTER_REJECT
        return NodeFilter.FILTER_ACCEPT
      },
    },
  )
  let current = walker.nextNode()
  while (current) {
    const textNode = current as Text
    const text = String(textNode.textContent || '')
    nodes.push({ node: textNode, start: raw.length, end: raw.length + text.length })
    raw += text
    current = walker.nextNode()
  }
  return { raw, nodes }
}

export function rawOffsetToDomPoint(
  segments: TextNodeCorpusSegment[],
  offset: number,
): { node: Text; offset: number } | null {
  for (const seg of segments) {
    if (offset < seg.start || offset > seg.end) continue
    return {
      node: seg.node,
      offset: Math.max(0, Math.min(seg.node.textContent?.length || 0, offset - seg.start)),
    }
  }
  const last = segments[segments.length - 1] || null
  if (!last) return null
  return {
    node: last.node,
    offset: last.node.textContent?.length || 0,
  }
}

function firstTextNodeWithin(node: Node | null): Text | null {
  if (!node) return null
  if (node.nodeType === Node.TEXT_NODE) return node as Text
  const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT)
  return walker.nextNode() as Text | null
}

function lastTextNodeWithin(node: Node | null): Text | null {
  if (!node) return null
  if (node.nodeType === Node.TEXT_NODE) return node as Text
  const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT)
  let current = walker.nextNode() as Text | null
  let last: Text | null = null
  while (current) {
    last = current
    current = walker.nextNode() as Text | null
  }
  return last
}

function domPointToRawOffset(
  segments: TextNodeCorpusSegment[],
  node: Node,
  offset: number,
  bias: 'start' | 'end',
): number | null {
  const last = segments[segments.length - 1] || null
  if (node.nodeType === Node.TEXT_NODE) {
    const textNode = node as Text
    const seg = segments.find((item) => item.node === textNode) || null
    if (!seg) return null
    return Math.max(seg.start, Math.min(seg.end, seg.start + Math.max(0, offset)))
  }
  const children = Array.from(node.childNodes)
  if (bias === 'start') {
    for (let idx = Math.max(0, offset); idx < children.length; idx += 1) {
      const textNode = firstTextNodeWithin(children[idx])
      if (!textNode) continue
      const seg = segments.find((item) => item.node === textNode) || null
      if (seg) return seg.start
    }
    for (let idx = Math.min(children.length - 1, offset - 1); idx >= 0; idx -= 1) {
      const textNode = lastTextNodeWithin(children[idx])
      if (!textNode) continue
      const seg = segments.find((item) => item.node === textNode) || null
      if (seg) return seg.end
    }
    return 0
  }
  for (let idx = Math.min(children.length - 1, offset - 1); idx >= 0; idx -= 1) {
    const textNode = lastTextNodeWithin(children[idx])
    if (!textNode) continue
    const seg = segments.find((item) => item.node === textNode) || null
    if (seg) return seg.end
  }
  for (let idx = Math.max(0, offset); idx < children.length; idx += 1) {
    const textNode = firstTextNodeWithin(children[idx])
    if (!textNode) continue
    const seg = segments.find((item) => item.node === textNode) || null
    if (seg) return seg.start
  }
  return last ? last.end : 0
}

export function rawOffsetsFromRange(container: HTMLElement, range: Range): { startOffset: number; endOffset: number } | null {
  const corpus = buildTextNodeCorpus(container)
  if (corpus.nodes.length <= 0) return null
  const startOffset = domPointToRawOffset(corpus.nodes, range.startContainer, range.startOffset, 'start')
  const endOffset = domPointToRawOffset(corpus.nodes, range.endContainer, range.endOffset, 'end')
  if (!Number.isFinite(startOffset) || !Number.isFinite(endOffset)) return null
  const start = Math.max(0, Math.min(Number(startOffset), Number(endOffset)))
  const end = Math.max(start, Math.max(Number(startOffset), Number(endOffset)))
  if (end <= start) return null
  return { startOffset: start, endOffset: end }
}

export function supportsCustomHighlights(): boolean {
  const scope = globalThis as {
    CSS?: { highlights?: { set?: (...args: unknown[]) => unknown; delete?: (...args: unknown[]) => unknown } }
    Highlight?: new (...ranges: Range[]) => unknown
  }
  return Boolean(scope.CSS?.highlights && typeof scope.Highlight === 'function')
}

export function ensureReaderCustomHighlightStyle() {
  if (typeof document === 'undefined') return
  if (document.getElementById(USER_HIGHLIGHT_STYLE_ID)) return
  const style = document.createElement('style')
  style.id = USER_HIGHLIGHT_STYLE_ID
  style.textContent = [
    `::highlight(${USER_HIGHLIGHT_NAME}) {`,
    '  background-color: color-mix(in srgb, #ffe08a 74%, white 26%);',
    '  color: inherit;',
    '}',
  ].join('\n')
  document.head.appendChild(style)
}

export function highlightRawRangeInCorpus(
  segments: TextNodeCorpusSegment[],
  startOffset: number,
  endOffset: number,
  className: string,
  attributes?: Record<string, string>,
) {
  for (let idx = segments.length - 1; idx >= 0; idx -= 1) {
    const seg = segments[idx]
    if (endOffset <= seg.start || startOffset >= seg.end) continue
    const localStart = Math.max(startOffset, seg.start) - seg.start
    const localEnd = Math.min(endOffset, seg.end) - seg.start
    if (localEnd <= localStart) continue
    let target = seg.node
    if (localEnd < seg.node.textContent!.length) {
      target.splitText(localEnd)
    }
    if (localStart > 0) {
      target = target.splitText(localStart)
    }
    const wrapper = document.createElement('mark')
    wrapper.className = className
    Object.entries(attributes || {}).forEach(([key, value]) => wrapper.setAttribute(key, value))
    target.parentNode?.insertBefore(wrapper, target)
    wrapper.appendChild(target)
  }
}

function wrapExactTextMatchInContainer(
  container: HTMLElement,
  query: string,
  opts: {
    className: string
    minLength?: number
    occurrence?: number
    attributes?: Record<string, string>
  },
): HTMLElement | null {
  const probe = String(query || '').replace(/\s+/g, ' ').trim()
  const minLength = Number.isFinite(Number(opts.minLength)) ? Math.max(1, Math.floor(Number(opts.minLength))) : 8
  const targetOccurrence = Number.isFinite(Number(opts.occurrence)) ? Math.max(0, Math.floor(Number(opts.occurrence))) : 0
  if (!probe || probe.length < minLength) return null
  const corpus = buildTextNodeCorpus(container)
  if (!corpus.raw) return null
  const normCorpus = normalizeWithMap(corpus.raw)
  const normQuery = normalizeWithMap(probe).norm
  if (!normCorpus.norm || !normQuery) return null
  let hitAt = -1
  let occurrence = 0
  let searchAt = 0
  while (searchAt <= normCorpus.norm.length) {
    const nextHit = normCorpus.norm.indexOf(normQuery, searchAt)
    if (nextHit < 0) break
    if (occurrence === targetOccurrence) {
      hitAt = nextHit
      break
    }
    occurrence += 1
    searchAt = nextHit + Math.max(1, normQuery.length)
  }
  if (hitAt < 0) return null
  const startRaw = normCorpus.map[hitAt]
  const endRaw = (normCorpus.map[hitAt + normQuery.length - 1] || startRaw) + 1
  const start = rawOffsetToDomPoint(corpus.nodes, startRaw)
  const end = rawOffsetToDomPoint(corpus.nodes, endRaw)
  if (!start || !end) return null
  const range = document.createRange()
  try {
    range.setStart(start.node, start.offset)
    range.setEnd(end.node, end.offset)
    if (range.collapsed) return null
    const wrapper = document.createElement('mark')
    wrapper.className = opts.className
    Object.entries(opts.attributes || {}).forEach(([key, value]) => {
      wrapper.setAttribute(key, value)
    })
    const frag = range.extractContents()
    wrapper.appendChild(frag)
    range.insertNode(wrapper)
    return wrapper
  } catch {
    return null
  }
}

export function highlightExactTextInContainer(container: HTMLElement, query: string): HTMLElement | null {
  return wrapExactTextMatchInContainer(container, query, {
    className: 'kb-reader-inline-hit',
    minLength: 8,
  })
}

export function highlightSessionTextInContainer(
  container: HTMLElement,
  query: string,
  highlightId: string,
  occurrence = 0,
): HTMLElement | null {
  return wrapExactTextMatchInContainer(container, query, {
    className: 'kb-reader-user-highlight',
    minLength: 2,
    occurrence,
    attributes: {
      'data-kb-session-highlight-id': highlightId,
    },
  })
}

function findSelectionReadableBlock(node: Node | null): HTMLElement | null {
  if (!node) return null
  const base = node instanceof HTMLElement ? node : node.parentElement
  if (!base) return null
  return closestReadableBlock(base) || base.closest('[data-kb-block-id], [data-kb-anchor-id]') as HTMLElement | null
}

function locateSelectionOccurrence(
  root: HTMLElement,
  block: HTMLElement,
  range: Range,
  text: string,
): { blockId: string; anchorId: string; occurrence: number; readableIndex: number } | null {
  const blockId = String(block.getAttribute('data-kb-block-id') || '').trim()
  const anchorId = String(block.getAttribute('data-kb-anchor-id') || '').trim()
  const readableIndex = readableBlocks(root).findIndex((item) => item === block)
  if (!blockId && !anchorId && readableIndex < 0) return null
  const corpus = buildTextNodeCorpus(block)
  const normQuery = normalizeWithMap(text).norm
  if (!corpus.raw || !normQuery) return null
  const preRange = document.createRange()
  try {
    preRange.selectNodeContents(block)
    preRange.setEnd(range.startContainer, range.startOffset)
  } catch {
    return { blockId, anchorId, occurrence: 0, readableIndex }
  }
  const selectionStartRaw = preRange.toString().length
  const normCorpus = normalizeWithMap(corpus.raw)
  if (!normCorpus.norm) return { blockId, anchorId, occurrence: 0, readableIndex }
  let occurrence = 0
  let bestOccurrence = 0
  let bestDistance = Number.POSITIVE_INFINITY
  let searchAt = 0
  while (searchAt <= normCorpus.norm.length) {
    const hitAt = normCorpus.norm.indexOf(normQuery, searchAt)
    if (hitAt < 0) break
    const hitRaw = normCorpus.map[hitAt] ?? 0
    const distance = Math.abs(hitRaw - selectionStartRaw)
    if (distance < bestDistance) {
      bestDistance = distance
      bestOccurrence = occurrence
    }
    occurrence += 1
    searchAt = hitAt + Math.max(1, normQuery.length)
  }
  return { blockId, anchorId, occurrence: bestOccurrence, readableIndex }
}

function locateTextOccurrenceInContainer(
  container: HTMLElement,
  range: Range,
  text: string,
): number {
  const corpus = buildTextNodeCorpus(container)
  const normQuery = normalizeWithMap(text).norm
  if (!corpus.raw || !normQuery) return -1
  const preRange = document.createRange()
  try {
    preRange.selectNodeContents(container)
    preRange.setEnd(range.startContainer, range.startOffset)
  } catch {
    return 0
  }
  const selectionStartRaw = preRange.toString().length
  const normCorpus = normalizeWithMap(corpus.raw)
  if (!normCorpus.norm) return 0
  let occurrence = 0
  let bestOccurrence = 0
  let bestDistance = Number.POSITIVE_INFINITY
  let searchAt = 0
  while (searchAt <= normCorpus.norm.length) {
    const hitAt = normCorpus.norm.indexOf(normQuery, searchAt)
    if (hitAt < 0) break
    const hitRaw = normCorpus.map[hitAt] ?? 0
    const distance = Math.abs(hitRaw - selectionStartRaw)
    if (distance < bestDistance) {
      bestDistance = distance
      bestOccurrence = occurrence
    }
    occurrence += 1
    searchAt = hitAt + Math.max(1, normQuery.length)
  }
  return bestOccurrence
}

export function sameHighlightTarget(
  left: ReaderHighlightTargetLike,
  right: ReaderHighlightTargetLike,
): boolean {
  const leftStart = Number(left.startOffset ?? -1)
  const leftEnd = Number(left.endOffset ?? -1)
  const rightStart = Number(right.startOffset ?? -1)
  const rightEnd = Number(right.endOffset ?? -1)
  if (leftStart >= 0 && leftEnd > leftStart && rightStart >= 0 && rightEnd > rightStart) {
    return leftStart === rightStart && leftEnd === rightEnd
  }
  return normalizeText(left.text || '') === normalizeText(right.text || '')
    && normalizeText(left.blockId || '') === normalizeText(right.blockId || '')
    && normalizeText(left.anchorId || '') === normalizeText(right.anchorId || '')
    && Math.max(0, Math.floor(Number(left.occurrence || 0))) === Math.max(0, Math.floor(Number(right.occurrence || 0)))
    && Math.max(-1, Math.floor(Number(left.readableIndex ?? -1))) === Math.max(-1, Math.floor(Number(right.readableIndex ?? -1)))
    && Math.max(-1, Math.floor(Number(left.documentOccurrence ?? -1))) === Math.max(-1, Math.floor(Number(right.documentOccurrence ?? -1)))
    && Math.max(-1, Math.floor(Number(left.startReadableIndex ?? -1))) === Math.max(-1, Math.floor(Number(right.startReadableIndex ?? -1)))
    && Math.max(-1, Math.floor(Number(left.endReadableIndex ?? -1))) === Math.max(-1, Math.floor(Number(right.endReadableIndex ?? -1)))
}

export function createSessionHighlightId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `hl-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
}

export function buildHighlightQueries(text: string, opts?: { anchorKind?: string; anchorNumber?: number }): string[] {
  const raw = String(text || '').replace(/\s+/g, ' ').trim()
  if (!raw) return []
  const out: string[] = []
  const seen = new Set<string>()
  const push = (value: string) => {
    const item = String(value || '').replace(/\s+/g, ' ').trim()
    if (!item) return
    const key = normalizeText(item)
    if (!key || seen.has(key)) return
    seen.add(key)
    out.push(item)
  }
  const quoted = extractQuotedSpans(raw, 10).sort((a, b) => b.length - a.length)
  quoted.forEach(push)
  push(raw)
  push(snippetProbeText(raw))
  const anchorKind = String(opts?.anchorKind || '').trim().toLowerCase()
  const anchorNumber = Number.isFinite(Number(opts?.anchorNumber)) ? Math.floor(Number(opts?.anchorNumber)) : 0
  if (anchorKind === 'equation' && anchorNumber > 0) {
    push(`equation ${anchorNumber}`)
    push(`閸忣剙绱?${anchorNumber}`)
    push(`閸忣剙绱?${anchorNumber})`)
  }
  if (anchorKind === 'figure' && anchorNumber > 0) {
    push(`Figure ${anchorNumber}`)
    push(`Fig. ${anchorNumber}`)
    push(`閸?{anchorNumber}`)
  }
  return out
}

export function nearbyReadableBlocks(root: HTMLElement, target: HTMLElement, maxDistance = 2): HTMLElement[] {
  const blocks = Array.from(root.querySelectorAll<HTMLElement>('p,li,blockquote,pre,code,figcaption,td,th,.katex-display,[data-kb-anchor-kind="figure"]'))
  const idx = blocks.findIndex((item) => item === target)
  if (idx < 0) return []
  const out: HTMLElement[] = []
  for (let cursor = Math.max(0, idx - maxDistance); cursor <= Math.min(blocks.length - 1, idx + maxDistance); cursor += 1) {
    if (cursor === idx) continue
    out.push(blocks[cursor])
  }
  return out
}

export function scrollReaderTargetIntoView(root: HTMLElement, target: HTMLElement) {
  const rootRect = root.getBoundingClientRect()
  const targetRect = target.getBoundingClientRect()
  const maxScrollTop = Math.max(0, root.scrollHeight - root.clientHeight)
  if (rootRect.height > 0 && targetRect.height > 0) {
    const topPadding = Math.max(20, Math.min(52, root.clientHeight * 0.08))
    const bottomPadding = Math.max(28, Math.min(96, root.clientHeight * 0.18))
    const visibleTop = rootRect.top + topPadding
    const visibleBottom = rootRect.bottom - bottomPadding
    let nextTop = root.scrollTop

    if (targetRect.top < visibleTop) {
      nextTop = root.scrollTop + (targetRect.top - visibleTop)
    } else if (targetRect.bottom > visibleBottom) {
      nextTop = root.scrollTop + (targetRect.bottom - visibleBottom)
    } else {
      return
    }

    root.scrollTo({
      top: Math.max(0, Math.min(maxScrollTop, nextTop)),
      behavior: 'auto',
    })
    return
  }

  let offsetTop = 0
  let cursor: HTMLElement | null = target
  while (cursor && cursor !== root) {
    offsetTop += cursor.offsetTop
    cursor = cursor.offsetParent as HTMLElement | null
  }
  root.scrollTo({
    top: Math.max(0, Math.min(maxScrollTop, offsetTop - 32)),
    behavior: 'auto',
  })
}

export function resolveSessionHighlightScrollTarget(
  root: HTMLElement,
  readerBlocks: ReaderDocBlock[],
  highlight: ReaderHighlightTargetLike & { id?: string },
): HTMLElement | null {
  const highlightId = String(highlight?.id || '').trim()
  if (highlightId) {
    const marked = root.querySelector<HTMLElement>(`[data-kb-session-highlight-id="${CSS.escape(highlightId)}"]`)
    if (marked) return closestReadableBlock(marked) || marked
  }

  const startOffset = Number(highlight?.startOffset ?? -1)
  const endOffset = Number(highlight?.endOffset ?? -1)
  if (startOffset >= 0 && endOffset > startOffset) {
    const corpus = buildTextNodeCorpus(root)
    const start = rawOffsetToDomPoint(corpus.nodes, startOffset)
    const base = start?.node?.parentElement || null
    const target = closestReadableBlock(base) || base
    if (target) return target
  }

  const blockId = String(highlight?.blockId || '').trim()
  const anchorId = String(highlight?.anchorId || '').trim()
  const resolved = resolveDirectTargetNode(root, readerBlocks, { blockId, anchorId })
  const direct = closestReadableBlock(resolved.target) || resolved.target
  if (direct) return direct

  const readableIndex = Number.isFinite(Number(highlight?.readableIndex ?? -1))
    ? Math.max(-1, Math.floor(Number(highlight?.readableIndex ?? -1)))
    : -1
  if (readableIndex >= 0) {
    return readableBlocks(root)[readableIndex] || null
  }

  const startReadableIndex = Number.isFinite(Number(highlight?.startReadableIndex ?? -1))
    ? Math.max(-1, Math.floor(Number(highlight?.startReadableIndex ?? -1)))
    : -1
  if (startReadableIndex >= 0) {
    return readableBlocks(root)[startReadableIndex] || null
  }

  return null
}

export function resolveStickyHighlightTarget(
  root: HTMLElement,
  readerBlocks: ReaderDocBlock[],
  sticky: {
    blockId: string
    anchorId: string
    anchorKind: string
    anchorNumber: number
    headingPath: string
    highlightSeed: string
    highlightQueries: string[]
    relatedBlockIds: string[]
    strictLocate: boolean
  },
): HTMLElement | null {
  const direct = resolveDirectTargetNode(root, readerBlocks, {
    blockId: sticky.blockId,
    anchorId: sticky.anchorId,
    anchorKind: sticky.anchorKind,
  })
  if (direct.target) return direct.target
  if (sticky.strictLocate) return null

  const seed = String(sticky.highlightSeed || '').trim()
  if (!seed) return null

  if (sticky.anchorKind === 'equation') {
    const eqNumbersAll = [
      ...extractEquationNumbers(seed),
      ...(Number.isFinite(Number(sticky.anchorNumber)) && Number(sticky.anchorNumber) > 0
        ? [Math.floor(Number(sticky.anchorNumber))]
        : []),
    ]
    const eqNumbers = Array.from(new Set(eqNumbersAll.filter((item) => Number.isFinite(item) && item > 0)))
    const blocks = visibleEquationBlocks(root)
    let best: HTMLElement | null = null
    let bestScore = 0
    for (const block of blocks) {
      const text = String(block.textContent || '')
      let score = 0.5 * formulaOverlapScore(seed, text)
      score += 0.35 * snippetMatchScore(seed, text)
      if (eqNumbers.length > 0) score += 0.65 * equationNumberMatchScore(text, eqNumbers)
      if (score > bestScore) {
        best = block
        bestScore = score
      }
    }
    return bestScore >= 0.18 ? best : null
  }

  const blocks = Array.from(root.querySelectorAll<HTMLElement>('p,li,blockquote,pre,code,figcaption,td,th'))
  let best: HTMLElement | null = null
  let bestScore = 0
  for (const block of blocks) {
    const text = String(block.textContent || '')
    const score = snippetMatchScore(seed, text)
    if (score > bestScore) {
      best = block
      bestScore = score
    }
  }
  return bestScore >= 0.12 ? best : null
}

export function visibleEquationBlocks(root: HTMLElement): HTMLElement[] {
  const out: HTMLElement[] = []
  const seen = new Set<HTMLElement>()
  const push = (node: HTMLElement | null) => {
    if (!node || seen.has(node)) return
    seen.add(node)
    out.push(node)
  }
  Array.from(root.querySelectorAll<HTMLElement>('[data-kb-anchor-kind="equation"]')).forEach((node) => {
    push(closestReadableBlock(node) || node)
  })
  Array.from(root.querySelectorAll<HTMLElement>('.katex-display')).forEach((node) => {
    push(closestReadableBlock(node) || node)
  })
  return out
}

function clearVisibleEquationBindings(root: HTMLElement) {
  Array.from(root.querySelectorAll<HTMLElement>('[data-kb-visible-equation-bound="1"]')).forEach((node) => {
    node.removeAttribute('data-kb-block-id')
    node.removeAttribute('data-kb-anchor-id')
    node.removeAttribute('data-kb-anchor-kind')
    node.removeAttribute('data-kb-anchor-number')
    node.removeAttribute('data-kb-visible-equation-bound')
  })
}

export function orderedEquationReaderBlocks(readerBlocks: ReaderDocBlock[]): ReaderDocBlock[] {
  return [...readerBlocks]
    .filter((block) => String(block?.kind || '').trim().toLowerCase() === 'equation')
    .sort((a, b) => {
      const orderA = Number(a?.order_index || 0)
      const orderB = Number(b?.order_index || 0)
      if (orderA > 0 || orderB > 0) {
        if (orderA !== orderB) return orderA - orderB
      }
      const lineA = Number(a?.line_start || 0)
      const lineB = Number(b?.line_start || 0)
      if (lineA !== lineB) return lineA - lineB
      return String(a?.block_id || '').localeCompare(String(b?.block_id || ''))
    })
}

export function bindVisibleEquationAnchors(root: HTMLElement, readerBlocks: ReaderDocBlock[]): number {
  if (!root || !Array.isArray(readerBlocks) || readerBlocks.length <= 0) return 0
  const equationBlocks = orderedEquationReaderBlocks(readerBlocks)
  if (equationBlocks.length <= 0) return 0
  const visibleNodes = Array.from(root.querySelectorAll<HTMLElement>('.katex-display'))
  if (visibleNodes.length <= 0) return 0

  clearVisibleEquationBindings(root)

  const bind = (node: HTMLElement, block: ReaderDocBlock) => {
    const blockId = String(block?.block_id || '').trim()
    const anchorId = String(block?.anchor_id || '').trim()
    if (!blockId || !anchorId) return
    node.setAttribute('data-kb-block-id', blockId)
    node.setAttribute('data-kb-anchor-id', anchorId)
    node.setAttribute('data-kb-anchor-kind', 'equation')
    node.setAttribute('data-kb-visible-equation-bound', '1')
    const number = Number(block?.number || 0)
    if (Number.isFinite(number) && number > 0) {
      node.setAttribute('data-kb-anchor-number', String(Math.floor(number)))
    } else {
      node.removeAttribute('data-kb-anchor-number')
    }
  }

  const nodeNumbers = visibleNodes.map((node) => Array.from(new Set(extractEquationNumbers(node.textContent || ''))))
  const usedNodeIdx = new Set<number>()
  const usedBlockId = new Set<string>()

  for (const block of equationBlocks) {
    const blockId = String(block?.block_id || '').trim()
    const number = Number(block?.number || 0)
    if (!blockId || !Number.isFinite(number) || number <= 0) continue
    let matchedIdx = -1
    for (let idx = 0; idx < visibleNodes.length; idx += 1) {
      if (usedNodeIdx.has(idx)) continue
      if (!nodeNumbers[idx].includes(Math.floor(number))) continue
      matchedIdx = idx
      break
    }
    if (matchedIdx < 0) continue
    bind(visibleNodes[matchedIdx], block)
    usedNodeIdx.add(matchedIdx)
    usedBlockId.add(blockId)
  }

  for (const block of equationBlocks) {
    const blockId = String(block?.block_id || '').trim()
    if (!blockId || usedBlockId.has(blockId)) continue
    const nextIdx = visibleNodes.findIndex((_node, idx) => !usedNodeIdx.has(idx))
    if (nextIdx < 0) break
    bind(visibleNodes[nextIdx], block)
    usedNodeIdx.add(nextIdx)
    usedBlockId.add(blockId)
  }

  return usedNodeIdx.size
}

function resolveVisibleEquationTarget(
  root: HTMLElement,
  opts: { blockId?: string; anchorId?: string },
): HTMLElement | null {
  const blockId = String(opts?.blockId || '').trim()
  const anchorId = String(opts?.anchorId || '').trim()
  const equations = visibleEquationBlocks(root)
  if (blockId) {
    const byBlock = equations.find((node) => String(node.getAttribute('data-kb-block-id') || '').trim() === blockId)
    if (byBlock) return byBlock
  }
  if (anchorId) {
    const byAnchor = equations.find((node) => String(node.getAttribute('data-kb-anchor-id') || '').trim() === anchorId)
    if (byAnchor) return byAnchor
  }
  return null
}

function visibleInlineFormulaNodes(container: HTMLElement | null): HTMLElement[] {
  if (!container) return []
  return Array.from(container.querySelectorAll<HTMLElement>('.katex'))
    .filter((node) => !node.closest('.katex-display'))
}

export function resolveInlineFormulaTarget(container: HTMLElement | null, seed: string): HTMLElement | null {
  const nodes = visibleInlineFormulaNodes(container)
  if (nodes.length <= 0) return null
  const probe = String(seed || '').trim()
  if (!probe) return nodes[0] || null
  let best: HTMLElement | null = null
  let bestScore = 0
  for (const node of nodes) {
    const text = String(node.textContent || '').trim()
    if (!text) continue
    let score = 0.78 * formulaOverlapScore(probe, text)
    score += 0.24 * snippetMatchScore(probe, text)
    if (score > bestScore) {
      best = node
      bestScore = score
    }
  }
  return bestScore >= 0.14 ? best : null
}

export function resolveDirectTargetNode(
  root: HTMLElement,
  readerBlocks: ReaderDocBlock[],
  opts: { blockId?: string; anchorId?: string; anchorKind?: string },
): DirectTargetResult {
  const blockId = String(opts?.blockId || '').trim()
  const anchorId = String(opts?.anchorId || '').trim()
  const anchorKind = String(opts?.anchorKind || '').trim().toLowerCase()
  const hintBlock = (() => {
    if (!Array.isArray(readerBlocks) || readerBlocks.length <= 0) return null
    if (blockId) {
      const byBlock = readerBlocks.find((item) => String(item?.block_id || '').trim() === blockId)
      if (byBlock) return byBlock
    }
    if (anchorId) {
      const byAnchor = readerBlocks.find((item) => String(item?.anchor_id || '').trim() === anchorId)
      if (byAnchor) return byAnchor
    }
    return null
  })()
  const preferEquation = anchorKind === 'equation' || String(hintBlock?.kind || '').trim().toLowerCase() === 'equation'
  let target: HTMLElement | null = null
  if (preferEquation) {
    target = resolveVisibleEquationTarget(root, {
      blockId: blockId || String(hintBlock?.block_id || '').trim(),
      anchorId: anchorId || String(hintBlock?.anchor_id || '').trim(),
    })
  }
  if (blockId) {
    target = target || root.querySelector<HTMLElement>(`[data-kb-block-id="${CSS.escape(blockId)}"]`)
  }
  if (!target && anchorId) {
    target = Array.from(root.querySelectorAll<HTMLElement>('[data-kb-anchor-id]'))
      .find((node) => String(node.getAttribute('data-kb-anchor-id') || '') === anchorId) || null
  }
  if (!target && hintBlock) {
    const hintBlockId = String(hintBlock.block_id || '').trim()
    const hintAnchorId = String(hintBlock.anchor_id || '').trim()
    if (preferEquation) {
      target = resolveVisibleEquationTarget(root, {
        blockId: hintBlockId,
        anchorId: hintAnchorId,
      })
    }
    if (hintBlockId) {
      target = target || root.querySelector<HTMLElement>(`[data-kb-block-id="${CSS.escape(hintBlockId)}"]`)
    }
    if (!target && hintAnchorId) {
      target = Array.from(root.querySelectorAll<HTMLElement>('[data-kb-anchor-id]'))
        .find((node) => String(node.getAttribute('data-kb-anchor-id') || '') === hintAnchorId) || null
    }
  }
  return { target, hintBlock }
}

export function resolveRelatedTargetNodes(
  root: HTMLElement,
  readerBlocks: ReaderDocBlock[],
  relatedBlockIds: string[],
  primaryNode: HTMLElement | null,
): HTMLElement[] {
  const primaryReadable = closestReadableBlock(primaryNode) || primaryNode
  const out: HTMLElement[] = []
  const seen = new Set<string>()
  for (const rawId of relatedBlockIds) {
    const blockId = String(rawId || '').trim()
    if (!blockId) continue
    const resolved = resolveDirectTargetNode(root, readerBlocks, { blockId })
    const target = closestReadableBlock(resolved.target) || resolved.target
    if (!target) continue
    if (primaryReadable && target === primaryReadable) continue
    const key = String(target.getAttribute('data-kb-block-id') || blockId).trim() || blockId
    if (seen.has(key)) continue
    seen.add(key)
    out.push(target)
  }
  return out
}

export function selectionStateInside(container: HTMLElement | null): ReaderSelectionState | null {
  if (!container) return null
  const sel = window.getSelection()
  if (!sel || sel.isCollapsed || sel.rangeCount <= 0) return null
  const range = sel.getRangeAt(0)
  const anchorNode = sel.anchorNode
  const focusNode = sel.focusNode
  if (!anchorNode || !focusNode) return null
  if (!container.contains(anchorNode) || !container.contains(focusNode)) return null
  const text = String(range.toString() || '').trim().slice(0, 2000)
  if (!text) return null
  const rawOffsets = rawOffsetsFromRange(container, range)
  if (!rawOffsets) return null
  const startBlock = findSelectionReadableBlock(range.startContainer)
  const endBlock = findSelectionReadableBlock(range.endContainer)
  const blocks = readableBlocks(container)
  const startReadableIndex = startBlock ? blocks.findIndex((item) => item === startBlock) : -1
  const endReadableIndex = endBlock ? blocks.findIndex((item) => item === endBlock) : -1
  const rangeStartIndex = startReadableIndex >= 0 && endReadableIndex >= 0
    ? Math.min(startReadableIndex, endReadableIndex)
    : -1
  const rangeEndIndex = startReadableIndex >= 0 && endReadableIndex >= 0
    ? Math.max(startReadableIndex, endReadableIndex)
    : -1
  const sameBlock = rangeStartIndex >= 0 && rangeStartIndex === rangeEndIndex
  const highlightMeta = sameBlock && startBlock ? locateSelectionOccurrence(container, startBlock, range, text) : null
  const documentOccurrence = locateTextOccurrenceInContainer(container, range, text)
  const rect = range.getBoundingClientRect()
  if (!rect || (rect.width <= 0 && rect.height <= 0)) return null
  const containerRect = container.getBoundingClientRect()
  if (rect.bottom < containerRect.top || rect.top > containerRect.bottom) return null
  const centerX = rect.left + (rect.width / 2)
  const aboveY = rect.top - containerRect.top - 10
  const belowY = rect.bottom - containerRect.top + 10
  const x = Math.max(18, Math.min(containerRect.width - 18, centerX - containerRect.left))
  const y = aboveY >= 16 ? aboveY : belowY
  return {
    text,
    x,
    y,
    startOffset: rawOffsets.startOffset,
    endOffset: rawOffsets.endOffset,
    blockId: String(highlightMeta?.blockId || '').trim(),
    anchorId: String(highlightMeta?.anchorId || '').trim(),
    occurrence: Number.isFinite(Number(highlightMeta?.occurrence || 0))
      ? Math.max(0, Math.floor(Number(highlightMeta?.occurrence || 0)))
      : 0,
    readableIndex: Number.isFinite(Number(highlightMeta?.readableIndex ?? -1))
      ? Math.max(-1, Math.floor(Number(highlightMeta?.readableIndex ?? -1)))
      : -1,
    documentOccurrence: Number.isFinite(Number(documentOccurrence))
      ? Math.max(-1, Math.floor(Number(documentOccurrence)))
      : -1,
    startReadableIndex: rangeStartIndex,
    endReadableIndex: rangeEndIndex,
    canHighlight: Boolean(highlightMeta) || (rangeStartIndex >= 0 && rangeEndIndex >= 0) || documentOccurrence >= 0,
    highlightId: '',
  }
}
