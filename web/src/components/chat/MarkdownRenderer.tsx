import { Children, cloneElement, isValidElement, useMemo, type CSSProperties, type MouseEvent, type ReactNode } from 'react'
import { createContext, useContext } from 'react'
import { message } from 'antd'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import rehypeHighlight from 'rehype-highlight'
import { citationInlineLabel, type CiteDetail } from './citationState'
import type { ReaderDocAnchor, ReaderDocBlock } from '../../api/references'

const TABLE_SEPARATOR_RE = /^\s*\|?(?:\s*:?-{2,}:?\s*\|)+\s*:?-{2,}:?\s*\|?\s*$/
const TABLE_ROW_RE = /^\s*\|?.+\|.+\|?\s*$/

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

function normalize(text: string) {
  return repairCollapsedGfmTables(text)
    .replace(/\\\(/g, '$').replace(/\\\)/g, '$')
    .replace(/\\\[/g, '$$').replace(/\\\]/g, '$$')
}

interface Props {
  content: string
  citeDetails?: CiteDetail[]
  onCitationClick?: (detail: CiteDetail, event: MouseEvent<HTMLElement>) => void
  onLocateSnippet?: (snippet: string, meta?: LocateRenderMeta) => void
  canLocateSnippet?: (snippet: string, meta?: LocateRenderMeta) => boolean
  locateTitleResolver?: (snippet: string) => string
  inlineLocateTokenPolicy?: Partial<Record<InlineLocateTokenKind, boolean>>
  inlineTextLocateEnabled?: boolean
  locateSurfacePolicy?: Partial<Record<LocateSurfaceKind, boolean>>
  variant?: 'chat' | 'reader'
  readerAnchors?: ReaderDocAnchor[]
  readerBlocks?: ReaderDocBlock[]
}

type LocateSurfaceKind = 'paragraph' | 'list_item' | 'blockquote' | 'equation' | 'figure'

interface LocateRenderMeta {
  kind: LocateSurfaceKind
  order: number
}
type InlineLocateTokenKind = 'quote' | 'figure_ref'
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
  const tail = (
    <span key={`${keyBase}:btn`} className="kb-md-loc-tail">
      {btn}
    </span>
  )

  const append = (node: ReactNode, keyPath: string): ReactNode => {
    if (node === null || node === undefined || typeof node === 'boolean') return node
    if (typeof node === 'string' || typeof node === 'number') {
      return (
        <>
          {node}
          {tail}
        </>
      )
    }
    if (Array.isArray(node)) {
      const items = [...node]
      for (let idx = items.length - 1; idx >= 0; idx -= 1) {
        if (isEmptyReactNode(items[idx])) continue
        items[idx] = append(items[idx], `${keyPath}:${idx}`)
        return items
      }
      return [...items, tail]
    }
    if (!isValidElement(node)) {
      return (
        <>
          {node}
          {tail}
        </>
      )
    }
    if (isTailBoundaryElement(node)) {
      return (
        <>
          {node}
          {tail}
        </>
      )
    }

    const props = node.props as { children?: ReactNode }
    if (props.children !== undefined) {
      return cloneElement(node, undefined, append(props.children, `${keyPath}:child`))
    }
    return (
      <>
        {node}
        {tail}
      </>
    )
  }

  return append(children, keyBase)
}

function normalizeInlineLocateTokenPolicy(
  policy?: Partial<Record<InlineLocateTokenKind, boolean>>,
): Record<InlineLocateTokenKind, boolean> {
  return {
    quote: policy?.quote !== false,
    figure_ref: policy?.figure_ref !== false,
  }
}

function normalizeLocateSurfacePolicy(
  policy?: Partial<Record<LocateSurfaceKind, boolean>>,
): Record<LocateSurfaceKind, boolean> {
  return {
    paragraph: policy?.paragraph !== false,
    list_item: policy?.list_item !== false,
    blockquote: policy?.blockquote !== false,
    equation: policy?.equation !== false,
    figure: policy?.figure !== false,
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
  const quoteTokens = collectInlineLocateTokens(raw, { quote: true, figure_ref: false })
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
  const tokens = collectInlineLocateTokens(raw, { quote: false, figure_ref: true })
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
  const push = (start: number, end: number, text0: string, kind: InlineLocateTokenKind) => {
    const text = String(text0 || '').replace(/\s+/g, ' ').trim()
    if (!text) return
    if (kind === 'quote') {
      if (text.length < 18) return
      if (isHeadingLikeQuotedToken(start, text)) return
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
    const token: ReaderAnchorToken = { anchorId, blockId: blockId || undefined, kind }
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
      if (!range) return null
      const preferred = new Set((kinds || []).map((k) => normalizeReaderAnchorKind(k)))
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
  onCitationClick?: (detail: CiteDetail, event: MouseEvent<HTMLElement>) => void,
  toneBySource?: Map<string, CiteChipTone>,
  onLocateSnippet?: (snippet: string, meta?: LocateRenderMeta) => void,
  canLocateSnippet?: (snippet: string, meta?: LocateRenderMeta) => boolean,
  locateTitleResolver?: (snippet: string) => string,
  inlineLocateTokenPolicy?: Partial<Record<InlineLocateTokenKind, boolean>>,
  inlineTextLocateEnabled: boolean = true,
  locateSurfacePolicy?: Partial<Record<LocateSurfaceKind, boolean>>,
  variant: 'chat' | 'reader' = 'chat',
  readerAnchorAllocator?: ReaderAnchorAllocator | null,
  readerBlockResolver?: ReaderBlockResolver | null,
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
    const label = '定位到原文证据'
    const title = String(locateTitleResolver?.(snippet) || '').trim() || '定位到原文证据'
    const kind = meta?.kind || 'paragraph'
    const badgeText = kind === 'equation'
      ? '式'
      : kind === 'figure'
        ? '图'
        : '原文'
    return (
      <button
        type="button"
        className={`kb-md-locate-inline-btn kb-md-locate-inline-btn-${kind}`}
        aria-label={label}
        title={title || label}
        data-locate-kind={kind}
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

  const decorateInlineLocateAnchors = (
    children: ReactNode,
    meta: LocateRenderMeta,
  ): { content: ReactNode; count: number } => {
    const metaForToken = (kind: InlineLocateTokenKind): LocateRenderMeta => {
      if (kind === 'quote') return { ...meta, kind: 'blockquote' }
      if (kind === 'figure_ref') return { ...meta, kind: 'figure' }
      return meta
    }
    const renderStringNode = (text0: string, keyBase: string): { content: ReactNode; count: number } => {
      const tokens = collectInlineLocateTokens(text0, effectiveInlineLocateTokenPolicy)
      if (tokens.length <= 0) return { content: text0, count: 0 }
      const parts: ReactNode[] = []
      let last = 0
      let count = 0
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
        } else {
          parts.push(raw)
        }
        last = token.end
      })
      if (last < text0.length) {
        parts.push(text0.slice(last))
      }
      return { content: parts, count }
    }

    const visit = (node: ReactNode, keyBase: string): { content: ReactNode; count: number } => {
      if (node === null || node === undefined || typeof node === 'boolean') {
        return { content: node, count: 0 }
      }
      if (typeof node === 'string' || typeof node === 'number') {
        return renderStringNode(String(node), keyBase)
      }
      if (Array.isArray(node)) {
        let count = 0
        const content = node.map((item, idx) => {
          const rendered = visit(item, `${keyBase}:${idx}`)
          count += rendered.count
          return rendered.content
        })
        return { content, count }
      }
      if (!isValidElement(node)) {
        return { content: node, count: 0 }
      }
      const nodeType = typeof node.type === 'string' ? node.type.toLowerCase() : ''
      const props = node.props as { children?: ReactNode; className?: string }
      const className = String(props.className || '')
      if (isInlineMathClass(className)) {
        // Inline KaTeX variables create noisy duplicate entrances; keep entrances
        // only on numbered equation refs / block formulas.
        return { content: node, count: 0 }
      }
      if (['a', 'button', 'img', 'code', 'pre', 'script', 'style'].includes(nodeType)) {
        return { content: node, count: 0 }
      }
      if (/\bkb-cite-chip\b/.test(className) || /\bkb-md-locate-inline-btn\b/.test(className)) {
        return { content: node, count: 0 }
      }
      const rendered = visit(props.children, `${keyBase}:child`)
      if (rendered.count <= 0) return { content: node, count: 0 }
      return {
        content: cloneElement(node, undefined, rendered.content),
        count: rendered.count,
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
                navigator.clipboard.writeText(text).then(() => message.success('代码已复制'))
              }}
            >
              复制代码
            </button>
          </div>
          <pre>{children}</pre>
        </div>
      )
    },
    table: ({ node, children }: { node?: unknown; children?: ReactNode }) => {
      const attrs = variant === 'reader'
        ? readerAnchorAttrs(pickReaderAnchor(node, ['table']))
        : undefined
      return (
        <div className="kb-table-wrap">
          <table {...attrs}>{children}</table>
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
      const detail = key ? byAnchor.get(key) : undefined
      if (detail) {
        const tone = toneBySource?.get(sourceKey(detail))
        const toneStyle: CSSProperties | undefined = tone
          ? ({
              ['--kb-cite-fg' as string]: tone.fg,
              ['--kb-cite-fg-hover' as string]: tone.fgHover,
            } as CSSProperties)
          : undefined
        return (
          <button
            type="button"
            className="kb-cite-chip"
            style={toneStyle}
            title={detail.sourceName || detail.sourcePath || undefined}
            onClick={(event) => {
              event.preventDefault()
              onCitationClick?.(detail, event)
            }}
          >
            {citationInlineLabel(detail, { includeSource: false })}
          </button>
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
      const attrs = variant === 'reader'
        ? readerAnchorAttrs(pickReaderAnchor(node, ['figure']))
        : undefined
      return (
        <span className={btn ? 'kb-md-figure-shell' : undefined} {...attrs}>
          <a href={resolvedSrc} target="_blank" rel="noreferrer" className="kb-md-image-link">
            <img
              src={resolvedSrc}
              alt={String(alt || 'figure')}
              className="kb-md-image"
              loading="lazy"
            />
          </a>
          {btn ? <span className="kb-md-figure-tail">{btn}</span> : null}
        </span>
      )
    },
    p: ({ node, children }: { node?: unknown; children?: ReactNode }) => {
      const insideBlockquote = useContext(BlockquoteLocateContext)
      const attrs = variant === 'reader'
        ? readerAnchorAttrs(pickReaderAnchor(node, ['paragraph']))
        : undefined
      const renderOrder = nextLocateRenderOrder()
      const meta = { kind: 'paragraph' as const, order: renderOrder }
      const inline = (variant === 'chat' && inlineTextLocateEnabled && !insideBlockquote)
        ? decorateInlineLocateAnchors(children, meta)
        : { content: children, count: 0 }
      const content = inline.count > 0 ? inline.content : children
      if (variant !== 'chat') {
        return <p {...attrs}>{content}</p>
      }
      if (isFigureHostParagraph(content)) {
        return <p {...attrs} className="kb-md-figure-host">{content}</p>
      }
      const figureCaptionSnippet = preferredFigureCaptionSnippet(content)
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
      return <p {...attrs}>{content}</p>
    },
    li: ({ node, children }: { node?: unknown; children?: ReactNode }) => {
      const insideBlockquote = useContext(BlockquoteLocateContext)
      const attrs = variant === 'reader'
        ? readerAnchorAttrs(pickReaderAnchor(node, ['list_item']))
        : undefined
      const renderOrder = nextLocateRenderOrder()
      const meta = { kind: 'list_item' as const, order: renderOrder }
      const inline = (variant === 'chat' && inlineTextLocateEnabled && !insideBlockquote)
        ? decorateInlineLocateAnchors(children, meta)
        : { content: children, count: 0 }
      return <li {...attrs}>{inline.count > 0 ? inline.content : children}</li>
    },
    div: (props: any) => {
      const { node, className, children, ...rest } = props || {}
      const cls = String(className || '').trim()
      const displayMath = isDisplayMathClass(cls)
      if (!displayMath) return <div className={cls || undefined} {...(rest as Record<string, unknown>)}>{children}</div>
      if (variant === 'reader') {
        // Display equations are bound at runtime to visible .katex-display nodes.
        // Static line-based binding is too unstable here and can mis-assign them
        // to neighboring paragraph blocks in the browser render path.
        return <div className={cls || undefined} data-kb-display-equation="1" {...(rest as Record<string, unknown>)}>{children}</div>
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
    span: (props: any) => {
      const { node, className, children, ...rest } = props || {}
      const cls = String(className || '').trim()
      const displayMath = isDisplayMathClass(cls)
      if (!displayMath) return <span className={cls || undefined} {...(rest as Record<string, unknown>)}>{children}</span>
      if (variant === 'reader') {
        return <span className={cls || undefined} data-kb-display-equation="1" {...(rest as Record<string, unknown>)}>{children}</span>
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
  onLocateSnippet,
  canLocateSnippet,
  locateTitleResolver,
  inlineLocateTokenPolicy,
  inlineTextLocateEnabled = true,
  locateSurfacePolicy,
  variant = 'chat',
  readerAnchors,
  readerBlocks,
}: Props) {
  const normalizedContent = normalize(content)
  const byAnchor = new Map(citeDetails.map((detail) => [detail.anchor, detail]))
  const toneBySource = useMemo(() => buildToneMap(citeDetails), [citeDetails])
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
    onCitationClick,
    toneBySource,
    onLocateSnippet,
    canLocateSnippet,
    locateTitleResolver,
    inlineLocateTokenPolicy,
    inlineTextLocateEnabled,
    locateSurfacePolicy,
    variant,
    readerAnchorAllocator,
    readerBlockResolver,
  )
  const parsedContract = parseAnswerContract(normalizedContent)

  return (
    <div className={`kb-markdown prose dark:prose-invert max-w-none text-sm ${variant === 'reader' ? 'kb-markdown-reader' : 'kb-markdown-chat'}`}>
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
              <div className="kb-answer-title">{section.label}</div>
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
          {normalizedContent}
        </ReactMarkdown>
      )}
    </div>
  )
}

