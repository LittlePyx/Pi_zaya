import { Children, isValidElement, useMemo, type CSSProperties, type MouseEvent, type ReactNode } from 'react'
import { message } from 'antd'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import rehypeHighlight from 'rehype-highlight'
import { citationInlineLabel, type CiteDetail } from './citationState'

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
  onLocateSnippet?: (snippet: string) => void
  locateLabelResolver?: (snippet: string) => string
  locateTitleResolver?: (snippet: string) => string
  variant?: 'chat' | 'reader'
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
  limits: '边界',
  next_steps: '下一步',
}

const ANSWER_SECTION_HEAD_RE =
  /^\s*(?:#{1,6}\s*)?(Conclusion|Evidence|Limits|Next\s*Steps|结论|依据|证据|边界|限制|局限|下一步建议|下一步)(?:\s*[:：]\s*(.*))?$/i

interface ParsedAnswerSection {
  key: AnswerSectionKey
  label: string
  body: string
}

function toSectionKey(raw: string): AnswerSectionKey | '' {
  const t = String(raw || '').replace(/\s+/g, '').toLowerCase()
  if (t === 'conclusion' || t === '结论') return 'conclusion'
  if (t === 'evidence' || t === '依据' || t === '证据') return 'evidence'
  if (t === 'limits' || t === '边界' || t === '限制' || t === '局限') return 'limits'
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

function plainText(node: ReactNode): string {
  if (node === null || node === undefined || typeof node === 'boolean') return ''
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map((item) => plainText(item)).join(' ')
  if (isValidElement(node)) {
    const props = node.props as { children?: ReactNode }
    return plainText(props.children)
  }
  return ''
}

function hasMathSignalInline(text: string): boolean {
  const src = String(text || '')
  if (!src) return false
  if (/[=^_]/.test(src)) return true
  if (/\\[a-zA-Z]{2,}/.test(src)) return true
  if (/\$[^$]{1,120}\$/.test(src) || /\$\$[^]{1,260}\$\$/.test(src)) return true
  return false
}

function toLocateSnippet(node: ReactNode): string {
  const text = plainText(node).replace(/\s+/g, ' ').trim()
  if (!text) return ''
  if (hasMathSignalInline(text)) {
    return text.length <= 480 ? text : `${text.slice(0, 480).trimEnd()}...`
  }
  if (text.length <= 420) return text
  const sentences = text
    .split(/(?<=[\u3002\uff01\uff1f.!;:\uff1b\uff1a])\s+/)
    .map((item) => String(item || '').trim())
    .filter(Boolean)
  const merged = sentences.slice(0, 3).join(' ').trim()
  if (merged.length >= 40) {
    return merged.length <= 480 ? merged : `${merged.slice(0, 480).trimEnd()}...`
  }
  return `${text.slice(0, 480).trimEnd()}...`
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
  onLocateSnippet?: (snippet: string) => void,
  locateLabelResolver?: (snippet: string) => string,
  locateTitleResolver?: (snippet: string) => string,
  variant: 'chat' | 'reader' = 'chat',
) {
  const renderLocateButton = (children: ReactNode) => {
    if (!onLocateSnippet) return null
    const snippet = toLocateSnippet(children)
    if (!snippet) return null
    const label = String(locateLabelResolver?.(snippet) || '').trim() || '定位原文'
    const title = String(locateTitleResolver?.(snippet) || '').trim()
    return (
      <button
        type="button"
        className="kb-md-locate-btn"
        title={title || label}
        onClick={(event) => {
          event.preventDefault()
          event.stopPropagation()
          onLocateSnippet(snippet)
        }}
      >
        {label}
      </button>
    )
  }

  return {
    pre: ({ children }: { children?: ReactNode }) => {
      const { text, language } = extractCode(children)
      if (variant === 'reader') {
        return (
          <pre>
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
    table: ({ children }: { children?: ReactNode }) => (
      <div className="kb-table-wrap">
        <table>{children}</table>
      </div>
    ),
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
    img: ({ src, alt }: { src?: string; alt?: string }) => {
      const resolvedSrc = String(src || '').trim()
      if (!resolvedSrc) return null
      return (
        <a href={resolvedSrc} target="_blank" rel="noreferrer" className="kb-md-image-link">
          <img
            src={resolvedSrc}
            alt={String(alt || 'figure')}
            className="kb-md-image"
            loading="lazy"
          />
        </a>
      )
    },
    p: ({ children }: { children?: ReactNode }) => {
      const btn = renderLocateButton(children)
      if (!btn) return <p>{children}</p>
      return (
        <div className="kb-md-loc-block">
          <p>{children}</p>
          {btn}
        </div>
      )
    },
    li: ({ children }: { children?: ReactNode }) => {
      const btn = renderLocateButton(children)
      if (!btn) return <li>{children}</li>
      return (
        <li className="kb-md-loc-li">
          {children}
          {btn}
        </li>
      )
    },
  }
}

export function MarkdownRenderer({
  content,
  citeDetails = [],
  onCitationClick,
  onLocateSnippet,
  locateLabelResolver,
  locateTitleResolver,
  variant = 'chat',
}: Props) {
  const normalizedContent = normalize(content)
  const byAnchor = new Map(citeDetails.map((detail) => [detail.anchor, detail]))
  const toneBySource = useMemo(() => buildToneMap(citeDetails), [citeDetails])
  const components = buildMarkdownComponents(
    byAnchor,
    onCitationClick,
    toneBySource,
    onLocateSnippet,
    locateLabelResolver,
    locateTitleResolver,
    variant,
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
