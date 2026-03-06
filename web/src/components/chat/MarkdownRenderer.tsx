import { Children, isValidElement, type MouseEvent, type ReactNode } from 'react'
import { message } from 'antd'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import rehypeHighlight from 'rehype-highlight'
import type { CiteDetail } from './citationState'

function normalize(text: string) {
  return text
    .replace(/\\\(/g, '$').replace(/\\\)/g, '$')
    .replace(/\\\[/g, '$$').replace(/\\\]/g, '$$')
}

interface Props {
  content: string
  citeDetails?: CiteDetail[]
  onCitationClick?: (detail: CiteDetail, event: MouseEvent<HTMLElement>) => void
}

type AnswerSectionKey = 'conclusion' | 'evidence' | 'limits' | 'next_steps'

const ANSWER_SECTION_LABEL: Record<AnswerSectionKey, string> = {
  conclusion: '结论',
  evidence: '依据',
  limits: '边界',
  next_steps: '下一步',
}

const ANSWER_SECTION_HEAD_RE = /^\s*(?:#{1,6}\s*)?(Conclusion|Evidence|Limits|Next\s*Steps|结论|依据|边界|下一步建议|下一步)(?:\s*[:：]\s*(.*))?$/i

interface ParsedAnswerSection {
  key: AnswerSectionKey
  label: string
  body: string
}

function toSectionKey(raw: string): AnswerSectionKey | '' {
  const t = String(raw || '').replace(/\s+/g, '').toLowerCase()
  if (t === 'conclusion' || t === '结论') return 'conclusion'
  if (t === 'evidence' || t === '依据') return 'evidence'
  if (t === 'limits' || t === '边界') return 'limits'
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
) {
  return {
    pre: ({ children }: { children?: ReactNode }) => {
      const { text, language } = extractCode(children)
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
        return (
          <button
            type="button"
            className="kb-cite-chip"
            onClick={(event) => {
              event.preventDefault()
              onCitationClick?.(detail, event)
            }}
          >
            {children}
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
  }
}

export function MarkdownRenderer({ content, citeDetails = [], onCitationClick }: Props) {
  const normalizedContent = normalize(content)
  const byAnchor = new Map(citeDetails.map((detail) => [detail.anchor, detail]))
  const components = buildMarkdownComponents(byAnchor, onCitationClick)
  const parsedContract = parseAnswerContract(normalizedContent)

  return (
    <div className="kb-markdown prose dark:prose-invert max-w-none text-sm">
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
