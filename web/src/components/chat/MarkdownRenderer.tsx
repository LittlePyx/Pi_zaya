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

function extractCode(node: ReactNode): { text: string; language: string } {
  const child = Children.toArray(node)[0]
  if (isValidElement(child)) {
    const props = child.props as { className?: string; children?: ReactNode }
    const language = String(props.className || '').replace('language-', '').trim()
    const text = String(Array.isArray(props.children) ? props.children.join('') : props.children || '').replace(/\n$/, '')
    return { text, language }
  }
  return { text: String(node || ''), language: '' }
}

export function MarkdownRenderer({ content, citeDetails = [], onCitationClick }: Props) {
  const byAnchor = new Map(citeDetails.map((detail) => [detail.anchor, detail]))

  return (
    <div className="kb-markdown prose dark:prose-invert max-w-none text-sm">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
        components={{
          pre: ({ children }) => {
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
          table: ({ children }) => (
            <div className="kb-table-wrap">
              <table>{children}</table>
            </div>
          ),
          a: ({ href, children }) => {
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
          img: ({ src, alt }) => {
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
        }}
      >
        {normalize(content)}
      </ReactMarkdown>
    </div>
  )
}
