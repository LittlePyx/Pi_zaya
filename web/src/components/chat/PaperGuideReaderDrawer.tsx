import { useEffect, useMemo, useRef, useState } from 'react'
import { Button, Drawer, Empty, Spin, Typography } from 'antd'
import { MarkdownRenderer } from './MarkdownRenderer'
import { referencesApi } from '../../api/references'

const { Text } = Typography

export interface ReaderOpenPayload {
  sourcePath: string
  sourceName?: string
  headingPath?: string
  snippet?: string
}

interface Props {
  open: boolean
  payload: ReaderOpenPayload | null
  onClose: () => void
  onAppendSelection: (text: string) => void
}

function normalizeText(input: string) {
  return String(input || '').replace(/\s+/g, ' ').trim().toLowerCase()
}

function tokenizeText(input: string): string[] {
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

function snippetMatchScore(snippet: string, block: string): number {
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

function headingMatchScore(needle: string, heading: string): number {
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

function snippetProbeText(text: string): string {
  const src = String(text || '').replace(/\s+/g, ' ').trim()
  if (!src) return ''
  const pieces = src
    .split(/(?<=[\u3002\uff01\uff1f.!;:\uff1b\uff1a])\s+/)
    .map((item) => item.trim())
    .filter(Boolean)
  if (pieces.length <= 0) return src.slice(0, 260)
  return pieces.slice(0, 2).join(' ').slice(0, 320)
}

function hasFormulaSignal(text: string): boolean {
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

function formulaOverlapScore(a: string, b: string): number {
  const ta = new Set(formulaTokens(a))
  const tb = new Set(formulaTokens(b))
  if (ta.size <= 0 || tb.size <= 0) return 0
  let overlap = 0
  for (const token of ta) {
    if (tb.has(token)) overlap += 1
  }
  return overlap / Math.sqrt(ta.size * tb.size)
}

function closestReadableBlock(node: Element | null): HTMLElement | null {
  if (!node) return null
  return node.closest('p,li,blockquote,pre,code,figcaption,td,th,.katex-display,h1,h2,h3,h4,h5,h6') as HTMLElement | null
}

function headingCandidates(path: string) {
  const parts = String(path || '')
    .split(' / ')
    .map((item) => item.trim())
    .filter(Boolean)
  if (parts.length <= 1) return parts
  return [...parts.slice().reverse(), parts.join(' / ')]
}

function selectionTextInside(container: HTMLElement | null): string {
  if (!container) return ''
  const sel = window.getSelection()
  if (!sel || sel.isCollapsed || sel.rangeCount <= 0) return ''
  const range = sel.getRangeAt(0)
  const anchorNode = sel.anchorNode
  const focusNode = sel.focusNode
  if (!anchorNode || !focusNode) return ''
  if (!container.contains(anchorNode) || !container.contains(focusNode)) return ''
  const text = String(range.toString() || '').trim()
  if (!text) return ''
  return text.slice(0, 2000)
}

export function PaperGuideReaderDrawer({ open, payload, onClose, onAppendSelection }: Props) {
  const contentRef = useRef<HTMLDivElement>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [markdown, setMarkdown] = useState('')
  const [resolvedName, setResolvedName] = useState('')
  const [selection, setSelection] = useState('')
  const [locateHint, setLocateHint] = useState('')

  const sourcePath = String(payload?.sourcePath || '').trim()
  const sourceName = String(payload?.sourceName || '').trim()
  const headingPath = String(payload?.headingPath || '').trim()
  const focusSnippet = String(payload?.snippet || '').trim()

  useEffect(() => {
    if (!open || !sourcePath) return
    let cancelled = false
    setLoading(true)
    setError('')
    setSelection('')
    setLocateHint('')
    referencesApi.readerDoc(sourcePath)
      .then((res) => {
        if (cancelled) return
        setMarkdown(String(res.markdown || ''))
        setResolvedName(String(res.source_name || sourceName || '').trim())
      })
      .catch((err) => {
        if (cancelled) return
        setMarkdown('')
        setError(err instanceof Error ? err.message : '读取文献内容失败')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [open, sourcePath, sourceName])

  useEffect(() => {
    if (!open || !markdown) return
    const timer = window.setTimeout(() => {
      const root = contentRef.current
      if (!root) return
      root.querySelectorAll('.kb-reader-focus').forEach((node) => node.classList.remove('kb-reader-focus'))
      setLocateHint('')

      let target: HTMLElement | null = null
      let headingTarget: HTMLElement | null = null
      const headingNeedles = headingCandidates(headingPath).map(normalizeText).filter(Boolean)
      if (headingNeedles.length > 0) {
        const headings = Array.from(root.querySelectorAll<HTMLElement>('h1,h2,h3,h4,h5,h6'))
        let bestHeading: HTMLElement | null = null
        let bestHeadingScore = 0
        for (const heading of headings) {
          const text = String(heading.textContent || '').trim()
          for (const needle of headingNeedles) {
            const score = headingMatchScore(needle, text)
            if (score > bestHeadingScore) {
              bestHeading = heading
              bestHeadingScore = score
            }
          }
        }
        if (bestHeading && bestHeadingScore >= 0.18) {
          headingTarget = bestHeading
        }
      }
      if (focusSnippet) {
        const probe = snippetProbeText(focusSnippet)
        const blocks = Array.from(root.querySelectorAll<HTMLElement>('p,li,blockquote,pre,code,figcaption,td,th,.katex-display'))
        const allNodes = Array.from(root.querySelectorAll<HTMLElement>('h1,h2,h3,h4,h5,h6,p,li,blockquote,pre,code,figcaption,td,th,.katex-display'))
        const nodeIndex = new Map<HTMLElement, number>()
        allNodes.forEach((node, idx) => nodeIndex.set(node, idx))
        const headingIndex = headingTarget ? Number(nodeIndex.get(headingTarget) ?? -1) : -1

        // Formula-first mapping: if query has math signal, try KaTeX annotation nodes first.
        if (hasFormulaSignal(probe)) {
          const anns = Array.from(root.querySelectorAll('annotation[encoding=\"application/x-tex\"]'))
          let eqBest: HTMLElement | null = null
          let eqBestScore = 0
          for (const ann of anns) {
            const annTex = String(ann.textContent || '').trim()
            if (!annTex) continue
            const host = closestReadableBlock(ann)
            if (!host) continue
            let score = (0.9 * formulaOverlapScore(probe, annTex)) + (0.45 * snippetMatchScore(probe, `${annTex} ${host.textContent || ''}`))
            if (headingIndex >= 0) {
              const hostIndex = Number(nodeIndex.get(host) ?? -1)
              if (hostIndex >= 0) {
                const distance = Math.abs(hostIndex - headingIndex)
                score += Math.max(0, 0.14 - distance * 0.003)
              }
            }
            if (score > eqBestScore) {
              eqBest = host
              eqBestScore = score
            }
          }
          if (eqBest && eqBestScore >= 0.16) {
            target = eqBest
          }
        }

        if (!target) {
          let best: HTMLElement | null = null
          let bestScore = 0
          for (const block of blocks) {
            let score = snippetMatchScore(probe, block.textContent || '')
            if (hasFormulaSignal(probe)) {
              score += 0.6 * formulaOverlapScore(probe, block.textContent || '')
            }
            if (headingIndex >= 0) {
              const blockIndex = Number(nodeIndex.get(block) ?? -1)
              if (blockIndex >= 0) {
                const distance = Math.abs(blockIndex - headingIndex)
                score += Math.max(0, 0.18 - distance * 0.004)
              }
            }
            if (score > bestScore) {
              best = block
              bestScore = score
            }
          }
          const dynamicThreshold = hasFormulaSignal(probe)
            ? 0.13
            : (tokenizeText(probe).length >= 8 ? 0.12 : 0.09)
          if (best && bestScore >= dynamicThreshold) {
            target = best
          }
        }
      }
      if (!target && headingTarget) target = headingTarget
      if (!target) {
        if (focusSnippet || headingPath) {
          setLocateHint('未命中精确句段，建议重问一次以生成更细映射。')
        }
        return
      }
      target.classList.add('kb-reader-focus')
      target.scrollIntoView({ behavior: 'smooth', block: 'center' })
      if (!focusSnippet && headingPath) {
        setLocateHint('已按章节定位。')
      }
    }, 80)
    return () => window.clearTimeout(timer)
  }, [open, markdown, headingPath, focusSnippet])

  const title = useMemo(
    () => resolvedName || sourceName || '文献阅读器',
    [resolvedName, sourceName],
  )

  const appendSelection = () => {
    const text = String(selection || '').trim()
    if (!text) return
    const quoted = text.split('\n').map((line) => `> ${line}`).join('\n')
    const sourceLine = title ? `> 来源：${title}\n` : ''
    onAppendSelection(`${sourceLine}${quoted}\n\n`)
    setSelection('')
    try {
      const sel = window.getSelection()
      sel?.removeAllRanges()
    } catch {
      // ignore
    }
  }

  return (
    <Drawer
      open={open}
      width={560}
      mask={false}
      title={title}
      onClose={onClose}
      destroyOnClose={false}
      extra={(
        <Button size="small" disabled={!selection} onClick={appendSelection}>
          追加选中到提问
        </Button>
      )}
    >
      <div className="mb-3 flex items-center justify-between">
        <Text type="secondary" className="text-xs">
          {headingPath ? `定位：${headingPath}` : '定位：文档开头'}
        </Text>
        {selection ? (
          <Text type="secondary" className="text-xs">
            已选中 {selection.length} 字
          </Text>
        ) : null}
      </div>
      {locateHint ? (
        <div className="mb-3">
          <Text type="secondary" className="text-xs">{locateHint}</Text>
        </div>
      ) : null}
      {loading ? (
        <div className="flex h-56 items-center justify-center">
          <Spin />
        </div>
      ) : error ? (
        <Empty description={error} />
      ) : markdown ? (
        <div
          ref={contentRef}
          className="kb-reader-content max-h-[calc(100vh-180px)] overflow-auto pr-1"
          onMouseUp={() => setSelection(selectionTextInside(contentRef.current))}
          onKeyUp={() => setSelection(selectionTextInside(contentRef.current))}
        >
          <MarkdownRenderer content={markdown} variant="reader" />
        </div>
      ) : (
        <Empty description="暂无可读内容" />
      )}
    </Drawer>
  )
}
