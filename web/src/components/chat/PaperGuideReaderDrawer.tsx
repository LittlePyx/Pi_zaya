import { useEffect, useMemo, useRef, useState } from 'react'
import { Button, Drawer, Empty, Spin, Typography } from 'antd'
import { MarkdownRenderer } from './MarkdownRenderer'
import { referencesApi, type ReaderDocAnchor, type ReaderDocBlock } from '../../api/references'

const { Text } = Typography

export interface ReaderOpenPayload {
  sourcePath: string
  sourceName?: string
  headingPath?: string
  snippet?: string
  highlightSnippet?: string
  anchorId?: string
  blockId?: string
  strictLocate?: boolean
  alternatives?: Array<{
    headingPath?: string
    snippet?: string
    highlightSnippet?: string
    anchorId?: string
    blockId?: string
  }>
  initialAltIndex?: number
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

function shortCandidateLabel(input: string, maxLen = 22): string {
  const text = String(input || '').replace(/\s+/g, ' ').trim()
  if (!text) return ''
  if (text.length <= maxLen) return text
  return `${text.slice(0, Math.max(8, maxLen - 3)).trimEnd()}...`
}

function extractEquationNumbers(text: string): number[] {
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
  for (const m of src.matchAll(/\b(?:eq|equation|公式)\s*[#(（]?\s*(\d{1,4})\s*[)）]?/gi)) {
    push(String(m[1] || ''))
  }
  for (const m of src.matchAll(/\((\d{1,4})\)/g)) {
    push(String(m[1] || ''))
  }
  return out
}

function equationNumberMatchScore(blockText: string, numbers: number[]): number {
  if (numbers.length <= 0) return 0
  const text = normalizeText(blockText)
  if (!text) return 0
  let best = 0
  for (const num of numbers) {
    if (new RegExp(`\\(\\s*${num}\\s*\\)`).test(text)) best = Math.max(best, 1.0)
    if (new RegExp(`\\[\\s*${num}\\s*\\]`).test(text)) best = Math.max(best, 0.92)
    if (new RegExp(`\\beq(?:uation)?\\s*\\.?\\s*${num}\\b`, 'i').test(text)) best = Math.max(best, 0.9)
    if (new RegExp(`\\b公式\\s*${num}\\b`).test(text)) best = Math.max(best, 0.92)
  }
  return best
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
  const [readerAnchors, setReaderAnchors] = useState<ReaderDocAnchor[]>([])
  const [readerBlocks, setReaderBlocks] = useState<ReaderDocBlock[]>([])
  const [resolvedName, setResolvedName] = useState('')
  const [selection, setSelection] = useState('')
  const [locateHint, setLocateHint] = useState('')

  const sourcePath = String(payload?.sourcePath || '').trim()
  const sourceName = String(payload?.sourceName || '').trim()
  const headingPath = String(payload?.headingPath || '').trim()
  const focusSnippet = String(payload?.snippet || '').trim()
  const highlightSnippet = String(payload?.highlightSnippet || '').trim()
  const anchorId = String(payload?.anchorId || '').trim()
  const blockId = String(payload?.blockId || '').trim()
  const strictLocate = Boolean(payload?.strictLocate)
  const alternatives = useMemo(() => {
    const listRaw = Array.isArray(payload?.alternatives) ? payload?.alternatives : []
    const out: Array<{ headingPath: string; snippet: string; highlightSnippet: string; anchorId: string; blockId: string }> = []
    const seen = new Set<string>()
    const push = (
      headingPath0: string,
      snippet0: string,
      highlightSnippet0: string,
      anchorId0: string,
      blockId0: string,
    ) => {
      const heading = String(headingPath0 || '').trim()
      const snippet = String(snippet0 || '').trim()
      const highlightSnippet = String(highlightSnippet0 || '').trim()
      const anchorId = String(anchorId0 || '').trim()
      const blockId = String(blockId0 || '').trim()
      if (!heading && !snippet && !highlightSnippet && !anchorId && !blockId) return
      const key = `${blockId.toLowerCase()}::${anchorId.toLowerCase()}::${heading.toLowerCase()}::${highlightSnippet.toLowerCase().slice(0, 180)}::${snippet.toLowerCase().slice(0, 180)}`
      if (seen.has(key)) return
      seen.add(key)
      out.push({ headingPath: heading, snippet, highlightSnippet, anchorId, blockId })
    }
    push(headingPath, focusSnippet, highlightSnippet, anchorId, blockId)
    for (const item of listRaw) {
      if (!item || typeof item !== 'object') continue
      push(
        String(item.headingPath || ''),
        String(item.snippet || ''),
        String(item.highlightSnippet || ''),
        String(item.anchorId || ''),
        String(item.blockId || ''),
      )
      if (out.length >= 6) break
    }
    return out
  }, [payload, headingPath, focusSnippet, highlightSnippet, anchorId, blockId])
  const [activeAltIndex, setActiveAltIndex] = useState(0)

  useEffect(() => {
    const maxIndex = Math.max(0, alternatives.length - 1)
    const hintIndex = Number(payload?.initialAltIndex || 0)
    const nextIndex = Number.isFinite(hintIndex) ? Math.max(0, Math.min(maxIndex, Math.floor(hintIndex))) : 0
    setActiveAltIndex(nextIndex)
  }, [payload, alternatives.length])

  const activeAlt = alternatives[activeAltIndex] || null
  const activeHeadingPath = String(activeAlt?.headingPath || headingPath).trim()
  const activeFocusSnippet = String(activeAlt?.snippet || focusSnippet).trim()
  const activeHighlightSnippet = String(activeAlt?.highlightSnippet || highlightSnippet || activeFocusSnippet).trim()
  const activeAnchorId = String(activeAlt?.anchorId || anchorId).trim()
  const activeBlockId = String(activeAlt?.blockId || blockId).trim()

  useEffect(() => {
    if (!open || !sourcePath) return
    let cancelled = false
    setLoading(true)
    setError('')
    setSelection('')
    setLocateHint('')
    setReaderAnchors([])
    setReaderBlocks([])
    referencesApi.readerDoc(sourcePath)
      .then((res) => {
        if (cancelled) return
        setMarkdown(String(res.markdown || ''))
        setReaderAnchors(Array.isArray(res.anchors) ? res.anchors : [])
        setReaderBlocks(Array.isArray(res.blocks) ? res.blocks : [])
        setResolvedName(String(res.source_name || sourceName || '').trim())
      })
      .catch((err) => {
        if (cancelled) return
        setMarkdown('')
        setReaderAnchors([])
        setReaderBlocks([])
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
      const readerBlockHint = (() => {
        if (!Array.isArray(readerBlocks) || readerBlocks.length <= 0) return null
        if (activeBlockId) {
          const byBlock = readerBlocks.find((item) => String(item?.block_id || '').trim() === activeBlockId)
          if (byBlock) return byBlock
        }
        if (activeAnchorId) {
          const byAnchor = readerBlocks.find((item) => String(item?.anchor_id || '').trim() === activeAnchorId)
          if (byAnchor) return byAnchor
        }
        return null
      })()
      if (activeBlockId) {
        const byBlock = root.querySelector<HTMLElement>(`[data-kb-block-id="${CSS.escape(activeBlockId)}"]`)
        if (byBlock) target = byBlock
      }
      if (activeAnchorId) {
        const direct = Array.from(root.querySelectorAll<HTMLElement>('[data-kb-anchor-id]'))
          .find((node) => String(node.getAttribute('data-kb-anchor-id') || '') === activeAnchorId) || null
        if (direct) target = direct
      }
      if (!target && readerBlockHint) {
        const hintBlockId = String(readerBlockHint.block_id || '').trim()
        const hintAnchorId = String(readerBlockHint.anchor_id || '').trim()
        if (hintBlockId) {
          const byBlock = root.querySelector<HTMLElement>(`[data-kb-block-id="${CSS.escape(hintBlockId)}"]`)
          if (byBlock) target = byBlock
        }
        if (!target && hintAnchorId) {
          const byAnchor = Array.from(root.querySelectorAll<HTMLElement>('[data-kb-anchor-id]'))
            .find((node) => String(node.getAttribute('data-kb-anchor-id') || '') === hintAnchorId) || null
          if (byAnchor) target = byAnchor
        }
      }
      const hasDirectIdentityHint = Boolean(
        activeBlockId
        || activeAnchorId
        || String(readerBlockHint?.block_id || '').trim()
        || String(readerBlockHint?.anchor_id || '').trim(),
      )
      if (!target && hasDirectIdentityHint && alternatives.length > 1) {
        let resolvedAltIndex = -1
        for (let idx = 0; idx < alternatives.length; idx += 1) {
          if (idx === activeAltIndex) continue
          const alt = alternatives[idx]
          if (!alt || typeof alt !== 'object') continue
          const altBlockId = String(alt.blockId || '').trim()
          const altAnchorId = String(alt.anchorId || '').trim()
          if (!altBlockId && !altAnchorId) continue

          let altTarget: HTMLElement | null = null
          if (altBlockId) {
            altTarget = root.querySelector<HTMLElement>(`[data-kb-block-id="${CSS.escape(altBlockId)}"]`)
          }
          if (!altTarget && altAnchorId) {
            altTarget = Array.from(root.querySelectorAll<HTMLElement>('[data-kb-anchor-id]'))
              .find((node) => String(node.getAttribute('data-kb-anchor-id') || '') === altAnchorId) || null
          }
          if (!altTarget && Array.isArray(readerBlocks) && readerBlocks.length > 0) {
            const hint = readerBlocks.find((item) => {
              const bid = String(item?.block_id || '').trim()
              const aid = String(item?.anchor_id || '').trim()
              return (altBlockId && bid === altBlockId) || (altAnchorId && aid === altAnchorId)
            }) || null
            if (hint) {
              const hintBlockId = String(hint.block_id || '').trim()
              const hintAnchorId = String(hint.anchor_id || '').trim()
              if (hintBlockId) {
                altTarget = root.querySelector<HTMLElement>(`[data-kb-block-id="${CSS.escape(hintBlockId)}"]`)
              }
              if (!altTarget && hintAnchorId) {
                altTarget = Array.from(root.querySelectorAll<HTMLElement>('[data-kb-anchor-id]'))
                  .find((node) => String(node.getAttribute('data-kb-anchor-id') || '') === hintAnchorId) || null
              }
            }
          }

          if (altTarget) {
            resolvedAltIndex = idx
            break
          }
        }
        if (resolvedAltIndex >= 0) {
          setActiveAltIndex(resolvedAltIndex)
          return
        }
      }
      if (!target && hasDirectIdentityHint) {
        setLocateHint('未命中精确依据块，已停止模糊定位。建议重问一次以刷新映射。')
        return
      }
      if (!target && strictLocate) {
        setLocateHint('未命中精确依据块，已停止模糊定位。建议重问一次以刷新映射。')
        return
      }
      if (!target) {
        const hintHeadingPath = String(readerBlockHint?.heading_path || '').trim()
        const headingNeedles = headingCandidates(activeHeadingPath || hintHeadingPath).map(normalizeText).filter(Boolean)
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
        const hintBlockText = String(readerBlockHint?.text || '').trim()
        const focusSeed = String(activeHighlightSnippet || activeFocusSnippet || hintBlockText).trim()
        if (focusSeed) {
          const probe = snippetProbeText(focusSeed)
          const eqNumbersAll = [
            ...extractEquationNumbers(`${activeFocusSnippet} ${activeHeadingPath}`),
            ...extractEquationNumbers(`${hintBlockText} ${hintHeadingPath}`),
          ]
          const hintNumber = Number(readerBlockHint?.number || 0)
          if (Number.isFinite(hintNumber) && hintNumber > 0) {
            eqNumbersAll.push(Math.floor(hintNumber))
          }
          const eqNumbers = Array.from(new Set(eqNumbersAll.filter((item) => Number.isFinite(item) && item > 0)))
          const preferFormula = Boolean(
            hasFormulaSignal(probe)
            || hasFormulaSignal(hintBlockText)
            || String(readerBlockHint?.kind || '').trim().toLowerCase() === 'equation',
          )
          const blockSelector = preferFormula
            ? 'p,li,blockquote,pre,code,figcaption,td,th,.katex-display,.katex'
            : 'p,li,blockquote,pre,code,figcaption,td,th,.katex-display'
          const blocks = Array.from(root.querySelectorAll<HTMLElement>(blockSelector))
          const allNodes = Array.from(root.querySelectorAll<HTMLElement>('h1,h2,h3,h4,h5,h6,p,li,blockquote,pre,code,figcaption,td,th,.katex-display,.katex'))
          const nodeIndex = new Map<HTMLElement, number>()
          allNodes.forEach((node, idx) => nodeIndex.set(node, idx))
          const headingIndex = headingTarget ? Number(nodeIndex.get(headingTarget) ?? -1) : -1

          // Formula-first mapping: if query has math signal, try KaTeX annotation nodes first.
          if (preferFormula) {
            const anns = Array.from(root.querySelectorAll('annotation[encoding="application/x-tex"]'))
            let eqBest: HTMLElement | null = null
            let eqBestScore = 0
            for (const ann of anns) {
              const annTex = String(ann.textContent || '').trim()
              if (!annTex) continue
              const host = closestReadableBlock(ann)
              if (!host) continue
              let score = (0.9 * formulaOverlapScore(probe, annTex)) + (0.45 * snippetMatchScore(probe, `${annTex} ${host.textContent || ''}`))
              if (eqNumbers.length > 0) {
                score += 1.05 * equationNumberMatchScore(host.textContent || '', eqNumbers)
              }
              const hostAnchor = String(host.getAttribute('data-kb-anchor-id') || '').trim()
              if (activeAnchorId && hostAnchor === activeAnchorId) {
                score += 1.8
              }
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
            if (eqBest && eqBestScore >= (eqNumbers.length > 0 ? 0.22 : 0.15)) {
              target = eqBest
            }
          }

          if (!target && eqNumbers.length > 0) {
            let eqNumBest: HTMLElement | null = null
            let eqNumBestScore = 0
            for (const block of blocks) {
              const text = String(block.textContent || '')
              let score = equationNumberMatchScore(text, eqNumbers)
              if (score <= 0) continue
              score += 0.45 * formulaOverlapScore(probe, text)
              score += 0.35 * snippetMatchScore(probe, text)
              if (headingIndex >= 0) {
                const blockIndex = Number(nodeIndex.get(block) ?? -1)
                if (blockIndex >= 0) {
                  const distance = Math.abs(blockIndex - headingIndex)
                  score += Math.max(0, 0.1 - distance * 0.002)
                }
              }
              if (score > eqNumBestScore) {
                eqNumBest = block
                eqNumBestScore = score
              }
            }
            if (eqNumBest && eqNumBestScore >= 0.18) {
              target = eqNumBest
            }
          }

          if (!target) {
            let best: HTMLElement | null = null
            let bestScore = 0
            for (const block of blocks) {
              let score = snippetMatchScore(probe, block.textContent || '')
              if (preferFormula) {
                score += 0.6 * formulaOverlapScore(probe, block.textContent || '')
              }
              if (eqNumbers.length > 0) {
                score += 0.55 * equationNumberMatchScore(block.textContent || '', eqNumbers)
              }
              const blockAnchor = String(block.getAttribute('data-kb-anchor-id') || '').trim()
              if (activeAnchorId && blockAnchor === activeAnchorId) {
                score += 0.9
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
            const dynamicThreshold = preferFormula
              ? 0.13
              : (tokenizeText(probe).length >= 8 ? 0.12 : 0.09)
            if (best && bestScore >= dynamicThreshold) {
              target = best
            }
          }
        }
      }
      if (!target && headingTarget) target = headingTarget
      if (!target) {
        if (activeFocusSnippet || activeHeadingPath) {
          setLocateHint('未命中精确句段，建议重问一次以生成更细映射。')
        }
        return
      }
      target.classList.add('kb-reader-focus')
      target.scrollIntoView({ behavior: 'smooth', block: 'center' })
      if (!activeFocusSnippet && activeHeadingPath) {
        setLocateHint('已按章节定位。')
      }
    }, 80)
    return () => window.clearTimeout(timer)
  }, [open, markdown, activeHeadingPath, activeFocusSnippet, activeHighlightSnippet, activeAltIndex, activeAnchorId, activeBlockId, readerBlocks, alternatives, strictLocate])

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
          {activeHeadingPath ? `定位：${activeHeadingPath}` : '定位：文档开头'}
        </Text>
        {selection ? (
          <Text type="secondary" className="text-xs">
            已选中 {selection.length} 字
          </Text>
        ) : null}
      </div>
      {alternatives.length > 1 ? (
        <div className="mb-3 flex flex-wrap gap-2">
          {alternatives.map((item, idx) => {
            const label = shortCandidateLabel(item.headingPath || item.snippet || '')
            const isActive = idx === activeAltIndex
            return (
              <Button
                key={`${idx}:${item.headingPath}:${item.snippet.slice(0, 64)}`}
                size="small"
                type={isActive ? 'primary' : 'default'}
                onClick={() => setActiveAltIndex(idx)}
              >
                {label ? `定位点 ${idx + 1}: ${label}` : `定位点 ${idx + 1}`}
              </Button>
            )
          })}
        </div>
      ) : null}
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
          <MarkdownRenderer
            content={markdown}
            variant="reader"
            readerAnchors={readerAnchors}
            readerBlocks={readerBlocks}
          />
        </div>
      ) : (
        <Empty description="暂无可读内容" />
      )}
    </Drawer>
  )
}
