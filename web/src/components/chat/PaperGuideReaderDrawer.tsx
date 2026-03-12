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
  relatedBlockIds?: string[]
  anchorKind?: string
  anchorNumber?: number
  strictLocate?: boolean
  locateRequestId?: number
  alternatives?: Array<{
    headingPath?: string
    snippet?: string
    highlightSnippet?: string
    anchorId?: string
    blockId?: string
    anchorKind?: string
    anchorNumber?: number
  }>
  initialAltIndex?: number
}

interface Props {
  open: boolean
  payload: ReaderOpenPayload | null
  onClose: () => void
  onAppendSelection: (text: string) => void
}

interface StickyLocateHighlight {
  blockId: string
  anchorId: string
  anchorKind: string
  anchorNumber: number
  headingPath: string
  highlightSeed: string
  highlightQueries: string[]
  relatedBlockIds: string[]
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
  const displayEquation = node.closest('.katex-display')
  if (displayEquation) return displayEquation as HTMLElement
  return node.closest(
    'p,li,blockquote,pre,code,figcaption,td,th,.katex-display,.katex,[data-kb-anchor-kind="figure"],h1,h2,h3,h4,h5,h6',
  ) as HTMLElement | null
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

function extractFigureNumbers(text: string): number[] {
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
  for (const m of src.matchAll(/图\s*(\d{1,4})\b/g)) {
    push(String(m[1] || ''))
  }
  return out
}

function extractQuotedSpans(text: string, minLen = 12): string[] {
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

function clearReaderInlineHits(root: HTMLElement | null) {
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

function clearReaderFocusClasses(root: HTMLElement | null) {
  if (!root) return
  root.querySelectorAll<HTMLElement>('.kb-reader-focus, .kb-reader-focus-secondary')
    .forEach((node) => {
      node.classList.remove('kb-reader-focus')
      node.classList.remove('kb-reader-focus-secondary')
    })
}

function buildTextNodeCorpus(container: HTMLElement): { raw: string; nodes: Array<{ node: Text; start: number; end: number }> } {
  const nodes: Array<{ node: Text; start: number; end: number }> = []
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

function rawOffsetToDomPoint(
  segments: Array<{ node: Text; start: number; end: number }>,
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

function highlightExactTextInContainer(container: HTMLElement, query: string): HTMLElement | null {
  const probe = String(query || '').replace(/\s+/g, ' ').trim()
  if (!probe || probe.length < 8) return null
  const corpus = buildTextNodeCorpus(container)
  if (!corpus.raw) return null
  const normCorpus = normalizeWithMap(corpus.raw)
  const normQuery = normalizeWithMap(probe).norm
  if (!normCorpus.norm || !normQuery) return null
  const hitAt = normCorpus.norm.indexOf(normQuery)
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
    wrapper.className = 'kb-reader-inline-hit'
    const frag = range.extractContents()
    wrapper.appendChild(frag)
    range.insertNode(wrapper)
    return wrapper
  } catch {
    return null
  }
}

function buildHighlightQueries(text: string, opts?: { anchorKind?: string; anchorNumber?: number }): string[] {
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
    push(`公式 ${anchorNumber}`)
    push(`公式(${anchorNumber})`)
  }
  if (anchorKind === 'figure' && anchorNumber > 0) {
    push(`Figure ${anchorNumber}`)
    push(`Fig. ${anchorNumber}`)
    push(`图${anchorNumber}`)
  }
  return out
}

function nearbyReadableBlocks(root: HTMLElement, target: HTMLElement, maxDistance = 2): HTMLElement[] {
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

function scrollReaderTargetIntoView(root: HTMLElement, target: HTMLElement) {
  const rootRect = root.getBoundingClientRect()
  const targetRect = target.getBoundingClientRect()
  const maxScrollTop = Math.max(0, root.scrollHeight - root.clientHeight)
  let nextTop = root.scrollTop

  if (rootRect.height > 0 && targetRect.height > 0) {
    const relativeTop = targetRect.top - rootRect.top
    nextTop = root.scrollTop + relativeTop - ((root.clientHeight - targetRect.height) / 2)
  } else {
    let offsetTop = 0
    let cursor: HTMLElement | null = target
    while (cursor && cursor !== root) {
      offsetTop += cursor.offsetTop
      cursor = cursor.offsetParent as HTMLElement | null
    }
    nextTop = offsetTop - (root.clientHeight / 2)
  }

  root.scrollTo({
    top: Math.max(0, Math.min(maxScrollTop, nextTop)),
    behavior: 'smooth',
  })
}

function resolveStickyHighlightTarget(
  root: HTMLElement,
  readerBlocks: ReaderDocBlock[],
  sticky: StickyLocateHighlight,
): HTMLElement | null {
  const direct = resolveDirectTargetNode(root, readerBlocks, {
    blockId: sticky.blockId,
    anchorId: sticky.anchorId,
    anchorKind: sticky.anchorKind,
  })
  if (direct.target) return direct.target

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
    let score = snippetMatchScore(seed, text)
    if (score > bestScore) {
      best = block
      bestScore = score
    }
  }
  return bestScore >= 0.12 ? best : null
}

function visibleEquationBlocks(root: HTMLElement): HTMLElement[] {
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

function orderedEquationReaderBlocks(readerBlocks: ReaderDocBlock[]): ReaderDocBlock[] {
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

function bindVisibleEquationAnchors(root: HTMLElement, readerBlocks: ReaderDocBlock[]): number {
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

function resolveDirectTargetNode(
  root: HTMLElement,
  readerBlocks: ReaderDocBlock[],
  opts: { blockId?: string; anchorId?: string; anchorKind?: string },
): { target: HTMLElement | null; hintBlock: ReaderDocBlock | null } {
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

function resolveRelatedTargetNodes(
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
  const stickyLocateHighlightRef = useRef<StickyLocateHighlight | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [markdown, setMarkdown] = useState('')
  const [readerAnchors, setReaderAnchors] = useState<ReaderDocAnchor[]>([])
  const [readerBlocks, setReaderBlocks] = useState<ReaderDocBlock[]>([])
  const [resolvedName, setResolvedName] = useState('')
  const [selection, setSelection] = useState('')
  const [locateHint, setLocateHint] = useState('')
  const [drawerReady, setDrawerReady] = useState(false)
  const [equationBindingReady, setEquationBindingReady] = useState(false)
  const [equationBindingBoundCount, setEquationBindingBoundCount] = useState(0)

  const sourcePath = String(payload?.sourcePath || '').trim()
  const sourceName = String(payload?.sourceName || '').trim()
  const headingPath = String(payload?.headingPath || '').trim()
  const focusSnippet = String(payload?.snippet || '').trim()
  const highlightSnippet = String(payload?.highlightSnippet || '').trim()
  const anchorId = String(payload?.anchorId || '').trim()
  const blockId = String(payload?.blockId || '').trim()
  const relatedBlockIds = Array.isArray(payload?.relatedBlockIds)
    ? payload.relatedBlockIds.map((item) => String(item || '').trim()).filter(Boolean)
    : []
  const strictLocate = Boolean(payload?.strictLocate)
  const locateRequestId = Number.isFinite(Number(payload?.locateRequestId || 0))
    ? Math.max(0, Math.floor(Number(payload?.locateRequestId || 0)))
    : 0
  const alternatives = useMemo(() => {
    const listRaw = Array.isArray(payload?.alternatives) ? payload?.alternatives : []
    const out: Array<{
      headingPath: string
      snippet: string
      highlightSnippet: string
      anchorId: string
      blockId: string
      anchorKind: string
      anchorNumber: number
    }> = []
    const seen = new Set<string>()
    const push = (
      headingPath0: string,
      snippet0: string,
      highlightSnippet0: string,
      anchorId0: string,
      blockId0: string,
      anchorKind0: string,
      anchorNumber0: number,
    ) => {
      const heading = String(headingPath0 || '').trim()
      const snippet = String(snippet0 || '').trim()
      const highlightSnippet = String(highlightSnippet0 || '').trim()
      const anchorId = String(anchorId0 || '').trim()
      const blockId = String(blockId0 || '').trim()
      const anchorKind = String(anchorKind0 || '').trim().toLowerCase()
      const anchorNumber = Number.isFinite(Number(anchorNumber0)) ? Math.floor(Number(anchorNumber0)) : 0
      if (!heading && !snippet && !highlightSnippet && !anchorId && !blockId && !anchorKind && anchorNumber <= 0) return
      const key = `${blockId.toLowerCase()}::${anchorId.toLowerCase()}::${anchorKind}::${anchorNumber}::${heading.toLowerCase()}::${highlightSnippet.toLowerCase().slice(0, 180)}::${snippet.toLowerCase().slice(0, 180)}`
      if (seen.has(key)) return
      seen.add(key)
      out.push({ headingPath: heading, snippet, highlightSnippet, anchorId, blockId, anchorKind, anchorNumber })
    }
    push(headingPath, focusSnippet, highlightSnippet, anchorId, blockId, String(payload?.anchorKind || ''), Number(payload?.anchorNumber || 0))
    for (const item of listRaw) {
      if (!item || typeof item !== 'object') continue
      push(
        String(item.headingPath || ''),
        String(item.snippet || ''),
        String(item.highlightSnippet || ''),
        String(item.anchorId || ''),
        String(item.blockId || ''),
        String(item.anchorKind || ''),
        Number(item.anchorNumber || 0),
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
  const activeAnchorKind = String(activeAlt?.anchorKind || payload?.anchorKind || '').trim().toLowerCase()
  const activeAnchorNumber = Number.isFinite(Number(activeAlt?.anchorNumber || payload?.anchorNumber || 0))
    ? Math.floor(Number(activeAlt?.anchorNumber || payload?.anchorNumber || 0))
    : 0
  const expectsEquationBinding = useMemo(() => {
    if (activeAnchorKind === 'equation') return true
    if (alternatives.some((item) => String(item?.anchorKind || '').trim().toLowerCase() === 'equation')) return true
    return false
  }, [activeAnchorKind, alternatives])

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
    if (!open) {
      setDrawerReady(false)
    }
  }, [open])

  useEffect(() => {
    if (!open) return
    if (drawerReady) return
    // Fallback: some environments may not reliably emit Drawer.afterOpenChange.
    const timer = window.setTimeout(() => {
      setDrawerReady(true)
    }, 240)
    return () => {
      window.clearTimeout(timer)
    }
  }, [open, drawerReady, locateRequestId, sourcePath])

  useEffect(() => {
    if (!open) return
    stickyLocateHighlightRef.current = null
    setLocateHint('')
    const root = contentRef.current
    if (!root) return
    clearReaderFocusClasses(root)
    clearReaderInlineHits(root)
  }, [open, locateRequestId])

  useEffect(() => {
    if (!open || !drawerReady || !markdown) return
    const sticky = stickyLocateHighlightRef.current
    if (!sticky) return
    const root = contentRef.current
    if (!root) return

    clearReaderFocusClasses(root)
    const target = resolveStickyHighlightTarget(root, readerBlocks, sticky)
    if (!target) return
    const focusedBlock = closestReadableBlock(target) || target
    focusedBlock.classList.add('kb-reader-focus')
    resolveRelatedTargetNodes(root, readerBlocks, sticky.relatedBlockIds || [], focusedBlock)
      .forEach((node) => node.classList.add('kb-reader-focus-secondary'))
    if (sticky.highlightQueries.length > 0 && !root.querySelector('.kb-reader-inline-hit')) {
      for (const query of sticky.highlightQueries) {
        const hit = highlightExactTextInContainer(focusedBlock, query)
        if (hit) break
      }
    }
  }, [open, drawerReady, markdown, readerBlocks, equationBindingReady, locateHint])

  useEffect(() => {
    if (!open || !drawerReady) {
      setEquationBindingReady(false)
      setEquationBindingBoundCount(0)
      return
    }
    if (!markdown) {
      setEquationBindingReady(false)
      setEquationBindingBoundCount(0)
      return
    }
    if (!expectsEquationBinding) {
      setEquationBindingReady(true)
      setEquationBindingBoundCount(0)
      return
    }
    setEquationBindingReady(false)
    setEquationBindingBoundCount(0)
  }, [open, drawerReady, markdown, expectsEquationBinding, locateRequestId, sourcePath])

  useEffect(() => {
    if (!open || !drawerReady || !markdown) return
    if (!expectsEquationBinding) return
    const equationBlockCount = orderedEquationReaderBlocks(readerBlocks).length
    if (equationBlockCount <= 0) {
      setEquationBindingReady(true)
      setEquationBindingBoundCount(0)
      return
    }
    let cancelled = false
    let raf = 0
    let timer = 0
    let observer: MutationObserver | null = null
    let lastVisibleCount = -1
    let stablePasses = 0
    const deadline = Date.now() + (strictLocate ? 2600 : 1600)
    const finalize = (boundCount: number) => {
      if (cancelled) return
      observer?.disconnect()
      window.cancelAnimationFrame(raf)
      window.clearTimeout(timer)
      setEquationBindingBoundCount(boundCount)
      setEquationBindingReady(true)
    }
    const scheduleBind = (delayMs = 0) => {
      if (cancelled) return
      window.cancelAnimationFrame(raf)
      window.clearTimeout(timer)
      const trigger = () => {
        if (cancelled) return
        raf = window.requestAnimationFrame(bind)
      }
      if (delayMs > 0) {
        timer = window.setTimeout(trigger, delayMs)
      } else {
        trigger()
      }
    }
    const bind = () => {
      if (cancelled) return
      const root = contentRef.current
      if (!root) {
        if (Date.now() < deadline) {
          scheduleBind(80)
          return
        }
        finalize(0)
        return
      }
      const boundCount = bindVisibleEquationAnchors(root, readerBlocks)
      const visibleCount = root.querySelectorAll('.katex-display').length
      const targetBindCount = Math.min(Math.max(0, visibleCount), Math.max(0, equationBlockCount))
      const bindingSatisfied = targetBindCount <= 0 || boundCount >= targetBindCount
      if (visibleCount === lastVisibleCount && bindingSatisfied) {
        stablePasses += 1
      } else {
        stablePasses = 0
      }
      lastVisibleCount = visibleCount
      if (bindingSatisfied && stablePasses >= 1) {
        finalize(boundCount)
        return
      }
      if (Date.now() < deadline) {
        scheduleBind(bindingSatisfied ? 40 : 90)
        return
      }
      finalize(boundCount)
    }
    const root = contentRef.current
    if (root) {
      observer = new MutationObserver(() => {
        if (cancelled) return
        scheduleBind(20)
      })
      observer.observe(root, { childList: true, subtree: true })
    }
    scheduleBind(0)
    return () => {
      cancelled = true
      observer?.disconnect()
      window.cancelAnimationFrame(raf)
      window.clearTimeout(timer)
    }
  }, [open, drawerReady, markdown, readerBlocks, expectsEquationBinding, strictLocate, locateRequestId])

  useEffect(() => {
    if (!open || !drawerReady || !markdown) return
    if (expectsEquationBinding && !equationBindingReady) return
    let cancelled = false
    let attempts = 0
    let locateRaf = 0
    let scrollRaf = 0
    let retryTimer = 0
    let observer: MutationObserver | null = null
    const deadline = Date.now() + (strictLocate ? 2800 : 1800)
    const scheduleLocate = (delayMs = 0) => {
      if (cancelled) return
      window.cancelAnimationFrame(locateRaf)
      window.clearTimeout(retryTimer)
      const trigger = () => {
        if (cancelled) return
        locateRaf = window.requestAnimationFrame(runLocate)
      }
      if (delayMs > 0) {
        retryTimer = window.setTimeout(trigger, delayMs)
      } else {
        trigger()
      }
    }
    const finishLocate = () => {
      observer?.disconnect()
      window.cancelAnimationFrame(locateRaf)
      window.clearTimeout(retryTimer)
    }
    const retryLocate = () => {
      if (Date.now() >= deadline) return false
      attempts += 1
      scheduleLocate(Math.min(60 + attempts * 35, 220))
      return true
    }
    const runLocate = () => {
      if (cancelled) return
      const root = contentRef.current
      if (!root || root.clientHeight <= 0 || root.scrollHeight <= 0) {
        retryLocate()
        return
      }
      if (expectsEquationBinding) bindVisibleEquationAnchors(root, readerBlocks)
      clearReaderFocusClasses(root)
      clearReaderInlineHits(root)

      const directResolved = resolveDirectTargetNode(root, readerBlocks, {
        blockId: activeBlockId,
        anchorId: activeAnchorId,
        anchorKind: activeAnchorKind,
      })
      let target: HTMLElement | null = directResolved.target
      let headingTarget: HTMLElement | null = null
      const readerBlockHint = directResolved.hintBlock
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

          const altResolved = resolveDirectTargetNode(root, readerBlocks, {
            blockId: altBlockId,
            anchorId: altAnchorId,
          })
          if (altResolved.target) {
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
        if (retryLocate()) return
        setLocateHint('Exact evidence block not found. Falling back to fuzzy locate.')
      }
      if (!target && strictLocate) {
        if (retryLocate()) return
        setLocateHint('Exact evidence block not found. Falling back to fuzzy locate.')
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
          const hintKind = String(readerBlockHint?.kind || '').trim().toLowerCase()
          const eqNumbersAll = [
            ...extractEquationNumbers(`${activeFocusSnippet} ${activeHeadingPath}`),
            ...extractEquationNumbers(`${hintBlockText} ${hintHeadingPath}`),
          ]
          const hintNumber = Number(readerBlockHint?.number || 0)
          if (Number.isFinite(hintNumber) && hintNumber > 0) {
            eqNumbersAll.push(Math.floor(hintNumber))
          }
          const eqNumbers = Array.from(new Set(eqNumbersAll.filter((item) => Number.isFinite(item) && item > 0)))
          const figNumbersAll = [
            ...extractFigureNumbers(`${activeFocusSnippet} ${activeHeadingPath}`),
            ...extractFigureNumbers(`${hintBlockText} ${hintHeadingPath}`),
          ]
          if (Number.isFinite(hintNumber) && hintNumber > 0 && (activeAnchorKind === 'figure' || hintKind === 'figure')) {
            figNumbersAll.push(Math.floor(hintNumber))
          }
          const figNumbers = Array.from(new Set(figNumbersAll.filter((item) => Number.isFinite(item) && item > 0)))
          const preferFormula = Boolean(
            hasFormulaSignal(probe)
            || hasFormulaSignal(hintBlockText)
            || hintKind === 'equation'
            || activeAnchorKind === 'equation'
          )
          const preferFigure = Boolean(activeAnchorKind === 'figure' || hintKind === 'figure' || figNumbers.length > 0)
          if (!target && preferFigure && Array.isArray(readerBlocks) && readerBlocks.length > 0) {
            const figBlock = readerBlocks.find((item) => {
              if (String(item?.kind || '').trim().toLowerCase() !== 'figure') return false
              const blockNumber = Number(item?.number || 0)
              return figNumbers.length <= 0 || (Number.isFinite(blockNumber) && figNumbers.includes(Math.floor(blockNumber)))
            }) || null
            if (figBlock) {
              target = root.querySelector<HTMLElement>(`[data-kb-block-id="${CSS.escape(String(figBlock.block_id || '').trim())}"]`)
              if (!target) {
                target = Array.from(root.querySelectorAll<HTMLElement>('[data-kb-anchor-id]'))
                  .find((node) => String(node.getAttribute('data-kb-anchor-id') || '') === String(figBlock.anchor_id || '').trim()) || null
              }
            }
          }
          const equationBlocks = visibleEquationBlocks(root)
          const blocks = preferFormula
            ? equationBlocks
            : Array.from(root.querySelectorAll<HTMLElement>('p,li,blockquote,pre,code,figcaption,td,th,.katex-display,[data-kb-anchor-kind="figure"]'))
          const allNodes = Array.from(root.querySelectorAll<HTMLElement>('h1,h2,h3,h4,h5,h6,p,li,blockquote,pre,code,figcaption,td,th,.katex-display,[data-kb-anchor-kind="equation"],[data-kb-anchor-kind="figure"]'))
          const nodeIndex = new Map<HTMLElement, number>()
          allNodes.forEach((node, idx) => nodeIndex.set(node, idx))
          const headingIndex = headingTarget ? Number(nodeIndex.get(headingTarget) ?? -1) : -1

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
        const anyReadable = root.querySelector<HTMLElement>('h1,h2,h3,p,li,blockquote,.katex-display,[data-kb-anchor-kind="equation"],[data-kb-anchor-kind="figure"]')
        if (anyReadable) {
          target = anyReadable
          setLocateHint((prev) => prev || 'Fuzzy locate fallback used.')
        }
      }
      if (!target) {
        if (retryLocate()) return
        if (activeFocusSnippet || activeHeadingPath) {
          setLocateHint('Exact snippet not found. Ask again to generate a finer mapping.')
        }
        finishLocate()
        return
      }

      const anchorKindForLocate = String(activeAnchorKind || readerBlockHint?.kind || '').trim().toLowerCase()
      const anchorNumberForLocate = Number.isFinite(Number(activeAnchorNumber || readerBlockHint?.number || 0))
        ? Math.floor(Number(activeAnchorNumber || readerBlockHint?.number || 0))
        : 0
      let nextLocateHint = ''
      const highlightSeed = String(activeHighlightSnippet || activeFocusSnippet || readerBlockHint?.text || '').trim()
      const highlightQueries = buildHighlightQueries(highlightSeed, {
        anchorKind: anchorKindForLocate,
        anchorNumber: anchorNumberForLocate,
      })
      const tryExactHighlight = (container: HTMLElement | null): HTMLElement | null => {
        if (!container || highlightQueries.length <= 0) return null
        for (const query of highlightQueries) {
          const hit = highlightExactTextInContainer(container, query)
          if (hit) return hit
        }
        return null
      }

      let focusedBlock = closestReadableBlock(target) || target
      let exactHit: HTMLElement | null = null
      let usedNeighbor = false
      if (anchorKindForLocate !== 'figure' && anchorKindForLocate !== 'equation' && highlightQueries.length > 0) {
        exactHit = tryExactHighlight(focusedBlock)
        if (!exactHit && strictLocate) {
          const maxDistance = anchorKindForLocate === 'equation' ? 1 : 2
          for (const neighbor of nearbyReadableBlocks(root, focusedBlock, maxDistance)) {
            const hit = tryExactHighlight(neighbor)
            if (!hit) continue
            focusedBlock = closestReadableBlock(neighbor) || neighbor
            exactHit = hit
            usedNeighbor = true
            break
          }
        }
      }

      focusedBlock.classList.add('kb-reader-focus')
      const relatedTargets = resolveRelatedTargetNodes(root, readerBlocks, relatedBlockIds, focusedBlock)
      relatedTargets.forEach((node) => node.classList.add('kb-reader-focus-secondary'))
      const focusNode = exactHit || focusedBlock
      stickyLocateHighlightRef.current = {
        blockId: String(focusedBlock.getAttribute('data-kb-block-id') || target.getAttribute('data-kb-block-id') || activeBlockId || readerBlockHint?.block_id || '').trim(),
        anchorId: String(focusedBlock.getAttribute('data-kb-anchor-id') || target.getAttribute('data-kb-anchor-id') || activeAnchorId || readerBlockHint?.anchor_id || '').trim(),
        anchorKind: anchorKindForLocate,
        anchorNumber: anchorNumberForLocate,
        headingPath: String(activeHeadingPath || readerBlockHint?.heading_path || '').trim(),
        highlightSeed,
        highlightQueries: anchorKindForLocate === 'equation' || anchorKindForLocate === 'figure'
          ? []
          : [...highlightQueries],
        relatedBlockIds: [...relatedBlockIds],
      }
      scrollRaf = window.requestAnimationFrame(() => {
        if (cancelled) return
        scrollRaf = window.requestAnimationFrame(() => {
          if (!cancelled) {
            scrollReaderTargetIntoView(root, focusNode)
            focusNode.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'nearest' })
            if (nextLocateHint) {
              window.setTimeout(() => {
                if (cancelled) return
                setLocateHint(nextLocateHint)
                window.requestAnimationFrame(() => {
                  if (cancelled) return
                  focusedBlock.classList.add('kb-reader-focus')
                  relatedTargets.forEach((node) => node.classList.add('kb-reader-focus-secondary'))
                })
              }, 90)
            }
          }
        })
      })

      if (strictLocate) {
        if (exactHit) {
          if (anchorKindForLocate === 'figure') {
            nextLocateHint = 'Exact figure block match.'
          } else {
            nextLocateHint = 'Exact source phrase match.'
          }
        } else if (anchorKindForLocate === 'figure') {
          nextLocateHint = 'Figure block matched.'
        } else if (anchorKindForLocate === 'equation') {
          nextLocateHint = 'Equation block matched.'
        } else if (highlightQueries.length > 0) {
          nextLocateHint = usedNeighbor ? 'Neighbor evidence block matched, but exact inline phrase was not found.' : 'Evidence block matched, but exact inline phrase was not found.'
        } else {
          nextLocateHint = 'Evidence block matched.'
        }
      } else if (!activeFocusSnippet && activeHeadingPath) {
        nextLocateHint = 'Located by heading.'
      }
      finishLocate()
    }
    const root = contentRef.current
    if (root) {
      observer = new MutationObserver(() => {
        if (cancelled) return
        scheduleLocate(20)
      })
      observer.observe(root, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['data-kb-block-id', 'data-kb-anchor-id', 'class'],
      })
    }
    scheduleLocate(0)
    return () => {
      cancelled = true
      finishLocate()
      window.cancelAnimationFrame(scrollRaf)
    }
  }, [
    open,
    drawerReady,
    markdown,
    locateRequestId,
    activeHeadingPath,
    activeFocusSnippet,
    activeHighlightSnippet,
    activeAltIndex,
    activeAnchorId,
    activeBlockId,
    activeAnchorKind,
    activeAnchorNumber,
    readerBlocks,
    alternatives,
    relatedBlockIds,
    strictLocate,
    expectsEquationBinding,
    equationBindingReady,
  ])

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
      afterOpenChange={setDrawerReady}
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
      ) : (expectsEquationBinding && !equationBindingReady) ? (
        <div className="mb-3">
          <Text type="secondary" className="text-xs">
            正在绑定公式锚点{equationBindingBoundCount > 0 ? `（已绑定 ${equationBindingBoundCount} 个）` : ''}
          </Text>
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
