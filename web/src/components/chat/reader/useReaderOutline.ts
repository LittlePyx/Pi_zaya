import { useEffect, useMemo, useState, type RefObject } from 'react'
import type { ReaderDocBlock } from '../../../api/references'
import {
  clearReaderFocusClasses,
  headingMatchScore,
  normalizeText,
  resolveDirectTargetNode,
} from './readerDomUtils'

export interface ReaderOutlineItem {
  id: string
  label: string
  headingPath: string
  depth: number
  blockId: string
  anchorId: string
  order: number
}

interface UseReaderOutlineArgs {
  open: boolean
  sourcePath: string
  isInlinePresentation: boolean
  contentRef: RefObject<HTMLDivElement | null>
  readerBlocks: ReaderDocBlock[]
}

interface UseReaderOutlineResult {
  outlineItems: ReaderOutlineItem[]
  outlineOpen: boolean
  activeOutlineId: string
  activeOutlineHeadingPath: string
  hasOutline: boolean
  toggleOutline: () => void
  jumpToOutlineItem: (item: ReaderOutlineItem) => void
}

function normalizeHeadingPath(path: string) {
  return String(path || '')
    .split(' / ')
    .map((item) => item.trim())
    .filter(Boolean)
}

function deriveOutlineItems(readerBlocks: ReaderDocBlock[]): ReaderOutlineItem[] {
  const rows = Array.isArray(readerBlocks) ? [...readerBlocks] : []
  if (rows.length <= 0) return []
  rows.sort((left, right) => {
    const leftOrder = Number(left.order_index || left.line_start || 0)
    const rightOrder = Number(right.order_index || right.line_start || 0)
    return leftOrder - rightOrder
  })

  const headingRows = rows.filter((row) => String(row.kind || '').trim().toLowerCase() === 'heading')
  const sourceRows = headingRows.length > 0
    ? headingRows
    : rows.filter((row, index, list) => {
      const path = String(row.heading_path || '').trim()
      if (!path) return false
      return list.findIndex((candidate) => String(candidate.heading_path || '').trim() === path) === index
    })

  const seen = new Set<string>()
  return sourceRows.flatMap((row, index) => {
    const headingPath = String(row.heading_path || '').trim()
    if (!headingPath) return []
    const blockId = String(row.block_id || '').trim()
    const anchorId = String(row.anchor_id || '').trim()
    if (!blockId && !anchorId) return []
    const key = `${blockId}::${anchorId}::${headingPath}`
    if (seen.has(key)) return []
    seen.add(key)
    const parts = normalizeHeadingPath(headingPath)
    const label = String(row.text || '').trim() || parts[parts.length - 1] || headingPath
    const depth = Math.max(0, parts.length - 2)
    return [{
      id: blockId || anchorId || `outline-${index + 1}`,
      label,
      headingPath,
      depth,
      blockId,
      anchorId,
      order: Number(row.order_index || row.line_start || index),
    }]
  })
}

function resolveOutlineHeadingNode(
  root: HTMLElement,
  readerBlocks: ReaderDocBlock[],
  item: ReaderOutlineItem,
): HTMLElement | null {
  const headingCandidates = Array.from(root.querySelectorAll<HTMLElement>('h1,h2,h3,h4,h5,h6'))
  const parts = normalizeHeadingPath(item.headingPath)
  const exactNeedles = [item.label, parts[parts.length - 1] || '']
    .map(normalizeText)
    .filter(Boolean)
  for (const needle of exactNeedles) {
    const exactHeading = headingCandidates.find((heading) => normalizeText(String(heading.textContent || '')) === needle)
    if (exactHeading) return exactHeading
  }
  const needles = [item.label, parts[parts.length - 1] || '', item.headingPath]
    .map(normalizeText)
    .filter(Boolean)
  let bestHeading: HTMLElement | null = null
  let bestScore = 0
  for (const heading of headingCandidates) {
    const headingText = String(heading.textContent || '').trim()
    for (const needle of needles) {
      const score = headingMatchScore(needle, headingText)
      if (score > bestScore) {
        bestHeading = heading
        bestScore = score
      }
    }
  }
  if (bestHeading && bestScore >= 0.18) return bestHeading

  const direct = resolveDirectTargetNode(root, readerBlocks, {
    blockId: item.blockId,
    anchorId: item.anchorId,
    anchorKind: 'heading',
  }).target
  if (direct) return direct

  return (item.blockId
    ? root.querySelector<HTMLElement>(`[data-kb-block-id="${CSS.escape(item.blockId)}"]`)
    : null)
    || (item.anchorId
      ? root.querySelector<HTMLElement>(`[data-kb-anchor-id="${CSS.escape(item.anchorId)}"]`)
      : null)
}

function offsetWithinRoot(root: HTMLElement, target: HTMLElement): number {
  const rootRect = root.getBoundingClientRect()
  const targetRect = target.getBoundingClientRect()
  const rectTop = targetRect.top - rootRect.top + root.scrollTop
  if (Number.isFinite(rectTop)) return rectTop

  let offsetTop = 0
  let cursor: HTMLElement | null = target
  while (cursor && cursor !== root) {
    offsetTop += cursor.offsetTop
    cursor = cursor.offsetParent as HTMLElement | null
  }
  return offsetTop
}

function collectOutlinePositions(
  root: HTMLElement,
  readerBlocks: ReaderDocBlock[],
  outlineItems: ReaderOutlineItem[],
): Array<{ id: string; top: number }> {
  return outlineItems
    .flatMap((item) => {
      const target = resolveOutlineHeadingNode(root, readerBlocks, item)
      if (!target) return []
      return [{
        id: item.id,
        top: offsetWithinRoot(root, target),
      }]
    })
    .sort((left, right) => left.top - right.top)
}

function scrollOutlineHeadingIntoView(root: HTMLElement, target: HTMLElement) {
  const maxScrollTop = Math.max(0, root.scrollHeight - root.clientHeight)
  const targetTop = offsetWithinRoot(root, target)
  root.scrollTo({
    top: Math.max(0, Math.min(maxScrollTop, targetTop - 28)),
    behavior: 'auto',
  })
}

export function useReaderOutline({
  open,
  sourcePath,
  isInlinePresentation,
  contentRef,
  readerBlocks,
}: UseReaderOutlineArgs): UseReaderOutlineResult {
  const outlineItems = useMemo(() => deriveOutlineItems(readerBlocks), [readerBlocks])
  const [outlineOpen, setOutlineOpen] = useState(() => isInlinePresentation)
  const [activeOutlineId, setActiveOutlineId] = useState('')

  useEffect(() => {
    if (!open) return
    setOutlineOpen(isInlinePresentation)
  }, [open, sourcePath, isInlinePresentation])

  useEffect(() => {
    if (!open) {
      setActiveOutlineId('')
      return
    }
    setActiveOutlineId((current) => (
      current && outlineItems.some((item) => item.id === current)
        ? current
        : (outlineItems[0]?.id || '')
    ))
  }, [open, outlineItems])

  useEffect(() => {
    if (!open) return
    const root = contentRef.current
    if (!root) return
    if (outlineItems.length <= 0) {
      setActiveOutlineId('')
      return
    }

    let measureFrame = 0
    let syncFrame = 0
    let positions: Array<{ id: string; top: number }> = []

    const syncActive = () => {
      syncFrame = 0
      if (positions.length <= 0) return
      const bottomReached = root.scrollTop + root.clientHeight >= root.scrollHeight - 20
      if (bottomReached) {
        const lastId = positions[positions.length - 1]?.id || ''
        if (lastId) {
          setActiveOutlineId((current) => (current === lastId ? current : lastId))
        }
        return
      }

      const probeTop = root.scrollTop + Math.min(148, Math.max(52, root.clientHeight * 0.28))
      let activeIndex = 0
      for (let idx = 0; idx < positions.length; idx += 1) {
        if (positions[idx]!.top > probeTop) break
        activeIndex = idx
      }
      const nextId = positions[activeIndex]?.id || positions[0]?.id || ''
      if (nextId) {
        setActiveOutlineId((current) => (current === nextId ? current : nextId))
      }
    }

    const scheduleSync = () => {
      if (syncFrame) return
      syncFrame = window.requestAnimationFrame(syncActive)
    }

    const measurePositions = () => {
      measureFrame = 0
      positions = collectOutlinePositions(root, readerBlocks, outlineItems)
      scheduleSync()
    }

    const scheduleMeasure = () => {
      if (measureFrame) return
      measureFrame = window.requestAnimationFrame(measurePositions)
    }

    const resizeObserver = typeof ResizeObserver !== 'undefined'
      ? new ResizeObserver(() => {
        scheduleMeasure()
      })
      : null
    const mutationObserver = typeof MutationObserver !== 'undefined'
      ? new MutationObserver(() => {
        scheduleMeasure()
      })
      : null

    root.addEventListener('scroll', scheduleSync, { passive: true })
    window.addEventListener('resize', scheduleMeasure)
    resizeObserver?.observe(root)
    if (root.firstElementChild instanceof HTMLElement) {
      resizeObserver?.observe(root.firstElementChild)
    }
    mutationObserver?.observe(root, {
      childList: true,
      subtree: true,
      characterData: true,
      attributes: true,
    })
    scheduleMeasure()

    return () => {
      root.removeEventListener('scroll', scheduleSync)
      window.removeEventListener('resize', scheduleMeasure)
      resizeObserver?.disconnect()
      mutationObserver?.disconnect()
      if (measureFrame) window.cancelAnimationFrame(measureFrame)
      if (syncFrame) window.cancelAnimationFrame(syncFrame)
    }
  }, [open, contentRef, outlineItems, readerBlocks])

  const jumpToOutlineItem = (item: ReaderOutlineItem) => {
    const root = contentRef.current
    if (!root) return
    const target = resolveOutlineHeadingNode(root, readerBlocks, item)
    if (!target) return
    clearReaderFocusClasses(root)
    target.classList.add('kb-reader-focus')
    setActiveOutlineId(item.id)
    scrollOutlineHeadingIntoView(root, target)
  }

  return {
    outlineItems,
    outlineOpen,
    activeOutlineId,
    activeOutlineHeadingPath: outlineItems.find((item) => item.id === activeOutlineId)?.headingPath || '',
    hasOutline: outlineItems.length > 1,
    toggleOutline: () => setOutlineOpen((current) => !current),
    jumpToOutlineItem,
  }
}
