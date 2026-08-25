import { useEffect, type RefObject } from 'react'

import type { ReaderDocBlock } from '../../api/references'
import { orderedEquationReaderBlocks } from './reader/readerDomUtils'
import type { ReaderSelectionShelfPayload } from './reader/readerTypes'

export interface ReaderEquationShelfLabels {
  locate_badge_eq?: string
  reader_add_to_shelf?: string
  reader_add_to_shelf_title?: string
  reader_add_to_note?: string
  reader_add_to_note_title?: string
}

export interface BuildReaderEquationShelfPayloadOptions {
  block: ReaderDocBlock | null | undefined
  nodeText?: string
  now?: () => number
  sourceName: string
  sourcePath: string
  textLimit?: number
}

export interface UseReaderEquationShelfActionsOptions {
  contentRef: RefObject<HTMLDivElement | null>
  labels: ReaderEquationShelfLabels
  markdown: string
  now?: () => number
  onAddSelectionToShelf?: (payload: ReaderSelectionShelfPayload) => void
  onAddSelectionToResearchNote?: (payload: ReaderSelectionShelfPayload) => void
  open: boolean
  readerBlocks: ReaderDocBlock[]
  sourceName: string
  sourcePath: string
}

function compactEquationText(value: unknown): string {
  return String(value || '').replace(/\s+/g, ' ').trim()
}

export function buildReaderEquationShelfPayload({
  block,
  nodeText,
  now = Date.now,
  sourceName,
  sourcePath,
  textLimit = 1400,
}: BuildReaderEquationShelfPayloadOptions): ReaderSelectionShelfPayload | null {
  const text = compactEquationText(block?.text || block?.raw_text || nodeText)
  if (!text) return null
  const blockId = String(block?.block_id || '').trim()
  const anchorId = String(block?.anchor_id || '').trim()
  return {
    text: text.length <= textLimit ? text : `${text.slice(0, textLimit).trimEnd()}...`,
    sourcePath,
    sourceName,
    headingPath: String(block?.heading_path || '').trim() || undefined,
    blockId: blockId || undefined,
    anchorId: anchorId || undefined,
    anchorKind: 'equation',
    createdAt: now(),
    captureKind: 'equation',
  }
}

export function useReaderEquationShelfActions({
  contentRef,
  labels,
  markdown,
  now,
  onAddSelectionToShelf,
  onAddSelectionToResearchNote,
  open,
  readerBlocks,
  sourceName,
  sourcePath,
}: UseReaderEquationShelfActionsOptions) {
  useEffect(() => {
    if (!open || (!onAddSelectionToShelf && !onAddSelectionToResearchNote) || !contentRef.current || !sourcePath) return undefined
    const root = contentRef.current
    const equationBlocks = orderedEquationReaderBlocks(readerBlocks)
    if (equationBlocks.length <= 0) return undefined
    const equationNodes = Array.from(root.querySelectorAll<HTMLElement>('.katex-display'))
    if (equationNodes.length <= 0) return undefined
    const cleanup: Array<() => void> = []
    const limit = Math.min(equationNodes.length, equationBlocks.length)
    const label = labels.reader_add_to_shelf || 'Shelf'
    const titleLabel = labels.reader_add_to_shelf_title || 'Add to research basket'
    const kindLabel = labels.locate_badge_eq || 'Eq'
    for (let idx = 0; idx < limit; idx += 1) {
      const node = equationNodes[idx]
      const block = equationBlocks[idx]
      const blockId = String(block?.block_id || '').trim()
      const anchorId = String(block?.anchor_id || '').trim()
      if (!node || (!blockId && !anchorId)) continue
      if (node.querySelector(':scope > .kb-md-reader-block-shelf-tail[data-kb-imperative-equation="1"]')) continue
      node.classList.add('kb-md-reader-equation-action-host')
      node.setAttribute('data-kb-block-id', blockId)
      node.setAttribute('data-kb-anchor-id', anchorId)
      node.setAttribute('data-kb-anchor-kind', 'equation')
      const number = Number(block?.number || 0)
      if (Number.isFinite(number) && number > 0) {
        node.setAttribute('data-kb-anchor-number', String(Math.floor(number)))
      }
      const tail = document.createElement('span')
      tail.className = 'kb-md-reader-block-shelf-tail'
      tail.setAttribute('data-kb-imperative-equation', '1')
      const button = document.createElement('button')
      button.type = 'button'
      button.className = 'kb-md-reader-block-shelf kb-md-reader-block-shelf-equation'
      button.title = titleLabel
      button.setAttribute('aria-label', titleLabel)
      button.setAttribute('data-testid', 'reader-block-shelf')
      button.setAttribute('data-kb-reader-block-kind', 'equation')
      const kindSpan = document.createElement('span')
      kindSpan.className = 'kb-md-reader-block-shelf-kind'
      kindSpan.setAttribute('aria-hidden', 'true')
      kindSpan.textContent = kindLabel
      const textSpan = document.createElement('span')
      textSpan.className = 'kb-md-reader-block-shelf-text'
      textSpan.textContent = label
      button.append(kindSpan, textSpan)
      const handleClick = (event: Event) => {
        event.preventDefault()
        event.stopPropagation()
        const payload = buildReaderEquationShelfPayload({
          block,
          nodeText: node.textContent || '',
          now,
          sourceName,
          sourcePath,
        })
        if (!payload) return
        onAddSelectionToShelf?.(payload)
      }
      if (onAddSelectionToShelf) {
        button.addEventListener('click', handleClick)
        tail.appendChild(button)
      }
      const noteButton = document.createElement('button')
      noteButton.type = 'button'
      noteButton.className = 'kb-md-reader-block-shelf kb-md-reader-block-note kb-md-reader-block-shelf-equation'
      noteButton.title = labels.reader_add_to_note_title || 'Save to research note'
      noteButton.setAttribute('aria-label', noteButton.title)
      noteButton.setAttribute('data-testid', 'reader-block-note')
      noteButton.setAttribute('data-kb-reader-block-kind', 'equation')
      const noteKindSpan = kindSpan.cloneNode(true) as HTMLSpanElement
      const noteTextSpan = textSpan.cloneNode(true) as HTMLSpanElement
      noteTextSpan.textContent = labels.reader_add_to_note || 'Note'
      noteButton.append(noteKindSpan, noteTextSpan)
      const handleNoteClick = (event: Event) => {
        event.preventDefault()
        event.stopPropagation()
        const payload = buildReaderEquationShelfPayload({
          block,
          nodeText: node.textContent || '',
          now,
          sourceName,
          sourcePath,
        })
        if (!payload) return
        onAddSelectionToResearchNote?.(payload)
      }
      if (onAddSelectionToResearchNote) {
        noteButton.addEventListener('click', handleNoteClick)
        tail.appendChild(noteButton)
      }
      node.appendChild(tail)
      cleanup.push(() => {
        button.removeEventListener('click', handleClick)
        noteButton.removeEventListener('click', handleNoteClick)
        tail.remove()
        node.classList.remove('kb-md-reader-equation-action-host')
      })
    }
    return () => {
      cleanup.forEach((dispose) => dispose())
    }
  }, [
    contentRef,
    labels.locate_badge_eq,
    labels.reader_add_to_shelf,
    labels.reader_add_to_shelf_title,
    labels.reader_add_to_note,
    labels.reader_add_to_note_title,
    markdown,
    now,
    onAddSelectionToShelf,
    onAddSelectionToResearchNote,
    open,
    readerBlocks,
    sourceName,
    sourcePath,
  ])
}
