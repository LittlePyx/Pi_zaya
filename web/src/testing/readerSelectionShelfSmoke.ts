import { createElement, useLayoutEffect } from 'react'
import { flushSync } from 'react-dom'
import { createRoot } from 'react-dom/client'

import type { ReaderSelectionState } from '../components/chat/reader/readerDomUtils'
import type {
  ReaderSelectionShelfPayload,
  ReaderSessionHighlight,
} from '../components/chat/reader/readerTypes'
import {
  buildReaderHighlightShelfPayload,
  buildReaderSelectionShelfPayload,
  useReaderSelectionShelf,
  type ReaderSelectionShelfController,
} from '../components/chat/useReaderSelectionShelf'

export interface ReaderSelectionShelfSmokeResult {
  clearEvents: string[]
  closeEvents: string[]
  directHighlightPayload: ReaderSelectionShelfPayload | null
  directSelectionPayload: ReaderSelectionShelfPayload | null
  events: ReaderSelectionShelfPayload[]
  invalidSelectionPayload: ReaderSelectionShelfPayload | null
  renderedText: string
}

const selectionBubble = {
  anchorId: ' selected-anchor ',
  blockId: ' selected-block ',
  canHighlight: true,
  documentOccurrence: 6,
  endOffset: 13,
  endReadableIndex: 5,
  highlightId: '',
  occurrence: 2,
  readableIndex: 4,
  startOffset: 3,
  startReadableIndex: 4,
  text: ' Selected text ',
  x: 10,
  y: 20,
} as ReaderSelectionState

const activeHighlight = {
  anchorId: ' highlight-anchor ',
  blockId: ' highlight-block ',
  documentOccurrence: 3,
  endOffset: 9,
  endReadableIndex: 4,
  headingPath: ' Highlight Heading ',
  id: 'highlight-a',
  occurrence: 1,
  readableIndex: 2,
  startOffset: 5,
  startReadableIndex: 2,
  text: ' Highlight text ',
} as ReaderSessionHighlight

const baseOptions = {
  activeAnchorId: ' active-anchor ',
  activeAnchorKind: ' paragraph ',
  activeBlockId: ' active-block ',
  activeHeadingPath: ' Active Heading ',
  sourceName: 'Reader Paper',
  sourcePath: '/tmp/reader.md',
}

export function runReaderSelectionShelfSmoke(): ReaderSelectionShelfSmokeResult {
  const host = document.createElement('div')
  document.body.append(host)
  const root = createRoot(host)
  const clearEvents: string[] = []
  const closeEvents: string[] = []
  const events: ReaderSelectionShelfPayload[] = []
  let controller: ReaderSelectionShelfController | null = null

  const directSelectionPayload = buildReaderSelectionShelfPayload({
    ...baseOptions,
    now: () => 12345,
    selection: 'fallback',
    selectionBubble,
  })
  const directHighlightPayload = buildReaderHighlightShelfPayload({
    ...baseOptions,
    activeHighlight,
    now: () => 23456,
  })
  const invalidSelectionPayload = buildReaderSelectionShelfPayload({
    ...baseOptions,
    now: () => 12345,
    selection: 'orphan text',
    selectionBubble: null,
  })

  const readController = () => {
    if (!controller) throw new Error('reader selection shelf smoke did not mount')
    return controller
  }

  function Harness() {
    const shelf = useReaderSelectionShelf({
      ...baseOptions,
      activeHighlight,
      now: () => 34567,
      onAddSelectionToShelf: (payload) => {
        events.push(payload)
      },
      onClearSelection: (clearNative) => {
        clearEvents.push(String(clearNative))
      },
      onCloseHighlight: () => {
        closeEvents.push('close')
      },
      selection: 'fallback',
      selectionBubble,
    })
    useLayoutEffect(() => {
      controller = shelf
    })
    return createElement(
      'div',
      { id: 'reader-selection-shelf-smoke' },
      [
        String(shelf.canAddSelectionToShelf),
        String(Boolean(shelf.addActiveHighlightToShelf)),
      ].join('|'),
    )
  }

  flushSync(() => {
    root.render(createElement(Harness))
  })
  flushSync(() => {
    readController().addSelectionToShelf()
  })
  flushSync(() => {
    readController().addActiveHighlightToShelf?.()
  })

  const renderedText = host.textContent || ''
  root.unmount()
  host.remove()

  return {
    clearEvents,
    closeEvents,
    directHighlightPayload,
    directSelectionPayload,
    events,
    invalidSelectionPayload,
    renderedText,
  }
}
