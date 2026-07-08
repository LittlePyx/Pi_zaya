import { createElement, useLayoutEffect } from 'react'
import { flushSync } from 'react-dom'
import { createRoot } from 'react-dom/client'

import type { ReaderSessionHighlight } from '../components/chat/reader/readerTypes'
import {
  enrichReaderSessionHighlight,
  sameHighlightUndoAction,
  useReaderHighlightActions,
  type ReaderHighlightActionsController,
  type ReaderHighlightMessenger,
} from '../components/chat/useReaderHighlightActions'

export interface ReaderHighlightActionsSmokeResult {
  addedHighlights: ReaderSessionHighlight[]
  appendedText: string[]
  closeCount: number
  directEnriched: ReaderSessionHighlight
  differentUndoMatch: boolean
  feedbackUpdates: ReaderSessionHighlight[]
  messageEvents: string[]
  removedIds: string[]
  renderedText: string
  sameUndoMatch: boolean
  undoAfterClear: boolean
}

const existingHighlight = {
  blockId: 'block-existing',
  endOffset: 22,
  headingPath: 'Existing Heading',
  id: 'existing',
  occurrence: 1,
  readableIndex: 2,
  startOffset: 10,
  text: 'Existing quote\nSecond line',
} as ReaderSessionHighlight

const newHighlight = {
  blockId: 'block-new',
  endOffset: 18,
  id: 'new-highlight',
  startOffset: 4,
  text: 'New quote',
} as ReaderSessionHighlight

const baseOptions = {
  activeHeadingPath: 'Active Heading',
  conversationId: 'conv-reader',
  locateFeedbackKey: 'locate-key',
  locateRequestId: 12,
  messageId: 99,
  sourceLabel: 'Reader Title / Active Heading',
  sourceName: 'Fallback Source',
  sourcePath: '/tmp/reader.md',
  title: 'Reader Title',
}

export function runReaderHighlightActionsSmoke(): ReaderHighlightActionsSmokeResult {
  const host = document.createElement('div')
  document.body.append(host)
  const root = createRoot(host)
  const addedHighlights: ReaderSessionHighlight[] = []
  const appendedText: string[] = []
  const feedbackUpdates: ReaderSessionHighlight[] = []
  const messageEvents: string[] = []
  const removedIds: string[] = []
  let closeCount = 0
  let controller: ReaderHighlightActionsController | null = null

  const messenger: ReaderHighlightMessenger = {
    open: (config) => {
      messageEvents.push(`open:${config.type}`)
    },
    success: (content) => {
      messageEvents.push(`success:${String(content)}`)
    },
  }

  const directEnriched = enrichReaderSessionHighlight(
    { id: 'direct', text: 'Direct quote' },
    {
      activeHeadingPath: 'Direct Heading',
      conversationId: 'conv-direct',
      locateFeedbackKey: 'locate-direct',
      locateRequestId: 77,
      messageId: 43,
      now: () => 12345,
      sourceName: 'Direct Fallback',
      sourcePath: '/tmp/direct.md',
      title: 'Direct Title',
    },
  )
  const sameUndoMatch = sameHighlightUndoAction(
    { kind: 'remove', highlight: newHighlight },
    { kind: 'remove', highlight: { ...newHighlight } },
  )
  const differentUndoMatch = sameHighlightUndoAction(
    { kind: 'remove', highlight: newHighlight },
    { kind: 'restore', highlight: newHighlight },
  )

  const readController = () => {
    if (!controller) throw new Error('reader highlight actions smoke did not mount')
    return controller
  }

  function Harness() {
    const actions = useReaderHighlightActions({
      ...baseOptions,
      activeHighlight: existingHighlight,
      labels: {
        reader_feedback_saved: 'Evidence note updated',
        reader_highlight_removed: 'Highlight removed',
        reader_undo: 'Undo',
      },
      messenger,
      now: () => 45678,
      onAddSessionHighlight: (highlight) => {
        addedHighlights.push(highlight)
      },
      onAppendSelection: (text) => {
        appendedText.push(text)
      },
      onCloseHighlight: () => {
        closeCount += 1
      },
      onRemoveSessionHighlight: (highlightId) => {
        removedIds.push(highlightId)
      },
      onUpdateSessionHighlight: (highlight) => {
        feedbackUpdates.push(highlight)
      },
      sessionHighlights: [existingHighlight],
    })
    useLayoutEffect(() => {
      controller = actions
    })
    return createElement('div', { id: 'reader-highlight-actions-smoke' }, 'ready')
  }

  flushSync(() => {
    root.render(createElement(Harness))
  })
  flushSync(() => {
    readController().addHighlightWithUndo(newHighlight)
  })
  flushSync(() => {
    readController().undoHighlightAction()
  })
  flushSync(() => {
    readController().removeHighlightWithUndo(existingHighlight.id)
  })
  flushSync(() => {
    readController().undoHighlightAction()
  })
  flushSync(() => {
    readController().setActiveHighlightFeedback('useful')
  })
  flushSync(() => {
    readController().appendActiveHighlight()
  })
  flushSync(() => {
    readController().removeActiveHighlight()
  })
  flushSync(() => {
    readController().clearHighlightUndoStack()
  })
  const undoAfterClear = readController().undoHighlightAction()

  const renderedText = host.textContent || ''
  root.unmount()
  host.remove()

  return {
    addedHighlights,
    appendedText,
    closeCount,
    directEnriched,
    differentUndoMatch,
    feedbackUpdates,
    messageEvents,
    removedIds,
    renderedText,
    sameUndoMatch,
    undoAfterClear,
  }
}
