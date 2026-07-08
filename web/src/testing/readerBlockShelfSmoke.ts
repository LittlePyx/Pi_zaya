import { createElement, useLayoutEffect } from 'react'
import { flushSync } from 'react-dom'
import { createRoot } from 'react-dom/client'

import type { ReaderSelectionShelfPayload } from '../components/chat/reader/readerTypes'
import {
  buildReaderBlockShelfPayload,
  useReaderBlockShelf,
  type ReaderBlockShelfController,
} from '../components/chat/useReaderBlockShelf'

export interface ReaderBlockShelfSmokeResult {
  canAddBlockToShelf: boolean
  directPayload: ReaderSelectionShelfPayload | null
  emptyPayload: ReaderSelectionShelfPayload | null
  events: ReaderSelectionShelfPayload[]
  renderedText: string
}

export function runReaderBlockShelfSmoke(): ReaderBlockShelfSmokeResult {
  const host = document.createElement('div')
  document.body.append(host)
  const root = createRoot(host)
  const events: ReaderSelectionShelfPayload[] = []
  let controller: ReaderBlockShelfController | null = null
  let canAddBlockToShelf = false

  const directPayload = buildReaderBlockShelfPayload({
    block: {
      anchorId: ' anchor-direct ',
      anchorKind: ' figure ',
      blockId: ' block-direct ',
      headingPath: ' Intro / Figure ',
      text: ' Direct figure text ',
    },
    now: () => 12345,
    sourceName: 'Reader Paper',
    sourcePath: ' /tmp/reader.md ',
  })
  const emptyPayload = buildReaderBlockShelfPayload({
    block: {
      text: '   ',
    },
    now: () => 12345,
    sourceName: 'Reader Paper',
    sourcePath: '/tmp/reader.md',
  })

  const readController = () => {
    if (!controller) throw new Error('reader block shelf smoke did not mount')
    return controller
  }

  function Harness() {
    const shelf = useReaderBlockShelf({
      now: () => 67890,
      onAddSelectionToShelf: (payload) => {
        events.push(payload)
      },
      sourceName: 'Reader Paper',
      sourcePath: '/tmp/reader.md',
    })
    useLayoutEffect(() => {
      controller = shelf
      canAddBlockToShelf = shelf.canAddBlockToShelf
    })
    return createElement('div', { id: 'reader-block-shelf-smoke' }, String(shelf.canAddBlockToShelf))
  }

  flushSync(() => {
    root.render(createElement(Harness))
  })
  flushSync(() => {
    readController().addBlockToShelf({
      anchorId: ' anchor-a ',
      anchorKind: ' table ',
      blockId: ' block-a ',
      headingPath: ' Methods ',
      text: ' Table text ',
    })
  })
  flushSync(() => {
    readController().addBlockToShelf({
      text: '   ',
    })
  })

  const renderedText = host.textContent || ''
  root.unmount()
  host.remove()

  return {
    canAddBlockToShelf,
    directPayload,
    emptyPayload,
    events,
    renderedText,
  }
}
