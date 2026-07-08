import { createElement, useLayoutEffect, useState } from 'react'
import { flushSync } from 'react-dom'
import { createRoot } from 'react-dom/client'

import type { ReaderSessionHighlight } from '../components/chat/reader/readerTypes'
import {
  buildReaderHighlightBubble,
  findReaderHighlightMark,
  isReaderHighlightBubbleStale,
  useReaderHighlightMenu,
  type ReaderHighlightBubble,
  type ReaderHighlightMenuController,
} from '../components/chat/useReaderHighlightMenu'

export interface ReaderHighlightMenuSmokeResult {
  activeHighlightText: string
  beforeOpenCount: number
  bubbleAfterOpen: ReaderHighlightBubble | null
  bubbleAfterStale: ReaderHighlightBubble | null
  eventCounts: {
    prevented: number
    stopped: number
  }
  foundInside: boolean
  foundOutside: boolean
  invalidBubble: ReaderHighlightBubble | null
  renderedText: string
  staleChecks: {
    active: boolean
    stale: boolean
  }
}

const highlight = {
  id: 'highlight-a',
  text: ' Highlight text ',
} as ReaderSessionHighlight

function rect(left: number, top: number, width: number, height: number): DOMRect {
  return {
    bottom: top + height,
    height,
    left,
    right: left + width,
    top,
    width,
    x: left,
    y: top,
    toJSON: () => ({}),
  } as DOMRect
}

function setRect(element: HTMLElement, nextRect: DOMRect) {
  Object.defineProperty(element, 'getBoundingClientRect', {
    configurable: true,
    value: () => nextRect,
  })
}

function nextFrame(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => resolve())
  })
}

export async function runReaderHighlightMenuSmoke(): Promise<ReaderHighlightMenuSmokeResult> {
  const host = document.createElement('div')
  const rootElement = document.createElement('div')
  const mark = document.createElement('mark')
  const markLabel = document.createElement('span')
  const outside = document.createElement('button')
  rootElement.className = 'kb-reader-root'
  mark.className = 'kb-reader-user-highlight'
  mark.setAttribute('data-kb-session-highlight-id', highlight.id)
  mark.append(markLabel)
  rootElement.append(mark)
  document.body.append(host, rootElement, outside)
  setRect(rootElement, rect(10, 20, 100, 160))
  setRect(mark, rect(160, 62, 40, 12))

  const root = createRoot(host)
  const contentRef = { current: rootElement }
  let beforeOpenCount = 0
  let prevented = 0
  let stopped = 0
  let controller: ReaderHighlightMenuController | null = null
  let setHighlights: ((nextHighlights: ReaderSessionHighlight[]) => void) | null = null

  const foundInside = findReaderHighlightMark(rootElement, markLabel) === mark
  const foundOutside = findReaderHighlightMark(rootElement, outside) === mark
  const invalidBubble = buildReaderHighlightBubble(rootElement, mark, [])
  const directBubble = buildReaderHighlightBubble(rootElement, mark, [highlight])
  const staleChecks = {
    active: isReaderHighlightBubbleStale(directBubble, [highlight]),
    stale: isReaderHighlightBubbleStale(directBubble, []),
  }

  const readController = () => {
    if (!controller) throw new Error('reader highlight menu smoke did not mount')
    return controller
  }
  const readSetHighlights = () => {
    if (!setHighlights) throw new Error('reader highlight menu setter did not mount')
    return setHighlights
  }

  function Harness() {
    const [sessionHighlights, updateHighlights] = useState<ReaderSessionHighlight[]>([highlight])
    const menu = useReaderHighlightMenu({
      contentRef,
      open: true,
      sessionHighlights,
      sourcePath: '/tmp/reader.md',
    })
    useLayoutEffect(() => {
      controller = menu
      setHighlights = updateHighlights
    }, [menu])
    return createElement(
      'div',
      { id: 'reader-highlight-menu-smoke' },
      menu.highlightBubble?.highlightId || 'closed',
    )
  }

  flushSync(() => {
    root.render(createElement(Harness))
  })
  flushSync(() => {
    readController().openHighlightMenuFromClick({
      target: markLabel,
      preventDefault: () => {
        prevented += 1
      },
      stopPropagation: () => {
        stopped += 1
      },
    } as never, () => {
      beforeOpenCount += 1
    })
  })
  const bubbleAfterOpen = readController().highlightBubble
  const activeHighlightText = String(readController().activeHighlight?.text || '')

  flushSync(() => {
    readSetHighlights()([])
  })
  await nextFrame()
  await nextFrame()
  const bubbleAfterStale = readController().highlightBubble

  const renderedText = host.textContent || ''
  root.unmount()
  host.remove()
  rootElement.remove()
  outside.remove()

  return {
    activeHighlightText,
    beforeOpenCount,
    bubbleAfterOpen,
    bubbleAfterStale,
    eventCounts: {
      prevented,
      stopped,
    },
    foundInside,
    foundOutside,
    invalidBubble,
    renderedText,
    staleChecks,
  }
}
