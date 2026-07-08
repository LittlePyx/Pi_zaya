import { createElement } from 'react'
import { flushSync } from 'react-dom'
import { createRoot } from 'react-dom/client'

import type { ReaderDocBlock } from '../api/references'
import type { ReaderSelectionShelfPayload } from '../components/chat/reader/readerTypes'
import {
  buildReaderEquationShelfPayload,
  useReaderEquationShelfActions,
} from '../components/chat/useReaderEquationShelfActions'

export interface ReaderEquationShelfActionsSmokeResult {
  anchorAttrs: Record<string, string | null>
  buttonText: string
  directPayload: ReaderSelectionShelfPayload | null
  emptyPayload: ReaderSelectionShelfPayload | null
  events: ReaderSelectionShelfPayload[]
  hostClassBeforeCleanup: string
  tailCountAfterCleanup: number
  tailCountBeforeClick: number
  truncatedPayload: ReaderSelectionShelfPayload | null
}

const equationBlock = {
  anchor_id: ' equation-anchor ',
  block_id: ' equation-block ',
  doc_id: 'doc-a',
  heading_path: ' Methods / Equation ',
  kind: 'equation',
  line_start: 12,
  number: 7,
  order_index: 2,
  text: ' E = mc^2 ',
} as ReaderDocBlock

const laterEquationBlock = {
  anchor_id: ' later-anchor ',
  block_id: ' later-block ',
  doc_id: 'doc-a',
  heading_path: ' Later ',
  kind: 'equation',
  line_start: 20,
  number: 8,
  order_index: 3,
  text: ' a^2 + b^2 = c^2 ',
} as ReaderDocBlock

function nextFrame(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => resolve())
  })
}

export async function runReaderEquationShelfActionsSmoke(): Promise<ReaderEquationShelfActionsSmokeResult> {
  const host = document.createElement('div')
  const readerRoot = document.createElement('div')
  const firstEquation = document.createElement('span')
  const secondEquation = document.createElement('span')
  firstEquation.className = 'katex-display'
  firstEquation.textContent = 'fallback equation text'
  secondEquation.className = 'katex-display'
  secondEquation.textContent = 'second fallback equation text'
  readerRoot.append(firstEquation, secondEquation)
  document.body.append(host, readerRoot)

  const root = createRoot(host)
  const events: ReaderSelectionShelfPayload[] = []
  const contentRef = { current: readerRoot }
  const directPayload = buildReaderEquationShelfPayload({
    block: equationBlock,
    now: () => 12345,
    sourceName: 'Reader Paper',
    sourcePath: '/tmp/reader.md',
  })
  const emptyPayload = buildReaderEquationShelfPayload({
    block: {
      ...equationBlock,
      raw_text: '',
      text: '',
    },
    nodeText: '   ',
    now: () => 12345,
    sourceName: 'Reader Paper',
    sourcePath: '/tmp/reader.md',
  })
  const truncatedPayload = buildReaderEquationShelfPayload({
    block: {
      ...equationBlock,
      text: 'abcdef',
    },
    now: () => 12345,
    sourceName: 'Reader Paper',
    sourcePath: '/tmp/reader.md',
    textLimit: 4,
  })

  function Harness() {
    useReaderEquationShelfActions({
      contentRef,
      labels: {
        locate_badge_eq: 'Eq',
        reader_add_to_shelf: 'Shelf',
        reader_add_to_shelf_title: 'Add to research basket',
      },
      markdown: 'markdown-version-a',
      now: () => 67890,
      onAddSelectionToShelf: (payload) => {
        events.push(payload)
      },
      open: true,
      readerBlocks: [laterEquationBlock, equationBlock],
      sourceName: 'Reader Paper',
      sourcePath: '/tmp/reader.md',
    })
    return createElement('div', { id: 'reader-equation-shelf-actions-smoke' }, 'ready')
  }

  flushSync(() => {
    root.render(createElement(Harness))
  })
  await nextFrame()
  const tailCountBeforeClick = readerRoot.querySelectorAll('.kb-md-reader-block-shelf-tail').length
  const button = firstEquation.querySelector<HTMLButtonElement>('[data-testid="reader-block-shelf"]')
  const buttonText = button?.textContent || ''
  const anchorAttrs = {
    anchorId: firstEquation.getAttribute('data-kb-anchor-id'),
    anchorKind: firstEquation.getAttribute('data-kb-anchor-kind'),
    anchorNumber: firstEquation.getAttribute('data-kb-anchor-number'),
    blockId: firstEquation.getAttribute('data-kb-block-id'),
  }
  button?.click()
  const hostClassBeforeCleanup = firstEquation.className

  root.unmount()
  await nextFrame()
  const tailCountAfterCleanup = readerRoot.querySelectorAll('.kb-md-reader-block-shelf-tail').length
  host.remove()
  readerRoot.remove()

  return {
    anchorAttrs,
    buttonText,
    directPayload,
    emptyPayload,
    events,
    hostClassBeforeCleanup,
    tailCountAfterCleanup,
    tailCountBeforeClick,
    truncatedPayload,
  }
}
