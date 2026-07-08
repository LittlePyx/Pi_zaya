import { createElement, useLayoutEffect } from 'react'
import { flushSync } from 'react-dom'
import { createRoot } from 'react-dom/client'

import type { ReaderDocBlock } from '../api/references'
import type { ReaderLocateResult } from '../components/chat/reader/readerTypes'
import {
  resolveReaderReturnToEvidenceTarget,
  useReaderReturnToEvidence,
} from '../components/chat/useReaderReturnToEvidence'

export interface ReaderReturnToEvidenceSmokeResult {
  directTargetId: string
  fallbackTargetId: string
  focusedIds: string[]
  missingTarget: boolean
  previousFocusCleared: boolean
  renderedText: string
  scrollCalls: number
}

const directBlock = {
  anchor_id: 'anchor-a',
  block_id: 'block-a',
  doc_id: 'doc-a',
  heading_path: 'Methods',
  kind: 'paragraph',
  order_index: 1,
  text: 'Direct evidence text',
} as ReaderDocBlock

const fallbackBlock = {
  anchor_id: 'anchor-fallback',
  block_id: 'block-fallback',
  doc_id: 'doc-a',
  heading_path: 'Results',
  kind: 'paragraph',
  order_index: 2,
  text: 'Fallback evidence sentence carries the exact quoted claim.',
} as ReaderDocBlock

const locateResult = {
  anchorId: 'anchor-a',
  anchorKind: 'paragraph',
  blockId: 'block-a',
  headingPath: 'Methods',
  hint: 'Located block-a',
  locateRequestId: 7,
  ok: true,
  precision: 'block',
  reason: 'Located block-a',
  repairable: false,
  sourcePath: '/tmp/reader.md',
  status: 'block',
  strictLocate: false,
} as ReaderLocateResult

function baseOptions(locate: ReaderLocateResult | null = locateResult) {
  return {
    activeAnchorId: 'anchor-a',
    activeAnchorKind: 'paragraph',
    activeAnchorNumber: 0,
    activeBlockId: 'block-a',
    activeFocusSnippet: 'Direct evidence text',
    activeHeadingPath: 'Methods',
    activeHighlightSnippet: '',
    locateResult: locate,
    relatedBlockIds: ['block-fallback'],
  }
}

export function runReaderReturnToEvidenceSmoke(): ReaderReturnToEvidenceSmokeResult {
  const host = document.createElement('div')
  const readerRoot = document.createElement('div')
  const previousFocus = document.createElement('p')
  const directReadable = document.createElement('p')
  const directAnchor = document.createElement('span')
  const fallbackReadable = document.createElement('p')
  previousFocus.id = 'previous-focus'
  previousFocus.className = 'kb-reader-focus kb-reader-focus-secondary'
  previousFocus.textContent = 'Previous focus'
  directReadable.id = 'direct-readable'
  directAnchor.id = 'direct-anchor'
  directAnchor.setAttribute('data-kb-block-id', 'block-a')
  directAnchor.setAttribute('data-kb-anchor-id', 'anchor-a')
  directAnchor.setAttribute('data-kb-anchor-kind', 'paragraph')
  directAnchor.textContent = 'Direct evidence text'
  directReadable.append(directAnchor)
  fallbackReadable.id = 'fallback-readable'
  fallbackReadable.textContent = 'Fallback evidence sentence carries the exact quoted claim.'
  readerRoot.append(previousFocus, directReadable, fallbackReadable)
  document.body.append(host, readerRoot)

  let scrollCalls = 0
  readerRoot.scrollTo = () => {
    scrollCalls += 1
  }

  const readerBlocks = [directBlock, fallbackBlock]
  const directTarget = resolveReaderReturnToEvidenceTarget(readerRoot, readerBlocks, baseOptions())
  const fallbackTarget = resolveReaderReturnToEvidenceTarget(readerRoot, readerBlocks, {
    ...baseOptions(null),
    activeAnchorId: '',
    activeBlockId: '',
    activeFocusSnippet: '',
    activeHighlightSnippet: 'exact quoted claim',
  })
  const missingTarget = resolveReaderReturnToEvidenceTarget(readerRoot, readerBlocks, {
    ...baseOptions(null),
    activeAnchorId: '',
    activeBlockId: '',
    activeFocusSnippet: '',
    activeHighlightSnippet: '',
  }) === null

  const root = createRoot(host)
  const contentRef = { current: readerRoot }
  let returnToEvidence: (() => void) | null = null
  const readReturnToEvidence = () => {
    if (!returnToEvidence) throw new Error('reader return-to-evidence smoke did not mount')
    return returnToEvidence
  }

  function Harness() {
    const handler = useReaderReturnToEvidence({
      ...baseOptions(),
      contentRef,
      readerBlocks,
    })
    useLayoutEffect(() => {
      returnToEvidence = handler
    }, [handler])
    return createElement('div', { id: 'reader-return-to-evidence-smoke' }, 'ready')
  }

  flushSync(() => {
    root.render(createElement(Harness))
  })
  flushSync(() => {
    readReturnToEvidence()()
  })

  const focusedIds = Array.from(readerRoot.querySelectorAll<HTMLElement>('.kb-reader-focus'))
    .map((node) => node.id)
  const previousFocusCleared = !previousFocus.classList.contains('kb-reader-focus')
    && !previousFocus.classList.contains('kb-reader-focus-secondary')
  const renderedText = host.textContent || ''

  root.unmount()
  host.remove()
  readerRoot.remove()

  return {
    directTargetId: directTarget?.id || '',
    fallbackTargetId: fallbackTarget?.id || '',
    focusedIds,
    missingTarget,
    previousFocusCleared,
    renderedText,
    scrollCalls,
  }
}
