import { useCallback, type RefObject } from 'react'

import type { ReaderDocBlock } from '../../api/references'
import {
  buildHighlightQueries,
  clearReaderFocusClasses,
  closestReadableBlock,
  resolveDirectTargetNode,
  resolveStickyHighlightTarget,
  scrollReaderTargetIntoView,
} from './reader/readerDomUtils'
import type { ReaderLocateResult } from './reader/readerTypes'

export interface ReaderReturnToEvidenceTargetOptions {
  activeAnchorId: string
  activeAnchorKind: string
  activeAnchorNumber: number
  activeBlockId: string
  activeFocusSnippet: string
  activeHeadingPath: string
  activeHighlightSnippet: string
  locateResult: ReaderLocateResult | null
  relatedBlockIds: string[]
}

export interface UseReaderReturnToEvidenceOptions extends ReaderReturnToEvidenceTargetOptions {
  contentRef: RefObject<HTMLDivElement | null>
  readerBlocks: ReaderDocBlock[]
}

export function resolveReaderReturnToEvidenceTarget(
  root: HTMLElement,
  readerBlocks: ReaderDocBlock[],
  {
    activeAnchorId,
    activeAnchorKind,
    activeAnchorNumber,
    activeBlockId,
    activeFocusSnippet,
    activeHeadingPath,
    activeHighlightSnippet,
    locateResult,
    relatedBlockIds,
  }: ReaderReturnToEvidenceTargetOptions,
): HTMLElement | null {
  const resultBlockId = String(locateResult?.blockId || activeBlockId || '').trim()
  const resultAnchorId = String(locateResult?.anchorId || activeAnchorId || '').trim()
  const resultAnchorKind = String(locateResult?.anchorKind || activeAnchorKind || '').trim().toLowerCase()
  const seed = String(activeHighlightSnippet || activeFocusSnippet || '').trim()
  const direct = resolveDirectTargetNode(root, readerBlocks, {
    blockId: resultBlockId,
    anchorId: resultAnchorId,
    anchorKind: resultAnchorKind,
  })
  return closestReadableBlock(direct.target) || direct.target || resolveStickyHighlightTarget(root, readerBlocks, {
    blockId: resultBlockId,
    anchorId: resultAnchorId,
    anchorKind: resultAnchorKind,
    anchorNumber: activeAnchorNumber,
    headingPath: String(locateResult?.headingPath || activeHeadingPath || '').trim(),
    highlightSeed: seed,
    highlightQueries: buildHighlightQueries(seed, {
      anchorKind: resultAnchorKind,
      anchorNumber: activeAnchorNumber,
    }),
    relatedBlockIds,
    strictLocate: false,
  })
}

export function useReaderReturnToEvidence({
  activeAnchorId,
  activeAnchorKind,
  activeAnchorNumber,
  activeBlockId,
  activeFocusSnippet,
  activeHeadingPath,
  activeHighlightSnippet,
  contentRef,
  locateResult,
  readerBlocks,
  relatedBlockIds,
}: UseReaderReturnToEvidenceOptions) {
  return useCallback(() => {
    const root = contentRef.current
    if (!root) return
    const target = resolveReaderReturnToEvidenceTarget(root, readerBlocks, {
      activeAnchorId,
      activeAnchorKind,
      activeAnchorNumber,
      activeBlockId,
      activeFocusSnippet,
      activeHeadingPath,
      activeHighlightSnippet,
      locateResult,
      relatedBlockIds,
    })
    if (!target) return
    clearReaderFocusClasses(root)
    target.classList.add('kb-reader-focus')
    scrollReaderTargetIntoView(root, target, { force: true })
  }, [
    activeAnchorId,
    activeAnchorKind,
    activeAnchorNumber,
    activeBlockId,
    activeFocusSnippet,
    activeHeadingPath,
    activeHighlightSnippet,
    contentRef,
    locateResult,
    readerBlocks,
    relatedBlockIds,
  ])
}
