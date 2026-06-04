import { useEffect, type RefObject } from 'react'
import type { ReaderDocBlock } from '../../../api/references'
import type { ReaderSessionHighlight } from './readerTypes'
import {
  buildTextNodeCorpus,
  clearReaderUserHighlights,
  closestReadableBlock,
  highlightRawRangeInCorpus,
  highlightSessionTextInContainer,
  readableBlocks,
  resolveDirectTargetNode,
} from './readerDomUtils'

interface UseReaderSessionHighlightLayerArgs {
  open: boolean
  drawerReady: boolean
  markdown: string
  contentRef: RefObject<HTMLDivElement | null>
  readerBlocks: ReaderDocBlock[]
  sessionHighlights: ReaderSessionHighlight[]
}

export function useReaderSessionHighlightLayer({
  open,
  drawerReady,
  markdown,
  contentRef,
  readerBlocks,
  sessionHighlights,
}: UseReaderSessionHighlightLayerArgs) {
  useEffect(() => {
    if (!open || !drawerReady || !markdown) return
    const root = contentRef.current
    if (!root) return
    clearReaderUserHighlights(root)
    const corpus = buildTextNodeCorpus(root)
    for (const item of sessionHighlights) {
      const startOffset = Number(item?.startOffset ?? -1)
      const endOffset = Number(item?.endOffset ?? -1)
      if (startOffset >= 0 && endOffset > startOffset) {
        highlightRawRangeInCorpus(corpus.nodes, startOffset, endOffset, 'kb-reader-user-highlight', {
          'data-kb-session-highlight-id': String(item?.id || '').trim(),
          role: 'button',
          tabindex: '0',
        })
        if (root.querySelector(`[data-kb-session-highlight-id="${CSS.escape(String(item?.id || '').trim())}"]`)) {
          continue
        }
      }
      const rangeStartIndex = Number.isFinite(Number(item?.startReadableIndex ?? -1))
        ? Math.max(-1, Math.floor(Number(item?.startReadableIndex ?? -1)))
        : -1
      const rangeEndIndex = Number.isFinite(Number(item?.endReadableIndex ?? -1))
        ? Math.max(-1, Math.floor(Number(item?.endReadableIndex ?? -1)))
        : -1
      const blockHint = String(item?.blockId || '').trim()
      const anchorHint = String(item?.anchorId || '').trim()
      const resolved = resolveDirectTargetNode(root, readerBlocks, {
        blockId: blockHint,
        anchorId: anchorHint,
      })
      const fallbackReadableIndex = Number.isFinite(Number(item?.readableIndex ?? -1))
        ? Math.max(-1, Math.floor(Number(item?.readableIndex ?? -1)))
        : -1
      const fallbackReadable = fallbackReadableIndex >= 0 ? (readableBlocks(root)[fallbackReadableIndex] || null) : null
      const container: HTMLElement = (closestReadableBlock(resolved.target) || resolved.target || fallbackReadable || root) as HTMLElement
      if (rangeStartIndex >= 0 && rangeEndIndex > rangeStartIndex) {
        continue
      }
      const hit = highlightSessionTextInContainer(
        container,
        String(item?.text || ''),
        String(item?.id || '').trim(),
        Number(item?.occurrence || 0),
      )
      const rootOccurrence = Number.isFinite(Number(item?.documentOccurrence ?? -1))
        ? Math.max(0, Math.floor(Number(item?.documentOccurrence ?? 0)))
        : Math.max(0, Math.floor(Number(item?.occurrence || 0)))
      if (!hit && container !== root) {
        highlightSessionTextInContainer(
          root,
          String(item?.text || ''),
          String(item?.id || '').trim(),
          rootOccurrence,
        )
      }
    }
  }, [
    open,
    drawerReady,
    markdown,
    contentRef,
    readerBlocks,
    sessionHighlights,
  ])
}
