/* eslint-disable react-hooks/set-state-in-effect */

import { useEffect, useState, type RefObject } from 'react'
import type { ReaderDocBlock } from '../../../api/references'
import type { ReaderSessionHighlight } from './readerTypes'
import {
  clearReaderFocusClasses,
  resolveSessionHighlightScrollTarget,
  scrollReaderTargetIntoView,
} from './readerDomUtils'

interface UseReaderHighlightWorkspaceArgs {
  open: boolean
  sourcePath: string
  contentRef: RefObject<HTMLDivElement | null>
  readerBlocks: ReaderDocBlock[]
  sessionHighlights: ReaderSessionHighlight[]
  onRemoveSessionHighlight?: (highlightId: string) => void
}

export function useReaderHighlightWorkspace({
  open,
  sourcePath,
  contentRef,
  readerBlocks,
  sessionHighlights,
  onRemoveSessionHighlight,
}: UseReaderHighlightWorkspaceArgs) {
  const [highlightsOpen, setHighlightsOpen] = useState(false)
  const [activeHighlightId, setActiveHighlightId] = useState('')

  useEffect(() => {
    if (!open) {
      setHighlightsOpen(false)
      setActiveHighlightId('')
      return
    }
    setHighlightsOpen(false)
    setActiveHighlightId('')
  }, [open, sourcePath])

  useEffect(() => {
    if (sessionHighlights.length > 0) return
    setHighlightsOpen(false)
    setActiveHighlightId('')
  }, [sessionHighlights])

  useEffect(() => {
    if (!activeHighlightId) return
    if (sessionHighlights.some((item) => item.id === activeHighlightId)) return
    setActiveHighlightId('')
  }, [activeHighlightId, sessionHighlights])

  const jumpToSessionHighlight = (item: ReaderSessionHighlight) => {
    const root = contentRef.current
    if (!root) return
    const target = resolveSessionHighlightScrollTarget(root, readerBlocks, item)
    if (!target) return
    clearReaderFocusClasses(root)
    target.classList.add('kb-reader-focus-secondary')
    setActiveHighlightId(String(item.id || '').trim())
    scrollReaderTargetIntoView(root, target)
  }

  const removeSessionHighlight = (highlightId: string) => {
    const targetId = String(highlightId || '').trim()
    if (!targetId) return
    onRemoveSessionHighlight?.(targetId)
    setActiveHighlightId((current) => (current === targetId ? '' : current))
  }

  return {
    hasHighlights: sessionHighlights.length > 0,
    highlightsOpen,
    activeHighlightId,
    toggleHighlights: () => setHighlightsOpen((current) => !current),
    jumpToSessionHighlight,
    removeSessionHighlight,
  }
}
