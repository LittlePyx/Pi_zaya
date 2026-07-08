/* eslint-disable react-hooks/set-state-in-effect */

import { useCallback, useEffect, useState, type MouseEvent, type RefObject } from 'react'

import type { ReaderSessionHighlight } from './reader/readerTypes'

export interface ReaderHighlightBubble {
  x: number
  y: number
  highlightId: string
  text: string
}

export interface UseReaderHighlightMenuOptions {
  contentRef: RefObject<HTMLDivElement | null>
  open: boolean
  sessionHighlights: ReaderSessionHighlight[]
  sourcePath: string
}

export interface ReaderHighlightMenuController {
  activeHighlight: ReaderSessionHighlight | null
  closeHighlightBubble: () => void
  highlightBubble: ReaderHighlightBubble | null
  openHighlightMenuFromClick: (
    event: MouseEvent<HTMLDivElement>,
    beforeOpen?: () => void,
  ) => boolean
}

export function findReaderHighlightMark(
  root: HTMLElement | null,
  target: EventTarget | null,
): HTMLElement | null {
  if (!root || !(target instanceof HTMLElement)) return null
  const mark = target.closest<HTMLElement>('.kb-reader-user-highlight') || null
  if (!mark || !root.contains(mark)) return null
  return mark
}

export function buildReaderHighlightBubble(
  root: HTMLElement | null,
  mark: HTMLElement | null,
  sessionHighlights: ReaderSessionHighlight[],
): ReaderHighlightBubble | null {
  if (!root || !mark || !root.contains(mark)) return null
  const highlightId = String(mark.getAttribute('data-kb-session-highlight-id') || '').trim()
  const item = sessionHighlights.find((entry) => entry.id === highlightId) || null
  if (!item) return null

  const rect = mark.getBoundingClientRect()
  const containerRect = root.getBoundingClientRect()
  const x = Math.max(18, Math.min(
    containerRect.width - 18,
    rect.left + (rect.width / 2) - containerRect.left,
  ))
  const aboveY = rect.top - containerRect.top - 10
  const belowY = rect.bottom - containerRect.top + 10
  const y = aboveY >= 16 ? aboveY : belowY

  return {
    x,
    y,
    highlightId,
    text: String(item.text || '').trim(),
  }
}

export function isReaderHighlightBubbleStale(
  highlightBubble: ReaderHighlightBubble | null,
  sessionHighlights: ReaderSessionHighlight[],
): boolean {
  if (!highlightBubble) return false
  return !sessionHighlights.some((item) => item.id === highlightBubble.highlightId)
}

export function useReaderHighlightMenu({
  contentRef,
  open,
  sessionHighlights,
  sourcePath,
}: UseReaderHighlightMenuOptions): ReaderHighlightMenuController {
  const [highlightBubble, setHighlightBubble] = useState<ReaderHighlightBubble | null>(null)
  const closeHighlightBubble = useCallback(() => {
    setHighlightBubble(null)
  }, [])
  const activeHighlight = highlightBubble
    ? sessionHighlights.find((item) => item.id === highlightBubble.highlightId) || null
    : null

  const openHighlightMenuFromClick = useCallback((
    event: MouseEvent<HTMLDivElement>,
    beforeOpen?: () => void,
  ): boolean => {
    const root = contentRef.current
    const mark = findReaderHighlightMark(root, event.target)
    const nextBubble = buildReaderHighlightBubble(root, mark, sessionHighlights)
    if (!nextBubble) {
      closeHighlightBubble()
      return false
    }
    event.preventDefault()
    event.stopPropagation()
    beforeOpen?.()
    setHighlightBubble(nextBubble)
    return true
  }, [closeHighlightBubble, contentRef, sessionHighlights])

  useEffect(() => {
    if (!isReaderHighlightBubbleStale(highlightBubble, sessionHighlights)) return
    closeHighlightBubble()
  }, [closeHighlightBubble, highlightBubble, sessionHighlights])

  useEffect(() => {
    closeHighlightBubble()
  }, [closeHighlightBubble, open, sourcePath])

  return {
    activeHighlight,
    closeHighlightBubble,
    highlightBubble,
    openHighlightMenuFromClick,
  }
}
