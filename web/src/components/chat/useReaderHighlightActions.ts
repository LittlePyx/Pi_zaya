import { createElement, useCallback, useRef, type ReactNode } from 'react'
import { message } from 'antd'

import { sameHighlightTarget } from './reader/readerDomUtils'
import type { ReaderSessionHighlight } from './reader/readerTypes'

export type ReaderHighlightFeedback = 'useful' | 'needs_check'

export type HighlightUndoAction =
  | { kind: 'remove'; highlight: ReaderSessionHighlight }
  | { kind: 'restore'; highlight: ReaderSessionHighlight }

export interface ReaderHighlightActionLabels {
  reader_feedback_saved?: string
  reader_highlight_removed?: string
  reader_undo?: string
}

export interface ReaderHighlightMessenger {
  open: (config: { type: 'success'; content: ReactNode }) => void
  success: (content: ReactNode) => void
}

export interface EnrichReaderSessionHighlightOptions {
  activeHeadingPath: string
  conversationId?: string
  locateFeedbackKey?: string
  locateRequestId?: number
  messageId?: number | null
  now?: () => number
  sourceName: string
  sourcePath: string
  title: string
}

export interface UseReaderHighlightActionsOptions extends EnrichReaderSessionHighlightOptions {
  activeHighlight: ReaderSessionHighlight | null
  labels: ReaderHighlightActionLabels
  messenger?: ReaderHighlightMessenger
  onAddSessionHighlight?: (highlight: ReaderSessionHighlight) => void
  onAppendSelection: (text: string) => void
  onCloseHighlight: () => void
  onRemoveSessionHighlight?: (highlightId: string) => void
  onUpdateSessionHighlight?: (highlight: ReaderSessionHighlight) => void
  sessionHighlights: ReaderSessionHighlight[]
  sourceLabel: string
}

export interface ReaderHighlightActionsController {
  addHighlightWithUndo: (highlight: ReaderSessionHighlight) => void
  appendActiveHighlight: () => void
  clearHighlightUndoStack: () => void
  removeActiveHighlight: () => void
  removeHighlightWithUndo: (highlightId: string) => void
  setActiveHighlightFeedback: (feedback: ReaderHighlightFeedback) => void
  undoHighlightAction: (specificAction?: HighlightUndoAction) => boolean
}

const defaultHighlightMessenger: ReaderHighlightMessenger = {
  open: (config) => {
    message.open(config)
  },
  success: (content) => {
    message.success(content)
  },
}

export function sameHighlightUndoAction(left: HighlightUndoAction, right: HighlightUndoAction): boolean {
  return left.kind === right.kind && String(left.highlight.id || '').trim() === String(right.highlight.id || '').trim()
}

export function enrichReaderSessionHighlight(
  highlight: ReaderSessionHighlight,
  {
    activeHeadingPath,
    conversationId,
    locateFeedbackKey,
    locateRequestId,
    messageId,
    now = Date.now,
    sourceName,
    sourcePath,
    title,
  }: EnrichReaderSessionHighlightOptions,
): ReaderSessionHighlight {
  const timestamp = now()
  const rawMessageId = highlight.messageId ?? messageId
  const nextMessageId = rawMessageId == null || !Number.isFinite(Number(rawMessageId))
    ? undefined
    : Number(rawMessageId)
  const rawLocateRequestId = highlight.locateRequestId ?? locateRequestId
  const nextLocateRequestId = rawLocateRequestId == null
    || !Number.isFinite(Number(rawLocateRequestId))
    || Number(rawLocateRequestId) <= 0
    ? undefined
    : Number(rawLocateRequestId)

  return {
    ...highlight,
    noteKind: highlight.noteKind || 'highlight',
    sourcePath: highlight.sourcePath || sourcePath || undefined,
    sourceName: highlight.sourceName || title || sourceName || undefined,
    conversationId: highlight.conversationId || String(conversationId || '').trim() || undefined,
    messageId: nextMessageId,
    locateRequestId: nextLocateRequestId,
    locateFeedbackKey: highlight.locateFeedbackKey || locateFeedbackKey || undefined,
    headingPath: highlight.headingPath || activeHeadingPath || undefined,
    createdAt: Number.isFinite(Number(highlight.createdAt)) ? Number(highlight.createdAt) : timestamp,
    updatedAt: timestamp,
  }
}

export function useReaderHighlightActions({
  activeHeadingPath,
  activeHighlight,
  conversationId,
  labels,
  locateFeedbackKey,
  locateRequestId,
  messageId,
  messenger = defaultHighlightMessenger,
  now = Date.now,
  onAddSessionHighlight,
  onAppendSelection,
  onCloseHighlight,
  onRemoveSessionHighlight,
  onUpdateSessionHighlight,
  sessionHighlights,
  sourceLabel,
  sourceName,
  sourcePath,
  title,
}: UseReaderHighlightActionsOptions): ReaderHighlightActionsController {
  const highlightUndoStackRef = useRef<HighlightUndoAction[]>([])
  const feedbackSavedLabel = labels.reader_feedback_saved || 'Evidence note updated'
  const highlightRemovedLabel = labels.reader_highlight_removed || 'Highlight removed'
  const undoLabel = labels.reader_undo || 'Undo'

  const enrichHighlight = useCallback((highlight: ReaderSessionHighlight): ReaderSessionHighlight => (
    enrichReaderSessionHighlight(highlight, {
      activeHeadingPath,
      conversationId,
      locateFeedbackKey,
      locateRequestId,
      messageId,
      now,
      sourceName,
      sourcePath,
      title,
    })
  ), [
    activeHeadingPath,
    conversationId,
    locateFeedbackKey,
    locateRequestId,
    messageId,
    now,
    sourceName,
    sourcePath,
    title,
  ])

  const clearHighlightUndoStack = useCallback(() => {
    highlightUndoStackRef.current = []
  }, [])

  const applyHighlightUndoAction = useCallback((action: HighlightUndoAction): boolean => {
    const highlightId = String(action.highlight.id || '').trim()
    if (!highlightId) return false
    if (action.kind === 'remove') {
      onRemoveSessionHighlight?.(highlightId)
    } else {
      onAddSessionHighlight?.(action.highlight)
    }
    onCloseHighlight()
    return true
  }, [onAddSessionHighlight, onCloseHighlight, onRemoveSessionHighlight])

  const undoHighlightAction = useCallback((specificAction?: HighlightUndoAction): boolean => {
    let action = specificAction || highlightUndoStackRef.current.pop()
    if (specificAction) {
      const idx = [...highlightUndoStackRef.current]
        .reverse()
        .findIndex((item) => sameHighlightUndoAction(item, specificAction))
      if (idx < 0) return false
      const removeAt = highlightUndoStackRef.current.length - 1 - idx
      action = highlightUndoStackRef.current[removeAt]
      highlightUndoStackRef.current.splice(removeAt, 1)
    }
    if (!action) return false
    return applyHighlightUndoAction(action)
  }, [applyHighlightUndoAction])

  const addHighlightWithUndo = useCallback((highlight: ReaderSessionHighlight) => {
    const nextHighlight = enrichHighlight(highlight)
    const nextId = String(nextHighlight?.id || '').trim()
    const alreadyExists = sessionHighlights.some((item) => (
      String(item.id || '').trim() === nextId || sameHighlightTarget(item, nextHighlight)
    ))
    if (alreadyExists) return
    onAddSessionHighlight?.(nextHighlight)
    highlightUndoStackRef.current.push({ kind: 'remove', highlight: nextHighlight })
  }, [enrichHighlight, onAddSessionHighlight, sessionHighlights])

  const removeHighlightWithUndo = useCallback((highlightId: string) => {
    const targetId = String(highlightId || '').trim()
    if (!targetId) return
    const removed = sessionHighlights.find((item) => item.id === targetId) || null
    onRemoveSessionHighlight?.(targetId)
    onCloseHighlight()
    if (removed && onAddSessionHighlight) {
      const undoAction: HighlightUndoAction = { kind: 'restore', highlight: removed }
      highlightUndoStackRef.current.push(undoAction)
      messenger.open({
        type: 'success',
        content: createElement(
          'span',
          { className: 'kb-reader-toast-content' },
          createElement('span', null, highlightRemovedLabel),
          createElement(
            'button',
            {
              type: 'button',
              className: 'kb-reader-toast-action',
              onClick: () => undoHighlightAction(undoAction),
            },
            undoLabel,
          ),
        ),
      })
      return
    }
    messenger.success(highlightRemovedLabel)
  }, [
    highlightRemovedLabel,
    messenger,
    onAddSessionHighlight,
    onCloseHighlight,
    onRemoveSessionHighlight,
    sessionHighlights,
    undoHighlightAction,
    undoLabel,
  ])

  const setActiveHighlightFeedback = useCallback((feedback: ReaderHighlightFeedback) => {
    const item = activeHighlight
    if (!item || !onUpdateSessionHighlight) return
    const nextFeedback = item.feedback === feedback ? undefined : feedback
    const updated = enrichHighlight({
      ...item,
      feedback: nextFeedback,
      feedbackAt: nextFeedback ? now() : undefined,
    })
    onUpdateSessionHighlight(updated)
    messenger.success(feedbackSavedLabel)
    onCloseHighlight()
  }, [
    activeHighlight,
    enrichHighlight,
    feedbackSavedLabel,
    messenger,
    now,
    onCloseHighlight,
    onUpdateSessionHighlight,
  ])

  const appendActiveHighlight = useCallback(() => {
    const item = activeHighlight
    const text = String(item?.text || '').trim()
    if (!item || !text) return
    const quoted = text.split('\n').map((line) => `> ${line}`).join('\n')
    const sourceLine = sourceLabel ? `> Source: ${sourceLabel}\n` : ''
    onAppendSelection(`${sourceLine}${quoted}\n\n`)
    onCloseHighlight()
  }, [activeHighlight, onAppendSelection, onCloseHighlight, sourceLabel])

  const removeActiveHighlight = useCallback(() => {
    if (!activeHighlight) return
    removeHighlightWithUndo(activeHighlight.id)
  }, [activeHighlight, removeHighlightWithUndo])

  return {
    addHighlightWithUndo,
    appendActiveHighlight,
    clearHighlightUndoStack,
    removeActiveHighlight,
    removeHighlightWithUndo,
    setActiveHighlightFeedback,
    undoHighlightAction,
  }
}
