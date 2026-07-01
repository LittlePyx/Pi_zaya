import { useCallback, useState, type RefObject } from 'react'
import { message } from 'antd'
import { chatApi } from '../../api/chat'
import { basenameFromSourcePath } from '../../utils/sourcePath'
import type { CiteDetail } from './citationState'
import type { RightDockPanel } from './useReaderDock'
import {
  READER_CITATION_SHELF_EVENT,
  READER_SELECTION_SHELF_EVENT,
  READER_STANDALONE_WINDOW_NAME,
  type ReaderOpenPayload,
  type ReaderSelectionShelfPayload,
  type ReaderSessionHighlight,
} from './reader/readerTypes'
import {
  sanitizeReaderLocateCandidates,
  sanitizeReaderLocateTarget,
} from './reader/readerOpenPayloadUtils'

interface RegisterReaderLocateRequestInput {
  feedbackKey?: string
  locateRequestId: number
  sourcePath: string
  payload: ReaderOpenPayload
}

export function useReaderWorkspaceActions({
  labels,
  activeConversationId,
  shelfProjectId,
  desktopReaderEligible,
  readerPayloadRef,
  activeReaderSessionHighlightsRef,
  nextEventToken,
  nextReaderLocateRequestId,
  registerReaderLocateRequest,
  openReaderDock,
  showDockPanel,
  openTimeline,
}: {
  labels: Record<string, string>
  activeConversationId?: string | null
  shelfProjectId?: string | null
  desktopReaderEligible: boolean
  readerPayloadRef: RefObject<ReaderOpenPayload | null>
  activeReaderSessionHighlightsRef: RefObject<ReaderSessionHighlight[]>
  nextEventToken: () => number
  nextReaderLocateRequestId: () => number
  registerReaderLocateRequest: (input: RegisterReaderLocateRequestInput) => void
  openReaderDock: (payload: ReaderOpenPayload) => void
  showDockPanel: (panel: RightDockPanel) => void
  openTimeline: () => void
}) {
  const [citationShelfOpen, setCitationShelfOpen] = useState(false)
  const [citationShelfCount, setCitationShelfCount] = useState(0)
  const [openShelfSignal, setOpenShelfSignal] = useState(0)
  const [appendSignal, setAppendSignal] = useState<{ token: number; text: string } | null>(null)

  const resetReaderWorkspaceTransientState = useCallback((resetShelf: boolean) => {
    if (resetShelf) {
      setCitationShelfOpen(false)
      setCitationShelfCount(0)
    }
    setAppendSignal(null)
  }, [])

  const openReader = useCallback((payload: ReaderOpenPayload) => {
    const sourcePath = String(payload?.sourcePath || '').trim()
    if (!sourcePath) {
      message.info(labels.reader_missing_path)
      return
    }
    const locateRequestId = nextReaderLocateRequestId()
    const locateTarget = sanitizeReaderLocateTarget(payload.locateTarget)
    const claimGroup = (payload.claimGroup && typeof payload.claimGroup === 'object')
      ? {
        id: String(payload.claimGroup.id || '').trim() || undefined,
        kind: String(payload.claimGroup.kind || '').trim() || undefined,
        leadText: String(payload.claimGroup.leadText || '').trim() || undefined,
        distance: Number.isFinite(Number(payload.claimGroup.distance))
          ? Number(payload.claimGroup.distance)
          : undefined,
      }
      : undefined
    const alternatives = sanitizeReaderLocateCandidates(payload.alternatives)
    const visibleAlternatives = sanitizeReaderLocateCandidates(payload.visibleAlternatives)
    const evidenceAlternatives = sanitizeReaderLocateCandidates(payload.evidenceAlternatives)
    const initialAltCandidateCount = evidenceAlternatives.length || visibleAlternatives.length || alternatives.length
    const initialAltIndexRaw = Number(payload.initialAltIndex)
    const initialAltIndex = Number.isFinite(initialAltIndexRaw)
      ? Math.min(
        Math.max(0, Math.floor(initialAltIndexRaw)),
        Math.max(0, initialAltCandidateCount - 1),
      )
      : undefined
    const nextPayload: ReaderOpenPayload = {
      sourcePath,
      sourceName: String(payload.sourceName || '').trim(),
      headingPath: String(payload.headingPath || '').trim(),
      snippet: String(payload.snippet || '').trim(),
      highlightSnippet: String(payload.highlightSnippet || '').trim(),
      blockId: String(payload.blockId || '').trim() || undefined,
      anchorId: String(payload.anchorId || '').trim() || undefined,
      relatedBlockIds: Array.isArray(payload.relatedBlockIds)
        ? payload.relatedBlockIds.map((item) => String(item || '').trim()).filter(Boolean)
        : undefined,
      anchorKind: String(payload.anchorKind || '').trim() || undefined,
      anchorNumber: Number.isFinite(Number(payload.anchorNumber))
        ? Number(payload.anchorNumber)
        : undefined,
      strictLocate: Boolean(payload.strictLocate),
      locateMode: payload.locateMode === 'heuristic' ? 'heuristic' : undefined,
      locateTarget,
      claimGroup,
      locateRequestId,
      alternatives: alternatives.length > 0
        ? alternatives
        : undefined,
      visibleAlternatives: visibleAlternatives.length > 0
        ? visibleAlternatives
        : undefined,
      evidenceAlternatives: evidenceAlternatives.length > 0
        ? evidenceAlternatives
        : undefined,
      initialAltIndex,
      locateFeedbackKey: String(payload.locateFeedbackKey || '').trim() || undefined,
    }
    const feedbackKey = String(nextPayload.locateFeedbackKey || '').trim()
    if (feedbackKey) {
      registerReaderLocateRequest({
        feedbackKey,
        locateRequestId,
        sourcePath,
        payload: nextPayload,
      })
    }
    openReaderDock(nextPayload)
  }, [labels.reader_missing_path, nextReaderLocateRequestId, openReaderDock, registerReaderLocateRequest])

  const openReaderStandalone = useCallback(async (payloadInput?: ReaderOpenPayload | null) => {
    const payload = payloadInput || readerPayloadRef.current
    const sourcePath = String(payload?.sourcePath || '').trim()
    if (!payload || !sourcePath) {
      message.info(labels.reader_missing_path)
      return
    }
    const sourceName = String(payload.sourceName || '').trim()
      || basenameFromSourcePath(sourcePath)
      || labels.side_dock_reader
    let popup: Window | null = null
    try {
      popup = window.open('', READER_STANDALONE_WINDOW_NAME)
      if (popup) {
        popup.document.title = sourceName
        popup.document.body.style.margin = '0'
        popup.document.body.style.fontFamily = 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
        popup.document.body.innerHTML = `<div style="display:grid;place-items:center;min-height:100vh;color:#64748b;font-size:14px;">${labels.reader_opening_window || 'Opening reader...'}</div>`
        popup.focus()
      }
    } catch {
      popup = null
    }
    try {
      const session = await chatApi.createReaderSession(payload, {
        title: sourceName,
        conversationId: activeConversationId,
        state: {
          sourcePath,
          conversationId: activeConversationId || '',
          projectId: shelfProjectId || '',
          highlights: activeReaderSessionHighlightsRef.current,
          evidenceNotes: activeReaderSessionHighlightsRef.current,
        },
      })
      const linkedConversationId = String(session.conversation_id || activeConversationId || '').trim()
      const readerUrl = new URL(`/reader/session/${encodeURIComponent(session.id)}`, window.location.origin)
      if (linkedConversationId) readerUrl.searchParams.set('conversation', linkedConversationId)
      const url = readerUrl.toString()
      if (popup && !popup.closed) {
        popup.location.href = url
        popup.focus()
      } else {
        const opened = window.open(url, READER_STANDALONE_WINDOW_NAME)
        opened?.focus()
        if (!opened) message.info(labels.reader_window_blocked || 'The browser blocked the reader window.')
      }
    } catch (err) {
      if (popup && !popup.closed) {
        popup.close()
      }
      message.error(err instanceof Error ? err.message : (labels.reader_open_window_failed || 'Failed to open reader window'))
    }
  }, [
    activeConversationId,
    activeReaderSessionHighlightsRef,
    labels.reader_missing_path,
    labels.reader_open_window_failed,
    labels.reader_opening_window,
    labels.reader_window_blocked,
    labels.side_dock_reader,
    readerPayloadRef,
    shelfProjectId,
  ])

  const handleCitationShelfOpenChange = useCallback((open: boolean) => {
    setCitationShelfOpen(open)
    if (open && desktopReaderEligible) {
      showDockPanel('shelf')
    }
  }, [desktopReaderEligible, showDockPanel])

  const handleCitationShelfStateChange = useCallback((state: { open: boolean; count: number }) => {
    setCitationShelfCount(Math.max(0, Math.floor(Number(state.count || 0))))
    setCitationShelfOpen(Boolean(state.open))
    if (state.open && desktopReaderEligible) {
      showDockPanel('shelf')
    }
  }, [desktopReaderEligible, showDockPanel])

  const openReaderCitationShelf = useCallback(() => {
    setOpenShelfSignal((value) => value + 1)
    setCitationShelfOpen(true)
    showDockPanel('shelf')
  }, [showDockPanel])

  const activateDockPanel = useCallback((panel: RightDockPanel) => {
    showDockPanel(panel)
    if (panel === 'timeline') {
      openTimeline()
      return
    }
    if (panel === 'shelf') {
      openReaderCitationShelf()
    }
  }, [openReaderCitationShelf, openTimeline, showDockPanel])

  const appendReaderSelection = useCallback((text: string) => {
    const raw = String(text || '')
    if (!raw.trim()) return
    setAppendSignal({
      token: nextEventToken(),
      text: raw,
    })
  }, [nextEventToken])

  const addReaderSelectionToShelf = useCallback((payload: ReaderSelectionShelfPayload) => {
    const text = String(payload?.text || '').trim()
    const sourcePath = String(payload?.sourcePath || '').trim()
    if (!text || !sourcePath) return
    const detail: ReaderSelectionShelfPayload = {
      ...payload,
      text,
      sourcePath,
      conversationId: activeConversationId || payload.conversationId || '',
      projectId: shelfProjectId || payload.projectId || '',
      createdAt: Number(payload.createdAt || Date.now()),
    }
    window.dispatchEvent(new CustomEvent(READER_SELECTION_SHELF_EVENT, { detail }))
    openReaderCitationShelf()
    message.success(labels.reader_added_to_shelf || 'Added to citation shelf')
  }, [activeConversationId, labels.reader_added_to_shelf, openReaderCitationShelf, shelfProjectId])

  const addReaderCitationToShelf = useCallback((detail: CiteDetail) => {
    if (!detail) return
    const payload = {
      type: 'reader-citation-shelf',
      detail: detail as unknown as Record<string, unknown>,
      conversationId: activeConversationId || '',
      projectId: shelfProjectId || '',
      createdAt: Date.now(),
    }
    window.dispatchEvent(new CustomEvent(READER_CITATION_SHELF_EVENT, { detail: payload }))
    openReaderCitationShelf()
    message.success(labels.reader_added_to_shelf || 'Added to citation shelf')
  }, [activeConversationId, labels.reader_added_to_shelf, openReaderCitationShelf, shelfProjectId])

  return {
    citationShelfOpen,
    citationShelfCount,
    openShelfSignal,
    appendSignal,
    resetReaderWorkspaceTransientState,
    openReader,
    openReaderStandalone,
    handleCitationShelfOpenChange,
    handleCitationShelfStateChange,
    activateDockPanel,
    appendReaderSelection,
    addReaderSelectionToShelf,
    addReaderCitationToShelf,
    openReaderCitationShelf,
  }
}
