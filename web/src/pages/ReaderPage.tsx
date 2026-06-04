import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Button, Empty, Spin, message } from 'antd'
import { ArrowLeftOutlined, ReloadOutlined } from '@ant-design/icons'
import { useNavigate, useParams } from 'react-router-dom'
import { chatApi, type ReaderSessionRecord } from '../api/chat'
import { PaperGuideReaderDrawer } from '../components/chat/PaperGuideReaderDrawer'
import type {
  ReaderSelectionShelfPayload,
  ReaderLocateCandidate,
  ReaderLocateResult,
  ReaderLocateTarget,
  ReaderOpenPayload,
  ReaderSessionHighlight,
} from '../components/chat/reader/readerTypes'
import { READER_SELECTION_SHELF_CHANNEL } from '../components/chat/reader/readerTypes'
import { useT } from '../i18n'

const READER_SESSION_SYNC_CHANNEL = 'kb:reader-session-sync'

function stringField(rec: Record<string, unknown>, key: string) {
  return String(rec[key] || '').trim()
}

function numberField(rec: Record<string, unknown>, key: string) {
  const value = Number(rec[key])
  return Number.isFinite(value) ? value : undefined
}

function candidateList(value: unknown): ReaderLocateCandidate[] | undefined {
  if (!Array.isArray(value)) return undefined
  const out = value
    .filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === 'object')
    .map((item) => ({
      headingPath: stringField(item, 'headingPath') || undefined,
      snippet: stringField(item, 'snippet') || undefined,
      highlightSnippet: stringField(item, 'highlightSnippet') || undefined,
      anchorId: stringField(item, 'anchorId') || undefined,
      blockId: stringField(item, 'blockId') || undefined,
      anchorKind: stringField(item, 'anchorKind') || undefined,
      anchorNumber: numberField(item, 'anchorNumber'),
    }))
    .filter((item) => Boolean(
      item.headingPath
      || item.snippet
      || item.highlightSnippet
      || item.anchorId
      || item.blockId
      || item.anchorKind
      || item.anchorNumber,
    ))
  return out.length > 0 ? out : undefined
}

function normalizeSessionHighlights(value: unknown): ReaderSessionHighlight[] {
  if (!Array.isArray(value)) return []
  return value
    .filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === 'object')
    .map((item) => ({
      id: stringField(item, 'id') || `imported-${Math.random().toString(36).slice(2, 10)}`,
      text: stringField(item, 'text'),
      startOffset: numberField(item, 'startOffset'),
      endOffset: numberField(item, 'endOffset'),
      blockId: stringField(item, 'blockId') || undefined,
      anchorId: stringField(item, 'anchorId') || undefined,
      occurrence: numberField(item, 'occurrence'),
      readableIndex: numberField(item, 'readableIndex'),
      documentOccurrence: numberField(item, 'documentOccurrence'),
      startReadableIndex: numberField(item, 'startReadableIndex'),
      endReadableIndex: numberField(item, 'endReadableIndex'),
    }))
    .filter((item) => Boolean(item.id && item.text))
}

function normalizeLocateTarget(value: unknown): ReaderLocateTarget | undefined {
  if (!value || typeof value !== 'object') return undefined
  const rec = value as Record<string, unknown>
  const target: ReaderLocateTarget = {
    segmentId: stringField(rec, 'segmentId') || undefined,
    sourceSegmentId: stringField(rec, 'sourceSegmentId') || undefined,
    headingPath: stringField(rec, 'headingPath') || undefined,
    snippet: stringField(rec, 'snippet') || undefined,
    highlightSnippet: stringField(rec, 'highlightSnippet') || undefined,
    evidenceQuote: stringField(rec, 'evidenceQuote') || undefined,
    anchorText: stringField(rec, 'anchorText') || undefined,
    hitLevel: stringField(rec, 'hitLevel') || undefined,
    blockId: stringField(rec, 'blockId') || undefined,
    anchorId: stringField(rec, 'anchorId') || undefined,
    anchorKind: stringField(rec, 'anchorKind') || undefined,
    anchorNumber: numberField(rec, 'anchorNumber'),
    claimType: stringField(rec, 'claimType') || undefined,
    locatePolicy: stringField(rec, 'locatePolicy') || undefined,
    locateSurfacePolicy: stringField(rec, 'locateSurfacePolicy') || undefined,
    snippetAliases: Array.isArray(rec.snippetAliases)
      ? rec.snippetAliases.map((item) => String(item || '').trim()).filter(Boolean)
      : undefined,
    relatedBlockIds: Array.isArray(rec.relatedBlockIds)
      ? rec.relatedBlockIds.map((item) => String(item || '').trim()).filter(Boolean)
      : undefined,
  }
  if (Object.values(target).some((item) => Boolean(item))) return target
  return undefined
}

function normalizeReaderPayload(record: ReaderSessionRecord | null): ReaderOpenPayload | null {
  const rec = (record?.payload && typeof record.payload === 'object')
    ? record.payload as Record<string, unknown>
    : null
  if (!rec) return null
  const sourcePath = stringField(rec, 'sourcePath') || stringField(rec, 'source_path')
  if (!sourcePath) return null
  return {
    sourcePath,
    sourceName: stringField(rec, 'sourceName') || stringField(rec, 'source_name') || record?.title || undefined,
    headingPath: stringField(rec, 'headingPath') || undefined,
    snippet: stringField(rec, 'snippet') || undefined,
    highlightSnippet: stringField(rec, 'highlightSnippet') || undefined,
    anchorId: stringField(rec, 'anchorId') || undefined,
    blockId: stringField(rec, 'blockId') || undefined,
    relatedBlockIds: Array.isArray(rec.relatedBlockIds)
      ? rec.relatedBlockIds.map((item) => String(item || '').trim()).filter(Boolean)
      : undefined,
    anchorKind: stringField(rec, 'anchorKind') || undefined,
    anchorNumber: numberField(rec, 'anchorNumber'),
    strictLocate: Boolean(rec.strictLocate),
    locateMode: rec.locateMode === 'heuristic' ? 'heuristic' : undefined,
    locateTarget: normalizeLocateTarget(rec.locateTarget),
    claimGroup: rec.claimGroup && typeof rec.claimGroup === 'object'
      ? {
        id: stringField(rec.claimGroup as Record<string, unknown>, 'id') || undefined,
        kind: stringField(rec.claimGroup as Record<string, unknown>, 'kind') || undefined,
        leadText: stringField(rec.claimGroup as Record<string, unknown>, 'leadText') || undefined,
        distance: numberField(rec.claimGroup as Record<string, unknown>, 'distance'),
      }
      : undefined,
    locateRequestId: numberField(rec, 'locateRequestId'),
    alternatives: candidateList(rec.alternatives),
    visibleAlternatives: candidateList(rec.visibleAlternatives),
    evidenceAlternatives: candidateList(rec.evidenceAlternatives),
    initialAltIndex: numberField(rec, 'initialAltIndex'),
    locateFeedbackKey: stringField(rec, 'locateFeedbackKey') || undefined,
  }
}

export default function ReaderPage() {
  const S = useT()
  const navigate = useNavigate()
  const params = useParams<{ sessionId: string }>()
  const sessionId = String(params.sessionId || '').trim()
  const [session, setSession] = useState<ReaderSessionRecord | null>(null)
  const [loading, setLoading] = useState(() => Boolean(sessionId))
  const [error, setError] = useState('')
  const [reloadToken, setReloadToken] = useState(0)
  const [sessionHighlights, setSessionHighlights] = useState<ReaderSessionHighlight[]>([])
  const stateHydratedRef = useRef(false)
  const broadcastRef = useRef<BroadcastChannel | null>(null)

  useEffect(() => {
    let cancelled = false
    if (!sessionId) {
      return
    }
    Promise.resolve().then(() => {
      if (cancelled) return
      setLoading(true)
      setError('')
    })
    chatApi.getReaderSession(sessionId)
      .then((res) => {
        if (cancelled) return
        stateHydratedRef.current = false
        setSession(res)
        const state = (res.state && typeof res.state === 'object')
          ? res.state as Record<string, unknown>
          : {}
        setSessionHighlights(normalizeSessionHighlights(state.highlights))
        window.requestAnimationFrame(() => {
          stateHydratedRef.current = true
        })
      })
      .catch((err) => {
        if (cancelled) return
        stateHydratedRef.current = false
        setSession(null)
        setSessionHighlights([])
        setError(err instanceof Error ? err.message : (S.reader_standalone_missing || 'Reader session not found.'))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [S.reader_standalone_missing, reloadToken, sessionId])

  const payload = useMemo(() => normalizeReaderPayload(session), [session])
  const payloadSourcePath = String(payload?.sourcePath || '').trim()
  const sessionConversationId = String(session?.conversation_id || '').trim()
  const title = String(session?.title || payload?.sourceName || payload?.sourcePath || S.side_dock_reader || 'Reader').trim()

  useEffect(() => {
    if (typeof BroadcastChannel === 'undefined') return undefined
    const channel = new BroadcastChannel(READER_SESSION_SYNC_CHANNEL)
    broadcastRef.current = channel
    channel.onmessage = (event) => {
      const data = (event?.data && typeof event.data === 'object')
        ? event.data as Record<string, unknown>
        : {}
      if (String(data.type || '') !== 'reader-session-state') return
      const sourcePath = String(data.sourcePath || '').trim()
      if (!sourcePath || sourcePath !== payloadSourcePath) return
      if (String(data.sessionId || '').trim() === sessionId) return
      const highlights = normalizeSessionHighlights(data.highlights)
      setSessionHighlights((current) => {
        if (JSON.stringify(current) === JSON.stringify(highlights)) return current
        return highlights
      })
    }
    return () => {
      channel.close()
      if (broadcastRef.current === channel) broadcastRef.current = null
    }
  }, [payloadSourcePath, sessionId])

  const publishState = useCallback((state: Record<string, unknown>) => {
    if (!sessionId || !payloadSourcePath) return
    const nextState = {
      ...state,
      sourcePath: payloadSourcePath,
      conversationId: sessionConversationId,
    }
    void chatApi.updateReaderSessionState(sessionId, nextState).catch(() => {})
    broadcastRef.current?.postMessage({
      type: 'reader-session-state',
      sessionId,
      ...nextState,
    })
  }, [payloadSourcePath, sessionConversationId, sessionId])

  useEffect(() => {
    if (!stateHydratedRef.current || !payloadSourcePath) return
    publishState({ highlights: sessionHighlights, updatedAt: Date.now() })
  }, [payloadSourcePath, publishState, sessionHighlights])

  const goBack = useCallback(() => {
    if (window.opener && window.history.length <= 1) {
      window.close()
      return
    }
    navigate('/')
  }, [navigate])

  const appendSelection = useCallback((text: string) => {
    const raw = String(text || '').trim()
    if (!raw) return
    if (navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(raw)
        .then(() => message.success(S.reader_selection_copied || 'Selection copied'))
        .catch(() => message.info(S.reader_selection_ready || 'Selection ready to copy'))
    } else {
      message.info(S.reader_selection_ready || 'Selection ready to copy')
    }
    publishState({
      selection: {
        text: raw,
        sourcePath: payload?.sourcePath || '',
        updatedAt: Date.now(),
      },
    })
  }, [S.reader_selection_copied, S.reader_selection_ready, payload?.sourcePath, publishState])

  const addSelectionToShelf = useCallback((selection: ReaderSelectionShelfPayload) => {
    const text = String(selection?.text || '').trim()
    const sourcePath = String(selection?.sourcePath || payload?.sourcePath || '').trim()
    if (!text || !sourcePath) return
    const next: ReaderSelectionShelfPayload = {
      ...selection,
      text,
      sourcePath,
      sourceName: selection.sourceName || payload?.sourceName || title,
      conversationId: session?.conversation_id || selection.conversationId || '',
      createdAt: Number(selection.createdAt || Date.now()),
    }
    if (typeof BroadcastChannel !== 'undefined') {
      const channel = new BroadcastChannel(READER_SELECTION_SHELF_CHANNEL)
      channel.postMessage({
        type: 'reader-selection-shelf',
        ...next,
      })
      channel.close()
    }
    message.success(S.reader_added_to_shelf || 'Added to citation shelf')
  }, [S.reader_added_to_shelf, payload?.sourceName, payload?.sourcePath, session?.conversation_id, title])

  const addSessionHighlight = useCallback((highlight: ReaderSessionHighlight) => {
    setSessionHighlights((current) => {
      if (current.some((item) => item.id === highlight.id)) return current
      return [...current, highlight]
    })
  }, [])

  const removeSessionHighlight = useCallback((highlightId: string) => {
    const target = String(highlightId || '').trim()
    if (!target) return
    setSessionHighlights((current) => current.filter((item) => item.id !== target))
  }, [])

  const recordLocateResult = useCallback((result: ReaderLocateResult) => {
    publishState({
      locateResult: {
        ...result,
        updatedAt: Date.now(),
      },
    })
  }, [publishState])

  return (
    <div className="kb-reader-page">
      <header className="kb-reader-page-bar">
        <Button
          type="text"
          icon={<ArrowLeftOutlined />}
          className="kb-reader-page-back"
          onClick={goBack}
        >
          {S.reader_standalone_back || 'Back'}
        </Button>
        <div className="kb-reader-page-heading">
          <div className="kb-reader-page-kicker">{S.side_dock_reader || 'Reader'}</div>
          <div className="kb-reader-page-title" title={title}>{title}</div>
        </div>
        <Button
          type="text"
          icon={<ReloadOutlined />}
          className="kb-reader-page-back"
          onClick={() => setReloadToken((value) => value + 1)}
        >
          {S.reader_reload || 'Reload'}
        </Button>
      </header>
      <main className="kb-reader-page-main">
        {loading ? (
          <div className="kb-reader-page-state">
            <Spin />
            <span>{S.reader_standalone_loading || 'Loading reader...'}</span>
          </div>
        ) : error || !payload ? (
          <div className="kb-reader-page-empty">
            <Empty description={error || (S.reader_standalone_missing || 'Reader session not found.')} />
          </div>
        ) : (
          <PaperGuideReaderDrawer
            open
            payload={payload}
            onClose={goBack}
            onAppendSelection={appendSelection}
            presentation="inline"
            surface="page"
            sessionHighlights={sessionHighlights}
            onAddSessionHighlight={addSessionHighlight}
            onRemoveSessionHighlight={removeSessionHighlight}
            onLocateResult={recordLocateResult}
            onAddSelectionToShelf={addSelectionToShelf}
          />
        )}
      </main>
    </div>
  )
}
