import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Alert, Button, Empty, Spin, message } from 'antd'
import { ArrowLeftOutlined, ReloadOutlined } from '@ant-design/icons'
import { useLocation, useNavigate, useParams } from 'react-router-dom'
import {
  chatApi,
  type ReaderSessionRecord,
  type ResearchNoteEvidenceContextSnapshot,
  type ResearchNoteEvidenceContextSnapshotBlock,
  type ResearchNoteEvidenceRepairIntent,
  type ResearchNoteEvidenceReviewIntent,
} from '../api/chat'
import { PaperGuideReaderDrawer } from '../components/chat/PaperGuideReaderDrawer'
import type {
  ReaderSelectionShelfPayload,
  ReaderLocateCandidate,
  ReaderLocateResult,
  ReaderLocateTarget,
  ReaderOpenPayload,
  ReaderSessionHighlight,
} from '../components/chat/reader/readerTypes'
import {
  CHAT_MAIN_WINDOW_NAME,
  READER_CITATION_SHELF_CHANNEL,
  READER_SELECTION_SHELF_CHANNEL,
  READER_SESSION_NAV_CHANNEL,
  READER_STANDALONE_WINDOW_NAME,
  READER_SESSION_SYNC_CHANNEL,
} from '../components/chat/reader/readerTypes'
import { readerSourcePathsMatch } from '../components/chat/reader/readerLocateGuard'
import {
  buildHeuristicReaderLocateRecoveryPayload,
  shouldRecoverReaderLocateHeuristically,
} from '../components/chat/reader/readerLocateRecovery'
import {
  sanitizeReaderLocateCandidates,
  sanitizeReaderLocateTarget,
} from '../components/chat/reader/readerOpenPayloadUtils'
import { readerHighlightsSignature, stableReaderHighlightId } from '../components/chat/reader/readerSessionState'
import type { CiteDetail } from '../components/chat/citationState'
import { useT } from '../i18n'

const LINKED_CONVERSATION_QUERY_KEYS = ['conversation', 'conversation_id', 'conv'] as const

function linkedConversationIdFromSearch(search: string) {
  const params = new URLSearchParams(search)
  for (const key of LINKED_CONVERSATION_QUERY_KEYS) {
    const value = String(params.get(key) || '').trim()
    if (value) return value
  }
  return ''
}

function stringField(rec: Record<string, unknown>, key: string) {
  return String(rec[key] || '').trim()
}

function readerNavRequestId() {
  return `reader-nav-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`
}

function conversationUrl(conversationId: string) {
  const params = new URLSearchParams()
  if (conversationId) params.set('conversation', conversationId)
  const url = new URL('/', window.location.origin)
  const search = params.toString()
  if (search) url.search = search
  return url.toString()
}

function readerSelectionShelfKey(selection: ReaderSelectionShelfPayload) {
  return [
    'reader-selection',
    selection.sourcePath,
    selection.blockId || selection.anchorId || '',
    selection.startOffset ?? '',
    selection.endOffset ?? '',
    String(selection.text || '').slice(0, 80),
  ].join(':')
}

function numberField(rec: Record<string, unknown>, key: string) {
  const value = Number(rec[key])
  return Number.isFinite(value) ? value : undefined
}

function candidateList(value: unknown): ReaderLocateCandidate[] | undefined {
  const out = sanitizeReaderLocateCandidates(value)
  return out.length > 0 ? out : undefined
}

function normalizeSessionHighlights(value: unknown): ReaderSessionHighlight[] {
  if (!Array.isArray(value)) return []
  return value
    .filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === 'object')
    .map((item) => {
      const highlight = {
        id: stringField(item, 'id'),
        text: stringField(item, 'text'),
        noteKind: stringField(item, 'noteKind') || undefined,
        sourcePath: stringField(item, 'sourcePath') || undefined,
        sourceName: stringField(item, 'sourceName') || undefined,
        conversationId: stringField(item, 'conversationId') || undefined,
        messageId: numberField(item, 'messageId'),
        locateRequestId: numberField(item, 'locateRequestId'),
        locateFeedbackKey: stringField(item, 'locateFeedbackKey') || undefined,
        createdAt: numberField(item, 'createdAt'),
        updatedAt: numberField(item, 'updatedAt'),
        feedback: stringField(item, 'feedback') || undefined,
        feedbackAt: numberField(item, 'feedbackAt'),
        headingPath: stringField(item, 'headingPath') || undefined,
        startOffset: numberField(item, 'startOffset'),
        endOffset: numberField(item, 'endOffset'),
        blockId: stringField(item, 'blockId') || undefined,
        anchorId: stringField(item, 'anchorId') || undefined,
        occurrence: numberField(item, 'occurrence'),
        readableIndex: numberField(item, 'readableIndex'),
        documentOccurrence: numberField(item, 'documentOccurrence'),
        startReadableIndex: numberField(item, 'startReadableIndex'),
        endReadableIndex: numberField(item, 'endReadableIndex'),
      }
      return {
        ...highlight,
        id: highlight.id || stableReaderHighlightId(highlight),
      }
    })
    .filter((item) => Boolean(item.id && item.text))
}

function normalizeLocateTarget(value: unknown): ReaderLocateTarget | undefined {
  return sanitizeReaderLocateTarget(value)
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

function normalizeEvidenceRepairIntent(value: unknown): ResearchNoteEvidenceRepairIntent | null {
  const record = value && typeof value === 'object' ? value as Record<string, unknown> : null
  const repair = record?.repair && typeof record.repair === 'object'
    ? record.repair as Record<string, unknown>
    : null
  const noteId = String(record?.note_id || '').trim()
  const sourcePath = String(repair?.source_path || '').trim()
  const evidenceQuote = String(repair?.candidate_evidence_quote || '').trim()
  const fingerprint = String(repair?.source_fingerprint || '').trim()
  const sourceIndex = Number(repair?.source_index)
  if (
    Number(record?.version) !== 1
    || !noteId
    || !sourcePath
    || !evidenceQuote
    || !fingerprint
    || !Number.isInteger(sourceIndex)
    || sourceIndex < 0
  ) return null
  return {
    version: 1,
    note_id: noteId,
    note_title: String(record?.note_title || '').trim(),
    source_label: String(record?.source_label || '').trim(),
    repair: {
      source_index: sourceIndex,
      source_path: sourcePath,
      capture_id: String(repair?.capture_id || '').trim() || undefined,
      old_block_id: String(repair?.old_block_id || '').trim() || undefined,
      old_anchor_id: String(repair?.old_anchor_id || '').trim() || undefined,
      old_evidence_quote: String(repair?.old_evidence_quote || '').trim() || undefined,
      candidate_block_id: String(repair?.candidate_block_id || '').trim() || undefined,
      candidate_anchor_id: String(repair?.candidate_anchor_id || '').trim() || undefined,
      candidate_anchor_kind: String(repair?.candidate_anchor_kind || '').trim() || undefined,
      candidate_evidence_quote: evidenceQuote,
      source_fingerprint: fingerprint,
    },
  }
}

function normalizeEvidenceContextSnapshotBlock(value: unknown): ResearchNoteEvidenceContextSnapshotBlock | null {
  const record = value && typeof value === 'object' ? value as Record<string, unknown> : null
  const text = String(record?.text || '').trim()
  if (!record || !text) return null
  const positiveNumber = (key: string) => {
    const number = Number(record[key])
    return Number.isFinite(number) && number > 0 ? number : undefined
  }
  return {
    block_id: String(record.block_id || '').trim() || undefined,
    anchor_id: String(record.anchor_id || '').trim() || undefined,
    kind: String(record.kind || '').trim() || undefined,
    heading_path: String(record.heading_path || '').trim() || undefined,
    page_start: positiveNumber('page_start'),
    page_end: positiveNumber('page_end'),
    line_start: positiveNumber('line_start'),
    line_end: positiveNumber('line_end'),
    text,
  }
}

function normalizeEvidenceContextSnapshot(value: unknown): ResearchNoteEvidenceContextSnapshot | null {
  const record = value && typeof value === 'object' ? value as Record<string, unknown> : null
  const target = normalizeEvidenceContextSnapshotBlock(record?.target)
  const fingerprint = String(record?.source_fingerprint || '').trim()
  const capturedAt = Number(record?.captured_at)
  if (!record || Number(record.version) !== 1 || !target || !fingerprint.startsWith('sha256:')) return null
  const blockList = (input: unknown) => Array.isArray(input)
    ? input.map(normalizeEvidenceContextSnapshotBlock).filter((item): item is ResearchNoteEvidenceContextSnapshotBlock => Boolean(item)).slice(0, 1)
    : []
  return {
    version: 1,
    captured_at: Number.isFinite(capturedAt) && capturedAt > 0 ? capturedAt : 0,
    source_fingerprint: fingerprint,
    reason: String(record.reason || '').trim() || undefined,
    before: blockList(record.before),
    target,
    after: blockList(record.after),
  }
}

type EvidenceContextRole = 'before' | 'target' | 'after'

function evidenceContextRoleText(
  snapshot: ResearchNoteEvidenceContextSnapshot | null | undefined,
  role: EvidenceContextRole,
): string {
  if (!snapshot) return ''
  if (role === 'target') return String(snapshot.target?.text || '').trim()
  const blocks = role === 'before' ? snapshot.before : snapshot.after
  return String(blocks?.[0]?.text || '').trim()
}

function evidenceContextChanged(
  previous: ResearchNoteEvidenceContextSnapshot | null | undefined,
  current: ResearchNoteEvidenceContextSnapshot | null | undefined,
): boolean {
  if (!previous || !current) return true
  return (['before', 'target', 'after'] as EvidenceContextRole[]).some(
    role => evidenceContextRoleText(previous, role) !== evidenceContextRoleText(current, role),
  )
}

function renderEvidenceContextDifference(previousText: string, currentText: string, side: 'previous' | 'current') {
  const left = Array.from(previousText)
  const right = Array.from(currentText)
  let prefixLength = 0
  while (prefixLength < left.length && prefixLength < right.length && left[prefixLength] === right[prefixLength]) {
    prefixLength += 1
  }
  let suffixLength = 0
  while (
    suffixLength < left.length - prefixLength
    && suffixLength < right.length - prefixLength
    && left[left.length - 1 - suffixLength] === right[right.length - 1 - suffixLength]
  ) {
    suffixLength += 1
  }
  const source = side === 'previous' ? left : right
  const prefix = source.slice(0, prefixLength).join('')
  const changed = source.slice(prefixLength, source.length - suffixLength).join('')
  const suffix = suffixLength > 0 ? source.slice(source.length - suffixLength).join('') : ''
  if (!changed && previousText === currentText) return <>{source.join('')}</>
  return (
    <>
      {prefix}
      {changed ? <mark className={`is-${side}`}>{changed}</mark> : <mark className={`is-${side} is-empty`}>∅</mark>}
      {suffix}
    </>
  )
}

function EvidenceContextSnapshotColumn({
  snapshot,
  counterpart,
  side,
  title,
  roleLabels,
  emptyLabel,
  testId,
}: {
  snapshot: ResearchNoteEvidenceContextSnapshot
  counterpart?: ResearchNoteEvidenceContextSnapshot | null
  side: 'previous' | 'current'
  title: string
  roleLabels: Record<EvidenceContextRole, string>
  emptyLabel: string
  testId: string
}) {
  return (
    <div data-testid={testId}>
      <header>
        <strong>{title}</strong>
        {snapshot.captured_at > 0 ? <small>{new Date(snapshot.captured_at * 1000).toLocaleString()}</small> : null}
      </header>
      <div className="kb-reader-evidence-context-segments">
        {(['before', 'target', 'after'] as EvidenceContextRole[]).map((role) => {
          const text = evidenceContextRoleText(snapshot, role)
          const otherText = evidenceContextRoleText(counterpart, role)
          return (
            <section className={`is-${role}`} key={role} data-testid={`${testId}-${role}`}>
              <span>{roleLabels[role]}</span>
              <p>
                {text
                  ? counterpart
                    ? renderEvidenceContextDifference(
                      side === 'previous' ? text : otherText,
                      side === 'previous' ? otherText : text,
                      side,
                    )
                    : text
                  : <em>{emptyLabel}</em>}
              </p>
            </section>
          )
        })}
      </div>
    </div>
  )
}

function normalizeEvidenceReviewIntent(value: unknown): ResearchNoteEvidenceReviewIntent | null {
  const record = value && typeof value === 'object' ? value as Record<string, unknown> : null
  const review = record?.review && typeof record.review === 'object'
    ? record.review as Record<string, unknown>
    : null
  const noteId = String(record?.note_id || '').trim()
  const sourcePath = String(review?.source_path || '').trim()
  const expectedFingerprint = String(review?.expected_source_fingerprint || '').trim()
  const currentFingerprint = String(review?.source_fingerprint || '').trim()
  const currentQuote = String(review?.current_evidence_quote || '').trim()
  const sourceIndex = Number(review?.source_index)
  if (
    Number(record?.version) !== 1
    || !noteId
    || !sourcePath
    || !expectedFingerprint
    || !currentFingerprint
    || !currentQuote
    || !Number.isInteger(sourceIndex)
    || sourceIndex < 0
  ) return null
  return {
    version: 1,
    note_id: noteId,
    note_title: String(record?.note_title || '').trim(),
    source_label: String(record?.source_label || '').trim(),
    saved_evidence_quote: String(record?.saved_evidence_quote || '').trim(),
    current_evidence_quote: String(record?.current_evidence_quote || currentQuote).trim(),
    previously_invalid: Boolean(record?.previously_invalid),
    saved_context_snapshot: normalizeEvidenceContextSnapshot(record?.saved_context_snapshot),
    current_context_snapshot: normalizeEvidenceContextSnapshot(record?.current_context_snapshot),
    review: {
      source_index: sourceIndex,
      source_path: sourcePath,
      capture_id: String(review?.capture_id || '').trim() || undefined,
      old_block_id: String(review?.old_block_id || '').trim() || undefined,
      old_anchor_id: String(review?.old_anchor_id || '').trim() || undefined,
      old_evidence_quote: String(review?.old_evidence_quote || '').trim() || undefined,
      expected_source_fingerprint: expectedFingerprint,
      current_block_id: String(review?.current_block_id || '').trim() || undefined,
      current_anchor_id: String(review?.current_anchor_id || '').trim() || undefined,
      current_anchor_kind: String(review?.current_anchor_kind || '').trim() || undefined,
      current_evidence_quote: currentQuote,
      source_fingerprint: currentFingerprint,
    },
  }
}

function normalizeSavedLocateResult(value: unknown): ReaderLocateResult | null {
  const record = value && typeof value === 'object' ? value as Record<string, unknown> : null
  const status = String(record?.status || '') as ReaderLocateResult['status']
  const precision = String(record?.precision || '') as ReaderLocateResult['precision']
  const sourcePath = String(record?.sourcePath || '').trim()
  if (!record || !sourcePath || !status || !precision) return null
  return {
    locateRequestId: Number(record.locateRequestId || 0),
    sourcePath,
    sourceName: String(record.sourceName || '').trim() || undefined,
    locateFeedbackKey: String(record.locateFeedbackKey || '').trim() || undefined,
    status,
    precision,
    ok: Boolean(record.ok),
    repairable: Boolean(record.repairable),
    strictLocate: Boolean(record.strictLocate),
    hint: String(record.hint || '').trim(),
    reason: String(record.reason || '').trim(),
    activeAltIndex: Number.isFinite(Number(record.activeAltIndex)) ? Number(record.activeAltIndex) : undefined,
    blockId: String(record.blockId || '').trim() || undefined,
    anchorId: String(record.anchorId || '').trim() || undefined,
    anchorKind: String(record.anchorKind || '').trim() || undefined,
    headingPath: String(record.headingPath || '').trim() || undefined,
  }
}

export default function ReaderPage() {
  const S = useT()
  const navigate = useNavigate()
  const location = useLocation()
  const params = useParams<{ sessionId: string }>()
  const sessionId = String(params.sessionId || '').trim()
  const [session, setSession] = useState<ReaderSessionRecord | null>(null)
  const [loading, setLoading] = useState(() => Boolean(sessionId))
  const [error, setError] = useState('')
  const [reloadToken, setReloadToken] = useState(0)
  const [sessionHighlights, setSessionHighlights] = useState<ReaderSessionHighlight[]>([])
  const [locateResult, setLocateResult] = useState<ReaderLocateResult | null>(null)
  const [standaloneRecoveryPayload, setStandaloneRecoveryPayload] = useState<ReaderOpenPayload | null>(null)
  const [repairingEvidence, setRepairingEvidence] = useState(false)
  const [repairError, setRepairError] = useState('')
  const [evidenceReviewDecision, setEvidenceReviewDecision] = useState<'confirmed' | 'invalid' | null>(null)
  const [evidenceReviewError, setEvidenceReviewError] = useState('')
  const stateHydratedRef = useRef(false)
  const broadcastRef = useRef<BroadcastChannel | null>(null)
  const pendingStateRef = useRef<Record<string, unknown>>({})
  const saveTimerRef = useRef<number | null>(null)

  useEffect(() => {
    if (typeof window === 'undefined') return
    if (window.name !== READER_STANDALONE_WINDOW_NAME) {
      window.name = READER_STANDALONE_WINDOW_NAME
    }
  }, [])

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
      .then(async (res) => {
        if (cancelled) return
        stateHydratedRef.current = false
        setSession(res)
        setStandaloneRecoveryPayload(null)
        const state = (res.state && typeof res.state === 'object')
          ? res.state as Record<string, unknown>
          : {}
        setLocateResult(normalizeSavedLocateResult(state.locateResult))
        setRepairError('')
        setEvidenceReviewError('')
        setEvidenceReviewDecision(null)
        let highlights = normalizeSessionHighlights(state.highlights || state.evidenceNotes)
        const payloadFromRecord = normalizeReaderPayload(res)
        const sourcePath = String(payloadFromRecord?.sourcePath || '').trim()
        const conversationId = String(res.conversation_id || linkedConversationIdFromSearch(location.search)).trim()
        if (highlights.length === 0 && conversationId && sourcePath) {
          try {
            const readerState = await chatApi.getConversationReaderState(conversationId, sourcePath)
            if (!cancelled) {
              highlights = normalizeSessionHighlights(readerState.state?.highlights || readerState.state?.evidenceNotes)
            }
          } catch {
            // Reader sessions remain usable even when the linked conversation state is unavailable.
          }
        }
        if (cancelled) return
        setSessionHighlights(highlights)
        window.requestAnimationFrame(() => {
          stateHydratedRef.current = true
        })
      })
      .catch((err) => {
        if (cancelled) return
        stateHydratedRef.current = false
        setSession(null)
        setSessionHighlights([])
        setLocateResult(null)
        setStandaloneRecoveryPayload(null)
        setError(err instanceof Error ? err.message : (S.reader_standalone_missing || 'Reader session not found.'))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [S.reader_standalone_missing, location.search, reloadToken, sessionId])

  const sessionPayload = useMemo(() => normalizeReaderPayload(session), [session])
  const repairIntent = useMemo(
    () => normalizeEvidenceRepairIntent(
      session?.state && typeof session.state === 'object'
        ? (session.state as Record<string, unknown>).researchNoteEvidenceRepair
        : null,
    ),
    [session?.state],
  )
  const evidenceReviewIntent = useMemo(
    () => normalizeEvidenceReviewIntent(
      session?.state && typeof session.state === 'object'
        ? (session.state as Record<string, unknown>).researchNoteEvidenceReview
        : null,
    ),
    [session?.state],
  )
  const payload = standaloneRecoveryPayload || sessionPayload
  const payloadSourcePath = String(payload?.sourcePath || '').trim()
  const sessionConversationId = String(session?.conversation_id || linkedConversationIdFromSearch(location.search)).trim()
  const sessionProjectId = String(
    (session?.state && typeof session.state === 'object' ? (session.state as Record<string, unknown>).projectId || (session.state as Record<string, unknown>).project_id : '')
    || (session?.payload && typeof session.payload === 'object' ? (session.payload as Record<string, unknown>).projectId || (session.payload as Record<string, unknown>).project_id : '')
    || '',
  ).trim()
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
      if (!readerSourcePathsMatch(sourcePath, payloadSourcePath)) return
      if (String(data.sessionId || '').trim() === sessionId) return
      const highlights = normalizeSessionHighlights(data.highlights)
      setSessionHighlights((current) => {
        if (readerHighlightsSignature(current) === readerHighlightsSignature(highlights)) return current
        return highlights
      })
    }
    return () => {
      channel.close()
      if (broadcastRef.current === channel) broadcastRef.current = null
    }
  }, [payloadSourcePath, sessionId])

  const flushPendingState = useCallback(() => {
    const pending = pendingStateRef.current
    if (Object.keys(pending).length === 0) return
    pendingStateRef.current = {}
    if (saveTimerRef.current !== null) {
      window.clearTimeout(saveTimerRef.current)
      saveTimerRef.current = null
    }
    if (sessionId && payloadSourcePath) {
      void chatApi.updateReaderSessionState(sessionId, pending).catch(() => {})
    }
    if (sessionConversationId && payloadSourcePath) {
      void chatApi.updateConversationReaderState(sessionConversationId, payloadSourcePath, pending).catch(() => {})
    }
  }, [payloadSourcePath, sessionConversationId, sessionId])

  const scheduleStateSave = useCallback(() => {
    if (saveTimerRef.current !== null) window.clearTimeout(saveTimerRef.current)
    saveTimerRef.current = window.setTimeout(() => {
      saveTimerRef.current = null
      flushPendingState()
    }, 600)
  }, [flushPendingState])

  useEffect(() => () => {
    flushPendingState()
  }, [flushPendingState])

  const publishState = useCallback((state: Record<string, unknown>, opts?: { flushNow?: boolean }) => {
    if (!sessionId || !payloadSourcePath) return
    const nextState = {
      ...state,
      sourcePath: payloadSourcePath,
      conversationId: sessionConversationId,
    }
    pendingStateRef.current = {
      ...pendingStateRef.current,
      ...nextState,
    }
    broadcastRef.current?.postMessage({
      type: 'reader-session-state',
      sessionId,
      ...nextState,
    })
    if (opts?.flushNow) {
      flushPendingState()
    } else {
      scheduleStateSave()
    }
  }, [flushPendingState, payloadSourcePath, scheduleStateSave, sessionConversationId, sessionId])

  useEffect(() => {
    if (!stateHydratedRef.current || !payloadSourcePath) return
    publishState({ highlights: sessionHighlights, evidenceNotes: sessionHighlights, updatedAt: Date.now() })
  }, [payloadSourcePath, publishState, sessionHighlights])

  const returnToConversationWindow = useCallback((opts?: { closeReader?: boolean; openMainWindow?: boolean }) => {
    if (!sessionConversationId) return false
    flushPendingState()
    const requestId = readerNavRequestId()
    const navPayload = {
      type: 'reader-return-to-conversation',
      requestId,
      conversationId: sessionConversationId,
      sessionId,
      sourcePath: payloadSourcePath,
    }
    let channel: BroadcastChannel | null = null
    let acknowledged = false
    if (typeof BroadcastChannel !== 'undefined') {
      channel = new BroadcastChannel(READER_SESSION_NAV_CHANNEL)
      channel.onmessage = (event) => {
        const data = (event?.data && typeof event.data === 'object')
          ? event.data as Record<string, unknown>
          : {}
        if (
          String(data.type || '') === 'reader-return-to-conversation-ack'
          && String(data.requestId || '') === requestId
        ) {
          acknowledged = true
          channel?.close()
          channel = null
          if (opts?.closeReader) {
            window.setTimeout(() => window.close(), 60)
          }
        }
      }
      channel.postMessage(navPayload)
    }
    if (window.opener && !window.opener.closed) {
      try {
        window.opener.postMessage(navPayload, window.location.origin)
        window.opener.focus()
      } catch {
        // BroadcastChannel remains the fallback when direct opener messaging is unavailable.
      }
      if (opts?.closeReader) {
        window.setTimeout(() => window.close(), 80)
        window.setTimeout(() => {
          if (!window.closed) {
            message.info(S.reader_returned_to_session || 'Switched the original session window. You can close this reader window.')
          }
        }, 220)
      }
      return true
    }

    if (opts?.openMainWindow !== false) {
      window.setTimeout(() => {
        if (acknowledged) return
        try {
          const target = window.open(conversationUrl(sessionConversationId), CHAT_MAIN_WINDOW_NAME)
          if (target) {
            target.focus()
            if (opts?.closeReader) {
              window.setTimeout(() => window.close(), 80)
            }
            window.setTimeout(() => {
              channel?.close()
            }, 800)
            return
          }
        } catch {
          // If focusing the named chat window is blocked, keep the reader open with a clear hint.
        }
        channel?.close()
        if (opts?.closeReader) {
          message.info(S.reader_returned_to_session || 'Switched the original session window. You can close this reader window.')
        }
      }, 180)
      return true
    }

    window.setTimeout(() => {
      channel?.close()
      if (!acknowledged && opts?.closeReader) {
        message.info(S.reader_returned_to_session || 'Switched the original session window. You can close this reader window.')
      }
    }, 360)
    return acknowledged
  }, [S.reader_returned_to_session, flushPendingState, payloadSourcePath, sessionConversationId, sessionId])

  const goBack = useCallback(() => {
    const noteIntent = repairIntent || evidenceReviewIntent
    if (noteIntent) {
      flushPendingState()
      navigate(`/notes?note=${encodeURIComponent(noteIntent.note_id)}`)
      return
    }
    if (sessionConversationId) {
      if (returnToConversationWindow({ closeReader: true, openMainWindow: true })) return
      if (typeof BroadcastChannel !== 'undefined') {
        return
      }
      message.info(S.reader_returned_to_session || 'Switched the original session window. You can close this reader window.')
      return
    }
    if (window.opener && window.history.length <= 1) {
      window.close()
      return
    }
    navigate('/')
  }, [S.reader_returned_to_session, evidenceReviewIntent, flushPendingState, navigate, repairIntent, returnToConversationWindow, sessionConversationId])

  const persistReaderShelfItem = useCallback((item: Record<string, unknown>) => {
    if (!item || Object.keys(item).length <= 0) return
    const convId = String(session?.conversation_id || sessionConversationId || '').trim()
    const projectId = String(sessionProjectId || '').trim()
    void chatApi.appendCitationShelfItem({
      convId: convId || undefined,
      projectId: projectId || undefined,
      scope: 'project',
      item,
      open: true,
    }).catch(() => {
      message.warning(S.reader_shelf_save_failed || 'Saved locally; backend basket sync failed.')
    })
  }, [S.reader_shelf_save_failed, session?.conversation_id, sessionConversationId, sessionProjectId])

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
      projectId: sessionProjectId || selection.projectId || '',
      createdAt: Number(selection.createdAt || Date.now()),
    }
    if (typeof BroadcastChannel !== 'undefined') {
      const requestId = readerNavRequestId()
      const channel = new BroadcastChannel(READER_SELECTION_SHELF_CHANNEL)
      channel.postMessage({
        type: 'reader-selection-shelf',
        requestId,
        ...next,
      })
      channel.close()
    }
    persistReaderShelfItem({
      key: readerSelectionShelfKey(next),
      main: next.sourceName || title,
      title: next.sourceName || title,
      sourcePath,
      source_path: sourcePath,
      sourceName: next.sourceName || payload?.sourceName || title,
      source_name: next.sourceName || payload?.sourceName || title,
      headingPath: next.headingPath || '',
      heading_path: next.headingPath || '',
      blockId: next.blockId || '',
      block_id: next.blockId || '',
      anchorId: next.anchorId || '',
      anchor_id: next.anchorId || '',
      anchorKind: next.anchorKind || '',
      anchor_kind: next.anchorKind || '',
      raw: text,
      citeFmt: text,
      cite_fmt: text,
      evidenceQuote: text,
      evidence_quote: text,
      evidenceSource: 'reader_selection',
      evidence_source: 'reader_selection',
      shelfItemKind: 'reader_selection',
      shelf_item_kind: 'reader_selection',
      shelfOrigin: 'reader_selection',
      shelf_origin: 'reader_selection',
      shelfExcerpt: text,
      shelf_excerpt: text,
      shelfExcerptLabel: 'Selected text',
      shelf_excerpt_label: 'Selected text',
      traceConvId: next.conversationId || '',
      trace_conv_id: next.conversationId || '',
      projectId: next.projectId || '',
      project_id: next.projectId || '',
    })
    message.success(S.reader_added_to_shelf || 'Added to citation shelf')
  }, [S.reader_added_to_shelf, payload?.sourceName, payload?.sourcePath, persistReaderShelfItem, session?.conversation_id, sessionProjectId, title])

  const addCitationToShelf = useCallback((detail: CiteDetail) => {
    if (!detail) return
    if (typeof BroadcastChannel !== 'undefined') {
      const requestId = readerNavRequestId()
      const channel = new BroadcastChannel(READER_CITATION_SHELF_CHANNEL)
      channel.postMessage({
        type: 'reader-citation-shelf',
        requestId,
        detail: detail as unknown as Record<string, unknown>,
        conversationId: session?.conversation_id || sessionConversationId || '',
        projectId: sessionProjectId,
        createdAt: Date.now(),
      })
      channel.close()
    }
    persistReaderShelfItem(detail as unknown as Record<string, unknown>)
    message.success(S.reader_added_to_shelf || 'Added to citation shelf')
  }, [S.reader_added_to_shelf, persistReaderShelfItem, session?.conversation_id, sessionConversationId, sessionProjectId])

  const openCitationShelf = useCallback(() => {
    if (sessionConversationId) {
      returnToConversationWindow({ closeReader: false, openMainWindow: true })
    }
  }, [returnToConversationWindow, sessionConversationId])

  const addSessionHighlight = useCallback((highlight: ReaderSessionHighlight) => {
    setSessionHighlights((current) => {
      if (current.some((item) => item.id === highlight.id)) return current
      return [...current, highlight]
    })
  }, [])

  const updateSessionHighlight = useCallback((highlight: ReaderSessionHighlight) => {
    const targetId = String(highlight?.id || '').trim()
    if (!targetId) return
    setSessionHighlights((current) => {
      let changed = false
      const next = current.map((item) => {
        if (String(item.id || '').trim() !== targetId) return item
        changed = true
        return { ...item, ...highlight }
      })
      return changed ? next : current
    })
  }, [])

  const removeSessionHighlight = useCallback((highlightId: string) => {
    const target = String(highlightId || '').trim()
    if (!target) return
    setSessionHighlights((current) => current.filter((item) => item.id !== target))
  }, [])

  const recordLocateResult = useCallback((result: ReaderLocateResult) => {
    if (
      !repairIntent
      && !evidenceReviewIntent
      && payload
      && shouldRecoverReaderLocateHeuristically(result, payload)
    ) {
      const nextRequestId = Math.max(
        Number(result.locateRequestId || 0),
        Number(payload.locateRequestId || 0),
      ) + 1
      setLocateResult(null)
      setStandaloneRecoveryPayload(
        buildHeuristicReaderLocateRecoveryPayload(payload, nextRequestId),
      )
      return
    }
    setLocateResult(result)
    setRepairError('')
    setEvidenceReviewError('')
    publishState({
      locateResult: {
        ...result,
        updatedAt: Date.now(),
      },
    })
  }, [evidenceReviewIntent, payload, publishState, repairIntent])

  const repairCandidateLocated = useMemo(() => {
    if (!repairIntent || !locateResult) return false
    const repair = repairIntent.repair
    if (!locateResult.ok || !locateResult.strictLocate || locateResult.status !== 'exact') return false
    if (!readerSourcePathsMatch(locateResult.sourcePath, repair.source_path)) return false
    if (repair.candidate_block_id && locateResult.blockId !== repair.candidate_block_id) return false
    if (repair.candidate_anchor_id && locateResult.anchorId !== repair.candidate_anchor_id) return false
    return Boolean(repair.candidate_block_id || repair.candidate_anchor_id)
  }, [locateResult, repairIntent])

  const confirmEvidenceRepair = useCallback(async () => {
    if (!repairIntent || !repairCandidateLocated || repairingEvidence) return
    setRepairingEvidence(true)
    setRepairError('')
    try {
      await chatApi.repairResearchNoteEvidence(repairIntent.note_id, repairIntent.repair)
      flushPendingState()
      navigate(
        `/notes?note=${encodeURIComponent(repairIntent.note_id)}&evidence_repaired=1`,
      )
    } catch {
      setRepairError('')
      message.info(S.reader_evidence_repair_failed)
      flushPendingState()
      navigate(
        `/notes?note=${encodeURIComponent(repairIntent.note_id)}&evidence_recheck=1`,
      )
    } finally {
      setRepairingEvidence(false)
    }
  }, [S.reader_evidence_repair_failed, flushPendingState, navigate, repairCandidateLocated, repairIntent, repairingEvidence])

  const evidenceReviewTargetLocated = useMemo(() => {
    if (!evidenceReviewIntent || !locateResult) return false
    const review = evidenceReviewIntent.review
    if (!locateResult.ok || !locateResult.strictLocate || locateResult.status !== 'exact') return false
    if (!readerSourcePathsMatch(locateResult.sourcePath, review.source_path)) return false
    if (review.current_block_id && locateResult.blockId !== review.current_block_id) return false
    if (review.current_anchor_id && locateResult.anchorId !== review.current_anchor_id) return false
    return Boolean(review.current_block_id || review.current_anchor_id)
  }, [evidenceReviewIntent, locateResult])

  const decideEvidenceReview = useCallback(async (decision: 'confirmed' | 'invalid') => {
    if (!evidenceReviewIntent || !evidenceReviewTargetLocated || evidenceReviewDecision) return
    setEvidenceReviewDecision(decision)
    setEvidenceReviewError('')
    try {
      await chatApi.decideResearchNoteEvidenceReview(evidenceReviewIntent.note_id, {
        ...evidenceReviewIntent.review,
        decision,
      })
      flushPendingState()
      navigate(
        `/notes?note=${encodeURIComponent(evidenceReviewIntent.note_id)}&evidence_reviewed=${decision}`,
      )
    } catch (err) {
      const detail = err instanceof Error ? err.message : S.reader_evidence_review_failed
      setEvidenceReviewError(detail)
      message.error(S.reader_evidence_review_failed)
    } finally {
      setEvidenceReviewDecision(null)
    }
  }, [
    S.reader_evidence_review_failed,
    evidenceReviewDecision,
    evidenceReviewIntent,
    evidenceReviewTargetLocated,
    flushPendingState,
    navigate,
  ])

  const savedContextSnapshot = evidenceReviewIntent?.saved_context_snapshot || null
  const currentContextSnapshot = evidenceReviewIntent?.current_context_snapshot || null
  const localContextChanged = evidenceContextChanged(savedContextSnapshot, currentContextSnapshot)
  const contextRoleLabels: Record<EvidenceContextRole, string> = {
    before: S.reader_evidence_context_before,
    target: S.reader_evidence_context_target,
    after: S.reader_evidence_context_after,
  }

  return (
    <div className="kb-reader-page">
      <header className="kb-reader-page-bar">
        <Button
          type="text"
          icon={<ArrowLeftOutlined />}
          className="kb-reader-page-back"
          onClick={goBack}
        >
          {repairIntent || evidenceReviewIntent
            ? S.reader_evidence_repair_back
            : sessionConversationId
              ? (S.reader_standalone_back_to_session || S.reader_standalone_back || 'Back')
              : (S.reader_standalone_back || 'Back')}
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
      {repairIntent ? (
        <section className="kb-reader-evidence-repair" data-testid="reader-evidence-repair-panel">
          <Alert
            type={repairError ? 'error' : repairCandidateLocated ? 'success' : 'info'}
            showIcon
            message={repairCandidateLocated
              ? S.reader_evidence_repair_exact
              : S.reader_evidence_repair_reviewing}
            description={repairError || (repairCandidateLocated
              ? S.reader_evidence_repair_exact_hint
              : S.reader_evidence_repair_reviewing_hint)}
          />
          <div className="kb-reader-evidence-repair-actions">
            <Button onClick={goBack}>{S.reader_evidence_repair_cancel}</Button>
            <Button
              type="primary"
              disabled={!repairCandidateLocated}
              loading={repairingEvidence}
              data-testid="reader-confirm-evidence-repair"
              onClick={() => { void confirmEvidenceRepair() }}
            >
              {S.reader_evidence_repair_confirm}
            </Button>
          </div>
        </section>
      ) : null}
      {evidenceReviewIntent ? (
        <section
          className="kb-reader-evidence-repair kb-reader-evidence-version-review"
          data-testid="reader-evidence-review-panel"
        >
          <div className="kb-reader-evidence-version-review-main">
            <Alert
              type={evidenceReviewError ? 'error' : evidenceReviewTargetLocated ? 'success' : 'info'}
              showIcon
              message={evidenceReviewTargetLocated
                ? S.reader_evidence_review_ready
                : S.reader_evidence_review_locating}
              description={evidenceReviewError || (evidenceReviewTargetLocated
                ? S.reader_evidence_review_ready_hint
                : S.reader_evidence_review_locating_hint)}
            />
            {evidenceReviewIntent.previously_invalid ? (
              <Alert
                type="warning"
                showIcon
                message={S.reader_evidence_review_previously_invalid}
              />
            ) : null}
            {!savedContextSnapshot ? (
              <Alert
                type="warning"
                showIcon
                message={S.reader_evidence_context_baseline_missing}
                description={S.reader_evidence_context_baseline_missing_hint}
              />
            ) : null}
            {savedContextSnapshot && currentContextSnapshot && !localContextChanged ? (
              <Alert
                type="info"
                showIcon
                message={S.reader_evidence_context_no_local_change}
                description={S.reader_evidence_context_no_local_change_hint}
              />
            ) : null}
            <div className="kb-reader-evidence-version-compare">
              {savedContextSnapshot ? (
                <EvidenceContextSnapshotColumn
                  snapshot={savedContextSnapshot}
                  counterpart={currentContextSnapshot}
                  side="previous"
                  title={S.reader_evidence_context_saved}
                  roleLabels={contextRoleLabels}
                  emptyLabel={S.reader_evidence_context_empty}
                  testId="reader-evidence-review-saved"
                />
              ) : (
                <div data-testid="reader-evidence-review-saved" className="is-context-fallback">
                  <header>
                    <strong>{S.reader_evidence_review_saved_quote}</strong>
                    <small>{S.reader_evidence_context_quote_only}</small>
                  </header>
                  <blockquote>{evidenceReviewIntent.saved_evidence_quote}</blockquote>
                </div>
              )}
              {currentContextSnapshot ? (
                <EvidenceContextSnapshotColumn
                  snapshot={currentContextSnapshot}
                  counterpart={savedContextSnapshot}
                  side="current"
                  title={S.reader_evidence_context_current}
                  roleLabels={contextRoleLabels}
                  emptyLabel={S.reader_evidence_context_empty}
                  testId="reader-evidence-review-current"
                />
              ) : (
                <div data-testid="reader-evidence-review-current" className="is-context-fallback">
                  <header>
                    <strong>{S.reader_evidence_review_current_quote}</strong>
                    <small>{S.reader_evidence_context_quote_only}</small>
                  </header>
                  <blockquote>{evidenceReviewIntent.current_evidence_quote}</blockquote>
                </div>
              )}
            </div>
          </div>
          <div className="kb-reader-evidence-repair-actions">
            <Button onClick={goBack}>{S.reader_evidence_review_cancel}</Button>
            <Button
              danger
              disabled={!evidenceReviewTargetLocated || Boolean(evidenceReviewDecision)}
              loading={evidenceReviewDecision === 'invalid'}
              data-testid="reader-invalidate-evidence-review"
              onClick={() => { void decideEvidenceReview('invalid') }}
            >
              {S.reader_evidence_review_invalidate}
            </Button>
            <Button
              type="primary"
              disabled={!evidenceReviewTargetLocated || Boolean(evidenceReviewDecision)}
              loading={evidenceReviewDecision === 'confirmed'}
              data-testid="reader-confirm-evidence-review"
              onClick={() => { void decideEvidenceReview('confirmed') }}
            >
              {S.reader_evidence_review_confirm}
            </Button>
          </div>
        </section>
      ) : null}
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
            conversationId={sessionConversationId}
            projectId={sessionProjectId}
            messageId={session?.message_id ?? null}
            sessionHighlights={sessionHighlights}
            onAddSessionHighlight={addSessionHighlight}
            onUpdateSessionHighlight={updateSessionHighlight}
            onRemoveSessionHighlight={removeSessionHighlight}
            onLocateResult={recordLocateResult}
            onAddSelectionToShelf={addSelectionToShelf}
            onAddCitationToShelf={addCitationToShelf}
            onOpenCitationShelf={openCitationShelf}
          />
        )}
      </main>
    </div>
  )
}
