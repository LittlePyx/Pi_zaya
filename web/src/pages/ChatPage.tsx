/* eslint-disable react-hooks/set-state-in-effect */

import { startTransition, useCallback, useDeferredValue, useEffect, useLayoutEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import { Alert, Button, message, Typography } from 'antd'
import { BookOutlined, BugOutlined, ClockCircleOutlined, LoadingOutlined, MenuFoldOutlined, MenuUnfoldOutlined, ReadOutlined } from '@ant-design/icons'
import { useChatStore } from '../stores/chatStore'
import { useSettingsStore } from '../stores/settingsStore'
import { MessageList, type ShelfActivityState } from '../components/chat/MessageList'
import { ChatInput } from '../components/chat/ChatInput'
import { PaperGuideReaderDrawer } from '../components/chat/PaperGuideReaderDrawer'
import type { CiteDetail } from '../components/chat/citationState'
import { sameHighlightTarget } from '../components/chat/reader/readerDomUtils'
import {
  READER_CITATION_SHELF_EVENT,
  READER_SELECTION_SHELF_EVENT,
  READER_SESSION_SYNC_CHANNEL,
  READER_STANDALONE_WINDOW_NAME,
  type ReaderLocateResult,
  type ReaderOpenPayload,
  type ReaderSelectionShelfPayload,
  type ReaderSessionHighlight,
} from '../components/chat/reader/readerTypes'
import {
  normalizeReaderSourcePathForMatch,
  normalizeReaderLocateRequestId,
  readerSourcePathsMatch,
  readerLocateRepairRunMatchesActiveRequest,
  readerLocateResultMatchesActiveRequest,
  type ReaderLocateRequestGuard,
} from '../components/chat/reader/readerLocateGuard'
import {
  sanitizeReaderLocateCandidates,
  sanitizeReaderLocateTarget,
} from '../components/chat/reader/readerOpenPayloadUtils'
import { readerHighlightsSignature } from '../components/chat/reader/readerSessionState'
import { buildResearchContext } from '../components/chat/researchContext'
import {
  normalizeSelectedResearchContextPack,
  type SelectedResearchContextPack,
} from '../components/chat/researchContextPack'
import { dispatchOpenSettings, type ApiSettingsTarget } from '../components/layout/settingsEvents'
import { chatApi, type ChatUploadItem, type Message, type QueryScope } from '../api/chat'
import { libraryApi } from '../api/library'
import { useT } from '../i18n'
import { internalDebugBrowserEnabled } from '../utils/internalDebug'
import { qualityDiagnosticsVisible } from '../utils/qualityDiagnostics'
import { basenameFromSourcePath } from '../utils/sourcePath'
import { reportUserIssue } from '../userIssueReporter'

const { Text } = Typography

const HISTORY_PAGE_SIZE = 24
const LIVE_WINDOW = 16
const READY_DISMISS_MS = 2600
const DUPLICATE_DISMISS_MS = 3600
const DESKTOP_READER_BREAKPOINT = 1280
const DESKTOP_DOCK_DEFAULT_WIDTH = 392
const DESKTOP_DOCK_MIN_WIDTH = 320
const DESKTOP_DOCK_MAX_WIDTH = 760
const DESKTOP_DOCK_COLLAPSED_WIDTH = 48
const DESKTOP_DOCK_WIDTH_TRANSITION = 'width 160ms cubic-bezier(0.2, 0, 0, 1)'
const RIGHT_DOCK_WIDTH_STORAGE_KEY = 'kb:chat-side-dock-width'
const RIGHT_DOCK_COLLAPSED_STORAGE_KEY = 'kb:chat-side-dock-collapsed'
const AGENT_MODE_STORAGE_KEY = 'kb:chat-research-agent-mode'
const SELECTED_RESEARCH_CONTEXT_STORAGE_PREFIX = 'kb:chat:selected-research-context:v1'
const SELECTED_RESEARCH_CONTEXT_STATE_KEY = 'selected_research_context'
const SELECTED_RESEARCH_CONTEXT_SCOPE_STATE_KEY = 'selected_research_context_scope'
const SELECTED_RESEARCH_CONTEXT_PROJECT_STATE_KEY = 'selected_research_context_project_id'
const SELECTED_RESEARCH_CONTEXT_CLEARED_AT_STATE_KEY = 'selected_research_context_cleared_at'
const READER_LOCATE_AUTO_REPAIR_RETRY_MS = 60_000
const showLegacyUiBlocks = false

function resolveQueryScope(scope: QueryScope, opts: { hasCurrentPaper: boolean; hasBasket: boolean }): QueryScope {
  if (scope === 'current_paper' && !opts.hasCurrentPaper) return 'library'
  if (scope === 'basket' && !opts.hasBasket) return opts.hasCurrentPaper ? 'current_paper' : 'library'
  return scope
}

function uploadItemKey(item: ChatUploadItem) {
  if (item.kind === 'pdf' && item.ingest_job_id) {
    return `pdf-job:${item.ingest_job_id}`
  }
  return [item.kind, item.sha1 || '', item.path || '', item.name].join(':')
}

function compactTimelineText(content: string, maxLen = 68, txt?: Record<string, string>) {
  const raw = String(content || '').replace(/\s+/g, ' ').trim()
  if (!raw) return txt?.timeline_blank_question || '空白提问'
  const imgOnly = raw.match(/^\[Image attachment x(\d+)\]$/i)
  if (imgOnly) {
    return (txt?.timeline_image_question || '图片提问 x{n}').replace('{n}', imgOnly[1] || '1')
  }
  if (raw.length <= maxLen) return raw
  return `${raw.slice(0, Math.max(8, maxLen - 1)).trimEnd()}...`
}

function stripSourceExt(name: string) {
  return String(name || '')
    .replace(/\.en\.md$/i, '')
    .replace(/\.md$/i, '')
    .replace(/\.pdf$/i, '')
    .trim()
}

function clampRightDockWidth(value: number) {
  if (!Number.isFinite(value)) return DESKTOP_DOCK_DEFAULT_WIDTH
  return Math.max(DESKTOP_DOCK_MIN_WIDTH, Math.min(DESKTOP_DOCK_MAX_WIDTH, Math.round(value)))
}

function loadStoredRightDockWidth() {
  if (typeof window === 'undefined') return DESKTOP_DOCK_DEFAULT_WIDTH
  const raw = Number(window.localStorage.getItem(RIGHT_DOCK_WIDTH_STORAGE_KEY) || 0)
  return clampRightDockWidth(raw || DESKTOP_DOCK_DEFAULT_WIDTH)
}

function loadStoredRightDockCollapsed() {
  if (typeof window === 'undefined') return false
  return window.localStorage.getItem(RIGHT_DOCK_COLLAPSED_STORAGE_KEY) === '1'
}

function loadStoredAgentMode() {
  if (typeof window === 'undefined') return false
  try {
    const params = new URLSearchParams(window.location.search)
    const queryValue = String(params.get('agent_mode') || params.get('research_agent') || '').trim().toLowerCase()
    if (['1', 'true', 'yes', 'on'].includes(queryValue)) return true
  } catch { /* ignore */ }
  try {
    return window.localStorage.getItem(AGENT_MODE_STORAGE_KEY) === '1'
  } catch {
    return false
  }
}

function selectedResearchContextStorageKey(conversationId?: string | null, shelfScope?: string | null) {
  const conv = String(conversationId || '').trim()
  if (!conv) return ''
  const scope = String(shelfScope || '__default__').trim() || '__default__'
  return `${SELECTED_RESEARCH_CONTEXT_STORAGE_PREFIX}:${encodeURIComponent(conv)}:${encodeURIComponent(scope)}`
}

function loadStoredSelectedResearchContext(storageKey: string): SelectedResearchContextPack | null {
  if (!storageKey || typeof window === 'undefined') return null
  try {
    const raw = window.localStorage.getItem(storageKey)
    if (!raw) return null
    const pack = normalizeSelectedResearchContextPack(JSON.parse(raw))
    if (!pack) {
      window.localStorage.removeItem(storageKey)
      return null
    }
    return pack
  } catch {
    try {
      window.localStorage.removeItem(storageKey)
    } catch {
      // Best-effort cleanup only.
    }
    return null
  }
}

function saveStoredSelectedResearchContext(storageKey: string, pack: SelectedResearchContextPack | null) {
  if (!storageKey || typeof window === 'undefined') return
  try {
    if (!pack) {
      window.localStorage.removeItem(storageKey)
      return
    }
    window.localStorage.setItem(storageKey, JSON.stringify(pack))
  } catch {
    // Storage can fail in private mode or under quota pressure; the in-memory state still works.
  }
}

function selectedResearchContextFromState(state: Record<string, unknown> | undefined | null): SelectedResearchContextPack | null {
  const raw = state && typeof state === 'object' ? state[SELECTED_RESEARCH_CONTEXT_STATE_KEY] : null
  return normalizeSelectedResearchContextPack(raw)
}

function researchContextStateMatchesShelf(
  state: Record<string, unknown> | undefined | null,
  shelfScope?: string | null,
  shelfProjectId?: string | null,
) {
  if (!state || typeof state !== 'object') return true
  const storedScope = String(state[SELECTED_RESEARCH_CONTEXT_SCOPE_STATE_KEY] || '').trim()
  const currentScope = String(shelfScope || '').trim()
  if (storedScope && currentScope && storedScope !== currentScope) return false
  const storedProjectId = String(state[SELECTED_RESEARCH_CONTEXT_PROJECT_STATE_KEY] || '').trim()
  const currentProjectId = String(shelfProjectId || '').trim()
  if (storedProjectId && storedProjectId !== currentProjectId) return false
  return true
}

function isModelConnectionError(err: unknown) {
  const text = err instanceof Error ? err.message : String(err || '')
  return /api key|authentication|unauthorized|forbidden|401|403|connection|network|timeout|timed out|base_url|model/i.test(text)
}

function chatSendFailureKind(messageText: string, labels: Record<string, string>) {
  const text = String(messageText || '').trim()
  const low = text.toLowerCase()
  if (text === labels.chat_generation_start_failed) return 'generation_start_failed'
  if (text === labels.chat_generation_stream_failed) return 'generation_stream_failed'
  if (text === labels.chat_generation_stream_incomplete) return 'generation_stream_incomplete'
  if (text === labels.chat_generation_refresh_failed) return 'generation_refresh_failed'
  if (/generation_start_failed|未能启动/.test(text)) return 'generation_start_failed'
  if (/interrupted before completion|尚未完成|ended before completion/.test(low) || /尚未完成|中断/.test(text)) return 'generation_stream_incomplete'
  if (/stream failed|stream temporarily unavailable|readable body|回答连接/.test(low) || /回答连接/.test(text)) return 'generation_stream_failed'
  if (/latest message|messages page|messages fallback|最新消息/.test(low) || /最新消息/.test(text)) return 'generation_refresh_failed'
  if (/401|403|api key|authentication|unauthorized|forbidden|connection|network|timeout|base_url|model/i.test(text)) return 'model_connection'
  return 'chat_send_failed'
}

function httpStatusFromError(messageText: string) {
  const match = String(messageText || '').trim().match(/^(\d{3})\b/)
  return match ? Number(match[1]) : 0
}

function readerHighlightScopeKey(convId: string | null | undefined, sourcePath: string) {
  const path = normalizeReaderSourcePathForMatch(sourcePath)
  if (!path) return ''
  const conv = String(convId || '__detached__').trim().toLowerCase()
  return `${conv}::${path}`
}

function sameReaderSessionHighlight(
  left: Pick<ReaderSessionHighlight, 'text' | 'startOffset' | 'endOffset' | 'blockId' | 'anchorId' | 'occurrence' | 'readableIndex' | 'documentOccurrence' | 'startReadableIndex' | 'endReadableIndex'>,
  right: Pick<ReaderSessionHighlight, 'text' | 'startOffset' | 'endOffset' | 'blockId' | 'anchorId' | 'occurrence' | 'readableIndex' | 'documentOccurrence' | 'startReadableIndex' | 'endReadableIndex'>,
) {
  return sameHighlightTarget(left, right)
}

function normalizeReaderSessionHighlights(value: unknown): ReaderSessionHighlight[] {
  if (!Array.isArray(value)) return []
  return value
    .filter((item): item is ReaderSessionHighlight => Boolean(item) && typeof item === 'object')
    .filter((item) => Boolean(String(item.id || '').trim() && String(item.text || '').trim()))
}

interface TimelineItem {
  order: number
  userMsgId: number
  targetMsgId: number
  questionPreview: string
  hasAnswer: boolean
}

type RightDockPanel = 'timeline' | 'shelf' | 'reader'

interface RefsActivitySummary {
  packCount: number
  pendingPackCount: number
  hitCount: number
}

interface ChatPerfSnapshot {
  switchTotal: number
  switchAvgMs: number
  refsTotal: number
  refsAvgMs: number
  openPhases: number
  messagePrep: number
}

interface ChatDebugApi {
  getLogs?: () => unknown[]
  summary?: () => Record<string, unknown>
}

interface ChatDebugWindow extends Window {
  __kbSwitchPerf?: ChatDebugApi
  __kbRefsPerf?: ChatDebugApi
  __kbConversationOpenPerf?: ChatDebugApi
  __kbMessageListPerf?: ChatDebugApi
}

function summarizeRefsActivity(refs: Record<string, unknown>): RefsActivitySummary {
  const summary: RefsActivitySummary = { packCount: 0, pendingPackCount: 0, hitCount: 0 }
  for (const value of Object.values(refs || {})) {
    if (!value || typeof value !== 'object') continue
    const rec = value as { hits?: unknown[]; enrichment_pending?: boolean; payload_mode?: string; display_state?: string }
    const hits = Array.isArray(rec.hits) ? rec.hits : []
    const mode = String(rec.payload_mode || '').trim().toLowerCase()
    const displayState = String(rec.display_state || '').trim().toLowerCase()
    summary.packCount += 1
    summary.hitCount += hits.length
    if (mode === 'pending' || displayState === 'pending' || Boolean(rec.enrichment_pending)) {
      summary.pendingPackCount += 1
    }
  }
  return summary
}

function loadChatDebugPanelEnabled() {
  return internalDebugBrowserEnabled()
}

function safeNumber(value: unknown) {
  const num = Number(value)
  return Number.isFinite(num) ? num : 0
}

function safeLogCount(api?: ChatDebugApi) {
  try {
    const logs = api?.getLogs?.()
    return Array.isArray(logs) ? logs.length : 0
  } catch {
    return 0
  }
}

function collectChatPerfSnapshot(): ChatPerfSnapshot {
  if (typeof window === 'undefined') {
    return { switchTotal: 0, switchAvgMs: 0, refsTotal: 0, refsAvgMs: 0, openPhases: 0, messagePrep: 0 }
  }
  const w = window as ChatDebugWindow
  const switchSummary = w.__kbSwitchPerf?.summary?.() || {}
  const refsSummary = w.__kbRefsPerf?.summary?.() || {}
  return {
    switchTotal: safeNumber(switchSummary.total),
    switchAvgMs: safeNumber(switchSummary.avgSuccessMs),
    refsTotal: safeNumber(refsSummary.total),
    refsAvgMs: safeNumber(refsSummary.avgFetchMs),
    openPhases: safeLogCount(w.__kbConversationOpenPerf),
    messagePrep: safeLogCount(w.__kbMessageListPerf),
  }
}

export default function ChatPage() {
  const S = useT()
  const messages = useChatStore((s) => s.messages)
  const conversationLoading = useChatStore((s) => s.conversationLoading)
  const messagesLoadingMore = useChatStore((s) => s.messagesLoadingMore)
  const messagesHasMoreBefore = useChatStore((s) => s.messagesHasMoreBefore)
  const loadOlderMessages = useChatStore((s) => s.loadOlderMessages)
  const generation = useChatStore((s) => s.generation)
  const refs = useChatStore((s) => s.refs)
  const activeConvId = useChatStore((s) => s.activeConvId)
  const activeProjectId = useChatStore((s) => s.activeProjectId)
  const activeConversation = useChatStore((s) => s.activeConversation)
  const guideBindings = useChatStore((s) => s.guideBindings)
  const uploadItems = useChatStore((s) => s.uploadItems)
  const pendingImages = useChatStore((s) => s.pendingImages)
  const uploading = useChatStore((s) => s.uploading)
  const uploadFiles = useChatStore((s) => s.uploadFiles)
  const retryUploadItem = useChatStore((s) => s.retryUploadItem)
  const cancelUploadItem = useChatStore((s) => s.cancelUploadItem)
  const removePendingImage = useChatStore((s) => s.removePendingImage)
  const dismissUploadItem = useChatStore((s) => s.dismissUploadItem)
  const sendMessage = useChatStore((s) => s.sendMessage)
  const createPaperGuideConversation = useChatStore((s) => s.createPaperGuideConversation)
  const cancelGen = useChatStore((s) => s.cancelGeneration)
  const settings = useSettingsStore()
  const [timelineOpen, setTimelineOpen] = useState(true)
  const [timelineJump, setTimelineJump] = useState<{ messageId: number; token: number } | null>(null)
  const [activeTimelineUserMsgId, setActiveTimelineUserMsgId] = useState<number | null>(null)
  const [readerOpen, setReaderOpen] = useState(false)
  const [readerPayload, setReaderPayload] = useState<ReaderOpenPayload | null>(null)
  const [rightDockCollapsed, setRightDockCollapsed] = useState(loadStoredRightDockCollapsed)
  const [rightDockWidth, setRightDockWidth] = useState(loadStoredRightDockWidth)
  const [readerSessionHighlights, setReaderSessionHighlights] = useState<Record<string, ReaderSessionHighlight[]>>({})
  const [readerLocateResults, setReaderLocateResults] = useState<Record<string, ReaderLocateResult>>({})
  const [sourceQualityRefreshToken, setSourceQualityRefreshToken] = useState(0)
  const [citationShelfOpen, setCitationShelfOpen] = useState(false)
  const [citationShelfCount, setCitationShelfCount] = useState(0)
  const [selectedResearchContext, setSelectedResearchContext] = useState<SelectedResearchContextPack | null>(null)
  const [queryScope, setQueryScope] = useState<QueryScope>('library')
  const [agentMode, setAgentMode] = useState(loadStoredAgentMode)
  const [selectedResearchContextLoadedKey, setSelectedResearchContextLoadedKey] = useState('')
  const [selectedResearchContextOwnerKey, setSelectedResearchContextOwnerKey] = useState('')
  const [shelfActivity, setShelfActivity] = useState<ShelfActivityState>({ summary: false, repair: false, autoRepair: false, background: false, count: 0 })
  const [debugPanelEnabled] = useState(loadChatDebugPanelEnabled)
  const [qualityDiagnosticsEnabled] = useState(qualityDiagnosticsVisible)
  const [debugSnapshot, setDebugSnapshot] = useState<ChatPerfSnapshot>(() => collectChatPerfSnapshot())
  const [openShelfSignal, setOpenShelfSignal] = useState(0)
  const [rightDockPanel, setRightDockPanel] = useState<RightDockPanel>('timeline')
  const [shelfDockTarget, setShelfDockTarget] = useState<HTMLDivElement | null>(null)
  const [desktopReaderEligible, setDesktopReaderEligible] = useState(
    () => (typeof window !== 'undefined' ? window.innerWidth >= DESKTOP_READER_BREAKPOINT : false),
  )
  const [rightDockResizing, setRightDockResizing] = useState(false)
  const [appendSignal, setAppendSignal] = useState<{ token: number; text: string } | null>(null)
  const uploadNoticeRef = useRef<Record<string, string>>({})
  const dismissTimerRef = useRef<Record<string, number>>({})
  const timelineJumpTokenRef = useRef(1)
  const readerLocateRequestRef = useRef(1)
  const readerLocateQualitySubmittedRef = useRef<Set<string>>(new Set())
  const readerLocateSourceRepairAtRef = useRef<Record<string, number>>({})
  const readerPayloadRef = useRef<ReaderOpenPayload | null>(null)
  const readerOpenRef = useRef(readerOpen)
  const readerPayloadByFeedbackKeyRef = useRef<Record<string, ReaderOpenPayload>>({})
  const readerLocateGuardByFeedbackKeyRef = useRef<Record<string, ReaderLocateRequestGuard>>({})
  const activeConvIdRef = useRef(String(activeConvId || '').trim())
  const activeReaderSessionHighlightsRef = useRef<ReaderSessionHighlight[]>([])
  const readerStateHydratedKeysRef = useRef<Set<string>>(new Set())
  const readerStateSaveTimersRef = useRef<Record<string, number>>({})
  const readerLocateSourceRepairStreamRef = useRef<AbortController | null>(null)
  const readerLocateSourceRepairRunTokenRef = useRef(0)
  const splitLayoutRef = useRef<HTMLDivElement | null>(null)
  const rightDockResizeGuideRef = useRef<HTMLDivElement | null>(null)
  const rightDockResizeRef = useRef<{ startX: number; startWidth: number } | null>(null)
  const rightDockActivePointerIdRef = useRef<number | null>(null)
  const rightDockResizeFocusRestoreRef = useRef<HTMLElement | null>(null)
  const selectedResearchContextLoadSeqRef = useRef(0)
  const rightDockWidthLiveRef = useRef(rightDockWidth)
  const rightDockResizePreviewWidthRef = useRef(rightDockWidth)
  const timelineScrollRestoreTopRef = useRef<number | null>(null)
  const activeGuideBinding = useMemo(() => {
    const convId = String(activeConvId || '').trim()
    return convId ? guideBindings?.[convId] : undefined
  }, [activeConvId, guideBindings])
  const researchContext = useMemo(() => buildResearchContext({
    activeConvId,
    activeProjectId,
    activeConversation,
    guideBinding: activeGuideBinding,
    readerOpen,
    readerPayload,
    settingsLoaded: settings.loaded,
    hasTextApiKey: settings.hasTextApiKey,
    hasVisionApiKey: settings.hasVisionApiKey,
    visionUsesTextFallback: settings.visionUsesTextFallback,
    readiness: settings.llmReadiness,
    pendingImageCount: pendingImages.length,
  }), [
    activeConvId,
    activeConversation,
    activeGuideBinding,
    activeProjectId,
    pendingImages.length,
    readerOpen,
    readerPayload,
    settings.hasTextApiKey,
    settings.hasVisionApiKey,
    settings.loaded,
    settings.llmReadiness,
    settings.visionUsesTextFallback,
  ])
  const shelfProjectId = researchContext.shelfProjectId || null
  const shelfProjectScope = researchContext.shelfScope
  const selectedResearchContextDraftKey = useMemo(
    () => selectedResearchContextStorageKey(activeConvId, shelfProjectScope),
    [activeConvId, shelfProjectScope],
  )
  const currentSelectedResearchContext = selectedResearchContextOwnerKey === selectedResearchContextDraftKey
    ? selectedResearchContext
    : null
  const previousShelfProjectScopeRef = useRef(shelfProjectScope)
  const selectedResearchContextKeys = useMemo(() => {
    const out: Record<string, boolean> = {}
    for (const item of currentSelectedResearchContext?.items || []) {
      if (item.key) out[item.key] = true
    }
    return out
  }, [currentSelectedResearchContext])
  const handleResearchContextPackChange = useCallback((pack: SelectedResearchContextPack | null) => {
    setSelectedResearchContextOwnerKey(selectedResearchContextDraftKey)
    setSelectedResearchContext(pack)
    if (pack?.items?.length) setQueryScope('basket')
  }, [selectedResearchContextDraftKey])
  const openApiSettings = useCallback((target: ApiSettingsTarget | '' = '') => {
    dispatchOpenSettings(target)
  }, [])

  useEffect(() => {
    const draftKey = selectedResearchContextDraftKey
    const convId = String(activeConvId || '').trim()
    const loadSeq = selectedResearchContextLoadSeqRef.current + 1
    selectedResearchContextLoadSeqRef.current = loadSeq
    const localPack = loadStoredSelectedResearchContext(draftKey)
    setSelectedResearchContextLoadedKey('')
    setSelectedResearchContextOwnerKey(draftKey)
    setSelectedResearchContext(localPack)
    if (localPack?.items?.length) setQueryScope('basket')
    if (!draftKey || !convId) {
      setSelectedResearchContextLoadedKey(draftKey)
      return undefined
    }
    let cancelled = false
    void chatApi.getConversationResearchState(convId).then((record) => {
      if (cancelled || selectedResearchContextLoadSeqRef.current !== loadSeq) return
      const state = record?.state && typeof record.state === 'object' ? record.state : {}
      const backendMatchesShelf = researchContextStateMatchesShelf(state, shelfProjectScope, shelfProjectId)
      const backendPack = backendMatchesShelf ? selectedResearchContextFromState(state) : null
      const backendTouched = Boolean(
        Object.prototype.hasOwnProperty.call(state, SELECTED_RESEARCH_CONTEXT_STATE_KEY)
        || Object.prototype.hasOwnProperty.call(state, SELECTED_RESEARCH_CONTEXT_SCOPE_STATE_KEY)
        || Object.prototype.hasOwnProperty.call(state, SELECTED_RESEARCH_CONTEXT_PROJECT_STATE_KEY)
        || Object.prototype.hasOwnProperty.call(state, SELECTED_RESEARCH_CONTEXT_CLEARED_AT_STATE_KEY)
      )
      const nextPack = backendTouched ? backendPack : localPack
      setSelectedResearchContextOwnerKey(draftKey)
      setSelectedResearchContext(nextPack)
      if (nextPack?.items?.length) setQueryScope('basket')
      if (backendTouched) {
        saveStoredSelectedResearchContext(draftKey, backendPack)
      }
      setSelectedResearchContextLoadedKey(draftKey)
    }).catch(() => {
      if (cancelled || selectedResearchContextLoadSeqRef.current !== loadSeq) return
      setSelectedResearchContextLoadedKey(draftKey)
    })
    return () => {
      cancelled = true
    }
  }, [activeConvId, selectedResearchContextDraftKey, shelfProjectId, shelfProjectScope])

  useEffect(() => {
    if (selectedResearchContextLoadedKey !== selectedResearchContextDraftKey) return
    const packForCurrentScope = selectedResearchContextOwnerKey === selectedResearchContextDraftKey
      ? selectedResearchContext
      : null
    saveStoredSelectedResearchContext(selectedResearchContextDraftKey, packForCurrentScope)
    const convId = String(activeConvId || '').trim()
    if (!convId) return undefined
    const timer = window.setTimeout(() => {
      void chatApi.patchConversationResearchState(convId, {
        [SELECTED_RESEARCH_CONTEXT_STATE_KEY]: packForCurrentScope || null,
        [SELECTED_RESEARCH_CONTEXT_SCOPE_STATE_KEY]: packForCurrentScope ? shelfProjectScope : null,
        [SELECTED_RESEARCH_CONTEXT_PROJECT_STATE_KEY]: packForCurrentScope ? shelfProjectId : null,
        [SELECTED_RESEARCH_CONTEXT_CLEARED_AT_STATE_KEY]: packForCurrentScope ? null : Date.now(),
      }).catch(() => {
        // The local draft remains the fallback if the backend is temporarily unavailable.
      })
    }, 160)
    return () => window.clearTimeout(timer)
  }, [
    activeConvId,
    selectedResearchContext,
    selectedResearchContextDraftKey,
    selectedResearchContextLoadedKey,
    selectedResearchContextOwnerKey,
    shelfProjectId,
    shelfProjectScope,
  ])

  useEffect(() => {
    if (!debugPanelEnabled || typeof window === 'undefined') return undefined
    const update = () => setDebugSnapshot(collectChatPerfSnapshot())
    update()
    const timer = window.setInterval(update, 1000)
    return () => window.clearInterval(timer)
  }, [debugPanelEnabled])

  const nextEventToken = useCallback(() => {
    timelineJumpTokenRef.current += 1
    return timelineJumpTokenRef.current
  }, [])

  const handleResearchContextFollowUp = useCallback((pack: SelectedResearchContextPack, promptText: string) => {
    handleResearchContextPackChange(pack)
    setAppendSignal({
      token: nextEventToken(),
      text: promptText,
    })
  }, [handleResearchContextPackChange, nextEventToken])

  const nextReaderLocateRequestId = useCallback(() => {
    readerLocateRequestRef.current += 1
    return readerLocateRequestRef.current
  }, [])

  const clearTimelineSelection = () => {
    setTimelineJump(null)
    setActiveTimelineUserMsgId(null)
  }

  const captureTimelineScrollTop = () => {
    const scrollHost = splitLayoutRef.current?.querySelector<HTMLElement>('.kb-main-scroll')
    timelineScrollRestoreTopRef.current = scrollHost ? scrollHost.scrollTop : null
  }

  const toggleTimelineOpen = () => {
    captureTimelineScrollTop()
    clearTimelineSelection()
    setTimelineOpen((value) => !value)
  }

  useEffect(() => {
    const projectChanged = previousShelfProjectScopeRef.current !== shelfProjectScope
    previousShelfProjectScopeRef.current = shelfProjectScope
    setTimelineOpen(true)
    clearTimelineSelection()
    setReaderOpen(false)
    setReaderPayload(null)
    readerPayloadRef.current = null
    readerPayloadByFeedbackKeyRef.current = {}
    readerLocateGuardByFeedbackKeyRef.current = {}
    readerLocateSourceRepairRunTokenRef.current += 1
    readerLocateSourceRepairStreamRef.current?.abort()
    readerLocateSourceRepairStreamRef.current = null
    if (projectChanged) {
      setCitationShelfOpen(false)
      setCitationShelfCount(0)
      setRightDockPanel('timeline')
    } else {
      setRightDockPanel((current) => (current === 'reader' ? 'timeline' : current))
    }
    setAppendSignal(null)
  }, [activeConvId, shelfProjectScope])

  useEffect(() => {
    const hasCurrentPaper = Boolean(researchContext.activeSource.ready)
    const hasBasket = Boolean(currentSelectedResearchContext?.items?.length)
    setQueryScope((current) => {
      if (current === 'library') return current
      return resolveQueryScope(current, { hasCurrentPaper, hasBasket })
    })
  }, [researchContext.activeSource.ready, currentSelectedResearchContext])

  useEffect(() => () => {
    Object.values(dismissTimerRef.current).forEach((timer) => window.clearTimeout(timer))
    dismissTimerRef.current = {}
    readerLocateSourceRepairRunTokenRef.current += 1
    readerLocateSourceRepairStreamRef.current?.abort()
    readerLocateSourceRepairStreamRef.current = null
  }, [])

  useEffect(() => {
    readerOpenRef.current = readerOpen
  }, [readerOpen])

  useEffect(() => {
    activeConvIdRef.current = String(activeConvId || '').trim()
  }, [activeConvId])

  useEffect(() => {
    readerPayloadRef.current = readerPayload
  }, [readerPayload])

  useEffect(() => {
    const syncLayout = () => {
      setDesktopReaderEligible(window.innerWidth >= DESKTOP_READER_BREAKPOINT)
    }
    syncLayout()
    window.addEventListener('resize', syncLayout)
    return () => {
      window.removeEventListener('resize', syncLayout)
    }
  }, [])

  useEffect(() => {
    window.localStorage.setItem(RIGHT_DOCK_WIDTH_STORAGE_KEY, String(clampRightDockWidth(rightDockWidth)))
  }, [rightDockWidth])

  useEffect(() => {
    window.localStorage.setItem(RIGHT_DOCK_COLLAPSED_STORAGE_KEY, rightDockCollapsed ? '1' : '0')
  }, [rightDockCollapsed])

  useEffect(() => {
    try {
      window.localStorage.setItem(AGENT_MODE_STORAGE_KEY, agentMode ? '1' : '0')
    } catch { /* ignore */ }
  }, [agentMode])

  useEffect(() => {
    rightDockWidthLiveRef.current = rightDockWidth
    if (!rightDockResizing) {
      rightDockResizePreviewWidthRef.current = rightDockWidth
    }
  }, [rightDockResizing, rightDockWidth])

  useLayoutEffect(() => {
    const targetTop = timelineScrollRestoreTopRef.current
    if (targetTop == null) return
    timelineScrollRestoreTopRef.current = null
    const scrollHost = splitLayoutRef.current?.querySelector<HTMLElement>('.kb-main-scroll')
    if (!scrollHost) return
    let cancelled = false
    const restore = () => {
      if (cancelled) return
      scrollHost.scrollTop = targetTop
    }
    const frameA = window.requestAnimationFrame(restore)
    const frameB = window.requestAnimationFrame(() => {
      window.requestAnimationFrame(restore)
    })
    return () => {
      cancelled = true
      window.cancelAnimationFrame(frameA)
      window.cancelAnimationFrame(frameB)
    }
  }, [timelineOpen, desktopReaderEligible, readerOpen, rightDockCollapsed])

  const restoreRightDockResizeFocus = () => {
    const target = rightDockResizeFocusRestoreRef.current
    rightDockResizeFocusRestoreRef.current = null
    if (!target || !target.isConnected) return
    try {
      target.focus({ preventScroll: true })
    } catch {
      target.focus()
    }
  }

  const clearRightDockResizeSession = () => {
    rightDockResizeRef.current = null
    rightDockActivePointerIdRef.current = null
    rightDockResizePreviewWidthRef.current = rightDockWidthLiveRef.current
    document.body.classList.remove('kb-right-dock-resizing')
    document.body.style.removeProperty('cursor')
    document.body.style.removeProperty('user-select')
    const guide = rightDockResizeGuideRef.current
    if (guide) {
      guide.style.removeProperty('left')
    }
  }

  useEffect(() => () => {
    clearRightDockResizeSession()
  }, [])

  const updateRightDockResizeGuide = (nextWidth: number) => {
    const guide = rightDockResizeGuideRef.current
    const layout = splitLayoutRef.current
    const clampedWidth = clampRightDockWidth(nextWidth)
    rightDockResizePreviewWidthRef.current = clampedWidth
    if (!guide || !layout || rightDockCollapsed) return
    const nextLeft = Math.max(0, layout.clientWidth - clampedWidth)
    guide.style.left = `${Math.round(nextLeft)}px`
  }

  const finishRightDockResize = (commit: boolean) => {
    const finalWidth = clampRightDockWidth(
      commit ? rightDockResizePreviewWidthRef.current || rightDockWidthLiveRef.current : rightDockWidthLiveRef.current,
    )
    clearRightDockResizeSession()
    setRightDockResizing(false)
    if (commit && !rightDockCollapsed) {
      setRightDockWidth(finalWidth)
    }
    window.requestAnimationFrame(restoreRightDockResizeFocus)
  }

  useEffect(() => {
    const liveKeys = new Set<string>()
    for (const item of uploadItems) {
      if (item.kind !== 'pdf') continue
      const key = uploadItemKey(item)
      liveKeys.add(key)
      const terminalState =
        item.status === 'duplicate'
          ? 'duplicate'
          : item.ingest_status === 'cancelled'
            ? 'cancelled'
            : (item.status === 'error' || item.ingest_status === 'error')
            ? 'error'
            : (item.ready || item.ingest_status === 'ready')
              ? 'ready'
              : ''
      if (!terminalState || uploadNoticeRef.current[key] === terminalState) {
        continue
      }
      uploadNoticeRef.current[key] = terminalState
      if (terminalState === 'ready') {
        message.success(`${S.upload_pdf_ready}: ${item.name}`)
        if (dismissTimerRef.current[key] == null) {
          dismissTimerRef.current[key] = window.setTimeout(() => {
            dismissUploadItem(key)
            delete dismissTimerRef.current[key]
          }, READY_DISMISS_MS)
        }
      } else if (terminalState === 'duplicate') {
        message.info(`${S.upload_pdf_duplicate}: ${item.name}`)
        if (dismissTimerRef.current[key] == null) {
          dismissTimerRef.current[key] = window.setTimeout(() => {
            dismissUploadItem(key)
            delete dismissTimerRef.current[key]
          }, DUPLICATE_DISMISS_MS)
        }
      } else if (terminalState === 'cancelled') {
        message.info(`${S.upload_pdf_cancelled}: ${item.name}`)
      } else if (terminalState === 'error') {
        message.error(`${S.upload_pdf_error}: ${item.name}`)
      }
    }

    for (const key of Object.keys(uploadNoticeRef.current)) {
      if (liveKeys.has(key)) continue
      delete uploadNoticeRef.current[key]
      const timer = dismissTimerRef.current[key]
      if (timer != null) {
        window.clearTimeout(timer)
        delete dismissTimerRef.current[key]
      }
    }
  }, [dismissUploadItem, S.upload_pdf_cancelled, S.upload_pdf_duplicate, S.upload_pdf_error, S.upload_pdf_ready, uploadItems])

  const onSend = (text: string) => {
    if (researchContext.api.sendBlockTarget === 'text') {
      message.warning(S.chat_api_missing_toast)
      openApiSettings('text')
      return
    }
    if (researchContext.api.sendBlockTarget === 'vision') {
      message.warning(S.chat_vision_api_missing_toast)
      openApiSettings('vision')
      return
    }
    const hasCurrentPaper = Boolean(researchContext.activeSource.ready)
    const hasBasket = Boolean(currentSelectedResearchContext?.items?.length)
    const resolvedScope = resolveQueryScope(queryScope, { hasCurrentPaper, hasBasket })
    const contextPackForSend = resolvedScope === 'basket' ? currentSelectedResearchContext : null
    void sendMessage(text, {
      topK: settings.topK,
      temperature: settings.temperature,
      maxTokens: settings.maxTokens,
      deepRead: true,
      promptContext: contextPackForSend,
      queryScope: resolvedScope,
      agentMode,
    }).then(() => {
      if (!contextPackForSend) return
      setSelectedResearchContext((current) => (
        current?.id === contextPackForSend.id ? null : current
      ))
    }).catch((err: unknown) => {
      const fallback = err instanceof Error ? err.message : String(err || '')
      const failureKind = chatSendFailureKind(fallback, S)
      reportUserIssue({
        source: 'frontend',
        domain: 'chat_generation',
        severity: 'error',
        summary: `Chat send failed: ${failureKind}`,
        detail: fallback || S.settings_test_unknown_error,
        route: '/',
        context: {
          ui_locale: settings.uiLocale,
          query_scope: resolvedScope,
          active_conversation: Boolean(activeConvId),
          active_project: Boolean(activeProjectId),
          paper_guide_mode: Boolean(
            activeConversation?.mode === 'paper_guide'
            || activeConversation?.bound_source_path
            || guideBindings?.[String(activeConvId || '')]?.sourcePath,
          ),
          message_count: messages.length,
          pending_image_count: pendingImages.length,
          upload_item_count: uploadItems.length,
          ready_upload_count: uploadItems.filter((item) => item.kind === 'pdf' && item.ready).length,
          running_upload_count: uploadItems.filter((item) => item.kind === 'pdf' && !item.ready && item.status !== 'error').length,
          selected_context: Boolean(contextPackForSend),
          selected_context_item_count: Array.isArray(contextPackForSend?.items) ? contextPackForSend.items.length : 0,
          agent_mode: agentMode,
          prompt_length: text.trim().length,
          prompt_empty: text.trim().length === 0,
        },
        payload: {
          error_kind: failureKind,
          http_status: httpStatusFromError(fallback),
        },
        fingerprint: `chat-send:${failureKind}:${resolvedScope}:${settings.uiLocale}`,
      })
      if (isModelConnectionError(err)) {
        message.error(S.chat_api_connection_failed.replace('{error}', fallback || S.settings_test_unknown_error))
        void settings.refreshReadiness().catch(() => {})
        openApiSettings('text')
        return
      }
      message.error(fallback || S.upload_failed_generic)
    })
  }

  const onUpload = async (files: File[]) => {
    try {
      await uploadFiles(files, { quickIngest: true, speedMode: 'balanced' })
    } catch {
      message.error(S.upload_failed_generic)
    }
  }

  const onRetryUpload = async (key: string) => {
    try {
      await retryUploadItem(key)
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.retry_ingest_failed)
    }
  }

  const onCancelUpload = async (key: string) => {
    try {
      await cancelUploadItem(key)
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.cancel_ingest_failed)
    }
  }

  const onStartGuideFromUpload = async (item: ChatUploadItem) => {
    const sourcePath = String(item.md_path || '').trim()
    if (!sourcePath) {
      message.info(S.reader_pdf_not_ready)
      return
    }
    const sourceName = stripSourceExt(item.name) || item.name
    const hide = message.loading(S.reader_creating_guide, 0)
    try {
      await createPaperGuideConversation({
        sourcePath,
        sourceName,
        title: S.default_guide_title.replace('{name}', sourceName),
      })
      hide()
      message.success(S.reader_entered_guide)
    } catch (err) {
      hide()
      message.error(err instanceof Error ? err.message : S.reader_create_guide_failed)
    }
  }

  const liveRunning = Boolean(generation)
  const visibleMessages = liveRunning
    ? messages.slice(-Math.min(messages.length, LIVE_WINDOW))
    : messages
  const deferredTimelineMessages = useDeferredValue(messages)
  const deferredRefs = useDeferredValue(refs)
  const hiddenCount = liveRunning
    ? Math.max(0, messages.length - visibleMessages.length)
    : 0
  const messageIndexById = useMemo(() => {
    const map = new Map<number, number>()
    messages.forEach((msg, idx) => {
      map.set(msg.id, idx)
    })
    return map
  }, [messages])
  const timelineItems = useMemo(() => {
    const out: TimelineItem[] = []
    let pendingUser: Message | null = null
    let order = 0
    for (const msg of deferredTimelineMessages) {
      if (msg.role === 'user') {
        pendingUser = msg
        continue
      }
      if (msg.role !== 'assistant' || !pendingUser) continue
      order += 1
      out.push({
        order,
        userMsgId: pendingUser.id,
        targetMsgId: msg.id,
        questionPreview: compactTimelineText(pendingUser.content, 68, S),
        hasAnswer: true,
      })
      pendingUser = null
    }
    if (pendingUser) {
      order += 1
      out.push({
        order,
        userMsgId: pendingUser.id,
        targetMsgId: pendingUser.id,
        questionPreview: compactTimelineText(pendingUser.content, 68, S),
        hasAnswer: false,
      })
    }
    return out
  }, [S, deferredTimelineMessages])
  const timelineTrackedMessageIds = useMemo(
    () => timelineItems.map((item) => item.targetMsgId),
    [timelineItems],
  )
  const timelineUserMsgIdByTargetMsgId = useMemo(() => {
    const map = new Map<number, number>()
    timelineItems.forEach((item) => {
      map.set(item.targetMsgId, item.userMsgId)
    })
    return map
  }, [timelineItems])
  const effectiveGuide = useMemo(() => {
    const sourcePath = researchContext.guideSource.sourcePath
    const sourceName = researchContext.guideSource.sourceName
    return { sourcePath, sourceName }
  }, [researchContext.guideSource.sourceName, researchContext.guideSource.sourcePath])

  const jumpToTimelineItem = (item: TimelineItem) => {
    if (liveRunning) {
      message.info(S.timeline_jump_blocked)
      return
    }
    const idx = messageIndexById.get(item.targetMsgId)
    if (idx == null) return
    setActiveTimelineUserMsgId(null)
    const token = nextEventToken()
    window.setTimeout(() => {
      setTimelineJump({ messageId: item.targetMsgId, token })
    }, 0)
  }

  const openReader = (payload: ReaderOpenPayload) => {
    const sourcePath = String(payload?.sourcePath || '').trim()
    if (!sourcePath) {
      message.info(S.reader_missing_path)
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
      readerPayloadByFeedbackKeyRef.current[feedbackKey] = nextPayload
      readerLocateGuardByFeedbackKeyRef.current[feedbackKey] = {
        locateRequestId,
        sourcePath,
        conversationId: String(activeConvId || '').trim(),
      }
    }
    readerPayloadRef.current = nextPayload
    setReaderPayload(nextPayload)
    setRightDockPanel('reader')
    setRightDockCollapsed(false)
    setReaderOpen(true)
  }

  const openReaderStandalone = useCallback(async (payloadInput?: ReaderOpenPayload | null) => {
    const payload = payloadInput || readerPayloadRef.current
    const sourcePath = String(payload?.sourcePath || '').trim()
    if (!payload || !sourcePath) {
      message.info(S.reader_missing_path)
      return
    }
    const sourceName = String(payload.sourceName || '').trim()
      || basenameFromSourcePath(sourcePath)
      || S.side_dock_reader
    let popup: Window | null = null
    try {
      popup = window.open('', READER_STANDALONE_WINDOW_NAME)
      if (popup) {
        popup.document.title = sourceName
        popup.document.body.style.margin = '0'
        popup.document.body.style.fontFamily = 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
        popup.document.body.innerHTML = `<div style="display:grid;place-items:center;min-height:100vh;color:#64748b;font-size:14px;">${S.reader_opening_window || 'Opening reader...'}</div>`
        popup.focus()
      }
    } catch {
      popup = null
    }
    try {
      const session = await chatApi.createReaderSession(payload, {
        title: sourceName,
        conversationId: activeConvId,
        state: {
          sourcePath,
          conversationId: activeConvId || '',
          projectId: shelfProjectId || '',
          highlights: activeReaderSessionHighlightsRef.current,
          evidenceNotes: activeReaderSessionHighlightsRef.current,
        },
      })
      const linkedConversationId = String(session.conversation_id || activeConvId || '').trim()
      const readerUrl = new URL(`/reader/session/${encodeURIComponent(session.id)}`, window.location.origin)
      if (linkedConversationId) readerUrl.searchParams.set('conversation', linkedConversationId)
      const url = readerUrl.toString()
      if (popup && !popup.closed) {
        popup.location.href = url
        popup.focus()
      } else {
        const opened = window.open(url, READER_STANDALONE_WINDOW_NAME)
        opened?.focus()
        if (!opened) message.info(S.reader_window_blocked || 'The browser blocked the reader window.')
      }
    } catch (err) {
      if (popup && !popup.closed) {
        popup.close()
      }
      message.error(err instanceof Error ? err.message : (S.reader_open_window_failed || 'Failed to open reader window'))
    }
  }, [
    S.reader_missing_path,
    S.reader_open_window_failed,
    S.reader_opening_window,
    S.reader_window_blocked,
    S.side_dock_reader,
    activeConvId,
    shelfProjectId,
  ])

  const handleCitationShelfOpenChange = useCallback((open: boolean) => {
    setCitationShelfOpen(open)
    if (open && desktopReaderEligible) {
      setRightDockPanel('shelf')
      setRightDockCollapsed(false)
    }
  }, [desktopReaderEligible])

  const handleCitationShelfStateChange = useCallback((state: { open: boolean; count: number }) => {
    setCitationShelfCount(Math.max(0, Math.floor(Number(state.count || 0))))
    setCitationShelfOpen(Boolean(state.open))
    if (state.open && desktopReaderEligible) {
      setRightDockPanel('shelf')
      setRightDockCollapsed(false)
    }
  }, [desktopReaderEligible])

  const handleShelfActivityChange = useCallback((state: ShelfActivityState) => {
    setShelfActivity((current) => (
      current.summary === state.summary
        && current.repair === state.repair
        && current.autoRepair === state.autoRepair
        && current.background === state.background
        && current.count === state.count
        ? current
        : state
    ))
  }, [])

  const activateDockPanel = useCallback((panel: RightDockPanel) => {
    setRightDockPanel(panel)
    setRightDockCollapsed(false)
    if (panel === 'timeline') {
      setTimelineOpen(true)
      return
    }
    if (panel === 'shelf') {
      setOpenShelfSignal((value) => value + 1)
      setCitationShelfOpen(true)
      return
    }
  }, [])

  const refreshShelfSourceQuality = useCallback(() => {
    setSourceQualityRefreshToken((value) => value + 1)
  }, [])

  const retryReaderLocateAfterRepair = useCallback((feedbackKey: string, sourcePath: string) => {
    const key = String(feedbackKey || '').trim()
    const path = String(sourcePath || '').trim()
    if (!key || !readerOpenRef.current) return
    const currentPayload = readerPayloadRef.current
    if (!currentPayload) return
    if (String(currentPayload.locateFeedbackKey || '').trim() !== key) return
    if (path && !readerSourcePathsMatch(currentPayload.sourcePath, path)) return
    const locateRequestId = nextReaderLocateRequestId()
    const nextPayload: ReaderOpenPayload = {
      ...currentPayload,
      locateRequestId,
    }
    readerPayloadByFeedbackKeyRef.current[key] = nextPayload
    readerLocateGuardByFeedbackKeyRef.current[key] = {
      locateRequestId,
      sourcePath: String(nextPayload.sourcePath || '').trim(),
      conversationId: String(activeConvIdRef.current || '').trim(),
    }
    readerPayloadRef.current = nextPayload
    setReaderPayload(nextPayload)
    setRightDockPanel('reader')
    setRightDockCollapsed(false)
    setReaderOpen(true)
  }, [nextReaderLocateRequestId])

  const completeReaderLocateSourceRepair = useCallback(async (
    runId: string,
    options: {
      needsReindex: boolean
      shouldRetryLocate: boolean
      feedbackKey: string
      sourcePath: string
      isCurrentRepair?: () => boolean
    },
  ) => {
    if (options.isCurrentRepair && !options.isCurrentRepair()) return
    let waiting = false
    if (runId && options.needsReindex) {
      try {
        const advanced = await libraryApi.advanceQualityRepairRun(runId)
        waiting = Boolean(advanced.waiting)
      } catch {
        waiting = false
      }
    }
    if (options.isCurrentRepair && !options.isCurrentRepair()) return
    refreshShelfSourceQuality()
    if (!waiting && options.shouldRetryLocate) {
      retryReaderLocateAfterRepair(options.feedbackKey, options.sourcePath)
    }
  }, [refreshShelfSourceQuality, retryReaderLocateAfterRepair])

  const handleReaderLocateResult = useCallback((result: ReaderLocateResult) => {
    const feedbackKey = String(result.locateFeedbackKey || '').trim()
    if (!feedbackKey) return
    const sourcePath = String(result.sourcePath || '').trim()
    const sourceName = String(result.sourceName || '').trim()
    const locateRequestId = normalizeReaderLocateRequestId(result.locateRequestId)
    const guard = readerLocateGuardByFeedbackKeyRef.current[feedbackKey]
    const currentPayload = readerPayloadRef.current
    const currentConversationId = String(activeConvIdRef.current || '').trim()
    if (!readerLocateResultMatchesActiveRequest({
      result: { ...result, locateRequestId },
      guard,
      currentPayload,
      currentConversationId,
      readerOpen: readerOpenRef.current,
    })) {
      return
    }
    const submitKey = [
      feedbackKey,
      locateRequestId,
      result.status,
      result.precision,
      result.hint,
      result.reason,
    ].join('|')
    if (qualityDiagnosticsEnabled && !readerLocateQualitySubmittedRef.current.has(submitKey)) {
      readerLocateQualitySubmittedRef.current.add(submitKey)
      libraryApi.recordReaderLocateQuality({
        source_path: sourcePath,
        source_name: sourceName,
        locate_feedback_key: feedbackKey,
        locate_request_id: locateRequestId,
        status: result.status,
        precision: result.precision,
        ok: result.ok,
        repairable: result.repairable,
        strict_locate: result.strictLocate,
        hint: result.hint,
        reason: result.reason,
        active_alt_index: result.activeAltIndex,
        block_id: result.blockId,
        anchor_id: result.anchorId,
        anchor_kind: result.anchorKind,
        heading_path: result.headingPath,
      }).catch(() => {})
    }
    const locateStatus = String(result.status || '').trim().toLowerCase()
    const needsSourceRepair = Boolean(
      sourcePath
      && (
        result.repairable
        || locateStatus === 'failed'
        || (result.strictLocate && !['exact', 'block'].includes(locateStatus))
      ),
    )
    if (qualityDiagnosticsEnabled && needsSourceRepair) {
      const repairKey = sourcePath || sourceName
      const now = Date.now()
      const last = Number(readerLocateSourceRepairAtRef.current[repairKey] || 0)
      if (repairKey && now - last >= READER_LOCATE_AUTO_REPAIR_RETRY_MS) {
        readerLocateSourceRepairAtRef.current[repairKey] = now
        const repairToken = readerLocateSourceRepairRunTokenRef.current + 1
        readerLocateSourceRepairRunTokenRef.current = repairToken
        const repairResult: ReaderLocateResult = { ...result, locateRequestId }
        const isCurrentSourceRepair = () => (
          readerLocateRepairRunMatchesActiveRequest({
            expectedRunToken: repairToken,
            currentRunToken: readerLocateSourceRepairRunTokenRef.current,
            result: repairResult,
            guard: readerLocateGuardByFeedbackKeyRef.current[feedbackKey],
            currentPayload: readerPayloadRef.current,
            currentConversationId: activeConvIdRef.current,
            readerOpen: readerOpenRef.current,
          })
        )
        libraryApi.repairQuality({
          sources: [{ source_path: sourcePath, source_name: sourceName }],
          speed_mode: 'balanced',
          replace: true,
          md_autofix: true,
        })
          .then((res) => {
            if (!isCurrentSourceRepair()) return undefined
            const runId = String(res.repair_run_id || res.repair_run?.run_id || '').trim()
            const queued = Number(res.enqueued || 0)
            const needsReindex = Boolean(res.needs_reindex || res.impact?.needs_reindex)
            const repaired = Number(res.repaired || res.impact?.repaired || 0)
            const readerLocateReindex = Number(res.impact?.reader_locate_reindex || 0)
            const shouldRetryLocate = Boolean(
              needsReindex
              || repaired > 0
              || readerLocateReindex > 0
              || (res.items || []).some((item) => Boolean(item.reader_locate_reindex_required)),
            )
            if (!runId) {
              if (!isCurrentSourceRepair()) return undefined
              refreshShelfSourceQuality()
              if (shouldRetryLocate) retryReaderLocateAfterRepair(feedbackKey, sourcePath)
              return undefined
            }
            if (queued > 0) {
              readerLocateSourceRepairStreamRef.current?.abort()
              let streamCtrl: AbortController | null = null
              const clearStreamIfCurrent = () => {
                if (!isCurrentSourceRepair() || readerLocateSourceRepairStreamRef.current !== streamCtrl) return false
                readerLocateSourceRepairStreamRef.current = null
                return true
              }
              streamCtrl = libraryApi.streamConvertStatus(
                () => {},
                () => {
                  if (!clearStreamIfCurrent()) return
                  void completeReaderLocateSourceRepair(runId, {
                    needsReindex,
                    shouldRetryLocate,
                    feedbackKey,
                    sourcePath,
                    isCurrentRepair: isCurrentSourceRepair,
                  })
                },
                () => {
                  if (!clearStreamIfCurrent()) return
                  refreshShelfSourceQuality()
                },
              )
              readerLocateSourceRepairStreamRef.current = streamCtrl
              return undefined
            }
            return completeReaderLocateSourceRepair(runId, {
              needsReindex,
              shouldRetryLocate,
              feedbackKey,
              sourcePath,
              isCurrentRepair: isCurrentSourceRepair,
            })
          })
          .catch(() => {
            if (isCurrentSourceRepair()) delete readerLocateSourceRepairAtRef.current[repairKey]
          })
      }
    }
    setReaderLocateResults((current) => {
      const prev = current[feedbackKey]
      if (
        prev
        && prev.locateRequestId === locateRequestId
        && prev.status === result.status
        && prev.precision === result.precision
        && prev.hint === result.hint
      ) {
        return current
      }
      return { ...current, [feedbackKey]: { ...result, locateRequestId } }
    })
  }, [completeReaderLocateSourceRepair, qualityDiagnosticsEnabled, refreshShelfSourceQuality, retryReaderLocateAfterRepair])

  const appendReaderSelection = (text: string) => {
    const raw = String(text || '')
    if (!raw.trim()) return
    setAppendSignal({
      token: nextEventToken(),
      text: raw,
    })
  }

  const addReaderSelectionToShelf = useCallback((payload: ReaderSelectionShelfPayload) => {
    const text = String(payload?.text || '').trim()
    const sourcePath = String(payload?.sourcePath || '').trim()
    if (!text || !sourcePath) return
    const detail: ReaderSelectionShelfPayload = {
      ...payload,
      text,
      sourcePath,
      conversationId: activeConvId || payload.conversationId || '',
      projectId: shelfProjectId || payload.projectId || '',
      createdAt: Number(payload.createdAt || Date.now()),
    }
    window.dispatchEvent(new CustomEvent(READER_SELECTION_SHELF_EVENT, { detail }))
    setOpenShelfSignal((value) => value + 1)
    setCitationShelfOpen(true)
    setRightDockPanel('shelf')
    setRightDockCollapsed(false)
    message.success(S.reader_added_to_shelf || 'Added to citation shelf')
  }, [S.reader_added_to_shelf, activeConvId, shelfProjectId])

  const addReaderCitationToShelf = useCallback((detail: CiteDetail) => {
    if (!detail) return
    const payload = {
      type: 'reader-citation-shelf',
      detail: detail as unknown as Record<string, unknown>,
      conversationId: activeConvId || '',
      projectId: shelfProjectId || '',
      createdAt: Date.now(),
    }
    window.dispatchEvent(new CustomEvent(READER_CITATION_SHELF_EVENT, { detail: payload }))
    setOpenShelfSignal((value) => value + 1)
    setCitationShelfOpen(true)
    setRightDockPanel('shelf')
    setRightDockCollapsed(false)
    message.success(S.reader_added_to_shelf || 'Added to citation shelf')
  }, [S.reader_added_to_shelf, activeConvId, shelfProjectId])

  const openReaderCitationShelf = useCallback(() => {
    setOpenShelfSignal((value) => value + 1)
    setCitationShelfOpen(true)
    setRightDockPanel('shelf')
    setRightDockCollapsed(false)
  }, [])

  const activeReaderSourcePath = useMemo(() => String(readerPayload?.sourcePath || '').trim(), [readerPayload?.sourcePath])
  const activeReaderHighlightScope = useMemo(
    () => readerHighlightScopeKey(activeConvId, activeReaderSourcePath),
    [activeConvId, activeReaderSourcePath],
  )
  const activeReaderSessionHighlights = useMemo(
    () => (activeReaderHighlightScope ? readerSessionHighlights[activeReaderHighlightScope] || [] : []),
    [activeReaderHighlightScope, readerSessionHighlights],
  )
  useEffect(() => {
    activeReaderSessionHighlightsRef.current = activeReaderSessionHighlights
  }, [activeReaderSessionHighlights])

  const persistReaderHighlights = useCallback((convId: string, sourcePath: string, highlights: ReaderSessionHighlight[]) => {
    const cid = String(convId || '').trim()
    const src = String(sourcePath || '').trim()
    if (!cid || !src) return
    void chatApi.updateConversationReaderState(cid, src, {
      highlights,
      evidenceNotes: highlights,
      updatedAt: Date.now(),
    }).catch(() => {})
  }, [])

  useEffect(() => {
    const convId = String(activeConvId || '').trim()
    const sourcePath = String(activeReaderSourcePath || '').trim()
    const scopeKey = activeReaderHighlightScope
    if (!convId || !sourcePath || !scopeKey) return undefined
    let cancelled = false
    readerStateHydratedKeysRef.current.delete(scopeKey)
    chatApi.getConversationReaderState(convId, sourcePath)
      .then((record) => {
        if (cancelled) return
        const highlights = normalizeReaderSessionHighlights(record.state?.highlights || record.state?.evidenceNotes)
        setReaderSessionHighlights((current) => {
          const prev = current[scopeKey] || []
          if (readerHighlightsSignature(prev) === readerHighlightsSignature(highlights)) return current
          if (highlights.length === 0 && prev.length > 0) return current
          return { ...current, [scopeKey]: highlights }
        })
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) readerStateHydratedKeysRef.current.add(scopeKey)
      })
    return () => {
      cancelled = true
    }
  }, [activeConvId, activeReaderHighlightScope, activeReaderSourcePath])

  useEffect(() => {
    const convId = String(activeConvId || '').trim()
    const sourcePath = String(activeReaderSourcePath || '').trim()
    const scopeKey = activeReaderHighlightScope
    if (!convId || !sourcePath || !scopeKey) return undefined
    if (!readerStateHydratedKeysRef.current.has(scopeKey)) return undefined
    const highlights = activeReaderSessionHighlights
    const previousTimer = readerStateSaveTimersRef.current[scopeKey]
    if (previousTimer) window.clearTimeout(previousTimer)
    const timer = window.setTimeout(() => {
      if (readerStateSaveTimersRef.current[scopeKey] === timer) {
        delete readerStateSaveTimersRef.current[scopeKey]
      }
      persistReaderHighlights(convId, sourcePath, highlights)
    }, 700)
    readerStateSaveTimersRef.current[scopeKey] = timer
    return undefined
  }, [
    activeConvId,
    activeReaderHighlightScope,
    activeReaderSessionHighlights,
    activeReaderSourcePath,
    persistReaderHighlights,
  ])

  useEffect(() => {
    if (typeof BroadcastChannel === 'undefined') return undefined
    const channel = new BroadcastChannel(READER_SESSION_SYNC_CHANNEL)
    channel.onmessage = (event) => {
      const data = (event?.data && typeof event.data === 'object')
        ? event.data as Record<string, unknown>
        : {}
      if (String(data.type || '') !== 'reader-session-state') return
      const sourcePath = String(data.sourcePath || '').trim()
      if (!sourcePath) return
      const conversationId = String(data.conversationId || '').trim()
      if (conversationId && activeConvId && conversationId !== activeConvId) return
      const highlights = Array.isArray(data.highlights)
        ? data.highlights.filter((item): item is ReaderSessionHighlight => Boolean(item) && typeof item === 'object')
        : null
      if (!highlights) return
      const scopeKey = readerHighlightScopeKey(activeConvId, sourcePath)
      if (!scopeKey) return
      readerStateHydratedKeysRef.current.add(scopeKey)
      setReaderSessionHighlights((current) => {
        const prev = current[scopeKey] || []
        if (readerHighlightsSignature(prev) === readerHighlightsSignature(highlights)) return current
        return { ...current, [scopeKey]: highlights }
      })
    }
    return () => {
      channel.close()
    }
  }, [activeConvId])
  const addReaderSessionHighlight = (highlight: ReaderSessionHighlight) => {
    const scopeKey = activeReaderHighlightScope
    if (!scopeKey) return
    setReaderSessionHighlights((current) => {
      const list = Array.isArray(current[scopeKey]) ? current[scopeKey] : []
      if (list.some((item) => sameReaderSessionHighlight(item, highlight))) {
        return current
      }
      return {
        ...current,
        [scopeKey]: [...list, highlight],
      }
    })
  }
  const removeReaderSessionHighlight = (highlightId: string) => {
    const scopeKey = activeReaderHighlightScope
    const targetId = String(highlightId || '').trim()
    if (!scopeKey || !targetId) return
    setReaderSessionHighlights((current) => {
      const list = Array.isArray(current[scopeKey]) ? current[scopeKey] : []
      const next = list.filter((item) => String(item.id || '').trim() !== targetId)
      if (next.length === list.length) return current
      return {
        ...current,
        [scopeKey]: next,
      }
    })
  }
  const updateReaderSessionHighlight = (highlight: ReaderSessionHighlight) => {
    const scopeKey = activeReaderHighlightScope
    const targetId = String(highlight?.id || '').trim()
    if (!scopeKey || !targetId) return
    setReaderSessionHighlights((current) => {
      const list = Array.isArray(current[scopeKey]) ? current[scopeKey] : []
      let changed = false
      const next = list.map((item) => {
        if (String(item.id || '').trim() !== targetId) return item
        changed = true
        return { ...item, ...highlight }
      })
      if (!changed) return current
      return {
        ...current,
        [scopeKey]: next,
      }
    })
  }

  const timelineUiReady = !conversationLoading && timelineItems.length > 0
  const dockTimelineAvailable = timelineUiReady && timelineItems.length > 1
  const dockShelfAvailable = citationShelfOpen || citationShelfCount > 0
  const dockReaderAvailable = readerOpen
  const showRightDock = desktopReaderEligible && (dockTimelineAvailable || dockShelfAvailable || dockReaderAvailable)
  const activeRightDockPanel: RightDockPanel | null = showRightDock
    ? (
      rightDockPanel === 'reader' && dockReaderAvailable
        ? 'reader'
        : rightDockPanel === 'shelf' && (dockShelfAvailable || citationShelfOpen)
          ? 'shelf'
          : rightDockPanel === 'timeline' && dockTimelineAvailable
            ? 'timeline'
            : dockReaderAvailable
              ? 'reader'
              : dockShelfAvailable
                ? 'shelf'
                : dockTimelineAvailable
                  ? 'timeline'
                  : null
  )
    : null
  const desktopReaderVisible = readerOpen && desktopReaderEligible
  const rightDockExpanded = showRightDock && !rightDockCollapsed
  const showDesktopTimeline = false
  const showInlineTimelineToggle = timelineUiReady && !desktopReaderEligible
  const showConversationMeta = !conversationLoading && (timelineUiReady || researchContext.mode === 'paper_guide')
  const hideConversationMetaOnDesktop = showRightDock && researchContext.mode !== 'paper_guide'
  const guideSourceLabel = researchContext.guideSource.label || S.guide_unbound
  const guideSourceReady = researchContext.guideSource.ready
  const guideStatusLabel = guideSourceReady ? S.timeline_guide_ready : S.timeline_guide_pending
  const beginRightDockResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!rightDockExpanded || !event.isPrimary) return
    const currentWidth = clampRightDockWidth(rightDockWidthLiveRef.current)
    rightDockResizeRef.current = {
      startX: event.clientX,
      startWidth: currentWidth,
    }
    rightDockActivePointerIdRef.current = event.pointerId
    rightDockResizePreviewWidthRef.current = currentWidth
    updateRightDockResizeGuide(currentWidth)
    const activeElement = document.activeElement
    if (
      activeElement instanceof HTMLElement
      && (activeElement.tagName === 'TEXTAREA' || activeElement.tagName === 'INPUT')
    ) {
      rightDockResizeFocusRestoreRef.current = activeElement
      activeElement.blur()
    } else {
      rightDockResizeFocusRestoreRef.current = null
    }
    setRightDockResizing(true)
    document.body.classList.add('kb-right-dock-resizing')
    document.body.style.setProperty('cursor', 'col-resize')
    document.body.style.setProperty('user-select', 'none')
    event.currentTarget.setPointerCapture(event.pointerId)
    event.preventDefault()
  }

  const handleRightDockResizeMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (rightDockActivePointerIdRef.current !== event.pointerId) return
    const state = rightDockResizeRef.current
    if (!state) return
    updateRightDockResizeGuide(state.startWidth + (state.startX - event.clientX))
    event.preventDefault()
  }

  const commitRightDockResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (rightDockActivePointerIdRef.current !== event.pointerId) return
    finishRightDockResize(true)
    event.preventDefault()
  }

  const cancelRightDockResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (rightDockActivePointerIdRef.current !== event.pointerId) return
    finishRightDockResize(false)
    event.preventDefault()
  }
  const chatComposer = (
    <>
      {currentSelectedResearchContext ? (
        <div className="kb-chat-context-pack-wrap" data-testid="chat-context-pack">
          <div className="kb-chat-context-pack">
            <div className="kb-chat-context-pack-main">
              <span className="kb-chat-context-pack-label">
                {S.research_context_pack_label || 'Next answer context'}
              </span>
              <span className="kb-chat-context-pack-text">
                {(S.research_context_pack_summary || '{n} excerpts · ~{tokens} tokens')
                  .replace('{n}', String(currentSelectedResearchContext.items.length))
                  .replace('{tokens}', String(currentSelectedResearchContext.tokenEstimate))}
              </span>
            </div>
            <button
              type="button"
              className="kb-chat-context-pack-clear"
              onClick={() => setSelectedResearchContext(null)}
              data-testid="chat-context-pack-clear"
            >
              {S.research_context_pack_clear || 'Clear'}
            </button>
          </div>
        </div>
      ) : null}
      <ChatInput
        onSend={onSend}
        onStop={cancelGen}
        onUpload={onUpload}
        onRetryUploadItem={onRetryUpload}
        onCancelUploadItem={onCancelUpload}
        onRemoveImage={removePendingImage}
        onDismissUploadItem={dismissUploadItem}
        onStartGuideFromUpload={onStartGuideFromUpload}
        uploadItems={uploadItems}
        pendingImages={pendingImages}
        uploading={uploading}
        generating={!!generation}
        appendSignal={appendSignal}
        queryScope={resolveQueryScope(queryScope, {
          hasCurrentPaper: Boolean(researchContext.activeSource.ready),
          hasBasket: Boolean(currentSelectedResearchContext?.items?.length),
        })}
        queryScopeOptions={[
          { value: 'current_paper', disabled: !researchContext.activeSource.ready },
          { value: 'basket', disabled: !currentSelectedResearchContext?.items?.length },
          { value: 'library' },
        ]}
        onQueryScopeChange={setQueryScope}
        agentMode={agentMode}
        onAgentModeChange={setAgentMode}
      />
    </>
  )
  const apiConnectionAlertTarget = researchContext.api.connectionAlertTarget
  const apiConnectionProvider = apiConnectionAlertTarget === 'vision'
    ? researchContext.api.vision
    : researchContext.api.text
  const apiConnectionAlertDesc = apiConnectionAlertTarget === 'vision'
    ? S.settings_missing_vision_api_desc
    : apiConnectionProvider.status === 'failed' && (apiConnectionProvider.lastError || apiConnectionProvider.reason)
      ? S.chat_api_failed_desc.replace('{error}', apiConnectionProvider.lastError || apiConnectionProvider.reason)
      : S.chat_api_missing_desc
  const connectionAlert = apiConnectionAlertTarget ? (
    <div className="kb-chat-connection-alert">
      <Alert
        type="warning"
        showIcon
        message={apiConnectionAlertTarget === 'vision' ? S.settings_missing_vision_api_title : S.chat_api_missing_title}
        description={apiConnectionAlertDesc}
        action={(
          <Button size="small" onClick={() => openApiSettings(apiConnectionAlertTarget)}>
            {S.chat_open_api_settings}
          </Button>
        )}
      />
    </div>
  ) : null
  const refsActivity = useMemo(() => summarizeRefsActivity(deferredRefs), [deferredRefs])
  const chatActivityItems = useMemo(() => {
    const items: Array<{ key: string; label: string; tone: 'active' | 'ready' | 'warning' }> = []
    if (conversationLoading || messagesLoadingMore) {
      items.push({ key: 'messages', label: S.chat_activity_messages, tone: 'active' })
    }
    if (liveRunning) {
      const stage = String(generation?.stage || '').trim()
      items.push({
        key: 'generation',
        label: stage ? `${S.chat_activity_generation} · ${stage}` : S.chat_activity_generation,
        tone: 'active',
      })
    }
    if (uploading) {
      items.push({ key: 'upload', label: S.chat_activity_upload, tone: 'active' })
    }
    if (refsActivity.pendingPackCount > 0) {
      items.push({
        key: 'refs',
        label: S.chat_activity_refs.replace('{n}', String(refsActivity.pendingPackCount)),
        tone: 'active',
      })
    }
    if (shelfActivity.count > 0) {
      items.push({
        key: 'shelf',
        label: S.chat_activity_shelf.replace('{n}', String(shelfActivity.count)),
        tone: 'active',
      })
    }
    if (researchContext.reader.open && researchContext.mode === 'paper_guide') {
      items.push({ key: 'reader', label: S.chat_activity_reader, tone: 'ready' })
    }
    if (apiConnectionAlertTarget && items.length > 0) {
      items.push({ key: 'api', label: S.chat_activity_api_attention, tone: 'warning' })
    }
    return items
  }, [
    S.chat_activity_api_attention,
    S.chat_activity_generation,
    S.chat_activity_messages,
    S.chat_activity_reader,
    S.chat_activity_refs,
    S.chat_activity_shelf,
    S.chat_activity_upload,
    conversationLoading,
    generation?.stage,
    liveRunning,
    messagesLoadingMore,
    researchContext.mode,
    researchContext.reader.open,
    refsActivity.pendingPackCount,
    shelfActivity.count,
    apiConnectionAlertTarget,
    uploading,
  ])
  const showActivityStrip = debugPanelEnabled || chatActivityItems.length > 0
  const chatActivityStrip = showActivityStrip ? (
    <div className="kb-chat-activity-shell">
      <div className="kb-chat-activity-strip" data-testid="chat-activity-strip" aria-live="polite">
        {chatActivityItems.map((item) => (
          <span
            key={item.key}
            className={`kb-chat-activity-pill is-${item.tone}`}
            data-testid={`chat-activity-${item.key}`}
          >
            {item.tone === 'active' ? <LoadingOutlined spin /> : <span className="kb-chat-activity-dot" aria-hidden="true" />}
            <span>{item.label}</span>
          </span>
        ))}
        {debugPanelEnabled ? (
          <span className="kb-chat-debug-strip" data-testid="chat-perf-panel">
            <BugOutlined />
            <span>{S.chat_debug_switch.replace('{n}', String(debugSnapshot.switchTotal)).replace('{ms}', String(debugSnapshot.switchAvgMs))}</span>
            <span>{S.chat_debug_refs.replace('{n}', String(debugSnapshot.refsTotal)).replace('{ms}', String(debugSnapshot.refsAvgMs))}</span>
            <span>{S.chat_debug_open.replace('{n}', String(debugSnapshot.openPhases))}</span>
            <span>{S.chat_debug_prep.replace('{n}', String(debugSnapshot.messagePrep))}</span>
          </span>
        ) : null}
      </div>
    </div>
  ) : null
  const researchContextAttrs = {
    'data-research-conversation-id': researchContext.conversationId,
    'data-research-project-id': researchContext.projectId,
    'data-research-mode': researchContext.mode,
    'data-research-task-mode': researchContext.taskMode,
    'data-research-source-kind': researchContext.activeSource.kind,
    'data-research-source-ready': researchContext.activeSource.ready ? '1' : '0',
    'data-research-reader-linked': researchContext.reader.linkedToConversation ? '1' : '0',
    'data-research-shelf-scope': researchContext.shelfScope,
    'data-research-api-text': researchContext.api.text.status,
    'data-research-api-vision': researchContext.api.vision.status,
    'data-research-api-block-target': researchContext.api.sendBlockTarget,
  } as const

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div data-testid="research-context-state" hidden {...researchContextAttrs} />
      {!activeConvId && messages.length === 0 ? (
        <>
          {connectionAlert}
          {chatActivityStrip}
          <div className="kb-empty-state flex flex-1 flex-col items-center justify-center gap-4 px-4">
            <div className="kb-empty-brand">
              <div className="kb-empty-logo-wrap flex h-14 w-14 items-center justify-center overflow-hidden rounded-full">
                <img src="/pi_logo.png" alt="Pi_zaya logo" className="kb-empty-logo h-9 w-9 object-contain" loading="lazy" />
              </div>
              <div className="kb-empty-copy">
                <div className="kb-empty-product">{S.brand_name}</div>
                <div className="kb-empty-typewriter" aria-label={S.brand_home_title}>
                  {S.brand_home_title}
                </div>
              </div>
            </div>
          </div>
          {chatComposer}
        </>
      ) : (
        <>
          {connectionAlert}
          {chatActivityStrip}
          {showLegacyUiBlocks ? (
            <div className="border-b border-[var(--border)] bg-[var(--panel)]/60 px-4 py-3">
              <div className="mx-auto flex max-w-5xl items-center gap-3">
                <Button size="small" loading={messagesLoadingMore} onClick={() => { void loadOlderMessages() }}>
                  {S.show_older.replace('{n}', String(Math.min(HISTORY_PAGE_SIZE, hiddenCount)))}
                </Button>
                <Button size="small" onClick={() => {}}>
                  展开全部
                </Button>
                <Text type="secondary" className="text-xs">
                  为了打开更快，当前先显示最近 {visibleMessages.length} 条消息，较早消息 {hiddenCount} 条已折叠。
                </Text>
              </div>
            </div>
          ) : null}

          {showLegacyUiBlocks ? (
            <div className="border-b border-[var(--border)] bg-[var(--panel)]/40 px-4 py-2">
              <div className="mx-auto flex max-w-5xl items-center gap-3">
                <Button size="small" onClick={() => {}}>
                  收起较早消息
                </Button>
                <Text type="secondary" className="text-xs">
                  当前已展开 {visibleMessages.length} 条消息。
                </Text>
              </div>
            </div>
          ) : null}

          {!liveRunning && messagesHasMoreBefore ? (
            <div className="border-b border-[var(--border)] bg-[var(--panel)]/60 px-4 py-3">
              <div className="mx-auto flex max-w-5xl items-center gap-3">
                <Button size="small" loading={messagesLoadingMore} onClick={() => { void loadOlderMessages() }}>
                  {S.show_older.replace('{n}', String(HISTORY_PAGE_SIZE))}
                </Button>
                <Text type="secondary" className="text-xs">
                  {S.show_older_paged}
                </Text>
              </div>
            </div>
          ) : null}

          {conversationLoading ? (
            <div className="border-b border-[var(--border)] bg-[var(--panel)]/40 px-4 py-2">
              <div className="mx-auto max-w-5xl">
                <Text type="secondary" className="text-xs">
                  {S.loading_conversation}
                </Text>
              </div>
            </div>
          ) : null}

          {liveRunning && hiddenCount > 0 ? (
            <div className="border-b border-[var(--border)] bg-[var(--panel)]/40 px-4 py-2">
              <div className="mx-auto max-w-5xl">
                <Text type="secondary" className="text-xs">
                  {S.live_stream_hint.replace('{n}', String(visibleMessages.length))}
                </Text>
              </div>
            </div>
          ) : null}

          {showConversationMeta ? (
            <div className={`px-4 pb-2 pt-3 ${hideConversationMetaOnDesktop ? 'lg:hidden' : ''}`}>
              <div className="mx-auto max-w-7xl">
                <section className="kb-chat-meta-shell">
                  <div
                    className="kb-chat-meta-strip"
                    data-testid="research-context-strip"
                    {...researchContextAttrs}
                  >
                    {timelineItems.length > 0 ? (
                      <div className="kb-chat-meta-inline-block">
                        <span className="kb-chat-meta-label">{S.timeline_label}</span>
                        <span className="kb-chat-meta-badge">{S.timeline_badge.replace('{n}', String(timelineItems.length))}</span>
                        {showInlineTimelineToggle ? (
                          <Button
                            size="small"
                            type="text"
                            className="kb-chat-meta-action"
                            onClick={toggleTimelineOpen}
                          >
                          {timelineOpen ? S.timeline_collapse : S.timeline_expand}
                          </Button>
                        ) : null}
                      </div>
                    ) : null}
                    {researchContext.mode === 'paper_guide' ? (
                      <div className="kb-chat-meta-inline-block kb-chat-meta-inline-guide">
                        <span className="kb-chat-meta-label">{S.timeline_guide_label}</span>
                        <span className="kb-chat-meta-source" title={guideSourceLabel}>{guideSourceLabel}</span>
                        <span className={`kb-chat-meta-state ${guideSourceReady ? 'is-ready' : 'is-pending'}`}>
                          {guideStatusLabel}
                        </span>
                      </div>
                    ) : null}
                  </div>
                  {timelineOpen && timelineItems.length > 0 ? (
                    <div className="kb-chat-meta-mobile-rail lg:hidden">
                      <div className="flex gap-2 overflow-x-auto">
                        {timelineItems.map((item) => (
                          <button
                            key={`m-timeline-mobile-${item.userMsgId}-${item.order}`}
                            type="button"
                            className={`kb-chat-meta-rail-pill ${activeTimelineUserMsgId === item.userMsgId ? 'is-active' : ''}`}
                            onClick={() => jumpToTimelineItem(item)}
                          >
                            Q{item.order}
                          </button>
                        ))}
                      </div>
                    </div>
                  ) : null}
                </section>
              </div>
            </div>
          ) : null}

          {showLegacyUiBlocks ? (
            <div className="border-b border-[var(--border)] bg-[var(--panel)]/30 px-4 py-2">
              <div className="mx-auto flex max-w-7xl items-center justify-between gap-3">
                <Text type="secondary" className="text-xs">
                  会话时间线：{timelineItems.length} 个提问节点
                </Text>
                <Button size="small" onClick={() => setTimelineOpen((v) => !v)}>
                  {timelineOpen ? '收起时间线' : '打开时间线'}
                </Button>
              </div>
            </div>
          ) : null}

          {showLegacyUiBlocks ? (
            <div className="border-b border-[var(--border)] bg-[var(--panel)]/20 px-3 py-2 lg:hidden">
              <div className="flex gap-2 overflow-x-auto">
                {timelineItems.map((item) => (
                  <button
                    key={`m-timeline-mobile-${item.userMsgId}-${item.order}`}
                    type="button"
                    className={`shrink-0 rounded-full border px-3 py-1 text-xs ${
                      activeTimelineUserMsgId === item.userMsgId
                        ? 'border-[var(--accent)] bg-[var(--accent)]/10 text-[var(--accent)]'
                        : 'border-[var(--border)] bg-[var(--panel)] text-black/70 dark:text-white/70'
                    }`}
                    onClick={() => jumpToTimelineItem(item)}
                  >
                    Q{item.order}
                  </button>
                ))}
              </div>
            </div>
          ) : null}

          {showLegacyUiBlocks ? (
            <div className="border-b border-[var(--border)] bg-[var(--panel)]/40 px-4 py-2">
              <div className="mx-auto flex max-w-7xl items-center justify-between gap-3">
                <Text className="text-xs">
                  阅读指导模式：
                  <span className="ml-1 font-medium">
                    {guideSourceLabel}
                  </span>
                </Text>
                <Text type="secondary" className="text-xs">
                  {guideStatusLabel}
                </Text>
              </div>
            </div>
          ) : null}

          <div
            ref={splitLayoutRef}
            className={`kb-chat-main-region relative flex min-h-0 flex-1 ${
              showDesktopTimeline ? 'has-timeline-rail' : ''
            }`}
          >
            <div className={`kb-chat-workspace flex min-h-0 min-w-0 flex-1 flex-col ${
              citationShelfOpen ? 'is-citation-shelf-open' : ''
            }`}>
              <div className="flex min-h-0 min-w-0 flex-1 flex-col">
                {conversationLoading ? (
                  <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-hidden px-6 py-6">
                    <div className="h-5 w-40 animate-pulse rounded-full bg-black/[0.06] dark:bg-white/[0.08]" />
                    <div className="ml-auto h-24 w-[68%] animate-pulse rounded-[28px] bg-black/[0.05] dark:bg-white/[0.06]" />
                    <div className="h-32 w-[82%] animate-pulse rounded-[28px] bg-black/[0.04] dark:bg-white/[0.05]" />
                    <div className="ml-auto h-20 w-[58%] animate-pulse rounded-[28px] bg-black/[0.05] dark:bg-white/[0.06]" />
                  </div>
                ) : (
                  <MessageList
                    activeConvId={activeConvId}
                    shelfProjectId={shelfProjectId}
                    messages={visibleMessages}
                    refs={deferredRefs}
                    generationPartial={generation?.partial}
                    generationStage={generation?.stage}
                    generationTrace={generation?.researchTrace}
                    generationAgentTrace={generation?.agentTrace}
                    jumpTarget={timelineJump}
                    onJumpHandled={(handled) => {
                      setTimelineJump((current) => (
                        current?.token === handled.token && current?.messageId === handled.messageId
                          ? null
                          : current
                      ))
                      setActiveTimelineUserMsgId(null)
                    }}
                    trackedMessageIds={timelineTrackedMessageIds}
                    onTrackedMessageActive={(messageId) => {
                      const nextUserMsgId = messageId != null
                        ? (timelineUserMsgIdByTargetMsgId.get(messageId) ?? null)
                        : null
                      startTransition(() => {
                        setActiveTimelineUserMsgId((current) => (
                          current === nextUserMsgId ? current : nextUserMsgId
                        ))
                      })
                    }}
                    onOpenReader={openReader}
                    readerLocateResults={readerLocateResults}
                    onShelfOpenChange={handleCitationShelfOpenChange}
                    onShelfStateChange={handleCitationShelfStateChange}
                    onShelfActivityChange={handleShelfActivityChange}
                    openShelfSignal={openShelfSignal}
                    shelfDockMode={showRightDock}
                    shelfPortalTarget={shelfDockTarget}
                    shelfVisible={activeRightDockPanel === 'shelf'}
                    sourceQualityRefreshToken={sourceQualityRefreshToken}
                    paperGuideSourcePath={effectiveGuide.sourcePath}
                    paperGuideSourceName={effectiveGuide.sourceName}
                    selectedResearchContextKeys={selectedResearchContextKeys}
                    onResearchContextPackChange={handleResearchContextPackChange}
                    onResearchContextFollowUp={handleResearchContextFollowUp}
                  />
                )}
              </div>
              {chatComposer}
            </div>
            {rightDockExpanded ? (
              <div
                ref={rightDockResizeGuideRef}
                className={`pointer-events-none absolute inset-y-0 z-20 hidden w-0 xl:block ${
                  rightDockResizing ? 'opacity-100' : 'opacity-0'
                }`}
                aria-hidden="true"
              >
                <div className="absolute inset-y-0 -translate-x-1/2 border-l-2 border-[var(--accent)]/75 shadow-[0_0_0_1px_rgba(22,119,255,0.15)]" />
              </div>
            ) : null}
            {rightDockExpanded ? (
              <div
                className={`kb-chat-side-resize-handle ${rightDockResizing ? 'is-resizing' : ''}`}
                aria-label={S.side_dock_resize || 'Resize right sidebar'}
                onPointerDown={beginRightDockResize}
                onPointerMove={handleRightDockResizeMove}
                onPointerUp={commitRightDockResize}
                onPointerCancel={cancelRightDockResize}
              />
            ) : null}
            {showRightDock ? (
              <aside
                className={`kb-chat-side-dock is-${activeRightDockPanel || 'empty'} ${rightDockCollapsed ? 'is-collapsed' : ''} ${rightDockResizing ? 'is-resizing' : ''}`}
                style={{
                  width: rightDockCollapsed ? `${DESKTOP_DOCK_COLLAPSED_WIDTH}px` : `${rightDockWidth}px`,
                  transition: rightDockResizing ? 'none' : DESKTOP_DOCK_WIDTH_TRANSITION,
                }}
              >
                <div className="kb-chat-side-workspace-head">
                  <span>{S.side_dock_workspace || 'Research workspace'}</span>
                </div>
                <div className="kb-chat-side-tabs" role="tablist" aria-label={S.side_dock_workspace || 'Research side workspace'}>
                  <button
                    type="button"
                    className="kb-chat-side-collapse-btn"
                    aria-label={rightDockCollapsed ? (S.side_dock_expand || 'Expand right sidebar') : (S.side_dock_collapse || 'Collapse right sidebar')}
                    title={rightDockCollapsed ? (S.side_dock_expand || 'Expand right sidebar') : (S.side_dock_collapse || 'Collapse right sidebar')}
                    onClick={() => setRightDockCollapsed((value) => !value)}
                  >
                    {rightDockCollapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
                  </button>
                  <button
                    type="button"
                    role="tab"
                    aria-selected={activeRightDockPanel === 'shelf'}
                    className={`kb-chat-side-tab ${activeRightDockPanel === 'shelf' ? 'is-active' : ''}`}
                    onClick={() => activateDockPanel('shelf')}
                  >
                    <BookOutlined />
                    <span>{S.side_dock_basket || S.shelf_title}</span>
                    {citationShelfCount > 0 ? <strong>{citationShelfCount}</strong> : null}
                  </button>
                  {dockReaderAvailable ? (
                    <button
                      type="button"
                      role="tab"
                      aria-selected={activeRightDockPanel === 'reader'}
                      className={`kb-chat-side-tab ${activeRightDockPanel === 'reader' ? 'is-active' : ''}`}
                      onClick={() => activateDockPanel('reader')}
                    >
                      <ReadOutlined />
                      <span>{S.side_dock_reader_locate || S.side_dock_reader || 'Reader'}</span>
                    </button>
                  ) : null}
                  {dockTimelineAvailable ? (
                    <button
                      type="button"
                      role="tab"
                      aria-selected={activeRightDockPanel === 'timeline'}
                      className={`kb-chat-side-tab ${activeRightDockPanel === 'timeline' ? 'is-active' : ''}`}
                      onClick={() => activateDockPanel('timeline')}
                    >
                      <ClockCircleOutlined />
                      <span>{S.side_dock_path || S.timeline_label}</span>
                      <strong>{timelineItems.length}</strong>
                    </button>
                  ) : null}
                </div>
                <div className="kb-chat-side-body">
                  {dockTimelineAvailable ? (
                    <section className={`kb-chat-side-panel kb-chat-side-timeline ${activeRightDockPanel === 'timeline' ? 'is-active' : ''}`}>
                      <div className="kb-chat-side-panel-head">
                        <div>
                          <div className="kb-chat-side-panel-title">{S.side_dock_path_title || S.timeline_label}</div>
                          <div className="kb-chat-side-panel-subtitle">{S.timeline_badge.replace('{n}', String(timelineItems.length))}</div>
                        </div>
                      </div>
                      <div className="kb-chat-timeline-list is-docked">
                        {timelineItems.map((item) => (
                          <button
                            key={`m-timeline-dock-${item.userMsgId}-${item.order}`}
                            type="button"
                            className={`kb-chat-timeline-item ${
                              activeTimelineUserMsgId === item.userMsgId
                                ? 'is-active'
                                : ''
                            } ${item.hasAnswer ? 'is-ready' : 'is-pending'}`}
                            aria-current={activeTimelineUserMsgId === item.userMsgId ? 'step' : undefined}
                            aria-label={`Q${item.order}: ${item.questionPreview}`}
                            title={item.questionPreview}
                            onClick={() => jumpToTimelineItem(item)}
                          >
                            <span className="kb-chat-timeline-item-node" aria-hidden="true" />
                            <div className="kb-chat-timeline-item-meta">
                              <span className="kb-chat-timeline-item-order">Q{item.order}</span>
                              <span className={`kb-chat-timeline-item-status ${item.hasAnswer ? 'is-ready' : 'is-pending'}`}>
                                {item.hasAnswer ? S.timeline_answered : S.timeline_pending_qa}
                              </span>
                            </div>
                            <div className="kb-chat-timeline-item-text">
                              {item.questionPreview}
                            </div>
                          </button>
                        ))}
                      </div>
                    </section>
                  ) : null}
                  <section className={`kb-chat-side-panel kb-chat-side-shelf ${activeRightDockPanel === 'shelf' ? 'is-active' : ''}`}>
                    <div ref={setShelfDockTarget} className="kb-chat-side-shelf-host" />
                  </section>
                  {dockReaderAvailable ? (
                    <section className={`kb-chat-side-panel kb-chat-side-reader ${activeRightDockPanel === 'reader' ? 'is-active' : ''}`}>
                      <PaperGuideReaderDrawer
                        open={readerOpen}
                        payload={readerPayload}
                        onClose={() => setReaderOpen(false)}
                        onAppendSelection={appendReaderSelection}
                        presentation="inline"
                        onCollapse={() => setRightDockCollapsed(true)}
                        onOpenStandalone={() => { void openReaderStandalone(readerPayload) }}
                        conversationId={activeConvId || ''}
                        sessionHighlights={activeReaderSessionHighlights}
                        onAddSessionHighlight={addReaderSessionHighlight}
                        onUpdateSessionHighlight={updateReaderSessionHighlight}
                        onRemoveSessionHighlight={removeReaderSessionHighlight}
                        onLocateResult={handleReaderLocateResult}
                        onAddSelectionToShelf={addReaderSelectionToShelf}
                        onAddCitationToShelf={addReaderCitationToShelf}
                        onOpenCitationShelf={openReaderCitationShelf}
                      />
                    </section>
                  ) : null}
                </div>
              </aside>
            ) : null}
          </div>
        </>
      )}
      {!desktopReaderVisible ? (
        <PaperGuideReaderDrawer
          open={readerOpen}
          payload={readerPayload}
          onClose={() => setReaderOpen(false)}
          onAppendSelection={appendReaderSelection}
          onOpenStandalone={() => { void openReaderStandalone(readerPayload) }}
          conversationId={activeConvId || ''}
          sessionHighlights={activeReaderSessionHighlights}
          onAddSessionHighlight={addReaderSessionHighlight}
          onUpdateSessionHighlight={updateReaderSessionHighlight}
          onRemoveSessionHighlight={removeReaderSessionHighlight}
          onLocateResult={handleReaderLocateResult}
          onAddSelectionToShelf={addReaderSelectionToShelf}
          onAddCitationToShelf={addReaderCitationToShelf}
          onOpenCitationShelf={openReaderCitationShelf}
        />
      ) : null}
    </div>
  )
}
