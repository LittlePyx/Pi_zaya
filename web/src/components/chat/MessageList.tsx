import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type MouseEvent } from 'react'
import { createPortal } from 'react-dom'
import { Button, Typography, message } from 'antd'
import { ReloadOutlined, UserOutlined } from '@ant-design/icons'
import { MarkdownRenderer } from './MarkdownRenderer'
import { CopyBar } from './CopyBar'
import { CitationPopover } from './CitationPopover'
import { CiteShelf } from './CiteShelf'
import {
  buildCitationPopoverMetadataPlan,
  loadCitationPopoverMetadata,
} from './citationPopoverMetadata'
import { useCitationPopoverPreview } from './useCitationPopoverPreview'
import { useCitationPopoverState } from './useCitationPopoverState'
import type {
  ReaderLocateResult,
  ReaderOpenPayload,
} from './reader/readerTypes'
import {
  READER_CITATION_SHELF_CHANNEL,
  READER_CITATION_SHELF_EVENT,
  READER_SELECTION_SHELF_CHANNEL,
  READER_SELECTION_SHELF_EVENT,
} from './reader/readerTypes'
import { buildBasicReaderOpenPayload } from './reader/readerOpenPayloadUtils'
import {
  buildGuideLocateCandidates,
  type LocateCandidate,
  type RefHitLite,
} from './reader/messageLocateCandidates'
import {
  buildHeuristicReaderOpenPayload,
  buildStructuredEntryReaderOpenPayload,
} from './reader/messageReaderLocatePayload'
import {
  type ProvenanceLocateEntry,
} from './reader/messageStructuredProvenance'
import {
  createStructuredInlineLocateResolver,
  type StructuredRenderLocateSlot,
} from './reader/messageStructuredInlineLocate'
import {
  mergeCiteMeta,
  normalizeCiteDetail,
  normalizeShelfNote,
  normalizeShelfTags,
  shelfProjectScopeId,
  shelfItemNeedsMetadataRepair,
  shelfItemRepairFingerprint,
  shelfStorageKey,
  strictRepairMerge,
  toShelfItem,
  type CiteDetail,
  type CiteShelfItem,
} from './citationState'
import {
  SHELF_MAX_ITEMS,
  articleSummaryPatchFromMeta,
  dedupeShelfItems,
  looksLowValueShelfSummary,
  mergeCitationDetailIntoShelfItems,
  mergeReaderSelectionDetailIntoShelfItems,
  mergeShelfItemWithLive,
  sameShelfItem,
  sameShelfItems,
  shelfDiscoverySourceDetail,
  shelfItemHasDisplayableArticleSummary,
  shelfLibraryFullTextDetail,
  shelfItemNeedsPersistedMetadataHydrate,
  shelfItemNeedsSummaryBackfill,
  shelfItemsForBackend,
  shelfMetadataHydrateAttemptKey,
  shelfPaperIdentity,
  shelfRepairMetaFromEntry,
  shelfRepairPayloads,
  shelfSummaryBackfillAttemptKey,
  shouldRequestCitationCardPolish,
  snapshotDiffCounts,
} from './citeShelfRuntime'
import {
  SHELF_SAVED_MAX_ITEMS,
  SHELF_SAVED_SUFFIX,
  invalidateSavedShelfSnapshotCache,
  invalidateShelfSnapshotCache,
  legacyShelfStorageKeys,
  migrateLegacySavedShelfSnapshots,
  migrateLegacyShelfSnapshot,
  persistSavedShelfSnapshots,
  persistShelfSnapshot,
  readSavedShelfSnapshots,
  readShelfSnapshot,
  restoreShelfItems,
  shelfSavedStorageKey,
  type ShelfSavedSnapshot,
} from './citeShelfStorage'
import {
  citeDetailFromReaderSelection,
  normalizeReaderCitationShelfPayload,
  normalizeReaderSelectionShelfPayload,
  readerSelectionNote,
} from './readerShelfPayload'
import { RefsPanel } from '../refs/RefsPanel'
import { chatApi, type Message } from '../../api/chat'
import { referencesApi, type ShelfMetadataRepairImpact } from '../../api/references'
import { useT } from '../../i18n'
import { useChatStore } from '../../stores/chatStore'
import { basenameFromSourcePath } from '../../utils/sourcePath'
import {
  cleanAssistantAnswerPresentationText,
  getMessageCiteDetailRecords,
  getMessageCopyMarkdownValue,
  getMessageCopyTextValue,
  getMessageRenderedBodyContent,
  getMessageRenderPacket,
  splitLeadingAssistantSourceNotice,
} from './messageRenderPacket'
import { remapStructuredEntryToGuideAnchors } from './messageStructuredLocateRemap'
import {
  buildSelectedResearchContextPack,
  buildSelectedResearchContextPackFromItems,
  type SelectedResearchContextPack,
  type SelectedResearchContextItem,
} from './researchContextPack'
import {
  pushMessageListPrepPerf,
  type MessageListPrepPerfEvent,
} from './messageListPerf'
import {
  buildAssistantLocatePrepByMsgId,
  shouldSuppressLooseInlineLocate,
  type AssistantLocatePrep,
} from './messageLocatePrep'
import { withBibliometricsLocale } from './bibliometricsLocale'
import {
  buildUnlinkedReferenceViews,
  enrichCiteDetailsWithVisibleRefContext,
} from './messageCitationViews'
import {
  buildAssistantTraceByMsgId,
  buildLiveCiteMap,
  buildMessageRows,
  buildSelectedResearchContextByAssistantId,
  type MessageRow,
} from './messageListDerived'
import { buildFallbackCiteDetailsFromRefHits } from './messageFallbackCitations'
import { resolveLowConfidenceMeta, stripLeadingLowConfidenceNotice } from './messageLowConfidence'
import { createMessageLocateResolvers } from './messageLocateResolvers'
import { buildMessageMarkdownLocateProps } from './messageMarkdownLocateProps'
import { MessageProvenanceChips } from './MessageProvenanceChips'
import { MessageReferenceCandidates } from './MessageReferenceCandidates'
import { UserMessageBubble } from './UserMessageBubble'
import { AssistantMessageNotices, AssistantSourceNotice, AssistantSourceSummaryNotice } from './AssistantMessageNotices'
import {
  lookupGuideCandidatesBySourcePath,
  sourcePathLookupKeys,
} from './messageSourceIdentity'
import {
  contextItemTitle,
  getMessageAgentTrace,
  getMessageResearchTrace,
  imageAttachmentsOf,
  isImageOnlyPlaceholder,
  messageHasAgentTraceHint,
} from './messageTraceUtils'
import {
  sourceSummaryFromAnswerContract,
  type AnswerSourceNoticeViewModel,
} from './answerSourceNoticeViewModel'
import { AgentTracePanel } from './AgentTracePanel'
import { ResearchTracePanel } from './ResearchTracePanel'
import { ResearchContextReceipt } from './ResearchContextReceipt'
import { EvidenceDrawer } from './EvidenceDrawer'
import { generationRetryPrompt, isGenerationFailureAnswer } from './generationFailureUi'

const { Text } = Typography
const SHELF_BACKEND_PERSIST_MS = 320
const SHELF_AUTO_REPAIR_BATCH_SIZE = 8
const SHELF_AUTO_REPAIR_RETRY_MS = 15000
const SHELF_METADATA_HYDRATE_BATCH_SIZE = 8
const SHELF_METADATA_HYDRATE_RETRY_MS = 15000
const SHELF_SUMMARY_BACKFILL_BATCH_SIZE = 24
const SHELF_SUMMARY_BACKFILL_RETRY_MS = 60000

type ShelfAsyncScopeToken = {
  epoch: number
  storageKey: string
}

export interface ShelfActivityState {
  summary: boolean
  repair: boolean
  autoRepair: boolean
  background: boolean
  count: number
}

interface Props {
  activeConvId?: string | null
  shelfProjectId?: string | null
  messages: Message[]
  refs: Record<string, unknown>
  generationPartial?: string
  generationStage?: string
  generationTrace?: Record<string, unknown>
  generationAgentTrace?: Record<string, unknown>
  generationAgentSourceSummary?: Record<string, unknown>
  generationAnswerContract?: Record<string, unknown>
  jumpTarget?: { messageId: number; token: number } | null
  onJumpHandled?: (jumpTarget: { messageId: number; token: number }) => void
  trackedMessageIds?: number[]
  onTrackedMessageActive?: (messageId: number | null) => void
  onOpenReader?: (payload: ReaderOpenPayload) => void
  onShelfOpenChange?: (open: boolean) => void
  onShelfStateChange?: (state: { open: boolean; count: number }) => void
  onShelfActivityChange?: (state: ShelfActivityState) => void
  closeShelfSignal?: number
  openShelfSignal?: number
  shelfDockMode?: boolean
  shelfPortalTarget?: HTMLElement | null
  shelfVisible?: boolean
  readerLocateResults?: Record<string, ReaderLocateResult>
  sourceQualityRefreshToken?: number
  paperGuideSourcePath?: string
  paperGuideSourceName?: string
  selectedResearchContextKeys?: Record<string, boolean>
  onResearchContextPackChange?: (pack: SelectedResearchContextPack | null) => void
  onResearchContextFollowUp?: (pack: SelectedResearchContextPack, promptText: string) => void
  onRetryMessage?: (promptText: string) => void
}

interface RefEntryLite {
  hits?: RefHitLite[]
  display_state?: string
  suppression_reason?: string
  suggestion?: string
  guide_filter?: { hidden_self_source?: boolean; filtered_hit_count?: number }
}

function bibliometricsSummaryFetchFailed(meta: Record<string, unknown>): boolean {
  const status = String(meta.summary_fetch_status || meta.summaryFetchStatus || '').trim().toLowerCase()
  return status === 'failed' || status === 'retryable'
}

function AssistantAvatar() {
  return (
    <div className="kb-msg-avatar kb-msg-avatar-assistant">
      <img src="/pi_logo.png" alt="Pi assistant" className="h-5 w-5 object-contain" loading="lazy" />
    </div>
  )
}

export function MessageList({
  activeConvId,
  shelfProjectId,
  messages,
  refs,
  generationPartial,
  generationStage,
  generationTrace,
  generationAgentTrace,
  generationAgentSourceSummary,
  generationAnswerContract,
  jumpTarget,
  onJumpHandled,
  trackedMessageIds,
  onTrackedMessageActive,
  onOpenReader,
  onShelfOpenChange,
  onShelfStateChange,
  onShelfActivityChange,
  closeShelfSignal = 0,
  openShelfSignal = 0,
  shelfDockMode = false,
  shelfPortalTarget = null,
  shelfVisible,
  readerLocateResults = {},
  sourceQualityRefreshToken = 0,
  paperGuideSourcePath,
  paperGuideSourceName,
  selectedResearchContextKeys = {},
  onResearchContextPackChange,
  onResearchContextFollowUp,
  onRetryMessage,
}: Props) {
  const createPaperGuideConversation = useChatStore((s) => s.createPaperGuideConversation)
  const scrollRef = useRef<HTMLDivElement>(null)
  const {
    activeRequestKeyRef: activePopoverRequestKeyRef,
    close: closeCitationPopoverState,
    detail: popoverDetail,
    guideLoading: popoverGuideLoading,
    loading: popoverLoading,
    mergeDetailForKey: mergePopoverDetailForItemKey,
    open: openCitationPopoverState,
    pinned: popoverPinned,
    position: popoverPos,
    setGuideLoading: setPopoverGuideLoading,
    setLoading: setPopoverLoading,
  } = useCitationPopoverState()
  const citationPreview = useCitationPopoverPreview()
  const [evidenceDrawerSource, setEvidenceDrawerSource] = useState<AnswerSourceNoticeViewModel | null>(null)
  const [evidenceDrawerCiteDetails, setEvidenceDrawerCiteDetails] = useState<CiteDetail[]>([])
  const citationPolishPrewarmKeysRef = useRef(new Set<string>())
  const [shelfOpen, setShelfOpen] = useState(false)
  const [shelfItems, setShelfItems] = useState<CiteShelfItem[]>([])
  const [focusedShelfKey, setFocusedShelfKey] = useState('')
  const [shelfSummaryLoadingKey, setShelfSummaryLoadingKey] = useState('')
  const [shelfSummaryStatusByKey, setShelfSummaryStatusByKey] = useState<Record<string, 'loading' | 'unavailable' | 'failed' | 'ready'>>({})
  const [shelfRepairLoadingKey, setShelfRepairLoadingKey] = useState('')
  const [shelfAutoRepairingKeys, setShelfAutoRepairingKeys] = useState<string[]>([])
  const [shelfBackgroundBusy, setShelfBackgroundBusy] = useState(false)
  const [shelfRepairImpact, setShelfRepairImpact] = useState<ShelfMetadataRepairImpact | null>(null)
  const [savedShelfSnapshots, setSavedShelfSnapshots] = useState<ShelfSavedSnapshot[]>([])
  const [selectedSavedSnapshotId, setSelectedSavedSnapshotId] = useState('')
  const [shelfMessageFlashId, setShelfMessageFlashId] = useState<number | null>(null)
  const assistantLocatePrepCacheRef = useRef(new Map<string, AssistantLocatePrep>())
  const assistantLocatePrepPerfRef = useRef<MessageListPrepPerfEvent | null>(null)
  const [guideDocCandidates, setGuideDocCandidates] = useState<LocateCandidate[]>([])
  const S = useT()
  const shelfScopeId = shelfProjectScopeId(shelfProjectId)
  const skipShelfPersistOnceRef = useRef(false)
  const persistShelfTimerRef = useRef<number | null>(null)
  const persistShelfBackendTimerRef = useRef<number | null>(null)
  const activeStorageKeyRef = useRef(shelfStorageKey(shelfScopeId))
  const shelfRevisionByKeyRef = useRef<Record<string, number>>({})
  const shelfBackendRevisionByKeyRef = useRef<Record<string, number>>({})
  const shelfBackendHydratedKeysRef = useRef(new Set<string>())
  const shelfEmptyBackendSaveIntentRef = useRef<Record<string, number>>({})
  const shelfBackendHydrateSeqRef = useRef(0)
  const shelfAsyncScopeEpochRef = useRef(0)
  const shelfStateTouchedAtRef = useRef(Date.now())
  const latestShelfStateRef = useRef<{ convId?: string | null; projectId?: string | null; open: boolean; items: CiteShelfItem[] }>({
    convId: activeConvId,
    projectId: shelfScopeId,
    open: false,
    items: [],
  })
  const flushShelfSnapshotRef = useRef<(() => void) | null>(null)
  const flushShelfBackendRef = useRef<(() => void) | null>(null)
  const shelfAutoRepairTimerRef = useRef<number | null>(null)
  const shelfAutoRepairingKeySetRef = useRef(new Set<string>())
  const shelfAutoRepairFingerprintsRef = useRef<Record<string, string>>({})
  const shelfAutoRepairRetryAfterRef = useRef<Record<string, number>>({})
  const shelfMetadataHydrateTimerRef = useRef<number | null>(null)
  const shelfMetadataHydrateInFlightRef = useRef(new Set<string>())
  const shelfMetadataHydrateAttemptedAtRef = useRef<Record<string, number>>({})
  const shelfSummaryBackfillTimerRef = useRef<number | null>(null)
  const shelfSummaryBackfillInFlightRef = useRef(new Set<string>())
  const shelfSummaryBackfillAttemptedAtRef = useRef<Record<string, number>>({})
  const shelfMessageFlashTimerRef = useRef<number | null>(null)
  const setShelfAutoRepairingKeySet = useCallback((nextSet: Set<string>) => {
    shelfAutoRepairingKeySetRef.current = nextSet
    setShelfAutoRepairingKeys(Array.from(nextSet))
  }, [])
  const captureShelfAsyncScope = useCallback((): ShelfAsyncScopeToken => ({
    epoch: shelfAsyncScopeEpochRef.current,
    storageKey: shelfStorageKey(shelfScopeId),
  }), [shelfScopeId])
  const shelfAsyncScopeIsCurrent = useCallback((token: ShelfAsyncScopeToken): boolean => (
    shelfAsyncScopeEpochRef.current === token.epoch
    && shelfStorageKey(latestShelfStateRef.current.projectId) === token.storageKey
  ), [])
  const currentShelfItemForAsync = useCallback((
    token: ShelfAsyncScopeToken,
    itemKey: string,
    expectedRepairFingerprint?: string,
  ): CiteShelfItem | null => {
    if (!shelfAsyncScopeIsCurrent(token)) return null
    const key = String(itemKey || '').trim()
    if (!key) return null
    const current = latestShelfStateRef.current.items.find((entry) => entry.key === key)
    if (!current) return null
    if (expectedRepairFingerprint && shelfItemRepairFingerprint(current) !== expectedRepairFingerprint) return null
    return current
  }, [shelfAsyncScopeIsCurrent])

  const persistShelfLocalNow = useCallback((items: CiteShelfItem[], open: boolean) => {
    const storageKey = shelfStorageKey(shelfScopeId)
    const currentRevision = Number(shelfRevisionByKeyRef.current[storageKey] || 0)
    const nextItems = dedupeShelfItems(items).slice(0, SHELF_MAX_ITEMS)
    const nextRevision = persistShelfSnapshot(storageKey, { open, items: nextItems }, currentRevision)
    shelfRevisionByKeyRef.current[storageKey] = nextRevision
    activeStorageKeyRef.current = storageKey
    latestShelfStateRef.current = {
      convId: activeConvId,
      projectId: shelfScopeId,
      open,
      items: nextItems,
    }
  }, [activeConvId, shelfScopeId])

  const markShelfEmptyBackendSaveIntent = useCallback((projectId?: string | null) => {
    const storageKey = shelfStorageKey(projectId)
    shelfEmptyBackendSaveIntentRef.current[storageKey] = Date.now() + 5000
  }, [])

  const saveShelfBackendNow = useCallback((
    state: { convId?: string | null; projectId?: string | null; open: boolean; items: CiteShelfItem[] },
    options?: { allowEmptyOverwrite?: boolean },
  ) => {
    const projectScopeId = shelfProjectScopeId(state.projectId)
    const storageKey = shelfStorageKey(projectScopeId)
    if (!shelfBackendHydratedKeysRef.current.has(storageKey)) return
    const items = shelfItemsForBackend(state.items)
    const emptyClosed = items.length <= 0 && !state.open
    const emptyIntentUntil = Number(shelfEmptyBackendSaveIntentRef.current[storageKey] || 0)
    const allowEmptyOverwrite = Boolean(options?.allowEmptyOverwrite || emptyIntentUntil > Date.now())
    if (emptyClosed && !allowEmptyOverwrite && Number(shelfBackendRevisionByKeyRef.current[storageKey] || 0) > 0) {
      return
    }
    void chatApi.saveCitationShelf({
      convId: state.convId || undefined,
      projectId: projectScopeId === '__default__' ? undefined : projectScopeId,
      scope: 'project',
      open: state.open,
      items,
      allowEmptyOverwrite,
    })
      .then((record) => {
        const latestKey = shelfStorageKey(projectScopeId)
        shelfBackendRevisionByKeyRef.current[latestKey] = Math.max(0, Number(record.revision || 0))
        shelfBackendHydratedKeysRef.current.add(latestKey)
        if (emptyClosed && allowEmptyOverwrite) {
          delete shelfEmptyBackendSaveIntentRef.current[latestKey]
        }
      })
      .catch(() => {
        // Local shelf storage remains the immediate fallback when the API is unavailable.
      })
  }, [])

  useEffect(() => {
    onShelfOpenChange?.(shelfOpen)
  }, [onShelfOpenChange, shelfOpen])

  useEffect(() => {
    onShelfStateChange?.({ open: shelfOpen, count: shelfItems.length })
  }, [onShelfStateChange, shelfItems.length, shelfOpen])

  useEffect(() => {
    const summary = Boolean(shelfSummaryLoadingKey)
    const repair = Boolean(shelfRepairLoadingKey)
    const autoRepair = shelfAutoRepairingKeys.length > 0
    const backgroundOnly = shelfBackgroundBusy && !summary && !repair && !autoRepair
    onShelfActivityChange?.({
      summary,
      repair,
      autoRepair,
      background: shelfBackgroundBusy,
      count: (summary ? 1 : 0) + (repair ? 1 : 0) + shelfAutoRepairingKeys.length + (backgroundOnly ? 1 : 0),
    })
  }, [onShelfActivityChange, shelfAutoRepairingKeys.length, shelfBackgroundBusy, shelfRepairLoadingKey, shelfSummaryLoadingKey])

  useEffect(() => () => {
    shelfAsyncScopeEpochRef.current += 1
    onShelfActivityChange?.({ summary: false, repair: false, autoRepair: false, background: false, count: 0 })
  }, [onShelfActivityChange])

  useEffect(() => {
    if (closeShelfSignal <= 0) return
    setShelfOpen(false)
  }, [closeShelfSignal])

  useEffect(() => {
    if (openShelfSignal <= 0) return
    setShelfOpen(true)
  }, [openShelfSignal])

  useEffect(() => {
    return () => {
      if (shelfAutoRepairTimerRef.current !== null) {
        window.clearTimeout(shelfAutoRepairTimerRef.current)
      }
      if (shelfMetadataHydrateTimerRef.current !== null) {
        window.clearTimeout(shelfMetadataHydrateTimerRef.current)
      }
      if (shelfSummaryBackfillTimerRef.current !== null) {
        window.clearTimeout(shelfSummaryBackfillTimerRef.current)
      }
      if (shelfMessageFlashTimerRef.current !== null) {
        window.clearTimeout(shelfMessageFlashTimerRef.current)
      }
      if (persistShelfBackendTimerRef.current !== null) {
        window.clearTimeout(persistShelfBackendTimerRef.current)
      }
    }
  }, [])

  useEffect(() => {
    const sourcePath = String(paperGuideSourcePath || '').trim()
    const sourceName = String(paperGuideSourceName || '').trim()
    if (!sourcePath) {
      setGuideDocCandidates([])
      return
    }
    let cancelled = false
    const ctrl = new AbortController()
    referencesApi.readerDoc(sourcePath, { signal: ctrl.signal })
      .then((res) => {
        if (cancelled) return
        const markdown = String(res.markdown || '')
        if (!markdown.trim()) {
          setGuideDocCandidates([])
          return
        }
        const resolvedName = String(res.source_name || sourceName || '').trim()
        const anchors = Array.isArray(res.anchors) ? res.anchors : []
        setGuideDocCandidates(
          buildGuideLocateCandidates(
            markdown,
            sourcePath,
            resolvedName || sourceName || sourcePath,
            'guide',
            anchors,
          ),
        )
      })
      .catch(() => {
        if (!cancelled) setGuideDocCandidates([])
      })
    return () => {
      cancelled = true
      ctrl.abort()
    }
  }, [paperGuideSourcePath, paperGuideSourceName])

  useLayoutEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const timer = window.requestAnimationFrame(() => {
      el.scrollTop = el.scrollHeight
    })
    return () => window.cancelAnimationFrame(timer)
  }, [activeConvId, generationPartial])

  useEffect(() => {
    if (!jumpTarget || !Number.isFinite(jumpTarget.messageId)) return
    const el = scrollRef.current
    if (!el) return
    const target = el.querySelector<HTMLElement>(`[data-msg-id="${jumpTarget.messageId}"]`)
    if (!target) return
    const targetRect = target.getBoundingClientRect()
    const containerRect = el.getBoundingClientRect()
    const top = targetRect.top - containerRect.top + el.scrollTop - 12
    el.scrollTo({ top: Math.max(0, top), behavior: 'smooth' })
    try {
      target.animate(
        [
          { boxShadow: '0 0 0 0 rgba(24,144,255,0.0)', backgroundColor: 'rgba(24,144,255,0.0)' },
          { boxShadow: '0 0 0 3px rgba(24,144,255,0.24)', backgroundColor: 'rgba(24,144,255,0.10)' },
          { boxShadow: '0 0 0 0 rgba(24,144,255,0.0)', backgroundColor: 'rgba(24,144,255,0.0)' },
        ],
        { duration: 900, easing: 'ease-out' },
      )
    } catch {
      // no-op: Web Animations may not be available in all envs.
    }
    onJumpHandled?.(jumpTarget)
  }, [jumpTarget, messages, onJumpHandled])

  useEffect(() => {
    if (!onTrackedMessageActive) return
    const el = scrollRef.current
    if (!el) return
    const trackedIds = Array.isArray(trackedMessageIds)
      ? trackedMessageIds.filter((id) => Number.isFinite(id))
      : []
    if (trackedIds.length <= 0) {
      onTrackedMessageActive(null)
      return
    }
    let syncFrameId = 0
    let measureFrameId = 0
    let lastReported: number | null = null
    let lastActiveIndex = -1
    let lastScrollTop = el.scrollTop
    let trackedAnchors: Array<{ id: number; top: number }> = []
    const SWITCH_HYSTERESIS_PX = 28

    const transitionMargin = (leftIndex: number, rightIndex: number) => {
      const leftTop = trackedAnchors[leftIndex]?.top ?? 0
      const rightTop = trackedAnchors[rightIndex]?.top ?? leftTop
      const gap = Math.max(0, rightTop - leftTop)
      return Math.min(SWITCH_HYSTERESIS_PX, Math.max(10, gap * 0.2))
    }

    const syncActiveMessage = () => {
      syncFrameId = 0
      if (trackedAnchors.length <= 0) {
        lastActiveIndex = -1
        if (lastReported !== null) {
          lastReported = null
          onTrackedMessageActive(null)
        }
        return
      }

      const currentScrollTop = el.scrollTop
      const anchorTop = currentScrollTop + Math.min(120, Math.max(48, el.clientHeight * 0.2))
      let low = 0
      let high = trackedAnchors.length - 1
      let activeIndex = 0
      while (low <= high) {
        const mid = Math.floor((low + high) / 2)
        if (trackedAnchors[mid]!.top <= anchorTop) {
          activeIndex = mid
          low = mid + 1
        } else {
          high = mid - 1
        }
      }
      if (lastActiveIndex >= 0 && lastActiveIndex < trackedAnchors.length && activeIndex !== lastActiveIndex) {
        const direction = currentScrollTop - lastScrollTop
        if (activeIndex === lastActiveIndex + 1 && direction >= 0) {
          const nextTop = trackedAnchors[activeIndex]?.top ?? 0
          if (anchorTop < nextTop + transitionMargin(lastActiveIndex, activeIndex)) {
            activeIndex = lastActiveIndex
          }
        } else if (activeIndex === lastActiveIndex - 1 && direction <= 0) {
          const currentTop = trackedAnchors[lastActiveIndex]?.top ?? 0
          if (anchorTop >= currentTop - transitionMargin(activeIndex, lastActiveIndex)) {
            activeIndex = lastActiveIndex
          }
        }
      }
      const activeMessageId = trackedAnchors[activeIndex]?.id ?? null
      lastScrollTop = currentScrollTop
      lastActiveIndex = activeMessageId != null ? activeIndex : -1

      if (activeMessageId !== lastReported) {
        lastReported = activeMessageId
        onTrackedMessageActive(activeMessageId)
      }
    }

    const scheduleSync = () => {
      if (syncFrameId) return
      syncFrameId = window.requestAnimationFrame(syncActiveMessage)
    }

    const measureTrackedAnchors = () => {
      measureFrameId = 0
      const containerRect = el.getBoundingClientRect()
      const currentScrollTop = el.scrollTop
      trackedAnchors = trackedIds
        .map((id) => {
          const node = el.querySelector<HTMLElement>(`[data-msg-id="${id}"]`)
          if (!node) return null
          const rect = node.getBoundingClientRect()
          return {
            id,
            top: rect.top - containerRect.top + currentScrollTop,
          }
        })
        .filter((item): item is { id: number; top: number } => Boolean(item))
        .sort((left, right) => left.top - right.top)
      if (lastReported != null) {
        lastActiveIndex = trackedAnchors.findIndex((item) => item.id === lastReported)
      } else {
        lastActiveIndex = -1
      }
      scheduleSync()
    }

    const scheduleMeasure = () => {
      if (measureFrameId) return
      measureFrameId = window.requestAnimationFrame(measureTrackedAnchors)
    }

    const resizeObserver = typeof ResizeObserver !== 'undefined'
      ? new ResizeObserver(() => {
        scheduleMeasure()
      })
      : null

    el.addEventListener('scroll', scheduleSync, { passive: true })
    window.addEventListener('resize', scheduleMeasure)
    resizeObserver?.observe(el)
    if (el.firstElementChild instanceof HTMLElement) {
      resizeObserver?.observe(el.firstElementChild)
    }
    scheduleMeasure()

    return () => {
      el.removeEventListener('scroll', scheduleSync)
      window.removeEventListener('resize', scheduleMeasure)
      resizeObserver?.disconnect()
      if (syncFrameId) {
        window.cancelAnimationFrame(syncFrameId)
      }
      if (measureFrameId) {
        window.cancelAnimationFrame(measureFrameId)
      }
    }
  }, [messages, onTrackedMessageActive, trackedMessageIds])

  useEffect(() => {
    shelfAsyncScopeEpochRef.current += 1
    const nextStorageKey = shelfStorageKey(shelfScopeId)
    const nextSavedStorageKey = shelfSavedStorageKey(shelfScopeId)
    const legacyKeys = legacyShelfStorageKeys(activeConvId)
    const legacySavedKeys = legacyKeys.map((key) => `${key}${SHELF_SAVED_SUFFIX}`)
    const prevStorageKey = activeStorageKeyRef.current
    flushShelfBackendRef.current?.()
    if (persistShelfTimerRef.current !== null) {
      window.clearTimeout(persistShelfTimerRef.current)
      persistShelfTimerRef.current = null
    }
    if (persistShelfBackendTimerRef.current !== null) {
      window.clearTimeout(persistShelfBackendTimerRef.current)
      persistShelfBackendTimerRef.current = null
    }
    shelfBackendHydratedKeysRef.current.delete(nextStorageKey)
    if (shelfAutoRepairTimerRef.current !== null) {
      window.clearTimeout(shelfAutoRepairTimerRef.current)
      shelfAutoRepairTimerRef.current = null
    }
    if (shelfMetadataHydrateTimerRef.current !== null) {
      window.clearTimeout(shelfMetadataHydrateTimerRef.current)
      shelfMetadataHydrateTimerRef.current = null
    }
    if (shelfSummaryBackfillTimerRef.current !== null) {
      window.clearTimeout(shelfSummaryBackfillTimerRef.current)
      shelfSummaryBackfillTimerRef.current = null
    }
    setShelfAutoRepairingKeySet(new Set())
    setShelfSummaryLoadingKey('')
    setShelfSummaryStatusByKey({})
    setShelfRepairLoadingKey('')
    setShelfRepairImpact(null)
    shelfAutoRepairFingerprintsRef.current = {}
    shelfAutoRepairRetryAfterRef.current = {}
    shelfMetadataHydrateInFlightRef.current = new Set()
    shelfMetadataHydrateAttemptedAtRef.current = {}
    shelfSummaryBackfillInFlightRef.current = new Set()
    shelfSummaryBackfillAttemptedAtRef.current = {}
    if (prevStorageKey !== nextStorageKey) {
      const prevRevision = Number(shelfRevisionByKeyRef.current[prevStorageKey] || 0)
      const latest = latestShelfStateRef.current
      const flushedRevision = persistShelfSnapshot(
        prevStorageKey,
        { open: latest.open, items: latest.items },
        prevRevision,
      )
      shelfRevisionByKeyRef.current[prevStorageKey] = flushedRevision
    }

    // Switching shelf scope changes storage key; skip one persist cycle to avoid
    // writing previous scope state into the new key before hydration.
    skipShelfPersistOnceRef.current = true
    const savedSnapshots = migrateLegacySavedShelfSnapshots(nextSavedStorageKey, legacySavedKeys)
    setSavedShelfSnapshots(savedSnapshots)
    setSelectedSavedSnapshotId((current) => {
      if (current && savedSnapshots.some((item) => item.id === current)) return current
      return savedSnapshots[0]?.id || ''
    })
    const snapshot = migrateLegacyShelfSnapshot(nextStorageKey, legacyKeys)
    if (!snapshot) {
      shelfRevisionByKeyRef.current[nextStorageKey] = 0
      latestShelfStateRef.current = { convId: activeConvId, projectId: shelfScopeId, open: false, items: [] }
      setShelfItems([])
      setShelfOpen(false)
      setFocusedShelfKey('')
      activeStorageKeyRef.current = nextStorageKey
      return
    }
    shelfRevisionByKeyRef.current[nextStorageKey] = Math.max(0, snapshot.revision || 0)
    latestShelfStateRef.current = {
      convId: activeConvId,
      projectId: shelfScopeId,
      open: snapshot.open,
      items: snapshot.items,
    }
    setShelfItems(snapshot.items)
    setShelfOpen(snapshot.open)
    setFocusedShelfKey('')
    activeStorageKeyRef.current = nextStorageKey
  }, [activeConvId, setShelfAutoRepairingKeySet, shelfScopeId])

  useEffect(() => {
    const storageKey = shelfStorageKey(shelfScopeId)
    const requestProjectId = shelfScopeId === '__default__' ? undefined : shelfScopeId
    const requestSeq = shelfBackendHydrateSeqRef.current + 1
    shelfBackendHydrateSeqRef.current = requestSeq
    let cancelled = false
    let requestStartedAt = Date.now()
    const timer = window.setTimeout(() => {
      requestStartedAt = Date.now()
      chatApi.getCitationShelf({ convId: activeConvId || undefined, projectId: requestProjectId, scope: 'project' })
        .then((record) => {
          if (cancelled || shelfBackendHydrateSeqRef.current !== requestSeq) return
          const latest = latestShelfStateRef.current
          if (shelfStorageKey(latest.projectId) !== storageKey) return
          const backendRevision = Math.max(0, Number(record.revision || 0))
          shelfBackendRevisionByKeyRef.current[storageKey] = backendRevision
          shelfBackendHydratedKeysRef.current.add(storageKey)

          const backendItems = restoreShelfItems(Array.isArray(record.items) ? record.items : [])
          const currentItems = dedupeShelfItems(latest.items || []).slice(0, SHELF_MAX_ITEMS)
          const rawBackendUpdatedAt = Number(record.updated_at || 0)
          const backendUpdatedAtMs = rawBackendUpdatedAt > 1000000000000
            ? rawBackendUpdatedAt
            : rawBackendUpdatedAt * 1000
          const localSnapshot = readShelfSnapshot(storageKey)
          const localUpdatedAtMs = Number(localSnapshot?.updatedAt || 0)
          const stateChangedAfterRequest = shelfStateTouchedAtRef.current > requestStartedAt + 10
          const localLooksNewer = localUpdatedAtMs > backendUpdatedAtMs + 500

          let nextItems: CiteShelfItem[]
          let nextOpen = Boolean(record.open)
          let shouldSaveBackend = false

          if (backendRevision <= 0) {
            nextItems = currentItems.length > 0 ? currentItems : backendItems
            nextOpen = latest.open || Boolean(record.open)
            shouldSaveBackend = nextItems.length > 0 || nextOpen
          } else if (backendItems.length <= 0 && currentItems.length > 0) {
            if (stateChangedAfterRequest || localLooksNewer) {
              nextItems = currentItems
              nextOpen = latest.open
              shouldSaveBackend = true
            } else {
              nextItems = []
              nextOpen = Boolean(record.open)
            }
          } else if (currentItems.length <= 0) {
            nextItems = backendItems
            nextOpen = Boolean(record.open)
          } else if (backendUpdatedAtMs > localUpdatedAtMs + 500 && !stateChangedAfterRequest) {
            nextItems = backendItems
            nextOpen = Boolean(record.open)
          } else {
            nextItems = dedupeShelfItems([...currentItems, ...backendItems]).slice(0, SHELF_MAX_ITEMS)
            nextOpen = latest.open || Boolean(record.open)
            shouldSaveBackend = !sameShelfItems(nextItems, backendItems) || nextOpen !== Boolean(record.open)
          }

          latestShelfStateRef.current = {
            convId: latest.convId,
            projectId: latest.projectId,
            open: nextOpen,
            items: nextItems,
          }
          if (!sameShelfItems(currentItems, nextItems)) {
            setShelfItems(nextItems)
            setFocusedShelfKey((current) => (
              current && nextItems.some((item) => item.key === current) ? current : ''
            ))
          }
          if (latest.open !== nextOpen) {
            setShelfOpen(nextOpen)
          }
          if (shouldSaveBackend) {
            saveShelfBackendNow({ convId: latest.convId, projectId: latest.projectId, open: nextOpen, items: nextItems })
          }
        })
        .catch(() => {
          if (cancelled || shelfBackendHydrateSeqRef.current !== requestSeq) return
          shelfBackendHydratedKeysRef.current.delete(storageKey)
        })
    }, 0)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [activeConvId, saveShelfBackendNow, shelfScopeId])

  useEffect(() => {
    const storageKey = shelfStorageKey(shelfScopeId)
    const savedStorageKey = shelfSavedStorageKey(shelfScopeId)
    const onStorage = (event: StorageEvent) => {
      if (event.key === savedStorageKey) {
        if (event.newValue === null) {
          invalidateSavedShelfSnapshotCache(savedStorageKey)
          setSavedShelfSnapshots([])
          setSelectedSavedSnapshotId('')
          return
        }
        const snapshots = readSavedShelfSnapshots(savedStorageKey, event.newValue)
        setSavedShelfSnapshots(snapshots)
        setSelectedSavedSnapshotId((current) => {
          if (current && snapshots.some((item) => item.id === current)) return current
          return snapshots[0]?.id || ''
        })
        return
      }
      if (event.key !== storageKey) return
      if (event.newValue === null) {
        invalidateShelfSnapshotCache(storageKey)
        skipShelfPersistOnceRef.current = true
        shelfRevisionByKeyRef.current[storageKey] = 0
        latestShelfStateRef.current = { convId: activeConvId, projectId: shelfScopeId, open: false, items: [] }
        setShelfItems([])
        setShelfOpen(false)
        setFocusedShelfKey('')
        return
      }
      const snapshot = readShelfSnapshot(storageKey, event.newValue)
      if (!snapshot) return
      const currentRevision = Number(shelfRevisionByKeyRef.current[storageKey] || 0)
      if (snapshot.revision <= currentRevision) return
      skipShelfPersistOnceRef.current = true
      shelfRevisionByKeyRef.current[storageKey] = snapshot.revision
      latestShelfStateRef.current = {
        convId: activeConvId,
        projectId: shelfScopeId,
        open: snapshot.open,
        items: snapshot.items,
      }
      setShelfItems(snapshot.items)
      setShelfOpen(snapshot.open)
      setFocusedShelfKey('')
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [activeConvId, shelfScopeId])

  useLayoutEffect(() => {
    shelfStateTouchedAtRef.current = Date.now()
    latestShelfStateRef.current = { convId: activeConvId, projectId: shelfScopeId, open: shelfOpen, items: shelfItems }
  }, [activeConvId, shelfItems, shelfOpen, shelfScopeId])

  useEffect(() => {
    flushShelfSnapshotRef.current = () => {
      if (persistShelfTimerRef.current !== null) {
        window.clearTimeout(persistShelfTimerRef.current)
        persistShelfTimerRef.current = null
      }
      const latest = latestShelfStateRef.current
      const storageKey = shelfStorageKey(latest.projectId)
      const currentRevision = Number(shelfRevisionByKeyRef.current[storageKey] || 0)
      const nextRevision = persistShelfSnapshot(
        storageKey,
        { open: latest.open, items: latest.items },
        currentRevision,
      )
      shelfRevisionByKeyRef.current[storageKey] = nextRevision
      activeStorageKeyRef.current = storageKey
    }
    flushShelfBackendRef.current = () => {
      if (persistShelfBackendTimerRef.current !== null) {
        window.clearTimeout(persistShelfBackendTimerRef.current)
        persistShelfBackendTimerRef.current = null
      }
      saveShelfBackendNow(latestShelfStateRef.current)
    }
    return () => {
      if (flushShelfSnapshotRef.current) {
        flushShelfSnapshotRef.current()
      }
      if (flushShelfBackendRef.current) {
        flushShelfBackendRef.current()
      }
      flushShelfSnapshotRef.current = null
      flushShelfBackendRef.current = null
    }
  }, [saveShelfBackendNow])

  useEffect(() => {
    setSelectedSavedSnapshotId((current) => {
      if (current && savedShelfSnapshots.some((item) => item.id === current)) return current
      return savedShelfSnapshots[0]?.id || ''
    })
  }, [savedShelfSnapshots])

  useEffect(() => {
    return () => {
      if (persistShelfTimerRef.current !== null) {
        window.clearTimeout(persistShelfTimerRef.current)
        persistShelfTimerRef.current = null
      }
      if (persistShelfBackendTimerRef.current !== null) {
        window.clearTimeout(persistShelfBackendTimerRef.current)
        persistShelfBackendTimerRef.current = null
      }
    }
  }, [])

  useEffect(() => {
    const flushShelfSnapshot = () => {
      flushShelfSnapshotRef.current?.()
      flushShelfBackendRef.current?.()
    }
    window.addEventListener('pagehide', flushShelfSnapshot)
    window.addEventListener('beforeunload', flushShelfSnapshot)
    return () => {
      window.removeEventListener('pagehide', flushShelfSnapshot)
      window.removeEventListener('beforeunload', flushShelfSnapshot)
    }
  }, [])

  useEffect(() => {
    if (skipShelfPersistOnceRef.current) {
      skipShelfPersistOnceRef.current = false
      return
    }
    const storageKey = shelfStorageKey(shelfScopeId)
    if (persistShelfTimerRef.current !== null) {
      window.clearTimeout(persistShelfTimerRef.current)
      persistShelfTimerRef.current = null
    }
    persistShelfTimerRef.current = window.setTimeout(() => {
      const latest = latestShelfStateRef.current
      const latestStorageKey = shelfStorageKey(latest.projectId)
      if (latestStorageKey !== storageKey) {
        persistShelfTimerRef.current = null
        return
      }
      const currentRevision = Number(shelfRevisionByKeyRef.current[storageKey] || 0)
      const nextRevision = persistShelfSnapshot(
        storageKey,
        { open: latest.open, items: latest.items },
        currentRevision,
      )
      shelfRevisionByKeyRef.current[storageKey] = nextRevision
      persistShelfTimerRef.current = null
    }, 80)
    return () => {
      if (persistShelfTimerRef.current !== null) {
        window.clearTimeout(persistShelfTimerRef.current)
        persistShelfTimerRef.current = null
      }
    }
  }, [shelfItems, shelfOpen, shelfScopeId])

  useEffect(() => {
    const storageKey = shelfStorageKey(shelfScopeId)
    if (!shelfBackendHydratedKeysRef.current.has(storageKey)) return
    if (persistShelfBackendTimerRef.current !== null) {
      window.clearTimeout(persistShelfBackendTimerRef.current)
      persistShelfBackendTimerRef.current = null
    }
    persistShelfBackendTimerRef.current = window.setTimeout(() => {
      const latest = latestShelfStateRef.current
      const latestStorageKey = shelfStorageKey(latest.projectId)
      if (latestStorageKey !== storageKey) {
        persistShelfBackendTimerRef.current = null
        return
      }
      saveShelfBackendNow(latest)
      persistShelfBackendTimerRef.current = null
    }, SHELF_BACKEND_PERSIST_MS)
    return () => {
      if (persistShelfBackendTimerRef.current !== null) {
        window.clearTimeout(persistShelfBackendTimerRef.current)
        persistShelfBackendTimerRef.current = null
      }
    }
  }, [saveShelfBackendNow, shelfItems, shelfOpen, shelfScopeId])

  const rows = useMemo<MessageRow[]>(() => buildMessageRows(messages, refs, {
    activeSourcePath: paperGuideSourcePath,
    activeSourceName: paperGuideSourceName,
  }), [messages, paperGuideSourceName, paperGuideSourcePath, refs])

  const assistantTraceByMsgId = useMemo(() => buildAssistantTraceByMsgId(messages), [messages])

  const selectedResearchContextByAssistantId = useMemo(
    () => buildSelectedResearchContextByAssistantId(messages),
    [messages],
  )

  const liveCiteMap = useMemo(
    () => buildLiveCiteMap(messages, activeConvId, assistantTraceByMsgId),
    [activeConvId, assistantTraceByMsgId, messages],
  )

  useEffect(() => {
    const candidates = Array.from(liveCiteMap.values())
      .filter(shouldRequestCitationCardPolish)
      .slice(0, 18)
    for (const item of candidates) {
      const itemKey = toShelfItem(item).key
      const warmKey = `${itemKey}|${item.citationCardPolishKey || ''}|v3`
      if (citationPolishPrewarmKeysRef.current.has(warmKey)) continue
      citationPolishPrewarmKeysRef.current.add(warmKey)
      referencesApi.citationCardPolishCached(item as unknown as Record<string, unknown>, 0.25)
        .catch(() => {
          citationPolishPrewarmKeysRef.current.delete(warmKey)
        })
    }
  }, [liveCiteMap])

  useEffect(() => {
    setShelfItems((current) => {
      let changed = false
      const next = current.map((item) => {
        const live = liveCiteMap.get(item.key)
        if (!live) return item
        const merged = mergeShelfItemWithLive(item, live)
        if (!sameShelfItem(merged, item)) {
          changed = true
          return merged
        }
        return item
      })
      const deduped = dedupeShelfItems(next)
      if (deduped.length !== current.length) changed = true
      return changed ? deduped : current
    })
  }, [liveCiteMap])

  const fetchShelfSummaryForItem = (item: CiteShelfItem, options?: { force?: boolean }) => {
    const summaryLine = String(item.summaryLine || '').trim()
    const lowValueSummary = Boolean(summaryLine && looksLowValueShelfSummary(summaryLine))
    if (!options?.force && shelfItemHasDisplayableArticleSummary(item)) return
    const itemIdentity = shelfPaperIdentity(item)
    const scopeToken = captureShelfAsyncScope()
    const requestItem = (lowValueSummary || options?.force)
      ? {
        ...item,
        summaryLine: '',
        summarySource: '',
        summaryProvider: '',
        summaryQuality: null,
        summary_line: '',
        summary_source: '',
        summary_provider: '',
        summary_quality: null,
      }
      : item
    setShelfSummaryLoadingKey(item.key)
    const loadBibliometrics = options?.force
      ? referencesApi.bibliometrics
      : referencesApi.bibliometricsCached
    shelfSummaryBackfillAttemptedAtRef.current[shelfSummaryBackfillAttemptKey(item)] = Date.now()
    setShelfSummaryStatusByKey((current) => ({ ...current, [item.key]: 'loading' }))
    loadBibliometrics(withBibliometricsLocale(requestItem as unknown as Record<string, unknown>))
      .then((meta) => {
        const currentItem = currentShelfItemForAsync(scopeToken, item.key)
        if (!currentItem) return
        if (!meta || Object.keys(meta).length === 0) {
          setShelfSummaryStatusByKey((current) => ({ ...current, [item.key]: 'unavailable' }))
          return
        }
        const currentMerged = mergeCiteMeta(currentItem, meta)
        const currentPatch = articleSummaryPatchFromMeta(currentItem, meta)
        const summaryReady = shelfItemHasDisplayableArticleSummary({
          ...toShelfItem(currentMerged),
          ...currentPatch,
          key: currentItem.key,
        })
        setShelfItems((current) => current.map((entry) => {
          if (entry.key !== item.key && shelfPaperIdentity(entry) !== itemIdentity) return entry
          const merged = mergeCiteMeta(entry, meta)
          const articleSummaryPatch = articleSummaryPatchFromMeta(entry, meta)
          const next = {
            ...toShelfItem(merged),
            ...articleSummaryPatch,
            key: entry.key,
            tags: normalizeShelfTags(entry.tags),
            note: normalizeShelfNote(entry.note),
          }
          return next
        }))
        setShelfSummaryStatusByKey((current) => ({
          ...current,
          [item.key]: summaryReady
            ? 'ready'
            : bibliometricsSummaryFetchFailed(meta)
              ? 'failed'
              : 'unavailable',
        }))
      })
      .catch(() => {
        if (!shelfAsyncScopeIsCurrent(scopeToken)) return
        setShelfSummaryStatusByKey((current) => ({ ...current, [item.key]: 'failed' }))
      })
      .finally(() => {
        if (!shelfAsyncScopeIsCurrent(scopeToken)) return
        setShelfSummaryLoadingKey((current) => (current === item.key ? '' : current))
      })
  }

  useEffect(() => {
    if (shelfSummaryBackfillTimerRef.current !== null) {
      window.clearTimeout(shelfSummaryBackfillTimerRef.current)
      shelfSummaryBackfillTimerRef.current = null
    }
    if (!shelfOpen || shelfItems.length <= 0) return
    shelfSummaryBackfillTimerRef.current = window.setTimeout(() => {
      shelfSummaryBackfillTimerRef.current = null
      const now = Date.now()
      const targets: Array<{ item: CiteShelfItem; attemptKey: string }> = []
      for (const item of shelfItems) {
        if (targets.length >= SHELF_SUMMARY_BACKFILL_BATCH_SIZE) break
        if (shelfSummaryBackfillInFlightRef.current.has(item.key)) continue
        if (shelfSummaryStatusByKey[item.key] === 'loading') continue
        if (!shelfItemNeedsSummaryBackfill(item)) continue
        const attemptKey = shelfSummaryBackfillAttemptKey(item)
        const lastAttempt = Number(shelfSummaryBackfillAttemptedAtRef.current[attemptKey] || 0)
        if (lastAttempt > 0 && now - lastAttempt < SHELF_SUMMARY_BACKFILL_RETRY_MS) continue
        targets.push({ item, attemptKey })
      }
      if (targets.length <= 0) return

      const inFlight = new Set(shelfSummaryBackfillInFlightRef.current)
      for (const target of targets) {
        inFlight.add(target.item.key)
        shelfSummaryBackfillAttemptedAtRef.current[target.attemptKey] = now
      }
      shelfSummaryBackfillInFlightRef.current = inFlight
      setShelfSummaryLoadingKey((current) => current || targets[0]?.item.key || '')
      setShelfSummaryStatusByKey((current) => {
        const next = { ...current }
        for (const target of targets) next[target.item.key] = 'loading'
        return next
      })
      const scopeToken = captureShelfAsyncScope()

      void Promise.all(targets.map(({ item }) => (
        referencesApi.bibliometrics(withBibliometricsLocale({
          ...item,
          summaryLine: '',
          summarySource: '',
          summaryProvider: '',
          summaryQuality: null,
          summary_line: '',
          summary_source: '',
          summary_provider: '',
          summary_quality: null,
        } as unknown as Record<string, unknown>))
          .then((meta) => ({ key: item.key, meta, failed: false }))
          .catch(() => ({ key: item.key, meta: {} as Record<string, unknown>, failed: true }))
      ))).then((results) => {
        if (!shelfAsyncScopeIsCurrent(scopeToken)) return
        const targetByKey = new Map(targets.map((target) => [target.item.key, target.item]))
        const statusPatch: Record<string, 'unavailable' | 'failed' | 'ready'> = {}
        for (const result of results) {
          if (result.failed) {
            statusPatch[result.key] = 'failed'
            continue
          }
          const target = targetByKey.get(result.key)
          if (!target || !result.meta || Object.keys(result.meta).length <= 0) {
            statusPatch[result.key] = 'unavailable'
            continue
          }
          const merged = mergeCiteMeta(target, result.meta)
          const articleSummaryPatch = articleSummaryPatchFromMeta(target, result.meta)
          const candidate = {
            ...toShelfItem(merged),
            ...articleSummaryPatch,
            key: target.key,
          }
          statusPatch[result.key] = shelfItemHasDisplayableArticleSummary(candidate)
            ? 'ready'
            : bibliometricsSummaryFetchFailed(result.meta)
              ? 'failed'
              : 'unavailable'
        }
        setShelfSummaryStatusByKey((current) => ({ ...current, ...statusPatch }))
        const usable = results.filter((entry) => !entry.failed && entry.meta && Object.keys(entry.meta).length > 0)
        setShelfItems((current) => current.map((entry) => {
          if (!currentShelfItemForAsync(scopeToken, entry.key)) return entry
          const result = usable.find((item) => item.key === entry.key)
          if (!result) return entry
          const merged = mergeCiteMeta(entry, result.meta)
          const articleSummaryPatch = articleSummaryPatchFromMeta(entry, result.meta)
          return {
            ...toShelfItem(merged),
            ...articleSummaryPatch,
            key: entry.key,
            tags: normalizeShelfTags(entry.tags),
            note: normalizeShelfNote(entry.note),
          }
        }))
      }).finally(() => {
        if (!shelfAsyncScopeIsCurrent(scopeToken)) return
        const nextInFlight = new Set(shelfSummaryBackfillInFlightRef.current)
        for (const target of targets) nextInFlight.delete(target.item.key)
        shelfSummaryBackfillInFlightRef.current = nextInFlight
        setShelfSummaryLoadingKey((current) => (
          targets.some((target) => target.item.key === current) ? '' : current
        ))
      })
    }, 220)
    return () => {
      if (shelfSummaryBackfillTimerRef.current !== null) {
        window.clearTimeout(shelfSummaryBackfillTimerRef.current)
        shelfSummaryBackfillTimerRef.current = null
      }
    }
  }, [captureShelfAsyncScope, currentShelfItemForAsync, shelfAsyncScopeIsCurrent, shelfItems, shelfOpen, shelfSummaryStatusByKey])

  const applyShelfMetadataRepairCandidates = useCallback((
    updates: Array<{ key: string; metas: Array<Record<string, unknown>> }>,
  ): boolean => {
    if (updates.length <= 0) return false
    const byKey = new Map<string, Array<Record<string, unknown>>>()
    for (const update of updates) {
      const key = String(update.key || '').trim()
      const metas = (update.metas || []).filter((meta) => meta && Object.keys(meta).length > 0)
      if (!key || metas.length <= 0) continue
      byKey.set(key, [...(byKey.get(key) || []), ...metas])
    }
    if (byKey.size <= 0) return false
    let didUpdate = false
    setShelfItems((current) => current.map((entry) => {
      const candidates = byKey.get(entry.key)
      if (!candidates || candidates.length <= 0) return entry
      for (const meta of candidates) {
        const accepted = strictRepairMerge(entry, meta)
        if (!accepted) continue
        if (!sameShelfItem(accepted, entry)) {
          didUpdate = true
          return accepted
        }
        return entry
      }
      return entry
    }))
    return didUpdate
  }, [])

  const repairShelfItemMeta = (item: CiteShelfItem, options?: { silent?: boolean }) => {
    if (shelfRepairLoadingKey === item.key) return
    const silent = Boolean(options?.silent)
    const scopeToken = captureShelfAsyncScope()
    const requestedFingerprint = shelfItemRepairFingerprint(item)
    setShelfRepairLoadingKey(item.key)
    const payloads = shelfRepairPayloads(item)
    const loadRepairCandidates = referencesApi.repairShelfMetadata(payloads, payloads.length)
      .then((res) => {
        if (!currentShelfItemForAsync(scopeToken, item.key, requestedFingerprint)) return []
        setShelfRepairImpact(res.impact || null)
        const repaired = Array.isArray(res.items) ? res.items : []
        return repaired
          .map(shelfRepairMetaFromEntry)
          .filter((meta) => meta && Object.keys(meta).length > 0)
      })
      .catch(() => {
        if (!currentShelfItemForAsync(scopeToken, item.key, requestedFingerprint)) return []
        return Promise.all([
          ...payloads.map((payload) => referencesApi.bibliometrics(withBibliometricsLocale(payload)).catch(() => ({}))),
        ])
      })

    loadRepairCandidates
      .then((metas) => {
        if (!currentShelfItemForAsync(scopeToken, item.key, requestedFingerprint)) return
        const candidates = metas.filter((meta) => meta && Object.keys(meta).length > 0)
        const didUpdate = applyShelfMetadataRepairCandidates([{ key: item.key, metas: candidates }])
        if (!silent) {
          if (didUpdate) message.success('Metadata repaired with strict rules')
          else message.info('Strict match did not pass; original metadata kept')
        }
      })
      .catch(() => {
        if (!currentShelfItemForAsync(scopeToken, item.key, requestedFingerprint)) return
        if (!silent) message.error('Repair failed, please retry.')
      })
      .finally(() => {
        if (!shelfAsyncScopeIsCurrent(scopeToken)) return
        setShelfRepairLoadingKey((current) => (current === item.key ? '' : current))
      })
  }

  const repairShelfItemsMetadataBatch = useCallback(async (targets: CiteShelfItem[]) => {
    const uniqueTargets: CiteShelfItem[] = []
    const seen = new Set<string>()
    for (const item of targets) {
      const key = String(item.key || '').trim()
      if (!key || seen.has(key)) continue
      seen.add(key)
      uniqueTargets.push(item)
      if (uniqueTargets.length >= SHELF_AUTO_REPAIR_BATCH_SIZE) break
    }
    if (uniqueTargets.length <= 0) return

    const scopeToken = captureShelfAsyncScope()
    const inFlight = new Set(shelfAutoRepairingKeySetRef.current)
    for (const item of uniqueTargets) {
      inFlight.add(item.key)
    }
    setShelfAutoRepairingKeySet(inFlight)
    const requestedFingerprints = new Map(uniqueTargets.map((item) => [
      item.key,
      shelfItemRepairFingerprint(item),
    ]))

    try {
      const payloads = uniqueTargets.flatMap(shelfRepairPayloads)
      const res = await referencesApi.repairShelfMetadata(payloads, payloads.length)
      if (shelfAsyncScopeIsCurrent(scopeToken)) {
        setShelfRepairImpact(res.impact || null)
        const metasByKey = new Map<string, Array<Record<string, unknown>>>()
        for (const entry of Array.isArray(res.items) ? res.items : []) {
          const meta = shelfRepairMetaFromEntry(entry)
          if (!meta || Object.keys(meta).length <= 0) continue
          const key = String(entry.key || meta.key || '').trim()
          if (!key) continue
          metasByKey.set(key, [...(metasByKey.get(key) || []), meta])
        }
        const updates = Array.from(metasByKey.entries())
          .filter(([key]) => currentShelfItemForAsync(scopeToken, key, requestedFingerprints.get(key) || ''))
          .map(([key, metas]) => ({ key, metas }))
        applyShelfMetadataRepairCandidates(updates)
        for (const item of uniqueTargets) {
          if (!currentShelfItemForAsync(scopeToken, item.key, requestedFingerprints.get(item.key) || '')) continue
          const fingerprint = requestedFingerprints.get(item.key)
          if (fingerprint) shelfAutoRepairFingerprintsRef.current[item.key] = fingerprint
          delete shelfAutoRepairRetryAfterRef.current[item.key]
        }
      }
    } catch {
      if (shelfAsyncScopeIsCurrent(scopeToken)) {
        const retryAt = Date.now() + SHELF_AUTO_REPAIR_RETRY_MS
        for (const item of uniqueTargets) {
          shelfAutoRepairRetryAfterRef.current[item.key] = retryAt
        }
      }
    } finally {
      if (shelfAsyncScopeIsCurrent(scopeToken)) {
        const nextInFlight = new Set(shelfAutoRepairingKeySetRef.current)
        for (const item of uniqueTargets) {
          nextInFlight.delete(item.key)
        }
        setShelfAutoRepairingKeySet(nextInFlight)
      }
    }
  }, [applyShelfMetadataRepairCandidates, captureShelfAsyncScope, currentShelfItemForAsync, setShelfAutoRepairingKeySet, shelfAsyncScopeIsCurrent])

  useEffect(() => {
    if (shelfMetadataHydrateTimerRef.current !== null) {
      window.clearTimeout(shelfMetadataHydrateTimerRef.current)
      shelfMetadataHydrateTimerRef.current = null
    }
    if (!shelfOpen || shelfItems.length <= 0) return
    shelfMetadataHydrateTimerRef.current = window.setTimeout(() => {
      shelfMetadataHydrateTimerRef.current = null
      const now = Date.now()
      const targets: Array<{ item: CiteShelfItem; attemptKey: string }> = []
      for (const item of shelfItems) {
        if (targets.length >= SHELF_METADATA_HYDRATE_BATCH_SIZE) break
        if (item.key === shelfRepairLoadingKey) continue
        if (shelfAutoRepairingKeySetRef.current.has(item.key)) continue
        if (shelfMetadataHydrateInFlightRef.current.has(item.key)) continue
        if (!shelfItemNeedsPersistedMetadataHydrate(item)) continue
        const attemptKey = shelfMetadataHydrateAttemptKey(item)
        const lastAttempt = Number(shelfMetadataHydrateAttemptedAtRef.current[attemptKey] || 0)
        if (lastAttempt > 0 && now - lastAttempt < SHELF_METADATA_HYDRATE_RETRY_MS) continue
        targets.push({ item, attemptKey })
      }
      if (targets.length <= 0) return

      const inFlight = new Set(shelfMetadataHydrateInFlightRef.current)
      for (const target of targets) {
        inFlight.add(target.item.key)
        shelfMetadataHydrateAttemptedAtRef.current[target.attemptKey] = now
      }
      shelfMetadataHydrateInFlightRef.current = inFlight
      const scopeToken = captureShelfAsyncScope()
      const requestedFingerprints = new Map(targets.map((target) => [
        target.item.key,
        shelfItemRepairFingerprint(target.item),
      ]))
      void Promise.all(targets.map(({ item }) => (
        referencesApi.bibliometrics(withBibliometricsLocale(item as unknown as Record<string, unknown>))
          .catch(() => ({}))
          .then((meta) => ({ key: item.key, meta }))
      ))).then((results) => {
        if (!shelfAsyncScopeIsCurrent(scopeToken)) return
        const updates = results
          .filter((entry) => entry.meta && Object.keys(entry.meta).length > 0)
          .filter((entry) => currentShelfItemForAsync(scopeToken, entry.key, requestedFingerprints.get(entry.key) || ''))
          .map((entry) => ({ key: entry.key, metas: [entry.meta] }))
        if (updates.length > 0) {
          applyShelfMetadataRepairCandidates(updates)
        }
      }).finally(() => {
        if (!shelfAsyncScopeIsCurrent(scopeToken)) return
        const nextInFlight = new Set(shelfMetadataHydrateInFlightRef.current)
        for (const target of targets) nextInFlight.delete(target.item.key)
        shelfMetadataHydrateInFlightRef.current = nextInFlight
      })
    }, 160)
    return () => {
      if (shelfMetadataHydrateTimerRef.current !== null) {
        window.clearTimeout(shelfMetadataHydrateTimerRef.current)
        shelfMetadataHydrateTimerRef.current = null
      }
    }
  }, [applyShelfMetadataRepairCandidates, captureShelfAsyncScope, currentShelfItemForAsync, shelfAsyncScopeIsCurrent, shelfItems, shelfOpen, shelfRepairLoadingKey])

  useEffect(() => {
    if (shelfAutoRepairTimerRef.current !== null) {
      window.clearTimeout(shelfAutoRepairTimerRef.current)
      shelfAutoRepairTimerRef.current = null
    }
    if (shelfItems.length <= 0) return
    shelfAutoRepairTimerRef.current = window.setTimeout(() => {
      shelfAutoRepairTimerRef.current = null
      const now = Date.now()
      const inFlight = shelfAutoRepairingKeySetRef.current
      const targets: CiteShelfItem[] = []
      for (const item of shelfItems) {
        if (targets.length >= SHELF_AUTO_REPAIR_BATCH_SIZE) break
        if (inFlight.has(item.key) || item.key === shelfRepairLoadingKey) continue
        if (!shelfItemNeedsMetadataRepair(item)) continue
        const fingerprint = shelfItemRepairFingerprint(item)
        if (shelfAutoRepairFingerprintsRef.current[item.key] === fingerprint) continue
        if ((shelfAutoRepairRetryAfterRef.current[item.key] || 0) > now) continue
        targets.push(item)
      }
      if (targets.length > 0) {
        void repairShelfItemsMetadataBatch(targets)
      }
    }, 250)
    return () => {
      if (shelfAutoRepairTimerRef.current !== null) {
        window.clearTimeout(shelfAutoRepairTimerRef.current)
        shelfAutoRepairTimerRef.current = null
      }
    }
  }, [repairShelfItemsMetadataBatch, shelfItems, shelfRepairLoadingKey])

  const mergeCitationMetaForItemKey = (itemKey: string, metas: Array<Record<string, unknown>>) => {
    const usable = mergePopoverDetailForItemKey(itemKey, metas)
    if (!usable.length) return
    setShelfItems((current) => current.map((item) => {
      if (item.key !== itemKey) return item
      let merged: CiteDetail = item
      for (const meta of usable) {
        merged = mergeCiteMeta(merged, meta)
      }
      return {
        ...toShelfItem(merged),
        tags: normalizeShelfTags(item.tags),
        note: normalizeShelfNote(item.note),
      }
    }))
  }

  const showCitationAt = (detail: CiteDetail, position: { x: number; y: number }, pinned: boolean) => {
    citationPreview.clearTimers()
    if (!pinned && popoverPinned) return
    const itemKey = toShelfItem(detail).key
    const metadataPlan = buildCitationPopoverMetadataPlan(detail, itemKey)
    openCitationPopoverState(detail, position, { pinned, requestKey: itemKey })
    if (shouldRequestCitationCardPolish(detail)) {
      citationPreview.requestPolish({
        activeRequestKeyRef: activePopoverRequestKeyRef,
        detail,
        itemKey,
        onMeta: mergeCitationMetaForItemKey,
      })
    }
    if (metadataPlan.requestCount <= 0) {
      setPopoverLoading(false)
      return
    }

    setPopoverLoading(true)
    loadCitationPopoverMetadata(detail, { plan: metadataPlan })
      .then(({ metas }) => {
        mergeCitationMetaForItemKey(itemKey, metas)
      })
      .finally(() => {
        if (activePopoverRequestKeyRef.current === itemKey) {
          setPopoverLoading(false)
        }
      })
  }

  const openCitation = (detail: CiteDetail, event: MouseEvent<HTMLElement>) => {
    showCitationAt(detail, { x: event.clientX, y: event.clientY }, true)
  }

  const previewCitation = (detail: CiteDetail, event: MouseEvent<HTMLElement>) => {
    if (popoverPinned) return
    const position = { x: event.clientX, y: event.clientY }
    citationPreview.schedulePreviewOpen(() => {
      showCitationAt(detail, position, false)
    })
  }

  const scheduleCitationPreviewClose = () => {
    if (popoverPinned) return
    citationPreview.schedulePreviewClose(closeCitationPopoverState)
  }

  const keepCitationPreviewOpen = () => {
    citationPreview.keepPreviewOpen()
  }

  const closeCitationPopover = () => {
    citationPreview.clearTimers()
    closeCitationPopoverState()
  }

  const closeEvidenceDrawer = () => {
    setEvidenceDrawerSource(null)
    setEvidenceDrawerCiteDetails([])
  }

  const openEvidenceDrawer = (sourceNotice: AnswerSourceNoticeViewModel, details: CiteDetail[]) => {
    closeCitationPopover()
    setEvidenceDrawerSource(sourceNotice)
    setEvidenceDrawerCiteDetails(details)
  }

  const openCitationShelfFromPopover = () => {
    setShelfOpen(true)
    closeCitationPopover()
  }

  const addToShelf = (detail: CiteDetail) => {
    const currentItems = latestShelfStateRef.current.items
    const { nextItems, focusKey, summaryTarget } = mergeCitationDetailIntoShelfItems(currentItems, detail)
    setShelfItems(nextItems)
    setFocusedShelfKey(focusKey)
    setShelfOpen(true)
    persistShelfLocalNow(nextItems, true)
    window.setTimeout(() => {
      fetchShelfSummaryForItem(summaryTarget)
    }, 160)
  }

  const addReaderCitationToShelf = (rawPayload: unknown) => {
    const payload = normalizeReaderCitationShelfPayload(rawPayload)
    if (!payload) return
    const payloadProjectId = String(payload.projectId || '').trim()
    if (payloadProjectId && shelfProjectScopeId(payloadProjectId) !== shelfScopeId) return
    const detail = normalizeCiteDetail(payload.detail)
    if (!detail) return
    addToShelf(detail)
  }
  const addReaderCitationToShelfRef = useRef(addReaderCitationToShelf)
  addReaderCitationToShelfRef.current = addReaderCitationToShelf

  useEffect(() => {
    const handleWindowEvent = (event: Event) => {
      const custom = event as CustomEvent<unknown>
      addReaderCitationToShelfRef.current(custom.detail)
    }
    window.addEventListener(READER_CITATION_SHELF_EVENT, handleWindowEvent)

    let channel: BroadcastChannel | null = null
    if (typeof BroadcastChannel !== 'undefined') {
      channel = new BroadcastChannel(READER_CITATION_SHELF_CHANNEL)
      channel.onmessage = (event) => {
        const data = event?.data && typeof event.data === 'object'
          ? event.data as Record<string, unknown>
          : {}
        if (String(data.type || '') !== 'reader-citation-shelf') return
        addReaderCitationToShelfRef.current(data)
        const requestId = String(data.requestId || '').trim()
        if (requestId) {
          channel?.postMessage({ type: 'reader-citation-shelf-ack', requestId })
        }
      }
    }
    return () => {
      window.removeEventListener(READER_CITATION_SHELF_EVENT, handleWindowEvent)
      channel?.close()
    }
  }, [])

  const addReaderSelectionToShelf = (rawPayload: unknown) => {
    const payload = normalizeReaderSelectionShelfPayload(rawPayload)
    if (!payload) return
    const payloadProjectId = String(payload.projectId || '').trim()
    if (payloadProjectId && shelfProjectScopeId(payloadProjectId) !== shelfScopeId) return
    const detail = citeDetailFromReaderSelection(payload, payload.conversationId || activeConvId)
    if (!detail) return
    const note = readerSelectionNote(payload, S)
    const currentItems = latestShelfStateRef.current.items
    const { nextItems, focusKey, summaryTarget } = mergeReaderSelectionDetailIntoShelfItems(currentItems, detail, {
      text: payload.text,
      note,
      headingPath: payload.headingPath,
      blockId: payload.blockId,
      anchorId: payload.anchorId,
      anchorKind: payload.anchorKind,
    })
    setShelfItems(nextItems)
    setFocusedShelfKey(focusKey)
    setShelfOpen(true)
    persistShelfLocalNow(nextItems, true)
    window.setTimeout(() => {
      fetchShelfSummaryForItem(summaryTarget, { force: true })
    }, 420)
  }
  const addReaderSelectionToShelfRef = useRef(addReaderSelectionToShelf)
  addReaderSelectionToShelfRef.current = addReaderSelectionToShelf

  useEffect(() => {
    const handleWindowEvent = (event: Event) => {
      const custom = event as CustomEvent<unknown>
      addReaderSelectionToShelfRef.current(custom.detail)
    }
    window.addEventListener(READER_SELECTION_SHELF_EVENT, handleWindowEvent)

    let channel: BroadcastChannel | null = null
    if (typeof BroadcastChannel !== 'undefined') {
      channel = new BroadcastChannel(READER_SELECTION_SHELF_CHANNEL)
      channel.onmessage = (event) => {
        const data = event?.data && typeof event.data === 'object'
          ? event.data as Record<string, unknown>
          : {}
        if (String(data.type || '') !== 'reader-selection-shelf') return
        addReaderSelectionToShelfRef.current(data)
        const requestId = String(data.requestId || '').trim()
        if (requestId) {
          channel?.postMessage({ type: 'reader-selection-shelf-ack', requestId })
        }
      }
    }
    return () => {
      window.removeEventListener(READER_SELECTION_SHELF_EVENT, handleWindowEvent)
      channel?.close()
    }
  }, [])

  const startPaperGuideFromDetail = async (detail: CiteDetail) => {
    const isInPaperReference = Boolean(detail.isInpaper)
    const sourcePath = String(detail.sourcePath || '').trim()
    if (!sourcePath) {
      message.info(isInPaperReference ? S.reader_pdf_not_ready : S.reader_missing_path)
      return
    }
    const sourceName = String(detail.sourceName || detail.title || '').trim() || basenameFromSourcePath(sourcePath) || S.default_source_fallback
    setPopoverGuideLoading(true)
    try {
      await createPaperGuideConversation({
        sourcePath,
        sourceName,
        title: `${S.timeline_guide_label} · ${sourceName}`,
      })
      message.success(S.reader_entered_guide)
      citationPreview.clearTimers()
      closeCitationPopoverState()
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.reader_create_guide_failed)
    } finally {
      setPopoverGuideLoading(false)
    }
  }

  const openReaderFromDetail = (detail: CiteDetail) => {
    if (!onOpenReader) return
    const sourcePath = String(detail.sourcePath || '').trim()
    if (!sourcePath) {
      message.info(S.reader_missing_path)
      return
    }
    const payload = buildBasicReaderOpenPayload({
      sourcePath,
      sourceName: String(detail.sourceName || detail.title || '').trim(),
      headingPath: String(detail.headingPath || (!detail.isInpaper ? detail.title : '') || '').trim(),
      snippet: String(detail.evidenceQuote || detail.summaryLine || detail.title || detail.raw || '').trim(),
      highlightSnippet: String(detail.evidenceQuote || detail.summaryLine || detail.raw || '').trim(),
      blockId: String(detail.blockId || '').trim(),
      anchorId: String(detail.anchorId || '').trim(),
      anchorKind: String(detail.anchorKind || '').trim(),
      strictLocate: Boolean(detail.blockId || detail.anchorId),
      locateFeedbackKey: String((detail as CiteShelfItem).key || toShelfItem(detail).key || '').trim(),
    })
    if (!payload) return
    closeCitationPopoverState()
    onOpenReader(payload)
  }

  const openMessageFromShelfItem = (item: CiteShelfItem) => {
    const targetId = Number(item.traceAssistantMsgId || item.traceUserMsgId || 0)
    if (!Number.isFinite(targetId) || targetId <= 0) {
      message.info(S.shelf_message_missing)
      return
    }
    const el = scrollRef.current
    if (!el) return
    const target = el.querySelector<HTMLElement>(`[data-msg-id="${targetId}"]`)
    if (!target) {
      message.info(S.shelf_message_not_loaded)
      return
    }
    const targetRect = target.getBoundingClientRect()
    const containerRect = el.getBoundingClientRect()
    const top = targetRect.top - containerRect.top + el.scrollTop - 12
    el.scrollTo({ top: Math.max(0, top), behavior: 'smooth' })
    setShelfMessageFlashId(targetId)
    if (shelfMessageFlashTimerRef.current !== null) {
      window.clearTimeout(shelfMessageFlashTimerRef.current)
    }
    shelfMessageFlashTimerRef.current = window.setTimeout(() => {
      setShelfMessageFlashId((current) => (current === targetId ? null : current))
      shelfMessageFlashTimerRef.current = null
    }, 1400)
  }

  const selectedSavedSnapshot = useMemo(
    () => savedShelfSnapshots.find((item) => item.id === selectedSavedSnapshotId) || null,
    [savedShelfSnapshots, selectedSavedSnapshotId],
  )

  const selectedSnapshotDiff = useMemo(() => {
    if (!selectedSavedSnapshot) return ''
    const diff = snapshotDiffCounts(shelfItems, selectedSavedSnapshot.items)
    if (diff.added <= 0 && diff.removed <= 0) return S.shelf_snapshot_no_diff
    return S.shelf_snapshot_diff
      .replace('{added}', String(diff.added))
      .replace('{removed}', String(diff.removed))
  }, [S, selectedSavedSnapshot, shelfItems])

  const guideSourcePathSet = useMemo(() => {
    const out = new Set<string>()
    for (const item of guideDocCandidates) {
      const sourcePath = String(item.sourcePath || '').trim()
      for (const key of sourcePathLookupKeys(sourcePath)) {
        out.add(key)
      }
    }
    return out
  }, [guideDocCandidates])

  const guideDocCandidatesBySourcePath = useMemo(() => {
    const out = new Map<string, LocateCandidate[]>()
    for (const item of guideDocCandidates) {
      const sourcePath = String(item.sourcePath || '').trim()
      if (!sourcePath) continue
      for (const key of sourcePathLookupKeys(sourcePath)) {
        const list = out.get(key) || []
        list.push(item)
        out.set(key, list)
      }
    }
    return out
  }, [guideDocCandidates])

  const assistantLocatePrepByMsgId = useMemo(() => {
    const result = buildAssistantLocatePrepByMsgId({
      activeConvId,
      messages,
      refs,
      assistantTraceByMsgId,
      guideDocCandidates,
      guideDocCandidatesBySourcePath,
      guideSourcePathSet,
      paperGuideSourcePath,
      paperGuideSourceName,
      onOpenReaderAvailable: Boolean(onOpenReader),
      previousCache: assistantLocatePrepCacheRef.current,
      S,
    })
    assistantLocatePrepCacheRef.current = result.nextCache
    assistantLocatePrepPerfRef.current = result.perf
    return result.prepByMsgId
  }, [
    activeConvId,
    assistantTraceByMsgId,
    guideDocCandidates,
    guideDocCandidatesBySourcePath,
    guideSourcePathSet,
    messages,
    onOpenReader,
    paperGuideSourceName,
    paperGuideSourcePath,
    refs,
    S,
  ])

  useEffect(() => {
    const perf = assistantLocatePrepPerfRef.current
    if (!perf) return
    pushMessageListPrepPerf(perf)
  }, [activeConvId, assistantLocatePrepByMsgId])

  const saveShelfSnapshot = () => {
    const currentItems = dedupeShelfItems(shelfItems).slice(0, SHELF_MAX_ITEMS)
    if (currentItems.length <= 0) {
      message.info(S.shelf_version_empty || 'Shelf is empty; cannot save local snapshot')
      return
    }
    const now = Date.now()
    const d = new Date(now)
    const pad = (value: number) => String(value).padStart(2, '0')
    const versionTime = `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
    const entry: ShelfSavedSnapshot = {
      id: `s_${now.toString(36)}_${Math.random().toString(36).slice(2, 7)}`,
      name: (S.shelf_version_name || 'Version {time}').replace('{time}', versionTime),
      createdAt: now,
      items: currentItems.map((item) => ({ ...item })),
    }
    setSavedShelfSnapshots((current) => {
      const next = [entry, ...current].slice(0, SHELF_SAVED_MAX_ITEMS)
      persistSavedShelfSnapshots(shelfSavedStorageKey(shelfScopeId), next)
      return next
    })
    setSelectedSavedSnapshotId(entry.id)
    message.success(S.shelf_version_saved || 'Saved to this browser')
  }

  const loadShelfSnapshot = () => {
    if (!selectedSavedSnapshot) return
    const restored = dedupeShelfItems(selectedSavedSnapshot.items).slice(0, SHELF_MAX_ITEMS).map((item) => ({ ...item }))
    setShelfItems(restored)
    setFocusedShelfKey('')
    setShelfSummaryLoadingKey('')
    setShelfRepairLoadingKey('')
    message.success((S.shelf_version_loaded || 'Restored local snapshot: {name}').replace('{name}', selectedSavedSnapshot.name))
  }

  const deleteShelfSnapshot = () => {
    if (!selectedSavedSnapshot) return
    const removedName = selectedSavedSnapshot.name
    setSavedShelfSnapshots((current) => {
      const next = current.filter((item) => item.id !== selectedSavedSnapshot.id)
      persistSavedShelfSnapshots(shelfSavedStorageKey(shelfScopeId), next)
      return next
    })
    setSelectedSavedSnapshotId((current) => (current === selectedSavedSnapshot.id ? '' : current))
    message.success((S.shelf_version_deleted || 'Deleted local snapshot: {name}').replace('{name}', removedName))
  }

  const useSelectedShelfItemsAsContext = useCallback((items: CiteShelfItem[]) => {
    const pack = buildSelectedResearchContextPack(items, {
      conversationId: activeConvId || '',
      guideSourcePath: paperGuideSourcePath || '',
      guideSourceName: paperGuideSourceName || '',
    })
    if (!pack) {
      message.info(S.research_context_empty_toast || 'No usable context in the selected items')
      return
    }
    onResearchContextPackChange?.(pack)
    message.success(
      (S.research_context_selected_toast || 'Added {n} selected items to the next answer context')
      .replace('{n}', String(pack.items.length)),
    )
  }, [S, activeConvId, onResearchContextPackChange, paperGuideSourceName, paperGuideSourcePath])

  const useReceiptItemAsFollowUp = useCallback((sourcePack: SelectedResearchContextPack, item: SelectedResearchContextItem) => {
    if (!onResearchContextFollowUp) return
    const pack = buildSelectedResearchContextPackFromItems([item], {
      conversationId: activeConvId || sourcePack.conversationId || '',
      guideSourcePath: sourcePack.guideSourcePath || paperGuideSourcePath || '',
      guideSourceName: sourcePack.guideSourceName || paperGuideSourceName || '',
    })
    if (!pack) {
      message.info(S.research_context_empty_toast || 'No usable context in this item')
      return
    }
    const title = contextItemTitle(item, S.default_source_fallback || 'Untitled')
    const promptText = (S.research_context_followup_prompt || 'Continue with this selected context: {title}\n\n')
      .replace('{title}', title)
    onResearchContextFollowUp(pack, promptText)
    message.success(S.research_context_followup_toast || 'Ready for a follow-up question')
  }, [S, activeConvId, onResearchContextFollowUp, paperGuideSourceName, paperGuideSourcePath])

  const shelfNode = (
    <CiteShelf
      open={shelfOpen}
      visible={shelfDockMode ? shelfVisible : undefined}
      presentation={shelfDockMode ? 'dock' : 'floating'}
      items={shelfItems}
      activeConvId={activeConvId}
      activeSourcePath={paperGuideSourcePath}
      readerLocateResults={readerLocateResults}
      sourceQualityRefreshToken={sourceQualityRefreshToken}
      focusedKey={focusedShelfKey}
      summaryLoadingKey={shelfSummaryLoadingKey}
      summaryStatusByKey={shelfSummaryStatusByKey}
      repairLoadingKey={shelfRepairLoadingKey}
      repairingKeys={shelfAutoRepairingKeys}
      repairImpact={shelfRepairImpact}
      activeContextKeys={selectedResearchContextKeys}
      snapshots={savedShelfSnapshots}
      selectedSnapshotId={selectedSavedSnapshotId}
      snapshotDiff={selectedSnapshotDiff}
      onToggle={() => setShelfOpen((value) => !value)}
      onSelect={(item) => {
        setFocusedShelfKey(item.key)
        fetchShelfSummaryForItem(item)
      }}
      onRetrySummary={(item) => fetchShelfSummaryForItem(item, { force: true })}
      onOpenSource={(item) => {
        openReaderFromDetail(shelfLibraryFullTextDetail(item) || item as unknown as CiteDetail)
      }}
      onOpenDiscoverySource={(item) => {
        openReaderFromDetail(shelfDiscoverySourceDetail(item) || item as unknown as CiteDetail)
      }}
      onOpenMessage={openMessageFromShelfItem}
      onUseSelectedAsContext={onResearchContextPackChange ? useSelectedShelfItemsAsContext : undefined}
      onRemove={(key) => {
        const willBeEmpty = latestShelfStateRef.current.items.filter((item) => item.key !== key).length <= 0
        if (willBeEmpty) markShelfEmptyBackendSaveIntent(shelfScopeId)
        setShelfItems((current) => current.filter((item) => item.key !== key))
        if (focusedShelfKey === key) setFocusedShelfKey('')
        if (shelfSummaryLoadingKey === key) setShelfSummaryLoadingKey('')
        setShelfSummaryStatusByKey((current) => {
          if (!(key in current)) return current
          const next = { ...current }
          delete next[key]
          return next
        })
        if (shelfRepairLoadingKey === key) setShelfRepairLoadingKey('')
        const nextRepairing = new Set(shelfAutoRepairingKeySetRef.current)
        nextRepairing.delete(key)
        setShelfAutoRepairingKeySet(nextRepairing)
        delete shelfAutoRepairFingerprintsRef.current[key]
        delete shelfAutoRepairRetryAfterRef.current[key]
      }}
      onClear={() => {
        markShelfEmptyBackendSaveIntent(shelfScopeId)
        setShelfItems([])
        setFocusedShelfKey('')
        setShelfSummaryLoadingKey('')
        setShelfSummaryStatusByKey({})
        setShelfRepairLoadingKey('')
        setShelfAutoRepairingKeySet(new Set())
        shelfAutoRepairFingerprintsRef.current = {}
        shelfAutoRepairRetryAfterRef.current = {}
        setShelfRepairImpact(null)
        const projectScopeId = shelfProjectScopeId(shelfScopeId)
        const storageKey = shelfStorageKey(projectScopeId)
        void chatApi.deleteCitationShelf({
          convId: activeConvId || undefined,
          projectId: projectScopeId === '__default__' ? undefined : projectScopeId,
          scope: 'project',
        })
          .then((record) => {
            shelfBackendRevisionByKeyRef.current[storageKey] = Math.max(0, Number(record.revision || 0))
            shelfBackendHydratedKeysRef.current.add(storageKey)
            delete shelfEmptyBackendSaveIntentRef.current[storageKey]
          })
          .catch(() => {
            // Local state remains cleared; the guarded save path will avoid accidental backend overwrite.
          })
      }}
      onUpdateTags={(key, tags) => {
        const nextTags = normalizeShelfTags(tags)
        setShelfItems((current) => current.map((item) => (
          item.key === key ? { ...item, tags: nextTags } : item
        )))
      }}
      onUpdateNote={(key, note) => {
        const nextNote = normalizeShelfNote(note)
        setShelfItems((current) => current.map((item) => (
          item.key === key ? { ...item, note: nextNote } : item
        )))
      }}
      onRepair={(item, options) => {
        repairShelfItemMeta(item, options)
      }}
      onApplyRepairCandidates={applyShelfMetadataRepairCandidates}
      onSelectSnapshot={setSelectedSavedSnapshotId}
      onSaveSnapshot={saveShelfSnapshot}
      onLoadSnapshot={loadShelfSnapshot}
      onDeleteSnapshot={deleteShelfSnapshot}
      onBackgroundActivityChange={setShelfBackgroundBusy}
    />
  )
  const renderedShelfNode = shelfDockMode
    ? (shelfPortalTarget ? createPortal(shelfNode, shelfPortalTarget) : null)
    : shelfNode
  const cleanGenerationPartial = generationPartial !== undefined && generationPartial !== null
    ? cleanAssistantAnswerPresentationText(generationPartial)
    : ''
  const generationSourceNotice = splitLeadingAssistantSourceNotice(cleanGenerationPartial)
  const visibleGenerationPartial = generationSourceNotice.notice ? generationSourceNotice.body : cleanGenerationPartial
  const generationContractSourceSummary = sourceSummaryFromAnswerContract(generationAnswerContract)
  const effectiveGenerationSourceSummary = generationContractSourceSummary || generationAgentSourceSummary
  const hasGenerationAnswerContract = Boolean(
    generationAnswerContract && Object.keys(generationAnswerContract).length > 0,
  )
  const hasGenerationAgentSourceSummary = Boolean(
    hasGenerationAnswerContract
      || (effectiveGenerationSourceSummary && Object.keys(effectiveGenerationSourceSummary).length > 0),
  )

  return (
    <>
      <div ref={scrollRef} className="kb-message-scroll kb-main-scroll">
        <div className="kb-message-stack">
          {rows.map((row, index) => {
            if (row.kind === 'refs') {
              return (
                <div key={`refs-${row.userMsgId}-${index}`} className="kb-message-row kb-message-row-refs">
                  <div className="kb-msg-avatar-spacer" />
                  <div className="kb-message-refs-wrap">
                    <RefsPanel
                      refs={refs}
                      msgId={row.userMsgId}
                      onOpenReader={onOpenReader}
                      activeSourcePath={paperGuideSourcePath}
                      activeSourceName={paperGuideSourceName}
                    />
                  </div>
                </div>
              )
            }

            const message = row.message
            const isUser = message.role === 'user'
            const trace = assistantTraceByMsgId.get(message.id)
            const agentTrace = !isUser ? getMessageAgentTrace(message) : null
            const canLoadAgentTrace = !isUser ? messageHasAgentTraceHint(message) : false
            const researchTrace = !isUser ? getMessageResearchTrace(message) : null
            const selectedResearchContextPack = !isUser
              ? selectedResearchContextByAssistantId.get(Number(message.id)) || null
              : null
            const renderPacket = !isUser ? getMessageRenderPacket(message) : null
            const citeDetails = getMessageCiteDetailRecords(message)
              .map(normalizeCiteDetail)
              .filter((detail): detail is CiteDetail => Boolean(detail))
              .map((detail) => ({
                ...detail,
                traceConvId: String(activeConvId || ''),
                traceAssistantMsgId: message.id,
                traceAssistantOrder: Number(trace?.answerOrder || 0),
                traceUserMsgId: Number(trace?.userMsgId || 0),
              }))
            const imageAttachments = imageAttachmentsOf(message)
            const showUserText = !(isUser && imageAttachments.length > 0 && isImageOnlyPlaceholder(message.content))
            const isImageOnlyUserMessage = isUser && imageAttachments.length > 0 && !showUserText
            const prep = !isUser ? assistantLocatePrepByMsgId.get(message.id) : undefined
            const rawBodyContent = prep?.bodyContent || getMessageRenderedBodyContent(message)
            const lowConfidenceMeta = !isUser
              ? resolveLowConfidenceMeta(
                (message.meta && typeof message.meta === 'object')
                  ? message.meta as Record<string, unknown>
                  : null,
                String(rawBodyContent || ''),
                S,
              )
              : null
            const bodyContent = lowConfidenceMeta
              ? stripLeadingLowConfidenceNotice(rawBodyContent)
              : rawBodyContent
            const retryPrompt = !isUser && onRetryMessage && isGenerationFailureAnswer(message.content)
              ? generationRetryPrompt(messages, message.id, trace?.userMsgId)
              : ''
            const refsUserMsgIdForCitations = Number(prep?.refsUserMsgId || message.refs_user_msg_id || trace?.userMsgId || 0)
            const refEntryForCitations = refsUserMsgIdForCitations > 0
              ? refs[String(refsUserMsgIdForCitations)] as RefEntryLite | undefined
              : undefined
            const fallbackCiteDetails = (!isUser && citeDetails.length <= 0 && Array.isArray(refEntryForCitations?.hits))
              ? buildFallbackCiteDetailsFromRefHits({
                bodyContent: String(bodyContent || ''),
                refHits: refEntryForCitations?.hits || [],
                messageId: message.id,
                traceConvId: String(activeConvId || ''),
                traceAssistantOrder: Number(trace?.answerOrder || 0),
                traceUserMsgId: Number(trace?.userMsgId || refsUserMsgIdForCitations || 0),
                S,
              })
              : []
            const effectiveCiteDetails = enrichCiteDetailsWithVisibleRefContext(
              citeDetails.length > 0 ? citeDetails : fallbackCiteDetails,
              refEntryForCitations,
            )
            const unlinkedReferenceViews = !isUser
              ? buildUnlinkedReferenceViews({
                packet: renderPacket,
                linkedDetails: effectiveCiteDetails,
                messageId: message.id,
                traceConvId: String(activeConvId || ''),
                traceAssistantOrder: Number(trace?.answerOrder || 0),
                traceUserMsgId: Number(trace?.userMsgId || refsUserMsgIdForCitations || 0),
                S,
              })
              : []
            const guideSourcePath = String(paperGuideSourcePath || '').trim()
            const locateSourceName = prep?.locateSourceName || String(paperGuideSourceName || '').trim()
            const messageProvenance = prep?.messageProvenance || (
              message.provenance && typeof message.provenance === 'object'
                ? message.provenance as Record<string, unknown>
                : null
            )
            const provenanceSourcePath = prep?.provenanceSourcePath || ''
            const provenanceSourceName = prep?.provenanceSourceName || ''
            const provenanceBlockMap = prep?.provenanceBlockMap || {} as Record<string, Record<string, unknown>>
            const provenanceDirectSegments = prep?.provenanceDirectSegments || []
            const hasDirectProvenance = prep?.hasDirectProvenance || false
            const hasStructuredProvenance = prep?.hasStructuredProvenance || false
            const effectiveGuideSourcePath = prep?.effectiveGuideSourcePath || guideSourcePath
            const strictProvenanceLocate = prep?.strictProvenanceLocate || false
            const provenanceLocateEntries = prep?.provenanceLocateEntries || []
            const structuredProvenanceSegmentsAll = prep?.structuredProvenanceSegmentsAll || []
            const provenanceStrictIdentityReady = prep?.provenanceStrictIdentityReady || false
            const hasStrictMustLocateEntries = prep?.hasStrictMustLocateEntries || false
            const strictStructuredLocateOnly = prep?.strictStructuredLocateOnly || false
            const strictStructuredInlineLocate = prep?.strictStructuredInlineLocate || false
            const suppressLooseInlineLocate = shouldSuppressLooseInlineLocate({
              guideSourcePath,
              bodyContent: String(bodyContent || ''),
              hasRawCiteDetails: effectiveCiteDetails.length > 0,
              hasStructuredProvenance,
              hasDirectProvenance,
            })
            const guideInlineTextTailLocate = Boolean(
              !suppressLooseInlineLocate
              && (
              guideSourcePath
              && provenanceLocateEntries.length > 0
              ),
            )
            const provenanceModeLabel = prep?.provenanceModeLabel || ''
            const structuredRenderSlotMap = prep?.structuredRenderSlotMap || new Map<number, StructuredRenderLocateSlot>()
            const structuredLocateOrderBySegmentId = prep?.structuredLocateOrderBySegmentId || new Map<string, number>()
            const allowedStructuredRenderOrders = prep?.allowedStructuredRenderOrders || new Set<number>()
            const structuredInlineLocateResolver = createStructuredInlineLocateResolver({
              strictStructuredInlineLocate,
              provenanceLocateEntries,
              structuredRenderSlotMap,
              structuredLocateOrderBySegmentId,
              messageProvenance,
              structuredProvenanceSegmentsAll,
              provenanceBlockMap,
              provenanceSourcePath,
              effectiveGuideSourcePath,
              provenanceSourceName,
              locateSourceName,
            })
            const {
              resolveExactStructuredInlineResolution,
              resolveStrictParagraphEntry,
              isStrictStructuredTargetCompatible,
            } = structuredInlineLocateResolver
            const locateCandidates = prep?.locateCandidates || (guideSourcePath ? guideDocCandidates : [])
            const enableLocateUi = Boolean(onOpenReader) && (
              strictStructuredLocateOnly
              || strictStructuredInlineLocate
              || hasDirectProvenance
              || provenanceLocateEntries.length > 0
              || locateCandidates.length > 0
            )
            const hasInlineLocateSurface = Boolean(enableLocateUi && (
              guideInlineTextTailLocate
              || strictStructuredInlineLocate
              || (!guideSourcePath && !suppressLooseInlineLocate)
            ))
            const showProvenanceLocateChips = Boolean(onOpenReader)
              && provenanceLocateEntries.length > 0
              && !hasInlineLocateSurface
            const {
              resolveProvenanceLocateCandidates,
              resolveLocateCandidates,
              locateCandidateKey,
            } = createMessageLocateResolvers({
              locateCandidates,
              provenanceSourcePath,
              provenanceSourceName,
              locateSourceName,
              provenanceDirectSegments,
              provenanceBlockMap,
              strictProvenanceLocate,
              hasStructuredProvenance,
              provenanceStrictIdentityReady,
              hasStrictMustLocateEntries,
              hasDirectProvenance,
            })
            const openReaderByCandidates = (
              pickedList: LocateCandidate[],
              snippet: string,
              opts?: { strictLocate?: boolean; highlightSnippet?: string; relatedBlockIds?: string[] },
            ) => {
              if (!onOpenReader) return
              const payload = buildHeuristicReaderOpenPayload(pickedList, snippet, opts)
              if (!payload) return
              onOpenReader(payload)
            }
            const openReaderByStructuredEntry = (entry: ProvenanceLocateEntry, snippet: string) => {
              if (!onOpenReader) return
              const sourcePath = String(entry.primary?.sourcePath || '').trim()
              const resolvedEntry = remapStructuredEntryToGuideAnchors(
                entry,
                sourcePath
                  ? lookupGuideCandidatesBySourcePath(guideDocCandidatesBySourcePath, sourcePath)
                  : [],
              )
              const payload = buildStructuredEntryReaderOpenPayload(resolvedEntry, snippet)
              if (!payload) return
              onOpenReader(payload)
            }
            const markdownLocateProps = buildMessageMarkdownLocateProps({
              enableLocateUi,
              guideSourcePath,
              guideInlineTextTailLocate,
              strictStructuredInlineLocate,
              suppressLooseInlineLocate,
              strictStructuredLocateOnly,
              allowedStructuredRenderOrders,
              resolveStrictParagraphEntry,
              resolveExactStructuredInlineResolution,
              isStrictStructuredTargetCompatible,
              resolveProvenanceLocateCandidates,
              resolveLocateCandidates,
              locateCandidateKey,
              openReaderByCandidates,
              openReaderByStructuredEntry,
            })
            return (
              <div
                key={message.id}
                data-msg-id={message.id}
                className={`kb-message-row ${isUser ? 'is-user' : 'is-assistant'} ${shelfMessageFlashId === message.id ? 'is-shelf-jump' : ''}`}
              >
                {!isUser ? <AssistantAvatar /> : null}
                {isUser ? (
                  <UserMessageBubble
                    content={message.content}
                    imageAttachments={imageAttachments}
                    showText={showUserText}
                    imageOnly={isImageOnlyUserMessage}
                  />
                ) : (
                  <div className="kb-msg-bubble kb-msg-bubble-assistant">
                    <>
                      <AssistantMessageNotices
                        message={message}
                        lowConfidenceMeta={lowConfidenceMeta}
                        provenanceModeLabel={provenanceModeLabel}
                        onOpenEvidence={(sourceNotice) => openEvidenceDrawer(sourceNotice, effectiveCiteDetails)}
                        S={S}
                      />
                      <MarkdownRenderer
                        content={bodyContent}
                        citeDetails={effectiveCiteDetails}
                        onCitationClick={openCitation}
                        onCitationHover={previewCitation}
                        onCitationLeave={scheduleCitationPreviewClose}
                        {...markdownLocateProps}
                      />
                      <ResearchContextReceipt
                        pack={selectedResearchContextPack}
                        onOpenReader={onOpenReader}
                        onFollowUp={onResearchContextFollowUp ? useReceiptItemAsFollowUp : undefined}
                        S={S}
                      />
                      <MessageReferenceCandidates
                        views={unlinkedReferenceViews}
                        messageId={message.id}
                        canOpenReader={Boolean(onOpenReader)}
                        onOpenReader={openReaderFromDetail}
                        onAddToShelf={addToShelf}
                        S={S}
                      />
                      <MessageProvenanceChips
                        entries={showProvenanceLocateChips ? provenanceLocateEntries : []}
                        messageId={message.id}
                        onOpenEntry={openReaderByStructuredEntry}
                      />
                      <AgentTracePanel
                        trace={agentTrace}
                        messageId={message.id}
                        canLoadTrace={canLoadAgentTrace}
                        onLoadTrace={(messageId) => chatApi.getMessageAgentTrace(messageId, activeConvId)}
                        onOpenReference={openReaderFromDetail}
                        onAddReferenceToShelf={addToShelf}
                      />
                      <ResearchTracePanel trace={researchTrace} />
                      <CopyBar
                        text={getMessageCopyTextValue(message)}
                        markdown={getMessageCopyMarkdownValue(message)}
                      />
                      {retryPrompt ? (
                        <Button
                          className="kb-generation-retry-btn"
                          type="text"
                          size="small"
                          icon={<ReloadOutlined />}
                          disabled={generationPartial !== undefined && generationPartial !== null}
                          onClick={() => onRetryMessage?.(retryPrompt)}
                        >
                          {S.chat_retry_answer}
                        </Button>
                      ) : null}
                    </>
                  </div>
                )}
                {isUser ? (
                  <div className="kb-msg-avatar kb-msg-avatar-user">
                    <UserOutlined className="text-xs" />
                  </div>
                ) : null}
              </div>
            )
          })}

          {generationPartial !== undefined && generationPartial !== null ? (
            <div className="kb-message-row is-assistant">
              <AssistantAvatar />
              <div className="kb-msg-bubble kb-msg-bubble-assistant is-streaming">
                {generationStage ? (
                  <div className="mb-2 flex items-center gap-2">
                    <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-[var(--accent)]" />
                    <Text type="secondary" className="text-xs">
                      {generationStage}
                    </Text>
                  </div>
                ) : null}
                {hasGenerationAgentSourceSummary ? (
                  <AssistantSourceSummaryNotice
                    answerContract={generationAnswerContract}
                    sourceSummary={effectiveGenerationSourceSummary}
                    fallbackNoticeText={generationSourceNotice.notice}
                    onOpenEvidence={(sourceNotice) => openEvidenceDrawer(sourceNotice, [])}
                    S={S}
                  />
                ) : generationSourceNotice.notice ? (
                  <AssistantSourceNotice noticeText={generationSourceNotice.notice} S={S} />
                ) : null}
                {visibleGenerationPartial ? (
                  <div className="whitespace-pre-wrap break-words text-sm leading-7 text-[var(--text)]">
                    {visibleGenerationPartial}
                  </div>
                ) : (
                  <div className="flex items-center gap-1 py-1">
                    <span className="typing-dot" />
                    <span className="typing-dot" style={{ animationDelay: '0.15s' }} />
                    <span className="typing-dot" style={{ animationDelay: '0.3s' }} />
                  </div>
                )}
                <AgentTracePanel
                  trace={generationAgentTrace}
                  onOpenReference={openReaderFromDetail}
                  onAddReferenceToShelf={addToShelf}
                />
                <ResearchTracePanel trace={generationTrace} />
              </div>
            </div>
          ) : null}
        </div>
      </div>
      <CitationPopover
        detail={popoverDetail}
        position={popoverPos}
        loading={popoverLoading}
        guideLoading={popoverGuideLoading}
        inShelf={Boolean(popoverDetail && (() => {
          const popoverItem = toShelfItem(popoverDetail)
          const identity = shelfPaperIdentity(popoverItem)
          return shelfItems.some((item) => item.key === popoverItem.key || shelfPaperIdentity(item) === identity)
        })())}
        onClose={closeCitationPopover}
        onAddToShelf={addToShelf}
        onOpenShelf={openCitationShelfFromPopover}
        onOpenReader={openReaderFromDetail}
        onStartGuide={startPaperGuideFromDetail}
        onMouseEnter={keepCitationPreviewOpen}
        onMouseLeave={scheduleCitationPreviewClose}
      />
      <EvidenceDrawer
        open={Boolean(evidenceDrawerSource)}
        sourceNotice={evidenceDrawerSource}
        citeDetails={evidenceDrawerCiteDetails}
        onClose={closeEvidenceDrawer}
        onOpenReader={onOpenReader ? openReaderFromDetail : undefined}
        onAddToShelf={addToShelf}
        S={S}
      />
      {renderedShelfNode}
    </>
  )
}
