/* eslint-disable react-hooks/set-state-in-effect */

import { useCallback, useDeferredValue, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Button, message, Typography } from 'antd'
import { useChatStore } from '../stores/chatStore'
import { useSettingsStore } from '../stores/settingsStore'
import { MessageList, type ShelfActivityState } from '../components/chat/MessageList'
import { ChatInput } from '../components/chat/ChatInput'
import { PaperGuideReaderDrawer } from '../components/chat/PaperGuideReaderDrawer'
import { ChatActivityStrip } from '../components/chat/ChatActivityStrip'
import { ChatConnectionAlert } from '../components/chat/ChatWorkspaceStatus'
import { ReaderWorkspaceDock } from '../components/chat/ReaderWorkspaceDock'
import { useChatPerfSnapshot } from '../components/chat/useChatPerfSnapshot'
import { useAgentMode } from '../components/chat/useAgentMode'
import { resolveQueryScope, useChatSendFlow } from '../components/chat/useChatSendFlow'
import { useChatTimeline } from '../components/chat/useChatTimeline'
import { useChatUploadFlow } from '../components/chat/useChatUploadFlow'
import { useChatActivityItems, useChatWorkspaceChrome } from '../components/chat/useChatWorkspaceChrome'
import { useReaderWorkspaceActions } from '../components/chat/useReaderWorkspaceActions'
import { useReaderDock } from '../components/chat/useReaderDock'
import { useSelectedResearchContext } from '../components/chat/useSelectedResearchContext'
import { useReaderSessionHighlights } from '../components/chat/reader/useReaderSessionHighlights'
import { useReaderLocateRepair } from '../components/chat/reader/useReaderLocateRepair'
import { buildResearchContext } from '../components/chat/researchContext'
import { useResearchContextAttrs } from '../components/chat/researchContextAttrs'
import type { SelectedResearchContextPack } from '../components/chat/researchContextPack'
import { dispatchOpenSettings, type ApiSettingsTarget } from '../components/layout/settingsEvents'
import type { QueryScope } from '../api/chat'
import { useT } from '../i18n'
import { internalDebugBrowserEnabled } from '../utils/internalDebug'

const { Text } = Typography

const HISTORY_PAGE_SIZE = 24
const LIVE_WINDOW = 16

function loadChatDebugPanelEnabled() {
  return internalDebugBrowserEnabled()
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
  const removePendingImage = useChatStore((s) => s.removePendingImage)
  const cancelGen = useChatStore((s) => s.cancelGeneration)
  const settings = useSettingsStore()
  const liveRunning = Boolean(generation)
  const { agentMode, setAgentMode: handleAgentModeChange } = useAgentMode(activeConvId)
  const {
    readerOpen,
    readerPayload,
    readerOpenRef,
    readerPayloadRef,
    openReaderDock,
    closeReader,
    resetReaderDock,
    rightDockCollapsed,
    rightDockPanel,
    setRightDockPanel,
    showDockPanel,
    collapseRightDock,
    toggleRightDockCollapsed,
    desktopReaderEligible,
    rightDockResizing,
    rightDockStyle,
    splitLayoutRef,
    rightDockResizeGuideRef,
    beginRightDockResize,
    handleRightDockResizeMove,
    commitRightDockResize,
    cancelRightDockResize,
  } = useReaderDock()
  const {
    activeReaderSessionHighlights,
    activeReaderSessionHighlightsRef,
    addReaderSessionHighlight,
    removeReaderSessionHighlight,
    updateReaderSessionHighlight,
  } = useReaderSessionHighlights({
    activeConversationId: activeConvId,
    readerPayload,
  })
  const {
    readerLocateResults,
    sourceQualityRefreshToken,
    nextReaderLocateRequestId,
    registerReaderLocateRequest,
    resetReaderLocateRepair,
    handleReaderLocateResult,
  } = useReaderLocateRepair({
    activeConversationId: activeConvId,
    readerOpenRef,
    readerPayloadRef,
    openReaderDock,
  })
  const [queryScope, setQueryScope] = useState<QueryScope>('library')
  const [shelfActivity, setShelfActivity] = useState<ShelfActivityState>({ summary: false, repair: false, autoRepair: false, background: false, count: 0 })
  const [debugPanelEnabled] = useState(loadChatDebugPanelEnabled)
  const debugSnapshot = useChatPerfSnapshot(debugPanelEnabled)
  const [shelfDockTarget, setShelfDockTarget] = useState<HTMLDivElement | null>(null)
  const eventTokenRef = useRef(1)
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
  const switchToBasketScope = useCallback(() => {
    setQueryScope('basket')
  }, [])
  const {
    currentSelectedResearchContext,
    selectedResearchContextKeys,
    handleResearchContextPackChange,
    clearSelectedResearchContext,
    clearSelectedResearchContextIfCurrent,
  } = useSelectedResearchContext({
    activeConversationId: activeConvId,
    shelfProjectId,
    shelfScope: shelfProjectScope,
    onBasketContextReady: switchToBasketScope,
  })
  const previousShelfProjectScopeRef = useRef(shelfProjectScope)
  const openApiSettings = useCallback((target: ApiSettingsTarget | '' = '') => {
    dispatchOpenSettings(target)
  }, [])

  const nextEventToken = useCallback(() => {
    eventTokenRef.current += 1
    return eventTokenRef.current
  }, [])

  const captureTimelineScrollTop = useCallback(() => {
    const scrollHost = splitLayoutRef.current?.querySelector<HTMLElement>('.kb-main-scroll')
    timelineScrollRestoreTopRef.current = scrollHost ? scrollHost.scrollTop : null
  }, [splitLayoutRef])

  const handleTimelineBlockedJump = useCallback(() => {
    message.info(S.timeline_jump_blocked)
  }, [S.timeline_jump_blocked])

  const {
    timelineOpen,
    timelineItems,
    timelineJump,
    timelineTrackedMessageIds,
    activeTimelineUserMsgId,
    toggleTimelineOpen,
    resetTimeline,
    openTimeline,
    jumpToTimelineItem,
    handleTimelineJumpHandled,
    handleTrackedMessageActive,
  } = useChatTimeline({
    messages,
    labels: S,
    liveRunning,
    onBlockedJump: handleTimelineBlockedJump,
    nextToken: nextEventToken,
    onBeforeToggle: captureTimelineScrollTop,
  })

  const {
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
  } = useReaderWorkspaceActions({
    labels: S,
    activeConversationId: activeConvId,
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
  })

  const handleResearchContextFollowUp = useCallback((pack: SelectedResearchContextPack, promptText: string) => {
    handleResearchContextPackChange(pack)
    appendReaderSelection(promptText)
  }, [appendReaderSelection, handleResearchContextPackChange])

  useEffect(() => {
    const projectChanged = previousShelfProjectScopeRef.current !== shelfProjectScope
    previousShelfProjectScopeRef.current = shelfProjectScope
    resetTimeline()
    resetReaderDock()
    resetReaderLocateRepair()
    if (projectChanged) {
      setRightDockPanel('timeline')
    } else {
      setRightDockPanel((current) => (current === 'reader' ? 'timeline' : current))
    }
    resetReaderWorkspaceTransientState(projectChanged)
  }, [
    activeConvId,
    resetReaderDock,
    resetReaderLocateRepair,
    resetReaderWorkspaceTransientState,
    resetTimeline,
    setRightDockPanel,
    shelfProjectScope,
  ])

  useEffect(() => {
    const hasCurrentPaper = Boolean(researchContext.activeSource.ready)
    const hasBasket = Boolean(currentSelectedResearchContext?.items?.length)
    setQueryScope((current) => {
      if (current === 'library') return current
      return resolveQueryScope(current, { hasCurrentPaper, hasBasket })
    })
  }, [researchContext.activeSource.ready, currentSelectedResearchContext])

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
  }, [desktopReaderEligible, readerOpen, rightDockCollapsed, splitLayoutRef, timelineOpen])

  const {
    onUpload,
    onRetryUpload,
    onCancelUpload,
    onDismissUploadItem,
    onStartGuideFromUpload,
  } = useChatUploadFlow(S)

  const onSend = useChatSendFlow({
    labels: S,
    researchContext,
    queryScope,
    selectedResearchContext: currentSelectedResearchContext,
    agentMode,
    onOpenApiSettings: openApiSettings,
    onSelectedResearchContextConsumed: clearSelectedResearchContextIfCurrent,
  })

  const visibleMessages = liveRunning
    ? messages.slice(-Math.min(messages.length, LIVE_WINDOW))
    : messages
  const deferredRefs = useDeferredValue(refs)
  const hiddenCount = liveRunning
    ? Math.max(0, messages.length - visibleMessages.length)
    : 0
  const effectiveGuide = useMemo(() => {
    const sourcePath = researchContext.guideSource.sourcePath
    const sourceName = researchContext.guideSource.sourceName
    return { sourcePath, sourceName }
  }, [researchContext.guideSource.sourceName, researchContext.guideSource.sourcePath])

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

  const {
    dockTimelineAvailable,
    dockReaderAvailable,
    showRightDock,
    activeRightDockPanel,
    desktopReaderVisible,
    rightDockExpanded,
    showDesktopTimeline,
    showInlineTimelineToggle,
    showConversationMeta,
    hideConversationMetaOnDesktop,
    guideSourceLabel,
    guideSourceReady,
    guideStatusLabel,
  } = useChatWorkspaceChrome({
    labels: S,
    researchContext,
    conversationLoading,
    timelineItemCount: timelineItems.length,
    citationShelfOpen,
    citationShelfCount,
    readerOpen,
    desktopReaderEligible,
    rightDockPanel,
    rightDockCollapsed,
  })
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
              onClick={clearSelectedResearchContext}
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
        onDismissUploadItem={onDismissUploadItem}
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
        onAgentModeChange={handleAgentModeChange}
      />
    </>
  )
  const apiConnectionAlertTarget = researchContext.api.connectionAlertTarget
  const connectionAlert = (
    <ChatConnectionAlert
      labels={S}
      researchContext={researchContext}
      onOpenSettings={openApiSettings}
    />
  )
  const chatActivityItems = useChatActivityItems({
    labels: S,
    refs: deferredRefs,
    conversationLoading,
    messagesLoadingMore,
    liveRunning,
    generationStage: generation?.stage,
    uploading,
    shelfActivity,
    researchContext,
    apiConnectionAlertTarget,
  })
  const chatActivityStrip = (
    <ChatActivityStrip
      items={chatActivityItems}
      debugEnabled={debugPanelEnabled}
      debugSnapshot={debugSnapshot}
      labels={S}
    />
  )
  const researchContextAttrs = useResearchContextAttrs(researchContext)

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
                    generationAgentSourceSummary={generation?.agentSourceSummary}
                    generationAnswerContract={generation?.answerContract}
                    jumpTarget={timelineJump}
                    onJumpHandled={handleTimelineJumpHandled}
                    trackedMessageIds={timelineTrackedMessageIds}
                    onTrackedMessageActive={handleTrackedMessageActive}
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
            <ReaderWorkspaceDock
              labels={S}
              rightDockExpanded={rightDockExpanded}
              rightDockResizeGuideRef={rightDockResizeGuideRef}
              rightDockResizing={rightDockResizing}
              onBeginResize={beginRightDockResize}
              onResizeMove={handleRightDockResizeMove}
              onCommitResize={commitRightDockResize}
              onCancelResize={cancelRightDockResize}
              showRightDock={showRightDock}
              activeRightDockPanel={activeRightDockPanel}
              rightDockCollapsed={rightDockCollapsed}
              rightDockStyle={rightDockStyle}
              onToggleRightDockCollapsed={toggleRightDockCollapsed}
              onActivateDockPanel={activateDockPanel}
              citationShelfCount={citationShelfCount}
              dockReaderAvailable={dockReaderAvailable}
              dockTimelineAvailable={dockTimelineAvailable}
              timelineItems={timelineItems}
              activeTimelineUserMsgId={activeTimelineUserMsgId}
              onTimelineItemClick={jumpToTimelineItem}
              setShelfDockTarget={setShelfDockTarget}
              readerOpen={readerOpen}
              readerPayload={readerPayload}
              onCloseReader={closeReader}
              onAppendSelection={appendReaderSelection}
              onCollapseReader={collapseRightDock}
              onOpenReaderStandalone={openReaderStandalone}
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
          </div>
        </>
      )}
      {!desktopReaderVisible ? (
        <PaperGuideReaderDrawer
          open={readerOpen}
          payload={readerPayload}
          onClose={closeReader}
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
