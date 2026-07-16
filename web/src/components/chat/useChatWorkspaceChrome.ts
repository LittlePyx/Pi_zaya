import { useMemo } from 'react'
import type { ShelfActivityState } from './MessageList'
import type { ChatActivityItem } from './ChatActivityStrip'
import type { ResearchRuntimeContext } from './researchContext'
import type { RightDockPanel } from './useReaderDock'

interface RefsActivitySummary {
  packCount: number
  pendingPackCount: number
  hitCount: number
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

export function useChatWorkspaceChrome({
  labels,
  researchContext,
  conversationLoading,
  timelineItemCount,
  citationShelfOpen,
  citationShelfCount,
  readerOpen,
  desktopReaderEligible,
  rightDockPanel,
  rightDockCollapsed,
}: {
  labels: Record<string, string>
  researchContext: ResearchRuntimeContext
  conversationLoading: boolean
  timelineItemCount: number
  citationShelfOpen: boolean
  citationShelfCount: number
  readerOpen: boolean
  desktopReaderEligible: boolean
  rightDockPanel: RightDockPanel
  rightDockCollapsed: boolean
}) {
  return useMemo(() => {
    const timelineUiReady = !conversationLoading && timelineItemCount > 0
    const dockTimelineAvailable = timelineUiReady && timelineItemCount > 1
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

    const guideSourceReady = researchContext.guideSource.ready
    return {
      timelineUiReady,
      dockTimelineAvailable,
      dockShelfAvailable,
      dockReaderAvailable,
      showRightDock,
      activeRightDockPanel,
      desktopReaderVisible: readerOpen && desktopReaderEligible,
      rightDockExpanded: showRightDock && !rightDockCollapsed,
      showDesktopTimeline: false,
      showInlineTimelineToggle: timelineUiReady && !desktopReaderEligible,
      showConversationMeta: !conversationLoading && (timelineUiReady || researchContext.mode === 'paper_guide'),
      hideConversationMetaOnDesktop: showRightDock && researchContext.mode !== 'paper_guide',
      guideSourceLabel: researchContext.guideSource.label || labels.guide_unbound,
      guideSourceReady,
      guideStatusLabel: guideSourceReady ? labels.timeline_guide_ready : labels.timeline_guide_pending,
    }
  }, [
    citationShelfCount,
    citationShelfOpen,
    conversationLoading,
    desktopReaderEligible,
    labels.guide_unbound,
    labels.timeline_guide_pending,
    labels.timeline_guide_ready,
    readerOpen,
    researchContext.guideSource.label,
    researchContext.guideSource.ready,
    researchContext.mode,
    rightDockCollapsed,
    rightDockPanel,
    timelineItemCount,
  ])
}

export function useChatActivityItems({
  labels,
  refs,
  conversationLoading,
  messagesLoadingMore,
  liveRunning,
  uploading,
  shelfActivity,
  researchContext,
  apiConnectionAlertTarget,
}: {
  labels: Record<string, string>
  refs: Record<string, unknown>
  conversationLoading: boolean
  messagesLoadingMore: boolean
  liveRunning: boolean
  uploading: boolean
  shelfActivity: ShelfActivityState
  researchContext: ResearchRuntimeContext
  apiConnectionAlertTarget: string
}) {
  return useMemo(() => {
    const refsActivity = summarizeRefsActivity(refs)
    const items: ChatActivityItem[] = []
    if (conversationLoading || messagesLoadingMore) {
      items.push({ key: 'messages', label: labels.chat_activity_messages, tone: 'active' })
    }
    if (liveRunning) {
      items.push({
        key: 'generation',
        label: labels.chat_activity_generation,
        tone: 'active',
      })
    }
    if (uploading) {
      items.push({ key: 'upload', label: labels.chat_activity_upload, tone: 'active' })
    }
    if (refsActivity.pendingPackCount > 0) {
      items.push({
        key: 'refs',
        label: labels.chat_activity_refs.replace('{n}', String(refsActivity.pendingPackCount)),
        tone: 'active',
      })
    }
    if (shelfActivity.count > 0) {
      items.push({
        key: 'shelf',
        label: labels.chat_activity_shelf.replace('{n}', String(shelfActivity.count)),
        tone: 'active',
      })
    }
    if (researchContext.reader.open && researchContext.mode === 'paper_guide') {
      items.push({ key: 'reader', label: labels.chat_activity_reader, tone: 'ready' })
    }
    if (apiConnectionAlertTarget && items.length > 0) {
      items.push({ key: 'api', label: labels.chat_activity_api_attention, tone: 'warning' })
    }
    return items
  }, [
    apiConnectionAlertTarget,
    conversationLoading,
    labels.chat_activity_api_attention,
    labels.chat_activity_generation,
    labels.chat_activity_messages,
    labels.chat_activity_reader,
    labels.chat_activity_refs,
    labels.chat_activity_shelf,
    labels.chat_activity_upload,
    liveRunning,
    messagesLoadingMore,
    refs,
    researchContext.mode,
    researchContext.reader.open,
    shelfActivity.count,
    uploading,
  ])
}
