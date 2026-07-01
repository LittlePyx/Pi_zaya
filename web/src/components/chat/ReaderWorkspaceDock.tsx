import type { CSSProperties, PointerEventHandler, RefObject } from 'react'
import { BookOutlined, ClockCircleOutlined, MenuFoldOutlined, MenuUnfoldOutlined, ReadOutlined } from '@ant-design/icons'
import { PaperGuideReaderDrawer } from './PaperGuideReaderDrawer'
import type { CiteDetail } from './citationState'
import type { TimelineItem } from './useChatTimeline'
import type { RightDockPanel } from './useReaderDock'
import type {
  ReaderLocateResult,
  ReaderOpenPayload,
  ReaderSelectionShelfPayload,
  ReaderSessionHighlight,
} from './reader/readerTypes'

interface ReaderWorkspaceDockProps {
  labels: Record<string, string>
  rightDockExpanded: boolean
  rightDockResizeGuideRef: RefObject<HTMLDivElement | null>
  rightDockResizing: boolean
  onBeginResize: PointerEventHandler<HTMLDivElement>
  onResizeMove: PointerEventHandler<HTMLDivElement>
  onCommitResize: PointerEventHandler<HTMLDivElement>
  onCancelResize: PointerEventHandler<HTMLDivElement>
  showRightDock: boolean
  activeRightDockPanel: RightDockPanel | null
  rightDockCollapsed: boolean
  rightDockStyle: CSSProperties
  onToggleRightDockCollapsed: () => void
  onActivateDockPanel: (panel: RightDockPanel) => void
  citationShelfCount: number
  dockReaderAvailable: boolean
  dockTimelineAvailable: boolean
  timelineItems: TimelineItem[]
  activeTimelineUserMsgId: number | null
  onTimelineItemClick: (item: TimelineItem) => void
  setShelfDockTarget: (node: HTMLDivElement | null) => void
  readerOpen: boolean
  readerPayload: ReaderOpenPayload | null
  onCloseReader: () => void
  onAppendSelection: (text: string) => void
  onCollapseReader: () => void
  onOpenReaderStandalone: (payload: ReaderOpenPayload | null) => void | Promise<void>
  conversationId: string
  sessionHighlights: ReaderSessionHighlight[]
  onAddSessionHighlight: (highlight: ReaderSessionHighlight) => void
  onUpdateSessionHighlight: (highlight: ReaderSessionHighlight) => void
  onRemoveSessionHighlight: (highlightId: string) => void
  onLocateResult: (result: ReaderLocateResult) => void
  onAddSelectionToShelf: (payload: ReaderSelectionShelfPayload) => void
  onAddCitationToShelf: (detail: CiteDetail) => void
  onOpenCitationShelf: () => void
}

export function ReaderWorkspaceDock({
  labels: S,
  rightDockExpanded,
  rightDockResizeGuideRef,
  rightDockResizing,
  onBeginResize,
  onResizeMove,
  onCommitResize,
  onCancelResize,
  showRightDock,
  activeRightDockPanel,
  rightDockCollapsed,
  rightDockStyle,
  onToggleRightDockCollapsed,
  onActivateDockPanel,
  citationShelfCount,
  dockReaderAvailable,
  dockTimelineAvailable,
  timelineItems,
  activeTimelineUserMsgId,
  onTimelineItemClick,
  setShelfDockTarget,
  readerOpen,
  readerPayload,
  onCloseReader,
  onAppendSelection,
  onCollapseReader,
  onOpenReaderStandalone,
  conversationId,
  sessionHighlights,
  onAddSessionHighlight,
  onUpdateSessionHighlight,
  onRemoveSessionHighlight,
  onLocateResult,
  onAddSelectionToShelf,
  onAddCitationToShelf,
  onOpenCitationShelf,
}: ReaderWorkspaceDockProps) {
  return (
    <>
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
          onPointerDown={onBeginResize}
          onPointerMove={onResizeMove}
          onPointerUp={onCommitResize}
          onPointerCancel={onCancelResize}
        />
      ) : null}
      {showRightDock ? (
        <aside
          className={`kb-chat-side-dock is-${activeRightDockPanel || 'empty'} ${rightDockCollapsed ? 'is-collapsed' : ''} ${rightDockResizing ? 'is-resizing' : ''}`}
          style={rightDockStyle}
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
              onClick={onToggleRightDockCollapsed}
            >
              {rightDockCollapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={activeRightDockPanel === 'shelf'}
              className={`kb-chat-side-tab ${activeRightDockPanel === 'shelf' ? 'is-active' : ''}`}
              onClick={() => onActivateDockPanel('shelf')}
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
                onClick={() => onActivateDockPanel('reader')}
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
                onClick={() => onActivateDockPanel('timeline')}
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
                      onClick={() => onTimelineItemClick(item)}
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
                  onClose={onCloseReader}
                  onAppendSelection={onAppendSelection}
                  presentation="inline"
                  onCollapse={onCollapseReader}
                  onOpenStandalone={() => { void onOpenReaderStandalone(readerPayload) }}
                  conversationId={conversationId}
                  sessionHighlights={sessionHighlights}
                  onAddSessionHighlight={onAddSessionHighlight}
                  onUpdateSessionHighlight={onUpdateSessionHighlight}
                  onRemoveSessionHighlight={onRemoveSessionHighlight}
                  onLocateResult={onLocateResult}
                  onAddSelectionToShelf={onAddSelectionToShelf}
                  onAddCitationToShelf={onAddCitationToShelf}
                  onOpenCitationShelf={onOpenCitationShelf}
                />
              </section>
            ) : null}
          </div>
        </aside>
      ) : null}
    </>
  )
}
