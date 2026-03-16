import type { ReactNode, RefObject } from 'react'
import { Empty, Spin } from 'antd'
import { ReaderOutlinePanel } from './ReaderOutlinePanel'
import { ReaderHighlightsPanel } from './ReaderHighlightsPanel'
import { ReaderEvidenceNav } from './ReaderEvidenceNav'
import type { ReaderSessionHighlight } from './readerTypes'
import type { ReaderOutlineItem } from './useReaderOutline'

interface ReaderCandidateOption {
  displayIndex: number
  targetIndex: number
  label: string
  distinctKey: string
}

interface ReaderSelectionBubbleState {
  x: number
  y: number
  canHighlight: boolean
  highlightId: string
}

interface PaperGuideReaderPanelProps {
  metaLocationText: string
  activeHeadingPath: string
  statusTextCompact: string
  statusTextFull: string
  selectionText: string
  hasOutline: boolean
  outlineOpen: boolean
  outlineItems: ReaderOutlineItem[]
  activeOutlineId: string
  hasHighlights: boolean
  highlightsOpen: boolean
  highlightItems: ReaderSessionHighlight[]
  activeHighlightId: string
  hasEvidenceNav: boolean
  evidencePositionLabel: string
  activeEvidenceLabel: string
  canGoPrevEvidence: boolean
  canGoNextEvidence: boolean
  hasDistinctAlternatives: boolean
  candidatePickerExpanded: boolean
  outlineToggleLabel: string
  highlightsToggleLabel: string
  candidateToggleLabel: string
  candidateOptions: ReaderCandidateOption[]
  activeAltIndex: number
  onToggleOutline: () => void
  onSelectOutline: (item: ReaderOutlineItem) => void
  onToggleHighlights: () => void
  onSelectHighlight: (item: ReaderSessionHighlight) => void
  onRemoveHighlight: (highlightId: string) => void
  onGoPrevEvidence: () => void
  onGoNextEvidence: () => void
  onToggleCandidatePicker: () => void
  onSelectCandidate: (idx: number) => void
  loading: boolean
  error: string
  hasMarkdown: boolean
  selectionBubble: ReaderSelectionBubbleState | null
  onToggleSelectionHighlight: () => void
  onAskSelection: () => void
  isInlinePresentation: boolean
  contentRef: RefObject<HTMLDivElement | null>
  onContentMouseUp: () => void
  onContentKeyUp: () => void
  children: ReactNode
}

export function PaperGuideReaderPanel({
  metaLocationText,
  activeHeadingPath,
  statusTextCompact,
  statusTextFull,
  selectionText,
  hasOutline,
  outlineOpen,
  outlineItems,
  activeOutlineId,
  hasHighlights,
  highlightsOpen,
  highlightItems,
  activeHighlightId,
  hasEvidenceNav,
  evidencePositionLabel,
  activeEvidenceLabel,
  canGoPrevEvidence,
  canGoNextEvidence,
  hasDistinctAlternatives,
  candidatePickerExpanded,
  outlineToggleLabel,
  highlightsToggleLabel,
  candidateToggleLabel,
  candidateOptions,
  activeAltIndex,
  onToggleOutline,
  onSelectOutline,
  onToggleHighlights,
  onSelectHighlight,
  onRemoveHighlight,
  onGoPrevEvidence,
  onGoNextEvidence,
  onToggleCandidatePicker,
  onSelectCandidate,
  loading,
  error,
  hasMarkdown,
  selectionBubble,
  onToggleSelectionHighlight,
  onAskSelection,
  isInlinePresentation,
  contentRef,
  onContentMouseUp,
  onContentKeyUp,
  children,
}: PaperGuideReaderPanelProps) {
  const showSidebar = (hasOutline && outlineOpen) || (hasHighlights && highlightsOpen)

  return (
    <>
      <div className="kb-reader-meta-stack">
        <div
          className="kb-reader-meta-location"
          title={activeHeadingPath ? `Located: ${activeHeadingPath}` : 'Located: document start'}
        >
          {metaLocationText}
        </div>
        {hasOutline || statusTextCompact || selectionText || hasDistinctAlternatives ? (
          <div className="kb-reader-meta-side">
            {hasOutline ? (
              <button
                type="button"
                className={`kb-reader-candidate-toggle ${outlineOpen ? 'is-open' : ''}`}
                onClick={onToggleOutline}
                title={outlineOpen ? 'Hide section outline' : 'Show section outline'}
                data-testid="reader-outline-toggle"
              >
                {outlineToggleLabel}
              </button>
            ) : null}
            {hasHighlights ? (
              <button
                type="button"
                className={`kb-reader-candidate-toggle ${highlightsOpen ? 'is-open' : ''}`}
                onClick={onToggleHighlights}
                title={highlightsOpen ? 'Hide highlights' : 'Show highlights'}
                data-testid="reader-highlights-toggle"
              >
                {highlightsToggleLabel}
              </button>
            ) : null}
            {hasEvidenceNav ? (
              <ReaderEvidenceNav
                activeLabel={activeEvidenceLabel}
                positionLabel={evidencePositionLabel}
                canGoPrev={canGoPrevEvidence}
                canGoNext={canGoNextEvidence}
                onGoPrev={onGoPrevEvidence}
                onGoNext={onGoNextEvidence}
              />
            ) : null}
            {statusTextCompact ? (
              <span className="kb-reader-meta-pill" title={statusTextFull} data-testid="reader-locate-status">
                {statusTextCompact}
              </span>
            ) : null}
            {selectionText ? (
              <span className="kb-reader-meta-pill">
                {`${selectionText.length} chars`}
              </span>
            ) : null}
            {hasDistinctAlternatives ? (
              <button
                type="button"
                className={`kb-reader-candidate-toggle ${candidatePickerExpanded ? 'is-open' : ''}`}
                onClick={onToggleCandidatePicker}
                title={candidatePickerExpanded ? 'Hide candidates' : 'View candidates'}
                data-testid="reader-candidate-toggle"
              >
                {candidateToggleLabel}
              </button>
            ) : null}
          </div>
        ) : null}
        {candidatePickerExpanded && hasDistinctAlternatives ? (
          <div className="kb-reader-candidate-list">
            {candidateOptions.map((option) => {
              const isActive = option.targetIndex === activeAltIndex
              return (
                <button
                  key={`${option.displayIndex}:${option.targetIndex}:${option.distinctKey}`}
                  type="button"
                  className={`kb-reader-candidate-chip ${isActive ? 'is-active' : ''}`}
                  onClick={() => onSelectCandidate(option.targetIndex)}
                  title={option.label}
                  data-testid={`reader-candidate-chip-${option.displayIndex}`}
                >
                  <span className="kb-reader-candidate-index">{option.displayIndex + 1}</span>
                  <span className="kb-reader-candidate-label">{option.label}</span>
                </button>
              )
            })}
          </div>
        ) : null}
      </div>
      {loading ? (
        <div className="flex h-56 items-center justify-center">
          <Spin />
        </div>
      ) : error ? (
        <Empty description={error} />
      ) : hasMarkdown ? (
        <div className={`kb-reader-body ${showSidebar ? (isInlinePresentation ? 'is-split' : 'is-stacked') : ''}`}>
          {showSidebar ? (
            <aside className={`kb-reader-outline-shell ${isInlinePresentation ? 'is-inline' : 'is-stacked'}`}>
              <div className="kb-reader-side-stack">
                {outlineOpen && hasOutline ? (
                  <ReaderOutlinePanel
                    items={outlineItems}
                    activeItemId={activeOutlineId}
                    onSelectItem={onSelectOutline}
                  />
                ) : null}
                {highlightsOpen && hasHighlights ? (
                  <ReaderHighlightsPanel
                    items={highlightItems}
                    activeItemId={activeHighlightId}
                    onSelectItem={onSelectHighlight}
                    onRemoveItem={onRemoveHighlight}
                  />
                ) : null}
              </div>
            </aside>
          ) : null}
          <div className="relative flex min-h-0 flex-1 flex-col overflow-hidden">
            {selectionBubble ? (
              <div
                className="kb-reader-selection-bubble"
                style={{ left: `${selectionBubble.x}px`, top: `${selectionBubble.y}px` }}
                onMouseDown={(event) => event.preventDefault()}
                data-testid="reader-selection-bubble"
              >
                {selectionBubble.canHighlight ? (
                  <button
                    type="button"
                    className={`kb-reader-selection-action ${selectionBubble.highlightId ? 'is-active' : ''}`}
                    onClick={onToggleSelectionHighlight}
                    title={selectionBubble.highlightId ? 'Remove highlight' : 'Highlight this selection'}
                    data-testid="reader-selection-highlight"
                  >
                    {selectionBubble.highlightId ? 'Marked' : 'Highlight'}
                  </button>
                ) : null}
                <button
                  type="button"
                  className="kb-reader-selection-action is-accent"
                  onClick={onAskSelection}
                  title="Ask about this selection"
                  data-testid="reader-selection-ask"
                >
                  Ask
                </button>
              </div>
            ) : null}
            <div
              ref={contentRef}
              className={isInlinePresentation
                ? 'kb-reader-content min-w-0 w-full flex-1 min-h-0 overflow-x-auto overflow-y-auto pr-1'
                : 'kb-reader-content min-w-0 max-h-[calc(100vh-180px)] overflow-x-auto overflow-y-auto pr-1'}
              onMouseUp={onContentMouseUp}
              onKeyUp={onContentKeyUp}
              data-testid="reader-content"
            >
              {children}
            </div>
          </div>
        </div>
      ) : (
        <Empty description="No readable content" />
      )}
    </>
  )
}
