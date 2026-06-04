import type { MouseEvent, ReactNode, RefObject, UIEvent } from 'react'
import { AimOutlined } from '@ant-design/icons'
import { Empty, Spin } from 'antd'
import { ReaderOutlinePanel } from './ReaderOutlinePanel'
import { ReaderHighlightsPanel } from './ReaderHighlightsPanel'
import { ReaderEvidenceNav } from './ReaderEvidenceNav'
import type { ReaderSessionHighlight } from './readerTypes'
import type { ReaderOutlineItem } from './useReaderOutline'
import { useT } from '../../../i18n'

interface ReaderCandidateOption {
  displayIndex: number
  targetIndex: number
  label: string
  distinctKey: string
  roleLabel?: string
  roleTone?: 'neutral' | 'accent' | 'success' | 'warning' | 'danger'
}

interface ReaderMetaBadge {
  key: string
  label: string
  title?: string
  tone?: 'neutral' | 'accent' | 'success' | 'warning' | 'danger'
  testId?: string
}

interface ReaderSelectionBubbleState {
  x: number
  y: number
  canHighlight: boolean
  highlightId: string
}

interface ReaderHighlightActionState {
  x: number
  y: number
  highlightId: string
  text: string
}

interface PaperGuideReaderPanelProps {
  metaLocationText: string
  activeHeadingPath: string
  locateBadges: ReaderMetaBadge[]
  statusTextCompact: string
  statusTextFull: string
  decisionText: string
  decisionTitle?: string
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
  activeCandidateDistinctKey: string
  onToggleOutline: () => void
  onSelectOutline: (item: ReaderOutlineItem) => void
  onToggleHighlights: () => void
  onSelectHighlight: (item: ReaderSessionHighlight) => void
  onRemoveHighlight: (highlightId: string) => void
  onGoPrevEvidence: () => void
  onGoNextEvidence: () => void
  onToggleCandidatePicker: () => void
  onSelectCandidate: (idx: number) => void
  onReturnToEvidence: () => void
  returnToEvidenceLabel: string
  returnToEvidenceTitle: string
  loading: boolean
  error: string
  hasMarkdown: boolean
  selectionBubble: ReaderSelectionBubbleState | null
  highlightBubble: ReaderHighlightActionState | null
  activeHighlightFeedback: string
  onToggleSelectionHighlight: () => void
  onAddSelectionToShelf?: () => void
  onRemoveActiveHighlight: () => void
  onAddActiveHighlightToShelf?: () => void
  onSetActiveHighlightFeedback?: (feedback: 'useful' | 'needs_check') => void
  onAskActiveHighlight: () => void
  onAskSelection: () => void
  isInlinePresentation: boolean
  isPageSurface: boolean
  contentRef: RefObject<HTMLDivElement | null>
  onContentClick: (event: MouseEvent<HTMLDivElement>) => void
  onContentMouseUp: () => void
  onContentKeyUp: () => void
  onContentScroll: (event: UIEvent<HTMLDivElement>) => void
  children: ReactNode
}

export function PaperGuideReaderPanel({
  metaLocationText,
  activeHeadingPath,
  locateBadges,
  statusTextCompact,
  statusTextFull,
  decisionText,
  decisionTitle,
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
  activeCandidateDistinctKey,
  onToggleOutline,
  onSelectOutline,
  onToggleHighlights,
  onSelectHighlight,
  onRemoveHighlight,
  onGoPrevEvidence,
  onGoNextEvidence,
  onToggleCandidatePicker,
  onSelectCandidate,
  onReturnToEvidence,
  returnToEvidenceLabel,
  returnToEvidenceTitle,
  loading,
  error,
  hasMarkdown,
  selectionBubble,
  highlightBubble,
  activeHighlightFeedback,
  onToggleSelectionHighlight,
  onAddSelectionToShelf,
  onRemoveActiveHighlight,
  onAddActiveHighlightToShelf,
  onSetActiveHighlightFeedback,
  onAskActiveHighlight,
  onAskSelection,
  isInlinePresentation,
  isPageSurface,
  contentRef,
  onContentClick,
  onContentMouseUp,
  onContentKeyUp,
  onContentScroll,
  children,
}: PaperGuideReaderPanelProps) {
  const S = useT()
  const showSidebar = (hasOutline && outlineOpen) || (hasHighlights && highlightsOpen)
  const canReturnToEvidence = Boolean(activeHeadingPath || statusTextCompact || activeEvidenceLabel || hasEvidenceNav || hasDistinctAlternatives)
  const visibleLocateBadges = isPageSurface
    ? locateBadges.filter((badge) => badge.key !== 'mode')
    : locateBadges

  return (
    <>
      <div className={`kb-reader-meta-stack ${isPageSurface ? 'is-page' : 'is-dock'}`}>
        <div
          className="kb-reader-meta-location"
          title={activeHeadingPath
            ? `${S.reader_located_prefix || 'Located'}: ${activeHeadingPath}`
            : `${S.reader_located_prefix || 'Located'}: ${S.reader_document_start || 'document start'}`}
        >
          {metaLocationText}
        </div>
        {hasOutline || visibleLocateBadges.length > 0 || statusTextCompact || selectionText || hasDistinctAlternatives ? (
          <div className="kb-reader-meta-side">
            {canReturnToEvidence ? (
              <button
                type="button"
                className="kb-reader-return-btn"
                onClick={onReturnToEvidence}
                title={returnToEvidenceTitle}
                data-testid="reader-return-evidence"
              >
                <AimOutlined />
                <span>{returnToEvidenceLabel}</span>
              </button>
            ) : null}
            {hasOutline ? (
              <button
                type="button"
                className={`kb-reader-candidate-toggle ${outlineOpen ? 'is-open' : ''}`}
                onClick={onToggleOutline}
                title={outlineOpen
                  ? (S.reader_hide_sections || 'Hide section outline')
                  : (S.reader_show_sections || 'Show section outline')}
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
                title={highlightsOpen
                  ? (S.reader_hide_highlights || 'Hide highlights')
                  : (S.reader_show_highlights || 'Show highlights')}
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
                prevLabel={S.reader_prev_evidence || 'Previous evidence'}
                nextLabel={S.reader_next_evidence || 'Next evidence'}
              />
            ) : null}
            {visibleLocateBadges.map((badge) => (
              <span
                key={badge.key}
                className={`kb-reader-meta-pill is-${badge.tone || 'neutral'}`}
                title={badge.title || badge.label}
                data-testid={badge.testId}
              >
                {badge.label}
              </span>
            ))}
            {statusTextCompact ? (
              <span
                className="kb-reader-meta-pill"
                title={statusTextFull}
                data-testid="reader-locate-status"
              >
                {statusTextCompact}
              </span>
            ) : null}
            {selectionText ? (
              <span className="kb-reader-meta-pill">
                {(S.reader_selection_chars || '{n} chars').replace('{n}', String(selectionText.length))}
              </span>
            ) : null}
            {hasDistinctAlternatives ? (
              <button
                type="button"
                className={`kb-reader-candidate-toggle ${candidatePickerExpanded ? 'is-open' : ''}`}
                onClick={onToggleCandidatePicker}
                title={candidatePickerExpanded
                  ? (S.reader_hide_candidates || 'Hide candidates')
                  : (S.reader_show_candidates || 'View candidates')}
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
              const isActive = option.distinctKey === activeCandidateDistinctKey
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
                  {option.roleLabel ? (
                    <span className={`kb-reader-candidate-role is-${option.roleTone || 'neutral'}`}>
                      {option.roleLabel}
                    </span>
                  ) : null}
                  <span className="kb-reader-candidate-label">{option.label}</span>
                </button>
              )
            })}
          </div>
        ) : null}
        {decisionText ? (
          <div
            className="kb-reader-locate-note"
            title={decisionTitle || decisionText}
            data-testid="reader-locate-decision"
          >
            {decisionText}
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
                    titleLabel={S.reader_sections || 'Sections'}
                  />
                ) : null}
                {highlightsOpen && hasHighlights ? (
                  <ReaderHighlightsPanel
                    items={highlightItems}
                    activeItemId={activeHighlightId}
                    onSelectItem={onSelectHighlight}
                    onRemoveItem={onRemoveHighlight}
                    titleLabel={S.reader_highlights || 'Highlights'}
                    removeLabel={S.reader_remove_highlight || 'Remove'}
                    usefulLabel={S.reader_feedback_useful || 'Useful'}
                    checkLabel={S.reader_feedback_check || 'Check'}
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
                    title={selectionBubble.highlightId
                      ? (S.reader_remove_highlight || 'Remove highlight')
                      : (S.reader_highlight_selection || 'Highlight this selection')}
                    data-testid="reader-selection-highlight"
                  >
                    {selectionBubble.highlightId
                      ? (S.reader_undo_highlight || 'Undo')
                      : (S.reader_highlight || 'Highlight')}
                  </button>
                ) : null}
                {onAddSelectionToShelf ? (
                  <button
                    type="button"
                    className="kb-reader-selection-action"
                    onClick={onAddSelectionToShelf}
                    title={S.reader_add_to_shelf_title || 'Add this selection to the citation shelf'}
                    data-testid="reader-selection-shelf"
                  >
                    {S.reader_add_to_shelf || 'Shelf'}
                  </button>
                ) : null}
                <button
                  type="button"
                  className="kb-reader-selection-action is-accent"
                  onClick={onAskSelection}
                  title={S.reader_ask_selection_title || 'Ask about this selection'}
                  data-testid="reader-selection-ask"
                >
                  {S.reader_ask_selection || 'Ask'}
                </button>
              </div>
            ) : null}
            {highlightBubble ? (
              <div
                className="kb-reader-selection-bubble is-highlight-menu"
                style={{ left: `${highlightBubble.x}px`, top: `${highlightBubble.y}px` }}
                onMouseDown={(event) => event.preventDefault()}
                data-testid="reader-highlight-menu"
              >
                <span className="kb-reader-highlight-menu-text" title={highlightBubble.text}>
                  {S.reader_evidence_note || S.reader_highlight || 'Evidence'}
                </span>
                {onSetActiveHighlightFeedback ? (
                  <>
                    <button
                      type="button"
                      className={`kb-reader-selection-action is-feedback ${activeHighlightFeedback === 'useful' ? 'is-active' : ''}`}
                      onClick={() => onSetActiveHighlightFeedback('useful')}
                      title={S.reader_feedback_useful_title || 'Mark this evidence as useful'}
                      data-testid="reader-highlight-menu-feedback-useful"
                    >
                      {S.reader_feedback_useful || 'Useful'}
                    </button>
                    <button
                      type="button"
                      className={`kb-reader-selection-action is-feedback ${activeHighlightFeedback === 'needs_check' ? 'is-active' : ''}`}
                      onClick={() => onSetActiveHighlightFeedback('needs_check')}
                      title={S.reader_feedback_check_title || 'Mark this evidence as needing review'}
                      data-testid="reader-highlight-menu-feedback-check"
                    >
                      {S.reader_feedback_check || 'Check'}
                    </button>
                  </>
                ) : null}
                <button
                  type="button"
                  className="kb-reader-selection-action is-danger"
                  onClick={onRemoveActiveHighlight}
                  title={S.reader_remove_highlight || 'Remove highlight'}
                  data-testid="reader-highlight-menu-remove"
                >
                  {S.reader_remove_highlight || 'Remove'}
                </button>
                {onAddActiveHighlightToShelf ? (
                  <button
                    type="button"
                    className="kb-reader-selection-action"
                    onClick={onAddActiveHighlightToShelf}
                    title={S.reader_add_to_shelf_title || 'Add this selection to the citation shelf'}
                    data-testid="reader-highlight-menu-shelf"
                  >
                    {S.reader_add_to_shelf || 'Shelf'}
                  </button>
                ) : null}
                <button
                  type="button"
                  className="kb-reader-selection-action is-accent"
                  onClick={onAskActiveHighlight}
                  title={S.reader_ask_selection_title || 'Ask about this selection'}
                  data-testid="reader-highlight-menu-ask"
                >
                  {S.reader_ask_selection || 'Ask'}
                </button>
              </div>
            ) : null}
            <div
              ref={contentRef}
              className={isInlinePresentation
                ? 'kb-reader-content min-w-0 w-full flex-1 min-h-0 overflow-x-auto overflow-y-auto pr-1'
                : 'kb-reader-content min-w-0 max-h-[calc(100vh-180px)] overflow-x-auto overflow-y-auto pr-1'}
              onClick={onContentClick}
              onMouseUp={onContentMouseUp}
              onKeyUp={onContentKeyUp}
              onScroll={onContentScroll}
              data-testid="reader-content"
            >
              {children}
            </div>
          </div>
        </div>
      ) : (
        <Empty description={S.reader_no_readable_content || 'No readable content'} />
      )}
    </>
  )
}
