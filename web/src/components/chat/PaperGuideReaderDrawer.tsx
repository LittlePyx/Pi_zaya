/* eslint-disable react-hooks/set-state-in-effect */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { MarkdownRenderer } from './MarkdownRenderer'
import { CitationPopover } from './CitationPopover'
import { PaperGuideReaderPanel } from './reader/PaperGuideReaderPanel'
import { useReaderDocument } from './reader/useReaderDocument'
import { PaperGuideReaderShell } from './reader/PaperGuideReaderShell'
import { useReaderSelectionInteractions } from './reader/useReaderSelectionInteractions'
import { useReaderLocateEngine } from './reader/useReaderLocateEngine'
import { useReaderSessionHighlightLayer } from './reader/useReaderSessionHighlightLayer'
import { useReaderOutline } from './reader/useReaderOutline'
import type { ReaderDocResponse } from '../../api/references'
import {
  type CiteDetail,
} from './citationState'
import { useReaderCitationPopover } from './useReaderCitationPopover'
import { useReaderCitationShelf } from './useReaderCitationShelf'
import { useReaderBlockShelf } from './useReaderBlockShelf'
import { useReaderSelectionShelf } from './useReaderSelectionShelf'
import { useReaderHighlightActions } from './useReaderHighlightActions'
import { useReaderHighlightMenu } from './useReaderHighlightMenu'
import { useReaderHighlightUndoShortcut } from './useReaderHighlightUndoShortcut'
import { useReaderEquationShelfActions } from './useReaderEquationShelfActions'
import { useReaderReturnToEvidence } from './useReaderReturnToEvidence'
import { useReaderLocateResultReporting } from './useReaderLocateResultReporting'
import {
  buildReaderSourceLabel,
  useReaderSidePanelNavigation,
} from './useReaderSidePanelNavigation'
import {
  buildReaderActiveLocateCandidate,
  buildReaderOpenPayloadViewModel,
} from './readerOpenPayloadViewModel'
import {
  buildReaderLocateStatusViewModel,
} from './readerLocateStatusViewModel'
import { buildReaderLocateCandidateViewModel } from './readerLocateCandidateViewModel'
import type {
  ReaderLocateResult,
  ReaderOpenPayload,
  ReaderSelectionShelfPayload,
  ReaderSessionHighlight,
} from './reader/readerTypes'
import {
  compactLocateHintLabel,
} from './reader/readerDomUtils'
import { useT } from '../../i18n'
export type {
  ReaderLocateCandidate,
  ReaderLocateClaimGroup,
  ReaderLocateTarget,
  ReaderOpenPayload,
  ReaderSessionHighlight,
} from './reader/readerTypes'

interface Props {
  open: boolean
  payload: ReaderOpenPayload | null
  onClose: () => void
  onAppendSelection: (text: string) => void
  presentation?: 'drawer' | 'inline'
  surface?: 'dock' | 'page'
  onCollapse?: () => void
  onOpenStandalone?: () => void
  conversationId?: string
  messageId?: number | null
  sessionHighlights?: ReaderSessionHighlight[]
  onAddSessionHighlight?: (highlight: ReaderSessionHighlight) => void
  onUpdateSessionHighlight?: (highlight: ReaderSessionHighlight) => void
  onRemoveSessionHighlight?: (highlightId: string) => void
  onLocateResult?: (result: ReaderLocateResult) => void
  onAddSelectionToShelf?: (payload: ReaderSelectionShelfPayload) => void
  onAddCitationToShelf?: (detail: CiteDetail) => void
  onOpenCitationShelf?: () => void
  documentOverride?: ReaderDocResponse | null
}

function compactReaderEvidenceText(value: string, limit = 180): string {
  const text = String(value || '').replace(/\s+/g, ' ').trim()
  if (!text) return ''
  if (text.length <= limit) return text
  return `${text.slice(0, Math.max(0, limit - 1)).trimEnd()}...`
}

export function PaperGuideReaderDrawer({
  open,
  payload,
  onClose,
  onAppendSelection,
  presentation = 'drawer',
  surface = 'dock',
  onCollapse,
  onOpenStandalone,
  conversationId,
  messageId,
  sessionHighlights = [],
  onAddSessionHighlight,
  onUpdateSessionHighlight,
  onRemoveSessionHighlight,
  onLocateResult,
  onAddSelectionToShelf,
  onAddCitationToShelf,
  onOpenCitationShelf,
  documentOverride,
}: Props) {
  const S = useT()
  const contentRef = useRef<HTMLDivElement>(null)
  const [drawerReady, setDrawerReady] = useState(false)
  const [altChangeSource, setAltChangeSource] = useState<'system' | 'manual'>('system')
  const {
    close: closeReaderCitationPopover,
    detail: citationPopoverDetail,
    loading: citationPopoverLoading,
    position: citationPopoverPos,
    showCitation: showReaderCitation,
  } = useReaderCitationPopover()
  const {
    addCitationToShelf: addReaderCitationToShelf,
    hasCitation: hasReaderCitationInShelf,
  } = useReaderCitationShelf({ onAddCitationToShelf })
  const isInlinePresentation = presentation === 'inline'
  const isPageSurface = isInlinePresentation && surface === 'page'

  const openPayloadViewModel = useMemo(() => buildReaderOpenPayloadViewModel(payload), [payload])
  const {
    activeHitLevel,
    alternatives,
    evidenceAlternatives: payloadEvidenceAlternatives,
    initialAltIndex,
    locateFeedbackKey,
    locateRequestId,
    primaryCandidate,
    relatedBlockIds,
    sourceName,
    sourcePath,
    strictLocate,
    visibleAlternatives,
  } = openPayloadViewModel
  const {
    anchorId,
    anchorKind: primaryAnchorKind,
    anchorNumber: primaryAnchorNumber,
    blockId,
    headingPath: primaryHeadingPath,
    highlightSnippet: primaryHighlightSnippet,
    snippet: primaryFocusSnippet,
  } = primaryCandidate
  const [activeAltIndex, setActiveAltIndexState] = useState(0)
  const [candidatePickerExpanded, setCandidatePickerExpanded] = useState(false)
  const setActiveAltIndex = (idx: number, source: 'system' | 'manual' = 'system') => {
    setAltChangeSource(source)
    setActiveAltIndexState(idx)
  }
  const {
    loading,
    error,
    markdown,
    readerAnchors,
    readerBlocks,
    citeDetails,
    resolvedName,
  } = useReaderDocument({
    open,
    sourcePath,
    sourceName,
    documentOverride,
  })

  const title = resolvedName || sourceName || 'Document reader'
  const {
    addBlockToShelf: addReaderBlockToShelf,
    canAddBlockToShelf: canAddReaderBlockToShelf,
  } = useReaderBlockShelf({
    onAddSelectionToShelf,
    sourceName: title,
    sourcePath,
  })
  const {
    activeAnchorId,
    activeAnchorKind,
    activeAnchorNumber,
    activeBlockId,
    activeFocusSnippet,
    activeHeadingPath,
    activeHighlightSnippet,
    expectsEquationBinding,
  } = useMemo(() => buildReaderActiveLocateCandidate({
    activeAltIndex,
    alternatives,
    primaryCandidate,
  }), [
    activeAltIndex,
    alternatives,
    primaryCandidate,
  ])

  const {
    locateHint,
    locateResult,
    equationBindingReady,
    equationBindingBoundCount,
  } = useReaderLocateEngine({
    open,
    drawerReady,
    markdown,
    locateRequestId,
    sourcePath,
    strictLocate,
    contentRef,
    readerBlocks,
    alternatives,
    relatedBlockIds,
    activeAltIndex,
    setActiveAltIndex: (idx) => setActiveAltIndex(idx, 'system'),
    activeHeadingPath,
    activeFocusSnippet,
    activeHighlightSnippet,
    activeAnchorId,
    activeBlockId,
    activeAnchorKind,
    activeAnchorNumber,
    activeHitLevel,
    expectsEquationBinding,
  })

  const returnToEvidence = useReaderReturnToEvidence({
    activeAnchorId,
    activeAnchorKind,
    activeAnchorNumber,
    activeBlockId,
    activeFocusSnippet,
    activeHeadingPath,
    activeHighlightSnippet,
    contentRef,
    locateResult,
    readerBlocks,
    relatedBlockIds,
  })

  useReaderLocateResultReporting({
    activeAltIndex,
    activeAnchorId,
    activeAnchorKind,
    activeBlockId,
    activeHeadingPath,
    error,
    locateFeedbackKey,
    locateRequestId,
    locateResult,
    onLocateResult,
    open,
    sourceName,
    sourcePath,
    strictLocate,
    title,
  })

  const sourceTitleAttr = String(sourcePath || sourceName || title || '').trim()
  const metaLocationText = activeHeadingPath || (S.reader_document_start || 'Document start')
  const evidenceFocusText = compactReaderEvidenceText(activeHighlightSnippet || activeFocusSnippet)
  const bindingStatusText = expectsEquationBinding && !equationBindingReady
    ? `${S.reader_binding_equations || 'Binding equations'}${equationBindingBoundCount > 0 ? ` (${equationBindingBoundCount})` : ''}`
    : ''
  const statusTextFull = String(locateHint || bindingStatusText).trim()
  const statusTextCompact = compactLocateHintLabel(statusTextFull)
  const {
    activeCandidateDistinctKey,
    candidateOptions,
    candidateToggleLabel,
    evidenceAlternatives,
    hasDistinctAlternatives,
    requestedAltIndex,
    shouldAutoExpandCandidatePicker,
  } = useMemo(() => buildReaderLocateCandidateViewModel({
    activeAltIndex,
    altChangeSource,
    alternatives,
    candidatePickerExpanded,
    evidenceAlternatives: payloadEvidenceAlternatives,
    initialAltIndex,
    locateHint,
    requestedCandidate: {
      headingPath: primaryHeadingPath,
      snippet: primaryFocusSnippet,
      highlightSnippet: primaryHighlightSnippet,
      anchorId,
      blockId,
      anchorKind: primaryAnchorKind,
      anchorNumber: primaryAnchorNumber,
    },
    S,
    strictLocate,
    title,
    visibleAlternatives,
  }), [
    S,
    activeAltIndex,
    altChangeSource,
    alternatives,
    anchorId,
    blockId,
    candidatePickerExpanded,
    initialAltIndex,
    locateHint,
    payloadEvidenceAlternatives,
    primaryAnchorKind,
    primaryAnchorNumber,
    primaryFocusSnippet,
    primaryHeadingPath,
    primaryHighlightSnippet,
    strictLocate,
    title,
    visibleAlternatives,
  ])
  const {
    decisionText,
    decisionTitle,
    locateBadges,
  } = useMemo(() => buildReaderLocateStatusViewModel({
    activeAltIndex,
    activeAnchorKind,
    activeHeadingPath,
    activeHitLevel,
    altChangeSource,
    hasDistinctAlternatives,
    requestedAltIndex,
    S,
    statusTextFull,
    strictLocate,
  }), [
    S,
    activeAltIndex,
    activeAnchorKind,
    activeHeadingPath,
    activeHitLevel,
    altChangeSource,
    hasDistinctAlternatives,
    requestedAltIndex,
    statusTextFull,
    strictLocate,
  ])

  useEffect(() => {
    closeReaderCitationPopover()
  }, [closeReaderCitationPopover, open, sourcePath])

  useReaderEquationShelfActions({
    contentRef,
    labels: S,
    markdown,
    onAddSelectionToShelf,
    open,
    readerBlocks,
    sourceName: title,
    sourcePath,
  })

  const readerMarkdownNode = useMemo(() => (
    <MarkdownRenderer
      content={markdown}
      variant="reader"
      citeDetails={citeDetails}
      onCitationClick={showReaderCitation}
      onCitationAddToShelf={addReaderCitationToShelf}
      onReaderBlockAddToShelf={canAddReaderBlockToShelf ? addReaderBlockToShelf : undefined}
      readerAnchors={readerAnchors}
      readerBlocks={readerBlocks}
    />
  ), [
    addReaderBlockToShelf,
    addReaderCitationToShelf,
    citeDetails,
    canAddReaderBlockToShelf,
    markdown,
    readerAnchors,
    readerBlocks,
    showReaderCitation,
  ])

  const sourceLabel = buildReaderSourceLabel(title, activeHeadingPath)
  const {
    activeHighlight: activeHighlightAction,
    closeHighlightBubble,
    highlightBubble,
    openHighlightMenuFromClick,
  } = useReaderHighlightMenu({
    contentRef,
    open,
    sessionHighlights,
    sourcePath,
  })
  const {
    addHighlightWithUndo,
    appendActiveHighlight,
    clearHighlightUndoStack,
    removeActiveHighlight,
    removeHighlightWithUndo,
    setActiveHighlightFeedback,
    undoHighlightAction,
  } = useReaderHighlightActions({
    activeHeadingPath,
    activeHighlight: activeHighlightAction,
    conversationId,
    labels: S,
    locateFeedbackKey,
    locateRequestId,
    messageId,
    onAddSessionHighlight,
    onAppendSelection,
    onCloseHighlight: closeHighlightBubble,
    onRemoveSessionHighlight,
    onUpdateSessionHighlight,
    sessionHighlights,
    sourceLabel,
    sourceName,
    sourcePath,
    title,
  })

  const {
    outlineItems,
    outlineOpen,
    activeOutlineId,
    hasOutline,
    toggleOutline,
    jumpToOutlineItem,
  } = useReaderOutline({
    open,
    sourcePath,
    isInlinePresentation,
    defaultOutlineOpen: isInlinePresentation && !isPageSurface,
    contentRef,
    readerBlocks,
  })
  const {
    selection,
    selectionBubble,
    clearSelectionState,
    queueSelectionStateSync,
    appendSelection,
    toggleSelectionHighlight,
  } = useReaderSelectionInteractions({
    open,
    sourcePath,
    markdown,
    locateRequestId,
    headingPath: activeHeadingPath,
    contentRef,
    sessionHighlights,
    onAddSessionHighlight: addHighlightWithUndo,
    onRemoveSessionHighlight: removeHighlightWithUndo,
    onAppendSelection,
    sourceLabel,
  })

  const {
    addActiveHighlightToShelf,
    addSelectionToShelf,
    canAddSelectionToShelf,
  } = useReaderSelectionShelf({
    activeAnchorId,
    activeAnchorKind,
    activeBlockId,
    activeHeadingPath,
    activeHighlight: activeHighlightAction,
    onAddSelectionToShelf,
    onClearSelection: clearSelectionState,
    onCloseHighlight: closeHighlightBubble,
    selection,
    selectionBubble,
    sourceName: title,
    sourcePath,
  })

  const handleHighlightMenuClick = useCallback((
    event: Parameters<typeof openHighlightMenuFromClick>[0],
  ) => {
    openHighlightMenuFromClick(event, () => clearSelectionState(true))
  }, [clearSelectionState, openHighlightMenuFromClick])

  const handleContentScroll = useCallback(() => {
    queueSelectionStateSync()
    closeHighlightBubble()
  }, [closeHighlightBubble, queueSelectionStateSync])

  useEffect(() => {
    clearHighlightUndoStack()
  }, [clearHighlightUndoStack, open, sourcePath])

  useReaderHighlightUndoShortcut({
    enabled: open,
    onUndo: undoHighlightAction,
    successLabel: S.reader_undo_complete || 'Undone',
  })

  useReaderSessionHighlightLayer({
    open,
    drawerReady,
    markdown,
    contentRef,
    readerBlocks,
    sessionHighlights,
  })

  const {
    hasEvidenceNav,
    activeEvidenceLabel,
    canGoPrevEvidence,
    canGoNextEvidence,
    evidencePositionLabel,
    goPrevEvidence,
    goNextEvidence,
    hasHighlights,
    highlightsOpen,
    activeHighlightId,
    toggleHighlights,
    jumpToSessionHighlight,
    removeSessionHighlight,
    outlineToggleLabel,
    highlightsToggleLabel,
  } = useReaderSidePanelNavigation({
    activeAltIndex,
    alternatives,
    contentRef,
    evidenceAlternatives,
    isPageSurface,
    onRemoveSessionHighlight: removeHighlightWithUndo,
    open,
    outlineOpen,
    readerBlocks,
    sessionHighlights,
    setActiveAltIndex: (idx) => setActiveAltIndex(idx, 'manual'),
    sourcePath,
    title,
    S,
  })

  useEffect(() => {
    setActiveAltIndex(requestedAltIndex, 'system')
  }, [openPayloadViewModel, requestedAltIndex])

  useEffect(() => {
    if (!open) {
      setCandidatePickerExpanded(false)
      return
    }
    setCandidatePickerExpanded(false)
  }, [open, locateRequestId, sourcePath])

  useEffect(() => {
    if (!shouldAutoExpandCandidatePicker) return
    setCandidatePickerExpanded(true)
  }, [shouldAutoExpandCandidatePicker])

  useEffect(() => {
    if (!open) {
      setDrawerReady(false)
      return
    }
    if (isInlinePresentation) {
      setDrawerReady(true)
      return
    }
    if (drawerReady) return
    // Fallback: some environments may not reliably emit Drawer.afterOpenChange.
    const timer = window.setTimeout(() => {
      setDrawerReady(true)
    }, 240)
    return () => {
      window.clearTimeout(timer)
    }
  }, [open, drawerReady, locateRequestId, sourcePath, isInlinePresentation])

  const panel = (
    <PaperGuideReaderPanel
      metaLocationText={metaLocationText}
      activeHeadingPath={activeHeadingPath}
      evidenceFocusText={evidenceFocusText}
      locateBadges={locateBadges}
      statusTextCompact={statusTextCompact}
      statusTextFull={statusTextFull}
      decisionText={decisionText}
      decisionTitle={decisionTitle}
      selectionText={selection}
      hasOutline={hasOutline}
      outlineOpen={outlineOpen}
      outlineItems={outlineItems}
      activeOutlineId={activeOutlineId}
      hasHighlights={hasHighlights}
      highlightsOpen={highlightsOpen}
      highlightItems={sessionHighlights}
      activeHighlightId={activeHighlightId}
      hasEvidenceNav={hasEvidenceNav}
      evidencePositionLabel={evidencePositionLabel}
      activeEvidenceLabel={activeEvidenceLabel}
      canGoPrevEvidence={canGoPrevEvidence}
      canGoNextEvidence={canGoNextEvidence}
      hasDistinctAlternatives={hasDistinctAlternatives}
      candidatePickerExpanded={candidatePickerExpanded}
      outlineToggleLabel={outlineToggleLabel}
      highlightsToggleLabel={highlightsToggleLabel}
      candidateToggleLabel={candidateToggleLabel}
      candidateOptions={candidateOptions}
      activeCandidateDistinctKey={activeCandidateDistinctKey}
      onToggleOutline={toggleOutline}
      onSelectOutline={jumpToOutlineItem}
      onToggleHighlights={toggleHighlights}
      onSelectHighlight={jumpToSessionHighlight}
      onRemoveHighlight={removeSessionHighlight}
      onGoPrevEvidence={goPrevEvidence}
      onGoNextEvidence={goNextEvidence}
      onToggleCandidatePicker={() => setCandidatePickerExpanded((prev) => !prev)}
      onSelectCandidate={(idx) => setActiveAltIndex(idx, 'manual')}
      onReturnToEvidence={returnToEvidence}
      returnToEvidenceLabel={S.reader_return_to_evidence || 'Back to evidence'}
      returnToEvidenceTitle={S.reader_return_to_evidence_title || 'Return to the located evidence'}
      loading={loading}
      error={error}
      hasMarkdown={Boolean(markdown)}
      selectionBubble={selectionBubble}
      highlightBubble={highlightBubble}
      activeHighlightFeedback={String(activeHighlightAction?.feedback || '')}
      onToggleSelectionHighlight={toggleSelectionHighlight}
      onAddSelectionToShelf={canAddSelectionToShelf ? addSelectionToShelf : undefined}
      onRemoveActiveHighlight={removeActiveHighlight}
      onAddActiveHighlightToShelf={addActiveHighlightToShelf}
      onSetActiveHighlightFeedback={onUpdateSessionHighlight ? setActiveHighlightFeedback : undefined}
      onAskActiveHighlight={appendActiveHighlight}
      onAskSelection={appendSelection}
      isInlinePresentation={isInlinePresentation}
      isPageSurface={isPageSurface}
      contentRef={contentRef}
      onContentClick={handleHighlightMenuClick}
      onContentMouseUp={queueSelectionStateSync}
      onContentKeyUp={queueSelectionStateSync}
      onContentScroll={handleContentScroll}
    >
      {readerMarkdownNode}
    </PaperGuideReaderPanel>
  )
  const citationPopoverInShelf = hasReaderCitationInShelf(citationPopoverDetail)

  return (
    <PaperGuideReaderShell
      open={open}
      isInlinePresentation={isInlinePresentation}
      surface={surface}
      title={title}
      titleTooltip={sourceTitleAttr || title}
      onClose={onClose}
      onCollapse={onCollapse}
      onOpenStandalone={onOpenStandalone}
      openStandaloneLabel={S.reader_open_window || 'Open window'}
      collapseLabel={S.reader_fold || 'Fold'}
      closeLabel={S.shelf_close || 'Close'}
      onAfterOpenChange={setDrawerReady}
    >
      {panel}
      <CitationPopover
        detail={citationPopoverDetail}
        position={citationPopoverPos}
        loading={citationPopoverLoading}
        guideLoading={false}
        inShelf={citationPopoverInShelf}
        onClose={closeReaderCitationPopover}
        onAddToShelf={addReaderCitationToShelf}
        onOpenShelf={() => onOpenCitationShelf?.()}
        onOpenReader={() => {}}
        onStartGuide={() => {}}
        showOpenReaderAction={false}
        showStartGuideAction={false}
      />
    </PaperGuideReaderShell>
  )
}
