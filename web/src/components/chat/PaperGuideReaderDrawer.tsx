/* eslint-disable react-hooks/set-state-in-effect */

import { useEffect, useMemo, useRef, useState } from 'react'
import { MarkdownRenderer } from './MarkdownRenderer'
import { PaperGuideReaderPanel } from './reader/PaperGuideReaderPanel'
import { useReaderDocument } from './reader/useReaderDocument'
import { PaperGuideReaderShell } from './reader/PaperGuideReaderShell'
import { useReaderSelectionInteractions } from './reader/useReaderSelectionInteractions'
import { useReaderLocateEngine } from './reader/useReaderLocateEngine'
import { useReaderSessionHighlightLayer } from './reader/useReaderSessionHighlightLayer'
import { useReaderOutline } from './reader/useReaderOutline'
import { useReaderHighlightWorkspace } from './reader/useReaderHighlightWorkspace'
import { useReaderEvidenceNavigator } from './reader/useReaderEvidenceNavigator'
import type {
  ReaderLocateCandidate,
  ReaderOpenPayload,
  ReaderSessionHighlight,
} from './reader/readerTypes'
import {
  candidateDisplayLabel,
  candidateIdentityKey,
  candidateVisibilityKey,
  compactLocateHintLabel,
} from './reader/readerDomUtils'
export type {
  ReaderLocateCandidate,
  ReaderLocateClaimGroup,
  ReaderLocateTarget,
  ReaderOpenPayload,
  ReaderSessionHighlight,
} from './reader/readerTypes'

type LocateBadgeTone = 'neutral' | 'accent' | 'success' | 'warning' | 'danger'

interface LocateMetaBadge {
  key: string
  label: string
  title?: string
  tone?: LocateBadgeTone
  testId?: string
}

interface Props {
  open: boolean
  payload: ReaderOpenPayload | null
  onClose: () => void
  onAppendSelection: (text: string) => void
  presentation?: 'drawer' | 'inline'
  onCollapse?: () => void
  sessionHighlights?: ReaderSessionHighlight[]
  onAddSessionHighlight?: (highlight: ReaderSessionHighlight) => void
  onRemoveSessionHighlight?: (highlightId: string) => void
}

function locateResultBadge(
  statusTextFull: string,
  activeHitLevel: string,
  activeAnchorKind: string,
  strictLocate: boolean,
  activeHeadingPath: string,
): LocateMetaBadge | null {
  const hint = String(statusTextFull || '').trim().toLowerCase()
  const hitLevel = String(activeHitLevel || '').trim().toLowerCase()
  const anchorKind = String(activeAnchorKind || '').trim().toLowerCase()
  const title = statusTextFull || undefined

  if (/\b(strict locate stopped|not found)\b/i.test(hint)) {
    return {
      key: 'result',
      label: 'Unresolved',
      title,
      tone: 'danger',
      testId: 'reader-locate-resolution',
    }
  }
  if (hitLevel === 'heading' || /\bheading\b/i.test(hint)) {
    return {
      key: 'result',
      label: 'Section only',
      title,
      tone: strictLocate ? 'warning' : 'neutral',
      testId: 'reader-locate-resolution',
    }
  }
  if (
    anchorKind === 'equation'
    || anchorKind === 'figure'
    || /\b(equation block|figure block|inline formula|neighbor formula|exact figure block)\b/i.test(hint)
  ) {
    return {
      key: 'result',
      label: 'Bound anchor',
      title,
      tone: 'success',
      testId: 'reader-locate-resolution',
    }
  }
  if (/\b(neighbor evidence|fallback|block only)\b/i.test(hint)) {
    return {
      key: 'result',
      label: 'Fallback evidence',
      title,
      tone: 'warning',
      testId: 'reader-locate-resolution',
    }
  }
  if (hitLevel === 'block' || /\bevidence block matched\b/i.test(hint)) {
    return {
      key: 'result',
      label: 'Bound block',
      title,
      tone: 'success',
      testId: 'reader-locate-resolution',
    }
  }
  if (hitLevel === 'exact' || /\bexact\b/i.test(hint)) {
    return {
      key: 'result',
      label: 'Exact target',
      title,
      tone: 'success',
      testId: 'reader-locate-resolution',
    }
  }
  if (activeHeadingPath) {
    return {
      key: 'result',
      label: strictLocate ? 'Requested section' : 'Section open',
      title,
      tone: 'neutral',
      testId: 'reader-locate-resolution',
    }
  }
  return null
}

export function PaperGuideReaderDrawer({
  open,
  payload,
  onClose,
  onAppendSelection,
  presentation = 'drawer',
  onCollapse,
  sessionHighlights = [],
  onAddSessionHighlight,
  onRemoveSessionHighlight,
}: Props) {
  const contentRef = useRef<HTMLDivElement>(null)
  const [drawerReady, setDrawerReady] = useState(false)
  const [altChangeSource, setAltChangeSource] = useState<'system' | 'manual'>('system')
  const isInlinePresentation = presentation === 'inline'

  const sourcePath = String(payload?.sourcePath || '').trim()
  const sourceName = String(payload?.sourceName || '').trim()
  const headingPath = String(payload?.headingPath || '').trim()
  const focusSnippet = String(payload?.snippet || '').trim()
  const highlightSnippet = String(payload?.highlightSnippet || '').trim()
  const locateTarget = (payload?.locateTarget && typeof payload.locateTarget === 'object')
    ? payload.locateTarget
    : null
  const hasStructuredLocateTarget = Boolean(locateTarget)
  const primaryHeadingPath = String(locateTarget?.headingPath || headingPath).trim()
  const primaryFocusSnippet = String(locateTarget?.snippet || focusSnippet).trim()
  const primaryHighlightSnippet = String(
    locateTarget?.highlightSnippet
    || highlightSnippet
    || primaryFocusSnippet,
  ).trim()
  const anchorId = String(locateTarget?.anchorId || payload?.anchorId || '').trim()
  const blockId = String(locateTarget?.blockId || payload?.blockId || '').trim()
  const relatedBlockIds = Array.isArray(locateTarget?.relatedBlockIds)
    ? locateTarget.relatedBlockIds.map((item) => String(item || '').trim()).filter(Boolean)
    : Array.isArray(payload?.relatedBlockIds)
      ? payload.relatedBlockIds.map((item) => String(item || '').trim()).filter(Boolean)
      : []
  const primaryAnchorKind = String(locateTarget?.anchorKind || payload?.anchorKind || '').trim().toLowerCase()
  const primaryAnchorNumber = Number.isFinite(Number(locateTarget?.anchorNumber || payload?.anchorNumber || 0))
    ? Math.floor(Number(locateTarget?.anchorNumber || payload?.anchorNumber || 0))
    : 0
  const activeHitLevel = String(locateTarget?.hitLevel || '').trim().toLowerCase()
  const strictLocate = Boolean(payload?.strictLocate || hasStructuredLocateTarget)
  const locateRequestId = Number.isFinite(Number(payload?.locateRequestId || 0))
    ? Math.max(0, Math.floor(Number(payload?.locateRequestId || 0)))
    : 0

  const alternatives = useMemo(() => {
    const listRaw = [
      ...(Array.isArray(payload?.visibleAlternatives) ? payload.visibleAlternatives : []),
      ...(Array.isArray(payload?.evidenceAlternatives) ? payload.evidenceAlternatives : []),
      ...(Array.isArray(payload?.alternatives) ? payload.alternatives : []),
    ]
    const out: Array<Required<Pick<ReaderLocateCandidate, 'headingPath' | 'snippet' | 'highlightSnippet' | 'anchorId' | 'blockId' | 'anchorKind' | 'anchorNumber'>>> = []
    const seen = new Set<string>()
    const push = (
      headingPath0: string,
      snippet0: string,
      highlightSnippet0: string,
      anchorId0: string,
      blockId0: string,
      anchorKind0: string,
      anchorNumber0: number,
    ) => {
      const heading = String(headingPath0 || '').trim()
      const snippet = String(snippet0 || '').trim()
      const highlightSnippet = String(highlightSnippet0 || '').trim()
      const anchorId = String(anchorId0 || '').trim()
      const blockId = String(blockId0 || '').trim()
      const anchorKind = String(anchorKind0 || '').trim().toLowerCase()
      const anchorNumber = Number.isFinite(Number(anchorNumber0)) ? Math.floor(Number(anchorNumber0)) : 0
      if (!heading && !snippet && !highlightSnippet && !anchorId && !blockId && !anchorKind && anchorNumber <= 0) return
      const key = candidateIdentityKey({
        headingPath: heading,
        snippet,
        highlightSnippet,
        anchorId,
        blockId,
        anchorKind,
        anchorNumber,
      })
      if (seen.has(key)) return
      seen.add(key)
      out.push({ headingPath: heading, snippet, highlightSnippet, anchorId, blockId, anchorKind, anchorNumber })
    }
    push(
      primaryHeadingPath,
      primaryFocusSnippet,
      primaryHighlightSnippet,
      anchorId,
      blockId,
      primaryAnchorKind,
      primaryAnchorNumber,
    )
    for (const item of listRaw) {
      if (!item || typeof item !== 'object') continue
      push(
        String(item.headingPath || ''),
        String(item.snippet || ''),
        String(item.highlightSnippet || ''),
        String(item.anchorId || ''),
        String(item.blockId || ''),
        String(item.anchorKind || ''),
        Number(item.anchorNumber || 0),
      )
      if (out.length >= 6) break
    }
    return out
  }, [
    payload,
    primaryHeadingPath,
    primaryFocusSnippet,
    primaryHighlightSnippet,
    anchorId,
    blockId,
    primaryAnchorKind,
    primaryAnchorNumber,
  ])
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
    resolvedName,
  } = useReaderDocument({
    open,
    sourcePath,
    sourceName,
  })

  const title = useMemo(
    () => resolvedName || sourceName || 'Document reader',
    [resolvedName, sourceName],
  )
  const requestedCandidateIdentity = useMemo(() => candidateIdentityKey({
    headingPath: primaryHeadingPath,
    snippet: primaryFocusSnippet,
    highlightSnippet: primaryHighlightSnippet,
    anchorId,
    blockId,
    anchorKind: primaryAnchorKind,
    anchorNumber: primaryAnchorNumber,
  }), [
    primaryHeadingPath,
    primaryFocusSnippet,
    primaryHighlightSnippet,
    anchorId,
    blockId,
    primaryAnchorKind,
    primaryAnchorNumber,
  ])
  const visibleCandidateOptions = useMemo(() => {
    const rawList = Array.isArray(payload?.visibleAlternatives) && payload.visibleAlternatives.length > 0
      ? payload.visibleAlternatives
      : alternatives
    if (!Array.isArray(rawList) || rawList.length <= 0) return []
    const internalIndexByKey = new Map<string, number>()
    alternatives.forEach((item, idx) => {
      internalIndexByKey.set(candidateIdentityKey(item), idx)
    })
    const out: Array<{ targetIndex: number; label: string; distinctKey: string }> = []
    const seenDistinct = new Set<string>()
    for (const raw of rawList) {
      if (!raw || typeof raw !== 'object') continue
      const key = candidateIdentityKey(raw)
      const targetIndex = internalIndexByKey.get(key)
      if (!Number.isFinite(targetIndex)) continue
      const safeIndex = Number(targetIndex)
      const item = alternatives[safeIndex]
      if (!item) continue
      const distinctKey = candidateVisibilityKey(item, title) || `alt:${safeIndex + 1}`
      if (seenDistinct.has(distinctKey)) continue
      seenDistinct.add(distinctKey)
      out.push({
        targetIndex: safeIndex,
        label: candidateDisplayLabel(item, title) || `Candidate ${safeIndex + 1}`,
        distinctKey,
      })
    }
    return out
  }, [payload, alternatives, title])
  const evidenceAlternatives = useMemo(() => {
    const rawList = Array.isArray(payload?.evidenceAlternatives)
      ? payload.evidenceAlternatives
      : []
    if (!Array.isArray(rawList) || rawList.length <= 0) return []
    const out: ReaderLocateCandidate[] = []
    const seen = new Set<string>()
    for (const item of rawList) {
      if (!item || typeof item !== 'object') continue
      const key = candidateIdentityKey(item)
      if (!key || seen.has(key)) continue
      seen.add(key)
      out.push({
        headingPath: String(item.headingPath || '').trim() || undefined,
        snippet: String(item.snippet || '').trim() || undefined,
        highlightSnippet: String(item.highlightSnippet || '').trim() || undefined,
        blockId: String(item.blockId || '').trim() || undefined,
        anchorId: String(item.anchorId || '').trim() || undefined,
        anchorKind: String(item.anchorKind || '').trim() || undefined,
        anchorNumber: Number.isFinite(Number(item.anchorNumber || 0))
          ? Math.floor(Number(item.anchorNumber || 0))
          : undefined,
      })
    }
    return out
  }, [payload])
  const evidenceCandidateIdentitySet = useMemo(() => new Set(
    evidenceAlternatives.map((item) => candidateIdentityKey(item)).filter(Boolean),
  ), [evidenceAlternatives])

  const activeAlt = alternatives[activeAltIndex] || null
  const activeCandidateDistinctKey = useMemo(() => {
    if (!activeAlt) return ''
    return candidateVisibilityKey(activeAlt, title) || candidateIdentityKey(activeAlt)
  }, [activeAlt, title])
  const requestedAltIndex = useMemo(() => {
    const hintIndex = Number(payload?.initialAltIndex || 0)
    return Number.isFinite(hintIndex) ? Math.max(0, Math.min(alternatives.length - 1, Math.floor(hintIndex))) : 0
  }, [payload, alternatives.length])
  const candidateOptions = useMemo(() => {
    const describeCandidateRole = (
      candidate: ReaderLocateCandidate | null | undefined,
    ): { roleLabel?: string; roleTone?: LocateBadgeTone } => {
      const identity = candidateIdentityKey(candidate)
      if (!identity) return {}
      const isActive = identity === candidateIdentityKey(activeAlt)
      if (requestedCandidateIdentity && identity === requestedCandidateIdentity) {
        return {
          roleLabel: strictLocate ? 'Requested' : 'Primary',
          roleTone: 'accent',
        }
      }
      if (isActive && strictLocate && altChangeSource === 'system' && activeAltIndex !== requestedAltIndex) {
        return {
          roleLabel: 'Resolved',
          roleTone: 'success',
        }
      }
      if (isActive && strictLocate && altChangeSource === 'manual' && activeAltIndex !== requestedAltIndex) {
        return {
          roleLabel: 'Manual',
          roleTone: 'accent',
        }
      }
      if (evidenceCandidateIdentitySet.has(identity)) {
        return {
          roleLabel: 'Evidence',
          roleTone: 'success',
        }
      }
      return {
        roleLabel: strictLocate ? 'Backup' : 'Alt',
        roleTone: 'neutral',
      }
    }

    const out = visibleCandidateOptions.map((item, displayIndex) => {
      const candidate = alternatives[item.targetIndex] || null
      const role = describeCandidateRole(candidate)
      return {
        displayIndex,
        targetIndex: item.targetIndex,
        label: item.label,
        distinctKey: item.distinctKey,
        roleLabel: role.roleLabel,
        roleTone: role.roleTone,
      }
    })

    const activeOptionExists = out.some((item) => item.distinctKey === activeCandidateDistinctKey)
    if (activeOptionExists || !activeAlt || !activeCandidateDistinctKey) return out
    const role = describeCandidateRole(activeAlt)
    return [
      ...out,
      {
        displayIndex: out.length,
        targetIndex: activeAltIndex,
        label: candidateDisplayLabel(activeAlt, title) || `Candidate ${activeAltIndex + 1}`,
        distinctKey: activeCandidateDistinctKey,
        roleLabel: role.roleLabel,
        roleTone: role.roleTone,
      },
    ]
  }, [
    visibleCandidateOptions,
    alternatives,
    activeAlt,
    activeAltIndex,
    activeCandidateDistinctKey,
    requestedCandidateIdentity,
    evidenceCandidateIdentitySet,
    strictLocate,
    altChangeSource,
    requestedAltIndex,
    title,
  ])
  const hasDistinctAlternatives = useMemo(() => {
    if (candidateOptions.length <= 1) return false
    const distinct = new Set(candidateOptions.map((item) => item.distinctKey).filter(Boolean))
    return distinct.size > 1
  }, [candidateOptions])
  const activeHeadingPath = String(activeAlt?.headingPath || primaryHeadingPath).trim()
  const activeFocusSnippet = String(activeAlt?.snippet || primaryFocusSnippet).trim()
  const activeHighlightSnippet = String(activeAlt?.highlightSnippet || primaryHighlightSnippet || activeFocusSnippet).trim()
  const activeAnchorId = String(activeAlt?.anchorId || anchorId).trim()
  const activeBlockId = String(activeAlt?.blockId || blockId).trim()
  const activeAnchorKind = String(activeAlt?.anchorKind || primaryAnchorKind).trim().toLowerCase()
  const activeAnchorNumber = Number.isFinite(Number(activeAlt?.anchorNumber || primaryAnchorNumber || 0))
    ? Math.floor(Number(activeAlt?.anchorNumber || primaryAnchorNumber || 0))
    : 0
  const expectsEquationBinding = useMemo(() => {
    if (activeAnchorKind === 'equation') return true
    if (alternatives.some((item) => String(item?.anchorKind || '').trim().toLowerCase() === 'equation')) return true
    return false
  }, [activeAnchorKind, alternatives])

  const {
    locateHint,
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

  const sourceTitleAttr = String(sourcePath || sourceName || title || '').trim()
  const metaLocationText = activeHeadingPath || 'Document start'
  const bindingStatusText = expectsEquationBinding && !equationBindingReady
    ? `Binding eq anchors${equationBindingBoundCount > 0 ? ` (${equationBindingBoundCount})` : ''}`
    : ''
  const statusTextFull = String(locateHint || bindingStatusText).trim()
  const statusTextCompact = compactLocateHintLabel(statusTextFull)
  const shouldAutoExpandCandidatePicker = useMemo(() => {
    if (!hasDistinctAlternatives) return false
    if (altChangeSource === 'system' && activeAltIndex > requestedAltIndex) return true
    return /\b(not found|fallback|strict locate|neighbor evidence|was not found)\b/i.test(String(locateHint || ''))
  }, [hasDistinctAlternatives, activeAltIndex, locateHint, altChangeSource, requestedAltIndex])
  const candidateToggleLabel = hasDistinctAlternatives
      ? (candidatePickerExpanded
      ? 'Hide list'
      : activeAltIndex > 0
        ? `Alt ${Math.max(1, candidateOptions.findIndex((item) => item.distinctKey === activeCandidateDistinctKey) + 1)}/${candidateOptions.length}`
        : `${candidateOptions.length} candidates`)
    : ''
  const locateBadges = useMemo(() => {
    const out: LocateMetaBadge[] = []
    out.push({
      key: 'mode',
      label: strictLocate ? 'Strict locate' : 'Section locate',
      title: strictLocate
        ? 'This reader open expects a direct evidence location before softer fallbacks.'
        : 'This reader open starts from the best matched section or snippet.',
      tone: strictLocate ? 'accent' : 'neutral',
      testId: 'reader-locate-mode',
    })
    const resultBadge = locateResultBadge(
      statusTextFull,
      activeHitLevel,
      activeAnchorKind,
      strictLocate,
      activeHeadingPath,
    )
    if (resultBadge) out.push(resultBadge)
    if (hasDistinctAlternatives && altChangeSource === 'system' && activeAltIndex !== requestedAltIndex) {
      out.push({
        key: 'switch',
        label: 'Auto-switched',
        title: 'The requested candidate could not be bound directly, so the reader moved to a backup candidate.',
        tone: 'warning',
        testId: 'reader-locate-switch',
      })
    } else if (hasDistinctAlternatives && altChangeSource === 'manual' && activeAltIndex !== requestedAltIndex) {
      out.push({
        key: 'switch',
        label: 'Manual alt',
        title: 'You are viewing a manually selected alternate candidate.',
        tone: 'accent',
        testId: 'reader-locate-switch',
      })
    }
    return out
  }, [
    strictLocate,
    statusTextFull,
    activeHitLevel,
    activeAnchorKind,
    activeHeadingPath,
    hasDistinctAlternatives,
    altChangeSource,
    activeAltIndex,
    requestedAltIndex,
  ])
  const decisionText = useMemo(() => {
    if (hasDistinctAlternatives && altChangeSource === 'system' && activeAltIndex !== requestedAltIndex) {
      return 'The requested target missed, so the reader moved to the best backup evidence.'
    }
    if (hasDistinctAlternatives && altChangeSource === 'manual' && activeAltIndex !== requestedAltIndex) {
      return 'Showing a manually selected alternate candidate.'
    }
    return ''
  }, [hasDistinctAlternatives, altChangeSource, activeAltIndex, requestedAltIndex])
  const decisionTitle = useMemo(() => {
    if (!decisionText) return undefined
    return statusTextFull || decisionText
  }, [decisionText, statusTextFull])

  const readerMarkdownNode = useMemo(() => (
    <MarkdownRenderer
      content={markdown}
      variant="reader"
      readerAnchors={readerAnchors}
      readerBlocks={readerBlocks}
    />
  ), [markdown, readerAnchors, readerBlocks])

  const sourceLabel = [title, activeHeadingPath].filter(Boolean).join(' / ')
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
    contentRef,
    readerBlocks,
  })
  const {
    selection,
    selectionBubble,
    queueSelectionStateSync,
    appendSelection,
    toggleSelectionHighlight,
  } = useReaderSelectionInteractions({
    open,
    sourcePath,
    markdown,
    locateRequestId,
    contentRef,
    sessionHighlights,
    onAddSessionHighlight,
    onRemoveSessionHighlight,
    onAppendSelection,
    sourceLabel,
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
    hasHighlights,
    highlightsOpen,
    activeHighlightId,
    toggleHighlights,
    jumpToSessionHighlight,
    removeSessionHighlight,
  } = useReaderHighlightWorkspace({
    open,
    sourcePath,
    contentRef,
    readerBlocks,
    sessionHighlights,
    onRemoveSessionHighlight,
  })

  const {
    hasEvidenceNav,
    activeEvidenceItem,
    canGoPrevEvidence,
    canGoNextEvidence,
    evidencePositionLabel,
    goPrevEvidence,
    goNextEvidence,
  } = useReaderEvidenceNavigator({
    open,
    sourcePath,
    title,
    evidenceAlternatives,
    alternatives,
    activeAltIndex,
    setActiveAltIndex: (idx) => setActiveAltIndex(idx, 'manual'),
  })

  useEffect(() => {
    setActiveAltIndex(requestedAltIndex, 'system')
  }, [payload, requestedAltIndex])

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
      activeEvidenceLabel={String(activeEvidenceItem?.label || '').trim()}
      canGoPrevEvidence={canGoPrevEvidence}
      canGoNextEvidence={canGoNextEvidence}
      hasDistinctAlternatives={hasDistinctAlternatives}
      candidatePickerExpanded={candidatePickerExpanded}
      outlineToggleLabel={outlineOpen ? 'Hide sections' : 'Sections'}
      highlightsToggleLabel={highlightsOpen ? 'Hide highlights' : `${sessionHighlights.length} highlights`}
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
      loading={loading}
      error={error}
      hasMarkdown={Boolean(markdown)}
      selectionBubble={selectionBubble}
      onToggleSelectionHighlight={toggleSelectionHighlight}
      onAskSelection={appendSelection}
      isInlinePresentation={isInlinePresentation}
      contentRef={contentRef}
      onContentMouseUp={queueSelectionStateSync}
      onContentKeyUp={queueSelectionStateSync}
    >
      {readerMarkdownNode}
    </PaperGuideReaderPanel>
  )

  return (
    <PaperGuideReaderShell
      open={open}
      isInlinePresentation={isInlinePresentation}
      title={title}
      titleTooltip={sourceTitleAttr || title}
      onClose={onClose}
      onCollapse={onCollapse}
      onAfterOpenChange={setDrawerReady}
    >
      {panel}
    </PaperGuideReaderShell>
  )
}
