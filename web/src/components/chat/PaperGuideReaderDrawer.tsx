/* eslint-disable react-hooks/set-state-in-effect */

import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent } from 'react'
import { message } from 'antd'
import { MarkdownRenderer } from './MarkdownRenderer'
import { CitationPopover } from './CitationPopover'
import { PaperGuideReaderPanel } from './reader/PaperGuideReaderPanel'
import { useReaderDocument } from './reader/useReaderDocument'
import { PaperGuideReaderShell } from './reader/PaperGuideReaderShell'
import { useReaderSelectionInteractions } from './reader/useReaderSelectionInteractions'
import { useReaderLocateEngine } from './reader/useReaderLocateEngine'
import { useReaderSessionHighlightLayer } from './reader/useReaderSessionHighlightLayer'
import { useReaderOutline } from './reader/useReaderOutline'
import { useReaderHighlightWorkspace } from './reader/useReaderHighlightWorkspace'
import { useReaderEvidenceNavigator } from './reader/useReaderEvidenceNavigator'
import { referencesApi, type ReaderDocResponse } from '../../api/references'
import {
  mergeCiteMeta,
  toShelfItem,
  type CiteDetail,
} from './citationState'
import type {
  ReaderLocateCandidate,
  ReaderLocateResult,
  ReaderOpenPayload,
  ReaderSelectionShelfPayload,
  ReaderSessionHighlight,
} from './reader/readerTypes'
import {
  buildHighlightQueries,
  candidateDisplayLabel,
  candidateIdentityKey,
  candidateVisibilityKey,
  clearReaderFocusClasses,
  closestReadableBlock,
  compactLocateHintLabel,
  resolveDirectTargetNode,
  resolveStickyHighlightTarget,
  sameHighlightTarget,
  scrollReaderTargetIntoView,
} from './reader/readerDomUtils'
import { useT } from '../../i18n'
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

type HighlightUndoAction =
  | { kind: 'remove'; highlight: ReaderSessionHighlight }
  | { kind: 'restore'; highlight: ReaderSessionHighlight }

function sameHighlightUndoAction(left: HighlightUndoAction, right: HighlightUndoAction): boolean {
  return left.kind === right.kind && String(left.highlight.id || '').trim() === String(right.highlight.id || '').trim()
}

function isEditableUndoTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  if (target.isContentEditable) return true
  return Boolean(target.closest('input, textarea, select, [contenteditable="true"], .ant-input'))
}

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

function locateResultBadge(
  statusTextFull: string,
  activeHitLevel: string,
  activeAnchorKind: string,
  strictLocate: boolean,
  activeHeadingPath: string,
  S: Record<string, string>,
): LocateMetaBadge | null {
  const hint = String(statusTextFull || '').trim().toLowerCase()
  const hitLevel = String(activeHitLevel || '').trim().toLowerCase()
  const anchorKind = String(activeAnchorKind || '').trim().toLowerCase()
  const title = statusTextFull || undefined

  if (/\b(strict locate stopped|not found)\b/i.test(hint)) {
    return {
      key: 'result',
      label: S.reader_locate_unresolved || 'Unresolved',
      title,
      tone: 'danger',
      testId: 'reader-locate-resolution',
    }
  }
  if (hitLevel === 'heading' || /\bheading\b/i.test(hint)) {
    return {
      key: 'result',
      label: S.reader_locate_section_only || 'Section only',
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
      label: S.reader_locate_bound_anchor || 'Bound anchor',
      title,
      tone: 'success',
      testId: 'reader-locate-resolution',
    }
  }
  if (/\b(neighbor evidence|fallback|block only)\b/i.test(hint)) {
    return {
      key: 'result',
      label: S.reader_locate_fallback_evidence || 'Fallback evidence',
      title,
      tone: 'warning',
      testId: 'reader-locate-resolution',
    }
  }
  if (hitLevel === 'block' || /\bevidence block matched\b/i.test(hint)) {
    return {
      key: 'result',
      label: S.reader_locate_bound_block || 'Bound block',
      title,
      tone: 'success',
      testId: 'reader-locate-resolution',
    }
  }
  if (hitLevel === 'exact' || /\bexact\b/i.test(hint)) {
    return {
      key: 'result',
      label: S.reader_locate_exact_target || 'Exact target',
      title,
      tone: 'success',
      testId: 'reader-locate-resolution',
    }
  }
  if (activeHeadingPath) {
    return {
      key: 'result',
      label: strictLocate
        ? (S.reader_locate_requested_section || 'Requested section')
        : (S.reader_locate_section_open || 'Section open'),
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
  const highlightUndoStackRef = useRef<HighlightUndoAction[]>([])
  const [drawerReady, setDrawerReady] = useState(false)
  const [altChangeSource, setAltChangeSource] = useState<'system' | 'manual'>('system')
  const [highlightBubble, setHighlightBubble] = useState<{
    x: number
    y: number
    highlightId: string
    text: string
  } | null>(null)
  const [citationPopoverDetail, setCitationPopoverDetail] = useState<CiteDetail | null>(null)
  const [citationPopoverPos, setCitationPopoverPos] = useState<{ x: number; y: number } | null>(null)
  const [citationPopoverLoading, setCitationPopoverLoading] = useState(false)
  const [readerCitationShelfKeys, setReaderCitationShelfKeys] = useState<Set<string>>(() => new Set())
  const activeCitationRequestKeyRef = useRef('')
  const isInlinePresentation = presentation === 'inline'
  const isPageSurface = isInlinePresentation && surface === 'page'

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
  const locateFeedbackKey = String(payload?.locateFeedbackKey || '').trim()

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
    citeDetails,
    resolvedName,
  } = useReaderDocument({
    open,
    sourcePath,
    sourceName,
    documentOverride,
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
          roleLabel: strictLocate
            ? (S.reader_candidate_requested || 'Requested')
            : (S.reader_candidate_primary || 'Primary'),
          roleTone: 'accent',
        }
      }
      if (isActive && strictLocate && altChangeSource === 'system' && activeAltIndex !== requestedAltIndex) {
        return {
          roleLabel: S.reader_candidate_resolved || 'Resolved',
          roleTone: 'success',
        }
      }
      if (isActive && strictLocate && altChangeSource === 'manual' && activeAltIndex !== requestedAltIndex) {
        return {
          roleLabel: S.reader_candidate_manual || 'Manual',
          roleTone: 'accent',
        }
      }
      if (evidenceCandidateIdentitySet.has(identity)) {
        return {
          roleLabel: S.reader_candidate_evidence || 'Evidence',
          roleTone: 'success',
        }
      }
      return {
        roleLabel: strictLocate
          ? (S.reader_candidate_backup || 'Backup')
          : (S.reader_candidate_alt || 'Alt'),
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
    S,
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

  const returnToEvidence = () => {
    const root = contentRef.current
    if (!root) return
    const resultBlockId = String(locateResult?.blockId || activeBlockId || '').trim()
    const resultAnchorId = String(locateResult?.anchorId || activeAnchorId || '').trim()
    const resultAnchorKind = String(locateResult?.anchorKind || activeAnchorKind || '').trim().toLowerCase()
    const seed = String(activeHighlightSnippet || activeFocusSnippet || '').trim()
    const direct = resolveDirectTargetNode(root, readerBlocks, {
      blockId: resultBlockId,
      anchorId: resultAnchorId,
      anchorKind: resultAnchorKind,
    })
    const target = closestReadableBlock(direct.target) || direct.target || resolveStickyHighlightTarget(root, readerBlocks, {
      blockId: resultBlockId,
      anchorId: resultAnchorId,
      anchorKind: resultAnchorKind,
      anchorNumber: activeAnchorNumber,
      headingPath: String(locateResult?.headingPath || activeHeadingPath || '').trim(),
      highlightSeed: seed,
      highlightQueries: buildHighlightQueries(seed, {
        anchorKind: resultAnchorKind,
        anchorNumber: activeAnchorNumber,
      }),
      relatedBlockIds,
      strictLocate: false,
    })
    if (!target) return
    clearReaderFocusClasses(root)
    target.classList.add('kb-reader-focus')
    scrollReaderTargetIntoView(root, target, { force: true })
  }

  useEffect(() => {
    if (!open || !locateResult || !onLocateResult) return
    onLocateResult({
      ...locateResult,
      sourceName: sourceName || title || undefined,
      locateFeedbackKey: String(payload?.locateFeedbackKey || locateResult.locateFeedbackKey || '').trim() || undefined,
    })
  }, [locateResult, onLocateResult, open, payload?.locateFeedbackKey, sourceName, title])

  useEffect(() => {
    if (!open || !error || !onLocateResult || !sourcePath) return
    onLocateResult({
      locateRequestId,
      sourcePath,
      sourceName: sourceName || title || undefined,
      locateFeedbackKey: String(payload?.locateFeedbackKey || '').trim() || undefined,
      status: 'failed',
      precision: 'failed',
      ok: false,
      repairable: true,
      strictLocate,
      hint: String(error || '').trim() || 'Reader source could not be loaded.',
      reason: String(error || '').trim() || 'Reader source could not be loaded.',
      activeAltIndex,
      blockId: activeBlockId || undefined,
      anchorId: activeAnchorId || undefined,
      anchorKind: activeAnchorKind || undefined,
      headingPath: activeHeadingPath || undefined,
    })
  }, [
    activeAltIndex,
    activeAnchorId,
    activeAnchorKind,
    activeBlockId,
    activeHeadingPath,
    error,
    locateRequestId,
    onLocateResult,
    open,
    payload?.locateFeedbackKey,
    sourceName,
    sourcePath,
    strictLocate,
    title,
  ])

  const sourceTitleAttr = String(sourcePath || sourceName || title || '').trim()
  const metaLocationText = activeHeadingPath || (S.reader_document_start || 'Document start')
  const bindingStatusText = expectsEquationBinding && !equationBindingReady
    ? `${S.reader_binding_equations || 'Binding equations'}${equationBindingBoundCount > 0 ? ` (${equationBindingBoundCount})` : ''}`
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
      ? (S.reader_hide_list || 'Hide list')
      : activeAltIndex > 0
        ? (S.reader_alt_index || 'Alt {i}/{n}')
          .replace('{i}', String(Math.max(1, candidateOptions.findIndex((item) => item.distinctKey === activeCandidateDistinctKey) + 1)))
          .replace('{n}', String(candidateOptions.length))
        : (S.reader_candidates_count || '{n} candidates').replace('{n}', String(candidateOptions.length)))
    : ''
  const locateBadges = useMemo(() => {
    const out: LocateMetaBadge[] = []
    out.push({
      key: 'mode',
      label: strictLocate
        ? (S.reader_locate_mode_strict || 'Strict locate')
        : (S.reader_locate_mode_section || 'Section locate'),
      title: strictLocate
        ? (S.reader_locate_mode_strict_title || 'This reader open expects a direct evidence location before softer fallbacks.')
        : (S.reader_locate_mode_section_title || 'This reader open starts from the best matched section or snippet.'),
      tone: strictLocate ? 'accent' : 'neutral',
      testId: 'reader-locate-mode',
    })
    const resultBadge = locateResultBadge(
      statusTextFull,
      activeHitLevel,
      activeAnchorKind,
      strictLocate,
      activeHeadingPath,
      S,
    )
    if (resultBadge) out.push(resultBadge)
    if (hasDistinctAlternatives && altChangeSource === 'system' && activeAltIndex !== requestedAltIndex) {
      out.push({
        key: 'switch',
        label: S.reader_auto_switched || 'Auto-switched',
        title: S.reader_auto_switched_title || 'The requested candidate could not be bound directly, so the reader moved to a backup candidate.',
        tone: 'warning',
        testId: 'reader-locate-switch',
      })
    } else if (hasDistinctAlternatives && altChangeSource === 'manual' && activeAltIndex !== requestedAltIndex) {
      out.push({
        key: 'switch',
        label: S.reader_manual_alt || 'Manual alt',
        title: S.reader_manual_alt_title || 'You are viewing a manually selected alternate candidate.',
        tone: 'accent',
        testId: 'reader-locate-switch',
      })
    }
    return out
  }, [
    S,
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
      return S.reader_auto_switched_note || 'The requested target missed, so the reader moved to the best backup evidence.'
    }
    if (hasDistinctAlternatives && altChangeSource === 'manual' && activeAltIndex !== requestedAltIndex) {
      return S.reader_manual_alt_note || 'Showing a manually selected alternate candidate.'
    }
    return ''
  }, [S, hasDistinctAlternatives, altChangeSource, activeAltIndex, requestedAltIndex])
  const decisionTitle = useMemo(() => {
    if (!decisionText) return undefined
    return statusTextFull || decisionText
  }, [decisionText, statusTextFull])

  useEffect(() => {
    setCitationPopoverDetail(null)
    setCitationPopoverPos(null)
    setCitationPopoverLoading(false)
    activeCitationRequestKeyRef.current = ''
  }, [open, sourcePath])

  const mergeReaderCitationMeta = useCallback((itemKey: string, metas: Array<Record<string, unknown>>) => {
    const usable = metas.filter((meta) => meta && Object.keys(meta).length > 0)
    if (usable.length <= 0) return
    setCitationPopoverDetail((current) => {
      if (!current) return current
      if (toShelfItem(current).key !== itemKey) return current
      let merged = current
      for (const meta of usable) {
        merged = mergeCiteMeta(merged, meta)
      }
      return merged
    })
  }, [])

  const showReaderCitation = useCallback((detail: CiteDetail, event: MouseEvent<HTMLElement>) => {
    const itemKey = toShelfItem(detail).key
    activeCitationRequestKeyRef.current = itemKey
    setCitationPopoverDetail(detail)
    setCitationPopoverPos({ x: event.clientX, y: event.clientY })
    const hasDoi = Boolean(String(detail.doi || detail.doiUrl || '').trim())
    const reqs: Array<Promise<Record<string, unknown>>> = []
    if (!detail.bibliometricsChecked && (hasDoi || detail.title || detail.raw || detail.citeFmt)) {
      reqs.push(referencesApi.bibliometricsCached(detail as unknown as Record<string, unknown>).catch(() => ({})))
    }
    reqs.push(referencesApi.citationCardPolishCached(detail as unknown as Record<string, unknown>, 1.5).catch(() => ({})))
    setCitationPopoverLoading(reqs.length > 0)
    Promise.all(reqs)
      .then((metas) => {
        if (activeCitationRequestKeyRef.current !== itemKey) return
        mergeReaderCitationMeta(itemKey, metas)
      })
      .finally(() => {
        if (activeCitationRequestKeyRef.current === itemKey) {
          setCitationPopoverLoading(false)
        }
      })
  }, [mergeReaderCitationMeta])

  const closeReaderCitationPopover = useCallback(() => {
    setCitationPopoverDetail(null)
    setCitationPopoverPos(null)
    setCitationPopoverLoading(false)
    activeCitationRequestKeyRef.current = ''
  }, [])

  const addReaderCitationToShelf = useCallback((detail: CiteDetail) => {
    onAddCitationToShelf?.(detail)
    const key = toShelfItem(detail).key
    setReaderCitationShelfKeys((current) => {
      const next = new Set(current)
      next.add(key)
      return next
    })
  }, [onAddCitationToShelf])

  const readerMarkdownNode = useMemo(() => (
    <MarkdownRenderer
      content={markdown}
      variant="reader"
      citeDetails={citeDetails}
      onCitationClick={showReaderCitation}
      onCitationAddToShelf={addReaderCitationToShelf}
      readerAnchors={readerAnchors}
      readerBlocks={readerBlocks}
    />
  ), [addReaderCitationToShelf, citeDetails, markdown, readerAnchors, readerBlocks, showReaderCitation])

  const sourceLabel = [title, activeHeadingPath].filter(Boolean).join(' / ')
  const enrichSessionHighlight = useCallback((highlight: ReaderSessionHighlight): ReaderSessionHighlight => {
    const now = Date.now()
    const rawMessageId = highlight.messageId ?? messageId
    const nextMessageId = rawMessageId == null || !Number.isFinite(Number(rawMessageId))
      ? undefined
      : Number(rawMessageId)
    const rawLocateRequestId = highlight.locateRequestId ?? locateRequestId
    const nextLocateRequestId = rawLocateRequestId == null || !Number.isFinite(Number(rawLocateRequestId)) || Number(rawLocateRequestId) <= 0
      ? undefined
      : Number(rawLocateRequestId)
    return {
      ...highlight,
      noteKind: highlight.noteKind || 'highlight',
      sourcePath: highlight.sourcePath || sourcePath || undefined,
      sourceName: highlight.sourceName || title || sourceName || undefined,
      conversationId: highlight.conversationId || String(conversationId || '').trim() || undefined,
      messageId: nextMessageId,
      locateRequestId: nextLocateRequestId,
      locateFeedbackKey: highlight.locateFeedbackKey || locateFeedbackKey || undefined,
      headingPath: highlight.headingPath || activeHeadingPath || undefined,
      createdAt: Number.isFinite(Number(highlight.createdAt)) ? Number(highlight.createdAt) : now,
      updatedAt: now,
    }
  }, [activeHeadingPath, conversationId, locateFeedbackKey, locateRequestId, messageId, sourceName, sourcePath, title])

  const applyHighlightUndoAction = useCallback((action: HighlightUndoAction): boolean => {
    const highlightId = String(action.highlight.id || '').trim()
    if (!highlightId) return false
    if (action.kind === 'remove') {
      onRemoveSessionHighlight?.(highlightId)
    } else {
      onAddSessionHighlight?.(action.highlight)
    }
    setHighlightBubble(null)
    return true
  }, [onAddSessionHighlight, onRemoveSessionHighlight])

  const undoHighlightAction = useCallback((specificAction?: HighlightUndoAction): boolean => {
    let action = specificAction || highlightUndoStackRef.current.pop()
    if (specificAction) {
      const idx = [...highlightUndoStackRef.current]
        .reverse()
        .findIndex((item) => sameHighlightUndoAction(item, specificAction))
      if (idx < 0) return false
      const removeAt = highlightUndoStackRef.current.length - 1 - idx
      action = highlightUndoStackRef.current[removeAt]
      highlightUndoStackRef.current.splice(removeAt, 1)
    }
    if (!action) return false
    return applyHighlightUndoAction(action)
  }, [applyHighlightUndoAction])

  const addHighlightWithUndo = useCallback((highlight: ReaderSessionHighlight) => {
    const nextHighlight = enrichSessionHighlight(highlight)
    const nextId = String(nextHighlight?.id || '').trim()
    const alreadyExists = sessionHighlights.some((item) => (
      String(item.id || '').trim() === nextId || sameHighlightTarget(item, nextHighlight)
    ))
    if (alreadyExists) return
    onAddSessionHighlight?.(nextHighlight)
    highlightUndoStackRef.current.push({ kind: 'remove', highlight: nextHighlight })
  }, [enrichSessionHighlight, onAddSessionHighlight, sessionHighlights])

  const removeHighlightWithUndo = useCallback((highlightId: string) => {
    const targetId = String(highlightId || '').trim()
    if (!targetId) return
    const removed = sessionHighlights.find((item) => item.id === targetId) || null
    onRemoveSessionHighlight?.(targetId)
    setHighlightBubble(null)
    if (removed && onAddSessionHighlight) {
      const undoAction: HighlightUndoAction = { kind: 'restore', highlight: removed }
      highlightUndoStackRef.current.push(undoAction)
      message.open({
        type: 'success',
        content: (
          <span className="kb-reader-toast-content">
            <span>{S.reader_highlight_removed || 'Highlight removed'}</span>
            <button
              type="button"
              className="kb-reader-toast-action"
              onClick={() => undoHighlightAction(undoAction)}
            >
              {S.reader_undo || 'Undo'}
            </button>
          </span>
        ),
      })
      return
    }
    message.success(S.reader_highlight_removed || 'Highlight removed')
  }, [S, onAddSessionHighlight, onRemoveSessionHighlight, sessionHighlights, undoHighlightAction])

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

  const addSelectionToShelf = () => {
    const selected = selectionBubble
    const text = String(selected?.text || selection || '').trim()
    if (!selected || !text || !onAddSelectionToShelf) return
    onAddSelectionToShelf({
      text,
      sourcePath,
      sourceName: title,
      headingPath: String(activeHeadingPath || '').trim() || undefined,
      blockId: String(selected.blockId || activeBlockId || '').trim() || undefined,
      anchorId: String(selected.anchorId || activeAnchorId || '').trim() || undefined,
      anchorKind: String(activeAnchorKind || '').trim() || undefined,
      startOffset: selected.startOffset >= 0 ? selected.startOffset : undefined,
      endOffset: selected.endOffset > selected.startOffset ? selected.endOffset : undefined,
      occurrence: Number.isFinite(Number(selected.occurrence)) ? Number(selected.occurrence) : undefined,
      readableIndex: selected.readableIndex >= 0 ? selected.readableIndex : undefined,
      documentOccurrence: selected.documentOccurrence >= 0 ? selected.documentOccurrence : undefined,
      startReadableIndex: selected.startReadableIndex >= 0 ? selected.startReadableIndex : undefined,
      endReadableIndex: selected.endReadableIndex >= 0 ? selected.endReadableIndex : undefined,
      createdAt: Date.now(),
    })
    clearSelectionState(true)
  }

  const activeHighlightAction = highlightBubble
    ? sessionHighlights.find((item) => item.id === highlightBubble.highlightId) || null
    : null

  const setActiveHighlightFeedback = useCallback((feedback: 'useful' | 'needs_check') => {
    const item = activeHighlightAction
    if (!item || !onUpdateSessionHighlight) return
    const nextFeedback = item.feedback === feedback ? undefined : feedback
    const updated = enrichSessionHighlight({
      ...item,
      feedback: nextFeedback,
      feedbackAt: nextFeedback ? Date.now() : undefined,
    })
    onUpdateSessionHighlight(updated)
    message.success(S.reader_feedback_saved || 'Evidence note updated')
    setHighlightBubble(null)
  }, [S.reader_feedback_saved, activeHighlightAction, enrichSessionHighlight, onUpdateSessionHighlight])

  const appendActiveHighlight = () => {
    const item = activeHighlightAction
    const text = String(item?.text || '').trim()
    if (!item || !text) return
    const quoted = text.split('\n').map((line) => `> ${line}`).join('\n')
    const sourceLine = sourceLabel ? `> Source: ${sourceLabel}\n` : ''
    onAppendSelection(`${sourceLine}${quoted}\n\n`)
    setHighlightBubble(null)
  }

  const addActiveHighlightToShelf = activeHighlightAction && onAddSelectionToShelf
    ? () => {
      const item = activeHighlightAction
      const text = String(item.text || '').trim()
      if (!text) return
      onAddSelectionToShelf({
        text,
        sourcePath,
        sourceName: title,
        headingPath: String(item.headingPath || activeHeadingPath || '').trim() || undefined,
        blockId: String(item.blockId || activeBlockId || '').trim() || undefined,
        anchorId: String(item.anchorId || activeAnchorId || '').trim() || undefined,
        anchorKind: String(activeAnchorKind || '').trim() || undefined,
        startOffset: Number.isFinite(Number(item.startOffset ?? -1)) && Number(item.startOffset) >= 0 ? Number(item.startOffset) : undefined,
        endOffset: Number.isFinite(Number(item.endOffset ?? -1)) && Number(item.endOffset) >= 0 ? Number(item.endOffset) : undefined,
        occurrence: Number.isFinite(Number(item.occurrence)) ? Number(item.occurrence) : undefined,
        readableIndex: Number.isFinite(Number(item.readableIndex ?? -1)) && Number(item.readableIndex) >= 0 ? Number(item.readableIndex) : undefined,
        documentOccurrence: Number.isFinite(Number(item.documentOccurrence ?? -1)) && Number(item.documentOccurrence) >= 0 ? Number(item.documentOccurrence) : undefined,
        startReadableIndex: Number.isFinite(Number(item.startReadableIndex ?? -1)) && Number(item.startReadableIndex) >= 0 ? Number(item.startReadableIndex) : undefined,
        endReadableIndex: Number.isFinite(Number(item.endReadableIndex ?? -1)) && Number(item.endReadableIndex) >= 0 ? Number(item.endReadableIndex) : undefined,
        createdAt: Date.now(),
      })
      setHighlightBubble(null)
    }
    : undefined

  const removeActiveHighlight = () => {
    if (!activeHighlightAction) return
    removeHighlightWithUndo(activeHighlightAction.id)
  }

  const openHighlightMenuFromClick = (event: MouseEvent<HTMLDivElement>) => {
    const root = contentRef.current
    const target = event.target instanceof HTMLElement ? event.target : null
    const mark = target?.closest<HTMLElement>('.kb-reader-user-highlight') || null
    if (!root || !mark || !root.contains(mark)) {
      setHighlightBubble(null)
      return
    }
    const highlightId = String(mark.getAttribute('data-kb-session-highlight-id') || '').trim()
    const item = sessionHighlights.find((entry) => entry.id === highlightId) || null
    if (!item) {
      setHighlightBubble(null)
      return
    }
    event.preventDefault()
    event.stopPropagation()
    clearSelectionState(true)
    const rect = mark.getBoundingClientRect()
    const containerRect = root.getBoundingClientRect()
    const x = Math.max(18, Math.min(containerRect.width - 18, rect.left + (rect.width / 2) - containerRect.left))
    const aboveY = rect.top - containerRect.top - 10
    const belowY = rect.bottom - containerRect.top + 10
    const y = aboveY >= 16 ? aboveY : belowY
    setHighlightBubble({
      x,
      y,
      highlightId,
      text: String(item.text || '').trim(),
    })
  }

  const handleContentScroll = () => {
    queueSelectionStateSync()
    setHighlightBubble(null)
  }

  useEffect(() => {
    if (!highlightBubble) return
    if (sessionHighlights.some((item) => item.id === highlightBubble.highlightId)) return
    setHighlightBubble(null)
  }, [highlightBubble, sessionHighlights])

  useEffect(() => {
    highlightUndoStackRef.current = []
    setHighlightBubble(null)
  }, [open, sourcePath])

  useEffect(() => {
    if (!open) return undefined
    const handleKeyDown = (event: KeyboardEvent) => {
      const key = String(event.key || '').toLowerCase()
      const isUndo = (event.ctrlKey || event.metaKey) && !event.shiftKey && key === 'z'
      if (!isUndo || isEditableUndoTarget(event.target)) return
      if (!undoHighlightAction()) return
      event.preventDefault()
      event.stopPropagation()
      message.success(S.reader_undo_complete || 'Undone')
    }
    window.addEventListener('keydown', handleKeyDown, true)
    return () => {
      window.removeEventListener('keydown', handleKeyDown, true)
    }
  }, [S.reader_undo_complete, open, undoHighlightAction])

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
    onRemoveSessionHighlight: removeHighlightWithUndo,
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
      outlineToggleLabel={outlineOpen && !isPageSurface
        ? (S.reader_hide_sections || 'Hide sections')
        : (S.reader_sections || 'Sections')}
      highlightsToggleLabel={highlightsOpen && !isPageSurface
        ? (S.reader_hide_highlights || 'Hide highlights')
        : (S.reader_highlights_count || '{n} highlights').replace('{n}', String(sessionHighlights.length))}
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
      onAddSelectionToShelf={onAddSelectionToShelf ? addSelectionToShelf : undefined}
      onRemoveActiveHighlight={removeActiveHighlight}
      onAddActiveHighlightToShelf={addActiveHighlightToShelf}
      onSetActiveHighlightFeedback={onUpdateSessionHighlight ? setActiveHighlightFeedback : undefined}
      onAskActiveHighlight={appendActiveHighlight}
      onAskSelection={appendSelection}
      isInlinePresentation={isInlinePresentation}
      isPageSurface={isPageSurface}
      contentRef={contentRef}
      onContentClick={openHighlightMenuFromClick}
      onContentMouseUp={queueSelectionStateSync}
      onContentKeyUp={queueSelectionStateSync}
      onContentScroll={handleContentScroll}
    >
      {readerMarkdownNode}
    </PaperGuideReaderPanel>
  )
  const citationPopoverInShelf = Boolean(
    citationPopoverDetail && readerCitationShelfKeys.has(toShelfItem(citationPopoverDetail).key),
  )

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
