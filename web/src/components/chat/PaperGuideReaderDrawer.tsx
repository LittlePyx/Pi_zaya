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
import type { ReaderDocResponse } from '../../api/references'
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

interface Props {
  open: boolean
  payload: ReaderOpenPayload | null
  onClose: () => void
  onAppendSelection: (text: string) => void
  presentation?: 'drawer' | 'inline'
  surface?: 'dock' | 'page'
  onCollapse?: () => void
  onOpenStandalone?: () => void
  sessionHighlights?: ReaderSessionHighlight[]
  onAddSessionHighlight?: (highlight: ReaderSessionHighlight) => void
  onRemoveSessionHighlight?: (highlightId: string) => void
  onLocateResult?: (result: ReaderLocateResult) => void
  onAddSelectionToShelf?: (payload: ReaderSelectionShelfPayload) => void
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
  sessionHighlights = [],
  onAddSessionHighlight,
  onRemoveSessionHighlight,
  onLocateResult,
  onAddSelectionToShelf,
  documentOverride,
}: Props) {
  const S = useT()
  const contentRef = useRef<HTMLDivElement>(null)
  const [drawerReady, setDrawerReady] = useState(false)
  const [altChangeSource, setAltChangeSource] = useState<'system' | 'manual'>('system')
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
    contentRef,
    sessionHighlights,
    onAddSessionHighlight,
    onRemoveSessionHighlight,
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
      onToggleSelectionHighlight={toggleSelectionHighlight}
      onAddSelectionToShelf={onAddSelectionToShelf ? addSelectionToShelf : undefined}
      onAskSelection={appendSelection}
      isInlinePresentation={isInlinePresentation}
      isPageSurface={isPageSurface}
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
    </PaperGuideReaderShell>
  )
}
