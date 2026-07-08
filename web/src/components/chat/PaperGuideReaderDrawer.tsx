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
import { useReaderHighlightWorkspace } from './reader/useReaderHighlightWorkspace'
import { useReaderEvidenceNavigator } from './reader/useReaderEvidenceNavigator'
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
  orderedEquationReaderBlocks,
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

  const title = resolvedName || sourceName || 'Document reader'
  const {
    addBlockToShelf: addReaderBlockToShelf,
    canAddBlockToShelf: canAddReaderBlockToShelf,
  } = useReaderBlockShelf({
    onAddSelectionToShelf,
    sourceName: title,
    sourcePath,
  })
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
  const visibleCandidateOptions = (() => {
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
  })()
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
  const activeCandidateDistinctKey = activeAlt
    ? candidateVisibilityKey(activeAlt, title) || candidateIdentityKey(activeAlt)
    : ''
  const requestedAltIndex = useMemo(() => {
    const hintIndex = Number(payload?.initialAltIndex || 0)
    return Number.isFinite(hintIndex) ? Math.max(0, Math.min(alternatives.length - 1, Math.floor(hintIndex))) : 0
  }, [payload, alternatives.length])
  const candidateOptions = (() => {
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
  })()
  const hasDistinctAlternatives = (() => {
    if (candidateOptions.length <= 1) return false
    const distinct = new Set(candidateOptions.map((item) => item.distinctKey).filter(Boolean))
    return distinct.size > 1
  })()
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
  const evidenceFocusText = compactReaderEvidenceText(activeHighlightSnippet || activeFocusSnippet)
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
    closeReaderCitationPopover()
  }, [closeReaderCitationPopover, open, sourcePath])

  useEffect(() => {
    if (!open || !onAddSelectionToShelf || !contentRef.current || !sourcePath) return undefined
    const root = contentRef.current
    const equationBlocks = orderedEquationReaderBlocks(readerBlocks)
    if (equationBlocks.length <= 0) return undefined
    const equationNodes = Array.from(root.querySelectorAll<HTMLElement>('.katex-display'))
    if (equationNodes.length <= 0) return undefined
    const cleanup: Array<() => void> = []
    const limit = Math.min(equationNodes.length, equationBlocks.length)
    const label = S.reader_add_to_shelf || 'Shelf'
    const titleLabel = S.reader_add_to_shelf_title || 'Add to research basket'
    const kindLabel = S.locate_badge_eq || 'Eq'
    for (let idx = 0; idx < limit; idx += 1) {
      const node = equationNodes[idx]
      const block = equationBlocks[idx]
      const blockId = String(block?.block_id || '').trim()
      const anchorId = String(block?.anchor_id || '').trim()
      if (!node || (!blockId && !anchorId)) continue
      if (node.querySelector(':scope > .kb-md-reader-block-shelf-tail[data-kb-imperative-equation="1"]')) continue
      node.classList.add('kb-md-reader-equation-action-host')
      node.setAttribute('data-kb-block-id', blockId)
      node.setAttribute('data-kb-anchor-id', anchorId)
      node.setAttribute('data-kb-anchor-kind', 'equation')
      const number = Number(block?.number || 0)
      if (Number.isFinite(number) && number > 0) {
        node.setAttribute('data-kb-anchor-number', String(Math.floor(number)))
      }
      const tail = document.createElement('span')
      tail.className = 'kb-md-reader-block-shelf-tail'
      tail.setAttribute('data-kb-imperative-equation', '1')
      const button = document.createElement('button')
      button.type = 'button'
      button.className = 'kb-md-reader-block-shelf kb-md-reader-block-shelf-equation'
      button.title = titleLabel
      button.setAttribute('aria-label', titleLabel)
      button.setAttribute('data-testid', 'reader-block-shelf')
      button.setAttribute('data-kb-reader-block-kind', 'equation')
      const kindSpan = document.createElement('span')
      kindSpan.className = 'kb-md-reader-block-shelf-kind'
      kindSpan.setAttribute('aria-hidden', 'true')
      kindSpan.textContent = kindLabel
      const textSpan = document.createElement('span')
      textSpan.className = 'kb-md-reader-block-shelf-text'
      textSpan.textContent = label
      button.append(kindSpan, textSpan)
      const handleClick = (event: Event) => {
        event.preventDefault()
        event.stopPropagation()
        const text = String(block.text || block.raw_text || node.textContent || '').replace(/\s+/g, ' ').trim()
        if (!text) return
        onAddSelectionToShelf({
          text: text.length <= 1400 ? text : `${text.slice(0, 1400).trimEnd()}...`,
          sourcePath,
          sourceName: title,
          headingPath: String(block.heading_path || '').trim() || undefined,
          blockId: blockId || undefined,
          anchorId: anchorId || undefined,
          anchorKind: 'equation',
          createdAt: Date.now(),
        })
      }
      button.addEventListener('click', handleClick)
      tail.appendChild(button)
      node.appendChild(tail)
      cleanup.push(() => {
        button.removeEventListener('click', handleClick)
        tail.remove()
        node.classList.remove('kb-md-reader-equation-action-host')
      })
    }
    return () => {
      cleanup.forEach((dispose) => dispose())
    }
  }, [
    S.locate_badge_eq,
    S.reader_add_to_shelf,
    S.reader_add_to_shelf_title,
    markdown,
    onAddSelectionToShelf,
    open,
    readerBlocks,
    sourcePath,
    title,
  ])

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

  const sourceLabel = [title, activeHeadingPath].filter(Boolean).join(' / ')
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
