import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type MouseEvent } from 'react'
import { createPortal } from 'react-dom'
import { Typography, message } from 'antd'
import { UserOutlined } from '@ant-design/icons'
import { MarkdownRenderer } from './MarkdownRenderer'
import { CopyBar } from './CopyBar'
import { CitationPopover } from './CitationPopover'
import { CiteShelf } from './CiteShelf'
import type {
  ReaderLocateResult,
  ReaderOpenPayload,
} from './reader/readerTypes'
import {
  READER_CITATION_SHELF_CHANNEL,
  READER_CITATION_SHELF_EVENT,
  READER_SELECTION_SHELF_CHANNEL,
  READER_SELECTION_SHELF_EVENT,
} from './reader/readerTypes'
import { buildBasicReaderOpenPayload } from './reader/readerOpenPayloadUtils'
import {
  buildGuideLocateCandidates,
  buildRefsLocateCandidatesAll,
  dedupeLocateCandidates,
  hasFormulaSignal,
  normalizeLocateText,
  stripMarkdownInline,
  stripProvenanceNoise,
  type LocateCandidate,
  type RefHitLite,
} from './reader/messageLocateCandidates'
import {
  buildHeuristicReaderOpenPayload,
  buildStructuredEntryReaderOpenPayload,
  toPositiveIntOrUndefined,
} from './reader/messageReaderLocatePayload'
import {
  buildStructuredProvenanceLocateEntries,
  isLikelyRhetoricalLocateShell,
  listStructuredProvenanceSegments,
  shortSegmentLabel,
  type ProvenanceLocateEntry,
  type StructuredProvenanceSegment,
} from './reader/messageStructuredProvenance'
import {
  buildStructuredRenderLocateSlotMap,
  createStructuredInlineLocateResolver,
  isEquationLocateCandidate,
  type StructuredRenderLocateSlot,
} from './reader/messageStructuredInlineLocate'
import {
  extractEquationNumbersFromText,
  extractFigureNumbersFromText,
  isPreferredStrictFigureRefSnippet,
  normalizeStructuredLocateKind,
  panelLetterMatchScore,
  scoreLocateCandidate,
  scoreProvenanceSegment,
} from './reader/messageStructuredLocateScoring'
import {
  citationDisplay,
  mergeCiteMeta,
  normalizeCiteDetail,
  normalizeShelfNote,
  normalizeShelfTags,
  shelfProjectScopeId,
  shelfItemNeedsMetadataRepair,
  shelfItemRepairFingerprint,
  shelfStorageKey,
  strictRepairMerge,
  toShelfItem,
  type CiteDetail,
  type CiteShelfItem,
} from './citationState'
import {
  SHELF_MAX_ITEMS,
  articleSummaryPatchFromMeta,
  dedupeShelfItems,
  looksLowValueShelfSummary,
  mergeCitationDetailIntoShelfItems,
  mergeReaderSelectionDetailIntoShelfItems,
  mergeShelfItemWithLive,
  sameShelfItem,
  sameShelfItems,
  shelfItemHasDisplayableArticleSummary,
  shelfItemNeedsPersistedMetadataHydrate,
  shelfItemNeedsSummaryBackfill,
  shelfItemsForBackend,
  shelfMetadataHydrateAttemptKey,
  shelfPaperIdentity,
  shelfRepairMetaFromEntry,
  shelfRepairPayloads,
  shelfSummaryBackfillAttemptKey,
  shouldRequestCitationCardPolish,
  snapshotDiffCounts,
} from './citeShelfRuntime'
import {
  SHELF_SAVED_MAX_ITEMS,
  SHELF_SAVED_SUFFIX,
  invalidateSavedShelfSnapshotCache,
  invalidateShelfSnapshotCache,
  legacyShelfStorageKeys,
  migrateLegacySavedShelfSnapshots,
  migrateLegacyShelfSnapshot,
  persistSavedShelfSnapshots,
  persistShelfSnapshot,
  readSavedShelfSnapshots,
  readShelfSnapshot,
  restoreShelfItems,
  shelfSavedStorageKey,
  type ShelfSavedSnapshot,
} from './citeShelfStorage'
import {
  citeDetailFromReaderSelection,
  normalizeReaderCitationShelfPayload,
  normalizeReaderSelectionShelfPayload,
  readerSelectionNote,
} from './readerShelfPayload'
import { RefsPanel } from '../refs/RefsPanel'
import { hasRefsPanelContent } from '../refs/refsPanelDisplay'
import { chatApi, type Message } from '../../api/chat'
import { referencesApi, type ShelfMetadataRepairImpact } from '../../api/references'
import { useT } from '../../i18n'
import { useChatStore } from '../../stores/chatStore'
import { basenameFromSourcePath } from '../../utils/sourcePath'
import {
  cleanAssistantAnswerPresentationText,
  getMessageCiteDetailRecords,
  getMessageCopyMarkdownValue,
  getMessageCopyTextValue,
  getMessageNoticeValue,
  getMessageRenderedBodyContent,
  getMessageRenderPacket,
} from './messageRenderPacket'
import { buildRenderPacketLocateEntry } from './messageRenderPacketLocate'
import {
  buildSelectedResearchContextPack,
  buildSelectedResearchContextPackFromItems,
  type SelectedResearchContextPack,
  type SelectedResearchContextItem,
} from './researchContextPack'
import {
  messageListPerfNow,
  pushMessageListPrepPerf,
  type MessageListPrepPerfEvent,
} from './messageListPerf'
import { withBibliometricsLocale } from './bibliometricsLocale'
import {
  buildUnlinkedReferenceViews,
  enrichCiteDetailsWithVisibleRefContext,
} from './messageCitationViews'
import { buildFallbackCiteDetailsFromRefHits } from './messageFallbackCitations'
import { resolveLowConfidenceMeta, stripLeadingLowConfidenceNotice } from './messageLowConfidence'
import { compactHeadingPath, extractQuotedSpans, quoteMatchStats } from './messageQuoteUtils'
import {
  lookupGuideCandidatesBySourcePath,
  sourcePathLookupKeys,
  sourcePathsReferToSameDocument,
} from './messageSourceIdentity'
import {
  contextItemTitle,
  getAssistantSelectedResearchContext,
  getMessageAgentTrace,
  getMessageResearchTrace,
  getUserPromptResearchContext,
  imageAttachmentsOf,
  isImageOnlyPlaceholder,
  messageHasAgentTraceHint,
} from './messageTraceUtils'
import { AgentTracePanel } from './AgentTracePanel'
import { ResearchTracePanel } from './ResearchTracePanel'
import { ResearchContextReceipt } from './ResearchContextReceipt'

const { Text } = Typography
const SHELF_BACKEND_PERSIST_MS = 320
const SHELF_AUTO_REPAIR_BATCH_SIZE = 8
const SHELF_AUTO_REPAIR_RETRY_MS = 15000
const SHELF_METADATA_HYDRATE_BATCH_SIZE = 8
const SHELF_METADATA_HYDRATE_RETRY_MS = 15000
const SHELF_SUMMARY_BACKFILL_BATCH_SIZE = 4
const SHELF_SUMMARY_BACKFILL_RETRY_MS = 60000

type ShelfAsyncScopeToken = {
  epoch: number
  storageKey: string
}

export interface ShelfActivityState {
  summary: boolean
  repair: boolean
  autoRepair: boolean
  background: boolean
  count: number
}

interface Props {
  activeConvId?: string | null
  shelfProjectId?: string | null
  messages: Message[]
  refs: Record<string, unknown>
  generationPartial?: string
  generationStage?: string
  generationTrace?: Record<string, unknown>
  generationAgentTrace?: Record<string, unknown>
  jumpTarget?: { messageId: number; token: number } | null
  onJumpHandled?: (jumpTarget: { messageId: number; token: number }) => void
  trackedMessageIds?: number[]
  onTrackedMessageActive?: (messageId: number | null) => void
  onOpenReader?: (payload: ReaderOpenPayload) => void
  onShelfOpenChange?: (open: boolean) => void
  onShelfStateChange?: (state: { open: boolean; count: number }) => void
  onShelfActivityChange?: (state: ShelfActivityState) => void
  closeShelfSignal?: number
  openShelfSignal?: number
  shelfDockMode?: boolean
  shelfPortalTarget?: HTMLElement | null
  shelfVisible?: boolean
  readerLocateResults?: Record<string, ReaderLocateResult>
  sourceQualityRefreshToken?: number
  paperGuideSourcePath?: string
  paperGuideSourceName?: string
  selectedResearchContextKeys?: Record<string, boolean>
  onResearchContextPackChange?: (pack: SelectedResearchContextPack | null) => void
  onResearchContextFollowUp?: (pack: SelectedResearchContextPack, promptText: string) => void
}

interface RefEntryLite {
  hits?: RefHitLite[]
  display_state?: string
  suppression_reason?: string
  suggestion?: string
  guide_filter?: { hidden_self_source?: boolean; filtered_hit_count?: number }
}

interface AssistantLocatePrep {
  bodyContent: string
  refsUserMsgId: number
  locateSourcePath: string
  locateSourceName: string
  refsLocateCandidatesAll: LocateCandidate[]
  guideLocateCandidates: LocateCandidate[]
  refsScopedCandidates: LocateCandidate[]
  messageProvenance: Record<string, unknown> | null
  provenanceSourcePath: string
  provenanceSourceName: string
  provenanceBlockMap: Record<string, Record<string, unknown>>
  provenanceDirectSegments: Array<Record<string, unknown>>
  hasDirectProvenance: boolean
  hasStructuredProvenance: boolean
  effectiveGuideSourcePath: string
  strictProvenanceLocate: boolean
  structuredLocateButtonCap: number
  provenanceLocateEntries: ProvenanceLocateEntry[]
  structuredProvenanceSegmentsAll: StructuredProvenanceSegment[]
  provenanceStrictIdentityReady: boolean
  hasStrictMustLocateEntries: boolean
  strictStructuredLocateOnly: boolean
  strictStructuredInlineLocate: boolean
  provenanceModeLabel: string
  structuredRenderSlotMap: Map<number, StructuredRenderLocateSlot>
  structuredLocateOrderBySegmentId: Map<string, number>
  allowedStructuredRenderOrders: Set<number>
  locateCandidates: LocateCandidate[]
}

function createEmptyAssistantLocatePrep(bodyContent: string, refsUserMsgId = 0): AssistantLocatePrep {
  return {
    bodyContent,
    refsUserMsgId,
    locateSourcePath: '',
    locateSourceName: '',
    refsLocateCandidatesAll: [],
    guideLocateCandidates: [],
    refsScopedCandidates: [],
    messageProvenance: null,
    provenanceSourcePath: '',
    provenanceSourceName: '',
    provenanceBlockMap: {},
    provenanceDirectSegments: [],
    hasDirectProvenance: false,
    hasStructuredProvenance: false,
    effectiveGuideSourcePath: '',
    strictProvenanceLocate: false,
    structuredLocateButtonCap: 12,
    provenanceLocateEntries: [],
    structuredProvenanceSegmentsAll: [],
    provenanceStrictIdentityReady: false,
    hasStrictMustLocateEntries: false,
    strictStructuredLocateOnly: false,
    strictStructuredInlineLocate: false,
    provenanceModeLabel: '',
    structuredRenderSlotMap: new Map<number, StructuredRenderLocateSlot>(),
    structuredLocateOrderBySegmentId: new Map<string, number>(),
    allowedStructuredRenderOrders: new Set<number>(),
    locateCandidates: [],
  }
}

function hasRenderableRefsForGuide(
  refs: Record<string, unknown>,
  msgId: number,
  opts: { activeSourcePath?: string; activeSourceName?: string },
) {
  return hasRefsPanelContent(refs, msgId, opts)
}

function countDistinctDocLabels(text: string): number {
  const seen = new Set<string>()
  for (const match of String(text || '').matchAll(/\bDOC-(\d{1,4})\b/gi)) {
    const docId = String(match[1] || '').trim()
    if (docId) seen.add(docId)
  }
  return seen.size
}

function shouldSuppressLooseInlineLocate(args: {
  guideSourcePath: string
  bodyContent: string
  hasRawCiteDetails: boolean
  hasStructuredProvenance: boolean
  hasDirectProvenance: boolean
}): boolean {
  if (String(args.guideSourcePath || '').trim()) return false
  if (args.hasRawCiteDetails) return false
  if (args.hasStructuredProvenance) return false
  if (args.hasDirectProvenance) return false
  return countDistinctDocLabels(args.bodyContent) >= 2
}

function messageLocatePayloadSignature(message: Message, renderPacketValue: unknown): string {
  const provenance = (message.provenance && typeof message.provenance === 'object')
    ? message.provenance as Record<string, unknown>
    : null
  const renderPacket = renderPacketValue && typeof renderPacketValue === 'object'
    ? renderPacketValue as Record<string, unknown>
    : null
  if (!provenance && !renderPacket) return 'no-locate-payload'

  const provenanceSig = (() => {
    if (!provenance) return ''
    const segments = Array.isArray(provenance.segments) ? provenance.segments : []
    const segmentSig = segments
      .slice(0, 24)
      .map((item, idx) => {
        const seg = item && typeof item === 'object' ? item as Record<string, unknown> : {}
        const evidenceIds = Array.isArray(seg.evidence_block_ids)
          ? seg.evidence_block_ids.map((value) => String(value || '').trim()).filter(Boolean).slice(0, 6)
          : []
        return [
          String(seg.segment_id || idx).trim(),
          String(seg.evidence_mode || '').trim(),
          String(seg.locate_policy || '').trim(),
          String(seg.locate_surface_policy || '').trim(),
          seg.must_locate ? 'must' : '',
          String(seg.primary_block_id || '').trim(),
          String(seg.primary_anchor_id || '').trim(),
          evidenceIds.join(','),
        ].join(':')
      })
      .join(';')
    const blockMap = provenance.block_map && typeof provenance.block_map === 'object'
      ? provenance.block_map as Record<string, unknown>
      : {}
    return [
      String(provenance.status || '').trim(),
      String(provenance.mapping_mode || '').trim(),
      provenance.strict_identity_ready ? 'strict-ready' : 'strict-pending',
      String(provenance.source_path || '').trim(),
      String(provenance.md_path || '').trim(),
      Number(provenance.must_locate_count || 0) || 0,
      Number(provenance.must_locate_candidate_count || 0) || 0,
      Number(provenance.strict_identity_count || 0) || 0,
      segments.length,
      Object.keys(blockMap).sort().slice(0, 48).join(','),
      segmentSig,
    ].join('|')
  })()

  const renderPacketSig = (() => {
    if (!renderPacket) return ''
    const locateTarget = (
      renderPacket.locateTarget
      || renderPacket.locate_target
      || null
    ) as Record<string, unknown> | null
    const readerOpen = (
      renderPacket.readerOpen
      || renderPacket.reader_open
      || null
    ) as Record<string, unknown> | null
    const segmentIds = Array.isArray(renderPacket.segment_ids)
      ? renderPacket.segment_ids.map((item) => String(item || '').trim()).filter(Boolean).slice(0, 24)
      : []
    const visibleSegmentIds = Array.isArray(renderPacket.visible_segment_ids)
      ? renderPacket.visible_segment_ids.map((item) => String(item || '').trim()).filter(Boolean).slice(0, 24)
      : []
    return [
      segmentIds.join(','),
      visibleSegmentIds.join(','),
      String(locateTarget?.blockId || locateTarget?.block_id || '').trim(),
      String(locateTarget?.anchorId || locateTarget?.anchor_id || '').trim(),
      String(locateTarget?.locatePolicy || locateTarget?.locate_policy || '').trim(),
      String(readerOpen?.blockId || readerOpen?.block_id || '').trim(),
      String(readerOpen?.anchorId || readerOpen?.anchor_id || '').trim(),
    ].join('|')
  })()

  return `${provenanceSig}::${renderPacketSig}`
}

function getStructuredEntryRemapTarget(
  entry: ProvenanceLocateEntry,
  primary: LocateCandidate,
): {
  targetKind: 'equation' | 'figure' | ''
  targetNumber: number
  panelLetters: string[]
  seed: string
} {
  const claimType = String(entry.claimType || '').trim().toLowerCase()
  const targetKind: 'equation' | 'figure' | '' = (() => {
    if (claimType === 'formula_claim' || claimType === 'inline_formula_claim' || claimType === 'equation_explanation_claim') {
      return 'equation'
    }
    if (claimType === 'figure_claim' || claimType === 'figure_panel') {
      return 'figure'
    }
    const rawKind = normalizeStructuredLocateKind(String(entry.anchorKind || primary.anchorKind || ''))
    return rawKind === 'equation' || rawKind === 'figure' ? rawKind : ''
  })()
  const targetNumber = (() => {
    if (targetKind === 'equation') {
      const eqNumbers = extractEquationNumbersFromText(
        `${entry.anchorText || ''} ${entry.evidenceQuote || ''} ${entry.segmentText || ''} ${primary.headingPath || ''}`,
      )
      const merged = [
        Number(entry.equationNumber || 0),
        Number(primary.anchorNumber || 0),
        ...eqNumbers,
      ].filter((item) => Number.isFinite(item) && Number(item) > 0)
      return merged.length > 0 ? Math.floor(Number(merged[0])) : 0
    }
    const figNumbers = extractFigureNumbersFromText(
      `${entry.anchorText || ''} ${entry.evidenceQuote || ''} ${entry.segmentText || ''} ${primary.headingPath || ''}`,
    )
    const merged = [
      Number(entry.supportFigureNumber || 0),
      Number(primary.anchorNumber || 0),
      ...figNumbers,
    ].filter((item) => Number.isFinite(item) && Number(item) > 0)
    return merged.length > 0 ? Math.floor(Number(merged[0])) : 0
  })()
  const panelLetters = Array.isArray(entry.supportPanelLetters)
    ? entry.supportPanelLetters.map((item) => String(item || '').trim().toLowerCase()).filter((item) => /^[a-z]$/.test(item))
    : []
  const seed = stripProvenanceNoise(
    stripMarkdownInline(String(entry.anchorText || entry.evidenceQuote || entry.segmentText || primary.focusSnippet || '')),
  ).trim()
  return {
    targetKind,
    targetNumber,
    panelLetters,
    seed,
  }
}

function getScopedGuideCandidatesForRemap(
  primary: LocateCandidate,
  guideCandidates: LocateCandidate[],
): LocateCandidate[] {
  const sourcePath = String(primary.sourcePath || '').trim()
  return (guideCandidates || []).filter((cand) => {
    if (!cand || typeof cand !== 'object') return false
    if (String(cand.sourceType || '').trim().toLowerCase() !== 'guide') return false
    const candSourcePath = String(cand.sourcePath || '').trim()
    if (sourcePath && candSourcePath && !sourcePathsReferToSameDocument(sourcePath, candSourcePath)) {
      return false
    }
    return Boolean(String(cand.blockId || cand.anchorId || '').trim())
  })
}

function findGuideCandidateIdentityMatch(
  primary: LocateCandidate,
  guideCandidates: LocateCandidate[],
): LocateCandidate | null {
  const primaryBlockId = String(primary.blockId || '').trim()
  const primaryAnchorId = String(primary.anchorId || '').trim()
  if (!(primaryBlockId || primaryAnchorId)) return null
  for (const cand of guideCandidates) {
    if (!cand || typeof cand !== 'object') continue
    const candBlockId = String(cand.blockId || '').trim()
    const candAnchorId = String(cand.anchorId || '').trim()
    if (primaryBlockId && candBlockId && candBlockId === primaryBlockId) return cand
    if (primaryAnchorId && candAnchorId && candAnchorId === primaryAnchorId) return cand
  }
  return null
}

function inferLocateCandidateTargetNumber(
  cand: LocateCandidate,
  targetKind: 'equation' | 'figure',
): number {
  const anchorNumber = Number(cand.anchorNumber || 0)
  if (Number.isFinite(anchorNumber) && anchorNumber > 0) {
    return Math.floor(anchorNumber)
  }
  const raw = `${cand.headingPath || ''} ${cand.focusSnippet || ''} ${cand.matchText || ''}`
  const nums = targetKind === 'equation'
    ? extractEquationNumbersFromText(raw)
    : extractFigureNumbersFromText(raw)
  return nums.length > 0 ? Math.floor(Number(nums[0])) : 0
}

function isGuideCandidateCanonicalForEntry(
  cand: LocateCandidate | null,
  opts: {
    targetKind: 'equation' | 'figure'
    targetNumber: number
  },
): boolean {
  if (!cand) return false
  const targetKind = opts.targetKind
  const targetNumber = opts.targetNumber
  const candKind = normalizeStructuredLocateKind(String(cand.anchorKind || ''))
  if (candKind !== targetKind) return false
  if (targetNumber > 0) {
    const candNumber = inferLocateCandidateTargetNumber(cand, targetKind)
    if (candNumber !== targetNumber) return false
  }
  return true
}

function remapStructuredEntryToGuideAnchors(
  entry: ProvenanceLocateEntry,
  guideCandidates: LocateCandidate[],
): ProvenanceLocateEntry {
  const primary = entry.primary
  if (!primary) return entry
  const scoped = getScopedGuideCandidatesForRemap(primary, guideCandidates)
  if (scoped.length <= 0) return entry

  const { targetKind, targetNumber, panelLetters, seed } = getStructuredEntryRemapTarget(entry, primary)
  if (!targetKind) return entry
  const primaryIdentityMatch = findGuideCandidateIdentityMatch(primary, scoped)
  if (isGuideCandidateCanonicalForEntry(primaryIdentityMatch, { targetKind, targetNumber })) {
    return entry
  }

  let best: LocateCandidate | null = null
  let bestScore = Number.NEGATIVE_INFINITY
  for (const cand of scoped) {
    const candKind = normalizeStructuredLocateKind(String(cand.anchorKind || ''))
    let score = scoreLocateCandidate(seed || String(primary.focusSnippet || ''), cand)
    if (candKind === targetKind) score += 1.22
    else if (candKind) score -= 1.08
    if (targetNumber > 0) {
      const candNumber = Number.isFinite(Number(cand.anchorNumber || 0))
        ? Math.floor(Number(cand.anchorNumber || 0))
        : 0
      if (candNumber === targetNumber) score += 1.48
      else if (candNumber > 0) score -= 0.46
    }
    if (targetKind === 'figure') {
      if (String(cand.headingPath || '').toLowerCase().includes('figure')) score += 0.22
      if (panelLetters.length > 0) {
        score += 0.28 * panelLetterMatchScore(
          `${cand.headingPath || ''} ${cand.focusSnippet || ''} ${cand.matchText || ''}`,
          panelLetters,
        )
      }
    }
    if (targetKind === 'equation' && hasFormulaSignal(String(cand.focusSnippet || cand.matchText || ''))) {
      score += 0.2
    }
    if (String(cand.blockId || '').trim() === String(primary.blockId || '').trim()) score += 0.08
    if (String(cand.anchorId || '').trim() === String(primary.anchorId || '').trim()) score += 0.06
    if (score > bestScore) {
      best = cand
      bestScore = score
    }
  }

  const acceptFloor = targetNumber > 0 ? 0.48 : 0.7
  if (!best || bestScore < acceptFloor) return entry
  const sameIdentity = (
    String(best.blockId || '').trim() === String(primary.blockId || '').trim()
    && String(best.anchorId || '').trim() === String(primary.anchorId || '').trim()
  )
  if (sameIdentity) return entry

  const relatedBlockIds = Array.from(new Set([
    ...((entry.relatedBlockIds || []).map((item) => String(item || '').trim()).filter(Boolean)),
    ...((String(primary.blockId || '').trim() && String(primary.blockId || '').trim() !== String(best.blockId || '').trim())
      ? [String(primary.blockId || '').trim()]
      : []),
  ]))
  const remappedAnchorKind = String(best.anchorKind || entry.anchorKind || entry.locateTarget?.anchorKind || entry.readerOpen?.anchorKind || '').trim().toLowerCase() || undefined
  const remappedAnchorNumber = toPositiveIntOrUndefined(
    best.anchorNumber
    || entry.equationNumber
    || entry.supportFigureNumber
    || entry.locateTarget?.anchorNumber
    || entry.readerOpen?.anchorNumber
    || 0,
  )
  const remappedLocateTarget = (() => {
    const baseLocateTarget = entry.locateTarget || entry.readerOpen?.locateTarget || null
    if (!baseLocateTarget) return entry.locateTarget
    return {
      ...baseLocateTarget,
      headingPath: String(best.headingPath || baseLocateTarget.headingPath || '').trim() || undefined,
      blockId: String(best.blockId || baseLocateTarget.blockId || '').trim() || undefined,
      anchorId: String(best.anchorId || baseLocateTarget.anchorId || '').trim() || undefined,
      anchorKind: remappedAnchorKind || baseLocateTarget.anchorKind,
      anchorNumber: remappedAnchorNumber ?? baseLocateTarget.anchorNumber,
      relatedBlockIds: relatedBlockIds.length > 0 ? relatedBlockIds : baseLocateTarget.relatedBlockIds,
    }
  })()
  const remappedReaderOpen = (() => {
    if (!entry.readerOpen) return entry.readerOpen
    return {
      ...entry.readerOpen,
      headingPath: String(best.headingPath || entry.readerOpen.headingPath || '').trim() || undefined,
      blockId: String(best.blockId || entry.readerOpen.blockId || '').trim() || undefined,
      anchorId: String(best.anchorId || entry.readerOpen.anchorId || '').trim() || undefined,
      relatedBlockIds: relatedBlockIds.length > 0 ? relatedBlockIds : entry.readerOpen.relatedBlockIds,
      anchorKind: remappedAnchorKind || entry.readerOpen.anchorKind,
      anchorNumber: remappedAnchorNumber ?? entry.readerOpen.anchorNumber,
      locateTarget: remappedLocateTarget || entry.readerOpen.locateTarget,
    }
  })()
  return {
    ...entry,
    primary: best,
    alternatives: dedupeLocateCandidates([best, primary, ...(entry.alternatives || [])]),
    relatedBlockIds: relatedBlockIds.length > 0 ? relatedBlockIds : entry.relatedBlockIds,
    locateTarget: remappedLocateTarget || entry.locateTarget,
    readerOpen: remappedReaderOpen || entry.readerOpen,
  }
}

function AssistantAvatar() {
  return (
    <div className="kb-msg-avatar kb-msg-avatar-assistant">
      <img src="/pi_logo.png" alt="Pi assistant" className="h-5 w-5 object-contain" loading="lazy" />
    </div>
  )
}

export function MessageList({
  activeConvId,
  shelfProjectId,
  messages,
  refs,
  generationPartial,
  generationStage,
  generationTrace,
  generationAgentTrace,
  jumpTarget,
  onJumpHandled,
  trackedMessageIds,
  onTrackedMessageActive,
  onOpenReader,
  onShelfOpenChange,
  onShelfStateChange,
  onShelfActivityChange,
  closeShelfSignal = 0,
  openShelfSignal = 0,
  shelfDockMode = false,
  shelfPortalTarget = null,
  shelfVisible,
  readerLocateResults = {},
  sourceQualityRefreshToken = 0,
  paperGuideSourcePath,
  paperGuideSourceName,
  selectedResearchContextKeys = {},
  onResearchContextPackChange,
  onResearchContextFollowUp,
}: Props) {
  const createPaperGuideConversation = useChatStore((s) => s.createPaperGuideConversation)
  const scrollRef = useRef<HTMLDivElement>(null)
  const [popoverDetail, setPopoverDetail] = useState<CiteDetail | null>(null)
  const [popoverPos, setPopoverPos] = useState<{ x: number; y: number } | null>(null)
  const [popoverLoading, setPopoverLoading] = useState(false)
  const [popoverGuideLoading, setPopoverGuideLoading] = useState(false)
  const [popoverPinned, setPopoverPinned] = useState(false)
  const citationHoverOpenTimerRef = useRef<number | null>(null)
  const citationHoverCloseTimerRef = useRef<number | null>(null)
  const citationPolishRetryTimerRef = useRef<number | null>(null)
  const activePopoverRequestKeyRef = useRef('')
  const citationPolishPrewarmKeysRef = useRef(new Set<string>())
  const [shelfOpen, setShelfOpen] = useState(false)
  const [shelfItems, setShelfItems] = useState<CiteShelfItem[]>([])
  const [focusedShelfKey, setFocusedShelfKey] = useState('')
  const [shelfSummaryLoadingKey, setShelfSummaryLoadingKey] = useState('')
  const [shelfRepairLoadingKey, setShelfRepairLoadingKey] = useState('')
  const [shelfAutoRepairingKeys, setShelfAutoRepairingKeys] = useState<string[]>([])
  const [shelfBackgroundBusy, setShelfBackgroundBusy] = useState(false)
  const [shelfRepairImpact, setShelfRepairImpact] = useState<ShelfMetadataRepairImpact | null>(null)
  const [savedShelfSnapshots, setSavedShelfSnapshots] = useState<ShelfSavedSnapshot[]>([])
  const [selectedSavedSnapshotId, setSelectedSavedSnapshotId] = useState('')
  const [shelfMessageFlashId, setShelfMessageFlashId] = useState<number | null>(null)
  const assistantLocatePrepCacheRef = useRef(new Map<string, AssistantLocatePrep>())
  const assistantLocatePrepPerfRef = useRef<MessageListPrepPerfEvent | null>(null)
  const [guideDocCandidates, setGuideDocCandidates] = useState<LocateCandidate[]>([])
  const S = useT()
  const shelfScopeId = shelfProjectScopeId(shelfProjectId)
  const skipShelfPersistOnceRef = useRef(false)
  const persistShelfTimerRef = useRef<number | null>(null)
  const persistShelfBackendTimerRef = useRef<number | null>(null)
  const activeStorageKeyRef = useRef(shelfStorageKey(shelfScopeId))
  const shelfRevisionByKeyRef = useRef<Record<string, number>>({})
  const shelfBackendRevisionByKeyRef = useRef<Record<string, number>>({})
  const shelfBackendHydratedKeysRef = useRef(new Set<string>())
  const shelfEmptyBackendSaveIntentRef = useRef<Record<string, number>>({})
  const shelfBackendHydrateSeqRef = useRef(0)
  const shelfAsyncScopeEpochRef = useRef(0)
  const shelfStateTouchedAtRef = useRef(Date.now())
  const latestShelfStateRef = useRef<{ convId?: string | null; projectId?: string | null; open: boolean; items: CiteShelfItem[] }>({
    convId: activeConvId,
    projectId: shelfScopeId,
    open: false,
    items: [],
  })
  const flushShelfSnapshotRef = useRef<(() => void) | null>(null)
  const flushShelfBackendRef = useRef<(() => void) | null>(null)
  const shelfAutoRepairTimerRef = useRef<number | null>(null)
  const shelfAutoRepairingKeySetRef = useRef(new Set<string>())
  const shelfAutoRepairFingerprintsRef = useRef<Record<string, string>>({})
  const shelfAutoRepairRetryAfterRef = useRef<Record<string, number>>({})
  const shelfMetadataHydrateTimerRef = useRef<number | null>(null)
  const shelfMetadataHydrateInFlightRef = useRef(new Set<string>())
  const shelfMetadataHydrateAttemptedAtRef = useRef<Record<string, number>>({})
  const shelfSummaryBackfillTimerRef = useRef<number | null>(null)
  const shelfSummaryBackfillInFlightRef = useRef(new Set<string>())
  const shelfSummaryBackfillAttemptedAtRef = useRef<Record<string, number>>({})
  const shelfMessageFlashTimerRef = useRef<number | null>(null)
  const setShelfAutoRepairingKeySet = useCallback((nextSet: Set<string>) => {
    shelfAutoRepairingKeySetRef.current = nextSet
    setShelfAutoRepairingKeys(Array.from(nextSet))
  }, [])
  const captureShelfAsyncScope = useCallback((): ShelfAsyncScopeToken => ({
    epoch: shelfAsyncScopeEpochRef.current,
    storageKey: shelfStorageKey(shelfScopeId),
  }), [shelfScopeId])
  const shelfAsyncScopeIsCurrent = useCallback((token: ShelfAsyncScopeToken): boolean => (
    shelfAsyncScopeEpochRef.current === token.epoch
    && shelfStorageKey(latestShelfStateRef.current.projectId) === token.storageKey
  ), [])
  const currentShelfItemForAsync = useCallback((
    token: ShelfAsyncScopeToken,
    itemKey: string,
    expectedRepairFingerprint?: string,
  ): CiteShelfItem | null => {
    if (!shelfAsyncScopeIsCurrent(token)) return null
    const key = String(itemKey || '').trim()
    if (!key) return null
    const current = latestShelfStateRef.current.items.find((entry) => entry.key === key)
    if (!current) return null
    if (expectedRepairFingerprint && shelfItemRepairFingerprint(current) !== expectedRepairFingerprint) return null
    return current
  }, [shelfAsyncScopeIsCurrent])

  const persistShelfLocalNow = useCallback((items: CiteShelfItem[], open: boolean) => {
    const storageKey = shelfStorageKey(shelfScopeId)
    const currentRevision = Number(shelfRevisionByKeyRef.current[storageKey] || 0)
    const nextItems = dedupeShelfItems(items).slice(0, SHELF_MAX_ITEMS)
    const nextRevision = persistShelfSnapshot(storageKey, { open, items: nextItems }, currentRevision)
    shelfRevisionByKeyRef.current[storageKey] = nextRevision
    activeStorageKeyRef.current = storageKey
    latestShelfStateRef.current = {
      convId: activeConvId,
      projectId: shelfScopeId,
      open,
      items: nextItems,
    }
  }, [activeConvId, shelfScopeId])

  const markShelfEmptyBackendSaveIntent = useCallback((projectId?: string | null) => {
    const storageKey = shelfStorageKey(projectId)
    shelfEmptyBackendSaveIntentRef.current[storageKey] = Date.now() + 5000
  }, [])

  const saveShelfBackendNow = useCallback((
    state: { convId?: string | null; projectId?: string | null; open: boolean; items: CiteShelfItem[] },
    options?: { allowEmptyOverwrite?: boolean },
  ) => {
    const projectScopeId = shelfProjectScopeId(state.projectId)
    const storageKey = shelfStorageKey(projectScopeId)
    if (!shelfBackendHydratedKeysRef.current.has(storageKey)) return
    const items = shelfItemsForBackend(state.items)
    const emptyClosed = items.length <= 0 && !state.open
    const emptyIntentUntil = Number(shelfEmptyBackendSaveIntentRef.current[storageKey] || 0)
    const allowEmptyOverwrite = Boolean(options?.allowEmptyOverwrite || emptyIntentUntil > Date.now())
    if (emptyClosed && !allowEmptyOverwrite && Number(shelfBackendRevisionByKeyRef.current[storageKey] || 0) > 0) {
      return
    }
    void chatApi.saveCitationShelf({
      convId: state.convId || undefined,
      projectId: projectScopeId === '__default__' ? undefined : projectScopeId,
      scope: 'project',
      open: state.open,
      items,
      allowEmptyOverwrite,
    })
      .then((record) => {
        const latestKey = shelfStorageKey(projectScopeId)
        shelfBackendRevisionByKeyRef.current[latestKey] = Math.max(0, Number(record.revision || 0))
        shelfBackendHydratedKeysRef.current.add(latestKey)
        if (emptyClosed && allowEmptyOverwrite) {
          delete shelfEmptyBackendSaveIntentRef.current[latestKey]
        }
      })
      .catch(() => {
        // Local shelf storage remains the immediate fallback when the API is unavailable.
      })
  }, [])

  useEffect(() => {
    onShelfOpenChange?.(shelfOpen)
  }, [onShelfOpenChange, shelfOpen])

  useEffect(() => {
    onShelfStateChange?.({ open: shelfOpen, count: shelfItems.length })
  }, [onShelfStateChange, shelfItems.length, shelfOpen])

  useEffect(() => {
    const summary = Boolean(shelfSummaryLoadingKey)
    const repair = Boolean(shelfRepairLoadingKey)
    const autoRepair = shelfAutoRepairingKeys.length > 0
    const backgroundOnly = shelfBackgroundBusy && !summary && !repair && !autoRepair
    onShelfActivityChange?.({
      summary,
      repair,
      autoRepair,
      background: shelfBackgroundBusy,
      count: (summary ? 1 : 0) + (repair ? 1 : 0) + shelfAutoRepairingKeys.length + (backgroundOnly ? 1 : 0),
    })
  }, [onShelfActivityChange, shelfAutoRepairingKeys.length, shelfBackgroundBusy, shelfRepairLoadingKey, shelfSummaryLoadingKey])

  useEffect(() => () => {
    shelfAsyncScopeEpochRef.current += 1
    onShelfActivityChange?.({ summary: false, repair: false, autoRepair: false, background: false, count: 0 })
  }, [onShelfActivityChange])

  useEffect(() => {
    if (closeShelfSignal <= 0) return
    setShelfOpen(false)
  }, [closeShelfSignal])

  useEffect(() => {
    if (openShelfSignal <= 0) return
    setShelfOpen(true)
  }, [openShelfSignal])

  useEffect(() => {
    return () => {
      if (citationHoverOpenTimerRef.current !== null) {
        window.clearTimeout(citationHoverOpenTimerRef.current)
      }
      if (citationHoverCloseTimerRef.current !== null) {
        window.clearTimeout(citationHoverCloseTimerRef.current)
      }
      if (citationPolishRetryTimerRef.current !== null) {
        window.clearTimeout(citationPolishRetryTimerRef.current)
      }
      if (shelfAutoRepairTimerRef.current !== null) {
        window.clearTimeout(shelfAutoRepairTimerRef.current)
      }
      if (shelfMetadataHydrateTimerRef.current !== null) {
        window.clearTimeout(shelfMetadataHydrateTimerRef.current)
      }
      if (shelfSummaryBackfillTimerRef.current !== null) {
        window.clearTimeout(shelfSummaryBackfillTimerRef.current)
      }
      if (shelfMessageFlashTimerRef.current !== null) {
        window.clearTimeout(shelfMessageFlashTimerRef.current)
      }
      if (persistShelfBackendTimerRef.current !== null) {
        window.clearTimeout(persistShelfBackendTimerRef.current)
      }
    }
  }, [])

  useEffect(() => {
    const sourcePath = String(paperGuideSourcePath || '').trim()
    const sourceName = String(paperGuideSourceName || '').trim()
    if (!sourcePath) {
      setGuideDocCandidates([])
      return
    }
    let cancelled = false
    const ctrl = new AbortController()
    referencesApi.readerDoc(sourcePath, { signal: ctrl.signal })
      .then((res) => {
        if (cancelled) return
        const markdown = String(res.markdown || '')
        if (!markdown.trim()) {
          setGuideDocCandidates([])
          return
        }
        const resolvedName = String(res.source_name || sourceName || '').trim()
        const anchors = Array.isArray(res.anchors) ? res.anchors : []
        setGuideDocCandidates(
          buildGuideLocateCandidates(
            markdown,
            sourcePath,
            resolvedName || sourceName || sourcePath,
            'guide',
            anchors,
          ),
        )
      })
      .catch(() => {
        if (!cancelled) setGuideDocCandidates([])
      })
    return () => {
      cancelled = true
      ctrl.abort()
    }
  }, [paperGuideSourcePath, paperGuideSourceName])

  useLayoutEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const timer = window.requestAnimationFrame(() => {
      el.scrollTop = el.scrollHeight
    })
    return () => window.cancelAnimationFrame(timer)
  }, [activeConvId, generationPartial])

  useEffect(() => {
    if (!jumpTarget || !Number.isFinite(jumpTarget.messageId)) return
    const el = scrollRef.current
    if (!el) return
    const target = el.querySelector<HTMLElement>(`[data-msg-id="${jumpTarget.messageId}"]`)
    if (!target) return
    const targetRect = target.getBoundingClientRect()
    const containerRect = el.getBoundingClientRect()
    const top = targetRect.top - containerRect.top + el.scrollTop - 12
    el.scrollTo({ top: Math.max(0, top), behavior: 'smooth' })
    try {
      target.animate(
        [
          { boxShadow: '0 0 0 0 rgba(24,144,255,0.0)', backgroundColor: 'rgba(24,144,255,0.0)' },
          { boxShadow: '0 0 0 3px rgba(24,144,255,0.24)', backgroundColor: 'rgba(24,144,255,0.10)' },
          { boxShadow: '0 0 0 0 rgba(24,144,255,0.0)', backgroundColor: 'rgba(24,144,255,0.0)' },
        ],
        { duration: 900, easing: 'ease-out' },
      )
    } catch {
      // no-op: Web Animations may not be available in all envs.
    }
    onJumpHandled?.(jumpTarget)
  }, [jumpTarget, messages, onJumpHandled])

  useEffect(() => {
    if (!onTrackedMessageActive) return
    const el = scrollRef.current
    if (!el) return
    const trackedIds = Array.isArray(trackedMessageIds)
      ? trackedMessageIds.filter((id) => Number.isFinite(id))
      : []
    if (trackedIds.length <= 0) {
      onTrackedMessageActive(null)
      return
    }
    let syncFrameId = 0
    let measureFrameId = 0
    let lastReported: number | null = null
    let lastActiveIndex = -1
    let lastScrollTop = el.scrollTop
    let trackedAnchors: Array<{ id: number; top: number }> = []
    const SWITCH_HYSTERESIS_PX = 28

    const transitionMargin = (leftIndex: number, rightIndex: number) => {
      const leftTop = trackedAnchors[leftIndex]?.top ?? 0
      const rightTop = trackedAnchors[rightIndex]?.top ?? leftTop
      const gap = Math.max(0, rightTop - leftTop)
      return Math.min(SWITCH_HYSTERESIS_PX, Math.max(10, gap * 0.2))
    }

    const syncActiveMessage = () => {
      syncFrameId = 0
      if (trackedAnchors.length <= 0) {
        lastActiveIndex = -1
        if (lastReported !== null) {
          lastReported = null
          onTrackedMessageActive(null)
        }
        return
      }

      const currentScrollTop = el.scrollTop
      const anchorTop = currentScrollTop + Math.min(120, Math.max(48, el.clientHeight * 0.2))
      let low = 0
      let high = trackedAnchors.length - 1
      let activeIndex = 0
      while (low <= high) {
        const mid = Math.floor((low + high) / 2)
        if (trackedAnchors[mid]!.top <= anchorTop) {
          activeIndex = mid
          low = mid + 1
        } else {
          high = mid - 1
        }
      }
      if (lastActiveIndex >= 0 && lastActiveIndex < trackedAnchors.length && activeIndex !== lastActiveIndex) {
        const direction = currentScrollTop - lastScrollTop
        if (activeIndex === lastActiveIndex + 1 && direction >= 0) {
          const nextTop = trackedAnchors[activeIndex]?.top ?? 0
          if (anchorTop < nextTop + transitionMargin(lastActiveIndex, activeIndex)) {
            activeIndex = lastActiveIndex
          }
        } else if (activeIndex === lastActiveIndex - 1 && direction <= 0) {
          const currentTop = trackedAnchors[lastActiveIndex]?.top ?? 0
          if (anchorTop >= currentTop - transitionMargin(activeIndex, lastActiveIndex)) {
            activeIndex = lastActiveIndex
          }
        }
      }
      const activeMessageId = trackedAnchors[activeIndex]?.id ?? null
      lastScrollTop = currentScrollTop
      lastActiveIndex = activeMessageId != null ? activeIndex : -1

      if (activeMessageId !== lastReported) {
        lastReported = activeMessageId
        onTrackedMessageActive(activeMessageId)
      }
    }

    const scheduleSync = () => {
      if (syncFrameId) return
      syncFrameId = window.requestAnimationFrame(syncActiveMessage)
    }

    const measureTrackedAnchors = () => {
      measureFrameId = 0
      const containerRect = el.getBoundingClientRect()
      const currentScrollTop = el.scrollTop
      trackedAnchors = trackedIds
        .map((id) => {
          const node = el.querySelector<HTMLElement>(`[data-msg-id="${id}"]`)
          if (!node) return null
          const rect = node.getBoundingClientRect()
          return {
            id,
            top: rect.top - containerRect.top + currentScrollTop,
          }
        })
        .filter((item): item is { id: number; top: number } => Boolean(item))
        .sort((left, right) => left.top - right.top)
      if (lastReported != null) {
        lastActiveIndex = trackedAnchors.findIndex((item) => item.id === lastReported)
      } else {
        lastActiveIndex = -1
      }
      scheduleSync()
    }

    const scheduleMeasure = () => {
      if (measureFrameId) return
      measureFrameId = window.requestAnimationFrame(measureTrackedAnchors)
    }

    const resizeObserver = typeof ResizeObserver !== 'undefined'
      ? new ResizeObserver(() => {
        scheduleMeasure()
      })
      : null

    el.addEventListener('scroll', scheduleSync, { passive: true })
    window.addEventListener('resize', scheduleMeasure)
    resizeObserver?.observe(el)
    if (el.firstElementChild instanceof HTMLElement) {
      resizeObserver?.observe(el.firstElementChild)
    }
    scheduleMeasure()

    return () => {
      el.removeEventListener('scroll', scheduleSync)
      window.removeEventListener('resize', scheduleMeasure)
      resizeObserver?.disconnect()
      if (syncFrameId) {
        window.cancelAnimationFrame(syncFrameId)
      }
      if (measureFrameId) {
        window.cancelAnimationFrame(measureFrameId)
      }
    }
  }, [messages, onTrackedMessageActive, trackedMessageIds])

  useEffect(() => {
    shelfAsyncScopeEpochRef.current += 1
    const nextStorageKey = shelfStorageKey(shelfScopeId)
    const nextSavedStorageKey = shelfSavedStorageKey(shelfScopeId)
    const legacyKeys = legacyShelfStorageKeys(activeConvId)
    const legacySavedKeys = legacyKeys.map((key) => `${key}${SHELF_SAVED_SUFFIX}`)
    const prevStorageKey = activeStorageKeyRef.current
    flushShelfBackendRef.current?.()
    if (persistShelfTimerRef.current !== null) {
      window.clearTimeout(persistShelfTimerRef.current)
      persistShelfTimerRef.current = null
    }
    if (persistShelfBackendTimerRef.current !== null) {
      window.clearTimeout(persistShelfBackendTimerRef.current)
      persistShelfBackendTimerRef.current = null
    }
    shelfBackendHydratedKeysRef.current.delete(nextStorageKey)
    if (shelfAutoRepairTimerRef.current !== null) {
      window.clearTimeout(shelfAutoRepairTimerRef.current)
      shelfAutoRepairTimerRef.current = null
    }
    if (shelfMetadataHydrateTimerRef.current !== null) {
      window.clearTimeout(shelfMetadataHydrateTimerRef.current)
      shelfMetadataHydrateTimerRef.current = null
    }
    if (shelfSummaryBackfillTimerRef.current !== null) {
      window.clearTimeout(shelfSummaryBackfillTimerRef.current)
      shelfSummaryBackfillTimerRef.current = null
    }
    setShelfAutoRepairingKeySet(new Set())
    setShelfSummaryLoadingKey('')
    setShelfRepairLoadingKey('')
    setShelfRepairImpact(null)
    shelfAutoRepairFingerprintsRef.current = {}
    shelfAutoRepairRetryAfterRef.current = {}
    shelfMetadataHydrateInFlightRef.current = new Set()
    shelfMetadataHydrateAttemptedAtRef.current = {}
    shelfSummaryBackfillInFlightRef.current = new Set()
    shelfSummaryBackfillAttemptedAtRef.current = {}
    if (prevStorageKey !== nextStorageKey) {
      const prevRevision = Number(shelfRevisionByKeyRef.current[prevStorageKey] || 0)
      const latest = latestShelfStateRef.current
      const flushedRevision = persistShelfSnapshot(
        prevStorageKey,
        { open: latest.open, items: latest.items },
        prevRevision,
      )
      shelfRevisionByKeyRef.current[prevStorageKey] = flushedRevision
    }

    // Switching shelf scope changes storage key; skip one persist cycle to avoid
    // writing previous scope state into the new key before hydration.
    skipShelfPersistOnceRef.current = true
    const savedSnapshots = migrateLegacySavedShelfSnapshots(nextSavedStorageKey, legacySavedKeys)
    setSavedShelfSnapshots(savedSnapshots)
    setSelectedSavedSnapshotId((current) => {
      if (current && savedSnapshots.some((item) => item.id === current)) return current
      return savedSnapshots[0]?.id || ''
    })
    const snapshot = migrateLegacyShelfSnapshot(nextStorageKey, legacyKeys)
    if (!snapshot) {
      shelfRevisionByKeyRef.current[nextStorageKey] = 0
      latestShelfStateRef.current = { convId: activeConvId, projectId: shelfScopeId, open: false, items: [] }
      setShelfItems([])
      setShelfOpen(false)
      setFocusedShelfKey('')
      activeStorageKeyRef.current = nextStorageKey
      return
    }
    shelfRevisionByKeyRef.current[nextStorageKey] = Math.max(0, snapshot.revision || 0)
    latestShelfStateRef.current = {
      convId: activeConvId,
      projectId: shelfScopeId,
      open: snapshot.open,
      items: snapshot.items,
    }
    setShelfItems(snapshot.items)
    setShelfOpen(snapshot.open)
    setFocusedShelfKey('')
    activeStorageKeyRef.current = nextStorageKey
  }, [activeConvId, setShelfAutoRepairingKeySet, shelfScopeId])

  useEffect(() => {
    const storageKey = shelfStorageKey(shelfScopeId)
    const requestProjectId = shelfScopeId === '__default__' ? undefined : shelfScopeId
    const requestSeq = shelfBackendHydrateSeqRef.current + 1
    shelfBackendHydrateSeqRef.current = requestSeq
    let cancelled = false
    let requestStartedAt = Date.now()
    const timer = window.setTimeout(() => {
      requestStartedAt = Date.now()
      chatApi.getCitationShelf({ convId: activeConvId || undefined, projectId: requestProjectId, scope: 'project' })
        .then((record) => {
          if (cancelled || shelfBackendHydrateSeqRef.current !== requestSeq) return
          const latest = latestShelfStateRef.current
          if (shelfStorageKey(latest.projectId) !== storageKey) return
          const backendRevision = Math.max(0, Number(record.revision || 0))
          shelfBackendRevisionByKeyRef.current[storageKey] = backendRevision
          shelfBackendHydratedKeysRef.current.add(storageKey)

          const backendItems = restoreShelfItems(Array.isArray(record.items) ? record.items : [])
          const currentItems = dedupeShelfItems(latest.items || []).slice(0, SHELF_MAX_ITEMS)
          const rawBackendUpdatedAt = Number(record.updated_at || 0)
          const backendUpdatedAtMs = rawBackendUpdatedAt > 1000000000000
            ? rawBackendUpdatedAt
            : rawBackendUpdatedAt * 1000
          const localSnapshot = readShelfSnapshot(storageKey)
          const localUpdatedAtMs = Number(localSnapshot?.updatedAt || 0)
          const stateChangedAfterRequest = shelfStateTouchedAtRef.current > requestStartedAt + 10
          const localLooksNewer = localUpdatedAtMs > backendUpdatedAtMs + 500

          let nextItems: CiteShelfItem[]
          let nextOpen = Boolean(record.open)
          let shouldSaveBackend = false

          if (backendRevision <= 0) {
            nextItems = currentItems.length > 0 ? currentItems : backendItems
            nextOpen = latest.open || Boolean(record.open)
            shouldSaveBackend = nextItems.length > 0 || nextOpen
          } else if (backendItems.length <= 0 && currentItems.length > 0) {
            if (stateChangedAfterRequest || localLooksNewer) {
              nextItems = currentItems
              nextOpen = latest.open
              shouldSaveBackend = true
            } else {
              nextItems = []
              nextOpen = Boolean(record.open)
            }
          } else if (currentItems.length <= 0) {
            nextItems = backendItems
            nextOpen = Boolean(record.open)
          } else if (backendUpdatedAtMs > localUpdatedAtMs + 500 && !stateChangedAfterRequest) {
            nextItems = backendItems
            nextOpen = Boolean(record.open)
          } else {
            nextItems = dedupeShelfItems([...currentItems, ...backendItems]).slice(0, SHELF_MAX_ITEMS)
            nextOpen = latest.open || Boolean(record.open)
            shouldSaveBackend = !sameShelfItems(nextItems, backendItems) || nextOpen !== Boolean(record.open)
          }

          latestShelfStateRef.current = {
            convId: latest.convId,
            projectId: latest.projectId,
            open: nextOpen,
            items: nextItems,
          }
          if (!sameShelfItems(currentItems, nextItems)) {
            setShelfItems(nextItems)
            setFocusedShelfKey((current) => (
              current && nextItems.some((item) => item.key === current) ? current : ''
            ))
          }
          if (latest.open !== nextOpen) {
            setShelfOpen(nextOpen)
          }
          if (shouldSaveBackend) {
            saveShelfBackendNow({ convId: latest.convId, projectId: latest.projectId, open: nextOpen, items: nextItems })
          }
        })
        .catch(() => {
          if (cancelled || shelfBackendHydrateSeqRef.current !== requestSeq) return
          shelfBackendHydratedKeysRef.current.delete(storageKey)
        })
    }, 0)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [activeConvId, saveShelfBackendNow, shelfScopeId])

  useEffect(() => {
    const storageKey = shelfStorageKey(shelfScopeId)
    const savedStorageKey = shelfSavedStorageKey(shelfScopeId)
    const onStorage = (event: StorageEvent) => {
      if (event.key === savedStorageKey) {
        if (event.newValue === null) {
          invalidateSavedShelfSnapshotCache(savedStorageKey)
          setSavedShelfSnapshots([])
          setSelectedSavedSnapshotId('')
          return
        }
        const snapshots = readSavedShelfSnapshots(savedStorageKey, event.newValue)
        setSavedShelfSnapshots(snapshots)
        setSelectedSavedSnapshotId((current) => {
          if (current && snapshots.some((item) => item.id === current)) return current
          return snapshots[0]?.id || ''
        })
        return
      }
      if (event.key !== storageKey) return
      if (event.newValue === null) {
        invalidateShelfSnapshotCache(storageKey)
        skipShelfPersistOnceRef.current = true
        shelfRevisionByKeyRef.current[storageKey] = 0
        latestShelfStateRef.current = { convId: activeConvId, projectId: shelfScopeId, open: false, items: [] }
        setShelfItems([])
        setShelfOpen(false)
        setFocusedShelfKey('')
        return
      }
      const snapshot = readShelfSnapshot(storageKey, event.newValue)
      if (!snapshot) return
      const currentRevision = Number(shelfRevisionByKeyRef.current[storageKey] || 0)
      if (snapshot.revision <= currentRevision) return
      skipShelfPersistOnceRef.current = true
      shelfRevisionByKeyRef.current[storageKey] = snapshot.revision
      latestShelfStateRef.current = {
        convId: activeConvId,
        projectId: shelfScopeId,
        open: snapshot.open,
        items: snapshot.items,
      }
      setShelfItems(snapshot.items)
      setShelfOpen(snapshot.open)
      setFocusedShelfKey('')
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [activeConvId, shelfScopeId])

  useLayoutEffect(() => {
    shelfStateTouchedAtRef.current = Date.now()
    latestShelfStateRef.current = { convId: activeConvId, projectId: shelfScopeId, open: shelfOpen, items: shelfItems }
  }, [activeConvId, shelfItems, shelfOpen, shelfScopeId])

  useEffect(() => {
    flushShelfSnapshotRef.current = () => {
      if (persistShelfTimerRef.current !== null) {
        window.clearTimeout(persistShelfTimerRef.current)
        persistShelfTimerRef.current = null
      }
      const latest = latestShelfStateRef.current
      const storageKey = shelfStorageKey(latest.projectId)
      const currentRevision = Number(shelfRevisionByKeyRef.current[storageKey] || 0)
      const nextRevision = persistShelfSnapshot(
        storageKey,
        { open: latest.open, items: latest.items },
        currentRevision,
      )
      shelfRevisionByKeyRef.current[storageKey] = nextRevision
      activeStorageKeyRef.current = storageKey
    }
    flushShelfBackendRef.current = () => {
      if (persistShelfBackendTimerRef.current !== null) {
        window.clearTimeout(persistShelfBackendTimerRef.current)
        persistShelfBackendTimerRef.current = null
      }
      saveShelfBackendNow(latestShelfStateRef.current)
    }
    return () => {
      if (flushShelfSnapshotRef.current) {
        flushShelfSnapshotRef.current()
      }
      if (flushShelfBackendRef.current) {
        flushShelfBackendRef.current()
      }
      flushShelfSnapshotRef.current = null
      flushShelfBackendRef.current = null
    }
  }, [saveShelfBackendNow])

  useEffect(() => {
    setSelectedSavedSnapshotId((current) => {
      if (current && savedShelfSnapshots.some((item) => item.id === current)) return current
      return savedShelfSnapshots[0]?.id || ''
    })
  }, [savedShelfSnapshots])

  useEffect(() => {
    return () => {
      if (persistShelfTimerRef.current !== null) {
        window.clearTimeout(persistShelfTimerRef.current)
        persistShelfTimerRef.current = null
      }
      if (persistShelfBackendTimerRef.current !== null) {
        window.clearTimeout(persistShelfBackendTimerRef.current)
        persistShelfBackendTimerRef.current = null
      }
    }
  }, [])

  useEffect(() => {
    const flushShelfSnapshot = () => {
      flushShelfSnapshotRef.current?.()
      flushShelfBackendRef.current?.()
    }
    window.addEventListener('pagehide', flushShelfSnapshot)
    window.addEventListener('beforeunload', flushShelfSnapshot)
    return () => {
      window.removeEventListener('pagehide', flushShelfSnapshot)
      window.removeEventListener('beforeunload', flushShelfSnapshot)
    }
  }, [])

  useEffect(() => {
    if (skipShelfPersistOnceRef.current) {
      skipShelfPersistOnceRef.current = false
      return
    }
    const storageKey = shelfStorageKey(shelfScopeId)
    if (persistShelfTimerRef.current !== null) {
      window.clearTimeout(persistShelfTimerRef.current)
      persistShelfTimerRef.current = null
    }
    persistShelfTimerRef.current = window.setTimeout(() => {
      const latest = latestShelfStateRef.current
      const latestStorageKey = shelfStorageKey(latest.projectId)
      if (latestStorageKey !== storageKey) {
        persistShelfTimerRef.current = null
        return
      }
      const currentRevision = Number(shelfRevisionByKeyRef.current[storageKey] || 0)
      const nextRevision = persistShelfSnapshot(
        storageKey,
        { open: latest.open, items: latest.items },
        currentRevision,
      )
      shelfRevisionByKeyRef.current[storageKey] = nextRevision
      persistShelfTimerRef.current = null
    }, 80)
    return () => {
      if (persistShelfTimerRef.current !== null) {
        window.clearTimeout(persistShelfTimerRef.current)
        persistShelfTimerRef.current = null
      }
    }
  }, [shelfItems, shelfOpen, shelfScopeId])

  useEffect(() => {
    const storageKey = shelfStorageKey(shelfScopeId)
    if (!shelfBackendHydratedKeysRef.current.has(storageKey)) return
    if (persistShelfBackendTimerRef.current !== null) {
      window.clearTimeout(persistShelfBackendTimerRef.current)
      persistShelfBackendTimerRef.current = null
    }
    persistShelfBackendTimerRef.current = window.setTimeout(() => {
      const latest = latestShelfStateRef.current
      const latestStorageKey = shelfStorageKey(latest.projectId)
      if (latestStorageKey !== storageKey) {
        persistShelfBackendTimerRef.current = null
        return
      }
      saveShelfBackendNow(latest)
      persistShelfBackendTimerRef.current = null
    }, SHELF_BACKEND_PERSIST_MS)
    return () => {
      if (persistShelfBackendTimerRef.current !== null) {
        window.clearTimeout(persistShelfBackendTimerRef.current)
        persistShelfBackendTimerRef.current = null
      }
    }
  }, [saveShelfBackendNow, shelfItems, shelfOpen, shelfScopeId])

  const rows = useMemo(() => {
    const out: Array<
      | { kind: 'message'; message: Message }
      | { kind: 'refs'; userMsgId: number }
    > = []
    let lastUserMsgId = 0
    const renderedRefs = new Set<number>()
    const refsPanelFilter = {
      activeSourcePath: paperGuideSourcePath,
      activeSourceName: paperGuideSourceName,
    }

    for (const message of messages) {
      out.push({ kind: 'message', message })
      if (message.role === 'user') {
        lastUserMsgId = message.id
        continue
      }
      if (lastUserMsgId > 0 && !renderedRefs.has(lastUserMsgId) && hasRenderableRefsForGuide(refs, lastUserMsgId, refsPanelFilter)) {
        out.push({ kind: 'refs', userMsgId: lastUserMsgId })
        renderedRefs.add(lastUserMsgId)
      }
    }
    if (lastUserMsgId > 0 && !renderedRefs.has(lastUserMsgId) && hasRenderableRefsForGuide(refs, lastUserMsgId, refsPanelFilter)) {
      out.push({ kind: 'refs', userMsgId: lastUserMsgId })
    }

    return out
  }, [messages, paperGuideSourceName, paperGuideSourcePath, refs])

  const assistantTraceByMsgId = useMemo(() => {
    const out = new Map<number, { answerOrder: number; userMsgId: number }>()
    let answerOrder = 0
    let lastUserMsgId = 0
    for (const message of messages) {
      if (message.role === 'user') {
        lastUserMsgId = message.id
        continue
      }
      if (message.role !== 'assistant') continue
      answerOrder += 1
      out.set(message.id, { answerOrder, userMsgId: lastUserMsgId })
    }
    return out
  }, [messages])

  const selectedResearchContextByAssistantId = useMemo(() => {
    const out = new Map<number, SelectedResearchContextPack>()
    let pendingUserContext: SelectedResearchContextPack | null = null
    for (const message of messages) {
      if (message.role === 'user') {
        pendingUserContext = getUserPromptResearchContext(message)
        continue
      }
      if (message.role !== 'assistant') continue
      const assistantContext = getAssistantSelectedResearchContext(message)
      if (assistantContext) {
        out.set(Number(message.id), assistantContext)
      } else if (pendingUserContext) {
        out.set(Number(message.id), pendingUserContext)
      }
      pendingUserContext = null
    }
    return out
  }, [messages])

  const liveCiteMap = useMemo(() => {
    const map = new Map<string, CiteShelfItem>()
    const convTraceId = String(activeConvId || '')
    for (const message of messages) {
      if (message.role !== 'assistant') continue
      const rawCiteDetails = getMessageCiteDetailRecords(message)
      if (rawCiteDetails.length <= 0) continue
      const trace = assistantTraceByMsgId.get(message.id)
      for (const rawDetail of rawCiteDetails) {
        const detail = normalizeCiteDetail(rawDetail)
        if (!detail) continue
        const tracedDetail: CiteDetail = {
          ...detail,
          traceConvId: convTraceId,
          traceAssistantMsgId: message.id,
          traceAssistantOrder: Number(trace?.answerOrder || 0),
          traceUserMsgId: Number(trace?.userMsgId || 0),
        }
        const item = toShelfItem(tracedDetail)
        map.set(item.key, item)
      }
    }
    return map
  }, [activeConvId, assistantTraceByMsgId, messages])

  useEffect(() => {
    const candidates = Array.from(liveCiteMap.values())
      .filter(shouldRequestCitationCardPolish)
      .slice(0, 18)
    for (const item of candidates) {
      const itemKey = toShelfItem(item).key
      const warmKey = `${itemKey}|${item.citationCardPolishKey || ''}|v3`
      if (citationPolishPrewarmKeysRef.current.has(warmKey)) continue
      citationPolishPrewarmKeysRef.current.add(warmKey)
      referencesApi.citationCardPolishCached(item as unknown as Record<string, unknown>, 0.25)
        .catch(() => {
          citationPolishPrewarmKeysRef.current.delete(warmKey)
        })
    }
  }, [liveCiteMap])

  useEffect(() => {
    setShelfItems((current) => {
      let changed = false
      const next = current.map((item) => {
        const live = liveCiteMap.get(item.key)
        if (!live) return item
        const merged = mergeShelfItemWithLive(item, live)
        if (!sameShelfItem(merged, item)) {
          changed = true
          return merged
        }
        return item
      })
      const deduped = dedupeShelfItems(next)
      if (deduped.length !== current.length) changed = true
      return changed ? deduped : current
    })
  }, [liveCiteMap])

  const fetchShelfSummaryForItem = (item: CiteShelfItem, options?: { force?: boolean }) => {
    const summaryLine = String(item.summaryLine || '').trim()
    const lowValueSummary = Boolean(summaryLine && looksLowValueShelfSummary(summaryLine))
    if (!options?.force && shelfItemHasDisplayableArticleSummary(item)) return
    const itemIdentity = shelfPaperIdentity(item)
    const scopeToken = captureShelfAsyncScope()
    const requestItem = (lowValueSummary || options?.force)
      ? {
        ...item,
        summaryLine: '',
        summarySource: '',
        summaryProvider: '',
        summaryQuality: null,
        summary_line: '',
        summary_source: '',
        summary_provider: '',
        summary_quality: null,
      }
      : item
    setShelfSummaryLoadingKey(item.key)
    const loadBibliometrics = options?.force
      ? referencesApi.bibliometrics
      : referencesApi.bibliometricsCached
    loadBibliometrics(withBibliometricsLocale(requestItem as unknown as Record<string, unknown>))
      .then((meta) => {
        if (!currentShelfItemForAsync(scopeToken, item.key)) return
        if (!meta || Object.keys(meta).length === 0) return
        const articleSummaryPatch = articleSummaryPatchFromMeta(meta)
        setShelfItems((current) => current.map((entry) => {
          if (entry.key !== item.key && shelfPaperIdentity(entry) !== itemIdentity) return entry
          const merged = mergeCiteMeta(entry, meta)
          return {
            ...toShelfItem(merged),
            ...articleSummaryPatch,
            key: entry.key,
            tags: normalizeShelfTags(entry.tags),
            note: normalizeShelfNote(entry.note),
          }
        }))
      })
      .finally(() => {
        if (!shelfAsyncScopeIsCurrent(scopeToken)) return
        setShelfSummaryLoadingKey((current) => (current === item.key ? '' : current))
      })
  }

  useEffect(() => {
    if (shelfSummaryBackfillTimerRef.current !== null) {
      window.clearTimeout(shelfSummaryBackfillTimerRef.current)
      shelfSummaryBackfillTimerRef.current = null
    }
    if (!shelfOpen || shelfItems.length <= 0) return
    shelfSummaryBackfillTimerRef.current = window.setTimeout(() => {
      shelfSummaryBackfillTimerRef.current = null
      const now = Date.now()
      const targets: Array<{ item: CiteShelfItem; attemptKey: string }> = []
      for (const item of shelfItems) {
        if (targets.length >= SHELF_SUMMARY_BACKFILL_BATCH_SIZE) break
        if (shelfSummaryBackfillInFlightRef.current.has(item.key)) continue
        if (!shelfItemNeedsSummaryBackfill(item)) continue
        const attemptKey = shelfSummaryBackfillAttemptKey(item)
        const lastAttempt = Number(shelfSummaryBackfillAttemptedAtRef.current[attemptKey] || 0)
        if (lastAttempt > 0 && now - lastAttempt < SHELF_SUMMARY_BACKFILL_RETRY_MS) continue
        targets.push({ item, attemptKey })
      }
      if (targets.length <= 0) return

      const inFlight = new Set(shelfSummaryBackfillInFlightRef.current)
      for (const target of targets) {
        inFlight.add(target.item.key)
        shelfSummaryBackfillAttemptedAtRef.current[target.attemptKey] = now
      }
      shelfSummaryBackfillInFlightRef.current = inFlight
      setShelfSummaryLoadingKey((current) => current || targets[0]?.item.key || '')
      const scopeToken = captureShelfAsyncScope()

      void Promise.all(targets.map(({ item }) => (
        referencesApi.bibliometrics(withBibliometricsLocale({
          ...item,
          summaryLine: '',
          summarySource: '',
          summaryProvider: '',
          summaryQuality: null,
          summary_line: '',
          summary_source: '',
          summary_provider: '',
          summary_quality: null,
        } as unknown as Record<string, unknown>))
          .catch(() => ({}))
          .then((meta) => ({ key: item.key, meta }))
      ))).then((results) => {
        if (!shelfAsyncScopeIsCurrent(scopeToken)) return
        const usable = results.filter((entry) => entry.meta && Object.keys(entry.meta).length > 0)
        if (usable.length <= 0) return
        setShelfItems((current) => current.map((entry) => {
          if (!currentShelfItemForAsync(scopeToken, entry.key)) return entry
          const result = usable.find((item) => item.key === entry.key)
          if (!result) return entry
          const merged = mergeCiteMeta(entry, result.meta)
          const articleSummaryPatch = articleSummaryPatchFromMeta(result.meta)
          return {
            ...toShelfItem(merged),
            ...articleSummaryPatch,
            key: entry.key,
            tags: normalizeShelfTags(entry.tags),
            note: normalizeShelfNote(entry.note),
          }
        }))
      }).finally(() => {
        if (!shelfAsyncScopeIsCurrent(scopeToken)) return
        const nextInFlight = new Set(shelfSummaryBackfillInFlightRef.current)
        for (const target of targets) nextInFlight.delete(target.item.key)
        shelfSummaryBackfillInFlightRef.current = nextInFlight
        setShelfSummaryLoadingKey((current) => (
          targets.some((target) => target.item.key === current) ? '' : current
        ))
      })
    }, 220)
    return () => {
      if (shelfSummaryBackfillTimerRef.current !== null) {
        window.clearTimeout(shelfSummaryBackfillTimerRef.current)
        shelfSummaryBackfillTimerRef.current = null
      }
    }
  }, [captureShelfAsyncScope, currentShelfItemForAsync, shelfAsyncScopeIsCurrent, shelfItems, shelfOpen])

  const applyShelfMetadataRepairCandidates = useCallback((
    updates: Array<{ key: string; metas: Array<Record<string, unknown>> }>,
  ): boolean => {
    if (updates.length <= 0) return false
    const byKey = new Map<string, Array<Record<string, unknown>>>()
    for (const update of updates) {
      const key = String(update.key || '').trim()
      const metas = (update.metas || []).filter((meta) => meta && Object.keys(meta).length > 0)
      if (!key || metas.length <= 0) continue
      byKey.set(key, [...(byKey.get(key) || []), ...metas])
    }
    if (byKey.size <= 0) return false
    let didUpdate = false
    setShelfItems((current) => current.map((entry) => {
      const candidates = byKey.get(entry.key)
      if (!candidates || candidates.length <= 0) return entry
      for (const meta of candidates) {
        const accepted = strictRepairMerge(entry, meta)
        if (!accepted) continue
        if (!sameShelfItem(accepted, entry)) {
          didUpdate = true
          return accepted
        }
        return entry
      }
      return entry
    }))
    return didUpdate
  }, [])

  const repairShelfItemMeta = (item: CiteShelfItem, options?: { silent?: boolean }) => {
    if (shelfRepairLoadingKey === item.key) return
    const silent = Boolean(options?.silent)
    const scopeToken = captureShelfAsyncScope()
    const requestedFingerprint = shelfItemRepairFingerprint(item)
    setShelfRepairLoadingKey(item.key)
    const payloads = shelfRepairPayloads(item)
    const loadRepairCandidates = referencesApi.repairShelfMetadata(payloads, payloads.length)
      .then((res) => {
        if (!currentShelfItemForAsync(scopeToken, item.key, requestedFingerprint)) return []
        setShelfRepairImpact(res.impact || null)
        const repaired = Array.isArray(res.items) ? res.items : []
        return repaired
          .map(shelfRepairMetaFromEntry)
          .filter((meta) => meta && Object.keys(meta).length > 0)
      })
      .catch(() => {
        if (!currentShelfItemForAsync(scopeToken, item.key, requestedFingerprint)) return []
        return Promise.all([
          ...payloads.map((payload) => referencesApi.bibliometrics(withBibliometricsLocale(payload)).catch(() => ({}))),
        ])
      })

    loadRepairCandidates
      .then((metas) => {
        if (!currentShelfItemForAsync(scopeToken, item.key, requestedFingerprint)) return
        const candidates = metas.filter((meta) => meta && Object.keys(meta).length > 0)
        const didUpdate = applyShelfMetadataRepairCandidates([{ key: item.key, metas: candidates }])
        if (!silent) {
          if (didUpdate) message.success('Metadata repaired with strict rules')
          else message.info('Strict match did not pass; original metadata kept')
        }
      })
      .catch(() => {
        if (!currentShelfItemForAsync(scopeToken, item.key, requestedFingerprint)) return
        if (!silent) message.error('Repair failed, please retry.')
      })
      .finally(() => {
        if (!shelfAsyncScopeIsCurrent(scopeToken)) return
        setShelfRepairLoadingKey((current) => (current === item.key ? '' : current))
      })
  }

  const repairShelfItemsMetadataBatch = useCallback(async (targets: CiteShelfItem[]) => {
    const uniqueTargets: CiteShelfItem[] = []
    const seen = new Set<string>()
    for (const item of targets) {
      const key = String(item.key || '').trim()
      if (!key || seen.has(key)) continue
      seen.add(key)
      uniqueTargets.push(item)
      if (uniqueTargets.length >= SHELF_AUTO_REPAIR_BATCH_SIZE) break
    }
    if (uniqueTargets.length <= 0) return

    const scopeToken = captureShelfAsyncScope()
    const inFlight = new Set(shelfAutoRepairingKeySetRef.current)
    for (const item of uniqueTargets) {
      inFlight.add(item.key)
    }
    setShelfAutoRepairingKeySet(inFlight)
    const requestedFingerprints = new Map(uniqueTargets.map((item) => [
      item.key,
      shelfItemRepairFingerprint(item),
    ]))

    try {
      const payloads = uniqueTargets.flatMap(shelfRepairPayloads)
      const res = await referencesApi.repairShelfMetadata(payloads, payloads.length)
      if (shelfAsyncScopeIsCurrent(scopeToken)) {
        setShelfRepairImpact(res.impact || null)
        const metasByKey = new Map<string, Array<Record<string, unknown>>>()
        for (const entry of Array.isArray(res.items) ? res.items : []) {
          const meta = shelfRepairMetaFromEntry(entry)
          if (!meta || Object.keys(meta).length <= 0) continue
          const key = String(entry.key || meta.key || '').trim()
          if (!key) continue
          metasByKey.set(key, [...(metasByKey.get(key) || []), meta])
        }
        const updates = Array.from(metasByKey.entries())
          .filter(([key]) => currentShelfItemForAsync(scopeToken, key, requestedFingerprints.get(key) || ''))
          .map(([key, metas]) => ({ key, metas }))
        applyShelfMetadataRepairCandidates(updates)
        for (const item of uniqueTargets) {
          if (!currentShelfItemForAsync(scopeToken, item.key, requestedFingerprints.get(item.key) || '')) continue
          const fingerprint = requestedFingerprints.get(item.key)
          if (fingerprint) shelfAutoRepairFingerprintsRef.current[item.key] = fingerprint
          delete shelfAutoRepairRetryAfterRef.current[item.key]
        }
      }
    } catch {
      if (shelfAsyncScopeIsCurrent(scopeToken)) {
        const retryAt = Date.now() + SHELF_AUTO_REPAIR_RETRY_MS
        for (const item of uniqueTargets) {
          shelfAutoRepairRetryAfterRef.current[item.key] = retryAt
        }
      }
    } finally {
      if (shelfAsyncScopeIsCurrent(scopeToken)) {
        const nextInFlight = new Set(shelfAutoRepairingKeySetRef.current)
        for (const item of uniqueTargets) {
          nextInFlight.delete(item.key)
        }
        setShelfAutoRepairingKeySet(nextInFlight)
      }
    }
  }, [applyShelfMetadataRepairCandidates, captureShelfAsyncScope, currentShelfItemForAsync, setShelfAutoRepairingKeySet, shelfAsyncScopeIsCurrent])

  useEffect(() => {
    if (shelfMetadataHydrateTimerRef.current !== null) {
      window.clearTimeout(shelfMetadataHydrateTimerRef.current)
      shelfMetadataHydrateTimerRef.current = null
    }
    if (!shelfOpen || shelfItems.length <= 0) return
    shelfMetadataHydrateTimerRef.current = window.setTimeout(() => {
      shelfMetadataHydrateTimerRef.current = null
      const now = Date.now()
      const targets: Array<{ item: CiteShelfItem; attemptKey: string }> = []
      for (const item of shelfItems) {
        if (targets.length >= SHELF_METADATA_HYDRATE_BATCH_SIZE) break
        if (item.key === shelfRepairLoadingKey) continue
        if (shelfAutoRepairingKeySetRef.current.has(item.key)) continue
        if (shelfMetadataHydrateInFlightRef.current.has(item.key)) continue
        if (!shelfItemNeedsPersistedMetadataHydrate(item)) continue
        const attemptKey = shelfMetadataHydrateAttemptKey(item)
        const lastAttempt = Number(shelfMetadataHydrateAttemptedAtRef.current[attemptKey] || 0)
        if (lastAttempt > 0 && now - lastAttempt < SHELF_METADATA_HYDRATE_RETRY_MS) continue
        targets.push({ item, attemptKey })
      }
      if (targets.length <= 0) return

      const inFlight = new Set(shelfMetadataHydrateInFlightRef.current)
      for (const target of targets) {
        inFlight.add(target.item.key)
        shelfMetadataHydrateAttemptedAtRef.current[target.attemptKey] = now
      }
      shelfMetadataHydrateInFlightRef.current = inFlight
      const scopeToken = captureShelfAsyncScope()
      const requestedFingerprints = new Map(targets.map((target) => [
        target.item.key,
        shelfItemRepairFingerprint(target.item),
      ]))
      void Promise.all(targets.map(({ item }) => (
        referencesApi.bibliometrics(withBibliometricsLocale(item as unknown as Record<string, unknown>))
          .catch(() => ({}))
          .then((meta) => ({ key: item.key, meta }))
      ))).then((results) => {
        if (!shelfAsyncScopeIsCurrent(scopeToken)) return
        const updates = results
          .filter((entry) => entry.meta && Object.keys(entry.meta).length > 0)
          .filter((entry) => currentShelfItemForAsync(scopeToken, entry.key, requestedFingerprints.get(entry.key) || ''))
          .map((entry) => ({ key: entry.key, metas: [entry.meta] }))
        if (updates.length > 0) {
          applyShelfMetadataRepairCandidates(updates)
        }
      }).finally(() => {
        if (!shelfAsyncScopeIsCurrent(scopeToken)) return
        const nextInFlight = new Set(shelfMetadataHydrateInFlightRef.current)
        for (const target of targets) nextInFlight.delete(target.item.key)
        shelfMetadataHydrateInFlightRef.current = nextInFlight
      })
    }, 160)
    return () => {
      if (shelfMetadataHydrateTimerRef.current !== null) {
        window.clearTimeout(shelfMetadataHydrateTimerRef.current)
        shelfMetadataHydrateTimerRef.current = null
      }
    }
  }, [applyShelfMetadataRepairCandidates, captureShelfAsyncScope, currentShelfItemForAsync, shelfAsyncScopeIsCurrent, shelfItems, shelfOpen, shelfRepairLoadingKey])

  useEffect(() => {
    if (shelfAutoRepairTimerRef.current !== null) {
      window.clearTimeout(shelfAutoRepairTimerRef.current)
      shelfAutoRepairTimerRef.current = null
    }
    if (shelfItems.length <= 0) return
    shelfAutoRepairTimerRef.current = window.setTimeout(() => {
      shelfAutoRepairTimerRef.current = null
      const now = Date.now()
      const inFlight = shelfAutoRepairingKeySetRef.current
      const targets: CiteShelfItem[] = []
      for (const item of shelfItems) {
        if (targets.length >= SHELF_AUTO_REPAIR_BATCH_SIZE) break
        if (inFlight.has(item.key) || item.key === shelfRepairLoadingKey) continue
        if (!shelfItemNeedsMetadataRepair(item)) continue
        const fingerprint = shelfItemRepairFingerprint(item)
        if (shelfAutoRepairFingerprintsRef.current[item.key] === fingerprint) continue
        if ((shelfAutoRepairRetryAfterRef.current[item.key] || 0) > now) continue
        targets.push(item)
      }
      if (targets.length > 0) {
        void repairShelfItemsMetadataBatch(targets)
      }
    }, 250)
    return () => {
      if (shelfAutoRepairTimerRef.current !== null) {
        window.clearTimeout(shelfAutoRepairTimerRef.current)
        shelfAutoRepairTimerRef.current = null
      }
    }
  }, [repairShelfItemsMetadataBatch, shelfItems, shelfRepairLoadingKey])

  const clearCitationHoverTimers = () => {
    if (citationHoverOpenTimerRef.current !== null) {
      window.clearTimeout(citationHoverOpenTimerRef.current)
      citationHoverOpenTimerRef.current = null
    }
    if (citationHoverCloseTimerRef.current !== null) {
      window.clearTimeout(citationHoverCloseTimerRef.current)
      citationHoverCloseTimerRef.current = null
    }
    if (citationPolishRetryTimerRef.current !== null) {
      window.clearTimeout(citationPolishRetryTimerRef.current)
      citationPolishRetryTimerRef.current = null
    }
  }

  const mergeCitationMetaForItemKey = (itemKey: string, metas: Array<Record<string, unknown>>) => {
    const usable = metas.filter((meta) => meta && Object.keys(meta).length > 0)
    if (!usable.length) return
    setPopoverDetail((current) => {
      if (!current) return current
      if (toShelfItem(current).key !== itemKey) return current
      let merged = current
      for (const meta of usable) {
        merged = mergeCiteMeta(merged, meta)
      }
      return merged
    })
    setShelfItems((current) => current.map((item) => {
      if (item.key !== itemKey) return item
      let merged: CiteDetail = item
      for (const meta of usable) {
        merged = mergeCiteMeta(merged, meta)
      }
      return {
        ...toShelfItem(merged),
        tags: normalizeShelfTags(item.tags),
        note: normalizeShelfNote(item.note),
      }
    }))
  }

  const requestCitationCardPolish = (detail: CiteDetail, itemKey: string, attempt = 0) => {
    const waitSeconds = attempt <= 1 ? 4 : 2
    referencesApi.citationCardPolishCached(detail as unknown as Record<string, unknown>, waitSeconds)
      .then((meta) => {
        if (activePopoverRequestKeyRef.current !== itemKey) return
        const status = String(meta?.citation_card_polish_status || meta?.citationCardPolishStatus || '').trim().toLowerCase()
        if (status === 'pending') {
          if (attempt >= 8) return
          if (citationPolishRetryTimerRef.current !== null) {
            window.clearTimeout(citationPolishRetryTimerRef.current)
          }
          citationPolishRetryTimerRef.current = window.setTimeout(() => {
            citationPolishRetryTimerRef.current = null
            requestCitationCardPolish(detail, itemKey, attempt + 1)
          }, 900 + attempt * 700)
          return
        }
        mergeCitationMetaForItemKey(itemKey, [meta])
      })
      .catch(() => {
        // The card already has deterministic fallback text; LLM polish is a best-effort enhancement.
      })
  }

  const showCitationAt = (detail: CiteDetail, position: { x: number; y: number }, pinned: boolean) => {
    clearCitationHoverTimers()
    if (!pinned && popoverPinned) return
    setPopoverPinned(pinned)
    setPopoverDetail(detail)
    setPopoverPos(position)
    setPopoverGuideLoading(false)
    const sourcePath = String(detail.sourcePath || '').trim()
    const isInPaperReference = Boolean(detail.isInpaper)
    const shouldFetchCitationMeta = Boolean(sourcePath) && !isInPaperReference
    const hasDoi = Boolean(String(detail.doi || '').trim())
    const itemKey = toShelfItem(detail).key
    const needsSummaryBackfill = shelfItemNeedsSummaryBackfill(toShelfItem(detail))
    const shouldFetchBibliometrics = (!detail.bibliometricsChecked || needsSummaryBackfill) && (
      isInPaperReference
        ? hasDoi
        : (detail.doi || detail.title || detail.venue || detail.raw || detail.citeFmt)
    )
    activePopoverRequestKeyRef.current = itemKey
    if (shouldRequestCitationCardPolish(detail)) {
      requestCitationCardPolish(detail, itemKey)
    }
    if (!shouldFetchCitationMeta && !shouldFetchBibliometrics) {
      setPopoverLoading(false)
      return
    }

    const reqs: Array<Promise<Record<string, unknown>>> = []
    if (shouldFetchCitationMeta && sourcePath) {
      reqs.push(referencesApi.citationMetaCached(sourcePath).catch(() => ({})))
    }
    if (shouldFetchBibliometrics) {
      const loadBibliometrics = needsSummaryBackfill
        ? referencesApi.bibliometrics(withBibliometricsLocale(detail as unknown as Record<string, unknown>))
        : referencesApi.bibliometricsCached(withBibliometricsLocale(detail as unknown as Record<string, unknown>))
      reqs.push(loadBibliometrics.catch(() => ({})))
    }

    setPopoverLoading(true)
    Promise.all(reqs)
      .then((metas) => {
        mergeCitationMetaForItemKey(itemKey, metas)
      })
      .finally(() => {
        if (activePopoverRequestKeyRef.current === itemKey) {
          setPopoverLoading(false)
        }
      })
  }

  const openCitation = (detail: CiteDetail, event: MouseEvent<HTMLElement>) => {
    showCitationAt(detail, { x: event.clientX, y: event.clientY }, true)
  }

  const previewCitation = (detail: CiteDetail, event: MouseEvent<HTMLElement>) => {
    if (popoverPinned) return
    const position = { x: event.clientX, y: event.clientY }
    if (citationHoverOpenTimerRef.current !== null) {
      window.clearTimeout(citationHoverOpenTimerRef.current)
    }
    if (citationHoverCloseTimerRef.current !== null) {
      window.clearTimeout(citationHoverCloseTimerRef.current)
      citationHoverCloseTimerRef.current = null
    }
    citationHoverOpenTimerRef.current = window.setTimeout(() => {
      citationHoverOpenTimerRef.current = null
      showCitationAt(detail, position, false)
    }, 180)
  }

  const scheduleCitationPreviewClose = () => {
    if (popoverPinned) return
    if (citationHoverOpenTimerRef.current !== null) {
      window.clearTimeout(citationHoverOpenTimerRef.current)
      citationHoverOpenTimerRef.current = null
    }
    if (citationHoverCloseTimerRef.current !== null) {
      window.clearTimeout(citationHoverCloseTimerRef.current)
    }
    citationHoverCloseTimerRef.current = window.setTimeout(() => {
      citationHoverCloseTimerRef.current = null
      if (citationPolishRetryTimerRef.current !== null) {
        window.clearTimeout(citationPolishRetryTimerRef.current)
        citationPolishRetryTimerRef.current = null
      }
      setPopoverDetail(null)
      setPopoverPos(null)
      activePopoverRequestKeyRef.current = ''
      setPopoverLoading(false)
      setPopoverGuideLoading(false)
    }, 260)
  }

  const keepCitationPreviewOpen = () => {
    if (citationHoverCloseTimerRef.current !== null) {
      window.clearTimeout(citationHoverCloseTimerRef.current)
      citationHoverCloseTimerRef.current = null
    }
  }

  const closeCitationPopover = () => {
    clearCitationHoverTimers()
    setPopoverPinned(false)
    setPopoverDetail(null)
    setPopoverPos(null)
    activePopoverRequestKeyRef.current = ''
    setPopoverLoading(false)
    setPopoverGuideLoading(false)
  }

  const openCitationShelfFromPopover = () => {
    setShelfOpen(true)
    closeCitationPopover()
  }

  const addToShelf = (detail: CiteDetail) => {
    const currentItems = latestShelfStateRef.current.items
    const { nextItems, focusKey, summaryTarget } = mergeCitationDetailIntoShelfItems(currentItems, detail)
    setShelfItems(nextItems)
    setFocusedShelfKey(focusKey)
    setShelfOpen(true)
    persistShelfLocalNow(nextItems, true)
    window.setTimeout(() => {
      fetchShelfSummaryForItem(summaryTarget)
    }, 160)
  }

  const addReaderCitationToShelf = (rawPayload: unknown) => {
    const payload = normalizeReaderCitationShelfPayload(rawPayload)
    if (!payload) return
    const payloadProjectId = String(payload.projectId || '').trim()
    if (payloadProjectId && shelfProjectScopeId(payloadProjectId) !== shelfScopeId) return
    const detail = normalizeCiteDetail(payload.detail)
    if (!detail) return
    addToShelf(detail)
  }
  const addReaderCitationToShelfRef = useRef(addReaderCitationToShelf)
  addReaderCitationToShelfRef.current = addReaderCitationToShelf

  useEffect(() => {
    const handleWindowEvent = (event: Event) => {
      const custom = event as CustomEvent<unknown>
      addReaderCitationToShelfRef.current(custom.detail)
    }
    window.addEventListener(READER_CITATION_SHELF_EVENT, handleWindowEvent)

    let channel: BroadcastChannel | null = null
    if (typeof BroadcastChannel !== 'undefined') {
      channel = new BroadcastChannel(READER_CITATION_SHELF_CHANNEL)
      channel.onmessage = (event) => {
        const data = event?.data && typeof event.data === 'object'
          ? event.data as Record<string, unknown>
          : {}
        if (String(data.type || '') !== 'reader-citation-shelf') return
        addReaderCitationToShelfRef.current(data)
        const requestId = String(data.requestId || '').trim()
        if (requestId) {
          channel?.postMessage({ type: 'reader-citation-shelf-ack', requestId })
        }
      }
    }
    return () => {
      window.removeEventListener(READER_CITATION_SHELF_EVENT, handleWindowEvent)
      channel?.close()
    }
  }, [])

  const addReaderSelectionToShelf = (rawPayload: unknown) => {
    const payload = normalizeReaderSelectionShelfPayload(rawPayload)
    if (!payload) return
    const payloadProjectId = String(payload.projectId || '').trim()
    if (payloadProjectId && shelfProjectScopeId(payloadProjectId) !== shelfScopeId) return
    const detail = citeDetailFromReaderSelection(payload, payload.conversationId || activeConvId)
    if (!detail) return
    const note = readerSelectionNote(payload, S)
    const currentItems = latestShelfStateRef.current.items
    const { nextItems, focusKey, summaryTarget } = mergeReaderSelectionDetailIntoShelfItems(currentItems, detail, {
      text: payload.text,
      note,
      headingPath: payload.headingPath,
      blockId: payload.blockId,
      anchorId: payload.anchorId,
      anchorKind: payload.anchorKind,
    })
    setShelfItems(nextItems)
    setFocusedShelfKey(focusKey)
    setShelfOpen(true)
    persistShelfLocalNow(nextItems, true)
    window.setTimeout(() => {
      fetchShelfSummaryForItem(summaryTarget, { force: true })
    }, 420)
  }
  const addReaderSelectionToShelfRef = useRef(addReaderSelectionToShelf)
  addReaderSelectionToShelfRef.current = addReaderSelectionToShelf

  useEffect(() => {
    const handleWindowEvent = (event: Event) => {
      const custom = event as CustomEvent<unknown>
      addReaderSelectionToShelfRef.current(custom.detail)
    }
    window.addEventListener(READER_SELECTION_SHELF_EVENT, handleWindowEvent)

    let channel: BroadcastChannel | null = null
    if (typeof BroadcastChannel !== 'undefined') {
      channel = new BroadcastChannel(READER_SELECTION_SHELF_CHANNEL)
      channel.onmessage = (event) => {
        const data = event?.data && typeof event.data === 'object'
          ? event.data as Record<string, unknown>
          : {}
        if (String(data.type || '') !== 'reader-selection-shelf') return
        addReaderSelectionToShelfRef.current(data)
        const requestId = String(data.requestId || '').trim()
        if (requestId) {
          channel?.postMessage({ type: 'reader-selection-shelf-ack', requestId })
        }
      }
    }
    return () => {
      window.removeEventListener(READER_SELECTION_SHELF_EVENT, handleWindowEvent)
      channel?.close()
    }
  }, [])

  const startPaperGuideFromDetail = async (detail: CiteDetail) => {
    const isInPaperReference = Boolean(detail.isInpaper)
    const sourcePath = String(detail.sourcePath || '').trim()
    if (!sourcePath) {
      message.info(isInPaperReference ? S.reader_pdf_not_ready : S.reader_missing_path)
      return
    }
    const sourceName = String(detail.sourceName || detail.title || '').trim() || basenameFromSourcePath(sourcePath) || S.default_source_fallback
    setPopoverGuideLoading(true)
    try {
      await createPaperGuideConversation({
        sourcePath,
        sourceName,
        title: `${S.timeline_guide_label} · ${sourceName}`,
      })
      message.success(S.reader_entered_guide)
      setPopoverPinned(false)
      clearCitationHoverTimers()
      setPopoverDetail(null)
      setPopoverPos(null)
      activePopoverRequestKeyRef.current = ''
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.reader_create_guide_failed)
    } finally {
      setPopoverGuideLoading(false)
    }
  }

  const openReaderFromDetail = (detail: CiteDetail) => {
    if (!onOpenReader) return
    const sourcePath = String(detail.sourcePath || '').trim()
    if (!sourcePath) {
      message.info(S.reader_missing_path)
      return
    }
    const payload = buildBasicReaderOpenPayload({
      sourcePath,
      sourceName: String(detail.sourceName || detail.title || '').trim(),
      headingPath: String(detail.headingPath || (!detail.isInpaper ? detail.title : '') || '').trim(),
      snippet: String(detail.evidenceQuote || detail.summaryLine || detail.title || detail.raw || '').trim(),
      highlightSnippet: String(detail.evidenceQuote || detail.summaryLine || detail.raw || '').trim(),
      blockId: String(detail.blockId || '').trim(),
      anchorId: String(detail.anchorId || '').trim(),
      anchorKind: String(detail.anchorKind || '').trim(),
      strictLocate: Boolean(detail.blockId || detail.anchorId),
      locateFeedbackKey: String((detail as CiteShelfItem).key || toShelfItem(detail).key || '').trim(),
    })
    if (!payload) return
    onOpenReader(payload)
  }

  const openMessageFromShelfItem = (item: CiteShelfItem) => {
    const targetId = Number(item.traceAssistantMsgId || item.traceUserMsgId || 0)
    if (!Number.isFinite(targetId) || targetId <= 0) {
      message.info(S.shelf_message_missing)
      return
    }
    const el = scrollRef.current
    if (!el) return
    const target = el.querySelector<HTMLElement>(`[data-msg-id="${targetId}"]`)
    if (!target) {
      message.info(S.shelf_message_not_loaded)
      return
    }
    const targetRect = target.getBoundingClientRect()
    const containerRect = el.getBoundingClientRect()
    const top = targetRect.top - containerRect.top + el.scrollTop - 12
    el.scrollTo({ top: Math.max(0, top), behavior: 'smooth' })
    setShelfMessageFlashId(targetId)
    if (shelfMessageFlashTimerRef.current !== null) {
      window.clearTimeout(shelfMessageFlashTimerRef.current)
    }
    shelfMessageFlashTimerRef.current = window.setTimeout(() => {
      setShelfMessageFlashId((current) => (current === targetId ? null : current))
      shelfMessageFlashTimerRef.current = null
    }, 1400)
  }

  const selectedSavedSnapshot = useMemo(
    () => savedShelfSnapshots.find((item) => item.id === selectedSavedSnapshotId) || null,
    [savedShelfSnapshots, selectedSavedSnapshotId],
  )

  const selectedSnapshotDiff = useMemo(() => {
    if (!selectedSavedSnapshot) return ''
    const diff = snapshotDiffCounts(shelfItems, selectedSavedSnapshot.items)
    if (diff.added <= 0 && diff.removed <= 0) return S.shelf_snapshot_no_diff
    return S.shelf_snapshot_diff
      .replace('{added}', String(diff.added))
      .replace('{removed}', String(diff.removed))
  }, [S, selectedSavedSnapshot, shelfItems])

  const guideSourcePathSet = useMemo(() => {
    const out = new Set<string>()
    for (const item of guideDocCandidates) {
      const sourcePath = String(item.sourcePath || '').trim()
      for (const key of sourcePathLookupKeys(sourcePath)) {
        out.add(key)
      }
    }
    return out
  }, [guideDocCandidates])

  const guideDocCandidatesBySourcePath = useMemo(() => {
    const out = new Map<string, LocateCandidate[]>()
    for (const item of guideDocCandidates) {
      const sourcePath = String(item.sourcePath || '').trim()
      if (!sourcePath) continue
      for (const key of sourcePathLookupKeys(sourcePath)) {
        const list = out.get(key) || []
        list.push(item)
        out.set(key, list)
      }
    }
    return out
  }, [guideDocCandidates])

  const assistantLocatePrepByMsgId = useMemo(() => {
    const nextCache = new Map<string, AssistantLocatePrep>()
    const out = new Map<number, AssistantLocatePrep>()
    const guideSourcePath = String(paperGuideSourcePath || '').trim()
    const guideSourceName = String(paperGuideSourceName || '').trim()
    const prepStartedAt = messageListPerfNow()
    let assistantCount = 0
    let heavyCount = 0
    let lightCount = 0
    let cacheHits = 0
    for (const message of messages) {
      if (message.role !== 'assistant') continue
      assistantCount += 1
      const trace = assistantTraceByMsgId.get(message.id)
      const renderPacket = getMessageRenderPacket(message)
      const locatePayloadSig = messageLocatePayloadSignature(message, renderPacket)
      const rawBodyContent = getMessageRenderedBodyContent(message)
      const lowConfidenceMeta = resolveLowConfidenceMeta(
        (message.meta && typeof message.meta === 'object')
          ? message.meta as Record<string, unknown>
          : null,
        String(rawBodyContent || ''),
        S,
      )
      const bodyContent = lowConfidenceMeta
        ? stripLeadingLowConfidenceNotice(rawBodyContent)
        : rawBodyContent
      const refsUserMsgId = Number(message.refs_user_msg_id || trace?.userMsgId || 0)
      const refEntry = refsUserMsgId > 0 ? (refs[String(refsUserMsgId)] as RefEntryLite | undefined) : undefined
      const refHits = Array.isArray(refEntry?.hits) ? refEntry.hits : []
      const rawCiteDetails = getMessageCiteDetailRecords(message)
      const hasRawCiteDetails = rawCiteDetails.length > 0
      const hasProvenancePayload = Boolean(message.provenance && typeof message.provenance === 'object')
      const hasRenderPacketLocate = Boolean(renderPacket?.readerOpen || renderPacket?.locateTarget)
      const shouldBuildLocatePrep = Boolean(onOpenReader) && (
        Boolean(guideSourcePath)
        || hasRawCiteDetails
        || refHits.length > 0
        || hasProvenancePayload
        || hasRenderPacketLocate
      )
      if (!shouldBuildLocatePrep) {
        const prepKey = [
          message.id,
          String(message.render_cache_key || ''),
          locatePayloadSig,
          'light',
          refsUserMsgId,
        ].join('::')
        const cached = assistantLocatePrepCacheRef.current.get(prepKey)
        if (cached) {
          cacheHits += 1
          nextCache.set(prepKey, cached)
          out.set(message.id, cached)
          continue
        }
        const prep = createEmptyAssistantLocatePrep(bodyContent, refsUserMsgId)
        lightCount += 1
        nextCache.set(prepKey, prep)
        out.set(message.id, prep)
        continue
      }
      const citeDetails = rawCiteDetails
        .map(normalizeCiteDetail)
        .filter((detail): detail is CiteDetail => Boolean(detail))
        .map((detail) => ({
          ...detail,
          traceConvId: String(activeConvId || ''),
          traceAssistantMsgId: message.id,
          traceAssistantOrder: Number(trace?.answerOrder || 0),
          traceUserMsgId: Number(trace?.userMsgId || 0),
        }))
      const uniqueSourcePaths = Array.from(
        new Set(
          citeDetails
            .map((detail) => String(detail.sourcePath || '').trim())
            .filter(Boolean),
        ),
      )
      const guideDocAvailable = Boolean(guideSourcePath && guideSourcePathSet.has(guideSourcePath))
      const guideCandidateCount = guideSourcePath
        ? lookupGuideCandidatesBySourcePath(guideDocCandidatesBySourcePath, guideSourcePath).length
        : 0
      const locateSourcePath = (
        guideSourcePath && guideDocAvailable
          ? guideSourcePath
          : (uniqueSourcePaths.length === 1 ? uniqueSourcePaths[0] : guideSourcePath)
      )
      const locateSourceName = (
        (guideSourcePath && guideDocAvailable ? guideSourceName : '')
        || (citeDetails.find((detail) => String(detail.sourcePath || '').trim() === locateSourcePath)?.sourceName || '')
        || guideSourceName
      )
      const refSig = `${refsUserMsgId}:${String((refEntry as { prompt_sig?: string } | undefined)?.prompt_sig || '')}:${Number((refEntry as { updated_at?: number } | undefined)?.updated_at || 0)}:${refHits.length}`
      const prepKey = [
        message.id,
        String(message.render_cache_key || ''),
        locatePayloadSig,
        guideSourcePath,
        guideCandidateCount,
        locateSourcePath,
        refSig,
      ].join('::')
      const cached = assistantLocatePrepCacheRef.current.get(prepKey)
      if (cached) {
        cacheHits += 1
        nextCache.set(prepKey, cached)
        out.set(message.id, cached)
        continue
      }

      const refsLocateCandidatesAll = buildRefsLocateCandidatesAll(refHits)
      const guideSourceCandidates = guideSourcePath
        ? lookupGuideCandidatesBySourcePath(guideDocCandidatesBySourcePath, guideSourcePath)
        : []
      const refsScopedCandidates = guideSourcePath
        ? refsLocateCandidatesAll.filter((item) => sourcePathsReferToSameDocument(item.sourcePath, guideSourcePath))
        : refsLocateCandidatesAll
      const messageProvenance = (message.provenance && typeof message.provenance === 'object')
        ? message.provenance as Record<string, unknown>
        : null
      const provenanceSourcePath = String(messageProvenance?.source_path || '').trim()
      const provenanceSourceName = String(messageProvenance?.source_name || '').trim()
      const provenanceBlockMap = (messageProvenance?.block_map && typeof messageProvenance.block_map === 'object')
        ? messageProvenance.block_map as Record<string, Record<string, unknown>>
        : {}
      const provenanceDirectSegments = Array.isArray(messageProvenance?.segments)
        ? messageProvenance.segments.filter((segment) => {
          if (!segment || typeof segment !== 'object') return false
          const evidenceMode = String(segment.evidence_mode || '').trim().toLowerCase()
          const locatePolicy = String(segment.locate_policy || '').trim().toLowerCase()
          const evidenceIds = Array.isArray(segment.evidence_block_ids) ? segment.evidence_block_ids : []
          return evidenceMode === 'direct' && locatePolicy !== 'hidden' && evidenceIds.length > 0
        }) as Array<Record<string, unknown>>
        : []
      const hasDirectProvenance = Boolean(provenanceSourcePath) && provenanceDirectSegments.length > 0
      const hasStructuredProvenance = Boolean(
        provenanceSourcePath
        && Array.isArray(messageProvenance?.segments),
      )
      const effectiveGuideSourcePath = String(
        guideSourcePath
        || provenanceSourcePath
        || locateSourcePath
        || '',
      ).trim()
      const strictProvenanceLocate = Boolean(effectiveGuideSourcePath)
      const structuredLocateButtonCap = 12
      const effectiveGuideCandidates = effectiveGuideSourcePath
        ? lookupGuideCandidatesBySourcePath(guideDocCandidatesBySourcePath, effectiveGuideSourcePath)
        : []
      const renderPacketLocateEntry = buildRenderPacketLocateEntry(message, renderPacket, {
        fallbackSourcePath: effectiveGuideSourcePath || provenanceSourcePath || locateSourcePath || '',
        fallbackSourceName: locateSourceName || provenanceSourceName || guideSourceName,
      }, S)
      const provenanceLocateEntries = buildStructuredProvenanceLocateEntries(
        messageProvenance,
        {
          guideSourcePath: effectiveGuideSourcePath,
          fallbackSourceName: locateSourceName,
          maxEntries: structuredLocateButtonCap,
          minConfidence: 0.62,
        },
      ).map((entry) => remapStructuredEntryToGuideAnchors(entry, effectiveGuideCandidates))
      if (provenanceLocateEntries.length <= 0 && renderPacketLocateEntry) {
        provenanceLocateEntries.push(renderPacketLocateEntry)
      }
      const structuredProvenanceSegmentsAll = messageProvenance
        ? listStructuredProvenanceSegments(messageProvenance)
        : []
      const provenanceStrictIdentityReady = Boolean(messageProvenance?.strict_identity_ready)
      const hasStrictMustLocateEntries = provenanceLocateEntries.some((entry) => Boolean(entry.mustLocate || entry.locatePolicy === 'required'))
      const hasRenderPacketStrictLocateEntry = Boolean(
        renderPacketLocateEntry
        && (renderPacketLocateEntry.mustLocate || renderPacketLocateEntry.locatePolicy === 'required'),
      )
      const strictStructuredLocateOnly = Boolean(
        strictProvenanceLocate
        && hasStrictMustLocateEntries
        && (
          (hasStructuredProvenance && provenanceStrictIdentityReady)
          || hasRenderPacketStrictLocateEntry
        ),
      )
      const strictStructuredInlineLocate = Boolean(strictStructuredLocateOnly)
      const provenanceMappingMode = String(messageProvenance?.mapping_mode || '').trim().toLowerCase()
      const provenanceLlmCallsRaw = Number(messageProvenance?.llm_rerank_calls || 0)
      const provenanceLlmCalls = Number.isFinite(provenanceLlmCallsRaw) && provenanceLlmCallsRaw > 0
        ? Math.floor(provenanceLlmCallsRaw)
        : 0
      const provenanceModeLabel = (() => {
        if (!strictStructuredLocateOnly) return ''
        if (provenanceMappingMode === 'llm_refined') {
          if (provenanceLlmCalls > 0) return `\u5b9a\u4f4d\u6620\u5c04\uff1aLLM\u7cbe\u4fee\uff08${provenanceLlmCalls} \u6b21\uff09`
          return '\u5b9a\u4f4d\u6620\u5c04\uff1aLLM\u7cbe\u4fee'
        }
        if (provenanceMappingMode === 'fast') return '\u5b9a\u4f4d\u6620\u5c04\uff1a\u5feb\u901f\u6620\u5c04'
        if (hasStructuredProvenance) return '\u5b9a\u4f4d\u6620\u5c04\uff1a\u7ed3\u6784\u5316\u6620\u5c04'
        return ''
      })()
      const structuredRenderSlotMap = buildStructuredRenderLocateSlotMap(
        String(bodyContent || ''),
        messageProvenance,
        provenanceLocateEntries,
      )
      const structuredLocateOrderBySegmentId = (() => {
        const map = new Map<string, number>()
        for (const slot of structuredRenderSlotMap.values()) {
          const segmentId = String(slot.entry.segmentId || '').trim()
          if (!segmentId || map.has(segmentId)) continue
          map.set(segmentId, Number(slot.order || 0))
        }
        return map
      })()
      const allowedStructuredRenderOrders = (() => {
        const ordered = Array.from(structuredRenderSlotMap.values()).sort((a, b) => a.order - b.order)
        const allowed = new Set<number>()
        let optionalCount = 0
        for (const slot of ordered) {
          if (slot.entry.mustLocate || slot.entry.locatePolicy === 'required') {
            allowed.add(slot.order)
            continue
          }
          if (optionalCount >= structuredLocateButtonCap) continue
          allowed.add(slot.order)
          optionalCount += 1
        }
        return allowed
      })()
      const locateCandidates = (() => {
        if (guideSourceCandidates.length > 0) return [...guideSourceCandidates, ...refsScopedCandidates]
        if (refsScopedCandidates.length > 0) return refsScopedCandidates
        if (refsLocateCandidatesAll.length > 0) return refsLocateCandidatesAll
        if (guideSourcePath) return guideDocCandidates
        return []
      })()

      const prep: AssistantLocatePrep = {
        bodyContent,
        refsUserMsgId,
        locateSourcePath,
        locateSourceName,
        refsLocateCandidatesAll,
        guideLocateCandidates: guideSourceCandidates,
        refsScopedCandidates,
        messageProvenance,
        provenanceSourcePath,
        provenanceSourceName,
        provenanceBlockMap,
        provenanceDirectSegments,
        hasDirectProvenance,
        hasStructuredProvenance,
        effectiveGuideSourcePath,
        strictProvenanceLocate,
        structuredLocateButtonCap,
        provenanceLocateEntries,
        structuredProvenanceSegmentsAll,
        provenanceStrictIdentityReady,
        hasStrictMustLocateEntries,
        strictStructuredLocateOnly,
        strictStructuredInlineLocate,
        provenanceModeLabel,
        structuredRenderSlotMap,
        structuredLocateOrderBySegmentId,
        allowedStructuredRenderOrders,
        locateCandidates,
      }
      heavyCount += 1
      nextCache.set(prepKey, prep)
      out.set(message.id, prep)
    }
    assistantLocatePrepCacheRef.current = nextCache
    assistantLocatePrepPerfRef.current = {
      ts: Date.now(),
      convId: String(activeConvId || ''),
      messageCount: messages.length,
      assistantCount,
      heavyCount,
      lightCount,
      cacheHits,
      durationMs: Number((messageListPerfNow() - prepStartedAt).toFixed(2)),
    }
    return out
  }, [
    activeConvId,
    assistantTraceByMsgId,
    guideDocCandidates,
    guideDocCandidatesBySourcePath,
    guideSourcePathSet,
    messages,
    onOpenReader,
    paperGuideSourceName,
    paperGuideSourcePath,
    refs,
    S,
  ])

  useEffect(() => {
    const perf = assistantLocatePrepPerfRef.current
    if (!perf) return
    pushMessageListPrepPerf(perf)
  }, [activeConvId, assistantLocatePrepByMsgId])

  const saveShelfSnapshot = () => {
    const currentItems = dedupeShelfItems(shelfItems).slice(0, SHELF_MAX_ITEMS)
    if (currentItems.length <= 0) {
      message.info(S.shelf_version_empty || 'Shelf is empty; cannot save local snapshot')
      return
    }
    const now = Date.now()
    const d = new Date(now)
    const pad = (value: number) => String(value).padStart(2, '0')
    const versionTime = `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
    const entry: ShelfSavedSnapshot = {
      id: `s_${now.toString(36)}_${Math.random().toString(36).slice(2, 7)}`,
      name: (S.shelf_version_name || 'Version {time}').replace('{time}', versionTime),
      createdAt: now,
      items: currentItems.map((item) => ({ ...item })),
    }
    setSavedShelfSnapshots((current) => {
      const next = [entry, ...current].slice(0, SHELF_SAVED_MAX_ITEMS)
      persistSavedShelfSnapshots(shelfSavedStorageKey(shelfScopeId), next)
      return next
    })
    setSelectedSavedSnapshotId(entry.id)
    message.success(S.shelf_version_saved || 'Saved to this browser')
  }

  const loadShelfSnapshot = () => {
    if (!selectedSavedSnapshot) return
    const restored = dedupeShelfItems(selectedSavedSnapshot.items).slice(0, SHELF_MAX_ITEMS).map((item) => ({ ...item }))
    setShelfItems(restored)
    setFocusedShelfKey('')
    setShelfSummaryLoadingKey('')
    setShelfRepairLoadingKey('')
    message.success((S.shelf_version_loaded || 'Restored local snapshot: {name}').replace('{name}', selectedSavedSnapshot.name))
  }

  const deleteShelfSnapshot = () => {
    if (!selectedSavedSnapshot) return
    const removedName = selectedSavedSnapshot.name
    setSavedShelfSnapshots((current) => {
      const next = current.filter((item) => item.id !== selectedSavedSnapshot.id)
      persistSavedShelfSnapshots(shelfSavedStorageKey(shelfScopeId), next)
      return next
    })
    setSelectedSavedSnapshotId((current) => (current === selectedSavedSnapshot.id ? '' : current))
    message.success((S.shelf_version_deleted || 'Deleted local snapshot: {name}').replace('{name}', removedName))
  }

  const useSelectedShelfItemsAsContext = useCallback((items: CiteShelfItem[]) => {
    const pack = buildSelectedResearchContextPack(items, {
      conversationId: activeConvId || '',
      guideSourcePath: paperGuideSourcePath || '',
      guideSourceName: paperGuideSourceName || '',
    })
    if (!pack) {
      message.info(S.research_context_empty_toast || 'No usable context in the selected items')
      return
    }
    onResearchContextPackChange?.(pack)
    message.success(
      (S.research_context_selected_toast || 'Added {n} selected items to the next answer context')
      .replace('{n}', String(pack.items.length)),
    )
  }, [S, activeConvId, onResearchContextPackChange, paperGuideSourceName, paperGuideSourcePath])

  const useReceiptItemAsFollowUp = useCallback((sourcePack: SelectedResearchContextPack, item: SelectedResearchContextItem) => {
    if (!onResearchContextFollowUp) return
    const pack = buildSelectedResearchContextPackFromItems([item], {
      conversationId: activeConvId || sourcePack.conversationId || '',
      guideSourcePath: sourcePack.guideSourcePath || paperGuideSourcePath || '',
      guideSourceName: sourcePack.guideSourceName || paperGuideSourceName || '',
    })
    if (!pack) {
      message.info(S.research_context_empty_toast || 'No usable context in this item')
      return
    }
    const title = contextItemTitle(item, S.default_source_fallback || 'Untitled')
    const promptText = (S.research_context_followup_prompt || 'Continue with this selected context: {title}\n\n')
      .replace('{title}', title)
    onResearchContextFollowUp(pack, promptText)
    message.success(S.research_context_followup_toast || 'Ready for a follow-up question')
  }, [S, activeConvId, onResearchContextFollowUp, paperGuideSourceName, paperGuideSourcePath])

  const shelfNode = (
    <CiteShelf
      open={shelfOpen}
      visible={shelfDockMode ? shelfVisible : undefined}
      presentation={shelfDockMode ? 'dock' : 'floating'}
      items={shelfItems}
      activeConvId={activeConvId}
      activeSourcePath={paperGuideSourcePath}
      readerLocateResults={readerLocateResults}
      sourceQualityRefreshToken={sourceQualityRefreshToken}
      focusedKey={focusedShelfKey}
      summaryLoadingKey={shelfSummaryLoadingKey}
      repairLoadingKey={shelfRepairLoadingKey}
      repairingKeys={shelfAutoRepairingKeys}
      repairImpact={shelfRepairImpact}
      activeContextKeys={selectedResearchContextKeys}
      snapshots={savedShelfSnapshots}
      selectedSnapshotId={selectedSavedSnapshotId}
      snapshotDiff={selectedSnapshotDiff}
      onToggle={() => setShelfOpen((value) => !value)}
      onSelect={(item) => {
        setFocusedShelfKey(item.key)
        fetchShelfSummaryForItem(item)
      }}
      onOpenSource={(item) => {
        openReaderFromDetail(item as unknown as CiteDetail)
      }}
      onOpenMessage={openMessageFromShelfItem}
      onUseSelectedAsContext={onResearchContextPackChange ? useSelectedShelfItemsAsContext : undefined}
      onRemove={(key) => {
        const willBeEmpty = latestShelfStateRef.current.items.filter((item) => item.key !== key).length <= 0
        if (willBeEmpty) markShelfEmptyBackendSaveIntent(shelfScopeId)
        setShelfItems((current) => current.filter((item) => item.key !== key))
        if (focusedShelfKey === key) setFocusedShelfKey('')
        if (shelfSummaryLoadingKey === key) setShelfSummaryLoadingKey('')
        if (shelfRepairLoadingKey === key) setShelfRepairLoadingKey('')
        const nextRepairing = new Set(shelfAutoRepairingKeySetRef.current)
        nextRepairing.delete(key)
        setShelfAutoRepairingKeySet(nextRepairing)
        delete shelfAutoRepairFingerprintsRef.current[key]
        delete shelfAutoRepairRetryAfterRef.current[key]
      }}
      onClear={() => {
        markShelfEmptyBackendSaveIntent(shelfScopeId)
        setShelfItems([])
        setFocusedShelfKey('')
        setShelfSummaryLoadingKey('')
        setShelfRepairLoadingKey('')
        setShelfAutoRepairingKeySet(new Set())
        shelfAutoRepairFingerprintsRef.current = {}
        shelfAutoRepairRetryAfterRef.current = {}
        setShelfRepairImpact(null)
        const projectScopeId = shelfProjectScopeId(shelfScopeId)
        const storageKey = shelfStorageKey(projectScopeId)
        void chatApi.deleteCitationShelf({
          convId: activeConvId || undefined,
          projectId: projectScopeId === '__default__' ? undefined : projectScopeId,
          scope: 'project',
        })
          .then((record) => {
            shelfBackendRevisionByKeyRef.current[storageKey] = Math.max(0, Number(record.revision || 0))
            shelfBackendHydratedKeysRef.current.add(storageKey)
            delete shelfEmptyBackendSaveIntentRef.current[storageKey]
          })
          .catch(() => {
            // Local state remains cleared; the guarded save path will avoid accidental backend overwrite.
          })
      }}
      onUpdateTags={(key, tags) => {
        const nextTags = normalizeShelfTags(tags)
        setShelfItems((current) => current.map((item) => (
          item.key === key ? { ...item, tags: nextTags } : item
        )))
      }}
      onUpdateNote={(key, note) => {
        const nextNote = normalizeShelfNote(note)
        setShelfItems((current) => current.map((item) => (
          item.key === key ? { ...item, note: nextNote } : item
        )))
      }}
      onRepair={(item, options) => {
        repairShelfItemMeta(item, options)
      }}
      onApplyRepairCandidates={applyShelfMetadataRepairCandidates}
      onSelectSnapshot={setSelectedSavedSnapshotId}
      onSaveSnapshot={saveShelfSnapshot}
      onLoadSnapshot={loadShelfSnapshot}
      onDeleteSnapshot={deleteShelfSnapshot}
      onBackgroundActivityChange={setShelfBackgroundBusy}
    />
  )
  const renderedShelfNode = shelfDockMode
    ? (shelfPortalTarget ? createPortal(shelfNode, shelfPortalTarget) : null)
    : shelfNode
  const cleanGenerationPartial = generationPartial !== undefined && generationPartial !== null
    ? cleanAssistantAnswerPresentationText(generationPartial)
    : ''

  return (
    <>
      <div ref={scrollRef} className="kb-message-scroll kb-main-scroll">
        <div className="kb-message-stack">
          {rows.map((row, index) => {
            if (row.kind === 'refs') {
              return (
                <div key={`refs-${row.userMsgId}-${index}`} className="kb-message-row kb-message-row-refs">
                  <div className="kb-msg-avatar-spacer" />
                  <div className="kb-message-refs-wrap">
                    <RefsPanel
                      refs={refs}
                      msgId={row.userMsgId}
                      onOpenReader={onOpenReader}
                      activeSourcePath={paperGuideSourcePath}
                      activeSourceName={paperGuideSourceName}
                    />
                  </div>
                </div>
              )
            }

            const message = row.message
            const isUser = message.role === 'user'
            const trace = assistantTraceByMsgId.get(message.id)
            const agentTrace = !isUser ? getMessageAgentTrace(message) : null
            const canLoadAgentTrace = !isUser ? messageHasAgentTraceHint(message) : false
            const researchTrace = !isUser ? getMessageResearchTrace(message) : null
            const selectedResearchContextPack = !isUser
              ? selectedResearchContextByAssistantId.get(Number(message.id)) || null
              : null
            const renderPacket = !isUser ? getMessageRenderPacket(message) : null
            const citeDetails = getMessageCiteDetailRecords(message)
              .map(normalizeCiteDetail)
              .filter((detail): detail is CiteDetail => Boolean(detail))
              .map((detail) => ({
                ...detail,
                traceConvId: String(activeConvId || ''),
                traceAssistantMsgId: message.id,
                traceAssistantOrder: Number(trace?.answerOrder || 0),
                traceUserMsgId: Number(trace?.userMsgId || 0),
              }))
            const imageAttachments = imageAttachmentsOf(message)
            const showUserText = !(isUser && imageAttachments.length > 0 && isImageOnlyPlaceholder(message.content))
            const isImageOnlyUserMessage = isUser && imageAttachments.length > 0 && !showUserText
            const prep = !isUser ? assistantLocatePrepByMsgId.get(message.id) : undefined
            const rawBodyContent = prep?.bodyContent || getMessageRenderedBodyContent(message)
            const lowConfidenceMeta = !isUser
              ? resolveLowConfidenceMeta(
                (message.meta && typeof message.meta === 'object')
                  ? message.meta as Record<string, unknown>
                  : null,
                String(rawBodyContent || ''),
                S,
              )
              : null
            const bodyContent = lowConfidenceMeta
              ? stripLeadingLowConfidenceNotice(rawBodyContent)
              : rawBodyContent
            const refsUserMsgIdForCitations = Number(prep?.refsUserMsgId || message.refs_user_msg_id || trace?.userMsgId || 0)
            const refEntryForCitations = refsUserMsgIdForCitations > 0
              ? refs[String(refsUserMsgIdForCitations)] as RefEntryLite | undefined
              : undefined
            const fallbackCiteDetails = (!isUser && citeDetails.length <= 0 && Array.isArray(refEntryForCitations?.hits))
              ? buildFallbackCiteDetailsFromRefHits({
                bodyContent: String(bodyContent || ''),
                refHits: refEntryForCitations?.hits || [],
                messageId: message.id,
                traceConvId: String(activeConvId || ''),
                traceAssistantOrder: Number(trace?.answerOrder || 0),
                traceUserMsgId: Number(trace?.userMsgId || refsUserMsgIdForCitations || 0),
                S,
              })
              : []
            const effectiveCiteDetails = enrichCiteDetailsWithVisibleRefContext(
              citeDetails.length > 0 ? citeDetails : fallbackCiteDetails,
              refEntryForCitations,
            )
            const unlinkedReferenceViews = !isUser
              ? buildUnlinkedReferenceViews({
                packet: renderPacket,
                linkedDetails: effectiveCiteDetails,
                messageId: message.id,
                traceConvId: String(activeConvId || ''),
                traceAssistantOrder: Number(trace?.answerOrder || 0),
                traceUserMsgId: Number(trace?.userMsgId || refsUserMsgIdForCitations || 0),
                S,
              })
              : []
            const guideSourcePath = String(paperGuideSourcePath || '').trim()
            const locateSourceName = prep?.locateSourceName || String(paperGuideSourceName || '').trim()
            const messageProvenance = prep?.messageProvenance || (
              message.provenance && typeof message.provenance === 'object'
                ? message.provenance as Record<string, unknown>
                : null
            )
            const provenanceSourcePath = prep?.provenanceSourcePath || ''
            const provenanceSourceName = prep?.provenanceSourceName || ''
            const provenanceBlockMap = prep?.provenanceBlockMap || {} as Record<string, Record<string, unknown>>
            const provenanceDirectSegments = prep?.provenanceDirectSegments || []
            const hasDirectProvenance = prep?.hasDirectProvenance || false
            const hasStructuredProvenance = prep?.hasStructuredProvenance || false
            const effectiveGuideSourcePath = prep?.effectiveGuideSourcePath || guideSourcePath
            const strictProvenanceLocate = prep?.strictProvenanceLocate || false
            const provenanceLocateEntries = prep?.provenanceLocateEntries || []
            const structuredProvenanceSegmentsAll = prep?.structuredProvenanceSegmentsAll || []
            const provenanceStrictIdentityReady = prep?.provenanceStrictIdentityReady || false
            const hasStrictMustLocateEntries = prep?.hasStrictMustLocateEntries || false
            const strictStructuredLocateOnly = prep?.strictStructuredLocateOnly || false
            const strictStructuredInlineLocate = prep?.strictStructuredInlineLocate || false
            const suppressLooseInlineLocate = shouldSuppressLooseInlineLocate({
              guideSourcePath,
              bodyContent: String(bodyContent || ''),
              hasRawCiteDetails: effectiveCiteDetails.length > 0,
              hasStructuredProvenance,
              hasDirectProvenance,
            })
            const guideInlineTextTailLocate = Boolean(
              !suppressLooseInlineLocate
              && (
              guideSourcePath
              && provenanceLocateEntries.length > 0
              ),
            )
            const provenanceModeLabel = prep?.provenanceModeLabel || ''
            const structuredRenderSlotMap = prep?.structuredRenderSlotMap || new Map<number, StructuredRenderLocateSlot>()
            const structuredLocateOrderBySegmentId = prep?.structuredLocateOrderBySegmentId || new Map<string, number>()
            const allowedStructuredRenderOrders = prep?.allowedStructuredRenderOrders || new Set<number>()
            const structuredInlineLocateResolver = createStructuredInlineLocateResolver({
              strictStructuredInlineLocate,
              provenanceLocateEntries,
              structuredRenderSlotMap,
              structuredLocateOrderBySegmentId,
              messageProvenance,
              structuredProvenanceSegmentsAll,
              provenanceBlockMap,
              provenanceSourcePath,
              effectiveGuideSourcePath,
              provenanceSourceName,
              locateSourceName,
            })
            const {
              resolveExactStructuredInlineResolution,
              resolveStrictParagraphEntry,
              isStrictStructuredTargetCompatible,
            } = structuredInlineLocateResolver
            const resolveProvenanceLocateCandidates = (snippet: string, limit = 4): LocateCandidate[] => {
              const raw = stripProvenanceNoise(stripMarkdownInline(String(snippet || '')))
              const key = normalizeLocateText(raw).slice(0, 360)
              if (!key || !provenanceSourcePath) return []
              const formulaQuery = hasFormulaSignal(raw)
              const quoteSpans = formulaQuery ? [] : extractQuotedSpans(raw, 12)
              const rankedSegments: Array<{ segment: NonNullable<typeof provenanceDirectSegments[number]>; score: number }> = []
              for (const segment of provenanceDirectSegments) {
                const segmentText = String(segment.text || '')
                if (isLikelyRhetoricalLocateShell(segmentText)) continue
                const segmentKey = String(segment.snippet_key || '')
                const segmentConf = Number(segment.evidence_confidence || 0)
                const confFloor = formulaQuery ? 0.5 : 0.62
                if (segmentConf > 0 && segmentConf < confFloor) continue
                let score = scoreProvenanceSegment(raw, segmentText, segmentKey)
                if (quoteSpans.length > 0) {
                  const qSeg = quoteMatchStats(quoteSpans, segmentText, segmentKey)
                  if (qSeg.hits <= 0 && qSeg.score < 0.55) continue
                  score += 0.38 * qSeg.score + (qSeg.hits > 0 ? 0.22 : 0)
                }
                if (score > 0) rankedSegments.push({ segment, score })
              }
              rankedSegments.sort((a, b) => b.score - a.score)
              const scoreFloor = formulaQuery ? 0.45 : 0.5
              let matchedSegments = rankedSegments
                .filter((row) => row.score >= scoreFloor)
                .slice(0, 1)
              if (formulaQuery && matchedSegments.length <= 0) {
                matchedSegments = rankedSegments
                  .filter((row) => row.score >= 0.42)
                  .slice(0, 1)
              }
              const out: LocateCandidate[] = []
              const seen = new Set<string>()
              for (const row of matchedSegments) {
                const segment = row.segment
                const evidenceIds = Array.isArray(segment.evidence_block_ids) ? segment.evidence_block_ids : []
                for (const blockIdRaw of evidenceIds.slice(0, formulaQuery ? 2 : 1)) {
                  const blockId = String(blockIdRaw || '').trim()
                  if (!blockId) continue
                  const block = provenanceBlockMap[blockId]
                  if (!block) continue
                  const blockKind = String(block.kind || '').trim().toLowerCase()
                  if (formulaQuery && blockKind && blockKind !== 'equation') continue
                  if (!formulaQuery && blockKind === 'equation' && evidenceIds.length > 1) continue
                  const blockText = String(block.text || '').trim()
                  if (quoteSpans.length > 0) {
                    const qBlock = quoteMatchStats(quoteSpans, blockText, String(segment.text || ''), String(block.heading_path || ''))
                    if (qBlock.hits <= 0 && qBlock.score < 0.85) continue
                  }
                  const key0 = `${provenanceSourcePath}::${blockId}`
                  if (seen.has(key0)) continue
                  seen.add(key0)
                  const segmentFocus = String(segment.text || '').trim()
                  const blockFocus = blockText
                  const focusSnippet = (formulaQuery ? (blockFocus || segmentFocus) : (segmentFocus || blockFocus))
                  if (!focusSnippet) continue
                  out.push({
                    sourcePath: provenanceSourcePath,
                    sourceName: provenanceSourceName || locateSourceName || provenanceSourcePath.split(/[\\/]/).pop() || 'paper',
                    headingPath: String(block.heading_path || '').trim(),
                    focusSnippet,
                    matchText: [String(block.heading_path || '').trim(), String(block.text || segment.text || '').trim()].filter(Boolean).join('\n'),
                    sourceType: 'guide',
                    blockId,
                    anchorId: String(block.anchor_id || '').trim() || undefined,
                    anchorKind: String(block.kind || '').trim().toLowerCase() || undefined,
                    anchorNumber: Number(block.number || 0) > 0 ? Math.floor(Number(block.number || 0)) : undefined,
                  })
                  if (out.length >= Math.max(1, limit)) return out
                }
              }
              return out
            }
            const locateCandidates = prep?.locateCandidates || (guideSourcePath ? guideDocCandidates : [])
            const enableLocateUi = Boolean(onOpenReader) && (
              strictStructuredLocateOnly
              || strictStructuredInlineLocate
              || hasDirectProvenance
              || provenanceLocateEntries.length > 0
              || locateCandidates.length > 0
            )
            const hasInlineLocateSurface = Boolean(enableLocateUi && (
              guideInlineTextTailLocate
              || strictStructuredInlineLocate
              || (!guideSourcePath && !suppressLooseInlineLocate)
            ))
            const showProvenanceLocateChips = Boolean(onOpenReader)
              && provenanceLocateEntries.length > 0
              && !hasInlineLocateSurface
            const resolveCache = new Map<string, LocateCandidate[]>()
            const usedCount = new Map<string, number>()
            const resolveLocateCandidates = (snippet: string, limit = 4) => {
              const key = String(snippet || '').trim()
              if (!key) return []
              if (resolveCache.has(key)) return (resolveCache.get(key) || []).slice(0, Math.max(1, limit))
              const formulaQuery = hasFormulaSignal(key)
              const guideOnly = locateCandidates.filter((item) => item.sourceType === 'guide')
              const strictDirectMode = hasDirectProvenance && !formulaQuery && guideOnly.length > 0
              const provenancePicked = resolveProvenanceLocateCandidates(key, limit)
              if (provenancePicked.length > 0) {
                const picked = formulaQuery
                  ? (() => {
                    const eqProv = provenancePicked.filter((cand) => isEquationLocateCandidate(cand))
                    return eqProv.length > 0 ? eqProv : provenancePicked
                  })()
                  : provenancePicked
                resolveCache.set(key, picked)
                return picked.slice(0, Math.max(1, limit))
              }
              if (strictProvenanceLocate && hasStructuredProvenance && provenanceStrictIdentityReady && hasStrictMustLocateEntries) {
                // Paper-guide mode should not fall back to fuzzy locate if this
                // message already has strict-ready structured provenance but no
                // direct evidence for the current snippet.
                resolveCache.set(key, [])
                return []
              }
              const quoteSpans = formulaQuery ? [] : extractQuotedSpans(key, 12)
              if (!formulaQuery && quoteSpans.length > 0) {
                const quotePool = guideOnly.length > 0 ? guideOnly : locateCandidates
                const quoteRank = quotePool
                  .map((cand) => {
                    const q = quoteMatchStats(quoteSpans, cand.matchText, cand.focusSnippet, cand.headingPath)
                    let score = q.score + (0.35 * scoreLocateCandidate(key, cand))
                    if (q.hits > 0) score += 0.35
                    if (cand.sourceType === 'guide') score += 0.08
                    if (cand.anchorId || cand.blockId) score += 0.1
                    return { cand, score, hits: q.hits }
                  })
                  .sort((a, b) => b.score - a.score)
                const bestQuote = quoteRank[0]
                if (bestQuote && bestQuote.hits > 0 && bestQuote.score >= 1.05) {
                  resolveCache.set(key, [bestQuote.cand])
                  return [bestQuote.cand]
                }
              }

              const rankIn = (cands: LocateCandidate[]) => {
                const scored: Array<{ cand: LocateCandidate; score: number }> = []
                for (const cand of cands) {
                  const base = scoreLocateCandidate(key, cand)
                  const candKey = `${cand.sourcePath}::${cand.anchorId || ''}::${cand.headingPath}::${cand.focusSnippet.slice(0, 96)}`
                  const penalty = 0.03 * Number(usedCount.get(candKey) || 0)
                  const score = base - penalty
                  scored.push({ cand, score })
                }
                scored.sort((a, b) => b.score - a.score)
                return scored
              }

              const picked: LocateCandidate[] = []
              const pickedKeySet = new Set<string>()
              const pickedHeadingSet = new Set<string>()
              const addPicked = (cand: LocateCandidate, preferNewHeading = false) => {
                const candKey = `${cand.sourcePath}::${cand.anchorId || ''}::${cand.headingPath}::${cand.focusSnippet.slice(0, 96)}`
                if (pickedKeySet.has(candKey)) return false
                const headingRaw = String(cand.headingPath || '').trim()
                const headingKey = headingRaw
                  ? normalizeLocateText(headingRaw)
                  : normalizeLocateText(String(cand.focusSnippet || '').slice(0, 56))
                if (preferNewHeading && headingKey && pickedHeadingSet.has(headingKey)) return false
                picked.push(cand)
                pickedKeySet.add(candKey)
                if (headingKey) pickedHeadingSet.add(headingKey)
                return true
              }
              const ingestRank = (
                rankRows: Array<{ cand: LocateCandidate; score: number }>,
                floor: number,
                preferNewHeading: boolean,
              ) => {
                for (const row of rankRows) {
                  if (row.score < floor) break
                  addPicked(row.cand, preferNewHeading)
                  if (picked.length >= limit) break
                }
              }

              if (hasDirectProvenance && formulaQuery) {
                const eqNums = extractEquationNumbersFromText(key)
                const eqGuide = guideOnly.filter((cand) => isEquationLocateCandidate(cand))
                if (eqGuide.length > 0) {
                  let bestEq: LocateCandidate | null = null
                  let bestEqScore = -1
                  for (const cand of eqGuide) {
                    let s = scoreLocateCandidate(key, cand)
                    if (eqNums.length > 0 && Number(cand.anchorNumber || 0) > 0 && eqNums.includes(Math.floor(Number(cand.anchorNumber || 0)))) {
                      s += 0.45
                    }
                    if (cand.anchorId) s += 0.2
                    if (s > bestEqScore) {
                      bestEq = cand
                      bestEqScore = s
                    }
                  }
                  if (bestEq && bestEqScore >= 0.58) {
                    resolveCache.set(key, [bestEq])
                    return [bestEq]
                  }
                }
              }
              if (guideOnly.length > 0) {
                const guideRank = rankIn(guideOnly)
                const guideFloor = strictDirectMode
                  ? 0.34
                  : (hasFormulaSignal(key) ? 0.32 : 0.2)
                ingestRank(guideRank, guideFloor, true)
                if (picked.length < limit) ingestRank(guideRank, guideFloor, false)
              }
              if (picked.length < limit) {
                const strictPool = strictDirectMode
                  ? locateCandidates.filter((item) => String(item.sourcePath || '').trim() === provenanceSourcePath)
                  : []
                const rankBase = (strictDirectMode && strictPool.length > 0) ? strictPool : locateCandidates
                const rankAll = rankIn(rankBase)
                const allFloor = strictDirectMode
                  ? (hasFormulaSignal(key) ? 0.34 : 0.24)
                  : (hasFormulaSignal(key) ? 0.3 : 0.2)
                ingestRank(rankAll, allFloor, true)
                if (picked.length < limit) ingestRank(rankAll, allFloor, false)
                if (picked.length <= 0 && rankAll.length > 0) {
                  const best = rankAll[0]
                  const preferAnchor = Boolean(best?.cand?.anchorId)
                  const bestFloor = preferAnchor
                    ? (hasFormulaSignal(key) ? 0.3 : 0.24)
                    : (hasFormulaSignal(key) ? 0.38 : 0.3)
                  if ((best?.score || 0) >= bestFloor) {
                    addPicked(best.cand, false)
                  }
                }
              }
              if (picked.length <= 0 && hasFormulaSignal(key) && guideOnly.length > 0) {
                const eqNums = extractEquationNumbersFromText(key)
                const eqCandidates = guideOnly.filter((cand) => isEquationLocateCandidate(cand))
                if (eqCandidates.length > 0) {
                  const preferByNum = eqNums.length > 0
                    ? eqCandidates.filter((cand) => {
                      const n = Number(cand.anchorNumber || 0)
                      return Number.isFinite(n) && n > 0 && eqNums.includes(Math.floor(n))
                    })
                    : []
                  const pool = preferByNum.length > 0 ? preferByNum : eqCandidates
                  let bestEq: LocateCandidate | null = null
                  let bestEqScore = -1
                  for (const cand of pool) {
                    let s = scoreLocateCandidate(key, cand)
                    if (eqNums.length > 0 && Number(cand.anchorNumber || 0) > 0) s += 0.4
                    if (cand.anchorId) s += 0.2
                    if (s > bestEqScore) {
                      bestEq = cand
                      bestEqScore = s
                    }
                  }
                  if (bestEq && bestEqScore >= 0.34) addPicked(bestEq, false)
                }
              }
              const unique: LocateCandidate[] = []
              const seen = new Set<string>()
              for (const cand of picked) {
                const candKey = `${cand.sourcePath}::${cand.anchorId || ''}::${cand.headingPath}::${cand.focusSnippet.slice(0, 96)}`
                if (seen.has(candKey)) continue
                seen.add(candKey)
                unique.push(cand)
                if (unique.length >= limit) break
              }
              if (unique.length <= 0 && guideOnly.length > 0) {
                const relaxed = rankIn(guideOnly)
                const best = relaxed[0]
                if (best && (best.score || 0) >= 0.08) {
                  unique.push(best.cand)
                }
              }
              const first = unique[0]
              if (first) {
                const pickKey = `${first.sourcePath}::${first.anchorId || ''}::${first.headingPath}::${first.focusSnippet.slice(0, 96)}`
                usedCount.set(pickKey, Number(usedCount.get(pickKey) || 0) + 1)
              }
              resolveCache.set(key, unique)
              return unique.slice(0, Math.max(1, limit))
            }
            const locateButtonShownKeys = new Set<string>()
            const locateButtonCap = 5
            let optionalLocateButtonCount = 0
            const locateCandidateKey = (cand: LocateCandidate | null) => {
              if (!cand) return ''
              if (cand.blockId) return `${cand.sourcePath}::block::${cand.blockId}`
              if (cand.anchorId) return `${cand.sourcePath}::anchor::${cand.anchorId}`
              const headingKey = normalizeLocateText(String(cand.headingPath || ''))
              const focusKey = normalizeLocateText(String(cand.focusSnippet || '')).slice(0, 64)
              return `${cand.sourcePath}::${headingKey}::${focusKey}`
            }
            const openReaderByCandidates = (
              pickedList: LocateCandidate[],
              snippet: string,
              opts?: { strictLocate?: boolean; highlightSnippet?: string; relatedBlockIds?: string[] },
            ) => {
              if (!onOpenReader) return
              const payload = buildHeuristicReaderOpenPayload(pickedList, snippet, opts)
              if (!payload) return
              onOpenReader(payload)
            }
            const openReaderByStructuredEntry = (entry: ProvenanceLocateEntry, snippet: string) => {
              if (!onOpenReader) return
              const sourcePath = String(entry.primary?.sourcePath || '').trim()
              const resolvedEntry = remapStructuredEntryToGuideAnchors(
                entry,
                sourcePath
                  ? lookupGuideCandidatesBySourcePath(guideDocCandidatesBySourcePath, sourcePath)
                  : [],
              )
              const payload = buildStructuredEntryReaderOpenPayload(resolvedEntry, snippet)
              if (!payload) return
              onOpenReader(payload)
            }
            const bubbleClass = isUser
              ? `kb-msg-bubble kb-msg-bubble-user ${isImageOnlyUserMessage ? 'is-image-only' : ''}`
              : 'kb-msg-bubble kb-msg-bubble-assistant'

            return (
              <div
                key={message.id}
                data-msg-id={message.id}
                className={`kb-message-row ${isUser ? 'is-user' : 'is-assistant'} ${shelfMessageFlashId === message.id ? 'is-shelf-jump' : ''}`}
              >
                {!isUser ? <AssistantAvatar /> : null}
                <div className={bubbleClass}>
                  {isUser ? (
                    <>
                      {imageAttachments.length > 0 ? (
                        <div
                          className={`${
                            showUserText ? 'mb-3' : ''
                          } grid ${
                            imageAttachments.length === 1 ? 'max-w-[18rem] grid-cols-1' : 'max-w-[38rem] grid-cols-2 sm:grid-cols-3'
                          } gap-2`}
                        >
                          {imageAttachments.map((item) => {
                            const src = String(item.url || '').trim()
                            const key = `${item.sha1 || item.path}-${item.name}`
                            const frameClass = 'block overflow-hidden rounded-2xl border border-[var(--border)] bg-white/70'
                            if (src) {
                              return (
                                <a
                                  key={key}
                                  href={src}
                                  target="_blank"
                                  rel="noreferrer"
                                  className={frameClass}
                                >
                                  <img
                                    src={src}
                                    alt={item.name}
                                    className="block h-32 w-full object-cover"
                                    loading="lazy"
                                  />
                                </a>
                              )
                            }
                            return (
                              <div key={key} className={frameClass}>
                                <div className="flex h-32 items-center justify-center px-3 text-center text-xs text-black/45">
                                  {item.name}
                                </div>
                              </div>
                            )
                          })}
                        </div>
                      ) : null}
                      {showUserText ? (
                        <Text className="whitespace-pre-wrap">{message.content}</Text>
                      ) : null}
                    </>
                  ) : (
                    <>
                      {(() => {
                        const noticeText = getMessageNoticeValue(message)
                        return noticeText ? (
                          <div className="mb-4 rounded-2xl border border-[var(--border)] bg-black/[0.03] px-4 py-3 text-sm text-black/70 dark:bg-white/[0.04] dark:text-white/70">
                            {noticeText}
                          </div>
                        ) : null
                      })()}
                      {lowConfidenceMeta ? (
                        <div className="mb-4 rounded-2xl border border-amber-300/70 bg-amber-50/80 px-4 py-3 text-sm text-amber-900 dark:border-amber-300/50 dark:bg-amber-300/10 dark:text-amber-100">
                          <div className="font-medium">
                            {lowConfidenceMeta.isZh ? S.msg_retrieval_low_confidence : 'Lower retrieval confidence'}
                          </div>
                          <div className="mt-1">
                            {lowConfidenceMeta.isZh
                              ? S.msg_retrieval_low_reason.replace('{text}', lowConfidenceMeta.reasonText)
                              : `Reason: ${lowConfidenceMeta.reasonText}.`}
                          </div>
                          {lowConfidenceMeta.candidateRefs.length > 0 ? (
                            <div className="mt-1">
                              {lowConfidenceMeta.isZh
                                ? S.msg_retrieval_candidate_refs.replace('{refs}', lowConfidenceMeta.candidateRefs.map((num) => `[${num}]`).join(', '))
                                : `Candidate refs for cross-check: ${lowConfidenceMeta.candidateRefs.map((num) => `[${num}]`).join(', ')}.`}
                            </div>
                          ) : null}
                        </div>
                      ) : null}
                      {Boolean((globalThis as { __KB_SHOW_PROVENANCE_MODE_LABEL__?: boolean }).__KB_SHOW_PROVENANCE_MODE_LABEL__) && provenanceModeLabel ? (
                        <div className="mb-2">
                          <Text type="secondary" className="text-xs">{provenanceModeLabel}</Text>
                        </div>
                      ) : null}
                      <MarkdownRenderer
                        content={bodyContent}
                        citeDetails={effectiveCiteDetails}
                        onCitationClick={openCitation}
                        onCitationHover={previewCitation}
                        onCitationLeave={scheduleCitationPreviewClose}
                        inlineLocateTokenPolicy={enableLocateUi && guideSourcePath ? { quote: true, figure_ref: true } : undefined}
                        inlineTextLocateEnabled={enableLocateUi ? ((!guideSourcePath || strictStructuredInlineLocate) && !suppressLooseInlineLocate) : false}
                        inlineTextTailLocateEnabled={enableLocateUi ? guideInlineTextTailLocate : false}
                        locateSurfacePolicy={enableLocateUi && guideSourcePath
                          ? {
                            paragraph: guideInlineTextTailLocate,
                            list_item: guideInlineTextTailLocate,
                            quote: true,
                            blockquote: true,
                            equation: true,
                            figure: true,
                          }
                          : undefined}
                        canLocateSnippet={enableLocateUi ? ((snippet, meta) => {
                          if (strictStructuredLocateOnly) {
                            if (!strictStructuredInlineLocate) return false
                            const targetKind = normalizeStructuredLocateKind(String(meta?.kind || ''))
                            if (targetKind === 'paragraph' || targetKind === 'list_item') {
                              const raw = String(snippet || '').trim()
                              if (raw.length < 18) return false
                              const structured = resolveStrictParagraphEntry(snippet, meta)
                              const picked = structured?.entry?.primary || resolveProvenanceLocateCandidates(snippet, 1)[0] || null
                              if (!picked) return false
                              const keyBase = locateCandidateKey(picked)
                              const snippetKey = normalizeLocateText(raw).slice(0, 96)
                              const key = keyBase ? `${keyBase}::${snippetKey}` : snippetKey
                              if (!key) return false
                              if (locateButtonShownKeys.has(key)) return false
                              if (optionalLocateButtonCount >= locateButtonCap) return false
                              locateButtonShownKeys.add(key)
                              optionalLocateButtonCount += 1
                              return true
                            }
                            if (!['quote', 'blockquote', 'equation', 'figure'].includes(targetKind)) {
                              return false
                            }
                            const resolved = resolveExactStructuredInlineResolution(snippet, meta)
                            const entry = resolved?.entry || null
                            if (!entry) return false
                            const order = Number(resolved?.order || 0)
                            if (targetKind !== 'figure' && !allowedStructuredRenderOrders.has(order)) return false
                            if (!isStrictStructuredTargetCompatible(entry, targetKind)) {
                              return false
                            }
                            const claimType = String(entry.claimType || '').trim().toLowerCase()
                            const anchorKind = String(entry.anchorKind || '').trim().toLowerCase()
                            const formulaOrigin = String(entry.formulaOrigin || '').trim().toLowerCase()
                            const locateSurfacePolicy = String(entry.locateSurfacePolicy || '').trim().toLowerCase()
                            if ((anchorKind === 'quote' || claimType === 'quote_claim') && targetKind !== 'quote') {
                              return false
                            }
                            if ((anchorKind === 'blockquote' || claimType === 'blockquote_claim') && targetKind !== 'blockquote') {
                              return false
                            }
                            if ((anchorKind === 'figure' || claimType === 'figure_claim' || claimType === 'figure_panel') && targetKind !== 'figure') {
                              return false
                            }
                            if (targetKind === 'equation') {
                              if (claimType !== 'formula_claim' || anchorKind !== 'equation') {
                                return false
                              }
                              if (formulaOrigin !== 'source' || locateSurfacePolicy !== 'primary') {
                                return false
                              }
                            }
                            if (targetKind === 'figure') {
                              return isPreferredStrictFigureRefSnippet(snippet)
                            }
                            return true
                          }
                          const raw = String(snippet || '').trim()
                          const formulaSnippet = hasFormulaSignal(raw)
                          if (!formulaSnippet && raw.length < 18) return false
                          const directPickedList = resolveProvenanceLocateCandidates(snippet, 1)
                          const directPicked = formulaSnippet
                            ? (directPickedList.find((item) => isEquationLocateCandidate(item)) || directPickedList[0] || null)
                            : (directPickedList[0] || null)
                          const pickedList = directPicked
                            ? directPickedList
                            : resolveLocateCandidates(snippet, 1)
                          const picked = formulaSnippet
                            ? (pickedList.find((item) => isEquationLocateCandidate(item)) || pickedList[0] || null)
                            : (pickedList[0] || null)
                          if (!picked) return false
                          const keyBase = locateCandidateKey(picked)
                          const snippetKey = normalizeLocateText(raw).slice(0, 96)
                          const key = keyBase
                            ? `${keyBase}::${snippetKey}`
                            : snippetKey
                          if (!key) return false
                          if (locateButtonShownKeys.has(key)) return false
                          if (!directPicked && optionalLocateButtonCount >= locateButtonCap) return false
                          locateButtonShownKeys.add(key)
                          if (!directPicked) optionalLocateButtonCount += 1
                          return true
                        }) : undefined}
                        onLocateSnippet={enableLocateUi && onOpenReader
                          ? (snippet, meta) => {
                            if (strictStructuredLocateOnly) {
                              if (!strictStructuredInlineLocate) return
                              const targetKind = normalizeStructuredLocateKind(String(meta?.kind || ''))
                              if (targetKind === 'paragraph' || targetKind === 'list_item') {
                                const structured = resolveStrictParagraphEntry(snippet, meta)
                                const entry = structured?.entry || null
                                if (entry) {
                                  openReaderByStructuredEntry(entry, snippet)
                                  return
                                }
                                const pickedList = resolveProvenanceLocateCandidates(snippet, 6)
                                if (pickedList.length <= 0) return
                                openReaderByCandidates(pickedList, snippet, { strictLocate: true })
                                return
                              }
                              const resolved = resolveExactStructuredInlineResolution(snippet, meta)
                              const entry = resolved?.entry || null
                              if (!entry) return
                              if (targetKind !== 'figure' && !allowedStructuredRenderOrders.has(Number(resolved?.order || 0))) return
                              openReaderByStructuredEntry(entry, snippet)
                              return
                            }
                            const raw = String(snippet || '').trim()
                            const formulaSnippet = hasFormulaSignal(raw)
                            const pickedListRaw = resolveLocateCandidates(snippet, 6)
                            const pickedList = formulaSnippet
                              ? [
                                ...pickedListRaw.filter((item) => isEquationLocateCandidate(item)),
                                ...pickedListRaw.filter((item) => !isEquationLocateCandidate(item)),
                              ]
                              : pickedListRaw
                            if (pickedList.length <= 0) return
                            openReaderByCandidates(pickedList, snippet)
                          }
                          : undefined}
                        locateTitleResolver={enableLocateUi ? ((snippet) => {
                          if (strictStructuredLocateOnly) {
                            const resolved = resolveExactStructuredInlineResolution(snippet)
                              || (isPreferredStrictFigureRefSnippet(snippet)
                                ? resolveExactStructuredInlineResolution(snippet, { kind: 'figure', order: 0 })
                                : null)
                            const entry = resolved?.entry || null
                            if (entry) {
                              const heading = String(entry.primary.headingPath || '').trim()
                              return heading ? `\u5b9a\u4f4d\u5230\u539f\u6587\u8bc1\u636e\uff1a${heading}` : '\u5b9a\u4f4d\u5230\u539f\u6587\u8bc1\u636e'
                            }
                          }
                          const formulaSnippet = hasFormulaSignal(String(snippet || ''))
                          const pickedList = resolveLocateCandidates(snippet, formulaSnippet ? 2 : 1)
                          const picked = formulaSnippet
                            ? (pickedList.find((item) => isEquationLocateCandidate(item)) || pickedList[0] || null)
                            : (pickedList[0] || null)
                          if (!picked) return '\u5b9a\u4f4d\u5230\u539f\u6587'
                          const heading = String(picked.headingPath || '').trim()
                          return heading ? `\u5b9a\u4f4d\u5230\u539f\u6587\uff1a${heading}` : '\u5b9a\u4f4d\u5230\u539f\u6587'
                        }) : undefined}
                        locateButtonAttrsResolver={enableLocateUi ? ((snippet, meta) => {
                          if (!strictStructuredLocateOnly) return null
                          const toAttrs = (candidate: LocateCandidate | null | undefined) => {
                            if (!candidate) return null
                            return {
                              className: 'kb-prov-locate-chip',
                              focus: String(candidate.focusSnippet || candidate.matchText || snippet || '').trim().slice(0, 220),
                              blockId: String(candidate.blockId || '').trim(),
                              anchorId: String(candidate.anchorId || '').trim(),
                              anchorKind: String(candidate.anchorKind || '').trim(),
                              heading: String(candidate.headingPath || '').trim(),
                            }
                          }
                          const targetKind = normalizeStructuredLocateKind(String(meta?.kind || ''))
                          if (targetKind === 'paragraph' || targetKind === 'list_item') {
                            const structured = resolveStrictParagraphEntry(snippet, meta)
                            return toAttrs(structured?.entry?.primary || resolveProvenanceLocateCandidates(snippet, 1)[0] || null)
                          }
                          const resolved = resolveExactStructuredInlineResolution(snippet, meta)
                            || (targetKind === 'figure' || isPreferredStrictFigureRefSnippet(snippet)
                              ? resolveExactStructuredInlineResolution(snippet, { kind: 'figure', order: Number(meta?.order || 0) })
                              : null)
                          const entry = resolved?.entry || null
                          if (!entry) return null
                          if (targetKind !== 'figure' && !allowedStructuredRenderOrders.has(Number(resolved?.order || 0))) return null
                          return toAttrs(entry.primary)
                        }) : undefined}
                      />
                      <ResearchContextReceipt
                        pack={selectedResearchContextPack}
                        onOpenReader={onOpenReader}
                        onFollowUp={onResearchContextFollowUp ? useReceiptItemAsFollowUp : undefined}
                        S={S}
                      />
                      {unlinkedReferenceViews.length > 0 ? (
                        <div className="kb-unlinked-ref-strip" data-testid={`unlinked-reference-candidates-${message.id}`}>
                          <div className="kb-unlinked-ref-head">
                            <span>{S.msg_reference_candidates_title || 'Possible cited papers'}</span>
                            <span>{S.msg_reference_candidates_note || 'Found in this paper bibliography'}</span>
                          </div>
                          <div className="kb-unlinked-ref-list">
                            {unlinkedReferenceViews.map((view) => {
                              const display = citationDisplay(view.detail)
                              const title = display.main || view.detail.title || view.detail.raw || S.default_source_fallback
                              const metaText = [
                                display.authors,
                                display.venueYear || display.venue,
                              ].filter(Boolean).join(' · ')
                              const key = String((view.candidate as Record<string, unknown>).id || view.detail.anchor || title)
                              return (
                                <div className="kb-unlinked-ref-row" key={key}>
                                  <div className="kb-unlinked-ref-main">
                                    <div className="kb-unlinked-ref-title">{title}</div>
                                    {metaText ? <div className="kb-unlinked-ref-meta">{metaText}</div> : null}
                                  </div>
                                  <span className="kb-unlinked-ref-reason">{view.label}</span>
                                  <div className="kb-unlinked-ref-actions">
                                    {onOpenReader && view.detail.sourcePath ? (
                                      <button
                                        type="button"
                                        className="kb-unlinked-ref-action"
                                        onClick={() => openReaderFromDetail(view.detail)}
                                      >
                                        {S.msg_reference_candidate_open || 'Open'}
                                      </button>
                                    ) : null}
                                    <button
                                      type="button"
                                      className="kb-unlinked-ref-action is-primary"
                                      onClick={() => addToShelf(view.detail)}
                                    >
                                      {S.msg_reference_candidate_add || 'Add'}
                                    </button>
                                  </div>
                                </div>
                              )
                            })}
                          </div>
                        </div>
                      ) : null}
                      {showProvenanceLocateChips ? (
                        <div className="mt-3 flex flex-wrap gap-2">
                          {provenanceLocateEntries.map((entry, idx) => {
                            const heading = String(entry.primary?.headingPath || '').trim()
                            const label = String(entry.label || '').trim()
                            const snippet = shortSegmentLabel(
                              String(entry.anchorText || entry.evidenceQuote || entry.segmentText || label || ''),
                              72,
                            )
                            const headingLite = compactHeadingPath(heading, 56)
                            const text = snippet
                              || label
                              || headingLite
                              || '\u539f\u6587\u8bc1\u636e'
                            const seedSnippet = String(
                              entry.evidenceQuote
                              || entry.anchorText
                              || entry.segmentText
                              || entry.label
                              || '',
                            ).trim()
                            const focusSnippet = String(entry.primary?.focusSnippet || entry.primary?.matchText || seedSnippet || '').trim()
                            return (
                              <button
                                key={`${message.id}::prov::${String(entry.segmentId || idx)}::${idx}`}
                                type="button"
                                className="kb-prov-locate-chip"
                                aria-label={'\u5b9a\u4f4d\u5230\u539f\u6587\u8bc1\u636e'}
                                title={heading
                                  ? `\u5b9a\u4f4d\u5230\u539f\u6587\u8bc1\u636e\uff1a${heading}`
                                  : (headingLite ? `\u5b9a\u4f4d\u5230\u539f\u6587\u8bc1\u636e\uff1a${headingLite}` : '\u5b9a\u4f4d\u5230\u539f\u6587\u8bc1\u636e')}
                                data-kb-locate-focus={focusSnippet.slice(0, 220)}
                                data-kb-locate-block-id={String(entry.primary?.blockId || '').trim()}
                                data-kb-locate-anchor-id={String(entry.primary?.anchorId || '').trim()}
                                data-kb-locate-heading={String(entry.primary?.headingPath || '').trim()}
                                onClick={() => openReaderByStructuredEntry(entry, seedSnippet)}
                              >
                                <span className="kb-prov-locate-chip-num">{`\u8bc1\u636e${idx + 1}`}</span>
                                <span className="kb-prov-locate-chip-text">{text}</span>
                              </button>
                            )
                          })}
                        </div>
                      ) : null}
                      <AgentTracePanel
                        trace={agentTrace}
                        messageId={message.id}
                        canLoadTrace={canLoadAgentTrace}
                        onLoadTrace={(messageId) => chatApi.getMessageAgentTrace(messageId, activeConvId)}
                        onOpenReference={openReaderFromDetail}
                        onAddReferenceToShelf={addToShelf}
                      />
                      <ResearchTracePanel trace={researchTrace} />
                      <CopyBar
                        text={getMessageCopyTextValue(message)}
                        markdown={getMessageCopyMarkdownValue(message)}
                      />
                    </>
                  )}
                </div>
                {isUser ? (
                  <div className="kb-msg-avatar kb-msg-avatar-user">
                    <UserOutlined className="text-xs" />
                  </div>
                ) : null}
              </div>
            )
          })}

          {generationPartial !== undefined && generationPartial !== null ? (
            <div className="kb-message-row is-assistant">
              <AssistantAvatar />
              <div className="kb-msg-bubble kb-msg-bubble-assistant is-streaming">
                {generationStage ? (
                  <div className="mb-2 flex items-center gap-2">
                    <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-[var(--accent)]" />
                    <Text type="secondary" className="text-xs">
                      {generationStage}
                    </Text>
                  </div>
                ) : null}
                {cleanGenerationPartial ? (
                  <div className="whitespace-pre-wrap break-words text-sm leading-7 text-[var(--text)]">
                    {cleanGenerationPartial}
                  </div>
                ) : (
                  <div className="flex items-center gap-1 py-1">
                    <span className="typing-dot" />
                    <span className="typing-dot" style={{ animationDelay: '0.15s' }} />
                    <span className="typing-dot" style={{ animationDelay: '0.3s' }} />
                  </div>
                )}
                <AgentTracePanel
                  trace={generationAgentTrace}
                  onOpenReference={openReaderFromDetail}
                  onAddReferenceToShelf={addToShelf}
                />
                <ResearchTracePanel trace={generationTrace} />
              </div>
            </div>
          ) : null}
        </div>
      </div>
      <CitationPopover
        detail={popoverDetail}
        position={popoverPos}
        loading={popoverLoading}
        guideLoading={popoverGuideLoading}
        inShelf={Boolean(popoverDetail && (() => {
          const popoverItem = toShelfItem(popoverDetail)
          const identity = shelfPaperIdentity(popoverItem)
          return shelfItems.some((item) => item.key === popoverItem.key || shelfPaperIdentity(item) === identity)
        })())}
        onClose={closeCitationPopover}
        onAddToShelf={addToShelf}
        onOpenShelf={openCitationShelfFromPopover}
        onOpenReader={openReaderFromDetail}
        onStartGuide={startPaperGuideFromDetail}
        onMouseEnter={keepCitationPreviewOpen}
        onMouseLeave={scheduleCitationPreviewClose}
      />
      {renderedShelfNode}
    </>
  )
}
