import type { Message } from '../../api/chat'
import {
  buildRefsLocateCandidatesAll,
  type LocateCandidate,
  type RefHitLite,
} from './reader/messageLocateCandidates'
import {
  getMessageCiteDetailRecords,
  getMessageRenderedBodyContent,
  getMessageRenderPacket,
} from './messageRenderPacket'
import {
  buildStructuredProvenanceLocateEntries,
  listStructuredProvenanceSegments,
} from './reader/messageStructuredProvenance'
import type {
  ProvenanceLocateEntry,
  StructuredProvenanceSegment,
} from './reader/messageStructuredProvenance'
import {
  buildStructuredRenderLocateSlotMap,
  type StructuredRenderLocateSlot,
} from './reader/messageStructuredInlineLocate'
import { normalizeCiteDetail, type CiteDetail } from './citationState'
import { resolveLowConfidenceMeta, stripLeadingLowConfidenceNotice } from './messageLowConfidence'
import { messageListPerfNow, type MessageListPrepPerfEvent } from './messageListPerf'
import { buildRenderPacketLocateEntry } from './messageRenderPacketLocate'
import { remapStructuredEntryToGuideAnchors } from './messageStructuredLocateRemap'
import {
  lookupGuideCandidatesBySourcePath,
  sourcePathsReferToSameDocument,
} from './messageSourceIdentity'

export interface AssistantLocatePrep {
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

export function createEmptyAssistantLocatePrep(bodyContent: string, refsUserMsgId = 0): AssistantLocatePrep {
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

export interface AssistantTraceLite {
  answerOrder: number
  userMsgId: number
}

interface RefEntryLiteForLocatePrep {
  hits?: RefHitLite[]
  prompt_sig?: string
  updated_at?: number
}

function fastRevisionDigest(value: string): string {
  let hash = 0x811c9dc5
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index)
    hash = Math.imul(hash, 0x01000193)
  }
  return (hash >>> 0).toString(36)
}

function messagePresentationRevision(
  bodyContent: string,
  citeDetails: Array<Record<string, unknown>>,
): string {
  const body = String(bodyContent || '')
  const citations = JSON.stringify(citeDetails)
  return `${body.length}:${fastRevisionDigest(body)}:${citeDetails.length}:${fastRevisionDigest(citations)}`
}

export interface BuildAssistantLocatePrepByMsgIdOptions {
  activeConvId?: string | null
  messages: Message[]
  refs: Record<string, unknown>
  assistantTraceByMsgId: Map<number, AssistantTraceLite>
  guideDocCandidates: LocateCandidate[]
  guideDocCandidatesBySourcePath: Map<string, LocateCandidate[]>
  guideSourcePathSet: Set<string>
  paperGuideSourcePath?: string
  paperGuideSourceName?: string
  onOpenReaderAvailable: boolean
  previousCache: Map<string, AssistantLocatePrep>
  S: Record<string, string>
}

export interface BuildAssistantLocatePrepByMsgIdResult {
  prepByMsgId: Map<number, AssistantLocatePrep>
  nextCache: Map<string, AssistantLocatePrep>
  perf: MessageListPrepPerfEvent
}

export function buildAssistantLocatePrepByMsgId(
  opts: BuildAssistantLocatePrepByMsgIdOptions,
): BuildAssistantLocatePrepByMsgIdResult {
  const nextCache = new Map<string, AssistantLocatePrep>()
  const prepByMsgId = new Map<number, AssistantLocatePrep>()
  const guideSourcePath = String(opts.paperGuideSourcePath || '').trim()
  const guideSourceName = String(opts.paperGuideSourceName || '').trim()
  const prepStartedAt = messageListPerfNow()
  let assistantCount = 0
  let heavyCount = 0
  let lightCount = 0
  let cacheHits = 0

  for (const message of opts.messages) {
    if (message.role !== 'assistant') continue
    assistantCount += 1
    const trace = opts.assistantTraceByMsgId.get(message.id)
    const renderPacket = getMessageRenderPacket(message)
    const locatePayloadSig = messageLocatePayloadSignature(message, renderPacket)
    const rawBodyContent = getMessageRenderedBodyContent(message)
    const lowConfidenceMeta = resolveLowConfidenceMeta(
      (message.meta && typeof message.meta === 'object')
        ? message.meta as Record<string, unknown>
        : null,
      String(rawBodyContent || ''),
      opts.S,
    )
    const bodyContent = lowConfidenceMeta
      ? stripLeadingLowConfidenceNotice(rawBodyContent)
      : rawBodyContent
    const refsUserMsgId = Number(message.refs_user_msg_id || trace?.userMsgId || 0)
    const refEntry = refsUserMsgId > 0
      ? opts.refs[String(refsUserMsgId)] as RefEntryLiteForLocatePrep | undefined
      : undefined
    const refHits = Array.isArray(refEntry?.hits) ? refEntry.hits : []
    const rawCiteDetails = getMessageCiteDetailRecords(message)
    const presentationRevision = messagePresentationRevision(bodyContent, rawCiteDetails)
    const hasRawCiteDetails = rawCiteDetails.length > 0
    const hasProvenancePayload = Boolean(message.provenance && typeof message.provenance === 'object')
    const hasRenderPacketLocate = Boolean(renderPacket?.readerOpen || renderPacket?.locateTarget)
    const shouldBuildLocatePrep = opts.onOpenReaderAvailable && (
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
        presentationRevision,
        'light',
        refsUserMsgId,
      ].join('::')
      const cached = opts.previousCache.get(prepKey)
      if (cached) {
        cacheHits += 1
        nextCache.set(prepKey, cached)
        prepByMsgId.set(message.id, cached)
        continue
      }
      const prep = createEmptyAssistantLocatePrep(bodyContent, refsUserMsgId)
      lightCount += 1
      nextCache.set(prepKey, prep)
      prepByMsgId.set(message.id, prep)
      continue
    }
    const citeDetails = rawCiteDetails
      .map(normalizeCiteDetail)
      .filter((detail): detail is CiteDetail => Boolean(detail))
      .map((detail) => ({
        ...detail,
        traceConvId: String(opts.activeConvId || ''),
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
    const guideDocAvailable = Boolean(guideSourcePath && opts.guideSourcePathSet.has(guideSourcePath))
    const guideCandidateCount = guideSourcePath
      ? lookupGuideCandidatesBySourcePath(opts.guideDocCandidatesBySourcePath, guideSourcePath).length
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
    const refSig = `${refsUserMsgId}:${String(refEntry?.prompt_sig || '')}:${Number(refEntry?.updated_at || 0)}:${refHits.length}`
    const prepKey = [
      message.id,
      String(message.render_cache_key || ''),
      locatePayloadSig,
      presentationRevision,
      guideSourcePath,
      guideCandidateCount,
      locateSourcePath,
      refSig,
    ].join('::')
    const cached = opts.previousCache.get(prepKey)
    if (cached) {
      cacheHits += 1
      nextCache.set(prepKey, cached)
      prepByMsgId.set(message.id, cached)
      continue
    }

    const refsLocateCandidatesAll = buildRefsLocateCandidatesAll(refHits)
    const guideSourceCandidates = guideSourcePath
      ? lookupGuideCandidatesBySourcePath(opts.guideDocCandidatesBySourcePath, guideSourcePath)
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
        const segmentRecord = segment as Record<string, unknown>
        const evidenceMode = String(segmentRecord.evidence_mode || '').trim().toLowerCase()
        const locatePolicy = String(segmentRecord.locate_policy || '').trim().toLowerCase()
        const evidenceIds = Array.isArray(segmentRecord.evidence_block_ids)
          ? segmentRecord.evidence_block_ids
          : []
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
      ? lookupGuideCandidatesBySourcePath(opts.guideDocCandidatesBySourcePath, effectiveGuideSourcePath)
      : []
    const renderPacketLocateEntry = buildRenderPacketLocateEntry(message, renderPacket, {
      fallbackSourcePath: effectiveGuideSourcePath || provenanceSourcePath || locateSourcePath || '',
      fallbackSourceName: locateSourceName || provenanceSourceName || guideSourceName,
    }, opts.S)
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
      if (guideSourcePath) return opts.guideDocCandidates
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
    prepByMsgId.set(message.id, prep)
  }

  return {
    prepByMsgId,
    nextCache,
    perf: {
      ts: Date.now(),
      convId: String(opts.activeConvId || ''),
      messageCount: opts.messages.length,
      assistantCount,
      heavyCount,
      lightCount,
      cacheHits,
      durationMs: Number((messageListPerfNow() - prepStartedAt).toFixed(2)),
    },
  }
}

function countDistinctDocLabels(text: string): number {
  const seen = new Set<string>()
  for (const match of String(text || '').matchAll(/\bDOC-(\d{1,4})\b/gi)) {
    const docId = String(match[1] || '').trim()
    if (docId) seen.add(docId)
  }
  return seen.size
}

export function shouldSuppressLooseInlineLocate(args: {
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

export function messageLocatePayloadSignature(message: Message, renderPacketValue: unknown): string {
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
