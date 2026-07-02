import type { Message } from '../../api/chat'
import type { LocateCandidate } from './reader/messageLocateCandidates'
import type {
  ProvenanceLocateEntry,
  StructuredProvenanceSegment,
} from './reader/messageStructuredProvenance'
import type { StructuredRenderLocateSlot } from './reader/messageStructuredInlineLocate'

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
