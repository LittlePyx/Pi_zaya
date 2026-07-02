import type { Message } from '../../api/chat'

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
