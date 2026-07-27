import { basenameFromSourcePath } from '../../utils/sourcePath'
import {
  cleanCitationDisplayText,
  normalizeCiteDetail,
  type CiteDetail,
} from './citationState'
import type { MessageRenderPacketLite } from './messageRenderPacket'
import { sourcePathsReferToSameDocument } from './messageSourceIdentity'
import type { RefHitLite } from './reader/messageLocateCandidates'

type UnlinkedReferenceCandidate = MessageRenderPacketLite['unlinkedReferenceCandidates'][number]

export interface UnlinkedReferenceView {
  candidate: UnlinkedReferenceCandidate
  detail: CiteDetail
  label: string
}

interface RefEntryCitationContext {
  hits?: RefHitLite[]
}

function unlinkedReferenceMatchLabel(method: string, S?: Record<string, string>): string {
  const raw = String(method || '').trim().toLowerCase()
  if (raw.includes('doi')) return 'DOI'
  if (raw.includes('title')) return S?.msg_reference_candidate_exact || 'Title match'
  if (raw.includes('venue_year')) return S?.msg_reference_candidate_venue_year || 'Venue/year'
  return S?.msg_reference_candidate_index || 'Reference index'
}

function sameReferenceDetailIdentity(left: CiteDetail, right: CiteDetail): boolean {
  const leftDoi = String(left.doi || left.doiUrl || '').trim().toLowerCase()
  const rightDoi = String(right.doi || right.doiUrl || '').trim().toLowerCase()
  if (leftDoi && rightDoi && leftDoi === rightDoi) return true
  if (
    Number(left.num || 0) > 0
    && Number(right.num || 0) > 0
    && Number(left.num || 0) === Number(right.num || 0)
    && sourcePathsReferToSameDocument(left.sourcePath, right.sourcePath)
  ) {
    return true
  }
  const leftTitle = cleanCitationDisplayText(left.title || left.cardTitle || left.raw || '').toLowerCase()
  const rightTitle = cleanCitationDisplayText(right.title || right.cardTitle || right.raw || '').toLowerCase()
  return Boolean(leftTitle && rightTitle && leftTitle.length >= 18 && leftTitle === rightTitle)
}

export function buildUnlinkedReferenceViews(opts: {
  packet: MessageRenderPacketLite | null
  linkedDetails: CiteDetail[]
  messageId: number
  traceConvId: string
  traceAssistantOrder: number
  traceUserMsgId: number
  S?: Record<string, string>
}): UnlinkedReferenceView[] {
  const candidates = Array.isArray(opts.packet?.unlinkedReferenceCandidates)
    ? opts.packet?.unlinkedReferenceCandidates || []
    : []
  if (candidates.length <= 0) return []
  const out: UnlinkedReferenceView[] = []
  for (const candidate of candidates) {
    if (!candidate || typeof candidate !== 'object') continue
    const candidateRec = candidate as Record<string, unknown>
    const detailSeed = (
      candidateRec.cite_detail && typeof candidateRec.cite_detail === 'object'
        ? candidateRec.cite_detail as Record<string, unknown>
        : candidateRec
    )
    const seedIsInpaper = typeof detailSeed.is_inpaper === 'boolean'
      ? detailSeed.is_inpaper
      : typeof detailSeed.isInpaper === 'boolean'
        ? detailSeed.isInpaper
        : true
    const seedNum = seedIsInpaper
      ? Number(detailSeed.num || candidateRec.ref_num || 0)
      : Number(detailSeed.num ?? 0)
    const detail = normalizeCiteDetail({
      ...candidateRec,
      ...detailSeed,
      anchor: String(detailSeed.anchor || candidateRec.id || `kb-unlinked-ref-${opts.messageId}-${out.length + 1}`),
      num: seedNum,
      source_name: String(detailSeed.source_name || candidateRec.source_name || ''),
      source_path: String(detailSeed.source_path || candidateRec.source_path || ''),
      citation_route: String(detailSeed.citation_route || 'system_b'),
      is_inpaper: seedIsInpaper,
      binding_status: String(detailSeed.binding_status || 'candidate'),
      binding_confidence: Number(detailSeed.binding_confidence || candidateRec.confidence || 0),
    })
    if (!detail) continue
    const tracedDetail: CiteDetail = {
      ...detail,
      traceConvId: opts.traceConvId,
      traceAssistantMsgId: opts.messageId,
      traceAssistantOrder: opts.traceAssistantOrder,
      traceUserMsgId: opts.traceUserMsgId,
    }
    if (opts.linkedDetails.some((item) => sameReferenceDetailIdentity(item, tracedDetail))) continue
    if (out.some((item) => sameReferenceDetailIdentity(item.detail, tracedDetail))) continue
    out.push({
      candidate,
      detail: tracedDetail,
      label: !tracedDetail.isInpaper && tracedDetail.libraryMatchStatus === 'in_library'
        ? opts.S?.msg_reference_candidate_library || 'In library'
        : unlinkedReferenceMatchLabel(String(candidateRec.match_method || ''), opts.S),
    })
  }
  return out.slice(0, 5)
}

function refDisplaySourceKey(value: string): string {
  return String(value || '').trim().replace(/\\/g, '/').toLowerCase()
}

function refDisplayNameKey(value: string): string {
  return cleanCitationDisplayText(String(value || ''))
    .replace(/\.(?:en\.)?md$/i, '')
    .replace(/\.pdf$/i, '')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase()
}

function refDisplayDoiKey(value: string): string {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/^doi:\s*/i, '')
    .replace(/^https?:\/\/(?:dx\.)?doi\.org\//i, '')
    .replace(/[?#].*$/, '')
    .replace(/[\s.,;]+$/, '')
}

function uiTextMostlySame(left: string, right: string): boolean {
  const a = cleanCitationDisplayText(left).replace(/\s+/g, ' ').toLowerCase()
  const b = cleanCitationDisplayText(right).replace(/\s+/g, ' ').toLowerCase()
  if (!a || !b) return false
  if (a === b) return true
  if (a.length >= 36 && b.includes(a)) return true
  if (b.length >= 36 && a.includes(b)) return true
  const at = new Set(a.match(/[a-z0-9\u4e00-\u9fff]{2,}/g) || [])
  const bt = new Set(b.match(/[a-z0-9\u4e00-\u9fff]{2,}/g) || [])
  if (at.size < 5 || bt.size < 5) return false
  let overlap = 0
  for (const token of at) {
    if (bt.has(token)) overlap += 1
  }
  return overlap / Math.min(at.size, bt.size) >= 0.82
}

export function enrichCiteDetailsWithVisibleRefContext(
  details: CiteDetail[],
  refEntry: RefEntryCitationContext | undefined,
): CiteDetail[] {
  const hits = Array.isArray(refEntry?.hits) ? refEntry.hits : []
  if (details.length <= 0 || hits.length <= 0) return details
  const bySourcePath = new Map<string, { displayNum: number; summaryLine: string }>()
  const bySourceName = new Map<string, { displayNum: number; summaryLine: string }>()
  const byDoi = new Map<string, { displayNum: number; summaryLine: string }>()
  for (const [index, hit] of hits.entries()) {
    const ui = hit?.ui_meta || {}
    const meta = hit?.meta || {}
    const displayNum = index + 1
    const summaryLine = cleanCitationDisplayText(String(ui.summary_line || ''))
    const sourcePath = String(ui.source_path || meta.source_path || '').trim()
    const sourceName = String(ui.display_name || basenameFromSourcePath(sourcePath) || '').trim()
    const citationMeta = ui.citation_meta || {}
    const row = { displayNum, summaryLine }
    const pathKey = refDisplaySourceKey(sourcePath)
    if (pathKey && !bySourcePath.has(pathKey)) bySourcePath.set(pathKey, row)
    const nameKey = refDisplayNameKey(sourceName)
    if (nameKey && !bySourceName.has(nameKey)) bySourceName.set(nameKey, row)
    const doiKey = refDisplayDoiKey(String(citationMeta.doi || citationMeta.doi_url || ''))
    if (doiKey && !byDoi.has(doiKey)) byDoi.set(doiKey, row)
  }
  if (bySourcePath.size <= 0 && bySourceName.size <= 0 && byDoi.size <= 0) return details

  return details.map((detail) => {
    if (detail.isInpaper) return detail
    const row = byDoi.get(refDisplayDoiKey(detail.doi || detail.doiUrl))
      || bySourcePath.get(refDisplaySourceKey(detail.sourcePath))
      || bySourceName.get(refDisplayNameKey(detail.sourceName))
    if (!row) return detail
    const next: CiteDetail = {
      ...detail,
      displayNum: row.displayNum,
      displayNums: [row.displayNum],
    }
    const currentTakeaway = cleanCitationDisplayText(detail.cardTakeaway)
    const answerClaim = cleanCitationDisplayText(detail.cardClaim || detail.answerClaim)
    const evidenceText = cleanCitationDisplayText(detail.cardEvidence || detail.evidenceQuote || detail.summaryLine)
    const summary = row.summaryLine
    const hasOccurrenceSpecificClaim = (detail.cardQualityFlags || []).includes('occurrence_specific_claim')
    const shouldReplaceTakeaway = Boolean(
      summary
      && !hasOccurrenceSpecificClaim
      && (
        !currentTakeaway
        || uiTextMostlySame(currentTakeaway, answerClaim)
        || uiTextMostlySame(currentTakeaway, evidenceText)
      )
      && !uiTextMostlySame(summary, answerClaim)
      && !uiTextMostlySame(summary, evidenceText)
    )
    if (shouldReplaceTakeaway) {
      next.cardTakeaway = summary
      next.cardTakeawayLabel = detail.cardTakeawayLabel || '证据重点'
    }
    return next
  })
}
