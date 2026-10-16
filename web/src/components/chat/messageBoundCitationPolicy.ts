import type { Message } from '../../api/chat'
import { getMessageRenderPacket } from './messageRenderPacket'
import { getMessageAnswerContract } from './messageTraceUtils'

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown> : {}
}

function source(value: unknown): string {
  return String(value || '').replace(/\\/g, '/').toLowerCase()
}

function quote(value: unknown): string {
  return String(value || '').replace(/\s+/g, ' ').trim()
}

export function hasIndependentlyBoundLocalCitations(message: Message): boolean {
  const meta = record(message.meta)
  const quality = record(meta.answer_quality)
  const packet = getMessageRenderPacket(message)
  // A formatting failure stays a failure. Only the server's atomic, audited
  // cards can retain their own actions; stale legacy details cannot opt in.
  const topClaim = record(meta.claim_evidence ?? meta.claimEvidence)
  const topMinimum = topClaim.minimum_ok ?? topClaim.minimumOk
  if (Object.keys(record(getMessageAnswerContract(message))).length
    || quality.minimum_ok !== false || quality.incomplete_fragment !== true
    || quality.contract_enabled !== false || quality.has_hits !== true
    || record(quality.claim_evidence).minimum_ok !== true
    || (topMinimum !== undefined && topMinimum !== true)
    || ['insufficient_evidence', 'not_found', 'scoped_not_found'].includes(String(quality.resolution_status || '').trim().toLowerCase())
    || !packet?.citeDetails.length) return false
  const plan = record(quality.citation_plan)
  const slots = Array.isArray(plan.slots) ? plan.slots.map(record) : []
  return packet.citeDetails.every((value) => {
    const card = record(value)
    const evidence = quote(card.evidence_quote)
    if (card.is_inpaper !== false || card.citation_route !== 'system_a'
      || card.binding_status !== 'grounded' || card.strict_locate !== true
      || !card.source_path || !card.block_id || !card.anchor_id
      || !Number.isInteger(card.page_start) || Number(card.page_start) <= 0
      || !Number.isInteger(card.num) || Number(card.num) <= 0 || evidence.length < 40) return false
    return slots.some(slot => source(slot.source_path) === source(card.source_path)
      && slot.block_id === card.block_id && slot.anchor_id === card.anchor_id
      && slot.page_start === card.page_start && slot.page_end === card.page_end
      && slot.heading_path === card.heading_path && slot.strict_locate === true
      && quote(slot.evidence_quote).includes(evidence))
  })
}
