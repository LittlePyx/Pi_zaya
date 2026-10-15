import type {
  ResearchNoteEvidenceContext,
  ResearchNoteSourceLink,
} from '../../api/chat'

export function researchNoteSourceLinkWithContext(
  link: ResearchNoteSourceLink,
  context: ResearchNoteEvidenceContext,
): ResearchNoteSourceLink {
  const snapshot = context.context_snapshot
  const target = context.target
  if (
    context.status !== 'exact'
    || !snapshot
    || !target
    || !String(context.source_fingerprint || '').startsWith('sha256:')
    || snapshot.source_fingerprint !== context.source_fingerprint
  ) return link
  if (link.block_id && target.block_id !== link.block_id) return link
  if (link.anchor_id && target.anchor_id !== link.anchor_id) return link
  if (link.block_id && snapshot.target.block_id !== link.block_id) return link
  if (link.anchor_id && snapshot.target.anchor_id !== link.anchor_id) return link
  if (!link.block_id && !link.anchor_id) return link
  return {
    ...link,
    evidence_context_snapshot: snapshot,
    evidence_verification: {
      status: 'exact',
      reason: 'captured_context',
      verified_at: snapshot.captured_at,
      source_fingerprint: context.source_fingerprint,
    },
  }
}
