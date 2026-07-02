import type { Message } from '../../api/chat'
import { basenameFromSourcePath } from '../../utils/sourcePath'
import type { MessageRenderPacketLite } from './messageRenderPacket'
import {
  coerceStringArray,
  dedupeLocateCandidates,
  normalizeLocateText,
  stripMarkdownInline,
  type LocateCandidate,
} from './reader/messageLocateCandidates'
import {
  buildLocateCandidateFromReaderLocateCandidate,
  coerceReaderLocateTarget,
  coerceReaderOpenPayload,
  toPositiveIntOrUndefined,
} from './reader/messageReaderLocatePayload'
import {
  mergeStructuredSnippetAliases,
  normalizeStructuredLocateSnippet,
  shortSegmentLabel,
  shouldSuppressNegativeLocateSurface,
  type ProvenanceLocateEntry,
} from './reader/messageStructuredProvenance'

export function buildRenderPacketLocateEntry(
  message: Message,
  packet: MessageRenderPacketLite | null,
  opts: {
    fallbackSourcePath?: string
    fallbackSourceName?: string
  },
  S: Record<string, string>,
): ProvenanceLocateEntry | null {
  if (!packet) return null
  const readerOpen = coerceReaderOpenPayload(packet.readerOpen)
  const locateTarget = coerceReaderLocateTarget(packet.locateTarget) || readerOpen?.locateTarget || null
  const locatePolicyNorm = String(locateTarget?.locatePolicy || '').trim().toLowerCase()
  const locateSurfacePolicyNorm = String(locateTarget?.locateSurfacePolicy || '').trim().toLowerCase()
  if (locatePolicyNorm === 'hidden' || locateSurfacePolicyNorm === 'hidden') {
    return null
  }
  const sourcePath = String(readerOpen?.sourcePath || opts.fallbackSourcePath || '').trim()
  if (!sourcePath) return null
  const sourceName = String(
    readerOpen?.sourceName
    || opts.fallbackSourceName
    || basenameFromSourcePath(sourcePath)
    || 'paper',
  ).trim()
  const snippet = String(
    locateTarget?.snippet
    || readerOpen?.snippet
    || packet.renderedBody
    || packet.answerMarkdown
    || message.content
    || '',
  ).trim()
  const highlightSnippet = String(
    locateTarget?.highlightSnippet
    || readerOpen?.highlightSnippet
    || locateTarget?.evidenceQuote
    || snippet,
  ).trim()
  const primary = buildLocateCandidateFromReaderLocateCandidate(
    {
      headingPath: locateTarget?.headingPath || readerOpen?.headingPath,
      snippet,
      highlightSnippet,
      blockId: locateTarget?.blockId || readerOpen?.blockId,
      anchorId: locateTarget?.anchorId || readerOpen?.anchorId,
      anchorKind: locateTarget?.anchorKind || readerOpen?.anchorKind,
      anchorNumber: locateTarget?.anchorNumber || readerOpen?.anchorNumber,
    },
    {
      sourcePath,
      sourceName,
      sourceType: 'guide',
    },
  )
  if (!primary) return null
  const alternatives = dedupeLocateCandidates(
    [
      ...(readerOpen?.alternatives || []),
      ...(readerOpen?.visibleAlternatives || []),
      ...(readerOpen?.evidenceAlternatives || []),
    ]
      .map((candidate) => buildLocateCandidateFromReaderLocateCandidate(candidate, {
        sourcePath,
        sourceName,
        sourceType: 'guide',
      }))
      .filter((candidate): candidate is LocateCandidate => Boolean(candidate)),
  )
  const claimGroup = readerOpen?.claimGroup || null
  const anchorKind = String(locateTarget?.anchorKind || readerOpen?.anchorKind || '').trim().toLowerCase()
  const anchorNumber = toPositiveIntOrUndefined(
    locateTarget?.anchorNumber
    || readerOpen?.anchorNumber
    || 0,
  )
  const evidenceQuote = String(locateTarget?.evidenceQuote || highlightSnippet || '').trim()
  const segmentText = stripMarkdownInline(String(packet.renderedBody || packet.answerMarkdown || snippet || '')).trim()
  if (shouldSuppressNegativeLocateSurface({
    claimType: String(locateTarget?.claimType || '').trim(),
    anchorKind: String(locateTarget?.anchorKind || readerOpen?.anchorKind || '').trim(),
    segmentText,
    evidenceQuote,
    anchorText: String(locateTarget?.anchorText || '').trim(),
    snippet,
    highlightSnippet,
  })) {
    return null
  }
  const snippetAliases = mergeStructuredSnippetAliases(
    coerceStringArray(locateTarget?.snippetAliases, 8, 360),
    [snippet, highlightSnippet, String(locateTarget?.anchorText || '').trim()],
  )
  const snippetKey = normalizeStructuredLocateSnippet(
    snippet
    || packet.renderedBody
    || packet.answerMarkdown,
  ) || normalizeLocateText(segmentText || evidenceQuote || primary.focusSnippet).slice(0, 360)
  const relatedBlockIds = Array.isArray(locateTarget?.relatedBlockIds)
    ? locateTarget.relatedBlockIds.map((item) => String(item || '').trim()).filter(Boolean)
    : (Array.isArray(readerOpen?.relatedBlockIds)
      ? readerOpen.relatedBlockIds.map((item) => String(item || '').trim()).filter(Boolean)
      : undefined)
  return {
    segmentId: String(locateTarget?.segmentId || locateTarget?.sourceSegmentId || `render-packet-${message.id}`).trim(),
    label: shortSegmentLabel(highlightSnippet || snippet || packet.renderedBody || S.msg_evidence_label),
    segmentText,
    evidenceQuote,
    locateTarget: locateTarget || undefined,
    readerOpen: readerOpen || undefined,
    hitLevel: String(locateTarget?.hitLevel || '').trim(),
    claimType: String(locateTarget?.claimType || '').trim(),
    mustLocate: locatePolicyNorm === 'required' || Boolean(readerOpen?.strictLocate),
    locatePolicy: String(locateTarget?.locatePolicy || (readerOpen?.strictLocate ? 'required' : '')).trim(),
    locateSurfacePolicy: String(locateTarget?.locateSurfacePolicy || '').trim(),
    claimGroupId: String(claimGroup?.id || '').trim(),
    claimGroupKind: String(claimGroup?.kind || '').trim(),
    formulaOrigin: '',
    anchorKind,
    anchorText: String(locateTarget?.anchorText || '').trim(),
    equationNumber: anchorKind === 'equation' ? (anchorNumber || 0) : 0,
    supportFigureNumber: anchorKind === 'figure' ? (anchorNumber || 0) : 0,
    supportPanelLetters: [],
    snippetKey,
    snippetAliases,
    primary,
    alternatives,
    relatedBlockIds,
    sourceSegmentId: String(locateTarget?.sourceSegmentId || locateTarget?.segmentId || '').trim(),
    groupLeadText: String(claimGroup?.leadText || '').trim(),
    groupDistance: toPositiveIntOrUndefined(claimGroup?.distance || 0),
  }
}
