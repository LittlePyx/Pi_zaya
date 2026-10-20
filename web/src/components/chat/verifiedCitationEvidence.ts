import type { CiteDetail } from './citationState'

export function preserveVerifiedCitationEvidence(current: CiteDetail, enriched: CiteDetail): CiteDetail {
  // Bibliographic metadata is shared by a paper. A verified excerpt and its
  // reader target belong to this particular occurrence of a citation.
  if (current.isInpaper || current.citationRoute !== 'system_a'
    || current.evidenceSource !== 'verified_source_excerpt'
    || !current.sourcePath || !current.blockId || !current.anchorId || !current.headingPath
    || !current.pageStart || !current.pageEnd || !current.evidenceQuote) return enriched
  return {
    ...enriched,
    anchor: current.anchor,
    num: current.num,
    displayNum: current.displayNum,
    displayNums: current.displayNums,
    isInpaper: current.isInpaper,
    citationRoute: current.citationRoute,
    sourcePath: current.sourcePath,
    sourceName: current.sourceName,
    blockId: current.blockId,
    anchorId: current.anchorId,
    anchorKind: current.anchorKind,
    headingPath: current.headingPath,
    pageStart: current.pageStart,
    pageEnd: current.pageEnd,
    evidenceSource: current.evidenceSource,
    evidenceFingerprint: current.evidenceFingerprint,
    evidenceQuote: current.evidenceQuote,
    readerEvidenceQuote: current.readerEvidenceQuote,
    answerClaim: current.answerClaim,
    cardClaim: current.cardClaim,
    cardEvidence: current.cardEvidence,
    cardLocator: current.cardLocator,
    cardView: current.cardView,
  }
}
