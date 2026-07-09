import {
  buildReaderActiveLocateCandidate,
  buildReaderOpenPayloadViewModel,
  type ReaderActiveLocateCandidateViewModel,
  type ReaderNormalizedLocateCandidate,
} from '../components/chat/readerOpenPayloadViewModel'

export interface ReaderOpenPayloadViewModelSmokeResult {
  activeFigure: ReaderActiveLocateCandidateViewModel
  fallbackPrimary: ReaderActiveLocateCandidateViewModel
  empty: {
    alternativesCount: number
    locateRequestId: number
    sourcePath: string
    strictLocate: boolean
  }
  rich: {
    activeHitLevel: string
    alternatives: ReaderNormalizedLocateCandidate[]
    evidenceCount: number
    hasStructuredLocateTarget: boolean
    initialAltIndex?: number
    locateFeedbackKey: string
    locateRequestId: number
    primaryCandidate: ReaderNormalizedLocateCandidate
    relatedBlockIds: string[]
    sourceName: string
    sourcePath: string
    strictLocate: boolean
    visibleCount: number
  }
}

export function runReaderOpenPayloadViewModelSmoke(): ReaderOpenPayloadViewModelSmokeResult {
  const richViewModel = buildReaderOpenPayloadViewModel({
    sourcePath: ' /tmp/paper.md ',
    sourceName: ' Reader Paper ',
    headingPath: 'Payload Heading',
    snippet: 'payload snippet',
    highlightSnippet: '',
    anchorId: 'payload-anchor',
    blockId: 'payload-block',
    anchorKind: 'figure',
    anchorNumber: 2,
    relatedBlockIds: ['payload-related'],
    locateTarget: {
      headingPath: ' Target Heading ',
      snippet: ' target snippet ',
      highlightSnippet: ' target quote ',
      anchorId: 'target-anchor',
      blockId: 'target-block',
      anchorKind: 'Equation',
      anchorNumber: 4.8,
      hitLevel: 'Exact',
      relatedBlockIds: [' rel-a ', ''],
    },
    locateRequestId: 9.7,
    locateFeedbackKey: ' feedback-key ',
    visibleAlternatives: [
      {
        headingPath: 'Target Heading',
        snippet: 'target snippet',
        highlightSnippet: 'target quote',
        anchorId: 'target-anchor',
        blockId: 'target-block',
        anchorKind: 'equation',
        anchorNumber: 4,
      },
      {
        headingPath: 'Visible Section',
        snippet: 'visible quote',
      },
    ],
    evidenceAlternatives: [
      {
        blockId: 'evidence-block',
        snippet: 'evidence quote',
      },
    ],
    alternatives: [
      {
        anchorKind: 'Figure',
        anchorNumber: 3,
        headingPath: 'Figure Section',
      },
    ],
    initialAltIndex: 2,
  })
  const emptyViewModel = buildReaderOpenPayloadViewModel(null)

  return {
    activeFigure: buildReaderActiveLocateCandidate({
      activeAltIndex: 3,
      alternatives: richViewModel.alternatives,
      primaryCandidate: richViewModel.primaryCandidate,
    }),
    fallbackPrimary: buildReaderActiveLocateCandidate({
      activeAltIndex: 99,
      alternatives: richViewModel.alternatives,
      primaryCandidate: richViewModel.primaryCandidate,
    }),
    empty: {
      alternativesCount: emptyViewModel.alternatives.length,
      locateRequestId: emptyViewModel.locateRequestId,
      sourcePath: emptyViewModel.sourcePath,
      strictLocate: emptyViewModel.strictLocate,
    },
    rich: {
      activeHitLevel: richViewModel.activeHitLevel,
      alternatives: richViewModel.alternatives,
      evidenceCount: richViewModel.evidenceAlternatives?.length || 0,
      hasStructuredLocateTarget: richViewModel.hasStructuredLocateTarget,
      initialAltIndex: richViewModel.initialAltIndex,
      locateFeedbackKey: richViewModel.locateFeedbackKey,
      locateRequestId: richViewModel.locateRequestId,
      primaryCandidate: richViewModel.primaryCandidate,
      relatedBlockIds: richViewModel.relatedBlockIds,
      sourceName: richViewModel.sourceName,
      sourcePath: richViewModel.sourcePath,
      strictLocate: richViewModel.strictLocate,
      visibleCount: richViewModel.visibleAlternatives?.length || 0,
    },
  }
}
