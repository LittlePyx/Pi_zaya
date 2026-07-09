import {
  buildReaderLocateCandidateViewModel,
  type ReaderLocateCandidateViewModel,
} from '../components/chat/readerLocateCandidateViewModel'
import type { ReaderLocateCandidate } from '../components/chat/reader/readerTypes'

export interface ReaderLocateCandidateViewModelSmokeResult {
  hiddenActive: ReaderLocateCandidateViewModel
  manualActive: ReaderLocateCandidateViewModel
  single: ReaderLocateCandidateViewModel
}

const labels = {
  reader_alt_index: 'Alt {i}/{n}',
  reader_candidate_alt: 'Alt role',
  reader_candidate_backup: 'Backup',
  reader_candidate_evidence: 'Evidence',
  reader_candidate_manual: 'Manual',
  reader_candidate_primary: 'Primary',
  reader_candidate_requested: 'Requested',
  reader_candidate_resolved: 'Resolved',
  reader_candidates_count: '{n} candidates',
  reader_hide_list: 'Hide list',
}

const requested = {
  anchorId: 'anchor-requested',
  anchorKind: 'paragraph',
  blockId: 'block-requested',
  headingPath: 'Methods',
  highlightSnippet: 'requested quote',
  snippet: 'requested quote',
} as ReaderLocateCandidate

const evidence = {
  anchorId: 'anchor-evidence',
  anchorKind: 'paragraph',
  blockId: 'block-evidence',
  headingPath: 'Results',
  highlightSnippet: 'evidence quote',
  snippet: 'evidence quote',
} as ReaderLocateCandidate

const hidden = {
  anchorId: 'anchor-hidden',
  anchorKind: 'figure',
  anchorNumber: 2,
  blockId: 'block-hidden',
  headingPath: 'Discussion',
  highlightSnippet: 'hidden quote',
  snippet: 'hidden quote',
} as ReaderLocateCandidate

export function runReaderLocateCandidateViewModelSmoke(): ReaderLocateCandidateViewModelSmokeResult {
  const alternatives = [requested, evidence, hidden]
  const hiddenActive = buildReaderLocateCandidateViewModel({
    activeAltIndex: 2,
    altChangeSource: 'system',
    alternatives,
    candidatePickerExpanded: false,
    evidenceAlternatives: [evidence, { ...evidence }],
    initialAltIndex: 0,
    locateHint: 'Neighbor evidence block matched',
    requestedCandidate: requested,
    S: labels,
    strictLocate: true,
    title: 'Reader Paper',
    visibleAlternatives: [requested, evidence],
  })
  const manualActive = buildReaderLocateCandidateViewModel({
    activeAltIndex: 1,
    altChangeSource: 'manual',
    alternatives,
    candidatePickerExpanded: true,
    evidenceAlternatives: [evidence],
    initialAltIndex: 0,
    locateHint: '',
    requestedCandidate: requested,
    S: labels,
    strictLocate: true,
    title: 'Reader Paper',
    visibleAlternatives: alternatives,
  })
  const single = buildReaderLocateCandidateViewModel({
    activeAltIndex: 0,
    altChangeSource: '',
    alternatives: [requested],
    candidatePickerExpanded: false,
    evidenceAlternatives: [],
    initialAltIndex: 0,
    locateHint: 'Exact target matched',
    requestedCandidate: requested,
    S: labels,
    strictLocate: false,
    title: 'Reader Paper',
    visibleAlternatives: [requested],
  })
  return {
    hiddenActive,
    manualActive,
    single,
  }
}
