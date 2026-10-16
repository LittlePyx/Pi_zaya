import {
  buildReaderLocateResultBadge,
  buildReaderLocateStatusViewModel,
  readerLocateBadgesHaveReturnTarget,
  type ReaderLocateMetaBadge,
  type ReaderLocateStatusViewModel,
} from '../components/chat/readerLocateStatusViewModel'

export interface ReaderLocateStatusViewModelSmokeResult {
  exactCanReturn: boolean
  exactBadge: ReaderLocateMetaBadge | null
  phraseBadge: ReaderLocateMetaBadge | null
  failedPhraseBadge: ReaderLocateMetaBadge | null
  fallbackBlock: ReaderLocateStatusViewModel
  manualSwitch: ReaderLocateStatusViewModel
  quiet: ReaderLocateStatusViewModel
  systemSwitch: ReaderLocateStatusViewModel
  unresolved: ReaderLocateStatusViewModel
  unresolvedCanReturn: boolean
}

const labels = {
  reader_auto_switched: 'Auto switched',
  reader_auto_switched_note: 'Auto note',
  reader_auto_switched_title: 'Auto title',
  reader_locate_bound_anchor: 'Anchor',
  reader_locate_exact_target: 'Exact',
  reader_locate_mode_section: 'Section',
  reader_locate_mode_section_title: 'Section title',
  reader_locate_mode_strict: 'Strict',
  reader_locate_mode_strict_title: 'Strict title',
  reader_locate_unresolved: 'Unresolved',
  reader_manual_alt: 'Manual alt',
  reader_manual_alt_note: 'Manual note',
  reader_manual_alt_title: 'Manual title',
}

export function runReaderLocateStatusViewModelSmoke(): ReaderLocateStatusViewModelSmokeResult {
  const unresolved = buildReaderLocateStatusViewModel({
    activeAltIndex: 0,
    activeAnchorKind: '',
    activeHeadingPath: '',
    activeHitLevel: '',
    altChangeSource: 'system',
    hasDistinctAlternatives: false,
    requestedAltIndex: 0,
    S: labels,
    statusTextFull: 'Strict locate stopped: not found',
    strictLocate: true,
  })
  const systemSwitch = buildReaderLocateStatusViewModel({
    activeAltIndex: 2,
    activeAnchorKind: 'equation',
    activeHeadingPath: 'Methods',
    activeHitLevel: 'block',
    altChangeSource: 'system',
    hasDistinctAlternatives: true,
    requestedAltIndex: 0,
    S: labels,
    statusTextFull: 'Equation block matched',
    strictLocate: true,
  })
  const manualSwitch = buildReaderLocateStatusViewModel({
    activeAltIndex: 1,
    activeAnchorKind: '',
    activeHeadingPath: 'Results',
    activeHitLevel: 'heading',
    altChangeSource: 'manual',
    hasDistinctAlternatives: true,
    requestedAltIndex: 0,
    S: labels,
    statusTextFull: '',
    strictLocate: false,
  })
  const quiet = buildReaderLocateStatusViewModel({
    activeAltIndex: 0,
    activeAnchorKind: '',
    activeHeadingPath: '',
    activeHitLevel: '',
    altChangeSource: '',
    hasDistinctAlternatives: false,
    requestedAltIndex: 0,
    S: labels,
    statusTextFull: '',
    strictLocate: false,
  })
  const exactBadge = buildReaderLocateResultBadge({
    activeAnchorKind: '',
    activeHeadingPath: '',
    activeHitLevel: 'exact',
    S: labels,
    statusTextFull: 'Exact target matched',
    strictLocate: true,
  })
  const fallbackBlock = buildReaderLocateStatusViewModel({
    activeAltIndex: 0,
    activeAnchorKind: 'sentence',
    activeHeadingPath: 'Results / Structured degradation modeling',
    activeHitLevel: 'block',
    altChangeSource: '',
    hasDistinctAlternatives: false,
    requestedAltIndex: 0,
    S: {
      ...labels,
      reader_locate_fallback_evidence: 'Fallback evidence',
    },
    statusTextFull: 'Evidence block matched, but exact inline phrase was not found.',
    strictLocate: true,
  })
  const phraseInput = {
    activeAnchorKind: 'figure', activeHeadingPath: '', activeHitLevel: 'exact',
    S: labels, statusTextFull: 'Exact phrase matched', strictLocate: true,
    locateResult: {
      locateRequestId: 1, sourcePath: 'paper.en.md', status: 'exact' as const,
      precision: 'phrase' as const, ok: true, repairable: false, strictLocate: true,
      hint: 'Exact phrase matched', reason: 'Exact phrase matched',
    },
  }
  const phraseBadge = buildReaderLocateResultBadge(phraseInput)
  const failedPhraseBadge = buildReaderLocateResultBadge({
    ...phraseInput, locateResult: { ...phraseInput.locateResult, status: 'failed', ok: false },
  })
  return {
    exactCanReturn: readerLocateBadgesHaveReturnTarget(exactBadge ? [exactBadge] : []),
    exactBadge,
    phraseBadge,
    failedPhraseBadge,
    fallbackBlock,
    manualSwitch,
    quiet,
    systemSwitch,
    unresolved,
    unresolvedCanReturn: readerLocateBadgesHaveReturnTarget(unresolved.locateBadges),
  }
}
