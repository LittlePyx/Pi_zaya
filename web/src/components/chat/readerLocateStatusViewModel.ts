export type ReaderLocateBadgeTone = 'neutral' | 'accent' | 'success' | 'warning' | 'danger'

export interface ReaderLocateMetaBadge {
  key: string
  label: string
  title?: string
  tone?: ReaderLocateBadgeTone
  testId?: string
}

export interface ReaderLocateStatusViewModelInput {
  activeAltIndex: number
  activeAnchorKind: string
  activeHeadingPath: string
  activeHitLevel: string
  altChangeSource: string
  hasDistinctAlternatives: boolean
  requestedAltIndex: number
  statusTextFull: string
  strictLocate: boolean
  S: Record<string, string>
}

export interface ReaderLocateStatusViewModel {
  decisionText: string
  decisionTitle?: string
  locateBadges: ReaderLocateMetaBadge[]
}

export function readerLocateBadgesHaveReturnTarget(badges: ReaderLocateMetaBadge[]): boolean {
  return badges.some((badge) => badge.key === 'result' && badge.tone !== 'danger')
}

export function buildReaderLocateResultBadge({
  activeAnchorKind,
  activeHeadingPath,
  activeHitLevel,
  S,
  statusTextFull,
  strictLocate,
}: Pick<
  ReaderLocateStatusViewModelInput,
  'activeAnchorKind' | 'activeHeadingPath' | 'activeHitLevel' | 'S' | 'statusTextFull' | 'strictLocate'
>): ReaderLocateMetaBadge | null {
  const hint = String(statusTextFull || '').trim().toLowerCase()
  const hitLevel = String(activeHitLevel || '').trim().toLowerCase()
  const anchorKind = String(activeAnchorKind || '').trim().toLowerCase()
  const title = statusTextFull || undefined

  if (
    /\b(strict locate stopped|not found)\b/i.test(hint)
    && !/\b(?:neighbor )?evidence block matched\b/i.test(hint)
  ) {
    return {
      key: 'result',
      label: S.reader_locate_unresolved || 'Unresolved',
      title,
      tone: 'danger',
      testId: 'reader-locate-resolution',
    }
  }
  if (hitLevel === 'heading' || /\bheading\b/i.test(hint)) {
    return {
      key: 'result',
      label: S.reader_locate_section_only || 'Section only',
      title,
      tone: strictLocate ? 'warning' : 'neutral',
      testId: 'reader-locate-resolution',
    }
  }
  if (
    anchorKind === 'equation'
    || anchorKind === 'figure'
    || /\b(equation block|figure block|inline formula|neighbor formula|exact figure block)\b/i.test(hint)
  ) {
    return {
      key: 'result',
      label: S.reader_locate_bound_anchor || 'Bound anchor',
      title,
      tone: 'success',
      testId: 'reader-locate-resolution',
    }
  }
  if (/\b(neighbor evidence|fallback|block only)\b/i.test(hint)) {
    return {
      key: 'result',
      label: S.reader_locate_fallback_evidence || 'Fallback evidence',
      title,
      tone: 'warning',
      testId: 'reader-locate-resolution',
    }
  }
  if (hitLevel === 'block' || /\bevidence block matched\b/i.test(hint)) {
    return {
      key: 'result',
      label: S.reader_locate_bound_block || 'Bound block',
      title,
      tone: 'success',
      testId: 'reader-locate-resolution',
    }
  }
  if (hitLevel === 'exact' || /\bexact\b/i.test(hint)) {
    return {
      key: 'result',
      label: S.reader_locate_exact_target || 'Exact target',
      title,
      tone: 'success',
      testId: 'reader-locate-resolution',
    }
  }
  if (activeHeadingPath) {
    return {
      key: 'result',
      label: strictLocate
        ? (S.reader_locate_requested_section || 'Requested section')
        : (S.reader_locate_section_open || 'Section open'),
      title,
      tone: 'neutral',
      testId: 'reader-locate-resolution',
    }
  }
  return null
}

export function buildReaderLocateStatusViewModel({
  activeAltIndex,
  activeAnchorKind,
  activeHeadingPath,
  activeHitLevel,
  altChangeSource,
  hasDistinctAlternatives,
  requestedAltIndex,
  S,
  statusTextFull,
  strictLocate,
}: ReaderLocateStatusViewModelInput): ReaderLocateStatusViewModel {
  const locateBadges: ReaderLocateMetaBadge[] = [{
    key: 'mode',
    label: strictLocate
      ? (S.reader_locate_mode_strict || 'Strict locate')
      : (S.reader_locate_mode_section || 'Section locate'),
    title: strictLocate
      ? (S.reader_locate_mode_strict_title || 'This reader open expects a direct evidence location before softer fallbacks.')
      : (S.reader_locate_mode_section_title || 'This reader open starts from the best matched section or snippet.'),
    tone: strictLocate ? 'accent' : 'neutral',
    testId: 'reader-locate-mode',
  }]

  const resultBadge = buildReaderLocateResultBadge({
    activeAnchorKind,
    activeHeadingPath,
    activeHitLevel,
    S,
    statusTextFull,
    strictLocate,
  })
  if (resultBadge) locateBadges.push(resultBadge)

  let decisionText = ''
  if (hasDistinctAlternatives && altChangeSource === 'system' && activeAltIndex !== requestedAltIndex) {
    locateBadges.push({
      key: 'switch',
      label: S.reader_auto_switched || 'Auto-switched',
      title: S.reader_auto_switched_title || 'The requested candidate could not be bound directly, so the reader moved to a backup candidate.',
      tone: 'warning',
      testId: 'reader-locate-switch',
    })
    decisionText = S.reader_auto_switched_note || 'The requested target missed, so the reader moved to the best backup evidence.'
  } else if (hasDistinctAlternatives && altChangeSource === 'manual' && activeAltIndex !== requestedAltIndex) {
    locateBadges.push({
      key: 'switch',
      label: S.reader_manual_alt || 'Manual alt',
      title: S.reader_manual_alt_title || 'You are viewing a manually selected alternate candidate.',
      tone: 'accent',
      testId: 'reader-locate-switch',
    })
    decisionText = S.reader_manual_alt_note || 'Showing a manually selected alternate candidate.'
  }

  return {
    decisionText,
    decisionTitle: decisionText ? (statusTextFull || decisionText) : undefined,
    locateBadges,
  }
}
