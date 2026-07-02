import {
  dedupeLocateCandidates,
  hasFormulaSignal,
  stripMarkdownInline,
  stripProvenanceNoise,
  type LocateCandidate,
} from './reader/messageLocateCandidates'
import { toPositiveIntOrUndefined } from './reader/messageReaderLocatePayload'
import type { ProvenanceLocateEntry } from './reader/messageStructuredProvenance'
import {
  extractEquationNumbersFromText,
  extractFigureNumbersFromText,
  normalizeStructuredLocateKind,
  panelLetterMatchScore,
  scoreLocateCandidate,
} from './reader/messageStructuredLocateScoring'
import { sourcePathsReferToSameDocument } from './messageSourceIdentity'

function getStructuredEntryRemapTarget(
  entry: ProvenanceLocateEntry,
  primary: LocateCandidate,
): {
  targetKind: 'equation' | 'figure' | ''
  targetNumber: number
  panelLetters: string[]
  seed: string
} {
  const claimType = String(entry.claimType || '').trim().toLowerCase()
  const targetKind: 'equation' | 'figure' | '' = (() => {
    if (claimType === 'formula_claim' || claimType === 'inline_formula_claim' || claimType === 'equation_explanation_claim') {
      return 'equation'
    }
    if (claimType === 'figure_claim' || claimType === 'figure_panel') {
      return 'figure'
    }
    const rawKind = normalizeStructuredLocateKind(String(entry.anchorKind || primary.anchorKind || ''))
    return rawKind === 'equation' || rawKind === 'figure' ? rawKind : ''
  })()
  const targetNumber = (() => {
    if (targetKind === 'equation') {
      const eqNumbers = extractEquationNumbersFromText(
        `${entry.anchorText || ''} ${entry.evidenceQuote || ''} ${entry.segmentText || ''} ${primary.headingPath || ''}`,
      )
      const merged = [
        Number(entry.equationNumber || 0),
        Number(primary.anchorNumber || 0),
        ...eqNumbers,
      ].filter((item) => Number.isFinite(item) && Number(item) > 0)
      return merged.length > 0 ? Math.floor(Number(merged[0])) : 0
    }
    const figNumbers = extractFigureNumbersFromText(
      `${entry.anchorText || ''} ${entry.evidenceQuote || ''} ${entry.segmentText || ''} ${primary.headingPath || ''}`,
    )
    const merged = [
      Number(entry.supportFigureNumber || 0),
      Number(primary.anchorNumber || 0),
      ...figNumbers,
    ].filter((item) => Number.isFinite(item) && Number(item) > 0)
    return merged.length > 0 ? Math.floor(Number(merged[0])) : 0
  })()
  const panelLetters = Array.isArray(entry.supportPanelLetters)
    ? entry.supportPanelLetters.map((item) => String(item || '').trim().toLowerCase()).filter((item) => /^[a-z]$/.test(item))
    : []
  const seed = stripProvenanceNoise(
    stripMarkdownInline(String(entry.anchorText || entry.evidenceQuote || entry.segmentText || primary.focusSnippet || '')),
  ).trim()
  return {
    targetKind,
    targetNumber,
    panelLetters,
    seed,
  }
}

function getScopedGuideCandidatesForRemap(
  primary: LocateCandidate,
  guideCandidates: LocateCandidate[],
): LocateCandidate[] {
  const sourcePath = String(primary.sourcePath || '').trim()
  return (guideCandidates || []).filter((cand) => {
    if (!cand || typeof cand !== 'object') return false
    if (String(cand.sourceType || '').trim().toLowerCase() !== 'guide') return false
    const candSourcePath = String(cand.sourcePath || '').trim()
    if (sourcePath && candSourcePath && !sourcePathsReferToSameDocument(sourcePath, candSourcePath)) {
      return false
    }
    return Boolean(String(cand.blockId || cand.anchorId || '').trim())
  })
}

function findGuideCandidateIdentityMatch(
  primary: LocateCandidate,
  guideCandidates: LocateCandidate[],
): LocateCandidate | null {
  const primaryBlockId = String(primary.blockId || '').trim()
  const primaryAnchorId = String(primary.anchorId || '').trim()
  if (!(primaryBlockId || primaryAnchorId)) return null
  for (const cand of guideCandidates) {
    if (!cand || typeof cand !== 'object') continue
    const candBlockId = String(cand.blockId || '').trim()
    const candAnchorId = String(cand.anchorId || '').trim()
    if (primaryBlockId && candBlockId && candBlockId === primaryBlockId) return cand
    if (primaryAnchorId && candAnchorId && candAnchorId === primaryAnchorId) return cand
  }
  return null
}

function inferLocateCandidateTargetNumber(
  cand: LocateCandidate,
  targetKind: 'equation' | 'figure',
): number {
  const anchorNumber = Number(cand.anchorNumber || 0)
  if (Number.isFinite(anchorNumber) && anchorNumber > 0) {
    return Math.floor(anchorNumber)
  }
  const raw = `${cand.headingPath || ''} ${cand.focusSnippet || ''} ${cand.matchText || ''}`
  const nums = targetKind === 'equation'
    ? extractEquationNumbersFromText(raw)
    : extractFigureNumbersFromText(raw)
  return nums.length > 0 ? Math.floor(Number(nums[0])) : 0
}

function isGuideCandidateCanonicalForEntry(
  cand: LocateCandidate | null,
  opts: {
    targetKind: 'equation' | 'figure'
    targetNumber: number
  },
): boolean {
  if (!cand) return false
  const targetKind = opts.targetKind
  const targetNumber = opts.targetNumber
  const candKind = normalizeStructuredLocateKind(String(cand.anchorKind || ''))
  if (candKind !== targetKind) return false
  if (targetNumber > 0) {
    const candNumber = inferLocateCandidateTargetNumber(cand, targetKind)
    if (candNumber !== targetNumber) return false
  }
  return true
}

export function remapStructuredEntryToGuideAnchors(
  entry: ProvenanceLocateEntry,
  guideCandidates: LocateCandidate[],
): ProvenanceLocateEntry {
  const primary = entry.primary
  if (!primary) return entry
  const scoped = getScopedGuideCandidatesForRemap(primary, guideCandidates)
  if (scoped.length <= 0) return entry

  const { targetKind, targetNumber, panelLetters, seed } = getStructuredEntryRemapTarget(entry, primary)
  if (!targetKind) return entry
  const primaryIdentityMatch = findGuideCandidateIdentityMatch(primary, scoped)
  if (isGuideCandidateCanonicalForEntry(primaryIdentityMatch, { targetKind, targetNumber })) {
    return entry
  }

  let best: LocateCandidate | null = null
  let bestScore = Number.NEGATIVE_INFINITY
  for (const cand of scoped) {
    const candKind = normalizeStructuredLocateKind(String(cand.anchorKind || ''))
    let score = scoreLocateCandidate(seed || String(primary.focusSnippet || ''), cand)
    if (candKind === targetKind) score += 1.22
    else if (candKind) score -= 1.08
    if (targetNumber > 0) {
      const candNumber = Number.isFinite(Number(cand.anchorNumber || 0))
        ? Math.floor(Number(cand.anchorNumber || 0))
        : 0
      if (candNumber === targetNumber) score += 1.48
      else if (candNumber > 0) score -= 0.46
    }
    if (targetKind === 'figure') {
      if (String(cand.headingPath || '').toLowerCase().includes('figure')) score += 0.22
      if (panelLetters.length > 0) {
        score += 0.28 * panelLetterMatchScore(
          `${cand.headingPath || ''} ${cand.focusSnippet || ''} ${cand.matchText || ''}`,
          panelLetters,
        )
      }
    }
    if (targetKind === 'equation' && hasFormulaSignal(String(cand.focusSnippet || cand.matchText || ''))) {
      score += 0.2
    }
    if (String(cand.blockId || '').trim() === String(primary.blockId || '').trim()) score += 0.08
    if (String(cand.anchorId || '').trim() === String(primary.anchorId || '').trim()) score += 0.06
    if (score > bestScore) {
      best = cand
      bestScore = score
    }
  }

  const acceptFloor = targetNumber > 0 ? 0.48 : 0.7
  if (!best || bestScore < acceptFloor) return entry
  const sameIdentity = (
    String(best.blockId || '').trim() === String(primary.blockId || '').trim()
    && String(best.anchorId || '').trim() === String(primary.anchorId || '').trim()
  )
  if (sameIdentity) return entry

  const relatedBlockIds = Array.from(new Set([
    ...((entry.relatedBlockIds || []).map((item) => String(item || '').trim()).filter(Boolean)),
    ...((String(primary.blockId || '').trim() && String(primary.blockId || '').trim() !== String(best.blockId || '').trim())
      ? [String(primary.blockId || '').trim()]
      : []),
  ]))
  const remappedAnchorKind = String(best.anchorKind || entry.anchorKind || entry.locateTarget?.anchorKind || entry.readerOpen?.anchorKind || '').trim().toLowerCase() || undefined
  const remappedAnchorNumber = toPositiveIntOrUndefined(
    best.anchorNumber
    || entry.equationNumber
    || entry.supportFigureNumber
    || entry.locateTarget?.anchorNumber
    || entry.readerOpen?.anchorNumber
    || 0,
  )
  const remappedLocateTarget = (() => {
    const baseLocateTarget = entry.locateTarget || entry.readerOpen?.locateTarget || null
    if (!baseLocateTarget) return entry.locateTarget
    return {
      ...baseLocateTarget,
      headingPath: String(best.headingPath || baseLocateTarget.headingPath || '').trim() || undefined,
      blockId: String(best.blockId || baseLocateTarget.blockId || '').trim() || undefined,
      anchorId: String(best.anchorId || baseLocateTarget.anchorId || '').trim() || undefined,
      anchorKind: remappedAnchorKind || baseLocateTarget.anchorKind,
      anchorNumber: remappedAnchorNumber ?? baseLocateTarget.anchorNumber,
      relatedBlockIds: relatedBlockIds.length > 0 ? relatedBlockIds : baseLocateTarget.relatedBlockIds,
    }
  })()
  const remappedReaderOpen = (() => {
    if (!entry.readerOpen) return entry.readerOpen
    return {
      ...entry.readerOpen,
      headingPath: String(best.headingPath || entry.readerOpen.headingPath || '').trim() || undefined,
      blockId: String(best.blockId || entry.readerOpen.blockId || '').trim() || undefined,
      anchorId: String(best.anchorId || entry.readerOpen.anchorId || '').trim() || undefined,
      relatedBlockIds: relatedBlockIds.length > 0 ? relatedBlockIds : entry.readerOpen.relatedBlockIds,
      anchorKind: remappedAnchorKind || entry.readerOpen.anchorKind,
      anchorNumber: remappedAnchorNumber ?? entry.readerOpen.anchorNumber,
      locateTarget: remappedLocateTarget || entry.readerOpen.locateTarget,
    }
  })()
  return {
    ...entry,
    primary: best,
    alternatives: dedupeLocateCandidates([best, primary, ...(entry.alternatives || [])]),
    relatedBlockIds: relatedBlockIds.length > 0 ? relatedBlockIds : entry.relatedBlockIds,
    locateTarget: remappedLocateTarget || entry.locateTarget,
    readerOpen: remappedReaderOpen || entry.readerOpen,
  }
}
