import {
  hasFormulaSignal,
  normalizeLocateText,
  stripMarkdownInline,
  stripProvenanceNoise,
  type LocateCandidate,
} from './reader/messageLocateCandidates'
import { isEquationLocateCandidate } from './reader/messageStructuredInlineLocate'
import { isLikelyRhetoricalLocateShell } from './reader/messageStructuredProvenance'
import {
  extractEquationNumbersFromText,
  scoreLocateCandidate,
  scoreProvenanceSegment,
} from './reader/messageStructuredLocateScoring'
import { extractQuotedSpans, quoteMatchStats } from './messageQuoteUtils'

export interface CreateMessageLocateResolversOptions {
  locateCandidates: LocateCandidate[]
  provenanceSourcePath: string
  provenanceSourceName: string
  locateSourceName: string
  provenanceDirectSegments: Array<Record<string, unknown>>
  provenanceBlockMap: Record<string, Record<string, unknown>>
  strictProvenanceLocate: boolean
  hasStructuredProvenance: boolean
  provenanceStrictIdentityReady: boolean
  hasStrictMustLocateEntries: boolean
  hasDirectProvenance: boolean
}

export interface MessageLocateResolvers {
  resolveProvenanceLocateCandidates: (snippet: string, limit?: number) => LocateCandidate[]
  resolveLocateCandidates: (snippet: string, limit?: number) => LocateCandidate[]
  locateCandidateKey: (cand: LocateCandidate | null | undefined) => string
}

export function createMessageLocateResolvers(opts: CreateMessageLocateResolversOptions): MessageLocateResolvers {
  const {
    locateCandidates,
    provenanceSourcePath,
    provenanceSourceName,
    locateSourceName,
    provenanceDirectSegments,
    provenanceBlockMap,
    strictProvenanceLocate,
    hasStructuredProvenance,
    provenanceStrictIdentityReady,
    hasStrictMustLocateEntries,
    hasDirectProvenance,
  } = opts
  const resolveCache = new Map<string, LocateCandidate[]>()
  const usedCount = new Map<string, number>()

  const resolveProvenanceLocateCandidates = (snippet: string, limit = 4): LocateCandidate[] => {
    const raw = stripProvenanceNoise(stripMarkdownInline(String(snippet || '')))
    const key = normalizeLocateText(raw).slice(0, 360)
    if (!key || !provenanceSourcePath) return []
    const formulaQuery = hasFormulaSignal(raw)
    const quoteSpans = formulaQuery ? [] : extractQuotedSpans(raw, 12)
    const rankedSegments: Array<{ segment: Record<string, unknown>; score: number }> = []
    for (const segment of provenanceDirectSegments) {
      const segmentText = String(segment.text || '')
      if (isLikelyRhetoricalLocateShell(segmentText)) continue
      const segmentKey = String(segment.snippet_key || '')
      const segmentConf = Number(segment.evidence_confidence || 0)
      const confFloor = formulaQuery ? 0.5 : 0.62
      if (segmentConf > 0 && segmentConf < confFloor) continue
      let score = scoreProvenanceSegment(raw, segmentText, segmentKey)
      if (quoteSpans.length > 0) {
        const qSeg = quoteMatchStats(quoteSpans, segmentText, segmentKey)
        if (qSeg.hits <= 0 && qSeg.score < 0.55) continue
        score += 0.38 * qSeg.score + (qSeg.hits > 0 ? 0.22 : 0)
      }
      if (score > 0) rankedSegments.push({ segment, score })
    }
    rankedSegments.sort((a, b) => b.score - a.score)
    const scoreFloor = formulaQuery ? 0.45 : 0.5
    let matchedSegments = rankedSegments
      .filter((row) => row.score >= scoreFloor)
      .slice(0, 1)
    if (formulaQuery && matchedSegments.length <= 0) {
      matchedSegments = rankedSegments
        .filter((row) => row.score >= 0.42)
        .slice(0, 1)
    }
    const out: LocateCandidate[] = []
    const seen = new Set<string>()
    for (const row of matchedSegments) {
      const segment = row.segment
      const evidenceIds = Array.isArray(segment.evidence_block_ids) ? segment.evidence_block_ids : []
      for (const blockIdRaw of evidenceIds.slice(0, formulaQuery ? 2 : 1)) {
        const blockId = String(blockIdRaw || '').trim()
        if (!blockId) continue
        const block = provenanceBlockMap[blockId]
        if (!block) continue
        const blockKind = String(block.kind || '').trim().toLowerCase()
        if (formulaQuery && blockKind && blockKind !== 'equation') continue
        if (!formulaQuery && blockKind === 'equation' && evidenceIds.length > 1) continue
        const blockText = String(block.text || '').trim()
        if (quoteSpans.length > 0) {
          const qBlock = quoteMatchStats(quoteSpans, blockText, String(segment.text || ''), String(block.heading_path || ''))
          if (qBlock.hits <= 0 && qBlock.score < 0.85) continue
        }
        const key0 = `${provenanceSourcePath}::${blockId}`
        if (seen.has(key0)) continue
        seen.add(key0)
        const segmentFocus = String(segment.text || '').trim()
        const blockFocus = blockText
        const focusSnippet = (formulaQuery ? (blockFocus || segmentFocus) : (segmentFocus || blockFocus))
        if (!focusSnippet) continue
        out.push({
          sourcePath: provenanceSourcePath,
          sourceName: provenanceSourceName || locateSourceName || provenanceSourcePath.split(/[\\/]/).pop() || 'paper',
          headingPath: String(block.heading_path || '').trim(),
          focusSnippet,
          matchText: [String(block.heading_path || '').trim(), String(block.text || segment.text || '').trim()].filter(Boolean).join('\n'),
          sourceType: 'guide',
          blockId,
          anchorId: String(block.anchor_id || '').trim() || undefined,
          anchorKind: String(block.kind || '').trim().toLowerCase() || undefined,
          anchorNumber: Number(block.number || 0) > 0 ? Math.floor(Number(block.number || 0)) : undefined,
        })
        if (out.length >= Math.max(1, limit)) return out
      }
    }
    return out
  }

  const resolveLocateCandidates = (snippet: string, limit = 4): LocateCandidate[] => {
    const key = String(snippet || '').trim()
    if (!key) return []
    if (resolveCache.has(key)) return (resolveCache.get(key) || []).slice(0, Math.max(1, limit))
    const formulaQuery = hasFormulaSignal(key)
    const guideOnly = locateCandidates.filter((item) => item.sourceType === 'guide')
    const strictDirectMode = hasDirectProvenance && !formulaQuery && guideOnly.length > 0
    const provenancePicked = resolveProvenanceLocateCandidates(key, limit)
    if (provenancePicked.length > 0) {
      const picked = formulaQuery
        ? (() => {
          const eqProv = provenancePicked.filter((cand) => isEquationLocateCandidate(cand))
          return eqProv.length > 0 ? eqProv : provenancePicked
        })()
        : provenancePicked
      resolveCache.set(key, picked)
      return picked.slice(0, Math.max(1, limit))
    }
    if (strictProvenanceLocate && hasStructuredProvenance && provenanceStrictIdentityReady && hasStrictMustLocateEntries) {
      resolveCache.set(key, [])
      return []
    }
    const quoteSpans = formulaQuery ? [] : extractQuotedSpans(key, 12)
    if (!formulaQuery && quoteSpans.length > 0) {
      const quotePool = guideOnly.length > 0 ? guideOnly : locateCandidates
      const quoteRank = quotePool
        .map((cand) => {
          const q = quoteMatchStats(quoteSpans, cand.matchText, cand.focusSnippet, cand.headingPath)
          let score = q.score + (0.35 * scoreLocateCandidate(key, cand))
          if (q.hits > 0) score += 0.35
          if (cand.sourceType === 'guide') score += 0.08
          if (cand.anchorId || cand.blockId) score += 0.1
          return { cand, score, hits: q.hits }
        })
        .sort((a, b) => b.score - a.score)
      const bestQuote = quoteRank[0]
      if (bestQuote && bestQuote.hits > 0 && bestQuote.score >= 1.05) {
        resolveCache.set(key, [bestQuote.cand])
        return [bestQuote.cand]
      }
    }

    const rankIn = (cands: LocateCandidate[]) => {
      const scored: Array<{ cand: LocateCandidate; score: number }> = []
      for (const cand of cands) {
        const base = scoreLocateCandidate(key, cand)
        const candKey = `${cand.sourcePath}::${cand.anchorId || ''}::${cand.headingPath}::${cand.focusSnippet.slice(0, 96)}`
        const penalty = 0.03 * Number(usedCount.get(candKey) || 0)
        const score = base - penalty
        scored.push({ cand, score })
      }
      scored.sort((a, b) => b.score - a.score)
      return scored
    }

    const picked: LocateCandidate[] = []
    const pickedKeySet = new Set<string>()
    const pickedHeadingSet = new Set<string>()
    const addPicked = (cand: LocateCandidate, preferNewHeading = false) => {
      const candKey = `${cand.sourcePath}::${cand.anchorId || ''}::${cand.headingPath}::${cand.focusSnippet.slice(0, 96)}`
      if (pickedKeySet.has(candKey)) return false
      const headingRaw = String(cand.headingPath || '').trim()
      const headingKey = headingRaw
        ? normalizeLocateText(headingRaw)
        : normalizeLocateText(String(cand.focusSnippet || '').slice(0, 56))
      if (preferNewHeading && headingKey && pickedHeadingSet.has(headingKey)) return false
      picked.push(cand)
      pickedKeySet.add(candKey)
      if (headingKey) pickedHeadingSet.add(headingKey)
      return true
    }
    const ingestRank = (
      rankRows: Array<{ cand: LocateCandidate; score: number }>,
      floor: number,
      preferNewHeading: boolean,
    ) => {
      for (const row of rankRows) {
        if (row.score < floor) break
        addPicked(row.cand, preferNewHeading)
        if (picked.length >= limit) break
      }
    }

    if (hasDirectProvenance && formulaQuery) {
      const eqNums = extractEquationNumbersFromText(key)
      const eqGuide = guideOnly.filter((cand) => isEquationLocateCandidate(cand))
      if (eqGuide.length > 0) {
        let bestEq: LocateCandidate | null = null
        let bestEqScore = -1
        for (const cand of eqGuide) {
          let s = scoreLocateCandidate(key, cand)
          if (eqNums.length > 0 && Number(cand.anchorNumber || 0) > 0 && eqNums.includes(Math.floor(Number(cand.anchorNumber || 0)))) {
            s += 0.45
          }
          if (cand.anchorId) s += 0.2
          if (s > bestEqScore) {
            bestEq = cand
            bestEqScore = s
          }
        }
        if (bestEq && bestEqScore >= 0.58) {
          resolveCache.set(key, [bestEq])
          return [bestEq]
        }
      }
    }
    if (guideOnly.length > 0) {
      const guideRank = rankIn(guideOnly)
      const guideFloor = strictDirectMode
        ? 0.34
        : (hasFormulaSignal(key) ? 0.32 : 0.2)
      ingestRank(guideRank, guideFloor, true)
      if (picked.length < limit) ingestRank(guideRank, guideFloor, false)
    }
    if (picked.length < limit) {
      const strictPool = strictDirectMode
        ? locateCandidates.filter((item) => String(item.sourcePath || '').trim() === provenanceSourcePath)
        : []
      const rankBase = (strictDirectMode && strictPool.length > 0) ? strictPool : locateCandidates
      const rankAll = rankIn(rankBase)
      const allFloor = strictDirectMode
        ? (hasFormulaSignal(key) ? 0.34 : 0.24)
        : (hasFormulaSignal(key) ? 0.3 : 0.2)
      ingestRank(rankAll, allFloor, true)
      if (picked.length < limit) ingestRank(rankAll, allFloor, false)
      if (picked.length <= 0 && rankAll.length > 0) {
        const best = rankAll[0]
        const preferAnchor = Boolean(best?.cand?.anchorId)
        const bestFloor = preferAnchor
          ? (hasFormulaSignal(key) ? 0.3 : 0.24)
          : (hasFormulaSignal(key) ? 0.38 : 0.3)
        if ((best?.score || 0) >= bestFloor) {
          addPicked(best.cand, false)
        }
      }
    }
    if (picked.length <= 0 && hasFormulaSignal(key) && guideOnly.length > 0) {
      const eqNums = extractEquationNumbersFromText(key)
      const eqCandidates = guideOnly.filter((cand) => isEquationLocateCandidate(cand))
      if (eqCandidates.length > 0) {
        const preferByNum = eqNums.length > 0
          ? eqCandidates.filter((cand) => {
            const n = Number(cand.anchorNumber || 0)
            return Number.isFinite(n) && n > 0 && eqNums.includes(Math.floor(n))
          })
          : []
        const pool = preferByNum.length > 0 ? preferByNum : eqCandidates
        let bestEq: LocateCandidate | null = null
        let bestEqScore = -1
        for (const cand of pool) {
          let s = scoreLocateCandidate(key, cand)
          if (eqNums.length > 0 && Number(cand.anchorNumber || 0) > 0) s += 0.4
          if (cand.anchorId) s += 0.2
          if (s > bestEqScore) {
            bestEq = cand
            bestEqScore = s
          }
        }
        if (bestEq && bestEqScore >= 0.34) addPicked(bestEq, false)
      }
    }
    const unique: LocateCandidate[] = []
    const seen = new Set<string>()
    for (const cand of picked) {
      const candKey = `${cand.sourcePath}::${cand.anchorId || ''}::${cand.headingPath}::${cand.focusSnippet.slice(0, 96)}`
      if (seen.has(candKey)) continue
      seen.add(candKey)
      unique.push(cand)
      if (unique.length >= limit) break
    }
    if (unique.length <= 0 && guideOnly.length > 0) {
      const relaxed = rankIn(guideOnly)
      const best = relaxed[0]
      if (best && (best.score || 0) >= 0.08) {
        unique.push(best.cand)
      }
    }
    const first = unique[0]
    if (first) {
      const pickKey = `${first.sourcePath}::${first.anchorId || ''}::${first.headingPath}::${first.focusSnippet.slice(0, 96)}`
      usedCount.set(pickKey, Number(usedCount.get(pickKey) || 0) + 1)
    }
    resolveCache.set(key, unique)
    return unique.slice(0, Math.max(1, limit))
  }

  const locateCandidateKey = (cand: LocateCandidate | null | undefined) => {
    if (!cand) return ''
    if (cand.blockId) return `${cand.sourcePath}::block::${cand.blockId}`
    if (cand.anchorId) return `${cand.sourcePath}::anchor::${cand.anchorId}`
    const headingKey = normalizeLocateText(String(cand.headingPath || ''))
    const focusKey = normalizeLocateText(String(cand.focusSnippet || '')).slice(0, 64)
    return `${cand.sourcePath}::${headingKey}::${focusKey}`
  }

  return {
    resolveProvenanceLocateCandidates,
    resolveLocateCandidates,
    locateCandidateKey,
  }
}
