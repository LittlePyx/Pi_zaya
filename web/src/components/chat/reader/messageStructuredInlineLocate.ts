import {
  hasDisplayFormulaSignal,
  hasFormulaSignal,
  normalizeLocateText,
  overlapScore,
  stripMarkdownInline,
  stripProvenanceNoise,
  type LocateCandidate,
} from './messageLocateCandidates'
import {
  hasSegmentStrictLocateIdentity,
  listStructuredProvenanceSegments,
  normalizeStrictAnchorText,
  normalizeStructuredLocateSnippet,
  shortSegmentLabel,
  type ProvenanceLocateEntry,
  type StructuredProvenanceSegment,
} from './messageStructuredProvenance'
import {
  extractEquationNumbersFromText,
  extractFigureNumbersFromText,
  extractPanelLettersFromText,
  figureNumberMatchScore,
  isPreferredStrictFigureRefSnippet,
  normalizeStructuredLocateKind,
  panelLetterMatchScore,
  scoreProvenanceSegment,
  scoreStructuredAnchorCompatibility,
  type StructuredLocateKind,
} from './messageStructuredLocateScoring'

interface StructuredRenderSegment {
  order: number
  kind: StructuredLocateKind
  text: string
  snippetKey: string
}

export interface StructuredRenderLocateSlot {
  order: number
  kind: StructuredLocateKind
  renderText: string
  renderSnippetKey: string
  entry: ProvenanceLocateEntry
  provenanceIndex: number
  score: number
}

export interface StructuredLocateResolution {
  entry: ProvenanceLocateEntry
  order: number
  fallback: boolean
}

export interface LocateRenderMetaLite {
  kind?: string
  order?: number
}

export function isEquationLocateCandidate(cand: LocateCandidate | null): boolean {
  if (!cand) return false
  const kind = String(cand.anchorKind || '').trim().toLowerCase()
  if (kind === 'equation') return true
  return hasDisplayFormulaSignal(String(cand.focusSnippet || cand.matchText || ''))
}

function splitAnswerRenderSegments(answerMarkdown: string): StructuredRenderSegment[] {
  const lines = String(answerMarkdown || '').replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n')
  const segments: StructuredRenderSegment[] = []
  let buf: string[] = []
  let quoteBuf: string[] = []
  let inFence = false
  let inDisplayMath = false
  let mathBuf: string[] = []
  let order = 0

  const push = (kind: StructuredLocateKind, rawText: string) => {
    const text = stripMarkdownInline(String(rawText || '')).replace(/\s+/g, ' ').trim()
    if (text.length < 10) return
    order += 1
    segments.push({
      order,
      kind,
      text: text.slice(0, 1600),
      snippetKey: normalizeLocateText(text.slice(0, 360)),
    })
  }

  const flushParagraph = () => {
    if (buf.length <= 0) return
    push('paragraph', buf.join('\n'))
    buf = []
  }

  const flushBlockquote = () => {
    if (quoteBuf.length <= 0) return
    push('blockquote', quoteBuf.join('\n'))
    quoteBuf = []
  }

  const flushDisplayMath = () => {
    if (mathBuf.length <= 0) return
    push('equation', mathBuf.join('\n'))
    mathBuf = []
  }

  for (const raw of lines) {
    const line = String(raw || '')
    if (/^\s*(```+|~~~+)\s*$/.test(line)) {
      if (inFence) {
        inFence = false
        flushParagraph()
        flushBlockquote()
        flushDisplayMath()
      } else {
        flushParagraph()
        flushBlockquote()
        inFence = true
      }
      continue
    }
    if (inFence) continue
    const trimmed = line.trim()
    const imageMatch = trimmed.match(/^!\[([^\]]*)\]\([^)]+\)\s*$/)
    if (imageMatch) {
      flushParagraph()
      flushBlockquote()
      flushDisplayMath()
      push('figure', String(imageMatch[1] || '').trim() || trimmed)
      continue
    }
    const eqStart = /^\s*(?:\$\$|\\\[|\\begin\{(?:equation|align|gather|multline|eqnarray)\*?\})/.test(line)
    const eqEnd = /(?:\$\$|\\\]|\\end\{(?:equation|align|gather|multline|eqnarray)\*?\})\s*$/.test(line)
    if (inDisplayMath) {
      mathBuf.push(line)
      if (eqEnd) {
        inDisplayMath = false
        flushDisplayMath()
      }
      continue
    }
    if (eqStart) {
      flushParagraph()
      flushBlockquote()
      mathBuf = [line]
      if (eqEnd && !/^\s*(?:\$\$|\\\[)\s*$/.test(line)) {
        flushDisplayMath()
      } else {
        inDisplayMath = true
      }
      continue
    }
    if (!line.trim()) {
      flushParagraph()
      flushBlockquote()
      flushDisplayMath()
      continue
    }
    if (/^\s{0,3}#{1,6}\s+/.test(line)) {
      flushParagraph()
      flushBlockquote()
      flushDisplayMath()
      continue
    }
    const listMatch = line.match(/^\s*(?:[-*+]|\d+[.)])\s+(.*)$/)
    if (listMatch) {
      flushParagraph()
      flushBlockquote()
      flushDisplayMath()
      push('list_item', String(listMatch[1] || ''))
      continue
    }
    if (/^\s*\|.*\|\s*$/.test(line)) {
      flushParagraph()
      flushBlockquote()
      flushDisplayMath()
      continue
    }
    const quoteMatch = line.match(/^\s*>\s?(.*)$/)
    if (quoteMatch) {
      flushParagraph()
      flushDisplayMath()
      quoteBuf.push(String(quoteMatch[1] || ''))
      continue
    }
    flushBlockquote()
    buf.push(line)
  }
  flushParagraph()
  flushBlockquote()
  flushDisplayMath()
  return segments
}

function scoreStructuredRenderBinding(
  renderSegment: StructuredRenderSegment,
  entry: ProvenanceLocateEntry,
  provenanceSegment: StructuredProvenanceSegment | null,
  targetOrder: number,
): number {
  const segText = String(provenanceSegment?.text || entry.segmentText || '').trim()
  const segKey = String(provenanceSegment?.snippetKey || entry.snippetKey || '').trim()
  let score = Math.max(
    scoreProvenanceSegment(renderSegment.text, segText, segKey),
    overlapScore(renderSegment.text, segText),
  )
  if (segKey && renderSegment.snippetKey === normalizeLocateText(segKey)) {
    score += 0.42
  }
  if (Array.isArray(entry.snippetAliases) && entry.snippetAliases.length > 0) {
    const aliasScore = entry.snippetAliases.reduce((acc, alias) => {
      return Math.max(acc, overlapScore(renderSegment.text, String(alias || '')))
    }, 0)
    score += 0.22 * aliasScore
  }
  if (Array.isArray(provenanceSegment?.snippetAliases) && provenanceSegment.snippetAliases.length > 0) {
    const aliasScore = provenanceSegment.snippetAliases.reduce((acc, alias) => {
      return Math.max(acc, overlapScore(renderSegment.text, String(alias || '')))
    }, 0)
    score += 0.14 * aliasScore
  }
  const figureNumbers = extractFigureNumbersFromText(
    `${entry.anchorText} ${entry.segmentText} ${provenanceSegment?.text || ''}`,
  )
  if (figureNumbers.length > 0) {
    score += 0.56 * figureNumberMatchScore(renderSegment.text, figureNumbers)
  }
  const renderKind = normalizeStructuredLocateKind(renderSegment.kind)
  const segKind = normalizeStructuredLocateKind(String(provenanceSegment?.kind || ''))
  const anchorCompat = scoreStructuredAnchorCompatibility(renderKind, entry)
  if (anchorCompat <= -0.9) return anchorCompat
  score += anchorCompat
  if (renderKind && segKind && renderKind === segKind) {
    score += 0.18
  }
  if (targetOrder > 0) {
    const distance = Math.abs(renderSegment.order - targetOrder)
    if (distance === 0) score += 0.26
    else score -= Math.min(0.48, distance * 0.1)
    if (distance <= 1) score += 0.05
  }
  return score
}

export function buildStructuredRenderLocateSlotMap(
  answerMarkdown: string,
  messageProvenance: Record<string, unknown> | null,
  provenanceLocateEntries: ProvenanceLocateEntry[],
): Map<number, StructuredRenderLocateSlot> {
  const renderSegments = splitAnswerRenderSegments(answerMarkdown)
  const provenanceSegments = listStructuredProvenanceSegments(messageProvenance)
  if (renderSegments.length <= 0 || provenanceSegments.length <= 0 || provenanceLocateEntries.length <= 0) {
    return new Map()
  }

  const provenanceById = new Map(provenanceSegments.map((segment) => [segment.segmentId, segment]))
  const renderableOrdinalBySegmentId = new Map<string, number>()
  let renderableOrdinal = 0
  for (const segment of provenanceSegments) {
    if (normalizeStructuredLocateKind(segment.kind)) {
      renderableOrdinal += 1
      renderableOrdinalBySegmentId.set(segment.segmentId, renderableOrdinal)
    }
  }

  const slotMap = new Map<number, StructuredRenderLocateSlot>()
  const assignedOrders = new Set<number>()
  const orderedEntries = provenanceLocateEntries
    .map((entry, entryIndex) => ({
      entry,
      provenanceSegment: provenanceById.get(entry.segmentId) || null,
      entryIndex,
    }))
    .sort((a, b) => {
      const aIndex = a.provenanceSegment?.index ?? a.entryIndex
      const bIndex = b.provenanceSegment?.index ?? b.entryIndex
      return aIndex - bIndex
    })

  for (const item of orderedEntries) {
    const { entry, provenanceSegment } = item
    const targetOrder = Number(renderableOrdinalBySegmentId.get(entry.segmentId) || 0)
    const formulaQuery = hasFormulaSignal(entry.segmentText || provenanceSegment?.text || '')
  const figureQuery = String(entry.anchorKind || '').trim().toLowerCase() === 'figure'
      || String(entry.claimType || '').trim().toLowerCase() === 'figure_claim'
      || String(entry.claimType || '').trim().toLowerCase() === 'figure_panel'
    let bestSegment: StructuredRenderSegment | null = null
    let bestScore = Number.NEGATIVE_INFINITY
    for (const renderSegment of renderSegments) {
      if (assignedOrders.has(renderSegment.order)) continue
      const score = scoreStructuredRenderBinding(renderSegment, entry, provenanceSegment, targetOrder)
      if (score > bestScore) {
        bestScore = score
        bestSegment = renderSegment
      }
    }
    if (!bestSegment) continue
    const distance = targetOrder > 0 ? Math.abs(bestSegment.order - targetOrder) : 0
    let floor = formulaQuery ? 0.3 : (figureQuery ? 0.26 : 0.44)
    if (targetOrder > 0 && distance === 0) floor -= 0.14
    else if (targetOrder > 0 && distance <= 1) floor -= 0.08
    if (bestScore < floor) continue
    assignedOrders.add(bestSegment.order)
    slotMap.set(bestSegment.order, {
      order: bestSegment.order,
      kind: bestSegment.kind,
      renderText: bestSegment.text,
      renderSnippetKey: bestSegment.snippetKey,
      entry,
      provenanceIndex: provenanceSegment?.index ?? item.entryIndex,
      score: bestScore,
    })
  }
  return slotMap
}

function resolveStructuredRenderLocateSlot(
  snippet: string,
  meta: LocateRenderMetaLite | undefined,
  slotMap: Map<number, StructuredRenderLocateSlot>,
): StructuredRenderLocateSlot | null {
  if (!(slotMap instanceof Map) || slotMap.size <= 0) return null
  const raw = stripProvenanceNoise(stripMarkdownInline(String(snippet || ''))).trim()
  const targetOrderRaw = Number(meta?.order || 0)
  const targetOrder = Number.isFinite(targetOrderRaw) && targetOrderRaw > 0 ? Math.floor(targetOrderRaw) : 0
  const targetKind = normalizeStructuredLocateKind(String(meta?.kind || ''))

  const scoreSlot = (slot: StructuredRenderLocateSlot): number => {
    const compat = scoreStructuredAnchorCompatibility(targetKind || slot.kind, slot.entry)
    if (targetKind && compat <= -0.9) return Number.NEGATIVE_INFINITY
    let score = 0
    if (raw) {
      score = Math.max(
        scoreProvenanceSegment(raw, slot.renderText, slot.renderSnippetKey),
        scoreProvenanceSegment(raw, slot.entry.segmentText, slot.entry.snippetKey),
        overlapScore(raw, slot.renderText),
      )
      if (Array.isArray(slot.entry.snippetAliases) && slot.entry.snippetAliases.length > 0) {
        const aliasScore = slot.entry.snippetAliases.reduce((acc, alias) => {
          return Math.max(acc, overlapScore(raw, String(alias || '')))
        }, 0)
        score += 0.18 * aliasScore
      }
    }
    const figureNumbers = extractFigureNumbersFromText(`${raw} ${slot.entry.anchorText} ${slot.entry.segmentText}`)
    if (figureNumbers.length > 0) {
      score += 0.62 * Math.max(
        figureNumberMatchScore(raw, figureNumbers),
        figureNumberMatchScore(slot.renderText, figureNumbers),
      )
    }
    if (targetKind && slot.kind === targetKind) score += 0.12
    score += Math.max(-0.6, compat)
    if (targetOrder > 0) {
      const distance = Math.abs(slot.order - targetOrder)
      if (distance === 0) score += 0.5
      else score -= Math.min(0.44, distance * 0.18)
    }
    return score
  }

  if (targetOrder > 0) {
    const direct = slotMap.get(targetOrder)
    if (direct) {
      const directScore = scoreSlot(direct)
      const directFloor = raw ? (hasFormulaSignal(raw) ? 0.16 : 0.1) : -1
      if ((!targetKind || direct.kind === targetKind) && directScore >= directFloor) {
        return direct
      }
    }
  }

  let best: StructuredRenderLocateSlot | null = null
  let bestScore = Number.NEGATIVE_INFINITY
  for (const slot of slotMap.values()) {
    const score = scoreSlot(slot)
    if (score > bestScore) {
      best = slot
      bestScore = score
    }
  }
  if (!best) return null
  if (targetKind) {
    const compat = scoreStructuredAnchorCompatibility(targetKind, best.entry)
    if (compat <= -0.9) return null
  }
  const floor = raw ? (hasFormulaSignal(raw) ? 0.34 : 0.48) : 0.22
  return bestScore >= floor ? best : null
}

function resolveStructuredFallbackLocateEntry(
  snippet: string,
  meta: LocateRenderMetaLite | undefined,
  provenanceLocateEntries: ProvenanceLocateEntry[],
): ProvenanceLocateEntry | null {
  const raw = stripProvenanceNoise(stripMarkdownInline(String(snippet || ''))).trim()
  const targetKind = normalizeStructuredLocateKind(String(meta?.kind || ''))
  if (!raw || provenanceLocateEntries.length <= 0) return null

  let best: ProvenanceLocateEntry | null = null
  let bestScore = Number.NEGATIVE_INFINITY
  for (const entry of provenanceLocateEntries) {
    const compat = scoreStructuredAnchorCompatibility(
      targetKind || normalizeStructuredLocateKind(String(entry.anchorKind || '')),
      entry,
    )
    if (targetKind && compat <= -0.9) continue
    let score = Math.max(
      scoreProvenanceSegment(raw, entry.segmentText, entry.snippetKey),
      overlapScore(raw, entry.anchorText || entry.segmentText),
    )
    if (Array.isArray(entry.snippetAliases) && entry.snippetAliases.length > 0) {
      const aliasScore = entry.snippetAliases.reduce((acc, alias) => {
        return Math.max(acc, overlapScore(raw, String(alias || '')))
      }, 0)
      score += 0.18 * aliasScore
    }
    const figureNumbers = extractFigureNumbersFromText(`${raw} ${entry.anchorText} ${entry.segmentText}`)
    if (figureNumbers.length > 0) {
      score += 0.72 * figureNumberMatchScore(`${entry.anchorText} ${entry.segmentText}`, figureNumbers)
    }
    if (targetKind && normalizeStructuredLocateKind(String(entry.anchorKind || '')) === targetKind) {
      score += 0.14
    }
    if (entry.mustLocate || entry.locatePolicy === 'required') {
      score += 0.08
    }
    score += Math.max(-0.4, compat)
    if (score > bestScore) {
      best = entry
      bestScore = score
    }
  }
  if (!best) return null
  const targetIsFigure = targetKind === 'figure'
  const floor = targetIsFigure ? 0.34 : (hasFormulaSignal(raw) ? 0.38 : 0.56)
  return bestScore >= floor ? best : null
}

export interface CreateStructuredInlineLocateResolverOptions {
  strictStructuredInlineLocate: boolean
  provenanceLocateEntries: ProvenanceLocateEntry[]
  structuredRenderSlotMap: Map<number, StructuredRenderLocateSlot>
  structuredLocateOrderBySegmentId: Map<string, number>
  messageProvenance: Record<string, unknown> | null
  structuredProvenanceSegmentsAll: StructuredProvenanceSegment[]
  provenanceBlockMap: Record<string, Record<string, unknown>>
  provenanceSourcePath: string
  effectiveGuideSourcePath: string
  provenanceSourceName: string
  locateSourceName: string
}

export function createStructuredInlineLocateResolver(opts: CreateStructuredInlineLocateResolverOptions): {
  resolveStructuredInlineResolution: (snippet: string, meta?: LocateRenderMetaLite) => StructuredLocateResolution | null
  resolveExactStructuredInlineResolution: (snippet: string, meta?: LocateRenderMetaLite) => StructuredLocateResolution | null
  resolveStrictParagraphEntry: (snippet: string, meta?: LocateRenderMetaLite) => StructuredLocateResolution | null
  isStrictStructuredTargetCompatible: (entry: ProvenanceLocateEntry | null | undefined, targetKindInput?: string) => boolean
} {
  const strictStructuredInlineLocate = Boolean(opts.strictStructuredInlineLocate)
  const provenanceLocateEntries = Array.isArray(opts.provenanceLocateEntries) ? opts.provenanceLocateEntries : []
  const structuredRenderSlotMap = opts.structuredRenderSlotMap instanceof Map ? opts.structuredRenderSlotMap : new Map<number, StructuredRenderLocateSlot>()
  const structuredLocateOrderBySegmentId = opts.structuredLocateOrderBySegmentId instanceof Map ? opts.structuredLocateOrderBySegmentId : new Map<string, number>()
  const messageProvenance = opts.messageProvenance
  const structuredProvenanceSegmentsAll = Array.isArray(opts.structuredProvenanceSegmentsAll) ? opts.structuredProvenanceSegmentsAll : []
  const provenanceBlockMap = opts.provenanceBlockMap || {}
  const provenanceSourcePath = String(opts.provenanceSourcePath || '').trim()
  const effectiveGuideSourcePath = String(opts.effectiveGuideSourcePath || '').trim()
  const provenanceSourceName = String(opts.provenanceSourceName || '').trim()
  const locateSourceName = String(opts.locateSourceName || '').trim()

  const resolveStructuredFigureEntry = (snippet: string): ProvenanceLocateEntry | null => {
    const raw = stripProvenanceNoise(stripMarkdownInline(String(snippet || ''))).trim()
    const figureNumbers = extractFigureNumbersFromText(raw)
    const panelLetters = extractPanelLettersFromText(raw)
    const pureFigureRef = isPreferredStrictFigureRefSnippet(raw) && panelLetters.length <= 0
    const figureEntries = provenanceLocateEntries.filter((entry) => {
      const anchorKind = String(entry.anchorKind || '').trim().toLowerCase()
      const claimType = String(entry.claimType || '').trim().toLowerCase()
      return anchorKind === 'figure' || claimType === 'figure_claim' || claimType === 'figure_panel'
    })
    if (!raw || figureEntries.length <= 0) return null
    const hasFigurePanelEntries = figureEntries.some((entry) => (
      String(entry.claimType || '').trim().toLowerCase() === 'figure_panel'
    ))
    let best: ProvenanceLocateEntry | null = null
    let bestScore = Number.NEGATIVE_INFINITY
    let bestFigureClaim: ProvenanceLocateEntry | null = null
    let bestFigureClaimScore = Number.NEGATIVE_INFINITY
    for (const entry of figureEntries) {
      const claimType = String(entry.claimType || '').trim().toLowerCase()
      const primaryAnchorKind = String(entry.primary?.anchorKind || '').trim().toLowerCase()
      const entryFigureText = `${entry.primary?.headingPath || ''} ${entry.anchorText || ''} ${entry.segmentText || ''}`
      const entryFigureNumbers = Array.from(new Set([
        ...extractFigureNumbersFromText(entryFigureText),
        ...(Number.isFinite(Number(entry.supportFigureNumber || 0)) && Number(entry.supportFigureNumber || 0) > 0
          ? [Math.floor(Number(entry.supportFigureNumber || 0))]
          : []),
      ]))
      const entryPanelLetters = Array.from(new Set([
        ...((Array.isArray(entry.supportPanelLetters) ? entry.supportPanelLetters : [])
          .map((item) => String(item || '').trim().toLowerCase())
          .filter((item) => /^[a-z]$/.test(item))),
        ...extractPanelLettersFromText(`${entry.anchorText || ''} ${entry.segmentText || ''}`),
      ]))
      let score = Math.max(
        scoreProvenanceSegment(raw, entry.segmentText, entry.snippetKey),
        overlapScore(raw, entry.anchorText || entry.segmentText),
      )
      if (figureNumbers.length > 0) {
        score += 0.92 * figureNumberMatchScore(entryFigureText, figureNumbers)
        const entryHasFigureMatch = entryFigureNumbers.some((num) => figureNumbers.includes(num))
        if (entryFigureNumbers.length > 0 && !entryHasFigureMatch) {
          score -= 0.42
        }
      }
      if (panelLetters.length > 0) {
        const panelScore = panelLetterMatchScore(
          `${entry.anchorText || ''} ${entry.segmentText || ''} ${entry.primary?.headingPath || ''} ${entryPanelLetters.join(' ')}`,
          panelLetters,
        )
        if (panelScore > 0) {
          score += 1.34 * panelScore
        } else if (entryPanelLetters.length > 0) {
          score -= 1.18
        } else if (claimType === 'figure_claim') {
          score -= 0.56
        }
      }
      if (pureFigureRef) {
        if (claimType === 'figure_claim') score += 1.12
        else if (claimType === 'figure_panel') score -= 1.02
        if (primaryAnchorKind === 'figure') score += 0.24
        else if (primaryAnchorKind) score -= 0.28
      } else if (hasFigurePanelEntries) {
        if (claimType === 'figure_panel') score += 0.2
        else if (claimType === 'figure_claim') score -= 0.18
      }
      if (entry.mustLocate || entry.locatePolicy === 'required') {
        score += 0.08
      }
      if (score > bestScore) {
        best = entry
        bestScore = score
      }
      if (claimType === 'figure_claim' && score > bestFigureClaimScore) {
        bestFigureClaim = entry
        bestFigureClaimScore = score
      }
    }
    if (
      pureFigureRef
      && best
      && String(best.claimType || '').trim().toLowerCase() === 'figure_panel'
      && bestFigureClaim
      && bestFigureClaimScore >= (bestScore - 0.38)
    ) {
      best = bestFigureClaim
      bestScore = bestFigureClaimScore
    }
    if (bestScore >= 0.26) return best
    if (!messageProvenance || !Array.isArray(messageProvenance?.segments)) return null
    const rawSegments = Array.isArray(messageProvenance.segments) ? messageProvenance.segments : []
    let rawBest: ProvenanceLocateEntry | null = null
    let rawBestScore = Number.NEGATIVE_INFINITY
    let rawBestFigureClaim: ProvenanceLocateEntry | null = null
    let rawBestFigureClaimScore = Number.NEGATIVE_INFINITY
    for (let idx = 0; idx < rawSegments.length; idx += 1) {
      const segment = rawSegments[idx] as unknown as Record<string, unknown> | null
      const currentSegment = structuredProvenanceSegmentsAll[idx] || null
      if (!segment || !currentSegment) continue
      const claimType = String(segment.claim_type || currentSegment.claimType || '').trim().toLowerCase()
      const locatePolicy = String(segment.locate_policy || currentSegment.locatePolicy || '').trim().toLowerCase()
      if ((claimType !== 'figure_claim' && claimType !== 'figure_panel') || locatePolicy === 'hidden') continue
      if (!hasSegmentStrictLocateIdentity(segment, currentSegment)) continue
      const primaryBlockId = String(segment.primary_block_id || '').trim()
      const supportBlockIds = Array.isArray(segment.support_block_ids) ? segment.support_block_ids : []
      const evidenceBlockIds = Array.isArray(segment.evidence_block_ids) ? segment.evidence_block_ids : []
      const blockIds = [
        ...[primaryBlockId].filter(Boolean),
        ...supportBlockIds.map((item) => String(item || '').trim()).filter(Boolean),
        ...evidenceBlockIds.map((item) => String(item || '').trim()).filter(Boolean),
      ]
      const candidates: LocateCandidate[] = []
      const seenBlock = new Set<string>()
      for (const blockIdRaw of blockIds) {
        const blockId = String(blockIdRaw || '').trim()
        if (!blockId || seenBlock.has(blockId)) continue
        const block = provenanceBlockMap[blockId]
        if (!block || typeof block !== 'object') continue
        seenBlock.add(blockId)
        const headingPath = String(block.heading_path || '').trim()
        const blockText = stripMarkdownInline(String(block.text || '')).trim()
        const anchorId = String(block.anchor_id || '').trim()
        const anchorText = normalizeStrictAnchorText(String(segment.anchor_text || currentSegment.anchorText || ''))
        const evidenceQuote = normalizeStrictAnchorText(String(segment.evidence_quote || anchorText || ''))
        const focusSnippet = anchorText || evidenceQuote || blockText || currentSegment.text || headingPath
        if (!focusSnippet) continue
        candidates.push({
          sourcePath: provenanceSourcePath || effectiveGuideSourcePath,
          sourceName: provenanceSourceName || locateSourceName || (provenanceSourcePath.split(/[\\/]/).pop() || 'paper'),
          headingPath,
          focusSnippet,
          matchText: [headingPath, anchorText || evidenceQuote || '', blockText || currentSegment.text].filter(Boolean).join('\n'),
          sourceType: 'guide',
          blockId,
          anchorId: anchorId || undefined,
          anchorKind: 'figure',
        })
      }
      if (candidates.length <= 0) continue
      const entry: ProvenanceLocateEntry = {
        segmentId: String(segment.segment_id || currentSegment.segmentId || `seg_${idx + 1}`).trim(),
        label: shortSegmentLabel(String(segment.anchor_text || currentSegment.anchorText || currentSegment.text || '')),
        segmentText: String(currentSegment.text || '').trim(),
        evidenceQuote: normalizeStrictAnchorText(String(segment.evidence_quote || segment.anchor_text || '')),
        hitLevel: String(segment.hit_level || currentSegment.hitLevel || '').trim().toLowerCase(),
        claimType,
        mustLocate: Boolean(segment.must_locate || locatePolicy === 'required'),
        locatePolicy,
        claimGroupId: String(segment.claim_group_id || currentSegment.claimGroupId || '').trim(),
        claimGroupKind: String(segment.claim_group_kind || currentSegment.claimGroupKind || '').trim().toLowerCase(),
        anchorKind: 'figure',
        anchorText: normalizeStrictAnchorText(String(segment.anchor_text || currentSegment.anchorText || '')),
        equationNumber: 0,
        supportFigureNumber: Number.isFinite(Number(segment.support_slot_figure_number || 0))
          ? Math.max(0, Math.floor(Number(segment.support_slot_figure_number || 0)))
          : 0,
        supportPanelLetters: Array.isArray(segment.support_slot_panel_letters)
          ? Array.from(new Set(
            segment.support_slot_panel_letters
              .map((item) => String(item || '').trim().toLowerCase())
              .filter((item) => /^[a-z]$/.test(item)),
          ))
          : [],
        snippetKey: normalizeStructuredLocateSnippet(String(currentSegment.snippetKey || currentSegment.text || '').trim()),
        snippetAliases: Array.isArray(currentSegment.snippetAliases) ? currentSegment.snippetAliases : [],
        primary: candidates[0],
        alternatives: candidates,
        sourceSegmentId: String(segment.segment_id || '').trim() || undefined,
      }
      let score = Math.max(
        scoreProvenanceSegment(raw, entry.segmentText, entry.snippetKey),
        overlapScore(raw, entry.anchorText || entry.segmentText),
      )
      if (figureNumbers.length > 0) {
        score += 0.92 * figureNumberMatchScore(`${entry.anchorText} ${entry.segmentText}`, figureNumbers)
      }
      if (panelLetters.length > 0) {
        const panelScore = panelLetterMatchScore(
          `${entry.anchorText || ''} ${entry.segmentText || ''} ${entry.primary?.headingPath || ''} ${(entry.supportPanelLetters || []).join(' ')}`,
          panelLetters,
        )
        if (panelScore > 0) score += 1.28 * panelScore
        else if (Array.isArray(entry.supportPanelLetters) && entry.supportPanelLetters.length > 0) score -= 1.05
      }
      if (pureFigureRef) {
        if (claimType === 'figure_claim') score += 1.08
        else if (claimType === 'figure_panel') score -= 0.92
      }
      if (score > rawBestScore) {
        rawBest = entry
        rawBestScore = score
      }
      if (claimType === 'figure_claim' && score > rawBestFigureClaimScore) {
        rawBestFigureClaim = entry
        rawBestFigureClaimScore = score
      }
    }
    if (
      pureFigureRef
      && rawBest
      && String(rawBest.claimType || '').trim().toLowerCase() === 'figure_panel'
      && rawBestFigureClaim
      && rawBestFigureClaimScore >= (rawBestScore - 0.38)
    ) {
      rawBest = rawBestFigureClaim
      rawBestScore = rawBestFigureClaimScore
    }
    return rawBestScore >= 0.26 ? rawBest : null
  }
  const resolveStructuredEquationEntry = (
    snippet: string,
  ): StructuredLocateResolution | null => {
    const raw = stripProvenanceNoise(stripMarkdownInline(String(snippet || ''))).trim()
    if (!raw) return null
    const eqNumbers = extractEquationNumbersFromText(raw)
    const equationEntries = provenanceLocateEntries.filter((entry) => {
      const anchorKind = String(entry.anchorKind || '').trim().toLowerCase()
      const claimType = String(entry.claimType || '').trim().toLowerCase()
      return anchorKind === 'equation' || claimType === 'formula_claim'
    })
    if (equationEntries.length <= 0) return null
    let best: ProvenanceLocateEntry | null = null
    let bestScore = Number.NEGATIVE_INFINITY
    for (const entry of equationEntries) {
      const claimType = String(entry.claimType || '').trim().toLowerCase()
      const anchorKind = String(entry.anchorKind || '').trim().toLowerCase()
      const formulaOrigin = String(entry.formulaOrigin || '').trim().toLowerCase()
      const locateSurfacePolicy = String(entry.locateSurfacePolicy || '').trim().toLowerCase()
      const entryText = [
        entry.anchorText,
        entry.evidenceQuote,
        entry.segmentText,
        entry.primary?.focusSnippet,
        entry.primary?.headingPath,
      ].filter(Boolean).join(' ')
      const entryNumbers = Array.from(new Set([
        Number(entry.equationNumber || 0),
        Number(entry.primary?.anchorNumber || 0),
        ...extractEquationNumbersFromText(entryText),
      ].filter((item) => Number.isFinite(Number(item)) && Number(item) > 0)
        .map((item) => Math.floor(Number(item)))))
      let score = Math.max(
        scoreProvenanceSegment(raw, entry.segmentText, entry.snippetKey),
        overlapScore(raw, entry.anchorText || entry.segmentText),
        overlapScore(raw, entry.evidenceQuote || entry.segmentText),
      )
      if (eqNumbers.length > 0) {
        const matchedNumber = entryNumbers.some((num) => eqNumbers.includes(num))
        if (matchedNumber) score += 1.65
        else if (entryNumbers.length > 0) score -= 0.95
      }
      if (claimType === 'formula_claim') score += 0.46
      if (anchorKind === 'equation') score += 0.42
      if (formulaOrigin === 'source') score += 0.22
      if (locateSurfacePolicy === 'primary') score += 0.18
      if (entry.mustLocate || entry.locatePolicy === 'required') score += 0.08
      if (score > bestScore) {
        best = entry
        bestScore = score
      }
    }
    const floor = eqNumbers.length > 0 ? 0.72 : 0.5
    if (!best || bestScore < floor) return null
    const order = Number(structuredLocateOrderBySegmentId.get(String(best.segmentId || '').trim()) || 0)
    return {
      entry: best,
      order: order > 0
        ? order
        : 10000 + Math.max(0, provenanceLocateEntries.findIndex((item) => item.segmentId === best.segmentId)),
      fallback: !(order > 0),
    }
  }
  const resolveStructuredQuoteEntry = (
    snippet: string,
    targetKindInput?: string,
  ): StructuredLocateResolution | null => {
    const raw = stripProvenanceNoise(stripMarkdownInline(String(snippet || ''))).trim()
    const targetKind = normalizeStructuredLocateKind(String(targetKindInput || ''))
    if (!raw) return null
    const quoteEntries = provenanceLocateEntries.filter((entry) => {
      const anchorKind = String(entry.anchorKind || '').trim().toLowerCase()
      const claimType = String(entry.claimType || '').trim().toLowerCase()
      if (targetKind === 'quote') {
        return anchorKind === 'quote' || claimType === 'quote_claim'
      }
      if (targetKind === 'blockquote') {
        return anchorKind === 'blockquote' || claimType === 'blockquote_claim' || claimType === 'quote_claim'
      }
      return anchorKind === 'quote' || anchorKind === 'blockquote' || claimType === 'quote_claim' || claimType === 'blockquote_claim'
    })
    if (quoteEntries.length <= 0) return null

    let best: ProvenanceLocateEntry | null = null
    let bestScore = Number.NEGATIVE_INFINITY
    for (const entry of quoteEntries) {
      const anchorKind = normalizeStructuredLocateKind(String(entry.anchorKind || ''))
      const compat = scoreStructuredAnchorCompatibility(targetKind || anchorKind, entry)
      if (targetKind && compat <= -0.9) continue
      let score = Math.max(
        scoreProvenanceSegment(raw, entry.segmentText, entry.snippetKey),
        overlapScore(raw, entry.anchorText || entry.segmentText),
      )
      if (Array.isArray(entry.snippetAliases) && entry.snippetAliases.length > 0) {
        const aliasScore = entry.snippetAliases.reduce((acc, alias) => {
          return Math.max(acc, overlapScore(raw, String(alias || '')))
        }, 0)
        score += 0.18 * aliasScore
      }
      if (entry.mustLocate || entry.locatePolicy === 'required') {
        score += 0.08
      }
      if (targetKind && anchorKind === targetKind) {
        score += 0.16
      }
      score += Math.max(-0.4, compat)
      if (score > bestScore) {
        best = entry
        bestScore = score
      }
    }
    const floor = targetKind === 'quote' ? 0.46 : 0.44
    if (!best || bestScore < floor) return null
    const order = Number(structuredLocateOrderBySegmentId.get(String(best.segmentId || '').trim()) || 0)
    return {
      entry: best,
      order: order > 0 ? order : 10000 + Math.max(0, provenanceLocateEntries.findIndex((item) => item.segmentId === best.segmentId)),
      fallback: !(order > 0),
    }
  }
  const isStrictStructuredTargetCompatible = (
    entry: ProvenanceLocateEntry | null | undefined,
    targetKindInput?: string,
  ): boolean => {
    const targetKind = normalizeStructuredLocateKind(String(targetKindInput || ''))
    if (!entry) return false
    if (!targetKind) return true
    const claimType = String(entry.claimType || '').trim().toLowerCase()
    const anchorKind = String(entry.anchorKind || '').trim().toLowerCase()
    if (targetKind === 'quote') {
      return anchorKind === 'quote' || claimType === 'quote_claim'
    }
    if (targetKind === 'blockquote') {
      return anchorKind === 'blockquote' || claimType === 'blockquote_claim' || claimType === 'quote_claim'
    }
    if (targetKind === 'figure') {
      return anchorKind === 'figure' || claimType === 'figure_claim' || claimType === 'figure_panel'
    }
    if (targetKind === 'equation') {
      return anchorKind === 'equation' && claimType === 'formula_claim'
    }
    return true
  }
  const resolveStructuredInlineResolution = (
    snippet: string,
    meta?: LocateRenderMetaLite,
  ): StructuredLocateResolution | null => {
    if (!strictStructuredInlineLocate) return null
    const targetKind = normalizeStructuredLocateKind(String(meta?.kind || ''))
    const quoteEntry = (targetKind === 'quote' || targetKind === 'blockquote')
      ? resolveStructuredQuoteEntry(snippet, targetKind)
      : null
    if (quoteEntry) return quoteEntry
    if (targetKind === 'figure') {
      const figureEntry = resolveStructuredFigureEntry(snippet)
      if (!figureEntry || !isStrictStructuredTargetCompatible(figureEntry, targetKind)) return null
      const order = Number(structuredLocateOrderBySegmentId.get(String(figureEntry.segmentId || '').trim()) || 0)
      return {
        entry: figureEntry,
        order: order > 0
          ? order
          : 10000 + Math.max(0, provenanceLocateEntries.findIndex((item) => item.segmentId === figureEntry.segmentId)),
        fallback: !(order > 0),
      }
    }
    const slot = resolveStructuredRenderLocateSlot(snippet, meta, structuredRenderSlotMap)
    if (slot && isStrictStructuredTargetCompatible(slot.entry, targetKind)) {
      return {
        entry: slot.entry,
        order: slot.order,
        fallback: false,
      }
    }
    if (targetKind === 'equation') return resolveStructuredEquationEntry(snippet)
    const fallbackEntry = resolveStructuredFallbackLocateEntry(snippet, meta, provenanceLocateEntries)
    const finalEntry = fallbackEntry
    if (!finalEntry || !isStrictStructuredTargetCompatible(finalEntry, targetKind)) return null
    return {
      entry: finalEntry,
      order: 10000 + Math.max(0, provenanceLocateEntries.findIndex((item) => item.segmentId === finalEntry.segmentId)),
      fallback: true,
    }
  }
  const resolveExactStructuredInlineResolution = (
    snippet: string,
    meta?: LocateRenderMetaLite,
  ): StructuredLocateResolution | null => {
    const targetKind = normalizeStructuredLocateKind(String(meta?.kind || ''))
    const resolved = resolveStructuredInlineResolution(snippet, meta)
    if (!resolved) return null
    if (resolved.fallback && targetKind !== 'figure') return null
    return resolved
  }
  const resolveStrictParagraphEntry = (
    snippet: string,
    meta?: LocateRenderMetaLite,
  ): StructuredLocateResolution | null => {
    const slotResolved = resolveStructuredInlineResolution(snippet, meta)
    if (slotResolved?.entry) return slotResolved
    const raw = stripProvenanceNoise(stripMarkdownInline(String(snippet || ''))).trim()
    if (!raw || provenanceLocateEntries.length <= 0) return null
    const eqNumbers = extractEquationNumbersFromText(raw)
    const figureNumbers = extractFigureNumbersFromText(raw)
    const wantsEquation = eqNumbers.length > 0
      || hasFormulaSignal(raw)
      || /\b(?:eq|eq\.|equation|formula)\b/i.test(raw)
    const wantsFigure = figureNumbers.length > 0
      || /\b(?:fig|fig\.|figure)\b/i.test(raw)
    let best: ProvenanceLocateEntry | null = null
    let bestScore = Number.NEGATIVE_INFINITY
    for (const entry of provenanceLocateEntries) {
      const anchorKind = String(entry.anchorKind || entry.primary?.anchorKind || '').trim().toLowerCase()
      const claimType = String(entry.claimType || '').trim().toLowerCase()
      const entryText = [
        entry.anchorText,
        entry.evidenceQuote,
        entry.segmentText,
        entry.primary?.focusSnippet,
        entry.primary?.matchText,
        entry.primary?.headingPath,
      ].filter(Boolean).join(' ')
      let score = Math.max(
        scoreProvenanceSegment(raw, entry.segmentText, entry.snippetKey),
        overlapScore(raw, entry.anchorText || entry.segmentText),
        overlapScore(raw, entry.evidenceQuote || entry.segmentText),
        overlapScore(raw, entry.primary?.focusSnippet || entry.primary?.matchText || ''),
      )
      if (Array.isArray(entry.snippetAliases) && entry.snippetAliases.length > 0) {
        score += 0.14 * entry.snippetAliases.reduce((acc, alias) => (
          Math.max(acc, overlapScore(raw, String(alias || '')))
        ), 0)
      }
      if (wantsEquation) {
        const entryEqNumbers = Array.from(new Set([
          Number(entry.equationNumber || 0),
          Number(entry.primary?.anchorNumber || 0),
          ...extractEquationNumbersFromText(entryText),
        ].filter((item) => Number.isFinite(Number(item)) && Number(item) > 0)
          .map((item) => Math.floor(Number(item)))))
        if (anchorKind === 'equation' || claimType === 'formula_claim' || isEquationLocateCandidate(entry.primary)) {
          score += 1.18
        } else if (anchorKind === 'figure' || claimType === 'figure_claim' || claimType === 'figure_panel') {
          score -= 0.78
        }
        if (eqNumbers.length > 0) {
          const matched = entryEqNumbers.some((num) => eqNumbers.includes(num))
          if (matched) score += 1.25
          else if (entryEqNumbers.length > 0) score -= 0.42
        }
        if (String(entry.formulaOrigin || '').trim().toLowerCase() === 'source') score += 0.16
        if (String(entry.locateSurfacePolicy || '').trim().toLowerCase() === 'primary') score += 0.14
      }
      if (wantsFigure) {
        if (anchorKind === 'figure' || claimType === 'figure_claim' || claimType === 'figure_panel') {
          score += 1.04
        } else if (anchorKind === 'equation' || claimType === 'formula_claim') {
          score -= 0.72
        }
        if (figureNumbers.length > 0) {
          score += 0.92 * figureNumberMatchScore(entryText, figureNumbers)
        }
      }
      if (entry.mustLocate || entry.locatePolicy === 'required') score += 0.08
      if (score > bestScore) {
        best = entry
        bestScore = score
      }
    }
    if (!best) return null
    const floor = wantsEquation || wantsFigure ? 0.36 : 0.52
    if (bestScore < floor) return null
    const order = Number(structuredLocateOrderBySegmentId.get(String(best.segmentId || '').trim()) || 0)
    return {
      entry: best,
      order: order > 0
        ? order
        : 10000 + Math.max(0, provenanceLocateEntries.findIndex((item) => item.segmentId === best?.segmentId)),
      fallback: !(order > 0),
    }
  }

  return {
    resolveStructuredInlineResolution,
    resolveExactStructuredInlineResolution,
    resolveStrictParagraphEntry,
    isStrictStructuredTargetCompatible,
  }
}
