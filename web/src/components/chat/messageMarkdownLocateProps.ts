import {
  hasFormulaSignal,
  normalizeLocateText,
  type LocateCandidate,
} from './reader/messageLocateCandidates'
import {
  isEquationLocateCandidate,
  type LocateRenderMetaLite,
  type StructuredLocateResolution,
} from './reader/messageStructuredInlineLocate'
import {
  isPreferredStrictFigureRefSnippet,
  normalizeStructuredLocateKind,
} from './reader/messageStructuredLocateScoring'
import type { ProvenanceLocateEntry } from './reader/messageStructuredProvenance'

type LocateSurfaceKind = 'paragraph' | 'list_item' | 'quote' | 'blockquote' | 'equation' | 'figure' | 'table'
type InlineLocateTokenKind = 'quote' | 'figure_ref' | 'equation_ref' | 'table_ref'

interface LocateButtonAttrs {
  className?: string
  focus?: string
  blockId?: string
  anchorId?: string
  anchorKind?: string
  heading?: string
}

export interface MessageMarkdownLocateProps {
  inlineLocateTokenPolicy?: Partial<Record<InlineLocateTokenKind, boolean>>
  inlineTextLocateEnabled: boolean
  inlineTextTailLocateEnabled: boolean
  locateSurfacePolicy?: Partial<Record<LocateSurfaceKind, boolean>>
  canLocateSnippet?: (snippet: string, meta?: LocateRenderMetaLite) => boolean
  onLocateSnippet?: (snippet: string, meta?: LocateRenderMetaLite) => void
  locateTitleResolver?: (snippet: string) => string
  locateButtonAttrsResolver?: (snippet: string, meta?: LocateRenderMetaLite) => LocateButtonAttrs | null | undefined
}

export interface BuildMessageMarkdownLocatePropsOptions {
  enableLocateUi: boolean
  guideSourcePath: string
  guideInlineTextTailLocate: boolean
  strictStructuredInlineLocate: boolean
  suppressLooseInlineLocate: boolean
  strictStructuredLocateOnly: boolean
  allowedStructuredRenderOrders: Set<number>
  resolveStrictParagraphEntry: (snippet: string, meta?: LocateRenderMetaLite) => StructuredLocateResolution | null
  resolveExactStructuredInlineResolution: (snippet: string, meta?: LocateRenderMetaLite) => StructuredLocateResolution | null
  isStrictStructuredTargetCompatible: (entry: ProvenanceLocateEntry, targetKind: string) => boolean
  resolveProvenanceLocateCandidates: (snippet: string, limit?: number) => LocateCandidate[]
  resolveLocateCandidates: (snippet: string, limit?: number) => LocateCandidate[]
  locateCandidateKey: (cand: LocateCandidate | null | undefined) => string
  openReaderByCandidates: (
    pickedList: LocateCandidate[],
    snippet: string,
    opts?: { strictLocate?: boolean; highlightSnippet?: string; relatedBlockIds?: string[] },
  ) => void
  openReaderByStructuredEntry: (entry: ProvenanceLocateEntry, snippet: string) => void
}

export function buildMessageMarkdownLocateProps(
  opts: BuildMessageMarkdownLocatePropsOptions,
): MessageMarkdownLocateProps {
  if (!opts.enableLocateUi) {
    return {
      inlineTextLocateEnabled: false,
      inlineTextTailLocateEnabled: false,
    }
  }

  const locateButtonShownKeys = new Set<string>()
  const locateButtonCap = 5
  let optionalLocateButtonCount = 0

  const canLocateSnippet = (snippet: string, meta?: LocateRenderMetaLite): boolean => {
    if (opts.strictStructuredLocateOnly) {
      if (!opts.strictStructuredInlineLocate) return false
      const targetKind = normalizeStructuredLocateKind(String(meta?.kind || ''))
      if (targetKind === 'paragraph' || targetKind === 'list_item') {
        const raw = String(snippet || '').trim()
        if (raw.length < 18) return false
        const structured = opts.resolveStrictParagraphEntry(snippet, meta)
        const picked = structured?.entry?.primary || opts.resolveProvenanceLocateCandidates(snippet, 1)[0] || null
        if (!picked) return false
        const keyBase = opts.locateCandidateKey(picked)
        const snippetKey = normalizeLocateText(raw).slice(0, 96)
        const key = keyBase ? `${keyBase}::${snippetKey}` : snippetKey
        if (!key) return false
        if (locateButtonShownKeys.has(key)) return false
        if (optionalLocateButtonCount >= locateButtonCap) return false
        locateButtonShownKeys.add(key)
        optionalLocateButtonCount += 1
        return true
      }
      if (!['quote', 'blockquote', 'equation', 'figure'].includes(targetKind)) {
        return false
      }
      const resolved = opts.resolveExactStructuredInlineResolution(snippet, meta)
      const entry = resolved?.entry || null
      if (!entry) return false
      const order = Number(resolved?.order || 0)
      if (targetKind !== 'figure' && !opts.allowedStructuredRenderOrders.has(order)) return false
      if (!opts.isStrictStructuredTargetCompatible(entry, targetKind)) {
        return false
      }
      const claimType = String(entry.claimType || '').trim().toLowerCase()
      const anchorKind = String(entry.anchorKind || '').trim().toLowerCase()
      const formulaOrigin = String(entry.formulaOrigin || '').trim().toLowerCase()
      const locateSurfacePolicy = String(entry.locateSurfacePolicy || '').trim().toLowerCase()
      if ((anchorKind === 'quote' || claimType === 'quote_claim') && targetKind !== 'quote') {
        return false
      }
      if ((anchorKind === 'blockquote' || claimType === 'blockquote_claim') && targetKind !== 'blockquote') {
        return false
      }
      if ((anchorKind === 'figure' || claimType === 'figure_claim' || claimType === 'figure_panel') && targetKind !== 'figure') {
        return false
      }
      if (targetKind === 'equation') {
        if (claimType !== 'formula_claim' || anchorKind !== 'equation') {
          return false
        }
        if (formulaOrigin !== 'source' || locateSurfacePolicy !== 'primary') {
          return false
        }
      }
      if (targetKind === 'figure') {
        return isPreferredStrictFigureRefSnippet(snippet)
      }
      return true
    }

    const raw = String(snippet || '').trim()
    const formulaSnippet = hasFormulaSignal(raw)
    if (!formulaSnippet && raw.length < 18) return false
    const directPickedList = opts.resolveProvenanceLocateCandidates(snippet, 1)
    const directPicked = formulaSnippet
      ? (directPickedList.find((item) => isEquationLocateCandidate(item)) || directPickedList[0] || null)
      : (directPickedList[0] || null)
    const pickedList = directPicked
      ? directPickedList
      : opts.resolveLocateCandidates(snippet, 1)
    const picked = formulaSnippet
      ? (pickedList.find((item) => isEquationLocateCandidate(item)) || pickedList[0] || null)
      : (pickedList[0] || null)
    if (!picked) return false
    const keyBase = opts.locateCandidateKey(picked)
    const snippetKey = normalizeLocateText(raw).slice(0, 96)
    const key = keyBase
      ? `${keyBase}::${snippetKey}`
      : snippetKey
    if (!key) return false
    if (locateButtonShownKeys.has(key)) return false
    if (!directPicked && optionalLocateButtonCount >= locateButtonCap) return false
    locateButtonShownKeys.add(key)
    if (!directPicked) optionalLocateButtonCount += 1
    return true
  }

  const onLocateSnippet = (snippet: string, meta?: LocateRenderMetaLite) => {
    if (opts.strictStructuredLocateOnly) {
      if (!opts.strictStructuredInlineLocate) return
      const targetKind = normalizeStructuredLocateKind(String(meta?.kind || ''))
      if (targetKind === 'paragraph' || targetKind === 'list_item') {
        const structured = opts.resolveStrictParagraphEntry(snippet, meta)
        const entry = structured?.entry || null
        if (entry) {
          opts.openReaderByStructuredEntry(entry, snippet)
          return
        }
        const pickedList = opts.resolveProvenanceLocateCandidates(snippet, 6)
        if (pickedList.length <= 0) return
        opts.openReaderByCandidates(pickedList, snippet, { strictLocate: true })
        return
      }
      const resolved = opts.resolveExactStructuredInlineResolution(snippet, meta)
      const entry = resolved?.entry || null
      if (!entry) return
      if (targetKind !== 'figure' && !opts.allowedStructuredRenderOrders.has(Number(resolved?.order || 0))) return
      opts.openReaderByStructuredEntry(entry, snippet)
      return
    }
    const raw = String(snippet || '').trim()
    const formulaSnippet = hasFormulaSignal(raw)
    const pickedListRaw = opts.resolveLocateCandidates(snippet, 6)
    const pickedList = formulaSnippet
      ? [
        ...pickedListRaw.filter((item) => isEquationLocateCandidate(item)),
        ...pickedListRaw.filter((item) => !isEquationLocateCandidate(item)),
      ]
      : pickedListRaw
    if (pickedList.length <= 0) return
    opts.openReaderByCandidates(pickedList, snippet)
  }

  const locateTitleResolver = (snippet: string) => {
    if (opts.strictStructuredLocateOnly) {
      const resolved = opts.resolveExactStructuredInlineResolution(snippet)
        || (isPreferredStrictFigureRefSnippet(snippet)
          ? opts.resolveExactStructuredInlineResolution(snippet, { kind: 'figure', order: 0 })
          : null)
      const entry = resolved?.entry || null
      if (entry) {
        const heading = String(entry.primary.headingPath || '').trim()
        return heading ? `\u5b9a\u4f4d\u5230\u539f\u6587\u8bc1\u636e\uff1a${heading}` : '\u5b9a\u4f4d\u5230\u539f\u6587\u8bc1\u636e'
      }
    }
    const formulaSnippet = hasFormulaSignal(String(snippet || ''))
    const pickedList = opts.resolveLocateCandidates(snippet, formulaSnippet ? 2 : 1)
    const picked = formulaSnippet
      ? (pickedList.find((item) => isEquationLocateCandidate(item)) || pickedList[0] || null)
      : (pickedList[0] || null)
    if (!picked) return '\u5b9a\u4f4d\u5230\u539f\u6587'
    const heading = String(picked.headingPath || '').trim()
    return heading ? `\u5b9a\u4f4d\u5230\u539f\u6587\uff1a${heading}` : '\u5b9a\u4f4d\u5230\u539f\u6587'
  }

  const locateButtonAttrsResolver = (snippet: string, meta?: LocateRenderMetaLite): LocateButtonAttrs | null => {
    if (!opts.strictStructuredLocateOnly) return null
    const toAttrs = (candidate: LocateCandidate | null | undefined): LocateButtonAttrs | null => {
      if (!candidate) return null
      return {
        className: 'kb-prov-locate-chip',
        focus: String(candidate.focusSnippet || candidate.matchText || snippet || '').trim().slice(0, 220),
        blockId: String(candidate.blockId || '').trim(),
        anchorId: String(candidate.anchorId || '').trim(),
        anchorKind: String(candidate.anchorKind || '').trim(),
        heading: String(candidate.headingPath || '').trim(),
      }
    }
    const targetKind = normalizeStructuredLocateKind(String(meta?.kind || ''))
    if (targetKind === 'paragraph' || targetKind === 'list_item') {
      const structured = opts.resolveStrictParagraphEntry(snippet, meta)
      return toAttrs(structured?.entry?.primary || opts.resolveProvenanceLocateCandidates(snippet, 1)[0] || null)
    }
    const resolved = opts.resolveExactStructuredInlineResolution(snippet, meta)
      || (targetKind === 'figure' || isPreferredStrictFigureRefSnippet(snippet)
        ? opts.resolveExactStructuredInlineResolution(snippet, { kind: 'figure', order: Number(meta?.order || 0) })
        : null)
    const entry = resolved?.entry || null
    if (!entry) return null
    if (targetKind !== 'figure' && !opts.allowedStructuredRenderOrders.has(Number(resolved?.order || 0))) return null
    return toAttrs(entry.primary)
  }

  return {
    inlineLocateTokenPolicy: opts.guideSourcePath ? { quote: true, figure_ref: true } : undefined,
    inlineTextLocateEnabled: (!opts.guideSourcePath || opts.strictStructuredInlineLocate) && !opts.suppressLooseInlineLocate,
    inlineTextTailLocateEnabled: opts.guideInlineTextTailLocate,
    locateSurfacePolicy: opts.guideSourcePath
      ? {
        paragraph: opts.guideInlineTextTailLocate,
        list_item: opts.guideInlineTextTailLocate,
        quote: true,
        blockquote: true,
        equation: true,
        figure: true,
      }
      : undefined,
    canLocateSnippet,
    onLocateSnippet,
    locateTitleResolver,
    locateButtonAttrsResolver,
  }
}
