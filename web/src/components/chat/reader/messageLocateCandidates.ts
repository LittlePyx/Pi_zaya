import type { ReaderDocAnchor } from '../../../api/references'
import { basenameFromSourcePath } from '../../../utils/sourcePath'

export interface RefUiMetaLite {
  display_name?: string
  heading_path?: string
  section_label?: string
  subsection_label?: string
  summary_line?: string
  why_line?: string
  source_path?: string
  anchor_target_kind?: string
  anchor_target_number?: number
}

export interface RefMetaLite {
  source_path?: string
  heading_path?: string
  ref_best_heading_path?: string
  ref_headings?: unknown
  ref_locs?: unknown
  ref_show_snippets?: unknown
  ref_overview_snippets?: unknown
  ref_snippets?: unknown
}

export interface RefHitLite {
  score?: number
  text?: string
  ui_meta?: RefUiMetaLite
  meta?: RefMetaLite
}

export interface LocateCandidate {
  sourcePath: string
  sourceName: string
  headingPath: string
  focusSnippet: string
  matchText: string
  sourceType: 'guide' | 'refs'
  blockId?: string
  anchorId?: string
  anchorKind?: string
  anchorNumber?: number
}

const GUIDE_LOCATE_CANDIDATE_LIMIT = 1600
const REF_LOCATE_CANDIDATE_LIMIT = 900

export function stripMarkdownInline(input: string): string {
  return String(input || '')
    .replace(/!\[[^\]]*\]\([^)]+\)/g, ' ')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/~~([^~]+)~~/g, '$1')
    .replace(/<[^>]+>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

export function normalizeLocateText(input: string): string {
  return String(input || '')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase()
}

export function stripProvenanceNoise(input: string): string {
  return String(input || '')
    .replace(/\[\d{1,3}(?:\s*[-,\u2013\u2014]\s*\d{1,3})*\]/g, ' ')
    .replace(/\(\s*\d{1,3}(?:\s*[-,\u2013\u2014]\s*\d{1,3})*\s*\)/g, ' ')
    .replace(/(?:see|\u53C2\u89C1)\s*\[\d{1,3}(?:\s*[-,\u2013\u2014]\s*\d{1,3})*\]/gi, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

export function tokenizeLocateText(input: string): string[] {
  const src = normalizeLocateText(input)
  if (!src) return []
  const out: string[] = []
  const latin = src.match(/[a-z0-9]{2,}/g) || []
  out.push(...latin)
  const cjkSeq = src.match(/[\u4e00-\u9fff]{1,}/g) || []
  for (const seq of cjkSeq) {
    if (seq.length <= 2) {
      out.push(seq)
      continue
    }
    for (let i = 0; i < seq.length - 1; i += 1) {
      out.push(seq.slice(i, i + 2))
    }
  }
  return out
}

export function overlapScore(a: string, b: string): number {
  const ta = tokenizeLocateText(a)
  const tb = tokenizeLocateText(b)
  if (ta.length === 0 || tb.length === 0) return 0
  const sa = new Set(ta)
  const sb = new Set(tb)
  let overlap = 0
  for (const token of sa) {
    if (sb.has(token)) overlap += 1
  }
  const denom = Math.sqrt(Math.max(1, sa.size) * Math.max(1, sb.size))
  return overlap / denom
}

export function coerceStringArray(input: unknown, maxItems = 8, maxChars = 2200): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  const push = (value: unknown) => {
    if (out.length >= maxItems) return
    const text = String(value || '').replace(/\s+/g, ' ').trim()
    if (!text) return
    const normalized = normalizeLocateText(text)
    if (!normalized || seen.has(normalized)) return
    seen.add(normalized)
    out.push(text.length > maxChars ? `${text.slice(0, maxChars).trimEnd()}...` : text)
  }
  if (Array.isArray(input)) {
    for (const value of input) {
      if (out.length >= maxItems) break
      push(value)
    }
    return out
  }
  push(input)
  return out
}

function pickFirstRefText(loc: Record<string, unknown>): string {
  const keys = ['snippet', 'text', 'quote', 'content', 'summary', 'why']
  for (const key of keys) {
    const value = String(loc[key] || '').trim()
    if (value) return value
  }
  return ''
}

function formulaTokens(text: string): string[] {
  const src = String(text || '')
  if (!src) return []
  const out: string[] = []
  const texCmds = src.match(/\\[a-zA-Z]{2,}/g) || []
  out.push(...texCmds.map((item) => item.toLowerCase()))
  const symbols = src.match(/[A-Za-z](?:_[A-Za-z0-9]+)?(?:\^[A-Za-z0-9]+)?/g) || []
  out.push(...symbols.map((item) => item.toLowerCase()))
  const numbers = src.match(/\b\d{1,4}\b/g) || []
  out.push(...numbers)
  return out
}

export function hasFormulaSignal(text: string): boolean {
  const src = String(text || '')
  if (!src) return false
  if (/[=^_]/.test(src)) return true
  if (/\\[a-zA-Z]{2,}/.test(src)) return true
  if (/\$[^$]{1,80}\$/.test(src) || /\$\$[^]{1,200}\$\$/.test(src)) return true
  return false
}

export function formulaOverlapScore(a: string, b: string): number {
  const ta = new Set(formulaTokens(a))
  const tb = new Set(formulaTokens(b))
  if (ta.size <= 0 || tb.size <= 0) return 0
  let overlap = 0
  for (const token of ta) {
    if (tb.has(token)) overlap += 1
  }
  return overlap / Math.sqrt(ta.size * tb.size)
}

export function hasDisplayFormulaSignal(text: string): boolean {
  const src = String(text || '')
  if (!src) return false
  if (/\$\$[^]{1,400}\$\$/.test(src)) return true
  if (/\\begin\{(?:equation|align|gather|multline|eqnarray)\*?\}/i.test(src)) return true
  const hasMathCore = /=/.test(src) || /\\tag\{\s*\d{1,4}\s*\}/.test(src)
  const hasMathToken = /\\[a-zA-Z]{2,}/.test(src) || /\$[^$]{1,220}\$/.test(src)
  return Boolean(hasMathCore && hasMathToken)
}

export function buildGuideLocateCandidates(
  markdown: string,
  sourcePath: string,
  sourceName: string,
  sourceType: 'guide' | 'refs' = 'guide',
  readerAnchors?: ReaderDocAnchor[],
): LocateCandidate[] {
  const out: LocateCandidate[] = []
  const seen = new Set<string>()
  const pushCandidate = (
    headingPathRaw: string,
    snippetRaw: string,
    extra?: { blockId?: string; anchorId?: string; anchorKind?: string; anchorNumber?: number },
  ) => {
    const headingPath = stripMarkdownInline(headingPathRaw)
    const text = stripMarkdownInline(snippetRaw)
    const formulaLike = hasFormulaSignal(text)
    if (text.length < 24 && !formulaLike) return
    if (formulaLike && text.length < 6) return
    const blockId = String(extra?.blockId || '').trim()
    const anchorId = String(extra?.anchorId || '').trim()
    const anchorKind = String(extra?.anchorKind || '').trim().toLowerCase()
    const anchorNumber = Number(extra?.anchorNumber || 0)
    const key = `${normalizeLocateText(sourcePath)}::${blockId.toLowerCase()}::${anchorId.toLowerCase()}::${normalizeLocateText(headingPath)}::${normalizeLocateText(text).slice(0, 260)}`
    if (seen.has(key)) return
    seen.add(key)
    out.push({
      sourcePath,
      sourceName,
      headingPath,
      focusSnippet: text,
      matchText: [headingPath, text].filter(Boolean).join('\n'),
      sourceType,
      blockId: blockId || undefined,
      anchorId: anchorId || undefined,
      anchorKind: anchorKind || undefined,
      anchorNumber: Number.isFinite(anchorNumber) && anchorNumber > 0 ? Math.floor(anchorNumber) : undefined,
    })
  }

  const pushSentenceCandidates = (
    headingPath: string,
    text: string,
    extra?: { blockId?: string; anchorId?: string; anchorKind?: string; anchorNumber?: number },
  ) => {
    const src = stripMarkdownInline(text)
    if (src.length < 24 && !hasFormulaSignal(src)) return
    const sentenceList = src
      .split(/(?<=[\u3002\uff01\uff1f.!;:\uff1b\uff1a])\s+/)
      .map((item) => item.trim())
      .filter((item) => item.length >= 16)
      .slice(0, 14)
    for (let i = 0; i < sentenceList.length; i += 1) {
      const sentence = sentenceList[i]
      if (sentence.length >= 24) pushCandidate(headingPath, sentence, extra)
      const pair = [sentence, sentenceList[i + 1] || ''].filter(Boolean).join(' ').trim()
      if (pair.length >= 30 && pair.length <= 260) {
        pushCandidate(headingPath, pair, extra)
      }
    }
  }

  const anchorList = Array.isArray(readerAnchors) ? readerAnchors : []
  if (anchorList.length > 0) {
    for (const item of anchorList) {
      const blockId = String(item?.block_id || '').trim()
      const anchorId = String(item?.anchor_id || '').trim()
      const headingPath = String(item?.heading_path || '').trim()
      const kind = String(item?.kind || '').trim().toLowerCase()
      const number = Number(item?.number || 0)
      const text = String(item?.text || '').trim()
      if (!text) continue
      pushCandidate(headingPath, text, {
        blockId,
        anchorId,
        anchorKind: kind,
        anchorNumber: number,
      })
      pushSentenceCandidates(headingPath, text, {
        blockId,
        anchorId,
        anchorKind: kind,
        anchorNumber: number,
      })
    }
    return out.slice(0, GUIDE_LOCATE_CANDIDATE_LIMIT)
  }

  const lines = String(markdown || '').replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n')
  const headingStack: Array<{ level: number; text: string }> = []
  let bucket: string[] = []

  const flush = () => {
    if (bucket.length <= 0) return
    const text = stripMarkdownInline(bucket.join(' ').trim())
    bucket = []
    if (text.length < 24) return
    const headingPath = headingStack.map((item) => item.text).filter(Boolean).join(' / ')
    pushCandidate(headingPath, text)
    pushSentenceCandidates(headingPath, text)
  }

  for (const raw of lines) {
    const line = String(raw || '')
    const heading = line.match(/^\s{0,3}(#{1,6})\s+(.*)$/)
    if (heading) {
      flush()
      const level = heading[1].length
      const text = stripMarkdownInline(heading[2] || '')
      if (text) {
        while (headingStack.length > 0 && headingStack[headingStack.length - 1].level >= level) {
          headingStack.pop()
        }
        headingStack.push({ level, text })
      }
      continue
    }
    if (/^\s*([-*_]\s*){3,}\s*$/.test(line) || /^\s*```/.test(line) || /^\s*~~~/.test(line)) {
      flush()
      continue
    }
    if (!line.trim()) {
      flush()
      continue
    }
    if (/^\s*\|/.test(line) || /^\s*>/.test(line)) {
      flush()
      const text = stripMarkdownInline(line.replace(/^\s*[>|]+\s*/, ''))
      if (text.length >= 24) {
        const headingPath = headingStack.map((item) => item.text).filter(Boolean).join(' / ')
        pushCandidate(headingPath, text)
      }
      continue
    }
    bucket.push(line)
  }
  flush()
  if (out.length <= 0) return out
  // Keep a practical upper bound for runtime matching cost.
  return out.slice(0, GUIDE_LOCATE_CANDIDATE_LIMIT)
}

export function buildRefsLocateCandidatesAll(refHits: RefHitLite[]): LocateCandidate[] {
  const out: LocateCandidate[] = []
  const seen = new Set<string>()
  const push = (candidate: LocateCandidate | null) => {
    if (!candidate) return
    if (out.length >= REF_LOCATE_CANDIDATE_LIMIT) return
    const sourcePath = String(candidate.sourcePath || '').trim()
    const matchText = String(candidate.matchText || '').trim()
    if (!sourcePath || !matchText) return
    const key = `${normalizeLocateText(sourcePath)}::${normalizeLocateText(candidate.headingPath || '')}::${normalizeLocateText(matchText).slice(0, 220)}`
    if (seen.has(key)) return
    seen.add(key)
    out.push(candidate)
  }

  for (const hit of refHits) {
    const ui = hit?.ui_meta || {}
    const meta = hit?.meta || {}
    const sourcePath = String(ui.source_path || meta.source_path || '').trim()
    if (!sourcePath) continue
    const sourceName = String(ui.display_name || '').trim() || basenameFromSourcePath(sourcePath) || 'paper'

    const headingCandidates = new Set<string>([
      String(ui.heading_path || '').trim(),
      String(ui.section_label || '').trim(),
      String(ui.subsection_label || '').trim(),
      String(meta.ref_best_heading_path || '').trim(),
      String(meta.heading_path || '').trim(),
    ].filter(Boolean))
    const anchorKind = String(ui.anchor_target_kind || '').trim().toLowerCase()
    const anchorNum = Number(ui.anchor_target_number || 0)
    for (const heading of coerceStringArray(meta.ref_headings, 8, 160)) {
      headingCandidates.add(String(heading || '').trim())
    }

    const refLocs = Array.isArray(meta.ref_locs) ? meta.ref_locs : []
    for (const loc0 of refLocs.slice(0, 10)) {
      const loc = (loc0 || {}) as Record<string, unknown>
      const headingPath = String(loc.heading_path || loc.heading || '').trim()
      if (headingPath) headingCandidates.add(headingPath)
      const locText = pickFirstRefText(loc)
      const locAnchorId = String(loc.anchor_id || loc.anchorId || '').trim()
      const locAnchorKind = String(loc.anchor_kind || loc.kind || anchorKind || '').trim().toLowerCase()
      const locAnchorNumber = Number(loc.anchor_number || loc.number || anchorNum || 0)
      if (locText) {
        push({
          sourcePath,
          sourceName,
          headingPath: headingPath || String(ui.heading_path || '').trim(),
          focusSnippet: locText,
          matchText: [headingPath, locText].filter(Boolean).join('\n'),
          sourceType: 'refs',
          anchorId: locAnchorId || undefined,
          anchorKind: locAnchorKind || undefined,
          anchorNumber: Number.isFinite(locAnchorNumber) && locAnchorNumber > 0 ? Math.floor(locAnchorNumber) : undefined,
        })
      }
    }

    const snippetSeeds = [
      ...coerceStringArray(ui.summary_line, 1, 360),
      ...coerceStringArray(ui.why_line, 1, 360),
      ...coerceStringArray(meta.ref_show_snippets, 4, 2600),
      ...coerceStringArray(meta.ref_snippets, 4, 2600),
      ...coerceStringArray(meta.ref_overview_snippets, 2, 2600),
    ]
    if (anchorKind === 'equation' && Number.isFinite(anchorNum) && anchorNum > 0) {
      snippetSeeds.push(
        `equation ${anchorNum}`,
        `eq ${anchorNum}`,
        `(${anchorNum})`,
      )
    }
    const headingFallback = Array.from(headingCandidates).find(Boolean) || ''
    for (const seed of snippetSeeds) {
      const pieces = buildGuideLocateCandidates(seed, sourcePath, sourceName, 'refs')
      if (pieces.length > 0) {
        for (const piece of pieces.slice(0, 40)) push(piece)
        continue
      }
      push({
        sourcePath,
        sourceName,
        headingPath: headingFallback,
        focusSnippet: seed,
        matchText: [headingFallback, seed].filter(Boolean).join('\n'),
        sourceType: 'refs',
        anchorKind: anchorKind || undefined,
        anchorNumber: Number.isFinite(anchorNum) && anchorNum > 0 ? Math.floor(anchorNum) : undefined,
      })
    }

    for (const headingPath of headingCandidates) {
      push({
        sourcePath,
        sourceName,
        headingPath,
        focusSnippet: headingPath,
        matchText: headingPath,
        sourceType: 'refs',
        anchorKind: anchorKind || undefined,
        anchorNumber: Number.isFinite(anchorNum) && anchorNum > 0 ? Math.floor(anchorNum) : undefined,
      })
    }
  }
  return out
}

export function dedupeLocateCandidates(candidates: LocateCandidate[]): LocateCandidate[] {
  const out: LocateCandidate[] = []
  const seen = new Set<string>()
  for (const cand of candidates) {
    if (!cand || typeof cand !== 'object') continue
    const sourcePath = String(cand.sourcePath || '').trim()
    const blockId = String(cand.blockId || '').trim()
    const anchorId = String(cand.anchorId || '').trim()
    const headingPath = normalizeLocateText(String(cand.headingPath || ''))
    const snippet = normalizeLocateText(String(cand.focusSnippet || cand.matchText || '')).slice(0, 220)
    const key = `${normalizeLocateText(sourcePath)}::${blockId.toLowerCase()}::${anchorId.toLowerCase()}::${headingPath}::${snippet}`
    if (seen.has(key)) continue
    seen.add(key)
    out.push(cand)
  }
  return out
}
