import type { ReaderOpenPayload } from '../chat/reader/readerTypes'
import { basenameFromSourcePath, normalizeSourcePathForMatch as normalizeSourcePathForMatchShared } from '../../utils/sourcePath'

export interface RefsPanelRefUiMeta {
  display_name?: string
  heading_path?: string
  section_label?: string
  subsection_label?: string
  page_start?: number
  page_end?: number
  score?: number | null
  score_pending?: boolean
  summary_line?: string
  summary_kind?: string
  summary_display_role?: string
  summary_label?: string
  summary_title?: string
  summary_generation?: string
  summary_basis?: string
  polish_status?: string
  polish_source?: string
  polish_detail?: string
  summary_polish_status?: string
  why_polish_status?: string
  why_line?: string
  why_generation?: string
  why_basis?: string
  semantic_badges?: Array<{
    text?: string
    score?: number
  }>
  can_open?: boolean
  citation_meta?: Record<string, unknown>
  source_path?: string
  source_kind?: string
  reader_open?: Partial<ReaderOpenPayload>
  card_view?: unknown
  cardView?: unknown
}

export interface RefsPanelRefHit {
  score?: number
  text?: string
  meta?: {
    source_path?: string
    ref_pack_state?: string
  }
  ui_meta?: RefsPanelRefUiMeta
}

export interface RefsPanelRefEntry {
  prompt?: string
  hits?: RefsPanelRefHit[]
  display_state?: string
  suppression_reason?: string
  suggestion?: string
  guide_filter?: {
    active?: boolean
    hidden_self_source?: boolean
    filtered_hit_count?: number
    guide_source_name?: string
  }
}

interface RefsPanelFilterOptions {
  activeSourcePath?: string
  activeSourceName?: string
}

function positiveNumber(input: unknown): number {
  const value = Number(input)
  return Number.isFinite(value) && value > 0 ? value : 0
}

function normalizeRefFocusText(input: unknown) {
  return String(input || '')
    .toLowerCase()
    .replace(/\.en\.md$/g, ' ')
    .replace(/\.md$|\.pdf$/g, ' ')
    .replace(/[_/\\]+/g, ' ')
    .replace(/[^a-z0-9\u4e00-\u9fff]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function normalizeSourcePathForMatch(input: unknown): string {
  return normalizeSourcePathForMatchShared(input)
}

function sourceDocumentIdentityKey(input: unknown): string {
  const normalized = normalizeSourcePathForMatch(input)
  if (!normalized) return ''
  return normalized
    .replace(/\.en\.md$/i, '')
    .replace(/\.md$/i, '')
    .replace(/\.pdf$/i, '')
    .replace(/\s+/g, ' ')
    .trim()
}

function normalizeSourceNameIdentity(input: unknown): string {
  const file = basenameFromSourcePath(input) || String(input || '').trim()
  return file
    .replace(/\.en\.md$/i, '')
    .replace(/\.md$/i, '')
    .replace(/\.pdf$/i, '')
    .replace(/[^a-z0-9\u4e00-\u9fff]+/gi, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase()
}

function hitSourcePath(hit: RefsPanelRefHit): string {
  const ui = hit.ui_meta || {}
  const readerOpen = (ui.reader_open && typeof ui.reader_open === 'object') ? ui.reader_open : {}
  return String(ui.source_path || readerOpen.sourcePath || hit.meta?.source_path || '').trim()
}

function hitSourceName(hit: RefsPanelRefHit): string {
  const ui = hit.ui_meta || {}
  const readerOpen = (ui.reader_open && typeof ui.reader_open === 'object') ? ui.reader_open : {}
  return String(ui.display_name || readerOpen.sourceName || hit.meta?.source_path || '').trim()
}

function sourcePathsReferToSameDocument(left: unknown, right: unknown): boolean {
  const leftNorm = normalizeSourcePathForMatch(left)
  const rightNorm = normalizeSourcePathForMatch(right)
  if (!leftNorm || !rightNorm) return false
  if (leftNorm === rightNorm) return true
  const leftHasDirectory = leftNorm.includes('/')
  const rightHasDirectory = rightNorm.includes('/')
  if (leftHasDirectory && rightHasDirectory) return false
  const leftName = normalizeSourceNameIdentity(leftNorm)
  const rightName = normalizeSourceNameIdentity(rightNorm)
  return Boolean(leftName && rightName && leftName === rightName)
}

function sourcePathsFormPdfMarkdownPair(left: unknown, right: unknown): boolean {
  const leftNorm = normalizeSourcePathForMatch(left)
  const rightNorm = normalizeSourcePathForMatch(right)
  if (!leftNorm || !rightNorm) return false
  const leftKind = leftNorm.endsWith('.pdf') ? 'pdf' : /(?:\.en)?\.md$/.test(leftNorm) ? 'markdown' : ''
  const rightKind = rightNorm.endsWith('.pdf') ? 'pdf' : /(?:\.en)?\.md$/.test(rightNorm) ? 'markdown' : ''
  if (new Set([leftKind, rightKind]).size !== 2 || !leftKind || !rightKind) return false
  const leftName = normalizeSourceNameIdentity(leftNorm)
  const rightName = normalizeSourceNameIdentity(rightNorm)
  if (!leftName || leftName !== rightName) return false
  const leftParent = leftNorm.split('/').slice(0, -1).join('/')
  const rightParent = rightNorm.split('/').slice(0, -1).join('/')
  if (leftParent && leftParent === rightParent) return true
  const markdownPath = leftKind === 'markdown' ? leftNorm : rightNorm
  const markdownParts = markdownPath.split('/').filter(Boolean)
  const markdownParentName = markdownParts.length >= 2 ? markdownParts[markdownParts.length - 2] : ''
  return normalizeSourceNameIdentity(markdownParentName) === leftName
}

function refHitDocumentKey(hit: RefsPanelRefHit, index: number): string {
  const sourcePath = hitSourcePath(hit)
  const pathKey = sourceDocumentIdentityKey(sourcePath)
  if (pathKey) return `path:${pathKey}`
  const nameKey = normalizeSourceNameIdentity(hitSourceName(hit))
  if (nameKey) return `name:${nameKey}`
  return `row:${index}`
}

function refHitScore(hit: RefsPanelRefHit): number {
  const uiScore = Number(hit.ui_meta?.score)
  if (Number.isFinite(uiScore)) return uiScore
  const hitScore = Number(hit.score)
  return Number.isFinite(hitScore) ? hitScore : 0
}

function preferRefHit(next: RefsPanelRefHit, current: RefsPanelRefHit): boolean {
  const nextReady = String(next.meta?.ref_pack_state || '').trim().toLowerCase() !== 'pending'
  const currentReady = String(current.meta?.ref_pack_state || '').trim().toLowerCase() !== 'pending'
  if (nextReady !== currentReady) return nextReady
  return refHitScore(next) > refHitScore(current) + 1e-6
}

function hitMatchesActiveSource(hit: RefsPanelRefHit, activeSourcePath?: string, activeSourceName?: string): boolean {
  const activePath = String(activeSourcePath || '').trim()
  const activeName = String(activeSourceName || '').trim()
  const sourcePath = hitSourcePath(hit)
  if (activePath && sourcePath) {
    if (sourcePathsReferToSameDocument(sourcePath, activePath)) return true
    if (!sourcePathsFormPdfMarkdownPair(sourcePath, activePath)) return false
    const activeNameKey = normalizeSourceNameIdentity(activeName || activePath)
    const hitNameKey = normalizeSourceNameIdentity(hitSourceName(hit) || sourcePath)
    return Boolean(activeNameKey && hitNameKey && activeNameKey === hitNameKey)
  }
  const activeNameKey = normalizeSourceNameIdentity(activeName || activePath)
  const hitNameKey = normalizeSourceNameIdentity(hitSourceName(hit) || sourcePath)
  return Boolean(activeNameKey && hitNameKey && activeNameKey === hitNameKey)
}

function promptFocusTerms(prompt: string) {
  const text = String(prompt || '').trim()
  if (!text) return [] as string[]
  const out: string[] = []
  const seen = new Set<string>()
  const push = (raw: string) => {
    const norm = normalizeRefFocusText(raw)
    if (!norm || norm.length < 3 || seen.has(norm)) return
    seen.add(norm)
    out.push(norm)
  }
  for (const m of text.matchAll(/[“"‘'`「『]([^“"‘'`」』]{2,80})[”"’'`」』]/g)) {
    push(String(m[1] || ''))
  }
  const stop = new Set([
    'the', 'and', 'for', 'with', 'from', 'into', 'using', 'about', 'where', 'which', 'what',
    'that', 'this', 'these', 'those', 'paper', 'papers', 'library', 'source', 'sources',
    'section', 'please', 'point', 'directly', 'most', 'does', 'do', 'did', 'discuss', 'discusses',
    'mentioned', 'mention', 'other', 'besides', 'find', 'show', 'explain',
  ])
  for (const m of text.matchAll(/\b[A-Za-z][A-Za-z0-9_-]{1,40}\b/g)) {
    const raw = String(m[0] || '').trim()
    const low = raw.toLowerCase()
    if (stop.has(low)) continue
    const hasSignal = /[A-Z]/.test(raw.slice(1)) || raw === raw.toUpperCase() || /\d/.test(raw) || raw.includes('-')
    if (!hasSignal) continue
    push(raw)
  }
  return out.slice(0, 8)
}

function promptNeedsStrictRefEvidence(prompt: string) {
  const low = String(prompt || '').toLowerCase()
  if (!low) return false
  const patterns = [
    'where is', 'where was', 'where are', 'discuss', 'mention', 'point me',
    'which paper', 'which papers', 'what other papers', 'besides this paper',
    '哪篇', '哪些论文', '提到', '哪里', '定位',
  ]
  return patterns.some((pattern) => low.includes(pattern))
}

function hitIdentityTerms(hit: RefsPanelRefHit) {
  const values = [
    hitSourceName(hit),
    hitSourcePath(hit),
  ]
  const out = new Set<string>()
  for (const raw of values) {
    const norm = normalizeRefFocusText(raw)
    if (!norm) continue
    out.add(norm)
    for (const token of norm.split(' ')) {
      if (token.length >= 3) out.add(token)
    }
  }
  return out
}

function hitSurfaceText(hit: RefsPanelRefHit) {
  const ui = hit.ui_meta || {}
  const readerOpen = (ui.reader_open && typeof ui.reader_open === 'object') ? ui.reader_open : {}
  const parts = [
    String(hit.text || ''),
    String(ui.heading_path || ''),
    String(ui.summary_line || ''),
    String(readerOpen.snippet || ''),
    String(readerOpen.highlightSnippet || ''),
  ]
  return normalizeRefFocusText(parts.filter(Boolean).join(' '))
}

function nonSourceFocusMatchCount(prompt: string, hit: RefsPanelRefHit) {
  const focusTerms = promptFocusTerms(prompt)
  if (!focusTerms.length) return 0
  const surface = hitSurfaceText(hit)
  if (!surface) return 0
  const identities = hitIdentityTerms(hit)
  let count = 0
  for (const term of focusTerms) {
    if (!surface.includes(term)) continue
    const isIdentity = Array.from(identities).some((ident) => term === ident || term.includes(ident) || ident.includes(term))
    if (!isIdentity) count += 1
  }
  return count
}

function looksNegativeReasonText(text: string) {
  const low = String(text || '').toLowerCase()
  if (!low) return false
  return [
    'not mentioned',
    'not discuss',
    'not discussed',
    'not stated',
    'no external paper matched',
    'no papers in your library',
    'cannot point',
    '未提及',
    '未提到',
    '没有提到',
    '没有命中',
    '无法定位',
    '不能指向',
  ].some((token) => low.includes(token))
}

function shouldSuppressRefHitCard(prompt: string, hit: RefsPanelRefHit) {
  if (!promptNeedsStrictRefEvidence(prompt)) return false
  const ui = hit.ui_meta || {}
  const why = String(ui.why_line || '').trim()
  const summary = String(ui.summary_line || '').trim()
  const focusTerms = promptFocusTerms(prompt)
  const nonSourceMatches = nonSourceFocusMatchCount(prompt, hit)
  if (focusTerms.length > 1 && nonSourceMatches <= 0) {
    return true
  }
  if (looksNegativeReasonText(why) && nonSourceMatches <= 0) {
    return true
  }
  if (looksNegativeReasonText(summary) && nonSourceMatches <= 0) {
    return true
  }
  return false
}

export function prepareRefsPanelHits(
  entry: RefsPanelRefEntry | undefined,
  opts?: RefsPanelFilterOptions,
): { hits: RefsPanelRefHit[]; suppressedHitCount: number; hiddenActiveSourceCount: number } {
  const prompt = String(entry?.prompt || '').trim()
  const displayState = String(entry?.display_state || '').trim().toLowerCase()
  const hasBackendDisplayState = Boolean(displayState)
  const rawHits = Array.isArray(entry?.hits) ? entry.hits : []
  const filteredByRelevance = hasBackendDisplayState
    ? rawHits
    : rawHits.filter((hit) => !shouldSuppressRefHitCard(prompt, hit))
  const activeSourcePath = String(opts?.activeSourcePath || '').trim()
  const activeSourceName = String(opts?.activeSourceName || '').trim()
  const hasPendingHit = rawHits.some((hit) => String(hit?.meta?.ref_pack_state || '').trim().toLowerCase() === 'pending')
  const hideActiveSource = Boolean(activeSourcePath || activeSourceName) && displayState !== 'pending' && !hasPendingHit
  const withoutActiveSource = hideActiveSource
    ? filteredByRelevance.filter((hit) => !hitMatchesActiveSource(hit, activeSourcePath, activeSourceName))
    : filteredByRelevance
  const hiddenActiveSourceCount = filteredByRelevance.length - withoutActiveSource.length
  const bestByDocument = new Map<string, { hit: RefsPanelRefHit; index: number }>()
  withoutActiveSource.forEach((hit, index) => {
    const key = refHitDocumentKey(hit, index)
    const current = bestByDocument.get(key)
    if (!current || preferRefHit(hit, current.hit)) {
      bestByDocument.set(key, { hit, index })
    }
  })
  const hits = Array.from(bestByDocument.values())
    .sort((a, b) => a.index - b.index)
    .map((item) => item.hit)
  return {
    hits,
    suppressedHitCount: Math.max(0, rawHits.length - filteredByRelevance.length),
    hiddenActiveSourceCount,
  }
}

export function hasRefsPanelContent(
  refs: Record<string, unknown>,
  msgId: number,
  opts?: RefsPanelFilterOptions,
): boolean {
  const entry = refs[String(msgId)] as RefsPanelRefEntry | undefined
  if (!entry) return false
  const displayState = String(entry.display_state || '').trim().toLowerCase()
  if (displayState === 'empty') return false
  if (displayState === 'suppressed') return false
  const { hits, suppressedHitCount, hiddenActiveSourceCount } = prepareRefsPanelHits(entry, opts)
  if (hits.length > 0) return true
  if (displayState === 'pending') return hiddenActiveSourceCount <= 0
  if (hiddenActiveSourceCount > 0 && String(opts?.activeSourcePath || opts?.activeSourceName || '').trim()) return false
  if (suppressedHitCount > 0) return false
  if (displayState === 'hidden_by_guide') return false
  const guideFilter = entry.guide_filter || {}
  const filteredCount = positiveNumber(guideFilter.filtered_hit_count)
  return Boolean(guideFilter.hidden_self_source || filteredCount > 0)
}
