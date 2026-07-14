type SummaryRecord = Record<string, unknown>

export type SystemBSummaryIdentityMode = 'embedded' | 'metadata'
export type SystemBSummarySourceKind = 'article' | 'context' | 'unknown'

export interface SystemBArticleSummaryDecision {
  visible: boolean
  isSystemB: boolean
  line: string
  source: string
  provider: string
  quality: SummaryRecord | null
  identityTitle: string
  identityDoi: string
  reason: string
}

const SUMMARY_SOURCE_KINDS: Record<string, Exclude<SystemBSummarySourceKind, 'unknown'>> = {
  abstract: 'article',
  fulltext: 'article',
  navigation: 'article',
  exact_anchor: 'article',
  section_intent_rescue: 'article',
  doc_list_seed: 'article',
  doc_list_prompt_aligned: 'article',
  answer_context: 'context',
  answer_reference_mention: 'context',
  citation_context: 'context',
  citation_card: 'context',
  citation_card_view: 'context',
  metadata: 'context',
  reference_primary_evidence: 'context',
  references_panel_hit: 'context',
  reader_occurrence: 'context',
  reader_reference_link: 'context',
  reader_references: 'context',
  retrieval_hit: 'context',
  source_markdown: 'context',
}

const LOCAL_ARTICLE_SUMMARY_PROVIDERS = new Set(['local_markdown'])
const LOCAL_ARTICLE_SUMMARY_GENERATIONS = new Set(['extractive_local_markdown'])

const LEGACY_ALIGNMENT_STOP_WORDS = new Set([
  'about', 'after', 'among', 'and', 'architecture', 'article', 'based', 'between', 'from',
  'into', 'method', 'methods', 'paper', 'proposes', 'study', 'system', 'systems', 'that',
  'the', 'their', 'this', 'through', 'using', 'via', 'with', 'work',
])

function recordValue(value: unknown): SummaryRecord {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as SummaryRecord
    : {}
}

function firstText(record: SummaryRecord, ...keys: string[]): string {
  for (const key of keys) {
    const value = String(record[key] || '').trim()
    if (value) return value
  }
  return ''
}

function boolValue(record: SummaryRecord, ...keys: string[]): boolean {
  return keys.some((key) => record[key] === true)
}

export function systemBSummarySourceKind(source: unknown): SystemBSummarySourceKind {
  const key = String(source || '').trim().toLowerCase()
  return SUMMARY_SOURCE_KINDS[key] || 'unknown'
}

export function isSystemBArticleSummarySource(source: unknown): boolean {
  return systemBSummarySourceKind(source) === 'article'
}

export function isSystemBContextSummarySource(source: unknown): boolean {
  return systemBSummarySourceKind(source) === 'context'
}

export function isSystemBReferenceSummaryTarget(value: unknown): boolean {
  const record = recordValue(value)
  const route = firstText(record, 'citationRoute', 'citation_route').toLowerCase()
  const shelfKind = firstText(record, 'shelfItemKind', 'shelf_item_kind').toLowerCase()
  return Boolean(
    boolValue(record, 'isInpaper', 'is_inpaper')
    || route === 'system_b'
    || shelfKind === 'reference',
  )
}

export function normalizeSystemBSummaryDoi(value: unknown): string {
  return String(value || '')
    .trim()
    .replace(/^https?:\/\/(?:dx\.)?doi\.org\//i, '')
    .replace(/^doi\s*:\s*/i, '')
    .replace(/[\s.,;:]+$/g, '')
    .toLowerCase()
}

function normalizedText(value: unknown): string {
  return String(value || '')
    .normalize('NFKD')
    .toLowerCase()
    .replace(/[^a-z0-9\u4e00-\u9fff]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function comparableTokens(value: unknown, options?: { dropLegacyStopWords?: boolean }): string[] {
  const tokens = normalizedText(value).match(/[a-z0-9]+|[\u4e00-\u9fff]+/g) || []
  return tokens.filter((token) => {
    if (/^\d+$/.test(token)) return false
    if (token.length <= 2 && !/[\u4e00-\u9fff]/.test(token)) return false
    if (options?.dropLegacyStopWords && LEGACY_ALIGNMENT_STOP_WORDS.has(token)) return false
    return true
  })
}

export function systemBSummaryTitlesMatch(left: unknown, right: unknown): boolean {
  const a = normalizedText(left)
  const b = normalizedText(right)
  if (!a || !b) return false
  if (a === b) return true
  if (a.length >= 28 && b.includes(a)) return true
  if (b.length >= 28 && a.includes(b)) return true
  const aTokens = new Set(comparableTokens(a))
  const bTokens = new Set(comparableTokens(b))
  if (aTokens.size < 3 || bTokens.size < 3) return false
  let overlap = 0
  for (const token of aTokens) {
    if (bTokens.has(token)) overlap += 1
  }
  return overlap >= 3 && overlap / Math.min(aTokens.size, bTokens.size) >= 0.72
}

function legacySummaryAlignsWithTitle(title: string, summary: string, doi: string): boolean {
  const normalizedSummary = normalizedText(summary)
  const normalizedTitle = normalizedText(title)
  if (!normalizedSummary || !normalizedTitle) return false
  if (normalizedTitle.length >= 28 && normalizedSummary.includes(normalizedTitle)) return true
  if (doi && normalizeSystemBSummaryDoi(summary).includes(doi)) return true

  const titleTokens = new Set(comparableTokens(title, { dropLegacyStopWords: true }))
  const summaryTokens = new Set(comparableTokens(summary))
  if (titleTokens.size < 3 || summaryTokens.size < 3) return false
  let overlap = 0
  for (const token of titleTokens) {
    if (summaryTokens.has(token)) overlap += 1
  }
  return overlap >= 3 && (overlap >= 5 || overlap / titleTokens.size >= 0.45)
}

function targetIdentity(record: SummaryRecord): { title: string; doi: string } {
  return {
    title: firstText(record, 'title', 'cardTitle', 'card_title', 'main'),
    doi: normalizeSystemBSummaryDoi(firstText(record, 'doi', 'doiUrl', 'doi_url')),
  }
}

function candidateIdentity(
  candidate: SummaryRecord,
  quality: SummaryRecord,
  mode: SystemBSummaryIdentityMode,
): { title: string; doi: string; explicit: boolean } {
  if (mode === 'embedded') {
    // Embedded historical cards can carry a stale summary bundle alongside a
    // newer top-level Crossref/library match.  Treat that match as the
    // authoritative identity instead of falling back to loose title/summary
    // token overlap.
    const externalTitle = firstText(candidate, 'external_title', 'externalTitle')
    const externalDoi = firstText(candidate, 'external_doi', 'externalDoi', 'external_doi_url', 'externalDoiUrl')
    if (externalTitle || externalDoi) {
      return {
        title: externalTitle,
        doi: normalizeSystemBSummaryDoi(externalDoi),
        explicit: true,
      }
    }
    const libraryTitle = firstText(candidate, 'library_match_title', 'libraryMatchTitle')
    const libraryDoi = firstText(candidate, 'library_match_doi', 'libraryMatchDoi')
    if (libraryTitle || libraryDoi) {
      return {
        title: libraryTitle,
        doi: normalizeSystemBSummaryDoi(libraryDoi),
        explicit: true,
      }
    }
  }
  const nestedIdentity = recordValue(quality.identity)
  const explicitTitle = (
    firstText(quality, 'identity_title', 'identityTitle', 'source_title', 'sourceTitle')
    || firstText(nestedIdentity, 'title')
  )
  const explicitDoi = (
    firstText(quality, 'identity_doi', 'identityDoi', 'source_doi', 'sourceDoi')
    || firstText(nestedIdentity, 'doi')
  )
  if (explicitTitle || explicitDoi) {
    return {
      title: explicitTitle,
      doi: normalizeSystemBSummaryDoi(explicitDoi),
      explicit: true,
    }
  }
  if (mode !== 'metadata') return { title: '', doi: '', explicit: false }

  const provider = (
    firstText(candidate, 'summary_provider', 'summaryProvider')
    || firstText(quality, 'provider')
  ).toLowerCase()
  const generation = (
    firstText(candidate, 'summary_generation', 'summaryGeneration')
    || firstText(quality, 'generation')
  ).toLowerCase()
  const isLocalSummary = (
    LOCAL_ARTICLE_SUMMARY_PROVIDERS.has(provider)
    || LOCAL_ARTICLE_SUMMARY_GENERATIONS.has(generation)
  )
  const title = isLocalSummary
    ? firstText(candidate, 'library_match_title', 'libraryMatchTitle')
    : firstText(
      candidate,
      'external_title',
      'externalTitle',
      'library_match_title',
      'libraryMatchTitle',
      'title',
    )
  const doi = isLocalSummary
    ? firstText(candidate, 'library_match_doi', 'libraryMatchDoi')
    : firstText(
      candidate,
      'external_doi',
      'externalDoi',
      'library_match_doi',
      'libraryMatchDoi',
      'doi',
      'doi_url',
      'doiUrl',
    )
  return {
    title,
    doi: normalizeSystemBSummaryDoi(doi),
    explicit: false,
  }
}

export function resolveSystemBArticleSummary(
  targetValue: unknown,
  candidateValue: unknown = targetValue,
  options: { identityMode?: SystemBSummaryIdentityMode; forceSystemB?: boolean } = {},
): SystemBArticleSummaryDecision {
  const target = recordValue(targetValue)
  const candidate = recordValue(candidateValue)
  const line = firstText(candidate, 'summaryLine', 'summary_line')
  const qualityRecord = recordValue(candidate.summaryQuality || candidate.summary_quality)
  const quality = Object.keys(qualityRecord).length > 0 ? qualityRecord : null
  const source = (
    firstText(candidate, 'summarySource', 'summary_source')
    || firstText(qualityRecord, 'source')
  ).toLowerCase()
  const provider = (
    firstText(candidate, 'summaryProvider', 'summary_provider')
    || firstText(qualityRecord, 'provider')
  )
  const isSystemB = Boolean(options.forceSystemB || isSystemBReferenceSummaryTarget(target))
  if (!line) {
    return { visible: false, isSystemB, line: '', source, provider, quality, identityTitle: '', identityDoi: '', reason: 'missing_summary' }
  }
  if (!isSystemBArticleSummarySource(source)) {
    return { visible: false, isSystemB, line, source, provider, quality, identityTitle: '', identityDoi: '', reason: 'not_article_source' }
  }
  if (!isSystemB) {
    return { visible: true, isSystemB, line, source, provider, quality, identityTitle: '', identityDoi: '', reason: 'not_system_b' }
  }

  const targetId = targetIdentity(target)
  const candidateId = candidateIdentity(candidate, qualityRecord, options.identityMode || 'embedded')
  const doiComparable = Boolean(targetId.doi && candidateId.doi)
  const doiMatches = doiComparable && targetId.doi === candidateId.doi
  if (doiComparable && !doiMatches) {
    return {
      visible: false,
      isSystemB,
      line,
      source,
      provider,
      quality,
      identityTitle: candidateId.title,
      identityDoi: candidateId.doi,
      reason: 'identity_doi_mismatch',
    }
  }
  const titleComparable = Boolean(targetId.title && candidateId.title)
  const titleMatches = titleComparable && systemBSummaryTitlesMatch(targetId.title, candidateId.title)
  if (titleComparable && !titleMatches && !doiMatches) {
    return {
      visible: false,
      isSystemB,
      line,
      source,
      provider,
      quality,
      identityTitle: candidateId.title,
      identityDoi: candidateId.doi,
      reason: 'identity_title_mismatch',
    }
  }

  const verifiedByIdentity = doiMatches || titleMatches
  const verifiedByLegacyAlignment = !doiComparable && !titleComparable
    && legacySummaryAlignsWithTitle(targetId.title, line, targetId.doi)
  if (!verifiedByIdentity && !verifiedByLegacyAlignment) {
    return {
      visible: false,
      isSystemB,
      line,
      source,
      provider,
      quality,
      identityTitle: candidateId.title,
      identityDoi: candidateId.doi,
      reason: candidateId.explicit ? 'identity_unverifiable' : 'legacy_alignment_failed',
    }
  }

  const identityTitle = candidateId.title || targetId.title
  const identityDoi = candidateId.doi || targetId.doi
  return {
    visible: true,
    isSystemB,
    line,
    source,
    provider,
    quality: {
      ...(quality || {}),
      identity_title: identityTitle,
      identity_doi: identityDoi,
      identity_basis: verifiedByLegacyAlignment ? 'legacy_title_summary_alignment' : 'metadata_identity',
    },
    identityTitle,
    identityDoi,
    reason: verifiedByLegacyAlignment ? 'legacy_title_summary_alignment' : 'identity_match',
  }
}
