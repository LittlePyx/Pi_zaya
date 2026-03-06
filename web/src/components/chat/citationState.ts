export interface CiteDetail {
  num: number
  anchor: string
  sourceName: string
  sourcePath: string
  raw: string
  citeFmt: string
  title: string
  authors: string
  venue: string
  year: string
  volume: string
  issue: string
  pages: string
  doi: string
  doiUrl: string
  citationCount: number
  citationSource: string
  venueKind: string
  venueVerifiedBy: string
  openalexVenue: string
  journalIf: string
  journalQuartile: string
  journalIfSource: string
  conferenceTier: string
  conferenceRankSource: string
  conferenceCcf: string
  conferenceCcfSource: string
  conferenceName: string
  conferenceAcronym: string
  bibliometricsChecked: boolean
}

export interface CiteShelfItem extends CiteDetail {
  key: string
  main: string
}

function asText(value: unknown): string {
  if (typeof value === 'string') return value.trim()
  if (typeof value === 'number' && Number.isFinite(value)) return String(value)
  return ''
}

function asNumber(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

function pickText(rec: Record<string, unknown>, ...keys: string[]): string {
  for (const key of keys) {
    const value = asText(rec[key])
    if (value) return value
  }
  return ''
}

function pickNumber(rec: Record<string, unknown>, ...keys: string[]): number {
  for (const key of keys) {
    const value = asNumber(rec[key])
    if (value) return value
  }
  return 0
}

function isWeakField(key: string, value: string): boolean {
  const s = String(value || '').trim()
  if (!s) return true
  if (key === 'title') {
    if (s.length <= 4 || (s.match(/[A-Za-z0-9\u4e00-\u9fff]+/g)?.length || 0) <= 1) return true
    if (/^[A-Z][A-Za-z'`-]+,\s*(?:[A-Z]\.?)(?:\s*[A-Z]\.?)?$/i.test(s)) return true
    if (/^[A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`-]+){0,2},\s*(?:[A-Z]\.?\s*){1,3}$/i.test(s)) return true
    if (/\bet\s+al\.?$/i.test(s) && (s.match(/[A-Za-z\u4e00-\u9fff]+/g)?.length || 0) <= 4) return true
  }
  if (key === 'authors') return s.length <= 3 || (s.match(/[A-Za-z\u4e00-\u9fff]+/g)?.length || 0) <= 1
  if (key === 'venue') return s.length <= 1
  return false
}

export function normalizeCiteDetail(value: unknown): CiteDetail | null {
  if (!value || typeof value !== 'object') return null
  const rec = value as Record<string, unknown>
  const anchor = pickText(rec, 'anchor')
  if (!anchor) return null
  return {
    num: pickNumber(rec, 'num'),
    anchor,
    sourceName: pickText(rec, 'source_name', 'sourceName'),
    sourcePath: pickText(rec, 'source_path', 'sourcePath'),
    raw: pickText(rec, 'raw'),
    citeFmt: pickText(rec, 'cite_fmt', 'citeFmt'),
    title: pickText(rec, 'title'),
    authors: pickText(rec, 'authors'),
    venue: pickText(rec, 'venue'),
    year: pickText(rec, 'year'),
    volume: pickText(rec, 'volume'),
    issue: pickText(rec, 'issue'),
    pages: pickText(rec, 'pages'),
    doi: pickText(rec, 'doi'),
    doiUrl: pickText(rec, 'doi_url', 'doiUrl'),
    citationCount: pickNumber(rec, 'citation_count', 'citationCount'),
    citationSource: pickText(rec, 'citation_source', 'citationSource'),
    venueKind: pickText(rec, 'venue_kind', 'venueKind'),
    venueVerifiedBy: pickText(rec, 'venue_verified_by', 'venueVerifiedBy'),
    openalexVenue: pickText(rec, 'openalex_venue', 'openalexVenue'),
    journalIf: pickText(rec, 'journal_if', 'journalIf'),
    journalQuartile: pickText(rec, 'journal_quartile', 'journalQuartile'),
    journalIfSource: pickText(rec, 'journal_if_source', 'journalIfSource'),
    conferenceTier: pickText(rec, 'conference_tier', 'conferenceTier'),
    conferenceRankSource: pickText(rec, 'conference_rank_source', 'conferenceRankSource'),
    conferenceCcf: pickText(rec, 'conference_ccf', 'conferenceCcf'),
    conferenceCcfSource: pickText(rec, 'conference_ccf_source', 'conferenceCcfSource'),
    conferenceName: pickText(rec, 'conference_name', 'conferenceName'),
    conferenceAcronym: pickText(rec, 'conference_acronym', 'conferenceAcronym'),
    bibliometricsChecked: Boolean(rec.bibliometrics_checked ?? rec.bibliometricsChecked),
  }
}

export function citationMain(detail: CiteDetail): string {
  const stripLeadLabel = (value: string) =>
    String(value || '')
      .replace(/^\s*(?:\[\s*\d{1,4}\s*\]\s*){1,3}/, '')
      .replace(/^\s*\d{1,4}\s*[.)]\s*/, '')
      .trim()

  if (detail.citeFmt) return stripLeadLabel(detail.citeFmt)
  const parts = [detail.authors, detail.title, detail.venue, detail.year].filter(Boolean)
  if (parts.length > 0) return parts.join('. ')
  return stripLeadLabel(detail.raw) || `[${detail.num || '?'}]`
}

export function toShelfItem(detail: CiteDetail): CiteShelfItem {
  const main = citationMain(detail)
  const baseKey = `${detail.anchor}|${detail.sourceName || detail.sourcePath}|${detail.num}`
  return {
    ...detail,
    key: baseKey,
    main,
  }
}

export function mergeCiteMeta(detail: CiteDetail, meta: Record<string, unknown>): CiteDetail {
  const merged: Record<string, unknown> = { ...detail }
  const overwriteKeys = new Set([
    'doi',
    'doi_url',
    'citation_count',
    'citation_source',
    'journal_if',
    'journal_quartile',
    'journal_if_source',
    'conference_tier',
    'conference_rank_source',
    'conference_ccf',
    'conference_ccf_source',
    'bibliometrics_checked',
    'venue_kind',
    'venue_verified_by',
    'openalex_venue',
    'conference_name',
    'conference_acronym',
  ])
  for (const [key, rawValue] of Object.entries(meta || {})) {
    if (rawValue === null || rawValue === undefined || rawValue === '' || (Array.isArray(rawValue) && rawValue.length === 0)) {
      continue
    }
    if (overwriteKeys.has(key)) {
      merged[key] = rawValue
      continue
    }
    if (typeof rawValue !== 'string') {
      merged[key] = rawValue
      continue
    }
    const current = String(merged[key] || '').trim()
    const incoming = rawValue.trim()
    if (!current) {
      merged[key] = incoming
      continue
    }
    const currentWeak = isWeakField(key, current)
    const incomingWeak = isWeakField(key, incoming)
    if (currentWeak && !incomingWeak) {
      merged[key] = incoming
      continue
    }
    if (!currentWeak && incomingWeak) continue
    if (incoming.length > current.length + 12) {
      merged[key] = incoming
    }
  }
  return normalizeCiteDetail(merged) || detail
}

export function citeMetricSummary(detail: CiteDetail): string[] {
  const items: string[] = []
  if (detail.citationCount > 0) {
    items.push(`被引 ${detail.citationCount}${detail.citationSource ? ` (${detail.citationSource})` : ''}`)
  }
  if (detail.venueKind === 'conference') {
    const confLabel = detail.conferenceAcronym || detail.conferenceName || detail.venue
    if (confLabel) items.push(`会议 ${confLabel}`)
    if (detail.year) items.push(`年份 ${detail.year}`)
    if (detail.conferenceTier) {
      items.push(`CORE ${detail.conferenceTier}${detail.conferenceRankSource ? ` (${detail.conferenceRankSource})` : ''}`)
    }
    if (detail.conferenceCcf) {
      items.push(`CCF ${detail.conferenceCcf}${detail.conferenceCcfSource ? ` (${detail.conferenceCcfSource})` : ''}`)
    }
  } else {
    if (detail.venue) items.push(`期刊 ${detail.venue}`)
    if (detail.year) items.push(`年份 ${detail.year}`)
  }
  if (detail.journalIf) items.push(`IF ${detail.journalIf}`)
  if (detail.journalQuartile) items.push(`JCR ${detail.journalQuartile}`)
  if (detail.journalIfSource) items.push(`IF来源 ${detail.journalIfSource}`)
  return items
}

export function shelfStorageKey(convId?: string | null): string {
  return `kb_cite_shelf:${String(convId || 'default')}`
}

function baseName(path: string): string {
  const text = String(path || '').trim()
  if (!text) return ''
  const parts = text.split(/[\\/]/)
  return String(parts[parts.length - 1] || '').trim()
}

function stripKnownExt(name: string): string {
  return String(name || '')
    .replace(/\.en\.md$/i, '')
    .replace(/\.md$/i, '')
    .replace(/\.pdf$/i, '')
    .trim()
}

function titleFromSourceName(sourceName: string, sourcePath: string): string {
  const raw = stripKnownExt(sourceName || baseName(sourcePath))
  if (!raw) return ''
  let candidate = raw.replace(/_/g, ' ').replace(/\s+/g, ' ').trim()
  const m = candidate.match(/^[A-Za-z]{2,20}-\d{4}-(.+)$/)
  if (m && m[1]) candidate = String(m[1]).trim()
  const m2 = candidate.match(/^\d{4}[-_ ]+(.+)$/)
  if (m2 && m2[1]) candidate = String(m2[1]).trim()
  return isWeakField('title', candidate) ? '' : candidate
}

function looksCitationLine(text: string): boolean {
  const s = String(text || '').replace(/\*+/g, '').replace(/\s+/g, ' ').trim()
  if (s.length < 24) return false
  const hasYear = /\b(?:19|20)\d{2}\b/.test(s)
  const hasVolumePagesTail = /,\s*\d{1,4}\s*,\s*\d{1,6}\.?$/.test(s)
  const hasVenueToken = /\b(?:Nat\.?|IEEE|ACM|Opt\.?|Phys\.?|Commun\.?|Journal|Proceedings|CVPR|ICCV|ICML|NeurIPS)\b/i.test(s)
  const startsLikeAuthors = /^(?:[A-Z][A-Za-z'`-]+,\s*(?:[A-Z]\.\s*){1,3})(?:,\s*[A-Z][A-Za-z'`-]+,\s*(?:[A-Z]\.\s*){1,3})*/.test(s)
  if (hasYear && hasVolumePagesTail) return true
  if (startsLikeAuthors && hasYear && hasVenueToken) return true
  return false
}

export function citationSourceLabel(detail: CiteDetail): string {
  return detail.sourceName || baseName(detail.sourcePath)
}

export function citationDisplay(detail: CiteDetail) {
  const main = (() => {
    const title = String(detail.title || '').trim()
    if (!isWeakField('title', title)) return title
    const sourceDerived = titleFromSourceName(detail.sourceName, detail.sourcePath)
    const fallbackMain = citationMain(detail)
    if (sourceDerived && (isWeakField('title', fallbackMain) || looksCitationLine(fallbackMain))) {
      return sourceDerived
    }
    return fallbackMain
  })()
  const authors = isWeakField('authors', detail.authors) ? '' : String(detail.authors || '').trim()
  const venue = isWeakField('venue', detail.venue) ? '' : String(detail.venue || '').trim()
  const source = citationSourceLabel(detail)
  const venueYear = [venue, String(detail.year || '').trim()].filter(Boolean).join(' | ')
  return {
    main,
    authors,
    source,
    venue,
    venueYear,
  }
}

export function buildCiteDetailFromMeta(
  meta: Record<string, unknown> | null | undefined,
  fallback: {
    sourceName?: string
    sourcePath?: string
    num?: number
    anchor?: string
  } = {},
): CiteDetail | null {
  const rec: Record<string, unknown> = { ...(meta || {}) }
  if (!pickText(rec, 'anchor')) {
    rec.anchor = fallback.anchor || `source:${fallback.sourcePath || fallback.sourceName || 'unknown'}`
  }
  if (!pickNumber(rec, 'num') && fallback.num) {
    rec.num = fallback.num
  }
  if (!pickText(rec, 'source_name', 'sourceName') && fallback.sourceName) {
    rec.source_name = fallback.sourceName
  }
  if (!pickText(rec, 'source_path', 'sourcePath') && fallback.sourcePath) {
    rec.source_path = fallback.sourcePath
  }
  return normalizeCiteDetail(rec)
}

export function citationFormats(detail: CiteDetail): { gbt: string; bibtex: string } {
  const title = asText(detail.title) || citationMain(detail)
  const authors = asText(detail.authors) || '[Unknown Authors]'
  const venue =
    asText(detail.conferenceName) ||
    asText(detail.conferenceAcronym) ||
    asText(detail.venue) ||
    'Unknown Venue'
  const year = asText(detail.year) || '20xx'
  const volume = asText(detail.volume)
  const issue = asText(detail.issue)
  const pages = asText(detail.pages)
  const doi = asText(detail.doi)
  const entryType = detail.venueKind === 'conference' ? 'inproceedings' : 'article'
  const gbtKind = detail.venueKind === 'conference' ? '[C]' : '[J]'

  let suffix = `, ${year}`
  if (volume) suffix += `, ${volume}`
  if (issue) suffix += `(${issue})`
  if (pages) suffix += `: ${pages}`
  const gbt = `${authors}. ${title} ${gbtKind}. ${venue}${suffix}.`

  const keyBase = title.toLowerCase().replace(/[^a-z0-9]+/g, '_').slice(0, 24) || 'reference'
  const venueField = detail.venueKind === 'conference' ? 'booktitle' : 'journal'
  const bibtex = `@${entryType}{ref_${year}_${keyBase},
  title={${title}},
  author={${authors}},
  ${venueField}={${venue}},
  year={${year}},${volume ? `\n  volume={${volume}},` : ''}${issue ? `\n  number={${issue}},` : ''}${pages ? `\n  pages={${pages}},` : ''}${doi ? `\n  doi={${doi}},` : ''}
}`

  return { gbt, bibtex }
}
