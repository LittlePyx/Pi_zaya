import type {
  ResearchNoteRecord,
  ResearchNoteSourceLink,
  ResearchNoteSourceState,
} from '../api/chat'
import { mergeResearchNoteSourceLinks } from '../components/chat/researchNote'

const OUTLINE_SOURCE_LIMIT = 100
const OUTLINE_REFRESH_HISTORY_LIMIT = 120

export interface ResearchNoteOutlineSource {
  id: string
  title: string
  revision: number
  content_fingerprint?: string
  last_refreshed_at?: number
}

export interface ResearchNoteOutlineRefreshEvent {
  source_id: string
  from_revision: number
  to_revision: number
  refreshed_at: number
  replaced_manual_edits: boolean
}

interface ResearchNoteOutlineComposition {
  kind: 'research_note_outline'
  version: 1
  source_notes: ResearchNoteOutlineSource[]
  refresh_history?: ResearchNoteOutlineRefreshEvent[]
}

export interface ResearchNoteOutlineRefreshResult {
  contentMarkdown: string
  sourceState: ResearchNoteSourceState
  currentSection: string
  hadManualEdits: boolean
}

function noteLinks(note: ResearchNoteRecord): ResearchNoteSourceLink[] {
  return Array.isArray(note.source_state?.links)
    ? note.source_state.links.filter((link): link is ResearchNoteSourceLink => Boolean(link && typeof link === 'object'))
    : []
}

function sharedValue(values: Array<string | null | undefined>): string | null {
  if (values.length <= 0) return null
  const normalized = values.map(value => String(value || '').trim())
  return new Set(normalized).size === 1 ? normalized[0] || null : null
}

function outlineSection(note: Pick<ResearchNoteRecord, 'title' | 'content_markdown'>, index: number): string {
  const body = String(note.content_markdown || '').trim()
  return `## ${index + 1}. ${note.title}\n\n${body}`.trim()
}

interface SectionRange {
  start: number
  end: number
  body: string
}

function sectionBody(value: string): string {
  const lines = String(value || '').trim().split('\n')
  if (/^##\s+\d+\.\s+/.test(String(lines[0] || '').trim())) lines.shift()
  return lines.join('\n').trim()
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function exactOutlineHeadingIndex(
  markdown: string,
  source: ResearchNoteOutlineSource | undefined,
  sourceIndex: number,
): number {
  if (!source) return -1
  const heading = `## ${sourceIndex + 1}. ${source.title}`
  return new RegExp(`^${escapeRegExp(heading)}$`, 'm').exec(markdown)?.index ?? -1
}

function generatedOutlineHeadings(markdown: string): RegExpMatchArray[] {
  return Array.from(markdown.matchAll(/^##\s+\d+\.\s+.*$/gm)).filter(match => {
    if (match.index === undefined) return false
    const previousLine = markdown
      .slice(0, match.index)
      .trimEnd()
      .split(/\r?\n/)
      .at(-1)
      ?.trim()
    return previousLine === '---' || /^#\s+/.test(previousLine || '')
  })
}

function outlineSectionRange(
  markdown: string,
  sources: ResearchNoteOutlineSource[],
  sourceId: string,
): SectionRange | null {
  const sourceIndex = sources.findIndex(source => source.id === sourceId)
  if (sourceIndex < 0) return null
  const generatedHeadings = generatedOutlineHeadings(markdown)
  const exactStart = exactOutlineHeadingIndex(markdown, sources[sourceIndex], sourceIndex)
  const headingStart = exactStart >= 0 ? exactStart : generatedHeadings[sourceIndex]?.index
  if (headingStart === undefined || headingStart < 0) return null
  const exactNext = exactOutlineHeadingIndex(markdown, sources[sourceIndex + 1], sourceIndex + 1)
  const fallbackNext = generatedHeadings[sourceIndex + 1]?.index
  const rangeEnd = sourceIndex >= sources.length - 1
    ? markdown.length
    : exactNext > headingStart
      ? exactNext
      : fallbackNext ?? markdown.length
  let contentEnd = rangeEnd
  const separator = markdown.lastIndexOf('\n\n---\n\n', rangeEnd)
  if (separator > headingStart) contentEnd = separator
  return {
    start: headingStart,
    end: contentEnd,
    body: sectionBody(markdown.slice(headingStart, contentEnd)),
  }
}

function parseComposition(sourceState: ResearchNoteSourceState | null | undefined): ResearchNoteOutlineComposition | null {
  const raw = sourceState?.composition
  if (!raw || typeof raw !== 'object') return null
  const composition = raw as Partial<ResearchNoteOutlineComposition>
  if (composition.kind !== 'research_note_outline' || composition.version !== 1) return null
  if (!Array.isArray(composition.source_notes)) return null
  return composition as ResearchNoteOutlineComposition
}

export function researchNoteContentFingerprint(value: string): string {
  const normalized = String(value || '').trim().replace(/\r\n/g, '\n')
  let hash = 0x811c9dc5
  for (let index = 0; index < normalized.length; index += 1) {
    hash ^= normalized.charCodeAt(index)
    hash = Math.imul(hash, 0x01000193)
  }
  return `fnv1a32:${(hash >>> 0).toString(16).padStart(8, '0')}:${normalized.length}`
}

export function buildResearchNoteOutline(title: string, notes: ResearchNoteRecord[]): string {
  const sections = notes.map((note, index) => outlineSection(note, index))
  return `# ${String(title || '').trim()}\n\n${sections.join('\n\n---\n\n')}`.trim()
}

export function researchNoteOutlineProjectId(notes: ResearchNoteRecord[]): string | null {
  return sharedValue(notes.map(note => note.project_id))
}

export function researchNoteOutlineHasMixedProjects(notes: ResearchNoteRecord[]): boolean {
  return new Set(notes.map(note => String(note.project_id || '').trim())).size > 1
}

export function researchNoteOutlineConversationId(notes: ResearchNoteRecord[]): string | null {
  return sharedValue(notes.map(note => note.source_conv_id))
}

export function researchNoteOutlineTags(notes: ResearchNoteRecord[]): string[] {
  const seen = new Set<string>()
  const tags: string[] = []
  for (const note of notes) {
    for (const rawTag of note.tags || []) {
      const tag = String(rawTag || '').trim()
      const identity = tag.toLocaleLowerCase()
      if (!tag || seen.has(identity)) continue
      seen.add(identity)
      tags.push(tag)
      if (tags.length >= 24) return tags
    }
  }
  return tags
}

export function researchNoteOutlineSourceState(notes: ResearchNoteRecord[]): ResearchNoteSourceState {
  const composition: ResearchNoteOutlineComposition = {
    kind: 'research_note_outline',
    version: 1,
    source_notes: notes.slice(0, OUTLINE_SOURCE_LIMIT).map(note => ({
      id: note.id,
      title: note.title,
      revision: note.revision,
      content_fingerprint: researchNoteContentFingerprint(note.content_markdown),
    })),
    refresh_history: [],
  }
  return {
    version: 1,
    links: mergeResearchNoteSourceLinks(...notes.map(noteLinks)),
    composition,
  }
}

export function researchNoteOutlineSourcesFromState(
  sourceState: ResearchNoteSourceState | null | undefined,
): ResearchNoteOutlineSource[] {
  const composition = parseComposition(sourceState)
  if (!composition) return []
  return composition.source_notes.flatMap((item) => {
    if (!item || typeof item !== 'object') return []
    const source = item as Partial<ResearchNoteOutlineSource>
    const id = String(source.id || '').trim()
    if (!id) return []
    const revision = Number(source.revision || 0)
    const refreshedAt = Number(source.last_refreshed_at || 0)
    return [{
      id,
      title: String(source.title || '').trim() || id,
      revision: Number.isInteger(revision) && revision > 0 ? revision : 1,
      content_fingerprint: String(source.content_fingerprint || '').trim() || undefined,
      last_refreshed_at: Number.isFinite(refreshedAt) && refreshedAt > 0 ? refreshedAt : undefined,
    }]
  }).slice(0, OUTLINE_SOURCE_LIMIT)
}

export function researchNoteOutlineSources(
  note: ResearchNoteRecord | null | undefined,
): ResearchNoteOutlineSource[] {
  return researchNoteOutlineSourcesFromState(note?.source_state)
}

export function researchNoteOutlineSectionContent(
  markdown: string,
  sources: ResearchNoteOutlineSource[],
  sourceId: string,
): string | null {
  return outlineSectionRange(markdown, sources, sourceId)?.body ?? null
}

export function researchNoteOutlineSectionHasManualEdits(
  markdown: string,
  sources: ResearchNoteOutlineSource[],
  sourceId: string,
): boolean {
  const source = sources.find(item => item.id === sourceId)
  const current = researchNoteOutlineSectionContent(markdown, sources, sourceId)
  if (!source || current === null || !source.content_fingerprint) return true
  return researchNoteContentFingerprint(current) !== source.content_fingerprint
}

export function refreshResearchNoteOutlineSource(
  markdown: string,
  sourceState: ResearchNoteSourceState,
  latest: ResearchNoteRecord,
  refreshedAt = Math.floor(Date.now() / 1000),
): ResearchNoteOutlineRefreshResult | null {
  const composition = parseComposition(sourceState)
  if (!composition) return null
  const sources = researchNoteOutlineSourcesFromState(sourceState)
  const sourceIndex = sources.findIndex(source => source.id === latest.id)
  const range = outlineSectionRange(markdown, sources, latest.id)
  if (sourceIndex < 0 || !range) return null
  const source = sources[sourceIndex]
  const hadManualEdits = researchNoteOutlineSectionHasManualEdits(markdown, sources, latest.id)
  const replacement = `\n${outlineSection(latest, sourceIndex)}\n`
  const contentMarkdown = `${markdown.slice(0, range.start)}${replacement}${markdown.slice(range.end)}`
    .trim()
  const sourceNotes = sources.map(item => item.id === latest.id
    ? {
        ...item,
        title: latest.title,
        revision: latest.revision,
        content_fingerprint: researchNoteContentFingerprint(latest.content_markdown),
        last_refreshed_at: refreshedAt,
      }
    : item)
  const rawHistory = Array.isArray(composition.refresh_history) ? composition.refresh_history : []
  const refreshHistory = [
    ...rawHistory,
    {
      source_id: latest.id,
      from_revision: source.revision,
      to_revision: latest.revision,
      refreshed_at: refreshedAt,
      replaced_manual_edits: hadManualEdits,
    },
  ].slice(-OUTLINE_REFRESH_HISTORY_LIMIT)
  return {
    contentMarkdown,
    currentSection: range.body,
    hadManualEdits,
    sourceState: {
      ...sourceState,
      version: 1,
      links: mergeResearchNoteSourceLinks(
        Array.isArray(sourceState.links) ? sourceState.links : [],
        noteLinks(latest),
      ),
      composition: {
        ...composition,
        source_notes: sourceNotes,
        refresh_history: refreshHistory,
      },
    },
  }
}
