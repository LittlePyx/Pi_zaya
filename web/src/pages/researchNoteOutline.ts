import type {
  ResearchNoteRecord,
  ResearchNoteSourceLink,
  ResearchNoteSourceState,
} from '../api/chat'
import { mergeResearchNoteSourceLinks } from '../components/chat/researchNote'

export interface ResearchNoteOutlineSource {
  id: string
  title: string
  revision: number
}

interface ResearchNoteOutlineComposition {
  kind: 'research_note_outline'
  version: 1
  source_notes: ResearchNoteOutlineSource[]
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

export function buildResearchNoteOutline(title: string, notes: ResearchNoteRecord[]): string {
  const sections = notes.map((note, index) => {
    const body = String(note.content_markdown || '').trim()
    return `## ${index + 1}. ${note.title}\n\n${body}`
  })
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
    source_notes: notes.map(note => ({
      id: note.id,
      title: note.title,
      revision: note.revision,
    })),
  }
  return {
    version: 1,
    links: mergeResearchNoteSourceLinks(...notes.map(noteLinks)),
    composition,
  }
}

export function researchNoteOutlineSources(
  note: ResearchNoteRecord | null | undefined,
): ResearchNoteOutlineSource[] {
  const raw = note?.source_state?.composition
  if (!raw || typeof raw !== 'object') return []
  const composition = raw as Partial<ResearchNoteOutlineComposition>
  if (composition.kind !== 'research_note_outline' || composition.version !== 1) return []
  if (!Array.isArray(composition.source_notes)) return []
  return composition.source_notes.flatMap((item) => {
    if (!item || typeof item !== 'object') return []
    const source = item as Partial<ResearchNoteOutlineSource>
    const id = String(source.id || '').trim()
    if (!id) return []
    const revision = Number(source.revision || 0)
    return [{
      id,
      title: String(source.title || '').trim() || id,
      revision: Number.isInteger(revision) && revision > 0 ? revision : 1,
    }]
  }).slice(0, 100)
}
