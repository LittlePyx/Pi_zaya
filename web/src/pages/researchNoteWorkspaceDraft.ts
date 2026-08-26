import type { ResearchNoteRecord } from '../api/chat'

const WORKSPACE_DRAFT_VERSION = 1
const WORKSPACE_DRAFT_PREFIX = 'kb:research-note-workspace-draft:'

export interface ResearchNoteWorkspaceDraft {
  version: number
  noteId: string
  baseRevision: number
  title: string
  contentMarkdown: string
  projectId: string | null
  tags: string[]
  savedAt: number
}

function draftKey(noteId: string): string {
  return `${WORKSPACE_DRAFT_PREFIX}${noteId}`
}

function cleanTags(value: unknown): string[] {
  return Array.isArray(value)
    ? value.map(tag => String(tag || '').trim()).filter(Boolean).slice(0, 24)
    : []
}

export function readResearchNoteWorkspaceDraft(noteId: string): ResearchNoteWorkspaceDraft | null {
  if (!noteId || typeof window === 'undefined') return null
  try {
    const raw = window.localStorage.getItem(draftKey(noteId))
    if (!raw) return null
    const value = JSON.parse(raw) as Partial<ResearchNoteWorkspaceDraft>
    if (value.version !== WORKSPACE_DRAFT_VERSION || value.noteId !== noteId) return null
    const baseRevision = Number(value.baseRevision || 0)
    if (!Number.isInteger(baseRevision) || baseRevision < 1) return null
    return {
      version: WORKSPACE_DRAFT_VERSION,
      noteId,
      baseRevision,
      title: String(value.title || ''),
      contentMarkdown: String(value.contentMarkdown || ''),
      projectId: value.projectId ? String(value.projectId) : null,
      tags: cleanTags(value.tags),
      savedAt: Number(value.savedAt || 0),
    }
  } catch {
    return null
  }
}

export function writeResearchNoteWorkspaceDraft(
  draft: Omit<ResearchNoteWorkspaceDraft, 'version' | 'savedAt'>,
): void {
  if (!draft.noteId || typeof window === 'undefined') return
  try {
    window.localStorage.setItem(draftKey(draft.noteId), JSON.stringify({
      ...draft,
      version: WORKSPACE_DRAFT_VERSION,
      savedAt: Date.now(),
    } satisfies ResearchNoteWorkspaceDraft))
  } catch {
    // Local draft persistence is best-effort; the in-memory editor remains usable.
  }
}

export function clearResearchNoteWorkspaceDraft(noteId: string): void {
  if (!noteId || typeof window === 'undefined') return
  try {
    window.localStorage.removeItem(draftKey(noteId))
  } catch {
    // Ignore unavailable or quota-restricted storage.
  }
}

export function workspaceDraftMatchesRecord(
  draft: ResearchNoteWorkspaceDraft,
  record: ResearchNoteRecord,
): boolean {
  return draft.title.trim() === String(record.title || '').trim()
    && draft.contentMarkdown.trim() === String(record.content_markdown || '').trim()
    && (draft.projectId || null) === (record.project_id || null)
    && JSON.stringify(draft.tags) === JSON.stringify(Array.isArray(record.tags) ? record.tags : [])
}
