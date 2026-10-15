import { useCallback, useEffect, useRef, useState } from 'react'
import { chatApi, type ResearchNoteRecord } from '../../api/chat'
import {
  RESEARCH_NOTES_CHANGED_EVENT,
  RESEARCH_NOTES_SYNC_CHANNEL,
} from './readerResearchNoteCapture'

interface UseResearchNoteCollectionOptions {
  onCountChange?: (state: { count: number }) => void
  onDeleteError?: () => void
  onDeleteSuccess?: () => void
  projectId?: string | null
}

export function useResearchNoteCollection({
  onCountChange,
  onDeleteError,
  onDeleteSuccess,
  projectId,
}: UseResearchNoteCollectionOptions) {
  const [notes, setNotes] = useState<ResearchNoteRecord[]>([])
  const [loading, setLoading] = useState(false)
  const loadSequenceRef = useRef(0)

  const reload = useCallback(async () => {
    const sequence = loadSequenceRef.current + 1
    loadSequenceRef.current = sequence
    setLoading(true)
    try {
      const records = await chatApi.listResearchNotes(String(projectId || '').trim() || null)
      if (loadSequenceRef.current !== sequence) return
      setNotes(Array.isArray(records) ? records : [])
    } catch {
      if (loadSequenceRef.current !== sequence) return
      setNotes([])
    } finally {
      if (loadSequenceRef.current === sequence) setLoading(false)
    }
  }, [projectId])

  useEffect(() => {
    setNotes([])
    void reload()
    return () => {
      loadSequenceRef.current += 1
    }
  }, [reload])

  useEffect(() => {
    const refresh = () => { void reload() }
    window.addEventListener(RESEARCH_NOTES_CHANGED_EVENT, refresh)
    const channel = typeof BroadcastChannel !== 'undefined'
      ? new BroadcastChannel(RESEARCH_NOTES_SYNC_CHANNEL)
      : null
    if (channel) {
      channel.onmessage = (event) => {
        const data = event?.data && typeof event.data === 'object'
          ? event.data as Record<string, unknown>
          : {}
        if (String(data.type || '') === 'research-notes-changed') refresh()
      }
    }
    return () => {
      window.removeEventListener(RESEARCH_NOTES_CHANGED_EVENT, refresh)
      channel?.close()
    }
  }, [reload])

  useEffect(() => {
    onCountChange?.({ count: notes.length })
  }, [notes.length, onCountChange])

  const upsertNote = useCallback((record: ResearchNoteRecord) => {
    const activeProjectId = String(projectId || '').trim()
    const recordProjectId = String(record.project_id || '').trim()
    if (activeProjectId && recordProjectId && activeProjectId !== recordProjectId) {
      void reload()
      return
    }
    loadSequenceRef.current += 1
    setLoading(false)
    setNotes((current) => [
      record,
      ...current.filter((note) => note.id !== record.id),
    ].sort((a, b) => Number(b.updated_at || 0) - Number(a.updated_at || 0)))
    void reload()
  }, [projectId, reload])

  const deleteNote = useCallback(async (noteId: string) => {
    try {
      await chatApi.deleteResearchNote(noteId)
      loadSequenceRef.current += 1
      setLoading(false)
      setNotes((current) => current.filter((note) => note.id !== noteId))
      onDeleteSuccess?.()
      void reload()
    } catch {
      onDeleteError?.()
    }
  }, [onDeleteError, onDeleteSuccess, reload])

  return {
    deleteNote,
    loading,
    notes,
    reload,
    upsertNote,
  }
}
