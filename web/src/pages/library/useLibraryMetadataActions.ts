import { useCallback, useMemo, useState } from 'react'
import { message } from 'antd'
import type { LibraryFileItem, LibrarySuggestionActionBody } from '../../api/library'
import { useLibraryStore } from '../../stores/libraryStore'
import {
  normalizeTextList,
  normalizeTextValue,
} from './libraryPageUtils'

type ReadingStatusValue = LibraryFileItem['reading_status']

export type LibraryMetaDraft = {
  paper_category: string
  reading_status: ReadingStatusValue
  note: string
  user_tags: string[]
}

export type MetadataSuggestionAction = Omit<LibrarySuggestionActionBody, 'pdf_name' | 'sha1' | 'path'>

export type LibraryMetadataActionsInput = {
  S: Record<string, string>
}

function draftFromLibraryItem(item: LibraryFileItem): LibraryMetaDraft {
  const confirmedCategory = normalizeTextValue(item.paper_category)
  const suggestedCategory = normalizeTextValue(item.suggested_category)
  const confirmedTags = normalizeTextList(Array.isArray(item.user_tags) ? item.user_tags : [])
  const suggestedTags = normalizeTextList(Array.isArray(item.suggested_tags) ? item.suggested_tags : [])
  return {
    paper_category: confirmedCategory || suggestedCategory,
    reading_status: String(item.reading_status || '') as ReadingStatusValue,
    note: String(item.note || ''),
    user_tags: confirmedTags.length > 0 ? confirmedTags : normalizeTextList([...confirmedTags, ...suggestedTags]),
  }
}

function draftFromPersistedItem(item: LibraryFileItem): LibraryMetaDraft {
  return {
    paper_category: normalizeTextValue(item.paper_category),
    reading_status: String(item.reading_status || '') as ReadingStatusValue,
    note: String(item.note || ''),
    user_tags: normalizeTextList(Array.isArray(item.user_tags) ? item.user_tags : []),
  }
}

export function useLibraryMetadataActions({
  S,
}: LibraryMetadataActionsInput) {
  const updatePaperMeta = useLibraryStore((s) => s.updatePaperMeta)
  const regenerateSuggestions = useLibraryStore((s) => s.regenerateSuggestions)
  const applySuggestionAction = useLibraryStore((s) => s.applySuggestionAction)

  const [metaDrawerOpen, setMetaDrawerOpen] = useState(false)
  const [metaSaving, setMetaSaving] = useState(false)
  const [metaSuggestionSaving, setMetaSuggestionSaving] = useState(false)
  const [metaItem, setMetaItem] = useState<LibraryFileItem | null>(null)
  const [metaDraft, setMetaDraft] = useState<LibraryMetaDraft>({
    paper_category: '',
    reading_status: '',
    note: '',
    user_tags: [],
  })

  const metaSuggestionCount = (metaItem?.suggested_category ? 1 : 0) + (metaItem?.suggested_tags?.length || 0)
  const metaDraftCategory = useMemo(() => normalizeTextValue(metaDraft.paper_category), [metaDraft.paper_category])
  const metaDraftTags = useMemo(() => normalizeTextList(metaDraft.user_tags), [metaDraft.user_tags])

  const openMetaEditor = useCallback((item: LibraryFileItem) => {
    setMetaItem(item)
    setMetaDraft(draftFromLibraryItem(item))
    setMetaDrawerOpen(true)
  }, [])

  const closeMetaEditor = useCallback(() => {
    setMetaDrawerOpen(false)
  }, [])

  const saveMetaEditor = useCallback(async () => {
    if (!metaItem) return
    const paperCategory = normalizeTextValue(metaDraft.paper_category)
    const userTags = normalizeTextList(metaDraft.user_tags)
    setMetaSaving(true)
    try {
      const updated = await updatePaperMeta({
        pdf_name: metaItem.name,
        paper_category: paperCategory,
        reading_status: metaDraft.reading_status,
        note: metaDraft.note,
        user_tags: userTags,
      })
      if (updated) setMetaItem(updated)
      setMetaDrawerOpen(false)
      message.success(S.lib_msg_meta_saved)
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.lib_msg_meta_save_fail)
    } finally {
      setMetaSaving(false)
    }
  }, [S.lib_msg_meta_save_fail, S.lib_msg_meta_saved, metaDraft, metaItem, updatePaperMeta])

  const applyMetaSuggestionAction = useCallback(async (body: MetadataSuggestionAction) => {
    if (!metaItem) return
    setMetaSuggestionSaving(true)
    try {
      const updated = await applySuggestionAction({
        pdf_name: metaItem.name,
        category_action: body.category_action,
        accept_tags: body.accept_tags,
        dismiss_tags: body.dismiss_tags,
        accept_all_tags: body.accept_all_tags,
        dismiss_all_tags: body.dismiss_all_tags,
      })
      if (updated) {
        setMetaItem(updated)
        setMetaDraft((cur) => ({
          ...cur,
          ...draftFromPersistedItem(updated),
        }))
      }
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.lib_msg_update_suggestion_fail)
    } finally {
      setMetaSuggestionSaving(false)
    }
  }, [S.lib_msg_update_suggestion_fail, applySuggestionAction, metaItem])

  const regenerateMetaSuggestions = useCallback(async () => {
    if (!metaItem) return
    setMetaSuggestionSaving(true)
    try {
      await regenerateSuggestions({ pdf_names: [metaItem.name] })
      const refreshed = useLibraryStore.getState().files.find((item) => item.name === metaItem.name) || null
      if (refreshed) {
        setMetaItem(refreshed)
        setMetaDraft((cur) => ({
          ...cur,
          ...draftFromPersistedItem(refreshed),
        }))
      }
      message.success(S.lib_msg_suggestion_refreshed)
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.lib_msg_refresh_suggestion_fail)
    } finally {
      setMetaSuggestionSaving(false)
    }
  }, [S.lib_msg_refresh_suggestion_fail, S.lib_msg_suggestion_refreshed, metaItem, regenerateSuggestions])

  return {
    applyMetaSuggestionAction,
    closeMetaEditor,
    metaDraft,
    metaDraftCategory,
    metaDraftTags,
    metaDrawerOpen,
    metaItem,
    metaSaving,
    metaSuggestionCount,
    metaSuggestionSaving,
    openMetaEditor,
    regenerateMetaSuggestions,
    saveMetaEditor,
    setMetaDraft,
  }
}
