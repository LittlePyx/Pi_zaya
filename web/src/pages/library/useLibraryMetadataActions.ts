import { useCallback, useMemo, useRef, useState } from 'react'
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

function normalizeMetaDraft(draft: LibraryMetaDraft): LibraryMetaDraft {
  return {
    paper_category: normalizeTextValue(draft.paper_category),
    reading_status: String(draft.reading_status || '') as ReadingStatusValue,
    note: String(draft.note || ''),
    user_tags: normalizeTextList(draft.user_tags),
  }
}

function metaDraftsEqual(left: LibraryMetaDraft, right: LibraryMetaDraft) {
  const a = normalizeMetaDraft(left)
  const b = normalizeMetaDraft(right)
  return a.paper_category === b.paper_category
    && a.reading_status === b.reading_status
    && a.note === b.note
    && a.user_tags.length === b.user_tags.length
    && a.user_tags.every((tag, index) => tag === b.user_tags[index])
}

function tagKey(value: string) {
  return value.trim().toLowerCase()
}

function mergeSuggestionTagDrafts({
  action,
  baseline,
  current,
  previousSuggestedTags,
  updatedItem,
}: {
  action: MetadataSuggestionAction
  baseline: string[]
  current: string[]
  previousSuggestedTags: string[]
  updatedItem: LibraryFileItem
}) {
  const baselineTags = normalizeTextList(baseline)
  const currentTags = normalizeTextList(current)
  const persistedTags = normalizeTextList(updatedItem.user_tags || [])
  const pendingSuggestionTags = normalizeTextList(updatedItem.suggested_tags || [])
  const acceptedTags = action.accept_all_tags
    ? normalizeTextList(previousSuggestedTags)
    : normalizeTextList(action.accept_tags || [])

  const baselineKeys = new Set(baselineTags.map(tagKey))
  const currentKeys = new Set(currentTags.map(tagKey))
  const persistedKeys = new Set(persistedTags.map(tagKey))
  const pendingSuggestionKeys = new Set(pendingSuggestionTags.map(tagKey))
  const acceptedKeys = new Set(acceptedTags.map(tagKey))
  const manualAdditionKeys = new Set(
    currentTags.map(tagKey).filter((key) => !baselineKeys.has(key)),
  )
  const manualRemovalKeys = new Set(
    baselineTags
      .map(tagKey)
      .filter((key) => !currentKeys.has(key) && !acceptedKeys.has(key)),
  )

  const nextDraftTags: string[] = []
  for (const tag of currentTags) {
    const key = tagKey(tag)
    if (
      persistedKeys.has(key)
      || pendingSuggestionKeys.has(key)
      || manualAdditionKeys.has(key)
    ) {
      nextDraftTags.push(tag)
    }
  }
  for (const tag of persistedTags) {
    const key = tagKey(tag)
    if (!manualRemovalKeys.has(key) || acceptedKeys.has(key)) {
      nextDraftTags.push(tag)
    }
  }

  return {
    nextBaselineTags: persistedTags,
    nextDraftTags: normalizeTextList(nextDraftTags),
  }
}

function mergeSuggestionActionDrafts({
  action,
  baseline,
  current,
  previousItem,
  updatedItem,
}: {
  action: MetadataSuggestionAction
  baseline: LibraryMetaDraft
  current: LibraryMetaDraft
  previousItem: LibraryFileItem
  updatedItem: LibraryFileItem
}) {
  let nextDraft = normalizeMetaDraft(current)
  let nextBaseline = normalizeMetaDraft(baseline)
  const previousSuggestedCategory = normalizeTextValue(previousItem.suggested_category)

  if (action.category_action === 'accept') {
    const acceptedCategory = normalizeTextValue(updatedItem.paper_category) || previousSuggestedCategory
    nextDraft = { ...nextDraft, paper_category: acceptedCategory }
    nextBaseline = { ...nextBaseline, paper_category: acceptedCategory }
  } else if (action.category_action === 'dismiss' && previousSuggestedCategory) {
    const persistedCategory = normalizeTextValue(updatedItem.paper_category)
    if (normalizeTextValue(nextDraft.paper_category) === previousSuggestedCategory) {
      nextDraft = { ...nextDraft, paper_category: persistedCategory }
    }
    if (normalizeTextValue(nextBaseline.paper_category) === previousSuggestedCategory) {
      nextBaseline = { ...nextBaseline, paper_category: persistedCategory }
    }
  }

  const tagMerge = mergeSuggestionTagDrafts({
    action,
    baseline: nextBaseline.user_tags,
    current: nextDraft.user_tags,
    previousSuggestedTags: previousItem.suggested_tags || [],
    updatedItem,
  })
  nextDraft = { ...nextDraft, user_tags: tagMerge.nextDraftTags }
  nextBaseline = { ...nextBaseline, user_tags: tagMerge.nextBaselineTags }

  return { nextBaseline, nextDraft }
}

export function useLibraryMetadataActions({
  S,
}: LibraryMetadataActionsInput) {
  const updatePaperMeta = useLibraryStore((s) => s.updatePaperMeta)
  const regenerateSuggestions = useLibraryStore((s) => s.regenerateSuggestions)
  const applySuggestionAction = useLibraryStore((s) => s.applySuggestionAction)

  const [metaDrawerOpen, setMetaDrawerOpen] = useState(false)
  const [metaCloseConfirmOpen, setMetaCloseConfirmOpen] = useState(false)
  const [metaSaving, setMetaSaving] = useState(false)
  const [metaSuggestionSaving, setMetaSuggestionSaving] = useState(false)
  const metaOperationRef = useRef<'save' | 'suggestion' | null>(null)
  const [metaItem, setMetaItem] = useState<LibraryFileItem | null>(null)
  const [metaDraft, setMetaDraft] = useState<LibraryMetaDraft>({
    paper_category: '',
    reading_status: '',
    note: '',
    user_tags: [],
  })
  const [metaBaseline, setMetaBaseline] = useState<LibraryMetaDraft>({
    paper_category: '',
    reading_status: '',
    note: '',
    user_tags: [],
  })

  const metaSuggestionCount = (metaItem?.suggested_category ? 1 : 0) + (metaItem?.suggested_tags?.length || 0)
  const metaDraftCategory = useMemo(() => normalizeTextValue(metaDraft.paper_category), [metaDraft.paper_category])
  const metaDraftTags = useMemo(() => normalizeTextList(metaDraft.user_tags), [metaDraft.user_tags])
  const metaDirty = useMemo(() => !metaDraftsEqual(metaDraft, metaBaseline), [metaBaseline, metaDraft])
  const metaBusy = metaSaving || metaSuggestionSaving

  const openMetaEditor = useCallback((item: LibraryFileItem) => {
    const initialDraft = draftFromLibraryItem(item)
    setMetaItem(item)
    setMetaDraft(initialDraft)
    setMetaBaseline(initialDraft)
    setMetaCloseConfirmOpen(false)
    setMetaDrawerOpen(true)
  }, [])

  const closeMetaEditor = useCallback(() => {
    if (metaOperationRef.current || metaBusy) return
    if (metaDirty) {
      setMetaCloseConfirmOpen(true)
      return
    }
    setMetaCloseConfirmOpen(false)
    setMetaDrawerOpen(false)
  }, [metaBusy, metaDirty])

  const continueMetaEditing = useCallback(() => {
    if (metaOperationRef.current || metaBusy) return
    setMetaCloseConfirmOpen(false)
  }, [metaBusy])

  const discardMetaEditor = useCallback(() => {
    if (metaOperationRef.current || metaBusy) return
    setMetaDraft(metaBaseline)
    setMetaCloseConfirmOpen(false)
    setMetaDrawerOpen(false)
  }, [metaBaseline, metaBusy])

  const saveMetaEditor = useCallback(async () => {
    if (!metaItem || metaOperationRef.current) return false
    metaOperationRef.current = 'save'
    const paperCategory = normalizeTextValue(metaDraft.paper_category)
    const userTags = normalizeTextList(metaDraft.user_tags)
    const savedDraft = normalizeMetaDraft({
      ...metaDraft,
      paper_category: paperCategory,
      user_tags: userTags,
    })
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
      setMetaDraft(savedDraft)
      setMetaBaseline(savedDraft)
      setMetaCloseConfirmOpen(false)
      setMetaDrawerOpen(false)
      message.success(S.lib_msg_meta_saved)
      return true
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.lib_msg_meta_save_fail)
      return false
    } finally {
      metaOperationRef.current = null
      setMetaSaving(false)
    }
  }, [S.lib_msg_meta_save_fail, S.lib_msg_meta_saved, metaDraft, metaItem, updatePaperMeta])

  const applyMetaSuggestionAction = useCallback(async (body: MetadataSuggestionAction) => {
    if (!metaItem || metaOperationRef.current) return
    metaOperationRef.current = 'suggestion'
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
        setMetaDraft((current) => mergeSuggestionActionDrafts({
          action: body,
          baseline: metaBaseline,
          current,
          previousItem: metaItem,
          updatedItem: updated,
        }).nextDraft)
        setMetaBaseline((baseline) => mergeSuggestionActionDrafts({
          action: body,
          baseline,
          current: baseline,
          previousItem: metaItem,
          updatedItem: updated,
        }).nextBaseline)
      }
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.lib_msg_update_suggestion_fail)
    } finally {
      metaOperationRef.current = null
      setMetaSuggestionSaving(false)
    }
  }, [S.lib_msg_update_suggestion_fail, applySuggestionAction, metaBaseline, metaItem])

  const regenerateMetaSuggestions = useCallback(async () => {
    if (!metaItem || metaOperationRef.current) return
    metaOperationRef.current = 'suggestion'
    setMetaSuggestionSaving(true)
    try {
      await regenerateSuggestions({ pdf_names: [metaItem.name] })
      const refreshed = useLibraryStore.getState().files.find((item) => item.name === metaItem.name) || null
      if (refreshed) {
        setMetaItem(refreshed)
      }
      message.success(S.lib_msg_suggestion_refreshed)
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.lib_msg_refresh_suggestion_fail)
    } finally {
      metaOperationRef.current = null
      setMetaSuggestionSaving(false)
    }
  }, [S.lib_msg_refresh_suggestion_fail, S.lib_msg_suggestion_refreshed, metaItem, regenerateSuggestions])

  return {
    applyMetaSuggestionAction,
    closeMetaEditor,
    continueMetaEditing,
    discardMetaEditor,
    metaDraft,
    metaDraftCategory,
    metaDraftTags,
    metaDrawerOpen,
    metaBusy,
    metaCloseConfirmOpen,
    metaDirty,
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
