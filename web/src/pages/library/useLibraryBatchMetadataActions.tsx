import { useCallback, useEffect, useMemo, useState } from 'react'
import { Modal, Typography, message } from 'antd'
import { ExclamationCircleOutlined } from '@ant-design/icons'
import type { LibraryFileItem } from '../../api/library'
import { useLibraryStore } from '../../stores/libraryStore'
import {
  hasConversionQualityIssue,
  normalizeTextList,
  normalizeTextValue,
} from './libraryPageUtils'
import type { LibraryBatchMetaDraft } from './LibraryBatchMetadataDrawer'

const { Text } = Typography

export type LibraryBatchMetadataActionsInput = {
  S: Record<string, string>
  currentListItems: LibraryFileItem[]
}

function emptyBatchDraft(): LibraryBatchMetaDraft {
  return {
    apply_paper_category: false,
    paper_category: '',
    apply_reading_status: false,
    reading_status: '',
    add_tags: [],
    remove_tags: [],
  }
}

export function useLibraryBatchMetadataActions({
  S,
  currentListItems,
}: LibraryBatchMetadataActionsInput) {
  const files = useLibraryStore((s) => s.files)
  const batchUpdatePaperMeta = useLibraryStore((s) => s.batchUpdatePaperMeta)

  const [selectedLibraryNames, setSelectedLibraryNames] = useState<Record<string, boolean>>({})
  const [batchDrawerOpen, setBatchDrawerOpen] = useState(false)
  const [batchSaving, setBatchSaving] = useState(false)
  const [batchDraft, setBatchDraft] = useState<LibraryBatchMetaDraft>(() => emptyBatchDraft())

  const selectedLibraryNamesList = useMemo(
    () => Object.keys(selectedLibraryNames).filter((name) => Boolean(selectedLibraryNames[name])),
    [selectedLibraryNames],
  )
  const selectedLibraryCount = selectedLibraryNamesList.length
  const selectedQualityReviewNames = useMemo(
    () => files
      .filter((item) => Boolean(selectedLibraryNames[item.name]) && hasConversionQualityIssue(item) && item.task_state === 'idle')
      .map((item) => item.name),
    [files, selectedLibraryNames],
  )

  useEffect(() => {
    const existing = new Set(files.map((item) => item.name))
    setSelectedLibraryNames((cur) => {
      let changed = false
      const next: Record<string, boolean> = {}
      for (const [name, selected] of Object.entries(cur)) {
        if (!selected) continue
        if (!existing.has(name)) {
          changed = true
          continue
        }
        next[name] = true
      }
      return changed ? next : cur
    })
  }, [files])

  const toggleLibrarySelection = useCallback((name: string, checked: boolean) => {
    setSelectedLibraryNames((cur) => {
      if (!checked && !cur[name]) return cur
      return {
        ...cur,
        [name]: checked,
      }
    })
  }, [])

  const selectCurrentListItems = useCallback(() => {
    if (!currentListItems.length) {
      message.info(S.lib_msg_no_selectable)
      return
    }
    setSelectedLibraryNames((cur) => {
      const next = { ...cur }
      for (const item of currentListItems) next[item.name] = true
      return next
    })
  }, [S.lib_msg_no_selectable, currentListItems])

  const clearLibrarySelection = useCallback(() => {
    setSelectedLibraryNames({})
  }, [])

  const closeBatchEditor = useCallback(() => {
    setBatchDrawerOpen(false)
  }, [])

  const openBatchEditor = useCallback(() => {
    if (!selectedLibraryCount) {
      message.info(S.lib_msg_select_batch_edit)
      return
    }
    setBatchDraft(emptyBatchDraft())
    setBatchDrawerOpen(true)
  }, [S.lib_msg_select_batch_edit, selectedLibraryCount])

  const confirmBatchEditorRisk = useCallback(async (params: { paperCategory: string, removeTags: string[] }): Promise<boolean> => {
    const willClearCategory = batchDraft.apply_paper_category && !params.paperCategory
    const willClearStatus = batchDraft.apply_reading_status && !batchDraft.reading_status
    const willRemoveTags = params.removeTags.length > 0
    if (!willClearCategory && !willClearStatus && !willRemoveTags) return true

    const previewTags = params.removeTags.slice(0, 8).join(', ')
    const removeTagsText = params.removeTags.length > 8 ? `${previewTags}...` : previewTags
    return new Promise<boolean>((resolve) => {
      let settled = false
      const done = (value: boolean) => {
        if (settled) return
        settled = true
        resolve(value)
      }
      Modal.confirm({
        title: S.lib_batch_confirm_title,
        icon: <ExclamationCircleOutlined />,
        content: (
          <div className="kb-lib-batch-confirm">
            <Text>{S.lib_batch_confirm_detail.replace('{n}', String(selectedLibraryCount))}</Text>
            {willClearCategory ? <Text type="warning">{S.lib_batch_confirm_clear_category}</Text> : null}
            {willClearStatus ? <Text type="warning">{S.lib_batch_confirm_clear_status}</Text> : null}
            {willRemoveTags ? (
              <Text type="danger">{S.lib_batch_confirm_remove_tags.replace('{tags}', removeTagsText)}</Text>
            ) : null}
          </div>
        ),
        okText: S.lib_batch_confirm_ok,
        cancelText: S.lib_batch_confirm_cancel,
        okButtonProps: { danger: true },
        onOk: () => done(true),
        onCancel: () => done(false),
        afterClose: () => done(false),
      })
    })
  }, [S, batchDraft, selectedLibraryCount])

  const saveBatchEditor = useCallback(async () => {
    if (!selectedLibraryCount) return
    const paperCategory = normalizeTextValue(batchDraft.paper_category)
    const addTags = normalizeTextList(batchDraft.add_tags)
    const removeTags = normalizeTextList(batchDraft.remove_tags)
    if (
      !batchDraft.apply_paper_category
      && !batchDraft.apply_reading_status
      && addTags.length === 0
      && removeTags.length === 0
    ) {
      message.info(S.lib_msg_set_batch_content)
      return
    }
    const confirmed = await confirmBatchEditorRisk({ paperCategory, removeTags })
    if (!confirmed) return
    setBatchSaving(true)
    try {
      const updated = await batchUpdatePaperMeta({
        pdf_names: selectedLibraryNamesList,
        apply_paper_category: batchDraft.apply_paper_category,
        paper_category: paperCategory,
        apply_reading_status: batchDraft.apply_reading_status,
        reading_status: batchDraft.reading_status,
        add_tags: addTags,
        remove_tags: removeTags,
      })
      setBatchDrawerOpen(false)
      setSelectedLibraryNames({})
      message.success(S.lib_msg_batch_updated_count.replace('{n}', String(updated)))
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.lib_msg_batch_edit_fail)
    } finally {
      setBatchSaving(false)
    }
  }, [S, batchDraft, batchUpdatePaperMeta, confirmBatchEditorRisk, selectedLibraryCount, selectedLibraryNamesList])

  return {
    batchDraft,
    batchDrawerOpen,
    batchSaving,
    clearLibrarySelection,
    closeBatchEditor,
    openBatchEditor,
    saveBatchEditor,
    selectCurrentListItems,
    selectedLibraryCount,
    selectedLibraryNames,
    selectedLibraryNamesList,
    selectedQualityReviewNames,
    setBatchDraft,
    toggleLibrarySelection,
  }
}
