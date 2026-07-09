import { useCallback, useEffect, useMemo, useState } from 'react'
import { message } from 'antd'
import { libraryApi, type RenameSuggestionItem } from '../../api/library'
import { useLibraryStore } from '../../stores/libraryStore'

export type LibraryRenameActionsInput = {
  S: Record<string, string>
  pageSize: number
  scope: string
  textModelReady: boolean
  warnLlmFallback: (action: string) => void
}

export function useLibraryRenameActions({
  S,
  pageSize,
  scope,
  textModelReady,
  warnLlmFallback,
}: LibraryRenameActionsInput) {
  const loadFiles = useLibraryStore((s) => s.loadFiles)

  const [renameScope, setRenameScope] = useState('30')
  const [renameLoading, setRenameLoading] = useState(false)
  const [renameApplying, setRenameApplying] = useState(false)
  const [renameItems, setRenameItems] = useState<RenameSuggestionItem[]>([])
  const [renameSelected, setRenameSelected] = useState<Record<string, boolean>>({})
  const [renameOverrides, setRenameOverrides] = useState<Record<string, string>>({})
  const [renameResultsOpen, setRenameResultsOpen] = useState(false)
  const [renamePage, setRenamePage] = useState(1)

  const renameVisible = useMemo(() => renameItems.filter((x) => x.diff), [renameItems])
  const selectedRenameCount = useMemo(
    () => renameItems.filter((x) => renameSelected[x.name]).length,
    [renameItems, renameSelected],
  )
  const renamePageCount = Math.max(1, Math.ceil(renameVisible.length / Math.max(1, pageSize)))
  const pagedRenameVisible = useMemo(
    () => renameVisible.slice((renamePage - 1) * pageSize, renamePage * pageSize),
    [pageSize, renamePage, renameVisible],
  )

  useEffect(() => {
    if (renamePage > renamePageCount) setRenamePage(renamePageCount)
  }, [renamePage, renamePageCount])

  const scanRenameSuggestions = useCallback(async () => {
    setRenameLoading(true)
    try {
      const effectiveUseLlm = textModelReady
      if (!textModelReady) {
        warnLlmFallback(S.lib_btn_rename_check)
      }
      const res = await libraryApi.listRenameSuggestions(renameScope, effectiveUseLlm)
      const items = Array.isArray(res.items) ? res.items : []
      setRenameItems(items)
      const selected: Record<string, boolean> = {}
      const overrides: Record<string, string> = {}
      for (const item of items) {
        selected[item.name] = Boolean(item.diff)
        overrides[item.name] = item.suggested_stem || item.name.replace(/\.pdf$/i, '')
      }
      setRenameSelected(selected)
      setRenameOverrides(overrides)
      setRenameResultsOpen(items.some((item) => item.diff))
      message.success(S.lib_msg_scan_result.replace('{changed}', String(res.changed)).replace('{total}', String(res.total_scanned)))
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.lib_msg_scan_rename_fail)
    } finally {
      setRenameLoading(false)
    }
  }, [S, renameScope, textModelReady, warnLlmFallback])

  const applyRenameSuggestions = useCallback(async () => {
    const names = renameItems.filter((x) => renameSelected[x.name]).map((x) => x.name)
    if (!names.length) {
      message.info(S.lib_msg_select_rename)
      return
    }
    setRenameApplying(true)
    try {
      const overrides: Record<string, string> = {}
      for (const name of names) overrides[name] = String(renameOverrides[name] || '').trim()
      const effectiveUseLlm = textModelReady
      if (!textModelReady) {
        warnLlmFallback(S.lib_btn_apply_rename)
      }
      const res = await libraryApi.applyRenameSuggestions(names, overrides, { useLlm: effectiveUseLlm, alsoMd: true })
      message[res.failed > 0 ? 'warning' : 'success'](S.lib_msg_rename_result.replace('{ok}', String(res.renamed)).replace('{skip}', String(res.skipped)).replace('{fail}', String(res.failed)))
      if (res.needs_reindex) message.info(S.lib_msg_rename_suggest_reindex)
      await loadFiles(scope)
      await scanRenameSuggestions()
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.lib_msg_apply_rename_fail)
    } finally {
      setRenameApplying(false)
    }
  }, [S, loadFiles, renameItems, renameOverrides, renameSelected, scanRenameSuggestions, scope, textModelReady, warnLlmFallback])

  const selectRenameDiffItems = useCallback(() => {
    setRenameSelected((cur) => {
      const next = { ...cur }
      for (const item of renameItems) {
        next[item.name] = Boolean(item.diff)
      }
      return next
    })
  }, [renameItems])

  const clearRenameSelection = useCallback(() => {
    setRenameSelected((cur) => {
      const next = { ...cur }
      for (const item of renameItems) {
        next[item.name] = false
      }
      return next
    })
  }, [renameItems])

  const toggleRenameResultsOpen = useCallback(() => {
    setRenameResultsOpen((open) => !open)
  }, [])

  const setRenameItemSelected = useCallback((name: string, selected: boolean) => {
    setRenameSelected((cur) => ({ ...cur, [name]: selected }))
  }, [])

  const setRenameOverride = useCallback((name: string, value: string) => {
    setRenameOverrides((cur) => ({ ...cur, [name]: value }))
  }, [])

  return {
    applyRenameSuggestions,
    clearRenameSelection,
    pagedRenameVisible,
    renameApplying,
    renameItems,
    renameLoading,
    renameOverrides,
    renamePage,
    renameResultsOpen,
    renameScope,
    renameSelected,
    renameVisible,
    scanRenameSuggestions,
    selectRenameDiffItems,
    selectedRenameCount,
    setRenameItemSelected,
    setRenameOverride,
    setRenamePage,
    setRenameScope,
    toggleRenameResultsOpen,
  }
}
