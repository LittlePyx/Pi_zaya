import { useCallback, useEffect, useMemo, useState } from 'react'
import { message } from 'antd'
import { settingsApi } from '../../api/settings'
import { useLibraryStore } from '../../stores/libraryStore'
import { useSettingsStore } from '../../stores/settingsStore'

export type LibraryDirectoryKind = 'pdf' | 'md'
export type LibraryDirectoryOpenTarget = 'pdf_dir' | 'md_dir'

export type LibraryDirectoryActionsInput = {
  S: Record<string, string>
  scope: string
}

export function useLibraryDirectoryActions({
  S,
  scope,
}: LibraryDirectoryActionsInput) {
  const settingsLoaded = useSettingsStore((s) => s.loaded)
  const settingsPdfDir = useSettingsStore((s) => s.pdfDir)
  const settingsMdDir = useSettingsStore((s) => s.mdDir)
  const updateSettings = useSettingsStore((s) => s.update)
  const loadFiles = useLibraryStore((s) => s.loadFiles)
  const openFile = useLibraryStore((s) => s.openFile)

  const [pdfDirDraft, setPdfDirDraft] = useState('')
  const [mdDirDraft, setMdDirDraft] = useState('')
  const [savingDirs, setSavingDirs] = useState(false)
  const [pickingDir, setPickingDir] = useState<LibraryDirectoryKind | null>(null)
  const [dirTouched, setDirTouched] = useState(false)
  const [dirEditorOpen, setDirEditorOpen] = useState(false)

  const dirDirty = useMemo(
    () =>
      pdfDirDraft.trim() !== String(settingsPdfDir || '').trim()
      || mdDirDraft.trim() !== String(settingsMdDir || '').trim(),
    [pdfDirDraft, mdDirDraft, settingsPdfDir, settingsMdDir],
  )
  const directoriesConfigured = Boolean(pdfDirDraft.trim() && mdDirDraft.trim())

  useEffect(() => {
    if (!settingsLoaded || dirTouched) return
    setPdfDirDraft(String(settingsPdfDir || ''))
    setMdDirDraft(String(settingsMdDir || ''))
  }, [settingsLoaded, settingsPdfDir, settingsMdDir, dirTouched])

  useEffect(() => {
    if (!settingsLoaded) return
    if (!String(settingsPdfDir || '').trim() || !String(settingsMdDir || '').trim()) {
      setDirEditorOpen(true)
    }
  }, [settingsLoaded, settingsPdfDir, settingsMdDir])

  const saveDirs = useCallback(async () => {
    if (!pdfDirDraft.trim() || !mdDirDraft.trim()) {
      message.warning(S.lib_msg_dir_empty)
      return false
    }
    setSavingDirs(true)
    try {
      await updateSettings({ pdfDir: pdfDirDraft.trim(), mdDir: mdDirDraft.trim() })
      setDirTouched(false)
      setDirEditorOpen(false)
      message.success(S.lib_msg_save_dir_success)
      await loadFiles(scope)
      return true
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.lib_msg_save_dir_fail)
      return false
    } finally {
      setSavingDirs(false)
    }
  }, [mdDirDraft, pdfDirDraft, scope, loadFiles, updateSettings, S])

  const ensureDirsReady = useCallback(async () => {
    if (!dirDirty) return true
    return saveDirs()
  }, [dirDirty, saveDirs])

  const openFolder = useCallback(async (target: LibraryDirectoryOpenTarget) => {
    const ready = await ensureDirsReady()
    if (!ready) return
    await openFile('', target)
  }, [ensureDirsReady, openFile])

  const pickDir = useCallback(async (target: LibraryDirectoryKind) => {
    const initial = target === 'pdf' ? pdfDirDraft : mdDirDraft
    setPickingDir(target)
    try {
      const res = await settingsApi.pickDir(target, initial)
      if (!res.ok || !res.path) {
        message.info(S.lib_msg_no_dir_selected)
        return
      }
      setDirTouched(true)
      if (target === 'pdf') setPdfDirDraft(res.path)
      else setMdDirDraft(res.path)
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.lib_msg_pick_dir_fail)
    } finally {
      setPickingDir(null)
    }
  }, [mdDirDraft, pdfDirDraft, S])

  const updatePdfDirDraft = useCallback((value: string) => {
    setDirTouched(true)
    setPdfDirDraft(value)
  }, [])

  const updateMdDirDraft = useCallback((value: string) => {
    setDirTouched(true)
    setMdDirDraft(value)
  }, [])

  const toggleDirEditor = useCallback(() => {
    setDirEditorOpen((open) => !open)
  }, [])

  return {
    directoriesConfigured,
    dirDirty,
    dirEditorOpen,
    ensureDirsReady,
    mdDirDraft,
    openFolder,
    pdfDirDraft,
    pickDir,
    pickingDir,
    saveDirs,
    savingDirs,
    toggleDirEditor,
    updateMdDirDraft,
    updatePdfDirDraft,
  }
}

