import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { message } from 'antd'
import { libraryApi } from '../../api/library'
import { useLibraryStore } from '../../stores/libraryStore'
import {
  classifyFailedReason,
  isDuplicateFailure,
  isUploadDraftConverted,
  type UploadDraft,
  type UploadDraftFilter,
  type UploadErrorReason,
} from './libraryPageUtils'

export type LibraryUploadDraftActionsInput = {
  S: Record<string, string>
  convertMode: string
  dirDirty: boolean
  ensureDirsReady: () => Promise<boolean>
  pageSize: number
  scope: string
  textModelReady: boolean
  uploadLocked: boolean
  warnLlmFallback: (action: string) => void
}

export function useLibraryUploadDraftActions({
  S,
  convertMode,
  dirDirty,
  ensureDirsReady,
  pageSize,
  scope,
  textModelReady,
  uploadLocked,
  warnLlmFallback,
}: LibraryUploadDraftActionsInput) {
  const files = useLibraryStore((s) => s.files)
  const loadFiles = useLibraryStore((s) => s.loadFiles)
  const startProgressStream = useLibraryStore((s) => s.startProgressStream)

  const [uploadDrafts, setUploadDrafts] = useState<UploadDraft[]>([])
  const [uploadUseLlm, setUploadUseLlm] = useState(true)
  const [uploadDraftFilter, setUploadDraftFilter] = useState<UploadDraftFilter>('all')
  const [uploadErrorReason, setUploadErrorReason] = useState<UploadErrorReason>('all')
  const [uploadInspecting, setUploadInspecting] = useState(false)
  const [uploadSaving, setUploadSaving] = useState(false)
  const [uploadWorkbenchOpen, setUploadWorkbenchOpen] = useState(false)
  const [uploadDraftPage, setUploadDraftPage] = useState(1)
  const autoInspectingRef = useRef(false)

  const selectedUploadCount = useMemo(() => uploadDrafts.filter((x) => x.selected).length, [uploadDrafts])
  const failedUploadDrafts = useMemo(() => uploadDrafts.filter((x) => x.status === 'error'), [uploadDrafts])
  const duplicateFailedDrafts = useMemo(
    () => failedUploadDrafts.filter((x) => isDuplicateFailure(x.note)),
    [failedUploadDrafts],
  )
  const retryableFailedUploadDrafts = useMemo(
    () => failedUploadDrafts.filter((x) => x.failureStage !== 'duplicate' && !isDuplicateFailure(x.note)),
    [failedUploadDrafts],
  )
  const failedUploadNotes = useMemo(
    () => Array.from(new Set(failedUploadDrafts.map((x) => String(x.note || '').trim()).filter(Boolean))).slice(0, 3),
    [failedUploadDrafts],
  )
  const failedReasonBuckets = useMemo(() => {
    const counter = new Map<Exclude<UploadErrorReason, 'all'>, number>()
    for (const item of failedUploadDrafts) {
      const key = classifyFailedReason(item.note)
      counter.set(key, (counter.get(key) || 0) + 1)
    }
    return Array.from(counter.entries())
      .map(([key, count]) => ({ key, count }))
      .sort((a, b) => b.count - a.count)
  }, [failedUploadDrafts])
  const filteredUploadDrafts = useMemo(() => {
    const withReason = (items: UploadDraft[]) => (
      uploadErrorReason === 'all'
        ? items
        : items.filter((x) => classifyFailedReason(x.note) === uploadErrorReason)
    )
    if (uploadDraftFilter === 'all') return uploadDrafts
    if (uploadDraftFilter === 'error') return withReason(uploadDrafts.filter((x) => x.status === 'error'))
    if (uploadDraftFilter === 'dup_error') return withReason(uploadDrafts.filter((x) => x.status === 'error' && isDuplicateFailure(x.note)))
    if (uploadDraftFilter === 'saved') return uploadDrafts.filter((x) => x.status === 'saved')
    return uploadDrafts.filter((x) => ['queued', 'inspecting', 'ready', 'saving'].includes(x.status))
  }, [uploadDrafts, uploadDraftFilter, uploadErrorReason])
  const uploadDraftFilterOptions = useMemo(
    () => [
      { value: 'all', label: S.lib_upload_filter_all.replace('{n}', String(uploadDrafts.length)) },
      { value: 'todo', label: S.lib_upload_filter_todo.replace('{n}', String(uploadDrafts.filter((x) => ['queued', 'inspecting', 'ready', 'saving'].includes(x.status)).length)) },
      { value: 'error', label: S.lib_upload_filter_error.replace('{n}', String(uploadDrafts.filter((x) => x.status === 'error').length)) },
      { value: 'dup_error', label: S.lib_upload_filter_dup.replace('{n}', String(uploadDrafts.filter((x) => x.status === 'error' && isDuplicateFailure(x.note)).length)) },
      { value: 'saved', label: S.lib_upload_filter_saved.replace('{n}', String(uploadDrafts.filter((x) => x.status === 'saved').length)) },
    ],
    [uploadDrafts, S],
  )
  const uploadDraftPageCount = Math.max(1, Math.ceil(filteredUploadDrafts.length / Math.max(1, pageSize)))
  const pagedUploadDrafts = useMemo(
    () => filteredUploadDrafts.slice((uploadDraftPage - 1) * pageSize, uploadDraftPage * pageSize),
    [filteredUploadDrafts, pageSize, uploadDraftPage],
  )

  const applyUploadFilter = useCallback((next: UploadDraftFilter) => {
    setUploadDraftFilter(next)
    if (next === 'dup_error') {
      setUploadErrorReason('duplicate')
      return
    }
    if (next !== 'error') {
      setUploadErrorReason('all')
    }
  }, [])

  const addDrafts = useCallback((filesToAdd: File[]) => {
    setUploadDrafts((cur) => {
      const seen = new Set(cur.map((x) => x.key))
      const next = [...cur]
      for (const file of filesToAdd) {
        const key = `${file.name}:${file.size}:${file.lastModified}`
        if (seen.has(key)) continue
        seen.add(key)
        next.push({
          key,
          file,
          name: file.name,
          selected: true,
          stem: file.name.replace(/\.pdf$/i, ''),
          status: 'queued',
          failureStage: '',
          displayName: file.name,
          note: '',
          savedName: '',
          savedSha1: '',
          taskId: '',
          convertRequested: false,
          suggestionBasisLabel: '',
          suggestionBasisDetail: '',
          suggestionMatchMethod: '',
          suggestionYearSource: '',
        })
      }
      return next
    })
  }, [])

  const inspectDraft = useCallback(async (key: string, opts?: { useLlm?: boolean }): Promise<{
    ok: boolean
    duplicate: boolean
    suggestedStem: string
  }> => {
    const ready = await ensureDirsReady()
    if (!ready) return { ok: false, duplicate: false, suggestedStem: '' }
    const target = uploadDrafts.find((x) => x.key === key)
    if (!target) return { ok: false, duplicate: false, suggestedStem: '' }
    const effectiveUseLlm = Boolean(opts?.useLlm ?? uploadUseLlm)
    setUploadDrafts((cur) => cur.map((x) => (x.key === key ? { ...x, status: 'inspecting', failureStage: '', note: '' } : x)))
    try {
      const res = await libraryApi.inspectUpload(target.file, effectiveUseLlm)
      const suggestedStem = String(res.suggested_stem || target.stem || '')
      setUploadDrafts((cur) => cur.map((x) => {
        if (x.key !== key) return x
        return {
          ...x,
          stem: suggestedStem || x.stem,
          displayName: res.display_full_name || x.displayName,
          suggestionBasisLabel: String(res.meta?.basis_label || ''),
          suggestionBasisDetail: String(res.meta?.basis_detail || ''),
          suggestionMatchMethod: String(res.meta?.match_method || ''),
          suggestionYearSource: String(res.meta?.year_source || ''),
          status: res.duplicate ? 'error' : 'ready',
          failureStage: res.duplicate ? 'duplicate' : '',
          note: res.duplicate ? `${S.lib_upload_dup_prefix}${String(res.existing || '')}` : S.lib_upload_scan_done,
        }
      }))
      return { ok: !res.duplicate, duplicate: Boolean(res.duplicate), suggestedStem }
    } catch (err) {
      setUploadDrafts((cur) => cur.map((x) => (
        x.key === key
          ? { ...x, status: 'error', failureStage: 'inspect', note: err instanceof Error ? err.message : S.lib_upload_scan_fail }
          : x
      )))
      return { ok: false, duplicate: false, suggestedStem: '' }
    }
  }, [ensureDirsReady, uploadDrafts, uploadUseLlm, S])

  const inspectSelectedDrafts = useCallback(async () => {
    const selected = uploadDrafts.filter((x) => x.selected && x.status !== 'inspecting')
    if (!selected.length) {
      message.info(S.lib_msg_select_scan)
      return
    }
    const effectiveUseLlm = uploadUseLlm && textModelReady
    if (uploadUseLlm && !textModelReady) {
      warnLlmFallback(S.lib_upload_use_llm)
    }
    setUploadInspecting(true)
    try {
      for (const x of selected) {
        await inspectDraft(x.key, { useLlm: effectiveUseLlm })
      }
      message.success(S.lib_msg_scanned_count.replace('{n}', String(selected.length)))
    } finally {
      setUploadInspecting(false)
    }
  }, [S, inspectDraft, textModelReady, uploadDrafts, uploadUseLlm, warnLlmFallback])

  const inspectSingleDraft = useCallback((key: string) => {
    const effectiveUseLlm = uploadUseLlm && textModelReady
    if (uploadUseLlm && !textModelReady) {
      warnLlmFallback(S.lib_upload_use_llm)
    }
    void inspectDraft(key, { useLlm: effectiveUseLlm })
  }, [S.lib_upload_use_llm, inspectDraft, textModelReady, uploadUseLlm, warnLlmFallback])

  const saveDraft = useCallback(async (key: string, convertNow: boolean, opts?: { syncUi?: boolean; baseName?: string }) => {
    const syncUi = opts?.syncUi ?? true
    const ready = await ensureDirsReady()
    if (!ready) return { saved: false, enqueued: false }
    const target = uploadDrafts.find((x) => x.key === key)
    if (!target) return { saved: false, enqueued: false }
    setUploadDrafts((cur) => cur.map((x) => (
      x.key === key
        ? {
          ...x,
          status: 'saving',
          failureStage: '',
          note: '',
          savedName: '',
          savedSha1: '',
          taskId: '',
          convertRequested: false,
        }
        : x
    )))
    try {
      const res = await libraryApi.commitUpload(target.file, {
        baseName: opts?.baseName ?? target.stem,
        convertNow,
        speedMode: convertMode,
        allowDuplicate: false,
      })
      const savedName = String(res.name || target.file.name || '')
      const enqueued = Boolean(convertNow && res.enqueued)
      setUploadDrafts((cur) => cur.map((x) => {
        if (x.key !== key) return x
        if (res.duplicate) {
          return {
            ...x,
            status: 'error',
            failureStage: 'duplicate',
            note: `${S.lib_upload_dup_prefix}${String(res.existing || '')}`,
          }
        }
        return {
          ...x,
          status: 'saved',
          failureStage: '',
          selected: false,
          stem: savedName.replace(/\.pdf$/i, '') || x.stem,
          displayName: savedName || x.displayName,
          savedName,
          savedSha1: String(res.sha1 || ''),
          taskId: String(res.task_id || ''),
          convertRequested: enqueued,
          note: enqueued ? S.lib_msg_saved_enqueued.replace('{name}', savedName) : S.lib_msg_saved_only.replace('{name}', savedName),
        }
      }))
      if (res.duplicate) return { saved: false, enqueued: false }
      if (syncUi) {
        await loadFiles(scope)
        if (enqueued) startProgressStream()
      }
      return { saved: true, enqueued }
    } catch (err) {
      setUploadDrafts((cur) => cur.map((x) => (
        x.key === key
          ? { ...x, status: 'error', failureStage: 'save', note: err instanceof Error ? err.message : S.lib_upload_save_fail }
          : x
      )))
      return { saved: false, enqueued: false }
    }
  }, [S, convertMode, ensureDirsReady, loadFiles, scope, startProgressStream, uploadDrafts])

  const saveSelectedDrafts = useCallback(async (convertNow: boolean) => {
    const ready = await ensureDirsReady()
    if (!ready) return
    const selected = uploadDrafts.filter((x) => x.selected && x.status !== 'saving' && x.status !== 'saved')
    if (!selected.length) {
      message.info(S.lib_msg_select_save)
      return
    }
    setUploadSaving(true)
    try {
      let anyEnqueued = false
      for (const x of selected) {
        const result = await saveDraft(x.key, convertNow, { syncUi: false })
        anyEnqueued = anyEnqueued || Boolean(result.enqueued)
      }
      await loadFiles(scope)
      if (anyEnqueued) startProgressStream()
      message.success(S.lib_msg_processed_count.replace('{n}', String(selected.length)))
    } finally {
      setUploadSaving(false)
    }
  }, [S.lib_msg_processed_count, S.lib_msg_select_save, ensureDirsReady, loadFiles, saveDraft, scope, startProgressStream, uploadDrafts])

  const selectFailedDrafts = useCallback(() => {
    if (!failedUploadDrafts.length) {
      message.info(S.lib_msg_no_failed_items)
      return
    }
    setUploadDrafts((cur) => cur.map((x) => ({ ...x, selected: x.status === 'error' })))
    message.info(S.lib_msg_selected_failed.replace('{n}', String(failedUploadDrafts.length)))
  }, [S, failedUploadDrafts.length])

  const showDuplicateFailedDrafts = useCallback(() => {
    if (!duplicateFailedDrafts.length) {
      message.info(S.lib_msg_no_dup_failures)
      return
    }
    applyUploadFilter('dup_error')
    message.info(S.lib_msg_switched_dup.replace('{n}', String(duplicateFailedDrafts.length)))
  }, [S, applyUploadFilter, duplicateFailedDrafts.length])

  const retryFailedDrafts = useCallback(async (convertNow: boolean) => {
    const failed = uploadDrafts.filter((x) => x.status === 'error')
    const retryable = failed.filter((x) => x.failureStage !== 'duplicate' && !isDuplicateFailure(x.note))
    if (!retryable.length) {
      message.info(S.lib_msg_no_retryable)
      return
    }
    setUploadSaving(true)
    setUploadInspecting(true)
    try {
      let anyEnqueued = false
      const effectiveUseLlm = uploadUseLlm && textModelReady
      if (uploadUseLlm && !textModelReady) {
        warnLlmFallback(S.lib_upload_use_llm)
      }
      for (const x of retryable) {
        if (x.failureStage === 'inspect') {
          const inspectResult = await inspectDraft(x.key, { useLlm: effectiveUseLlm })
          if (!inspectResult.ok || !convertNow) continue
          const result = await saveDraft(x.key, true, {
            syncUi: false,
            baseName: inspectResult.suggestedStem || x.stem,
          })
          anyEnqueued = anyEnqueued || Boolean(result.enqueued)
          continue
        }
        const result = await saveDraft(x.key, convertNow, { syncUi: false })
        anyEnqueued = anyEnqueued || Boolean(result.enqueued)
      }
      await loadFiles(scope)
      if (anyEnqueued) startProgressStream()
      message.success(S.lib_msg_retried_count.replace('{n}', String(retryable.length)))
    } finally {
      setUploadInspecting(false)
      setUploadSaving(false)
    }
  }, [S, inspectDraft, loadFiles, saveDraft, scope, startProgressStream, textModelReady, uploadDrafts, uploadUseLlm, warnLlmFallback])

  const clearSavedDrafts = useCallback(() => {
    setUploadDrafts((cur) => cur.filter((x) => x.status !== 'saved'))
  }, [])

  const selectFailedReason = useCallback((reason: Exclude<UploadErrorReason, 'all'>) => {
    applyUploadFilter('error')
    setUploadErrorReason(reason)
  }, [applyUploadFilter])

  const setDraftSelected = useCallback((key: string, selected: boolean) => {
    setUploadDrafts((cur) => cur.map((x) => (x.key === key ? { ...x, selected } : x)))
  }, [])

  const setDraftStem = useCallback((key: string, stem: string) => {
    setUploadDrafts((cur) => cur.map((x) => (x.key === key ? { ...x, stem } : x)))
  }, [])

  const selectAllUploadDrafts = useCallback(() => {
    setUploadDrafts((cur) => cur.map((item) => ({ ...item, selected: true })))
  }, [])

  const invertUploadDraftSelection = useCallback(() => {
    setUploadDrafts((cur) => cur.map((item) => ({ ...item, selected: !item.selected })))
  }, [])

  useEffect(() => {
    if (uploadDrafts.length === 0) {
      setUploadWorkbenchOpen(false)
      return
    }
    setUploadWorkbenchOpen(true)
  }, [uploadDrafts.length])

  useEffect(() => {
    if (uploadLocked || dirDirty || uploadInspecting || autoInspectingRef.current) return
    const queuedKeys = uploadDrafts
      .filter((x) => x.status === 'queued')
      .map((x) => x.key)
    if (!queuedKeys.length) return

    autoInspectingRef.current = true
    setUploadInspecting(true)

    void (async () => {
      try {
        const effectiveUseLlm = uploadUseLlm && textModelReady
        if (uploadUseLlm && !textModelReady) {
          warnLlmFallback(S.lib_upload_use_llm)
        }
        for (const key of queuedKeys) {
          // Auto-fill suggested names for newly added upload drafts.
          await inspectDraft(key, { useLlm: effectiveUseLlm })
        }
      } finally {
        autoInspectingRef.current = false
        setUploadInspecting(false)
      }
    })()
  }, [S.lib_upload_use_llm, dirDirty, inspectDraft, textModelReady, uploadDrafts, uploadInspecting, uploadLocked, uploadUseLlm, warnLlmFallback])

  useEffect(() => {
    setUploadDrafts((cur) => {
      const next = cur.filter((draft) => !isUploadDraftConverted(draft, files))
      return next.length === cur.length ? cur : next
    })
  }, [files])

  useEffect(() => {
    if (uploadDraftPage > uploadDraftPageCount) setUploadDraftPage(uploadDraftPageCount)
  }, [uploadDraftPage, uploadDraftPageCount])

  return {
    addDrafts,
    applyUploadFilter,
    clearSavedDrafts,
    duplicateFailedDrafts,
    failedReasonBuckets,
    failedUploadDrafts,
    failedUploadNotes,
    filteredUploadDrafts,
    inspectSelectedDrafts,
    inspectSingleDraft,
    invertUploadDraftSelection,
    pagedUploadDrafts,
    retryFailedDrafts,
    retryableFailedUploadDrafts,
    saveDraft,
    saveSelectedDrafts,
    selectAllUploadDrafts,
    selectFailedDrafts,
    selectFailedReason,
    selectedUploadCount,
    setDraftSelected,
    setDraftStem,
    setUploadDraftPage,
    setUploadErrorReason,
    setUploadUseLlm,
    setUploadWorkbenchOpen,
    showDuplicateFailedDrafts,
    uploadDraftFilter,
    uploadDraftFilterOptions,
    uploadDraftPage,
    uploadDrafts,
    uploadErrorReason,
    uploadInspecting,
    uploadSaving,
    uploadUseLlm,
    uploadWorkbenchOpen,
  }
}
