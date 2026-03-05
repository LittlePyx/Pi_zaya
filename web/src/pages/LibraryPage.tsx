
import { useEffect, useMemo, useState, type ReactNode } from 'react'
import {
  Upload,
  Button,
  List,
  message,
  Progress,
  Select,
  Typography,
  Tabs,
  Tag,
  Switch,
  Space,
  Empty,
  Input,
  Card,
  Checkbox,
  Collapse,
  Alert,
  Tooltip,
  Dropdown,
  Modal,
} from 'antd'
import {
  UploadOutlined,
  ReloadOutlined,
  StopOutlined,
  FolderOpenOutlined,
  DeleteOutlined,
  SaveOutlined,
  SearchOutlined,
  CheckOutlined,
  ClearOutlined,
  MoreOutlined,
  CopyOutlined,
  LockOutlined,
  ApiOutlined,
  ExclamationCircleOutlined,
} from '@ant-design/icons'
import type { LibraryFileItem, RenameSuggestionItem } from '../api/library'
import { libraryApi } from '../api/library'
import { settingsApi } from '../api/settings'
import { useLibraryStore } from '../stores/libraryStore'
import { useSettingsStore } from '../stores/settingsStore'
import VirtualList from 'rc-virtual-list'

const { Text } = Typography
const { Dragger } = Upload
const FILE_VIRTUAL_THRESHOLD = 60
const FILE_VIRTUAL_HEIGHT = 620
const FILE_VIRTUAL_ROW_HEIGHT = 88

type FileTabKey = 'pending' | 'converted' | 'all'
type DraftStatus = 'queued' | 'inspecting' | 'ready' | 'saving' | 'saved' | 'error'
type UploadDraftFilter = 'all' | 'todo' | 'error' | 'dup_error' | 'saved'
type UploadErrorReason = 'all' | 'duplicate' | 'path' | 'permission' | 'network' | 'other'

type UploadDraft = {
  key: string
  file: File
  name: string
  selected: boolean
  stem: string
  status: DraftStatus
  displayName: string
  note: string
}

const CONVERT_MODE = 'balanced'

const SCOPE_OPTIONS = [
  { value: '200', label: '最近 200 篇' },
  { value: '1000', label: '最近 1000 篇' },
  { value: 'all', label: '全部' },
]

const RENAME_SCOPE_OPTIONS = [
  { value: '30', label: '最近 30 篇' },
  { value: '50', label: '最近 50 篇' },
  { value: '100', label: '最近 100 篇' },
  { value: 'all', label: '全部' },
]

const DRAFT_STATUS_TEXT: Record<DraftStatus, string> = {
  queued: '待处理',
  inspecting: '扫描中',
  ready: '待保存',
  saving: '保存中',
  saved: '已保存',
  error: '失败',
}

const FAILED_REASON_META: Record<Exclude<UploadErrorReason, 'all'>, { label: string, icon: ReactNode }> = {
  duplicate: { label: '重复文件', icon: <CopyOutlined /> },
  path: { label: '路径/目录', icon: <FolderOpenOutlined /> },
  permission: { label: '权限', icon: <LockOutlined /> },
  network: { label: '网络', icon: <ApiOutlined /> },
  other: { label: '其他', icon: <ExclamationCircleOutlined /> },
}

function fileTag(item: LibraryFileItem) {
  if (item.task_state === 'running') return { color: 'processing' as const, text: '转换中' }
  if (item.task_state === 'queued') return { color: 'warning' as const, text: `排队中${item.queue_pos > 0 ? ` #${item.queue_pos}` : ''}` }
  return item.category === 'converted'
    ? { color: 'success' as const, text: '已转换' }
    : { color: 'default' as const, text: '待转换' }
}

function matchesKeyword(name: string, keyword: string) {
  if (!keyword) return true
  return name.toLowerCase().includes(keyword)
}

function isDuplicateFailure(note: string) {
  const t = String(note || '').toLowerCase()
  return t.includes('重复') || t.includes('duplicate') || t.includes('already exists') || t.includes('已存在')
}

function classifyFailedReason(note: string) {
  const t = String(note || '').toLowerCase()
  if (isDuplicateFailure(t)) return 'duplicate'
  if (t.includes('目录') || t.includes('路径') || t.includes('path') || t.includes('dir')) return 'path'
  if (t.includes('权限') || t.includes('permission') || t.includes('denied')) return 'permission'
  if (t.includes('网络') || t.includes('timeout') || t.includes('network')) return 'network'
  return 'other'
}

export default function LibraryPage() {
  const store = useLibraryStore()

  const settingsLoaded = useSettingsStore((s) => s.loaded)
  const settingsPdfDir = useSettingsStore((s) => s.pdfDir)
  const settingsMdDir = useSettingsStore((s) => s.mdDir)
  const updateSettings = useSettingsStore((s) => s.update)

  const [scope, setScope] = useState('200')
  const [tabKey, setTabKey] = useState<FileTabKey>('pending')
  const [replaceMd, setReplaceMd] = useState(true)
  const [onlyBusyFiles, setOnlyBusyFiles] = useState(false)
  const [fileKeyword, setFileKeyword] = useState('')

  const [pdfDirDraft, setPdfDirDraft] = useState('')
  const [mdDirDraft, setMdDirDraft] = useState('')
  const [savingDirs, setSavingDirs] = useState(false)
  const [pickingDir, setPickingDir] = useState<'pdf' | 'md' | null>(null)
  const [dirTouched, setDirTouched] = useState(false)

  const [uploadDrafts, setUploadDrafts] = useState<UploadDraft[]>([])
  const [uploadUseLlm, setUploadUseLlm] = useState(true)
  const [uploadDraftFilter, setUploadDraftFilter] = useState<UploadDraftFilter>('all')
  const [uploadErrorReason, setUploadErrorReason] = useState<UploadErrorReason>('all')
  const [uploadInspecting, setUploadInspecting] = useState(false)
  const [uploadSaving, setUploadSaving] = useState(false)

  const [renameScope, setRenameScope] = useState('30')
  const [renameUseLlm, setRenameUseLlm] = useState(false)
  const [renameOnlyDiff, setRenameOnlyDiff] = useState(true)
  const [renameLoading, setRenameLoading] = useState(false)
  const [renameApplying, setRenameApplying] = useState(false)
  const [renameItems, setRenameItems] = useState<RenameSuggestionItem[]>([])
  const [renameSelected, setRenameSelected] = useState<Record<string, boolean>>({})
  const [renameOverrides, setRenameOverrides] = useState<Record<string, string>>({})

  const uploadLocked = store.converting || Boolean(store.refSync?.running)
  const normalizedKeyword = fileKeyword.trim().toLowerCase()

  const dirDirty = useMemo(
    () =>
      pdfDirDraft.trim() !== String(settingsPdfDir || '').trim()
      || mdDirDraft.trim() !== String(settingsMdDir || '').trim(),
    [pdfDirDraft, mdDirDraft, settingsPdfDir, settingsMdDir],
  )

  const pendingFiles = useMemo(() => store.files.filter((x) => x.category === 'pending'), [store.files])
  const convertedFiles = useMemo(() => store.files.filter((x) => x.category === 'converted'), [store.files])
  const renameVisible = useMemo(() => (renameOnlyDiff ? renameItems.filter((x) => x.diff) : renameItems), [renameOnlyDiff, renameItems])
  const selectedUploadCount = useMemo(() => uploadDrafts.filter((x) => x.selected).length, [uploadDrafts])
  const selectedRenameCount = useMemo(() => renameItems.filter((x) => renameSelected[x.name]).length, [renameItems, renameSelected])
  const failedUploadDrafts = useMemo(() => uploadDrafts.filter((x) => x.status === 'error'), [uploadDrafts])
  const duplicateFailedDrafts = useMemo(
    () => failedUploadDrafts.filter((x) => isDuplicateFailure(x.note)),
    [failedUploadDrafts],
  )
  const failedUploadNotes = useMemo(
    () => Array.from(new Set(failedUploadDrafts.map((x) => String(x.note || '').trim()).filter(Boolean))).slice(0, 3),
    [failedUploadDrafts],
  )
  const failedReasonBuckets = useMemo(() => {
    const counter = new Map<Exclude<UploadErrorReason, 'all'>, number>()
    for (const item of failedUploadDrafts) {
      const key = classifyFailedReason(item.note) as Exclude<UploadErrorReason, 'all'>
      counter.set(key, (counter.get(key) || 0) + 1)
    }
    return Array.from(counter.entries())
      .map(([key, count]) => ({ key, count, label: FAILED_REASON_META[key].label }))
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
      { value: 'all', label: `全部 (${uploadDrafts.length})` },
      { value: 'todo', label: `待处理 (${uploadDrafts.filter((x) => ['queued', 'inspecting', 'ready', 'saving'].includes(x.status)).length})` },
      { value: 'error', label: `失败 (${uploadDrafts.filter((x) => x.status === 'error').length})` },
      { value: 'dup_error', label: `重复失败 (${uploadDrafts.filter((x) => x.status === 'error' && isDuplicateFailure(x.note)).length})` },
      { value: 'saved', label: `已保存 (${uploadDrafts.filter((x) => x.status === 'saved').length})` },
    ],
    [uploadDrafts],
  )
  const activeErrorReasonText = useMemo(() => {
    const map: Record<UploadErrorReason, string> = {
      all: '全部原因',
      duplicate: FAILED_REASON_META.duplicate.label,
      path: FAILED_REASON_META.path.label,
      permission: FAILED_REASON_META.permission.label,
      network: FAILED_REASON_META.network.label,
      other: FAILED_REASON_META.other.label,
    }
    return map[uploadErrorReason]
  }, [uploadErrorReason])
  const convertPercent = useMemo(
    () => (store.progress && store.progress.total > 0
      ? Math.round((store.progress.completed / store.progress.total) * 100)
      : 0),
    [store.progress],
  )
  const refSyncPercent = useMemo(
    () => (store.refSync && store.refSync.docsTotal > 0
      ? Math.round((store.refSync.docsDone / Math.max(1, store.refSync.docsTotal)) * 100)
      : 0),
    [store.refSync],
  )
  const showStickyStatus = Boolean((store.converting && store.progress) || store.refSync?.running)

  const filterFiles = (items: LibraryFileItem[]) =>
    items.filter((item) => {
      if (!matchesKeyword(item.name, normalizedKeyword)) return false
      if (onlyBusyFiles) return item.task_state !== 'idle'
      return true
    })

  const visiblePending = useMemo(() => filterFiles(pendingFiles), [pendingFiles, normalizedKeyword, onlyBusyFiles])
  const visibleConverted = useMemo(() => filterFiles(convertedFiles), [convertedFiles, normalizedKeyword, onlyBusyFiles])
  const visibleAll = useMemo(() => filterFiles(store.files), [store.files, normalizedKeyword, onlyBusyFiles])

  useEffect(() => {
    void store.loadFiles(scope)
    if (store.converting && !store.sseController) store.startProgressStream()
    if (!store.refSyncController) store.startRefSyncStream()
    return () => {
      store.stopProgressStream()
      store.stopRefSyncStream()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!settingsLoaded || dirTouched) return
    setPdfDirDraft(String(settingsPdfDir || ''))
    setMdDirDraft(String(settingsMdDir || ''))
  }, [settingsLoaded, settingsPdfDir, settingsMdDir, dirTouched])

  const saveDirs = async () => {
    if (!pdfDirDraft.trim() || !mdDirDraft.trim()) {
      message.warning('PDF/MD 目录不能为空')
      return false
    }
    setSavingDirs(true)
    try {
      await updateSettings({ pdfDir: pdfDirDraft.trim(), mdDir: mdDirDraft.trim() })
      setDirTouched(false)
      message.success('目录已保存')
      await store.loadFiles(scope)
      return true
    } catch (err) {
      message.error(err instanceof Error ? err.message : '目录保存失败')
      return false
    } finally {
      setSavingDirs(false)
    }
  }

  const ensureDirsReady = async () => {
    if (!dirDirty) return true
    return saveDirs()
  }

  const openFolder = async (target: 'pdf_dir' | 'md_dir') => {
    const ready = await ensureDirsReady()
    if (!ready) return
    await store.openFile('', target)
  }

  const pickDir = async (target: 'pdf' | 'md') => {
    const initial = target === 'pdf' ? pdfDirDraft : mdDirDraft
    setPickingDir(target)
    try {
      const res = await settingsApi.pickDir(target, initial)
      if (!res.ok || !res.path) {
        message.info('未选择目录')
        return
      }
      setDirTouched(true)
      if (target === 'pdf') setPdfDirDraft(res.path)
      else setMdDirDraft(res.path)
    } catch (err) {
      message.error(err instanceof Error ? err.message : '打开目录选择器失败')
    } finally {
      setPickingDir(null)
    }
  }

  const addDrafts = (files: File[]) => {
    setUploadDrafts((cur) => {
      const seen = new Set(cur.map((x) => x.key))
      const next = [...cur]
      for (const file of files) {
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
          displayName: file.name,
          note: '',
        })
      }
      return next
    })
  }

  const inspectDraft = async (key: string) => {
    const ready = await ensureDirsReady()
    if (!ready) return
    const target = uploadDrafts.find((x) => x.key === key)
    if (!target) return
    setUploadDrafts((cur) => cur.map((x) => (x.key === key ? { ...x, status: 'inspecting', note: '' } : x)))
    try {
      const res = await libraryApi.inspectUpload(target.file, uploadUseLlm)
      setUploadDrafts((cur) => cur.map((x) => {
        if (x.key !== key) return x
        return {
          ...x,
          stem: res.suggested_stem || x.stem,
          displayName: res.display_full_name || x.displayName,
          status: res.duplicate ? 'error' : 'ready',
          note: res.duplicate ? `重复：${String(res.existing || '')}` : '扫描完成',
        }
      }))
    } catch (err) {
      setUploadDrafts((cur) => cur.map((x) => (
        x.key === key
          ? { ...x, status: 'error', note: err instanceof Error ? err.message : '扫描失败' }
          : x
      )))
    }
  }

  const inspectSelectedDrafts = async () => {
    const selected = uploadDrafts.filter((x) => x.selected && x.status !== 'inspecting')
    if (!selected.length) {
      message.info('请先选择要扫描的文件')
      return
    }
    setUploadInspecting(true)
    try {
      for (const x of selected) {
        // eslint-disable-next-line no-await-in-loop
        await inspectDraft(x.key)
      }
      message.success(`已扫描 ${selected.length} 个文件`)
    } finally {
      setUploadInspecting(false)
    }
  }

  const saveDraft = async (key: string, convertNow: boolean) => {
    const ready = await ensureDirsReady()
    if (!ready) return
    const target = uploadDrafts.find((x) => x.key === key)
    if (!target) return
    setUploadDrafts((cur) => cur.map((x) => (x.key === key ? { ...x, status: 'saving', note: '' } : x)))
    try {
      const res = await libraryApi.commitUpload(target.file, {
        baseName: target.stem,
        convertNow,
        speedMode: CONVERT_MODE,
        allowDuplicate: false,
      })
      setUploadDrafts((cur) => cur.map((x) => {
        if (x.key !== key) return x
        if (res.duplicate) return { ...x, status: 'error', note: `重复：${String(res.existing || '')}` }
        return {
          ...x,
          status: 'saved',
          selected: false,
          note: convertNow && res.enqueued ? '已保存并加入转换队列' : '已保存',
        }
      }))
    } catch (err) {
      setUploadDrafts((cur) => cur.map((x) => (
        x.key === key
          ? { ...x, status: 'error', note: err instanceof Error ? err.message : '保存失败' }
          : x
      )))
    }
  }

  const saveSelectedDrafts = async (convertNow: boolean) => {
    const ready = await ensureDirsReady()
    if (!ready) return
    const selected = uploadDrafts.filter((x) => x.selected && x.status !== 'saving' && x.status !== 'saved')
    if (!selected.length) {
      message.info('请先选择要保存的文件')
      return
    }
    setUploadSaving(true)
    try {
      for (const x of selected) {
        // eslint-disable-next-line no-await-in-loop
        await saveDraft(x.key, convertNow)
      }
      await store.loadFiles(scope)
      message.success(`已处理 ${selected.length} 个文件`)
    } finally {
      setUploadSaving(false)
    }
  }

  const scanRenameSuggestions = async () => {
    setRenameLoading(true)
    try {
      const res = await libraryApi.listRenameSuggestions(renameScope, renameUseLlm)
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
      message.success(`扫描完成：${res.changed}/${res.total_scanned} 需要改名`)
    } catch (err) {
      message.error(err instanceof Error ? err.message : '扫描改名建议失败')
    } finally {
      setRenameLoading(false)
    }
  }

  const selectFailedDrafts = () => {
    if (!failedUploadDrafts.length) {
      message.info('暂无失败项')
      return
    }
    setUploadDrafts((cur) => cur.map((x) => ({ ...x, selected: x.status === 'error' })))
    message.info(`已选择 ${failedUploadDrafts.length} 个失败项`)
  }

  const showDuplicateFailedDrafts = () => {
    if (!duplicateFailedDrafts.length) {
      message.info('当前没有重复文件失败项')
      return
    }
    applyUploadFilter('dup_error')
    message.info(`已切换到重复失败项（${duplicateFailedDrafts.length}）`)
  }

  const retryFailedDrafts = async (convertNow: boolean) => {
    const failed = uploadDrafts.filter((x) => x.status === 'error')
    if (!failed.length) {
      message.info('没有可重试的失败项')
      return
    }
    setUploadSaving(true)
    try {
      for (const x of failed) {
        // eslint-disable-next-line no-await-in-loop
        await saveDraft(x.key, convertNow)
      }
      await store.loadFiles(scope)
      message.success(`已重试 ${failed.length} 个失败项`)
    } finally {
      setUploadSaving(false)
    }
  }

  const applyRenameSuggestions = async () => {
    const names = renameItems.filter((x) => renameSelected[x.name]).map((x) => x.name)
    if (!names.length) {
      message.info('请先选择要改名的条目')
      return
    }
    setRenameApplying(true)
    try {
      const overrides: Record<string, string> = {}
      for (const name of names) overrides[name] = String(renameOverrides[name] || '').trim()
      const res = await libraryApi.applyRenameSuggestions(names, overrides, { useLlm: renameUseLlm, alsoMd: true })
      message[res.failed > 0 ? 'warning' : 'success'](`改名完成：成功 ${res.renamed}，跳过 ${res.skipped}，失败 ${res.failed}`)
      if (res.needs_reindex) message.info('改名后建议更新知识库')
      await store.loadFiles(scope)
      await scanRenameSuggestions()
    } catch (err) {
      message.error(err instanceof Error ? err.message : '应用改名失败')
    } finally {
      setRenameApplying(false)
    }
  }

  const handleConvertPending = async () => {
    const res = await store.convertPending(CONVERT_MODE)
    message[res.enqueued > 0 ? 'success' : 'info'](
      res.enqueued > 0
        ? `已加入队列 ${res.enqueued} 个待转换文件`
        : '没有可入队的待转换文件',
    )
    await store.loadFiles(scope)
  }

  const handleConvertOne = async (item: LibraryFileItem) => {
    if (item.task_state !== 'idle') return
    const replace = item.md_exists ? replaceMd : false
    if (item.md_exists && !replaceMd) {
      message.info('请先开启“覆盖已有 Markdown”再重新转换')
      return
    }
    await store.convert(item.name, CONVERT_MODE, replace)
  }

  const handleDeleteOne = async (item: LibraryFileItem) => {
    const res = await store.deleteFile(item.name, true)
    if (res.ok) {
      message.success(`已删除 ${item.name}`)
      if (res.needs_reindex) {
        message.info('删除/改名后建议更新知识库')
      }
      return
    }
    const warning = Array.isArray(res.warnings) && res.warnings.length > 0
      ? `（${res.warnings.join('；')}）`
      : ''
    message.warning(`删除未完全成功${warning}`)
  }

  const confirmDeleteOne = (item: LibraryFileItem) => {
    Modal.confirm({
      title: '确认删除这个文献吗？',
      content: item.name,
      okText: '删除',
      okType: 'danger',
      cancelText: '取消',
      onOk: async () => {
        await handleDeleteOne(item)
      },
    })
  }

  const handleReindex = async () => {
    const hide = message.loading('正在更新知识库...', 0)
    try {
      const res = await store.reindex()
      hide()
      if (!res.ok) {
        message.error('执行失败')
        return
      }
      message.success('执行完成')
      if (res.refsync_error) {
        message.warning(`引用同步启动失败：${res.refsync_error}`)
      } else if (res.refsync?.started) {
        message.info('已在后台启动引用同步')
      }
    } catch (err) {
      hide()
      message.error(err instanceof Error ? err.message : '执行失败')
    }
  }

  const handleStartRefSync = async () => {
    const hide = message.loading('正在启动引用同步...', 0)
    try {
      const res = await store.startReferenceSync()
      hide()
      if (res.started) {
        message.success('引用同步已启动')
      } else if (res.reason === 'running') {
        message.info('引用同步已在运行')
      } else {
        message.warning('引用同步未启动')
      }
    } catch (err) {
      hide()
      message.error(err instanceof Error ? err.message : '启动引用同步失败')
    }
  }

  const selectAllUploadDrafts = () => {
    setUploadDrafts((cur) => cur.map((item) => ({ ...item, selected: true })))
  }

  const invertUploadDraftSelection = () => {
    setUploadDrafts((cur) => cur.map((item) => ({ ...item, selected: !item.selected })))
  }

  const selectRenameDiffItems = () => {
    setRenameSelected((cur) => {
      const next = { ...cur }
      for (const item of renameItems) {
        next[item.name] = Boolean(item.diff)
      }
      return next
    })
  }

  const clearRenameSelection = () => {
    setRenameSelected((cur) => {
      const next = { ...cur }
      for (const item of renameItems) {
        next[item.name] = false
      }
      return next
    })
  }

  const applyUploadFilter = (next: UploadDraftFilter) => {
    setUploadDraftFilter(next)
    if (next === 'dup_error') {
      setUploadErrorReason('duplicate')
      return
    }
    if (next !== 'error') {
      setUploadErrorReason('all')
    }
  }

  const renderFileRow = (item: LibraryFileItem) => {
    const tag = fileTag(item)
    return (
      <div className="kb-lib-file-row flex w-full flex-wrap items-center justify-between gap-3">
        <div className="kb-lib-file-main min-w-0 flex-1">
          <div className="mb-1 flex items-center gap-2">
            <Text className="min-w-0 truncate text-sm font-medium">{item.name}</Text>
            <Tag color={tag.color}>{tag.text}</Tag>
          </div>
          <Text type="secondary" className="text-xs">{item.md_exists ? '已存在 Markdown' : '暂无 Markdown'}</Text>
        </div>
        <Space className="kb-lib-file-actions">
          <Button
            className="kb-lib-file-action-main"
            size="small"
            type="primary"
            ghost
            disabled={item.task_state !== 'idle'}
            onClick={() => { void handleConvertOne(item) }}
          >
            {item.md_exists ? '重新转换' : '转换'}
          </Button>
          <Button className="kb-lib-file-action-main" size="small" onClick={() => { void store.openFile(item.name, 'pdf') }}>
            打开 PDF
          </Button>
          <Dropdown
            trigger={['click']}
            menu={{
              items: [
                { key: 'open-md', label: '打开 MD', disabled: !item.md_exists },
                { type: 'divider' },
                { key: 'delete', label: '删除文献', danger: true, disabled: item.task_state !== 'idle', icon: <DeleteOutlined /> },
              ],
              onClick: ({ key }) => {
                if (key === 'open-md') {
                  void store.openFile(item.name, 'md')
                  return
                }
                if (key === 'delete') {
                  confirmDeleteOne(item)
                }
              },
            }}
          >
            <Button size="small" className="kb-lib-file-more-btn" icon={<MoreOutlined />} />
          </Dropdown>
        </Space>
      </div>
    )
  }

  const renderFiles = (items: LibraryFileItem[], emptyText: string) => {
    if (!items.length) {
      return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={emptyText} />
    }

    if (items.length < FILE_VIRTUAL_THRESHOLD) {
      return (
        <List
          className="kb-lib-file-list"
          size="small"
          dataSource={items}
          renderItem={(item) => (
            <List.Item className="kb-lib-file-item">
              {renderFileRow(item)}
            </List.Item>
          )}
        />
      )
    }

    return (
      <div className="kb-lib-file-virtual-shell">
        <div className="kb-lib-file-virtual-tip">
          <Text type="secondary" className="text-xs">已启用虚拟滚动（{items.length} 条）</Text>
        </div>
        <VirtualList
          data={items}
          itemKey="name"
          height={FILE_VIRTUAL_HEIGHT}
          itemHeight={FILE_VIRTUAL_ROW_HEIGHT}
        >
          {(item: LibraryFileItem) => (
            <div className="ant-list-item kb-lib-file-item kb-lib-file-virtual-item">
              {renderFileRow(item)}
            </div>
          )}
        </VirtualList>
      </div>
    )
  }

  const counts = store.fileCounts || {
    total_view: store.files.length,
    total_all: store.files.length,
    pending: pendingFiles.length,
    converted: convertedFiles.length,
    queued: store.files.filter((x) => x.task_state === 'queued').length,
    running: store.files.filter((x) => x.task_state === 'running').length,
    reconverting: 0,
  }

  return (
    <div className="kb-library-page mx-auto w-full max-w-[1760px] space-y-4 p-5">
      <div className="kb-lib-head flex flex-wrap items-end justify-between gap-3">
        <div className="kb-lib-head-main">
          <Text className="text-2xl font-semibold">文献管理</Text>
          <div>
            <Text type="secondary" className="text-sm">先配置目录，再批量处理，最后在列表逐条检查。</Text>
          </div>
        </div>
        <Space wrap className="kb-lib-head-actions">
          <Button className="kb-lib-head-btn" icon={<ReloadOutlined />} type="primary" onClick={() => { void handleReindex() }}>更新知识库</Button>
          <Button className="kb-lib-head-btn" icon={<ReloadOutlined />} onClick={() => { void handleStartRefSync() }}>同步引用信息</Button>
        </Space>
      </div>

      <div className="kb-lib-stats-grid grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
        <Card size="small" className="kb-lib-stat"><Text type="secondary">当前视图</Text><div className="kb-lib-stat-value">{counts.total_view}</div></Card>
        <Card size="small" className="kb-lib-stat"><Text type="secondary">待转换</Text><div className="kb-lib-stat-value">{counts.pending}</div></Card>
        <Card size="small" className="kb-lib-stat"><Text type="secondary">已转换</Text><div className="kb-lib-stat-value">{counts.converted}</div></Card>
        <Card size="small" className="kb-lib-stat"><Text type="secondary">排队中</Text><div className="kb-lib-stat-value">{counts.queued}</div></Card>
        <Card size="small" className="kb-lib-stat"><Text type="secondary">运行中</Text><div className="kb-lib-stat-value">{counts.running}</div></Card>
      </div>

      <div className="kb-lib-ops-grid grid gap-4 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,0.85fr)]">
        <Card size="small" className="kb-lib-card" title="目录设置">
          <div className="space-y-3">
            <div className="grid items-center gap-2 md:grid-cols-[56px_minmax(0,1fr)_auto_auto]">
              <Text className="text-sm font-semibold">PDF</Text>
              <Input
                value={pdfDirDraft}
                placeholder="选择 PDF 文献目录"
                onChange={(e) => {
                  setDirTouched(true)
                  setPdfDirDraft(e.target.value)
                }}
              />
              <Button loading={pickingDir === 'pdf'} onClick={() => { void pickDir('pdf') }}>选择目录</Button>
              <Button icon={<FolderOpenOutlined />} onClick={() => { void openFolder('pdf_dir') }}>打开目录</Button>
            </div>

            <div className="grid items-center gap-2 md:grid-cols-[56px_minmax(0,1fr)_auto_auto]">
              <Text className="text-sm font-semibold">MD</Text>
              <Input
                value={mdDirDraft}
                placeholder="选择 Markdown 输出目录"
                onChange={(e) => {
                  setDirTouched(true)
                  setMdDirDraft(e.target.value)
                }}
              />
              <Button loading={pickingDir === 'md'} onClick={() => { void pickDir('md') }}>选择目录</Button>
              <Button icon={<FolderOpenOutlined />} onClick={() => { void openFolder('md_dir') }}>打开目录</Button>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <Button type="primary" icon={<SaveOutlined />} loading={savingDirs} disabled={!dirDirty} onClick={() => { void saveDirs() }}>
                保存目录设置
              </Button>
              <Text type="secondary" className="text-xs">支持手动输入路径，也支持点击“选择目录”弹窗挑选。</Text>
            </div>
          </div>
        </Card>

        <Card size="small" className="kb-lib-card kb-lib-upload-card" title="上传 PDF">
          <Dragger
            className="kb-lib-upload-drop"
            multiple
            accept=".pdf"
            disabled={uploadLocked}
            showUploadList={false}
            beforeUpload={(file) => {
              addDrafts([file as File])
              return false
            }}
          >
            <p className="text-lg"><UploadOutlined /></p>
            <p>上传 PDF（仅加入队列）</p>
          </Dragger>

          {uploadLocked ? (
            <Text type="secondary" className="text-xs">
              {store.converting ? '转换进行中，上传面板暂时锁定。' : '引用同步进行中，上传面板暂时锁定。'}
            </Text>
          ) : null}
        </Card>
      </div>

      {showStickyStatus ? (
        <Card size="small" className="kb-lib-card kb-lib-sticky-status">
          <div className="kb-lib-sticky-wrap">
            {store.converting && store.progress ? (
              <div className="kb-lib-sticky-item">
                <div className="kb-lib-sticky-main">
                  <Text className="kb-lib-sticky-title">转换中 {store.progress.completed}/{store.progress.total}</Text>
                  {store.progress.current ? <Text type="secondary" className="kb-lib-sticky-sub">{store.progress.current}</Text> : null}
                </div>
                <Progress className="kb-lib-sticky-progress" percent={convertPercent} status="active" size="small" />
                <Button size="small" danger icon={<StopOutlined />} onClick={() => { void store.cancelConvert() }}>
                  停止
                </Button>
              </div>
            ) : null}

            {store.refSync?.running ? (
              <div className="kb-lib-sticky-item">
                <div className="kb-lib-sticky-main">
                  <Text className="kb-lib-sticky-title">引用同步中</Text>
                  <Text type="secondary" className="kb-lib-sticky-sub">
                    {store.refSync.current
                      ? `${store.refSync.stage || '运行中'} | ${store.refSync.current}`
                      : (store.refSync.message || '等待同步任务')}
                  </Text>
                </div>
                <Progress className="kb-lib-sticky-progress" percent={refSyncPercent} status="active" size="small" />
                <Tag color="processing">运行中</Tag>
              </div>
            ) : null}
          </div>
        </Card>
      ) : null}

      <Collapse
        className="kb-lib-collapse"
        size="small"
        items={[
          {
            key: 'upload-workbench',
            label: `上传工作台（${uploadDrafts.length}）`,
            children: (
              <div className="space-y-3">
                <div className="kb-lib-upload-toolbar flex flex-wrap items-center gap-2">
                  <Switch checked={uploadUseLlm} onChange={setUploadUseLlm} />
                  <Text className="text-sm text-[var(--muted)]">使用 LLM 补全信息</Text>
                  <Select
                    value={uploadDraftFilter}
                    onChange={(value) => applyUploadFilter(value as UploadDraftFilter)}
                    options={uploadDraftFilterOptions}
                    className="kb-lib-upload-filter"
                  />
                  <Tooltip title="全选草稿"><Button icon={<CheckOutlined />} onClick={selectAllUploadDrafts}>全选</Button></Tooltip>
                  <Tooltip title="反选草稿"><Button icon={<ClearOutlined />} onClick={invertUploadDraftSelection}>反选</Button></Tooltip>
                  <Button loading={uploadInspecting} disabled={uploadLocked} onClick={() => { void inspectSelectedDrafts() }}>扫描已选</Button>
                  <Button loading={uploadSaving} disabled={uploadLocked} onClick={() => { void saveSelectedDrafts(false) }}>保存已选</Button>
                  <Button type="primary" loading={uploadSaving} disabled={uploadLocked} onClick={() => { void saveSelectedDrafts(true) }}>保存并转换</Button>
                  <Button disabled={uploadLocked} onClick={selectFailedDrafts}>选择失败项</Button>
                  <Button disabled={uploadLocked || duplicateFailedDrafts.length === 0} onClick={showDuplicateFailedDrafts}>仅看重复失败</Button>
                  <Button loading={uploadSaving} disabled={uploadLocked || failedUploadDrafts.length === 0} onClick={() => { void retryFailedDrafts(false) }}>重试失败项</Button>
                  <Button type="primary" loading={uploadSaving} disabled={uploadLocked || failedUploadDrafts.length === 0} onClick={() => { void retryFailedDrafts(true) }}>重试并转换</Button>
                  <Button disabled={uploadLocked} onClick={() => setUploadDrafts((cur) => cur.filter((x) => x.status !== 'saved'))}>清理已保存</Button>
                </div>

                <div className="kb-lib-upload-meta flex flex-wrap items-center gap-3">
                  <Text type="secondary" className="text-xs">已选 {selectedUploadCount} 项</Text>
                  <Text type="secondary" className="text-xs">显示 {filteredUploadDrafts.length}/{uploadDrafts.length} 项</Text>
                  {(uploadDraftFilter === 'error' || uploadDraftFilter === 'dup_error') && uploadErrorReason !== 'all' ? (
                    <Button size="small" onClick={() => setUploadErrorReason('all')}>
                      原因筛选：{activeErrorReasonText}（清除）
                    </Button>
                  ) : null}
                </div>

                {failedUploadDrafts.length > 0 ? (
                  <Alert
                    type="warning"
                    showIcon
                    message={`失败草稿：${failedUploadDrafts.length}`}
                    description={(
                      <div className="kb-lib-failed-summary">
                        <div className="kb-lib-failed-reasons">
                          {failedReasonBuckets.map((bucket) => (
                            <Button
                              key={bucket.key}
                              size="small"
                              icon={FAILED_REASON_META[bucket.key].icon}
                              className={`kb-lib-failed-reason-btn kb-lib-reason-tone is-${bucket.key}${uploadErrorReason === bucket.key ? ' is-active' : ''}`}
                              onClick={() => {
                                applyUploadFilter('error')
                                setUploadErrorReason(bucket.key)
                              }}
                            >
                              {bucket.label} ({bucket.count})
                            </Button>
                          ))}
                        </div>
                        <Text type="secondary" className="text-xs">
                          {failedUploadNotes.length > 0 ? failedUploadNotes.join(' | ') : '请查看行内错误信息后重试。'}
                        </Text>
                      </div>
                    )}
                  />
                ) : null}

                <List
                  size="small"
                  locale={{ emptyText: <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无上传草稿" /> }}
                  dataSource={filteredUploadDrafts}
                  pagination={{ pageSize: 8, size: 'small', showSizeChanger: false }}
                  renderItem={(x) => {
                    const reasonKey = x.status === 'error'
                      ? classifyFailedReason(x.note) as Exclude<UploadErrorReason, 'all'>
                      : null
                    return (
                      <List.Item>
                        <div className="w-full space-y-2">
                          <div className="flex flex-wrap items-center gap-2">
                            <Checkbox checked={x.selected} onChange={(e) => setUploadDrafts((cur) => cur.map((t) => (t.key === x.key ? { ...t, selected: e.target.checked } : t)))} />
                            <Text className="min-w-0 flex-1 truncate text-sm">{x.name}</Text>
                            <Tag color={x.status === 'saved' ? 'success' : x.status === 'error' ? 'error' : (x.status === 'saving' || x.status === 'inspecting') ? 'processing' : 'default'}>
                              {DRAFT_STATUS_TEXT[x.status]}
                            </Tag>
                            {reasonKey ? (
                              <span className={`kb-lib-inline-reason-chip kb-lib-reason-tone is-${reasonKey}`}>
                                {FAILED_REASON_META[reasonKey].icon}
                                <span>{FAILED_REASON_META[reasonKey].label}</span>
                              </span>
                            ) : null}
                          </div>
                          <div className="flex flex-wrap items-center gap-2 pl-6">
                            <Text type="secondary" className="text-xs">建议存储名</Text>
                            <Input value={x.stem} onChange={(e) => setUploadDrafts((cur) => cur.map((t) => (t.key === x.key ? { ...t, stem: e.target.value } : t)))} className="w-[24rem] max-w-full" />
                            <Button size="small" disabled={uploadLocked || x.status === 'saving' || x.status === 'inspecting'} onClick={() => { void inspectDraft(x.key) }}>扫描</Button>
                            <Button size="small" disabled={uploadLocked || x.status === 'saving' || x.status === 'saved' || x.status === 'inspecting'} onClick={() => { void saveDraft(x.key, false) }}>保存</Button>
                            <Button size="small" type="primary" disabled={uploadLocked || x.status === 'saving' || x.status === 'saved' || x.status === 'inspecting'} onClick={() => { void saveDraft(x.key, true) }}>保存并转换</Button>
                          </div>
                          <Text type="secondary" className="block pl-6 text-xs">{x.displayName}</Text>
                          {x.note ? (
                            <Text type="secondary" className={`block pl-6 text-xs${reasonKey ? ' kb-lib-fail-note' : ''}`}>
                              {x.note}
                            </Text>
                          ) : null}
                        </div>
                      </List.Item>
                    )
                  }}
                />
              </div>
            ),
          },
        ]}
      />

      <Card size="small" className="kb-lib-card" title="转换与列表筛选">
        <div className="flex flex-wrap items-center gap-2">
          <Select value={scope} onChange={(value) => { setScope(value); void store.loadFiles(value) }} className="w-40" options={SCOPE_OPTIONS} />
          <Input
            value={fileKeyword}
            onChange={(e) => setFileKeyword(e.target.value)}
            allowClear
            prefix={<SearchOutlined className="opacity-50" />}
            placeholder="筛选文件名"
            className="w-[280px] max-w-full"
          />
          <Switch checked={onlyBusyFiles} onChange={setOnlyBusyFiles} />
          <Text className="text-sm text-[var(--muted)]">仅显示排队/运行</Text>
          <Switch checked={replaceMd} onChange={setReplaceMd} />
          <Text className="text-sm text-[var(--muted)]">重新转换时覆盖已有 Markdown</Text>
          <Button onClick={() => { void handleConvertPending() }}>立即转换待处理</Button>
          {store.converting ? <Button icon={<StopOutlined />} danger onClick={() => { void store.cancelConvert() }}>停止</Button> : null}
          <Button icon={<ReloadOutlined />} onClick={() => { void store.loadFiles(scope) }}>刷新</Button>
        </div>
      </Card>

      {store.refSync && !store.refSync.running ? (
        <Card size="small" className="kb-lib-card" title="引用同步">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Text type="secondary" className="text-xs">
                {store.refSync.current
                  ? `${store.refSync.stage || '运行中'} | ${store.refSync.current}`
                  : (store.refSync.message || '等待同步任务')}
              </Text>
              <Tag color={store.refSync.running ? 'processing' : (store.refSync.status === 'error' ? 'error' : 'default')}>
                {store.refSync.running ? '运行中' : (store.refSync.status === 'idle' ? '空闲' : store.refSync.status)}
              </Tag>
            </div>
            {store.refSync.docsTotal > 0 ? (
              <Progress
                percent={Math.round((store.refSync.docsDone / Math.max(1, store.refSync.docsTotal)) * 100)}
                status={store.refSync.running ? 'active' : (store.refSync.status === 'error' ? 'exception' : 'normal')}
              />
            ) : null}
            {store.refSync.error ? <Text type="danger" className="text-xs">{store.refSync.error}</Text> : null}
          </div>
        </Card>
      ) : null}

      <Collapse
        className="kb-lib-collapse"
        size="small"
        items={[
          {
            key: 'rename',
            label: `文件名管理（${renameVisible.length}/${renameItems.length}）`,
            children: (
              <div className="space-y-3">
                <div className="flex flex-wrap items-center gap-2">
                  <Select value={renameScope} onChange={setRenameScope} className="w-40" options={RENAME_SCOPE_OPTIONS} />
                  <Switch checked={renameUseLlm} onChange={setRenameUseLlm} />
                  <Text className="text-sm text-[var(--muted)]">使用 LLM 补全信息</Text>
                  <Switch checked={renameOnlyDiff} onChange={setRenameOnlyDiff} />
                  <Text className="text-sm text-[var(--muted)]">仅显示需改名项</Text>
                  <Button onClick={selectRenameDiffItems}>选择需改名项</Button>
                  <Button onClick={clearRenameSelection}>清空选择</Button>
                  <Button loading={renameLoading} onClick={() => { void scanRenameSuggestions() }}>扫描建议</Button>
                  <Button type="primary" loading={renameApplying} disabled={renameItems.length === 0} onClick={() => { void applyRenameSuggestions() }}>应用已选改名</Button>
                </div>

                <Text type="secondary" className="text-xs">已选 {selectedRenameCount} 项</Text>

                <List
                  size="small"
                  locale={{ emptyText: <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无扫描结果" /> }}
                  dataSource={renameVisible}
                  pagination={{ pageSize: 12, size: 'small', showSizeChanger: false }}
                  renderItem={(item) => (
                    <List.Item>
                      <div className="w-full space-y-2">
                        <div className="flex items-center gap-2">
                          <Checkbox checked={Boolean(renameSelected[item.name])} onChange={(e) => setRenameSelected((cur) => ({ ...cur, [item.name]: e.target.checked }))} />
                          <Text className="min-w-0 flex-1 truncate text-sm">{item.name}</Text>
                          <Tag color={item.diff ? 'warning' : 'default'}>{item.diff ? '建议改名' : '无需改名'}</Tag>
                        </div>
                        <div className="flex flex-wrap items-center gap-2 pl-6">
                          <Input value={renameOverrides[item.name] || ''} onChange={(e) => setRenameOverrides((cur) => ({ ...cur, [item.name]: e.target.value }))} className="w-[26rem] max-w-full" />
                        </div>
                        <Text type="secondary" className="block pl-6 text-xs">{item.display_full_name}</Text>
                      </div>
                    </List.Item>
                  )}
                />
              </div>
            ),
          },
        ]}
      />

      <Tabs
        className="kb-lib-tabs"
        activeKey={tabKey}
        onChange={(key) => setTabKey(key as FileTabKey)}
        items={[
          { key: 'pending', label: `待转换 (${visiblePending.length})`, children: renderFiles(visiblePending, '暂无待转换文件') },
          { key: 'converted', label: `已转换 (${visibleConverted.length})`, children: renderFiles(visibleConverted, '暂无已转换文件') },
          { key: 'all', label: `当前视图 (${visibleAll.length})`, children: renderFiles(visibleAll, '暂无文件') },
        ]}
      />
    </div>
  )
}
