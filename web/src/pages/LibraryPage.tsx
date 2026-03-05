import { useEffect, useMemo, useState } from 'react'
import { Upload, Button, List, message, Progress, Select, Typography, Tabs, Tag, Switch, Space, Popconfirm, Empty, Input, Card, Checkbox, Collapse } from 'antd'
import { UploadOutlined, ReloadOutlined, StopOutlined, FolderOpenOutlined, DeleteOutlined, SaveOutlined } from '@ant-design/icons'
import type { LibraryFileItem, RenameSuggestionItem } from '../api/library'
import { libraryApi } from '../api/library'
import { useLibraryStore } from '../stores/libraryStore'
import { useSettingsStore } from '../stores/settingsStore'
import { S } from '../i18n/zh'

const { Text } = Typography
const { Dragger } = Upload

type FileTabKey = 'pending' | 'converted' | 'all'
type DraftStatus = 'queued' | 'saving' | 'saved' | 'error'

type UploadDraft = {
  key: string
  file: File
  name: string
  selected: boolean
  stem: string
  status: DraftStatus
  note: string
}

const CONVERT_MODE = 'balanced'

const SCOPE_OPTIONS = [
  { value: '200', label: 'Recent 200' },
  { value: '1000', label: 'Recent 1000' },
  { value: 'all', label: 'All' },
]

const RENAME_SCOPE_OPTIONS = [
  { value: '30', label: 'Recent 30' },
  { value: '50', label: 'Recent 50' },
  { value: '100', label: 'Recent 100' },
  { value: 'all', label: 'All' },
]

function fileTag(item: LibraryFileItem) {
  if (item.task_state === 'running') return { color: 'processing' as const, text: 'Running' }
  if (item.task_state === 'queued') return { color: 'warning' as const, text: `Queued${item.queue_pos > 0 ? ` #${item.queue_pos}` : ''}` }
  return item.category === 'converted'
    ? { color: 'success' as const, text: 'Converted' }
    : { color: 'default' as const, text: 'Pending' }
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

  const [pdfDirDraft, setPdfDirDraft] = useState('')
  const [mdDirDraft, setMdDirDraft] = useState('')
  const [savingDirs, setSavingDirs] = useState(false)
  const [dirTouched, setDirTouched] = useState(false)

  const [uploadDrafts, setUploadDrafts] = useState<UploadDraft[]>([])
  const [uploadSaving, setUploadSaving] = useState(false)

  const [renameScope, setRenameScope] = useState('30')
  const [renameUseLlm, setRenameUseLlm] = useState(false)
  const [renameOnlyDiff, setRenameOnlyDiff] = useState(true)
  const [renameLoading, setRenameLoading] = useState(false)
  const [renameApplying, setRenameApplying] = useState(false)
  const [renameItems, setRenameItems] = useState<RenameSuggestionItem[]>([])
  const [renameSelected, setRenameSelected] = useState<Record<string, boolean>>({})
  const [renameOverrides, setRenameOverrides] = useState<Record<string, string>>({})

  const dirDirty = useMemo(
    () => pdfDirDraft.trim() !== String(settingsPdfDir || '').trim() || mdDirDraft.trim() !== String(settingsMdDir || '').trim(),
    [pdfDirDraft, mdDirDraft, settingsPdfDir, settingsMdDir],
  )

  const pendingFiles = useMemo(() => store.files.filter((x) => x.category === 'pending'), [store.files])
  const convertedFiles = useMemo(() => store.files.filter((x) => x.category === 'converted'), [store.files])
  const renameVisible = useMemo(() => (renameOnlyDiff ? renameItems.filter((x) => x.diff) : renameItems), [renameOnlyDiff, renameItems])

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
      message.warning('PDF/MD directory cannot be empty')
      return false
    }
    setSavingDirs(true)
    try {
      await updateSettings({ pdfDir: pdfDirDraft.trim(), mdDir: mdDirDraft.trim() })
      setDirTouched(false)
      message.success('Directories saved')
      await store.loadFiles(scope)
      return true
    } catch (err) {
      message.error(err instanceof Error ? err.message : 'Failed to save directories')
      return false
    } finally {
      setSavingDirs(false)
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
        next.push({ key, file, name: file.name, selected: true, stem: file.name.replace(/\.pdf$/i, ''), status: 'queued', note: '' })
      }
      return next
    })
  }

  const saveDraft = async (key: string, convertNow: boolean) => {
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
        if (res.duplicate) return { ...x, status: 'error', note: `Duplicate: ${String(res.existing || '')}` }
        return { ...x, status: 'saved', selected: false, note: convertNow && res.enqueued ? 'Saved + converted' : 'Saved' }
      }))
    } catch (err) {
      setUploadDrafts((cur) => cur.map((x) => (x.key === key ? { ...x, status: 'error', note: err instanceof Error ? err.message : 'Save failed' } : x)))
    }
  }

  const saveSelectedDrafts = async (convertNow: boolean) => {
    const selected = uploadDrafts.filter((x) => x.selected && x.status !== 'saving' && x.status !== 'saved')
    if (!selected.length) {
      message.info('Please select files to save')
      return
    }
    setUploadSaving(true)
    try {
      for (const x of selected) {
        // eslint-disable-next-line no-await-in-loop
        await saveDraft(x.key, convertNow)
      }
      await store.loadFiles(scope)
      message.success(`Processed ${selected.length} file(s)`)
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
      message.success(`Scan done: ${res.changed}/${res.total_scanned} changed`)
    } catch (err) {
      message.error(err instanceof Error ? err.message : 'Rename suggestion scan failed')
    } finally {
      setRenameLoading(false)
    }
  }

  const applyRenameSuggestions = async () => {
    const names = renameItems.filter((x) => renameSelected[x.name]).map((x) => x.name)
    if (!names.length) {
      message.info('Please select items to rename')
      return
    }
    setRenameApplying(true)
    try {
      const overrides: Record<string, string> = {}
      for (const name of names) overrides[name] = String(renameOverrides[name] || '').trim()
      const res = await libraryApi.applyRenameSuggestions(names, overrides, { useLlm: renameUseLlm, alsoMd: true })
      message[res.failed > 0 ? 'warning' : 'success'](`Rename finished: success ${res.renamed}, skipped ${res.skipped}, failed ${res.failed}`)
      if (res.needs_reindex) message.info('Reindex is recommended after rename changes')
      await store.loadFiles(scope)
      await scanRenameSuggestions()
    } catch (err) {
      message.error(err instanceof Error ? err.message : 'Apply rename failed')
    } finally {
      setRenameApplying(false)
    }
  }

  const handleConvertPending = async () => {
    const res = await store.convertPending(CONVERT_MODE)
    message[res.enqueued > 0 ? 'success' : 'info'](res.enqueued > 0 ? `Enqueued ${res.enqueued} pending file(s)` : 'No idle pending files to enqueue')
    await store.loadFiles(scope)
  }

  const handleConvertOne = async (item: LibraryFileItem) => {
    if (item.task_state !== 'idle') return
    const replace = item.md_exists ? replaceMd : false
    if (item.md_exists && !replaceMd) {
      message.info('Enable replace mode to re-convert existing markdown')
      return
    }
    await store.convert(item.name, CONVERT_MODE, replace)
  }

  const renderFiles = (items: LibraryFileItem[], emptyText: string) => (
    <List
      size="small"
      locale={{ emptyText: <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={emptyText} /> }}
      dataSource={items}
      renderItem={(item) => {
        const tag = fileTag(item)
        return (
          <List.Item>
            <div className="flex w-full flex-wrap items-center justify-between gap-2">
              <div className="min-w-0 flex-1">
                <div className="mb-1 flex items-center gap-2">
                  <Text className="min-w-0 truncate text-sm">{item.name}</Text>
                  <Tag color={tag.color}>{tag.text}</Tag>
                </div>
                <Text type="secondary" className="text-xs">{item.md_exists ? 'Markdown exists' : 'No markdown yet'}</Text>
              </div>
              <Space wrap>
                <Button size="small" disabled={item.task_state !== 'idle'} onClick={() => { void handleConvertOne(item) }}>{item.md_exists ? 'Re-convert' : 'Convert'}</Button>
                <Button size="small" onClick={() => { void store.openFile(item.name, 'pdf') }}>Open PDF</Button>
                <Button size="small" disabled={!item.md_exists} onClick={() => { void store.openFile(item.name, 'md') }}>Open MD</Button>
                <Popconfirm title="Delete this file?" onConfirm={() => { void store.deleteFile(item.name, true) }}>
                  <Button size="small" danger icon={<DeleteOutlined />} disabled={item.task_state !== 'idle'}>Delete</Button>
                </Popconfirm>
              </Space>
            </div>
          </List.Item>
        )
      }}
    />
  )

  return (
    <div className="max-w-3xl mx-auto p-6 space-y-6">
      <Text className="text-xl font-bold">{S.page_library}</Text>

      <Card size="small" title="Directory Settings">
        <div className="space-y-3">
          <Input addonBefore="PDF" value={pdfDirDraft} onChange={(e) => { setDirTouched(true); setPdfDirDraft(e.target.value) }} />
          <Input addonBefore="MD" value={mdDirDraft} onChange={(e) => { setDirTouched(true); setMdDirDraft(e.target.value) }} />
          <Space wrap>
            <Button type="primary" icon={<SaveOutlined />} loading={savingDirs} disabled={!dirDirty} onClick={() => { void saveDirs() }}>Save</Button>
            <Button icon={<FolderOpenOutlined />} onClick={() => { void store.openFile('', 'pdf_dir') }}>Open PDF dir</Button>
            <Button icon={<FolderOpenOutlined />} onClick={() => { void store.openFile('', 'md_dir') }}>Open MD dir</Button>
          </Space>
        </div>
      </Card>

      <Dragger multiple accept=".pdf" showUploadList={false} beforeUpload={(file) => { addDrafts([file as File]); return false }}>
        <p className="text-lg"><UploadOutlined /></p>
        <p>{S.upload_pdf} (queue only)</p>
      </Dragger>

      <Card size="small" title={`Upload Workbench (${uploadDrafts.length})`}>
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            <Button loading={uploadSaving} onClick={() => { void saveSelectedDrafts(false) }}>Save Selected</Button>
            <Button type="primary" loading={uploadSaving} onClick={() => { void saveSelectedDrafts(true) }}>Save + Convert Selected</Button>
            <Button onClick={() => setUploadDrafts((cur) => cur.filter((x) => x.status !== 'saved'))}>Clear Saved</Button>
          </div>
          <List
            size="small"
            locale={{ emptyText: <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="No upload drafts" /> }}
            dataSource={uploadDrafts}
            pagination={{ pageSize: 8, size: 'small', showSizeChanger: false }}
            renderItem={(x) => (
              <List.Item>
                <div className="w-full space-y-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <Checkbox checked={x.selected} onChange={(e) => setUploadDrafts((cur) => cur.map((t) => (t.key === x.key ? { ...t, selected: e.target.checked } : t)))} />
                    <Text className="min-w-0 flex-1 truncate text-sm">{x.name}</Text>
                    <Tag color={x.status === 'saved' ? 'success' : x.status === 'error' ? 'error' : x.status === 'saving' ? 'processing' : 'default'}>{x.status}</Tag>
                  </div>
                  <div className="flex flex-wrap items-center gap-2 pl-6">
                    <Text type="secondary" className="text-xs">Suggested stem</Text>
                    <Input value={x.stem} onChange={(e) => setUploadDrafts((cur) => cur.map((t) => (t.key === x.key ? { ...t, stem: e.target.value } : t)))} className="w-[24rem] max-w-full" />
                    <Button size="small" disabled={x.status === 'saving' || x.status === 'saved'} onClick={() => { void saveDraft(x.key, false) }}>Save</Button>
                    <Button size="small" type="primary" disabled={x.status === 'saving' || x.status === 'saved'} onClick={() => { void saveDraft(x.key, true) }}>Save + Convert</Button>
                  </div>
                  {x.note ? <Text type="secondary" className="pl-6 text-xs block">{x.note}</Text> : null}
                </div>
              </List.Item>
            )}
          />
        </div>
      </Card>

      <div className="flex flex-wrap gap-2 items-center">
        <Select value={scope} onChange={(value) => { setScope(value); void store.loadFiles(value) }} className="w-36" options={SCOPE_OPTIONS} />
        <Switch checked={replaceMd} onChange={setReplaceMd} />
        <Text className="text-sm text-[var(--muted)]">Replace existing markdown on reconvert</Text>
        <Button onClick={() => { void handleConvertPending() }}>{S.convert_now}</Button>
        {store.converting ? <Button icon={<StopOutlined />} danger onClick={() => { void store.cancelConvert() }}>Stop</Button> : null}
        <Button icon={<ReloadOutlined />} onClick={() => { void store.loadFiles(scope) }}>Refresh</Button>
      </div>

      {store.converting && store.progress ? (
        <Card size="small" title={`${S.converting_files} ${store.progress.completed}/${store.progress.total}`}>
          <Progress percent={store.progress.total > 0 ? Math.round((store.progress.completed / store.progress.total) * 100) : 0} status="active" />
          {store.progress.current ? <Text type="secondary" className="text-xs">{store.progress.current}</Text> : null}
        </Card>
      ) : null}

      <Space wrap>
        <Button icon={<ReloadOutlined />} type="primary" onClick={() => { void store.reindex() }}>{S.reindex_now}</Button>
        <Button icon={<ReloadOutlined />} onClick={() => { void store.startReferenceSync() }}>Sync references</Button>
      </Space>

      <Collapse
        size="small"
        items={[
          {
            key: 'rename',
            label: `Filename Manager${renameItems.length > 0 ? ` - ${renameVisible.length}/${renameItems.length}` : ''}`,
            children: (
              <div className="space-y-3">
                <div className="flex flex-wrap items-center gap-2">
                  <Select value={renameScope} onChange={setRenameScope} className="w-40" options={RENAME_SCOPE_OPTIONS} />
                  <Switch checked={renameUseLlm} onChange={setRenameUseLlm} />
                  <Text className="text-sm text-[var(--muted)]">Use LLM metadata</Text>
                  <Switch checked={renameOnlyDiff} onChange={setRenameOnlyDiff} />
                  <Text className="text-sm text-[var(--muted)]">Only changed</Text>
                  <Button loading={renameLoading} onClick={() => { void scanRenameSuggestions() }}>Scan Suggestions</Button>
                  <Button type="primary" loading={renameApplying} disabled={renameItems.length === 0} onClick={() => { void applyRenameSuggestions() }}>Apply Selected</Button>
                </div>
                <List
                  size="small"
                  locale={{ emptyText: <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="No scan result yet" /> }}
                  dataSource={renameVisible}
                  pagination={{ pageSize: 12, size: 'small', showSizeChanger: false }}
                  renderItem={(item) => (
                    <List.Item>
                      <div className="w-full space-y-2">
                        <div className="flex items-center gap-2">
                          <Checkbox checked={Boolean(renameSelected[item.name])} onChange={(e) => setRenameSelected((cur) => ({ ...cur, [item.name]: e.target.checked }))} />
                          <Text className="min-w-0 flex-1 truncate text-sm">{item.name}</Text>
                          <Tag color={item.diff ? 'warning' : 'default'}>{item.diff ? 'Need rename' : 'No rename needed'}</Tag>
                        </div>
                        <div className="flex flex-wrap items-center gap-2 pl-6">
                          <Input value={renameOverrides[item.name] || ''} onChange={(e) => setRenameOverrides((cur) => ({ ...cur, [item.name]: e.target.value }))} className="w-[26rem] max-w-full" />
                        </div>
                        <Text type="secondary" className="pl-6 text-xs block">{item.display_full_name}</Text>
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
        activeKey={tabKey}
        onChange={(key) => setTabKey(key as FileTabKey)}
        items={[
          { key: 'pending', label: `Pending (${pendingFiles.length})`, children: renderFiles(pendingFiles, 'No pending files') },
          { key: 'converted', label: `Converted (${convertedFiles.length})`, children: renderFiles(convertedFiles, 'No converted files') },
          { key: 'all', label: `Current view (${store.files.length})`, children: renderFiles(store.files, 'No files') },
        ]}
      />
    </div>
  )
}
