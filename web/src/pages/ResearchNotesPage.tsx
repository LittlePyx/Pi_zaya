import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Button,
  Checkbox,
  Empty,
  Input,
  Modal,
  Popconfirm,
  Segmented,
  Select,
  Spin,
  Tag,
  Tooltip,
  message,
} from 'antd'
import {
  ArrowRightOutlined,
  CopyOutlined,
  DeleteOutlined,
  FileMarkdownOutlined,
  FileTextOutlined,
  FileWordOutlined,
  FolderOutlined,
  InboxOutlined,
  LinkOutlined,
  PlusOutlined,
  PushpinFilled,
  PushpinOutlined,
  ReloadOutlined,
  SearchOutlined,
} from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import {
  chatApi,
  type ResearchNoteRecord,
  type ResearchNoteSourceLink,
  type ResearchNoteUpdateBody,
} from '../api/chat'
import { MarkdownRenderer } from '../components/chat/MarkdownRenderer'
import { useT } from '../i18n'
import { useChatStore } from '../stores/chatStore'
import {
  RESEARCH_NOTES_CHANGED_EVENT,
  RESEARCH_NOTES_SYNC_CHANNEL,
} from '../components/chat/readerResearchNoteCapture'

const UNASSIGNED_PROJECT = '__unassigned__'
const ALL_FILTER = '__all__'

function noteLinks(note: ResearchNoteRecord | null | undefined): ResearchNoteSourceLink[] {
  return Array.isArray(note?.source_state?.links)
    ? note.source_state.links.filter((link): link is ResearchNoteSourceLink => Boolean(link && typeof link === 'object'))
    : []
}

function sourceLabel(link: ResearchNoteSourceLink): string {
  return String(link.source_name || link.label || '').trim()
}

function noteSourceNames(note: ResearchNoteRecord): string[] {
  return Array.from(new Set(noteLinks(note).filter(link => link.kind !== 'answer').map(sourceLabel).filter(Boolean)))
}

function formatUpdated(timestamp: number): string {
  const date = new Date(Number(timestamp || 0) * 1000)
  if (!Number.isFinite(date.getTime())) return ''
  return new Intl.DateTimeFormat(undefined, {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date)
}

function safeFilename(value: string, fallback: string): string {
  const cleaned = String(value || '').replace(/[\\/:*?"<>|]+/g, '-').replace(/\s+/g, ' ').trim()
  return (cleaned || fallback).slice(0, 90)
}

function downloadMarkdown(filename: string, content: string) {
  const url = URL.createObjectURL(new Blob([content], { type: 'text/markdown;charset=utf-8' }))
  try {
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = filename
    document.body.appendChild(anchor)
    anchor.click()
    anchor.remove()
  } finally {
    window.setTimeout(() => URL.revokeObjectURL(url), 2_000)
  }
}

async function copyText(value: string) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value)
    return
  }
  const textarea = document.createElement('textarea')
  textarea.value = value
  textarea.style.position = 'fixed'
  textarea.style.opacity = '0'
  document.body.appendChild(textarea)
  textarea.select()
  const copied = document.execCommand('copy')
  textarea.remove()
  if (!copied) throw new Error('clipboard-copy-failed')
}

function combinedOutline(title: string, notes: ResearchNoteRecord[]): string {
  const sections = notes.map((note, index) => {
    const body = String(note.content_markdown || '').trim()
    return `## ${index + 1}. ${note.title}\n\n${body}`
  })
  return `# ${title}\n\n${sections.join('\n\n---\n\n')}`.trim()
}

export default function ResearchNotesPage() {
  const S = useT()
  const navigate = useNavigate()
  const projects = useChatStore(state => state.projects)
  const [notes, setNotes] = useState<ResearchNoteRecord[]>([])
  const [activeNote, setActiveNote] = useState<ResearchNoteRecord | null>(null)
  const [activeId, setActiveId] = useState('')
  const [loading, setLoading] = useState(true)
  const [loadingNote, setLoadingNote] = useState(false)
  const [saving, setSaving] = useState(false)
  const [dirty, setDirty] = useState(false)
  const [saveFailed, setSaveFailed] = useState(false)
  const draftVersionRef = useRef(0)
  const listRequestRef = useRef(0)

  const [query, setQuery] = useState('')
  const [archiveView, setArchiveView] = useState<'active' | 'archived'>('active')
  const [projectFilter, setProjectFilter] = useState(ALL_FILTER)
  const [sourceFilter, setSourceFilter] = useState(ALL_FILTER)
  const [tagFilter, setTagFilter] = useState(ALL_FILTER)
  const [timeFilter, setTimeFilter] = useState<'all' | 'week' | 'month'>('all')
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [editorMode, setEditorMode] = useState<'edit' | 'preview'>('edit')

  const [draftTitle, setDraftTitle] = useState('')
  const [draftContent, setDraftContent] = useState('')
  const [draftProjectId, setDraftProjectId] = useState<string | null>(null)
  const [draftTags, setDraftTags] = useState<string[]>([])

  const [combineOpen, setCombineOpen] = useState(false)
  const [combineTitle, setCombineTitle] = useState(S.research_notes_combine_default_title)
  const [combineContent, setCombineContent] = useState('')
  const [combinePreview, setCombinePreview] = useState(false)
  const [combining, setCombining] = useState(false)

  const markDirty = useCallback(() => {
    draftVersionRef.current += 1
    setSaveFailed(false)
    setDirty(true)
  }, [])

  const applyRecord = useCallback((record: ResearchNoteRecord) => {
    setActiveNote(record)
    setActiveId(record.id)
    setDraftTitle(record.title)
    setDraftContent(record.content_markdown)
    setDraftProjectId(record.project_id || null)
    setDraftTags(Array.isArray(record.tags) ? record.tags : [])
    setSaveFailed(false)
    setDirty(false)
  }, [])

  const upsertSummary = useCallback((record: ResearchNoteRecord) => {
    setNotes(current => {
      const next = current.filter(item => item.id !== record.id)
      if ((archiveView === 'archived') !== Boolean(record.archived)) return next
      next.push({ ...record, content_markdown: '' })
      return next.sort((a, b) => Number(b.pinned) - Number(a.pinned) || b.updated_at - a.updated_at)
    })
  }, [archiveView])

  const loadNotes = useCallback(async () => {
    const requestId = ++listRequestRef.current
    setLoading(true)
    try {
      const records = await chatApi.listResearchNotes(null, 500, {
        scope: 'all',
        query: query.trim(),
        archived: archiveView,
      })
      if (requestId !== listRequestRef.current) return
      setNotes(records)
    } catch (error) {
      if (requestId === listRequestRef.current) {
        message.error(error instanceof Error ? error.message : S.research_notes_load_failed)
      }
    } finally {
      if (requestId === listRequestRef.current) setLoading(false)
    }
  }, [S.research_notes_load_failed, archiveView, query])

  useEffect(() => {
    const timer = window.setTimeout(() => { void loadNotes() }, 180)
    return () => window.clearTimeout(timer)
  }, [loadNotes])

  useEffect(() => {
    const refresh = () => { void loadNotes() }
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
  }, [loadNotes])

  const openNote = useCallback(async (noteId: string) => {
    if (!noteId) return
    setActiveId(noteId)
    setLoadingNote(true)
    try {
      const record = await chatApi.getResearchNote(noteId)
      applyRecord(record)
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_notes_load_failed)
    } finally {
      setLoadingNote(false)
    }
  }, [S.research_notes_load_failed, applyRecord])

  useEffect(() => {
    if (loading || activeId || notes.length <= 0) return
    void openNote(notes[0].id)
  }, [activeId, loading, notes, openNote])

  useEffect(() => {
    if (!activeId || notes.some(note => note.id === activeId)) return
    setActiveId('')
    setActiveNote(null)
  }, [activeId, notes])

  useEffect(() => {
    if (!activeNote || !dirty || saving || saveFailed) return
    const version = draftVersionRef.current
    const timer = window.setTimeout(async () => {
      setSaving(true)
      try {
        const record = await chatApi.updateResearchNote(activeNote.id, {
          expected_revision: activeNote.revision,
          title: draftTitle.trim() || S.research_notes_new_default_title,
          content_markdown: draftContent.trim() || S.research_notes_new_default_body,
          project_id: draftProjectId,
          tags: draftTags,
        })
        setActiveNote(record)
        upsertSummary(record)
        if (draftVersionRef.current === version) setDirty(false)
      } catch (error) {
        const text = error instanceof Error ? error.message : ''
        if (text.includes('409')) {
          message.error(S.research_notes_conflict)
          try {
            applyRecord(await chatApi.getResearchNote(activeNote.id))
          } catch {
            setSaveFailed(true)
          }
        } else {
          setSaveFailed(true)
          message.error(text || S.research_notes_save_failed)
        }
      } finally {
        setSaving(false)
      }
    }, 850)
    return () => window.clearTimeout(timer)
  }, [
    S.research_notes_conflict,
    S.research_notes_new_default_body,
    S.research_notes_new_default_title,
    S.research_notes_save_failed,
    activeNote,
    applyRecord,
    dirty,
    draftContent,
    draftProjectId,
    draftTags,
    draftTitle,
    saving,
    saveFailed,
    upsertSummary,
  ])

  const sourceOptions = useMemo(() => Array.from(new Set(notes.flatMap(noteSourceNames))).sort(), [notes])
  const tagOptions = useMemo(() => Array.from(new Set(notes.flatMap(note => note.tags || []))).sort(), [notes])
  const projectNameById = useMemo(() => new Map(projects.map(project => [project.id, project.name])), [projects])

  const filteredNotes = useMemo(() => {
    const now = Date.now() / 1000
    const cutoff = timeFilter === 'week'
      ? now - 7 * 86400
      : timeFilter === 'month'
        ? now - 30 * 86400
        : 0
    return notes.filter(note => {
      if (projectFilter === UNASSIGNED_PROJECT && note.project_id) return false
      if (projectFilter !== ALL_FILTER && projectFilter !== UNASSIGNED_PROJECT && note.project_id !== projectFilter) return false
      if (sourceFilter !== ALL_FILTER && !noteSourceNames(note).includes(sourceFilter)) return false
      if (tagFilter !== ALL_FILTER && !(note.tags || []).includes(tagFilter)) return false
      if (cutoff && note.updated_at < cutoff) return false
      return true
    })
  }, [notes, projectFilter, sourceFilter, tagFilter, timeFilter])

  const createNote = async () => {
    try {
      const projectId = projectFilter !== ALL_FILTER && projectFilter !== UNASSIGNED_PROJECT ? projectFilter : null
      const record = await chatApi.createResearchNote({
        title: S.research_notes_new_default_title,
        content_markdown: S.research_notes_new_default_body,
        project_id: projectId,
        source_state: { version: 1, links: [] },
      })
      setArchiveView('active')
      setNotes(current => [
        { ...record, content_markdown: '' },
        ...current.filter(item => item.id !== record.id && !item.archived),
      ])
      applyRecord(record)
      setEditorMode('edit')
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_notes_save_failed)
    }
  }

  const patchMetadata = async (patch: ResearchNoteUpdateBody) => {
    if (!activeNote) return
    try {
      setSaving(true)
      const record = await chatApi.updateResearchNote(activeNote.id, {
        expected_revision: activeNote.revision,
        ...patch,
      })
      setActiveNote(record)
      setDraftProjectId(record.project_id || null)
      setDraftTags(record.tags || [])
      upsertSummary(record)
      if (patch.archived !== undefined) {
        setActiveId('')
        setActiveNote(null)
        await loadNotes()
      }
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_notes_save_failed)
    } finally {
      setSaving(false)
    }
  }

  const deleteActive = async () => {
    if (!activeNote) return
    try {
      await chatApi.deleteResearchNote(activeNote.id)
      setNotes(current => current.filter(item => item.id !== activeNote.id))
      setSelectedIds(current => {
        const next = new Set(current)
        next.delete(activeNote.id)
        return next
      })
      setActiveId('')
      setActiveNote(null)
      message.success(S.research_notes_deleted)
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_notes_delete_failed)
    }
  }

  const openAnswer = (link: ResearchNoteSourceLink) => {
    const conversationId = String(link.conversation_id || '').trim()
    const messageId = Number(link.message_id || 0)
    if (!conversationId || !messageId) return
    const params = new URLSearchParams({ conversation: conversationId, note_message: String(messageId) })
    navigate({ pathname: '/', search: `?${params.toString()}` })
  }

  const openPaper = async (link: ResearchNoteSourceLink) => {
    const sourcePath = String(link.source_path || '').trim()
    if (!sourcePath) return
    try {
      const session = await chatApi.createReaderSession({
        sourcePath,
        sourceName: link.source_name || link.label,
        headingPath: link.heading_path,
        snippet: link.evidence_quote,
        highlightSnippet: link.evidence_quote,
        blockId: link.block_id,
        anchorId: link.anchor_id,
        anchorKind: link.anchor_kind,
        startOffset: link.start_offset,
        endOffset: link.end_offset,
        occurrence: link.occurrence,
        readableIndex: link.readable_index,
        documentOccurrence: link.document_occurrence,
        startReadableIndex: link.start_readable_index,
        endReadableIndex: link.end_readable_index,
        strictLocate: true,
        locateTarget: {
          headingPath: link.heading_path,
          snippet: link.evidence_quote,
          highlightSnippet: link.evidence_quote,
          evidenceQuote: link.evidence_quote,
          blockId: link.block_id,
          anchorId: link.anchor_id,
          anchorKind: link.anchor_kind,
          startOffset: link.start_offset,
          endOffset: link.end_offset,
          occurrence: link.occurrence,
          readableIndex: link.readable_index,
          documentOccurrence: link.document_occurrence,
          startReadableIndex: link.start_readable_index,
          endReadableIndex: link.end_readable_index,
        },
      }, {
        title: link.source_name || link.label,
        conversationId: link.conversation_id,
        messageId: link.message_id,
      })
      navigate(`/reader/session/${encodeURIComponent(session.id)}`)
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_notes_load_failed)
    }
  }

  const openCombine = async () => {
    const ids = Array.from(selectedIds)
    if (ids.length <= 0) return
    setCombining(true)
    try {
      const records = await Promise.all(ids.map(id => chatApi.getResearchNote(id)))
      const title = S.research_notes_combine_default_title
      setCombineTitle(title)
      setCombineContent(combinedOutline(title, records))
      setCombinePreview(false)
      setCombineOpen(true)
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_notes_load_failed)
    } finally {
      setCombining(false)
    }
  }

  const exportCurrent = async (format: 'copy' | 'markdown' | 'word') => {
    if (!activeNote) return
    const title = draftTitle.trim() || activeNote.title
    try {
      if (format === 'copy') {
        await copyText(draftContent)
        message.success(S.research_notes_copied)
      } else if (format === 'markdown') {
        downloadMarkdown(`${safeFilename(title, 'research-note')}.md`, draftContent)
      } else {
        await chatApi.downloadResearchNoteDocx({ title, content_markdown: draftContent })
      }
    } catch {
      message.error(S.research_notes_export_failed)
    }
  }

  const links = noteLinks(activeNote)
  const statusText = saving
    ? S.research_notes_saving
    : saveFailed
      ? S.research_notes_save_failed
    : dirty
      ? S.research_notes_unsaved_changes
      : S.research_notes_saved_status

  return (
    <main className="kb-notes-page" data-testid="research-notes-workspace">
      <header className="kb-notes-page-header">
        <div>
          <h1>{S.page_research_notes}</h1>
          <p>{S.research_notes_workspace_subtitle}</p>
        </div>
        <div className="kb-notes-header-actions">
          {selectedIds.size > 0 ? (
            <>
              <span>{S.research_notes_selected_count.replace('{n}', String(selectedIds.size))}</span>
              <Button size="small" onClick={() => setSelectedIds(new Set())}>{S.research_notes_clear_selection}</Button>
              <Button type="primary" loading={combining} onClick={() => { void openCombine() }}>
                {S.research_notes_combine}
              </Button>
            </>
          ) : null}
          <Button icon={<ReloadOutlined />} onClick={() => { void loadNotes() }} aria-label={S.reload} />
          <Button type="primary" icon={<PlusOutlined />} onClick={() => { void createNote() }}>
            {S.research_notes_new}
          </Button>
        </div>
      </header>

      <section className="kb-notes-workspace">
        <aside className="kb-notes-directory">
          <Segmented
            block
            value={archiveView}
            options={[
              { value: 'active', label: S.research_notes_active },
              { value: 'archived', label: S.research_notes_archived },
            ]}
            onChange={value => {
              setArchiveView(value as 'active' | 'archived')
              setActiveId('')
              setActiveNote(null)
              setSelectedIds(new Set())
            }}
          />
          <Input
            allowClear
            prefix={<SearchOutlined />}
            value={query}
            placeholder={S.research_notes_search}
            onChange={event => setQuery(event.target.value)}
          />
          <div className="kb-notes-filters">
            <Select
              aria-label={S.research_notes_filter_project}
              value={projectFilter}
              onChange={setProjectFilter}
              options={[
                { value: ALL_FILTER, label: S.research_notes_all_projects },
                { value: UNASSIGNED_PROJECT, label: S.research_notes_unassigned },
                ...projects.map(project => ({ value: project.id, label: project.name })),
              ]}
            />
            <Select
              aria-label={S.research_notes_filter_source}
              value={sourceFilter}
              onChange={setSourceFilter}
              options={[
                { value: ALL_FILTER, label: S.research_notes_all_sources },
                ...sourceOptions.map(value => ({ value, label: value })),
              ]}
            />
            <Select
              aria-label={S.research_notes_filter_tag}
              value={tagFilter}
              onChange={setTagFilter}
              options={[
                { value: ALL_FILTER, label: S.research_notes_all_tags },
                ...tagOptions.map(value => ({ value, label: value })),
              ]}
            />
            <Select
              aria-label={S.research_notes_filter_time}
              value={timeFilter}
              onChange={setTimeFilter}
              options={[
                { value: 'all', label: S.research_notes_all_time },
                { value: 'week', label: S.research_notes_last_week },
                { value: 'month', label: S.research_notes_last_month },
              ]}
            />
          </div>
          <div className="kb-notes-directory-list kb-main-scroll">
            {loading ? (
              <div className="kb-notes-centered"><Spin size="small" /> {S.research_notes_loading_workspace}</div>
            ) : filteredNotes.length <= 0 ? (
              <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={S.research_notes_empty_workspace} />
            ) : filteredNotes.map(note => {
              const selected = selectedIds.has(note.id)
              const sourceCount = noteLinks(note).length
              return (
                <article
                  key={note.id}
                  className={`kb-notes-directory-card ${activeId === note.id ? 'is-active' : ''}`}
                  data-testid={`workspace-note-${note.id}`}
                  onClick={() => { void openNote(note.id) }}
                >
                  <Checkbox
                    checked={selected}
                    aria-label={S.research_notes_select_for_combine}
                    onClick={event => event.stopPropagation()}
                    onChange={event => {
                      setSelectedIds(current => {
                        const next = new Set(current)
                        if (event.target.checked) next.add(note.id)
                        else next.delete(note.id)
                        return next
                      })
                    }}
                  />
                  <div>
                    <div className="kb-notes-card-title">
                      <strong>{note.title}</strong>
                      {note.pinned ? <PushpinFilled /> : null}
                    </div>
                    <small>{note.project_id ? projectNameById.get(note.project_id) : S.research_notes_unassigned}</small>
                    <div className="kb-notes-card-tags">
                      {(note.tags || []).slice(0, 3).map(tag => <Tag key={tag}>{tag}</Tag>)}
                    </div>
                    <div className="kb-notes-card-foot">
                      <span><LinkOutlined /> {sourceCount}</span>
                      <time>{formatUpdated(note.updated_at)}</time>
                    </div>
                  </div>
                </article>
              )
            })}
          </div>
        </aside>

        <section className="kb-notes-editor-pane">
          {loadingNote ? (
            <div className="kb-notes-centered"><Spin /> {S.research_notes_loading}</div>
          ) : !activeNote ? (
            <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={S.research_notes_select_one} />
          ) : (
            <>
              <div className="kb-notes-editor-toolbar">
                <Segmented
                  value={editorMode}
                  options={[
                    { value: 'edit', label: S.research_notes_editor },
                    { value: 'preview', label: S.research_notes_preview },
                  ]}
                  onChange={value => setEditorMode(value as 'edit' | 'preview')}
                />
                <span className={`kb-notes-save-state ${dirty ? 'is-dirty' : ''}`}>{statusText}</span>
                <div className="kb-notes-editor-actions">
                  <Tooltip title={activeNote.pinned ? S.research_notes_unpin : S.research_notes_pin}>
                    <Button
                      icon={activeNote.pinned ? <PushpinFilled /> : <PushpinOutlined />}
                      disabled={dirty || saving}
                      onClick={() => { void patchMetadata({ pinned: !activeNote.pinned }) }}
                    />
                  </Tooltip>
                  <Button disabled={dirty || saving} icon={<InboxOutlined />} onClick={() => { void patchMetadata({ archived: !activeNote.archived }) }}>
                    {activeNote.archived ? S.research_notes_restore : S.research_notes_archive}
                  </Button>
                  <Popconfirm
                    title={S.research_notes_delete_confirm}
                    okText={S.research_notes_delete}
                    cancelText={S.confirm_cancel}
                    onConfirm={deleteActive}
                  >
                    <Button danger icon={<DeleteOutlined />} aria-label={S.research_notes_delete} />
                  </Popconfirm>
                </div>
              </div>
              <div className="kb-notes-editor-fields">
                <Input
                  className="kb-notes-title-input"
                  value={draftTitle}
                  onChange={event => { setDraftTitle(event.target.value); markDirty() }}
                  maxLength={240}
                />
                <div className="kb-notes-meta-fields">
                  <label>
                    <span><FolderOutlined /> {S.research_notes_project}</span>
                    <Select
                      value={draftProjectId || UNASSIGNED_PROJECT}
                      onChange={value => {
                        setDraftProjectId(value === UNASSIGNED_PROJECT ? null : value)
                        markDirty()
                      }}
                      options={[
                        { value: UNASSIGNED_PROJECT, label: S.research_notes_unassigned },
                        ...projects.map(project => ({ value: project.id, label: project.name })),
                      ]}
                    />
                  </label>
                  <label>
                    <span>{S.research_notes_tags}</span>
                    <Select
                      mode="tags"
                      value={draftTags}
                      maxCount={24}
                      tokenSeparators={[',', '，']}
                      placeholder={S.research_notes_tags_placeholder}
                      onChange={value => { setDraftTags(value); markDirty() }}
                      options={tagOptions.map(value => ({ value, label: value }))}
                    />
                  </label>
                </div>
              </div>
              <div className="kb-notes-editor-body kb-main-scroll">
                {editorMode === 'edit' ? (
                  <Input.TextArea
                    value={draftContent}
                    onChange={event => { setDraftContent(event.target.value); markDirty() }}
                    autoSize={false}
                    spellCheck={false}
                  />
                ) : (
                  <div className="kb-notes-markdown-preview">
                    <MarkdownRenderer content={draftContent} citeDetails={[]} linkifyPlainCitations={false} />
                  </div>
                )}
              </div>
              <footer className="kb-notes-export-bar">
                <Button icon={<CopyOutlined />} onClick={() => { void exportCurrent('copy') }}>{S.research_notes_copy}</Button>
                <Button icon={<FileMarkdownOutlined />} onClick={() => { void exportCurrent('markdown') }}>{S.research_notes_export_markdown}</Button>
                <Button icon={<FileWordOutlined />} onClick={() => { void exportCurrent('word') }}>{S.research_notes_export_word}</Button>
              </footer>
            </>
          )}
        </section>

        <aside className="kb-notes-sources-pane">
          <div className="kb-notes-sources-head">
            <h2>{S.research_notes_sources_title}</h2>
            <p>{S.research_notes_sources_hint}</p>
          </div>
          <div className="kb-notes-sources-list kb-main-scroll">
            {!activeNote || links.length <= 0 ? (
              <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={S.research_notes_no_sources} />
            ) : links.map((link, index) => (
              <article key={`${link.kind}-${link.message_id || 0}-${link.source_path || ''}-${index}`} className="kb-notes-source-card">
                <div className="kb-notes-source-kind">
                  {link.kind === 'answer' ? <FileTextOutlined /> : <LinkOutlined />}
                  <span>{link.kind === 'answer' ? S.research_notes_answer_source : S.research_notes_paper_source}</span>
                </div>
                <strong>{link.label || link.source_name || S.research_note_untitled_source}</strong>
                {link.location_label || link.heading_path ? (
                  <small>{S.research_notes_source_location} · {link.location_label || link.heading_path}</small>
                ) : null}
                {link.evidence_quote ? (
                  <blockquote>
                    <span>{S.research_notes_source_quote}</span>
                    {link.evidence_quote}
                  </blockquote>
                ) : null}
                <Button
                  type="link"
                  icon={<ArrowRightOutlined />}
                  disabled={link.kind === 'answer'
                    ? !link.conversation_id || !link.message_id
                    : !link.source_path}
                  onClick={() => {
                    if (link.kind === 'answer') openAnswer(link)
                    else void openPaper(link)
                  }}
                >
                  {link.kind === 'answer' ? S.research_notes_open_answer : S.research_notes_open_paper}
                </Button>
              </article>
            ))}
          </div>
        </aside>
      </section>

      <Modal
        open={combineOpen}
        width={920}
        title={S.research_notes_combine_title}
        onCancel={() => setCombineOpen(false)}
        footer={[
          <Button key="copy" icon={<CopyOutlined />} onClick={() => {
            void copyText(combineContent).then(() => message.success(S.research_notes_copied))
          }}>{S.research_notes_copy}</Button>,
          <Button key="markdown" icon={<FileMarkdownOutlined />} onClick={() => {
            downloadMarkdown(`${safeFilename(combineTitle, 'research-outline')}.md`, combineContent)
          }}>{S.research_notes_export_markdown}</Button>,
          <Button key="word" type="primary" icon={<FileWordOutlined />} onClick={() => {
            void chatApi.downloadResearchNoteDocx({ title: combineTitle, content_markdown: combineContent })
              .catch(() => message.error(S.research_notes_export_failed))
          }}>{S.research_notes_export_word}</Button>,
        ]}
      >
        <div className="kb-notes-combine-modal">
          <Input value={combineTitle} onChange={event => setCombineTitle(event.target.value)} />
          <Segmented
            value={combinePreview ? 'preview' : 'edit'}
            options={[
              { value: 'edit', label: S.research_notes_editor },
              { value: 'preview', label: S.research_notes_preview },
            ]}
            onChange={value => setCombinePreview(value === 'preview')}
          />
          {combinePreview ? (
            <div className="kb-notes-combine-preview kb-main-scroll">
              <MarkdownRenderer content={combineContent} citeDetails={[]} linkifyPlainCitations={false} />
            </div>
          ) : (
            <Input.TextArea value={combineContent} onChange={event => setCombineContent(event.target.value)} autoSize={false} />
          )}
        </div>
      </Modal>
    </main>
  )
}
