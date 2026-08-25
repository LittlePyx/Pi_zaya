import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Button, Checkbox, Empty, Input, Modal, Segmented, Select, message } from 'antd'
import {
  CopyOutlined,
  DownloadOutlined,
  FileMarkdownOutlined,
  FileWordOutlined,
  ReloadOutlined,
  SaveOutlined,
} from '@ant-design/icons'
import type {
  Message,
  ResearchNoteRecord,
  ResearchNoteSourceLink,
  ResearchNoteSourceState,
} from '../../api/chat'
import { chatApi } from '../../api/chat'
import { useT } from '../../i18n'
import type { CiteShelfItem } from './citationState'
import { MarkdownRenderer } from './MarkdownRenderer'
import {
  appendResearchNoteBody,
  buildResearchNoteAnswers,
  buildResearchNoteBody,
  buildResearchNoteSourceLinks,
  researchNoteDefaultTitle,
  type ResearchNoteLabels,
} from './researchNote'

interface Props {
  open: boolean
  initialMessageId: number | null
  initialNoteId: string
  activeConvId?: string | null
  projectId?: string | null
  messages: Message[]
  shelfItems: CiteShelfItem[]
  notes: ResearchNoteRecord[]
  onSaved: (record: ResearchNoteRecord) => void
  onClose: () => void
}

function plainPreview(value: string): string {
  return String(value || '')
    .replace(/!\[[^\]]*]\([^)]+\)/g, ' ')
    .replace(/\[([^\]]+)]\([^)]+\)/g, '$1')
    .replace(/[`*_>#|~-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function downloadTextFile(filename: string, content: string) {
  const href = URL.createObjectURL(new Blob([content], { type: 'text/markdown;charset=utf-8' }))
  try {
    const link = document.createElement('a')
    link.href = href
    link.download = filename
    document.body.appendChild(link)
    link.click()
    link.remove()
  } finally {
    window.setTimeout(() => URL.revokeObjectURL(href), 2_000)
  }
}

async function writeClipboard(value: string): Promise<void> {
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

function mergeLinks(existing: ResearchNoteSourceLink[], incoming: ResearchNoteSourceLink[]): ResearchNoteSourceLink[] {
  const out: ResearchNoteSourceLink[] = []
  const seen = new Set<string>()
  for (const link of [...existing, ...incoming]) {
    const identity = link.kind === 'answer'
      ? `answer:${link.conversation_id || ''}:${Number(link.message_id || 0)}`
      : `source:${String(link.source_path || '').toLowerCase()}|${link.block_id || link.anchor_id || link.heading_path || link.label}`
    if (seen.has(identity)) continue
    seen.add(identity)
    out.push(link)
  }
  return out.slice(0, 240)
}

function draftKey(projectId: string | null | undefined, convId: string | null | undefined, messageId: number | null) {
  return `kb:research-note-draft:${projectId || 'unassigned'}:${convId || 'none'}:${Number(messageId || 0)}`
}

export function ResearchNoteModal({
  open,
  initialMessageId,
  initialNoteId,
  activeConvId,
  projectId,
  messages,
  shelfItems,
  notes,
  onSaved,
  onClose,
}: Props) {
  const S = useT()
  const answers = useMemo(() => buildResearchNoteAnswers(messages), [messages])
  const labels = useMemo<ResearchNoteLabels>(() => ({
    question: S.research_note_question,
    answer: S.research_note_answer,
    answerSources: S.research_note_answer_sources,
    bibliography: S.research_note_bibliography,
    researchBasket: S.research_note_basket_section,
    authors: S.research_note_authors,
    venue: S.research_note_venue,
    year: S.research_note_year,
    location: S.research_note_location,
    evidence: S.research_note_evidence,
    excerpt: S.research_note_excerpt,
    note: S.research_note_note,
    summary: S.research_note_summary,
    source: S.research_note_source,
    untitledSource: S.research_note_untitled_source,
  }), [S])
  const [selectedIds, setSelectedIds] = useState<number[]>([])
  const [includeShelf, setIncludeShelf] = useState(false)
  const [title, setTitle] = useState('')
  const [body, setBody] = useState('')
  const [bodyDirty, setBodyDirty] = useState(false)
  const [sourceState, setSourceState] = useState<ResearchNoteSourceState>({})
  const [viewMode, setViewMode] = useState<'edit' | 'preview'>('edit')
  const [wordExporting, setWordExporting] = useState(false)
  const [saving, setSaving] = useState(false)
  const [loadingNote, setLoadingNote] = useState(false)
  const [saveState, setSaveState] = useState<'draft' | 'saving' | 'saved' | 'error'>('draft')
  const [currentNote, setCurrentNote] = useState<ResearchNoteRecord | null>(null)
  const [targetValue, setTargetValue] = useState('new')
  const [autoSaveEnabled, setAutoSaveEnabled] = useState(false)
  const initializedRef = useRef(false)
  const editSignatureRef = useRef('')
  const savingRef = useRef(false)

  const compose = useCallback((ids: number[], withShelf: boolean) => buildResearchNoteBody({
    answers,
    selectedMessageIds: ids,
    includeShelf: withShelf,
    shelfItems,
    labels,
  }), [answers, labels, shelfItems])

  const currentLinks = useCallback((ids: number[], withShelf: boolean) => buildResearchNoteSourceLinks({
    answers,
    selectedMessageIds: ids,
    includeShelf: withShelf,
    shelfItems,
    conversationId: String(activeConvId || ''),
  }), [activeConvId, answers, shelfItems])

  const resetNewNote = useCallback(() => {
    const initial = answers.find((answer) => answer.messageId === Number(initialMessageId || 0))
    const ids = initial ? [initial.messageId] : []
    const key = draftKey(projectId, activeConvId, initialMessageId)
    let draft: { title?: string; body?: string; selectedIds?: number[]; includeShelf?: boolean } | null = null
    try {
      draft = JSON.parse(window.localStorage.getItem(key) || 'null')
    } catch {
      draft = null
    }
    const restoredIds = Array.isArray(draft?.selectedIds)
      ? draft.selectedIds.filter((id) => answers.some((answer) => answer.messageId === Number(id)))
      : ids
    const restoredShelf = Boolean(draft?.includeShelf && shelfItems.length > 0)
    const nextTitle = String(draft?.title || researchNoteDefaultTitle(initial, S.research_note_default_title))
    const nextBody = String(draft?.body || compose(restoredIds, restoredShelf))
    const links = currentLinks(restoredIds, restoredShelf)
    setCurrentNote(null)
    setTargetValue('new')
    setSelectedIds(restoredIds)
    setIncludeShelf(restoredShelf)
    setTitle(nextTitle)
    setBody(nextBody)
    setSourceState({ version: 1, selected_message_ids: restoredIds, include_shelf: restoredShelf, links })
    setBodyDirty(Boolean(draft?.body))
    setSaveState('draft')
    setAutoSaveEnabled(false)
  }, [activeConvId, answers, compose, currentLinks, initialMessageId, projectId, S.research_note_default_title, shelfItems.length])

  const loadExistingNote = useCallback(async (noteId: string, appendCurrentSelection: boolean) => {
    setLoadingNote(true)
    try {
      const record = await chatApi.getResearchNote(noteId)
      const ids = appendCurrentSelection
        ? [Number(initialMessageId || 0)].filter((id) => id > 0)
        : []
      const addition = ids.length > 0 ? compose(ids, false) : ''
      const incomingLinks = ids.length > 0 ? currentLinks(ids, false) : []
      const existingLinks = Array.isArray(record.source_state?.links) ? record.source_state.links : []
      const nextBody = addition ? appendResearchNoteBody(record.content_markdown, addition, labels) : record.content_markdown
      const nextSourceState: ResearchNoteSourceState = {
        ...record.source_state,
        version: 1,
        links: mergeLinks(existingLinks, incomingLinks),
      }
      setCurrentNote(record)
      setTargetValue(record.id)
      setSelectedIds(ids)
      setIncludeShelf(false)
      setTitle(record.title)
      setBody(nextBody)
      setSourceState(nextSourceState)
      setBodyDirty(Boolean(addition))
      setSaveState(addition ? 'draft' : 'saved')
      setAutoSaveEnabled(!appendCurrentSelection)
    } catch {
      message.error(S.research_notes_load_failed)
      resetNewNote()
    } finally {
      setLoadingNote(false)
    }
  }, [compose, currentLinks, initialMessageId, labels, resetNewNote, S.research_notes_load_failed])

  useEffect(() => {
    if (!open) {
      initializedRef.current = false
      return
    }
    if (initializedRef.current) return
    initializedRef.current = true
    setViewMode('edit')
    setWordExporting(false)
    setSaving(false)
    if (initialNoteId) {
      void loadExistingNote(initialNoteId, false)
    } else {
      resetNewNote()
    }
  }, [initialNoteId, loadExistingNote, open, resetNewNote])

  const editSignature = useMemo(
    () => JSON.stringify([title.trim(), body.trim(), sourceState]),
    [body, sourceState, title],
  )
  useEffect(() => {
    editSignatureRef.current = editSignature
  }, [editSignature])

  useEffect(() => {
    if (!open || currentNote || !initializedRef.current) return undefined
    const timer = window.setTimeout(() => {
      try {
        window.localStorage.setItem(draftKey(projectId, activeConvId, initialMessageId), JSON.stringify({
          title,
          body,
          selectedIds,
          includeShelf,
        }))
      } catch {
        // A browser may block local storage; the in-memory draft remains available while open.
      }
    }, 300)
    return () => window.clearTimeout(timer)
  }, [activeConvId, body, currentNote, includeShelf, initialMessageId, open, projectId, selectedIds, title])

  const safeTitle = title.trim() || S.research_note_default_title
  const fullMarkdown = `# ${safeTitle}\n\n${body.trim()}\n`
  const canSave = body.trim().length > 0
  const canExport = body.trim().length > 0

  const saveNow = useCallback(async (silent = false) => {
    if (!body.trim() || savingRef.current) return
    const signature = editSignatureRef.current
    savingRef.current = true
    setSaving(true)
    setSaveState('saving')
    try {
      const record = currentNote
        ? await chatApi.updateResearchNote(currentNote.id, {
          expected_revision: currentNote.revision,
          title: safeTitle,
          content_markdown: body.trim(),
          source_state: sourceState,
        })
        : await chatApi.createResearchNote({
          title: safeTitle,
          content_markdown: body.trim(),
          project_id: projectId || null,
          source_conv_id: activeConvId || null,
          source_state: sourceState,
        })
      setCurrentNote(record)
      setTargetValue(record.id)
      setAutoSaveEnabled(true)
      setSaveState('saved')
      if (editSignatureRef.current === signature) setBodyDirty(false)
      try {
        window.localStorage.removeItem(draftKey(projectId, activeConvId, initialMessageId))
      } catch {
        // Ignore blocked local storage cleanup.
      }
      onSaved(record)
      if (!silent) message.success(S.research_notes_saved)
    } catch (error) {
      setSaveState('error')
      if (!silent) message.error(S.research_notes_save_failed)
      if (String(error).includes('409')) message.warning(S.research_notes_conflict)
    } finally {
      savingRef.current = false
      setSaving(false)
    }
  }, [activeConvId, body, currentNote, initialMessageId, onSaved, projectId, safeTitle, sourceState, S.research_notes_conflict, S.research_notes_save_failed, S.research_notes_saved])

  useEffect(() => {
    if (!open || !currentNote || !autoSaveEnabled || !bodyDirty || savingRef.current) return undefined
    const timer = window.setTimeout(() => { void saveNow(true) }, 900)
    return () => window.clearTimeout(timer)
  }, [autoSaveEnabled, bodyDirty, currentNote, editSignature, open, saveNow])

  const applyFreshSourceState = (ids: number[], withShelf: boolean) => {
    const links = currentLinks(ids, withShelf)
    setSourceState((current) => ({
      ...current,
      version: 1,
      selected_message_ids: ids,
      include_shelf: withShelf,
      links: mergeLinks(Array.isArray(current.links) ? current.links : [], links),
    }))
  }

  const updateSelection = (messageId: number, checked: boolean) => {
    const next = checked
      ? Array.from(new Set([...selectedIds, messageId]))
      : selectedIds.filter((id) => id !== messageId)
    setSelectedIds(next)
    if (!currentNote && !bodyDirty) {
      setBody(compose(next, includeShelf))
      applyFreshSourceState(next, includeShelf)
    }
  }

  const updateIncludeShelf = (checked: boolean) => {
    setIncludeShelf(checked)
    if (!currentNote && !bodyDirty) {
      setBody(compose(selectedIds, checked))
      applyFreshSourceState(selectedIds, checked)
    }
  }

  const selectAll = () => {
    const next = answers.map((answer) => answer.messageId)
    setSelectedIds(next)
    if (!currentNote && !bodyDirty) {
      setBody(compose(next, includeShelf))
      applyFreshSourceState(next, includeShelf)
    }
  }

  const clearSelection = () => {
    setSelectedIds([])
    if (!currentNote && !bodyDirty) {
      setBody(compose([], includeShelf))
      applyFreshSourceState([], includeShelf)
    }
  }

  const regenerateOrAppend = () => {
    const generated = compose(selectedIds, includeShelf)
    if (currentNote) {
      if (!generated) return
      setBody((current) => appendResearchNoteBody(current, generated, labels))
      applyFreshSourceState(selectedIds, includeShelf)
    } else {
      setBody(generated)
      applyFreshSourceState(selectedIds, includeShelf)
    }
    setBodyDirty(true)
    setSaveState('draft')
    message.success(currentNote ? S.research_notes_appended : S.research_note_rebuilt)
  }

  const copyMarkdown = async () => {
    try {
      await writeClipboard(fullMarkdown)
      message.success(S.research_note_copied)
    } catch {
      message.error(S.research_note_copy_failed)
    }
  }

  const exportWord = async () => {
    if (!canExport || wordExporting) return
    setWordExporting(true)
    try {
      await chatApi.downloadResearchNoteDocx({ title: safeTitle, content_markdown: body.trim() })
      message.success(S.research_note_word_exported)
    } catch {
      message.error(S.research_note_export_failed)
    } finally {
      setWordExporting(false)
    }
  }

  const saveStatusLabel = saveState === 'saving'
    ? S.research_notes_saving
    : saveState === 'saved'
      ? S.research_notes_auto_saved
      : saveState === 'error'
        ? S.research_notes_save_failed
        : currentNote
          ? S.research_notes_unsaved_changes
          : S.research_notes_local_draft

  return (
    <Modal
      className="kb-research-note-modal"
      open={open}
      width={1120}
      title={currentNote ? S.research_notes_edit_title : S.research_note_title}
      onCancel={onClose}
      footer={null}
      destroyOnHidden
    >
      <div className="kb-research-note-intro">{S.research_note_intro}</div>
      {!initialNoteId && initialMessageId && notes.length > 0 ? (
        <label className="kb-research-note-target-field">
          <span>{S.research_notes_save_target}</span>
          <Select
            value={targetValue}
            options={[
              { value: 'new', label: S.research_notes_create_new },
              ...notes.map((note) => ({ value: note.id, label: note.title })),
            ]}
            onChange={(value) => {
              if (value === 'new') resetNewNote()
              else void loadExistingNote(value, true)
            }}
          />
        </label>
      ) : null}
      {loadingNote ? (
        <div className="kb-research-note-loading">{S.research_notes_loading}</div>
      ) : answers.length <= 0 && !currentNote ? (
        <Empty description={S.research_note_empty} />
      ) : (
        <div className="kb-research-note-layout">
          <aside className="kb-research-note-source-panel">
            <div className="kb-research-note-panel-head">
              <div>
                <strong>{currentNote ? S.research_notes_append_answers : S.research_note_choose_answers}</strong>
                <span>{S.research_note_selected.replace('{n}', String(selectedIds.length))}</span>
              </div>
              <div className="kb-research-note-select-actions">
                <Button type="link" size="small" onClick={selectAll}>{S.research_note_select_all}</Button>
                <Button type="link" size="small" onClick={clearSelection}>{S.research_note_clear}</Button>
              </div>
            </div>
            <div className="kb-research-note-answer-list">
              {answers.map((answer, index) => (
                <label key={answer.messageId} className="kb-research-note-answer-option">
                  <Checkbox checked={selectedIds.includes(answer.messageId)} onChange={(event) => updateSelection(answer.messageId, event.target.checked)} />
                  <span>
                    <strong>{answer.question || `${S.research_note_question} ${index + 1}`}</strong>
                    <small>{plainPreview(answer.answerMarkdown).slice(0, 118)}</small>
                    <em>{S.research_note_source_count.replace('{n}', String(answer.citations.length))}</em>
                  </span>
                </label>
              ))}
            </div>
            <div className="kb-research-note-basket-option">
              <Checkbox checked={includeShelf} disabled={shelfItems.length <= 0} onChange={(event) => updateIncludeShelf(event.target.checked)}>
                {S.research_note_include_basket.replace('{n}', String(shelfItems.length))}
              </Checkbox>
              <small>{shelfItems.length > 0 ? S.research_note_include_basket_tip : S.research_note_basket_empty}</small>
            </div>
          </aside>

          <section className="kb-research-note-editor-panel">
            <label className="kb-research-note-title-field">
              <span>{S.research_note_note_title}</span>
              <Input value={title} maxLength={240} onChange={(event) => {
                setTitle(event.target.value)
                setBodyDirty(true)
                setSaveState('draft')
              }} />
            </label>
            <div className="kb-research-note-editor-toolbar">
              <Segmented size="small" value={viewMode} options={[
                { label: S.research_note_edit, value: 'edit' },
                { label: S.research_note_preview, value: 'preview' },
              ]} onChange={(value) => setViewMode(value as 'edit' | 'preview')} />
              <Button size="small" type="text" icon={<ReloadOutlined />} disabled={selectedIds.length <= 0} onClick={regenerateOrAppend}>
                {currentNote ? S.research_notes_append_selected : S.research_note_rebuild}
              </Button>
            </div>
            {bodyDirty ? <div className="kb-research-note-dirty-tip">{S.research_note_manual_edits}</div> : null}
            {viewMode === 'edit' ? (
              <Input.TextArea className="kb-research-note-editor" value={body} autoSize={{ minRows: 21, maxRows: 28 }} onChange={(event) => {
                setBody(event.target.value)
                setBodyDirty(true)
                setSaveState('draft')
              }} />
            ) : (
              <div className="kb-research-note-preview kb-main-scroll">
                <MarkdownRenderer content={fullMarkdown} citeDetails={[]} linkifyPlainCitations={false} />
              </div>
            )}
          </section>
        </div>
      )}
      <div className="kb-research-note-footer">
        <span className={`is-${saveState}`}>{saveStatusLabel}</span>
        <div>
          <Button onClick={onClose}>{S.research_note_close}</Button>
          <Button icon={<CopyOutlined />} disabled={!canExport} onClick={() => { void copyMarkdown() }}>{S.research_note_copy}</Button>
          <Button icon={<FileMarkdownOutlined />} disabled={!canExport} onClick={() => downloadTextFile('pi-zaya-research-note.md', fullMarkdown)}>{S.research_note_markdown}</Button>
          <Button icon={wordExporting ? <DownloadOutlined /> : <FileWordOutlined />} loading={wordExporting} disabled={!canExport} onClick={() => { void exportWord() }}>{S.research_note_word}</Button>
          <Button type="primary" icon={<SaveOutlined />} loading={saving} disabled={!canSave} onClick={() => { void saveNow(false) }}>
            {currentNote ? S.research_notes_save : S.research_notes_save_to_library}
          </Button>
        </div>
      </div>
    </Modal>
  )
}
