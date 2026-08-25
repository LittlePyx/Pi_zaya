import { useEffect, useMemo, useRef, useState } from 'react'
import { Alert, Input, Modal, Select, Spin, Tag, message } from 'antd'
import { FileTextOutlined } from '@ant-design/icons'
import {
  chatApi,
  type Project,
  type ResearchNoteRecord,
  type ResearchNoteSourceLink,
} from '../../api/chat'
import { useT } from '../../i18n'
import {
  appendResearchNoteCapture,
  buildResearchNoteCaptureSection,
  RESEARCH_NOTES_CHANGED_EVENT,
  RESEARCH_NOTES_SYNC_CHANNEL,
  readerCaptureDraftKey,
  readerCaptureSourceLink,
  researchNoteHasCapture,
  type ReaderResearchNoteCapture,
  type ReaderResearchNoteCaptureKind,
} from './readerResearchNoteCapture'

const NEW_NOTE = '__new__'

interface Props {
  open: boolean
  capture: ReaderResearchNoteCapture | null
  conversationId?: string | null
  projectId?: string | null
  onClose: () => void
}

interface CaptureDraft {
  targetId?: string
  title?: string
  comment?: string
  projectId?: string | null
  tags?: string[]
}

function sourceBaseName(value: string): string {
  return String(value || '')
    .split(/[\\/]/)
    .pop()
    ?.replace(/\.(?:pdf|md)$/i, '')
    .replace(/\.en$/i, '')
    .trim() || ''
}

function readDraft(capture: ReaderResearchNoteCapture): CaptureDraft | null {
  try {
    const raw = window.localStorage.getItem(readerCaptureDraftKey(capture))
    if (!raw) return null
    const parsed = JSON.parse(raw)
    return parsed && typeof parsed === 'object' ? parsed as CaptureDraft : null
  } catch {
    return null
  }
}

function notifyResearchNotesChanged(record: ResearchNoteRecord) {
  window.dispatchEvent(new CustomEvent(RESEARCH_NOTES_CHANGED_EVENT, { detail: { noteId: record.id } }))
  if (typeof BroadcastChannel === 'undefined') return
  const channel = new BroadcastChannel(RESEARCH_NOTES_SYNC_CHANNEL)
  channel.postMessage({ type: 'research-notes-changed', noteId: record.id, updatedAt: Date.now() })
  channel.close()
}

function linkList(record: ResearchNoteRecord): ResearchNoteSourceLink[] {
  return Array.isArray(record.source_state?.links)
    ? record.source_state.links.filter((item): item is ResearchNoteSourceLink => Boolean(item && typeof item === 'object'))
    : []
}

function mergeTags(existing: string[], incoming: string[]): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  for (const value of [...existing, ...incoming]) {
    const tag = String(value || '').trim().slice(0, 48)
    const key = tag.toLocaleLowerCase()
    if (!tag || seen.has(key)) continue
    seen.add(key)
    out.push(tag)
  }
  return out.slice(0, 24)
}

export function ReaderResearchNoteCaptureModal({
  open,
  capture,
  conversationId,
  projectId,
  onClose,
}: Props) {
  const S = useT()
  const [notes, setNotes] = useState<ResearchNoteRecord[]>([])
  const [projects, setProjects] = useState<Project[]>([])
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [targetId, setTargetId] = useState(NEW_NOTE)
  const [title, setTitle] = useState('')
  const [comment, setComment] = useState('')
  const [targetProjectId, setTargetProjectId] = useState<string | null>(null)
  const [tags, setTags] = useState<string[]>([])
  const [saveError, setSaveError] = useState('')
  const loadSeqRef = useRef(0)

  const selectedNote = useMemo(
    () => notes.find((note) => note.id === targetId) || null,
    [notes, targetId],
  )
  const kindLabels = useMemo<Record<ReaderResearchNoteCaptureKind, string>>(() => ({
    selection: S.reader_note_kind_selection || '原文摘录',
    table: S.reader_note_kind_table || '表格摘录',
    equation: S.reader_note_kind_equation || '公式摘录',
    figure: S.reader_note_kind_figure || '图片摘录',
  }), [S])
  const capturePreview = capture
    ? buildResearchNoteCaptureSection(capture, comment, {
      kinds: kindLabels,
      location: S.reader_note_location || '原文位置',
      comment: S.reader_note_comment_heading || '我的批注',
    })
    : ''

  useEffect(() => {
    if (!open || !capture) return
    const draft = readDraft(capture)
    const fallbackTitle = `${sourceBaseName(capture.sourceName || capture.sourcePath) || S.research_note_default_title} · ${kindLabels[capture.captureKind]}`
    setTargetId(String(draft?.targetId || NEW_NOTE))
    setTitle(String(draft?.title || fallbackTitle))
    setComment(String(draft?.comment || ''))
    setTargetProjectId(String(draft?.projectId || projectId || '').trim() || null)
    setTags(Array.isArray(draft?.tags) ? draft.tags.map(String).filter(Boolean) : [])
    setSaveError('')
    const sequence = loadSeqRef.current + 1
    loadSeqRef.current = sequence
    setLoading(true)
    Promise.all([
      chatApi.listResearchNotes(null, 200, { scope: 'all', archived: 'active' }),
      chatApi.listProjects(),
    ]).then(([noteRecords, projectRecords]) => {
      if (loadSeqRef.current !== sequence) return
      const activeNotes = Array.isArray(noteRecords) ? noteRecords : []
      setNotes(activeNotes)
      setProjects(Array.isArray(projectRecords) ? projectRecords : [])
      if (draft?.targetId && draft.targetId !== NEW_NOTE && !activeNotes.some((note) => note.id === draft.targetId)) {
        setTargetId(NEW_NOTE)
      }
    }).catch((error) => {
      if (loadSeqRef.current !== sequence) return
      setNotes([])
      setProjects([])
      setSaveError(error instanceof Error ? error.message : S.reader_note_load_failed)
    }).finally(() => {
      if (loadSeqRef.current === sequence) setLoading(false)
    })
    return () => {
      loadSeqRef.current += 1
    }
  }, [S.reader_note_load_failed, S.research_note_default_title, capture, kindLabels, open, projectId])

  useEffect(() => {
    if (!open || !capture) return undefined
    const timer = window.setTimeout(() => {
      try {
        window.localStorage.setItem(readerCaptureDraftKey(capture), JSON.stringify({
          targetId,
          title,
          comment,
          projectId: targetProjectId,
          tags,
        } satisfies CaptureDraft))
      } catch {
        // The in-memory draft remains intact when storage is unavailable.
      }
    }, 250)
    return () => window.clearTimeout(timer)
  }, [capture, comment, open, tags, targetId, targetProjectId, title])

  useEffect(() => {
    if (!selectedNote) return
    setTargetProjectId(selectedNote.project_id || null)
  }, [selectedNote])

  const save = async () => {
    if (!capture || saving) return
    const cleanTitle = title.trim() || S.research_note_default_title
    const section = buildResearchNoteCaptureSection(capture, comment, {
      kinds: kindLabels,
      location: S.reader_note_location || '原文位置',
      comment: S.reader_note_comment_heading || '我的批注',
    })
    const sourceLink = readerCaptureSourceLink(capture, conversationId)
    setSaving(true)
    setSaveError('')
    try {
      let record: ResearchNoteRecord
      if (targetId === NEW_NOTE) {
        record = await chatApi.createResearchNote({
          title: cleanTitle,
          content_markdown: section,
          project_id: targetProjectId,
          source_conv_id: String(conversationId || '').trim() || null,
          source_state: { version: 1, links: [sourceLink] },
          tags,
        })
      } else {
        const latest = await chatApi.getResearchNote(targetId)
        if (researchNoteHasCapture(latest, capture)) {
          const duplicateMessage = S.reader_note_duplicate || '这段原文已在该笔记中。'
          setSaveError(duplicateMessage)
          message.info(duplicateMessage)
          return
        }
        record = await chatApi.updateResearchNote(latest.id, {
          expected_revision: latest.revision,
          content_markdown: appendResearchNoteCapture(latest.content_markdown, section),
          source_state: {
            ...latest.source_state,
            version: 1,
            links: [...linkList(latest), sourceLink].slice(0, 240),
          },
          tags: mergeTags(latest.tags || [], tags),
        })
      }
      try {
        window.localStorage.removeItem(readerCaptureDraftKey(capture))
      } catch {
        // Successful backend save is authoritative even if local storage is unavailable.
      }
      notifyResearchNotesChanged(record)
      message.success(S.reader_note_saved || '已加入研究笔记')
      onClose()
    } catch (error) {
      const detail = error instanceof Error ? error.message : S.research_notes_save_failed
      setSaveError(detail || S.research_notes_save_failed)
      message.error(S.reader_note_save_failed || S.research_notes_save_failed)
    } finally {
      setSaving(false)
    }
  }

  return (
    <Modal
      open={open}
      width={680}
      title={(
        <span className="kb-reader-note-modal-title">
          <FileTextOutlined />
          {S.reader_note_modal_title || '加入研究笔记'}
        </span>
      )}
      okText={S.reader_note_save || '加入笔记'}
      cancelText={S.confirm_cancel}
      okButtonProps={{ disabled: !capture || loading, loading: saving, 'data-testid': 'reader-note-save' }}
      cancelButtonProps={{ disabled: saving }}
      maskClosable={!saving}
      keyboard={!saving}
      onOk={() => { void save() }}
      onCancel={onClose}
      destroyOnHidden={false}
    >
      {loading ? (
        <div className="kb-reader-note-loading"><Spin size="small" /> {S.reader_note_loading || '正在读取研究笔记…'}</div>
      ) : (
        <div className="kb-reader-note-modal" data-testid="reader-note-modal">
          {saveError ? <Alert type="warning" showIcon message={saveError} /> : null}
          <label>
            <span>{S.research_notes_save_target}</span>
            <Select
              value={targetId}
              showSearch
              optionFilterProp="label"
              data-testid="reader-note-target"
              options={[
                { value: NEW_NOTE, label: S.research_notes_create_new },
                ...notes.map((note) => ({ value: note.id, label: note.title || S.research_note_default_title })),
              ]}
              onChange={(value) => {
                setTargetId(value)
                setSaveError('')
              }}
            />
          </label>
          {targetId === NEW_NOTE ? (
            <div className="kb-reader-note-new-fields">
              <label>
                <span>{S.research_note_note_title}</span>
                <Input value={title} maxLength={180} onChange={(event) => setTitle(event.target.value)} data-testid="reader-note-title" />
              </label>
              <label>
                <span>{S.research_notes_project}</span>
                <Select
                  allowClear
                  value={targetProjectId || undefined}
                  placeholder={S.research_notes_unassigned}
                  options={projects.map((project) => ({ value: project.id, label: project.name }))}
                  onChange={(value) => setTargetProjectId(value || null)}
                />
              </label>
            </div>
          ) : selectedNote ? (
            <div className="kb-reader-note-target-summary">
              <strong>{selectedNote.title}</strong>
              <span>{projects.find((project) => project.id === selectedNote.project_id)?.name || S.research_notes_unassigned}</span>
            </div>
          ) : null}
          <label>
            <span>{S.research_notes_tags}</span>
            <Select
              mode="tags"
              value={tags}
              maxCount={24}
              tokenSeparators={[',', '，']}
              placeholder={S.research_notes_tags_placeholder}
              onChange={(values) => setTags(values.map(String))}
            />
          </label>
          <label>
            <span>{S.reader_note_comment || '我的批注（可选）'}</span>
            <Input.TextArea
              value={comment}
              autoSize={{ minRows: 3, maxRows: 7 }}
              maxLength={4000}
              placeholder={S.reader_note_comment_placeholder || '记录这段原文为什么重要、与你的研究问题有什么关系。'}
              onChange={(event) => setComment(event.target.value)}
              data-testid="reader-note-comment"
            />
          </label>
          {capture ? (
            <section className="kb-reader-note-capture-card">
              <div>
                <Tag color="blue">{kindLabels[capture.captureKind]}</Tag>
                <strong>{capture.sourceName || sourceBaseName(capture.sourcePath)}</strong>
                {capture.locationLabel ? <small>{capture.locationLabel}</small> : null}
              </div>
              <pre>{capturePreview.replace(/^##[^\n]*\n+/, '').replace(/^\*\*[^\n]+\*\*[^\n]*\n+/, '').trim()}</pre>
            </section>
          ) : null}
          <div className="kb-reader-note-draft-hint">{S.reader_note_draft_hint || '保存失败时，当前摘录和输入会保留在本机。'}</div>
        </div>
      )}
    </Modal>
  )
}
