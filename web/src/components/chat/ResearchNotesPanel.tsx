import { useMemo, useState } from 'react'
import { Button, Empty, Input, Popconfirm, Spin } from 'antd'
import {
  DeleteOutlined,
  EditOutlined,
  FileTextOutlined,
  LinkOutlined,
  PlusOutlined,
  SearchOutlined,
} from '@ant-design/icons'
import type { ResearchNoteRecord, ResearchNoteSourceLink } from '../../api/chat'
import { useT } from '../../i18n'

interface Props {
  visible: boolean
  loading: boolean
  notes: ResearchNoteRecord[]
  onNew: () => void
  onEdit: (noteId: string) => void
  onDelete: (noteId: string) => void | Promise<void>
  onOpenMessage: (link: ResearchNoteSourceLink) => void | Promise<void>
  onOpenSource: (link: ResearchNoteSourceLink) => void
}

function noteLinks(note: ResearchNoteRecord): ResearchNoteSourceLink[] {
  return Array.isArray(note.source_state?.links)
    ? note.source_state.links.filter((link): link is ResearchNoteSourceLink => Boolean(link && typeof link === 'object'))
    : []
}

function updatedLabel(value: number): string {
  const date = new Date(Number(value || 0) * 1000)
  if (!Number.isFinite(date.getTime())) return ''
  return new Intl.DateTimeFormat(undefined, {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date)
}

export function ResearchNotesPanel({
  visible,
  loading,
  notes,
  onNew,
  onEdit,
  onDelete,
  onOpenMessage,
  onOpenSource,
}: Props) {
  const S = useT()
  const [search, setSearch] = useState('')
  const [expandedId, setExpandedId] = useState('')
  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase()
    if (!query) return notes
    return notes.filter((note) => {
      const sourceText = noteLinks(note).map((link) => link.label).join(' ')
      return `${note.title} ${sourceText}`.toLowerCase().includes(query)
    })
  }, [notes, search])

  if (!visible) return null
  return (
    <div className="kb-research-notes-panel" data-testid="research-notes-panel">
      <div className="kb-research-notes-panel-head">
        <div>
          <strong>{S.research_notes_title}</strong>
          <span>{S.research_notes_count.replace('{n}', String(notes.length))}</span>
        </div>
        <Button size="small" type="primary" icon={<PlusOutlined />} onClick={onNew}>
          {S.research_notes_new}
        </Button>
      </div>
      <Input
        allowClear
        prefix={<SearchOutlined />}
        value={search}
        placeholder={S.research_notes_search}
        onChange={(event) => setSearch(event.target.value)}
      />
      <div className="kb-research-notes-list kb-main-scroll">
        {loading ? (
          <div className="kb-research-notes-loading"><Spin size="small" /></div>
        ) : filtered.length <= 0 ? (
          <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={S.research_notes_empty} />
        ) : filtered.map((note) => {
          const links = noteLinks(note)
          const expanded = expandedId === note.id
          return (
            <article key={note.id} className="kb-research-note-card" data-testid={`research-note-card-${note.id}`}>
              <button type="button" className="kb-research-note-card-main" onClick={() => onEdit(note.id)}>
                <FileTextOutlined />
                <span>
                  <strong>{note.title || S.research_note_default_title}</strong>
                  <small>{S.research_notes_updated.replace('{time}', updatedLabel(note.updated_at))}</small>
                </span>
              </button>
              <div className="kb-research-note-card-meta">
                <button
                  type="button"
                  disabled={links.length <= 0}
                  onClick={() => setExpandedId((current) => current === note.id ? '' : note.id)}
                >
                  <LinkOutlined /> {S.research_notes_sources.replace('{n}', String(links.length))}
                </button>
                <span>
                  <Button size="small" type="text" icon={<EditOutlined />} aria-label={S.research_notes_edit} onClick={() => onEdit(note.id)} />
                  <Popconfirm
                    title={S.research_notes_delete_confirm}
                    okText={S.research_notes_delete}
                    cancelText={S.confirm_cancel}
                    onConfirm={() => onDelete(note.id)}
                  >
                    <Button danger size="small" type="text" icon={<DeleteOutlined />} aria-label={S.research_notes_delete} />
                  </Popconfirm>
                </span>
              </div>
              {expanded ? (
                <div className="kb-research-note-source-links">
                  {links.map((link, index) => (
                    <button
                      key={`${link.kind}-${link.message_id || 0}-${link.source_path || ''}-${index}`}
                      type="button"
                      onClick={() => {
                        if (link.kind === 'answer') void onOpenMessage(link)
                        else onOpenSource(link)
                      }}
                    >
                      <span>{link.kind === 'answer' ? S.research_notes_answer_source : S.research_notes_paper_source}</span>
                      <strong>{link.label || S.research_note_untitled_source}</strong>
                      {link.location_label || link.heading_path ? <small>{link.location_label || link.heading_path}</small> : null}
                    </button>
                  ))}
                </div>
              ) : null}
            </article>
          )
        })}
      </div>
    </div>
  )
}
