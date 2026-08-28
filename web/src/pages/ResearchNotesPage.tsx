import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Alert,
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
  DiffOutlined,
  FileMarkdownOutlined,
  FileTextOutlined,
  FileWordOutlined,
  FolderOutlined,
  HistoryOutlined,
  InboxOutlined,
  LinkOutlined,
  PlusOutlined,
  PushpinFilled,
  PushpinOutlined,
  ReloadOutlined,
  SaveOutlined,
  SearchOutlined,
} from '@ant-design/icons'
import { useLocation, useNavigate } from 'react-router-dom'
import {
  chatApi,
  type MessageProvenanceReaderOpen,
  type ResearchNoteEvidenceBlock,
  type ResearchNoteEvidenceBaselineItem,
  type ResearchNoteEvidenceBaselineSummary,
  type ResearchNoteEvidenceContext,
  type ResearchNoteEvidenceRepairIntent,
  type ResearchNoteEvidenceReviewIntent,
  type ResearchNoteRecord,
  type ResearchNoteSourceLink,
  type ResearchNoteSourceState,
  type ResearchNoteUpdateBody,
} from '../api/chat'
import { MarkdownRenderer } from '../components/chat/MarkdownRenderer'
import { useT } from '../i18n'
import { useChatStore } from '../stores/chatStore'
import {
  RESEARCH_NOTES_CHANGED_EVENT,
  RESEARCH_NOTES_SYNC_CHANNEL,
} from '../components/chat/readerResearchNoteCapture'
import {
  clearResearchNoteWorkspaceDraft,
  readResearchNoteWorkspaceDraft,
  workspaceDraftMatchesRecord,
  writeResearchNoteWorkspaceDraft,
} from './researchNoteWorkspaceDraft'
import {
  RESEARCH_NOTE_EVIDENCE_SYSTEM_RETRY_REASONS,
  RESEARCH_NOTE_EVIDENCE_USER_DECISION_REASONS,
  researchNoteEvidenceMaintenanceDue,
  researchNoteEvidenceMaintenanceTracked,
  researchNoteEvidenceNextRetryAt,
  researchNoteEvidenceReviewIssues,
  researchNoteEvidenceReviewSummary,
  researchNoteEvidenceSystemRetryRequired,
  type ResearchNoteEvidenceReviewSummary,
} from './researchNoteEvidenceReview'
import {
  buildResearchNoteOutline,
  researchNoteOutlineConversationId,
  researchNoteOutlineHasMixedProjects,
  researchNoteOutlineProjectId,
  researchNoteOutlineSectionContent,
  researchNoteOutlineSectionHasManualEdits,
  researchNoteOutlineSourceState,
  researchNoteOutlineSourcesFromState,
  researchNoteOutlineTags,
  refreshResearchNoteOutlineSource,
  type ResearchNoteOutlineSource,
} from './researchNoteOutline'

const UNASSIGNED_PROJECT = '__unassigned__'
const ALL_FILTER = '__all__'
type SaveIssue = 'failed' | 'conflict' | null
type OutlineFreshnessStatus = 'checking' | 'current' | 'updated' | 'missing' | 'error'
type EvidenceHealthStatus = 'exact' | 'changed' | 'candidate' | 'invalid' | 'recovering' | 'unchecked'
type EvidenceHealthFilter = 'all' | 'issues'
type DirectoryEvidenceFilter = 'all' | 'issues'

interface EvidenceReviewState {
  loading: boolean
  error?: string
  data?: ResearchNoteEvidenceContext
}

interface EvidenceHealthRunState {
  started: boolean
  running: boolean
  total: number
  completed: number
}

interface EvidenceBaselineRunState {
  running: boolean
  summary?: ResearchNoteEvidenceBaselineSummary
  items?: ResearchNoteEvidenceBaselineItem[]
  sourceSignature?: string
}

interface EvidenceLinkEntry {
  link: ResearchNoteSourceLink
  index: number
  reviewKey: string
}

const EMPTY_EVIDENCE_HEALTH_RUN: EvidenceHealthRunState = {
  started: false,
  running: false,
  total: 0,
  completed: 0,
}

const EMPTY_EVIDENCE_BASELINE_RUN: EvidenceBaselineRunState = {
  running: false,
}

interface OutlineSourceFreshness {
  source: ResearchNoteOutlineSource
  status: OutlineFreshnessStatus
  latestRevision?: number
  latestTitle?: string
}

interface OutlineSourceReview {
  source: ResearchNoteOutlineSource
  latest: ResearchNoteRecord
  currentSection: string
  hasManualEdits: boolean
}

function sourceStateLinks(sourceState: ResearchNoteSourceState | null | undefined): ResearchNoteSourceLink[] {
  return Array.isArray(sourceState?.links)
    ? sourceState.links.filter((link): link is ResearchNoteSourceLink => Boolean(link && typeof link === 'object'))
    : []
}

function noteLinks(note: ResearchNoteRecord | null | undefined): ResearchNoteSourceLink[] {
  return sourceStateLinks(note?.source_state)
}

function sourceLabel(link: ResearchNoteSourceLink): string {
  return String(link.source_name || link.label || '').trim()
}

function sourceReviewKey(link: ResearchNoteSourceLink, index: number): string {
  return [
    link.capture_id,
    link.source_path,
    link.block_id,
    link.anchor_id,
    index,
  ].map(value => String(value || '')).join(':')
}

function evidenceBaselineSourceSignature(sourceState: ResearchNoteSourceState | null | undefined): string {
  return JSON.stringify(sourceStateLinks(sourceState).map((link, index) => [
    index,
    link.kind,
    link.capture_id,
    link.source_path,
    link.block_id,
    link.anchor_id,
    link.evidence_quote,
    link.evidence_verification?.status,
    link.evidence_verification?.source_fingerprint,
    link.evidence_context_snapshot?.source_fingerprint,
    link.evidence_baseline_check?.status,
    link.evidence_baseline_check?.reason,
    link.evidence_baseline_check?.checked_at,
    link.evidence_baseline_check?.source_fingerprint,
  ].map(value => String(value || ''))))
}

function mergeEvidenceBaselineState(
  currentState: ResearchNoteSourceState,
  storedState: ResearchNoteSourceState,
): ResearchNoteSourceState {
  const currentLinks = Array.isArray(currentState.links) ? currentState.links : []
  const storedLinks = Array.isArray(storedState.links) ? storedState.links : []
  return {
    ...currentState,
    links: currentLinks.map((link, index) => {
      const stored = storedLinks[index]
      if (!stored || sourceReviewKey(stored, index) !== sourceReviewKey(link, index)) return link
      const merged: ResearchNoteSourceLink = { ...link }
      if (
        link.evidence_context_snapshot == null
        && stored.evidence_context_snapshot != null
        && stored.evidence_verification?.reason === 'baseline_backfilled'
      ) {
        merged.evidence_context_snapshot = stored.evidence_context_snapshot
        merged.evidence_verification = stored.evidence_verification
        merged.evidence_context_history = stored.evidence_context_history
      }
      if (stored.evidence_baseline_check) merged.evidence_baseline_check = stored.evidence_baseline_check
      else delete merged.evidence_baseline_check
      if (stored.evidence_baseline_check_history) {
        merged.evidence_baseline_check_history = stored.evidence_baseline_check_history
      } else {
        delete merged.evidence_baseline_check_history
      }
      return merged
    }),
  }
}

function evidenceHealthStatus(
  entry: EvidenceLinkEntry,
  review: EvidenceReviewState | undefined,
  started: boolean,
): EvidenceHealthStatus {
  if (entry.link.evidence_verification?.status === 'invalid') return 'invalid'
  const check = entry.link.evidence_baseline_check
  const checkReason = String(check?.reason || '')
  if (RESEARCH_NOTE_EVIDENCE_USER_DECISION_REASONS.has(checkReason)) {
    return checkReason === 'candidate' ? 'candidate' : 'changed'
  }
  if (
    check?.status === 'failed'
    || RESEARCH_NOTE_EVIDENCE_SYSTEM_RETRY_REASONS.has(checkReason)
  ) return 'recovering'
  if (review?.data?.source_changed) return 'changed'
  if (review?.data?.status === 'candidate') return 'candidate'
  if (review?.data?.status === 'exact') return 'exact'
  if (review?.error) return 'recovering'
  if (entry.link.evidence_context_snapshot != null) return 'exact'
  if (!started || review?.loading) return 'unchecked'
  if (!String(entry.link.source_path || '').trim()) return 'recovering'
  return 'unchecked'
}

function evidenceTargetMatchesSavedLocator(
  link: ResearchNoteSourceLink,
  target: ResearchNoteEvidenceBlock | null | undefined,
): boolean {
  if (!target) return false
  if (link.block_id && target.block_id !== link.block_id) return false
  if (link.anchor_id && target.anchor_id !== link.anchor_id) return false
  return Boolean(link.block_id || link.anchor_id)
}

function evidenceLocationLabel(
  S: ReturnType<typeof useT>,
  headingPath?: string,
  pageStart?: number | null,
  pageEnd?: number | null,
): string {
  const parts: string[] = []
  if (headingPath) parts.push(headingPath)
  if (pageStart) {
    const page = pageEnd && pageEnd !== pageStart ? `${pageStart}–${pageEnd}` : String(pageStart)
    parts.push(S.research_notes_evidence_page.replace('{page}', page))
  }
  return parts.join(' · ')
}

function evidenceReasonLabel(S: ReturnType<typeof useT>, reason: string): string {
  const labels: Record<string, string> = {
    exact_block: S.research_notes_evidence_reason_exact_block,
    exact_anchor: S.research_notes_evidence_reason_exact_anchor,
    quote_relocated: S.research_notes_evidence_reason_quote_relocated,
    context_disambiguated: S.research_notes_evidence_reason_context_disambiguated,
    ambiguous_quote: S.research_notes_evidence_reason_ambiguous,
    saved_target_missing: S.research_notes_evidence_reason_target_missing,
    converted_markdown_missing: S.research_notes_evidence_reason_markdown_missing,
    no_source_blocks: S.research_notes_evidence_reason_no_blocks,
    no_match: S.research_notes_evidence_reason_no_match,
  }
  return labels[reason] || S.research_notes_evidence_reason_unknown
}

function evidenceContextMatchFieldLabel(S: ReturnType<typeof useT>, field: string): string {
  const labels: Record<string, string> = {
    target: S.research_notes_evidence_candidate_match_target,
    before: S.research_notes_evidence_candidate_match_before,
    after: S.research_notes_evidence_candidate_match_after,
    heading: S.research_notes_evidence_candidate_match_heading,
    page: S.research_notes_evidence_candidate_match_page,
  }
  return labels[field] || field
}

function renderEvidenceTextDifference(
  savedText: string,
  candidateText: string,
  side: 'saved' | 'candidate',
) {
  const left = Array.from(savedText)
  const right = Array.from(candidateText)
  let prefixLength = 0
  while (prefixLength < left.length && prefixLength < right.length && left[prefixLength] === right[prefixLength]) {
    prefixLength += 1
  }
  let suffixLength = 0
  while (
    suffixLength < left.length - prefixLength
    && suffixLength < right.length - prefixLength
    && left[left.length - 1 - suffixLength] === right[right.length - 1 - suffixLength]
  ) {
    suffixLength += 1
  }
  const source = side === 'saved' ? left : right
  const prefix = source.slice(0, prefixLength).join('')
  const changed = source.slice(prefixLength, source.length - suffixLength).join('')
  const suffix = suffixLength > 0 ? source.slice(source.length - suffixLength).join('') : ''
  if (!changed && savedText === candidateText) return <>{source.join('')}</>
  return (
    <>
      {prefix}
      {changed ? <mark className={`is-${side}`}>{changed}</mark> : <mark className={`is-${side} is-empty`}>∅</mark>}
      {suffix}
    </>
  )
}

function evidenceBaselineReasonLabel(S: ReturnType<typeof useT>, reason: string): string {
  const labels: Record<string, string> = {
    source_changed: S.research_notes_evidence_baseline_reason_source_changed,
    location_changed: S.research_notes_evidence_baseline_reason_location_changed,
    candidate: S.research_notes_evidence_baseline_reason_candidate,
    invalid: S.research_notes_evidence_baseline_reason_invalid,
    unavailable: S.research_notes_evidence_baseline_reason_unavailable,
    verification_failed: S.research_notes_evidence_baseline_reason_verification_failed,
    source_changed_during_capture: S.research_notes_evidence_baseline_reason_source_changed_during_capture,
  }
  return labels[reason] || S.research_notes_evidence_baseline_reason_unknown
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

export default function ResearchNotesPage() {
  const S = useT()
  const navigate = useNavigate()
  const location = useLocation()
  const requestedNoteId = useMemo(
    () => String(new URLSearchParams(location.search).get('note') || '').trim(),
    [location.search],
  )
  const projects = useChatStore(state => state.projects)
  const [notes, setNotes] = useState<ResearchNoteRecord[]>([])
  const [activeNote, setActiveNote] = useState<ResearchNoteRecord | null>(null)
  const activeNoteRef = useRef<ResearchNoteRecord | null>(null)
  const [activeId, setActiveId] = useState('')
  const [loading, setLoading] = useState(true)
  const [evidenceMaintaining, setEvidenceMaintaining] = useState(false)
  const [loadingNote, setLoadingNote] = useState(false)
  const [saving, setSaving] = useState(false)
  const [dirty, setDirty] = useState(false)
  const dirtyRef = useRef(false)
  const savingRef = useRef(false)
  const [saveIssue, setSaveIssue] = useState<SaveIssue>(null)
  const [conflictRecord, setConflictRecord] = useState<ResearchNoteRecord | null>(null)
  const draftVersionRef = useRef(0)
  const listRequestRef = useRef(0)
  const noteRequestRef = useRef(0)
  const activeIdRef = useRef('')
  const requestedNoteHandledRef = useRef('')
  const evidenceMaintenanceFailureAtRef = useRef(0)

  const [query, setQuery] = useState('')
  const [archiveView, setArchiveView] = useState<'active' | 'archived'>('active')
  const [projectFilter, setProjectFilter] = useState(ALL_FILTER)
  const [sourceFilter, setSourceFilter] = useState(ALL_FILTER)
  const [tagFilter, setTagFilter] = useState(ALL_FILTER)
  const [timeFilter, setTimeFilter] = useState<'all' | 'week' | 'month'>('all')
  const [directoryEvidenceFilter, setDirectoryEvidenceFilter] = useState<DirectoryEvidenceFilter>('all')
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [editorMode, setEditorMode] = useState<'edit' | 'preview'>('edit')

  const [draftTitle, setDraftTitle] = useState('')
  const [draftContent, setDraftContent] = useState('')
  const [draftProjectId, setDraftProjectId] = useState<string | null>(null)
  const [draftTags, setDraftTags] = useState<string[]>([])
  const [draftSourceState, setDraftSourceState] = useState<ResearchNoteSourceState>({})
  const [draftBaseRevision, setDraftBaseRevision] = useState(1)
  const [evidenceReviews, setEvidenceReviews] = useState<Record<string, EvidenceReviewState>>({})
  const [candidateDecisionKey, setCandidateDecisionKey] = useState('')
  const [evidenceHealthRun, setEvidenceHealthRun] = useState<EvidenceHealthRunState>(EMPTY_EVIDENCE_HEALTH_RUN)
  const [evidenceHealthFilter, setEvidenceHealthFilter] = useState<EvidenceHealthFilter>('all')
  const [evidenceBaselineRun, setEvidenceBaselineRun] = useState<EvidenceBaselineRunState>(EMPTY_EVIDENCE_BASELINE_RUN)
  const [focusedEvidenceIndex, setFocusedEvidenceIndex] = useState<number | null>(null)
  const evidenceHealthRunRef = useRef(0)
  const evidenceBaselineRunRef = useRef(0)
  const links = useMemo(() => sourceStateLinks(draftSourceState), [draftSourceState])

  activeNoteRef.current = activeNote
  dirtyRef.current = dirty
  savingRef.current = saving

  const [outlineCheckVersion, setOutlineCheckVersion] = useState(0)
  const [outlineFreshness, setOutlineFreshness] = useState<OutlineSourceFreshness[]>([])
  const [outlineChecking, setOutlineChecking] = useState(false)
  const [outlineReviewLoadingId, setOutlineReviewLoadingId] = useState('')
  const [outlineReview, setOutlineReview] = useState<OutlineSourceReview | null>(null)
  const outlineCheckRequestRef = useRef(0)

  const [combineOpen, setCombineOpen] = useState(false)
  const [combineTitle, setCombineTitle] = useState(S.research_notes_combine_default_title)
  const [combineContent, setCombineContent] = useState('')
  const [combinePreview, setCombinePreview] = useState(false)
  const [combining, setCombining] = useState(false)
  const [combineSaving, setCombineSaving] = useState(false)
  const [combineRecords, setCombineRecords] = useState<ResearchNoteRecord[]>([])
  const [combineProjectId, setCombineProjectId] = useState<string | null>(null)

  useEffect(() => {
    evidenceHealthRunRef.current += 1
    evidenceBaselineRunRef.current += 1
    setEvidenceReviews({})
    setEvidenceHealthRun(EMPTY_EVIDENCE_HEALTH_RUN)
    setEvidenceHealthFilter('all')
    setEvidenceBaselineRun(EMPTY_EVIDENCE_BASELINE_RUN)
    setFocusedEvidenceIndex(null)
  }, [activeId])

  const markDirty = useCallback(() => {
    draftVersionRef.current += 1
    setSaveIssue(current => current === 'conflict' ? current : null)
    setDirty(true)
  }, [])

  const applyRecord = useCallback((
    record: ResearchNoteRecord,
    options: { focusFirstEvidenceIssue?: boolean } = {},
  ) => {
    const localDraft = readResearchNoteWorkspaceDraft(record.id)
    const shouldRestoreDraft = Boolean(localDraft && !workspaceDraftMatchesRecord(localDraft, record))
    draftVersionRef.current += 1
    activeIdRef.current = record.id
    setActiveNote(record)
    setActiveId(record.id)
    setOutlineReview(null)
    if (localDraft && shouldRestoreDraft) {
      const nextSourceState = localDraft.sourceState || record.source_state || {}
      setDraftTitle(localDraft.title)
      setDraftContent(localDraft.contentMarkdown)
      setDraftProjectId(localDraft.projectId)
      setDraftTags(localDraft.tags)
      setDraftSourceState(nextSourceState)
      setDraftBaseRevision(localDraft.baseRevision)
      const conflicted = localDraft.baseRevision !== record.revision
      setConflictRecord(conflicted ? record : null)
      setSaveIssue(conflicted ? 'conflict' : null)
      setDirty(true)
      if (options.focusFirstEvidenceIssue) {
        setFocusedEvidenceIndex(researchNoteEvidenceReviewSummary(nextSourceState).firstSourceIndex)
      }
      return
    }
    clearResearchNoteWorkspaceDraft(record.id)
    setDraftTitle(record.title)
    setDraftContent(record.content_markdown)
    setDraftProjectId(record.project_id || null)
    setDraftTags(Array.isArray(record.tags) ? record.tags : [])
    setDraftSourceState(record.source_state || {})
    setDraftBaseRevision(record.revision)
    setConflictRecord(null)
    setSaveIssue(null)
    setDirty(false)
    if (options.focusFirstEvidenceIssue) {
      setFocusedEvidenceIndex(researchNoteEvidenceReviewSummary(record.source_state).firstSourceIndex)
    }
  }, [])

  const upsertSummary = useCallback((record: ResearchNoteRecord) => {
    setNotes(current => {
      const next = current.filter(item => item.id !== record.id)
      if ((archiveView === 'archived') !== Boolean(record.archived)) return next
      next.push({ ...record, content_markdown: '' })
      return next.sort((a, b) => Number(b.pinned) - Number(a.pinned) || b.updated_at - a.updated_at)
    })
  }, [archiveView])

  const autoResolveEvidenceRecord = useCallback(async (
    record: ResearchNoteRecord,
  ): Promise<{ record: ResearchNoteRecord; checked: boolean }> => {
    const localDraft = readResearchNoteWorkspaceDraft(record.id)
    if (localDraft && !workspaceDraftMatchesRecord(localDraft, record)) return { record, checked: false }
    if (
      activeNoteRef.current?.id === record.id
      && (dirtyRef.current || savingRef.current)
    ) return { record, checked: false }

    try {
      const result = await chatApi.refreshResearchNoteEvidenceHealth(
        record.id,
        record.revision,
      )
      return {
        record: result?.note?.id === record.id ? result.note : record,
        checked: true,
      }
    } catch (error) {
      const detail = error instanceof Error ? error.message : ''
      if (detail.includes('409')) {
        try {
          return { record: await chatApi.getResearchNote(record.id), checked: true }
        } catch {
          return { record, checked: false }
        }
      }
      return { record, checked: false }
    }
  }, [])

  const autoResolveListedEvidence = useCallback(async (
    records: ResearchNoteRecord[],
    protectedNoteId = '',
  ): Promise<{ records: ResearchNoteRecord[]; updated: Map<string, ResearchNoteRecord> }> => {
    const candidates = records.filter(record => {
      if (
        (
          record.id === protectedNoteId
          && (
            activeNoteRef.current?.id !== record.id
            || !researchNoteEvidenceSystemRetryRequired(record.source_state)
          )
        )
        || !researchNoteEvidenceMaintenanceTracked(record.source_state)
        || !researchNoteEvidenceMaintenanceDue(record.source_state)
      ) return false
      if (
        activeNoteRef.current?.id === record.id
        && (dirtyRef.current || savingRef.current)
      ) return false
      const localDraft = readResearchNoteWorkspaceDraft(record.id)
      return !(localDraft && !workspaceDraftMatchesRecord(localDraft, record))
    }).slice(0, 64)
    if (candidates.length <= 0) return { records, updated: new Map() }
    try {
      const result = await chatApi.maintainResearchNoteEvidence(candidates.map(record => ({
        note_id: record.id,
        expected_revision: record.revision,
      })))
      if (!Array.isArray(result?.notes)) throw new Error('invalid evidence maintenance response')
      evidenceMaintenanceFailureAtRef.current = Number(result.summary?.errors || 0) > 0
        ? Date.now()
        : 0
      const updated = new Map(
        result.notes
          .filter(record => record && typeof record.id === 'string')
          .map(record => [record.id, record]),
      )
      return {
        records: records.map(record => updated.get(record.id) || record),
        updated,
      }
    } catch {
      evidenceMaintenanceFailureAtRef.current = Date.now()
      return { records, updated: new Map() }
    }
  }, [])

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
      setLoading(false)
      const requested = requestedNoteId && records.some(note => note.id === requestedNoteId)
        ? requestedNoteId
        : ''
      const protectedNoteId = activeIdRef.current || requested || records[0]?.id || ''
      setEvidenceMaintaining(true)
      const resolved = await autoResolveListedEvidence(records, protectedNoteId)
      if (requestId !== listRequestRef.current) return
      setNotes(resolved.records)
      const activeRecord = activeNoteRef.current
      const activeUpdate = activeRecord ? resolved.updated.get(activeRecord.id) : undefined
      if (activeRecord && activeUpdate && !dirtyRef.current && !savingRef.current) {
        const mergedRecord: ResearchNoteRecord = {
          ...activeRecord,
          ...activeUpdate,
          content_markdown: activeRecord.content_markdown,
        }
        setActiveNote(mergedRecord)
        setDraftSourceState(activeUpdate.source_state || {})
        setDraftBaseRevision(activeUpdate.revision)
        const total = noteLinks(mergedRecord).filter(link => link.kind !== 'answer').length
        setEvidenceHealthRun({ started: true, running: false, total, completed: total })
        setEvidenceHealthFilter(researchNoteEvidenceReviewIssues(mergedRecord.source_state).length > 0 ? 'issues' : 'all')
      }
    } catch (error) {
      if (requestId === listRequestRef.current) {
        message.error(error instanceof Error ? error.message : S.research_notes_load_failed)
      }
    } finally {
      if (requestId === listRequestRef.current) {
        setLoading(false)
        setEvidenceMaintaining(false)
      }
    }
  }, [S.research_notes_load_failed, archiveView, autoResolveListedEvidence, query, requestedNoteId])

  useEffect(() => {
    const timer = window.setTimeout(() => { void loadNotes() }, 180)
    return () => window.clearTimeout(timer)
  }, [loadNotes])

  useEffect(() => {
    if (loading || evidenceMaintaining || notes.length <= 0) return
    const eligible = notes.filter(note => {
      if (!researchNoteEvidenceMaintenanceTracked(note.source_state)) return false
      if (
        note.id === activeIdRef.current
        && (
          dirtyRef.current
          || savingRef.current
          || !researchNoteEvidenceSystemRetryRequired(note.source_state)
        )
      ) return false
      const localDraft = readResearchNoteWorkspaceDraft(note.id)
      return !(localDraft && !workspaceDraftMatchesRecord(localDraft, note))
    })
    const now = Date.now()
    const hasImmediate = eligible.some(note => researchNoteEvidenceMaintenanceDue(note.source_state, now / 1000))
    const retryTimes = eligible.flatMap(note => {
      const retryAt = researchNoteEvidenceNextRetryAt(note.source_state)
      return retryAt == null ? [] : [retryAt * 1000]
    })
    if (!hasImmediate && retryTimes.length <= 0) return
    const failureDelay = Math.max(0, evidenceMaintenanceFailureAtRef.current + 30_000 - now)
    const dueDelay = hasImmediate ? 250 : Math.max(250, Math.min(...retryTimes) - now + 50)
    const timer = window.setTimeout(() => { void loadNotes() }, Math.max(dueDelay, failureDelay))
    return () => window.clearTimeout(timer)
  }, [evidenceMaintaining, loading, loadNotes, notes])

  useEffect(() => {
    const refresh = () => {
      void loadNotes()
      setOutlineCheckVersion(current => current + 1)
    }
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
    if (noteId === activeIdRef.current && activeNote) {
      setFocusedEvidenceIndex(researchNoteEvidenceReviewSummary(draftSourceState).firstSourceIndex)
      return
    }
    const requestId = ++noteRequestRef.current
    const previousId = activeIdRef.current
    activeIdRef.current = noteId
    setActiveId(noteId)
    setLoadingNote(true)
    try {
      let record = await chatApi.getResearchNote(noteId)
      if (requestId !== noteRequestRef.current) return
      const evidenceRefresh = await autoResolveEvidenceRecord(record)
      record = evidenceRefresh.record
      if (requestId !== noteRequestRef.current) return
      upsertSummary(record)
      applyRecord(record, { focusFirstEvidenceIssue: true })
      if (evidenceRefresh.checked) {
        const total = noteLinks(record).filter(link => link.kind !== 'answer').length
        setEvidenceHealthRun({ started: true, running: false, total, completed: total })
        setEvidenceHealthFilter(researchNoteEvidenceReviewIssues(record.source_state).length > 0 ? 'issues' : 'all')
      }
    } catch (error) {
      if (requestId === noteRequestRef.current) {
        activeIdRef.current = previousId
        setActiveId(previousId)
        message.error(error instanceof Error ? error.message : S.research_notes_load_failed)
      }
    } finally {
      if (requestId === noteRequestRef.current) setLoadingNote(false)
    }
  }, [S.research_notes_load_failed, activeNote, applyRecord, autoResolveEvidenceRecord, draftSourceState, upsertSummary])

  useEffect(() => {
    if (!activeNote || focusedEvidenceIndex == null || loadingNote) return
    const frame = window.requestAnimationFrame(() => {
      const sourceCard = document.querySelector<HTMLElement>(
        `[data-testid="research-note-source-${focusedEvidenceIndex}"]`,
      )
      sourceCard?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
      sourceCard?.focus({ preventScroll: true })
    })
    return () => window.cancelAnimationFrame(frame)
  }, [activeNote, focusedEvidenceIndex, loadingNote])

  useEffect(() => {
    if (loading || notes.length <= 0) return
    const requested = requestedNoteId
      && requestedNoteHandledRef.current !== requestedNoteId
      && notes.some(note => note.id === requestedNoteId)
      ? requestedNoteId
      : ''
    if (requested) requestedNoteHandledRef.current = requested
    const nextId = requested || (!activeId ? notes[0].id : '')
    if (nextId && nextId !== activeId) void openNote(nextId)
  }, [activeId, loading, notes, openNote, requestedNoteId])

  useEffect(() => {
    if (!requestedNoteId) requestedNoteHandledRef.current = ''
  }, [requestedNoteId])

  useEffect(() => {
    const params = new URLSearchParams(location.search)
    const repaired = params.get('evidence_repaired') === '1'
    const reviewDecision = params.get('evidence_reviewed')
    if (!repaired && reviewDecision !== 'confirmed' && reviewDecision !== 'invalid') return
    if (repaired) message.success(S.research_notes_evidence_repair_saved)
    if (reviewDecision === 'confirmed') message.success(S.research_notes_evidence_review_confirmed)
    if (reviewDecision === 'invalid') message.success(S.research_notes_evidence_review_invalidated)
    params.delete('evidence_repaired')
    params.delete('evidence_reviewed')
    navigate({ pathname: location.pathname, search: params.toString() ? `?${params.toString()}` : '' }, { replace: true })
  }, [
    S.research_notes_evidence_repair_saved,
    S.research_notes_evidence_review_confirmed,
    S.research_notes_evidence_review_invalidated,
    location.pathname,
    location.search,
    navigate,
  ])

  useEffect(() => {
    if (!activeNote || !dirty) return
    writeResearchNoteWorkspaceDraft({
      noteId: activeNote.id,
      baseRevision: draftBaseRevision,
      title: draftTitle,
      contentMarkdown: draftContent,
      projectId: draftProjectId,
      tags: draftTags,
      sourceState: draftSourceState,
    })
  }, [activeNote, dirty, draftBaseRevision, draftContent, draftProjectId, draftSourceState, draftTags, draftTitle])

  useEffect(() => {
    if (!dirty && !saving) return
    const keepDraft = (event: BeforeUnloadEvent) => {
      event.preventDefault()
      event.returnValue = ''
    }
    window.addEventListener('beforeunload', keepDraft)
    return () => window.removeEventListener('beforeunload', keepDraft)
  }, [dirty, saving])

  const outlineSources = useMemo(
    () => researchNoteOutlineSourcesFromState(draftSourceState),
    [draftSourceState],
  )
  const outlineSourceSignature = useMemo(
    () => outlineSources.map(source => `${source.id}:${source.revision}`).join('|'),
    [outlineSources],
  )

  useEffect(() => {
    const requestId = ++outlineCheckRequestRef.current
    if (!activeNote || outlineSources.length <= 0) {
      setOutlineChecking(false)
      setOutlineFreshness([])
      return
    }
    setOutlineChecking(true)
    setOutlineFreshness(outlineSources.map(source => ({ source, status: 'checking' })))
    void chatApi.listResearchNotes(null, 500, { scope: 'all', archived: 'all' })
      .then(records => {
        if (requestId !== outlineCheckRequestRef.current || activeIdRef.current !== activeNote.id) return
        const latestById = new Map(records.map(record => [record.id, record]))
        setOutlineFreshness(outlineSources.map(source => {
          const latest = latestById.get(source.id)
          if (!latest) return { source, status: 'missing' }
          return {
            source,
            status: latest.revision > source.revision ? 'updated' : 'current',
            latestRevision: latest.revision,
            latestTitle: latest.title,
          }
        }))
      })
      .catch(() => {
        if (requestId !== outlineCheckRequestRef.current || activeIdRef.current !== activeNote.id) return
        setOutlineFreshness(outlineSources.map(source => ({ source, status: 'error' })))
      })
      .finally(() => {
        if (requestId === outlineCheckRequestRef.current && activeIdRef.current === activeNote.id) {
          setOutlineChecking(false)
        }
      })
    return () => {
      if (requestId === outlineCheckRequestRef.current) outlineCheckRequestRef.current += 1
    }
  }, [activeNote, outlineCheckVersion, outlineSourceSignature, outlineSources])

  const reviewOutlineSourceUpdate = async (item: OutlineSourceFreshness) => {
    if (!activeNote || item.status !== 'updated') return
    const noteId = activeNote.id
    setOutlineReviewLoadingId(item.source.id)
    try {
      const latest = await chatApi.getResearchNote(item.source.id)
      if (activeIdRef.current !== noteId) return
      const currentSection = researchNoteOutlineSectionContent(
        draftContent,
        outlineSources,
        item.source.id,
      )
      if (currentSection === null) {
        message.error(S.research_notes_outline_section_missing)
        return
      }
      setOutlineReview({
        source: item.source,
        latest,
        currentSection,
        hasManualEdits: researchNoteOutlineSectionHasManualEdits(
          draftContent,
          outlineSources,
          item.source.id,
        ),
      })
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_notes_outline_refresh_load_failed)
    } finally {
      setOutlineReviewLoadingId('')
    }
  }

  const applyOutlineSourceRefresh = () => {
    if (!outlineReview || saveIssue === 'conflict') return
    const result = refreshResearchNoteOutlineSource(
      draftContent,
      draftSourceState,
      outlineReview.latest,
    )
    if (!result) {
      message.error(S.research_notes_outline_section_missing)
      return
    }
    setDraftContent(result.contentMarkdown)
    setDraftSourceState(result.sourceState)
    markDirty()
    setOutlineReview(null)
    message.success(S.research_notes_outline_refresh_applied)
  }

  const saveDraftNow = useCallback(async (): Promise<boolean> => {
    if (!activeNote || !dirty || saving || saveIssue === 'conflict') return false
    const noteId = activeNote.id
    const version = draftVersionRef.current
    setSaving(true)
    setSaveIssue(null)
    try {
      const record = await chatApi.updateResearchNote(noteId, {
        expected_revision: draftBaseRevision,
        title: draftTitle.trim() || S.research_notes_new_default_title,
        content_markdown: draftContent.trim() || S.research_notes_new_default_body,
        project_id: draftProjectId,
        tags: draftTags,
        source_state: draftSourceState,
      })
      upsertSummary(record)
      if (activeIdRef.current === noteId) {
        setActiveNote(record)
        setDraftSourceState(record.source_state || {})
        setDraftBaseRevision(record.revision)
        setConflictRecord(null)
        setSaveIssue(null)
        if (draftVersionRef.current === version) {
          clearResearchNoteWorkspaceDraft(noteId)
          setDirty(false)
        }
      }
      return true
    } catch (error) {
      const text = error instanceof Error ? error.message : ''
      if (text.includes('409')) {
        let latest: ResearchNoteRecord | null = null
        try {
          latest = await chatApi.getResearchNote(noteId)
        } catch {
          // The local draft is already durable even when the latest record cannot be loaded.
        }
        if (activeIdRef.current === noteId) {
          if (latest) setActiveNote(latest)
          setConflictRecord(latest)
          setSaveIssue('conflict')
          setDirty(true)
          message.error(S.research_notes_conflict)
        }
      } else if (activeIdRef.current === noteId) {
        setSaveIssue('failed')
        setDirty(true)
        message.error(text || S.research_notes_save_failed)
      }
      return false
    } finally {
      setSaving(false)
    }
  }, [
    S.research_notes_conflict,
    S.research_notes_new_default_body,
    S.research_notes_new_default_title,
    S.research_notes_save_failed,
    activeNote,
    dirty,
    draftBaseRevision,
    draftContent,
    draftProjectId,
    draftSourceState,
    draftTags,
    draftTitle,
    saving,
    saveIssue,
    upsertSummary,
  ])

  useEffect(() => {
    if (!activeNote || !dirty || saving || saveIssue || evidenceBaselineRun.running) return
    const timer = window.setTimeout(() => { void saveDraftNow() }, 850)
    return () => window.clearTimeout(timer)
  }, [activeNote, dirty, evidenceBaselineRun.running, saveDraftNow, saveIssue, saving])

  const sourceOptions = useMemo(() => Array.from(new Set(notes.flatMap(noteSourceNames))).sort(), [notes])
  const tagOptions = useMemo(() => Array.from(new Set(notes.flatMap(note => note.tags || []))).sort(), [notes])
  const projectNameById = useMemo(() => new Map(projects.map(project => [project.id, project.name])), [projects])
  const noteEvidenceReviewSummaries = useMemo(() => new Map<string, ResearchNoteEvidenceReviewSummary>(
    notes.map(note => [note.id, researchNoteEvidenceReviewSummary(note.source_state)]),
  ), [notes])
  const evidenceReviewNoteCount = useMemo(() => Array.from(noteEvidenceReviewSummaries.values())
    .filter(summary => summary.total > 0).length, [noteEvidenceReviewSummaries])

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
      if (directoryEvidenceFilter === 'issues' && !(noteEvidenceReviewSummaries.get(note.id)?.total)) return false
      return true
    })
  }, [directoryEvidenceFilter, noteEvidenceReviewSummaries, notes, projectFilter, sourceFilter, tagFilter, timeFilter])

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
        expected_revision: draftBaseRevision,
        ...patch,
      })
      setActiveNote(record)
      setDraftBaseRevision(record.revision)
      setDraftProjectId(record.project_id || null)
      setDraftTags(record.tags || [])
      setDraftSourceState(record.source_state || {})
      upsertSummary(record)
      if (patch.archived !== undefined) {
        noteRequestRef.current += 1
        activeIdRef.current = ''
        setActiveId('')
        setActiveNote(null)
        await loadNotes()
      }
    } catch (error) {
      const text = error instanceof Error ? error.message : ''
      if (text.includes('409')) {
        message.error(S.research_notes_conflict)
        try {
          applyRecord(await chatApi.getResearchNote(activeNote.id))
        } catch {
          message.error(S.research_notes_load_failed)
        }
      } else {
        message.error(text || S.research_notes_save_failed)
      }
    } finally {
      setSaving(false)
    }
  }

  const deleteActive = async () => {
    if (!activeNote) return
    try {
      await chatApi.deleteResearchNote(activeNote.id)
      clearResearchNoteWorkspaceDraft(activeNote.id)
      setNotes(current => current.filter(item => item.id !== activeNote.id))
      setSelectedIds(current => {
        const next = new Set(current)
        next.delete(activeNote.id)
        return next
      })
      noteRequestRef.current += 1
      activeIdRef.current = ''
      setActiveId('')
      setActiveNote(null)
      message.success(S.research_notes_deleted)
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_notes_delete_failed)
    }
  }

  const saveConflictCopy = async () => {
    if (!activeNote || saveIssue !== 'conflict') return
    const originalId = activeNote.id
    const suffix = S.research_notes_conflict_copy_suffix.trim()
    const baseTitle = draftTitle.trim() || activeNote.title
    const copyTitle = `${baseTitle.slice(0, Math.max(1, 239 - suffix.length)).trim()} ${suffix}`.trim()
    setSaving(true)
    try {
      const record = await chatApi.createResearchNote({
        title: copyTitle,
        content_markdown: draftContent.trim() || S.research_notes_new_default_body,
        project_id: draftProjectId,
        source_conv_id: activeNote.source_conv_id || null,
        source_state: draftSourceState,
        tags: draftTags,
      })
      clearResearchNoteWorkspaceDraft(originalId)
      setArchiveView('active')
      setNotes(current => [
        { ...record, content_markdown: '' },
        ...current.filter(item => item.id !== record.id && !item.archived),
      ])
      applyRecord(record)
      setEditorMode('edit')
      message.success(S.research_notes_conflict_copy_saved)
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_notes_save_failed)
    } finally {
      setSaving(false)
    }
  }

  const loadLatestConflictRecord = async () => {
    if (!activeNote || saveIssue !== 'conflict') return
    const noteId = activeNote.id
    setLoadingNote(true)
    try {
      const record = conflictRecord?.id === noteId
        ? conflictRecord
        : await chatApi.getResearchNote(noteId)
      clearResearchNoteWorkspaceDraft(noteId)
      applyRecord(record)
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_notes_load_failed)
    } finally {
      setLoadingNote(false)
    }
  }

  const openAnswer = (link: ResearchNoteSourceLink) => {
    const conversationId = String(link.conversation_id || '').trim()
    const messageId = Number(link.message_id || 0)
    if (!conversationId || !messageId) return
    const params = new URLSearchParams({ conversation: conversationId, note_message: String(messageId) })
    navigate({ pathname: '/', search: `?${params.toString()}` })
  }

  const openPaper = async (
    link: ResearchNoteSourceLink,
    verifiedPayload: MessageProvenanceReaderOpen,
    state?: Record<string, unknown>,
  ) => {
    const sourcePath = String(verifiedPayload.sourcePath || '').trim()
    if (!sourcePath) return
    try {
      const session = await chatApi.createReaderSession(verifiedPayload, {
        title: link.source_name || link.label,
        conversationId: link.conversation_id,
        messageId: link.message_id,
        state,
      })
      navigate(`/reader/session/${encodeURIComponent(session.id)}`)
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_notes_load_failed)
    }
  }

  const openRepairCandidate = async (
    link: ResearchNoteSourceLink,
    sourceIndex: number,
    candidate: ResearchNoteEvidenceBlock,
    verifiedPayload: MessageProvenanceReaderOpen,
    context: ResearchNoteEvidenceContext,
  ) => {
    if (!activeNote || !link.source_path || !context.source_fingerprint) return
    if (saving) {
      message.info(S.research_notes_evidence_wait_for_save)
      return
    }
    if (saveIssue === 'conflict') {
      message.error(S.research_notes_evidence_conflict_blocked)
      return
    }
    if (dirty && !(await saveDraftNow())) return
    const intent: ResearchNoteEvidenceRepairIntent = {
      version: 1,
      note_id: activeNote.id,
      note_title: draftTitle.trim() || activeNote.title,
      source_label: sourceLabel(link),
      repair: {
        source_index: sourceIndex,
        source_path: link.source_path,
        capture_id: link.capture_id,
        old_block_id: link.block_id,
        old_anchor_id: link.anchor_id,
        old_evidence_quote: link.evidence_quote,
        candidate_block_id: candidate.block_id,
        candidate_anchor_id: candidate.anchor_id,
        candidate_anchor_kind: candidate.kind,
        candidate_evidence_quote: candidate.text,
        source_fingerprint: context.source_fingerprint,
      },
    }
    await openPaper(link, verifiedPayload, { researchNoteEvidenceRepair: intent })
  }

  const decideCandidateEvidence = async (
    link: ResearchNoteSourceLink,
    sourceIndex: number,
    candidate: ResearchNoteEvidenceBlock,
    context: ResearchNoteEvidenceContext,
    reviewKey: string,
    decision: 'confirmed' | 'invalid',
  ) => {
    if (!activeNote || !link.source_path || !context.source_fingerprint || candidateDecisionKey) return
    if (saving) {
      message.info(S.research_notes_evidence_wait_for_save)
      return
    }
    if (saveIssue === 'conflict') {
      message.error(S.research_notes_evidence_conflict_blocked)
      return
    }
    if (dirty && !(await saveDraftNow())) return

    const actionKey = `${reviewKey}:${candidate.block_id || candidate.anchor_id || 'candidate'}:${decision}`
    const noteId = activeNote.id
    const savedFingerprint = String(
      link.evidence_verification?.source_fingerprint
      || link.evidence_context_snapshot?.source_fingerprint
      || context.source_fingerprint,
    ).trim()
    setCandidateDecisionKey(actionKey)
    setSaving(true)
    try {
      const identity = {
        source_index: sourceIndex,
        source_path: link.source_path,
        capture_id: link.capture_id,
        old_block_id: link.block_id,
        old_anchor_id: link.anchor_id,
        old_evidence_quote: link.evidence_quote,
      }
      const record = decision === 'confirmed'
        ? await chatApi.repairResearchNoteEvidence(noteId, {
          ...identity,
          candidate_block_id: candidate.block_id,
          candidate_anchor_id: candidate.anchor_id,
          candidate_anchor_kind: candidate.kind,
          candidate_evidence_quote: candidate.text,
          source_fingerprint: context.source_fingerprint,
        })
        : await chatApi.decideResearchNoteEvidenceReview(noteId, {
          ...identity,
          decision: 'invalid',
          expected_source_fingerprint: savedFingerprint,
          current_block_id: candidate.block_id,
          current_anchor_id: candidate.anchor_id,
          current_anchor_kind: candidate.kind,
          current_evidence_quote: candidate.text,
          source_fingerprint: context.source_fingerprint,
        })
      setEvidenceReviews(current => {
        const next = { ...current }
        delete next[reviewKey]
        return next
      })
      upsertSummary(record)
      applyRecord(record)
      message.success(
        decision === 'confirmed'
          ? S.research_notes_evidence_repair_saved
          : S.research_notes_evidence_review_invalidated,
      )
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_notes_evidence_decision_failed)
    } finally {
      setSaving(false)
      setCandidateDecisionKey('')
    }
  }

  const openSourceChangeReview = async (
    link: ResearchNoteSourceLink,
    sourceIndex: number,
    target: ResearchNoteEvidenceBlock,
    verifiedPayload: MessageProvenanceReaderOpen,
    context: ResearchNoteEvidenceContext,
  ) => {
    const savedFingerprint = String(link.evidence_verification?.source_fingerprint || '').trim()
    if (
      !activeNote
      || !link.source_path
      || !savedFingerprint
      || !context.source_fingerprint
      || !target.text
      || !evidenceTargetMatchesSavedLocator(link, target)
    ) return
    if (saving) {
      message.info(S.research_notes_evidence_wait_for_save)
      return
    }
    if (saveIssue === 'conflict') {
      message.error(S.research_notes_evidence_conflict_blocked)
      return
    }
    if (dirty && !(await saveDraftNow())) return
    const intent: ResearchNoteEvidenceReviewIntent = {
      version: 1,
      note_id: activeNote.id,
      note_title: draftTitle.trim() || activeNote.title,
      source_label: sourceLabel(link),
      saved_evidence_quote: String(link.evidence_quote || '').trim(),
      current_evidence_quote: target.text,
      previously_invalid: link.evidence_verification?.status === 'invalid',
      saved_context_snapshot: link.evidence_context_snapshot || null,
      current_context_snapshot: context.context_snapshot || null,
      review: {
        source_index: sourceIndex,
        source_path: link.source_path,
        capture_id: link.capture_id,
        old_block_id: link.block_id,
        old_anchor_id: link.anchor_id,
        old_evidence_quote: link.evidence_quote,
        expected_source_fingerprint: savedFingerprint,
        current_block_id: target.block_id,
        current_anchor_id: target.anchor_id,
        current_anchor_kind: target.kind,
        current_evidence_quote: target.text,
        source_fingerprint: context.source_fingerprint,
      },
    }
    await openPaper(link, verifiedPayload, { researchNoteEvidenceReview: intent })
  }

  const reviewEvidence = async (
    link: ResearchNoteSourceLink,
    reviewKey: string,
    openIfExact = false,
  ) => {
    const cached = evidenceReviews[reviewKey]?.data
    if (openIfExact && cached?.status === 'exact' && cached.reader_open) {
      await openPaper(link, cached.reader_open)
      return
    }
    setEvidenceReviews(current => ({
      ...current,
      [reviewKey]: { ...current[reviewKey], loading: true, error: undefined },
    }))
    try {
      const data = await chatApi.reviewResearchNoteEvidence(link)
      setEvidenceReviews(current => ({ ...current, [reviewKey]: { loading: false, data } }))
      if (openIfExact && data.status === 'exact' && data.reader_open) {
        await openPaper(link, data.reader_open)
      }
    } catch (error) {
      const detail = error instanceof Error ? error.message : S.research_notes_evidence_load_failed
      setEvidenceReviews(current => ({
        ...current,
        [reviewKey]: { ...current[reviewKey], loading: false, error: detail },
      }))
    }
  }

  const runEvidenceHealthCheck = async (options: { silent?: boolean } = {}) => {
    if (!activeNote || evidenceHealthRun.running || evidenceBaselineRun.running) return
    const noteId = activeNote.id
    const initialTotal = links.filter(link => link.kind !== 'answer').length
    if (initialTotal <= 0) return
    const runId = ++evidenceHealthRunRef.current
    setEvidenceHealthFilter('all')
    setEvidenceHealthRun({
      started: true,
      running: true,
      total: initialTotal,
      completed: 0,
    })
    try {
      if (dirty) {
        if (options.silent || !(await saveDraftNow())) return
      }
      const latest = await chatApi.getResearchNote(noteId)
      if (runId !== evidenceHealthRunRef.current || activeIdRef.current !== noteId) return
      const result = await chatApi.refreshResearchNoteEvidenceHealth(noteId, latest.revision)
      if (runId !== evidenceHealthRunRef.current || activeIdRef.current !== noteId) return
      upsertSummary(result.note)
      applyRecord(result.note, { focusFirstEvidenceIssue: true })
      setEvidenceBaselineRun({
        running: false,
        summary: result.summary,
        items: result.items,
        sourceSignature: evidenceBaselineSourceSignature(result.note.source_state),
      })

      const refreshedEntries = noteLinks(result.note)
        .map((link, index) => ({ link, index, reviewKey: sourceReviewKey(link, index) }))
        .filter(entry => entry.link.kind !== 'answer')
      const decisionIndices = new Set(
        researchNoteEvidenceReviewIssues(result.note.source_state).map(issue => issue.sourceIndex),
      )
      const decisionEntries = refreshedEntries.filter(entry => decisionIndices.has(entry.index))
      const contextEntries = refreshedEntries.filter(entry => (
        decisionIndices.has(entry.index)
        || entry.link.evidence_verification?.status === 'invalid'
      ))
      const initialReviews: Record<string, EvidenceReviewState> = {}
      contextEntries.forEach(entry => { initialReviews[entry.reviewKey] = { loading: true } })
      setEvidenceReviews(initialReviews)
      await Promise.all(contextEntries.map(async entry => {
        let nextReview: EvidenceReviewState
        try {
          const data = await chatApi.reviewResearchNoteEvidence(entry.link)
          nextReview = { loading: false, data }
        } catch (error) {
          nextReview = {
            loading: false,
            error: error instanceof Error ? error.message : S.research_notes_evidence_load_failed,
          }
        }
        if (runId !== evidenceHealthRunRef.current || activeIdRef.current !== noteId) return
        setEvidenceReviews(current => ({ ...current, [entry.reviewKey]: nextReview }))
      }))
      if (runId !== evidenceHealthRunRef.current || activeIdRef.current !== noteId) return
      setEvidenceHealthRun({
        started: true,
        running: false,
        total: refreshedEntries.length,
        completed: refreshedEntries.length,
      })
      setEvidenceHealthFilter(decisionEntries.length > 0 ? 'issues' : 'all')
    } catch (error) {
      const detail = error instanceof Error ? error.message : ''
      if (detail.includes('409')) {
        try {
          const latest = await chatApi.getResearchNote(noteId)
          if (runId === evidenceHealthRunRef.current && activeIdRef.current === noteId) {
            upsertSummary(latest)
            applyRecord(latest, { focusFirstEvidenceIssue: true })
            const total = noteLinks(latest).filter(link => link.kind !== 'answer').length
            setEvidenceHealthRun({ started: true, running: false, total, completed: total })
            setEvidenceHealthFilter(researchNoteEvidenceReviewIssues(latest.source_state).length > 0 ? 'issues' : 'all')
          }
        } catch {
          // The next bounded maintenance pass will retry this transient conflict.
        }
      } else if (!options.silent) {
        message.info(S.research_notes_evidence_context_recovering)
      }
    } finally {
      if (runId === evidenceHealthRunRef.current && activeIdRef.current === noteId) {
        setEvidenceHealthRun(current => ({ ...current, running: false }))
      }
    }
  }

  const captureMissingEvidenceBaselines = async () => {
    if (!activeNote || saving || evidenceHealthRun.running || evidenceBaselineRun.running) return
    if (saveIssue === 'conflict') {
      message.error(S.research_notes_evidence_baseline_conflict)
      return
    }
    const noteId = activeNote.id
    const runId = ++evidenceBaselineRunRef.current
    const operationDraftVersion = draftVersionRef.current
    setEvidenceBaselineRun({ running: true })
    try {
      if (dirty && !(await saveDraftNow())) return
      const latest = await chatApi.getResearchNote(noteId)
      if (runId !== evidenceBaselineRunRef.current || activeIdRef.current !== noteId) return
      const result = await chatApi.captureResearchNoteEvidenceContextBaselines(
        noteId,
        latest.revision,
      )
      if (runId !== evidenceBaselineRunRef.current || activeIdRef.current !== noteId) return
      upsertSummary(result.note)
      if (draftVersionRef.current === operationDraftVersion) {
        applyRecord(result.note)
      } else {
        setActiveNote(result.note)
        setDraftBaseRevision(result.note.revision)
        setDraftSourceState(current => mergeEvidenceBaselineState(
          current,
          result.note.source_state || {},
        ))
        setDirty(true)
      }
      setEvidenceBaselineRun({
        running: false,
        summary: result.summary,
        items: result.items,
        sourceSignature: evidenceBaselineSourceSignature(result.note.source_state),
      })
      const decisionCount = result.items.filter(item => (
        item.status === 'skipped'
        && ['candidate', 'source_changed', 'location_changed'].includes(item.reason)
      )).length
      const feedback = S.research_notes_evidence_baseline_result
        .replace('{created}', String(result.summary.created))
        .replace('{review}', String(decisionCount))
      if (result.summary.failed > 0) {
        message.info(`${feedback} ${S.research_notes_evidence_baseline_background_retry}`)
      } else {
        message.success(feedback)
      }
    } catch (error) {
      const detail = error instanceof Error ? error.message : ''
      if (detail.includes('409')) {
        message.error(S.research_notes_evidence_baseline_conflict)
      } else {
        message.error(detail || S.research_notes_evidence_baseline_failed)
      }
    } finally {
      if (runId === evidenceBaselineRunRef.current && activeIdRef.current === noteId) {
        setEvidenceBaselineRun(current => ({ ...current, running: false }))
      }
    }
  }

  const openCombine = async () => {
    const ids = Array.from(selectedIds)
    if (ids.length <= 0) return
    setCombining(true)
    try {
      const records = await Promise.all(ids.map(id => chatApi.getResearchNote(id)))
      const title = S.research_notes_combine_default_title
      setCombineRecords(records)
      setCombineProjectId(researchNoteOutlineProjectId(records))
      setCombineTitle(title)
      setCombineContent(buildResearchNoteOutline(title, records))
      setCombinePreview(false)
      setCombineOpen(true)
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_notes_load_failed)
    } finally {
      setCombining(false)
    }
  }

  const updateCombineTitle = (nextTitle: string) => {
    const previousHeading = `# ${combineTitle.trim()}`
    setCombineContent(current => {
      const lines = current.split('\n')
      if (String(lines[0] || '').trim() !== previousHeading) return current
      lines[0] = `# ${nextTitle.trim()}`
      return lines.join('\n')
    })
    setCombineTitle(nextTitle)
  }

  const saveCombinedOutline = async () => {
    const title = combineTitle.trim()
    const content = combineContent.trim()
    if (!title || !content || combineRecords.length <= 0) return
    setCombineSaving(true)
    try {
      const record = await chatApi.createResearchNote({
        title,
        content_markdown: content,
        project_id: combineProjectId,
        source_conv_id: researchNoteOutlineConversationId(combineRecords),
        source_state: researchNoteOutlineSourceState(combineRecords),
        tags: researchNoteOutlineTags(combineRecords),
      })
      setArchiveView('active')
      setQuery('')
      setProjectFilter(ALL_FILTER)
      setSourceFilter(ALL_FILTER)
      setTagFilter(ALL_FILTER)
      setTimeFilter('all')
      setSelectedIds(new Set())
      setNotes(current => [
        { ...record, content_markdown: '' },
        ...current.filter(item => item.id !== record.id && !item.archived),
      ])
      setCombineOpen(false)
      applyRecord(record)
      setEditorMode('edit')
      message.success(S.research_notes_outline_saved)
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_notes_save_failed)
    } finally {
      setCombineSaving(false)
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

  const linkEntries: EvidenceLinkEntry[] = links.map((link, index) => ({
    link,
    index,
    reviewKey: sourceReviewKey(link, index),
  }))
  const paperLinkEntries = linkEntries.filter(entry => entry.link.kind !== 'answer')
  const missingEvidenceBaselineEntries = paperLinkEntries.filter(entry => (
    Boolean(String(entry.link.source_path || '').trim())
    && entry.link.evidence_context_snapshot == null
    && entry.link.evidence_verification?.status !== 'invalid'
    && entry.link.evidence_baseline_check?.status !== 'failed'
    && entry.link.evidence_baseline_check?.reason !== 'unavailable'
  ))
  const missingEvidenceBaselineCount = missingEvidenceBaselineEntries.length
  const baselineResultIsCurrent = Boolean(
    evidenceBaselineRun.sourceSignature
    && evidenceBaselineRun.sourceSignature === evidenceBaselineSourceSignature(draftSourceState),
  )
  const currentEvidenceBaselineSummary = baselineResultIsCurrent ? evidenceBaselineRun.summary : undefined
  const persistedEvidenceBaselineIssues: ResearchNoteEvidenceBaselineItem[] = researchNoteEvidenceReviewIssues(
    draftSourceState,
  ).map(issue => ({
    source_index: issue.sourceIndex,
    source_name: sourceLabel(issue.link),
    status: issue.status,
    reason: issue.reason,
    checked_at: issue.checkedAt,
  }))
  const evidenceBaselineIssues = baselineResultIsCurrent
    ? (evidenceBaselineRun.items || []).filter(item => (
      item.status === 'skipped'
      && ['candidate', 'source_changed', 'location_changed'].includes(item.reason)
    ))
    : persistedEvidenceBaselineIssues
  const evidenceBaselineIssueCount = evidenceBaselineIssues.length
  const hasUnclassifiedEvidenceBaselines = missingEvidenceBaselineCount > evidenceBaselineIssueCount
  const locateEvidenceBaselineIssue = (item: ResearchNoteEvidenceBaselineItem) => {
    const entry = linkEntries.find(candidate => candidate.index === item.source_index)
    if (!entry) return
    setEvidenceHealthFilter('all')
    setFocusedEvidenceIndex(item.source_index)
    window.requestAnimationFrame(() => {
      const sourceCard = document.querySelector<HTMLElement>(
        `[data-testid="research-note-source-${item.source_index}"]`,
      )
      sourceCard?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
      sourceCard?.focus({ preventScroll: true })
    })
    if (entry.link.source_path) void reviewEvidence(entry.link, entry.reviewKey)
  }
  const evidenceHealthCounts = paperLinkEntries.reduce((counts, entry) => {
    const status = evidenceHealthStatus(
      entry,
      evidenceReviews[entry.reviewKey],
      evidenceHealthRun.started,
    )
    counts[status] += 1
    return counts
  }, {
    exact: 0,
    changed: 0,
    candidate: 0,
    invalid: 0,
    recovering: 0,
    unchecked: 0,
  } as Record<EvidenceHealthStatus, number>)
  const evidenceIssueCount = evidenceHealthCounts.changed
    + evidenceHealthCounts.candidate
  const displayedLinkEntries = evidenceHealthFilter === 'issues'
    && evidenceHealthRun.started
    && !evidenceHealthRun.running
    ? linkEntries.filter(entry => (
      entry.link.kind !== 'answer'
      && ['changed', 'candidate'].includes(
        evidenceHealthStatus(entry, evidenceReviews[entry.reviewKey], true),
      )
    ))
    : linkEntries
  const combineHasMixedProjects = researchNoteOutlineHasMixedProjects(combineRecords)
  const freshnessMatchesOutline = outlineFreshness
    .map(item => `${item.source.id}:${item.source.revision}`)
    .join('|') === outlineSourceSignature
  const displayedOutlineFreshness: OutlineSourceFreshness[] = freshnessMatchesOutline && outlineFreshness.length > 0
    ? outlineFreshness
    : outlineSources.map(source => ({ source, status: 'checking' }))
  const outlineIsChecking = outlineChecking
    || displayedOutlineFreshness.some(item => item.status === 'checking')
  const outlineUpdatedCount = displayedOutlineFreshness.filter(item => item.status === 'updated').length
  const outlineMissingCount = displayedOutlineFreshness.filter(item => item.status === 'missing').length
  const outlineCheckFailed = displayedOutlineFreshness.some(item => item.status === 'error')
  const statusText = saving
    ? S.research_notes_saving
    : saveIssue === 'conflict'
      ? S.research_notes_conflict_status
    : saveIssue === 'failed'
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
          <Button
            icon={<ReloadOutlined />}
            onClick={() => {
              void loadNotes()
            }}
            aria-label={S.reload}
          />
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
              noteRequestRef.current += 1
              activeIdRef.current = ''
              setArchiveView(value as 'active' | 'archived')
              setActiveId('')
              setActiveNote(null)
              setDirty(false)
              setSaveIssue(null)
              setConflictRecord(null)
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
          <Segmented
            block
            size="small"
            className="kb-notes-evidence-filter"
            data-testid="research-notes-evidence-filter"
            aria-label={S.research_notes_filter_evidence_review}
            value={directoryEvidenceFilter}
            options={[
              { value: 'all', label: S.research_notes_evidence_filter_all },
              {
                value: 'issues',
                label: S.research_notes_evidence_filter_issues.replace('{n}', String(evidenceReviewNoteCount)),
              },
            ]}
            onChange={value => setDirectoryEvidenceFilter(value as DirectoryEvidenceFilter)}
          />
          <div className="kb-notes-directory-list kb-main-scroll">
            {loading ? (
              <div className="kb-notes-centered"><Spin size="small" /> {S.research_notes_loading_workspace}</div>
            ) : filteredNotes.length <= 0 ? (
              <Empty
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                description={directoryEvidenceFilter === 'issues'
                  ? S.research_notes_evidence_filter_empty
                  : S.research_notes_empty_workspace}
              />
            ) : filteredNotes.map(note => {
              const selected = selectedIds.has(note.id)
              const sourceCount = noteLinks(note).length
              const reviewSummary = noteEvidenceReviewSummaries.get(note.id)
              return (
                <article
                  key={note.id}
                  className={`kb-notes-directory-card ${activeId === note.id ? 'is-active' : ''}${reviewSummary?.total ? ' has-evidence-review' : ''}${reviewSummary?.failed ? ' has-evidence-failure' : ''}`}
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
                    {reviewSummary?.total ? (
                      <div
                        className="kb-notes-card-evidence-review"
                        data-testid={`workspace-note-evidence-review-${note.id}`}
                      >
                        {reviewSummary.needsReview > 0 ? (
                          <Tag color="gold" data-testid={`workspace-note-evidence-review-count-${note.id}`}>
                            {S.research_notes_evidence_card_review.replace('{n}', String(reviewSummary.needsReview))}
                          </Tag>
                        ) : null}
                      </div>
                    ) : null}
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
                <span
                  className={`kb-notes-save-state ${dirty ? 'is-dirty' : ''} ${saveIssue ? 'is-error' : ''}`}
                  data-testid="research-note-save-state"
                  role="status"
                  aria-live="polite"
                >
                  {statusText}
                </span>
                {dirty && saveIssue !== 'conflict' ? (
                  <Button
                    size="small"
                    icon={<SaveOutlined />}
                    loading={saving}
                    data-testid="research-note-save-now"
                    onClick={() => { void saveDraftNow() }}
                  >
                    {saveIssue === 'failed' ? S.research_notes_retry_save : S.research_notes_save_now}
                  </Button>
                ) : null}
                {saveIssue === 'conflict' ? (
                  <div className="kb-notes-conflict-actions">
                    <Button
                      size="small"
                      type="primary"
                      icon={<CopyOutlined />}
                      loading={saving}
                      data-testid="research-note-save-copy"
                      onClick={() => { void saveConflictCopy() }}
                    >
                      {S.research_notes_save_conflict_copy}
                    </Button>
                    <Popconfirm
                      title={S.research_notes_load_latest_confirm}
                      okText={S.research_notes_load_latest}
                      cancelText={S.confirm_cancel}
                      onConfirm={loadLatestConflictRecord}
                    >
                      <Button size="small" data-testid="research-note-load-latest">
                        {S.research_notes_load_latest}
                      </Button>
                    </Popconfirm>
                  </div>
                ) : null}
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
                    <Button danger disabled={dirty || saving} icon={<DeleteOutlined />} aria-label={S.research_notes_delete} />
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
            <div className="kb-notes-sources-head-row">
              <div>
                <h2>{S.research_notes_sources_title}</h2>
                <p>{S.research_notes_sources_hint}</p>
              </div>
              <Button
                size="small"
                icon={<SearchOutlined />}
                loading={evidenceHealthRun.running}
                disabled={!activeNote || paperLinkEntries.length <= 0 || evidenceBaselineRun.running}
                data-testid="research-note-evidence-check-all"
                onClick={() => { void runEvidenceHealthCheck() }}
              >
                {evidenceHealthRun.started
                  ? S.research_notes_evidence_health_recheck
                  : S.research_notes_evidence_health_check}
              </Button>
            </div>
            {paperLinkEntries.length > 0 && (
              missingEvidenceBaselineCount > 0
              || evidenceBaselineIssueCount > 0
              || currentEvidenceBaselineSummary
            ) ? (
              <section
                className="kb-notes-evidence-baseline"
                data-testid="research-note-evidence-baseline"
                role="status"
                aria-live="polite"
              >
                <div className="kb-notes-evidence-baseline-copy">
                  <div>
                    <HistoryOutlined />
                    <strong>
                      {evidenceBaselineIssueCount > 0
                        ? S.research_notes_evidence_baseline_needs_review.replace('{n}', String(evidenceBaselineIssueCount))
                        : missingEvidenceBaselineCount > 0
                          ? S.research_notes_evidence_baseline_missing.replace('{n}', String(missingEvidenceBaselineCount))
                        : S.research_notes_evidence_baseline_complete}
                    </strong>
                  </div>
                  <p>
                    {evidenceBaselineIssueCount > 0
                      ? S.research_notes_evidence_baseline_review_hint
                      : missingEvidenceBaselineCount > 0
                        ? S.research_notes_evidence_baseline_hint
                      : S.research_notes_evidence_baseline_complete_hint}
                  </p>
                </div>
                {missingEvidenceBaselineCount > 0 && hasUnclassifiedEvidenceBaselines ? (
                  <Popconfirm
                    title={S.research_notes_evidence_baseline_confirm}
                    description={S.research_notes_evidence_baseline_confirm_hint}
                    okText={S.research_notes_evidence_baseline_action}
                    cancelText={S.confirm_cancel}
                    onConfirm={captureMissingEvidenceBaselines}
                  >
                    <Button
                      size="small"
                      loading={evidenceBaselineRun.running}
                      disabled={saving || evidenceHealthRun.running || saveIssue === 'conflict'}
                      data-testid="research-note-evidence-baseline-create"
                    >
                      {S.research_notes_evidence_baseline_action}
                    </Button>
                  </Popconfirm>
                ) : null}
                {currentEvidenceBaselineSummary ? (
                  <div
                    className="kb-notes-evidence-baseline-result"
                    data-testid="research-note-evidence-baseline-result"
                  >
                    <Tag color="green">
                      {S.research_notes_evidence_baseline_direct_created.replace(
                        '{n}',
                        String(currentEvidenceBaselineSummary.created),
                      )}
                    </Tag>
                    <Tag color={evidenceBaselineIssueCount > 0 ? 'gold' : 'default'}>
                      {S.research_notes_evidence_baseline_needs_review_count.replace(
                        '{n}',
                        String(evidenceBaselineIssueCount),
                      )}
                    </Tag>
                  </div>
                ) : null}
                {evidenceBaselineIssues.length > 0 ? (
                  <div
                    className="kb-notes-evidence-baseline-issues"
                    data-testid="research-note-evidence-baseline-issues"
                  >
                    {evidenceBaselineIssues.map(item => {
                      const entry = linkEntries.find(candidate => candidate.index === item.source_index)
                      const canReview = Boolean(entry?.link.source_path)
                      return (
                        <div
                          key={`${item.source_index}:${item.reason}`}
                          className={`kb-notes-evidence-baseline-issue${item.status === 'failed' ? ' is-failed' : ''}`}
                          data-testid={`research-note-evidence-baseline-issue-${item.source_index}`}
                        >
                          <strong>{item.source_name || (entry ? sourceLabel(entry.link) : '') || S.research_note_untitled_source}</strong>
                          <span>{evidenceBaselineReasonLabel(S, item.reason)}</span>
                          {item.checked_at ? (
                            <time dateTime={new Date(item.checked_at * 1000).toISOString()}>
                              {S.research_notes_evidence_baseline_checked_at.replace(
                                '{time}',
                                formatUpdated(item.checked_at),
                              )}
                            </time>
                          ) : null}
                          <Button
                            type="link"
                            size="small"
                            icon={canReview ? <SearchOutlined /> : <ArrowRightOutlined />}
                            loading={Boolean(entry && evidenceReviews[entry.reviewKey]?.loading)}
                            disabled={evidenceHealthRun.running}
                            onClick={() => locateEvidenceBaselineIssue(item)}
                          >
                            {canReview
                              ? S.research_notes_evidence_baseline_locate_and_review
                              : S.research_notes_evidence_baseline_locate}
                          </Button>
                        </div>
                      )
                    })}
                  </div>
                ) : null}
              </section>
            ) : null}
            {evidenceHealthRun.started ? (
              <section
                className="kb-notes-evidence-health"
                data-testid="research-note-evidence-health"
                role="status"
                aria-live="polite"
              >
                <div className="kb-notes-evidence-health-progress">
                  {evidenceHealthRun.running ? <Spin size="small" /> : null}
                  <span>
                    {S.research_notes_evidence_health_progress
                      .replace('{done}', String(evidenceHealthRun.completed))
                      .replace('{total}', String(evidenceHealthRun.total))}
                  </span>
                </div>
                <div className="kb-notes-evidence-health-counts">
                  <Tag color="green">
                    {S.research_notes_evidence_health_exact.replace('{n}', String(evidenceHealthCounts.exact))}
                  </Tag>
                  <Tag color="gold">
                    {S.research_notes_evidence_health_changed.replace('{n}', String(evidenceHealthCounts.changed))}
                  </Tag>
                  <Tag color="orange">
                    {S.research_notes_evidence_health_candidate.replace('{n}', String(evidenceHealthCounts.candidate))}
                  </Tag>
                  <Tag color="magenta">
                    {S.research_notes_evidence_health_invalid.replace('{n}', String(evidenceHealthCounts.invalid))}
                  </Tag>
                  <Tag>
                    {S.research_notes_evidence_health_recovering.replace('{n}', String(evidenceHealthCounts.recovering))}
                  </Tag>
                </div>
                <Segmented
                  block
                  size="small"
                  disabled={evidenceHealthRun.running}
                  value={evidenceHealthFilter}
                  options={[
                    {
                      value: 'all',
                      label: S.research_notes_evidence_health_all.replace('{n}', String(links.length)),
                    },
                    {
                      value: 'issues',
                      label: S.research_notes_evidence_health_issues.replace('{n}', String(evidenceIssueCount)),
                    },
                  ]}
                  onChange={value => setEvidenceHealthFilter(value as EvidenceHealthFilter)}
                />
              </section>
            ) : null}
          </div>
          <div className="kb-notes-sources-list kb-main-scroll">
            {!activeNote || (links.length <= 0 && outlineSources.length <= 0) ? (
              <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={S.research_notes_no_sources} />
            ) : (
              <>
                {outlineSources.length > 0 ? (
                  <article className="kb-notes-composition-card" data-testid="research-note-outline-sources">
                    <div className="kb-notes-composition-head">
                      <div className="kb-notes-source-kind">
                        <FileTextOutlined />
                        <span>{S.research_notes_outline_sources.replace('{n}', String(outlineSources.length))}</span>
                      </div>
                      <Button
                        type="text"
                        size="small"
                        icon={<ReloadOutlined />}
                        loading={outlineIsChecking}
                        aria-label={S.research_notes_outline_check_updates}
                        onClick={() => setOutlineCheckVersion(current => current + 1)}
                      />
                    </div>
                    <div
                      className="kb-notes-outline-update-summary"
                      data-testid="research-note-outline-update-summary"
                      role="status"
                      aria-live="polite"
                    >
                      {outlineIsChecking ? S.research_notes_outline_checking : null}
                      {!outlineIsChecking && outlineUpdatedCount > 0 ? (
                        <Tag color="orange">
                          {S.research_notes_outline_updates_available.replace('{n}', String(outlineUpdatedCount))}
                        </Tag>
                      ) : null}
                      {!outlineIsChecking && outlineMissingCount > 0 ? (
                        <Tag color="red">
                          {S.research_notes_outline_missing_count.replace('{n}', String(outlineMissingCount))}
                        </Tag>
                      ) : null}
                      {!outlineIsChecking && !outlineCheckFailed && outlineUpdatedCount <= 0 && outlineMissingCount <= 0 ? (
                        <Tag color="green">{S.research_notes_outline_up_to_date}</Tag>
                      ) : null}
                    </div>
                    {outlineCheckFailed ? (
                      <Alert type="warning" showIcon title={S.research_notes_outline_check_failed} />
                    ) : null}
                    <div className="kb-notes-composition-links">
                      {displayedOutlineFreshness.map(item => (
                          <div
                            key={item.source.id}
                            className={`kb-notes-composition-source is-${item.status}`}
                            data-testid={`research-note-outline-status-${item.source.id}`}
                          >
                            <div>
                              <Button
                                type="link"
                                data-testid={`research-note-outline-source-${item.source.id}`}
                                disabled={item.status === 'missing' || item.status === 'error'}
                                onClick={() => { void openNote(item.source.id) }}
                              >
                                {item.latestTitle || item.source.title}
                              </Button>
                              <small>
                                {item.status === 'updated' && item.latestRevision
                                  ? S.research_notes_outline_revision_change
                                    .replace('{from}', String(item.source.revision))
                                    .replace('{to}', String(item.latestRevision))
                                  : S.research_notes_outline_revision.replace('{n}', String(item.source.revision))}
                                {item.source.last_refreshed_at
                                  ? ` · ${S.research_notes_outline_last_refreshed.replace('{time}', formatUpdated(item.source.last_refreshed_at))}`
                                  : ''}
                              </small>
                            </div>
                            <div className="kb-notes-composition-source-actions">
                              <Tag color={item.status === 'updated' ? 'orange' : item.status === 'current' ? 'green' : item.status === 'checking' ? 'default' : 'red'}>
                                {item.status === 'updated'
                                  ? S.research_notes_outline_source_updated
                                  : item.status === 'current'
                                    ? S.research_notes_outline_source_current
                                    : item.status === 'checking'
                                      ? S.research_notes_outline_checking_short
                                      : item.status === 'missing'
                                        ? S.research_notes_outline_source_missing
                                        : S.research_notes_outline_source_error}
                              </Tag>
                              {item.status === 'updated' ? (
                                <Button
                                  size="small"
                                  icon={<DiffOutlined />}
                                  loading={outlineReviewLoadingId === item.source.id}
                                  data-testid={`research-note-outline-review-${item.source.id}`}
                                  onClick={() => { void reviewOutlineSourceUpdate(item) }}
                                >
                                  {S.research_notes_outline_review_update}
                                </Button>
                              ) : null}
                            </div>
                          </div>
                        ))}
                    </div>
                  </article>
                ) : null}
                {displayedLinkEntries.length <= 0 && evidenceHealthFilter === 'issues' ? (
                  <Alert type="success" showIcon title={S.research_notes_evidence_health_no_issues} />
                ) : null}
                {displayedLinkEntries.map(({ link, index, reviewKey }) => {
                  const review = evidenceReviews[reviewKey]
                  const result = review?.data
                  const healthStatus = evidenceHealthStatus(
                    { link, index, reviewKey },
                    review,
                    evidenceHealthRun.started,
                  )
                  const healthColor = healthStatus === 'exact'
                    ? 'green'
                    : healthStatus === 'changed'
                      ? 'gold'
                      : healthStatus === 'candidate'
                        ? 'orange'
                        : healthStatus === 'invalid'
                          ? 'magenta'
                          : 'default'
                  const healthLabel = healthStatus === 'exact'
                    ? S.research_notes_evidence_health_status_exact
                    : healthStatus === 'changed'
                      ? S.research_notes_evidence_health_status_changed
                      : healthStatus === 'candidate'
                        ? S.research_notes_evidence_health_status_candidate
                        : healthStatus === 'invalid'
                          ? S.research_notes_evidence_health_status_invalid
                        : healthStatus === 'recovering'
                          ? S.research_notes_evidence_health_status_recovering
                            : ''
                  const targetMatchesSavedLocator = evidenceTargetMatchesSavedLocator(link, result?.target)
                  const requiresLocatorRepair = Boolean(
                    result?.status === 'exact'
                    && result.reader_open
                    && result.target
                    && (result.reason === 'quote_relocated' || !targetMatchesSavedLocator),
                  )
                  const canReviewSourceVersion = Boolean(
                    result?.status === 'exact'
                    && result.reader_open
                    && result.target
                    && targetMatchesSavedLocator
                    && result.reason !== 'quote_relocated'
                    && (result.source_changed || link.evidence_verification?.status === 'invalid')
                    && link.evidence_verification?.source_fingerprint,
                  )
                  return (
                    <article
                      key={`${link.kind}-${link.message_id || 0}-${link.source_path || ''}-${index}`}
                      className={`kb-notes-source-card${focusedEvidenceIndex === index ? ' is-baseline-focus' : ''}`}
                      data-testid={`research-note-source-${index}`}
                      tabIndex={-1}
                    >
                      <div className="kb-notes-source-kind">
                        {link.kind === 'answer' ? <FileTextOutlined /> : <LinkOutlined />}
                        <span>{link.kind === 'answer' ? S.research_notes_answer_source : S.research_notes_paper_source}</span>
                        {link.kind !== 'answer' && healthLabel ? (
                          <Tag
                            color={healthColor}
                            data-testid={`research-note-evidence-health-status-${index}`}
                          >
                            {healthLabel}
                          </Tag>
                        ) : null}
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
                      {link.kind === 'answer' ? (
                        <Button
                          type="link"
                          icon={<ArrowRightOutlined />}
                          disabled={!link.conversation_id || !link.message_id}
                          onClick={() => openAnswer(link)}
                        >
                          {S.research_notes_open_answer}
                        </Button>
                      ) : (
                        <>
                          <div className="kb-notes-evidence-actions">
                            {healthStatus !== 'recovering' ? (
                              <Button
                                type="link"
                                icon={<SearchOutlined />}
                                loading={Boolean(review?.loading)}
                                disabled={!link.source_path || evidenceHealthRun.running}
                                data-testid={`research-note-evidence-review-${index}`}
                                onClick={() => { void reviewEvidence(link, reviewKey) }}
                              >
                                {result ? S.research_notes_evidence_recheck : S.research_notes_evidence_review}
                              </Button>
                            ) : (
                              <Button
                                type="link"
                                icon={<ArrowRightOutlined />}
                                data-testid={`research-note-evidence-open-recovery-${index}`}
                                onClick={() => {
                                  const source = String(link.source_name || link.label || link.source_path || '').trim()
                                  navigate(`/library?source=${encodeURIComponent(source)}`)
                                }}
                              >
                                {S.research_notes_evidence_open_recovery}
                              </Button>
                            )}
                            {result?.status === 'exact' && result.reader_open && !requiresLocatorRepair && !canReviewSourceVersion ? (
                              <Button
                                type="link"
                                icon={<ArrowRightOutlined />}
                                data-testid={`research-note-evidence-open-${index}`}
                                onClick={() => { void reviewEvidence(link, reviewKey, true) }}
                              >
                                {S.research_notes_evidence_open_exact}
                              </Button>
                            ) : null}
                            {result?.status === 'exact' && result.reader_open && result.target && requiresLocatorRepair ? (
                              <Button
                                type="link"
                                icon={<ArrowRightOutlined />}
                                disabled={saving || saveIssue === 'conflict'}
                                data-testid={`research-note-evidence-repair-relocated-${index}`}
                                onClick={() => {
                                  void openRepairCandidate(link, index, result.target as ResearchNoteEvidenceBlock, result.reader_open as MessageProvenanceReaderOpen, result)
                                }}
                              >
                                {S.research_notes_evidence_review_and_repair}
                              </Button>
                            ) : null}
                            {result?.status === 'exact' && result.reader_open && result.target && canReviewSourceVersion ? (
                              <Button
                                type="link"
                                icon={<ArrowRightOutlined />}
                                disabled={saving || saveIssue === 'conflict'}
                                data-testid={`research-note-evidence-review-change-${index}`}
                                onClick={() => {
                                  void openSourceChangeReview(link, index, result.target as ResearchNoteEvidenceBlock, result.reader_open as MessageProvenanceReaderOpen, result)
                                }}
                              >
                                {link.evidence_verification?.status === 'invalid'
                                  ? S.research_notes_evidence_review_invalid
                                  : S.research_notes_evidence_review_change}
                              </Button>
                            ) : null}
                          </div>
                          {review?.loading && !result ? (
                            <div className="kb-notes-evidence-loading">
                              <Spin size="small" /> {S.research_notes_evidence_loading}
                            </div>
                          ) : null}
                          {review?.error ? (
                            <Alert
                              type="info"
                              showIcon
                              title={S.research_notes_evidence_context_recovering}
                            />
                          ) : null}
                          {result ? (
                            <section
                              className={`kb-notes-evidence-review kb-notes-evidence-review-${result.status}`}
                              data-testid={`research-note-evidence-result-${index}`}
                            >
                              <div className="kb-notes-evidence-status">
                                <Tag color={result.status === 'exact' ? 'green' : result.status === 'candidate' ? 'orange' : 'default'}>
                                  {result.status === 'exact'
                                    ? S.research_notes_evidence_exact
                                    : result.status === 'candidate'
                                      ? S.research_notes_evidence_candidate
                                      : S.research_notes_evidence_unavailable}
                                </Tag>
                                <span>{evidenceReasonLabel(S, result.reason)}</span>
                              </div>
                              {result.source_changed ? (
                                <Alert
                                  type="warning"
                                  showIcon
                                  title={S.research_notes_evidence_source_changed}
                                />
                              ) : null}
                              {link.evidence_verification?.status === 'invalid' ? (
                                <Alert
                                  type="error"
                                  showIcon
                                  title={S.research_notes_evidence_invalid_notice}
                                />
                              ) : null}
                              {evidenceLocationLabel(S, result.heading_path, result.page_start, result.page_end) ? (
                                <div className="kb-notes-evidence-location">
                                  {evidenceLocationLabel(S, result.heading_path, result.page_start, result.page_end)}
                                </div>
                              ) : null}
                              {result.status === 'exact' && result.target ? (
                                <div className="kb-notes-evidence-context">
                                  {result.before.map((block, blockIndex) => (
                                    <div key={`before-${block.block_id || blockIndex}`} className="kb-notes-evidence-context-block">
                                      <small>{S.research_notes_evidence_before}</small>
                                      <MarkdownRenderer content={block.markdown || block.text} citeDetails={[]} linkifyPlainCitations={false} />
                                    </div>
                                  ))}
                                  <div className="kb-notes-evidence-context-block is-target">
                                    <small>{S.research_notes_evidence_target}</small>
                                    <MarkdownRenderer content={result.target.markdown || result.target.text} citeDetails={[]} linkifyPlainCitations={false} />
                                  </div>
                                  {result.after.map((block, blockIndex) => (
                                    <div key={`after-${block.block_id || blockIndex}`} className="kb-notes-evidence-context-block">
                                      <small>{S.research_notes_evidence_after}</small>
                                      <MarkdownRenderer content={block.markdown || block.text} citeDetails={[]} linkifyPlainCitations={false} />
                                    </div>
                                  ))}
                                </div>
                              ) : null}
                              {result.status === 'candidate' ? (
                                <div className="kb-notes-evidence-candidates">
                                  <strong>{S.research_notes_evidence_candidates_title}</strong>
                                  <Alert
                                    type="info"
                                    showIcon
                                    title={S.research_notes_evidence_candidate_decision_title}
                                    description={result.reason === 'ambiguous_quote'
                                      ? S.research_notes_evidence_candidate_auto_blocked_ambiguous
                                      : result.reason === 'saved_target_missing'
                                        ? S.research_notes_evidence_candidate_auto_blocked_changed
                                        : S.research_notes_evidence_candidate_auto_blocked_generic}
                                    data-testid={`research-note-evidence-candidate-explanation-${index}`}
                                  />
                                  {result.candidates[0] ? (
                                    <div className="kb-notes-evidence-candidate-resolution-actions">
                                      <Button
                                        danger
                                        size="small"
                                        loading={candidateDecisionKey === `${reviewKey}:${result.candidates[0].block_id || result.candidates[0].anchor_id || 'candidate'}:invalid`}
                                        disabled={saving || saveIssue === 'conflict' || Boolean(candidateDecisionKey)}
                                        data-testid={`research-note-evidence-invalidate-candidate-${index}-0`}
                                        onClick={() => {
                                          void decideCandidateEvidence(
                                            link,
                                            index,
                                            result.candidates[0],
                                            result,
                                            reviewKey,
                                            'invalid',
                                          )
                                        }}
                                      >
                                        {S.research_notes_evidence_candidate_mark_invalid}
                                      </Button>
                                    </div>
                                  ) : null}
                                  {result.candidates.map((candidate, candidateIndex) => {
                                    const matchedFields = candidate.context_match?.matched_fields || []
                                    const confirmKey = `${reviewKey}:${candidate.block_id || candidate.anchor_id || 'candidate'}:confirmed`
                                    return (
                                      <article key={candidate.block_id || candidateIndex}>
                                        <div className="kb-notes-evidence-candidate-heading">
                                          <small>
                                            {evidenceLocationLabel(
                                              S,
                                              candidate.heading_path,
                                              candidate.page_start,
                                              candidate.page_end,
                                            ) || S.research_notes_evidence_location_unknown}
                                          </small>
                                          {candidate.context_match?.best_match ? (
                                            <Tag
                                              color="blue"
                                              data-testid={`research-note-evidence-candidate-best-${index}-${candidateIndex}`}
                                            >
                                              {S.research_notes_evidence_candidate_best_match}
                                            </Tag>
                                          ) : null}
                                        </div>
                                        {matchedFields.length > 0 ? (
                                          <div
                                            className="kb-notes-evidence-candidate-matches"
                                            data-testid={`research-note-evidence-candidate-matches-${index}-${candidateIndex}`}
                                          >
                                            <span>{S.research_notes_evidence_candidate_context_matches}</span>
                                            {matchedFields.map(field => (
                                              <Tag key={field} color="blue">
                                                {evidenceContextMatchFieldLabel(S, field)}
                                              </Tag>
                                            ))}
                                          </div>
                                        ) : null}
                                        <div
                                          className="kb-notes-evidence-candidate-diff"
                                          data-testid={`research-note-evidence-candidate-diff-${index}-${candidateIndex}`}
                                        >
                                          <section>
                                            <small>{S.research_notes_evidence_candidate_saved_text}</small>
                                            <p>{renderEvidenceTextDifference(
                                              String(link.evidence_quote || '').trim(),
                                              candidate.text,
                                              'saved',
                                            )}</p>
                                          </section>
                                          <section>
                                            <small>{S.research_notes_evidence_candidate_current_text}</small>
                                            <p>{renderEvidenceTextDifference(
                                              String(link.evidence_quote || '').trim(),
                                              candidate.text,
                                              'candidate',
                                            )}</p>
                                          </section>
                                        </div>
                                        <div className="kb-notes-evidence-candidate-passage">
                                          {(candidate.before || []).map((block, blockIndex) => (
                                            <div key={`before-${block.block_id || blockIndex}`} className="kb-notes-evidence-candidate-block">
                                              <small>{S.research_notes_evidence_before}</small>
                                              <MarkdownRenderer content={block.markdown || block.text} citeDetails={[]} linkifyPlainCitations={false} />
                                            </div>
                                          ))}
                                          <div className="kb-notes-evidence-candidate-block is-target">
                                            <small>{S.research_notes_evidence_target}</small>
                                            <MarkdownRenderer content={candidate.markdown || candidate.text} citeDetails={[]} linkifyPlainCitations={false} />
                                          </div>
                                          {(candidate.after || []).map((block, blockIndex) => (
                                            <div key={`after-${block.block_id || blockIndex}`} className="kb-notes-evidence-candidate-block">
                                              <small>{S.research_notes_evidence_after}</small>
                                              <MarkdownRenderer content={block.markdown || block.text} citeDetails={[]} linkifyPlainCitations={false} />
                                            </div>
                                          ))}
                                        </div>
                                        <div className="kb-notes-evidence-candidate-actions">
                                          <Button
                                            type="primary"
                                            size="small"
                                            loading={candidateDecisionKey === confirmKey}
                                            disabled={saving || saveIssue === 'conflict' || Boolean(candidateDecisionKey)}
                                            data-testid={`research-note-evidence-confirm-candidate-${index}-${candidateIndex}`}
                                            onClick={() => {
                                              void decideCandidateEvidence(
                                                link,
                                                index,
                                                candidate,
                                                result,
                                                reviewKey,
                                                'confirmed',
                                              )
                                            }}
                                          >
                                            {S.research_notes_evidence_candidate_confirm_same}
                                          </Button>
                                          <Button
                                            type="link"
                                            size="small"
                                            icon={<ArrowRightOutlined />}
                                            disabled={saving || saveIssue === 'conflict' || Boolean(candidateDecisionKey)}
                                            data-testid={`research-note-evidence-repair-candidate-${index}-${candidateIndex}`}
                                            onClick={() => {
                                              void openRepairCandidate(link, index, candidate, candidate.reader_open, result)
                                            }}
                                          >
                                            {S.research_notes_evidence_review_candidate}
                                          </Button>
                                        </div>
                                      </article>
                                    )
                                  })}
                                </div>
                              ) : null}
                            </section>
                          ) : null}
                        </>
                      )}
                    </article>
                  )
                })}
              </>
            )}
          </div>
        </aside>
      </section>

      <Modal
        open={Boolean(outlineReview)}
        width={980}
        title={outlineReview
          ? S.research_notes_outline_refresh_title.replace('{title}', outlineReview.latest.title)
          : S.research_notes_outline_refresh_title.replace('{title}', '')}
        onCancel={() => setOutlineReview(null)}
        mask={{ closable: !saving }}
        footer={[
          <Button key="cancel" onClick={() => setOutlineReview(null)}>{S.confirm_cancel}</Button>,
          <Button
            key="apply"
            type="primary"
            danger={Boolean(outlineReview?.hasManualEdits)}
            disabled={saving || saveIssue === 'conflict'}
            data-testid="research-note-outline-apply-refresh"
            onClick={applyOutlineSourceRefresh}
          >
            {S.research_notes_outline_refresh_apply}
          </Button>,
        ]}
      >
        {outlineReview ? (
          <div className="kb-notes-outline-review" data-testid="research-note-outline-review-dialog">
            <p className="kb-notes-outline-review-revision">
              {S.research_notes_outline_revision_change
                .replace('{from}', String(outlineReview.source.revision))
                .replace('{to}', String(outlineReview.latest.revision))}
            </p>
            {outlineReview.hasManualEdits ? (
              <Alert type="warning" showIcon title={S.research_notes_outline_manual_edits_warning} />
            ) : (
              <Alert type="info" showIcon title={S.research_notes_outline_safe_refresh_hint} />
            )}
            {saveIssue === 'conflict' ? (
              <Alert type="error" showIcon title={S.research_notes_outline_conflict_blocked} />
            ) : null}
            <div className="kb-notes-outline-review-columns">
              <section>
                <h3>{S.research_notes_outline_current_section}</h3>
                <pre>{outlineReview.currentSection}</pre>
              </section>
              <section>
                <h3>{S.research_notes_outline_latest_source}</h3>
                <pre>{outlineReview.latest.content_markdown}</pre>
              </section>
            </div>
          </div>
        ) : null}
      </Modal>

      <Modal
        open={combineOpen}
        width={920}
        title={S.research_notes_combine_title}
        closable={!combineSaving}
        mask={{ closable: !combineSaving }}
        onCancel={() => { if (!combineSaving) setCombineOpen(false) }}
        footer={[
          <Button key="copy" icon={<CopyOutlined />} onClick={() => {
            void copyText(combineContent).then(() => message.success(S.research_notes_copied))
          }}>{S.research_notes_copy}</Button>,
          <Button key="markdown" icon={<FileMarkdownOutlined />} onClick={() => {
            downloadMarkdown(`${safeFilename(combineTitle, 'evidence-outline')}.md`, combineContent)
          }}>{S.research_notes_export_markdown}</Button>,
          <Button key="word" icon={<FileWordOutlined />} onClick={() => {
            void chatApi.downloadResearchNoteDocx({ title: combineTitle, content_markdown: combineContent })
              .catch(() => message.error(S.research_notes_export_failed))
          }}>{S.research_notes_export_word}</Button>,
          <Button
            key="save"
            type="primary"
            icon={<SaveOutlined />}
            loading={combineSaving}
            disabled={!combineTitle.trim() || !combineContent.trim() || combineRecords.length <= 0}
            data-testid="research-note-outline-save"
            onClick={() => { void saveCombinedOutline() }}
          >
            {S.research_notes_save_outline}
          </Button>,
        ]}
      >
        <div className="kb-notes-combine-modal">
          <Input value={combineTitle} maxLength={240} onChange={event => updateCombineTitle(event.target.value)} />
          <div className="kb-notes-combine-project-row">
            <label>
              <span><FolderOutlined /> {S.research_notes_project}</span>
              <Select
                value={combineProjectId || UNASSIGNED_PROJECT}
                aria-label={S.research_notes_outline_project}
                onChange={value => setCombineProjectId(value === UNASSIGNED_PROJECT ? null : value)}
                options={[
                  { value: UNASSIGNED_PROJECT, label: S.research_notes_unassigned },
                  ...projects.map(project => ({ value: project.id, label: project.name })),
                ]}
              />
            </label>
            {combineHasMixedProjects && !combineProjectId ? (
              <small>{S.research_notes_outline_mixed_project_hint}</small>
            ) : null}
          </div>
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
