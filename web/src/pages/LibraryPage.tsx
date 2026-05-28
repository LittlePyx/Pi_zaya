
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import {
  Upload,
  AutoComplete,
  Button,
  List,
  Drawer,
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
  Alert,
  Tooltip,
  Dropdown,
  Modal,
  Segmented,
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
import type {
  ConversionQualitySummary,
  LibraryFileItem,
  LibraryQualityActionDelta,
  LibraryQualityActionHistoryItem,
  LibraryQualityActionSnapshot,
  LibraryQualityDomain,
  LibraryQualityFeatureHealth,
  LibraryQualityFeatureHealthItem,
  LibraryQualityFailureCase,
  LibraryQualityFullChain,
  LibraryQualityFullChainStage,
  LibraryQualityOverviewResponse,
  LibraryQualityPriorityAction,
  LibraryQualityRepairAction,
  LibraryQualityRepairImpact,
  LibraryQualityRepairRun,
  LibraryResearchQaRerunResponse,
  RenameSuggestionItem,
} from '../api/library'
import { libraryApi } from '../api/library'
import { referencesApi } from '../api/references'
import { useChatStore } from '../stores/chatStore'
import { settingsApi } from '../api/settings'
import { useLibraryStore } from '../stores/libraryStore'
import { useSettingsStore } from '../stores/settingsStore'
import VirtualList from 'rc-virtual-list'
import { useNavigate } from 'react-router-dom'
import { useT } from '../i18n'

const { Text } = Typography
const { Dragger } = Upload
const FILE_VIRTUAL_THRESHOLD = 60
const FILE_VIRTUAL_HEIGHT = 620
const FILE_VIRTUAL_ROW_HEIGHT = 88

type FileTabKey = 'pending' | 'converted' | 'all'
type LibraryBrowseMode = 'list' | 'categories' | 'tags'
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
  savedName?: string
  savedSha1?: string
  taskId?: string
  convertRequested?: boolean
  suggestionBasisLabel?: string
  suggestionBasisDetail?: string
  suggestionMatchMethod?: string
  suggestionYearSource?: string
}

const CONVERT_MODE = 'balanced'
const QUALITY_REPAIR_HISTORY_STORAGE_KEY = 'kb.library.qualityRepairHistory.v1'
const QUALITY_REPAIR_HISTORY_LIMIT = 40
const PAPER_CATEGORY_PRESETS = [
  'NeRF',
  '3DGS',
  'SCI',
  'Diffusion',
  'Single-Photon Imaging',
  'Single-Pixel Imaging',
  'Inverse Imaging',
  'Survey',
  'Dataset',
  'Benchmark',
] as const
const TAG_INPUT_SEPARATORS = [',', '，', ';', '；']
function READING_STATUS_OPTIONS(S: Record<string, string>) {
  return [
    { value: '', label: S.lib_reading_status_all },
    { value: 'unread', label: S.lib_reading_status_unread },
    { value: 'reading', label: S.lib_reading_status_reading },
    { value: 'done', label: S.lib_reading_status_done },
    { value: 'revisit', label: S.lib_reading_status_revisit },
  ] as const
}

type ReadingStatusValue = '' | 'unread' | 'reading' | 'done' | 'revisit'
type LibraryMetaDraft = {
  paper_category: string
  reading_status: ReadingStatusValue
  note: string
  user_tags: string[]
}

type LibraryBatchMetaDraft = {
  apply_paper_category: boolean
  paper_category: string
  apply_reading_status: boolean
  reading_status: ReadingStatusValue
  add_tags: string[]
  remove_tags: string[]
}

type QualityRepairBaseline = {
  quality: ConversionQualitySummary | null
  startedAt: number
}

type QualityRepairHistoryRecord = {
  name: string
  beforeScore: number
  afterScore: number
  beforeStatus: string
  afterStatus: string
  fixedIssues: string[]
  remainingIssues: string[]
  updatedAt: number
}

type QualityRepairRunOptions = {
  autoReindexImmediate?: boolean
  autoReindexQueued?: boolean
}

type QualityIssueStat = {
  key: string
  label: string
  severity: string
  papers: number
  count: number
  repairStrategy?: string
}

type QualityReportRecommendationView = {
  name: string
  score: number
  issues: string[]
}

type QualityDomainView = {
  key: 'conversion' | 'research_qa' | 'citation_cards'
  label: string
  available: boolean
  status: string
  statusLabel: string
  countText: string
  detailText: string
  failureText: string
}

type QualityFullChainActionResult = {
  status: 'success' | 'warning' | 'error' | 'info'
  summary: string
  detail?: string
  deltaText?: string
  verificationText?: string
  improved?: boolean | null
  updatedAt: number
}

type CategoryCardItem = {
  key: string
  label: string
  count: number
  unreadCount: number
  convertedCount: number
  pendingCount: number
  commonTags: string[]
  recentPapers: string[]
}

type TagCardItem = {
  key: string
  label: string
  count: number
  unreadCount: number
  categories: string[]
  recentPapers: string[]
}

type FilterFilesOptions = {
  ignoreCategoryFilter?: boolean
  ignoreTagFilter?: boolean
}

type TextOption = {
  value?: string | number
  label?: ReactNode
}

type SuggestionMetaInfo = {
  match_method?: string
  year_source?: string
  basis_label?: string
  basis_detail?: string
}

function SCOPE_OPTIONS(S: Record<string, string>) {
  return [
    { value: '200', label: S.lib_scope_recent_200 },
    { value: '1000', label: S.lib_scope_recent_1000 },
    { value: 'all', label: S.lib_scope_all },
  ]
}

function RENAME_SCOPE_OPTIONS(S: Record<string, string>) {
  return [
    { value: '30', label: S.lib_scope_recent_30 },
    { value: '50', label: S.lib_scope_recent_50 },
    { value: '100', label: S.lib_scope_recent_100 },
    { value: 'all', label: S.lib_scope_all },
  ]
}

function DRAFT_STATUS_TEXT(S: Record<string, string>): Record<DraftStatus, string> {
  return {
    queued: S.lib_draft_queued,
    inspecting: S.lib_draft_inspecting,
    ready: S.lib_draft_ready,
    saving: S.lib_draft_saving,
    saved: S.lib_draft_saved,
    error: S.lib_draft_error,
  }
}

function FAILED_REASON_META(S: Record<string, string>): Record<Exclude<UploadErrorReason, 'all'>, { label: string, icon: ReactNode }> {
  return {
    duplicate: { label: S.lib_fail_duplicate, icon: <CopyOutlined /> },
    path: { label: S.lib_fail_path, icon: <FolderOpenOutlined /> },
    permission: { label: S.lib_fail_permission, icon: <LockOutlined /> },
    network: { label: S.lib_fail_network, icon: <ApiOutlined /> },
    other: { label: S.lib_fail_other, icon: <ExclamationCircleOutlined /> },
  }
}

function fileTag(item: LibraryFileItem, S: Record<string, string>) {
  if (item.task_state === 'running') return { color: 'processing' as const, text: S.lib_tag_converting }
  if (item.task_state === 'queued') return { color: 'warning' as const, text: `${S.lib_tag_queued}${item.queue_pos > 0 ? ` #${item.queue_pos}` : ''}` }
  return item.category === 'converted'
    ? { color: 'success' as const, text: S.lib_tag_converted }
    : { color: 'default' as const, text: S.lib_tag_pending }
}

function derivePageProgress(done0: number, total0: number, msg0: string) {
  const done = Number(done0 || 0)
  const total = Number(total0 || 0)
  if (total > 0) return { done: Math.max(0, done), total: Math.max(0, total) }
  const msg = String(msg0 || '')
  const m = msg.match(/\b(\d{1,4})\s*\/\s*(\d{1,4})\b/)
  if (!m) return { done: 0, total: 0 }
  const parsedDone = Number(m[1] || 0)
  const parsedTotal = Number(m[2] || 0)
  if (!Number.isFinite(parsedDone) || !Number.isFinite(parsedTotal) || parsedTotal <= 0) {
    return { done: 0, total: 0 }
  }
  return { done: Math.max(0, parsedDone), total: Math.max(0, parsedTotal) }
}

function deriveConvertStageLabel(msg0: string, S_?: Record<string, string>) {
  const msg = String(msg0 || '').trim().toLowerCase()
  if (!msg) return ''
  if (msg.includes('ingesting')) return S_ ? S_.lib_convert_ingesting : '正在更新知识库索引...'
  if (msg.includes('cancel')) return S_ ? S_.lib_convert_cancelling : '正在取消转换...'
  return ''
}

function matchesKeyword(name: string, keyword: string) {
  if (!keyword) return true
  return name.toLowerCase().includes(keyword)
}

function readingStatusLabel(value: string, S_?: Record<string, string>) {
  if (value === 'unread') return S_ ? S_.lib_reading_status_unread : '未读'
  if (value === 'reading') return S_ ? S_.lib_reading_status_reading : '在读'
  if (value === 'done') return S_ ? S_.lib_reading_status_done : '已读'
  if (value === 'revisit') return S_ ? S_.lib_reading_status_revisit : '待回看'
  return ''
}

function stripKnownSourceExt(name: string) {
  return String(name || '')
    .replace(/\.en\.md$/i, '')
    .replace(/\.md$/i, '')
    .replace(/\.pdf$/i, '')
    .trim()
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

function normalizeTextValue(value: unknown) {
  return String(value || '').replace(/\s+/g, ' ').trim()
}

function qualityDomainNumber(domain: LibraryQualityDomain | undefined, key: string) {
  const value = domain?.summary?.[key]
  const num = Number(value || 0)
  return Number.isFinite(num) ? num : 0
}

function qualityDomainStatus(domain: LibraryQualityDomain | undefined, fallback = 'unknown') {
  return normalizeTextValue(domain?.status || fallback).toLowerCase() || 'unknown'
}

function qualityStatusText(status: string, S: Record<string, string>) {
  if (status === 'good') return S.lib_quality_domain_status_good
  if (status === 'error') return S.lib_quality_domain_status_error
  if (status === 'warning') return S.lib_quality_domain_status_warning
  return S.lib_quality_domain_status_unknown
}

function qualityCompareRank(status: string) {
  const clean = normalizeTextValue(status).toLowerCase()
  if (clean === 'error') return 3
  if (clean === 'warning') return 2
  if (clean === 'unknown') return 1
  if (clean === 'good' || clean === 'success') return 0
  return 1
}

function qualityWorstStatus(values: string[]) {
  let worst = ''
  for (const value of values) {
    const clean = normalizeTextValue(value).toLowerCase()
    if (!clean) continue
    if (!worst || qualityCompareRank(clean) > qualityCompareRank(worst)) worst = clean
  }
  return worst || 'unknown'
}

function qualityFeatureMatchesStage(item: LibraryQualityFeatureHealthItem, stageKey: string) {
  const key = normalizeTextValue(stageKey).toLowerCase()
  const itemKey = normalizeTextValue(item.key).toLowerCase()
  const targetStage = normalizeTextValue(item.target_stage).toLowerCase()
  if (targetStage === key || itemKey === key) return true
  if (key === 'conversion') return itemKey === 'pdf_conversion'
  if (key === 'research_qa') return itemKey === 'general_qa' || itemKey === 'paper_guide'
  if (key === 'citations') return itemKey === 'citation_cards' || itemKey === 'reader_locate'
  if (key === 'shelf') return itemKey === 'literature_basket'
  if (key === 'repair_loop') return itemKey === 'repair_loop'
  return false
}

function qualityOverviewStageSnapshot(
  overview: LibraryQualityOverviewResponse | null | undefined,
  stageKey: string,
): LibraryQualityActionSnapshot {
  const key = normalizeTextValue(stageKey).toLowerCase()
  const stages = Array.isArray(overview?.full_chain?.stages) ? overview?.full_chain?.stages || [] : []
  const stage = stages.find((item) => normalizeTextValue(item.key).toLowerCase() === key)
  const featureItems = (Array.isArray(overview?.feature_health?.items) ? overview?.feature_health?.items || [] : [])
    .filter((item) => qualityFeatureMatchesStage(item, key))
  const featureScores = featureItems
    .map((item) => Number(item.score || 0))
    .filter((score) => Number.isFinite(score) && score > 0)
  const stageCount = Number(stage?.count || 0)
  const featureCount = featureItems.reduce((sum, item) => sum + Number(item.count || 0), 0)
  const score = featureScores.length > 0
    ? Math.round(featureScores.reduce((sum, scoreValue) => sum + scoreValue, 0) / featureScores.length)
    : Math.round(Number(overview?.full_chain?.score || 0))
  const status = qualityWorstStatus([
    normalizeTextValue(stage?.status),
    ...featureItems.map((item) => normalizeTextValue(item.status)),
  ])
  return {
    status,
    score: Math.max(0, Math.min(100, score)),
    count: Math.max(0, stageCount || featureCount),
    summary: normalizeTextValue(stage?.label || key),
    detail: normalizeTextValue(stage?.detail || featureItems[0]?.summary || overview?.full_chain?.summary || overview?.status),
    blocking: Boolean(stage?.blocking || featureItems.some((item) => item.blocking)),
  }
}

function qualitySnapshotLabel(snapshot: LibraryQualityActionSnapshot | undefined) {
  if (!snapshot) return ''
  const bits: string[] = []
  const status = normalizeTextValue(snapshot.status)
  const score = Number(snapshot.score || 0)
  const count = Number(snapshot.count || 0)
  if (status) bits.push(status)
  if (score > 0) bits.push(`Q${Math.round(score)}`)
  if (count > 0) bits.push(`${count} open`)
  return bits.join(' / ')
}

function qualityBuildActionDelta(
  before: LibraryQualityActionSnapshot | undefined,
  after: LibraryQualityActionSnapshot | undefined,
): LibraryQualityActionDelta {
  const beforeStatus = normalizeTextValue(before?.status).toLowerCase()
  const afterStatus = normalizeTextValue(after?.status).toLowerCase()
  if (!beforeStatus || !afterStatus) {
    return { improved: null, summary: 'Verification pending' }
  }
  const scoreDelta = Math.round(Number(after?.score || 0) - Number(before?.score || 0))
  const countDelta = Math.round(Number(before?.count || 0) - Number(after?.count || 0))
  const statusDelta = qualityCompareRank(beforeStatus) - qualityCompareRank(afterStatus)
  const improved = statusDelta > 0 || scoreDelta >= 3 || countDelta > 0
  const worsened = statusDelta < 0 || scoreDelta <= -3 || countDelta < 0
  const beforeLabel = qualitySnapshotLabel(before)
  const afterLabel = qualitySnapshotLabel(after)
  let summary = 'No measurable change yet'
  if (improved) {
    summary = `Improved: ${beforeLabel || beforeStatus} -> ${afterLabel || afterStatus}`
  } else if (worsened) {
    summary = `Needs follow-up: ${beforeLabel || beforeStatus} -> ${afterLabel || afterStatus}`
  }
  return {
    improved,
    worsened,
    status_changed: beforeStatus !== afterStatus,
    score_delta: scoreDelta,
    count_delta: countDelta,
    summary,
  }
}

function qualityActionDeltaText(item: Pick<LibraryQualityActionHistoryItem, 'delta' | 'improved' | 'before' | 'after'>) {
  const explicit = normalizeTextValue(item.delta?.summary)
  if (explicit) return explicit
  const before = item.before
  const after = item.after
  if (before || after) return qualityBuildActionDelta(before, after).summary || ''
  if (item.improved === true) return 'Improved'
  if (item.improved === false) return 'No measurable change yet'
  return ''
}

function qualityVerificationFromRerun(rerun: LibraryResearchQaRerunResponse | null | undefined) {
  if (!rerun) return {}
  return {
    type: 'research_qa_rerun',
    case_id: normalizeTextValue(rerun.case_id),
    status: normalizeTextValue(rerun.status),
    quality_ok: Boolean(rerun.quality_ok || rerun.status === 'passed'),
    failure_count: Number(rerun.failures?.length || 0),
    error_kind: normalizeTextValue(rerun.error_kind),
    error_detail: normalizeTextValue(rerun.error_detail),
  }
}

function qualityVerificationText(verification: Record<string, unknown> | undefined) {
  if (!verification || !Object.keys(verification).length) return ''
  const type = normalizeTextValue(verification.type)
  if (type === 'research_qa_rerun') {
    const caseId = normalizeTextValue(verification.case_id)
    const status = normalizeTextValue(verification.status).toLowerCase()
    const errorKind = normalizeTextValue(verification.error_kind)
    if (verification.quality_ok === true || status === 'passed') return caseId ? `QA rerun passed: ${caseId}` : 'QA rerun passed'
    if (status === 'skipped') return 'No linked QA case'
    if (errorKind) return `QA rerun needs service: ${errorKind}`
    if (status) return caseId ? `QA rerun ${status}: ${caseId}` : `QA rerun ${status}`
  }
  return ''
}

function qualityTopFailureText(domain: LibraryQualityDomain | undefined) {
  const first = Array.isArray(domain?.top_failures) ? domain?.top_failures?.[0] : null
  const name = normalizeTextValue(first?.name)
  if (!name) return ''
  const count = Number(first?.count || 0)
  return count > 0 ? `${name} x${count}` : name
}

function qualityActionText(action: LibraryQualityPriorityAction, S: Record<string, string>) {
  const domain = normalizeTextValue(action.domain)
  const label = normalizeTextValue(action.label)
  if (domain === 'conversion') return S.lib_quality_action_conversion
  if (domain === 'research_qa' && label.toLowerCase().includes('run')) return S.lib_quality_action_research_qa_run
  if (domain === 'research_qa') return S.lib_quality_action_research_qa
  if (domain === 'citation_cards') return S.lib_quality_action_citation_cards
  return label || domain
}

function qualityFullChainActionText(stage: LibraryQualityFullChainStage) {
  const action = normalizeTextValue(stage.action).toLowerCase()
  const status = normalizeTextValue(stage.status).toLowerCase()
  if (status === 'good' && action.startsWith('monitor_')) return 'Verified'
  if (action === 'repair_conversion') return 'Repair'
  if (action === 'fix_failed_qa_cases') return 'Fix case'
  if (action === 'run_research_qa') return 'Open runbook'
  if (action === 'rebuild_index') return 'Rebuild'
  if (action === 'repair_citation_cards') return 'Repair cards'
  if (action === 'repair_shelf_metadata') return 'Repair metadata'
  if (action === 'rerun_failed_cases') return 'Rerun'
  if (action.startsWith('monitor_')) return 'Review'
  return normalizeTextValue(stage.action).replace(/_/g, ' ') || 'Review'
}

function qualityFailureCaseMatchesStage(item: LibraryQualityFailureCase, stageKey: string) {
  const key = normalizeTextValue(stageKey).toLowerCase()
  const names = new Set([
    ...(item.failure_names || []),
    ...((item.failures || []).map((failure) => failure.name)),
  ].map((value) => normalizeTextValue(value).toLowerCase()).filter(Boolean))
  const rootCodes = new Set(
    (item.root_causes || []).map((cause) => normalizeTextValue(cause.code).toLowerCase()).filter(Boolean),
  )
  const actions = new Set(
    (item.root_causes || []).map((cause) => normalizeTextValue(cause.action).toLowerCase()).filter(Boolean),
  )
  if (key === 'research_qa' || key === 'repair_loop') return true
  if (key === 'retrieval') {
    return Boolean(item.missing_expected_doc_ids?.length)
      || names.has('refs_include_required_docs')
      || rootCodes.has('retrieval_missing_expected_docs')
      || rootCodes.has('empty_reference_basket')
      || actions.has('rebuild_index')
  }
  if (key === 'citations') {
    return names.has('citation_card_quality')
      || names.has('refs_card_copy_quality')
      || names.has('system_b_audit')
      || rootCodes.has('citation_card_quality')
      || rootCodes.has('citation_missing_expected_docs')
      || rootCodes.has('system_b_mapping')
  }
  if (key === 'shelf') {
    return names.has('citation_shelf_quality')
      || names.has('citation_card_quality')
      || rootCodes.has('citation_card_quality')
      || actions.has('inspect_cards')
  }
  return false
}

function qualityActionHistoryActionText(item: LibraryQualityActionHistoryItem) {
  const stageKey = normalizeTextValue(item.stage_key).toLowerCase()
  const hasTarget = Boolean((item.target_ids || []).some((value) => normalizeTextValue(value)))
  if (stageKey === 'conversion') return hasTarget ? 'Focus source' : 'Review'
  if (['research_qa', 'retrieval', 'repair_loop', 'citations', 'shelf'].includes(stageKey)) {
    return hasTarget ? 'Open replay' : (stageKey === 'citations' || stageKey === 'shelf' ? 'Open report' : 'Review')
  }
  if (stageKey === 'citation_cards') return 'Open report'
  return hasTarget ? 'Open target' : 'Review'
}

function qualityFeatureActionText(item: LibraryQualityFeatureHealthItem) {
  const action = normalizeTextValue(item.action).toLowerCase()
  if (action.startsWith('repair_')) return 'Repair'
  if (action.startsWith('fix_')) return 'Fix'
  if (action.startsWith('run_')) return 'Run'
  if (action.startsWith('inspect_')) return 'Inspect'
  return 'Review'
}

function normalizeTextList(values: unknown[]) {
  const out: string[] = []
  const seen = new Set<string>()
  for (const value of values) {
    const clean = normalizeTextValue(value)
    if (!clean) continue
    const key = clean.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    out.push(clean)
  }
  return out
}

function uniqueTextValues(values: Iterable<unknown>) {
  const out: string[] = []
  const seen = new Set<string>()
  for (const value of values) {
    const clean = normalizeTextValue(value)
    if (!clean) continue
    const key = clean.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    out.push(clean)
  }
  return out
}

function isUploadDraftConverted(draft: UploadDraft, files: LibraryFileItem[]) {
  if (!draft.convertRequested || draft.status !== 'saved') return false
  const match = files.find((item) => {
    if (draft.savedSha1 && item.sha1) return item.sha1 === draft.savedSha1
    if (draft.savedName) return item.name === draft.savedName
    return false
  })
  if (!match) return false
  return match.md_exists && match.task_state === 'idle' && match.category === 'converted'
}

function suggestionBasisTagColor(meta?: SuggestionMetaInfo) {
  const method = String(meta?.match_method || '').trim().toLowerCase()
  const yearSource = String(meta?.year_source || '').trim().toLowerCase()
  if (method === 'doi') return 'success'
  if (method === 'crossref_strong') return 'processing'
  if (yearSource === 'filename') return 'gold'
  if (method === 'crossref_weak') return 'warning'
  return 'default'
}

function conversionQualityStatus(quality?: ConversionQualitySummary | null) {
  return String(quality?.status || '').trim().toLowerCase()
}

function conversionQualityToneClass(quality?: ConversionQualitySummary | null) {
  const status = conversionQualityStatus(quality)
  if (status === 'good') return 'is-good'
  if (status === 'error') return 'is-error'
  if (status === 'warning') return 'is-warning'
  return 'is-unknown'
}

function conversionQualityLabel(quality?: ConversionQualitySummary | null) {
  if (!quality) return ''
  const score = Number(quality.score || 0)
  const scoreText = Number.isFinite(score) ? `Q${Math.max(0, Math.min(100, Math.round(score)))}` : 'Q?'
  const status = conversionQualityStatus(quality)
  if (status === 'good') return scoreText
  if (status === 'error') return `Repair ${scoreText}`
  return `Review ${scoreText}`
}

function conversionMetric(quality: ConversionQualitySummary | null | undefined, key: keyof ConversionQualitySummary['metrics']) {
  const value = Number(quality?.metrics?.[key] || 0)
  return Number.isFinite(value) ? Math.max(0, Math.round(value)) : 0
}

function conversionQualityNeedsReview(quality?: ConversionQualitySummary | null) {
  const status = conversionQualityStatus(quality)
  return Boolean(quality?.has_review_issue) || status === 'warning' || status === 'error'
}

function conversionQualityScore(quality?: ConversionQualitySummary | null) {
  const value = Number(quality?.score || 0)
  return Number.isFinite(value) ? Math.max(0, Math.min(100, Math.round(value))) : 0
}

function conversionQualityIssueEntries(quality?: ConversionQualitySummary | null) {
  return (Array.isArray(quality?.issues) ? quality?.issues || [] : [])
    .map((issue) => {
      const label = normalizeTextValue(issue.label || issue.code)
      const key = normalizeTextValue(issue.code || issue.label).toLowerCase()
      return key && label ? { key, label } : null
    })
    .filter((item): item is { key: string; label: string } => Boolean(item))
}

function summarizeConversionQualityRepair(
  before: ConversionQualitySummary | null,
  after: ConversionQualitySummary | null,
  S: Record<string, string>,
) {
  const beforeScore = conversionQualityScore(before)
  const afterScore = conversionQualityScore(after)
  const beforeIssues = conversionQualityIssueEntries(before)
  const afterIssueKeys = new Set(conversionQualityIssueEntries(after).map((item) => item.key))
  const fixedIssues = beforeIssues
    .filter((item) => !afterIssueKeys.has(item.key))
    .map((item) => item.label)
    .slice(0, 3)
  const remaining = conversionQualityIssueEntries(after).length
  const template = after && !conversionQualityNeedsReview(after)
    ? S.quality_repair_result_pass
    : S.quality_repair_result_review
  const base = String(template || 'Repair checked: Q{before} -> Q{after}')
    .replace('{before}', String(beforeScore))
    .replace('{after}', String(afterScore))
    .replace('{n}', String(remaining))
  if (!fixedIssues.length) return base
  return `${base} · ${String(S.quality_repair_result_fixed_issues || 'Fixed: {issues}').replace('{issues}', fixedIssues.join(' / '))}`
}

function buildQualityRepairHistoryRecord(
  name: string,
  before: ConversionQualitySummary | null,
  after: ConversionQualitySummary | null,
): QualityRepairHistoryRecord {
  const afterIssues = conversionQualityIssueEntries(after)
  const afterIssueKeys = new Set(afterIssues.map((item) => item.key))
  const fixedIssues = conversionQualityIssueEntries(before)
    .filter((item) => !afterIssueKeys.has(item.key))
    .map((item) => item.label)
    .slice(0, 6)
  return {
    name,
    beforeScore: conversionQualityScore(before),
    afterScore: conversionQualityScore(after),
    beforeStatus: conversionQualityStatus(before),
    afterStatus: conversionQualityStatus(after),
    fixedIssues,
    remainingIssues: afterIssues.map((item) => item.label).slice(0, 6),
    updatedAt: Date.now(),
  }
}

function normalizeQualityRepairHistory(value: unknown): Record<string, QualityRepairHistoryRecord> {
  const source = (value && typeof value === 'object') ? value as Record<string, unknown> : {}
  const entries = Object.entries(source)
    .map(([key, raw]) => {
      const item = (raw && typeof raw === 'object') ? raw as Record<string, unknown> : {}
      const name = normalizeTextValue(item.name || key)
      if (!name) return null
      return {
        name,
        beforeScore: Math.max(0, Math.min(100, Math.round(Number(item.beforeScore || 0)))),
        afterScore: Math.max(0, Math.min(100, Math.round(Number(item.afterScore || 0)))),
        beforeStatus: normalizeTextValue(item.beforeStatus),
        afterStatus: normalizeTextValue(item.afterStatus),
        fixedIssues: uniqueTextValues(Array.isArray(item.fixedIssues) ? item.fixedIssues : []),
        remainingIssues: uniqueTextValues(Array.isArray(item.remainingIssues) ? item.remainingIssues : []),
        updatedAt: Math.max(0, Number(item.updatedAt || 0)),
      } satisfies QualityRepairHistoryRecord
    })
    .filter((item): item is QualityRepairHistoryRecord => Boolean(item))
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .slice(0, QUALITY_REPAIR_HISTORY_LIMIT)
  return Object.fromEntries(entries.map((item) => [item.name, item]))
}

function loadQualityRepairHistory(): Record<string, QualityRepairHistoryRecord> {
  if (typeof window === 'undefined') return {}
  try {
    const raw = window.localStorage.getItem(QUALITY_REPAIR_HISTORY_STORAGE_KEY)
    if (!raw) return {}
    return normalizeQualityRepairHistory(JSON.parse(raw))
  } catch {
    return {}
  }
}

function saveQualityRepairHistory(records: Record<string, QualityRepairHistoryRecord>) {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(QUALITY_REPAIR_HISTORY_STORAGE_KEY, JSON.stringify(normalizeQualityRepairHistory(records)))
  } catch {
    // Storage is best-effort; the current-session repair result remains visible.
  }
}

function formatQualityRepairHistoryTime(ts: number) {
  if (!Number.isFinite(ts) || ts <= 0) return ''
  try {
    return new Date(ts).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })
  } catch {
    return ''
  }
}

function formatQualityRepairRecordSummary(record: QualityRepairHistoryRecord, S: Record<string, string>) {
  const template = record.remainingIssues.length <= 0 ? S.quality_repair_result_pass : S.quality_repair_result_review
  const base = String(template || 'Repair checked: Q{before} -> Q{after}')
    .replace('{before}', String(record.beforeScore))
    .replace('{after}', String(record.afterScore))
    .replace('{n}', String(record.remainingIssues.length))
  if (!record.fixedIssues.length) return base
  return `${base} · ${String(S.quality_repair_result_fixed_issues || 'Fixed: {issues}').replace('{issues}', record.fixedIssues.slice(0, 3).join(' / '))}`
}

function formatSignedNumber(value: number) {
  const n = Math.round(Number(value || 0))
  return n > 0 ? `+${n}` : String(n)
}

function qualityRepairImpactIndexText(impact: LibraryQualityRepairImpact) {
  if (!impact.needs_reindex) return 'Index current'
  if (impact.reindexed === true) return 'Index refreshed'
  if (impact.reindexed === false && Number(impact.enqueued || 0) <= 0) return 'Index refresh needs retry'
  if (Number(impact.enqueued || 0) > 0) return 'Index will refresh after conversion'
  return 'Index refresh pending'
}

function qualityRepairRunStatusText(run: LibraryQualityRepairRun | null) {
  if (!run) return ''
  const status = normalizeTextValue(run.status).toLowerCase()
  const phase = normalizeTextValue(run.phase).toLowerCase()
  if (phase === 'verification_passed') return 'Run tracked: verified'
  if (phase === 'verification_failed') return 'Run tracked: verification failed'
  if (phase === 'verification_blocked') return 'Run tracked: verification blocked'
  if (status === 'completed' || phase === 'reindex_complete') return 'Run tracked: completed'
  if (status === 'failed' || phase === 'repair_failed') return 'Run tracked: failed'
  if (phase === 'source_reconversion_queued') return 'Run tracked: waiting for conversion'
  if (phase === 'reindex_pending' || status === 'reindex_pending') return 'Run tracked: index refresh pending'
  return `Run tracked: ${run.status || run.phase || 'recorded'}`
}

function qualityRepairRunTagColor(run: LibraryQualityRepairRun | null) {
  const status = normalizeTextValue(run?.status).toLowerCase()
  const phase = normalizeTextValue(run?.phase).toLowerCase()
  if (phase === 'verification_passed') return 'success'
  if (phase === 'verification_failed' || phase === 'verification_blocked') return 'warning'
  if (status === 'completed' || phase === 'reindex_complete') return 'success'
  if (status === 'failed' || phase === 'repair_failed') return 'error'
  if (status === 'queued' || phase === 'source_reconversion_queued') return 'processing'
  if (status === 'reindex_pending' || phase === 'reindex_pending') return 'warning'
  return 'default'
}

function qualityRepairRunCanAdvance(run: LibraryQualityRepairRun | null) {
  if (!run) return false
  const status = normalizeTextValue(run.status).toLowerCase()
  const phase = normalizeTextValue(run.phase).toLowerCase()
  if (phase === 'verification_failed' || phase === 'verification_blocked') return true
  if (status === 'completed' || phase === 'reindex_complete' || phase === 'repair_complete' || phase === 'verification_passed') return false
  return Boolean(run.needs_reindex || status === 'queued' || status === 'reindex_pending' || phase === 'source_reconversion_queued' || phase === 'reindex_pending' || phase === 'reindex_failed')
}

function hasConversionQualityIssue(item: LibraryFileItem) {
  return Boolean(item.conversion_quality?.has_review_issue)
}

function toTextOptions(values: string[]) {
  return values.map((value) => ({ value, label: value }))
}

function optionMatchesInput(input: string, option?: TextOption) {
  const needle = normalizeTextValue(input).toLowerCase()
  if (!needle) return true
  const hay = normalizeTextValue(option?.value || option?.label || '').toLowerCase()
  return hay.includes(needle)
}

function saveResearchQaReplayFailureCase(item: LibraryQualityFailureCase) {
  if (typeof window === 'undefined') return
  try {
    window.sessionStorage.setItem('kb.researchQaReplay.failureCase.v1', JSON.stringify(item))
  } catch {
    // Replay still works from fixture data if session storage is unavailable.
  }
}

export default function LibraryPage() {
  const S = useT()
  const store = useLibraryStore()
  const createPaperGuideConversation = useChatStore((s) => s.createPaperGuideConversation)
  const nav = useNavigate()

  const settingsLoaded = useSettingsStore((s) => s.loaded)
  const settingsPdfDir = useSettingsStore((s) => s.pdfDir)
  const settingsMdDir = useSettingsStore((s) => s.mdDir)
  const updateSettings = useSettingsStore((s) => s.update)

  const [scope, setScope] = useState('200')
  const [tabKey, setTabKey] = useState<FileTabKey>('all')
  const [browseMode, setBrowseMode] = useState<LibraryBrowseMode>('list')
  const [fileKeyword, setFileKeyword] = useState('')
  const [paperCategoryFilter, setPaperCategoryFilter] = useState('')
  const [paperTagFilter, setPaperTagFilter] = useState('')
  const [readingStatusFilter, setReadingStatusFilter] = useState<ReadingStatusValue>('')
  const [onlyUnread, setOnlyUnread] = useState(false)
  const [onlyUnclassified, setOnlyUnclassified] = useState(false)
  const [onlySuggested, setOnlySuggested] = useState(false)
  const [onlyQualityIssues, setOnlyQualityIssues] = useState(false)
  const [metaDrawerOpen, setMetaDrawerOpen] = useState(false)
  const [metaSaving, setMetaSaving] = useState(false)
  const [metaSuggestionSaving, setMetaSuggestionSaving] = useState(false)
  const [metaItem, setMetaItem] = useState<LibraryFileItem | null>(null)
  const [metaDraft, setMetaDraft] = useState<LibraryMetaDraft>({
    paper_category: '',
    reading_status: '',
    note: '',
    user_tags: [],
  })
  const [selectedLibraryNames, setSelectedLibraryNames] = useState<Record<string, boolean>>({})
  const [batchDrawerOpen, setBatchDrawerOpen] = useState(false)
  const [batchSaving, setBatchSaving] = useState(false)
  const [qualityRepairingNames, setQualityRepairingNames] = useState<Record<string, boolean>>({})
  const [qualityRepairResults, setQualityRepairResults] = useState<Record<string, string>>({})
  const [qualityRepairImpact, setQualityRepairImpact] = useState<LibraryQualityRepairImpact | null>(null)
  const [qualityRepairRun, setQualityRepairRun] = useState<LibraryQualityRepairRun | null>(null)
  const [qualityRepairAdvancing, setQualityRepairAdvancing] = useState(false)
  const [qualityRepairHistory, setQualityRepairHistory] = useState<Record<string, QualityRepairHistoryRecord>>(() => loadQualityRepairHistory())
  const [qualityHistoryFocusNames, setQualityHistoryFocusNames] = useState<string[]>([])
  const [qualityArtifactOpening, setQualityArtifactOpening] = useState('')
  const [qualityCaseActionKey, setQualityCaseActionKey] = useState('')
  const [qualityFullChainActionKey, setQualityFullChainActionKey] = useState('')
  const [qualityFullChainResults, setQualityFullChainResults] = useState<Record<string, QualityFullChainActionResult>>({})
  const [qualityCaseRerunResults, setQualityCaseRerunResults] = useState<Record<string, LibraryResearchQaRerunResponse>>({})
  const [qualityFailureFilter, setQualityFailureFilter] = useState('')
  const qualityRepairBaselinesRef = useRef<Record<string, QualityRepairBaseline>>({})
  const [batchDraft, setBatchDraft] = useState<LibraryBatchMetaDraft>({
    apply_paper_category: false,
    paper_category: '',
    apply_reading_status: false,
    reading_status: '',
    add_tags: [],
    remove_tags: [],
  })

  const [pdfDirDraft, setPdfDirDraft] = useState('')
  const [mdDirDraft, setMdDirDraft] = useState('')
  const [savingDirs, setSavingDirs] = useState(false)
  const [pickingDir, setPickingDir] = useState<'pdf' | 'md' | null>(null)
  const [dirTouched, setDirTouched] = useState(false)
  const [dirEditorOpen, setDirEditorOpen] = useState(false)

  const [uploadDrafts, setUploadDrafts] = useState<UploadDraft[]>([])
  const [uploadUseLlm, setUploadUseLlm] = useState(true)
  const [uploadDraftFilter, setUploadDraftFilter] = useState<UploadDraftFilter>('all')
  const [uploadErrorReason, setUploadErrorReason] = useState<UploadErrorReason>('all')
  const [uploadInspecting, setUploadInspecting] = useState(false)
  const [uploadSaving, setUploadSaving] = useState(false)
  const [uploadWorkbenchOpen, setUploadWorkbenchOpen] = useState(false)
  const autoInspectingRef = useRef(false)

  const [renameScope, setRenameScope] = useState('30')
  const [renameLoading, setRenameLoading] = useState(false)
  const [renameApplying, setRenameApplying] = useState(false)
  const [renameItems, setRenameItems] = useState<RenameSuggestionItem[]>([])
  const [renameSelected, setRenameSelected] = useState<Record<string, boolean>>({})
  const [renameOverrides, setRenameOverrides] = useState<Record<string, string>>({})
  const [renameResultsOpen, setRenameResultsOpen] = useState(false)
  const [suggestionsRefreshing, setSuggestionsRefreshing] = useState(false)

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
  const qualityReviewCount = useMemo(() => store.files.filter((x) => hasConversionQualityIssue(x)).length, [store.files])
  const qualityReadyCount = useMemo(
    () => store.files.filter((x) => conversionQualityStatus(x.conversion_quality) === 'good').length,
    [store.files],
  )
  const backendQualityOverview = store.qualityOverview?.ok ? store.qualityOverview : null
  const latestBackendRepairRun = backendQualityOverview?.repair_runs?.[0] || null
  useEffect(() => {
    if (!latestBackendRepairRun) return
    setQualityRepairRun((cur) => {
      if (!cur || cur.run_id !== latestBackendRepairRun.run_id) return latestBackendRepairRun
      const curUpdated = Number(cur.updated_at || 0)
      const nextUpdated = Number(latestBackendRepairRun.updated_at || 0)
      if (nextUpdated > curUpdated) return latestBackendRepairRun
      const curState = `${normalizeTextValue(cur.status)}:${normalizeTextValue(cur.phase)}:${String(cur.reindexed)}`
      const nextState = `${normalizeTextValue(latestBackendRepairRun.status)}:${normalizeTextValue(latestBackendRepairRun.phase)}:${String(latestBackendRepairRun.reindexed)}`
      return curState === nextState ? cur : latestBackendRepairRun
    })
  }, [latestBackendRepairRun])
  const fallbackQualityReportStats = useMemo(() => {
    const assessed = store.files.filter((item) => item.conversion_quality)
    const convertedWithoutQuality = store.files.filter((item) => item.category === 'converted' && !item.conversion_quality).length
    const scores = assessed
      .map((item) => conversionQualityScore(item.conversion_quality))
      .filter((score) => Number.isFinite(score) && score > 0)
    const avgScore = scores.length > 0
      ? Math.round(scores.reduce((acc, score) => acc + score, 0) / scores.length)
      : 0
    return {
      assessed: assessed.length,
      converted: convertedFiles.length,
      review: qualityReviewCount,
      good: qualityReadyCount,
      unknown: convertedWithoutQuality,
      avgScore,
    }
  }, [convertedFiles.length, qualityReadyCount, qualityReviewCount, store.files])
  const qualityReportStats = useMemo(() => {
    const summary = backendQualityOverview?.summary
    if (!summary) return fallbackQualityReportStats
    return {
      assessed: Number(summary.assessed || 0),
      converted: Number(summary.converted || 0),
      review: Number(summary.review || 0),
      good: Number(summary.good || 0),
      unknown: Number(summary.unknown || 0),
      avgScore: Number(summary.avg_score || 0),
    }
  }, [backendQualityOverview, fallbackQualityReportStats])
  const fallbackQualityIssueStats = useMemo<QualityIssueStat[]>(() => {
    const stats = new Map<string, QualityIssueStat>()
    for (const item of store.files) {
      const seenInPaper = new Set<string>()
      for (const issue of item.conversion_quality?.issues || []) {
        const label = normalizeTextValue(issue.label || issue.code)
        const key = normalizeTextValue(issue.code || issue.label).toLowerCase()
        if (!key || !label) continue
        const existing = stats.get(key) || {
          key,
          label,
          severity: String(issue.severity || '').trim().toLowerCase(),
          papers: 0,
          count: 0,
        }
        existing.count += Math.max(1, Math.round(Number(issue.count || 0) || 1))
        if (!seenInPaper.has(key)) {
          existing.papers += 1
          seenInPaper.add(key)
        }
        if (String(issue.severity || '').trim().toLowerCase() === 'error') existing.severity = 'error'
        stats.set(key, existing)
      }
    }
    const severityWeight = (severity: string) => (severity === 'error' ? 2 : severity === 'warning' ? 1 : 0)
    return Array.from(stats.values())
      .sort((a, b) => severityWeight(b.severity) - severityWeight(a.severity)
        || b.papers - a.papers
        || b.count - a.count
        || a.label.localeCompare(b.label, 'en'))
      .slice(0, 5)
  }, [store.files])
  const qualityIssueStats = useMemo<QualityIssueStat[]>(() => {
    const issues = Array.isArray(backendQualityOverview?.top_issues) ? backendQualityOverview.top_issues : []
    if (!issues.length) return fallbackQualityIssueStats
    return issues.slice(0, 5).map((issue) => ({
      key: normalizeTextValue(issue.code || issue.label).toLowerCase(),
      label: normalizeTextValue(issue.label || issue.code),
      severity: normalizeTextValue(issue.severity || 'warning').toLowerCase(),
      papers: Number(issue.papers || 0),
      count: Number(issue.count || 0),
      repairStrategy: normalizeTextValue(issue.repair_strategy),
    })).filter((issue) => Boolean(issue.key && issue.label))
  }, [backendQualityOverview, fallbackQualityIssueStats])
  const qualityRepairHistoryList = useMemo(
    () => Object.values(qualityRepairHistory).sort((a, b) => b.updatedAt - a.updatedAt),
    [qualityRepairHistory],
  )
  const qualityRepairHistoryStats = useMemo(() => {
    const total = qualityRepairHistoryList.length
    const fixedCount = qualityRepairHistoryList.reduce((acc, item) => acc + item.fixedIssues.length, 0)
    const improved = qualityRepairHistoryList.filter((item) => item.afterScore > item.beforeScore).length
    const avgDelta = total > 0
      ? Math.round(qualityRepairHistoryList.reduce((acc, item) => acc + (item.afterScore - item.beforeScore), 0) / total)
      : 0
    return { total, fixedCount, improved, avgDelta }
  }, [qualityRepairHistoryList])
  const qualityHistoryFocusSet = useMemo(
    () => new Set(qualityHistoryFocusNames.map((name) => String(name || '').trim()).filter(Boolean)),
    [qualityHistoryFocusNames],
  )
  const qualityHistoryRemainingNames = useMemo(() => {
    const availableNames = new Set(store.files.map((item) => item.name))
    return qualityRepairHistoryList
      .filter((record) => record.remainingIssues.length > 0 && availableNames.has(record.name))
      .map((record) => record.name)
  }, [qualityRepairHistoryList, store.files])
  const localQualityRepairRecommendedItems = useMemo(() => (
    store.files
      .filter((item) => item.task_state === 'idle' && hasConversionQualityIssue(item))
      .sort((a, b) => {
        const aHistory = qualityRepairHistory[a.name]
        const bHistory = qualityRepairHistory[b.name]
        const aRemaining = aHistory?.remainingIssues.length || 0
        const bRemaining = bHistory?.remainingIssues.length || 0
        const aScore = conversionQualityScore(a.conversion_quality)
        const bScore = conversionQualityScore(b.conversion_quality)
        return bRemaining - aRemaining
          || aScore - bScore
          || String(a.name || '').localeCompare(String(b.name || ''), 'en')
      })
      .slice(0, 5)
  ), [qualityRepairHistory, store.files])
  const qualityReportRecommendations = useMemo<QualityReportRecommendationView[]>(() => {
    const overviewItems = Array.isArray(backendQualityOverview?.recommended) ? backendQualityOverview.recommended : []
    if (overviewItems.length > 0) {
      return overviewItems.slice(0, 5)
        .map((item) => ({
          name: normalizeTextValue(item.name),
          score: Math.max(0, Math.min(100, Math.round(Number(item.score || 0)))),
          issues: (Array.isArray(item.issues) ? item.issues : [])
            .map((issue) => normalizeTextValue(issue.label || issue.code))
            .filter(Boolean)
            .slice(0, 2),
        }))
        .filter((item) => Boolean(item.name))
    }
    return localQualityRepairRecommendedItems.slice(0, 5).map((item) => ({
      name: item.name,
      score: conversionQualityScore(item.conversion_quality),
      issues: conversionQualityIssueEntries(item.conversion_quality).map((issue) => issue.label).slice(0, 2),
    }))
  }, [backendQualityOverview, localQualityRepairRecommendedItems])
  const qualityDomainViews = useMemo<QualityDomainView[]>(() => {
    const domains = backendQualityOverview?.domains || {}
    const conversion = domains.conversion
    const researchQa = domains.research_qa
    const citationCards = domains.citation_cards

    const conversionStatus = qualityDomainStatus(conversion, backendQualityOverview?.status || 'unknown')
    const conversionReview = conversion ? qualityDomainNumber(conversion, 'review') : qualityReportStats.review
    const conversionGood = conversion ? qualityDomainNumber(conversion, 'good') : qualityReportStats.good
    const conversionAvg = conversion ? qualityDomainNumber(conversion, 'avg_score') : qualityReportStats.avgScore
    const conversionUnknown = conversion ? qualityDomainNumber(conversion, 'unknown') : qualityReportStats.unknown

    const qaStatus = qualityDomainStatus(researchQa)
    const qaAvailable = researchQa?.available !== false && Boolean(researchQa)
    const qaTotal = qualityDomainNumber(researchQa, 'total')
    const qaPassed = qualityDomainNumber(researchQa, 'passed')
    const qaFailed = qualityDomainNumber(researchQa, 'failed')

    const cardStatus = qualityDomainStatus(citationCards)
    const cardsAvailable = citationCards?.available !== false && Boolean(citationCards)
    const trackedChecks = qualityDomainNumber(citationCards, 'tracked_checks')
    const failedChecks = qualityDomainNumber(citationCards, 'failed_checks')

    return [
      {
        key: 'conversion',
        label: S.lib_quality_domain_conversion,
        available: true,
        status: conversionStatus,
        statusLabel: qualityStatusText(conversionStatus, S),
        countText: conversionReview > 0
          ? `${conversionReview} ${S.lib_quality_domain_failed}`
          : `${conversionGood} ${S.lib_quality_domain_passed}`,
        detailText: `Q${Math.round(conversionAvg)} · ${conversionUnknown} ${S.lib_quality_report_unknown}`,
        failureText: qualityTopFailureText(conversion),
      },
      {
        key: 'research_qa',
        label: S.lib_quality_domain_research_qa,
        available: qaAvailable,
        status: qaAvailable ? qaStatus : 'unknown',
        statusLabel: qaAvailable ? qualityStatusText(qaStatus, S) : S.lib_quality_domain_unavailable,
        countText: qaAvailable
          ? (qaFailed > 0 ? `${qaFailed} ${S.lib_quality_domain_failed}` : `${qaPassed}/${qaTotal} ${S.lib_quality_domain_passed}`)
          : S.lib_quality_domain_unavailable,
        detailText: qaAvailable ? S.lib_quality_domain_cases.replace('{n}', String(qaTotal)) : '',
        failureText: qualityTopFailureText(researchQa),
      },
      {
        key: 'citation_cards',
        label: S.lib_quality_domain_citation_cards,
        available: cardsAvailable,
        status: cardsAvailable ? cardStatus : 'unknown',
        statusLabel: cardsAvailable ? qualityStatusText(cardStatus, S) : S.lib_quality_domain_unavailable,
        countText: cardsAvailable
          ? (failedChecks > 0 ? `${failedChecks} ${S.lib_quality_domain_failed}` : `${trackedChecks} ${S.lib_quality_domain_passed}`)
          : S.lib_quality_domain_unavailable,
        detailText: cardsAvailable ? S.lib_quality_domain_checks.replace('{n}', String(trackedChecks)) : '',
        failureText: qualityTopFailureText(citationCards),
      },
    ]
  }, [backendQualityOverview, qualityReportStats, S])
  const qualityPriorityActions = useMemo<LibraryQualityPriorityAction[]>(
    () => (Array.isArray(backendQualityOverview?.priority_actions) ? backendQualityOverview.priority_actions : [])
      .filter((item) => item && normalizeTextValue(item.domain))
      .slice(0, 4),
    [backendQualityOverview],
  )
  const qualityFullChain = useMemo<LibraryQualityFullChain | null>(() => {
    const fullChain = backendQualityOverview?.full_chain
    if (!fullChain || fullChain.available === false) return null
    return fullChain
  }, [backendQualityOverview])
  const qualityFullChainStages = useMemo(
    () => (Array.isArray(qualityFullChain?.stages) ? qualityFullChain.stages : [])
      .filter((stage) => stage && normalizeTextValue(stage.key))
      .slice(0, 6),
    [qualityFullChain],
  )
  const qualityFullChainRootCauses = useMemo(
    () => (Array.isArray(qualityFullChain?.root_causes) ? qualityFullChain.root_causes : [])
      .filter((cause) => cause && normalizeTextValue(cause.code || cause.label))
      .slice(0, 5),
    [qualityFullChain],
  )
  const qualityFullChainActionHistory = useMemo<LibraryQualityActionHistoryItem[]>(
    () => (Array.isArray(qualityFullChain?.action_history) ? qualityFullChain.action_history : [])
      .filter((item) => item && normalizeTextValue(item.stage_key) && normalizeTextValue(item.summary))
      .slice(0, 8),
    [qualityFullChain],
  )
  const qualityFullChainPersistedResults = useMemo<Record<string, QualityFullChainActionResult>>(() => {
    const out: Record<string, QualityFullChainActionResult> = {}
    for (const item of qualityFullChainActionHistory) {
      const key = normalizeTextValue(item.stage_key).toLowerCase()
      if (!key || out[key]) continue
      const status = normalizeTextValue(item.status).toLowerCase()
      out[key] = {
        status: status === 'success' || status === 'warning' || status === 'error' ? status : 'info',
        summary: normalizeTextValue(item.summary),
        detail: normalizeTextValue(item.detail),
        deltaText: qualityActionDeltaText(item),
        verificationText: qualityVerificationText(item.verification),
        improved: typeof item.improved === 'boolean' ? item.improved : item.delta?.improved,
        updatedAt: Number(item.created_at || 0) * 1000,
      }
    }
    return out
  }, [qualityFullChainActionHistory])
  const qualityFeatureHealth = useMemo<LibraryQualityFeatureHealth | null>(() => {
    const featureHealth = backendQualityOverview?.feature_health
    if (!featureHealth || featureHealth.available === false) return null
    return featureHealth
  }, [backendQualityOverview])
  const qualityFeatureHealthItems = useMemo<LibraryQualityFeatureHealthItem[]>(
    () => (Array.isArray(qualityFeatureHealth?.items) ? qualityFeatureHealth.items : [])
      .filter((item) => item && normalizeTextValue(item.key))
      .slice(0, 8),
    [qualityFeatureHealth],
  )
  const qualityRerunSummary = backendQualityOverview?.rerun_summary
  const qualityFailureCases = useMemo<LibraryQualityFailureCase[]>(
    () => (Array.isArray(backendQualityOverview?.failure_cases) ? backendQualityOverview.failure_cases : [])
      .filter((item) => item && normalizeTextValue(item.id))
      .slice(0, 12),
    [backendQualityOverview],
  )
  const qualityFailureFilters = useMemo(() => {
    const stats = new Map<string, number>()
    for (const item of qualityFailureCases) {
      for (const failure of item.failures || []) {
        const name = normalizeTextValue(failure.name)
        if (!name) continue
        stats.set(name, (stats.get(name) || 0) + 1)
      }
    }
    return Array.from(stats.entries())
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0], 'en'))
      .slice(0, 6)
      .map(([name, count]) => ({ name, count }))
  }, [qualityFailureCases])
  const visibleQualityFailureCases = useMemo(() => {
    const filter = normalizeTextValue(qualityFailureFilter)
    if (!filter) return qualityFailureCases
    return qualityFailureCases.filter((item) => (item.failures || []).some((failure) => normalizeTextValue(failure.name) === filter))
  }, [qualityFailureCases, qualityFailureFilter])
  const qualityRepairRecommendedNames = useMemo(
    () => qualityReportRecommendations.map((item) => item.name).filter(Boolean),
    [qualityReportRecommendations],
  )
  const renameOnlyDiff = true
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
      .map(([key, count]) => ({ key, count, label: FAILED_REASON_META(S)[key].label }))
      .sort((a, b) => b.count - a.count)
  }, [failedUploadDrafts, S])
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
  const activeErrorReasonText = useMemo(() => {
    const map: Record<UploadErrorReason, string> = {
      all: S.lib_error_filter_all,
      duplicate: FAILED_REASON_META(S).duplicate.label,
      path: FAILED_REASON_META(S).path.label,
      permission: FAILED_REASON_META(S).permission.label,
      network: FAILED_REASON_META(S).network.label,
      other: FAILED_REASON_META(S).other.label,
    }
    return map[uploadErrorReason]
  }, [uploadErrorReason, S])
  const convertPercent = useMemo(() => {
    if (!store.progress || store.progress.total <= 0) return 0
    const tasks = Array.isArray(store.progress.activeTasks) ? store.progress.activeTasks : []
    let activeFraction = 0
    if (tasks.length > 0) {
      for (const task of tasks) {
        const taskProgress = derivePageProgress(task.cur_page_done, task.cur_page_total, task.cur_page_msg)
        if (taskProgress.total <= 0) continue
        activeFraction += Math.min(0.999, taskProgress.done / Math.max(1, taskProgress.total))
      }
    } else {
      const fallback = derivePageProgress(
        store.progress.curPageDone,
        store.progress.curPageTotal,
        store.progress.curPageMsg,
      )
      if (fallback.total > 0) {
        activeFraction = Math.min(0.999, fallback.done / Math.max(1, fallback.total))
      }
    }
    const rawPercent = Math.min(100, Math.round(((store.progress.completed + activeFraction) / Math.max(1, store.progress.total)) * 100))
    const stillRunning = store.progress.completed < store.progress.total
      && (
        Number(store.progress.activeCount || 0) > 0
        || tasks.length > 0
        || Boolean(String(store.progress.current || '').trim())
      )
    return stillRunning ? Math.min(rawPercent, 99) : rawPercent
  }, [store.progress])
  const convertPageProgress = useMemo(() => (
    derivePageProgress(
      Number(store.progress?.curPageDone || 0),
      Number(store.progress?.curPageTotal || 0),
      String(store.progress?.curPageMsg || ''),
    )
  ), [store.progress])
  const convertPagePercent = useMemo(
    () => (convertPageProgress.total > 0
      ? Math.round((convertPageProgress.done / Math.max(1, convertPageProgress.total)) * 100)
      : 0),
    [convertPageProgress],
  )
  const convertActiveSummary = useMemo(() => {
    const tasks = Array.isArray(store.progress?.activeTasks) ? store.progress.activeTasks : []
    if (!tasks.length) return ''
    const names = tasks.map((task) => String(task.name || '').trim()).filter(Boolean)
    if (!names.length) return ''
    const preview = names.slice(0, 3).join('\u3001')
    const suffix = names.length > 3 ? ' ...' : ''
    return `\u5e76\u884c\u4e2d ${tasks.length} \u7bc7\uff1a${preview}${suffix}`
  }, [store.progress])
  const convertStageLabel = useMemo(
    () => deriveConvertStageLabel(String(store.progress?.curPageMsg || ''), S),
    [store.progress, S],
  )
  const refSyncPercent = useMemo(
    () => (store.refSync && store.refSync.docsTotal > 0
      ? Math.round((store.refSync.docsDone / Math.max(1, store.refSync.docsTotal)) * 100)
      : 0),
    [store.refSync],
  )
  const showStickyStatus = Boolean((store.converting && store.progress) || store.refSync?.running)

  const paperCategoryFilterOptions = useMemo(() => {
    const values = uniqueTextValues(store.files.map((item) => item.paper_category))
      .sort((a, b) => a.localeCompare(b, 'en'))
    return toTextOptions(values)
  }, [store.files])

  const paperCategoryOptions = useMemo(() => {
    const presetValues = Array.from(PAPER_CATEGORY_PRESETS)
    const dynamicValues = uniqueTextValues([
      ...store.files.map((item) => item.paper_category),
      ...store.files.map((item) => item.suggested_category),
    ]).sort((a, b) => a.localeCompare(b, 'en'))
    const presetKeys = new Set(presetValues.map((value) => value.toLowerCase()))
    const merged = [
      ...presetValues,
      ...dynamicValues.filter((value) => !presetKeys.has(value.toLowerCase())),
    ]
    return toTextOptions(merged)
  }, [store.files])

  const paperTagFilterOptions = useMemo(() => {
    const values = uniqueTextValues(store.files.flatMap((item) => item.user_tags || []))
      .sort((a, b) => a.localeCompare(b, 'en'))
    return toTextOptions(values)
  }, [store.files])

  const paperTagOptions = useMemo(() => {
    const values = uniqueTextValues([
      ...store.files.flatMap((item) => item.user_tags || []),
      ...store.files.flatMap((item) => item.suggested_tags || []),
    ]).sort((a, b) => a.localeCompare(b, 'en'))
    return toTextOptions(values)
  }, [store.files])

  const applyPaperCategoryFilter = (value: string) => {
    setOnlyUnclassified(false)
    setPaperCategoryFilter(value)
  }

  const applyPaperTagFilter = (value: string) => {
    setPaperTagFilter(value)
  }

  const clearTaxonomyFilters = () => {
    setFileKeyword('')
    setPaperCategoryFilter('')
    setPaperTagFilter('')
    setReadingStatusFilter('')
    setOnlyUnread(false)
    setOnlyUnclassified(false)
    setOnlySuggested(false)
    setOnlyQualityIssues(false)
    setQualityHistoryFocusNames([])
  }

  const hasActiveTaxonomyFilters = Boolean(
    normalizedKeyword
    || paperCategoryFilter
    || paperTagFilter
    || readingStatusFilter
    || onlyUnread
    || onlyUnclassified
    || onlySuggested
    || onlyQualityIssues
    || qualityHistoryFocusNames.length > 0
  )
  const activeTaxonomyFilterCount = [
    normalizedKeyword,
    paperCategoryFilter,
    paperTagFilter,
    readingStatusFilter,
    onlyUnread ? 'onlyUnread' : '',
    onlyUnclassified ? 'onlyUnclassified' : '',
    onlySuggested ? 'onlySuggested' : '',
    onlyQualityIssues ? 'onlyQualityIssues' : '',
    qualityHistoryFocusNames.length > 0 ? 'qualityHistoryFocus' : '',
  ].filter(Boolean).length

  const filterFiles = useCallback(
    (items: LibraryFileItem[], options: FilterFilesOptions = {}) =>
      items.filter((item) => {
        if (qualityHistoryFocusSet.size > 0 && !qualityHistoryFocusSet.has(item.name)) return false
        const keywordText = [
          item.name,
          item.paper_category,
          item.reading_status,
          item.note,
          item.suggested_category,
          item.conversion_quality?.label,
          item.conversion_quality?.summary,
          ...(item.conversion_quality?.issues || []).flatMap((issue) => [issue.code, issue.label]),
          ...(item.user_tags || []),
          ...(item.suggested_tags || []),
        ]
          .map((part) => String(part || '').toLowerCase())
          .join(' ')
        if (!matchesKeyword(keywordText, normalizedKeyword)) return false
        if (!options.ignoreCategoryFilter && paperCategoryFilter && String(item.paper_category || '') !== paperCategoryFilter) return false
        if (!options.ignoreTagFilter && paperTagFilter && !(item.user_tags || []).some((tag) => String(tag || '').toLowerCase() === paperTagFilter.toLowerCase())) return false
        if (readingStatusFilter && String(item.reading_status || '') !== readingStatusFilter) return false
        if (onlyUnread && String(item.reading_status || '') !== 'unread') return false
        if (onlyUnclassified && String(item.paper_category || '').trim()) return false
        if (onlySuggested && !item.has_suggestions) return false
        if (onlyQualityIssues && !hasConversionQualityIssue(item)) return false
        return true
      }),
    [normalizedKeyword, onlyQualityIssues, onlySuggested, onlyUnclassified, onlyUnread, paperCategoryFilter, paperTagFilter, qualityHistoryFocusSet, readingStatusFilter],
  )

  const visiblePending = useMemo(
    () => filterFiles(pendingFiles),
    [filterFiles, pendingFiles],
  )
  const visibleConverted = useMemo(
    () => filterFiles(convertedFiles),
    [convertedFiles, filterFiles],
  )
  const visibleAll = useMemo(
    () => filterFiles(store.files),
    [filterFiles, store.files],
  )
  const visibleAllWithoutCategory = useMemo(
    () => filterFiles(store.files, { ignoreCategoryFilter: true }),
    [filterFiles, store.files],
  )
  const visibleAllWithoutTag = useMemo(
    () => filterFiles(store.files, { ignoreTagFilter: true }),
    [filterFiles, store.files],
  )

  const categoryCards = useMemo<CategoryCardItem[]>(() => {
    const groups = new Map<string, LibraryFileItem[]>()
    for (const item of visibleAllWithoutCategory) {
      const rawLabel = String(item.paper_category || '').trim()
      const key = rawLabel ? `category:${rawLabel}` : 'category:__unclassified__'
      const list = groups.get(key)
      if (list) {
        list.push(item)
      } else {
        groups.set(key, [item])
      }
    }

    const out: CategoryCardItem[] = []
    for (const [key, items] of groups.entries()) {
      const label = key === 'category:__unclassified__'
        ? S.lib_category_unclassified
        : String(items[0]?.paper_category || '').trim()
      const tagCounts = new Map<string, number>()
      for (const item of items) {
        for (const tag of item.user_tags || []) {
          const value = String(tag || '').trim()
          if (!value) continue
          tagCounts.set(value, (tagCounts.get(value) || 0) + 1)
        }
      }
      const commonTags = Array.from(tagCounts.entries())
        .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0], 'en'))
        .slice(0, 4)
        .map(([tag]) => tag)
      out.push({
        key,
        label: label || S.lib_category_unclassified,
        count: items.length,
        unreadCount: items.filter((item) => item.reading_status === 'unread').length,
        convertedCount: items.filter((item) => item.category === 'converted').length,
        pendingCount: items.filter((item) => item.category === 'pending').length,
        commonTags,
        recentPapers: items.slice(0, 3).map((item) => item.name),
      })
    }

    return out.sort((a, b) => b.count - a.count || a.label.localeCompare(b.label, 'en'))
  }, [visibleAllWithoutCategory, S])

  const tagCards = useMemo<TagCardItem[]>(() => {
    const groups = new Map<string, LibraryFileItem[]>()
    for (const item of visibleAllWithoutTag) {
      for (const rawTag of item.user_tags || []) {
        const label = String(rawTag || '').trim()
        if (!label) continue
        const key = label.toLowerCase()
        const list = groups.get(key)
        if (list) {
          list.push(item)
        } else {
          groups.set(key, [item])
        }
      }
    }

    const out: TagCardItem[] = []
    for (const [key, items] of groups.entries()) {
      const label = items.find((item) => (item.user_tags || []).some((tag) => String(tag || '').trim().toLowerCase() === key))
        ?.user_tags.find((tag) => String(tag || '').trim().toLowerCase() === key) || key
      const categoryCounts = new Map<string, number>()
      for (const item of items) {
        const category = String(item.paper_category || '').trim() || S.lib_category_unclassified
        categoryCounts.set(category, (categoryCounts.get(category) || 0) + 1)
      }
      out.push({
        key,
        label: String(label),
        count: items.length,
        unreadCount: items.filter((item) => item.reading_status === 'unread').length,
        categories: Array.from(categoryCounts.entries())
          .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0], 'en'))
          .slice(0, 3)
          .map(([category]) => category),
        recentPapers: items.slice(0, 3).map((item) => item.name),
      })
    }

    return out.sort((a, b) => b.count - a.count || a.label.localeCompare(b.label, 'en'))
  }, [visibleAllWithoutTag, S])

  const currentListItems = useMemo(() => {
    if (tabKey === 'pending') return visiblePending
    if (tabKey === 'converted') return visibleConverted
    return visibleAll
  }, [tabKey, visiblePending, visibleConverted, visibleAll])

  const selectedLibraryNamesList = useMemo(
    () => Object.keys(selectedLibraryNames).filter((name) => Boolean(selectedLibraryNames[name])),
    [selectedLibraryNames],
  )

  const selectedLibraryCount = selectedLibraryNamesList.length
  const selectedQualityReviewNames = useMemo(
    () => store.files
      .filter((item) => Boolean(selectedLibraryNames[item.name]) && hasConversionQualityIssue(item) && item.task_state === 'idle')
      .map((item) => item.name),
    [store.files, selectedLibraryNames],
  )
  const metaSuggestionCount = (metaItem?.suggested_category ? 1 : 0) + (metaItem?.suggested_tags?.length || 0)
  const metaDraftCategory = normalizeTextValue(metaDraft.paper_category)
  const metaDraftTags = normalizeTextList(metaDraft.user_tags)

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

  useEffect(() => {
    if (!settingsLoaded) return
    if (!String(settingsPdfDir || '').trim() || !String(settingsMdDir || '').trim()) {
      setDirEditorOpen(true)
    }
  }, [settingsLoaded, settingsPdfDir, settingsMdDir])

  useEffect(() => {
    if (uploadDrafts.length === 0) {
      setUploadWorkbenchOpen(false)
      return
    }
    setUploadWorkbenchOpen(true)
  }, [uploadDrafts.length])

  useEffect(() => {
    const existing = new Set(store.files.map((item) => item.name))
    setSelectedLibraryNames((cur) => {
      let changed = false
      const next: Record<string, boolean> = {}
      for (const [name, selected] of Object.entries(cur)) {
        if (!selected) continue
        if (!existing.has(name)) {
          changed = true
          continue
        }
        next[name] = true
      }
      return changed ? next : cur
    })
  }, [store.files])

  useEffect(() => {
    const pending = qualityRepairBaselinesRef.current
    const pendingNames = Object.keys(pending)
    if (!pendingNames.length) return
    const nextPending = { ...pending }
    const nextResults: Record<string, string> = {}
    const nextHistory: Record<string, QualityRepairHistoryRecord> = {}
    for (const item of store.files) {
      const baseline = pending[item.name]
      if (!baseline) continue
      if (item.task_state !== 'idle' || !item.conversion_quality) continue
      nextResults[item.name] = summarizeConversionQualityRepair(baseline.quality, item.conversion_quality, S)
      nextHistory[item.name] = buildQualityRepairHistoryRecord(item.name, baseline.quality, item.conversion_quality)
      delete nextPending[item.name]
    }
    if (Object.keys(nextResults).length <= 0) return
    qualityRepairBaselinesRef.current = nextPending
    setQualityRepairResults((cur) => ({ ...cur, ...nextResults }))
    setQualityRepairHistory((cur) => {
      const merged = normalizeQualityRepairHistory({ ...cur, ...nextHistory })
      saveQualityRepairHistory(merged)
      return merged
    })
  }, [store.files, S])

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
      await store.loadFiles(scope)
      return true
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.lib_msg_save_dir_fail)
      return false
    } finally {
      setSavingDirs(false)
    }
  }, [mdDirDraft, pdfDirDraft, scope, store, updateSettings, S])

  const ensureDirsReady = useCallback(async () => {
    if (!dirDirty) return true
    return saveDirs()
  }, [dirDirty, saveDirs])

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
  }

  const inspectDraft = useCallback(async (key: string) => {
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
          suggestionBasisLabel: String(res.meta?.basis_label || ''),
          suggestionBasisDetail: String(res.meta?.basis_detail || ''),
          suggestionMatchMethod: String(res.meta?.match_method || ''),
          suggestionYearSource: String(res.meta?.year_source || ''),
          status: res.duplicate ? 'error' : 'ready',
          note: res.duplicate ? `${S.lib_upload_dup_prefix}${String(res.existing || '')}` : S.lib_upload_scan_done,
        }
      }))
    } catch (err) {
      setUploadDrafts((cur) => cur.map((x) => (
        x.key === key
          ? { ...x, status: 'error', note: err instanceof Error ? err.message : S.lib_upload_scan_fail }
          : x
      )))
    }
  }, [ensureDirsReady, uploadDrafts, uploadUseLlm, S])

  const inspectSelectedDrafts = async () => {
    const selected = uploadDrafts.filter((x) => x.selected && x.status !== 'inspecting')
    if (!selected.length) {
      message.info(S.lib_msg_select_scan)
      return
    }
    setUploadInspecting(true)
    try {
      for (const x of selected) {
        await inspectDraft(x.key)
      }
      message.success(S.lib_msg_scanned_count.replace('{n}', String(selected.length)))
    } finally {
      setUploadInspecting(false)
    }
  }

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
        for (const key of queuedKeys) {
          // Auto-fill suggested names for newly added upload drafts.
          await inspectDraft(key)
        }
      } finally {
        autoInspectingRef.current = false
        setUploadInspecting(false)
      }
    })()
  }, [dirDirty, inspectDraft, uploadDrafts, uploadInspecting, uploadLocked])

  useEffect(() => {
    setUploadDrafts((cur) => {
      const next = cur.filter((draft) => !isUploadDraftConverted(draft, store.files))
      return next.length === cur.length ? cur : next
    })
  }, [store.files])

  const saveDraft = async (key: string, convertNow: boolean, opts?: { syncUi?: boolean }) => {
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
        baseName: target.stem,
        convertNow,
        speedMode: CONVERT_MODE,
        allowDuplicate: false,
      })
      const savedName = String(res.name || target.file.name || '')
      const enqueued = Boolean(convertNow && res.enqueued)
      setUploadDrafts((cur) => cur.map((x) => {
        if (x.key !== key) return x
        if (res.duplicate) return { ...x, status: 'error', note: `${S.lib_upload_dup_prefix}${String(res.existing || '')}` }
        return {
          ...x,
          status: 'saved',
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
        await store.loadFiles(scope)
        if (enqueued) store.startProgressStream()
      }
      return { saved: true, enqueued }
    } catch (err) {
      setUploadDrafts((cur) => cur.map((x) => (
        x.key === key
          ? { ...x, status: 'error', note: err instanceof Error ? err.message : S.lib_upload_save_fail }
          : x
      )))
      return { saved: false, enqueued: false }
    }
  }

  const saveSelectedDrafts = async (convertNow: boolean) => {
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
      await store.loadFiles(scope)
      if (anyEnqueued) store.startProgressStream()
      message.success(S.lib_msg_processed_count.replace('{n}', String(selected.length)))
    } finally {
      setUploadSaving(false)
    }
  }

  const scanRenameSuggestions = async () => {
    setRenameLoading(true)
    try {
      const res = await libraryApi.listRenameSuggestions(renameScope, true)
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
  }

  const selectFailedDrafts = () => {
    if (!failedUploadDrafts.length) {
      message.info(S.lib_msg_no_failed_items)
      return
    }
    setUploadDrafts((cur) => cur.map((x) => ({ ...x, selected: x.status === 'error' })))
    message.info(S.lib_msg_selected_failed.replace('{n}', String(failedUploadDrafts.length)))
  }

  const showDuplicateFailedDrafts = () => {
    if (!duplicateFailedDrafts.length) {
      message.info(S.lib_msg_no_dup_failures)
      return
    }
    applyUploadFilter('dup_error')
    message.info(S.lib_msg_switched_dup.replace('{n}', String(duplicateFailedDrafts.length)))
  }

  const retryFailedDrafts = async (convertNow: boolean) => {
    const failed = uploadDrafts.filter((x) => x.status === 'error')
    if (!failed.length) {
      message.info(S.lib_msg_no_retryable)
      return
    }
    setUploadSaving(true)
    try {
      let anyEnqueued = false
      for (const x of failed) {
        const result = await saveDraft(x.key, convertNow, { syncUi: false })
        anyEnqueued = anyEnqueued || Boolean(result.enqueued)
      }
      await store.loadFiles(scope)
      if (anyEnqueued) store.startProgressStream()
      message.success(S.lib_msg_retried_count.replace('{n}', String(failed.length)))
    } finally {
      setUploadSaving(false)
    }
  }

  const applyRenameSuggestions = async () => {
    const names = renameItems.filter((x) => renameSelected[x.name]).map((x) => x.name)
    if (!names.length) {
      message.info(S.lib_msg_select_rename)
      return
    }
    setRenameApplying(true)
    try {
      const overrides: Record<string, string> = {}
      for (const name of names) overrides[name] = String(renameOverrides[name] || '').trim()
      const res = await libraryApi.applyRenameSuggestions(names, overrides, { useLlm: true, alsoMd: true })
      message[res.failed > 0 ? 'warning' : 'success'](S.lib_msg_rename_result.replace('{ok}', String(res.renamed)).replace('{skip}', String(res.skipped)).replace('{fail}', String(res.failed)))
      if (res.needs_reindex) message.info(S.lib_msg_rename_suggest_reindex)
      await store.loadFiles(scope)
      await scanRenameSuggestions()
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.lib_msg_apply_rename_fail)
    } finally {
      setRenameApplying(false)
    }
  }

  const handleConvertPending = async () => {
    const res = await store.convertPending(CONVERT_MODE)
    message[res.enqueued > 0 ? 'success' : 'info'](
      res.enqueued > 0
        ? S.lib_msg_enqueued_count.replace('{n}', String(res.enqueued))
        : S.lib_msg_no_convertible,
    )
    await store.loadFiles(scope)
  }

  const handleConvertOne = async (item: LibraryFileItem) => {
    if (item.task_state !== 'idle') return
    await store.convert(item.name, CONVERT_MODE, true)
  }

  const recordQualityFullChainResult = (
    stageKey: string,
    result: Omit<QualityFullChainActionResult, 'updatedAt'>,
    meta: {
      stageLabel?: string
      action?: string
      targetIds?: string[]
      metrics?: Record<string, string | number | boolean | null | undefined>
      before?: LibraryQualityActionSnapshot
      after?: LibraryQualityActionSnapshot
      verification?: Record<string, string | number | boolean | null | undefined>
    } = {},
  ) => {
    const key = normalizeTextValue(stageKey).toLowerCase()
    if (!key) return
    const nowMs = Date.now()
    const hasSnapshots = Boolean(meta.before || meta.after)
    const verificationOk = meta.verification?.quality_ok === true
    const rawDelta: LibraryQualityActionDelta = hasSnapshots ? qualityBuildActionDelta(meta.before, meta.after) : {}
    const delta: LibraryQualityActionDelta = hasSnapshots && rawDelta.improved === false && verificationOk
      ? {
        ...rawDelta,
        improved: true,
        worsened: false,
        summary: 'Improved: QA rerun passed; overview unchanged',
      }
      : rawDelta
    const improved = typeof delta.improved === 'boolean' ? delta.improved : null
    const verificationText = qualityVerificationText(meta.verification)
    const resultStatus = result.status === 'success' && improved === false ? 'warning' : result.status
    setQualityFullChainResults((cur) => ({
      ...cur,
      [key]: {
        ...result,
        status: resultStatus,
        deltaText: hasSnapshots ? delta.summary : '',
        verificationText,
        improved,
        updatedAt: nowMs,
      },
    }))
    void libraryApi.recordQualityAction({
      stage_key: key,
      stage_label: normalizeTextValue(meta.stageLabel || key),
      action: normalizeTextValue(meta.action),
      status: resultStatus,
      summary: result.summary,
      detail: result.detail,
      target_ids: meta.targetIds || [],
      metrics: meta.metrics || {},
      before: meta.before,
      after: meta.after,
      delta: hasSnapshots ? delta : {},
      improved,
      verification: meta.verification || {},
      created_at: Math.floor(nowMs / 1000),
    }).catch(() => {
      // Persisting the audit trail should never block the repair flow.
    })
  }

  const repairQualityByNames = async (names: string[], opts: QualityRepairRunOptions = {}) => {
    const targets = Array.from(new Set(names.map((name) => String(name || '').trim()).filter(Boolean)))
    if (!targets.length) {
      message.info(S.lib_msg_quality_repair_none)
      return { ok: true, targetCount: 0, queued: 0, repaired: 0, needsReindex: false, reindexed: false, impact: null as LibraryQualityRepairImpact | null }
    }
    const startedAt = Date.now()
    const baselineByName = new Map(store.files.map((item) => [item.name, item.conversion_quality || null]))
    qualityRepairBaselinesRef.current = {
      ...qualityRepairBaselinesRef.current,
      ...Object.fromEntries(targets.map((name) => [name, { quality: baselineByName.get(name) || null, startedAt }])),
    }
    setQualityRepairResults((cur) => {
      const next = { ...cur }
      for (const name of targets) delete next[name]
      return next
    })
    setQualityRepairingNames((cur) => {
      const next = { ...cur }
      for (const name of targets) next[name] = true
      return next
    })
    try {
      const res = await store.repairQuality({
        pdf_names: targets,
        speed_mode: CONVERT_MODE,
        replace: true,
      }, {
        autoReindexAfterQueued: opts.autoReindexQueued !== false,
      })
      const queued = Number(res.enqueued || 0)
      const repaired = Number(res.repaired || 0)
      const impact = res.impact || null
      const needsReindex = Boolean(res.needs_reindex || impact?.needs_reindex)
      let reindexed = false
      if (res.repair_run) {
        setQualityRepairRun(res.repair_run)
      }
      if (impact) {
        setQualityRepairImpact(impact)
      }
      if (queued > 0 || repaired > 0) {
        message.success(
          queued > 0
            ? S.lib_msg_quality_repair_enqueued.replace('{n}', String(queued))
            : `Markdown repaired: ${repaired}`,
        )
      } else {
        qualityRepairBaselinesRef.current = Object.fromEntries(
          Object.entries(qualityRepairBaselinesRef.current).filter(([name]) => !targets.includes(name)),
        )
        message.info(S.lib_msg_quality_repair_none)
      }
      await store.loadFiles(scope)
      if (needsReindex && repaired > 0 && queued <= 0 && opts.autoReindexImmediate !== false) {
        reindexed = await handleReindex()
        if (impact) {
          setQualityRepairImpact({ ...impact, reindexed })
        }
        if (res.repair_run?.run_id) {
          const status = reindexed ? 'completed' : 'warning'
          const phase = reindexed ? 'reindex_complete' : 'reindex_failed'
          setQualityRepairRun({ ...res.repair_run, status, phase, reindexed })
          libraryApi.updateQualityRepairRun(res.repair_run.run_id, { status, phase, reindexed }).catch(() => {})
        }
        if (reindexed) await store.loadFiles(scope)
      }
      return { ok: true, targetCount: targets.length, queued, repaired, needsReindex, reindexed, impact }
    } catch (err) {
      qualityRepairBaselinesRef.current = Object.fromEntries(
        Object.entries(qualityRepairBaselinesRef.current).filter(([name]) => !targets.includes(name)),
      )
      message.error(err instanceof Error ? err.message : S.lib_msg_quality_repair_failed)
      return { ok: false, targetCount: targets.length, queued: 0, repaired: 0, needsReindex: false, reindexed: false, impact: null as LibraryQualityRepairImpact | null }
    } finally {
      setQualityRepairingNames((cur) => {
        const next = { ...cur }
        for (const name of targets) delete next[name]
        return next
      })
    }
  }

  const handleRepairQualityOne = async (item: LibraryFileItem) => {
    if (item.task_state !== 'idle' || !hasConversionQualityIssue(item)) return
    await repairQualityByNames([item.name])
  }

  const handleRepairSelectedQuality = async () => {
    await repairQualityByNames(selectedQualityReviewNames)
  }

  const handleFocusQualityReview = () => {
    if (qualityReviewCount <= 0) {
      message.info(S.lib_quality_report_no_issues)
      return
    }
    setQualityHistoryFocusNames([])
    setOnlyQualityIssues(true)
    setBrowseMode('list')
    setTabKey('all')
  }

  const handleFocusQualityIssue = (label: string) => {
    const keyword = String(label || '').trim()
    if (!keyword) return
    setFileKeyword(keyword)
    setQualityHistoryFocusNames([])
    setOnlyQualityIssues(true)
    setBrowseMode('list')
    setTabKey('all')
  }

  const focusQualityHistoryNames = (names: string[]) => {
    const availableNames = new Set(store.files.map((item) => item.name))
    const rawTargets = Array.from(new Set(names.map((name) => String(name || '').trim()).filter(Boolean)))
    const targets = rawTargets.filter((name) => availableNames.has(name))
    if (!targets.length) {
      if (rawTargets.length > 0) {
        setQualityHistoryFocusNames(rawTargets)
        setBrowseMode('list')
        setTabKey('all')
        if (scope !== 'all') {
          setScope('all')
          void store.loadFiles('all')
        }
        return
      }
      message.info(S.lib_quality_history_no_remaining)
      return
    }
    setQualityHistoryFocusNames(targets)
    setBrowseMode('list')
    setTabKey('all')
  }

  const handleFocusQualityHistoryRemaining = () => {
    focusQualityHistoryNames(qualityHistoryRemainingNames)
  }

  const handleRepairRecommendedQuality = async (opts: QualityRepairRunOptions = {}) => {
    if (!qualityRepairRecommendedNames.length) {
      message.info(S.lib_quality_history_no_recommended)
      return { ok: true, targetCount: 0, queued: 0, repaired: 0, needsReindex: false, reindexed: false, impact: null as LibraryQualityRepairImpact | null }
    }
    return repairQualityByNames(qualityRepairRecommendedNames, opts)
  }

  const openQualityArtifact = async (domain: 'research_qa' | 'citation_cards', target: 'report' | 'folder' | 'raw' | 'summary' | 'runbook') => {
    const key = `${domain}:${target}`
    setQualityArtifactOpening(key)
    try {
      await libraryApi.openQualityArtifact(domain, target)
      message.success(S.lib_quality_artifact_opened)
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.lib_quality_artifact_open_failed)
    } finally {
      setQualityArtifactOpening('')
    }
  }

  const handleQualityPriorityAction = async (action: LibraryQualityPriorityAction) => {
    const domain = normalizeTextValue(action.domain)
    if (domain === 'conversion') {
      handleFocusQualityReview()
      return
    }
    if (domain === 'research_qa' || domain === 'citation_cards') {
      const label = normalizeTextValue(action.label).toLowerCase()
      await openQualityArtifact(domain, label.includes('run') ? 'runbook' : 'report')
    }
  }

  const openResearchQaReplayCase = (item: LibraryQualityFailureCase) => {
    const caseId = normalizeTextValue(item.id)
    if (!caseId) {
      void openQualityArtifact('research_qa', 'report')
      return
    }
    saveResearchQaReplayFailureCase(item)
    nav(`/__research_qa_replay__?case=${encodeURIComponent(caseId)}&source=quality`)
  }

  const qualityCaseRepairSources = (item: LibraryQualityFailureCase) => {
    return (item.source_diagnostics || [])
      .filter((source) => Boolean(source.repairable))
      .filter((source) => Boolean(source.needs_repair) || ['error', 'warning'].includes(normalizeTextValue(source.quality_status).toLowerCase()))
      .map((source) => ({
        source_path: normalizeTextValue(source.source_path || source.md_path || source.pdf_path),
        source_name: normalizeTextValue(source.source_name || source.title),
      }))
      .filter((source) => source.source_path || source.source_name)
  }

  const waitForLibraryConversionDone = async (timeoutMs = 180000) => {
    return new Promise<boolean>((resolve) => {
      let settled = false
      let ctrl: AbortController | null = null
      let timer = 0
      const finish = (ok: boolean) => {
        if (settled) return
        settled = true
        window.clearTimeout(timer)
        ctrl?.abort()
        resolve(ok)
      }
      timer = window.setTimeout(() => {
        finish(false)
      }, timeoutMs)
      ctrl = libraryApi.streamConvertStatus(
        (data) => {
          if (data.done) finish(true)
        },
        () => finish(true),
        () => finish(false),
      )
    })
  }

  const repairQualityCaseSources = async (
    item: LibraryQualityFailureCase,
    opts: { manageActionKey?: boolean; waitForCompletion?: boolean; silent?: boolean; actionKey?: string } & QualityRepairRunOptions = {},
  ): Promise<{
    queued: number
    completed: boolean
    repaired: number
    needsReindex: boolean
    reindexed: boolean
    impact: LibraryQualityRepairImpact | null
  }> => {
    const sources = qualityCaseRepairSources(item)
    if (!sources.length) {
      if (!opts.silent) message.info(S.lib_msg_quality_repair_none)
      return { queued: 0, completed: true, repaired: 0, needsReindex: false, reindexed: false, impact: null }
    }
    const key = opts.actionKey || `${item.id}:repair_sources`
    const manageActionKey = opts.manageActionKey !== false
    if (manageActionKey) setQualityCaseActionKey(key)
    try {
      const res = await store.repairQuality({
        sources,
        speed_mode: CONVERT_MODE,
        replace: true,
      }, {
        autoReindexAfterQueued: opts.waitForCompletion ? false : opts.autoReindexQueued !== false,
      })
      const queued = Number(res.enqueued || 0)
      const repaired = Number(res.repaired || 0)
      const impact = res.impact || null
      const needsReindex = Boolean(res.needs_reindex || impact?.needs_reindex)
      let reindexed = false
      if (res.repair_run) {
        setQualityRepairRun(res.repair_run)
      }
      if (impact) {
        setQualityRepairImpact(impact)
      }
      if (queued > 0) {
        if (!opts.silent) message.success(S.lib_msg_quality_repair_enqueued.replace('{n}', String(queued)))
        const completed = opts.waitForCompletion ? await waitForLibraryConversionDone() : false
        if (completed && needsReindex && opts.autoReindexImmediate !== false) {
          reindexed = await handleReindex()
          if (impact) setQualityRepairImpact({ ...impact, reindexed })
          if (res.repair_run?.run_id) {
            const status = reindexed ? 'completed' : 'warning'
            const phase = reindexed ? 'reindex_complete' : 'reindex_failed'
            setQualityRepairRun({ ...res.repair_run, status, phase, reindexed })
            libraryApi.updateQualityRepairRun(res.repair_run.run_id, { status, phase, reindexed }).catch(() => {})
          }
          if (reindexed) await store.loadFiles(scope)
        }
        return { queued, completed, repaired, needsReindex, reindexed, impact }
      } else if (repaired > 0) {
        if (!opts.silent) message.success(`Markdown repaired: ${repaired}`)
        if (needsReindex && opts.autoReindexImmediate !== false) {
          reindexed = await handleReindex()
          if (impact) setQualityRepairImpact({ ...impact, reindexed })
          if (res.repair_run?.run_id) {
            const status = reindexed ? 'completed' : 'warning'
            const phase = reindexed ? 'reindex_complete' : 'reindex_failed'
            setQualityRepairRun({ ...res.repair_run, status, phase, reindexed })
            libraryApi.updateQualityRepairRun(res.repair_run.run_id, { status, phase, reindexed }).catch(() => {})
          }
          if (reindexed) await store.loadFiles(scope)
        }
        return { queued: 0, completed: true, repaired, needsReindex, reindexed, impact }
      } else {
        if (!opts.silent) message.info(S.lib_msg_quality_repair_none)
        return { queued: 0, completed: true, repaired, needsReindex, reindexed, impact }
      }
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.lib_msg_quality_repair_failed)
      return { queued: 0, completed: false, repaired: 0, needsReindex: false, reindexed: false, impact: null }
    } finally {
      if (manageActionKey) setQualityCaseActionKey('')
    }
  }

  const copyQualityFailureSummary = async (item: LibraryQualityFailureCase) => {
    const lines = [
      `Case: ${normalizeTextValue(item.id)}`,
      `Question: ${normalizeTextValue(item.question)}`,
      `Failures: ${(item.failure_names || []).join(' / ') || 'none'}`,
      `Missing docs: ${(item.missing_expected_doc_ids || []).join(' / ') || 'none'}`,
      `Root causes: ${(item.root_causes || []).map((cause) => cause.label).join(' / ') || 'unknown'}`,
      `Sources: ${(item.source_diagnostics || []).map((source) => source.title || source.source_name || source.source_path).filter(Boolean).join(' / ') || 'none'}`,
    ].join('\n')
    try {
      await navigator.clipboard.writeText(lines)
      message.success(S.copied || 'Copied')
    } catch {
      message.error(S.copy_failed || 'Copy failed')
    }
  }

  const storeQualityCaseRerunResult = (caseId: string, res: LibraryResearchQaRerunResponse) => {
    setQualityCaseRerunResults((cur) => ({ ...cur, [caseId]: res }))
    if (res.status === 'passed' || res.quality_ok) {
      message.success(`QA case passed: ${caseId}`)
    } else if (res.status === 'failed') {
      message.warning(`QA case still failing: ${caseId}`)
    } else if (normalizeTextValue(res.error_kind).toLowerCase() === 'connection') {
      message.warning(`QA service is unreachable: ${caseId}`)
    } else if (normalizeTextValue(res.error_kind).toLowerCase() === 'timeout') {
      message.warning(`QA rerun timed out: ${caseId}`)
    } else {
      message.error(`QA rerun error: ${caseId}`)
    }
  }

  const runQualityFailureCaseRerun = async (item: LibraryQualityFailureCase) => {
    const caseId = normalizeTextValue(item.id)
    if (!caseId) return null
    const res = await libraryApi.rerunResearchQaCase({ case_id: caseId })
    storeQualityCaseRerunResult(caseId, res)
    await store.loadQualityOverview('all')
    return res
  }

  const rerunQualityFailureCase = async (item: LibraryQualityFailureCase) => {
    const caseId = normalizeTextValue(item.id)
    if (!caseId) return
    const key = `${item.id}:rerun_case:`
    setQualityCaseActionKey(key)
    try {
      await runQualityFailureCaseRerun(item)
    } catch (err) {
      message.error(err instanceof Error ? err.message : 'QA rerun failed')
    } finally {
      setQualityCaseActionKey('')
    }
  }

  const repairQualityCaseShelfMetadata = async (item: LibraryQualityFailureCase) => {
    const rawItems = [
      ...(Array.isArray(item.citation_diagnostics) ? item.citation_diagnostics : []),
      ...(Array.isArray(item.ref_diagnostics) ? item.ref_diagnostics : []),
    ]
    const candidates = rawItems
      .map((entry, index) => ({
        record: entry as unknown as Record<string, unknown>,
        index,
      }))
      .map(({ record, index }) => ({
        ...record,
        key: `${item.id}:meta:${index}`,
        anchor: normalizeTextValue(record.anchor) || `${item.id}-meta-${index}`,
        title: normalizeTextValue(record.title),
        source_path: normalizeTextValue(record.source_path),
        source_name: normalizeTextValue(record.source_name || record.title),
        raw: normalizeTextValue(record.raw || record.evidence_quote),
      }))
      .filter((entry) => entry.source_path || entry.source_name || entry.title || entry.raw)
      .slice(0, 12)
    if (!candidates.length) return { ready: 0, changed: 0, retryable: 0 }
    const res = await referencesApi.repairShelfMetadata(candidates as Array<Record<string, unknown>>, candidates.length)
    const ready = Number(res.ready || 0)
    const changed = Number(res.changed || 0)
    const retryable = Number(res.retryable || 0)
    if (retryable > 0) {
      message.warning(`Metadata repair queued for retry: ${retryable}`)
    } else if (changed > 0) {
      message.success(`Citation metadata repaired: ${changed}`)
    }
    return { ready, changed, retryable }
  }

  const applyQualityFailureRepairPlan = async (item: LibraryQualityFailureCase, action: LibraryQualityRepairAction) => {
    const caseId = normalizeTextValue(item.id)
    const steps = Array.isArray(action.steps) ? action.steps : []
    const stepKinds = new Set(steps.map((step) => normalizeTextValue(step.kind)))
    const key = `${item.id}:apply_repair_plan:${action.target || ''}`
    setQualityCaseActionKey(key)
    try {
      let sourceRepairImpact: LibraryQualityRepairImpact | null = null
      if (stepKinds.has('repair_sources')) {
        const result = await repairQualityCaseSources(item, {
          actionKey: key,
          manageActionKey: false,
          waitForCompletion: true,
          autoReindexImmediate: !stepKinds.has('rebuild_index'),
        })
        sourceRepairImpact = result.impact
        if (result.queued > 0 && !result.completed) {
          message.warning('Source repair is still running; QA rerun will wait for the next refresh.')
          await store.loadQualityOverview('all')
          return { ok: true, caseId, status: 'source_repair_running', rerun: null as LibraryResearchQaRerunResponse | null }
        }
      }
      if (stepKinds.has('repair_shelf_metadata')) {
        await repairQualityCaseShelfMetadata(item)
      }
      if (stepKinds.has('rebuild_index')) {
        const ok = await handleReindex()
        if (sourceRepairImpact) setQualityRepairImpact({ ...sourceRepairImpact, reindexed: ok })
        if (!ok) return { ok: false, caseId, status: 'reindex_failed', rerun: null as LibraryResearchQaRerunResponse | null }
      }
      if (stepKinds.has('rerun_case') && caseId) {
        const rerun = await runQualityFailureCaseRerun(item)
        return { ok: Boolean(rerun?.quality_ok || rerun?.status === 'passed'), caseId, status: String(rerun?.status || ''), rerun }
      } else {
        await store.loadQualityOverview('all')
      }
      return { ok: true, caseId, status: 'repaired', rerun: null as LibraryResearchQaRerunResponse | null }
    } catch (err) {
      message.error(err instanceof Error ? err.message : 'Quality repair plan failed')
      return { ok: false, caseId, status: 'error', rerun: null as LibraryResearchQaRerunResponse | null }
    } finally {
      setQualityCaseActionKey('')
    }
  }

  const handleQualityFailureAction = async (item: LibraryQualityFailureCase, actionOrKind: LibraryQualityRepairAction | string, target = '') => {
    const actionKind = typeof actionOrKind === 'string' ? actionOrKind : actionOrKind.kind
    const actionTarget = typeof actionOrKind === 'string' ? target : (actionOrKind.target || target)
    const key = `${item.id}:${actionKind}:${actionTarget}`
    if (actionKind === 'open_replay') {
      openResearchQaReplayCase(item)
      return
    }
    if (actionKind === 'apply_repair_plan' && typeof actionOrKind !== 'string') {
      await applyQualityFailureRepairPlan(item, actionOrKind)
      return
    }
    if (actionKind === 'rerun_case') {
      await rerunQualityFailureCase(item)
      return
    }
    if (actionKind === 'repair_sources') {
      await repairQualityCaseSources(item)
      return
    }
    if (actionKind === 'rebuild_index') {
      setQualityCaseActionKey(key)
      try {
        await handleReindex()
      } finally {
        setQualityCaseActionKey('')
      }
      return
    }
    if (actionKind === 'open_artifact') {
      await openQualityArtifact('research_qa', actionTarget === 'raw' ? 'raw' : 'report')
    }
  }

  const firstQualityCaseForStage = (stageKey: string) => (
    qualityFailureCases.find((item) => qualityFailureCaseMatchesStage(item, stageKey)) || qualityFailureCases[0] || null
  )

  const repairQualityStageShelfMetadata = async (stageKey: string) => {
    const targets = qualityFailureCases.filter((item) => qualityFailureCaseMatchesStage(item, stageKey)).slice(0, 3)
    if (!targets.length) {
      await openQualityArtifact('citation_cards', 'report')
      return { targetCount: 0, targetIds: [] as string[], ready: 0, changed: 0, retryable: 0 }
    }
    let changed = 0
    let ready = 0
    let retryable = 0
    for (const item of targets) {
      const res = await repairQualityCaseShelfMetadata(item)
      changed += Number(res.changed || 0)
      ready += Number(res.ready || 0)
      retryable += Number(res.retryable || 0)
    }
    if (changed <= 0 && ready <= 0) {
      message.info('No repairable citation metadata found in the current failed cases.')
    }
    await store.loadQualityOverview('all')
    return { targetCount: targets.length, targetIds: targets.map((item) => normalizeTextValue(item.id)).filter(Boolean), ready, changed, retryable }
  }

  const refreshQualityOverviewSnapshot = async () => {
    await store.loadQualityOverview('all')
    const overview = useLibraryStore.getState().qualityOverview
    return overview?.ok ? overview : null
  }

  const handleQualityFullChainStage = async (stage: LibraryQualityFullChainStage) => {
    const stageKey = normalizeTextValue(stage.key).toLowerCase()
    const action = normalizeTextValue(stage.action).toLowerCase()
    const caseTarget = firstQualityCaseForStage(stageKey)
    const beforeOverview = backendQualityOverview
    const beforeSnapshot = qualityOverviewStageSnapshot(beforeOverview, stageKey)
    const recordStageResult = (
      result: Omit<QualityFullChainActionResult, 'updatedAt'>,
      meta: {
        targetIds?: string[]
        metrics?: Record<string, string | number | boolean | null | undefined>
        afterOverview?: LibraryQualityOverviewResponse | null
        verification?: Record<string, string | number | boolean | null | undefined>
      } = {},
    ) => {
      const latestOverview = meta.afterOverview
        || (useLibraryStore.getState().qualityOverview?.ok ? useLibraryStore.getState().qualityOverview : null)
        || backendQualityOverview
      const afterSnapshot = qualityOverviewStageSnapshot(latestOverview, stageKey)
      recordQualityFullChainResult(stageKey, result, {
        stageLabel: stage.label,
        action: stage.action,
        before: beforeSnapshot,
        after: afterSnapshot,
        verification: meta.verification,
        targetIds: meta.targetIds,
        metrics: meta.metrics,
      })
    }
    setQualityFullChainActionKey(stageKey)
    try {
      if (stageKey === 'conversion' || action === 'repair_conversion') {
        if (qualityRepairRecommendedNames.length > 0) {
          const repair = await handleRepairRecommendedQuality({
            autoReindexImmediate: false,
            autoReindexQueued: false,
          })
          const completed = Number(repair?.queued || 0) > 0 ? await waitForLibraryConversionDone() : true
          const repaired = Number(repair?.repaired || 0)
          const queued = Number(repair?.queued || 0)
          const needsReindex = Boolean(repair?.needsReindex || repair?.impact?.needs_reindex)
          const reindexed = completed && needsReindex ? await handleReindex() : false
          if (repair?.impact && needsReindex) {
            setQualityRepairImpact({ ...repair.impact, reindexed })
          }
          if (reindexed) await store.loadFiles(scope)
          const rerun = completed && (!needsReindex || reindexed) && caseTarget ? await runQualityFailureCaseRerun(caseTarget) : null
          const afterOverview = await refreshQualityOverviewSnapshot()
          const repairOk = Boolean(repair?.ok)
          const reindexFailed = Boolean(completed && needsReindex && !reindexed)
          recordStageResult({
            status: reindexFailed ? 'warning' : (queued > 0 || repaired > 0 ? 'success' : (repairOk ? 'info' : 'error')),
            summary: queued > 0
              ? (completed
                ? (reindexFailed ? `Converted ${queued} sources; index refresh failed` : `Verified ${queued} conversion repairs`)
                : `Queued ${queued} conversion repairs`)
              : (repaired > 0
                ? (reindexFailed ? `Markdown autofix repaired ${repaired}; index refresh failed` : `Markdown autofix repaired ${repaired} sources`)
                : (repairOk ? 'No conversion repair was queued' : 'Conversion repair failed')),
            detail: rerun?.case_id
              ? `Regression check: ${rerun.case_id}`
              : (reindexFailed ? 'Rebuild the retrieval index before rerunning QA.' : (repair?.targetCount ? `${repair.targetCount} recommended sources checked` : undefined)),
          }, {
            targetIds: qualityRepairRecommendedNames.slice(0, 12),
            metrics: {
              queued,
              repaired,
              target_count: Number(repair?.targetCount || 0),
              conversion_completed: Boolean(completed),
              needs_reindex: needsReindex,
              reindexed,
              qa_rerun_quality_ok: Boolean(rerun?.quality_ok || rerun?.status === 'passed'),
            },
            afterOverview,
            verification: qualityVerificationFromRerun(rerun),
          })
        } else {
          handleFocusQualityReview()
          recordStageResult({
            status: 'info',
            summary: 'Focused the conversion review list',
          })
        }
        return
      }
      if (stageKey === 'retrieval' || action === 'rebuild_index') {
        const ok = await handleReindex()
        const rerun = ok && caseTarget ? await runQualityFailureCaseRerun(caseTarget) : null
        if (ok && !rerun) await store.loadQualityOverview('all')
        const afterOverview = await refreshQualityOverviewSnapshot()
        const rerunPassed = Boolean(rerun?.quality_ok || rerun?.status === 'passed')
        recordStageResult({
          status: ok ? (rerun && !rerunPassed ? 'warning' : 'success') : 'error',
          summary: ok
            ? (rerun ? (rerunPassed ? `Reindex verified: ${caseTarget?.id}` : `Reindex done; QA still failing: ${caseTarget?.id}`) : 'Rebuilt retrieval index')
            : 'Retrieval index rebuild failed',
          detail: rerun?.failures?.length ? `${rerun.failures.length} failures remain` : (caseTarget?.id ? `Regression check: ${caseTarget.id}` : undefined),
        }, {
          targetIds: caseTarget?.id ? [caseTarget.id] : [],
          metrics: {
            qa_rerun_quality_ok: rerun ? rerunPassed : false,
            failure_count: Number(rerun?.failures?.length || 0),
          },
          afterOverview,
          verification: qualityVerificationFromRerun(rerun),
        })
        return
      }
      if (stageKey === 'citations' || stageKey === 'shelf' || action === 'repair_citation_cards' || action === 'repair_shelf_metadata') {
        const result = await repairQualityStageShelfMetadata(stageKey === 'citations' ? 'citations' : 'shelf')
        const rerun = result.targetCount > 0 && caseTarget ? await runQualityFailureCaseRerun(caseTarget) : null
        const afterOverview = await refreshQualityOverviewSnapshot()
        const rerunPassed = Boolean(rerun?.quality_ok || rerun?.status === 'passed')
        recordStageResult({
          status: result.retryable > 0 || (rerun && !rerunPassed) ? 'warning' : (result.changed > 0 || result.ready > 0 ? 'success' : 'info'),
          summary: rerun
            ? (rerunPassed ? `Metadata repair verified: ${caseTarget?.id}` : `Metadata checked; QA still failing: ${caseTarget?.id}`)
            : (result.changed > 0
              ? `Metadata repaired: ${result.changed}`
              : (result.ready > 0 ? `Metadata already ready: ${result.ready}` : 'Opened citation quality report')),
          detail: rerun?.failures?.length ? `${rerun.failures.length} failures remain` : (result.targetCount > 0 ? `${result.targetCount} failed cases checked` : undefined),
        }, {
          targetIds: result.targetIds,
          metrics: {
            changed: result.changed,
            ready: result.ready,
            retryable: result.retryable,
            target_count: result.targetCount,
            qa_rerun_quality_ok: rerun ? rerunPassed : false,
          },
          afterOverview,
          verification: qualityVerificationFromRerun(rerun),
        })
        return
      }
      if (stageKey === 'repair_loop' || action === 'rerun_failed_cases') {
        if (caseTarget) {
          const rerun = await runQualityFailureCaseRerun(caseTarget)
          const afterOverview = await refreshQualityOverviewSnapshot()
          recordStageResult({
            status: rerun?.quality_ok || rerun?.status === 'passed' ? 'success' : 'warning',
            summary: rerun?.quality_ok || rerun?.status === 'passed'
              ? `Rerun passed: ${caseTarget.id}`
              : `Rerun still failing: ${caseTarget.id}`,
            detail: rerun?.failures?.length ? `${rerun.failures.length} failures remain` : undefined,
          }, {
            targetIds: [caseTarget.id],
            metrics: {
              quality_ok: Boolean(rerun?.quality_ok),
              failure_count: Number(rerun?.failures?.length || 0),
            },
            afterOverview,
            verification: qualityVerificationFromRerun(rerun),
          })
        } else {
          await openQualityArtifact('research_qa', 'report')
          recordStageResult({
            status: 'info',
            summary: 'Opened QA report',
          })
        }
        return
      }
      if (stageKey === 'research_qa' || action === 'fix_failed_qa_cases') {
        const plan = caseTarget?.repair_actions?.find((item) => item.kind === 'apply_repair_plan')
        if (caseTarget && plan) {
          const result = await applyQualityFailureRepairPlan(caseTarget, plan)
          const afterOverview = await refreshQualityOverviewSnapshot()
          recordStageResult({
            status: result?.ok ? 'success' : 'warning',
            summary: result?.rerun?.quality_ok || result?.rerun?.status === 'passed'
              ? `Repair plan passed: ${caseTarget.id}`
              : `Repair plan ran: ${caseTarget.id}`,
            detail: result?.status ? `Last status: ${result.status}` : undefined,
          }, {
            targetIds: [caseTarget.id],
            metrics: {
              quality_ok: Boolean(result?.rerun?.quality_ok),
              has_rerun: Boolean(result?.rerun),
            },
            afterOverview,
            verification: qualityVerificationFromRerun(result?.rerun),
          })
        } else if (caseTarget) {
          const rerun = await runQualityFailureCaseRerun(caseTarget)
          const afterOverview = await refreshQualityOverviewSnapshot()
          recordStageResult({
            status: rerun?.quality_ok || rerun?.status === 'passed' ? 'success' : 'warning',
            summary: rerun?.quality_ok || rerun?.status === 'passed'
              ? `QA case passed: ${caseTarget.id}`
              : `QA case still failing: ${caseTarget.id}`,
            detail: rerun?.failures?.length ? `${rerun.failures.length} failures remain` : undefined,
          }, {
            targetIds: [caseTarget.id],
            metrics: {
              quality_ok: Boolean(rerun?.quality_ok),
              failure_count: Number(rerun?.failures?.length || 0),
            },
            afterOverview,
            verification: qualityVerificationFromRerun(rerun),
          })
        } else {
          await openQualityArtifact('research_qa', action === 'run_research_qa' ? 'runbook' : 'report')
          recordStageResult({
            status: 'info',
            summary: action === 'run_research_qa' ? 'Opened QA runbook' : 'Opened QA report',
          })
        }
        return
      }
      if (stageKey === 'citation_cards') {
        await openQualityArtifact('citation_cards', 'report')
        recordStageResult({
          status: 'info',
          summary: 'Opened citation-card quality report',
        })
        return
      }
      message.info('No direct action is available for this quality stage yet.')
      recordStageResult({
        status: 'info',
        summary: 'No direct action available for this stage',
      })
    } catch (err) {
      message.error(err instanceof Error ? err.message : 'Quality stage action failed')
    } finally {
      setQualityFullChainActionKey('')
    }
  }

  const handleQualityFeatureHealthAction = async (item: LibraryQualityFeatureHealthItem) => {
    const targetStage = normalizeTextValue(item.target_stage).toLowerCase()
    const stage = qualityFullChainStages.find((stageItem) => normalizeTextValue(stageItem.key).toLowerCase() === targetStage)
    if (stage) {
      await handleQualityFullChainStage(stage)
      return
    }
    const featureKey = normalizeTextValue(item.key).toLowerCase()
    if (featureKey === 'pdf_conversion') {
      handleFocusQualityReview()
      return
    }
    if (featureKey === 'citation_cards' || featureKey === 'literature_basket') {
      await openQualityArtifact('citation_cards', 'report')
      return
    }
    await openQualityArtifact('research_qa', featureKey === 'general_qa' ? 'report' : 'runbook')
  }

  const handleQualityActionHistoryOpen = async (item: LibraryQualityActionHistoryItem) => {
    const stageKey = normalizeTextValue(item.stage_key).toLowerCase()
    const targetIds = normalizeTextList(item.target_ids || [])
    const firstTarget = targetIds[0] || ''
    if (stageKey === 'conversion') {
      if (targetIds.length > 0) {
        focusQualityHistoryNames(targetIds)
      } else {
        handleFocusQualityReview()
      }
      return
    }
    if (['research_qa', 'retrieval', 'repair_loop', 'citations', 'shelf'].includes(stageKey)) {
      if (firstTarget) {
        const failureCase = qualityFailureCases.find((row) => normalizeTextValue(row.id) === firstTarget)
        if (failureCase) saveResearchQaReplayFailureCase(failureCase)
        nav(`/__research_qa_replay__?case=${encodeURIComponent(firstTarget)}&source=quality-history`)
        return
      }
      await openQualityArtifact(stageKey === 'citations' || stageKey === 'shelf' ? 'citation_cards' : 'research_qa', 'report')
      return
    }
    if (stageKey === 'citation_cards') {
      await openQualityArtifact('citation_cards', 'report')
      return
    }
    await openQualityArtifact('research_qa', 'report')
  }

  const handleDeleteOne = async (item: LibraryFileItem) => {
    const res = await store.deleteFile(item.name, true)
    if (res.ok) {
      message.success(S.lib_msg_deleted_name.replace('{name}', item.name))
      if (res.needs_reindex) {
        message.info(S.lib_msg_delete_suggest_reindex)
      }
      return
    }
    const warning = Array.isArray(res.warnings) && res.warnings.length > 0
      ? `（${res.warnings.join('；')}）`
      : ''
    message.warning(S.lib_msg_delete_not_complete.replace('{warning}', warning))
  }

  const confirmDeleteOne = (item: LibraryFileItem) => {
    Modal.confirm({
      title: S.lib_menu_delete_confirm_title,
      content: item.name,
      okText: S.lib_menu_delete_ok,
      okType: 'danger',
      cancelText: S.lib_menu_delete_cancel,
      onOk: async () => {
        await handleDeleteOne(item)
      },
    })
  }

  const handleReindex = async (): Promise<boolean> => {
    const hide = message.loading(S.lib_msg_updating_kb, 0)
    try {
      const res = await store.reindex()
      hide()
      if (!res.ok) {
        message.error(S.lib_msg_exec_fail)
        return false
      }
      message.success(S.lib_msg_exec_done)
      if (res.refsync_error) {
        message.warning(S.lib_msg_refsync_fail_detail.replace('{error}', String(res.refsync_error)))
      } else if (res.refsync?.started) {
        message.info(S.lib_msg_refsync_started_bg)
      }
      return true
    } catch (err) {
      hide()
      message.error(err instanceof Error ? err.message : S.lib_msg_exec_fail)
      return false
    }
  }

  const handleAdvanceQualityRepairRun = async () => {
    const runId = normalizeTextValue(qualityRepairRun?.run_id)
    if (!runId) return
    const hide = message.loading('Continuing repair run...', 0)
    setQualityRepairAdvancing(true)
    try {
      const res = await libraryApi.advanceQualityRepairRun(runId)
      hide()
      if (res.item) {
        setQualityRepairRun(res.item)
        const impact = res.item.impact
        if (impact && typeof impact === 'object' && typeof (impact as LibraryQualityRepairImpact).needs_reindex === 'boolean') {
          setQualityRepairImpact(impact as LibraryQualityRepairImpact)
        }
      }
      if (res.waiting) {
        message.info('Conversion is still running; continue again after it finishes.')
      } else if (res.ok) {
        message.success(res.advanced ? 'Repair run advanced' : 'Repair run is already up to date')
      } else {
        message.error(res.detail || S.lib_msg_exec_fail)
      }
      if (res.reindex?.refsync_error) {
        message.warning(S.lib_msg_refsync_fail_detail.replace('{error}', String(res.reindex.refsync_error)))
      } else if (res.reindex?.refsync?.started) {
        message.info(S.lib_msg_refsync_started_bg)
      }
      await store.loadFiles(scope)
      await store.loadQualityOverview('all')
    } catch (err) {
      hide()
      message.error(err instanceof Error ? err.message : S.lib_msg_exec_fail)
    } finally {
      setQualityRepairAdvancing(false)
    }
  }

  const handleStartRefSync = async () => {
    const hide = message.loading(S.lib_msg_starting_refsync, 0)
    try {
      const res = await store.startReferenceSync()
      hide()
      if (res.started) {
        message.success(S.lib_msg_refsync_started)
      } else if (res.reason === 'running') {
        message.info(S.lib_msg_refsync_already_running)
      } else {
        message.warning(S.lib_msg_refsync_not_started)
      }
    } catch (err) {
      hide()
      message.error(err instanceof Error ? err.message : S.lib_msg_start_refsync_fail)
    }
  }

  const handleStartPaperGuide = async (item: LibraryFileItem) => {
    if (!item.md_exists || !item.md_path) {
      message.info(S.lib_msg_guide_not_converted)
      return
    }
    const hide = message.loading(S.lib_msg_creating_guide, 0)
    try {
      let sourcePath = ''
      let sourceName = stripKnownSourceExt(item.name) || item.name
      let resolvedMdPath = ''
      try {
        const resolved = await libraryApi.resolveGuideSource(item.name)
        sourcePath = String(resolved.source_path || '').trim()
        sourceName = String(resolved.source_name || '').trim() || sourceName
        resolvedMdPath = String(resolved.md_path || '').trim()
      } catch {
        // Backward-compatible fallback when backend route is not available yet.
        sourcePath = String(item.md_path || '').trim()
        message.warning(S.lib_msg_guide_source_fallback)
      }
      const convTitle = `阅读指导 · ${sourceName}`
      if (!sourcePath) throw new Error('source path not ready')
      await createPaperGuideConversation({
        sourcePath,
        sourceName,
        title: convTitle,
      })
      if (resolvedMdPath && resolvedMdPath !== String(item.md_path || '').trim()) {
        void store.loadFiles(scope)
      }
      hide()
      nav('/')
      message.success(S.lib_msg_guide_entered)
    } catch (err) {
      hide()
      message.error(err instanceof Error ? err.message : S.lib_msg_guide_create_fail)
    }
  }

  const openMetaEditor = (item: LibraryFileItem) => {
    setMetaItem(item)
    setMetaDraft({
      paper_category: normalizeTextValue(item.paper_category),
      reading_status: (String(item.reading_status || '') as ReadingStatusValue),
      note: String(item.note || ''),
      user_tags: normalizeTextList(Array.isArray(item.user_tags) ? item.user_tags : []),
    })
    setMetaDrawerOpen(true)
  }

  const saveMetaEditor = async () => {
    if (!metaItem) return
    const paperCategory = normalizeTextValue(metaDraft.paper_category)
    const userTags = normalizeTextList(metaDraft.user_tags)
    setMetaSaving(true)
    try {
      const updated = await store.updatePaperMeta({
        pdf_name: metaItem.name,
        paper_category: paperCategory,
        reading_status: metaDraft.reading_status,
        note: metaDraft.note,
        user_tags: userTags,
      })
      if (updated) setMetaItem(updated)
      setMetaDrawerOpen(false)
      message.success(S.lib_msg_meta_saved)
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.lib_msg_meta_save_fail)
    } finally {
      setMetaSaving(false)
    }
  }

  const regenerateSuggestionsForVisible = async () => {
    const targets = visibleAll.map((item) => item.name).filter(Boolean)
    if (!targets.length) {
      message.info(S.lib_msg_no_suggestion_candidates)
      return
    }
    setSuggestionsRefreshing(true)
    try {
      const updated = await store.regenerateSuggestions({ pdf_names: targets })
      message.success(S.lib_msg_suggestions_refreshed_count.replace('{n}', String(updated)))
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.lib_msg_refresh_suggestion_fail)
    } finally {
      setSuggestionsRefreshing(false)
    }
  }

  const applyMetaSuggestionAction = async (body: {
    category_action?: '' | 'accept' | 'dismiss'
    accept_tags?: string[]
    dismiss_tags?: string[]
    accept_all_tags?: boolean
    dismiss_all_tags?: boolean
  }) => {
    if (!metaItem) return
    setMetaSuggestionSaving(true)
    try {
      const updated = await store.applySuggestionAction({
        pdf_name: metaItem.name,
        category_action: body.category_action,
        accept_tags: body.accept_tags,
        dismiss_tags: body.dismiss_tags,
        accept_all_tags: body.accept_all_tags,
        dismiss_all_tags: body.dismiss_all_tags,
      })
      if (updated) {
        setMetaItem(updated)
        setMetaDraft((cur) => ({
          ...cur,
          paper_category: normalizeTextValue(updated.paper_category),
          reading_status: String(updated.reading_status || '') as ReadingStatusValue,
          note: String(updated.note || ''),
          user_tags: normalizeTextList(Array.isArray(updated.user_tags) ? updated.user_tags : []),
        }))
      }
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.lib_msg_update_suggestion_fail)
    } finally {
      setMetaSuggestionSaving(false)
    }
  }

  const regenerateMetaSuggestions = async () => {
    if (!metaItem) return
    setMetaSuggestionSaving(true)
    try {
      await store.regenerateSuggestions({ pdf_names: [metaItem.name] })
      const refreshed = useLibraryStore.getState().files.find((item) => item.name === metaItem.name) || null
      if (refreshed) {
        setMetaItem(refreshed)
        setMetaDraft((cur) => ({
          ...cur,
          paper_category: normalizeTextValue(refreshed.paper_category),
          reading_status: String(refreshed.reading_status || '') as ReadingStatusValue,
          note: String(refreshed.note || ''),
          user_tags: normalizeTextList(Array.isArray(refreshed.user_tags) ? refreshed.user_tags : []),
        }))
      }
      message.success(S.lib_msg_suggestion_refreshed)
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.lib_msg_refresh_suggestion_fail)
    } finally {
      setMetaSuggestionSaving(false)
    }
  }

  const toggleLibrarySelection = (name: string, checked: boolean) => {
    setSelectedLibraryNames((cur) => {
      if (!checked && !cur[name]) return cur
      return {
        ...cur,
        [name]: checked,
      }
    })
  }

  const selectCurrentListItems = () => {
    if (!currentListItems.length) {
      message.info(S.lib_msg_no_selectable)
      return
    }
    setSelectedLibraryNames((cur) => {
      const next = { ...cur }
      for (const item of currentListItems) next[item.name] = true
      return next
    })
  }

  const clearLibrarySelection = () => {
    setSelectedLibraryNames({})
  }

  const openBatchEditor = () => {
    if (!selectedLibraryCount) {
      message.info(S.lib_msg_select_batch_edit)
      return
    }
    setBatchDraft({
      apply_paper_category: false,
      paper_category: '',
      apply_reading_status: false,
      reading_status: '',
      add_tags: [],
      remove_tags: [],
    })
    setBatchDrawerOpen(true)
  }

  const saveBatchEditor = async () => {
    if (!selectedLibraryCount) return
    const paperCategory = normalizeTextValue(batchDraft.paper_category)
    const addTags = normalizeTextList(batchDraft.add_tags)
    const removeTags = normalizeTextList(batchDraft.remove_tags)
    if (
      !batchDraft.apply_paper_category
      && !batchDraft.apply_reading_status
      && addTags.length === 0
      && removeTags.length === 0
    ) {
      message.info(S.lib_msg_set_batch_content)
      return
    }
    setBatchSaving(true)
    try {
      const updated = await store.batchUpdatePaperMeta({
        pdf_names: selectedLibraryNamesList,
        apply_paper_category: batchDraft.apply_paper_category,
        paper_category: paperCategory,
        apply_reading_status: batchDraft.apply_reading_status,
        reading_status: batchDraft.reading_status,
        add_tags: addTags,
        remove_tags: removeTags,
      })
      setBatchDrawerOpen(false)
      setSelectedLibraryNames({})
      message.success(S.lib_msg_batch_updated_count.replace('{n}', String(updated)))
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.lib_msg_batch_edit_fail)
    } finally {
      setBatchSaving(false)
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
    const tag = fileTag(item, S)
    const statusTone =
      tag.color === 'success'
        ? 'is-success'
        : tag.color === 'processing'
          ? 'is-processing'
          : tag.color === 'warning'
            ? 'is-warning'
            : 'is-default'
    const readingLabel = readingStatusLabel(item.reading_status, S)
    const metaTags = item.user_tags || []
    const suggestionCount = (item.suggested_category ? 1 : 0) + (item.suggested_tags || []).length
    const categoryActive = !onlyUnclassified && paperCategoryFilter && String(item.paper_category || '') === paperCategoryFilter
    const statusActive = readingStatusFilter && item.reading_status === readingStatusFilter
    const isSelected = Boolean(selectedLibraryNames[item.name])
    const showPrimaryConvertAction = !item.md_exists
    const itemProgress = derivePageProgress(item.cur_page_done, item.cur_page_total, item.cur_page_msg)
    const itemProgressPercent = itemProgress.total > 0
      ? Math.round((itemProgress.done / Math.max(1, itemProgress.total)) * 100)
      : 0
    const quality = item.conversion_quality
    const qualityIssues = Array.isArray(quality?.issues) ? quality.issues.slice(0, 3) : []
    const qualityReport = quality?.conversion_report || null
    const qualityRepairPlan = qualityReport?.repair_plan || null
    const latestQualityRepairAttempt = qualityReport?.latest_repair_attempt || null
    const latestQualityRepairAttemptStatus = normalizeTextValue(latestQualityRepairAttempt?.status).toLowerCase()
    const latestQualityRepairAttemptTone =
      ['success', 'resolved', 'ready'].includes(latestQualityRepairAttemptStatus)
        ? 'is-success'
        : ['error', 'failed', 'blocked'].includes(latestQualityRepairAttemptStatus)
          ? 'is-error'
          : 'is-warning'
    const qualityAutoRepairApplied = Array.isArray(qualityReport?.auto_repair_applied)
      ? qualityReport?.auto_repair_applied || []
      : []
    const mathCount = conversionMetric(quality, 'display_math') + conversionMetric(quality, 'inline_math')
    const qualityNeedsRepair = hasConversionQualityIssue(item)
    const qualityRepairing = Boolean(qualityRepairingNames[item.name])
    const qualityRepairRecord = qualityRepairHistory[item.name]
    const qualityRepairResult = String(
      qualityRepairResults[item.name] || (qualityRepairRecord ? formatQualityRepairRecordSummary(qualityRepairRecord, S) : ''),
    ).trim()

    return (
      <div
        className={`kb-lib-file-row${isSelected ? ' is-selected' : ''}${suggestionCount > 0 ? ' has-suggestions' : ''}`}
        data-testid="library-file-row"
        data-library-file-name={item.name}
      >
        <div className="kb-lib-file-select">
          <Checkbox
            checked={isSelected}
            onChange={(event) => toggleLibrarySelection(item.name, event.target.checked)}
          />
        </div>

        <div className="kb-lib-file-main">
          <div className="kb-lib-file-head">
            <div className="kb-lib-file-title-wrap">
              <Text className="kb-lib-file-title" title={item.name}>{item.name}</Text>
            </div>
            <div className="kb-lib-file-submeta">
              <span className={`kb-lib-file-status-chip ${statusTone}`}>{tag.text}</span>
              {quality ? (
                <span
                  className={`kb-lib-file-quality-chip ${conversionQualityToneClass(quality)}`}
                  data-testid="library-file-quality-chip"
                  data-quality-status={conversionQualityStatus(quality)}
                  title={quality.summary}
                >
                  {conversionQualityLabel(quality)}
                </span>
              ) : null}
              {!item.md_exists ? <span className="kb-lib-file-meta-muted">{S.lib_file_no_md}</span> : null}
              {suggestionCount > 0 ? (
                <span className="kb-lib-file-submeta-chip is-suggestion">
                  {S.lib_file_suggestions.replace('{n}', String(suggestionCount))}
                </span>
              ) : null}
            </div>
          </div>

          {(item.paper_category || readingLabel || metaTags.length > 0) ? (
            <div className="kb-lib-file-taxonomy">
              {item.paper_category ? (
                <button
                  type="button"
                  className={`kb-lib-taxonomy-pill is-category${categoryActive ? ' is-active' : ''}`}
                  onClick={() => applyPaperCategoryFilter(String(item.paper_category || ''))}
                >
                  {item.paper_category}
                </button>
              ) : null}
              {readingLabel ? (
                <button
                  type="button"
                  className={`kb-lib-taxonomy-pill is-status${statusActive ? ' is-active' : ''}`}
                  onClick={() => setReadingStatusFilter(item.reading_status)}
                >
                  {readingLabel}
                </button>
              ) : null}
              {metaTags.map((tagValue) => (
                <button
                  key={`${item.name}-tag-${tagValue}`}
                  type="button"
                  className={`kb-lib-taxonomy-pill is-tag${paperTagFilter && tagValue.toLowerCase() === paperTagFilter.toLowerCase() ? ' is-active' : ''}`}
                  onClick={() => applyPaperTagFilter(tagValue)}
                >
                  #{tagValue}
                </button>
              ))}
            </div>
          ) : null}

          {quality ? (
            <div className="kb-lib-quality-line" data-testid="library-file-quality-line">
              <span className="kb-lib-quality-metric">pages {conversionMetric(quality, 'page_markers')}</span>
              <span className="kb-lib-quality-metric">refs {conversionMetric(quality, 'references') || conversionMetric(quality, 'reference_lines')}</span>
              <span className="kb-lib-quality-metric">fig {conversionMetric(quality, 'figures')}</span>
              <span className="kb-lib-quality-metric">math {mathCount}</span>
              {qualityIssues.map((issue) => (
                <span
                  key={`${item.name}-${issue.code}`}
                  className={`kb-lib-quality-issue ${String(issue.severity || '') === 'error' ? 'is-error' : 'is-warning'}`}
                  title={issue.label}
                >
                  {issue.label}
                </span>
              ))}
              {qualityReport?.auto_repair_changed ? (
                <span
                  className="kb-lib-quality-issue is-success"
                  title={qualityAutoRepairApplied.join(' / ') || 'Conversion auto-repair applied'}
                >
                  auto fixed {qualityAutoRepairApplied.length || 1}
                </span>
              ) : null}
              {qualityReport?.needs_reconvert ? (
                <span
                  className="kb-lib-quality-issue is-error"
                  title={qualityRepairPlan?.reason || 'Conversion report recommends re-conversion'}
                >
                  {qualityRepairPlan?.scope ? `reconvert ${qualityRepairPlan.scope}` : 'reconvert'}
                </span>
              ) : null}
              {latestQualityRepairAttempt ? (
                <span
                  className={`kb-lib-quality-issue ${latestQualityRepairAttemptTone}`}
                  title={latestQualityRepairAttempt.detail || latestQualityRepairAttempt.reason || latestQualityRepairAttempt.event}
                >
                  {latestQualityRepairAttemptStatus === 'queued'
                    ? 'source repair queued'
                    : latestQualityRepairAttemptStatus === 'success'
                      ? 'source repair ok'
                      : latestQualityRepairAttemptStatus === 'partial'
                        ? 'source repair partial'
                        : `source repair ${latestQualityRepairAttemptStatus || 'tracked'}`}
                </span>
              ) : null}
              {qualityNeedsRepair ? (
                <Button
                  size="small"
                  icon={<ReloadOutlined />}
                  className="kb-lib-quality-repair-btn"
                  data-testid="library-quality-repair"
                  loading={qualityRepairing}
                  disabled={item.task_state !== 'idle'}
                  onClick={() => { void handleRepairQualityOne(item) }}
                >
                  {S.lib_btn_repair_quality}
                </Button>
              ) : null}
            </div>
          ) : null}

          {qualityRepairResult ? (
            <div className="kb-lib-quality-repair-result" data-testid="library-quality-repair-result">
              {qualityRepairResult}
            </div>
          ) : null}

          {item.note ? <div className="kb-lib-file-note">{item.note}</div> : null}
          {item.task_state === 'running' ? (
            <div style={{ marginTop: 8 }}>
              {itemProgress.total > 0 ? (
                <>
                  <Progress percent={itemProgressPercent} status="active" size="small" showInfo={false} />
                  <Text type="secondary" className="text-xs">
                    {`\u9875\u8fdb\u5ea6 ${itemProgress.done}/${itemProgress.total}`}
                  </Text>
                </>
              ) : null}
              {item.cur_page_msg ? (
                <div>
                  <Text type="secondary" className="text-xs">{item.cur_page_msg}</Text>
                </div>
              ) : null}
            </div>
          ) : null}
        </div>

        <div className={`kb-lib-file-actions${showPrimaryConvertAction ? ' has-convert' : ' is-compact'}`}>
          <Button className="kb-lib-file-action-main" size="small" onClick={() => openMetaEditor(item)}>
            {S.lib_btn_categorize}
          </Button>
          {item.md_exists ? (
            <Button
              className="kb-lib-file-action-link"
              type="text"
              size="small"
              disabled={!item.md_path}
              onClick={() => { void handleStartPaperGuide(item) }}
            >
              {S.lib_btn_read}
            </Button>
          ) : null}
          {showPrimaryConvertAction ? (
            <Button
              className="kb-lib-file-action-link is-accent"
              type="text"
              size="small"
              disabled={item.task_state !== 'idle'}
              onClick={() => { void handleConvertOne(item) }}
            >
              {S.lib_btn_convert}
            </Button>
          ) : null}
          <Button className="kb-lib-file-action-link" type="text" size="small" onClick={() => { void store.openFile(item.name, 'pdf') }}>
            PDF
          </Button>
          <div className="kb-lib-file-more">
            <Dropdown
              trigger={['click']}
              menu={{
                items: [
                  ...(item.md_exists
                    ? [{ key: 'reconvert', label: S.lib_btn_reconvert, disabled: item.task_state !== 'idle', icon: <ReloadOutlined /> }]
                    : []),
                  { key: 'open-md', label: S.lib_btn_open_md, disabled: !item.md_exists },
                  { type: 'divider' },
                  { key: 'delete', label: S.lib_btn_delete, danger: true, disabled: item.task_state !== 'idle', icon: <DeleteOutlined /> },
                ],
                onClick: ({ key }) => {
                  if (key === 'reconvert') {
                    void handleConvertOne(item)
                    return
                  }
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
          </div>
        </div>
      </div>
    )
  }

  const renderCategoriesView = () => {
    if (!categoryCards.length) {
      return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={S.lib_empty_category} />
    }

    return (
      <div className="kb-lib-category-grid">
        {categoryCards.map((card) => {
          const isUnclassified = card.key === 'category:__unclassified__'
          const active = isUnclassified ? onlyUnclassified : (!onlyUnclassified && paperCategoryFilter === card.label)
          return (
            <button
              key={card.key}
              type="button"
              className={`kb-lib-category-card${active ? ' is-active' : ''}`}
              onClick={() => {
                if (isUnclassified) {
                  setPaperCategoryFilter('')
                  setOnlyUnclassified(true)
                } else {
                  applyPaperCategoryFilter(card.label)
                }
                setBrowseMode('list')
              }}
            >
              <div className="kb-lib-category-card-head">
                <div className="kb-lib-category-card-title">
                  <span>{card.label}</span>
                  <strong>{card.count}</strong>
                </div>
                <div className="kb-lib-category-card-meta">
                  <span>{card.unreadCount} unread</span>
                  <span>{card.convertedCount} converted</span>
                  {card.pendingCount > 0 ? <span>{card.pendingCount} pending</span> : null}
                </div>
              </div>

              {card.commonTags.length > 0 ? (
                <div className="kb-lib-category-card-tags">
                  {card.commonTags.map((tagValue) => (
                    <span key={`${card.key}-${tagValue}`} className="kb-lib-category-tag">
                      #{tagValue}
                    </span>
                  ))}
                </div>
              ) : (
                <div className="kb-lib-category-card-empty">{S.lib_tag_empty_common}</div>
              )}

              <div className="kb-lib-category-card-recent">
                {card.recentPapers.map((paper) => (
                  <span key={`${card.key}-${paper}`} className="kb-lib-category-paper">
                    {paper}
                  </span>
                ))}
              </div>
            </button>
          )
        })}
      </div>
    )
  }

  const renderTagsView = () => {
    if (!tagCards.length) {
      return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={S.lib_empty_tag} />
    }

    return (
      <div className="kb-lib-tag-grid">
        {tagCards.map((card) => {
          const active = paperTagFilter && card.label.toLowerCase() === paperTagFilter.toLowerCase()
          return (
            <button
              key={card.key}
              type="button"
              className={`kb-lib-tag-card${active ? ' is-active' : ''}`}
              onClick={() => {
                applyPaperTagFilter(card.label)
                setBrowseMode('list')
              }}
            >
              <div className="kb-lib-tag-card-head">
                <div className="kb-lib-tag-card-title">
                  <span>#{card.label}</span>
                  <strong>{card.count}</strong>
                </div>
                <div className="kb-lib-tag-card-meta">
                  <span>{S.lib_tag_unread_count.replace('{n}', String(card.unreadCount))}</span>
                </div>
              </div>

              {card.categories.length > 0 ? (
                <div className="kb-lib-tag-card-cats">
                  {card.categories.map((category) => (
                    <span key={`${card.key}-${category}`} className="kb-lib-tag-category">
                      {category}
                    </span>
                  ))}
                </div>
              ) : null}

              <div className="kb-lib-tag-card-recent">
                {card.recentPapers.map((paper) => (
                  <span key={`${card.key}-${paper}`} className="kb-lib-tag-paper">
                    {paper}
                  </span>
                ))}
              </div>
            </button>
          )
        })}
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
          <Text type="secondary" className="text-xs">{S.lib_virtual_scroll_hint.replace('{n}', String(items.length))}</Text>
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
    quality_review: qualityReviewCount,
    quality_ready: qualityReadyCount,
  }

  const directoriesConfigured = Boolean(pdfDirDraft.trim() && mdDirDraft.trim())
  const showDirEditor = dirEditorOpen || !directoriesConfigured
  const workbenchStats = [
    { key: 'view', label: S.lib_stats_view, value: counts.total_view },
    { key: 'pending', label: S.lib_stats_pending, value: counts.pending },
    { key: 'converted', label: S.lib_stats_converted, value: counts.converted },
    { key: 'queued', label: S.lib_stats_queued, value: counts.queued },
    { key: 'running', label: S.lib_stats_running, value: counts.running },
    { key: 'quality', label: 'Quality review', value: counts.quality_review },
  ]

  const renameHasResults = renameItems.length > 0
  const renameHasVisibleItems = renameVisible.length > 0
  const hasRenameSelection = selectedRenameCount > 0
  const showUploadWorkbench = uploadWorkbenchOpen && uploadDrafts.length > 0
  const showTaxonomySelectAction = browseMode === 'list' && currentListItems.length > 0
  const showTaxonomyRefreshAction = browseMode === 'list' && visibleAll.length > 0
  const showTaxonomyClearAction = hasActiveTaxonomyFilters
  const showTaxonomyTopActions = showTaxonomySelectAction || showTaxonomyRefreshAction || showTaxonomyClearAction

  const renameWorkbenchSection = (
    <section className="kb-lib-workbench-section kb-lib-workbench-section-rename">
      <div className="kb-lib-section-head">
        <div className="kb-lib-section-copy">
          <Text className="kb-lib-section-title">{S.lib_section_rename}</Text>
        </div>
      </div>

      <div className="kb-lib-rename-summary">
        <div className="kb-lib-rename-summary-main">
          <Select value={renameScope} onChange={setRenameScope} className="kb-lib-rename-scope" options={RENAME_SCOPE_OPTIONS(S)} />
          <Button size="small" className="kb-lib-action-tonal" loading={renameLoading} onClick={() => { void scanRenameSuggestions() }}>
            {renameHasResults ? S.lib_rename_recheck : S.lib_btn_rename_check}
          </Button>
          {renameHasVisibleItems ? (
            <Button className="kb-lib-action-quiet" size="small" onClick={() => setRenameResultsOpen((open) => !open)}>
              {renameResultsOpen ? S.lib_rename_collapse : S.lib_rename_expand}
            </Button>
          ) : null}
          {renameHasVisibleItems ? (
            <Button className="kb-lib-action-quiet" size="small" onClick={selectRenameDiffItems}>{S.lib_btn_select_all}</Button>
          ) : null}
          {hasRenameSelection ? (
            <Button className="kb-lib-action-quiet" size="small" onClick={clearRenameSelection}>{S.lib_btn_clear}</Button>
          ) : null}
          {hasRenameSelection ? (
            <Button className="kb-lib-action-tonal" size="small" type="primary" loading={renameApplying} onClick={() => { void applyRenameSuggestions() }}>
              {S.lib_btn_apply_rename}
            </Button>
          ) : null}
        </div>
        {renameHasResults ? (
          <div className="kb-lib-rename-summary-side">
            <div className="kb-lib-rename-badges">
              <span className="kb-lib-rename-meta">{S.lib_rename_meta_format.replace('{sel}', String(selectedRenameCount)).replace('{vis}', String(renameVisible.length)).replace('{total}', String(renameItems.length))}</span>
            </div>
          </div>
        ) : null}
      </div>

      {renameHasResults && renameHasVisibleItems && renameResultsOpen ? (
        <List
          className="kb-lib-rename-list"
          size="small"
          locale={{ emptyText: <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={S.lib_empty_rename} /> }}
          dataSource={renameVisible}
          pagination={{ pageSize: 6, size: 'small', showSizeChanger: false }}
          renderItem={(item) => (
            <List.Item className="kb-lib-rename-list-item">
              <div className="kb-lib-rename-item">
                <div className="kb-lib-rename-item-head">
                  <Checkbox
                    checked={Boolean(renameSelected[item.name])}
                    onChange={(e) => setRenameSelected((cur) => ({ ...cur, [item.name]: e.target.checked }))}
                  />
                  <Text className="kb-lib-rename-item-name">{item.name}</Text>
                  <Tag color={item.diff ? 'warning' : 'default'}>{item.diff ? S.lib_rename_suggest_rename : S.lib_rename_no_rename}</Tag>
                </div>
                <Input
                  value={renameOverrides[item.name] || ''}
                  onChange={(e) => setRenameOverrides((cur) => ({ ...cur, [item.name]: e.target.value }))}
                  className="kb-lib-rename-item-input"
                />
                <div className="flex flex-wrap items-center gap-2">
                  <Text type="secondary" className="kb-lib-rename-item-source">
                    {item.display_full_name}
                  </Text>
                  {item.meta?.basis_label ? (
                    <Tag color={suggestionBasisTagColor(item.meta)}>
                      {item.meta.basis_label}
                    </Tag>
                  ) : null}
                </div>
                {item.meta?.basis_detail ? (
                  <Text type="secondary" className="kb-lib-rename-item-source">
                    {item.meta.basis_detail}
                  </Text>
                ) : null}
              </div>
            </List.Item>
          )}
        />
      ) : null}
      {renameHasResults && !renameHasVisibleItems ? (
        <Text type="secondary" className="kb-lib-section-note">
          {S.lib_rename_no_files}
        </Text>
      ) : null}
    </section>
  )

  const preparationWorkbench = (
    <Card size="small" className="kb-lib-card kb-lib-workbench-card" title={S.lib_prep_workbench}>
      <div className="kb-lib-workbench">
        <div className="kb-lib-workbench-main">
          <section className="kb-lib-workbench-section">
            <div className="kb-lib-section-head">
              <div className="kb-lib-section-copy">
                <Text className="kb-lib-section-title">{S.lib_section_dir}</Text>
              </div>
              {directoriesConfigured ? (
                <Button className="kb-lib-action-quiet" onClick={() => setDirEditorOpen((open) => !open)}>
                  {showDirEditor ? S.lib_dir_collapse : S.lib_dir_edit}
                </Button>
              ) : null}
            </div>

            <div className="kb-lib-dir-summary">
              <div className={`kb-lib-dir-summary-row${showDirEditor ? ' is-editing' : ''}`}>
                <Text className="kb-lib-dir-summary-label">PDF</Text>
                {showDirEditor ? (
                  <Input
                    value={pdfDirDraft}
                    placeholder={S.lib_dir_select_pdf}
                    onChange={(e) => {
                      setDirTouched(true)
                      setPdfDirDraft(e.target.value)
                    }}
                  />
                ) : (
                  <Text className="kb-lib-dir-summary-path" ellipsis={{ tooltip: pdfDirDraft || S.lib_dir_no_pdf }}>
                    {pdfDirDraft || S.lib_dir_no_pdf}
                  </Text>
                )}
                {showDirEditor ? (
                  <Button className="kb-lib-action-quiet" loading={pickingDir === 'pdf'} onClick={() => { void pickDir('pdf') }}>{S.lib_dir_pick}</Button>
                ) : null}
                <Button className="kb-lib-action-quiet" icon={<FolderOpenOutlined />} disabled={!pdfDirDraft.trim()} onClick={() => { void openFolder('pdf_dir') }}>{S.lib_dir_open}</Button>
              </div>
              <div className={`kb-lib-dir-summary-row${showDirEditor ? ' is-editing' : ''}`}>
                <Text className="kb-lib-dir-summary-label">MD</Text>
                {showDirEditor ? (
                  <Input
                    value={mdDirDraft}
                    placeholder={S.lib_dir_select_md}
                    onChange={(e) => {
                      setDirTouched(true)
                      setMdDirDraft(e.target.value)
                    }}
                  />
                ) : (
                  <Text className="kb-lib-dir-summary-path" ellipsis={{ tooltip: mdDirDraft || S.lib_dir_no_md }}>
                    {mdDirDraft || S.lib_dir_no_md}
                  </Text>
                )}
                {showDirEditor ? (
                  <Button className="kb-lib-action-quiet" loading={pickingDir === 'md'} onClick={() => { void pickDir('md') }}>{S.lib_dir_pick}</Button>
                ) : null}
                <Button className="kb-lib-action-quiet" icon={<FolderOpenOutlined />} disabled={!mdDirDraft.trim()} onClick={() => { void openFolder('md_dir') }}>{S.lib_dir_open}</Button>
              </div>
            </div>

            {showDirEditor ? (
              <div className="kb-lib-section-actions">
                <Button className="kb-lib-action-tonal" type="primary" icon={<SaveOutlined />} loading={savingDirs} disabled={!dirDirty} onClick={() => { void saveDirs() }}>{S.lib_dir_save}</Button>
              </div>
            ) : null}
          </section>

          {renameWorkbenchSection}
        </div>

        <div className="kb-lib-workbench-side">
          <section className="kb-lib-workbench-section kb-lib-workbench-section-upload">
            <div className="kb-lib-section-head">
              <div className="kb-lib-section-copy">
                <Text className="kb-lib-section-title">{S.lib_upload_title}</Text>
              </div>
            </div>

            <Dragger
              multiple
              accept=".pdf"
              disabled={uploadLocked}
              showUploadList={false}
              className={`kb-lib-upload-dropzone${uploadLocked ? ' is-locked' : ''}`}
              beforeUpload={(file) => {
                addDrafts([file as File])
                return false
              }}
            >
              <div className="kb-lib-upload-dropzone-copy">
                <UploadOutlined className="kb-lib-upload-dropzone-icon" />
                <Text className="kb-lib-upload-dropzone-title">{S.lib_upload_drop_hint}</Text>
                <Text type="secondary" className="kb-lib-upload-dropzone-note">{S.lib_upload_click_hint}</Text>
              </div>
            </Dragger>

            {(uploadDrafts.length > 0 || uploadLocked) ? (
              <div className="kb-lib-upload-meta">
                {uploadDrafts.length > 0 ? (
                  <div className="kb-lib-upload-meta-main">
                    <span className="kb-lib-rename-meta">{S.lib_workbench_draft_count.replace('{n}', String(uploadDrafts.length))}</span>
                    <Button className="kb-lib-action-quiet" onClick={() => setUploadWorkbenchOpen((open) => !open)}>
                      {showUploadWorkbench ? S.lib_workbench_hide_queue : S.lib_workbench_upload_queue}
                    </Button>
                  </div>
                ) : null}
                {uploadLocked ? (
                  <Text type="secondary" className="kb-lib-upload-inline-note">
                    {store.converting ? S.lib_upload_locked_converting : S.lib_upload_locked_refsync}
                  </Text>
                ) : null}
              </div>
            ) : null}
          </section>

          <section className="kb-lib-workbench-section kb-lib-workbench-section-process">
            <div className="kb-lib-section-head">
              <div className="kb-lib-section-copy">
                <Text className="kb-lib-section-title">{S.lib_section_batch}</Text>
              </div>
            </div>

            <div className="kb-lib-process-toolbar">
              <div className="kb-lib-process-toolbar-main">
                <Select
                  value={scope}
                  onChange={(value) => { setScope(value); void store.loadFiles(value) }}
                  className="kb-lib-process-scope"
                  options={SCOPE_OPTIONS(S)}
                />
                <Button className="kb-lib-action-tonal" type="primary" onClick={() => { void handleConvertPending() }}>{S.lib_btn_convert_pending_short}</Button>
              </div>
              <div className="kb-lib-process-toolbar-side">
                <Button className="kb-lib-action-quiet kb-lib-process-refresh" icon={<ReloadOutlined />} onClick={() => { void store.loadFiles(scope) }}>{S.lib_btn_refresh}</Button>
                {store.converting ? <Button icon={<StopOutlined />} danger onClick={() => { void store.cancelConvert() }}>{S.lib_btn_stop}</Button> : null}
              </div>
            </div>
          </section>
        </div>
      </div>
    </Card>
  )

  const uploadWorkbenchCard = showUploadWorkbench ? (
    <Card
      size="small"
      className="kb-lib-card kb-lib-upload-workbench-card"
      title={S.lib_section_upload_workbench}
      extra={(
        <Space size={8}>
          <Text type="secondary" className="text-xs">{S.lib_upload_selected_count.replace('{n}', String(selectedUploadCount))}</Text>
          <Text type="secondary" className="text-xs">{S.lib_upload_show_count.replace('{n}', String(filteredUploadDrafts.length)).replace('{total}', String(uploadDrafts.length))}</Text>
          <Button size="small" onClick={() => setUploadWorkbenchOpen(false)}>{S.lib_btn_collapse}</Button>
        </Space>
      )}
    >
      <div className="space-y-3">
        <div className="kb-lib-upload-toolbar flex flex-wrap items-center gap-2">
          <Switch checked={uploadUseLlm} onChange={setUploadUseLlm} />
          <Text className="text-sm text-[var(--muted)]">{S.lib_upload_use_llm}</Text>
          <Select
            value={uploadDraftFilter}
            onChange={(value) => applyUploadFilter(value as UploadDraftFilter)}
            options={uploadDraftFilterOptions}
            className="kb-lib-upload-filter"
          />
          <Tooltip title={S.lib_btn_select_all}><Button icon={<CheckOutlined />} onClick={selectAllUploadDrafts}>{S.lib_btn_select_all}</Button></Tooltip>
          <Tooltip title={S.lib_btn_invert_select}><Button icon={<ClearOutlined />} onClick={invertUploadDraftSelection}>{S.lib_btn_invert_select}</Button></Tooltip>
          <Button loading={uploadInspecting} disabled={uploadLocked} onClick={() => { void inspectSelectedDrafts() }}>{S.lib_btn_scan_selected}</Button>
          <Button loading={uploadSaving} disabled={uploadLocked} onClick={() => { void saveSelectedDrafts(false) }}>{S.lib_btn_save_selected}</Button>
          <Button type="primary" loading={uploadSaving} disabled={uploadLocked} onClick={() => { void saveSelectedDrafts(true) }}>{S.lib_btn_save_and_convert}</Button>
          <Button disabled={uploadLocked} onClick={selectFailedDrafts}>{S.lib_btn_select_failed}</Button>
          <Button disabled={uploadLocked || duplicateFailedDrafts.length === 0} onClick={showDuplicateFailedDrafts}>{S.lib_btn_view_dup_failed}</Button>
          <Button loading={uploadSaving} disabled={uploadLocked || failedUploadDrafts.length === 0} onClick={() => { void retryFailedDrafts(false) }}>{S.lib_btn_retry_failed}</Button>
          <Button type="primary" loading={uploadSaving} disabled={uploadLocked || failedUploadDrafts.length === 0} onClick={() => { void retryFailedDrafts(true) }}>{S.lib_btn_retry_and_convert}</Button>
          <Button disabled={uploadLocked} onClick={() => setUploadDrafts((cur) => cur.filter((x) => x.status !== 'saved'))}>{S.lib_btn_clear_saved}</Button>
        </div>

        {(uploadDraftFilter === 'error' || uploadDraftFilter === 'dup_error') && uploadErrorReason !== 'all' ? (
          <div className="kb-lib-upload-meta flex flex-wrap items-center gap-3">
            <Button size="small" onClick={() => setUploadErrorReason('all')}>
              {S.lib_upload_filter_reason.replace('{reason}', activeErrorReasonText)}
            </Button>
          </div>
        ) : null}

        {failedUploadDrafts.length > 0 ? (
          <Alert
            type="warning"
            showIcon
            message={S.lib_upload_failed_drafts.replace('{n}', String(failedUploadDrafts.length))}
            description={(
              <div className="kb-lib-failed-summary">
                <div className="kb-lib-failed-reasons">
                  {failedReasonBuckets.map((bucket) => (
                    <Button
                      key={bucket.key}
                      size="small"
                      icon={FAILED_REASON_META(S)[bucket.key].icon}
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
                  {failedUploadNotes.length > 0 ? failedUploadNotes.join(' | ') : S.lib_upload_error_hint}
                </Text>
              </div>
            )}
          />
        ) : null}

        <List
          size="small"
          locale={{ emptyText: <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={S.lib_upload_empty} /> }}
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
                      {DRAFT_STATUS_TEXT(S)[x.status]}
                    </Tag>
                    {reasonKey ? (
                      <span className={`kb-lib-inline-reason-chip kb-lib-reason-tone is-${reasonKey}`}>
                        {FAILED_REASON_META(S)[reasonKey].icon}
                        <span>{FAILED_REASON_META(S)[reasonKey].label}</span>
                      </span>
                    ) : null}
                  </div>
                  <div className="flex flex-wrap items-center gap-2 pl-6">
                    <Text type="secondary" className="text-xs">{S.lib_upload_suggest_name}</Text>
                    <Input value={x.stem} onChange={(e) => setUploadDrafts((cur) => cur.map((t) => (t.key === x.key ? { ...t, stem: e.target.value } : t)))} className="w-[24rem] max-w-full" />
                    <Button size="small" disabled={uploadLocked || x.status === 'saving' || x.status === 'inspecting'} onClick={() => { void inspectDraft(x.key) }}>{S.lib_btn_scan}</Button>
                    <Button size="small" disabled={uploadLocked || x.status === 'saving' || x.status === 'saved' || x.status === 'inspecting'} onClick={() => { void saveDraft(x.key, false) }}>{S.lib_btn_save}</Button>
                    <Button size="small" type="primary" disabled={uploadLocked || x.status === 'saving' || x.status === 'saved' || x.status === 'inspecting'} onClick={() => { void saveDraft(x.key, true) }}>{S.lib_btn_save_and_convert}</Button>
                  </div>
                  <div className="flex flex-wrap items-center gap-2 pl-6">
                    <Text type="secondary" className="text-xs">{x.displayName}</Text>
                    {x.suggestionBasisLabel ? (
                      <Tag color={suggestionBasisTagColor({ match_method: x.suggestionMatchMethod, year_source: x.suggestionYearSource })}>
                        {x.suggestionBasisLabel}
                      </Tag>
                    ) : null}
                  </div>
                  {x.suggestionBasisDetail ? (
                    <Text type="secondary" className="block pl-6 text-xs">{x.suggestionBasisDetail}</Text>
                  ) : null}
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
    </Card>
  ) : null

  return (
    <div className="kb-library-page mx-auto w-full max-w-[1760px] space-y-5 p-5">
      <div className="kb-lib-head flex flex-wrap items-end justify-between gap-3">
        <div className="kb-lib-head-main">
          <Text className="text-2xl font-semibold">{S.page_library}</Text>
          <div>
            <Text type="secondary" className="text-sm">{S.lib_page_subtitle}</Text>
          </div>
        </div>
        <Space wrap className="kb-lib-head-actions">
          <Button className="kb-lib-head-btn" icon={<ReloadOutlined />} type="primary" onClick={() => { void handleReindex() }}>{S.reindex_now}</Button>
          <Button className="kb-lib-head-btn" icon={<ReloadOutlined />} onClick={() => { void handleStartRefSync() }}>{S.lib_btn_sync_refs}</Button>
        </Space>
      </div>

      <div className="kb-lib-summary-strip">
        {workbenchStats.map((item) => (
          <div key={item.key} className="kb-lib-summary-chip">
            <Text type="secondary" className="kb-lib-summary-label">{item.label}</Text>
            <div className="kb-lib-summary-value">{item.value}</div>
          </div>
        ))}
      </div>

      {preparationWorkbench}
      {uploadWorkbenchCard}

      <div className="kb-lib-stats-grid grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
        <Card size="small" className="kb-lib-stat"><Text type="secondary">{S.lib_stats_view}</Text><div className="kb-lib-stat-value">{counts.total_view}</div></Card>
        <Card size="small" className="kb-lib-stat"><Text type="secondary">{S.lib_stats_pending}</Text><div className="kb-lib-stat-value">{counts.pending}</div></Card>
        <Card size="small" className="kb-lib-stat"><Text type="secondary">{S.lib_stats_converted}</Text><div className="kb-lib-stat-value">{counts.converted}</div></Card>
        <Card size="small" className="kb-lib-stat"><Text type="secondary">{S.lib_stats_queued}</Text><div className="kb-lib-stat-value">{counts.queued}</div></Card>
        <Card size="small" className="kb-lib-stat"><Text type="secondary">{S.lib_stats_running}</Text><div className="kb-lib-stat-value">{counts.running}</div></Card>
      </div>

      {showStickyStatus ? (
        <Card size="small" className="kb-lib-card kb-lib-sticky-status">
          <div className="kb-lib-sticky-wrap">
            {store.converting && store.progress ? (
              <div className="kb-lib-sticky-item">
                <div className="kb-lib-sticky-main">
                  <Text className="kb-lib-sticky-title">{S.lib_convert_progress.replace('{done}', String(store.progress.completed)).replace('{total}', String(store.progress.total))}</Text>
                  {store.progress.current ? <Text type="secondary" className="kb-lib-sticky-sub">{store.progress.current}</Text> : null}
                  {convertActiveSummary ? <Text type="secondary" className="kb-lib-sticky-sub">{convertActiveSummary}</Text> : null}
                  {convertStageLabel ? <Text type="secondary" className="kb-lib-sticky-sub">{convertStageLabel}</Text> : null}
                  {convertPageProgress.total > 0 ? (
                    <Text type="secondary" className="kb-lib-sticky-sub">
                      {S.lib_convert_page_progress} {convertPageProgress.done}/{convertPageProgress.total}
                    </Text>
                  ) : null}
                </div>
                <div className="kb-lib-sticky-progress-stack">
                  <Progress className="kb-lib-sticky-progress" percent={convertPercent} status="active" size="small" />
                  {convertPageProgress.total > 0 ? (
                    <Progress className="kb-lib-sticky-progress kb-lib-sticky-progress-inner" percent={convertPagePercent} status="active" size="small" />
                  ) : null}
                </div>
                <Button size="small" danger icon={<StopOutlined />} onClick={() => { void store.cancelConvert() }}>
                  {S.lib_btn_stop}
                </Button>
              </div>
            ) : null}

            {store.refSync?.running ? (
              <div className="kb-lib-sticky-item">
                <div className="kb-lib-sticky-main">
                  <Text className="kb-lib-sticky-title">{S.lib_refsync_title}</Text>
                  <Text type="secondary" className="kb-lib-sticky-sub">
                    {store.refSync.current
                      ? `${store.refSync.stage || S.lib_refsync_running} | ${store.refSync.current}`
                      : (store.refSync.message || S.lib_refsync_waiting)}
                  </Text>
                </div>
                <Progress className="kb-lib-sticky-progress" percent={refSyncPercent} status="active" size="small" />
                <Tag color="processing">{S.lib_refsync_running}</Tag>
              </div>
            ) : null}
          </div>
        </Card>
      ) : null}

      <Card size="small" className="kb-lib-card kb-lib-legacy-convert-card" title={S.lib_convert_scope}>
        <div className="kb-lib-convert-shell">
          <div className="kb-lib-convert-row kb-lib-convert-row-top">
            <Select
              value={scope}
              onChange={(value) => { setScope(value); void store.loadFiles(value) }}
              className="kb-lib-convert-scope"
              options={SCOPE_OPTIONS(S)}
            />
            <Input
              value={fileKeyword}
              onChange={(e) => setFileKeyword(e.target.value)}
              allowClear
              prefix={<SearchOutlined className="opacity-50" />}
              placeholder={S.lib_filter_filename}
              className="kb-lib-convert-search"
            />
            <Button className="kb-lib-convert-refresh" icon={<ReloadOutlined />} onClick={() => { void store.loadFiles(scope) }}>
              {S.lib_btn_refresh}
            </Button>
          </div>

          <div className="kb-lib-convert-row kb-lib-convert-row-filters">
            <Select
              value={paperCategoryFilter || undefined}
              allowClear
              placeholder={S.lib_filter_category}
              className="kb-lib-convert-filter"
              options={paperCategoryFilterOptions}
              onChange={(value) => setPaperCategoryFilter(String(value || ''))}
            />
            <Select
              value={paperTagFilter || undefined}
              allowClear
              showSearch
              placeholder={S.lib_filter_tag}
              className="kb-lib-convert-filter"
              options={paperTagFilterOptions}
              optionFilterProp="label"
              onChange={(value) => setPaperTagFilter(String(value || ''))}
            />
            <Select
              value={readingStatusFilter || undefined}
              allowClear
              placeholder={S.lib_filter_reading}
              className="kb-lib-convert-filter"
              options={READING_STATUS_OPTIONS(S).filter((item) => item.value)}
              onChange={(value) => setReadingStatusFilter(String(value || '') as ReadingStatusValue)}
            />
            <Button
              className="kb-lib-convert-refresh"
              onClick={() => {
                setPaperCategoryFilter('')
                setPaperTagFilter('')
                setReadingStatusFilter('')
              }}
            >
              {S.lib_btn_clear_metadata_filter}
            </Button>
          </div>

          <div className="kb-lib-convert-row kb-lib-convert-row-actions">
            <Button type="primary" onClick={() => { void handleConvertPending() }}>{S.lib_btn_convert_pending}</Button>
            {store.converting ? <Button icon={<StopOutlined />} danger onClick={() => { void store.cancelConvert() }}>{S.lib_btn_stop}</Button> : null}
          </div>
        </div>
      </Card>

      {store.refSync && !store.refSync.running && (store.refSync.status === 'error' || Boolean(store.refSync.error)) ? (
        <Card size="small" className="kb-lib-card" title={S.lib_card_refsync}>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Text type="secondary" className="text-xs">
                {store.refSync.current
                  ? `${store.refSync.stage || S.lib_refsync_running} | ${store.refSync.current}`
                  : (store.refSync.message || S.lib_refsync_waiting)}
              </Text>
              <Tag color={store.refSync.running ? 'processing' : (store.refSync.status === 'error' ? 'error' : 'default')}>
                {store.refSync.running ? S.lib_refsync_running : (store.refSync.status === 'idle' ? S.lib_refsync_idle : store.refSync.status)}
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

      {(qualityReportStats.converted > 0 || qualityReportStats.assessed > 0) ? (
        <Card size="small" className="kb-lib-card kb-lib-quality-report" data-testid="library-quality-report">
          <div className="kb-lib-quality-report-head">
            <div className="kb-lib-quality-report-copy">
              <Text className="kb-lib-quality-report-title">{S.lib_quality_report_title}</Text>
              <Text type="secondary" className="kb-lib-quality-report-hint">
                {S.lib_quality_report_hint
                  .replace('{assessed}', String(qualityReportStats.assessed))
                  .replace('{converted}', String(qualityReportStats.converted))
                  .replace('{review}', String(qualityReportStats.review))
                  .replace('{avg}', String(qualityReportStats.avgScore))}
              </Text>
            </div>
            <div className="kb-lib-quality-report-actions">
              <Button
                size="small"
                className="kb-lib-action-quiet"
                disabled={qualityReportStats.review <= 0}
                data-testid="library-quality-report-focus-review"
                onClick={handleFocusQualityReview}
              >
                {S.lib_quality_report_focus_review}
              </Button>
              <Button
                size="small"
                type="primary"
                disabled={qualityRepairRecommendedNames.length <= 0}
                loading={qualityRepairRecommendedNames.some((name) => Boolean(qualityRepairingNames[name]))}
                data-testid="library-quality-report-repair-recommended"
                onClick={() => { void handleRepairRecommendedQuality() }}
              >
                {S.lib_quality_report_repair_top.replace('{n}', String(qualityRepairRecommendedNames.length))}
              </Button>
            </div>
          </div>
          <div className="kb-lib-quality-report-metrics">
            <span className="kb-lib-quality-report-metric is-good" data-testid="library-quality-report-good">
              <span>{S.lib_quality_report_good}</span>
              <strong>{qualityReportStats.good}</strong>
            </span>
            <button
              type="button"
              className="kb-lib-quality-report-metric is-review"
              disabled={qualityReportStats.review <= 0}
              data-testid="library-quality-report-review"
              onClick={handleFocusQualityReview}
            >
              <span>{S.lib_quality_report_review}</span>
              <strong>{qualityReportStats.review}</strong>
            </button>
            <span className="kb-lib-quality-report-metric is-unknown" data-testid="library-quality-report-unknown">
              <span>{S.lib_quality_report_unknown}</span>
              <strong>{qualityReportStats.unknown}</strong>
            </span>
            <span className="kb-lib-quality-report-metric is-score" data-testid="library-quality-report-avg">
              <span>{S.lib_quality_report_avg.replace('{score}', String(qualityReportStats.avgScore))}</span>
            </span>
          </div>
          {qualityRepairImpact ? (
            <div className="kb-lib-quality-repair-impact" data-testid="library-quality-repair-impact">
              <div className="kb-lib-quality-repair-impact-head">
                <Text className="kb-lib-quality-report-section-title">Repair impact</Text>
                <Tag color={qualityRepairImpact.reindexed === true ? 'success' : (qualityRepairImpact.needs_reindex ? (qualityRepairImpact.reindexed === false ? 'warning' : 'processing') : 'success')}>
                  {qualityRepairImpactIndexText(qualityRepairImpact)}
                </Tag>
              </div>
              {qualityRepairRun ? (
                <div className="kb-lib-quality-repair-run" data-testid="library-quality-repair-run">
                  <Tag color={qualityRepairRunTagColor(qualityRepairRun)}>
                    {qualityRepairRunStatusText(qualityRepairRun)}
                  </Tag>
                  <span>{qualityRepairRun.run_id.slice(0, 8)}</span>
                  {qualityRepairRun.detail ? <em>{qualityRepairRun.detail}</em> : null}
                  {qualityVerificationText(qualityRepairRun.verification as Record<string, unknown> | undefined) ? (
                    <em className="kb-lib-quality-repair-run-verification">
                      {qualityVerificationText(qualityRepairRun.verification as Record<string, unknown> | undefined)}
                    </em>
                  ) : null}
                  {qualityRepairRunCanAdvance(qualityRepairRun) ? (
                    <Button
                      size="small"
                      icon={<ReloadOutlined />}
                      loading={qualityRepairAdvancing}
                      data-testid="library-quality-repair-run-advance"
                      onClick={() => { void handleAdvanceQualityRepairRun() }}
                    >
                      Continue
                    </Button>
                  ) : null}
                </div>
              ) : null}
              <div className="kb-lib-quality-repair-impact-grid">
                <span>
                  <em>Repaired</em>
                  <strong>{qualityRepairImpact.repaired}</strong>
                </span>
                <span>
                  <em>Queued</em>
                  <strong>{qualityRepairImpact.enqueued}</strong>
                </span>
                <span>
                  <em>Improved</em>
                  <strong>{qualityRepairImpact.improved}</strong>
                </span>
                <span>
                  <em>Score</em>
                  <strong>Q{qualityRepairImpact.before_avg_score} -&gt; Q{qualityRepairImpact.after_avg_score} ({formatSignedNumber(qualityRepairImpact.score_delta)})</strong>
                </span>
              </div>
              {(qualityRepairImpact.fixed_issue_codes || []).length > 0 || (qualityRepairImpact.remaining_issue_codes || []).length > 0 ? (
                <div className="kb-lib-quality-repair-impact-issues">
                  {(qualityRepairImpact.fixed_issue_codes || []).slice(0, 5).map((issue) => (
                    <span key={`fixed-${issue.name}`} className="is-fixed">{issue.name} x{issue.count}</span>
                  ))}
                  {(qualityRepairImpact.remaining_issue_codes || []).slice(0, 3).map((issue) => (
                    <span key={`remaining-${issue.name}`} className="is-remaining">{issue.name} x{issue.count}</span>
                  ))}
                </div>
              ) : null}
            </div>
          ) : null}
          <div className="kb-lib-quality-domain-section">
            <Text className="kb-lib-quality-report-section-title">{S.lib_quality_domains_title}</Text>
            <div className="kb-lib-quality-domain-grid" data-testid="library-quality-domains">
              {qualityDomainViews.map((domain) => {
                const artifactDomain = domain.key === 'citation_cards' ? 'citation_cards' : 'research_qa'
                return (
                  <div
                    key={domain.key}
                    className={`kb-lib-quality-domain-card is-${domain.status}`}
                    data-quality-domain={domain.key}
                  >
                    <div className="kb-lib-quality-domain-head">
                      <span>{domain.label}</span>
                      <Tag color={domain.status === 'good' ? 'success' : domain.status === 'error' ? 'error' : domain.status === 'warning' ? 'warning' : 'default'}>
                        {domain.statusLabel}
                      </Tag>
                    </div>
                    <strong>{domain.countText}</strong>
                    {domain.detailText ? <span>{domain.detailText}</span> : null}
                    {domain.failureText ? <em>{domain.failureText}</em> : null}
                    <div className="kb-lib-quality-domain-actions">
                      {domain.key === 'conversion' ? (
                        <Button
                          size="small"
                          className="kb-lib-quality-domain-action"
                          disabled={qualityReportStats.review <= 0}
                          onClick={handleFocusQualityReview}
                        >
                          {S.lib_quality_report_focus_review}
                        </Button>
                      ) : (
                        <>
                          <Button
                            size="small"
                            className="kb-lib-quality-domain-action"
                            loading={qualityArtifactOpening === `${artifactDomain}:${domain.available ? 'report' : 'runbook'}`}
                            onClick={() => {
                              void openQualityArtifact(artifactDomain, domain.available ? 'report' : 'runbook')
                            }}
                          >
                            {domain.available ? S.lib_quality_artifact_open_report : S.lib_quality_artifact_open_runbook}
                          </Button>
                          {domain.available ? (
                            <Button
                              size="small"
                              className="kb-lib-quality-domain-action"
                              loading={qualityArtifactOpening === `${artifactDomain}:folder`}
                              onClick={() => {
                                void openQualityArtifact(artifactDomain, 'folder')
                              }}
                            >
                              {S.lib_quality_artifact_open_folder}
                            </Button>
                          ) : null}
                        </>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
          {qualityFeatureHealth && qualityFeatureHealthItems.length > 0 ? (
            <div
              className={`kb-lib-quality-feature-health is-${normalizeTextValue(qualityFeatureHealth.status).toLowerCase() || 'unknown'}`}
              data-testid="library-quality-feature-health"
            >
              <div className="kb-lib-quality-feature-health-head">
                <div>
                  <Text className="kb-lib-quality-report-section-title">Feature health</Text>
                  <strong>Q{Math.max(0, Math.min(100, Math.round(Number(qualityFeatureHealth.score || 0))))}</strong>
                </div>
                <Tag color={qualityFeatureHealth.status === 'good' ? 'success' : qualityFeatureHealth.status === 'error' ? 'error' : qualityFeatureHealth.status === 'warning' ? 'warning' : 'default'}>
                  {qualityStatusText(normalizeTextValue(qualityFeatureHealth.status).toLowerCase(), S)}
                </Tag>
              </div>
              {qualityFeatureHealth.summary ? <p>{qualityFeatureHealth.summary}</p> : null}
              <div className="kb-lib-quality-feature-health-grid">
                {qualityFeatureHealthItems.map((item) => {
                  const featureStatus = normalizeTextValue(item.status).toLowerCase() || 'unknown'
                  const featureAction = qualityFeatureActionText(item)
                  const featureStageKey = normalizeTextValue(item.target_stage || item.key).toLowerCase()
                  const featureStageResult = qualityFullChainResults[featureStageKey] || qualityFullChainPersistedResults[featureStageKey]
                  return (
                    <div
                      key={item.key}
                      className={`kb-lib-quality-feature-card is-${featureStatus}${item.blocking ? ' is-blocking' : ''}`}
                      data-quality-feature={item.key}
                      data-testid="library-quality-feature-card"
                    >
                      <div className="kb-lib-quality-feature-card-head">
                        <span>{item.label}</span>
                        <Tag color={featureStatus === 'good' ? 'success' : featureStatus === 'error' ? 'error' : featureStatus === 'warning' ? 'warning' : 'default'}>
                          {qualityStatusText(featureStatus, S)}
                        </Tag>
                      </div>
                      <strong>{item.summary || item.detail}</strong>
                      {item.detail ? <span>{item.detail}</span> : null}
                      {featureStageResult?.deltaText || featureStageResult?.verificationText ? (
                        <div
                          className={`kb-lib-quality-feature-result is-${featureStageResult.status}`}
                          data-testid="library-quality-feature-result"
                        >
                          {featureStageResult.deltaText ? <span>{featureStageResult.deltaText}</span> : null}
                          {featureStageResult.verificationText ? <em>{featureStageResult.verificationText}</em> : null}
                        </div>
                      ) : null}
                      <div className="kb-lib-quality-feature-card-foot">
                        <em>Q{Math.max(0, Math.min(100, Math.round(Number(item.score || 0))))}</em>
                        <button
                          type="button"
                          className="kb-lib-quality-feature-action"
                          data-testid="library-quality-feature-action"
                          onClick={() => { void handleQualityFeatureHealthAction(item) }}
                        >
                          {featureAction}
                        </button>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          ) : null}
          {qualityFullChain ? (
            <div
              className={`kb-lib-quality-full-chain is-${normalizeTextValue(qualityFullChain.status).toLowerCase() || 'unknown'}`}
              data-testid="library-quality-full-chain"
            >
              <div className="kb-lib-quality-full-chain-head">
                <div>
                  <Text className="kb-lib-quality-report-section-title">Full-chain health</Text>
                  <strong>Q{Math.max(0, Math.min(100, Math.round(Number(qualityFullChain.score || 0))))}</strong>
                </div>
                <Tag color={qualityFullChain.status === 'good' ? 'success' : qualityFullChain.status === 'error' ? 'error' : qualityFullChain.status === 'warning' ? 'warning' : 'default'}>
                  {qualityStatusText(normalizeTextValue(qualityFullChain.status).toLowerCase(), S)}
                </Tag>
              </div>
              {qualityFullChain.summary ? (
                <p>{qualityFullChain.summary}</p>
              ) : null}
              {qualityFullChainStages.length > 0 ? (
                <div className="kb-lib-quality-full-chain-stages">
                  {qualityFullChainStages.map((stage) => {
                    const stageStatus = normalizeTextValue(stage.status).toLowerCase() || 'unknown'
                    const stageCount = Number(stage.count || 0)
                    const stageActionText = qualityFullChainActionText(stage)
                    const cleanStageKey = normalizeTextValue(stage.key).toLowerCase()
                    const stageBusy = qualityFullChainActionKey === cleanStageKey
                    const stageResult = qualityFullChainResults[cleanStageKey] || qualityFullChainPersistedResults[cleanStageKey]
                    return (
                      <div
                        key={stage.key}
                        className={`kb-lib-quality-full-chain-stage is-${stageStatus}${stage.blocking ? ' is-blocking' : ''}`}
                        data-quality-stage={stage.key}
                        data-testid="library-quality-full-chain-stage"
                      >
                        <div>
                          <span>{stage.label}</span>
                          <Tag color={stageStatus === 'good' ? 'success' : stageStatus === 'error' ? 'error' : stageStatus === 'warning' ? 'warning' : 'default'}>
                            {qualityStatusText(stageStatus, S)}
                          </Tag>
                        </div>
                        <strong>{stage.detail || normalizeTextValue(stage.action).replace(/_/g, ' ')}</strong>
                        <div className="kb-lib-quality-full-chain-stage-foot">
                          <em>{stageCount > 0 ? `${stageCount}` : normalizeTextValue(stage.action).replace(/_/g, ' ')}</em>
                          <button
                            type="button"
                            className="kb-lib-quality-full-chain-stage-action"
                            disabled={Boolean(qualityFullChainActionKey) && !stageBusy}
                            data-testid="library-quality-full-chain-stage-action"
                            onClick={() => { void handleQualityFullChainStage(stage) }}
                          >
                            {stageBusy ? 'Working' : stageActionText}
                          </button>
                        </div>
                        {stageResult ? (
                          <div
                            className={`kb-lib-quality-full-chain-stage-result is-${stageResult.status}`}
                            data-testid="library-quality-full-chain-stage-result"
                          >
                            <span>{stageResult.summary}</span>
                            {stageResult.detail ? <em>{stageResult.detail}</em> : null}
                            {stageResult.deltaText ? <em>{stageResult.deltaText}</em> : null}
                            {stageResult.verificationText ? <em>{stageResult.verificationText}</em> : null}
                          </div>
                        ) : null}
                      </div>
                    )
                  })}
                </div>
              ) : null}
              {qualityFullChainRootCauses.length > 0 ? (
                <div className="kb-lib-quality-full-chain-roots">
                  {qualityFullChainRootCauses.map((cause) => {
                    const causeSeverity = normalizeTextValue(cause.severity).toLowerCase() || 'warning'
                    return (
                      <em
                        key={`${cause.domain}-${cause.code}`}
                        className={`is-${causeSeverity}`}
                        data-testid="library-quality-full-chain-root-cause"
                      >
                        {cause.label || cause.code}
                        <span>{cause.code} x{Number(cause.count || 0)}</span>
                      </em>
                    )
                  })}
                </div>
              ) : null}
              {qualityFullChainActionHistory.length > 0 ? (
                <div className="kb-lib-quality-full-chain-history" data-testid="library-quality-full-chain-history">
                  <Text className="kb-lib-quality-report-section-title">Recent actions</Text>
                  <div className="kb-lib-quality-full-chain-history-list">
                    {qualityFullChainActionHistory.slice(0, 4).map((item) => {
                      const actionStatus = normalizeTextValue(item.status).toLowerCase() || 'info'
                      const createdAt = Number(item.created_at || 0) * 1000
                      const actionText = qualityActionHistoryActionText(item)
                      const deltaText = qualityActionDeltaText(item)
                      return (
                        <div
                          key={item.id || `${item.stage_key}-${item.created_at}-${item.summary}`}
                          className={`kb-lib-quality-full-chain-history-row is-${actionStatus}`}
                          data-testid="library-quality-full-chain-history-row"
                        >
                          <span>{item.stage_label || item.stage_key}</span>
                          <strong>{item.summary}</strong>
                          <em>{deltaText || formatQualityRepairHistoryTime(createdAt)}</em>
                          <button
                            type="button"
                            className="kb-lib-quality-full-chain-history-open"
                            data-testid="library-quality-full-chain-history-open"
                            onClick={() => { void handleQualityActionHistoryOpen(item) }}
                          >
                            {actionText}
                          </button>
                        </div>
                      )
                    })}
                  </div>
                </div>
              ) : null}
            </div>
          ) : null}
          {qualityPriorityActions.length > 0 ? (
            <div className="kb-lib-quality-priority-actions" data-testid="library-quality-priority-actions">
              <Text className="kb-lib-quality-report-section-title">{S.lib_quality_priority_actions}</Text>
              <div className="kb-lib-quality-priority-list">
                {qualityPriorityActions.map((action) => {
                  const severity = normalizeTextValue(action.severity || 'warning').toLowerCase() || 'warning'
                  const count = Number(action.count || 0)
                  return (
                    <button
                      key={`${action.domain}-${action.label}`}
                      type="button"
                      className={`kb-lib-quality-priority-pill is-${severity}`}
                      data-quality-action-domain={action.domain}
                      onClick={() => { void handleQualityPriorityAction(action) }}
                    >
                      <strong>{qualityActionText(action, S)}</strong>
                      <em>{count > 0 ? String(count) : qualityStatusText(severity, S)}</em>
                    </button>
                  )
                })}
              </div>
            </div>
          ) : null}
          {qualityRerunSummary?.available ? (
            <div className="kb-lib-quality-rerun-summary" data-testid="library-quality-rerun-summary">
              <Text className="kb-lib-quality-report-section-title">Rerun history</Text>
              <div className="kb-lib-quality-rerun-summary-grid">
                <span>Runs <strong>{qualityRerunSummary.total}</strong></span>
                <span>Passed <strong>{qualityRerunSummary.passed}</strong></span>
                <span>Failed <strong>{qualityRerunSummary.failed + qualityRerunSummary.error}</strong></span>
                <span>Cases <strong>{qualityRerunSummary.case_count}</strong></span>
              </div>
              {qualityRerunSummary.top_failures?.length ? (
                <div className="kb-lib-quality-rerun-summary-failures">
                  {qualityRerunSummary.top_failures.slice(0, 3).map((item) => (
                    <em key={item.name}>{item.name} x{item.count}</em>
                  ))}
                </div>
              ) : null}
            </div>
          ) : null}
          {qualityFailureCases.length > 0 ? (
            <div className="kb-lib-quality-failure-cases" data-testid="library-quality-failure-cases">
              <div className="kb-lib-quality-failure-head">
                <Text className="kb-lib-quality-report-section-title">
                  {S.lib_quality_failure_cases.replace('{n}', String(qualityFailureCases.length))}
                </Text>
                <Button
                  size="small"
                  className="kb-lib-quality-domain-action"
                  loading={qualityArtifactOpening === 'research_qa:report'}
                  onClick={() => { void openQualityArtifact('research_qa', 'report') }}
                >
                  {S.lib_quality_failure_open_report}
                </Button>
              </div>
              {qualityFailureFilters.length > 0 ? (
                <div className="kb-lib-quality-failure-filters">
                  <button
                    type="button"
                    className={`kb-lib-quality-failure-filter${!qualityFailureFilter ? ' is-active' : ''}`}
                    data-testid="library-quality-failure-filter-all"
                    onClick={() => setQualityFailureFilter('')}
                  >
                    {S.lib_quality_failure_all}
                  </button>
                  {qualityFailureFilters.map((item) => (
                    <button
                      key={item.name}
                      type="button"
                      className={`kb-lib-quality-failure-filter${qualityFailureFilter === item.name ? ' is-active' : ''}`}
                      data-testid="library-quality-failure-filter"
                      onClick={() => setQualityFailureFilter(item.name)}
                    >
                      <span>{item.name}</span>
                      <strong>{item.count}</strong>
                    </button>
                  ))}
                </div>
              ) : null}
              {visibleQualityFailureCases.length > 0 ? (
                <div className="kb-lib-quality-failure-list">
                  {visibleQualityFailureCases.slice(0, 4).map((item) => {
                    const docIds = Array.isArray(item.doc_ids) ? item.doc_ids.filter(Boolean) : []
                    const failures = Array.isArray(item.failures) ? item.failures : []
                    const missingDocIds = Array.isArray(item.missing_expected_doc_ids) ? item.missing_expected_doc_ids.filter(Boolean) : []
                    const routeSummary = item.diagnostic_summary?.citation_routes || {}
                    const rootCauses = Array.isArray(item.root_causes) && item.root_causes.length > 0
                      ? item.root_causes
                      : failures.slice(0, 2).map((failure) => ({
                        code: failure.name,
                        label: failure.name,
                        severity: failure.domain === 'citation_cards' ? 'error' : 'warning',
                        detail: failure.detail || '',
                        action: 'inspect_replay',
                      }))
                    const sourceDiagnostics = Array.isArray(item.source_diagnostics) ? item.source_diagnostics : []
                    const repairActions = Array.isArray(item.repair_actions) && item.repair_actions.length > 0
                      ? item.repair_actions
                      : [
                        { id: 'open_replay', kind: 'open_replay', label: 'Open replay', severity: 'warning', enabled: true, detail: '' },
                        { id: 'rerun_case', kind: 'rerun_case', label: 'Rerun case', severity: 'warning', enabled: true, detail: '' },
                        { id: 'open_report', kind: 'open_artifact', target: 'report', label: 'Open report', severity: 'warning', enabled: true, detail: '' },
                      ]
                    const rerunResult = qualityCaseRerunResults[item.id]
                    const persistedRerun = item.rerun_status?.available ? item.rerun_status : null
                    const rerunView = rerunResult
                      ? {
                        label: 'Rerun',
                        status: rerunResult.status,
                        failures: (rerunResult.failures || []).map((failure) => failure.name),
                        latencyMs: rerunResult.latency_ms,
                        consecutiveFailed: 0,
                      }
                      : persistedRerun
                        ? {
                          label: 'Last rerun',
                          status: persistedRerun.last_status,
                          failures: persistedRerun.failure_names || [],
                          latencyMs: persistedRerun.last_latency_ms,
                          consecutiveFailed: persistedRerun.consecutive_failed,
                        }
                        : null
                    return (
                      <div
                        key={item.id}
                        role="button"
                        tabIndex={0}
                        className="kb-lib-quality-failure-case"
                        data-testid="library-quality-failure-case"
                        onClick={() => openResearchQaReplayCase(item)}
                        onKeyDown={(event) => {
                          if (event.key === 'Enter' || event.key === ' ') {
                            event.preventDefault()
                            openResearchQaReplayCase(item)
                          }
                        }}
                      >
                        <span className="kb-lib-quality-failure-case-title">{item.id}</span>
                        <span className="kb-lib-quality-failure-case-question">{item.question || S.lib_quality_failure_question_empty}</span>
                        <span className="kb-lib-quality-failure-case-badges">
                          {failures.slice(0, 3).map((failure) => (
                            <em key={`${item.id}-${failure.name}`}>{failure.name}</em>
                          ))}
                        </span>
                        {docIds.length > 0 ? (
                          <span className="kb-lib-quality-failure-case-docs">
                            {S.lib_quality_failure_case_docs.replace('{docs}', docIds.slice(0, 4).join(' / '))}
                          </span>
                        ) : null}
                        <span className="kb-lib-quality-failure-case-diagnostics">
                          {missingDocIds.length > 0 ? <em>Missing {missingDocIds.slice(0, 3).join(' / ')}</em> : null}
                          <em>A {routeSummary.system_a || 0} / B {routeSummary.system_b || 0}</em>
                          {sourceDiagnostics.length > 0 ? <em>Sources {sourceDiagnostics.length}</em> : null}
                        </span>
                        <span className="kb-lib-quality-root-causes" data-testid="library-quality-root-causes">
                          {rootCauses.slice(0, 3).map((cause) => {
                            const severity = normalizeTextValue(cause.severity).toLowerCase() || 'warning'
                            return (
                              <em key={`${item.id}-${cause.code}`} className={`is-${severity}`}>
                                {cause.label}
                              </em>
                            )
                          })}
                        </span>
                        {sourceDiagnostics.length > 0 ? (
                          <span className="kb-lib-quality-source-diag" data-testid="library-quality-source-diagnostics">
                            {sourceDiagnostics.slice(0, 2).map((source) => {
                              const status = normalizeTextValue(source.quality_status).toLowerCase() || 'unknown'
                              const label = source.title || source.source_name || source.source_path || 'source'
                              return (
                                <em key={`${item.id}-${source.source_path || source.source_name}`} className={`is-${status}`}>
                                  {label}{source.quality_score > 0 ? ` Q${source.quality_score}` : ''}
                                </em>
                              )
                            })}
                          </span>
                        ) : null}
                        {rerunView ? (
                          <span className={`kb-lib-quality-rerun-result is-${rerunView.status}`} data-testid="library-quality-rerun-result">
                            {rerunView.label} {rerunView.status}
                            {rerunView.consecutiveFailed ? ` · ${rerunView.consecutiveFailed}x failing` : ''}
                            {rerunView.failures?.length ? ` · ${rerunView.failures.slice(0, 2).join(' / ')}` : ''}
                            {rerunView.latencyMs ? ` · ${Math.round(rerunView.latencyMs / 1000)}s` : ''}
                          </span>
                        ) : null}
                        <span className="kb-lib-quality-failure-actions" data-testid="library-quality-failure-actions">
                          {repairActions.slice(0, 6).map((action) => {
                            const loadingKey = `${item.id}:${action.kind}:${action.target || ''}`
                            return (
                              <Button
                                key={`${item.id}-${action.id || action.kind}`}
                                size="small"
                                className="kb-lib-quality-failure-action"
                                disabled={action.enabled === false}
                                loading={
                                  qualityCaseActionKey === loadingKey
                                  || (action.kind === 'repair_sources' && qualityCaseActionKey === `${item.id}:repair_sources`)
                                }
                                onClick={(event) => {
                                  event.stopPropagation()
                                  void handleQualityFailureAction(item, action)
                                }}
                              >
                                {action.label}
                              </Button>
                            )
                          })}
                          <Button
                            size="small"
                            className="kb-lib-quality-failure-action"
                            onClick={(event) => {
                              event.stopPropagation()
                              void copyQualityFailureSummary(item)
                            }}
                          >
                            Copy summary
                          </Button>
                        </span>
                      </div>
                    )
                  })}
                </div>
              ) : (
                <Text type="secondary" className="kb-lib-quality-report-empty">{S.lib_quality_failure_no_match}</Text>
              )}
            </div>
          ) : null}
          <div className="kb-lib-quality-report-body">
            <div className="kb-lib-quality-report-section">
              <Text className="kb-lib-quality-report-section-title">{S.lib_quality_report_top_issues}</Text>
              {qualityIssueStats.length > 0 ? (
                <div className="kb-lib-quality-report-issues">
                  {qualityIssueStats.map((issue) => (
                    <button
                      key={issue.key}
                      type="button"
                      className={`kb-lib-quality-report-issue is-${issue.severity || 'warning'}`}
                      data-testid="library-quality-report-issue"
                      onClick={() => handleFocusQualityIssue(issue.label)}
                    >
                      <span>{issue.label}</span>
                      {issue.repairStrategy ? <em>{issue.repairStrategy}</em> : null}
                      <strong>{S.lib_quality_report_papers.replace('{n}', String(issue.papers))}</strong>
                    </button>
                  ))}
                </div>
              ) : (
                <Text type="secondary" className="kb-lib-quality-report-empty">{S.lib_quality_report_no_issues}</Text>
              )}
            </div>
            <div className="kb-lib-quality-report-section">
              <Text className="kb-lib-quality-report-section-title">{S.lib_quality_report_recommended}</Text>
              {qualityReportRecommendations.length > 0 ? (
                <div className="kb-lib-quality-report-recommendations" data-testid="library-quality-report-recommended">
                  {qualityReportRecommendations.slice(0, 3).map((item) => (
                    <button
                      key={item.name}
                      type="button"
                      className="kb-lib-quality-report-recommendation"
                      onClick={() => focusQualityHistoryNames([item.name])}
                    >
                      <span className="kb-lib-quality-report-rec-title">{stripKnownSourceExt(item.name) || item.name}</span>
                      <span className="kb-lib-quality-report-rec-meta">
                        Q{item.score}
                        {item.issues.length > 0 ? ` · ${item.issues.join(' / ')}` : ''}
                      </span>
                    </button>
                  ))}
                </div>
              ) : (
                <Text type="secondary" className="kb-lib-quality-report-empty">{S.lib_quality_report_no_issues}</Text>
              )}
            </div>
          </div>
        </Card>
      ) : null}

      {qualityRepairHistoryList.length > 0 ? (
        <Card size="small" className="kb-lib-card kb-lib-quality-history" data-testid="library-quality-history">
          <div className="kb-lib-quality-history-head">
            <div>
              <Text className="kb-lib-quality-history-title">{S.lib_quality_history_title}</Text>
              <Text type="secondary" className="kb-lib-quality-history-hint">
                {S.lib_quality_history_hint
                  .replace('{n}', String(qualityRepairHistoryStats.total))
                  .replace('{delta}', String(qualityRepairHistoryStats.avgDelta >= 0 ? `+${qualityRepairHistoryStats.avgDelta}` : qualityRepairHistoryStats.avgDelta))
                  .replace('{issues}', String(qualityRepairHistoryStats.fixedCount))}
              </Text>
            </div>
            <div className="kb-lib-quality-history-side">
              <div className="kb-lib-quality-history-metrics">
                <span data-testid="library-quality-history-count">{S.lib_quality_history_count.replace('{n}', String(qualityRepairHistoryStats.total))}</span>
                <span>{S.lib_quality_history_improved.replace('{n}', String(qualityRepairHistoryStats.improved))}</span>
              </div>
              <div className="kb-lib-quality-history-actions">
                <Button
                  size="small"
                  className="kb-lib-action-quiet"
                  disabled={qualityHistoryRemainingNames.length <= 0}
                  data-testid="library-quality-history-focus-remaining"
                  onClick={handleFocusQualityHistoryRemaining}
                >
                  {S.lib_quality_history_focus_remaining}
                </Button>
                <Button
                  size="small"
                  type="primary"
                  disabled={qualityRepairRecommendedNames.length <= 0}
                  loading={qualityRepairRecommendedNames.some((name) => Boolean(qualityRepairingNames[name]))}
                  data-testid="library-quality-history-repair-recommended"
                  onClick={() => { void handleRepairRecommendedQuality() }}
                >
                  {S.lib_quality_history_repair_recommended.replace('{n}', String(qualityRepairRecommendedNames.length))}
                </Button>
                {qualityHistoryFocusNames.length > 0 ? (
                  <Button
                    size="small"
                    className="kb-lib-action-quiet"
                    data-testid="library-quality-history-clear-focus"
                    onClick={() => setQualityHistoryFocusNames([])}
                  >
                    {S.lib_quality_history_clear_focus}
                  </Button>
                ) : null}
              </div>
            </div>
          </div>
          <div className="kb-lib-quality-history-list">
            {qualityRepairHistoryList.slice(0, 4).map((record) => (
              <div key={`${record.name}-${record.updatedAt}`} className="kb-lib-quality-history-row" data-testid="library-quality-history-row">
                <button
                  type="button"
                  className="kb-lib-quality-history-paper"
                  title={record.name}
                  data-testid="library-quality-history-paper"
                  onClick={() => focusQualityHistoryNames([record.name])}
                >
                  {stripKnownSourceExt(record.name) || record.name}
                </button>
                <div className="kb-lib-quality-history-result">
                  <span className="kb-lib-quality-history-score">Q{record.beforeScore} -&gt; Q{record.afterScore}</span>
                  {record.fixedIssues.length > 0 ? (
                    <span className="kb-lib-quality-history-fixed">
                      {S.lib_quality_history_fixed.replace('{issues}', record.fixedIssues.slice(0, 2).join(' / '))}
                    </span>
                  ) : null}
                  {record.remainingIssues.length > 0 ? (
                    <span className="kb-lib-quality-history-remaining">
                      {S.lib_quality_history_remaining.replace('{n}', String(record.remainingIssues.length))}
                    </span>
                  ) : null}
                </div>
                <div className="kb-lib-quality-history-time">{formatQualityRepairHistoryTime(record.updatedAt)}</div>
              </div>
            ))}
          </div>
        </Card>
      ) : null}

      <Card size="small" className="kb-lib-card kb-lib-taxonomy-bar" title={S.lib_taxonomy_title}>
        <div className="kb-lib-taxonomy-shell">
          <div className="kb-lib-taxonomy-top">
            <div className="kb-lib-taxonomy-view">
              <Segmented
                className="kb-lib-browse-switch"
                value={browseMode}
                onChange={(value) => setBrowseMode(value as LibraryBrowseMode)}
                options={[
                  { label: S.lib_browse_list, value: 'list' },
                  { label: S.lib_browse_categories, value: 'categories' },
                  { label: S.lib_browse_tags, value: 'tags' },
                ]}
              />
            </div>
            <div className="kb-lib-taxonomy-meta">
              <div className="kb-lib-taxonomy-summary">
                <Text type="secondary" className="kb-lib-taxonomy-result">
                  {S.lib_taxonomy_result.replace('{n}', String(visibleAll.length)).replace('{total}', String(store.files.length))}
                </Text>
                {hasActiveTaxonomyFilters ? (
                  <span className="kb-lib-taxonomy-status-pill">
                    {S.lib_taxonomy_filtering.replace('{n}', String(activeTaxonomyFilterCount))}
                  </span>
                ) : null}
              </div>
              {showTaxonomyTopActions ? (
                <div className="kb-lib-taxonomy-top-actions">
                  {showTaxonomySelectAction ? (
                    <Button className="kb-lib-action-quiet" onClick={selectCurrentListItems}>
                      {S.lib_btn_select_current_list}
                    </Button>
                  ) : null}
                  {showTaxonomyRefreshAction ? (
                    <Button
                      className="kb-lib-action-tonal"
                      loading={suggestionsRefreshing}
                      onClick={() => { void regenerateSuggestionsForVisible() }}
                    >
                      {S.lib_btn_refresh_suggestions}
                    </Button>
                  ) : null}
                  {showTaxonomyClearAction ? (
                    <Button className="kb-lib-action-quiet" onClick={clearTaxonomyFilters}>
                      {S.lib_btn_clear_filters}
                    </Button>
                  ) : null}
                </div>
              ) : null}
            </div>
          </div>

          <div className="kb-lib-taxonomy-controls">
            <div className="kb-lib-taxonomy-filters">
              <Input
                value={fileKeyword}
                onChange={(e) => setFileKeyword(e.target.value)}
                allowClear
                prefix={<SearchOutlined className="opacity-50" />}
                placeholder={S.lib_search_placeholder}
                className="kb-lib-taxonomy-search"
              />
              <Select
                value={paperCategoryFilter || undefined}
                allowClear
                placeholder={S.lib_search_category}
                className="kb-lib-taxonomy-select"
                options={paperCategoryFilterOptions}
                onChange={(value) => applyPaperCategoryFilter(String(value || ''))}
              />
              <Select
                value={paperTagFilter || undefined}
                allowClear
                showSearch
                placeholder={S.lib_search_tag}
                className="kb-lib-taxonomy-select"
                options={paperTagFilterOptions}
                optionFilterProp="label"
                onChange={(value) => applyPaperTagFilter(String(value || ''))}
              />
              <Select
                value={readingStatusFilter || undefined}
                allowClear
                placeholder={S.lib_search_reading}
                className="kb-lib-taxonomy-select"
                options={READING_STATUS_OPTIONS(S).filter((item) => item.value)}
                onChange={(value) => setReadingStatusFilter(String(value || '') as ReadingStatusValue)}
              />
            </div>

            <div className="kb-lib-taxonomy-quick">
              <div className="kb-lib-taxonomy-toggle-row">
                <button
                  type="button"
                  className={`kb-lib-taxonomy-pill is-status${onlyUnread ? ' is-active' : ''}`}
                  onClick={() => setOnlyUnread((value) => !value)}
                >
                  {S.lib_taxonomy_unread}
                </button>
                <button
                  type="button"
                  className={`kb-lib-taxonomy-pill is-category${onlyUnclassified ? ' is-active' : ''}`}
                  onClick={() => {
                    const next = !onlyUnclassified
                    setOnlyUnclassified(next)
                    if (next) setPaperCategoryFilter('')
                  }}
                >
                  {S.lib_category_unclassified}
                </button>
                <button
                  type="button"
                  className={`kb-lib-taxonomy-pill is-suggestion${onlySuggested ? ' is-active' : ''}`}
                  onClick={() => setOnlySuggested((value) => !value)}
                >
                  {S.lib_taxonomy_has_suggestions}
                </button>
                <button
                  type="button"
                  className={`kb-lib-taxonomy-pill is-quality${onlyQualityIssues ? ' is-active' : ''}`}
                  data-testid="library-quality-issues-filter"
                  onClick={() => setOnlyQualityIssues((value) => !value)}
                >
                  Quality review {qualityReviewCount}
                </button>
                {qualityHistoryFocusNames.length > 0 ? (
                  <button
                    type="button"
                    className="kb-lib-taxonomy-pill is-quality is-active"
                    data-testid="library-quality-history-active-filter"
                    onClick={() => setQualityHistoryFocusNames([])}
                  >
                    {S.lib_quality_history_focus_badge.replace('{n}', String(qualityHistoryFocusNames.length))}
                  </button>
                ) : null}
              </div>
            </div>
          </div>
        </div>
      </Card>

      {browseMode === 'list' && selectedLibraryCount > 0 ? (
        <Card size="small" className="kb-lib-card kb-lib-batch-card">
          <div className="kb-lib-batch-bar">
            <div className="kb-lib-batch-summary">
              <div className="kb-lib-batch-badges">
                <span className="kb-lib-batch-badge is-strong">{S.lib_batch_selected_count.replace('{n}', String(selectedLibraryCount))}</span>
                <span className="kb-lib-batch-badge">{S.lib_batch_current_count.replace('{n}', String(currentListItems.length))}</span>
              </div>
              <Text className="kb-lib-batch-count">{S.lib_batch_title_selected}</Text>
              <Text type="secondary" className="kb-lib-batch-hint">{S.lib_batch_hint_scope}</Text>
            </div>
            <div className="kb-lib-batch-actions">
              <Button onClick={selectCurrentListItems}>{S.lib_btn_select_current_list}</Button>
              <Button onClick={clearLibrarySelection} disabled={!selectedLibraryCount}>{S.lib_btn_clear_selection}</Button>
              {selectedQualityReviewNames.length > 0 ? (
                <Button
                  icon={<ReloadOutlined />}
                  loading={selectedQualityReviewNames.some((name) => Boolean(qualityRepairingNames[name]))}
                  onClick={() => { void handleRepairSelectedQuality() }}
                  data-testid="library-quality-repair-selected"
                >
                  {S.lib_btn_repair_quality_selected}
                </Button>
              ) : null}
              <Button type="primary" onClick={openBatchEditor} disabled={!selectedLibraryCount}>{S.lib_batch_title}</Button>
            </div>
          </div>
        </Card>
      ) : null}

      {browseMode === 'list' ? (
        <Tabs
        className="kb-lib-tabs"
        activeKey={tabKey}
        onChange={(key) => setTabKey(key as FileTabKey)}
        items={[
          { key: 'pending', label: S.lib_tab_pending.replace('{n}', String(visiblePending.length)), children: renderFiles(visiblePending, S.lib_empty_pending) },
          { key: 'converted', label: S.lib_tab_converted.replace('{n}', String(visibleConverted.length)), children: renderFiles(visibleConverted, S.lib_empty_converted) },
          { key: 'all', label: S.lib_tab_all.replace('{n}', String(visibleAll.length)), children: renderFiles(visibleAll, S.lib_empty_all) },
        ]}
      />
      ) : browseMode === 'categories' ? (
        <Card size="small" className="kb-lib-card">
          {renderCategoriesView()}
        </Card>
      ) : (
        <Card size="small" className="kb-lib-card">
          {renderTagsView()}
        </Card>
      )}

      <Drawer
        title={metaItem ? S.lib_meta_title.replace('{name}', metaItem.name) : S.lib_meta_title_fallback}
        open={metaDrawerOpen}
        width={420}
        onClose={() => setMetaDrawerOpen(false)}
        destroyOnClose={false}
      >
        <div className="kb-lib-meta-drawer">
          {metaItem ? (
            <div className="kb-lib-meta-hero">
              <div className="kb-lib-meta-hero-copy">
                <Text className="kb-lib-meta-hero-title">{stripKnownSourceExt(metaItem.name) || metaItem.name}</Text>
                <Text type="secondary" className="kb-lib-meta-hero-note">
                  {S.lib_meta_hero_hint}
                </Text>
              </div>
              <Space wrap size={[6, 6]} className="kb-lib-meta-chip-row">
                <Tag color={metaDraftCategory ? 'blue' : 'default'}>{metaDraftCategory || S.lib_category_unclassified}</Tag>
                {metaDraft.reading_status ? (
                  <Tag color="gold">{readingStatusLabel(metaDraft.reading_status, S)}</Tag>
                ) : (
                  <Tag>{S.lib_meta_status_not_set}</Tag>
                )}
                <Tag color={metaSuggestionCount ? 'processing' : 'default'}>
                  {metaSuggestionCount ? S.lib_meta_suggestions.replace('{n}', String(metaSuggestionCount)) : S.lib_meta_no_suggestions}
                </Tag>
              </Space>
              {metaDraftTags.length ? (
                <div className="kb-lib-meta-chip-row">
                  {metaDraftTags.slice(0, 8).map((tagValue) => (
                    <Tag key={`meta-current-${tagValue}`}>{tagValue}</Tag>
                  ))}
                </div>
              ) : null}
            </div>
          ) : null}

          <section className="kb-lib-meta-section">
            <div className="kb-lib-meta-section-head">
              <div className="kb-lib-meta-section-copy">
                <Text className="kb-lib-meta-section-title">{S.lib_meta_section_my_org}</Text>
                <Text type="secondary" className="kb-lib-meta-section-note">
                  {S.lib_meta_org_hint}
                </Text>
              </div>
            </div>

            <div className="kb-lib-meta-field">
              <Text type="secondary" className="kb-lib-meta-label">{S.lib_meta_label_category}</Text>
              <AutoComplete
                value={metaDraft.paper_category}
                allowClear
                options={paperCategoryOptions}
                placeholder={S.lib_meta_category_placeholder}
                filterOption={optionMatchesInput}
                onChange={(value) => setMetaDraft((cur) => ({ ...cur, paper_category: String(value || '') }))}
                onBlur={() => setMetaDraft((cur) => ({ ...cur, paper_category: normalizeTextValue(cur.paper_category) }))}
              />
              <Text type="secondary" className="kb-lib-meta-help">
                {S.lib_meta_category_hint}
              </Text>
            </div>

            <div className="kb-lib-meta-field">
              <Text type="secondary" className="kb-lib-meta-label">{S.lib_meta_label_status}</Text>
              <Select
                value={metaDraft.reading_status || undefined}
                allowClear
                placeholder={S.lib_meta_reading_placeholder}
                options={READING_STATUS_OPTIONS(S).filter((item) => item.value)}
                onChange={(value) => setMetaDraft((cur) => ({ ...cur, reading_status: String(value || '') as ReadingStatusValue }))}
              />
            </div>

            <div className="kb-lib-meta-field">
              <Text type="secondary" className="kb-lib-meta-label">{S.lib_meta_label_tags}</Text>
              <Select
                mode="tags"
                value={metaDraft.user_tags}
                showSearch
                maxTagCount="responsive"
                tokenSeparators={TAG_INPUT_SEPARATORS}
                placeholder={S.lib_meta_tag_placeholder}
                options={paperTagOptions}
                optionFilterProp="label"
                onChange={(value) => setMetaDraft((cur) => ({ ...cur, user_tags: normalizeTextList(value as unknown[]) }))}
              />
              <Text type="secondary" className="kb-lib-meta-help">
                {S.lib_meta_tags_hint}
              </Text>
            </div>

            <div className="kb-lib-meta-field">
              <Text type="secondary" className="kb-lib-meta-label">{S.lib_meta_label_note}</Text>
              <Input.TextArea
                autoSize={{ minRows: 5, maxRows: 9 }}
                value={metaDraft.note}
                placeholder={S.lib_meta_note_placeholder}
                onChange={(event) => setMetaDraft((cur) => ({ ...cur, note: event.target.value }))}
              />
            </div>
          </section>

          <section className="kb-lib-meta-section kb-lib-meta-section-suggest">
            <div className="kb-lib-suggest-head">
              <div className="kb-lib-meta-section-copy">
                <Text className="kb-lib-meta-section-title">{S.lib_meta_section_system}</Text>
                <Text type="secondary" className="kb-lib-meta-section-note">
                  {S.lib_meta_system_hint}
                </Text>
              </div>
              <Space size={8} wrap>
                <Button size="small" loading={metaSuggestionSaving} onClick={() => { void regenerateMetaSuggestions() }}>
                  {S.lib_btn_refresh_suggestions}
                </Button>
                {metaItem?.has_suggestions ? (
                  <>
                    <Button
                      size="small"
                      type="primary"
                      ghost
                      loading={metaSuggestionSaving}
                      onClick={() => {
                        void applyMetaSuggestionAction({
                          category_action: metaItem?.suggested_category ? 'accept' : '',
                          accept_all_tags: true,
                        })
                      }}
                    >
                      {S.lib_btn_accept_all}
                    </Button>
                    <Button
                      size="small"
                      loading={metaSuggestionSaving}
                      onClick={() => {
                        void applyMetaSuggestionAction({
                          category_action: metaItem?.suggested_category ? 'dismiss' : '',
                          dismiss_all_tags: true,
                        })
                      }}
                    >
                      {S.lib_btn_dismiss_all}
                    </Button>
                  </>
                ) : null}
              </Space>
            </div>

            {metaItem?.has_suggestions ? (
              <div className="kb-lib-suggest-list">
                {metaItem.suggested_category ? (
                  <div className="kb-lib-suggest-item">
                    <div className="kb-lib-suggest-copy">
                      <Text className="kb-lib-suggest-title">{S.lib_meta_suggest_category}</Text>
                      <div className="kb-lib-meta-chip-row">
                        <Tag color="blue">{metaItem.suggested_category}</Tag>
                      </div>
                    </div>
                    <Space size={8}>
                      <Button
                        size="small"
                        type="primary"
                        ghost
                        loading={metaSuggestionSaving}
                        onClick={() => { void applyMetaSuggestionAction({ category_action: 'accept' }) }}
                      >
                        {S.lib_btn_accept}
                      </Button>
                      <Button
                        size="small"
                        loading={metaSuggestionSaving}
                        onClick={() => { void applyMetaSuggestionAction({ category_action: 'dismiss' }) }}
                      >
                        {S.lib_btn_dismiss}
                      </Button>
                    </Space>
                  </div>
                ) : null}

                {(metaItem?.suggested_tags || []).map((tagValue) => (
                  <div key={`meta-suggest-${tagValue}`} className="kb-lib-suggest-item">
                    <div className="kb-lib-suggest-copy">
                      <Text className="kb-lib-suggest-title">{S.lib_meta_suggest_tags}</Text>
                      <div className="kb-lib-meta-chip-row">
                        <Tag>{tagValue}</Tag>
                      </div>
                    </div>
                    <Space size={8}>
                      <Button
                        size="small"
                        type="primary"
                        ghost
                        loading={metaSuggestionSaving}
                        onClick={() => { void applyMetaSuggestionAction({ accept_tags: [tagValue] }) }}
                      >
                        {S.lib_btn_accept}
                      </Button>
                      <Button
                        size="small"
                        loading={metaSuggestionSaving}
                        onClick={() => { void applyMetaSuggestionAction({ dismiss_tags: [tagValue] }) }}
                      >
                        {S.lib_btn_dismiss}
                      </Button>
                    </Space>
                  </div>
                ))}
              </div>
            ) : (
              <Alert
                type="info"
                showIcon
                className="kb-lib-suggest-empty"
                message={S.lib_meta_no_suggestions_msg}
                description={S.lib_batch_hint}
              />
            )}
          </section>

          <div className="kb-lib-meta-actions">
            <Button onClick={() => setMetaDrawerOpen(false)}>
              {S.lib_btn_cancel}
            </Button>
            <Button type="primary" loading={metaSaving} onClick={() => { void saveMetaEditor() }}>
              {S.lib_btn_save}
            </Button>
          </div>
        </div>
      </Drawer>

      <Drawer
        title={S.lib_batch_edit_count_format.replace('{n}', String(selectedLibraryCount))}
        open={batchDrawerOpen}
        width={420}
        onClose={() => setBatchDrawerOpen(false)}
        destroyOnClose={false}
      >
        <div className="kb-lib-meta-drawer">
          <div className="kb-lib-meta-hero kb-lib-meta-hero-batch">
            <div className="kb-lib-meta-hero-copy">
              <Text className="kb-lib-meta-hero-title">{S.lib_batch_edit_hero.replace('{n}', String(selectedLibraryCount))}</Text>
              <Text type="secondary" className="kb-lib-meta-hero-note">
                {S.lib_batch_notice}
              </Text>
            </div>
            <Space wrap size={[6, 6]} className="kb-lib-meta-chip-row">
              <Tag color={selectedLibraryCount ? 'blue' : 'default'}>{S.lib_batch_selected_tag.replace('{n}', String(selectedLibraryCount))}</Tag>
              {batchDraft.apply_paper_category && normalizeTextValue(batchDraft.paper_category) ? (
                <Tag color="processing">{S.lib_batch_set_category_label.replace('{category}', normalizeTextValue(batchDraft.paper_category))}</Tag>
              ) : null}
              {batchDraft.add_tags.length ? (
                <Tag color="green">{S.lib_batch_add_tag_count.replace('{n}', String(normalizeTextList(batchDraft.add_tags).length))}</Tag>
              ) : null}
            </Space>
          </div>

          <section className="kb-lib-meta-section">
            <div className="kb-lib-meta-section-head">
              <div className="kb-lib-meta-section-copy">
                <Text className="kb-lib-meta-section-title">{S.lib_batch_section_setting}</Text>
                <Text type="secondary" className="kb-lib-meta-section-note">
                  {S.lib_batch_setting_hint}
                </Text>
              </div>
            </div>

            <div className={`kb-lib-meta-field ${batchDraft.apply_paper_category ? '' : 'is-muted'}`}>
              <Checkbox
                checked={batchDraft.apply_paper_category}
                onChange={(event) => setBatchDraft((cur) => ({ ...cur, apply_paper_category: event.target.checked }))}
              >
                {S.lib_batch_set_category_cb}
              </Checkbox>
              <AutoComplete
                value={batchDraft.paper_category}
                allowClear
                disabled={!batchDraft.apply_paper_category}
                options={paperCategoryOptions}
                placeholder={S.lib_meta_category_placeholder}
                filterOption={optionMatchesInput}
                onChange={(value) => setBatchDraft((cur) => ({ ...cur, paper_category: String(value || '') }))}
                onBlur={() => setBatchDraft((cur) => ({ ...cur, paper_category: normalizeTextValue(cur.paper_category) }))}
              />
              <Text type="secondary" className="kb-lib-meta-help">
                {S.lib_batch_category_hint}
              </Text>
            </div>

            <div className={`kb-lib-meta-field ${batchDraft.apply_reading_status ? '' : 'is-muted'}`}>
              <Checkbox
                checked={batchDraft.apply_reading_status}
                onChange={(event) => setBatchDraft((cur) => ({ ...cur, apply_reading_status: event.target.checked }))}
              >
                {S.lib_batch_set_status_cb}
              </Checkbox>
              <Select
                value={batchDraft.reading_status || undefined}
                allowClear
                disabled={!batchDraft.apply_reading_status}
                placeholder={S.lib_meta_reading_placeholder}
                options={READING_STATUS_OPTIONS(S).filter((item) => item.value)}
                onChange={(value) => setBatchDraft((cur) => ({ ...cur, reading_status: String(value || '') as ReadingStatusValue }))}
              />
            </div>
          </section>

          <section className="kb-lib-meta-section">
            <div className="kb-lib-meta-section-head">
              <div className="kb-lib-meta-section-copy">
                <Text className="kb-lib-meta-section-title">{S.lib_batch_section_tags}</Text>
                <Text type="secondary" className="kb-lib-meta-section-note">
                  {S.lib_batch_tags_hint}
                </Text>
              </div>
            </div>

            <div className="kb-lib-meta-field">
              <Text type="secondary" className="kb-lib-meta-label">{S.lib_batch_label_add_tags}</Text>
              <Select
                mode="tags"
                value={batchDraft.add_tags}
                showSearch
                maxTagCount="responsive"
                tokenSeparators={TAG_INPUT_SEPARATORS}
                placeholder={S.lib_batch_add_tag_placeholder}
                options={paperTagOptions}
                optionFilterProp="label"
                onChange={(value) => setBatchDraft((cur) => ({ ...cur, add_tags: normalizeTextList(value as unknown[]) }))}
              />
            </div>

            <div className="kb-lib-meta-field">
              <Text type="secondary" className="kb-lib-meta-label">{S.lib_batch_label_remove_tags}</Text>
              <Select
                mode="multiple"
                value={batchDraft.remove_tags}
                maxTagCount="responsive"
                placeholder={S.lib_batch_remove_tag_placeholder}
                options={paperTagFilterOptions}
                optionFilterProp="label"
                onChange={(value) => setBatchDraft((cur) => ({ ...cur, remove_tags: normalizeTextList(value as unknown[]) }))}
              />
            </div>
          </section>

          <div className="kb-lib-meta-actions">
            <Button onClick={() => setBatchDrawerOpen(false)}>
              {S.lib_btn_cancel}
            </Button>
            <Button type="primary" loading={batchSaving} onClick={() => { void saveBatchEditor() }}>
              {S.lib_btn_apply_to_selected}
            </Button>
          </div>
        </div>
      </Drawer>
    </div>
  )
}
