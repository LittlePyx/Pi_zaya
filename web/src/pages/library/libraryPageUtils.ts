import type {
  ConversionRepairAttempt,
  ConversionStage,
  ConversionQualitySummary,
  LibraryFileItem,
  LibraryConversionQualityBatchResponse,
  LibraryQualityActionHistoryItem,
  LibraryQualityActionDelta,
  LibraryQualityActionSnapshot,
  LibraryQualityDomain,
  LibraryQualityFailureCase,
  LibraryQualityFeatureHealthItem,
  LibraryQualityFullChainStage,
  LibraryQualityOverviewResponse,
  LibraryQualityPriorityAction,
  LibraryQualityRepairImpact,
  LibraryQualityRepairRun,
  LibraryResearchQaRerunResponse,
} from '../../api/library'
import type { ReferenceSyncStatKey, ReferenceSyncStats } from '../../api/references'

export function SCOPE_OPTIONS(S: Record<string, string>) {
  return [
    { value: '200', label: S.lib_scope_recent_200 },
    { value: '1000', label: S.lib_scope_recent_1000 },
    { value: 'all', label: S.lib_scope_all },
  ]
}

export function RENAME_SCOPE_OPTIONS(S: Record<string, string>) {
  return [
    { value: '30', label: S.lib_scope_recent_30 },
    { value: '50', label: S.lib_scope_recent_50 },
    { value: '100', label: S.lib_scope_recent_100 },
    { value: 'all', label: S.lib_scope_all },
  ]
}

export function numericStat(stats: ReferenceSyncStats | null | undefined, key: ReferenceSyncStatKey): number {
  const value = stats?.[key]
  const n = typeof value === 'number' ? value : Number(value || 0)
  return Number.isFinite(n) ? n : 0
}

export function formatSeconds(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return '0s'
  if (value < 60) return `${value.toFixed(value >= 10 ? 0 : 1)}s`
  const minutes = Math.floor(value / 60)
  const seconds = Math.round(value % 60)
  return `${minutes}m ${seconds}s`
}

export function fileTag(item: LibraryFileItem, S: Record<string, string>) {
  if (item.task_state === 'running') return { color: 'processing' as const, text: S.lib_tag_converting }
  if (item.task_state === 'queued') return { color: 'warning' as const, text: `${S.lib_tag_queued}${item.queue_pos > 0 ? ` #${item.queue_pos}` : ''}` }
  return item.category === 'converted'
    ? { color: 'success' as const, text: S.lib_tag_converted }
    : { color: 'default' as const, text: S.lib_tag_pending }
}

export function derivePageProgress(done0: number, total0: number, msg0: string) {
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

export function conversionStageLabel(stage0: string, S: Record<string, string>) {
  const stage = String(stage0 || '').trim().toLowerCase() as ConversionStage
  if (stage === 'queued') return S.lib_convert_queued
  if (stage === 'converting') return S.lib_convert_pages
  if (stage === 'finalizing') return S.lib_convert_finalizing
  if (stage === 'indexing') return S.lib_convert_ingesting
  if (stage === 'retrying') return S.lib_convert_retrying
  if (stage === 'cancelling') return S.lib_convert_cancelling
  return ''
}

export function normalizeRunningPages(value: unknown, total0 = 0): number[] {
  const total = Number(total0 || 0)
  if (!Array.isArray(value)) return []
  return Array.from(new Set(
    value
      .map((item) => Number(item))
      .filter((item) => (
        Number.isInteger(item)
        && item > 0
        && (!(total > 0) || item <= total)
      )),
  )).sort((a, b) => a - b)
}

export function runningPagesLabel(
  stage0: string,
  pages0: unknown,
  pageCount0: number,
  pageTotal0: number,
  S: Record<string, string>,
): string {
  if (String(stage0 || '').trim().toLowerCase() !== 'converting') return ''
  const pages = normalizeRunningPages(pages0, pageTotal0)
  if (!pages.length) return ''

  const preview = pages.slice(0, 5)
  const separator = String(S.lib_convert_running_pages_separator || '、')
  const pageText = preview.join(separator)
  const reportedCount = Number(pageCount0 || 0)
  const totalCount = Math.max(
    pages.length,
    Number.isFinite(reportedCount) && reportedCount > 0 ? Math.floor(reportedCount) : 0,
  )
  const hasMore = totalCount > preview.length
  const template = hasMore
    ? String(S.lib_convert_running_pages_more || '剩余页：{pages}…（共 {count} 页）')
    : String(S.lib_convert_running_pages || '剩余页：{pages}')
  return template
    .replace('{pages}', pageText)
    .replace('{count}', String(totalCount))
}

export function conversionTaskFraction(stage0: string, pageDone: number, pageTotal: number) {
  const stage = String(stage0 || '').trim().toLowerCase() as ConversionStage
  const pageFraction = pageTotal > 0
    ? Math.max(0, Math.min(1, Number(pageDone || 0) / Math.max(1, Number(pageTotal || 0))))
    : 0
  if (stage === 'indexing') return 0.98
  if (stage === 'finalizing') return 0.94
  if (stage === 'retrying') return 0.9
  if (stage === 'cancelling') return Math.min(0.98, pageFraction * 0.9)
  return Math.min(0.9, pageFraction * 0.9)
}

export function matchesKeyword(name: string, keyword: string) {
  if (!keyword) return true
  return name.toLowerCase().includes(keyword)
}

export function stripKnownSourceExt(name: string) {
  return String(name || '')
    .replace(/\.en\.md$/i, '')
    .replace(/\.md$/i, '')
    .replace(/\.pdf$/i, '')
    .trim()
}

export type UploadDraft = {
  key: string
  file: File
  name: string
  selected: boolean
  stem: string
  status: 'queued' | 'inspecting' | 'ready' | 'saving' | 'saved' | 'error'
  failureStage?: 'inspect' | 'save' | 'duplicate' | ''
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

export type UploadDraftFilter = 'all' | 'todo' | 'error' | 'dup_error' | 'saved'
export type UploadErrorReason = 'all' | 'duplicate' | 'path' | 'permission' | 'network' | 'other'

export function isDuplicateFailure(note: string) {
  const text = String(note || '').toLowerCase()
  return text.includes('\u91cd\u590d') || text.includes('duplicate') || text.includes('already exists') || text.includes('\u5df2\u5b58\u5728')
}

export function classifyFailedReason(note: string): Exclude<UploadErrorReason, 'all'> {
  const text = String(note || '').toLowerCase()
  if (isDuplicateFailure(text)) return 'duplicate'
  if (text.includes('\u76ee\u5f55') || text.includes('\u8def\u5f84') || text.includes('path') || text.includes('dir')) return 'path'
  if (text.includes('\u6743\u9650') || text.includes('permission') || text.includes('denied')) return 'permission'
  if (text.includes('\u7f51\u7edc') || text.includes('timeout') || text.includes('network')) return 'network'
  return 'other'
}

export type QualityRepairHistoryRecord = {
  name: string
  beforeScore: number
  afterScore: number
  beforeStatus: string
  afterStatus: string
  fixedIssues: string[]
  remainingIssues: string[]
  updatedAt: number
}

export type SourceReadinessKind = 'ready' | 'autofixed' | 'confirmed' | 'blocked' | 'review' | 'processing' | 'pending' | 'index_stale' | 'unknown'

export interface SourceReadinessView {
  kind: SourceReadinessKind
  tone: 'ready' | 'autofixed' | 'blocked' | 'review' | 'processing' | 'pending' | 'index_stale' | 'unknown'
  label: string
  detail: string
  action: 'repair' | 'reconvert' | 'reindex' | ''
  qaReady: boolean
  blocked: boolean
}

const QUALITY_REPAIR_HISTORY_STORAGE_KEY = 'kb.library.qualityRepairHistory.v1'
const QUALITY_REPAIR_HISTORY_LIMIT = 40

export function normalizeTextValue(value: unknown) {
  return String(value || '').replace(/\s+/g, ' ').trim()
}

export function qualityDomainNumber(domain: LibraryQualityDomain | undefined, key: string) {
  const value = domain?.summary?.[key]
  const num = Number(value || 0)
  return Number.isFinite(num) ? num : 0
}

export function qualityDomainStatus(domain: LibraryQualityDomain | undefined, fallback = 'unknown') {
  return normalizeTextValue(domain?.status || fallback).toLowerCase() || 'unknown'
}

export function qualityStatusText(status: string, S: Record<string, string>) {
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

export function qualityOverviewStageSnapshot(
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

export function qualityBuildActionDelta(
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

export function qualityActionDeltaText(item: Pick<LibraryQualityActionHistoryItem, 'delta' | 'improved' | 'before' | 'after'>) {
  const explicit = normalizeTextValue(item.delta?.summary)
  if (explicit) return explicit
  const before = item.before
  const after = item.after
  if (before || after) return qualityBuildActionDelta(before, after).summary || ''
  if (item.improved === true) return 'Improved'
  if (item.improved === false) return 'No measurable change yet'
  return ''
}

export function qualityVerificationFromRerun(rerun: LibraryResearchQaRerunResponse | null | undefined) {
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

export function qualityVerificationText(verification: Record<string, unknown> | undefined): string {
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
  if (type === 'reader_locate_repair') {
    const status = normalizeTextValue(verification.status).toLowerCase()
    const passed = Number(verification.passed || 0)
    const total = Number(verification.target_count || 0)
    const failed = Number(verification.failed || 0)
    const needsReopen = Number(verification.needs_reader_reopen || 0)
    if (verification.quality_ok === true || status === 'passed') return `Reader locate verified: ${passed}/${total}`
    if (status === 'needs_reader_reopen') return `Reader locate needs reopen: ${needsReopen}/${total}`
    if (status === 'failed') return `Reader locate still failing: ${failed}/${total}`
    if (status === 'skipped') return 'No linked Reader locate event'
  }
  if (type === 'shelf_metadata_repair') {
    const status = normalizeTextValue(verification.status).toLowerCase()
    const ready = Number(verification.export_ready_after || 0)
    const total = Number(verification.target_count || 0)
    const retryable = Number(verification.retryable || 0)
    const unresolved = Number(verification.unresolved_after || 0)
    if (verification.quality_ok === true || status === 'passed') return `Metadata export verified: ${ready}/${total}`
    if (status === 'retryable') return `Metadata repair can retry: ${retryable}/${total}`
    if (status === 'failed') return `Metadata still missing fields: ${unresolved}/${total}`
    if (status === 'partial') return `Metadata partially ready: ${ready}/${total}`
    if (status === 'skipped') return 'No shelf metadata target'
  }
  if (type === 'combined_shelf_metadata_verification') {
    const shelf = verification.shelf_metadata as Record<string, unknown> | undefined
    const research = verification.research_qa as Record<string, unknown> | undefined
    const shelfText: string = qualityVerificationText(shelf)
    const researchText: string = qualityVerificationText(research)
    return [shelfText, researchText].filter(Boolean).join(' / ')
  }
  if (type === 'combined_repair_verification') {
    const research = verification.research_qa as Record<string, unknown> | undefined
    const reader = verification.reader_locate as Record<string, unknown> | undefined
    const researchText: string = qualityVerificationText(research)
    const readerText: string = qualityVerificationText(reader)
    return [researchText, readerText].filter(Boolean).join(' / ')
  }
  return ''
}

export function qualityTopFailureText(domain: LibraryQualityDomain | undefined) {
  const first = Array.isArray(domain?.top_failures) ? domain?.top_failures?.[0] : null
  const name = normalizeTextValue(first?.name)
  if (!name) return ''
  const count = Number(first?.count || 0)
  return count > 0 ? `${name} x${count}` : name
}

export function qualityActionText(action: LibraryQualityPriorityAction, S: Record<string, string>) {
  const domain = normalizeTextValue(action.domain)
  const label = normalizeTextValue(action.label)
  if (domain === 'conversion') return S.lib_quality_action_conversion
  if (domain === 'research_qa' && label.toLowerCase().includes('run')) return S.lib_quality_action_research_qa_run
  if (domain === 'research_qa') return S.lib_quality_action_research_qa
  if (domain === 'citation_cards') return S.lib_quality_action_citation_cards
  if (domain === 'reader_locate') return 'Repair reader locate'
  return label || domain
}

export function qualityFullChainActionText(stage: LibraryQualityFullChainStage) {
  const action = normalizeTextValue(stage.action).toLowerCase()
  const status = normalizeTextValue(stage.status).toLowerCase()
  if (status === 'good' && action.startsWith('monitor_')) return 'Verified'
  if (action === 'repair_conversion') return 'Repair'
  if (action === 'fix_failed_qa_cases') return 'Fix case'
  if (action === 'run_research_qa') return 'Open runbook'
  if (action === 'rebuild_index') return 'Rebuild'
  if (action === 'repair_citation_cards') return 'Repair cards'
  if (action === 'repair_shelf_metadata') return 'Repair metadata'
  if (action === 'monitor_literature_basket') return status === 'good' ? 'Verified' : 'Preflight'
  if (action === 'rerun_failed_cases') return 'Rerun'
  if (action.startsWith('monitor_')) return 'Review'
  return normalizeTextValue(stage.action).replace(/_/g, ' ') || 'Review'
}

export function qualityFailureCaseMatchesStage(item: LibraryQualityFailureCase, stageKey: string) {
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
      || rootCodes.has('shelf_metadata_export_fields')
      || actions.has('repair_shelf_metadata')
      || actions.has('inspect_cards')
  }
  return false
}

export function qualityActionHistoryActionText(item: LibraryQualityActionHistoryItem) {
  const stageKey = normalizeTextValue(item.stage_key).toLowerCase()
  const hasTarget = Boolean((item.target_ids || []).some((value) => normalizeTextValue(value)))
  if (stageKey === 'conversion') return hasTarget ? 'Focus source' : 'Review'
  if (['research_qa', 'retrieval', 'repair_loop', 'citations', 'shelf'].includes(stageKey)) {
    return hasTarget ? 'Open replay' : (stageKey === 'citations' || stageKey === 'shelf' ? 'Open report' : 'Review')
  }
  if (stageKey === 'citation_cards') return 'Open report'
  return hasTarget ? 'Open target' : 'Review'
}

export function qualityFeatureActionText(item: LibraryQualityFeatureHealthItem) {
  const action = normalizeTextValue(item.action).toLowerCase()
  if (action.startsWith('repair_')) return 'Repair'
  if (action.startsWith('fix_')) return 'Fix'
  if (action.startsWith('run_')) return 'Run'
  if (action.startsWith('inspect_')) return 'Inspect'
  return 'Review'
}

export function normalizeTextList(values: unknown[]) {
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

export function uniqueTextValues(values: Iterable<unknown>) {
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

export function isUploadDraftConverted(draft: UploadDraft, files: LibraryFileItem[]) {
  if (!draft.convertRequested || draft.status !== 'saved') return false
  const match = files.find((item) => {
    if (draft.savedSha1 && item.sha1) return item.sha1 === draft.savedSha1
    if (draft.savedName) return item.name === draft.savedName
    return false
  })
  if (!match) return false
  return match.md_exists && match.task_state === 'idle' && match.category === 'converted'
}

export function suggestionBasisTagColor(meta?: {
  match_method?: string
  year_source?: string
}) {
  const method = String(meta?.match_method || '').trim().toLowerCase()
  const yearSource = String(meta?.year_source || '').trim().toLowerCase()
  if (method === 'doi') return 'success'
  if (method === 'crossref_strong') return 'processing'
  if (yearSource === 'filename') return 'gold'
  if (method === 'crossref_weak') return 'warning'
  return 'default'
}

export function conversionQualityStatus(quality?: ConversionQualitySummary | null) {
  return String(quality?.status || '').trim().toLowerCase()
}

export function conversionQualityToneClass(quality?: ConversionQualitySummary | null) {
  const status = conversionQualityStatus(quality)
  if (status === 'good') return 'is-good'
  if (status === 'error') return 'is-error'
  if (status === 'warning') return 'is-warning'
  return 'is-unknown'
}

export function conversionQualityLabel(quality?: ConversionQualitySummary | null) {
  if (!quality) return ''
  const center = quality.conversion_report?.quality_center || null
  const centerActionLabel = normalizeTextValue(center?.action_label)
  const score = Number(quality.score || 0)
  const scoreText = Number.isFinite(score) ? `Q${Math.max(0, Math.min(100, Math.round(score)))}` : 'Q?'
  const status = conversionQualityStatus(quality)
  if (centerActionLabel && status !== 'good') return `${centerActionLabel} ${scoreText}`
  if (status === 'good') return scoreText
  if (status === 'error') return `Repair ${scoreText}`
  return `Review ${scoreText}`
}

export function conversionMetric(quality: ConversionQualitySummary | null | undefined, key: keyof ConversionQualitySummary['metrics']) {
  const value = Number(quality?.metrics?.[key] || 0)
  return Number.isFinite(value) ? Math.max(0, Math.round(value)) : 0
}

export function libraryDocumentTypeView(value: unknown) {
  const docType = normalizeTextValue(value).toLowerCase()
  if (docType === 'review') {
    return {
      label: 'Review paper',
      tone: 'is-review',
      title: 'Review paper: useful for background and broad reference trails.',
    }
  }
  if (docType === 'supplementary') {
    return {
      label: 'Supplement',
      tone: 'is-supplement',
      title: 'Supplementary material: a standalone reference list may not exist.',
    }
  }
  return null
}

export function conversionQualityNeedsReview(quality?: ConversionQualitySummary | null) {
  const status = conversionQualityStatus(quality)
  return Boolean(quality?.has_review_issue) || status === 'warning' || status === 'error'
}

export function conversionRepairAttemptLabel(
  attempt: ConversionRepairAttempt | null | undefined,
  S: Record<string, string>,
) {
  const status = normalizeTextValue(attempt?.status).toLowerCase()
  if (status === 'queued' || status === 'running') return S.lib_quality_gate_queued
  if (status === 'autofixed' || status === 'fixed') return S.lib_quality_gate_autofixed
  if (status === 'success' || status === 'resolved' || status === 'ready' || status === 'accepted') return S.lib_quality_gate_ready
  if (status === 'rolled_back') return S.lib_quality_gate_rolled_back
  if (status === 'partial') return S.lib_quality_gate_partial
  if (status === 'blocked' || status === 'failed' || status === 'error') return S.lib_quality_gate_blocked
  return S.lib_quality_gate_tracked.replace('{status}', status || 'tracked')
}

function normalizedConversionPages(value: unknown): number[] {
  const rows = Array.isArray(value) ? value : []
  return Array.from(new Set(rows
    .map((item) => Number(item))
    .filter((item) => Number.isInteger(item) && item > 0)))
    .sort((a, b) => a - b)
    .slice(0, 500)
}

export function conversionTargetedRepairPages(item: LibraryFileItem): number[] {
  const report = item.conversion_quality?.conversion_report
  if (!report || report.stale) return []
  const plan = report.repair_plan
  if (
    normalizeTextValue(plan?.action).toLowerCase() !== 'reconvert'
    || normalizeTextValue(plan?.scope).toLowerCase() !== 'pages'
  ) return []
  const planned = normalizedConversionPages(plan?.retry_pages)
  if (planned.length) return planned
  return normalizedConversionPages(report.source_quality?.evidence_unreliable_pages)
}

export function conversionTargetedRepairPageDetail(
  item: LibraryFileItem,
  S?: Record<string, string>,
): string {
  const pages = conversionTargetedRepairPages(item)
  if (!pages.length) return ''
  const profiles = Array.isArray(item.conversion_quality?.conversion_report?.source_quality?.page_evidence_profiles)
    ? item.conversion_quality?.conversion_report?.source_quality?.page_evidence_profiles || []
    : []
  const byPage = new Map(profiles
    .filter((profile) => Number.isInteger(Number(profile?.page)) && Number(profile?.page) > 0)
    .map((profile) => [Number(profile?.page), profile]))
  const reasonLabels: Record<string, string> = {
    empty_page_marker_segment: S?.lib_quality_page_reason_empty || 'no readable page content detected',
    low_local_page_overlap: S?.lib_quality_page_reason_low_local_overlap || 'converted content weakly matches this page',
    low_text_overlap: S?.lib_quality_page_reason_low_text_overlap || 'converted content weakly matches the source',
    missing_wrapped_word_prefixes: S?.lib_quality_page_reason_wrapped_words || 'line-wrapped words may be incomplete',
    source_prose_omission: S?.lib_quality_page_reason_prose_omission || 'source prose may be missing',
  }
  return pages.slice(0, 12).map((page) => {
    const profile = byPage.get(page)
    const reasons = normalizeTextList(profile?.reason_codes || [])
      .map((reason) => reasonLabels[reason] || reason.replaceAll('_', ' '))
    const coverage = Number(profile?.coverage)
    const evidence = [
      reasons.join(', '),
      Number.isFinite(coverage)
        ? `${S?.lib_quality_page_evidence_coverage || 'source coverage'} ${Math.round(coverage * 100)}%`
        : '',
    ].filter(Boolean).join(' · ')
    return evidence ? `p.${page}: ${evidence}` : `p.${page}`
  }).join('\n')
}

export function conversionSourceReadiness(item: LibraryFileItem, S: Record<string, string>): SourceReadinessView {
  const quality = item.conversion_quality
  const report = quality?.conversion_report || null
  const center = report?.quality_center || null
  const sourceQuality = report?.source_quality || center?.source_quality || null
  const repairPlan = report?.stale === true ? null : (report?.repair_plan || null)
  const latestAttempt = report?.latest_repair_attempt || null
  const latestStatus = normalizeTextValue(latestAttempt?.status).toLowerCase()
  const centerAction = normalizeTextValue(center?.action).toLowerCase()
  const gateAction = normalizeTextValue(item.quality_gate?.action).toLowerCase()
  const centerMessage = normalizeTextValue(center?.message || report?.source_quality_message)
  const repairAction = [
    normalizeTextValue(repairPlan?.action).toLowerCase(),
    centerAction,
    gateAction,
  ].find((action) => ['none', 'autofix', 'review', 'reconvert'].includes(action)) || ''
  const qualityStatus = conversionQualityStatus(quality)
  const needsReview = conversionQualityNeedsReview(quality)
  const remainingCodes = Array.isArray(report?.remaining_issue_codes) ? report?.remaining_issue_codes || [] : []
  const indexState = normalizeTextValue(item.index_state).toLowerCase()
  const indexStatus = normalizeTextValue(item.index_status).toLowerCase()
  const indexQualityGate = item.quality_gate && typeof item.quality_gate === 'object'
    ? item.quality_gate
    : null
  const manuallyConfirmed = indexState === 'ready'
    && indexStatus === 'quality_degraded'
    && indexQualityGate?.indexable === true
    && indexQualityGate?.override_applied === true
  const hasAuthoritativeIndexState = Boolean(indexState)
  const needsQualityRepair = ['review', 'autofix'].includes(repairAction)
  const isBlocked = repairAction === 'reconvert'
    || (!repairAction && (Boolean(report?.needs_reconvert) || Boolean(sourceQuality?.source_text_loss)))
  const wasAutofixed = Boolean(report?.auto_repair_changed) || ['autofixed', 'fixed'].includes(latestStatus)

  if (item.task_state === 'running' || item.task_state === 'queued') {
    return {
      kind: 'processing',
      tone: 'processing',
      label: S.lib_source_status_processing,
      detail: S.lib_source_status_processing_detail,
      action: '',
      qaReady: false,
      blocked: false,
    }
  }
  if (!item.md_exists) {
    return {
      kind: 'pending',
      tone: 'pending',
      label: S.lib_source_status_pending,
      detail: S.lib_source_status_pending_detail,
      action: '',
      qaReady: false,
      blocked: false,
    }
  }
  if (manuallyConfirmed) {
    return {
      kind: 'confirmed',
      tone: 'review',
      label: S.lib_source_status_confirmed,
      detail: S.lib_source_status_confirmed_detail,
      action: '',
      qaReady: true,
      blocked: false,
    }
  }
  const needsIndexRefresh = ['not_indexed', 'index_stale', 'not_ready'].includes(indexState)
    || (indexState === 'quality_blocked' && repairAction === 'none')
  if (!isBlocked && !needsQualityRepair && needsIndexRefresh) {
    return {
      kind: 'index_stale',
      tone: 'index_stale',
      label: S.lib_source_status_index_stale,
      detail: S.lib_source_status_index_stale_detail,
      action: 'reindex',
      qaReady: false,
      blocked: false,
    }
  }
  if (isBlocked) {
    return {
      kind: 'blocked',
      tone: 'blocked',
      label: S.lib_source_status_blocked,
      detail: centerMessage || repairPlan?.reason || latestAttempt?.detail || latestAttempt?.reason || S.lib_source_status_blocked_detail,
      action: 'reconvert',
      qaReady: false,
      blocked: true,
    }
  }
  if (latestStatus === 'queued' || latestStatus === 'running') {
    return {
      kind: 'processing',
      tone: 'processing',
      label: S.lib_source_status_processing,
      detail: latestAttempt?.detail || latestAttempt?.reason || S.lib_source_status_processing_detail,
      action: '',
      qaReady: false,
      blocked: false,
    }
  }
  if (wasAutofixed && !needsReview && remainingCodes.length === 0) {
    return {
      kind: 'autofixed',
      tone: 'autofixed',
      label: S.lib_source_status_autofixed,
      detail: centerMessage || S.lib_source_status_autofixed_detail,
      action: '',
      qaReady: true,
      blocked: false,
    }
  }
  if (
    (
      indexState === 'ready'
      || (!hasAuthoritativeIndexState && (qualityStatus === 'good' || latestStatus === 'ready' || latestStatus === 'success' || latestStatus === 'resolved'))
    )
    && !needsReview
  ) {
    return {
      kind: 'ready',
      tone: 'ready',
      label: S.lib_source_status_ready,
      detail: centerMessage || S.lib_source_status_ready_detail,
      action: '',
      qaReady: true,
      blocked: false,
    }
  }
  if (needsReview || remainingCodes.length > 0 || needsQualityRepair) {
    return {
      kind: 'review',
      tone: 'review',
      label: S.lib_source_status_review,
      detail: centerMessage || repairPlan?.reason || S.lib_source_status_review_detail,
      action: 'repair',
      qaReady: false,
      blocked: false,
    }
  }
  return {
    kind: 'unknown',
    tone: 'unknown',
    label: S.lib_source_status_unknown,
    detail: S.lib_source_status_unknown_detail,
    action: '',
    qaReady: false,
    blocked: false,
  }
}

export function conversionQualityScore(quality?: ConversionQualitySummary | null) {
  const value = Number(quality?.score || 0)
  return Number.isFinite(value) ? Math.max(0, Math.min(100, Math.round(value))) : 0
}

export function conversionQualityIssueEntries(quality?: ConversionQualitySummary | null) {
  return (Array.isArray(quality?.issues) ? quality?.issues || [] : [])
    .map((issue) => {
      const label = normalizeTextValue(issue.label || issue.code)
      const key = normalizeTextValue(issue.code || issue.label).toLowerCase()
      return key && label ? { key, label } : null
    })
    .filter((item): item is { key: string; label: string } => Boolean(item))
}

export function summarizeConversionQualityRepair(
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
  return `${base} / ${String(S.quality_repair_result_fixed_issues || 'Fixed: {issues}').replace('{issues}', fixedIssues.join(' / '))}`
}

export function buildQualityRepairHistoryRecord(
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

export function normalizeQualityRepairHistory(value: unknown): Record<string, QualityRepairHistoryRecord> {
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

export function loadQualityRepairHistory(): Record<string, QualityRepairHistoryRecord> {
  if (typeof window === 'undefined') return {}
  try {
    const raw = window.localStorage.getItem(QUALITY_REPAIR_HISTORY_STORAGE_KEY)
    if (!raw) return {}
    return normalizeQualityRepairHistory(JSON.parse(raw))
  } catch {
    return {}
  }
}

export function saveQualityRepairHistory(records: Record<string, QualityRepairHistoryRecord>) {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(QUALITY_REPAIR_HISTORY_STORAGE_KEY, JSON.stringify(normalizeQualityRepairHistory(records)))
  } catch {
    // Storage is best-effort; the current-session repair result remains visible.
  }
}

export function formatQualityRepairHistoryTime(ts: number) {
  if (!Number.isFinite(ts) || ts <= 0) return ''
  try {
    return new Date(ts).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })
  } catch {
    return ''
  }
}

export function formatQualityRepairRecordSummary(record: QualityRepairHistoryRecord, S: Record<string, string>) {
  const template = record.remainingIssues.length <= 0 ? S.quality_repair_result_pass : S.quality_repair_result_review
  const base = String(template || 'Repair checked: Q{before} -> Q{after}')
    .replace('{before}', String(record.beforeScore))
    .replace('{after}', String(record.afterScore))
    .replace('{n}', String(record.remainingIssues.length))
  if (!record.fixedIssues.length) return base
  return `${base} / ${String(S.quality_repair_result_fixed_issues || 'Fixed: {issues}').replace('{issues}', record.fixedIssues.slice(0, 3).join(' / '))}`
}

export function formatSignedNumber(value: number) {
  const n = Math.round(Number(value || 0))
  return n > 0 ? `+${n}` : String(n)
}

export function qualityRepairImpactIndexText(impact: LibraryQualityRepairImpact) {
  if (!impact.needs_reindex) return 'Index current'
  if (impact.reindexed === true) return 'Index refreshed'
  if (impact.reindexed === false && Number(impact.enqueued || 0) <= 0) return 'Index refresh needs retry'
  if (Number(impact.enqueued || 0) > 0) return 'Index will refresh after conversion'
  return 'Index refresh pending'
}

export function qualityBatchIndexText(result: LibraryConversionQualityBatchResponse) {
  if (!result.needs_reindex) return 'Index current'
  if (Number(result.rebuilt || 0) > 0) return `Structured index rebuilt: ${result.rebuilt}`
  return 'Index refresh needed'
}

export function qualityBatchStatusText(result: LibraryConversionQualityBatchResponse) {
  const mode = result.mode === 'repair' ? 'Safe repair' : 'Source scan'
  const failed = Number(result.failed || 0)
  const issueCount = Number(result.autofix || 0) + Number(result.reconvert || 0) + Number(result.review || 0)
  const core = `${mode}: ${result.scanned}/${result.target_count} scanned, ${result.ready} ready, ${issueCount} need action`
  if (failed > 0) return `${core}, ${failed} failed`
  return core
}

export function qualityRepairRunStatusText(run: LibraryQualityRepairRun | null) {
  if (!run) return ''
  const status = normalizeTextValue(run.status).toLowerCase()
  const phase = normalizeTextValue(run.phase).toLowerCase()
  if (phase === 'verification_passed') return 'Run tracked: verified'
  if (phase === 'verification_failed') return 'Run tracked: verification failed'
  if (phase === 'verification_blocked') return 'Run tracked: verification blocked'
  if (phase === 'verification_needs_reader_reopen') return 'Run tracked: needs reader reopen'
  if (phase === 'shelf_metadata_verified') return 'Run tracked: metadata verified'
  if (phase === 'shelf_metadata_retryable') return 'Run tracked: metadata retryable'
  if (phase === 'shelf_metadata_unresolved') return 'Run tracked: metadata unresolved'
  if (status === 'completed' || phase === 'reindex_complete') return 'Run tracked: completed'
  if (status === 'failed' || phase === 'repair_failed') return 'Run tracked: failed'
  if (phase === 'source_reconversion_queued') return 'Run tracked: waiting for conversion'
  if (phase === 'reindex_pending' || status === 'reindex_pending') return 'Run tracked: index refresh pending'
  return `Run tracked: ${run.status || run.phase || 'recorded'}`
}

export function qualityRepairRunTagColor(run: LibraryQualityRepairRun | null) {
  const status = normalizeTextValue(run?.status).toLowerCase()
  const phase = normalizeTextValue(run?.phase).toLowerCase()
  if (phase === 'verification_passed') return 'success'
  if (phase === 'verification_failed' || phase === 'verification_blocked' || phase === 'verification_needs_reader_reopen') return 'warning'
  if (phase === 'shelf_metadata_verified') return 'success'
  if (phase === 'shelf_metadata_retryable' || phase === 'shelf_metadata_unresolved') return 'warning'
  if (status === 'completed' || phase === 'reindex_complete') return 'success'
  if (status === 'failed' || phase === 'repair_failed') return 'error'
  if (status === 'queued' || phase === 'source_reconversion_queued') return 'processing'
  if (status === 'reindex_pending' || phase === 'reindex_pending') return 'warning'
  return 'default'
}

export function qualityRepairRunCanAdvance(run: LibraryQualityRepairRun | null) {
  if (!run) return false
  const status = normalizeTextValue(run.status).toLowerCase()
  const phase = normalizeTextValue(run.phase).toLowerCase()
  if (phase === 'verification_failed' || phase === 'verification_blocked' || phase === 'verification_needs_reader_reopen') return true
  if (status === 'completed' || phase === 'reindex_complete' || phase === 'repair_complete' || phase === 'verification_passed') return false
  return Boolean(run.needs_reindex || status === 'queued' || status === 'reindex_pending' || phase === 'source_reconversion_queued' || phase === 'reindex_pending' || phase === 'reindex_failed')
}

export function hasConversionQualityIssue(item: LibraryFileItem) {
  const quality = item.conversion_quality
  const report = quality?.conversion_report || null
  const centerStatus = normalizeTextValue(report?.quality_center?.status).toLowerCase()
  const sourceTextLoss = Boolean(report?.source_quality?.source_text_loss || report?.quality_center?.source_quality?.source_text_loss)
  const latestStatus = normalizeTextValue(report?.latest_repair_attempt?.status).toLowerCase()
  const planAction = normalizeTextValue(report?.repair_plan?.action || report?.quality_center?.action).toLowerCase()
  const indexState = normalizeTextValue(item.index_state).toLowerCase()
  const remainingCodes = Array.isArray(report?.remaining_issue_codes) ? report?.remaining_issue_codes || [] : []
  const gateReady = ['ready', 'success', 'resolved', 'autofixed', 'fixed'].includes(latestStatus)
    && !conversionQualityNeedsReview(quality)
    && remainingCodes.length === 0
    && !report?.needs_reconvert
    && !report?.auto_repair_unsafe
    && (!indexState || indexState === 'ready')
  if (gateReady) return false
  return Boolean(quality?.has_review_issue)
    || Boolean(report?.needs_reconvert)
    || sourceTextLoss
    || Boolean(report?.auto_repair_unsafe)
    || remainingCodes.length > 0
    || ['reconvert', 'review', 'autofix'].includes(planAction)
    || ['reconvert', 'review', 'autofix'].includes(centerStatus)
    || ['blocked', 'failed', 'error', 'partial'].includes(latestStatus)
    || indexState === 'quality_blocked'
}

export function toTextOptions(values: string[]) {
  return values.map((value) => ({ value, label: value }))
}

export function optionMatchesInput(input: string, option?: { value?: unknown; label?: unknown }) {
  const needle = normalizeTextValue(input).toLowerCase()
  if (!needle) return true
  const hay = normalizeTextValue(option?.value || option?.label || '').toLowerCase()
  return hay.includes(needle)
}

export function saveResearchQaReplayFailureCase(item: LibraryQualityFailureCase) {
  if (typeof window === 'undefined') return
  try {
    window.sessionStorage.setItem('kb.researchQaReplay.failureCase.v1', JSON.stringify(item))
  } catch {
    // Replay still works from fixture data if session storage is unavailable.
  }
}
