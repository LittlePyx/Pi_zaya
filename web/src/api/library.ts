import { api, authFetch, responseError, responseJson } from './client'

export interface ConvertActiveTask {
  task_id: string
  name: string
  pdf: string
  replace: boolean
  cur_page_done: number
  cur_page_total: number
  cur_page_msg: string
  conversion_stage: ConversionStage
  running_pages?: number[]
  running_page_count?: number
}

export type ConversionStage = '' | 'queued' | 'converting' | 'finalizing' | 'indexing' | 'retrying' | 'cancelling'

export interface ConvertProgress {
  running: boolean
  done: boolean
  total: number
  completed: number
  current: string
  active_count: number
  active_tasks: ConvertActiveTask[]
  cur_page_done: number
  cur_page_total: number
  cur_page_msg: string
  conversion_stage: ConversionStage
  running_pages?: number[]
  running_page_count?: number
  last: string
  recent_tasks?: ConversionTaskResult[]
}

export type ConversionOutcome = 'success' | 'cancelled' | 'conversion_failed' | 'quality_blocked' | 'index_failed'

export interface ConversionTaskResult {
  task_id: string
  name: string
  pdf: string
  outcome: ConversionOutcome
  operation: 'conversion' | 'index_retry'
  message: string
  detail: string
  retry_action: '' | 'reconvert' | 'reindex'
  replace: boolean
  speed_mode: string
  started_at: number
  finished_at: number
  duration_s: number
  page_done: number
  page_total: number
}

export interface CancelConversionResponse {
  ok: boolean
  scope: 'task' | 'all'
  matched: boolean
  task_id: string
  state: 'queued_removed' | 'cancelling' | 'not_found' | 'cancelling_all' | string
  removed_queued: number
}

export interface LibraryFileItem {
  name: string
  path: string
  sha1: string
  md_exists: boolean
  md_path: string
  md_folder: string
  conversion_quality: ConversionQualitySummary | null
  index_state?: 'ready' | 'quality_blocked' | 'index_stale' | 'not_indexed' | 'not_ready' | 'not_converted' | string
  index_status?: string
  index_ready?: boolean
  index_doc_id?: string
  index_path?: string
  index_num_chunks?: number
  index_chunk_exists?: boolean
  quality_gate?: Record<string, unknown> | null
  category: 'pending' | 'converted'
  task_state: 'idle' | 'queued' | 'running'
  status: string
  task_id?: string
  replace_task: boolean
  queue_pos: number
  cur_page_done: number
  cur_page_total: number
  cur_page_msg: string
  conversion_stage: ConversionStage
  last_conversion?: ConversionTaskResult | null
  running_pages?: number[]
  running_page_count?: number
  paper_category: string
  reading_status: '' | 'unread' | 'reading' | 'done' | 'revisit'
  note: string
  user_tags: string[]
  has_suggestions: boolean
  suggested_category: string
  suggested_tags: string[]
}

export interface ConversionQualityIssue {
  code: string
  label: string
  severity: 'warning' | 'error' | string
  count: number
  repairable?: boolean
  repair_strategy?: string
  repair_steps?: string[]
  repair_action?: string
  repair_scope?: string
  repair_speed_mode?: string
}

export interface ConversionRepairPlan {
  action: 'none' | 'autofix' | 'reconvert' | 'review' | string
  scope: string
  speed_mode: string
  no_llm: boolean
  replace: boolean
  md_autofix_first: boolean
  reason: string
  issue_codes: string[]
  reconvert_issue_codes: string[]
  autofix_issue_codes: string[]
  review_issue_codes: string[]
  issue_actions: Array<Record<string, unknown>>
}

export interface ConversionRepairAttempt {
  event: string
  status: string
  action?: string
  scope?: string
  speed_mode?: string
  issue_codes?: string[]
  task_id?: string
  source?: string
  reason?: string
  detail?: string
  created_at?: string
  extra?: Record<string, unknown>
}

export interface ConversionQualityReport {
  available: boolean
  stale: boolean
  path: string
  generated_at: string
  auto_repair_enabled: boolean
  auto_repair_changed: boolean
  auto_repair_unsafe: boolean
  auto_repair_applied: string[]
  issue_codes_before: string[]
  remaining_issue_codes: string[]
  regression_reasons: string[]
  repair_plan?: ConversionRepairPlan | null
  repair_attempt_count?: number
  latest_repair_attempt?: ConversionRepairAttempt | null
  repair_attempts?: ConversionRepairAttempt[]
  recommended_action: string
  needs_reconvert: boolean
  source_quality?: ConversionSourceQuality | null
  quality_center?: ConversionQualityCenterSummary | null
  source_quality_status?: string
  source_quality_message?: string
}

export interface ConversionSourceQuality {
  document_type?: 'research_article' | 'review' | 'supplementary' | string
  abstract_required?: boolean
  abstract_not_applicable?: boolean
  source_pdf_path?: string
  source_pdf_available?: boolean
  pdf_page_count?: number
  pdf_text_chars?: number
  pdf_pages?: number
  page_marker_count?: number
  matched_page_ratio?: number
  page_alignment_confidence?: 'high' | 'medium' | 'low' | 'missing' | 'unknown' | string
  references_line?: number
  references_char?: number
  references_char_ratio?: number
  body_heading_after_references_line?: number
  reference_line_count_before_body?: number
  references_before_body?: boolean
  abstract_autofix_likely?: boolean
  source_text_loss?: boolean
}

export interface ConversionQualityCenterSummary {
  available?: boolean
  status?: 'ready' | 'autofix' | 'reconvert' | 'review' | 'unknown' | string
  severity?: 'ok' | 'warning' | 'error' | string
  action?: string
  action_label?: string
  message?: string
  badges?: string[]
  issue_labels?: string[]
  issue_codes?: string[]
  source_quality?: ConversionSourceQuality | null
  report_path?: string
}

export interface ConversionQualitySummary {
  status: 'good' | 'warning' | 'error' | string
  label: string
  score: number
  summary: string
  has_review_issue: boolean
  issues: ConversionQualityIssue[]
  metrics: {
    chars?: number
    headings?: number
    page_markers?: number
    page_marker_gaps?: number
    figures?: number
    missing_images?: number
    captions?: number
    tables?: number
    display_math?: number
    inline_math?: number
    unclosed_display_math?: number
    references?: number
    reference_lines?: number
    body_citations?: number
    mojibake?: number
    analyzer_errors?: number
    analyzer_warnings?: number
  }
  conversion_report?: ConversionQualityReport | null
}

export interface LibraryMetaUpdateBody {
  pdf_name?: string
  sha1?: string
  path?: string
  paper_category: string
  reading_status: '' | 'unread' | 'reading' | 'done' | 'revisit'
  note: string
  user_tags: string[]
}

export interface LibraryMetaUpdateResponse {
  ok: boolean
  sha1: string
  path: string
  paper_category: string
  reading_status: '' | 'unread' | 'reading' | 'done' | 'revisit'
  note: string
  user_tags: string[]
  has_suggestions: boolean
  suggested_category: string
  suggested_tags: string[]
}

export interface LibraryMetaBatchUpdateBody {
  pdf_names?: string[]
  sha1s?: string[]
  apply_paper_category: boolean
  paper_category: string
  apply_reading_status: boolean
  reading_status: '' | 'unread' | 'reading' | 'done' | 'revisit'
  add_tags: string[]
  remove_tags: string[]
}

export interface LibraryMetaBatchUpdateItem {
  name: string
  sha1: string
  path: string
  paper_category: string
  reading_status: '' | 'unread' | 'reading' | 'done' | 'revisit'
  note: string
  user_tags: string[]
}

export interface LibraryMetaBatchUpdateResponse {
  ok: boolean
  requested: number
  updated: number
  items: LibraryMetaBatchUpdateItem[]
}

export interface LibrarySuggestionRegenerateBody {
  pdf_names?: string[]
  sha1s?: string[]
  auto_apply_empty?: boolean
}

export interface LibrarySuggestionActionBody {
  pdf_name?: string
  sha1?: string
  path?: string
  category_action?: '' | 'accept' | 'dismiss'
  accept_tags?: string[]
  dismiss_tags?: string[]
  accept_all_tags?: boolean
  dismiss_all_tags?: boolean
}

export interface LibrarySuggestionResponseItem {
  name: string
  sha1: string
  path: string
  paper_category: string
  reading_status: '' | 'unread' | 'reading' | 'done' | 'revisit'
  note: string
  user_tags: string[]
  has_suggestions: boolean
  suggested_category: string
  suggested_tags: string[]
}

export interface LibrarySuggestionRegenerateResponse {
  ok: boolean
  updated: number
  items: LibrarySuggestionResponseItem[]
}

export interface LibrarySuggestionActionResponse {
  ok: boolean
  sha1: string
  path: string
  paper_category: string
  reading_status: '' | 'unread' | 'reading' | 'done' | 'revisit'
  note: string
  user_tags: string[]
  has_suggestions: boolean
  suggested_category: string
  suggested_tags: string[]
}

export interface LibraryFilesResponse {
  items: LibraryFileItem[]
  counts: {
    total_view: number
    total_all: number
    pending: number
    converted: number
    queued: number
    running: number
    reconverting: number
    quality_review: number
    quality_ready: number
    index_ready?: number
    index_quality_blocked?: number
    index_stale?: number
  }
  truncated: boolean
  scope: string
  queue: {
    running: boolean
    queued_count?: number
    active_count: number
    active_tasks: ConvertActiveTask[]
    current: string
    done: number
    total: number
    recent_tasks?: ConversionTaskResult[]
  }
}

export interface RenameSuggestionItem {
  name: string
  path: string
  suggested_name: string
  suggested_stem: string
  display_full_name: string
  diff: boolean
  meta: {
    venue: string
    year: string
    title: string
    match_method: string
    year_source: string
    basis_label: string
    basis_detail: string
  }
  md_exists: boolean
  md_path: string
  md_folder: string
}

export interface RenameSuggestionsResponse {
  items: RenameSuggestionItem[]
  scope: string
  use_llm: boolean
  total_scanned: number
  changed: number
}

export interface RenameApplyResponse {
  ok: boolean
  renamed: number
  skipped: number
  failed: number
  needs_reindex: boolean
  index_cleanup?: {
    docs_removed?: number
    chunks_removed?: number
    reference_docs_removed?: number
    errors?: string[]
  }
  items: Array<Record<string, unknown>>
}

export interface UploadInspectResponse {
  name: string
  sha1: string
  duplicate: boolean
  existing: string
  existing_path: string
  suggested_name: string
  suggested_stem: string
  display_full_name: string
  meta: {
    venue: string
    year: string
    title: string
    match_method: string
    year_source: string
    basis_label: string
    basis_detail: string
  }
}

export interface UploadCommitResponse {
  duplicate?: boolean
  existing?: string
  path?: string
  name?: string
  sha1?: string
  citation_meta?: Record<string, unknown>
  enqueued: boolean
  task_id: string
}

export interface LibraryUploadResponse {
  name: string
  duplicate?: boolean
  existing?: string
}

export interface GuideSourceResponse {
  ok: boolean
  pdf_name: string
  pdf_path: string
  md_path: string
  md_exists: boolean
  source_path: string
  source_name: string
}

export interface LibraryReindexResponse {
  ok: boolean
  stdout: string
  stderr: string
  structured_indices: {
    version: number
    scanned: number
    rebuilt: number
    skipped: number
    failed: number
    citation_mention_count: number
    errors: Array<{ path: string; error: string }>
  } | null
  structured_indices_error: string
  refsync: {
    started?: boolean
    reason?: string
    run_id?: number
  } | null
  refsync_error: string
}

export interface LibrarySourceQualityItem {
  source_path: string
  source_name: string
  pdf_path: string
  md_path: string
  md_exists: boolean
  conversion_quality: ConversionQualitySummary | null
}

export interface LibrarySourceQualityResponse {
  ok: boolean
  items: LibrarySourceQualityItem[]
  review_count: number
}

export interface LibraryConversionQualityBatchBody {
  repair?: boolean
  rebuild_indices?: boolean
  limit?: number
}

export interface LibraryConversionQualityBatchResponse {
  ok: boolean
  mode: 'scan' | 'repair' | string
  target_count: number
  limit: number
  needs_reindex: boolean
  scanned: number
  repaired: number
  changed: number
  rebuilt: number
  ready: number
  autofix: number
  reconvert: number
  review: number
  unknown: number
  failed: number
  errors: Array<{ path: string; error: string }>
  changed_paths: string[]
  reconvert_paths: string[]
  review_paths: string[]
}

export interface LibraryFigureAssetIssue {
  code: 'missing_asset' | 'invalid_image' | 'low_resolution' | 'duplicate_asset' | 'suspicious_crop' | string
  severity: 'warning' | 'error' | string
  asset_name: string
  page: number
  figure_number: number
  message: string
  actual_width?: number
  actual_height?: number
  expected_width?: number
  expected_height?: number
  estimated_dpi?: number
  duplicates?: string[]
}

export interface LibraryFigureAssetDetail {
  asset_name: string
  page: number
  figure_number: number
  exists: boolean
  width: number
  height: number
  expected_width: number
  expected_height: number
  file_size?: number
  estimated_dpi?: number
  issue_codes: string[]
}

export interface LibraryFigureAssetScanItem {
  ok: boolean
  status: 'good' | 'warning' | 'error' | string
  source_name: string
  pdf_name: string
  pdf_path: string
  md_path: string
  assets_dir: string
  source_pdf_path: string
  source_pdf_available: boolean
  target_dpi: number
  figures: number
  issue_count: number
  issue_counts: Record<string, number>
  severity_counts?: Record<string, number>
  refresh_recommended: boolean
  issues: LibraryFigureAssetIssue[]
  assets: LibraryFigureAssetDetail[]
}

export interface LibraryFigureAssetScanResponse {
  ok: boolean
  target_count: number
  limit: number
  status: 'good' | 'warning' | 'error' | string
  scanned: number
  figures: number
  docs_with_issues: number
  refresh_recommended: number
  issue_counts: Record<string, number>
  severity_counts: Record<string, number>
  target_dpi?: number
  failed: number
  errors: Array<{ path?: string; name?: string; error: string }>
  items: LibraryFigureAssetScanItem[]
}

export interface LibraryFigureAssetRefreshBody {
  pdf_names?: string[]
  sources?: Array<{ source_path: string; source_name?: string }>
  limit?: number
  speed_mode?: string
  no_llm?: boolean
  replace?: boolean
  target_dpi?: number
}

export interface LibraryFigureAssetRefreshItem {
  source_name: string
  pdf_name: string
  pdf_path: string
  md_path: string
  issue_count: number
  issue_codes: string[]
  enqueued: boolean
  skipped_busy: boolean
  task_id: string
  error: string
}

export interface LibraryFigureAssetRefreshResponse {
  ok: boolean
  requested: number
  scanned: number
  figures: number
  docs_with_issues: number
  refresh_recommended: number
  issue_counts: Record<string, number>
  severity_counts: Record<string, number>
  enqueued: number
  skipped_busy: number
  failed: number
  errors: Array<{ path?: string; name?: string; error: string }>
  items: LibraryFigureAssetRefreshItem[]
}

export interface LibraryReaderLocateQualityPayload {
  source_path: string
  source_name?: string
  locate_feedback_key?: string
  locate_request_id?: number
  status: 'exact' | 'block' | 'fuzzy' | 'section' | 'source_only' | 'failed' | string
  precision: 'exact_anchor' | 'block' | 'phrase' | 'fuzzy' | 'section' | 'source_only' | 'failed' | string
  ok: boolean
  repairable: boolean
  strict_locate: boolean
  hint?: string
  reason?: string
  active_alt_index?: number
  block_id?: string
  anchor_id?: string
  anchor_kind?: string
  heading_path?: string
}

export interface LibraryReaderLocateSourceRecommendation {
  source_path: string
  source_name: string
  pdf_path: string
  md_path: string
  md_exists: boolean
  total: number
  failed: number
  degraded: number
  repairable: number
  strict_miss: number
  latest_status: string
  latest_precision: string
  latest_reason: string
  latest_at: number
  recommended_action: string
}

export interface LibraryReaderLocateQualitySummary {
  available: boolean
  status: 'good' | 'warning' | 'error' | 'unknown' | string
  summary: {
    total: number
    exact: number
    block: number
    degraded: number
    failed: number
    repairable: number
    strict_miss: number
    affected_sources: number
  }
  top_failures: Array<{ name: string, count: number }>
  recommended_sources: LibraryReaderLocateSourceRecommendation[]
  latest?: Array<Record<string, unknown>>
}

export interface LibraryReaderLocateQualityResponse {
  ok: boolean
  item: Record<string, unknown>
  summary: LibraryReaderLocateQualitySummary
}

export interface LibraryQualityOverviewIssue {
  code: string
  label: string
  severity: string
  papers: number
  count: number
  repairable?: boolean
  repair_strategy?: string
  repair_steps?: string[]
}

export interface LibraryQualityOverviewRecommendation {
  name: string
  path: string
  md_path: string
  status: string
  score: number
  summary: string
  task_state: string
  issues: ConversionQualityIssue[]
}

export interface LibraryQualityDomainSummary {
  [key: string]: string | number | boolean | null | undefined
}

export interface LibraryQualityDomain {
  available?: boolean
  status: 'good' | 'warning' | 'error' | 'unknown' | string
  summary?: LibraryQualityDomainSummary
  top_failures?: Array<{ name: string, count: number }>
  latest_path?: string
  report_path?: string
  updated_at?: number
}

export interface LibraryQualityPriorityAction {
  domain: string
  severity: 'good' | 'warning' | 'error' | 'unknown' | string
  label: string
  count: number
  detail?: string
}

export interface LibraryQualityFullChainStage {
  key: string
  label: string
  status: 'good' | 'warning' | 'error' | 'unknown' | string
  detail: string
  action: string
  count: number
  blocking: boolean
  metrics?: Record<string, string | number | boolean | null | undefined>
}

export interface LibraryQualityFullChainRootCause {
  code: string
  label: string
  domain: string
  count: number
  severity: 'good' | 'warning' | 'error' | 'unknown' | string
}

export interface LibraryQualityActionSnapshot {
  status?: 'good' | 'warning' | 'error' | 'unknown' | string
  score?: number
  count?: number
  summary?: string
  detail?: string
  blocking?: boolean
}

export interface LibraryQualityActionDelta {
  improved?: boolean | null
  worsened?: boolean
  status_changed?: boolean
  score_delta?: number
  count_delta?: number
  summary?: string
}

export interface LibraryQualityActionHistoryItem {
  id: string
  stage_key: string
  stage_label: string
  action: string
  status: 'success' | 'warning' | 'error' | 'info' | 'good' | string
  summary: string
  detail?: string
  target_ids?: string[]
  metrics?: Record<string, string | number | boolean | null | undefined>
  before?: LibraryQualityActionSnapshot
  after?: LibraryQualityActionSnapshot
  delta?: LibraryQualityActionDelta
  improved?: boolean | null
  verification?: Record<string, unknown>
  created_at: number
}

export interface LibraryQualityFullChain {
  available: boolean
  status: 'good' | 'warning' | 'error' | 'unknown' | string
  score: number
  summary: string
  stages: LibraryQualityFullChainStage[]
  root_causes: LibraryQualityFullChainRootCause[]
  next_actions: LibraryQualityPriorityAction[]
  action_history?: LibraryQualityActionHistoryItem[]
}

export interface LibraryQualityFeatureHealthItem {
  key: string
  label: string
  status: 'good' | 'warning' | 'error' | 'unknown' | string
  score: number
  summary: string
  detail: string
  action: string
  target_stage: string
  count: number
  blocking: boolean
  metrics?: Record<string, string | number | boolean | null | undefined>
}

export interface LibraryQualityFeatureHealth {
  available: boolean
  status: 'good' | 'warning' | 'error' | 'unknown' | string
  score: number
  summary: string
  items: LibraryQualityFeatureHealthItem[]
}

export interface LibraryQualityFailure {
  name: string
  domain: string
  detail?: string
}

export interface LibraryQualityDiagnosticIssue {
  name: string
  field?: string
  detail?: string
  severity?: 'warning' | 'error' | string
}

export interface LibraryQualityFieldCount {
  name: string
  count: number
}

export interface LibraryQualityCitationDiagnostic {
  route: 'system_a' | 'system_b' | string
  num: number
  anchor: string
  title: string
  source_name: string
  source_path: string
  authors?: string
  venue?: string
  year?: string
  doi?: string
  doi_url?: string
  raw?: string
  cite_fmt?: string
  summary_line?: string
  summary_quality?: Record<string, unknown>
  heading_path: string
  evidence_quote: string
  answer_claim?: string
  support_relation?: string
  trace?: string
  quality_issues?: LibraryQualityDiagnosticIssue[]
  shelf_quality_issues?: LibraryQualityDiagnosticIssue[]
  metadata_missing_fields?: string[]
  metadata_repairable?: boolean
  quality_issue_count?: number
}

export interface LibraryQualityRefDiagnostic {
  title: string
  source_name: string
  source_path: string
  heading_path: string
  score: number
  summary_line: string
  why_line: string
  polish_status: string
  ref_pack_state: string
  evidence_quote: string
  authors?: string
  venue?: string
  year?: string
  doi?: string
  doi_url?: string
  raw?: string
  cite_fmt?: string
  summary_quality?: Record<string, unknown>
  quality_issues?: LibraryQualityDiagnosticIssue[]
  quality_issue_count?: number
}

export interface LibraryQualitySourceDiagnostic {
  source_path: string
  source_name: string
  title: string
  roles: string[]
  pdf_path: string
  md_path: string
  md_exists: boolean
  repairable: boolean
  needs_repair: boolean
  quality_status: 'good' | 'warning' | 'error' | 'unknown' | string
  quality_score: number
  quality_summary: string
  quality_issues: ConversionQualityIssue[]
}

export interface LibraryQualityRootCause {
  code: string
  label: string
  severity: 'good' | 'warning' | 'error' | 'unknown' | string
  detail: string
  action: string
}

export interface LibraryQualityRepairAction {
  id: string
  kind: 'apply_repair_plan' | 'open_replay' | 'rerun_case' | 'repair_sources' | 'rebuild_index' | 'open_artifact' | string
  label: string
  severity: 'good' | 'warning' | 'error' | 'unknown' | string
  enabled: boolean
  detail: string
  target?: string
  source_count?: number
  steps?: Array<{
    kind: 'repair_sources' | 'repair_shelf_metadata' | 'rebuild_index' | 'rerun_case' | string
    label?: string
    target?: string
    source_count?: number
    target_count?: number
    missing_fields?: LibraryQualityFieldCount[]
  }>
  acceptance?: string
}

export interface LibraryQualityRerunStatus {
  available: boolean
  run_count: number
  last_status: 'passed' | 'failed' | 'error' | 'complete' | string
  last_quality_ok: boolean
  last_finished_at: number
  last_latency_ms: number
  last_passed_at: number
  consecutive_failed: number
  failure_names: string[]
  report_path: string
  raw_path: string
  error_kind?: string
  error_detail?: string
}

export interface LibraryQualityFailureCase {
  id: string
  question: string
  status: string
  conv_id: string
  latency_ms: number
  failures: LibraryQualityFailure[]
  failure_names: string[]
  expected_doc_ids: string[]
  ref_doc_ids: string[]
  citation_doc_ids: string[]
  missing_expected_doc_ids?: string[]
  doc_ids: string[]
  citation_count: number
  system_b_count: number
  ref_hit_count: number
  diagnostic_summary?: {
    citation_routes?: {
      system_a?: number
      system_b?: number
      [key: string]: number | undefined
    }
    missing_expected_doc_count?: number
    citation_diagnostic_count?: number
    ref_diagnostic_count?: number
    citation_card_failure_count?: number
    citation_card_warning_count?: number
    shelf_failure_count?: number
    shelf_warning_count?: number
    shelf_metadata_ready_count?: number
    shelf_export_ready_count?: number
    shelf_summary_export_ready_count?: number
    shelf_doi_count?: number
    shelf_source_clickable_count?: number
    shelf_review_count?: number
    shelf_missing_export_fields?: LibraryQualityFieldCount[]
    shelf_metadata_repair_target_count?: number
    ref_card_failure_count?: number
    ref_card_warning_count?: number
    system_b_needs_review_count?: number
    system_b_answer_context_only_count?: number
    system_b_reference_index_fallback_count?: number
  }
  citation_diagnostics?: LibraryQualityCitationDiagnostic[]
  ref_diagnostics?: LibraryQualityRefDiagnostic[]
  shelf_metadata_repair_targets?: Array<Record<string, unknown>>
  shelf_metadata_missing_fields?: LibraryQualityFieldCount[]
  source_diagnostics?: LibraryQualitySourceDiagnostic[]
  root_causes?: LibraryQualityRootCause[]
  repair_actions?: LibraryQualityRepairAction[]
  rerun_status?: LibraryQualityRerunStatus
  answer_preview: string
}

export interface LibraryQualityArtifactOpenResponse {
  ok: boolean
  domain: string
  target: string
  path: string
}

export interface LibraryQualityOverviewResponse {
  ok: boolean
  status: 'good' | 'warning' | 'error' | string
  summary: {
    total_view: number
    total_all: number
    converted: number
    pending: number
    queued: number
    running: number
    assessed: number
    good: number
    review: number
    unknown: number
    avg_score: number
  }
  top_issues: LibraryQualityOverviewIssue[]
  recommended: LibraryQualityOverviewRecommendation[]
  domains?: Record<string, LibraryQualityDomain>
  full_chain?: LibraryQualityFullChain
  feature_health?: LibraryQualityFeatureHealth
  reader_locate?: LibraryReaderLocateQualitySummary
  failure_cases?: LibraryQualityFailureCase[]
  rerun_summary?: {
    available: boolean
    total: number
    passed: number
    failed: number
    error: number
    case_count: number
    latest_finished_at: number
    latest_status: string
    top_failures?: Array<{ name: string, count: number }>
  }
  repair_runs?: LibraryQualityRepairRun[]
  priority_actions?: LibraryQualityPriorityAction[]
  queue: LibraryFilesResponse['queue']
  scope: string
  truncated: boolean
}

export interface LibraryQualityRepairItem {
  source_path: string
  source_name: string
  pdf_name: string
  pdf_path: string
  md_path?: string
  ok: boolean
  enqueued: boolean
  repaired?: boolean
  repair_changed?: boolean
  repair_applied?: string[]
  repair_before_score?: number
  repair_after_score?: number
  quality_before?: ConversionQualitySummary
  quality_after?: ConversionQualitySummary
  before_issue_codes?: string[]
  fixed_issue_codes?: string[]
  remaining_issue_codes?: string[]
  repair_plan?: ConversionRepairPlan
  planned_action?: string
  planned_scope?: string
  planned_speed_mode?: string
  planned_no_llm?: boolean
  reader_locate_problem_count?: number
  reader_locate_recommended_actions?: string[]
  reader_locate_problem_keys?: string[]
  reader_locate_reindex_required?: boolean
  repair_attempt?: ConversionRepairAttempt
  repair_error?: string
  skipped_busy: boolean
  error: string
  task_id: string
}

export interface LibraryQualityRepairImpact {
  requested: number
  repaired: number
  improved: number
  enqueued: number
  skipped_busy: number
  failed: number
  needs_reindex: boolean
  reader_locate_reindex?: number
  before_avg_score: number
  after_avg_score: number
  score_delta: number
  reindexed?: boolean
  fixed_issue_codes?: Array<{ name: string, count: number }>
  remaining_issue_codes?: Array<{ name: string, count: number }>
}

export interface LibraryQualityRepairRun {
  run_id: string
  status: string
  phase: string
  created_at: number
  updated_at: number
  requested: number
  enqueued: number
  repaired: number
  failed: number
  skipped_busy: number
  needs_reindex: boolean
  reindexed?: boolean | null
  target_names: string[]
  target_sources: string[]
  impact?: LibraryQualityRepairImpact | Record<string, unknown>
  verification?: Record<string, unknown>
  detail: string
}

export interface LibraryQualityRepairResponse {
  ok: boolean
  repair_run_id?: string
  repair_run?: LibraryQualityRepairRun
  requested: number
  enqueued: number
  repaired?: number
  needs_reindex?: boolean
  impact?: LibraryQualityRepairImpact
  skipped_busy: number
  failed: number
  items: LibraryQualityRepairItem[]
}

export interface LibraryQualityRepairRunAdvanceResponse {
  ok: boolean
  advanced: boolean
  waiting: boolean
  item: LibraryQualityRepairRun
  reindex: LibraryReindexResponse | null
  detail: string
}

export interface LibraryQualityRepairBody {
  pdf_names?: string[]
  sources?: Array<{ source_path: string; source_name?: string }>
  speed_mode?: string
  no_llm?: boolean
  replace?: boolean
  md_autofix?: boolean
}

export interface LibraryQualityActionHistoryBody {
  stage_key: string
  stage_label?: string
  action?: string
  status?: 'success' | 'warning' | 'error' | 'info' | 'good' | string
  summary: string
  detail?: string
  target_ids?: string[]
  metrics?: Record<string, string | number | boolean | null | undefined>
  before?: LibraryQualityActionSnapshot
  after?: LibraryQualityActionSnapshot
  delta?: LibraryQualityActionDelta
  improved?: boolean | null
  verification?: Record<string, unknown>
  created_at?: number
}

export interface LibraryResearchQaRerunResponse {
  ok: boolean
  case_id: string
  status: 'passed' | 'failed' | 'error' | 'complete' | string
  quality_ok: boolean
  returncode: number
  summary: Record<string, unknown>
  failures: LibraryQualityFailure[]
  output_dir: string
  report_path: string
  raw_path: string
  stdout_tail: string
  stderr_tail: string
  error_kind?: string
  error_detail?: string
  started_at: number
  finished_at: number
  latency_ms: number
}

export const libraryApi = {
  listPdfs: () => api.get<{ name: string; path: string }[]>('/api/library/pdfs'),
  listFiles: (scope = '200') =>
    api.get<LibraryFilesResponse>(`/api/library/files?scope=${encodeURIComponent(scope)}`),
  upload: async (file: File, baseName?: string) => {
    const fd = new FormData()
    fd.append('file', file)
    if (baseName) fd.append('base_name', baseName)
    const res = await authFetch('/api/library/upload', { method: 'POST', body: fd })
    return responseJson<LibraryUploadResponse>(res)
  },
  inspectUpload: async (file: File, useLlm = true) => {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('use_llm', String(useLlm))
    const res = await authFetch('/api/library/upload/inspect', { method: 'POST', body: fd })
    return responseJson<UploadInspectResponse>(res)
  },
  commitUpload: async (
    file: File,
    opts?: { baseName?: string; convertNow?: boolean; speedMode?: string; allowDuplicate?: boolean },
  ) => {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('base_name', String(opts?.baseName || ''))
    fd.append('convert_now', String(Boolean(opts?.convertNow)))
    fd.append('speed_mode', String(opts?.speedMode || 'balanced'))
    fd.append('allow_duplicate', String(Boolean(opts?.allowDuplicate)))
    const res = await authFetch('/api/library/upload/commit', { method: 'POST', body: fd })
    return responseJson<UploadCommitResponse>(res)
  },
  convert: (pdfName: string, speedMode = 'balanced', opts?: { replace?: boolean }) =>
    api.post('/api/library/convert', {
      pdf_name: pdfName,
      speed_mode: speedMode,
      no_llm: speedMode === 'no_llm',
      replace: opts?.replace ?? true,
    }),
  convertPending: (speedMode = 'balanced', limit = 0) =>
    api.post<{ ok: boolean; enqueued: number; skipped_busy: number; pending_total: number }>(
      '/api/library/convert/pending',
      { speed_mode: speedMode, limit, replace: true },
    ),
  cancelConvert: () => api.post<CancelConversionResponse>('/api/library/convert/cancel'),
  cancelConversionTask: (taskId: string) => api.post<CancelConversionResponse>(
    '/api/library/convert/cancel',
    { task_id: taskId },
  ),
  reindexFile: (pdfName: string) => api.post<{
    ok: boolean
    task_id: string
    pdf_name: string
    md_path: string
    outcome: ConversionOutcome
    message: string
    detail: string
  }>('/api/library/reindex/file', { pdf_name: pdfName }),
  openFile: (pdfName: string, target: 'pdf' | 'md' | 'pdf_dir' | 'md_dir' = 'pdf') =>
    api.post<{ ok: boolean; target: string; path: string }>('/api/library/file/open', {
      pdf_name: pdfName,
      target,
    }),
  deleteFile: (pdfName: string, alsoMd = true) =>
    api.post<{
      ok: boolean
      pdf_deleted: boolean
      md_deleted: boolean
      removed_queued: number
      index_cleanup?: {
        docs_removed?: number
        chunks_removed?: number
        reference_docs_removed?: number
        errors?: string[]
      }
      warnings: string[]
      needs_reindex: boolean
    }>(
      '/api/library/file/delete',
      {
        pdf_name: pdfName,
        also_md: alsoMd,
      },
    ),
  resolveGuideSource: (pdfName: string) =>
    api.post<GuideSourceResponse>('/api/library/file/guide_source', { pdf_name: pdfName }),
  listRenameSuggestions: (scope = '30', useLlm = true) =>
    api.get<RenameSuggestionsResponse>(
      `/api/library/rename/suggestions?scope=${encodeURIComponent(scope)}&use_llm=${String(useLlm)}`,
    ),
  applyRenameSuggestions: (
    pdfNames: string[],
    baseOverrides?: Record<string, string>,
    opts?: { useLlm?: boolean; alsoMd?: boolean },
  ) =>
    api.post<RenameApplyResponse>('/api/library/rename/apply', {
      pdf_names: pdfNames,
      base_overrides: baseOverrides || {},
      use_llm: Boolean(opts?.useLlm ?? true),
      also_md: Boolean(opts?.alsoMd ?? true),
    }),
  reindex: () => api.post<LibraryReindexResponse>('/api/library/reindex'),
  updateMeta: (body: LibraryMetaUpdateBody) =>
    api.post<LibraryMetaUpdateResponse>('/api/library/meta/update', body),
  batchUpdateMeta: (body: LibraryMetaBatchUpdateBody) =>
    api.post<LibraryMetaBatchUpdateResponse>('/api/library/meta/batch_update', body),
  regenerateSuggestions: (body: LibrarySuggestionRegenerateBody) =>
    api.post<LibrarySuggestionRegenerateResponse>('/api/library/meta/suggestions/regenerate', body),
  applySuggestionAction: (body: LibrarySuggestionActionBody) =>
    api.post<LibrarySuggestionActionResponse>('/api/library/meta/suggestions/apply', body),
  qualityOverview: (scope = 'all') =>
    api.get<LibraryQualityOverviewResponse>(`/api/library/quality/overview?scope=${encodeURIComponent(scope)}`),
  qualityActionHistory: (limit = 20) =>
    api.get<{ ok: boolean; items: LibraryQualityActionHistoryItem[] }>(`/api/library/quality/action-history?limit=${encodeURIComponent(String(limit))}`),
  recordQualityAction: (body: LibraryQualityActionHistoryBody) =>
    api.post<{ ok: boolean; item: LibraryQualityActionHistoryItem }>('/api/library/quality/action-history', body),
  sourceQuality: (sources: Array<{ source_path: string; source_name?: string }>) =>
    api.post<LibrarySourceQualityResponse>('/api/library/quality/sources', { sources }),
  conversionQualityBatch: (body: LibraryConversionQualityBatchBody = {}) =>
    api.post<LibraryConversionQualityBatchResponse>('/api/library/quality/conversion/batch', body),
  figureAssetQualityScan: (body: { limit?: number; include_all?: boolean; target_dpi?: number } = {}) =>
    api.post<LibraryFigureAssetScanResponse>('/api/library/quality/figure-assets/scan', body),
  refreshFigureAssets: (body: LibraryFigureAssetRefreshBody = {}) =>
    api.post<LibraryFigureAssetRefreshResponse>('/api/library/quality/figure-assets/refresh', body),
  recordReaderLocateQuality: (body: LibraryReaderLocateQualityPayload) =>
    api.post<LibraryReaderLocateQualityResponse>('/api/library/quality/reader-locate', body),
  openQualityArtifact: (domain: 'research_qa' | 'citation_cards' | string, target: 'report' | 'folder' | 'raw' | 'summary' | 'runbook' | string = 'report') =>
    api.post<LibraryQualityArtifactOpenResponse>('/api/library/quality/artifact/open', {
      domain,
      target,
    }),
  repairQuality: (body: LibraryQualityRepairBody) =>
    api.post<LibraryQualityRepairResponse>('/api/library/quality/repair', body),
  qualityRepairRuns: (limit = 20) =>
    api.get<{ ok: boolean; items: LibraryQualityRepairRun[] }>(`/api/library/quality/repair-runs?limit=${encodeURIComponent(String(limit))}`),
  qualityRepairRun: (runId: string) =>
    api.get<{ ok: boolean; item: LibraryQualityRepairRun }>(`/api/library/quality/repair-runs/${encodeURIComponent(runId)}`),
  updateQualityRepairRun: (runId: string, body: { status?: string; phase?: string; reindexed?: boolean; detail?: string; metrics?: Record<string, unknown> }) =>
    api.post<{ ok: boolean; item: LibraryQualityRepairRun }>(`/api/library/quality/repair-runs/${encodeURIComponent(runId)}`, body),
  advanceQualityRepairRun: (runId: string, body: { verify?: boolean; case_id?: string; base_url?: string; timeout_s?: number; top_k?: number; max_tokens?: number; dry_run?: boolean } = {}) =>
    api.post<LibraryQualityRepairRunAdvanceResponse>(`/api/library/quality/repair-runs/${encodeURIComponent(runId)}/advance`, body),
  rerunResearchQaCase: (body: { case_id: string, base_url?: string, timeout_s?: number, top_k?: number, max_tokens?: number, dry_run?: boolean }) =>
    api.post<LibraryResearchQaRerunResponse>('/api/library/quality/research-qa/rerun', body),

  streamConvertStatus: (
    onData: (data: ConvertProgress) => void,
    onDone: () => void,
    onError?: (err: unknown) => void,
  ): AbortController => {
    const ctrl = new AbortController()
    ;(async () => {
      try {
        const res = await authFetch('/api/library/convert/status', { signal: ctrl.signal })
        if (!res.ok) throw await responseError(res, 'conversion status failed')
        if (!res.body) throw new Error('conversion status stream is empty')
        const reader = res.body!.getReader()
        const decoder = new TextDecoder()
        let buf = ''
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buf += decoder.decode(value, { stream: true })
          const lines = buf.split('\n')
          buf = lines.pop() || ''
          for (const line of lines) {
            if (!line.startsWith('data: ')) continue
            try {
              const data = JSON.parse(line.slice(6)) as ConvertProgress
              onData(data)
              if (data.done) { onDone(); return }
            } catch { /* skip bad JSON */ }
          }
        }
        throw new Error('conversion status stream ended before completion')
      } catch (err) {
        if (!ctrl.signal.aborted) onError?.(err)
      }
    })()
    return ctrl
  },
}
