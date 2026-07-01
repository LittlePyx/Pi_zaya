import { api, authFetch, responseError, responseJson } from './client'

export interface Conversation {
  id: string
  title: string
  created_at: number
  updated_at: number
  project_id?: string | null
  mode?: 'normal' | 'paper_guide'
  bound_source_path?: string
  bound_source_name?: string
  bound_source_ready?: number | boolean
  archived?: number | boolean
  archived_at?: number | null
}

export interface Project {
  id: string
  name: string
  created_at: number
  updated_at: number
}

export interface SidebarSnapshot {
  projects: Project[]
  root_conversations?: Conversation[]
  rootConversations?: Conversation[]
  project_conversations?: Record<string, Conversation[]>
  projectConversations?: Record<string, Conversation[]>
}

export interface Message {
  id: number
  role: string
  content: string
  created_at: number
  attachments?: ChatImageAttachment[]
  meta?: MessageMeta
  provenance?: MessageProvenance
  rendered_content?: string
  rendered_body?: string
  notice?: string
  cite_details?: Array<Record<string, unknown>>
  copy_text?: string
  copy_markdown?: string
  refs_user_msg_id?: number
  render_cache_key?: string
}

export interface RefsResponseMeta {
  serverTiming: string
  mode: string
  counts: string
  durationMs: number
}

export interface RefsResponseWithMeta {
  data: Record<string, unknown>
  meta: RefsResponseMeta
}

export interface AgentTracePlanStep {
  goal?: string
  tool?: string
  status?: string
}

export interface AgentTraceExecutionStep {
  tool?: string
  status?: string
  observation?: string
  output?: Record<string, unknown>
  error?: string
  elapsed_ms?: number
}

export interface AgentTraceVerification {
  total_claims?: number
  supported_claims?: number
  unsupported_claims?: number
  local_claims?: number
  external_background_claims?: number
  source_notice_count?: number
  support_ratio?: number
  evidence_status?: 'grounded' | 'needs_review' | 'insufficient' | 'not_applicable' | string
  evidence_hit_count?: number
  evidence_status_reasons?: string[]
  claims?: Array<Record<string, unknown>>
}

export interface AgentTraceSummary {
  question_type?: string
  status?: string
  research_run_status?: string
  source_policy?: string
  subtask_count?: number
  evidence_matrix_rows?: number
  query_scope?: string
  requested_query_scope?: string
  retrieval_confidence?: string
  usable_hit_count?: number
  total_claims?: number
  supported_claims?: number
  unsupported_claims?: number
  local_claims?: number
  external_background_claims?: number
  source_notice_count?: number
  support_ratio?: number
  evidence_status?: 'grounded' | 'needs_review' | 'insufficient' | 'not_applicable' | string
  evidence_hit_count?: number
  evidence_status_reasons?: string[]
  quality_gate_status?: string
  quality_gate_reasons?: string[]
  quality_gate_warnings?: string[]
  plan_step_count?: number
  tool_call_count?: number
  has_errors?: boolean
  schema_ok?: boolean
  [key: string]: unknown
}

export interface EvidenceMatrixRow {
  paper?: string
  source_name?: string
  source_path?: string
  method?: string
  dataset_or_experiment?: string
  key_result?: string
  limitation?: string
  evidence_quote?: string
  citation?: string
  heading_path?: string
  support_status?: string
  [key: string]: unknown
}

export interface ResearchRun {
  run_id?: string
  status?: string
  source_policy?: string
  query_scope?: string
  question?: string
  subtasks?: Array<Record<string, unknown>>
  evidence_matrix?: EvidenceMatrixRow[]
  metrics?: Record<string, unknown>
  [key: string]: unknown
}

export interface AgentTrace {
  mode?: 'research_agent' | string
  question_type?: string
  context?: Record<string, unknown>
  plan?: AgentTracePlanStep[]
  steps?: AgentTraceExecutionStep[]
  verification?: AgentTraceVerification
  research_run?: ResearchRun | Record<string, unknown>
  summary?: AgentTraceSummary | Record<string, unknown>
  status?: string
  errors?: string[]
  [key: string]: unknown
}

export interface AgentTraceAuditResponse {
  message_id: number
  conv_id: string
  available: boolean
  agent_trace?: AgentTrace | Record<string, unknown>
  summary?: AgentTraceSummary | Record<string, unknown>
  schema_errors?: string[]
}

export interface MessageMeta {
  provenance?: MessageProvenance
  answer_quality?: Record<string, unknown>
  agent_trace?: AgentTrace | Record<string, unknown>
  paper_guide_contracts?: MessagePaperGuideContracts
  render_cache?: Record<string, unknown>
  [key: string]: unknown
}

export interface MessagePaperGuideContracts {
  version?: number
  intent?: Record<string, unknown>
  prompt_context?: Record<string, unknown>
  retrieval_bundle?: Record<string, unknown>
  support_pack?: Record<string, unknown>
  grounding_trace?: Array<Record<string, unknown>>
  render_packet?: MessageRenderPacket
  [key: string]: unknown
}

export interface MessageCitationDetail {
  num?: number
  anchor?: string
  source_name?: string
  source_path?: string
  raw?: string
  title?: string
  authors?: string
  venue?: string
  year?: string
  volume?: string
  issue?: string
  pages?: string
  doi?: string
  doi_url?: string
  cite_fmt?: string
  linked_nums?: number[]
  evidence_fingerprint?: string
  render_locale?: string
  citation_route?: string
  routing_reason?: string
  routing_confidence?: number
  heading_path?: string
  evidence_quote?: string
  why_line?: string
  summary_line?: string
  summary_source?: string
  answer_claim?: string
  evidence_source?: string
  citation_context?: string
  citation_context_source?: string
  upstream_work_role?: string
  user_question_relation?: string
  location_label?: string
  support_relation?: string
  block_id?: string
  anchor_id?: string
  anchor_kind?: string
  page_start?: number
  page_end?: number
  score?: number
  binding_status?: string
  binding_confidence?: number
  binding_reason?: string
  binding_overlap_terms?: string[]
  card_kind?: string
  card_title?: string
  card_subtitle?: string
  card_takeaway_label?: string
  card_takeaway?: string
  card_claim_label?: string
  card_claim?: string
  card_locator_label?: string
  card_locator?: string
  card_evidence_label?: string
  card_evidence?: string
  card_support_label?: string
  card_support_explanation?: string
  card_quality_label?: string
  card_quality_score?: number
  card_quality_flags?: string[]
  card_warning?: string
  card_flow?: string[]
  [key: string]: unknown
}

export interface MessageUnlinkedReferenceCandidate {
  id?: string
  status?: string
  match_method?: string
  confidence?: number
  mention?: string
  source_path?: string
  source_name?: string
  ref_num?: number
  title?: string
  authors?: string
  venue?: string
  year?: string
  doi?: string
  doi_url?: string
  raw?: string
  cite_detail?: MessageCitationDetail
  [key: string]: unknown
}

export interface MessageRenderPacket {
  answer_markdown?: string
  notice?: string
  rendered_body?: string
  rendered_content?: string
  copy_markdown?: string
  copy_text?: string
  cite_details?: MessageCitationDetail[]
  citation_validation?: Record<string, unknown>
  locate_target?: MessageProvenanceLocateTarget
  reader_open?: MessageProvenanceReaderOpen
  segment_ids?: string[]
  visible_segment_ids?: string[]
  provenance_segment_count?: number
  visible_segment_count?: number
  unlinked_reference_candidates?: MessageUnlinkedReferenceCandidate[]
  [key: string]: unknown
}

export interface MessagePage {
  messages: Message[]
  has_more_before: boolean
  oldest_loaded_id?: number | null
  newest_loaded_id?: number | null
}

export interface MessageProvenanceBlock {
  block_id: string
  anchor_id?: string
  kind?: string
  heading_path?: string
  text?: string
  line_start?: number
  line_end?: number
  number?: number
}

export interface MessageProvenanceLocateTarget {
  segmentId?: string
  sourceSegmentId?: string
  headingPath?: string
  snippet?: string
  highlightSnippet?: string
  evidenceQuote?: string
  anchorText?: string
  hitLevel?: 'exact' | 'block' | 'heading' | 'none' | string
  blockId?: string
  anchorId?: string
  anchorKind?: string
  anchorNumber?: number
  claimType?: string
  locatePolicy?: string
  locateSurfacePolicy?: string
  snippetAliases?: string[]
  relatedBlockIds?: string[]
}

export interface MessageProvenanceClaimGroup {
  id?: string
  kind?: string
  leadText?: string
  distance?: number
}

export interface MessageProvenanceReaderOpenCandidate {
  headingPath?: string
  snippet?: string
  highlightSnippet?: string
  anchorId?: string
  blockId?: string
  anchorKind?: string
  anchorNumber?: number
}

export interface MessageProvenanceReaderOpen {
  sourcePath?: string
  sourceName?: string
  headingPath?: string
  snippet?: string
  highlightSnippet?: string
  anchorId?: string
  blockId?: string
  relatedBlockIds?: string[]
  anchorKind?: string
  anchorNumber?: number
  strictLocate?: boolean
  locateTarget?: MessageProvenanceLocateTarget
  claimGroup?: MessageProvenanceClaimGroup
  alternatives?: MessageProvenanceReaderOpenCandidate[]
  visibleAlternatives?: MessageProvenanceReaderOpenCandidate[]
  evidenceAlternatives?: MessageProvenanceReaderOpenCandidate[]
  initialAltIndex?: number
}

export interface ReaderSessionRecord {
  id: string
  title?: string
  conversation_id?: string
  message_id?: number | null
  payload: MessageProvenanceReaderOpen & Record<string, unknown>
  state?: Record<string, unknown>
  created_at?: number
  updated_at?: number
}

export interface ConversationReaderStateRecord {
  conv_id: string
  source_path: string
  state: Record<string, unknown>
  created_at?: number
  updated_at?: number
}

export interface ConversationResearchStateRecord {
  conv_id: string
  state: Record<string, unknown>
  created_at?: number
  updated_at?: number
}

export type QueryScope = 'current_paper' | 'basket' | 'library'

export interface MessageProvenanceSegment {
  segment_id: string
  segment_index?: number
  kind?: string
  segment_type?: string
  claim_type?: 'quote_claim' | 'blockquote_claim' | 'formula_claim' | 'inline_formula_claim' | 'equation_explanation_claim' | 'figure_claim' | 'critical_fact_claim' | 'shell_sentence' | string
  must_locate?: boolean
  locate_policy?: 'required' | 'optional' | 'hidden' | string
  locate_surface_policy?: 'primary' | 'secondary' | 'hidden' | string
  claim_group_id?: string
  claim_group_kind?: 'formula_bundle' | 'quote_bundle' | string
  claim_group_target_segment_id?: string
  claim_group_target_distance?: number
  claim_group_lead_text?: string
  formula_origin?: 'source' | 'explanation' | 'derived' | string
  text: string
  raw_markdown?: string
  display_markdown?: string
  cite_details?: Array<Record<string, unknown>>
  snippet_key?: string
  snippet_aliases?: string[]
  evidence_mode?: 'direct' | 'synthesis' | 'none' | string
  hit_level?: 'exact' | 'block' | 'heading' | 'none' | string
  evidence_block_ids?: string[]
  primary_block_id?: string
  primary_anchor_id?: string
  primary_heading_path?: string
  support_block_ids?: string[]
  related_block_ids?: string[]
  evidence_quote?: string
  evidence_confidence?: number
  mapping_quality?: number
  mapping_source?: 'fast' | 'llm_refined' | string
  anchor_kind?: 'quote' | 'blockquote' | 'equation' | 'inline_formula' | 'figure' | 'sentence' | string
  anchor_text?: string
  equation_number?: number
  support_slot_figure_number?: number
  support_slot_panel_letters?: string[]
  strict_identity_missing_reasons?: string[]
  locate_target?: MessageProvenanceLocateTarget
  reader_open?: MessageProvenanceReaderOpen
}

export interface MessageProvenance {
  version?: number
  provenance_schema_version?: number
  status?: string
  mapping_mode?: 'fast' | 'llm_refined' | string
  llm_rerank_enabled?: boolean
  llm_rerank_calls?: number
  strict_identity_ready?: boolean
  must_locate_candidate_count?: number
  must_locate_count?: number
  strict_identity_count?: number
  identity_missing_reasons?: Record<string, number>
  identity_missing_segments?: Array<Record<string, unknown>>
  source_path?: string
  source_name?: string
  md_path?: string
  doc_id?: string
  candidate_block_count?: number
  block_map?: Record<string, MessageProvenanceBlock>
  segments?: MessageProvenanceSegment[]
}

const refsNowMs = () => (
  typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now()
)

async function getRefsWithMeta(convId: string, init?: RequestInit): Promise<RefsResponseWithMeta> {
  const startedAt = refsNowMs()
  const res = await authFetch(`/api/references/conversation/${convId}`, init)
  if (!res.ok) {
    throw await responseError(res)
  }
  const data = await responseJson<Record<string, unknown>>(res)
  return {
    data,
    meta: {
      serverTiming: res.headers.get('server-timing') || '',
      mode: res.headers.get('x-kb-refs-mode') || '',
      counts: res.headers.get('x-kb-refs-counts') || '',
      durationMs: Number((refsNowMs() - startedAt).toFixed(2)),
    },
  }
}

export interface ChatImageAttachment {
  sha1: string
  path: string
  name: string
  mime: string
  url?: string
}

export interface CitationShelfRecord {
  version: number
  scope: string
  scope_id: string
  project_id?: string | null
  items: Array<Record<string, unknown>>
  open: boolean
  revision: number
  created_at: number
  updated_at: number
}

export interface CitationShelfRequest {
  convId?: string | null
  projectId?: string | null
  scope?: string
}

export interface CitationShelfSaveBody extends CitationShelfRequest {
  items: Array<Record<string, unknown>>
  open?: boolean
  allowEmptyOverwrite?: boolean
}

export interface ChatUploadItem {
  kind: 'pdf' | 'image' | 'unknown'
  status: 'saved' | 'duplicate' | 'error' | 'unsupported'
  name: string
  sha1?: string
  path?: string
  mime?: string
  existing?: string
  error?: string
  ready?: boolean
  ingest_status?: 'idle' | 'processing' | 'renaming' | 'converting' | 'ingesting' | 'ready' | 'error' | 'cancelled'
  quality_status?: 'none' | 'pending' | 'running' | 'ready' | 'error' | 'cancelled'
  quality_stage?: string
  quality_error?: string
  ingest_job_id?: string
  md_path?: string
  attachment?: ChatImageAttachment
}

function citationShelfUrl(opts?: CitationShelfRequest): string {
  const params = new URLSearchParams()
  const convId = String(opts?.convId || '').trim()
  const projectId = String(opts?.projectId || '').trim()
  const scope = String(opts?.scope || 'project').trim()
  if (convId) params.set('conv_id', convId)
  if (projectId) params.set('project_id', projectId)
  if (scope) params.set('scope', scope)
  const query = params.toString()
  return `/api/chat/citation-shelf${query ? `?${query}` : ''}`
}

function citationShelfItemsUrl(opts?: CitationShelfRequest): string {
  const url = citationShelfUrl(opts)
  return url.replace('/api/chat/citation-shelf', '/api/chat/citation-shelf/items')
}

export const chatApi = {
  listProjects: () =>
    api.get<Project[]>('/api/projects'),
  createProject: (name: string) =>
    api.post<{ id: string }>('/api/projects', { name }),
  renameProject: (projectId: string, name: string) =>
    api.patch(`/api/projects/${projectId}`, { name }),
  deleteProject: (projectId: string) =>
    api.delete(`/api/projects/${projectId}`),
  getSidebar: (limit = 80, includeArchived = false) =>
    api.get<SidebarSnapshot>(
      `/api/sidebar?limit=${limit}${includeArchived ? '&include_archived=1' : ''}`,
    ),
  listConversations: (limit = 80, projectId?: string | null, includeArchived = false) =>
    api.get<Conversation[]>(
      `/api/conversations?limit=${limit}`
      + `${projectId ? `&project_id=${encodeURIComponent(projectId)}` : ''}`
      + `${includeArchived ? '&include_archived=1' : ''}`,
    ),
  getConversation: (convId: string) =>
    api.get<Conversation>(`/api/conversations/${convId}`),
  createConversation: (
    title: string,
    projectId?: string | null,
    guide?: {
      mode?: 'normal' | 'paper_guide'
      bound_source_path?: string
      bound_source_name?: string
      bound_source_ready?: boolean
    },
  ) =>
    api.post<{ id: string }>('/api/conversations', {
      title,
      project_id: projectId ?? null,
      mode: guide?.mode ?? 'normal',
      bound_source_path: guide?.bound_source_path ?? '',
      bound_source_name: guide?.bound_source_name ?? '',
      bound_source_ready: Boolean(guide?.bound_source_ready),
    }),
  deleteConversation: (id: string) =>
    api.delete(`/api/conversations/${id}`),
  getMessages: (convId: string, opts?: { renderPacketOnly?: boolean }) =>
    api.get<Message[]>(
      `/api/conversations/${convId}/messages`
      + `${typeof opts?.renderPacketOnly === 'boolean' ? `?render_packet_only=${opts.renderPacketOnly ? 1 : 0}` : ''}`,
    ),
  getMessagesPage: (convId: string, opts?: { limit?: number; beforeId?: number | null; renderPacketOnly?: boolean }) =>
    api.get<MessagePage>(
      (() => {
        const limit = Math.max(1, Math.floor(Number(opts?.limit || 24)))
        const beforeId = Number(opts?.beforeId || 0)
        const beforePart = Number.isFinite(beforeId) && beforeId > 0 ? `&before_id=${Math.floor(beforeId)}` : ''
        const packetPart = typeof opts?.renderPacketOnly === 'boolean' ? `&render_packet_only=${opts.renderPacketOnly ? 1 : 0}` : ''
        return `/api/conversations/${convId}/messages_page?limit=${limit}${beforePart}${packetPart}`
      })(),
    ),
  appendMessage: (convId: string, role: string, content: string) =>
    api.post<{ id: number }>(`/api/conversations/${convId}/messages`, { role, content }),
  getMessageAgentTrace: (messageId: number, convId?: string | null) =>
    api.get<AgentTraceAuditResponse>(
      `/api/messages/${Math.floor(Number(messageId || 0))}/agent-trace`
      + `${convId ? `?conv_id=${encodeURIComponent(convId)}` : ''}`,
    ),
  uploadFiles: async (files: File[], opts?: { quickIngest?: boolean; speedMode?: string; convId?: string | null }) => {
    const fd = new FormData()
    files.forEach((file) => fd.append('files', file))
    fd.append('quick_ingest', String(opts?.quickIngest ?? true))
    fd.append('speed_mode', opts?.speedMode ?? 'balanced')
    if (opts?.convId) fd.append('conv_id', String(opts.convId))
    const res = await authFetch('/api/chat/uploads', { method: 'POST', body: fd })
    return responseJson<{ items: ChatUploadItem[] }>(res)
  },
  getUploadStatuses: (jobIds: string[]) =>
    api.get<{ items: ChatUploadItem[] }>(`/api/chat/uploads/status?job_ids=${encodeURIComponent(jobIds.join(','))}`),
  retryUploadJob: (jobId: string) =>
    api.post<{ item: ChatUploadItem }>('/api/chat/uploads/retry', { job_id: jobId }),
  retryUploadQualityJob: (jobId: string) =>
    api.post<{ item: ChatUploadItem }>('/api/chat/uploads/quality/retry', { job_id: jobId }),
  cancelUploadJob: (jobId: string) =>
    api.post<{ item: ChatUploadItem }>('/api/chat/uploads/cancel', { job_id: jobId }),
  getCitationShelf: (opts?: CitationShelfRequest) =>
    api.get<CitationShelfRecord>(citationShelfUrl(opts)),
  saveCitationShelf: (body: CitationShelfSaveBody) =>
    api.patch<CitationShelfRecord>(citationShelfUrl(body), {
      items: body.items,
      open: Boolean(body.open),
      scope: body.scope || 'project',
      project_id: body.projectId ?? null,
      allow_empty_overwrite: body.allowEmptyOverwrite ?? false,
    }),
  appendCitationShelfItem: (body: CitationShelfRequest & { item: Record<string, unknown>; open?: boolean }) =>
    api.post<CitationShelfRecord>(citationShelfItemsUrl(body), {
      item: body.item,
      open: body.open ?? true,
      scope: body.scope || 'project',
      project_id: body.projectId ?? null,
    }),
  deleteCitationShelf: (opts?: CitationShelfRequest) =>
    api.delete<CitationShelfRecord>(citationShelfUrl(opts)),
  getRefsWithMeta,
  getRefs: async (convId: string, init?: RequestInit) =>
    (await getRefsWithMeta(convId, init)).data,
  updateTitle: (convId: string, title: string) =>
    api.patch(`/api/conversations/${convId}/title`, { title }),
  updateConversationProject: (convId: string, projectId?: string | null) =>
    api.patch(`/api/conversations/${convId}/project`, { project_id: projectId ?? null }),
  updateConversationGuide: (
    convId: string,
    guide: {
      mode?: 'normal' | 'paper_guide'
      bound_source_path?: string
      bound_source_name?: string
      bound_source_ready?: boolean
    },
  ) =>
    api.patch(`/api/conversations/${convId}/guide`, {
      mode: guide.mode,
      bound_source_path: guide.bound_source_path,
      bound_source_name: guide.bound_source_name,
      bound_source_ready: guide.bound_source_ready,
    }),
  getConversationResearchState: (convId: string) =>
    api.get<ConversationResearchStateRecord>(`/api/conversations/${encodeURIComponent(convId)}/research-state`),
  patchConversationResearchState: (convId: string, state: Record<string, unknown>) =>
    api.patch<ConversationResearchStateRecord>(`/api/conversations/${encodeURIComponent(convId)}/research-state`, { state }),
  createReaderSession: <T extends MessageProvenanceReaderOpen>(
    payload: T,
    opts?: {
      title?: string
      conversationId?: string | null
      messageId?: number | null
      state?: Record<string, unknown>
    },
  ) =>
    api.post<ReaderSessionRecord>('/api/reader/sessions', {
      payload,
      state: opts?.state ?? {},
      title: opts?.title ?? '',
      conversation_id: opts?.conversationId ?? '',
      message_id: opts?.messageId ?? null,
    }),
  getReaderSession: (sessionId: string) =>
    api.get<ReaderSessionRecord>(`/api/reader/sessions/${encodeURIComponent(sessionId)}`),
  updateReaderSessionState: (sessionId: string, state: Record<string, unknown>) =>
    api.patch<ReaderSessionRecord>(`/api/reader/sessions/${encodeURIComponent(sessionId)}/state`, { state }),
  getConversationReaderState: (convId: string, sourcePath: string) =>
    api.get<ConversationReaderStateRecord>(
      `/api/conversations/${encodeURIComponent(convId)}/reader-state?source_path=${encodeURIComponent(sourcePath)}`,
    ),
  updateConversationReaderState: (convId: string, sourcePath: string, state: Record<string, unknown>) =>
    api.patch<ConversationReaderStateRecord>(
      `/api/conversations/${encodeURIComponent(convId)}/reader-state?source_path=${encodeURIComponent(sourcePath)}`,
      { state },
    ),
}
