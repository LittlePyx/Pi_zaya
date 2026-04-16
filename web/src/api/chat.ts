import { api } from './client'

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

export interface MessageMeta {
  provenance?: MessageProvenance
  answer_quality?: Record<string, unknown>
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

export interface ChatImageAttachment {
  sha1: string
  path: string
  name: string
  mime: string
  url?: string
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

export const chatApi = {
  listProjects: () =>
    api.get<Project[]>('/api/projects'),
  createProject: (name = '未命名项目') =>
    api.post<{ id: string }>('/api/projects', { name }),
  renameProject: (projectId: string, name: string) =>
    api.patch(`/api/projects/${projectId}`, { name }),
  deleteProject: (projectId: string) =>
    api.delete(`/api/projects/${projectId}`),
  listConversations: (limit = 80, projectId?: string | null, includeArchived = false) =>
    api.get<Conversation[]>(
      `/api/conversations?limit=${limit}`
      + `${projectId ? `&project_id=${encodeURIComponent(projectId)}` : ''}`
      + `${includeArchived ? '&include_archived=1' : ''}`,
    ),
  getConversation: (convId: string) =>
    api.get<Conversation>(`/api/conversations/${convId}`),
  createConversation: (
    title = '新对话',
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
  uploadFiles: async (files: File[], opts?: { quickIngest?: boolean; speedMode?: string; convId?: string | null }) => {
    const fd = new FormData()
    files.forEach((file) => fd.append('files', file))
    fd.append('quick_ingest', String(opts?.quickIngest ?? true))
    fd.append('speed_mode', opts?.speedMode ?? 'balanced')
    if (opts?.convId) fd.append('conv_id', String(opts.convId))
    const res = await fetch('/api/chat/uploads', { method: 'POST', body: fd })
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
    return res.json() as Promise<{ items: ChatUploadItem[] }>
  },
  getUploadStatuses: (jobIds: string[]) =>
    api.get<{ items: ChatUploadItem[] }>(`/api/chat/uploads/status?job_ids=${encodeURIComponent(jobIds.join(','))}`),
  retryUploadJob: (jobId: string) =>
    api.post<{ item: ChatUploadItem }>('/api/chat/uploads/retry', { job_id: jobId }),
  retryUploadQualityJob: (jobId: string) =>
    api.post<{ item: ChatUploadItem }>('/api/chat/uploads/quality/retry', { job_id: jobId }),
  cancelUploadJob: (jobId: string) =>
    api.post<{ item: ChatUploadItem }>('/api/chat/uploads/cancel', { job_id: jobId }),
  getRefs: (convId: string) =>
    api.get<Record<string, unknown>>(`/api/references/conversation/${convId}`),
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
}
