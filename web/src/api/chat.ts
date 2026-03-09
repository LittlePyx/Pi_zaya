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
  rendered_content?: string
  rendered_body?: string
  notice?: string
  cite_details?: Array<Record<string, unknown>>
  copy_text?: string
  copy_markdown?: string
  refs_user_msg_id?: number
  render_cache_key?: string
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
  getMessages: (convId: string) =>
    api.get<Message[]>(`/api/conversations/${convId}/messages`),
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
