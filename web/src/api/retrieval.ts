import { api } from './client'

export type LocalSemanticPublicState =
  | 'not_installed'
  | 'preparing'
  | 'ready'
  | 'paused'
  | 'needs_space'
  | 'bm25_available'
  | 'removing'
  | 'unavailable'

export interface LocalSemanticLifecycleStatus {
  state: LocalSemanticPublicState
  available: boolean
  mode: 'hybrid_local'
  reason_id: string
  action_id: 'enable' | 'cancel' | 'remove' | 'resume' | 'free_space' | 'update_app' | 'none'
  task_id: string
  completed_bytes: number
  total_bytes: number
  eta_seconds: number | null
  required_additional_bytes: number
  bm25_available: true
}

export interface LocalSemanticComponentDisclosure {
  consent_scope: 'install_local_semantic_retrieval_component'
  archive_bytes: number
  installed_bytes: number
  network_required_for_install: boolean
  library_content_uploaded: false
  external_content_or_model_knowledge_added: false
  bm25_remains_available: true
}

export interface LocalSemanticComponentResponse {
  schema_version: 2
  default_mode: 'bm25'
  available_modes: Array<'bm25' | 'hybrid_local'>
  show_mode_control: boolean
  product_enabled: boolean
  install_available: boolean
  component: LocalSemanticComponentDisclosure
  hybrid_local: LocalSemanticLifecycleStatus
  created?: boolean
}

export const retrievalApi = {
  getLocalSemanticComponent: () =>
    api.get<LocalSemanticComponentResponse>('/api/retrieval/local-component'),
  enableLocalSemanticComponent: () =>
    api.post<LocalSemanticComponentResponse>('/api/retrieval/local-component/enable', {
      consent: true,
    }),
  resumeLocalSemanticComponent: () =>
    api.post<LocalSemanticComponentResponse>('/api/retrieval/local-component/resume', {}),
  cancelLocalSemanticComponent: (taskId: string) =>
    api.post<LocalSemanticComponentResponse>('/api/retrieval/local-component/cancel', {
      task_id: taskId,
    }),
  removeLocalSemanticComponent: () =>
    api.delete<LocalSemanticComponentResponse>('/api/retrieval/local-component'),
}
