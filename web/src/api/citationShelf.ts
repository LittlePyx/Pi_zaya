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
  acknowledged_mutation_id?: string
  concurrency_guarded?: boolean
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
  expectedRevision?: number
  mutationId?: string
  mutations?: Array<Record<string, unknown>>
}
