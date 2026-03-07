import { api } from './client'

export interface AnswerQualityIntentSummary {
  count: number
  structure_complete_rate: number
  evidence_coverage_rate: number
  next_steps_coverage_rate: number
  minimum_ok_rate: number
}

export interface AnswerQualityDepthSummary {
  count: number
  minimum_ok_rate: number
  avg_char_count: number
}

export interface AnswerQualitySummary {
  limit: number
  filters: {
    intent: string
    depth: string
    only_failed: boolean
  }
  total: number
  failed_count: number
  failed_rate: number
  structure_complete_rate: number
  evidence_coverage_rate: number
  next_steps_coverage_rate: number
  minimum_ok_rate: number
  avg_core_section_coverage: number
  by_intent: Record<string, AnswerQualityIntentSummary>
  by_depth: Record<string, AnswerQualityDepthSummary>
  fail_reasons: Record<string, number>
}

export interface AnswerQualitySummaryParams {
  limit?: number
  intent?: string
  depth?: string
  onlyFailed?: boolean
}

export const generateApi = {
  qualitySummary: (params: AnswerQualitySummaryParams = {}) => {
    const limit = Math.max(20, Math.min(2000, Math.floor(Number(params.limit || 200))))
    const q = new URLSearchParams()
    q.set('limit', String(limit))
    if (params.intent) q.set('intent', String(params.intent).trim())
    if (params.depth) q.set('depth', String(params.depth).trim())
    if (params.onlyFailed) q.set('only_failed', 'true')
    return api.get<AnswerQualitySummary>(`/api/generate/quality/summary?${q.toString()}`)
  },
}
