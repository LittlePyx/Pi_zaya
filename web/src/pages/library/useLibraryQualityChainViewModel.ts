import { useMemo } from 'react'
import type {
  LibraryQualityActionHistoryItem,
  LibraryQualityFeatureHealth,
  LibraryQualityFeatureHealthItem,
  LibraryQualityFullChain,
  LibraryQualityFullChainRootCause,
  LibraryQualityFullChainStage,
  LibraryQualityOverviewResponse,
  LibraryQualityPriorityAction,
  LibraryReaderLocateSourceRecommendation,
} from '../../api/library'
import {
  normalizeTextValue,
  qualityActionDeltaText,
  qualityVerificationText,
} from './libraryPageUtils'

export type QualityFullChainActionResult = {
  status: 'success' | 'warning' | 'error' | 'info'
  summary: string
  detail?: string
  deltaText?: string
  verificationText?: string
  improved?: boolean | null
  updatedAt: number
}

type UseLibraryQualityChainViewModelParams = {
  backendQualityOverview: LibraryQualityOverviewResponse | null
}

export function useLibraryQualityChainViewModel({
  backendQualityOverview,
}: UseLibraryQualityChainViewModelParams) {
  const qualityPriorityActions = useMemo<LibraryQualityPriorityAction[]>(
    () => (Array.isArray(backendQualityOverview?.priority_actions) ? backendQualityOverview.priority_actions : [])
      .filter((item) => item && normalizeTextValue(item.domain))
      .slice(0, 4),
    [backendQualityOverview],
  )

  const actionableQualityPriorityActions = useMemo(
    () => qualityPriorityActions.filter((item) => (
      Number(item.count || 0) > 0
      || normalizeTextValue(item.severity).toLowerCase() === 'error'
    )),
    [qualityPriorityActions],
  )

  const qualityFullChain = useMemo<LibraryQualityFullChain | null>(() => {
    const fullChain = backendQualityOverview?.full_chain
    if (!fullChain || fullChain.available === false) return null
    return fullChain
  }, [backendQualityOverview])

  const qualityFullChainStages = useMemo<LibraryQualityFullChainStage[]>(
    () => (Array.isArray(qualityFullChain?.stages) ? qualityFullChain.stages : [])
      .filter((stage) => stage && normalizeTextValue(stage.key))
      .slice(0, 6),
    [qualityFullChain],
  )

  const qualityFullChainRootCauses = useMemo<LibraryQualityFullChainRootCause[]>(
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

  const qualityReaderLocateRecommendedSources = useMemo<LibraryReaderLocateSourceRecommendation[]>(() => {
    const sources = backendQualityOverview?.reader_locate?.recommended_sources
    if (!Array.isArray(sources)) return []
    return sources
      .filter((item) => item && (normalizeTextValue(item.source_path) || normalizeTextValue(item.source_name)))
      .slice(0, 12)
  }, [backendQualityOverview])

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

  return {
    qualityPriorityActions,
    actionableQualityPriorityActions,
    qualityFullChain,
    qualityFullChainStages,
    qualityFullChainRootCauses,
    qualityFullChainActionHistory,
    qualityFullChainPersistedResults,
    qualityReaderLocateRecommendedSources,
    qualityFeatureHealth,
    qualityFeatureHealthItems,
  }
}
