import type { StringMap } from '../../i18n'
import {
  shortText,
  verificationHeaderText,
} from './agentTracePanelUtils'

export type AgentTraceHeaderSummaryInput = {
  evidenceLabel: string
  totalClaims: number
  supportedClaims: number
  unsupportedClaims: number
  hasErrors: boolean
  scopeSummary: string
  taskLabel: string
}

export type AgentTraceHeaderSummary = {
  headerEvidence: string
  headerContext: string
}

export function buildAgentTraceHeaderSummary(
  labels: Partial<StringMap>,
  input: AgentTraceHeaderSummaryInput,
): AgentTraceHeaderSummary {
  const {
    evidenceLabel,
    totalClaims,
    supportedClaims,
    unsupportedClaims,
    hasErrors,
    scopeSummary,
    taskLabel,
  } = input
  const claimSummary = verificationHeaderText(totalClaims, supportedClaims, unsupportedClaims, hasErrors, labels)

  return {
    headerEvidence: evidenceLabel || claimSummary,
    headerContext: totalClaims > 0 && evidenceLabel
      ? claimSummary
      : (scopeSummary ? shortText(scopeSummary, 42) : taskLabel),
  }
}
