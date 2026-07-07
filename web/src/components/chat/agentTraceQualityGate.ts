import {
  compactStringList,
} from './agentTracePanelUtils'

export type AgentTraceQualityGateTitleInput = {
  reasons: unknown
  warnings: unknown
}

export function buildAgentTraceQualityGateTitle({
  reasons,
  warnings,
}: AgentTraceQualityGateTitleInput): string {
  return [
    ...compactStringList(reasons),
    ...compactStringList(warnings),
  ].join(' / ')
}
