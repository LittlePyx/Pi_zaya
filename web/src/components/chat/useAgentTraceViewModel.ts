import { useMemo } from 'react'
import type { StringMap } from '../../i18n'
import { buildAgentTraceViewModel } from './agentTraceViewModel'
import type { AgentTraceViewModel } from './agentTraceViewModel'

export type {
  AgentSourceSummaryViewModel,
  AgentTraceDiagnosticsViewModel,
  AgentTraceViewModel,
} from './agentTraceViewModel'

export function useAgentTraceViewModel(trace: Record<string, unknown>, labels: Partial<StringMap>): AgentTraceViewModel {
  return useMemo(() => buildAgentTraceViewModel(trace, labels), [labels, trace])
}
