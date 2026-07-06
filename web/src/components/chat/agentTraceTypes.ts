import type { StringMap } from '../../i18n'

export type AgentTraceRecord = Record<string, unknown>

export type AgentTraceLabels = {
  labels: Partial<StringMap>
}
