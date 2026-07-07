import { useT } from '../../i18n'
import { internalDebugEnvEnabled } from '../../utils/internalDebug'
import { AgentTraceResolvedPanel } from './AgentTraceResolvedPanel'
import { AgentTraceStoredPrompt } from './AgentTraceStoredPrompt'
import { buildAgentTracePanelState } from './agentTracePanelState'
import type { AgentTraceReferenceHandlers } from './agentTraceReferenceTypes'
import { useArchivedAgentTrace, type LoadArchivedAgentTrace } from './useArchivedAgentTrace'
import { useAgentTraceViewModel } from './useAgentTraceViewModel'

export function AgentTracePanel({
  trace,
  messageId,
  canLoadTrace,
  onLoadTrace,
  onOpenReference,
  onAddReferenceToShelf,
}: AgentTraceReferenceHandlers & {
  trace?: Record<string, unknown> | null
  messageId?: number
  canLoadTrace?: boolean
  onLoadTrace?: LoadArchivedAgentTrace
}) {
  const S = useT()
  const {
    traceRecord,
    hasTrace,
    canLazyLoad,
    loadStatus,
    loadArchivedTrace,
  } = useArchivedAgentTrace({
    trace,
    messageId,
    canLoadTrace,
    onLoadTrace,
  })
  const viewModel = useAgentTraceViewModel(traceRecord, S)
  const panelState = buildAgentTracePanelState({
    traceRecord,
    hasTrace,
    canLazyLoad,
  })

  if (panelState === 'hidden') return null

  if (panelState === 'stored_prompt') {
    return (
      <AgentTraceStoredPrompt
        labels={S}
        loadStatus={loadStatus}
        onLoad={loadArchivedTrace}
      />
    )
  }

  const showDiagnostics = internalDebugEnvEnabled()

  return (
    <AgentTraceResolvedPanel
      labels={S}
      viewModel={viewModel}
      loadStatus={loadStatus}
      showDiagnostics={showDiagnostics}
      onOpen={loadArchivedTrace}
      onOpenReference={onOpenReference}
      onAddReferenceToShelf={onAddReferenceToShelf}
    />
  )
}
