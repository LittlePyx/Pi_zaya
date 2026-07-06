import { AgentTraceFrame } from './AgentTraceFrame'
import type { AgentTraceLabels } from './agentTraceTypes'
import { tx } from './agentTracePanelUtils'
import type { ArchivedAgentTraceLoadStatus } from './useArchivedAgentTrace'

function storedTraceNote(loadStatus: ArchivedAgentTraceLoadStatus, labels: AgentTraceLabels['labels']) {
  if (loadStatus === 'loading') return tx(labels, 'agent_trace_loading_stored', 'Loading saved source check...')
  if (loadStatus === 'error') return tx(labels, 'agent_trace_load_failed', 'Saved source check could not be loaded.')
  if (loadStatus === 'empty') return tx(labels, 'agent_trace_no_stored', 'No saved source check is available.')
  return tx(labels, 'agent_trace_open_to_load', 'Open to load saved source check.')
}

export function AgentTraceStoredPrompt({
  labels,
  loadStatus,
  onLoad,
}: AgentTraceLabels & {
  loadStatus: ArchivedAgentTraceLoadStatus
  onLoad: () => void | Promise<void>
}) {
  return (
    <AgentTraceFrame
      labels={labels}
      summaryStatus={tx(labels, 'agent_trace_stored', 'Saved check')}
      summaryContext={loadStatus === 'loading' ? tx(labels, 'agent_trace_loading', 'loading') : tx(labels, 'agent_trace_open_load', 'open to load')}
      onOpen={onLoad}
    >
      <div className="kb-agent-trace-empty">{storedTraceNote(loadStatus, labels)}</div>
    </AgentTraceFrame>
  )
}
