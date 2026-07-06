import type { StringMap } from '../../i18n'
import { tx } from './agentTracePanelUtils'
import type { ArchivedAgentTraceLoadStatus } from './useArchivedAgentTrace'

function storedTraceNote(loadStatus: ArchivedAgentTraceLoadStatus, labels: Partial<StringMap>) {
  if (loadStatus === 'loading') return tx(labels, 'agent_trace_loading_stored', 'Loading saved source check...')
  if (loadStatus === 'error') return tx(labels, 'agent_trace_load_failed', 'Saved source check could not be loaded.')
  if (loadStatus === 'empty') return tx(labels, 'agent_trace_no_stored', 'No saved source check is available.')
  return tx(labels, 'agent_trace_open_to_load', 'Open to load saved source check.')
}

export function AgentTraceStoredPrompt({
  labels,
  loadStatus,
  onLoad,
}: {
  labels: Partial<StringMap>
  loadStatus: ArchivedAgentTraceLoadStatus
  onLoad: () => void | Promise<void>
}) {
  return (
    <details className="kb-agent-trace" onToggle={(event) => {
      if ((event.currentTarget as HTMLDetailsElement).open) void onLoad()
    }}>
      <summary>
        <span>{tx(labels, 'agent_trace_title', 'Sources & evidence')}</span>
        <span>{tx(labels, 'agent_trace_stored', 'Saved check')}</span>
        <span>{loadStatus === 'loading' ? tx(labels, 'agent_trace_loading', 'loading') : tx(labels, 'agent_trace_open_load', 'open to load')}</span>
      </summary>
      <div className="kb-agent-trace-empty">{storedTraceNote(loadStatus, labels)}</div>
    </details>
  )
}
