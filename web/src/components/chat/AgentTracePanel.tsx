import { asTraceRecord, traceNum } from './messageTraceUtils'

function records(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value)
    ? value.map((item) => asTraceRecord(item)).filter((item) => Object.keys(item).length > 0)
    : []
}

function shortText(value: unknown, limit = 180): string {
  const text = String(value || '').replace(/\s+/g, ' ').trim()
  return text.length > limit ? `${text.slice(0, limit - 3).trim()}...` : text
}

function statusClass(status: unknown) {
  const s = String(status || '').trim().toLowerCase()
  if (s === 'error' || s === 'canceled') return 'is-warning'
  if (s === 'done') return 'is-done'
  return ''
}

export function AgentTracePanel({ trace }: { trace?: Record<string, unknown> | null }) {
  const tr = asTraceRecord(trace)
  if (Object.keys(tr).length <= 0) return null
  const mode = String(tr.mode || '').trim()
  if (mode && mode !== 'research_agent') return null

  const plan = records(tr.plan)
  const steps = records(tr.steps)
  const verification = asTraceRecord(tr.verification)
  const totalClaims = traceNum(verification.total_claims)
  const supportedClaims = traceNum(verification.supported_claims)
  const unsupportedClaims = traceNum(verification.unsupported_claims)
  const questionType = String(tr.question_type || 'unknown').trim()
  const status = String(tr.status || '').trim() || 'done'

  return (
    <details className="kb-agent-trace">
      <summary>
        <span>Research Agent Trace</span>
        <span>{questionType}</span>
        <span>{status}</span>
      </summary>
      <div className="kb-agent-trace-summary">
        <div>
          <span>Claims</span>
          <strong>{supportedClaims}/{totalClaims}</strong>
        </div>
        <div className={unsupportedClaims > 0 ? 'is-warning' : ''}>
          <span>Unsupported</span>
          <strong>{unsupportedClaims}</strong>
        </div>
      </div>
      {plan.length > 0 ? (
        <div className="kb-agent-trace-section">
          <div className="kb-agent-trace-heading">Plan</div>
          {plan.map((step, idx) => (
            <div className="kb-agent-trace-row" key={`${String(step.tool || 'plan')}-${idx}`}>
              <span className={`kb-agent-trace-status ${statusClass(step.status)}`}>{String(step.status || 'pending')}</span>
              <span className="kb-agent-trace-tool">{String(step.tool || '')}</span>
              <span className="kb-agent-trace-text">{shortText(step.goal)}</span>
            </div>
          ))}
        </div>
      ) : null}
      {steps.length > 0 ? (
        <div className="kb-agent-trace-section">
          <div className="kb-agent-trace-heading">Tool Calls</div>
          {steps.map((step, idx) => (
            <div className="kb-agent-trace-call" key={`${String(step.tool || 'tool')}-${idx}`}>
              <div className="kb-agent-trace-call-head">
                <span className={`kb-agent-trace-status ${statusClass(step.status)}`}>{String(step.status || '')}</span>
                <strong>{String(step.tool || '')}</strong>
                {traceNum(step.elapsed_ms) > 0 ? <span>{traceNum(step.elapsed_ms)}ms</span> : null}
              </div>
              {step.observation ? <div className="kb-agent-trace-observation">{shortText(step.observation, 260)}</div> : null}
              {step.error ? <div className="kb-agent-trace-error">{shortText(step.error, 260)}</div> : null}
            </div>
          ))}
        </div>
      ) : null}
    </details>
  )
}
