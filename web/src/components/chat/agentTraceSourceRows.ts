import { records, refText } from './agentTracePanelUtils'
import type { AgentTraceReferenceRecord } from './agentTraceReferenceTypes'
import type { AgentTraceRecord } from './agentTraceTypes'
import { asTraceRecord, traceNum } from './messageTraceUtils'

export type AgentTraceSourceRowsInput = {
  verification: AgentTraceRecord
  steps: AgentTraceRecord[]
  unsupportedClaimLimit?: number
  referenceLimit?: number
}

export type AgentTraceSourceRows = {
  unsupportedClaimRows: AgentTraceRecord[]
  references: AgentTraceReferenceRecord[]
}

export function buildUnsupportedClaimRows(
  claims: AgentTraceRecord[],
  limit = 3,
): AgentTraceRecord[] {
  return claims
    .filter((claim) => claim.supported === false || String(claim.unsupported_reason || '').trim())
    .slice(0, limit)
}

export function traceStepReferences(
  steps: AgentTraceRecord[],
  limit = 4,
): AgentTraceReferenceRecord[] {
  const out: AgentTraceReferenceRecord[] = []
  const seen = new Set<string>()
  for (const step of steps) {
    const output = asTraceRecord(step.output)
    for (const ref of records(output.references)) {
      const key = [
        refText(ref, 'source_path', 'sourcePath'),
        traceNum(ref.ref_num || ref.num),
        refText(ref, 'title', 'raw', 'source_name', 'sourceName'),
      ].join('|')
      if (seen.has(key)) continue
      seen.add(key)
      out.push(ref)
      if (out.length >= limit) return out
    }
  }
  return out
}

export function buildAgentTraceSourceRows({
  verification,
  steps,
  unsupportedClaimLimit,
  referenceLimit,
}: AgentTraceSourceRowsInput): AgentTraceSourceRows {
  return {
    unsupportedClaimRows: buildUnsupportedClaimRows(records(verification.claims), unsupportedClaimLimit),
    references: traceStepReferences(steps, referenceLimit),
  }
}
