import type { ChatImageAttachment, Message } from '../../api/chat'
import {
  normalizeSelectedResearchContextPack,
  type SelectedResearchContextItem,
  type SelectedResearchContextPack,
} from './researchContextPack'
import { internalDebugBrowserEnabled } from '../../utils/internalDebug'
import { sourceSummaryFromAnswerContract } from './answerContractViewModel'

export function isImageOnlyPlaceholder(content: string) {
  return /^\[Image attachment x\d+\]$/i.test(String(content || '').trim())
}

export function imageAttachmentsOf(message: Message): ChatImageAttachment[] {
  return Array.isArray(message.attachments)
    ? message.attachments.filter((item): item is ChatImageAttachment => Boolean(item && item.path))
    : []
}

export function asTraceRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
}

export function getMessageResearchTrace(message: Message): Record<string, unknown> | null {
  const meta = asTraceRecord(message.meta)
  const trace = asTraceRecord(meta.research_trace)
  if (Object.keys(trace).length > 0) return trace
  const traceId = String(meta.trace_id || '').trim()
  return traceId ? { trace_id: traceId } : null
}

export function getMessageAgentTrace(message: Message): Record<string, unknown> | null {
  const meta = asTraceRecord(message.meta)
  const trace = asTraceRecord(meta.agent_trace)
  return Object.keys(trace).length > 0 ? trace : null
}

export function getMessageAnswerContract(message: Message): Record<string, unknown> | null {
  const meta = asTraceRecord(message.meta)
  const contract = asTraceRecord(meta.answer_contract || meta.answerContract)
  return Object.keys(contract).length > 0 ? contract : null
}

export function getMessageAgentSourceSummary(message: Message): Record<string, unknown> | null {
  const meta = asTraceRecord(message.meta)
  const contractSummary = sourceSummaryFromAnswerContract(meta.answer_contract || meta.answerContract)
  if (contractSummary) return contractSummary
  const summary = asTraceRecord(meta.agent_source_summary || meta.agentSourceSummary)
  return Object.keys(summary).length > 0 ? summary : null
}

export function messageHasAgentTraceHint(message: Message): boolean {
  if (getMessageAgentTrace(message)) return true
  const meta = asTraceRecord(message.meta)
  const mode = String(meta.agent_mode || meta.agentMode || '').trim().toLowerCase()
  return mode === 'research_agent' || meta.agent_trace_available === true || meta.agentTraceAvailable === true
}

export function getAssistantSelectedResearchContext(message: Message): SelectedResearchContextPack | null {
  const meta = asTraceRecord(message.meta)
  const contracts = asTraceRecord(meta.paper_guide_contracts)
  return normalizeSelectedResearchContextPack(contracts.selected_research_context)
}

export function getUserPromptResearchContext(message: Message): SelectedResearchContextPack | null {
  const meta = asTraceRecord(message.meta)
  return normalizeSelectedResearchContextPack(meta.prompt_context)
}

export function traceNum(value: unknown): number {
  const n = Number(value)
  return Number.isFinite(n) ? n : 0
}

export function formatTraceMs(value: unknown): string {
  const ms = traceNum(value)
  if (ms <= 0) return '0ms'
  if (ms >= 1000) return `${(ms / 1000).toFixed(ms >= 10000 ? 1 : 2)}s`
  return `${Math.round(ms)}ms`
}

export function traceSourceLabels(items: unknown): string[] {
  if (!Array.isArray(items)) return []
  const out: string[] = []
  for (const item of items) {
    const rec = asTraceRecord(item)
    const label = String(rec.source_name || rec.source_path || '').trim()
    if (!label) continue
    out.push(label)
    if (out.length >= 3) break
  }
  return out
}

interface ResearchTraceDebugWindow extends Window {
  __KB_SHOW_RESEARCH_TRACE__?: boolean
}

export function shouldShowResearchTracePanel(): boolean {
  if (!internalDebugBrowserEnabled()) return false
  if (typeof window === 'undefined') return false
  const w = window as ResearchTraceDebugWindow
  if (w.__KB_SHOW_RESEARCH_TRACE__) return true
  const enabledValues = new Set(['1', 'true', 'yes', 'on', 'debug'])
  try {
    const params = new URLSearchParams(window.location.search)
    const queryValue = String(params.get('debug_trace') || params.get('kb_trace') || '').trim().toLowerCase()
    if (enabledValues.has(queryValue)) return true
  } catch { /* ignore */ }
  try {
    const stored = String(window.localStorage.getItem('kb_debug_trace') || '').trim().toLowerCase()
    return enabledValues.has(stored)
  } catch {
    return false
  }
}

export function contextItemTitle(item: SelectedResearchContextItem, fallback: string): string {
  return String(item.title || item.excerpt || item.summary || fallback || '').trim()
}

export function contextItemMeta(item: SelectedResearchContextItem): string {
  return [
    item.authors,
    item.year,
    item.sourceName,
    item.refNum ? `[${item.refNum}]` : '',
  ].filter(Boolean).join(' 路 ')
}
