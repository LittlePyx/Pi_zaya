import type { RefsResponseMeta } from '../api/chat'
import { internalDebugEnabled } from '../utils/internalDebug'

type SwitchPerfStatus = 'same_conv' | 'success' | 'stale' | 'error'

interface SwitchPerfEvent {
  ts: number
  convId: string
  token: number
  status: SwitchPerfStatus
  durationMs: number
  usedCache: boolean
  messageCount: number
  note: string
}

interface SwitchPerfSummary {
  total: number
  success: number
  stale: number
  error: number
  sameConv: number
  avgSuccessMs: number
}

interface SwitchPerfApi {
  getLogs: () => SwitchPerfEvent[]
  clear: () => void
  summary: () => SwitchPerfSummary
}

interface ConversationOpenPhaseEvent {
  ts: number
  convId: string
  token: number
  phase: string
  durationMs: number
  detail?: string
}

interface ConversationOpenPhaseApi {
  getLogs: () => ConversationOpenPhaseEvent[]
  clear: () => void
}

interface RefsPayloadSummary {
  packCount: number
  hitCount: number
  pendingPackCount: number
  fastPackCount: number
  readyPackCount: number
  emptyPackCount: number
  displayStates: Record<string, number>
}

interface RefsPerfEvent {
  ts: number
  convId: string
  phase: string
  token: number
  durationMs: number
  attempt?: number
  reason?: string
  active?: boolean
  needsEnrichment?: boolean
  keepPolling?: boolean
  nextDelayMs?: number
  error?: string
  backendMode?: string
  backendCounts?: string
  serverTiming?: string
  summary?: RefsPayloadSummary
}

interface RefsPerfSummary {
  total: number
  fetchSuccess: number
  fetchError: number
  stale: number
  avgFetchMs: number
  lastMode: string
  lastCounts: string
}

interface RefsPerfApi {
  getLogs: () => RefsPerfEvent[]
  clear: () => void
  summary: () => RefsPerfSummary
}

interface DebugWindow extends Window {
  __kbSwitchPerf?: SwitchPerfApi
  __kbConversationOpenPerf?: ConversationOpenPhaseApi
  __kbRefsPerf?: RefsPerfApi
}

const switchPerfLog: SwitchPerfEvent[] = []
const conversationOpenPhaseLog: ConversationOpenPhaseEvent[] = []
const refsPerfLog: RefsPerfEvent[] = []
const SWITCH_PERF_LIMIT = 240
const CONVERSATION_OPEN_PHASE_LIMIT = 480
const REFS_PERF_LIMIT = 720

function getSwitchPerfSummary(): SwitchPerfSummary {
  const total = switchPerfLog.length
  let success = 0
  let stale = 0
  let error = 0
  let sameConv = 0
  let successDuration = 0
  for (const event of switchPerfLog) {
    if (event.status === 'success') {
      success += 1
      successDuration += event.durationMs
    } else if (event.status === 'stale') {
      stale += 1
    } else if (event.status === 'error') {
      error += 1
    } else if (event.status === 'same_conv') {
      sameConv += 1
    }
  }
  return {
    total,
    success,
    stale,
    error,
    sameConv,
    avgSuccessMs: success > 0 ? Number((successDuration / success).toFixed(2)) : 0,
  }
}

export function summarizeRefsPayload(refs: Record<string, unknown>): RefsPayloadSummary {
  const summary: RefsPayloadSummary = {
    packCount: 0,
    hitCount: 0,
    pendingPackCount: 0,
    fastPackCount: 0,
    readyPackCount: 0,
    emptyPackCount: 0,
    displayStates: {},
  }
  for (const value of Object.values(refs || {})) {
    if (!value || typeof value !== 'object') continue
    const rec = value as {
      hits?: unknown[]
      enrichment_pending?: boolean
      payload_mode?: string
      display_state?: string
    }
    summary.packCount += 1
    const hits = Array.isArray(rec.hits) ? rec.hits : []
    summary.hitCount += hits.length
    const mode = String(rec.payload_mode || '').trim().toLowerCase()
    const displayState = String(rec.display_state || '').trim().toLowerCase() || 'unknown'
    summary.displayStates[displayState] = (summary.displayStates[displayState] || 0) + 1
    if (mode === 'pending' || Boolean(rec.enrichment_pending)) {
      summary.pendingPackCount += 1
    } else if (mode === 'fast') {
      summary.fastPackCount += 1
    } else if (hits.length > 0 || displayState === 'ready') {
      summary.readyPackCount += 1
    } else {
      summary.emptyPackCount += 1
    }
  }
  return summary
}

function getRefsPerfSummary(): RefsPerfSummary {
  let fetchSuccess = 0
  let fetchError = 0
  let stale = 0
  let fetchDuration = 0
  let lastMode = ''
  let lastCounts = ''
  const inferredMode = (summary?: RefsPayloadSummary): string => {
    if (!summary) return ''
    if (summary.pendingPackCount > 0) return 'pending'
    if (summary.fastPackCount > 0) return 'fast'
    if (summary.readyPackCount > 0) return 'ready'
    if (summary.packCount > 0) return 'empty'
    return ''
  }
  const inferredCounts = (summary?: RefsPayloadSummary): string => {
    if (!summary) return ''
    return [
      `packs=${summary.packCount}`,
      `hits=${summary.hitCount}`,
      `pending=${summary.pendingPackCount}`,
      `fast=${summary.fastPackCount}`,
      `ready=${summary.readyPackCount}`,
    ].join(',')
  }
  for (const event of refsPerfLog) {
    if (event.phase === 'fetch_success' || event.phase === 'poll_success') {
      fetchSuccess += 1
      fetchDuration += event.durationMs
      lastMode = event.backendMode || inferredMode(event.summary) || lastMode
      lastCounts = event.backendCounts || inferredCounts(event.summary) || lastCounts
    } else if (event.phase === 'fetch_error' || event.phase === 'poll_error') {
      fetchError += 1
    } else if (event.phase === 'fetch_stale') {
      stale += 1
    }
  }
  return {
    total: refsPerfLog.length,
    fetchSuccess,
    fetchError,
    stale,
    avgFetchMs: fetchSuccess > 0 ? Number((fetchDuration / fetchSuccess).toFixed(2)) : 0,
    lastMode,
    lastCounts,
  }
}

export function ensureChatStorePerfApi() {
  if (typeof window === 'undefined') return
  if (!internalDebugEnabled()) return
  const w = window as DebugWindow
  if (!w.__kbSwitchPerf) {
    w.__kbSwitchPerf = {
      getLogs: () => switchPerfLog.slice(),
      clear: () => {
        switchPerfLog.length = 0
      },
      summary: () => getSwitchPerfSummary(),
    }
  }
  if (!w.__kbConversationOpenPerf) {
    w.__kbConversationOpenPerf = {
      getLogs: () => conversationOpenPhaseLog.slice(),
      clear: () => {
        conversationOpenPhaseLog.length = 0
      },
    }
  }
  if (!w.__kbRefsPerf) {
    w.__kbRefsPerf = {
      getLogs: () => refsPerfLog.slice(),
      clear: () => {
        refsPerfLog.length = 0
      },
      summary: () => getRefsPerfSummary(),
    }
  }
}

export function pushSwitchPerf(event: SwitchPerfEvent) {
  switchPerfLog.push(event)
  if (switchPerfLog.length > SWITCH_PERF_LIMIT) {
    switchPerfLog.splice(0, switchPerfLog.length - SWITCH_PERF_LIMIT)
  }
  ensureChatStorePerfApi()
}

export function pushConversationOpenPhase(event: ConversationOpenPhaseEvent) {
  conversationOpenPhaseLog.push(event)
  if (conversationOpenPhaseLog.length > CONVERSATION_OPEN_PHASE_LIMIT) {
    conversationOpenPhaseLog.splice(0, conversationOpenPhaseLog.length - CONVERSATION_OPEN_PHASE_LIMIT)
  }
  ensureChatStorePerfApi()
}

export function pushRefsPerf(event: RefsPerfEvent) {
  refsPerfLog.push(event)
  if (refsPerfLog.length > REFS_PERF_LIMIT) {
    refsPerfLog.splice(0, refsPerfLog.length - REFS_PERF_LIMIT)
  }
  ensureChatStorePerfApi()
}

export function refsBackendPerf(meta?: RefsResponseMeta | null) {
  return {
    backendMode: String(meta?.mode || '').trim(),
    backendCounts: String(meta?.counts || '').trim(),
    serverTiming: String(meta?.serverTiming || '').trim(),
  }
}

ensureChatStorePerfApi()
