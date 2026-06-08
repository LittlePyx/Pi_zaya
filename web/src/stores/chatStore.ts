import { create } from 'zustand'
import {
  chatApi,
  type ChatImageAttachment,
  type ChatUploadItem,
  type Conversation,
  type Message,
  type MessagePage,
  type Project,
  type RefsResponseMeta,
} from '../api/chat'
import { api, authFetch } from '../api/client'
import { S as zh } from '../i18n/zh'
import { S as en } from '../i18n/en'
import { useSettingsStore } from './settingsStore'

let refsPollToken = 0
let refsPollTimer: number | null = null
let messagePostprocessPollToken = 0
let messagePostprocessPollTimer: number | null = null
let uploadPollToken = 0
let uploadPollTimer: number | null = null
let conversationSwitchToken = 0
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
const SIDEBAR_CONVERSATION_LIMIT = 80
const MESSAGE_PAGE_SIZE = 24

function conversationDraftTimestamp() {
  const d = new Date()
  const mm = `${d.getMonth() + 1}`.padStart(2, '0')
  const dd = `${d.getDate()}`.padStart(2, '0')
  const hh = `${d.getHours()}`.padStart(2, '0')
  const min = `${d.getMinutes()}`.padStart(2, '0')
  return `${mm}/${dd} ${hh}:${min}`
}

function buildDefaultConversationTitle(locale: string) {
  const prefix = locale.toLowerCase().startsWith('zh') ? '新会话' : 'New conversation'
  return `${prefix} · ${conversationDraftTimestamp()}`
}

function patchConversationTitle(list: Conversation[], convId: string, title: string): Conversation[] {
  if (!convId || !title) return list
  let changed = false
  const next = list.map((conversation) => {
    if (String(conversation.id || '') !== convId || conversation.title === title) return conversation
    changed = true
    return { ...conversation, title }
  })
  return changed ? next : list
}

function patchProjectConversationTitle(
  grouped: Record<string, Conversation[]>,
  convId: string,
  title: string,
): Record<string, Conversation[]> {
  if (!convId || !title) return grouped
  let changed = false
  const next: Record<string, Conversation[]> = {}
  for (const [projectId, conversations] of Object.entries(grouped || {})) {
    const patched = patchConversationTitle(conversations || [], convId, title)
    next[projectId] = patched
    if (patched !== conversations) changed = true
  }
  return changed ? next : grouped
}

function nowMs() {
  try {
    return performance.now()
  } catch {
    return Date.now()
  }
}

function buildFullMessagePage(messages: Message[]): MessagePage {
  return {
    messages,
    has_more_before: false,
    oldest_loaded_id: messages.length > 0 ? Number(messages[0]?.id || 0) || null : null,
    newest_loaded_id: messages.length > 0 ? Number(messages[messages.length - 1]?.id || 0) || null : null,
  }
}

function mergeLatestMessagePage(
  currentMessages: Message[],
  currentHasMoreBefore: boolean,
  page: MessagePage,
): { messages: Message[]; hasMoreBefore: boolean; oldestLoadedMessageId: number | null } {
  const latestMessages = Array.isArray(page?.messages) ? page.messages : []
  const latestIds = new Set(latestMessages.map((item) => Number(item.id || 0)).filter((id) => Number.isFinite(id) && id > 0))
  const latestOldestId = Number(page?.oldest_loaded_id || 0)
  const hasLatestOldestId = Number.isFinite(latestOldestId) && latestOldestId > 0
  const retainedOlder = currentMessages.filter((item) => {
    const id = Number(item.id || 0)
    if (!Number.isFinite(id) || id <= 0) return false
    if (latestIds.has(id)) return false
    return hasLatestOldestId ? id < latestOldestId : false
  })
  const merged = [...retainedOlder, ...latestMessages]
  return {
    messages: merged,
    hasMoreBefore: retainedOlder.length > 0 ? currentHasMoreBefore : Boolean(page?.has_more_before),
    oldestLoadedMessageId: merged.length > 0 ? Number(merged[0]?.id || 0) || null : null,
  }
}

async function getMessagesPageWithFallback(
  convId: string,
  opts?: { limit?: number; beforeId?: number | null; renderPacketOnly?: boolean },
): Promise<{ page: MessagePage; usedFallback: boolean }> {
  try {
    const page = await chatApi.getMessagesPage(convId, opts)
    return { page, usedFallback: false }
  } catch (error) {
    const beforeId = Number(opts?.beforeId || 0)
    if (beforeId > 0) {
      throw error
    }
    const messages = await chatApi.getMessages(convId, { renderPacketOnly: opts?.renderPacketOnly })
    return {
      page: buildFullMessagePage(Array.isArray(messages) ? messages : []),
      usedFallback: true,
    }
  }
}

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

function summarizeRefsPayload(refs: Record<string, unknown>): RefsPayloadSummary {
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
  for (const event of refsPerfLog) {
    if (event.phase === 'fetch_success' || event.phase === 'poll_success') {
      fetchSuccess += 1
      fetchDuration += event.durationMs
      lastMode = event.backendMode || lastMode
      lastCounts = event.backendCounts || lastCounts
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

function ensureSwitchPerfApi() {
  if (typeof window === 'undefined') return
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

function pushSwitchPerf(event: SwitchPerfEvent) {
  switchPerfLog.push(event)
  if (switchPerfLog.length > SWITCH_PERF_LIMIT) {
    switchPerfLog.splice(0, switchPerfLog.length - SWITCH_PERF_LIMIT)
  }
  ensureSwitchPerfApi()
}

function pushConversationOpenPhase(event: ConversationOpenPhaseEvent) {
  conversationOpenPhaseLog.push(event)
  if (conversationOpenPhaseLog.length > CONVERSATION_OPEN_PHASE_LIMIT) {
    conversationOpenPhaseLog.splice(0, conversationOpenPhaseLog.length - CONVERSATION_OPEN_PHASE_LIMIT)
  }
  ensureSwitchPerfApi()
}

function pushRefsPerf(event: RefsPerfEvent) {
  refsPerfLog.push(event)
  if (refsPerfLog.length > REFS_PERF_LIMIT) {
    refsPerfLog.splice(0, refsPerfLog.length - REFS_PERF_LIMIT)
  }
  ensureSwitchPerfApi()
}

function refsBackendPerf(meta?: RefsResponseMeta | null) {
  return {
    backendMode: String(meta?.mode || '').trim(),
    backendCounts: String(meta?.counts || '').trim(),
    serverTiming: String(meta?.serverTiming || '').trim(),
  }
}

if (typeof window !== 'undefined') {
  ensureSwitchPerfApi()
}

function stopRefsPolling() {
  refsPollToken += 1
  if (refsPollTimer !== null) {
    window.clearTimeout(refsPollTimer)
    refsPollTimer = null
  }
}

function stopMessagePostprocessPolling() {
  messagePostprocessPollToken += 1
  if (messagePostprocessPollTimer !== null) {
    window.clearTimeout(messagePostprocessPollTimer)
    messagePostprocessPollTimer = null
  }
}

function stopUploadPolling() {
  uploadPollToken += 1
  if (uploadPollTimer !== null) {
    window.clearTimeout(uploadPollTimer)
    uploadPollTimer = null
  }
}

function needsRefsEnrichment(refs: Record<string, unknown>) {
  for (const value of Object.values(refs || {})) {
    const rec = value as {
      hits?: Array<{ ui_meta?: Record<string, unknown>; meta?: Record<string, unknown> }>
      enrichment_pending?: boolean
      payload_mode?: string
    }
    if (rec?.enrichment_pending) {
      return true
    }
    const payloadMode = String(rec?.payload_mode || '').trim().toLowerCase()
    if (payloadMode === 'fast' || payloadMode === 'pending') {
      return true
    }
    const hits = Array.isArray(rec?.hits) ? rec.hits : []
    for (const hit of hits) {
      const meta = hit?.meta || {}
      if (String(meta.ref_pack_state || '').trim().toLowerCase() === 'pending') {
        return true
      }
    }
  }
  return false
}

function uploadItemKey(item: ChatUploadItem) {
  if (item.kind === 'pdf' && item.ingest_job_id) {
    return `pdf-job:${item.ingest_job_id}`
  }
  return [item.kind, item.sha1 || '', item.path || '', item.name].join(':')
}

function attachmentKey(item: ChatImageAttachment) {
  return item.sha1 || item.path
}

function mergeUploadItems(current: ChatUploadItem[], incoming: ChatUploadItem[]) {
  const next = [...current]
  const positions = new Map(next.map((item, index) => [uploadItemKey(item), index]))
  for (const item of incoming) {
    const key = uploadItemKey(item)
    const index = positions.get(key)
    if (index === undefined) {
      positions.set(key, next.length)
      next.push(item)
    } else {
      next[index] = item
    }
  }
  return next
}

function isPdfUploadJobRunning(item: ChatUploadItem) {
  if (item.kind !== 'pdf') return false
  const ingestRunning = ['processing', 'renaming', 'converting', 'ingesting'].includes(String(item.ingest_status || ''))
  const qualityRunning = ['pending', 'running'].includes(String(item.quality_status || ''))
  return ingestRunning || qualityRunning
}

function needsUploadStatusPolling(uploadItems: ChatUploadItem[]) {
  return uploadItems.some((item) =>
    isPdfUploadJobRunning(item) && item.ingest_job_id,
  )
}

async function startUploadPolling(set: (patch: Partial<ChatState> | ((state: ChatState) => Partial<ChatState>)) => void, getState: () => ChatState) {
  stopUploadPolling()
  const token = ++uploadPollToken
  let tries = 0
  const maxTries = 240
  const nextDelay = () => {
    if (tries <= 10) return 500
    if (tries <= 40) return 1000
    return 1800
  }

  const tick = async () => {
    if (token !== uploadPollToken) return
    tries += 1
    const state = getState()
    const jobIds = state.uploadItems
      .filter((item) => isPdfUploadJobRunning(item) && item.ingest_job_id)
      .map((item) => String(item.ingest_job_id || '').trim())
      .filter(Boolean)
    if (jobIds.length === 0) {
      uploadPollTimer = null
      return
    }
    try {
      const res = await chatApi.getUploadStatuses(jobIds)
      if (token !== uploadPollToken) return
      const items = Array.isArray(res.items) ? res.items : []
      set((cur) => {
        const nextItems = mergeUploadItems(cur.uploadItems, items)
        return { uploadItems: nextItems }
      })
      const nextState = getState()
      if (!needsUploadStatusPolling(nextState.uploadItems) || tries >= maxTries) {
        uploadPollTimer = null
        return
      }
    } catch {
      if (tries >= maxTries) {
        uploadPollTimer = null
        return
      }
    }
    uploadPollTimer = window.setTimeout(tick, nextDelay())
  }

  void tick()
}

function mergeImageAttachments(current: ChatImageAttachment[], incoming: ChatImageAttachment[]) {
  const next = [...current]
  const seen = new Set(next.map(attachmentKey))
  for (const item of incoming) {
    const key = attachmentKey(item)
    if (!key || seen.has(key)) continue
    seen.add(key)
    next.push(item)
  }
  return next
}

async function loadRefsForConversation(
  convId: string,
  set: (patch: Partial<ChatState> | ((state: ChatState) => Partial<ChatState>)) => void,
  getActiveConvId: () => string | null,
  shouldKeepPolling?: () => boolean,
  reason = 'load',
) {
  const startedAt = nowMs()
  const token = refsPollToken
  pushRefsPerf({
    ts: Date.now(),
    convId,
    phase: 'fetch_start',
    token,
    durationMs: 0,
    reason,
    active: getActiveConvId() === convId,
  })
  try {
    const { data: refs, meta } = await chatApi.getRefsWithMeta(convId)
    const durationMs = Number((nowMs() - startedAt).toFixed(2))
    if (getActiveConvId() !== convId) {
      pushRefsPerf({
        ts: Date.now(),
        convId,
        phase: 'fetch_stale',
        token,
        durationMs,
        reason,
        active: false,
        ...refsBackendPerf(meta),
        summary: summarizeRefsPayload(refs),
      })
      return
    }
    set((state) => ({
      refs,
      conversationCacheById: upsertConversationViewCache(state.conversationCacheById, convId, {
        refs,
        cachedAt: Date.now(),
      }),
    }))
    const needsEnrichment = needsRefsEnrichment(refs)
    const keepPolling = Boolean(shouldKeepPolling?.())
    pushRefsPerf({
      ts: Date.now(),
      convId,
      phase: 'fetch_success',
      token,
      durationMs,
      reason,
      active: true,
      needsEnrichment,
      keepPolling,
      ...refsBackendPerf(meta),
      summary: summarizeRefsPayload(refs),
    })
    if (needsEnrichment || keepPolling) {
      void startRefsPolling(convId, set, shouldKeepPolling, `${reason}:followup`)
    }
  } catch (err) {
    pushRefsPerf({
      ts: Date.now(),
      convId,
      phase: 'fetch_error',
      token,
      durationMs: Number((nowMs() - startedAt).toFixed(2)),
      reason,
      active: getActiveConvId() === convId,
      error: err instanceof Error ? err.message : String(err || 'unknown'),
    })
    if (getActiveConvId() === convId) {
      set((state) => ({
        refs: state.activeConvId === convId
          ? (
            state.refs && typeof state.refs === 'object'
              ? state.refs
              : (
                state.conversationCacheById[convId]?.refs
                && typeof state.conversationCacheById[convId]?.refs === 'object'
              )
                ? state.conversationCacheById[convId]?.refs
                : {}
          )
          : {},
        conversationCacheById: upsertConversationViewCache(state.conversationCacheById, convId, {
          refs: state.activeConvId === convId
            ? (
              state.refs && typeof state.refs === 'object'
                ? state.refs
                : (
                  state.conversationCacheById[convId]?.refs
                  && typeof state.conversationCacheById[convId]?.refs === 'object'
                )
                  ? state.conversationCacheById[convId]?.refs
                  : {}
            )
            : {},
          cachedAt: Date.now(),
        }),
      }))
      void startRefsPolling(convId, set, shouldKeepPolling, `${reason}:retry_after_error`)
    }
  }
}

function scheduleLoadRefsForConversation(
  convId: string,
  set: (patch: Partial<ChatState> | ((state: ChatState) => Partial<ChatState>)) => void,
  getActiveConvId: () => string | null,
  delayMs = 120,
  shouldKeepPolling?: () => boolean,
  reason = 'scheduled',
) {
  pushRefsPerf({
    ts: Date.now(),
    convId,
    phase: 'schedule',
    token: refsPollToken,
    durationMs: 0,
    reason,
    active: getActiveConvId() === convId,
    nextDelayMs: Math.max(0, delayMs),
  })
  if (typeof window === 'undefined') {
    void loadRefsForConversation(convId, set, getActiveConvId, shouldKeepPolling, reason)
    return
  }
  window.setTimeout(() => {
    if (getActiveConvId() !== convId) {
      pushRefsPerf({
        ts: Date.now(),
        convId,
        phase: 'schedule_stale',
        token: refsPollToken,
        durationMs: 0,
        reason,
        active: false,
      })
      return
    }
    void loadRefsForConversation(convId, set, getActiveConvId, shouldKeepPolling, reason)
  }, Math.max(0, delayMs))
}

async function startRefsPolling(
  convId: string,
  set: (patch: Partial<ChatState> | ((state: ChatState) => Partial<ChatState>)) => void,
  shouldKeepPolling?: () => boolean,
  reason = 'poll',
) {
  stopRefsPolling()
  const token = ++refsPollToken
  let tries = 0
  const maxTries = 180
  const nextDelay = () => {
    if (tries <= 6) return 350
    if (tries <= 18) return 700
    if (tries <= 60) return 1200
    return 1800
  }

  pushRefsPerf({
    ts: Date.now(),
    convId,
    phase: 'poll_start',
    token,
    durationMs: 0,
    reason,
  })

  const tick = async () => {
    if (token !== refsPollToken) return
    tries += 1
    const startedAt = nowMs()
    try {
      const { data: refs, meta } = await chatApi.getRefsWithMeta(convId)
      if (token !== refsPollToken) return
      set((state) => ({
        refs,
        conversationCacheById: upsertConversationViewCache(state.conversationCacheById, convId, {
          refs,
          cachedAt: Date.now(),
        }),
      }))
      const keepPolling = Boolean(shouldKeepPolling?.())
      const needsEnrichment = needsRefsEnrichment(refs)
      pushRefsPerf({
        ts: Date.now(),
        convId,
        phase: 'poll_success',
        token,
        durationMs: Number((nowMs() - startedAt).toFixed(2)),
        attempt: tries,
        reason,
        needsEnrichment,
        keepPolling,
        ...refsBackendPerf(meta),
        summary: summarizeRefsPayload(refs),
      })
      if ((!needsEnrichment && !keepPolling) || tries >= maxTries) {
        refsPollTimer = null
        pushRefsPerf({
          ts: Date.now(),
          convId,
          phase: 'poll_stop',
          token,
          durationMs: 0,
          attempt: tries,
          reason: tries >= maxTries ? 'max_tries' : 'settled',
          needsEnrichment,
          keepPolling,
        })
        return
      }
    } catch (err) {
      pushRefsPerf({
        ts: Date.now(),
        convId,
        phase: 'poll_error',
        token,
        durationMs: Number((nowMs() - startedAt).toFixed(2)),
        attempt: tries,
        reason,
        error: err instanceof Error ? err.message : String(err || 'unknown'),
      })
      if (tries >= maxTries) {
        refsPollTimer = null
        pushRefsPerf({
          ts: Date.now(),
          convId,
          phase: 'poll_stop',
          token,
          durationMs: 0,
          attempt: tries,
          reason: 'max_tries_after_error',
        })
        return
      }
    }
    const delay = nextDelay()
    pushRefsPerf({
      ts: Date.now(),
      convId,
      phase: 'poll_schedule_next',
      token,
      durationMs: 0,
      attempt: tries,
      reason,
      nextDelayMs: delay,
    })
    refsPollTimer = window.setTimeout(tick, delay)
  }

  void tick()
}

function getMessageProvenanceForPostprocess(message: Message | null | undefined): Record<string, unknown> | null {
  if (!message || typeof message !== 'object') return null
  if (message.provenance && typeof message.provenance === 'object') {
    return message.provenance as Record<string, unknown>
  }
  const meta = message.meta && typeof message.meta === 'object' ? message.meta : null
  const nested = meta?.provenance
  return nested && typeof nested === 'object' ? nested as Record<string, unknown> : null
}

function getMessageRenderPacketForPostprocess(message: Message | null | undefined): Record<string, unknown> | null {
  const meta = message?.meta && typeof message.meta === 'object' ? message.meta : null
  const contracts = meta?.paper_guide_contracts
  if (!contracts || typeof contracts !== 'object') return null
  const packet = (contracts as Record<string, unknown>).render_packet
  return packet && typeof packet === 'object' ? packet as Record<string, unknown> : null
}

function messageHasReadyLocatePostprocess(message: Message | null | undefined): boolean {
  if (!message || String(message.role || '').trim().toLowerCase() !== 'assistant') return false
  const provenance = getMessageProvenanceForPostprocess(message)
  if (provenance) {
    const status = String(provenance.status || '').trim().toLowerCase()
    const segments = Array.isArray(provenance.segments) ? provenance.segments : []
    const strictIdentityReady = Boolean(provenance.strict_identity_ready)
    const mustLocateCount = Math.max(
      0,
      Number(provenance.must_locate_count || 0) || 0,
      Number(provenance.must_locate_candidate_count || 0) || 0,
    )
    if (status === 'ready' && (strictIdentityReady || segments.length > 0 || mustLocateCount > 0)) {
      return true
    }
    if (!status && (strictIdentityReady || segments.length > 0 || mustLocateCount > 0)) {
      return true
    }
  }
  const packet = getMessageRenderPacketForPostprocess(message)
  if (packet) {
    const segmentIds = Array.isArray(packet.segment_ids) ? packet.segment_ids : []
    const visibleSegmentIds = Array.isArray(packet.visible_segment_ids) ? packet.visible_segment_ids : []
    if (
      packet.locate_target
      || packet.reader_open
      || segmentIds.length > 0
      || visibleSegmentIds.length > 0
    ) {
      return true
    }
  }
  return false
}

function messageNeedsPostprocessRefresh(
  message: Message | null | undefined,
  opts?: { paperGuideMode?: boolean },
): boolean {
  if (!message) return true
  if (String(message.role || '').trim().toLowerCase() !== 'assistant') return false
  if (messageHasReadyLocatePostprocess(message)) return false
  const provenance = getMessageProvenanceForPostprocess(message)
  const status = String(provenance?.status || '').trim().toLowerCase()
  if (status && status !== 'ready') return true
  return Boolean(opts?.paperGuideMode)
}

async function startMessagePostprocessPolling(
  convId: string,
  assistantMsgId: number,
  set: (patch: Partial<ChatState> | ((state: ChatState) => Partial<ChatState>)) => void,
  getState: () => ChatState,
  opts?: { paperGuideMode?: boolean; reason?: string },
) {
  stopMessagePostprocessPolling()
  const msgId = Number(assistantMsgId || 0)
  if (!convId || !Number.isFinite(msgId) || msgId <= 0 || typeof window === 'undefined') return
  const token = ++messagePostprocessPollToken
  let tries = 0
  const maxTries = opts?.paperGuideMode ? 60 : 18
  const minTries = opts?.paperGuideMode ? 4 : 1
  const nextDelay = () => {
    if (tries <= 4) return 350
    if (tries <= 14) return 700
    if (tries <= 36) return 1200
    return 1800
  }

  const tick = async () => {
    if (token !== messagePostprocessPollToken) return
    if (getState().activeConvId !== convId) {
      messagePostprocessPollTimer = null
      return
    }
    tries += 1
    try {
      const { page } = await getMessagesPageWithFallback(convId, {
        limit: MESSAGE_PAGE_SIZE,
        renderPacketOnly: false,
      })
      if (token !== messagePostprocessPollToken || getState().activeConvId !== convId) return
      set((state) => {
        if (state.activeConvId !== convId) return {}
        const merged = mergeLatestMessagePage(
          state.messages,
          state.messagesHasMoreBefore,
          page,
        )
        return {
          messages: merged.messages,
          messagesHasMoreBefore: merged.hasMoreBefore,
          oldestLoadedMessageId: merged.oldestLoadedMessageId,
          conversationCacheById: upsertConversationViewCache(state.conversationCacheById, convId, {
            messages: merged.messages,
            refs: state.refs,
            messagesHasMoreBefore: merged.hasMoreBefore,
            oldestLoadedMessageId: merged.oldestLoadedMessageId,
            cachedAt: Date.now(),
          }),
        }
      })
      const target = getState().messages.find((item) => Number(item.id || 0) === msgId) || null
      if ((!messageNeedsPostprocessRefresh(target, opts) && tries >= minTries) || tries >= maxTries) {
        messagePostprocessPollTimer = null
        return
      }
    } catch {
      if (tries >= maxTries) {
        messagePostprocessPollTimer = null
        return
      }
    }
    if (token !== messagePostprocessPollToken) return
    messagePostprocessPollTimer = window.setTimeout(tick, nextDelay())
  }

  void tick()
}

interface GenerationState {
  sessionId: string
  taskId: string
  traceId?: string
  stage: string
  partial: string
  done: boolean
  researchTrace?: Record<string, unknown>
}

interface GuideBinding {
  sourcePath: string
  sourceName: string
}

interface ConversationViewCache {
  messages: Message[]
  refs: Record<string, unknown>
  messagesHasMoreBefore: boolean
  oldestLoadedMessageId: number | null
  cachedAt: number
}

function upsertConversationViewCache(
  current: Record<string, ConversationViewCache>,
  convId: string,
  patch: Partial<ConversationViewCache>,
) {
  const key = String(convId || '').trim()
  if (!key) return current
  const prev = current[key]
  return {
    ...current,
    [key]: {
      messages: Array.isArray(patch.messages) ? patch.messages : Array.isArray(prev?.messages) ? prev.messages : [],
      refs: patch.refs && typeof patch.refs === 'object'
        ? patch.refs
        : (prev?.refs && typeof prev.refs === 'object' ? prev.refs : {}),
      messagesHasMoreBefore: typeof patch.messagesHasMoreBefore === 'boolean'
        ? patch.messagesHasMoreBefore
        : Boolean(prev?.messagesHasMoreBefore),
      oldestLoadedMessageId: patch.oldestLoadedMessageId !== undefined
        ? patch.oldestLoadedMessageId ?? null
        : (prev?.oldestLoadedMessageId ?? null),
      cachedAt: Number.isFinite(Number(patch.cachedAt))
        ? Number(patch.cachedAt)
        : (prev?.cachedAt ?? Date.now()),
    },
  }
}

interface ChatState {
  projects: Project[]
  activeProjectId: string | null
  projectConversations: Record<string, Conversation[]>
  rootConversations: Conversation[]
  activeConvId: string | null
  activeConversation: Conversation | null
  guideBindings: Record<string, GuideBinding>
  conversationCacheById: Record<string, ConversationViewCache>
  messages: Message[]
  conversationLoading: boolean
  messagesLoadingMore: boolean
  messagesHasMoreBefore: boolean
  oldestLoadedMessageId: number | null
  refs: Record<string, unknown>
  uploadItems: ChatUploadItem[]
  pendingImages: ChatImageAttachment[]
  uploading: boolean
  generation: GenerationState | null
  sseController: AbortController | null

  loadSidebarData: () => Promise<void>
  selectProject: (id: string | null) => void
  createProject: (name: string) => Promise<string>
  renameProject: (id: string, name: string) => Promise<void>
  deleteProject: (id: string) => Promise<void>
  selectConversation: (id: string) => Promise<void>
  createConversation: () => Promise<string>
  createPaperGuideConversation: (opts: {
    sourcePath: string
    sourceName?: string
    title?: string
    projectId?: string | null
  }) => Promise<string>
  renameConversation: (id: string, title: string) => Promise<void>
  deleteConversation: (id: string) => Promise<void>
  moveConversation: (convId: string, projectId: string | null) => Promise<void>
  loadOlderMessages: () => Promise<void>
  uploadFiles: (files: File[], opts?: { quickIngest?: boolean; speedMode?: string; convId?: string | null }) => Promise<void>
  retryUploadItem: (key: string) => Promise<void>
  cancelUploadItem: (key: string) => Promise<void>
  removePendingImage: (key: string) => void
  dismissUploadItem: (key: string) => void
  sendMessage: (prompt: string, opts: {
    topK: number; temperature: number; maxTokens: number; deepRead: boolean; promptContext?: unknown
  }) => Promise<void>
  cancelGeneration: () => void
  clearGeneration: () => void
}

async function loadGroupedConversations(projects: Project[]) {
  const rootConversations = await chatApi.listConversations(SIDEBAR_CONVERSATION_LIMIT, null)
  const groupedEntries = await Promise.all(
    projects.map(async (project) => {
      const conversations = await chatApi.listConversations(SIDEBAR_CONVERSATION_LIMIT, project.id)
      return [project.id, conversations] as const
    }),
  )
  return {
    rootConversations,
    projectConversations: Object.fromEntries(groupedEntries) as Record<string, Conversation[]>,
  }
}

function findConversationInState(state: ChatState, convId: string): Conversation | null {
  for (const item of state.rootConversations) {
    if (item.id === convId) return item
  }
  for (const items of Object.values(state.projectConversations)) {
    for (const item of items) {
      if (item.id === convId) return item
    }
  }
  return null
}

function findConversationInLists(
  rootConversations: Conversation[],
  projectConversations: Record<string, Conversation[]>,
  convId: string,
): Conversation | null {
  for (const item of rootConversations) {
    if (item.id === convId) return item
  }
  for (const items of Object.values(projectConversations)) {
    for (const item of items) {
      if (item.id === convId) return item
    }
  }
  return null
}

export const useChatStore = create<ChatState>((set, get) => ({
  projects: [],
  activeProjectId: null,
  projectConversations: {},
  rootConversations: [],
  activeConvId: null,
  activeConversation: null,
  guideBindings: {},
  conversationCacheById: {},
  messages: [],
  conversationLoading: false,
  messagesLoadingMore: false,
  messagesHasMoreBefore: false,
  oldestLoadedMessageId: null,
  refs: {},
  uploadItems: [],
  pendingImages: [],
  uploading: false,
  generation: null,
  sseController: null,

  loadSidebarData: async () => {
    const projects = await chatApi.listProjects()
    const grouped = await loadGroupedConversations(projects)
    set((state) => ({
      guideBindings: (() => {
        const next = { ...(state.guideBindings || {}) }
        const absorb = (conv: Conversation) => {
          const cid = String(conv?.id || '').trim()
          const sourcePath = String(conv?.bound_source_path || '').trim()
          if (!cid || !sourcePath) return
          const sourceName = String(conv?.bound_source_name || '').trim()
          next[cid] = { sourcePath, sourceName }
        }
        for (const conv of grouped.rootConversations) absorb(conv)
        for (const list of Object.values(grouped.projectConversations || {})) {
          for (const conv of list) absorb(conv)
        }
        return next
      })(),
      activeConversation: state.activeConvId
        ? findConversationInLists(grouped.rootConversations, grouped.projectConversations, state.activeConvId)
        : null,
      projects,
      projectConversations: grouped.projectConversations,
      rootConversations: grouped.rootConversations,
      activeProjectId:
        state.activeProjectId && projects.some((project) => project.id === state.activeProjectId)
          ? state.activeProjectId
          : null,
    }))
  },

  selectProject: (id) => {
    set({ activeProjectId: id })
  },

  createProject: async (name) => {
    const { id } = await chatApi.createProject(name)
    await get().loadSidebarData()
    set({ activeProjectId: id })
    return id
  },

  renameProject: async (id, name) => {
    await chatApi.renameProject(id, name)
    await get().loadSidebarData()
  },

  deleteProject: async (id) => {
    await chatApi.deleteProject(id)
    const state = get()
    if (state.activeProjectId === id) {
      set({ activeProjectId: null })
    }
    await get().loadSidebarData()
    const activeConvId = get().activeConvId
    if (activeConvId) {
      const conv = await chatApi.getConversation(activeConvId).catch(() => null)
      if (conv) {
        set({ activeProjectId: conv.project_id ?? null })
      }
    }
  },

  selectConversation: async (id) => {
    const convId = String(id || '').trim()
    if (!convId) return
    const startedAt = nowMs()
    const current = get()
    if (current.activeConvId === convId) {
      if (Object.keys(current.refs || {}).length === 0) {
        scheduleLoadRefsForConversation(convId, set, () => get().activeConvId, 120, undefined, 'open_empty_cache')
      }
      pushSwitchPerf({
        ts: Date.now(),
        convId,
        token: conversationSwitchToken,
        status: 'same_conv',
        durationMs: Number((nowMs() - startedAt).toFixed(2)),
        usedCache: true,
        messageCount: current.messages.length,
        note: 'skip_same_conversation',
      })
      return
    }
    const myToken = ++conversationSwitchToken
    const cachedConv = findConversationInState(current, convId)
    const cachedView = current.conversationCacheById[convId]
    stopRefsPolling()
    stopMessagePostprocessPolling()
    stopUploadPolling()
    const cacheShowStartedAt = nowMs()
    set({
      activeConvId: convId,
      activeConversation: cachedConv || null,
      messages: Array.isArray(cachedView?.messages) ? cachedView.messages : [],
      conversationLoading: !cachedView,
      messagesLoadingMore: false,
      messagesHasMoreBefore: Boolean(cachedView?.messagesHasMoreBefore),
      oldestLoadedMessageId: cachedView?.oldestLoadedMessageId ?? null,
      generation: null,
      refs: cachedView?.refs && typeof cachedView.refs === 'object' ? cachedView.refs : {},
      uploadItems: [],
      pendingImages: [],
    })
    pushConversationOpenPhase({
      ts: Date.now(),
      convId,
      token: myToken,
      phase: 'cache_show',
      durationMs: Number((nowMs() - cacheShowStartedAt).toFixed(2)),
      detail: cachedView ? `cache_hit:${cachedView.messages?.length || 0}` : 'cache_miss',
    })
    if (cachedConv) {
      set({ activeProjectId: cachedConv.project_id ?? null, activeConversation: cachedConv })
    }
    const fetchStartedAt = nowMs()
    try {
      const conv = cachedConv || await chatApi.getConversation(convId).catch(() => null)
      const paperGuideMode = (conv || cachedConv)?.mode === 'paper_guide'
      const pageResult = await getMessagesPageWithFallback(convId, {
        limit: MESSAGE_PAGE_SIZE,
        renderPacketOnly: paperGuideMode ? true : undefined,
      })
      pushConversationOpenPhase({
        ts: Date.now(),
        convId,
        token: myToken,
        phase: 'fetch_page',
        durationMs: Number((nowMs() - fetchStartedAt).toFixed(2)),
        detail: pageResult.usedFallback
          ? `fallback:${Array.isArray(pageResult.page?.messages) ? pageResult.page.messages.length : 0}`
          : `tail:${Array.isArray(pageResult.page?.messages) ? pageResult.page.messages.length : 0}`,
      })
      const page = pageResult.page
      if (myToken !== conversationSwitchToken || get().activeConvId !== convId) {
        pushSwitchPerf({
          ts: Date.now(),
          convId,
          token: myToken,
          status: 'stale',
          durationMs: Number((nowMs() - startedAt).toFixed(2)),
          usedCache: Boolean(cachedConv),
          messageCount: 0,
          note: 'stale_after_fetch',
        })
        return
      }
      const applyStartedAt = nowMs()
      set({
        activeProjectId: conv?.project_id ?? null,
        activeConversation: conv || cachedConv || null,
        messages: Array.isArray(page?.messages) ? page.messages : [],
        conversationLoading: false,
        messagesLoadingMore: false,
        messagesHasMoreBefore: Boolean(page?.has_more_before),
        oldestLoadedMessageId: Number.isFinite(Number(page?.oldest_loaded_id))
          ? Number(page?.oldest_loaded_id)
          : null,
      })
      set((state) => ({
        conversationCacheById: upsertConversationViewCache(state.conversationCacheById, convId, {
          messages: Array.isArray(page?.messages) ? page.messages : [],
          refs: state.refs,
          messagesHasMoreBefore: Boolean(page?.has_more_before),
          oldestLoadedMessageId: Number.isFinite(Number(page?.oldest_loaded_id))
            ? Number(page?.oldest_loaded_id)
            : null,
          cachedAt: Date.now(),
        }),
      }))
      pushConversationOpenPhase({
        ts: Date.now(),
        convId,
        token: myToken,
        phase: 'apply_page',
        durationMs: Number((nowMs() - applyStartedAt).toFixed(2)),
        detail: `${Array.isArray(page?.messages) ? page.messages.length : 0}`,
      })
      const active = conv || cachedConv || null
      const sourcePath = String(active?.bound_source_path || '').trim()
      if (sourcePath) {
        const sourceName = String(active?.bound_source_name || '').trim()
        set((state) => ({
          guideBindings: {
            ...(state.guideBindings || {}),
            [convId]: { sourcePath, sourceName },
          },
        }))
      }
      scheduleLoadRefsForConversation(convId, set, () => get().activeConvId, 120, undefined, 'open_after_messages')
      pushConversationOpenPhase({
        ts: Date.now(),
        convId,
        token: myToken,
        phase: 'schedule_refs',
        durationMs: 0,
        detail: 'deferred',
      })
      pushSwitchPerf({
        ts: Date.now(),
        convId,
        token: myToken,
        status: 'success',
        durationMs: Number((nowMs() - startedAt).toFixed(2)),
        usedCache: Boolean(cachedView),
        messageCount: Array.isArray(page?.messages) ? page.messages.length : 0,
        note: pageResult.usedFallback
          ? (conv ? 'ok_legacy_messages_fallback' : 'ok_without_conv_meta_legacy_messages_fallback')
          : (conv
            ? (cachedView ? 'ok_tail_refs_deferred_cache_refresh' : 'ok_tail_refs_deferred')
            : (cachedView ? 'ok_without_conv_meta_tail_refs_deferred_cache_refresh' : 'ok_without_conv_meta_tail_refs_deferred')),
      })
    } catch {
      pushConversationOpenPhase({
        ts: Date.now(),
        convId,
        token: myToken,
        phase: 'fetch_error',
        durationMs: Number((nowMs() - fetchStartedAt).toFixed(2)),
      })
      if (myToken !== conversationSwitchToken || get().activeConvId !== convId) {
        pushSwitchPerf({
          ts: Date.now(),
          convId,
          token: myToken,
          status: 'stale',
          durationMs: Number((nowMs() - startedAt).toFixed(2)),
          usedCache: Boolean(cachedConv),
          messageCount: 0,
          note: 'stale_after_error',
        })
        return
      }
      set({
        messages: Array.isArray(cachedView?.messages) ? cachedView.messages : [],
        refs: cachedView?.refs && typeof cachedView.refs === 'object' ? cachedView.refs : {},
        activeConversation: cachedConv || null,
        conversationLoading: false,
        messagesLoadingMore: false,
        messagesHasMoreBefore: Boolean(cachedView?.messagesHasMoreBefore),
        oldestLoadedMessageId: cachedView?.oldestLoadedMessageId ?? null,
      })
      pushSwitchPerf({
        ts: Date.now(),
        convId,
        token: myToken,
        status: 'error',
        durationMs: Number((nowMs() - startedAt).toFixed(2)),
        usedCache: Boolean(cachedConv),
        messageCount: 0,
        note: 'fetch_failed',
      })
    }
  },

  createConversation: async () => {
    const projectId = get().activeProjectId
    const locale = useSettingsStore.getState().uiLocale
    const defaultTitle = buildDefaultConversationTitle(locale)
    const { id } = await chatApi.createConversation(defaultTitle, projectId)
    await get().loadSidebarData()
    stopMessagePostprocessPolling()
    stopUploadPolling()
    set({ generation: null, uploadItems: [], pendingImages: [] })
    await get().selectConversation(id)
    return id
  },

  createPaperGuideConversation: async (opts) => {
    const sourcePath = String(opts.sourcePath || '').trim()
    if (!sourcePath) throw new Error('sourcePath required')
    const locale = useSettingsStore.getState().uiLocale
    const sourceName = String(opts.sourceName || '').trim() || sourcePath.split(/[\\/]/).pop() || (locale === 'zh' ? zh.default_source_fallback : en.default_source_fallback)
    const projectId = opts.projectId ?? get().activeProjectId
    const titleBase = String(opts.title || '').trim() || (locale === 'zh' ? zh.default_guide_title.replace('{name}', sourceName) : en.default_guide_title.replace('{name}', sourceName))
    const { id } = await chatApi.createConversation(titleBase, projectId, {
      mode: 'paper_guide',
      bound_source_path: sourcePath,
      bound_source_name: sourceName,
      bound_source_ready: true,
    })
    try {
      await chatApi.updateConversationGuide(id, {
        mode: 'paper_guide',
        bound_source_path: sourcePath,
        bound_source_name: sourceName,
        bound_source_ready: true,
      })
    } catch {
      // Backward compatible: old backend may not expose /guide.
    }
    await get().loadSidebarData()
    stopMessagePostprocessPolling()
    stopUploadPolling()
    set((state) => ({
      generation: null,
      uploadItems: [],
      pendingImages: [],
      guideBindings: {
        ...(state.guideBindings || {}),
        [id]: { sourcePath, sourceName },
      },
    }))
    await get().selectConversation(id)
    set((state) => ({
      activeConversation: state.activeConversation && state.activeConversation.id === id
        ? {
            ...state.activeConversation,
            mode: 'paper_guide',
            bound_source_path: sourcePath,
            bound_source_name: sourceName,
            bound_source_ready: true,
          }
        : state.activeConversation,
    }))
    return id
  },

  renameConversation: async (id, title) => {
    const nextTitle = String(title || '').trim()
    if (!nextTitle) return
    await chatApi.updateTitle(id, nextTitle)
    await get().loadSidebarData()
  },

  deleteConversation: async (id) => {
    stopRefsPolling()
    stopMessagePostprocessPolling()
    stopUploadPolling()
    await chatApi.deleteConversation(id)
    const state = get()
    set((cur) => {
      const nextBindings = { ...(cur.guideBindings || {}) }
      const nextCache = { ...(cur.conversationCacheById || {}) }
      delete nextBindings[id]
      delete nextCache[id]
      if (state.activeConvId === id) {
        return {
          activeConvId: null,
          activeConversation: null,
          messages: [],
          conversationLoading: false,
          messagesLoadingMore: false,
          messagesHasMoreBefore: false,
          oldestLoadedMessageId: null,
          refs: {},
          generation: null,
          uploadItems: [],
          pendingImages: [],
          guideBindings: nextBindings,
          conversationCacheById: nextCache,
        }
      }
      return {
        guideBindings: nextBindings,
        conversationCacheById: nextCache,
      }
    })
    await get().loadSidebarData()
  },

  moveConversation: async (convId, projectId) => {
    await chatApi.updateConversationProject(convId, projectId)
    await get().loadSidebarData()
    if (get().activeConvId === convId) {
      set((state) => ({
        activeProjectId: projectId,
        activeConversation: state.activeConversation ? { ...state.activeConversation, project_id: projectId } : state.activeConversation,
      }))
    }
  },

  loadOlderMessages: async () => {
    const state = get()
    const convId = String(state.activeConvId || '').trim()
    const beforeId = Number(state.oldestLoadedMessageId || 0)
    if (!convId || state.conversationLoading || state.messagesLoadingMore || !state.messagesHasMoreBefore || beforeId <= 0) {
      return
    }
    set({ messagesLoadingMore: true })
    try {
      const page = await chatApi.getMessagesPage(convId, {
        limit: MESSAGE_PAGE_SIZE,
        beforeId,
        renderPacketOnly: state.activeConversation?.mode === 'paper_guide' ? true : undefined,
      })
      if (get().activeConvId !== convId) return
      const olderMessages = Array.isArray(page?.messages) ? page.messages : []
      set((current) => {
        const seen = new Set(current.messages.map((item) => Number(item.id || 0)))
        const merged = [
          ...olderMessages.filter((item) => !seen.has(Number(item.id || 0))),
          ...current.messages,
        ]
        return {
          messages: merged,
          messagesLoadingMore: false,
          messagesHasMoreBefore: Boolean(page?.has_more_before),
          oldestLoadedMessageId: Number.isFinite(Number(page?.oldest_loaded_id))
            ? Number(page?.oldest_loaded_id)
            : (merged.length > 0 ? Number(merged[0]?.id || 0) || null : null),
          conversationCacheById: upsertConversationViewCache(current.conversationCacheById, convId, {
            messages: merged,
            refs: current.refs,
            messagesHasMoreBefore: Boolean(page?.has_more_before),
            oldestLoadedMessageId: Number.isFinite(Number(page?.oldest_loaded_id))
              ? Number(page?.oldest_loaded_id)
              : (merged.length > 0 ? Number(merged[0]?.id || 0) || null : null),
            cachedAt: Date.now(),
          }),
        }
      })
    } catch {
      if (get().activeConvId === convId) {
        set({ messagesLoadingMore: false })
      }
    }
  },

  uploadFiles: async (files, opts) => {
    if (!files.length) return
    set({ uploading: true })
    try {
      const hasPdf = files.some((file) => String(file.name || '').toLowerCase().endsWith('.pdf') || String(file.type || '').toLowerCase() === 'application/pdf')
      let convId = String(opts?.convId || '').trim()
      if (!convId) {
        convId = String(get().activeConvId || '').trim()
      }
      if (hasPdf && !convId) {
        convId = await get().createConversation()
      }
      const res = await chatApi.uploadFiles(files, { ...(opts || {}), convId: convId || null })
      const imageAttachments = (res.items || [])
        .map((item) => item.attachment)
        .filter((item): item is ChatImageAttachment => Boolean(item && item.path))
      set((state) => ({
        uploading: false,
        uploadItems: mergeUploadItems(state.uploadItems, res.items || []),
        pendingImages: mergeImageAttachments(state.pendingImages, imageAttachments),
      }))
      if (needsUploadStatusPolling(get().uploadItems)) {
        void startUploadPolling(set, get)
      }
    } catch {
      set((state) => ({
        uploading: false,
        uploadItems: mergeUploadItems(state.uploadItems, [{
          kind: 'unknown',
          status: 'error',
          name: 'upload',
          error: 'upload failed',
        }]),
      }))
      throw new Error('upload failed')
    }
  },

  retryUploadItem: async (key) => {
    const current = get().uploadItems.find((item) => uploadItemKey(item) === key)
    if (!current || current.kind !== 'pdf' || !current.ingest_job_id) return
    const shouldRetryQuality = (
      current.ready === true
      && String(current.ingest_status || '') === 'ready'
      && String(current.quality_status || '') === 'error'
    )
    const res = shouldRetryQuality
      ? await chatApi.retryUploadQualityJob(current.ingest_job_id)
      : await chatApi.retryUploadJob(current.ingest_job_id)
    const nextItem = res.item
    set((state) => ({
      uploadItems: mergeUploadItems(
        state.uploadItems.filter((item) => uploadItemKey(item) !== key),
        nextItem ? [nextItem] : [],
      ),
    }))
    if (needsUploadStatusPolling(get().uploadItems)) {
      void startUploadPolling(set, get)
    }
  },

  cancelUploadItem: async (key) => {
    const current = get().uploadItems.find((item) => uploadItemKey(item) === key)
    if (!current || current.kind !== 'pdf' || !current.ingest_job_id) return
    const res = await chatApi.cancelUploadJob(current.ingest_job_id)
    const nextItem = res.item
    set((state) => ({
      uploadItems: mergeUploadItems(state.uploadItems, nextItem ? [nextItem] : []),
    }))
    if (!needsUploadStatusPolling(get().uploadItems)) {
      stopUploadPolling()
    }
  },

  removePendingImage: (key) => {
    set((state) => ({
      pendingImages: state.pendingImages.filter((item) => attachmentKey(item) !== key),
      uploadItems: state.uploadItems.filter((item) => {
        const attachment = item.attachment
        return !attachment || attachmentKey(attachment) !== key
      }),
    }))
  },

  dismissUploadItem: (key) => {
    set((state) => ({
      uploadItems: state.uploadItems.filter((item) => uploadItemKey(item) !== key),
    }))
    if (!needsUploadStatusPolling(get().uploadItems)) {
      stopUploadPolling()
    }
  },

  sendMessage: async (prompt, opts) => {
    stopMessagePostprocessPolling()
    let convId = get().activeConvId
    if (!convId) {
      convId = await get().createConversation()
    }

    const stateNow = get()
    const pendingImages = stateNow.pendingImages
    const localGuide = (convId ? stateNow.guideBindings?.[convId] : undefined)
    const boundSourcePath = String(stateNow.activeConversation?.bound_source_path || localGuide?.sourcePath || '').trim()
    const boundSourceName = String(stateNow.activeConversation?.bound_source_name || localGuide?.sourceName || '').trim()
    const preferredSources = stateNow.uploadItems
      .filter((item) => item.kind === 'pdf' && (item.status === 'duplicate' || item.ingest_status === 'ready'))
      .flatMap((item) => [String(item.path || '').trim(), String(item.name || '').trim()])
      .filter(Boolean)
    for (const hint of [boundSourcePath, boundSourceName]) {
      const v = String(hint || '').trim()
      if (!v) continue
      if (!preferredSources.includes(v)) preferredSources.unshift(v)
    }
    const preferredSourcesFinal = preferredSources.slice(0, 4)
    const trimmedPrompt = prompt.trim()
    const userStoreText = trimmedPrompt || `[Image attachment x${pendingImages.length}]`
    const shouldKeepPollingRefs = () => {
      const state = get()
      return state.activeConvId === convId && Boolean(state.generation)
    }

    const res = await api.post<{
      session_id: string
      task_id: string
      trace_id?: string
      user_msg_id: number
      assistant_msg_id: number
      conversation_title?: string
    }>('/api/generate', {
      conv_id: convId,
      prompt: trimmedPrompt,
      image_attachments: pendingImages,
      preferred_sources: preferredSourcesFinal,
      source_lock_path: boundSourcePath,
      source_lock_name: boundSourceName,
      top_k: opts.topK,
      temperature: opts.temperature,
      max_tokens: opts.maxTokens,
      deep_read: opts.deepRead,
      prompt_context: opts.promptContext || undefined,
    })
    const conversationTitle = String(res.conversation_title || '').trim()
    const userMessage: Message = {
      id: res.user_msg_id,
      role: 'user',
      content: userStoreText,
      created_at: Date.now() / 1000,
      attachments: pendingImages,
      meta: opts.promptContext ? { prompt_context: opts.promptContext } : undefined,
    }

    set((state) => ({
      messages: [...state.messages, userMessage],
      pendingImages: [],
      uploadItems: state.uploadItems.filter((item) => item.kind !== 'image'),
      conversationLoading: false,
      generation: {
        sessionId: res.session_id,
        taskId: res.task_id,
        traceId: res.trace_id,
        stage: 'starting',
        partial: '',
        done: false,
      },
      activeConversation: conversationTitle && state.activeConversation?.id === convId
        ? { ...state.activeConversation, title: conversationTitle }
        : state.activeConversation,
      rootConversations: conversationTitle
        ? patchConversationTitle(state.rootConversations, convId!, conversationTitle)
        : state.rootConversations,
      projectConversations: conversationTitle
        ? patchProjectConversationTitle(state.projectConversations, convId!, conversationTitle)
        : state.projectConversations,
      conversationCacheById: convId
        ? upsertConversationViewCache(state.conversationCacheById, convId, {
          messages: [...state.messages, userMessage],
          refs: state.refs,
          messagesHasMoreBefore: state.messagesHasMoreBefore,
          oldestLoadedMessageId: state.oldestLoadedMessageId,
          cachedAt: Date.now(),
        })
        : state.conversationCacheById,
    }))
      scheduleLoadRefsForConversation(
        convId,
        set,
        () => get().activeConvId,
        350,
        shouldKeepPollingRefs,
        'generation_started',
      )

    const ctrl = new AbortController()
    set({ sseController: ctrl })

    try {
      const sseRes = await authFetch(`/api/generate/${res.session_id}/stream`, {
        signal: ctrl.signal,
      })
      const reader = sseRes.body!.getReader()
      const decoder = new TextDecoder()
      let buf = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop() || ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const data = JSON.parse(line.slice(6))
            set({
              generation: {
                sessionId: res.session_id,
                taskId: res.task_id,
                traceId: res.trace_id,
                stage: data.stage || '',
                partial: data.partial || '',
                done: !!data.done,
                researchTrace: data.research_trace && typeof data.research_trace === 'object'
                  ? data.research_trace as Record<string, unknown>
                  : undefined,
              },
            })
            if (data.done) {
              const { page } = await getMessagesPageWithFallback(convId!, {
                limit: MESSAGE_PAGE_SIZE,
                renderPacketOnly: get().activeConversation?.mode === 'paper_guide' ? true : undefined,
              })
              set((state) => {
                const merged = mergeLatestMessagePage(
                  state.messages,
                  state.messagesHasMoreBefore,
                  page,
                )
                return {
                  messages: merged.messages,
                  generation: null,
                  conversationLoading: false,
                  messagesLoadingMore: false,
                  messagesHasMoreBefore: merged.hasMoreBefore,
                  oldestLoadedMessageId: merged.oldestLoadedMessageId,
                  conversationCacheById: upsertConversationViewCache(state.conversationCacheById, convId!, {
                    messages: merged.messages,
                    refs: state.refs,
                    messagesHasMoreBefore: merged.hasMoreBefore,
                    oldestLoadedMessageId: merged.oldestLoadedMessageId,
                    cachedAt: Date.now(),
                  }),
                }
              })
              scheduleLoadRefsForConversation(convId!, set, () => get().activeConvId, 120, undefined, 'generation_done')
              const postprocessState = get()
              const paperGuideMode = Boolean(
                postprocessState.activeConversation?.mode === 'paper_guide'
                || postprocessState.guideBindings?.[convId!]?.sourcePath
                || boundSourcePath,
              )
              const assistantMessage = postprocessState.messages.find(
                (item) => Number(item.id || 0) === Number(res.assistant_msg_id || 0),
              ) || null
              if (paperGuideMode || messageNeedsPostprocessRefresh(assistantMessage, { paperGuideMode })) {
                void startMessagePostprocessPolling(
                  convId!,
                  res.assistant_msg_id,
                  set,
                  get,
                  { paperGuideMode, reason: 'generation_done' },
                )
              }
              await get().loadSidebarData()
              return
            }
          } catch {
            // ignore malformed SSE chunks
          }
        }
      }
    } catch {
      // aborted or network error
    } finally {
      set({ sseController: null })
    }
  },

  cancelGeneration: () => {
    const state = get()
    if (state.generation && state.sseController) {
      state.sseController.abort()
      api.post(`/api/generate/${state.generation.sessionId}/cancel?task_id=${state.generation.taskId}`)
        .catch(() => {})
    }
    stopRefsPolling()
    stopMessagePostprocessPolling()
    set({ generation: null, sseController: null })
  },

  clearGeneration: () => set({ generation: null }),
}))
