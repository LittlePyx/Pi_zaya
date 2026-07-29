import { create } from 'zustand'
import {
  chatApi,
  type ChatImageAttachment,
  type ChatUploadItem,
  type Conversation,
  type QueryScope,
  type Message,
  type MessagePage,
  type Project,
  type SidebarSnapshot,
} from '../api/chat'
import { api, authFetch, responseError } from '../api/client'
import { S as zh } from '../i18n/zh'
import { S as en } from '../i18n/en'
import { useSettingsStore } from './settingsStore'
import { basenameFromSourcePath } from '../utils/sourcePath'
import {
  ensureChatStorePerfApi,
  pushConversationOpenPhase,
  pushRefsPerf,
  pushSwitchPerf,
  refsBackendPerf,
  summarizeRefsPayload,
} from './chatStorePerf'
import { buildFullMessagePage, mergeLatestMessagePage } from './chatStoreMessages'
import {
  attachmentKey,
  cachedPendingImages,
  cachedUploadItems,
  isPdfUploadJobRunning,
  mergeImageAttachments,
  mergeUploadItems,
  needsAnyUploadStatusPolling,
  uploadItemKey,
} from './chatStoreUploads'
import { startUploadPolling, stopUploadPolling } from './chatStoreUploadPolling'

let refsPollToken = 0
let refsPollTimer: number | null = null
let refsPollController: AbortController | null = null
let refsLoadRequestSequence = 0
const latestRefsLoadRequestByConversation = new Map<string, number>()
const refsLoadControllerByConversation = new Map<string, AbortController>()
let localeRefreshSequence = 0
let messagePostprocessPollToken = 0
let messagePostprocessPollTimer: number | null = null
let conversationSwitchToken = 0
const NEW_CONVERSATION_GENERATION_LOCK = '__new_conversation_generation__'
const generationStartLocks = new Set<string>()
const SIDEBAR_CONVERSATION_LIMIT = 80
// Render packets can include source evidence and citation-card contracts for
// every assistant turn.  Loading twelve messages keeps the first screen useful
// while older turns remain available through the existing paged history UI.
const MESSAGE_PAGE_SIZE = 12
// Post-generation polling only needs the just-created user/assistant pair.
// Keep a small safety margin without repeatedly re-rendering a whole history.
const POSTPROCESS_MESSAGE_PAGE_SIZE = 4
const GENERATION_START_FAILED_CODE = 'generation_start_failed'
const GENERATION_START_FAILED_MESSAGE_EN = 'Generation could not be started. Please retry.'
const GENERATION_START_FAILED_MESSAGE_ZH = '回答任务未能启动，请稍后重试。'

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

function localizedGenerationStartFailedMessage(locale: string) {
  return locale.toLowerCase().startsWith('zh')
    ? zh.chat_generation_start_failed
    : en.chat_generation_start_failed
}

function generationStartFailureDisplayMessage(error: string, locale: string) {
  const raw = String(error || '').trim()
  if (
    raw === GENERATION_START_FAILED_CODE
    || raw === GENERATION_START_FAILED_MESSAGE_EN
    || raw === GENERATION_START_FAILED_MESSAGE_ZH
  ) {
    return localizedGenerationStartFailedMessage(locale)
  }
  return raw || localizedGenerationStartFailedMessage(locale)
}

function localizedGenerationStreamFailedMessage(locale: string) {
  return locale.toLowerCase().startsWith('zh')
    ? zh.chat_generation_stream_failed
    : en.chat_generation_stream_failed
}

function localizedGenerationStreamIncompleteMessage(locale: string) {
  return locale.toLowerCase().startsWith('zh')
    ? zh.chat_generation_stream_incomplete
    : en.chat_generation_stream_incomplete
}

function generationStreamFailureDisplayMessage(error: unknown, locale: string) {
  const raw = error instanceof Error ? error.message.trim() : String(error || '').trim()
  if (!raw) return localizedGenerationStreamFailedMessage(locale)
  if (/generation stream did not return a readable body/i.test(raw)) {
    return localizedGenerationStreamFailedMessage(locale)
  }
  if (/generation stream ended before completion/i.test(raw)) {
    return localizedGenerationStreamIncompleteMessage(locale)
  }
  if (/^(?:408|429|5\d\d)\b/.test(raw)) {
    return localizedGenerationStreamFailedMessage(locale)
  }
  if (/^generation stream failed\.?$/i.test(raw)) {
    return localizedGenerationStreamFailedMessage(locale)
  }
  return raw
}

function isCanonicalGenerationStartFailureText(text: string) {
  const raw = String(text || '').trim()
  return (
    raw === GENERATION_START_FAILED_CODE
    || raw === GENERATION_START_FAILED_MESSAGE_EN
    || raw === GENERATION_START_FAILED_MESSAGE_ZH
  )
}

function localizeGenerationStartFailurePage(
  page: MessagePage,
  assistantMsgId: number,
  message: string,
): MessagePage {
  const targetId = Number(assistantMsgId || 0)
  if (!targetId || !Array.isArray(page?.messages)) return page
  let changed = false
  const messages = page.messages.map((item) => {
    if (Number(item.id || 0) !== targetId || !isCanonicalGenerationStartFailureText(item.content)) return item
    changed = true
    return { ...item, content: message }
  })
  return changed ? { ...page, messages } : page
}

function generationFailureMessageContent(generation: GenerationState | null | undefined, failureMessage: string) {
  const partial = String(generation?.partial || '').trim()
  const message = String(failureMessage || '').trim()
  if (!partial) return message
  if (!message) return partial
  if (partial.includes(message)) return partial
  return `${partial}\n\n${message}`
}

function clearStaleMessagePresentation(message: Message): Message {
  const next: Message = { ...message }
  delete next.rendered_content
  delete next.rendered_body
  delete next.notice
  delete next.cite_details
  delete next.copy_text
  delete next.copy_markdown
  delete next.refs_user_msg_id
  delete next.render_cache_key
  delete next.provenance

  const meta = message.meta && typeof message.meta === 'object'
    ? { ...message.meta } as Record<string, unknown>
    : {}
  for (const key of [
    'paper_guide_contracts',
    'render_cache',
    'provenance',
    'answer_quality',
    'answer_runtime_check',
    'answer_contract',
    'answerContract',
    'research_trace',
    'agent_trace',
    'agent_source_summary',
    'agentSourceSummary',
    'agent_trace_available',
    'agentTraceAvailable',
  ]) {
    delete meta[key]
  }
  if (Object.keys(meta).length > 0) next.meta = meta
  else delete next.meta
  return next
}

function terminalMessageBase(message: Message): Message {
  // The terminal SSE event currently carries answer text, but no atomic
  // render packet whose refs/locale/citation-plan signatures can prove that
  // an existing packet belongs to this generation. Matching answer_markdown
  // alone is insufficient: the same prose can have different source cards or
  // locale. Clear presentation metadata and let message hydration install the
  // server-authoritative packet.
  return clearStaleMessagePresentation(message)
}

function currentGenerationMeta(generation: GenerationState | null | undefined): Record<string, unknown> {
  return {
    ...(generation?.traceId ? { trace_id: generation.traceId } : {}),
    ...(generation?.researchTrace ? { research_trace: generation.researchTrace } : {}),
    ...(generation?.agentTrace ? { agent_trace: generation.agentTrace } : {}),
    ...(generation?.agentSourceSummary ? { agent_source_summary: generation.agentSourceSummary } : {}),
    ...(generation?.answerContract ? { answer_contract: generation.answerContract } : {}),
  }
}

function upsertGenerationFailureMessage(
  messages: Message[],
  generation: GenerationState | null | undefined,
  failureMessage: string,
): Message[] {
  const content = generationFailureMessageContent(generation, failureMessage)
  if (!content) return messages
  const assistantMsgId = Number(generation?.assistantMsgId || 0)
  const fallbackId = Number.isFinite(assistantMsgId) && assistantMsgId > 0
    ? assistantMsgId
    : Math.floor(Date.now())
  const createdAt = Date.now() / 1000
  const patchMessage = (message: Message): Message => {
    const base = terminalMessageBase(message)
    return {
      ...base,
      role: 'assistant',
      content,
      created_at: Number.isFinite(Number(message.created_at)) ? message.created_at : createdAt,
      meta: {
        ...(base.meta || {}),
        ...currentGenerationMeta(generation),
        generation_status: 'failed',
      },
    }
  }
  if (assistantMsgId > 0) {
    let found = false
    const next = messages.map((message) => {
      if (Number(message.id || 0) !== assistantMsgId) return message
      found = true
      return patchMessage(message)
    })
    if (found) return next
  }
  return [
    ...messages,
    {
      id: fallbackId,
      role: 'assistant',
      content,
      created_at: createdAt,
      meta: {
        ...(generation?.traceId ? { trace_id: generation.traceId } : {}),
        generation_status: 'failed',
      },
    },
  ]
}

function upsertGenerationCompletedMessage(
  messages: Message[],
  generation: GenerationState | null | undefined,
  finalContent: string,
): Message[] {
  const content = String(finalContent || generation?.partial || '').trim()
  if (!content) return messages
  const assistantMsgId = Number(generation?.assistantMsgId || 0)
  const fallbackId = Number.isFinite(assistantMsgId) && assistantMsgId > 0
    ? assistantMsgId
    : Math.floor(Date.now())
  const createdAt = Date.now() / 1000
  const patchMessage = (message: Message): Message => {
    const base = terminalMessageBase(message)
    return {
      ...base,
      role: 'assistant',
      content,
      created_at: Number.isFinite(Number(message.created_at)) ? message.created_at : createdAt,
      meta: {
        ...(base.meta || {}),
        ...currentGenerationMeta(generation),
        generation_status: 'done',
      },
    }
  }
  if (assistantMsgId > 0) {
    let found = false
    const next = messages.map((message) => {
      if (Number(message.id || 0) !== assistantMsgId) return message
      found = true
      return patchMessage(message)
    })
    if (found) return next
  }
  return [
    ...messages,
    {
      id: fallbackId,
      role: 'assistant',
      content,
      created_at: createdAt,
      meta: {
        ...(generation?.traceId ? { trace_id: generation.traceId } : {}),
        generation_status: 'done',
      },
    },
  ]
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

function rehomeDeletedProjectLocally(
  state: ChatState,
  deletedProjectId: string,
): Partial<ChatState> {
  const projectId = String(deletedProjectId || '').trim()
  if (!projectId) return {}
  const nextProjects = state.projects.filter((project) => project.id !== projectId)
  const nextProjectConversations: Record<string, Conversation[]> = {}
  const rehomed: Conversation[] = []
  for (const [pid, conversations] of Object.entries(state.projectConversations || {})) {
    if (pid === projectId) {
      rehomed.push(...(conversations || []).map((conversation) => ({ ...conversation, project_id: null })))
      continue
    }
    nextProjectConversations[pid] = conversations || []
  }
  const nextRootConversations = [
    ...state.rootConversations.map((conversation) => (
      conversation.project_id === projectId ? { ...conversation, project_id: null } : conversation
    )),
    ...rehomed,
  ]
  const activeConversation = state.activeConversation?.project_id === projectId
    ? { ...state.activeConversation, project_id: null }
    : state.activeConversation
  return {
    projects: nextProjects,
    projectConversations: nextProjectConversations,
    rootConversations: nextRootConversations,
    activeProjectId: state.activeProjectId === projectId ? null : state.activeProjectId,
    activeConversation,
  }
}

function moveConversationLocally(
  state: ChatState,
  convId: string,
  projectId: string | null,
): Partial<ChatState> {
  const targetConvId = String(convId || '').trim()
  if (!targetConvId) return {}
  const targetProjectId = String(projectId || '').trim() || null
  const source = findConversationInLists(state.rootConversations, state.projectConversations, targetConvId)
    || (state.activeConversation?.id === targetConvId ? state.activeConversation : null)
  if (!source) return {}
  const moved: Conversation = { ...source, project_id: targetProjectId }
  const nextRootConversations = state.rootConversations
    .filter((conversation) => conversation.id !== targetConvId)
  if (!targetProjectId) {
    nextRootConversations.push(moved)
  }
  const nextProjectConversations: Record<string, Conversation[]> = {}
  const projectIds = new Set([
    ...Object.keys(state.projectConversations || {}),
    ...(targetProjectId ? [targetProjectId] : []),
  ])
  for (const pid of projectIds) {
    const existing = (state.projectConversations[pid] || [])
      .filter((conversation) => conversation.id !== targetConvId)
    nextProjectConversations[pid] = pid === targetProjectId
      ? [...existing, moved]
      : existing
  }
  return {
    rootConversations: nextRootConversations,
    projectConversations: nextProjectConversations,
    activeProjectId: state.activeConvId === targetConvId ? targetProjectId : state.activeProjectId,
    activeConversation: state.activeConversation?.id === targetConvId
      ? { ...state.activeConversation, project_id: targetProjectId }
      : state.activeConversation,
  }
}

function nowMs() {
  try {
    return performance.now()
  } catch {
    return Date.now()
  }
}

async function getMessagesPageWithFallback(
  convId: string,
  opts?: { limit?: number; beforeId?: number | null; renderPacketOnly?: boolean },
): Promise<{ page: MessagePage; usedFallback: boolean }> {
  try {
    const page = await chatApi.getMessagesPage(convId, opts)
    return { page, usedFallback: false }
  } catch (pageError) {
    const beforeId = Number(opts?.beforeId || 0)
    if (beforeId > 0) {
      throw pageError
    }
    let messages: Message[]
    try {
      messages = await chatApi.getMessages(convId, { renderPacketOnly: opts?.renderPacketOnly })
    } catch (fallbackError) {
      const pageMessage = pageError instanceof Error ? pageError.message : String(pageError || 'messages page failed')
      const fallbackMessage = fallbackError instanceof Error ? fallbackError.message : String(fallbackError || 'messages fallback failed')
      throw new Error(`${pageMessage}; fallback failed: ${fallbackMessage}`)
    }
    return {
      page: buildFullMessagePage(Array.isArray(messages) ? messages : []),
      usedFallback: true,
    }
  }
}

function scheduleRefreshLatestMessagesAfterCancel(
  convId: string,
  generation: GenerationState,
  set: (patch: Partial<ChatState> | ((state: ChatState) => Partial<ChatState>)) => void,
) {
  const targetConvId = String(convId || '').trim()
  const sessionId = String(generation.sessionId || '').trim()
  const assistantMsgId = Number(generation.assistantMsgId || 0)
  if (!targetConvId || !sessionId) return

  const run = async (attempt: number) => {
    try {
      const { page } = await getMessagesPageWithFallback(targetConvId, { limit: MESSAGE_PAGE_SIZE })
      const latestMessages = Array.isArray(page?.messages) ? page.messages : []
      const canceledAssistant = assistantMsgId > 0
        ? latestMessages.find((item) => Number(item.id || 0) === assistantMsgId)
        : null
      const canceledAssistantStatus = String(canceledAssistant?.meta?.generation_status || '').trim().toLowerCase()
      const canceledAssistantContent = String(canceledAssistant?.content || '').trim()
      const hasCanceledAssistant = assistantMsgId > 0 ? Boolean(canceledAssistant) : true
      const hasFinalCanceledAssistant = assistantMsgId <= 0 || Boolean(
        canceledAssistant
        && (
          canceledAssistantStatus === 'canceled'
          || /generation\s+cancell?ed(?:\s+by\s+user)?[.)]?\s*$/i.test(canceledAssistantContent)
        )
      )
      if ((!hasCanceledAssistant || !hasFinalCanceledAssistant) && attempt < 4) {
        schedule(attempt + 1)
        return
      }
      if (!hasCanceledAssistant || !hasFinalCanceledAssistant) return

      set((state) => {
        const previousCache = state.conversationCacheById[targetConvId]
        const visibleForRequest = state.activeConvId === targetConvId
        const conversationStillKnown = visibleForRequest
          || Boolean(previousCache)
          || Boolean(findConversationInLists(state.rootConversations, state.projectConversations, targetConvId))
        if (!conversationStillKnown) return {}
        const currentGeneration = visibleForRequest ? state.generation : previousCache?.generation
        if (currentGeneration && currentGeneration.sessionId !== sessionId) return {}
        const baseMessages = visibleForRequest
          ? state.messages
          : (Array.isArray(previousCache?.messages) ? previousCache.messages : [])
        const baseHasMoreBefore = visibleForRequest
          ? state.messagesHasMoreBefore
          : Boolean(previousCache?.messagesHasMoreBefore)
        const merged = mergeLatestMessagePage(baseMessages, baseHasMoreBefore, page)
        const nextCache = upsertConversationViewCache(state.conversationCacheById, targetConvId, {
          messages: merged.messages,
          refs: visibleForRequest
            ? state.refs
            : (previousCache?.refs && typeof previousCache.refs === 'object' ? previousCache.refs : {}),
          messagesHasMoreBefore: merged.hasMoreBefore,
          oldestLoadedMessageId: merged.oldestLoadedMessageId,
          generation: currentGeneration && currentGeneration.sessionId === sessionId ? null : previousCache?.generation ?? null,
          sseController: null,
          cachedAt: Date.now(),
        })
        if (!visibleForRequest) {
          return { conversationCacheById: nextCache }
        }
        return {
          messages: merged.messages,
          generation: currentGeneration && currentGeneration.sessionId === sessionId ? null : state.generation,
          sseController: null,
          conversationLoading: false,
          messagesHasMoreBefore: merged.hasMoreBefore,
          oldestLoadedMessageId: merged.oldestLoadedMessageId,
          conversationCacheById: nextCache,
        }
      })
    } catch {
      if (attempt < 4) schedule(attempt + 1)
    }
  }

  const schedule = (attempt: number) => {
    const delay = attempt <= 1 ? 180 : attempt === 2 ? 420 : 900
    if (typeof window === 'undefined') {
      void run(attempt)
      return
    }
    window.setTimeout(() => { void run(attempt) }, delay)
  }

  schedule(1)
}

if (typeof window !== 'undefined') {
  ensureChatStorePerfApi()
}

function stopRefsPolling() {
  refsPollToken += 1
  refsPollController?.abort()
  refsPollController = null
  if (refsPollTimer !== null) {
    window.clearTimeout(refsPollTimer)
    refsPollTimer = null
  }
}

function refsLocalePreferenceKey() {
  const settings = useSettingsStore.getState()
  return `${settings.uiLocale}:${settings.refsCardLocale}:${settings.localePreferencesRevision}`
}

function invalidateRefsLoadForConversation(convId: string) {
  const key = String(convId || '').trim()
  if (!key) return
  latestRefsLoadRequestByConversation.set(key, ++refsLoadRequestSequence)
  refsLoadControllerByConversation.get(key)?.abort()
  refsLoadControllerByConversation.delete(key)
}

function beginRefsLoadForConversation(convId: string) {
  const key = String(convId || '').trim()
  const previous = refsLoadControllerByConversation.get(key)
  previous?.abort()
  const controller = new AbortController()
  const requestSequence = ++refsLoadRequestSequence
  latestRefsLoadRequestByConversation.set(key, requestSequence)
  refsLoadControllerByConversation.set(key, controller)
  return { controller, requestSequence, localePreferenceKey: refsLocalePreferenceKey() }
}

function refsLoadRequestIsCurrent(
  convId: string,
  requestSequence: number,
  localePreferenceKey: string,
  getActiveConvId: () => string | null,
) {
  return latestRefsLoadRequestByConversation.get(convId) === requestSequence
    && refsLocalePreferenceKey() === localePreferenceKey
    && getActiveConvId() === convId
}

function finishRefsLoadForConversation(convId: string, controller: AbortController) {
  if (refsLoadControllerByConversation.get(convId) === controller) {
    refsLoadControllerByConversation.delete(convId)
  }
}

function stopMessagePostprocessPolling() {
  messagePostprocessPollToken += 1
  if (messagePostprocessPollTimer !== null) {
    window.clearTimeout(messagePostprocessPollTimer)
    messagePostprocessPollTimer = null
  }
}

function latestRefsPack(refs: Record<string, unknown>) {
  const entries = Object.entries(refs || {})
    .map(([key, value]) => ({ key, id: Number(key), value }))
    .filter((entry) => Number.isFinite(entry.id))
    .sort((a, b) => b.id - a.id)
  return entries[0]?.value as {
    hits?: Array<{ ui_meta?: Record<string, unknown>; meta?: Record<string, unknown> }>
    enrichment_pending?: boolean
    payload_mode?: string
    render_status?: string
    display_state?: string
    polish_status?: string
  } | undefined
}

function needsRefsEnrichment(refs: Record<string, unknown>) {
  // Historical pending packs must not keep a settled conversation polling for
  // five minutes. Only the latest user turn can still change what the user is
  // currently waiting to see.
  const rec = latestRefsPack(refs)
  if (!rec) return false
  if (rec.enrichment_pending === true) return true
  if (rec.enrichment_pending === false) return false
  const renderStatus = String(rec.render_status || '').trim().toLowerCase()
  const displayState = String(rec.display_state || '').trim().toLowerCase()
  const polishStatus = String(rec.polish_status || '').trim().toLowerCase()
  if (
    renderStatus === 'full'
    || renderStatus === 'failed'
    || displayState === 'empty'
    || displayState === 'hidden_by_guide'
    || displayState === 'suppressed'
    || polishStatus === 'full'
  ) {
    return false
  }
  if (rec.enrichment_pending) return true
  const payloadMode = String(rec.payload_mode || '').trim().toLowerCase()
  if (payloadMode === 'fast' || payloadMode === 'pending') return true
  const hits = Array.isArray(rec.hits) ? rec.hits : []
  return hits.some((hit) => (
    String(hit?.meta?.ref_pack_state || '').trim().toLowerCase() === 'pending'
  ))
}

function refsPayloadRevision(refs: Record<string, unknown>) {
  // Compare the full public response. A narrow projection can miss a later
  // DOI, bibliometrics, SystemB trace, or locate-target upgrade even when the
  // visible summary sentence stays unchanged.
  return JSON.stringify(refs || {})
}

async function loadRefsForConversation(
  convId: string,
  set: (patch: Partial<ChatState> | ((state: ChatState) => Partial<ChatState>)) => void,
  getActiveConvId: () => string | null,
  shouldKeepPolling?: () => boolean,
  reason = 'load',
) {
  const startedAt = nowMs()
  // A direct load supersedes any older polling response for the same visible
  // refs surface. Abort the poll before assigning this request's token.
  const supersededPoll = refsPollController !== null || refsPollTimer !== null
  const supersededPollToken = refsPollToken
  stopRefsPolling()
  if (supersededPoll) {
    pushRefsPerf({
      ts: Date.now(),
      convId,
      phase: 'poll_stop',
      token: supersededPollToken,
      durationMs: 0,
      reason: 'superseded_by_direct_load',
      active: getActiveConvId() === convId,
    })
  }
  const token = refsPollToken
  const { controller, requestSequence, localePreferenceKey } = beginRefsLoadForConversation(convId)
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
    const { data: refs, meta } = await chatApi.getRefsWithMeta(convId, { signal: controller.signal })
    const durationMs = Number((nowMs() - startedAt).toFixed(2))
    if (!refsLoadRequestIsCurrent(convId, requestSequence, localePreferenceKey, getActiveConvId)) {
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
    const nextRevision = refsPayloadRevision(refs)
    set((state) => {
      if (
        state.activeConvId !== convId
        || !refsLoadRequestIsCurrent(convId, requestSequence, localePreferenceKey, getActiveConvId)
      ) return {}
      if (refsPayloadRevision(state.refs) === nextRevision) return state
      return {
        refs,
        conversationCacheById: upsertConversationViewCache(state.conversationCacheById, convId, {
          refs,
          cachedAt: Date.now(),
        }),
      }
    })
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
    if (
      (needsEnrichment || keepPolling)
      && refsLoadRequestIsCurrent(convId, requestSequence, localePreferenceKey, getActiveConvId)
    ) {
      void startRefsPolling(convId, set, getActiveConvId, shouldKeepPolling, `${reason}:followup`)
    }
  } catch (err) {
    if (
      controller.signal.aborted
      || !refsLoadRequestIsCurrent(convId, requestSequence, localePreferenceKey, getActiveConvId)
    ) return
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
      void startRefsPolling(convId, set, getActiveConvId, shouldKeepPolling, `${reason}:retry_after_error`)
    }
  } finally {
    finishRefsLoadForConversation(convId, controller)
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
  getActiveConvId: () => string | null,
  shouldKeepPolling?: () => boolean,
  reason = 'poll',
) {
  stopRefsPolling()
  const token = ++refsPollToken
  const localePreferenceKey = refsLocalePreferenceKey()
  let tries = 0
  const pollingStartedAt = nowMs()
  let generationSettledAt = 0
  const maxTotalPollingMs = 120_000
  const maxSettledPollingMs = 30_000
  const nextDelay = () => {
    if (tries <= 3) return 500
    if (tries <= 8) return 1000
    if (tries <= 16) return 2000
    return 4000
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
    if (localePreferenceKey !== refsLocalePreferenceKey()) {
      refsPollTimer = null
      return
    }
    if (getActiveConvId() !== convId) {
      refsPollTimer = null
      pushRefsPerf({
        ts: Date.now(),
        convId,
        phase: 'poll_stop',
        token,
        durationMs: 0,
        attempt: tries,
        reason: 'inactive_conversation',
      })
      return
    }
    tries += 1
    const startedAt = nowMs()
    const ctrl = new AbortController()
    refsPollController = ctrl
    try {
      const { data: refs, meta } = await chatApi.getRefsWithMeta(convId, { signal: ctrl.signal })
      if (refsPollController === ctrl) refsPollController = null
      if (token !== refsPollToken) return
      if (localePreferenceKey !== refsLocalePreferenceKey()) return
      if (getActiveConvId() !== convId) {
        refsPollTimer = null
        pushRefsPerf({
          ts: Date.now(),
          convId,
          phase: 'poll_stale',
          token,
          durationMs: Number((nowMs() - startedAt).toFixed(2)),
          attempt: tries,
          reason,
          active: false,
          ...refsBackendPerf(meta),
          summary: summarizeRefsPayload(refs),
        })
        return
      }
      const nextRevision = refsPayloadRevision(refs)
      set((state) => {
        if (refsPayloadRevision(state.refs) === nextRevision) return state
        return {
          refs,
          conversationCacheById: upsertConversationViewCache(state.conversationCacheById, convId, {
            refs,
            cachedAt: Date.now(),
          }),
        }
      })
      const keepPolling = Boolean(shouldKeepPolling?.())
      const needsEnrichment = needsRefsEnrichment(refs)
      const now = nowMs()
      if (keepPolling) {
        generationSettledAt = 0
      } else if (!generationSettledAt) {
        generationSettledAt = now
      }
      const totalPollingExpired = now - pollingStartedAt >= maxTotalPollingMs
      const settledPollingExpired = Boolean(
        generationSettledAt
        && now - generationSettledAt >= maxSettledPollingMs
      )
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
      if (
        (!needsEnrichment && !keepPolling)
        || totalPollingExpired
        || settledPollingExpired
      ) {
        refsPollTimer = null
        pushRefsPerf({
          ts: Date.now(),
          convId,
          phase: 'poll_stop',
          token,
          durationMs: 0,
          attempt: tries,
          reason: totalPollingExpired
            ? 'max_total_wall_time'
            : settledPollingExpired
              ? 'max_settled_wall_time'
              : 'settled',
          needsEnrichment,
          keepPolling,
        })
        return
      }
    } catch (err) {
      if (refsPollController === ctrl) refsPollController = null
      if (
        ctrl.signal.aborted
        || token !== refsPollToken
        || localePreferenceKey !== refsLocalePreferenceKey()
      ) return
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
      if (nowMs() - pollingStartedAt >= maxTotalPollingMs) {
        refsPollTimer = null
        pushRefsPerf({
          ts: Date.now(),
          convId,
          phase: 'poll_stop',
          token,
          durationMs: 0,
          attempt: tries,
          reason: 'max_total_wall_time_after_error',
        })
        return
      }
    }
    if (refsPollController === ctrl) refsPollController = null
    if (token !== refsPollToken || localePreferenceKey !== refsLocalePreferenceKey()) {
      refsPollTimer = null
      return
    }
    if (getActiveConvId() !== convId) {
      refsPollTimer = null
      pushRefsPerf({
        ts: Date.now(),
        convId,
        phase: 'poll_stop',
        token,
        durationMs: 0,
        attempt: tries,
        reason: 'inactive_conversation',
      })
      return
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

function messageCitationRevisionForPostprocess(message: Message | null | undefined): string {
  const packet = getMessageRenderPacketForPostprocess(message)
  if (!message) return ''
  const details = packet && Array.isArray(packet.cite_details)
    ? packet.cite_details
    : Array.isArray(message.cite_details)
      ? message.cite_details
      : []
  return JSON.stringify({
    body: String(
      packet?.rendered_body
      || packet?.rendered_content
      || packet?.answer_markdown
      || message.rendered_body
      || message.rendered_content
      || message.content
      || '',
    ),
    details: details.map((raw) => {
      const detail = raw && typeof raw === 'object' ? raw as Record<string, unknown> : {}
      return {
        num: Number(detail.num || 0),
        displayNum: Number(detail.display_num || 0),
        answerHitNum: Number(detail.answer_hit_num || 0),
        route: String(detail.citation_route || ''),
        anchor: String(detail.anchor || ''),
        sourcePath: String(detail.source_path || ''),
        evidence: String(detail.evidence_quote || detail.summary_line || ''),
        takeaway: String(detail.card_takeaway || ''),
        claim: String(detail.card_claim || detail.answer_claim || ''),
      }
    }),
  })
}

function messageCitationPlanForPostprocess(message: Message | null | undefined): Record<string, unknown> | null {
  const meta = message?.meta && typeof message.meta === 'object'
    ? message.meta as Record<string, unknown>
    : null
  const answerQuality = meta?.answer_quality && typeof meta.answer_quality === 'object'
    ? meta.answer_quality as Record<string, unknown>
    : null
  const contracts = meta?.paper_guide_contracts && typeof meta.paper_guide_contracts === 'object'
    ? meta.paper_guide_contracts as Record<string, unknown>
    : null
  const plan = answerQuality?.citation_plan || contracts?.citation_plan || meta?.citation_plan
  return plan && typeof plan === 'object' ? plan as Record<string, unknown> : null
}

function messageExpectsSystemACitations(message: Message | null | undefined): boolean {
  const plan = messageCitationPlanForPostprocess(message)
  if (!plan) return false
  if (plan.system_a_enabled === true) return true
  if (plan.system_a_enabled === false) return false
  const budget = plan.budget && typeof plan.budget === 'object'
    ? plan.budget as Record<string, unknown>
    : null
  if (Math.max(0, Number(budget?.system_a || 0) || 0) > 0) return true
  const slots = Array.isArray(plan.slots) ? plan.slots : []
  return slots.some((raw) => {
    const slot = raw && typeof raw === 'object' ? raw as Record<string, unknown> : null
    return String(slot?.preferred_system || '').trim().toLowerCase() === 'system_a'
  })
}

function messageExpectsSystemBCitations(message: Message | null | undefined): boolean {
  const plan = messageCitationPlanForPostprocess(message)
  if (!plan) return false
  if (plan.system_b_enabled === true) return true
  if (plan.system_b_enabled === false) return false
  const budget = plan.budget && typeof plan.budget === 'object'
    ? plan.budget as Record<string, unknown>
    : null
  if (Math.max(0, Number(budget?.system_b || 0) || 0) > 0) return true
  const slots = Array.isArray(plan.slots) ? plan.slots : []
  return slots.some((raw) => {
    const slot = raw && typeof raw === 'object' ? raw as Record<string, unknown> : null
    return String(slot?.preferred_system || '').trim().toLowerCase() === 'system_b'
  })
}

function messageFinalRenderedBodyForPostprocess(message: Message | null | undefined): string {
  const packet = getMessageRenderPacketForPostprocess(message)
  if (packet) return String(packet.rendered_body || '').trim()
  return String(message?.rendered_body || '').trim()
}

function messageHasAnyAnchoredCitation(message: Message | null | undefined): boolean {
  const renderedBody = messageFinalRenderedBodyForPostprocess(message)
  if (!renderedBody) return false
  const packet = getMessageRenderPacketForPostprocess(message)
  const details = packet && Array.isArray(packet.cite_details)
    ? packet.cite_details
    : Array.isArray(message?.cite_details) ? message.cite_details : []
  return details.some((raw) => {
    const detail = raw && typeof raw === 'object' ? raw as Record<string, unknown> : null
    const anchor = String(detail?.anchor || '').trim().replace(/^#/, '')
    return Boolean(anchor && renderedBody.includes(`#${anchor}`))
  })
}

function messageHasAnchoredCitations(message: Message | null | undefined): boolean {
  const renderedBody = messageFinalRenderedBodyForPostprocess(message)
  if (!renderedBody) return false
  const packet = getMessageRenderPacketForPostprocess(message)
  const details = packet
    ? (Array.isArray(packet.cite_details) ? packet.cite_details : [])
    : (Array.isArray(message?.cite_details) ? message.cite_details : [])
  const plan = messageCitationPlanForPostprocess(message)
  const requiredCount = String(plan?.coverage_mode || '').trim().toLowerCase() === 'per_entity'
    ? Math.max(1, Number(plan?.coverage_target_count || 0) || 0)
    : 1
  const expectsSystemA = messageExpectsSystemACitations(message)
  const expectsSystemB = messageExpectsSystemBCitations(message)
  let anchoredSystemA = 0
  let anchoredSystemB = 0
  const countedAnchors = new Set<string>()
  for (const raw of details) {
    const detail = raw && typeof raw === 'object' ? raw as Record<string, unknown> : null
    const route = String(detail?.citation_route || '').trim().toLowerCase()
    const anchor = String(detail?.anchor || '').trim().replace(/^#/, '')
    if (!anchor || countedAnchors.has(anchor)) continue
    countedAnchors.add(anchor)
    let occurrenceCount = 0
    occurrenceCount += renderedBody.split(`](#${anchor})`).length - 1
    occurrenceCount += renderedBody.split(`](#${anchor} `).length - 1
    occurrenceCount += renderedBody.split(`href="#${anchor}"`).length - 1
    occurrenceCount += renderedBody.split(`href='#${anchor}'`).length - 1
    if (route === 'system_b') anchoredSystemB += occurrenceCount
    else anchoredSystemA += occurrenceCount
  }
  if (!expectsSystemA && !expectsSystemB) return anchoredSystemA + anchoredSystemB > 0
  return (!expectsSystemA || anchoredSystemA >= requiredCount)
    && (!expectsSystemB || anchoredSystemB >= 1)
}

function refsPackForPostprocessMessage(
  refs: Record<string, unknown> | null | undefined,
  message: Message | null | undefined,
): Record<string, unknown> | null {
  const refsUserMsgId = Number(message?.refs_user_msg_id || 0)
  if (!Number.isFinite(refsUserMsgId) || refsUserMsgId <= 0 || !refs || typeof refs !== 'object') {
    return null
  }
  const pack = refs[String(refsUserMsgId)]
  return pack && typeof pack === 'object' ? pack as Record<string, unknown> : null
}

function refsPackIsSettledForPostprocess(refPack: Record<string, unknown> | null): boolean {
  if (!refPack) return false
  const payloadMode = String(refPack.payload_mode || '').trim().toLowerCase()
  const renderStatus = String(refPack.render_status || '').trim().toLowerCase()
  const displayState = String(refPack.display_state || '').trim().toLowerCase()
  const pending = refPack.enrichment_pending === true
    || refPack.pending === true
    || payloadMode === 'pending'
    || displayState === 'pending'
  const settled = refPack.enrichment_pending === false
    || payloadMode === 'full'
    || renderStatus === 'full'
  return !pending && settled
}

function messageHasExplicitZeroCitationTerminal(
  message: Message | null | undefined,
  refs?: Record<string, unknown>,
): boolean {
  // A citation plan may legitimately end with no links when retrieval found
  // no candidate at all.  Use only the pack for this assistant message's
  // exact user turn; a settled historical pack must not stop current polling.
  const refPack = refsPackForPostprocessMessage(refs, message)
  if (!refPack) return false
  const displayState = String(refPack.display_state || '').trim().toLowerCase()
  const suppressionReason = String(refPack.suppression_reason || '').trim().toLowerCase()
  const pipelineDebug = refPack.pipeline_debug && typeof refPack.pipeline_debug === 'object'
    ? refPack.pipeline_debug as Record<string, unknown>
    : null
  const rawHitCount = Math.max(0, Number(pipelineDebug?.raw_hit_count || 0) || 0)
  return refsPackIsSettledForPostprocess(refPack)
    && displayState === 'empty'
    && suppressionReason === 'no_candidate_hits'
    && rawHitCount === 0
}

function messageNeedsPostprocessRefresh(
  message: Message | null | undefined,
  opts?: { paperGuideMode?: boolean; refs?: Record<string, unknown> },
): boolean {
  if (!message) return true
  if (String(message.role || '').trim().toLowerCase() !== 'assistant') return false
  if (messageHasAnchoredCitations(message)) return false
  if (messageHasExplicitZeroCitationTerminal(message, opts?.refs)) return false
  if (messageExpectsSystemACitations(message) || messageExpectsSystemBCitations(message)) return true
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
  let lastObservedRevision = ''
  let stableRevisionTries = 0
  const maxTries = opts?.paperGuideMode ? 24 : 18
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
    const previousTarget = getState().messages.find((item) => Number(item.id || 0) === msgId) || null
    const previousCitationRevision = messageCitationRevisionForPostprocess(previousTarget)
    try {
      const { page } = await getMessagesPageWithFallback(convId, {
        limit: POSTPROCESS_MESSAGE_PAGE_SIZE,
        renderPacketOnly: opts?.paperGuideMode ? true : undefined,
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
      const nextCitationRevision = messageCitationRevisionForPostprocess(target)
      if (nextCitationRevision && nextCitationRevision !== previousCitationRevision) {
        scheduleLoadRefsForConversation(
          convId,
          set,
          () => getState().activeConvId,
          40,
          undefined,
          'message_postprocess_citations_changed',
        )
      }
      const needsRefresh = messageNeedsPostprocessRefresh(target, {
        ...opts,
        refs: getState().refs,
      })
      if (nextCitationRevision && nextCitationRevision === lastObservedRevision) {
        stableRevisionTries += 1
      } else {
        lastObservedRevision = nextCitationRevision
        stableRevisionTries = 0
      }
      const hasSomeAnchoredCitation = messageHasAnyAnchoredCitation(target)
      const exactRefsSettled = refsPackIsSettledForPostprocess(
        refsPackForPostprocessMessage(getState().refs, target),
      )
      const provenanceStatus = String(
        getMessageProvenanceForPostprocess(target)?.status || '',
      ).trim().toLowerCase()
      const stableTerminal = Boolean(
        needsRefresh
        && tries >= minTries
        && provenanceStatus === 'ready'
        && exactRefsSettled
        && stableRevisionTries >= (hasSomeAnchoredCitation ? 4 : 8)
      )
      if ((!needsRefresh && tries >= minTries) || stableTerminal || tries >= maxTries) {
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
  assistantMsgId?: number
  traceId?: string
  stage: string
  partial: string
  done: boolean
  researchTrace?: Record<string, unknown>
  agentTrace?: Record<string, unknown>
  agentSourceSummary?: Record<string, unknown>
  answerContract?: Record<string, unknown>
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
  generation: GenerationState | null
  sseController: AbortController | null
  uploadItems: ChatUploadItem[]
  pendingImages: ChatImageAttachment[]
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
      generation: patch.generation !== undefined
        ? patch.generation
        : (prev?.generation ?? null),
      sseController: patch.sseController !== undefined
        ? patch.sseController
        : (prev?.sseController ?? null),
      uploadItems: Array.isArray(patch.uploadItems)
        ? patch.uploadItems
        : cachedUploadItems(prev),
      pendingImages: Array.isArray(patch.pendingImages)
        ? patch.pendingImages
        : cachedPendingImages(prev),
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
  refreshActiveConversationLocale: () => Promise<void>
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
    topK: number; temperature: number; maxTokens: number; deepRead: boolean; promptContext?: unknown; queryScope?: QueryScope; agentMode?: boolean
  }) => Promise<void>
  cancelGeneration: () => void
  clearGeneration: () => void
}

function generationCancelUrl(generation: GenerationState) {
  const sessionId = encodeURIComponent(String(generation.sessionId || ''))
  const taskId = encodeURIComponent(String(generation.taskId || ''))
  return `/api/generate/${sessionId}/cancel?task_id=${taskId}`
}

function generationForConversation(state: ChatState, convId: string): GenerationState | null {
  const targetConvId = String(convId || '').trim()
  if (!targetConvId) return null
  const cached = state.conversationCacheById[targetConvId]?.generation ?? null
  if (state.activeConvId === targetConvId) return state.generation || cached
  return cached
}

function generationStartPendingOrRunning(state: ChatState, convId: string): boolean {
  const targetConvId = String(convId || '').trim()
  if (!targetConvId) return false
  return generationStartLocks.has(targetConvId) || Boolean(generationForConversation(state, targetConvId))
}

function cancelConversationGeneration(
  state: ChatState,
  convId: string,
  opts?: {
    refreshMessages?: boolean
    set?: (patch: Partial<ChatState> | ((state: ChatState) => Partial<ChatState>)) => void
  },
) {
  const targetConvId = String(convId || '').trim()
  if (!targetConvId) return
  const cachedView = state.conversationCacheById[targetConvId]
  const active = state.activeConvId === targetConvId
  const generation = active ? (state.generation || cachedView?.generation) : cachedView?.generation
  if (!generation) return
  const ctrl = active ? (state.sseController || cachedView?.sseController) : cachedView?.sseController
  ctrl?.abort()
  api.post(generationCancelUrl(generation))
    .then(() => {
      if (opts?.refreshMessages && opts.set) {
        scheduleRefreshLatestMessagesAfterCancel(targetConvId, generation, opts.set)
      }
    })
    .catch(() => {})
}

function runningUploadJobIdsForConversation(state: ChatState, convId: string) {
  const targetConvId = String(convId || '').trim()
  if (!targetConvId) return []
  const cachedView = state.conversationCacheById[targetConvId]
  const items = state.activeConvId === targetConvId
    ? state.uploadItems
    : cachedUploadItems(cachedView)
  const seen = new Set<string>()
  for (const item of items || []) {
    if (!isPdfUploadJobRunning(item) || !item.ingest_job_id) continue
    const jobId = String(item.ingest_job_id || '').trim()
    if (jobId) seen.add(jobId)
  }
  return [...seen]
}

function cancelConversationUploadJobs(state: ChatState, convId: string) {
  for (const jobId of runningUploadJobIdsForConversation(state, convId)) {
    chatApi.cancelUploadJob(jobId).catch(() => {})
  }
}

function finiteNumber(value: unknown, fallback = 0) {
  const n = Number(value)
  return Number.isFinite(n) ? n : fallback
}

function normalizeProject(raw: unknown): Project | null {
  if (!raw || typeof raw !== 'object') return null
  const rec = raw as Partial<Project> & Record<string, unknown>
  const id = String(rec.id || '').trim()
  if (!id) return null
  return {
    ...rec,
    id,
    name: String(rec.name || '').trim() || 'Untitled project',
    created_at: finiteNumber(rec.created_at),
    updated_at: finiteNumber(rec.updated_at),
  }
}

function normalizeConversation(raw: unknown, fallbackProjectId?: string | null): Conversation | null {
  if (!raw || typeof raw !== 'object') return null
  const rec = raw as Partial<Conversation> & Record<string, unknown>
  const id = String(rec.id || '').trim()
  if (!id) return null
  const projectId = String(rec.project_id ?? '').trim() || String(fallbackProjectId ?? '').trim()
  return {
    ...rec,
    id,
    title: String(rec.title || '').trim() || 'Untitled conversation',
    created_at: finiteNumber(rec.created_at),
    updated_at: finiteNumber(rec.updated_at),
    project_id: projectId || null,
  }
}

function normalizeSidebarSnapshot(snapshot: SidebarSnapshot) {
  const projects: Project[] = []
  const seenProjectIds = new Set<string>()
  for (const rawProject of Array.isArray(snapshot.projects) ? snapshot.projects : []) {
    const project = normalizeProject(rawProject)
    if (!project || seenProjectIds.has(project.id)) continue
    seenProjectIds.add(project.id)
    projects.push(project)
  }

  const projectConversations: Record<string, Conversation[]> = Object.fromEntries(
    projects.map((project) => [project.id, []]),
  )
  const rootConversations: Conversation[] = []
  const seenConversationIds = new Set<string>()
  const addConversation = (rawConversation: unknown, fallbackProjectId?: string | null) => {
    const conversation = normalizeConversation(rawConversation, fallbackProjectId)
    if (!conversation || seenConversationIds.has(conversation.id)) return
    seenConversationIds.add(conversation.id)
    const projectId = String(conversation.project_id || '').trim()
    if (projectId && Object.prototype.hasOwnProperty.call(projectConversations, projectId)) {
      projectConversations[projectId].push({ ...conversation, project_id: projectId })
      return
    }
    rootConversations.push({ ...conversation, project_id: null })
  }

  const rootRaw = Array.isArray(snapshot.root_conversations)
    ? snapshot.root_conversations
    : Array.isArray(snapshot.rootConversations)
      ? snapshot.rootConversations
      : []
  for (const rawConversation of rootRaw) {
    addConversation(rawConversation, null)
  }

  const projectConversationsRaw = snapshot.project_conversations || snapshot.projectConversations || {}
  for (const [projectId, conversations] of Object.entries(projectConversationsRaw)) {
    const normalizedProjectId = String(projectId || '').trim()
    if (!Array.isArray(conversations)) continue
    for (const rawConversation of conversations) {
      addConversation(rawConversation, normalizedProjectId)
    }
  }

  return {
    projects,
    rootConversations,
    projectConversations,
  }
}

async function loadGroupedConversations() {
  try {
    return normalizeSidebarSnapshot(await chatApi.getSidebar(SIDEBAR_CONVERSATION_LIMIT))
  } catch {
    const projects = await chatApi.listProjects()
    const rootConversations = await chatApi.listConversations(SIDEBAR_CONVERSATION_LIMIT, null)
    const groupedEntries = await Promise.all(
      projects.map(async (project) => {
        const conversations = await chatApi.listConversations(SIDEBAR_CONVERSATION_LIMIT, project.id)
        return [project.id, conversations] as const
      }),
    )
    return normalizeSidebarSnapshot({
      projects,
      root_conversations: rootConversations,
      project_conversations: Object.fromEntries(groupedEntries) as Record<string, Conversation[]>,
    })
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
    const grouped = await loadGroupedConversations()
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
      activeConversation: (() => {
        if (!state.activeConvId) return null
        const found = findConversationInLists(grouped.rootConversations, grouped.projectConversations, state.activeConvId)
        if (found) return found
        return state.activeConversation?.id === state.activeConvId ? state.activeConversation : null
      })(),
      projects: grouped.projects,
      projectConversations: grouped.projectConversations,
      rootConversations: grouped.rootConversations,
      activeProjectId:
        state.activeProjectId && grouped.projects.some((project) => project.id === state.activeProjectId)
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
    const deletedProjectId = String(id || '').trim()
    if (!deletedProjectId) return
    await chatApi.deleteProject(deletedProjectId)
    set((state) => rehomeDeletedProjectLocally(state, deletedProjectId))
    await get().loadSidebarData().catch(() => {})
    const activeConvId = get().activeConvId
    if (activeConvId) {
      const conv = await chatApi.getConversation(activeConvId).catch(() => null)
      if (conv) {
        set((current) => ({
          activeProjectId: conv.project_id ?? null,
          activeConversation: current.activeConvId === activeConvId
            ? {
                ...(current.activeConversation || conv),
                ...conv,
                project_id: conv.project_id ?? null,
              }
            : current.activeConversation,
        }))
      } else {
        set((current) => {
          if (String(current.activeConversation?.project_id || '').trim() !== deletedProjectId) {
            return {}
          }
          return {
            activeProjectId: null,
            activeConversation: current.activeConversation
              ? { ...current.activeConversation, project_id: null }
              : current.activeConversation,
          }
        })
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
    localeRefreshSequence += 1
    if (current.activeConvId) invalidateRefsLoadForConversation(current.activeConvId)
    const cachedConv = findConversationInState(current, convId)
    const cacheAfterLeaving = current.activeConvId
      ? upsertConversationViewCache(current.conversationCacheById, current.activeConvId, {
        messages: current.messages,
        refs: current.refs,
        messagesHasMoreBefore: current.messagesHasMoreBefore,
        oldestLoadedMessageId: current.oldestLoadedMessageId,
        generation: current.generation,
        sseController: current.sseController,
        uploadItems: current.uploadItems,
        pendingImages: current.pendingImages,
        cachedAt: Date.now(),
      })
      : current.conversationCacheById
    const cachedView = cacheAfterLeaving[convId]
    const restoredGeneration = cachedView?.generation ?? null
    const restoredUploadItems = cachedUploadItems(cachedView)
    const restoredPendingImages = cachedPendingImages(cachedView)
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
      generation: restoredGeneration,
      sseController: restoredGeneration ? (cachedView?.sseController ?? null) : null,
      refs: cachedView?.refs && typeof cachedView.refs === 'object' ? cachedView.refs : {},
      uploadItems: restoredUploadItems,
      pendingImages: restoredPendingImages,
      conversationCacheById: cacheAfterLeaving,
    })
    if (needsAnyUploadStatusPolling(get())) {
      void startUploadPolling(set, get, upsertConversationViewCache)
    }
    if (restoredGeneration) {
      scheduleLoadRefsForConversation(
        convId,
        set,
        () => get().activeConvId,
        120,
        () => {
          const state = get()
          return state.activeConvId === convId && state.generation?.sessionId === restoredGeneration.sessionId
        },
        'generation_resumed',
      )
    }
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

  refreshActiveConversationLocale: async () => {
    const initial = get()
    const convId = String(initial.activeConvId || '').trim()
    if (!convId) return
    const refreshSequence = ++localeRefreshSequence
    const localePreferenceKey = refsLocalePreferenceKey()
    stopRefsPolling()
    invalidateRefsLoadForConversation(convId)
    const requestIsCurrent = () => (
      refreshSequence === localeRefreshSequence
      && refsLocalePreferenceKey() === localePreferenceKey
      && get().activeConvId === convId
    )
    const paperGuideMode = initial.activeConversation?.mode === 'paper_guide'
      || Boolean(initial.guideBindings?.[convId]?.sourcePath)
    const pageResult = await getMessagesPageWithFallback(convId, {
      limit: MESSAGE_PAGE_SIZE,
      renderPacketOnly: paperGuideMode ? true : undefined,
    })
    if (!requestIsCurrent()) return
    const page = pageResult.page
    set((state) => {
      if (!requestIsCurrent() || state.activeConvId !== convId) return {}
      return {
        messages: Array.isArray(page?.messages) ? page.messages : [],
        messagesHasMoreBefore: Boolean(page?.has_more_before),
        oldestLoadedMessageId: Number.isFinite(Number(page?.oldest_loaded_id))
          ? Number(page?.oldest_loaded_id)
          : null,
        conversationCacheById: upsertConversationViewCache(state.conversationCacheById, convId, {
          messages: Array.isArray(page?.messages) ? page.messages : [],
          refs: {},
          messagesHasMoreBefore: Boolean(page?.has_more_before),
          oldestLoadedMessageId: Number.isFinite(Number(page?.oldest_loaded_id))
            ? Number(page?.oldest_loaded_id)
            : null,
          cachedAt: Date.now(),
        }),
        refs: {},
      }
    })
    if (!requestIsCurrent()) return
    await loadRefsForConversation(
      convId,
      set,
      () => get().activeConvId,
      undefined,
      'locale_change',
    )
  },

  createConversation: async () => {
    const projectId = get().activeProjectId
    const locale = useSettingsStore.getState().uiLocale
    const defaultTitle = buildDefaultConversationTitle(locale)
    const { id } = await chatApi.createConversation(defaultTitle, projectId)
    await get().loadSidebarData()
    await get().selectConversation(id)
    return id
  },

  createPaperGuideConversation: async (opts) => {
    const sourcePath = String(opts.sourcePath || '').trim()
    if (!sourcePath) throw new Error('sourcePath required')
    const locale = useSettingsStore.getState().uiLocale
    const sourceName = String(opts.sourceName || '').trim() || basenameFromSourcePath(sourcePath) || (locale === 'zh' ? zh.default_source_fallback : en.default_source_fallback)
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
    set((state) => ({
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
    set((state) => ({
      activeConversation: state.activeConversation?.id === id
        ? { ...state.activeConversation, title: nextTitle }
        : state.activeConversation,
      rootConversations: patchConversationTitle(state.rootConversations, id, nextTitle),
      projectConversations: patchProjectConversationTitle(state.projectConversations, id, nextTitle),
    }))
    await get().loadSidebarData()
  },

  deleteConversation: async (id) => {
    const targetId = String(id || '').trim()
    if (!targetId) return
    const wasActive = get().activeConvId === targetId
    if (wasActive) {
      conversationSwitchToken += 1
      stopRefsPolling()
      stopMessagePostprocessPolling()
      stopUploadPolling()
    }
    const beforeDelete = get()
    cancelConversationGeneration(beforeDelete, targetId)
    cancelConversationUploadJobs(beforeDelete, targetId)
    await chatApi.deleteConversation(targetId)
    set((cur) => {
      const nextBindings = { ...(cur.guideBindings || {}) }
      const nextCache = { ...(cur.conversationCacheById || {}) }
      delete nextBindings[targetId]
      delete nextCache[targetId]
      if (cur.activeConvId === targetId) {
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
          sseController: null,
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
    if (!needsAnyUploadStatusPolling(get())) {
      stopUploadPolling()
    }
    if (wasActive) {
      set((cur) => {
        if (cur.activeConvId && cur.activeConvId !== targetId) return {}
        const nextBindings = { ...(cur.guideBindings || {}) }
        const nextCache = { ...(cur.conversationCacheById || {}) }
        delete nextBindings[targetId]
        delete nextCache[targetId]
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
          sseController: null,
          uploadItems: [],
          pendingImages: [],
          guideBindings: nextBindings,
          conversationCacheById: nextCache,
        }
      })
    }
  },

  moveConversation: async (convId, projectId) => {
    const targetConvId = String(convId || '').trim()
    const targetProjectId = String(projectId || '').trim() || null
    if (!targetConvId) return
    await chatApi.updateConversationProject(targetConvId, targetProjectId)
    set((state) => moveConversationLocally(state, targetConvId, targetProjectId))
    await get().loadSidebarData().catch(() => {})
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
    let convId = String(opts?.convId || '').trim()
    try {
      const hasPdf = files.some((file) => String(file.name || '').toLowerCase().endsWith('.pdf') || String(file.type || '').toLowerCase() === 'application/pdf')
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
        ...(() => {
          const targetConvId = String(convId || '').trim()
          const visibleForUpload = !targetConvId || state.activeConvId === targetConvId
          const previousCache = targetConvId ? state.conversationCacheById[targetConvId] : undefined
          const baseUploadItems = visibleForUpload ? state.uploadItems : cachedUploadItems(previousCache)
          const basePendingImages = visibleForUpload ? state.pendingImages : cachedPendingImages(previousCache)
          const nextUploadItems = mergeUploadItems(baseUploadItems, res.items || [])
          const nextPendingImages = mergeImageAttachments(basePendingImages, imageAttachments)
          return {
            ...(visibleForUpload ? { uploadItems: nextUploadItems, pendingImages: nextPendingImages } : {}),
            conversationCacheById: targetConvId
              ? upsertConversationViewCache(state.conversationCacheById, targetConvId, {
                uploadItems: nextUploadItems,
                pendingImages: nextPendingImages,
                cachedAt: Date.now(),
              })
              : state.conversationCacheById,
          }
        })(),
      }))
      if (needsAnyUploadStatusPolling(get())) {
        void startUploadPolling(set, get, upsertConversationViewCache)
      }
    } catch {
      set((state) => ({
        uploading: false,
        ...(() => {
          const targetConvId = String(convId || '').trim()
          const visibleForUpload = !targetConvId || state.activeConvId === targetConvId
          const previousCache = targetConvId ? state.conversationCacheById[targetConvId] : undefined
          const baseUploadItems = visibleForUpload ? state.uploadItems : cachedUploadItems(previousCache)
          const nextUploadItems = mergeUploadItems(baseUploadItems, [{
            kind: 'unknown',
            status: 'error',
            name: 'upload',
            error: 'upload failed',
          }])
          return {
            ...(visibleForUpload ? { uploadItems: nextUploadItems } : {}),
            conversationCacheById: targetConvId
              ? upsertConversationViewCache(state.conversationCacheById, targetConvId, {
                uploadItems: nextUploadItems,
                pendingImages: visibleForUpload ? state.pendingImages : cachedPendingImages(previousCache),
                cachedAt: Date.now(),
              })
              : state.conversationCacheById,
          }
        })(),
      }))
      throw new Error('upload failed')
    }
  },

  retryUploadItem: async (key) => {
    const current = get().uploadItems.find((item) => uploadItemKey(item) === key)
    if (!current || current.kind !== 'pdf' || !current.ingest_job_id) return
    const convId = String(get().activeConvId || '').trim()
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
      ...(() => {
        const nextUploadItems = mergeUploadItems(
          state.uploadItems.filter((item) => uploadItemKey(item) !== key),
          nextItem ? [nextItem] : [],
        )
        return {
          uploadItems: nextUploadItems,
          conversationCacheById: convId
            ? upsertConversationViewCache(state.conversationCacheById, convId, {
              uploadItems: nextUploadItems,
              pendingImages: state.pendingImages,
              cachedAt: Date.now(),
            })
            : state.conversationCacheById,
        }
      })(),
    }))
    if (needsAnyUploadStatusPolling(get())) {
      void startUploadPolling(set, get, upsertConversationViewCache)
    }
  },

  cancelUploadItem: async (key) => {
    const current = get().uploadItems.find((item) => uploadItemKey(item) === key)
    if (!current || current.kind !== 'pdf' || !current.ingest_job_id) return
    const convId = String(get().activeConvId || '').trim()
    const res = await chatApi.cancelUploadJob(current.ingest_job_id)
    const nextItem = res.item
    set((state) => ({
      ...(() => {
        const nextUploadItems = mergeUploadItems(state.uploadItems, nextItem ? [nextItem] : [])
        return {
          uploadItems: nextUploadItems,
          conversationCacheById: convId
            ? upsertConversationViewCache(state.conversationCacheById, convId, {
              uploadItems: nextUploadItems,
              pendingImages: state.pendingImages,
              cachedAt: Date.now(),
            })
            : state.conversationCacheById,
        }
      })(),
    }))
    if (!needsAnyUploadStatusPolling(get())) {
      stopUploadPolling()
    }
  },

  removePendingImage: (key) => {
    const convId = String(get().activeConvId || '').trim()
    set((state) => ({
      ...(() => {
        const nextPendingImages = state.pendingImages.filter((item) => attachmentKey(item) !== key)
        const nextUploadItems = state.uploadItems.filter((item) => {
          const attachment = item.attachment
          return !attachment || attachmentKey(attachment) !== key
        })
        return {
          pendingImages: nextPendingImages,
          uploadItems: nextUploadItems,
          conversationCacheById: convId
            ? upsertConversationViewCache(state.conversationCacheById, convId, {
              uploadItems: nextUploadItems,
              pendingImages: nextPendingImages,
              cachedAt: Date.now(),
            })
            : state.conversationCacheById,
        }
      })(),
    }))
  },

  dismissUploadItem: (key) => {
    const convId = String(get().activeConvId || '').trim()
    set((state) => ({
      ...(() => {
        const nextUploadItems = state.uploadItems.filter((item) => uploadItemKey(item) !== key)
        return {
          uploadItems: nextUploadItems,
          conversationCacheById: convId
            ? upsertConversationViewCache(state.conversationCacheById, convId, {
              uploadItems: nextUploadItems,
              pendingImages: state.pendingImages,
              cachedAt: Date.now(),
            })
            : state.conversationCacheById,
        }
      })(),
    }))
    if (!needsAnyUploadStatusPolling(get())) {
      stopUploadPolling()
    }
  },

  sendMessage: async (prompt, opts) => {
    stopMessagePostprocessPolling()
    let convId = get().activeConvId
    let provisionalNewConversationLock = false
    if (!convId) {
      if (generationStartLocks.has(NEW_CONVERSATION_GENERATION_LOCK)) return
      generationStartLocks.add(NEW_CONVERSATION_GENERATION_LOCK)
      provisionalNewConversationLock = true
      try {
        convId = await get().createConversation()
      } catch (err) {
        generationStartLocks.delete(NEW_CONVERSATION_GENERATION_LOCK)
        throw err
      }
    }
    convId = String(convId || '').trim()
    if (!convId) {
      if (provisionalNewConversationLock) generationStartLocks.delete(NEW_CONVERSATION_GENERATION_LOCK)
      return
    }

    const stateNow = get()
    if (generationStartPendingOrRunning(stateNow, convId)) {
      if (provisionalNewConversationLock) generationStartLocks.delete(NEW_CONVERSATION_GENERATION_LOCK)
      return
    }
    generationStartLocks.add(convId)
    if (provisionalNewConversationLock) {
      generationStartLocks.delete(NEW_CONVERSATION_GENERATION_LOCK)
      provisionalNewConversationLock = false
    }
    const releaseGenerationStartLock = () => {
      generationStartLocks.delete(convId!)
      if (provisionalNewConversationLock) generationStartLocks.delete(NEW_CONVERSATION_GENERATION_LOCK)
    }
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
    const sentImageKeys = new Set(pendingImages.map((item) => attachmentKey(item)))
    const requestPaperGuideMode = Boolean(
      stateNow.activeConversation?.mode === 'paper_guide'
      || localGuide?.sourcePath
      || boundSourcePath
    )

    let res: {
      session_id: string
      task_id: string
      trace_id?: string
      user_msg_id: number
      assistant_msg_id: number
      conversation_title?: string
      started?: boolean
      start_error?: string
    }
    try {
      res = await api.post<typeof res>('/api/generate', {
        conv_id: convId,
        prompt: trimmedPrompt,
        image_attachments: pendingImages,
        preferred_sources: preferredSourcesFinal,
        source_lock_path: boundSourcePath,
        source_lock_name: boundSourceName,
        query_scope: opts.queryScope || undefined,
        top_k: opts.topK,
        temperature: opts.temperature,
        max_tokens: opts.maxTokens,
        deep_read: opts.deepRead,
        agent_mode: opts.agentMode || undefined,
        prompt_context: opts.promptContext || undefined,
      })
    } catch (err) {
      releaseGenerationStartLock()
      throw err
    }
    const conversationTitle = String(res.conversation_title || '').trim()
    const uiLocale = useSettingsStore.getState().uiLocale
    const generationStarted = res.started !== false
    const generationStartError = String(res.start_error || '').trim()
    const startFailureMessage = generationStartFailureDisplayMessage(generationStartError, uiLocale)
    const userMessage: Message = {
      id: res.user_msg_id,
      role: 'user',
      content: userStoreText,
      created_at: Date.now() / 1000,
      attachments: pendingImages,
      meta: {
        ...(opts.promptContext ? { prompt_context: opts.promptContext } : {}),
        ...(opts.queryScope ? { query_scope: opts.queryScope } : {}),
        ...(opts.agentMode ? { agent_mode: 'research_agent', agent_mode_requested: true } : {}),
      },
    }
    const startFailureAssistantMessage: Message | null = generationStarted
      ? null
      : {
          id: res.assistant_msg_id,
          role: 'assistant',
          content: startFailureMessage,
          created_at: Date.now() / 1000,
          meta: {
            trace_id: res.trace_id,
            ...(opts.agentMode ? { agent_mode: 'research_agent' } : {}),
          },
        }
    const currentGeneration = {
      sessionId: res.session_id,
      taskId: res.task_id,
      assistantMsgId: Number(res.assistant_msg_id || 0) || undefined,
      traceId: res.trace_id,
      stage: 'starting',
      partial: '',
      done: false,
    }
    const ctrl = new AbortController()
    const shouldKeepPollingRefs = () => {
      const state = get()
      return state.activeConvId === convId && Boolean(state.generation)
    }
    const requestGenerationIsCurrent = () => {
      const state = get()
      const cachedGeneration = convId ? state.conversationCacheById[convId]?.generation : null
      const candidate = state.activeConvId === convId ? state.generation : cachedGeneration
      return candidate?.sessionId === res.session_id
    }

    set((state) => {
      const visibleForRequest = state.activeConvId === convId
      const previousCache = convId ? state.conversationCacheById[convId] : undefined
      const baseMessages = visibleForRequest
        ? state.messages
        : (Array.isArray(previousCache?.messages) ? previousCache.messages : [])
      const nextMessages = startFailureAssistantMessage
        ? [...baseMessages, userMessage, startFailureAssistantMessage]
        : [...baseMessages, userMessage]
      const basePendingImages = visibleForRequest ? state.pendingImages : cachedPendingImages(previousCache)
      const baseUploadItems = visibleForRequest ? state.uploadItems : cachedUploadItems(previousCache)
      const nextPendingImages = sentImageKeys.size > 0
        ? basePendingImages.filter((item) => !sentImageKeys.has(attachmentKey(item)))
        : basePendingImages
      const nextUploadItems = sentImageKeys.size > 0
        ? baseUploadItems.filter((item) => {
          if (item.kind !== 'image') return true
          const attachment = item.attachment
          return !attachment || !sentImageKeys.has(attachmentKey(attachment))
        })
        : baseUploadItems
      const nextCache = convId
        ? upsertConversationViewCache(state.conversationCacheById, convId, {
          messages: nextMessages,
          refs: visibleForRequest
            ? state.refs
            : (previousCache?.refs && typeof previousCache.refs === 'object' ? previousCache.refs : {}),
          messagesHasMoreBefore: visibleForRequest
            ? state.messagesHasMoreBefore
            : Boolean(previousCache?.messagesHasMoreBefore),
          oldestLoadedMessageId: visibleForRequest
            ? state.oldestLoadedMessageId
            : (previousCache?.oldestLoadedMessageId ?? null),
          generation: generationStarted ? currentGeneration : null,
          sseController: generationStarted ? ctrl : null,
          uploadItems: nextUploadItems,
          pendingImages: nextPendingImages,
          cachedAt: Date.now(),
        })
        : state.conversationCacheById
      return {
        ...(visibleForRequest
          ? {
              messages: nextMessages,
              conversationLoading: false,
              generation: generationStarted ? currentGeneration : null,
              sseController: generationStarted ? ctrl : null,
              uploadItems: nextUploadItems,
              pendingImages: nextPendingImages,
            }
          : {}),
        activeConversation: conversationTitle && state.activeConversation?.id === convId
          ? { ...state.activeConversation, title: conversationTitle }
          : state.activeConversation,
        rootConversations: conversationTitle
          ? patchConversationTitle(state.rootConversations, convId!, conversationTitle)
          : state.rootConversations,
        projectConversations: conversationTitle
          ? patchProjectConversationTitle(state.projectConversations, convId!, conversationTitle)
          : state.projectConversations,
        conversationCacheById: nextCache,
      }
    })
    releaseGenerationStartLock()

    if (!generationStarted) {
      try {
        const { page } = await getMessagesPageWithFallback(convId!, {
          limit: MESSAGE_PAGE_SIZE,
          renderPacketOnly: requestPaperGuideMode ? true : undefined,
        })
        const localizedPage = localizeGenerationStartFailurePage(
          page,
          Number(res.assistant_msg_id || 0),
          startFailureMessage,
        )
        set((state) => {
          const visibleForRequest = state.activeConvId === convId
          const previousCache = state.conversationCacheById[convId!]
          const baseMessages = visibleForRequest
            ? state.messages
            : (Array.isArray(previousCache?.messages) ? previousCache.messages : [])
          const baseHasMoreBefore = visibleForRequest
            ? state.messagesHasMoreBefore
            : Boolean(previousCache?.messagesHasMoreBefore)
          const merged = mergeLatestMessagePage(baseMessages, baseHasMoreBefore, localizedPage)
          const nextCache = upsertConversationViewCache(state.conversationCacheById, convId!, {
            messages: merged.messages,
            refs: visibleForRequest
              ? state.refs
              : (previousCache?.refs && typeof previousCache.refs === 'object' ? previousCache.refs : {}),
            messagesHasMoreBefore: merged.hasMoreBefore,
            oldestLoadedMessageId: merged.oldestLoadedMessageId,
            generation: null,
            sseController: null,
            cachedAt: Date.now(),
          })
          if (!visibleForRequest) {
            return { conversationCacheById: nextCache }
          }
          return {
            messages: merged.messages,
            generation: null,
            sseController: null,
            conversationLoading: false,
            messagesLoadingMore: false,
            messagesHasMoreBefore: merged.hasMoreBefore,
            oldestLoadedMessageId: merged.oldestLoadedMessageId,
            conversationCacheById: nextCache,
          }
        })
      } catch {
        set((state) => {
          const nextCache = convId
            ? upsertConversationViewCache(state.conversationCacheById, convId, {
              generation: null,
              sseController: null,
              cachedAt: Date.now(),
            })
            : state.conversationCacheById
          return {
            ...(state.activeConvId === convId ? { generation: null, sseController: null, conversationLoading: false } : {}),
            ...(nextCache !== state.conversationCacheById ? { conversationCacheById: nextCache } : {}),
          }
        })
      }
      await get().loadSidebarData().catch(() => {})
      throw new Error(startFailureMessage)
    }
    scheduleLoadRefsForConversation(
      convId,
      set,
      () => get().activeConvId,
      350,
      shouldKeepPollingRefs,
      'generation_started',
    )

    try {
      const sseRes = await authFetch(`/api/generate/${res.session_id}/stream`, {
        signal: ctrl.signal,
      })
      if (!sseRes.ok) {
        throw await responseError(sseRes)
      }
      if (!sseRes.body) {
        throw new Error('Generation stream did not return a readable body.')
      }
      const reader = sseRes.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''
      let streamDone = false

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop() || ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          let data: Record<string, unknown>
          try {
            data = JSON.parse(line.slice(6)) as Record<string, unknown>
          } catch {
            // ignore malformed SSE chunks
            continue
          }
          const parsedResearchTrace = data.research_trace && typeof data.research_trace === 'object'
            ? data.research_trace as Record<string, unknown>
            : undefined
          const parsedAgentTrace = data.agent_trace && typeof data.agent_trace === 'object'
            ? data.agent_trace as Record<string, unknown>
            : undefined
          const parsedAgentSourceSummary = data.agent_source_summary && typeof data.agent_source_summary === 'object'
            ? data.agent_source_summary as Record<string, unknown>
            : undefined
          const parsedAnswerContract = data.answer_contract && typeof data.answer_contract === 'object'
            ? data.answer_contract as Record<string, unknown>
            : undefined
          const nextGeneration: GenerationState = {
            sessionId: res.session_id,
            taskId: res.task_id,
            assistantMsgId: Number(res.assistant_msg_id || 0) || undefined,
            traceId: res.trace_id,
            stage: String(data.stage || ''),
            partial: String(data.partial || ''),
            done: !!data.done,
            researchTrace: parsedResearchTrace,
            agentTrace: parsedAgentTrace,
            agentSourceSummary: parsedAgentSourceSummary,
            answerContract: parsedAnswerContract,
          }
          set((state) => {
            const cachedGeneration = convId ? state.conversationCacheById[convId]?.generation : null
            const candidate = state.activeConvId === convId ? state.generation : cachedGeneration
            if (candidate?.sessionId !== res.session_id) return {}
            const generationWithTrace: GenerationState = {
              ...nextGeneration,
              researchTrace: parsedResearchTrace || candidate.researchTrace,
              agentTrace: parsedAgentTrace || candidate.agentTrace,
              agentSourceSummary: parsedAgentSourceSummary || candidate.agentSourceSummary,
              answerContract: parsedAnswerContract || candidate.answerContract,
            }
            const nextCache = convId
              ? upsertConversationViewCache(state.conversationCacheById, convId, {
                generation: generationWithTrace,
                sseController: ctrl,
                cachedAt: Date.now(),
              })
              : state.conversationCacheById
            if (state.activeConvId !== convId || state.generation?.sessionId !== res.session_id) {
              return { conversationCacheById: nextCache }
            }
            return {
              generation: generationWithTrace,
              conversationCacheById: nextCache,
            }
          })
          const terminalStatus = String(data.status || data.stage || '').trim().toLowerCase()
          if (data.done && terminalStatus === 'error') {
            streamDone = true
            throw new Error(String(
              data.error
              || data.partial
              || data.answer
              || localizedGenerationStreamFailedMessage(uiLocale),
            ))
          }
          if (data.done) {
            streamDone = true
            if (!requestGenerationIsCurrent()) return
            const finalContent = String(data.answer || data.partial || nextGeneration.partial || '').trim()
            let visibleAtCompletion = false
            set((state) => {
              const visibleForRequest = state.activeConvId === convId && (
                state.generation?.sessionId === res.session_id
                || state.conversationCacheById[convId!]?.generation?.sessionId === res.session_id
              )
              visibleAtCompletion = visibleForRequest
              const previousCache = state.conversationCacheById[convId!]
              const baseMessages = visibleForRequest
                ? state.messages
                : (Array.isArray(previousCache?.messages) ? previousCache.messages : [])
              const completedGeneration = visibleForRequest
                ? state.generation
                : previousCache?.generation
              const completedMessages = upsertGenerationCompletedMessage(
                baseMessages,
                completedGeneration,
                finalContent,
              )
              const nextCache = upsertConversationViewCache(state.conversationCacheById, convId!, {
                messages: completedMessages,
                refs: visibleForRequest
                  ? state.refs
                  : (previousCache?.refs && typeof previousCache.refs === 'object' ? previousCache.refs : {}),
                generation: null,
                sseController: null,
                cachedAt: Date.now(),
              })
              if (!visibleForRequest) {
                return { conversationCacheById: nextCache }
              }
              return {
                messages: completedMessages,
                generation: null,
                conversationLoading: false,
                messagesLoadingMore: false,
                sseController: state.sseController === ctrl ? null : state.sseController,
                conversationCacheById: nextCache,
              }
            })
            if (visibleAtCompletion) {
              scheduleLoadRefsForConversation(convId!, set, () => get().activeConvId, 120, undefined, 'generation_done')
            }

            // The completed answer is already visible. Hydrating server-side
            // metadata and post-processing must never keep the streaming UI
            // open or turn a successful generation into a visible failure.
            let page: MessagePage | null = null
            try {
              const result = await getMessagesPageWithFallback(convId!, {
                // Only the just-created user/assistant pair is needed here.
                // Rendering a full page delays citation hydration in long
                // conversations even though the streamed answer is complete.
                limit: 2,
                renderPacketOnly: requestPaperGuideMode ? true : undefined,
              })
              page = result.page
            } catch {
              page = null
            }
            if (page) set((state) => {
              const visibleForRequest = state.activeConvId === convId
              const previousCache = state.conversationCacheById[convId!]
              const currentRequestGeneration = visibleForRequest
                ? state.generation
                : previousCache?.generation
              const stillOwnsGenerationState = !currentRequestGeneration
                || currentRequestGeneration.sessionId === res.session_id
              // A newer request may have started in the same conversation
              // while this lightweight hydration was in flight. Its page can
              // only describe the completed request, so do not merge it into
              // or clear any state owned by the newer generation.
              if (!stillOwnsGenerationState) return {}
              const baseMessages = visibleForRequest
                ? state.messages
                : (Array.isArray(previousCache?.messages) ? previousCache.messages : [])
              const baseHasMoreBefore = visibleForRequest
                ? state.messagesHasMoreBefore
                : Boolean(previousCache?.messagesHasMoreBefore)
              const merged = mergeLatestMessagePage(
                baseMessages,
                baseHasMoreBefore,
                page,
              )
              const nextCache = upsertConversationViewCache(state.conversationCacheById, convId!, {
                messages: merged.messages,
                refs: visibleForRequest
                  ? state.refs
                  : (previousCache?.refs && typeof previousCache.refs === 'object' ? previousCache.refs : {}),
                messagesHasMoreBefore: merged.hasMoreBefore,
                oldestLoadedMessageId: merged.oldestLoadedMessageId,
                generation: null,
                sseController: null,
                cachedAt: Date.now(),
              })
              if (!visibleForRequest) {
                return { conversationCacheById: nextCache }
              }
              return {
                messages: merged.messages,
                conversationLoading: false,
                messagesLoadingMore: false,
                messagesHasMoreBefore: merged.hasMoreBefore,
                oldestLoadedMessageId: merged.oldestLoadedMessageId,
                generation: null,
                sseController: null,
                conversationCacheById: nextCache,
              }
            })
            if (page && visibleAtCompletion) {
              scheduleLoadRefsForConversation(
                convId!,
                set,
                () => get().activeConvId,
                40,
                undefined,
                'post_generation_message_hydration',
              )
            }
            const postprocessState = get()
            if (!visibleAtCompletion || postprocessState.activeConvId !== convId) {
              void get().loadSidebarData()
              return
            }
            const paperGuideMode = Boolean(
              requestPaperGuideMode
              || postprocessState.activeConversation?.mode === 'paper_guide'
              || postprocessState.guideBindings?.[convId!]?.sourcePath
              || boundSourcePath,
            )
            const assistantMessage = postprocessState.messages.find(
              (item) => Number(item.id || 0) === Number(res.assistant_msg_id || 0),
            ) || null
            if (paperGuideMode || messageNeedsPostprocessRefresh(assistantMessage, {
              paperGuideMode,
              refs: postprocessState.refs,
            })) {
              void startMessagePostprocessPolling(
                convId!,
                res.assistant_msg_id,
                set,
                get,
                { paperGuideMode, reason: 'generation_done' },
              )
            }
            void get().loadSidebarData()
            return
          }
        }
      }
      if (!streamDone && !ctrl.signal.aborted) {
        throw new Error('Generation stream ended before completion.')
      }
    } catch (err) {
      if (ctrl.signal.aborted) return
      const displayMessage = generationStreamFailureDisplayMessage(err, uiLocale)
      set((state) => {
        const activeMatches = state.activeConvId === convId && state.generation?.sessionId === res.session_id
        const previousCache = convId ? state.conversationCacheById[convId] : undefined
        const cachedMatches = Boolean(convId && previousCache?.generation?.sessionId === res.session_id)
        const shouldPersistFailure = activeMatches || cachedMatches
        const failureGeneration = activeMatches ? state.generation : previousCache?.generation
        const baseMessages = activeMatches
          ? state.messages
          : (Array.isArray(previousCache?.messages) ? previousCache.messages : [])
        const nextMessages = shouldPersistFailure
          ? upsertGenerationFailureMessage(baseMessages, failureGeneration, displayMessage)
          : baseMessages
        const nextCache = shouldPersistFailure && convId
          ? upsertConversationViewCache(state.conversationCacheById, convId, {
            messages: nextMessages,
            generation: null,
            sseController: null,
            cachedAt: Date.now(),
          })
          : state.conversationCacheById
        return {
          ...(activeMatches ? { messages: nextMessages, generation: null, conversationLoading: false } : {}),
          ...(state.sseController === ctrl ? { sseController: null } : {}),
          ...(nextCache !== state.conversationCacheById ? { conversationCacheById: nextCache } : {}),
        }
      })
      throw new Error(displayMessage)
    } finally {
      set((state) => {
        const previousCache = convId ? state.conversationCacheById[convId] : undefined
        const shouldClearCachedController = previousCache?.sseController === ctrl
        const nextCache = shouldClearCachedController && convId
          ? upsertConversationViewCache(state.conversationCacheById, convId, { sseController: null })
          : state.conversationCacheById
        return {
          ...(state.sseController === ctrl ? { sseController: null } : {}),
          ...(nextCache !== state.conversationCacheById ? { conversationCacheById: nextCache } : {}),
        }
      })
    }
  },

  cancelGeneration: () => {
    const state = get()
    if (state.activeConvId) {
      cancelConversationGeneration(state, state.activeConvId, {
        refreshMessages: true,
        set,
      })
    }
    stopRefsPolling()
    stopMessagePostprocessPolling()
    set((current) => {
      const convId = current.activeConvId
      const nextCache = convId
        ? upsertConversationViewCache(current.conversationCacheById, convId, {
          generation: null,
          sseController: null,
        })
        : current.conversationCacheById
      return { generation: null, sseController: null, conversationCacheById: nextCache }
    })
  },

  clearGeneration: () => set((state) => {
    const convId = state.activeConvId
    const nextCache = convId
      ? upsertConversationViewCache(state.conversationCacheById, convId, {
        generation: null,
        sseController: null,
      })
      : state.conversationCacheById
    return { generation: null, conversationCacheById: nextCache }
  }),
}))
