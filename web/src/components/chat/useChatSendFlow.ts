import { useCallback } from 'react'
import { message } from 'antd'
import type { QueryScope } from '../../api/chat'
import { useChatStore } from '../../stores/chatStore'
import { useSettingsStore } from '../../stores/settingsStore'
import { reportUserIssue } from '../../userIssueReporter'
import type { ApiSettingsTarget } from '../layout/settingsEvents'
import type { ResearchRuntimeContext } from './researchContext'
import type { SelectedResearchContextPack } from './researchContextPack'

export function resolveQueryScope(scope: QueryScope, opts: { hasCurrentPaper: boolean; hasBasket: boolean }): QueryScope {
  if (scope === 'current_paper' && !opts.hasCurrentPaper) return 'library'
  if (scope === 'basket' && !opts.hasBasket) return opts.hasCurrentPaper ? 'current_paper' : 'library'
  return scope
}

function isModelConnectionError(err: unknown) {
  const text = err instanceof Error ? err.message : String(err || '')
  return /api key|authentication|unauthorized|forbidden|401|403|connection|network|timeout|timed out|base_url|model/i.test(text)
}

function chatSendFailureKind(messageText: string, labels: Record<string, string>) {
  const text = String(messageText || '').trim()
  const low = text.toLowerCase()
  if (text === labels.chat_generation_start_failed) return 'generation_start_failed'
  if (text === labels.chat_generation_stream_failed) return 'generation_stream_failed'
  if (text === labels.chat_generation_stream_incomplete) return 'generation_stream_incomplete'
  if (text === labels.chat_generation_refresh_failed) return 'generation_refresh_failed'
  if (/generation_start_failed|未能启动/.test(text)) return 'generation_start_failed'
  if (/interrupted before completion|尚未完成|ended before completion/.test(low) || /尚未完成|中断/.test(text)) return 'generation_stream_incomplete'
  if (/stream failed|stream temporarily unavailable|readable body|回答连接/.test(low) || /回答连接/.test(text)) return 'generation_stream_failed'
  if (/latest message|messages page|messages fallback|最新消息/.test(low) || /最新消息/.test(text)) return 'generation_refresh_failed'
  if (/401|403|api key|authentication|unauthorized|forbidden|connection|network|timeout|base_url|model/i.test(text)) return 'model_connection'
  return 'chat_send_failed'
}

function httpStatusFromError(messageText: string) {
  const match = String(messageText || '').trim().match(/^(\d{3})\b/)
  return match ? Number(match[1]) : 0
}

export function useChatSendFlow({
  labels,
  researchContext,
  queryScope,
  selectedResearchContext,
  agentMode,
  onOpenApiSettings,
  onSelectedResearchContextConsumed,
}: {
  labels: Record<string, string>
  researchContext: ResearchRuntimeContext
  queryScope: QueryScope
  selectedResearchContext?: SelectedResearchContextPack | null
  agentMode: boolean
  onOpenApiSettings: (target: ApiSettingsTarget | '') => void
  onSelectedResearchContextConsumed?: (packId: string) => void
}) {
  const sendMessage = useChatStore((s) => s.sendMessage)
  const activeConvId = useChatStore((s) => s.activeConvId)
  const activeProjectId = useChatStore((s) => s.activeProjectId)
  const activeConversation = useChatStore((s) => s.activeConversation)
  const guideBindings = useChatStore((s) => s.guideBindings)
  const messages = useChatStore((s) => s.messages)
  const pendingImages = useChatStore((s) => s.pendingImages)
  const uploadItems = useChatStore((s) => s.uploadItems)
  const topK = useSettingsStore((s) => s.topK)
  const temperature = useSettingsStore((s) => s.temperature)
  const maxTokens = useSettingsStore((s) => s.maxTokens)
  const uiLocale = useSettingsStore((s) => s.uiLocale)
  const refreshReadiness = useSettingsStore((s) => s.refreshReadiness)

  return useCallback((text: string) => {
    if (researchContext.api.sendBlockTarget === 'text') {
      message.warning(labels.chat_api_missing_toast)
      onOpenApiSettings('text')
      return
    }
    if (researchContext.api.sendBlockTarget === 'vision') {
      message.warning(labels.chat_vision_api_missing_toast)
      onOpenApiSettings('vision')
      return
    }

    const hasCurrentPaper = Boolean(researchContext.activeSource.ready)
    const hasBasket = Boolean(selectedResearchContext?.items?.length)
    const resolvedScope = resolveQueryScope(queryScope, { hasCurrentPaper, hasBasket })
    const contextPackForSend = resolvedScope === 'basket' ? selectedResearchContext : null
    void sendMessage(text, {
      topK,
      temperature,
      maxTokens,
      deepRead: true,
      promptContext: contextPackForSend,
      queryScope: resolvedScope,
      agentMode,
    }).then(() => {
      if (!contextPackForSend) return
      onSelectedResearchContextConsumed?.(contextPackForSend.id)
    }).catch((err: unknown) => {
      const fallback = err instanceof Error ? err.message : String(err || '')
      const failureKind = chatSendFailureKind(fallback, labels)
      reportUserIssue({
        source: 'frontend',
        domain: 'chat_generation',
        severity: 'error',
        summary: `Chat send failed: ${failureKind}`,
        detail: fallback || labels.settings_test_unknown_error,
        route: '/',
        context: {
          ui_locale: uiLocale,
          query_scope: resolvedScope,
          active_conversation: Boolean(activeConvId),
          active_project: Boolean(activeProjectId),
          paper_guide_mode: Boolean(
            activeConversation?.mode === 'paper_guide'
            || activeConversation?.bound_source_path
            || guideBindings?.[String(activeConvId || '')]?.sourcePath,
          ),
          message_count: messages.length,
          pending_image_count: pendingImages.length,
          upload_item_count: uploadItems.length,
          ready_upload_count: uploadItems.filter((item) => item.kind === 'pdf' && item.ready).length,
          running_upload_count: uploadItems.filter((item) => item.kind === 'pdf' && !item.ready && item.status !== 'error').length,
          selected_context: Boolean(contextPackForSend),
          selected_context_item_count: Array.isArray(contextPackForSend?.items) ? contextPackForSend.items.length : 0,
          agent_mode: agentMode,
          prompt_length: text.trim().length,
          prompt_empty: text.trim().length === 0,
        },
        payload: {
          error_kind: failureKind,
          http_status: httpStatusFromError(fallback),
        },
        fingerprint: `chat-send:${failureKind}:${resolvedScope}:${uiLocale}`,
      })
      if (isModelConnectionError(err)) {
        message.error(labels.chat_api_connection_failed.replace('{error}', fallback || labels.settings_test_unknown_error))
        void refreshReadiness().catch(() => {})
        onOpenApiSettings('text')
        return
      }
      message.error(fallback || labels.upload_failed_generic)
    })
  }, [
    activeConvId,
    activeConversation?.bound_source_path,
    activeConversation?.mode,
    activeProjectId,
    agentMode,
    guideBindings,
    labels,
    maxTokens,
    messages.length,
    onOpenApiSettings,
    onSelectedResearchContextConsumed,
    pendingImages.length,
    queryScope,
    refreshReadiness,
    researchContext.activeSource.ready,
    researchContext.api.sendBlockTarget,
    selectedResearchContext,
    sendMessage,
    temperature,
    topK,
    uiLocale,
    uploadItems,
  ])
}
