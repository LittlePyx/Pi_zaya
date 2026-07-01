import type { Message, MessagePage } from '../api/chat'

export function buildFullMessagePage(messages: Message[]): MessagePage {
  return {
    messages,
    has_more_before: false,
    oldest_loaded_id: messages.length > 0 ? Number(messages[0]?.id || 0) || null : null,
    newest_loaded_id: messages.length > 0 ? Number(messages[messages.length - 1]?.id || 0) || null : null,
  }
}

export function mergeLatestMessagePage(
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
