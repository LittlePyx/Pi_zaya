import { startTransition, useCallback, useDeferredValue, useMemo, useState } from 'react'
import type { Message } from '../../api/chat'

export interface TimelineItem {
  order: number
  userMsgId: number
  targetMsgId: number
  questionPreview: string
  hasAnswer: boolean
}

export interface TimelineJumpTarget {
  messageId: number
  token: number
}

function compactTimelineText(content: string, maxLen = 68, txt?: Record<string, string>) {
  const raw = String(content || '').replace(/\s+/g, ' ').trim()
  if (!raw) return txt?.timeline_blank_question || 'Blank question'
  const imgOnly = raw.match(/^\[Image attachment x(\d+)\]$/i)
  if (imgOnly) {
    return (txt?.timeline_image_question || 'Image question x{n}').replace('{n}', imgOnly[1] || '1')
  }
  if (raw.length <= maxLen) return raw
  return `${raw.slice(0, Math.max(8, maxLen - 1)).trimEnd()}...`
}

export function useChatTimeline({
  messages,
  labels,
  liveRunning,
  onBlockedJump,
  nextToken,
  onBeforeToggle,
}: {
  messages: Message[]
  labels: Record<string, string>
  liveRunning: boolean
  onBlockedJump: () => void
  nextToken: () => number
  onBeforeToggle?: () => void
}) {
  const [timelineOpen, setTimelineOpen] = useState(true)
  const [timelineJump, setTimelineJump] = useState<TimelineJumpTarget | null>(null)
  const [activeTimelineUserMsgId, setActiveTimelineUserMsgId] = useState<number | null>(null)
  const deferredTimelineMessages = useDeferredValue(messages)

  const messageIndexById = useMemo(() => {
    const map = new Map<number, number>()
    messages.forEach((msg, idx) => {
      map.set(msg.id, idx)
    })
    return map
  }, [messages])

  const timelineItems = useMemo(() => {
    const out: TimelineItem[] = []
    let pendingUser: Message | null = null
    let order = 0
    for (const msg of deferredTimelineMessages) {
      if (msg.role === 'user') {
        pendingUser = msg
        continue
      }
      if (msg.role !== 'assistant' || !pendingUser) continue
      order += 1
      out.push({
        order,
        userMsgId: pendingUser.id,
        targetMsgId: msg.id,
        questionPreview: compactTimelineText(pendingUser.content, 68, labels),
        hasAnswer: true,
      })
      pendingUser = null
    }
    if (pendingUser) {
      order += 1
      out.push({
        order,
        userMsgId: pendingUser.id,
        targetMsgId: pendingUser.id,
        questionPreview: compactTimelineText(pendingUser.content, 68, labels),
        hasAnswer: false,
      })
    }
    return out
  }, [labels, deferredTimelineMessages])

  const timelineTrackedMessageIds = useMemo(
    () => timelineItems.map((item) => item.targetMsgId),
    [timelineItems],
  )

  const timelineUserMsgIdByTargetMsgId = useMemo(() => {
    const map = new Map<number, number>()
    timelineItems.forEach((item) => {
      map.set(item.targetMsgId, item.userMsgId)
    })
    return map
  }, [timelineItems])

  const clearTimelineSelection = useCallback(() => {
    setTimelineJump(null)
    setActiveTimelineUserMsgId(null)
  }, [])

  const toggleTimelineOpen = useCallback(() => {
    onBeforeToggle?.()
    clearTimelineSelection()
    setTimelineOpen((value) => !value)
  }, [clearTimelineSelection, onBeforeToggle])

  const resetTimeline = useCallback(() => {
    setTimelineOpen(true)
    clearTimelineSelection()
  }, [clearTimelineSelection])

  const openTimeline = useCallback(() => {
    setTimelineOpen(true)
  }, [])

  const jumpToTimelineItem = useCallback((item: TimelineItem) => {
    if (liveRunning) {
      onBlockedJump()
      return
    }
    const idx = messageIndexById.get(item.targetMsgId)
    if (idx == null) return
    setActiveTimelineUserMsgId(null)
    const token = nextToken()
    window.setTimeout(() => {
      setTimelineJump({ messageId: item.targetMsgId, token })
    }, 0)
  }, [liveRunning, messageIndexById, nextToken, onBlockedJump])

  const handleTimelineJumpHandled = useCallback((handled: TimelineJumpTarget) => {
    setTimelineJump((current) => (
      current?.token === handled.token && current?.messageId === handled.messageId
        ? null
        : current
    ))
    setActiveTimelineUserMsgId(null)
  }, [])

  const handleTrackedMessageActive = useCallback((messageId: number | null) => {
    const nextUserMsgId = messageId != null
      ? (timelineUserMsgIdByTargetMsgId.get(messageId) ?? null)
      : null
    startTransition(() => {
      setActiveTimelineUserMsgId((current) => (
        current === nextUserMsgId ? current : nextUserMsgId
      ))
    })
  }, [timelineUserMsgIdByTargetMsgId])

  return {
    timelineOpen,
    timelineItems,
    timelineJump,
    timelineTrackedMessageIds,
    activeTimelineUserMsgId,
    toggleTimelineOpen,
    resetTimeline,
    openTimeline,
    jumpToTimelineItem,
    handleTimelineJumpHandled,
    handleTrackedMessageActive,
  }
}
