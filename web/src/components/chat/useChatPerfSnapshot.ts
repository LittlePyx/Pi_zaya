import { useEffect, useState } from 'react'

export interface ChatPerfSnapshot {
  switchTotal: number
  switchAvgMs: number
  refsTotal: number
  refsAvgMs: number
  openPhases: number
  messagePrep: number
}

interface ChatDebugApi {
  getLogs?: () => unknown[]
  summary?: () => Record<string, unknown>
}

interface ChatDebugWindow extends Window {
  __kbSwitchPerf?: ChatDebugApi
  __kbRefsPerf?: ChatDebugApi
  __kbConversationOpenPerf?: ChatDebugApi
  __kbMessageListPerf?: ChatDebugApi
}

function emptyPerfSnapshot(): ChatPerfSnapshot {
  return { switchTotal: 0, switchAvgMs: 0, refsTotal: 0, refsAvgMs: 0, openPhases: 0, messagePrep: 0 }
}

function safeNumber(value: unknown) {
  const num = Number(value)
  return Number.isFinite(num) ? num : 0
}

function safeLogCount(api?: ChatDebugApi) {
  try {
    const logs = api?.getLogs?.()
    return Array.isArray(logs) ? logs.length : 0
  } catch {
    return 0
  }
}

function collectChatPerfSnapshot(): ChatPerfSnapshot {
  if (typeof window === 'undefined') return emptyPerfSnapshot()
  const w = window as ChatDebugWindow
  const switchSummary = w.__kbSwitchPerf?.summary?.() || {}
  const refsSummary = w.__kbRefsPerf?.summary?.() || {}
  return {
    switchTotal: safeNumber(switchSummary.total),
    switchAvgMs: safeNumber(switchSummary.avgSuccessMs),
    refsTotal: safeNumber(refsSummary.total),
    refsAvgMs: safeNumber(refsSummary.avgFetchMs),
    openPhases: safeLogCount(w.__kbConversationOpenPerf),
    messagePrep: safeLogCount(w.__kbMessageListPerf),
  }
}

export function useChatPerfSnapshot(enabled: boolean) {
  const [debugSnapshot, setDebugSnapshot] = useState<ChatPerfSnapshot>(() => collectChatPerfSnapshot())

  useEffect(() => {
    if (!enabled || typeof window === 'undefined') return undefined
    const update = () => setDebugSnapshot(collectChatPerfSnapshot())
    update()
    const timer = window.setInterval(update, 1000)
    return () => window.clearInterval(timer)
  }, [enabled])

  return debugSnapshot
}
