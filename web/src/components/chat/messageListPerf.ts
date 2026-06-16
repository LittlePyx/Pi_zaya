const MESSAGE_LIST_PREP_PERF_LIMIT = 180

export interface MessageListPrepPerfEvent {
  ts: number
  convId: string
  messageCount: number
  assistantCount: number
  heavyCount: number
  lightCount: number
  cacheHits: number
  durationMs: number
}

interface MessageListPrepPerfApi {
  getLogs: () => MessageListPrepPerfEvent[]
  clear: () => void
}

interface MessageListDebugWindow extends Window {
  __kbMessageListPerf?: MessageListPrepPerfApi
}

const messageListPrepPerfLog: MessageListPrepPerfEvent[] = []

export function messageListPerfNow() {
  try {
    return performance.now()
  } catch {
    return Date.now()
  }
}

function ensureMessageListPerfApi() {
  if (typeof window === 'undefined') return
  const w = window as MessageListDebugWindow
  if (w.__kbMessageListPerf) return
  w.__kbMessageListPerf = {
    getLogs: () => messageListPrepPerfLog.slice(),
    clear: () => {
      messageListPrepPerfLog.length = 0
    },
  }
}

export function pushMessageListPrepPerf(event: MessageListPrepPerfEvent) {
  messageListPrepPerfLog.push(event)
  if (messageListPrepPerfLog.length > MESSAGE_LIST_PREP_PERF_LIMIT) {
    messageListPrepPerfLog.splice(0, messageListPrepPerfLog.length - MESSAGE_LIST_PREP_PERF_LIMIT)
  }
  ensureMessageListPerfApi()
}

if (typeof window !== 'undefined') {
  ensureMessageListPerfApi()
}
