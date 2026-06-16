import { userIssuesApi, type UserIssuePayload } from './api/userIssues'

const RECENT_WINDOW_MS = 60_000
const recentIssues = new Map<string, number>()

function trimText(value: unknown, limit = 2000): string {
  const text = String(value ?? '').replace(/\s+/g, ' ').trim()
  return text.length > limit ? text.slice(0, limit) : text
}

function routeText(): string {
  try {
    return `${window.location.pathname}${window.location.search}${window.location.hash}`
  } catch {
    return ''
  }
}

function fingerprintFor(payload: UserIssuePayload): string {
  return [
    payload.source || 'frontend',
    payload.domain || 'runtime',
    payload.severity || 'error',
    payload.summary,
    trimText(payload.detail || '', 500),
    routeText(),
  ].join('|').toLowerCase()
}

function shouldSend(key: string): boolean {
  const now = Date.now()
  const last = recentIssues.get(key) || 0
  if (now - last < RECENT_WINDOW_MS) return false
  recentIssues.set(key, now)
  for (const [recentKey, ts] of Array.from(recentIssues.entries())) {
    if (now - ts > RECENT_WINDOW_MS * 5) recentIssues.delete(recentKey)
  }
  return true
}

function submitIssue(payload: UserIssuePayload) {
  const fingerprint = payload.fingerprint || fingerprintFor(payload)
  if (!shouldSend(fingerprint)) return
  void userIssuesApi.record({
    source: 'frontend',
    domain: 'runtime',
    severity: 'error',
    route: routeText(),
    ...payload,
    fingerprint,
    context: {
      url: routeText(),
      user_agent: typeof navigator !== 'undefined' ? navigator.userAgent : '',
      ...payload.context,
    },
  }).catch(() => {
    /* The reporter must never surface its own failures to users. */
  })
}

function detailFromReason(reason: unknown): string {
  if (reason instanceof Error) return trimText(reason.stack || reason.message, 4000)
  if (typeof reason === 'object' && reason) {
    try {
      return trimText(JSON.stringify(reason), 4000)
    } catch {
      return trimText(String(reason), 4000)
    }
  }
  return trimText(reason, 4000)
}

export function installUserIssueReporter() {
  if (typeof window === 'undefined') return
  window.addEventListener('error', (event) => {
    const err = event.error instanceof Error ? event.error : null
    submitIssue({
      summary: trimText(err?.message || event.message || 'Frontend runtime error', 500),
      detail: trimText(err?.stack || event.message || '', 4000),
      payload: {
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
      },
    })
  })
  window.addEventListener('unhandledrejection', (event) => {
    const reason = event.reason
    submitIssue({
      summary: reason instanceof Error ? trimText(reason.message, 500) : 'Unhandled promise rejection',
      detail: detailFromReason(reason),
    })
  })
}
