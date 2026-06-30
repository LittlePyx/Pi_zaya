import { userIssuesApi, type UserIssuePayload } from './api/userIssues'
import { redactSensitiveText } from './utils/redaction'

const RECENT_WINDOW_MS = 60_000
const recentIssues = new Map<string, number>()

const INSTALL_FLAG = '__kbUserIssueReporterInstalled'
const SENSITIVE_PAYLOAD_KEY_RE = /(?:api[_-]?key|token|secret|password|authorization|cookie|(?:^|[_-])user[_-]?agent(?:$|[_-])|^ua$|browser[_-]?agent|pdf[_-]?path|md[_-]?path|source[_-]?path|absolute[_-]?path|local[_-]?path|file[_-]?path|path|pdf[_-]?name|md[_-]?name|source[_-]?name|document[_-]?name|file[_-]?name|filename|(?:^|[_-])(?:title|main|raw|prompt|query|question|answer|message|content|body|excerpt|quote|abstract)(?:$|[_-]?(?:text|markdown|content|body|raw)$)|(?:pdf|md|markdown|raw|full|source|document|page)[_-]?text$)/i
const FREEFORM_SAMPLE_KEY_RE = /(?:^|[_-])(?:sample|samples|example|examples|evidence|snippet|snippets)(?:$|[_-]?(?:text|texts|markdown|content|body|raw|items?|list|names?|values?)$)/i
const DOCUMENT_COLLECTION_KEY_RE = /(?:^|[_-])(?:paper|papers|document|documents|file|files)(?:$|[_-]?(?:list|names?|titles?|items?)$)|(?:^|[_-])(?:source|sources)[_-](?:list|names?|titles?|items?)$/i
const ISSUE_PAYLOAD_DICT_LIMIT = 100
const ISSUE_PAYLOAD_LIST_LIMIT = 20
const ISSUE_PAYLOAD_STRING_LIMIT = 500

function redactText(value: unknown, limit = 2000): string {
  return redactSensitiveText(value, limit)
}

function routeText(): string {
  try {
    return window.location.pathname || '/'
  } catch {
    return ''
  }
}

function cleanRouteText(value: unknown): string {
  let text = redactText(value, 500)
  for (const sep of ['?', '#']) {
    const idx = text.indexOf(sep)
    if (idx >= 0) text = text.slice(0, idx)
  }
  return text.trim()
}

function sourceText(filename: string | undefined): string {
  const clean = redactText(filename || '', 500)
  if (!clean || clean === '[local-path]') return clean
  try {
    const url = new URL(clean, window.location.origin)
    if (url.protocol === 'http:' || url.protocol === 'https:') {
      if (url.origin === window.location.origin) return cleanRouteText(`${url.pathname}${url.search}${url.hash}`) || '/'
      return '[source-redacted]'
    }
  } catch {
    return clean
  }
  return '[source-redacted]'
}

function hashText(value: string): string {
  let hash = 0x811c9dc5
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i)
    hash = Math.imul(hash, 0x01000193)
  }
  return (hash >>> 0).toString(16).padStart(8, '0')
}

function hashedFingerprint(material: string): string {
  const clean = redactText(material, 1200).toLowerCase()
  return `frontend-${hashText(clean)}-${clean.length.toString(36)}`
}

function fingerprintFor(payload: UserIssuePayload): string {
  const material = [
    payload.source || 'frontend',
    payload.domain || 'runtime',
    payload.severity || 'error',
    redactText(payload.summary, 500),
    redactText(payload.detail || '', 500),
    routeText(),
  ].join('|').toLowerCase()
  return hashedFingerprint(material)
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

function safeIssueScalar(value: unknown): unknown {
  if (value === null || typeof value === 'boolean') return value
  if (typeof value === 'number') return Number.isFinite(value) ? value : null
  return redactText(value, ISSUE_PAYLOAD_STRING_LIMIT)
}

function payloadKeyRequiresRedaction(key: string, value: unknown): boolean {
  if (SENSITIVE_PAYLOAD_KEY_RE.test(key) || FREEFORM_SAMPLE_KEY_RE.test(key)) return true
  if (DOCUMENT_COLLECTION_KEY_RE.test(key)) {
    return !(value === null || typeof value === 'boolean' || typeof value === 'number')
  }
  return false
}

function safeIssuePayload(value: unknown, depth = 0, seen = new WeakSet<object>()): unknown {
  if (depth > 4) return '[depth-limit]'
  if (Array.isArray(value)) {
    return value.slice(0, ISSUE_PAYLOAD_LIST_LIMIT).map((item) => safeIssuePayload(item, depth + 1, seen))
  }
  if (value && typeof value === 'object') {
    if (seen.has(value)) return '[circular]'
    seen.add(value)
    const out: Record<string, unknown> = {}
    for (const [key, rawValue] of Object.entries(value).slice(0, ISSUE_PAYLOAD_DICT_LIMIT)) {
      const cleanKey = redactText(key, 120)
      if (!cleanKey) continue
      out[cleanKey] = payloadKeyRequiresRedaction(cleanKey, rawValue)
        ? '[redacted]'
        : safeIssuePayload(rawValue, depth + 1, seen)
    }
    return out
  }
  return safeIssueScalar(value)
}

function safeIssueRecord(value: unknown): Record<string, unknown> {
  const safe = safeIssuePayload(value)
  if (!safe || typeof safe !== 'object' || Array.isArray(safe)) return {}
  return safe as Record<string, unknown>
}

function submitIssue(payload: UserIssuePayload) {
  const fingerprint = payload.fingerprint ? hashedFingerprint(payload.fingerprint) : fingerprintFor(payload)
  if (!shouldSend(fingerprint)) return
  void userIssuesApi.record({
    source: redactText(payload.source || 'frontend', 120) || 'frontend',
    domain: redactText(payload.domain || 'runtime', 120) || 'runtime',
    severity: redactText(payload.severity || 'error', 40) || 'error',
    summary: redactText(payload.summary || 'User issue', 500) || 'User issue',
    detail: redactText(payload.detail || '', 4000),
    route: cleanRouteText(payload.route || routeText()),
    fingerprint,
    context: {
      ...safeIssueRecord(payload.context),
      url: routeText(),
    },
    payload: safeIssueRecord(payload.payload),
  }).catch(() => {
    /* The reporter must never surface its own failures to users. */
  })
}

export function reportUserIssue(payload: UserIssuePayload) {
  submitIssue(payload)
}

function detailFromReason(reason: unknown): string {
  if (reason instanceof Error) return redactText(reason.stack || reason.message, 4000)
  if (typeof reason === 'object' && reason) {
    try {
      return redactText(JSON.stringify(reason), 4000)
    } catch {
      return redactText(String(reason), 4000)
    }
  }
  return redactText(reason, 4000)
}

export function installUserIssueReporter() {
  if (typeof window === 'undefined') return
  const target = window as Window & { [INSTALL_FLAG]?: boolean }
  if (target[INSTALL_FLAG]) return
  target[INSTALL_FLAG] = true
  window.addEventListener('error', (event) => {
    const err = event.error instanceof Error ? event.error : null
    submitIssue({
      summary: redactText(err?.message || event.message || 'Frontend runtime error', 500),
      detail: redactText(err?.stack || event.message || '', 4000),
      payload: {
        source: sourceText(event.filename),
        lineno: event.lineno,
        colno: event.colno,
      },
    })
  })
  window.addEventListener('unhandledrejection', (event) => {
    const reason = event.reason
    submitIssue({
      summary: reason instanceof Error ? redactText(reason.message, 500) : 'Unhandled promise rejection',
      detail: detailFromReason(reason),
    })
  })
}
