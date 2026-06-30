const WINDOWS_PATH_RE = /(^|[\s("'=])([A-Za-z]:[\\/][^\s"'<>|]+)/g
const FILE_URL_RE = /file:\/\/\/[^\s"'<>|]+/gi
const UNC_PATH_RE = /\\\\[^\s"'<>|]+/g
const UNIX_PATH_RE = /(^|[\s("'=])(\/(?:Users|home|mnt|var|tmp|private)\/[^\s"'<>]+)/gi
const EMAIL_RE = /\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b/g
const AUTH_SECRET_RE = /\b((?:authorization|x[-_]?api[-_]?key|api[-_]?key|access[-_]?token|refresh[-_]?token|cookie|set-cookie)\s*[:=]\s*)(?:bearer\s+)?[A-Za-z0-9._~+/=-]{8,}/gi
const BEARER_RE = /\bbearer\s+[A-Za-z0-9._~+/=-]{8,}/gi
const TOKEN_RE = /\b(?:sk|pk|ghp|github_pat|xoxb|xoxp|ya29|AIza)[A-Za-z0-9_-]{12,}\b/g
const LONG_HASH_RE = /\b[A-Fa-f0-9]{32,}\b/g
const URL_QUERY_RE = /(https?:\/\/[^\s?#]+)(?:[?#][^ \t\r\n"'<>]*)?/g

export function normalizedText(value: unknown): string {
  return String(value ?? '').replace(/\s+/g, ' ').trim()
}

export function truncateText(value: string, limit = 2000): string {
  const max = Math.max(0, Math.round(limit))
  return value.length > max ? value.slice(0, max) : value
}

export function redactSensitiveText(value: unknown, limit = 2000): string {
  const text = normalizedText(value)
  return truncateText(text
    .replace(URL_QUERY_RE, '$1')
    .replace(FILE_URL_RE, '[local-path]')
    .replace(UNC_PATH_RE, '[local-path]')
    .replace(WINDOWS_PATH_RE, '$1[local-path]')
    .replace(UNIX_PATH_RE, '$1[local-path]')
    .replace(EMAIL_RE, '[email]')
    .replace(AUTH_SECRET_RE, '$1[token]')
    .replace(BEARER_RE, 'Bearer [token]')
    .replace(TOKEN_RE, '[token]')
    .replace(LONG_HASH_RE, '[hash]'), limit)
}
