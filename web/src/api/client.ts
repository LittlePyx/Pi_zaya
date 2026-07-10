import { redactSensitiveText } from '../utils/redaction'
import { authGateBuildEnabled } from './authGate'

export function normalizeApiBase(value: unknown): string {
  const raw = String(value ?? '').trim().replace(/\/+$/, '')
  if (!raw) return ''
  try {
    const url = new URL(raw)
    return url.protocol === 'http:' || url.protocol === 'https:' ? url.toString().replace(/\/+$/, '') : ''
  } catch {
    return raw.startsWith('/') && !raw.startsWith('//') ? raw : ''
  }
}

const VITE_ENV = ((import.meta as ImportMeta & {
  env?: Record<string, string | undefined>
}).env || {})

export const API_BASE = normalizeApiBase(VITE_ENV.VITE_BACKEND_URL)
export const ACCESS_TOKEN_STORAGE_KEY = 'kb_access_token'
export const AUTH_REQUIRED_EVENT = 'kb:auth-required'
export const MANAGEMENT_AUTH_REQUIRED_EVENT = 'kb:management-auth-required'
const AUTH_GATE_EVENTS_ENABLED = authGateBuildEnabled()
export const BACKEND_CONNECT_ERROR_MESSAGE =
  'Cannot connect to the Pi-zaya backend. For local development, run .\\run_new.ps1 -StopExisting so /api is proxied to FastAPI; for single-server mode, run python server.py after building web/dist.'
const ERROR_DETAIL_KEYS = ['detail', 'message', 'error', 'reason', 'title', 'msg', 'errors'] as const

let accessToken = ''

export function getAccessToken(): string {
  return accessToken
}

export function setAccessToken(token: string) {
  accessToken = token.trim()
  try {
    window.localStorage.removeItem(ACCESS_TOKEN_STORAGE_KEY)
  } catch {
    /* ignore */
  }
}

function authHeaders(init?: RequestInit): Headers {
  const headers = new Headers(init?.headers || {})
  const token = getAccessToken()
  if (token && !headers.has('X-KB-Access-Token')) {
    headers.set('X-KB-Access-Token', token)
  }
  return headers
}

function dispatchAuthRequired() {
  if (!AUTH_GATE_EVENTS_ENABLED) return
  try {
    window.dispatchEvent(new CustomEvent(AUTH_REQUIRED_EVENT))
  } catch {
    /* ignore */
  }
}

function dispatchManagementAuthRequired() {
  try {
    window.dispatchEvent(new CustomEvent(MANAGEMENT_AUTH_REQUIRED_EVENT))
  } catch {
    /* ignore */
  }
}

function joinErrorParts(parts: string[]): string {
  const seen = new Set<string>()
  const clean = parts
    .map((item) => redactSensitiveText(item))
    .filter((item) => {
      if (!item || seen.has(item)) return false
      seen.add(item)
      return true
    })
  return redactSensitiveText(clean.join('; '))
}

function cleanErrorDetail(value: unknown): string {
  if (typeof value === 'string') return redactSensitiveText(value)
  if (typeof value === 'number' || typeof value === 'boolean') return redactSensitiveText(value)
  if (Array.isArray(value)) {
    return joinErrorParts(value.map((item) => cleanErrorDetail(item)).filter(Boolean))
  }
  if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>
    const parts = ERROR_DETAIL_KEYS
      .filter((key) => Object.prototype.hasOwnProperty.call(record, key))
      .map((key) => cleanErrorDetail(record[key]))
      .filter(Boolean)
    return joinErrorParts(parts)
  }
  return ''
}

function responseErrorDetail(text: string): string {
  const raw = text.trim()
  if (!raw) return ''
  try {
    const parsed = JSON.parse(raw) as unknown
    return cleanErrorDetail(parsed) || redactSensitiveText(raw)
  } catch {
    return redactSensitiveText(raw)
  }
}

export function resolveApiUrl(input: string): string {
  const isApiPath = input === '/api'
    || input.startsWith('/api/')
    || input.startsWith('/api?')
    || input.startsWith('/api#')
  if (!API_BASE || !isApiPath) return input
  return `${API_BASE}${input}`
}

function resolveFetchInput(input: RequestInfo | URL): RequestInfo | URL {
  if (typeof input === 'string') return resolveApiUrl(input)
  if (input instanceof URL) {
    if (
      typeof window !== 'undefined'
      && input.origin === window.location.origin
    ) {
      return resolveApiUrl(`${input.pathname}${input.search}${input.hash}`)
    }
    return input
  }
  return input
}

export async function authFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  let response: Response
  try {
    response = await fetch(resolveFetchInput(input), {
      ...init,
      headers: authHeaders(init),
      credentials: init?.credentials || 'same-origin',
    })
  } catch (err) {
    if (init?.signal?.aborted || (err instanceof Error && err.name === 'AbortError')) {
      throw err
    }
    throw new Error(BACKEND_CONNECT_ERROR_MESSAGE)
  }
  if (response.headers.get('X-KB-Management-Auth') === 'required') {
    dispatchManagementAuthRequired()
  }
  if (response.status === 401) {
    dispatchAuthRequired()
  }
  return response
}

export async function responseError(response: Response, fallbackStatusText = 'Request failed'): Promise<Error> {
  const statusText = redactSensitiveText(response.statusText || fallbackStatusText, 240) || fallbackStatusText
  let detail = ''
  try {
    const text = (await response.text()).trim()
    const formatted = responseErrorDetail(text)
    detail = formatted ? `: ${formatted}` : ''
  } catch {
    detail = ''
  }
  return new Error(`${response.status} ${statusText}${detail}`)
}

export async function responseJson<T>(
  response: Response,
  fallbackStatusText = 'Request failed',
): Promise<T> {
  if (!response.ok) {
    throw await responseError(response, fallbackStatusText)
  }
  if (response.status === 204 || response.status === 205) {
    return undefined as T
  }
  const text = (await response.text()).trim()
  if (!text) {
    return undefined as T
  }
  return JSON.parse(text) as T
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await authFetch(url, init)
  if (!res.ok) {
    throw await responseError(res)
  }
  return responseJson<T>(res)
}

function jsonBody(body: unknown): string | undefined {
  return body === undefined ? undefined : JSON.stringify(body)
}

function jsonRequestInit(method: string, body: unknown, init?: RequestInit): RequestInit {
  const headers = new Headers(init?.headers || {})
  if (!headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }
  return {
    ...init,
    method,
    headers,
    body: jsonBody(body),
  }
}

export const api = {
  get: <T>(url: string, init?: RequestInit) => request<T>(url, init),
  post: <T>(url: string, body?: unknown, init?: RequestInit) => request<T>(url, jsonRequestInit('POST', body, init)),
  patch: <T>(url: string, body?: unknown, init?: RequestInit) => request<T>(url, jsonRequestInit('PATCH', body, init)),
  delete: <T>(url: string, init?: RequestInit) => request<T>(url, { ...init, method: 'DELETE' }),
}
