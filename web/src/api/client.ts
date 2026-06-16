const BASE = ''
export const ACCESS_TOKEN_STORAGE_KEY = 'kb_access_token'
export const AUTH_REQUIRED_EVENT = 'kb:auth-required'

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
  try {
    window.dispatchEvent(new CustomEvent(AUTH_REQUIRED_EVENT))
  } catch {
    /* ignore */
  }
}

export async function authFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  return fetch(input, {
    ...init,
    headers: authHeaders(init),
    credentials: init?.credentials || 'same-origin',
  })
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await authFetch(BASE + url, init)
  } catch {
    throw new Error(
      'Cannot connect to backend. Ensure the backend is running and Vite proxy /api targets the correct port.',
    )
  }
  if (!res.ok) {
    if (res.status === 401) dispatchAuthRequired()
    let detail = ''
    try {
      const text = (await res.text()).trim()
      detail = text ? `: ${text}` : ''
    } catch {
      detail = ''
    }
    throw new Error(`${res.status} ${res.statusText}${detail}`)
  }
  return res.json()
}

export const api = {
  get: <T>(url: string) => request<T>(url),
  post: <T>(url: string, body?: unknown) =>
    request<T>(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: body ? JSON.stringify(body) : undefined,
    }),
  patch: <T>(url: string, body?: unknown) =>
    request<T>(url, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: body ? JSON.stringify(body) : undefined,
    }),
  delete: <T>(url: string) => request<T>(url, { method: 'DELETE' }),
}
