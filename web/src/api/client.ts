const BASE = ''

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(BASE + url, init)
  } catch {
    throw new Error(
      'Cannot connect to backend. Ensure the backend is running and Vite proxy /api targets the correct port.',
    )
  }
  if (!res.ok) {
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

