import { expect, test, type Page, type Route } from '@playwright/test'

const AUTH_GATE_ENABLED = process.env.VITE_ENABLE_AUTH_GATE === '1'
  && process.env.VITE_PRIVATE_INSTANCE_AUTH === '1'
  && process.env.VITE_ALLOW_LOCAL_AUTH_GATE === '1'

async function fulfillJson(route: Route, body: unknown, status = 200) {
  await route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(body),
  })
}

function settingsPayload() {
  return {
    model: 'test-model',
    base_url: '',
    has_api_key: true,
    db_dir: '',
    connection: {
      text: { has_api_key: true, model: 'test-model', base_url: '' },
      vision: { has_api_key: true, model: 'test-vision', base_url: '' },
      auto_route: false,
    },
    readiness: {
      overall: { status: 'ok', severity: 'ok', reason: 'Ready' },
      providers: {},
    },
    app_readiness: {
      status: 'ok',
      env: 'development',
      production: false,
      auth_required: false,
      items: [],
    },
    prefs: {
      ui_locale: 'zh',
      theme: 'light',
      top_k: 6,
      temperature: 0.2,
      max_tokens: 1216,
      deep_read: false,
    },
  }
}

async function installClientMocks(page: Page, seenBodies: string[], seenUrls: string[] = []) {
  await page.route(/^https?:\/\/[^/]+\/api(?:\/|$)/, async (route) => {
    const url = new URL(route.request().url())
    const pathname = url.pathname.replace(/\/+$/, '')
    if (pathname === '/api/client/falsy') {
      seenUrls.push(route.request().url())
      seenBodies.push(route.request().postData() ?? '<missing>')
      await fulfillJson(route, { ok: true })
      return
    }
    if (pathname === '/api/client/empty-204') {
      await route.fulfill({ status: 204, body: '' })
      return
    }
    if (pathname === '/api/client/empty-200') {
      await route.fulfill({ status: 200, contentType: 'application/json', body: '' })
      return
    }
    if (pathname === '/api/client/private' || pathname === '/api/client/private-fetch') {
      await fulfillJson(route, { detail: 'Authentication required' }, 401)
      return
    }
    if (pathname === '/api/client/upload-fails') {
      await fulfillJson(route, { detail: [{ msg: 'File exceeds upload limit' }] }, 413)
      return
    }
    if (pathname === '/api/client/secret-json-fails') {
      await fulfillJson(route, {
        detail: 'Provider rejected Authorization: Bearer sk-secretsecretsecretsecret; api_key=abcdef1234567890; user alice@example.com; path C:\\Users\\Alice\\paper.pdf; callback https://proxy.example/v1?token=private; redirect https://proxy.example/callback#access_token=fragmentsecret',
      }, 500)
      return
    }
    if (pathname === '/api/client/message-json-fails') {
      await fulfillJson(route, {
        error: {
          message: 'Upstream provider rejected api_key=abcdef1234567890 for alice@example.com at C:\\Users\\Alice\\paper.pdf',
        },
        debug: 'Bearer sk-secretsecretsecretsecret should stay server-side',
      }, 500)
      return
    }
    if (pathname === '/api/client/errors-array-fails') {
      await fulfillJson(route, {
        reason: 'Batch validation failed',
        errors: [
          { msg: 'File path C:\\Users\\Alice\\paper.pdf is outside the workspace' },
          { message: 'Callback https://proxy.example/cb?token=fragmentsecret failed for alice@example.com' },
        ],
      }, 422)
      return
    }
    if (pathname === '/api/client/secret-text-fails') {
      await route.fulfill({
        status: 502,
        contentType: 'text/plain',
        body: 'upstream failed with Bearer github_pat_secretsecretsecretsecret at /Users/alice/project/.env?debug=1 and https://proxy.example/cb#token=fragmentsecret',
      })
      return
    }
    if (pathname === '/api/client/network-fails') {
      await route.abort('failed')
      return
    }
    if (pathname === '/api/client/slow-abort') {
      await new Promise((resolve) => setTimeout(resolve, 800))
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ ok: true }) })
        .catch(() => {})
      return
    }
    if (pathname === '/api/auth/status') {
      await fulfillJson(route, {
        required: false,
        configured: false,
        authenticated: false,
        env: 'development',
        production: false,
      })
      return
    }
    if (pathname === '/api/settings') {
      await fulfillJson(route, settingsPayload())
      return
    }
    if (pathname === '/api/settings/readiness') {
      await fulfillJson(route, settingsPayload().readiness)
      return
    }
    if (pathname === '/api/readiness') {
      await fulfillJson(route, settingsPayload().app_readiness)
      return
    }
    if (pathname === '/api/app/update-check') {
      await fulfillJson(route, { ok: true, status: 'current', update_available: false })
      return
    }
    if (pathname === '/api/sidebar') {
      await fulfillJson(route, { projects: [], root_conversations: [], project_conversations: {} })
      return
    }
    if (pathname === '/api/projects' || pathname === '/api/conversations') {
      await fulfillJson(route, [])
      return
    }
    if (pathname === '/api/chat/citation-shelf') {
      await fulfillJson(route, {
        version: 1,
        scope: 'project',
        scope_id: '__default__',
        project_id: null,
        items: [],
        open: false,
        revision: 0,
        created_at: 0,
        updated_at: 0,
      })
      return
    }
    await fulfillJson(route, {})
  })
}

test('api client normalizes runtime backend base URL for API requests only', async ({ page }) => {
  const seenBodies: string[] = []
  const seenUrls: string[] = []
  await installClientMocks(page, seenBodies, seenUrls)
  await page.goto('/')

  const result = await page.evaluate(async () => {
    const { API_BASE, api, normalizeApiBase, resolveApiUrl } = await import('/src/api/client.ts')
    await api.post('/api/client/falsy', { ok: true })
    return {
      apiBase: API_BASE,
      normalizedAbsolute: normalizeApiBase('https://backend.example///'),
      normalizedPath: normalizeApiBase('/backend///'),
      rejectsProtocolRelative: normalizeApiBase('//backend.example'),
      apiUrl: resolveApiUrl('/api/settings?x=1'),
      assetUrl: resolveApiUrl('/assets/app.js'),
      apiLookalikeUrl: resolveApiUrl('/apiary/logo.png'),
      absoluteUrl: resolveApiUrl('https://elsewhere.example/api/settings'),
    }
  })

  expect(result.normalizedAbsolute).toBe('https://backend.example')
  expect(result.normalizedPath).toBe('/backend')
  expect(result.rejectsProtocolRelative).toBe('')
  expect(result.assetUrl).toBe('/assets/app.js')
  expect(result.apiLookalikeUrl).toBe('/apiary/logo.png')
  expect(result.absoluteUrl).toBe('https://elsewhere.example/api/settings')
  expect(result.apiUrl).toBe(result.apiBase ? `${result.apiBase}/api/settings?x=1` : '/api/settings?x=1')
  if (result.apiBase) {
    expect(seenUrls[0]).toBe(`${result.apiBase}/api/client/falsy`)
  } else {
    expect(seenUrls[0]).toContain('/api/client/falsy')
  }
})

test('bibliometrics cached empty summaries expire quickly and can be retried', async ({ page }) => {
  let requestCount = 0
  await page.route('**/api/references/bibliometrics', async (route) => {
    requestCount += 1
    await fulfillJson(route, {
      bibliometrics_checked: true,
      doi: '10.1000/no-public-summary',
      metadata_export_acceptance: {
        summary_export_ready: false,
        summary_status: 'missing',
      },
    })
  })
  await page.goto('/')

  const counts = await page.evaluate(async () => {
    const originalNow = Date.now
    let now = 1_000_000
    Date.now = () => now
    try {
      const { referencesApi } = await import('/src/api/references.ts')
      const meta = { doi: '10.1000/no-public-summary', target_locale: 'zh' }
      await referencesApi.bibliometricsCached(meta)
      await referencesApi.bibliometricsCached(meta)
      const beforeExpiry = performance.getEntriesByType('resource')
        .filter((entry) => entry.name.includes('/api/references/bibliometrics')).length
      now += 16_000
      await referencesApi.bibliometricsCached(meta)
      const afterExpiry = performance.getEntriesByType('resource')
        .filter((entry) => entry.name.includes('/api/references/bibliometrics')).length
      return { beforeExpiry, afterExpiry }
    } finally {
      Date.now = originalNow
    }
  })

  expect(requestCount).toBe(2)
  expect(counts.afterExpiry).toBeGreaterThanOrEqual(counts.beforeExpiry)
})

test('api client preserves falsy JSON bodies and accepts empty success responses', async ({ page }) => {
  const seenBodies: string[] = []
  await installClientMocks(page, seenBodies)
  await page.goto('/')

  const result = await page.evaluate(async () => {
    const { api } = await import('/src/api/client.ts')
    await api.post('/api/client/falsy', false)
    await api.post('/api/client/falsy', 0)
    await api.post('/api/client/falsy', '')
    await api.post('/api/client/falsy', null)
    const empty204 = await api.post('/api/client/empty-204', { ok: true })
    const empty200 = await api.post('/api/client/empty-200', { ok: true })
    return {
      empty204IsUndefined: empty204 === undefined,
      empty200IsUndefined: empty200 === undefined,
    }
  })

  expect(seenBodies).toEqual(['false', '0', '""', 'null'])
  expect(result.empty204IsUndefined).toBe(true)
  expect(result.empty200IsUndefined).toBe(true)
})

test('api client only dispatches auth-required event in the private auth-gate build', async ({ page }) => {
  const seenBodies: string[] = []
  await installClientMocks(page, seenBodies)
  await page.goto('/')

  const result = await page.evaluate(async () => {
    const { AUTH_REQUIRED_EVENT, api } = await import('/src/api/client.ts')
    let eventCount = 0
    window.addEventListener(AUTH_REQUIRED_EVENT, () => {
      eventCount += 1
    })
    try {
      await api.get('/api/client/private')
    } catch (err) {
      return {
        eventCount,
        message: err instanceof Error ? err.message : String(err),
      }
    }
    return { eventCount, message: '' }
  })

  expect(result.eventCount).toBe(AUTH_GATE_ENABLED ? 1 : 0)
  expect(result.message).toContain('401')
  expect(result.message).toContain('Authentication required')
})

test('authFetch only dispatches auth-required event for direct callers in the private auth-gate build', async ({ page }) => {
  const seenBodies: string[] = []
  await installClientMocks(page, seenBodies)
  await page.goto('/')

  const result = await page.evaluate(async () => {
    const { AUTH_REQUIRED_EVENT, authFetch } = await import('/src/api/client.ts')
    let eventCount = 0
    window.addEventListener(AUTH_REQUIRED_EVENT, () => {
      eventCount += 1
    })
    const response = await authFetch('/api/client/private-fetch', { method: 'POST', body: new FormData() })
    return {
      eventCount,
      status: response.status,
      ok: response.ok,
    }
  })

  expect(result.eventCount).toBe(AUTH_GATE_ENABLED ? 1 : 0)
  expect(result.status).toBe(401)
  expect(result.ok).toBe(false)
})

test('responseJson preserves backend detail for direct authFetch callers', async ({ page }) => {
  const seenBodies: string[] = []
  await installClientMocks(page, seenBodies)
  await page.goto('/')

  const result = await page.evaluate(async () => {
    const { authFetch, responseJson } = await import('/src/api/client.ts')
    const response = await authFetch('/api/client/upload-fails', { method: 'POST', body: new FormData() })
    try {
      await responseJson(response, 'upload failed')
    } catch (err) {
      return err instanceof Error ? err.message : String(err)
    }
    return ''
  })

  expect(result).toContain('413')
  expect(result).toContain('File exceeds upload limit')
})

test('api client redacts secrets from JSON backend error details', async ({ page }) => {
  const seenBodies: string[] = []
  await installClientMocks(page, seenBodies)
  await page.goto('/')

  const messageText = await page.evaluate(async () => {
    const { api } = await import('/src/api/client.ts')
    try {
      await api.get('/api/client/secret-json-fails')
    } catch (err) {
      return err instanceof Error ? err.message : String(err)
    }
    return ''
  })

  expect(messageText).toContain('500')
  expect(messageText).toContain('Authorization: [token]')
  expect(messageText).toContain('api_key=[token]')
  expect(messageText).toContain('[email]')
  expect(messageText).toContain('[local-path]')
  expect(messageText).toContain('https://proxy.example/v1')
  expect(messageText).toContain('https://proxy.example/callback')
  expect(messageText).not.toContain('sk-secret')
  expect(messageText).not.toContain('abcdef1234567890')
  expect(messageText).not.toContain('alice@example.com')
  expect(messageText).not.toContain('C:\\Users')
  expect(messageText).not.toContain('?token=private')
  expect(messageText).not.toContain('#access_token=')
  expect(messageText).not.toContain('fragmentsecret')
})

test('api client extracts readable JSON error fields without exposing debug payloads', async ({ page }) => {
  const seenBodies: string[] = []
  await installClientMocks(page, seenBodies)
  await page.goto('/')

  const messageText = await page.evaluate(async () => {
    const { api } = await import('/src/api/client.ts')
    try {
      await api.get('/api/client/message-json-fails')
    } catch (err) {
      return err instanceof Error ? err.message : String(err)
    }
    return ''
  })

  expect(messageText).toContain('500')
  expect(messageText).toContain('Upstream provider rejected api_key=[token]')
  expect(messageText).toContain('[email]')
  expect(messageText).toContain('[local-path]')
  expect(messageText).not.toContain('debug')
  expect(messageText).not.toContain('sk-secret')
  expect(messageText).not.toContain('abcdef1234567890')
  expect(messageText).not.toContain('alice@example.com')
  expect(messageText).not.toContain('C:\\Users')
})

test('api client summarizes JSON errors arrays and redacts every item', async ({ page }) => {
  const seenBodies: string[] = []
  await installClientMocks(page, seenBodies)
  await page.goto('/')

  const messageText = await page.evaluate(async () => {
    const { api } = await import('/src/api/client.ts')
    try {
      await api.get('/api/client/errors-array-fails')
    } catch (err) {
      return err instanceof Error ? err.message : String(err)
    }
    return ''
  })

  expect(messageText).toContain('422')
  expect(messageText).toContain('Batch validation failed')
  expect(messageText).toContain('File path [local-path] is outside the workspace')
  expect(messageText).toContain('Callback https://proxy.example/cb failed for [email]')
  expect(messageText).not.toContain('C:\\Users')
  expect(messageText).not.toContain('?token=')
  expect(messageText).not.toContain('fragmentsecret')
  expect(messageText).not.toContain('alice@example.com')
})

test('api client redacts secrets from raw text error bodies', async ({ page }) => {
  const seenBodies: string[] = []
  await installClientMocks(page, seenBodies)
  await page.goto('/')

  const messageText = await page.evaluate(async () => {
    const { authFetch, responseJson } = await import('/src/api/client.ts')
    const response = await authFetch('/api/client/secret-text-fails')
    try {
      await responseJson(response, 'raw secret failed')
    } catch (err) {
      return err instanceof Error ? err.message : String(err)
    }
    return ''
  })

  expect(messageText).toContain('502')
  expect(messageText).toContain('Bearer [token]')
  expect(messageText).toContain('[local-path]')
  expect(messageText).toContain('https://proxy.example/cb')
  expect(messageText).not.toContain('github_pat_secret')
  expect(messageText).not.toContain('/Users/alice/project')
  expect(messageText).not.toContain('#token=')
  expect(messageText).not.toContain('fragmentsecret')
})

test('api client redacts sensitive status text as well as response bodies', async ({ page }) => {
  const seenBodies: string[] = []
  await installClientMocks(page, seenBodies)
  await page.goto('/')

  const messageText = await page.evaluate(async () => {
    const { responseJson } = await import('/src/api/client.ts')
    const response = new Response(
      JSON.stringify({ detail: 'Body includes /Users/alice/project/.env?debug=1 and alice@example.com' }),
      {
        status: 503,
        statusText: 'Bearer sk-secretsecretsecretsecret at C:\\Users\\Alice\\paper.pdf',
        headers: { 'content-type': 'application/json' },
      },
    )
    try {
      await responseJson(response, 'status text secret failed')
    } catch (err) {
      return err instanceof Error ? err.message : String(err)
    }
    return ''
  })

  expect(messageText).toContain('503')
  expect(messageText).toContain('Bearer [token]')
  expect(messageText).toContain('[local-path]')
  expect(messageText).toContain('[email]')
  expect(messageText).not.toContain('sk-secret')
  expect(messageText).not.toContain('C:\\Users')
  expect(messageText).not.toContain('/Users/alice')
  expect(messageText).not.toContain('alice@example.com')
})

test('authFetch reports a stable backend connection error on network failure', async ({ page }) => {
  const seenBodies: string[] = []
  await installClientMocks(page, seenBodies)
  await page.goto('/')

  const result = await page.evaluate(async () => {
    const { BACKEND_CONNECT_ERROR_MESSAGE, authFetch } = await import('/src/api/client.ts')
    try {
      await authFetch('/api/client/network-fails')
    } catch (err) {
      return {
        expected: BACKEND_CONNECT_ERROR_MESSAGE,
        message: err instanceof Error ? err.message : String(err),
      }
    }
    return { expected: BACKEND_CONNECT_ERROR_MESSAGE, message: '' }
  })

  expect(result.message).toBe(result.expected)
  expect(result.message).toContain('.\\run_new.ps1 -StopExisting')
  expect(result.message).toContain('/api is proxied to FastAPI')
  expect(result.message).toContain('python server.py')
  expect(result.message).not.toContain('Failed to fetch')
})

test('authFetch preserves AbortError for intentionally cancelled requests', async ({ page }) => {
  const seenBodies: string[] = []
  await installClientMocks(page, seenBodies)
  await page.goto('/')

  const result = await page.evaluate(async () => {
    const { BACKEND_CONNECT_ERROR_MESSAGE, authFetch } = await import('/src/api/client.ts')
    const controller = new AbortController()
    const pending = authFetch('/api/client/slow-abort', { signal: controller.signal })
    controller.abort()
    try {
      await pending
    } catch (err) {
      return {
        expected: BACKEND_CONNECT_ERROR_MESSAGE,
        name: err instanceof Error ? err.name : '',
        message: err instanceof Error ? err.message : String(err),
      }
    }
    return { expected: BACKEND_CONNECT_ERROR_MESSAGE, name: '', message: '' }
  })

  expect(result.name).toBe('AbortError')
  expect(result.message).not.toBe(result.expected)
})

test('api request wrapper preserves AbortError for intentionally cancelled requests', async ({ page }) => {
  const seenBodies: string[] = []
  await installClientMocks(page, seenBodies)
  await page.goto('/')

  const result = await page.evaluate(async () => {
    const { BACKEND_CONNECT_ERROR_MESSAGE, api } = await import('/src/api/client.ts')
    const controller = new AbortController()
    const pending = api.get('/api/client/slow-abort', { signal: controller.signal })
    controller.abort()
    try {
      await pending
    } catch (err) {
      return {
        expected: BACKEND_CONNECT_ERROR_MESSAGE,
        name: err instanceof Error ? err.name : '',
        message: err instanceof Error ? err.message : String(err),
      }
    }
    return { expected: BACKEND_CONNECT_ERROR_MESSAGE, name: '', message: '' }
  })

  expect(result.name).toBe('AbortError')
  expect(result.message).not.toBe(result.expected)
})
