import { expect, test, type Route } from '@playwright/test'

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

function settingsPayload(authRequired = false) {
  return {
    model: 'test-model',
    base_url: '',
    has_api_key: true,
    db_dir: '',
    connection: {
      text: { configured: true, connected: true, has_api_key: true, model: 'test-model', base_url: '' },
      vision: { configured: true, connected: true, has_api_key: true, model: 'test-vision', base_url: '' },
      auto_route: false,
    },
    readiness: {
      overall: { status: 'ok', severity: 'ok', reason: 'Ready' },
      providers: {},
      issues: [],
    },
    app_readiness: {
      status: 'ok',
      env: 'development',
      production: false,
      auth_required: authRequired,
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

async function installUserFacingBackend(routeAuthRequired = false) {
  let protectedCalls = 0
  return {
    async route(route: Route) {
      const url = new URL(route.request().url())
      const pathname = url.pathname.replace(/\/+$/, '')
      if (pathname === '/api/auth/status') {
        await fulfillJson(route, {
          required: routeAuthRequired,
          configured: true,
          authenticated: false,
          env: 'development',
          production: false,
        })
        return
      }
      if (routeAuthRequired && pathname !== '/api/health') {
        protectedCalls += 1
        await fulfillJson(route, { detail: 'Authentication required' }, 401)
        return
      }
      if (pathname === '/api/settings') {
        await fulfillJson(route, settingsPayload(false))
        return
      }
      if (pathname === '/api/settings/readiness') {
        await fulfillJson(route, settingsPayload(false).readiness)
        return
      }
      if (pathname === '/api/readiness') {
        await fulfillJson(route, {
          status: 'ok',
          env: 'development',
          production: false,
          auth_required: false,
          items: [],
        })
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
    },
    protectedCalls: () => protectedCalls,
  }
}

async function installPrivateAuthGateBackend() {
  return async function route(route: Route) {
    const url = new URL(route.request().url())
    const pathname = url.pathname.replace(/\/+$/, '')
    if (pathname === '/api/auth/status') {
      await fulfillJson(route, {
        required: true,
        configured: true,
        authenticated: false,
        env: 'development',
        production: false,
      })
      return
    }
    if (pathname === '/api/settings') {
      await fulfillJson(route, settingsPayload(true))
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
  }
}

test('user-facing build does not render an access-token dialog', async ({ page }) => {
  test.skip(AUTH_GATE_ENABLED, 'this assertion covers the default user-facing build')
  const backend = await installUserFacingBackend(true)
  await page.route(/^https?:\/\/[^/]+\/api(?:\/|$)/, backend.route)

  await page.goto('/')
  await expect.poll(() => backend.protectedCalls()).toBeGreaterThan(0)

  await expect(page.locator('.kb-auth-gate')).toHaveCount(0)
  await expect(page.getByRole('dialog')).toHaveCount(0)
  await expect(page.getByText(/Access Token Required|需要访问令牌/)).toHaveCount(0)
})

test('configured access token stays invisible when auth is not required', async ({ page }) => {
  await page.route(/^https?:\/\/[^/]+\/api(?:\/|$)/, (await installUserFacingBackend(false)).route)

  await page.goto('/')

  await expect(page.locator('.kb-auth-gate')).toHaveCount(0)
  await expect(page.getByRole('dialog')).toHaveCount(0)
})

test('stray api 401 cannot force an access-token dialog in the user build', async ({ page }) => {
  test.skip(AUTH_GATE_ENABLED, 'this assertion covers the default user-facing build')
  let settingsCalls = 0
  await page.route(/^https?:\/\/[^/]+\/api(?:\/|$)/, async (route) => {
    const url = new URL(route.request().url())
    const pathname = url.pathname.replace(/\/+$/, '')
    if (pathname === '/api/auth/status') {
      await fulfillJson(route, {
        required: false,
        configured: true,
        authenticated: false,
        env: 'development',
        production: false,
      })
      return
    }
    if (pathname === '/api/settings') {
      settingsCalls += 1
      await fulfillJson(route, { detail: 'unrelated route auth failure' }, 401)
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

  await page.goto('/')
  await expect.poll(() => settingsCalls).toBeGreaterThan(0)

  await expect(page.locator('.kb-auth-gate')).toHaveCount(0)
  await expect(page.getByRole('dialog')).toHaveCount(0)
})

test('enabled auth gate shows readable Chinese private-instance copy', async ({ page }) => {
  test.skip(!AUTH_GATE_ENABLED, 'auth gate component is lazy-loaded only in the private-instance build')
  await page.route(/^https?:\/\/[^/]+\/api(?:\/|$)/, await installPrivateAuthGateBackend())

  await page.goto('/')

  await expect(page.locator('.kb-auth-gate')).toHaveCount(1)
  await expect(page.getByRole('dialog')).toContainText('需要访问令牌')
  await expect(page.getByRole('dialog')).toContainText('这个 Pi-zaya 实例已启用 API 访问保护。')
  await expect(page.getByTestId('auth-gate-token-help')).toContainText('这个私有实例已通过 KB_REQUIRE_AUTH=1 显式锁定')
  await expect(page.getByPlaceholder('输入访问令牌')).toHaveCount(1)
  await expect(page.getByRole('dialog')).not.toContainText('\u95c7')
  await expect(page.getByRole('dialog')).not.toContainText('\u6769')
})
