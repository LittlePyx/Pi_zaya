import type { Page, Route } from '@playwright/test'

async function fulfillJson(route: Route, body: unknown, status = 200) {
  await route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(body),
  })
}

function postJson<T>(route: Route): T | undefined {
  try {
    return route.request().postDataJSON() as T
  } catch {
    return undefined
  }
}

export async function installAppShellMocks(
  page: Page,
  options: {
    authStatus?: Record<string, unknown>
    readiness?: Record<string, unknown>
    updateCheck?: Record<string, unknown>
    projects?: unknown[]
    rootConversations?: unknown[]
    projectConversations?: Record<string, unknown[]>
  } = {},
) {
  await page.route('**/api/auth/status', async (route) => {
    await fulfillJson(route, options.authStatus || {
      required: false,
      configured: false,
      authenticated: true,
      env: 'test',
      production: false,
    })
  })
  await page.route('**/api/readiness', async (route) => {
    await fulfillJson(route, options.readiness || { status: 'ok', checks: [] })
  })
  await page.route('**/api/app/update-check**', async (route) => {
    await fulfillJson(route, options.updateCheck || { ok: true, update_available: false })
  })
  await page.route('**/api/sidebar**', async (route) => {
    await fulfillJson(route, {
      projects: options.projects || [],
      root_conversations: options.rootConversations || [],
      project_conversations: options.projectConversations || {},
    })
  })
  await page.route('**/api/projects', async (route) => {
    await fulfillJson(route, options.projects || [])
  })
}

export async function installIdleReferenceMocks(page: Page) {
  const context = page.context()
  await context.route('**/api/references/sync/status', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: 'data: {"running":false,"done":true,"status":"idle","stage":"","message":"","current":"","docs_done":0,"docs_total":0}\n\n',
    })
  })
  await context.route('**/api/references/shelf/metadata/backfill/status', async (route) => {
    await fulfillJson(route, {
      ok: true,
      status: 'idle',
      running: false,
      progress: { percent: 0, processed: 0, total: 0 },
    })
  })
  await context.route('**/api/references/shelf/metadata/repair', async (route) => {
    const payload = postJson<{ items?: unknown[] }>(route)
    const requested = Array.isArray(payload?.items) ? payload.items.length : 0
    await fulfillJson(route, {
      ok: true,
      requested,
      ready: requested,
      partial: 0,
      retryable: 0,
      failed: 0,
      changed: 0,
      persisted: 0,
      export_ready: requested,
      unresolved: 0,
      items: [],
    })
  })
  await context.route('**/api/references/reader/doc**', async (route) => {
    await fulfillJson(route, { detail: 'reader source fixture not configured' }, 404)
  })
}

export async function installEmptyCitationShelfMock(
  page: Page,
  options: {
    scopeId?: string
    projectId?: string | null
    initialItems?: Array<Record<string, unknown>>
    initialOpen?: boolean
  } = {},
) {
  let items = (options.initialItems || []).map((item) => ({ ...item }))
  let open = Boolean(options.initialOpen)
  let revision = items.length > 0 ? 1 : 0
  const scopeId = options.scopeId || options.projectId || 'e2e-project'
  const projectId = options.projectId === undefined ? String(scopeId) : options.projectId
  const fulfillShelf = async (route: Route) => {
    await fulfillJson(route, {
      version: 1,
      scope: 'project',
      scope_id: scopeId,
      project_id: projectId,
      items,
      open,
      revision,
      created_at: 0,
      updated_at: revision,
    })
  }

  await page.context().route('**/api/chat/citation-shelf**', async (route) => {
    const request = route.request()
    if (request.method() === 'POST' && new URL(request.url()).pathname.endsWith('/items')) {
      const payload = postJson<{ item?: Record<string, unknown>; open?: boolean }>(route)
      if (payload?.item && typeof payload.item === 'object') {
        items = [payload.item, ...items]
        revision += 1
      }
      open = payload?.open ?? true
      await fulfillShelf(route)
      return
    }
    if (request.method() === 'PATCH') {
      const payload = postJson<{ items?: unknown[]; open?: boolean }>(route)
      if (Array.isArray(payload?.items)) {
        items = payload.items.filter((item): item is Record<string, unknown> => Boolean(item && typeof item === 'object'))
        revision += 1
      }
      open = Boolean(payload?.open)
      await fulfillShelf(route)
      return
    }
    if (request.method() === 'DELETE') {
      items = []
      open = false
      revision += 1
      await fulfillShelf(route)
      return
    }
    await fulfillShelf(route)
  })
}
