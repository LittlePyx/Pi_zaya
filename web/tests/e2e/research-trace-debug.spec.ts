import { expect, test, type Page, type Route } from '@playwright/test'
import {
  installAppShellMocks,
  installEmptyCitationShelfMock,
  installIdleReferenceMocks,
} from './mockAppShell'

const CONV_ID = 'conv-trace-hidden'

async function fulfillJson(route: Route, body: unknown) {
  await route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify(body),
  })
}

async function installTraceMocks(page: Page) {
  const conversation = {
    id: CONV_ID,
    title: 'Trace hidden fixture',
    created_at: 1_780_000_000,
    updated_at: 1_780_000_060,
    project_id: null,
    mode: 'normal',
  }
  const messages = [
    {
      id: 901,
      role: 'user',
      content: 'Summarize the fixture.',
      created_at: 1_780_000_001,
    },
    {
      id: 902,
      role: 'assistant',
      content: 'Answer with hidden trace.',
      created_at: 1_780_000_002,
      meta: {
        research_trace: {
          trace_id: 'trace-should-not-render',
          timings_ms: { total: 42, retrieve: 10 },
          retrieval: { raw_hit_count: 3, top_hits: [{ source_name: 'Private Debug Source.pdf' }] },
        },
      },
    },
  ]

  await installAppShellMocks(page, { rootConversations: [conversation] })
  await installEmptyCitationShelfMock(page, { scopeId: '__default__', projectId: null })
  await installIdleReferenceMocks(page)

  await page.route('**/api/settings', async (route) => {
    await fulfillJson(route, {
      model: 'test-model',
      base_url: '',
      has_api_key: true,
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
      db_dir: '',
      prefs: {
        ui_locale: 'zh',
        theme: 'light',
        top_k: 6,
        temperature: 0.2,
        max_tokens: 1216,
        deep_read: false,
      },
    })
  })

  await page.route(/\/api\/conversations(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, [conversation])
  })
  await page.route(`**/api/conversations/${CONV_ID}`, async (route) => {
    await fulfillJson(route, conversation)
  })
  await page.route(`**/api/conversations/${CONV_ID}/research-state`, async (route) => {
    await fulfillJson(route, { ok: true, state: null })
  })
  await page.route(`**/api/conversations/${CONV_ID}/messages_page**`, async (route) => {
    await fulfillJson(route, {
      messages,
      has_more_before: false,
      oldest_loaded_id: 901,
      newest_loaded_id: 902,
    })
  })
  await page.route(new RegExp(`/api/conversations/${CONV_ID}/messages(?:\\?.*)?$`), async (route) => {
    await fulfillJson(route, messages)
  })
  await page.route(`**/api/references/conversation/${CONV_ID}`, async (route) => {
    await fulfillJson(route, {})
  })
  await page.route('**/api/references/citation-meta', async (route) => {
    await fulfillJson(route, {})
  })
  await page.route('**/api/references/citation-card-polish', async (route) => {
    await fulfillJson(route, {})
  })
}

test('default user build ignores research trace debug URL flags', async ({ page }) => {
  await installTraceMocks(page)

  await page.goto('/?debug_trace=1')
  const conversationRow = page.locator('.kb-conv-row', { hasText: 'Trace hidden fixture' })
  await expect(conversationRow).toBeVisible()
  await conversationRow.click()

  await expect(page.locator('body')).toContainText('Answer with hidden trace.')
  await expect(page.locator('.kb-research-trace')).toHaveCount(0)
  await expect(page.locator('body')).not.toContainText('trace-should-not-render')
  await expect(page.locator('body')).not.toContainText('Private Debug Source.pdf')
})

test('internal debug mode requires both general and trace-specific flags', async ({ page }) => {
  await installTraceMocks(page)

  await page.goto('/?debug=1&debug_trace=1')
  const debugBuildEnabled = await page.evaluate(async () => {
    const mod = await import('/src/utils/internalDebug.ts')
    return mod.internalDebugEnvEnabled()
  })
  test.skip(!debugBuildEnabled, 'positive trace assertion needs an internal debug build')

  const conversationRow = page.locator('.kb-conv-row', { hasText: 'Trace hidden fixture' })
  await expect(conversationRow).toBeVisible()
  await conversationRow.click()

  await expect(page.locator('body')).toContainText('Answer with hidden trace.')
  await expect(page.locator('.kb-research-trace')).toHaveCount(1)
  await expect(page.locator('body')).toContainText('trace-should-not-render')
  await expect(page.locator('body')).toContainText('Private Debug Source.pdf')
})
