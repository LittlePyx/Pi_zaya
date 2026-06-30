import { expect, test, type Page, type Route } from '@playwright/test'
import {
  installAppShellMocks,
  installEmptyCitationShelfMock,
  installIdleReferenceMocks,
} from './mockAppShell'

type Conversation = {
  id: string
  title: string
  created_at: number
  updated_at: number
  project_id: string | null
  mode: 'normal'
}

const OLD_CONV_ID = 'conv-old-title'
const NEW_CONV_ID = 'conv-new-title'

async function fulfillJson(route: Route, body: unknown, status = 200) {
  await route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(body),
  })
}

async function installBackend(page: Page) {
  const conversations: Record<string, Conversation> = {
    [NEW_CONV_ID]: {
      id: NEW_CONV_ID,
      title: 'Newer Conversation',
      created_at: 2,
      updated_at: 20,
      project_id: null,
      mode: 'normal',
    },
    [OLD_CONV_ID]: {
      id: OLD_CONV_ID,
      title: 'Older Conversation',
      created_at: 1,
      updated_at: 10,
      project_id: null,
      mode: 'normal',
    },
  }
  const titleUpdates: Array<{ convId: string; title: string }> = []

  await installAppShellMocks(page)
  await installEmptyCitationShelfMock(page, { scopeId: '__default__', projectId: null })
  await installIdleReferenceMocks(page)

  await page.route('**/api/settings', async (route) => {
    await fulfillJson(route, {
      model: 'test-model',
      base_url: '',
      has_api_key: true,
      connection: {
        text: { has_api_key: true, model: 'test-model', base_url: '' },
        vision: { has_api_key: true, model: 'test-vision', base_url: '', uses_text_fallback: false },
        auto_route: false,
      },
      readiness: {
        overall: { status: 'ok', severity: 'ok', message: 'Ready' },
        providers: {},
        issues: [],
      },
      app_readiness: { status: 'ok', env: 'test', production: false, auth_required: false, items: [] },
      prefs: {
        ui_locale: 'en',
        theme: 'light',
        sidebar_collapsed: false,
        top_k: 6,
        temperature: 0.2,
        max_tokens: 1200,
        deep_read: false,
      },
    })
  })
  await page.route('**/api/sidebar**', async (route) => {
    await fulfillJson(route, {
      projects: [],
      root_conversations: Object.values(conversations).sort((a, b) => b.updated_at - a.updated_at),
      project_conversations: {},
    })
  })
  await page.route(/\/api\/conversations\/([^/]+)\/title$/, async (route) => {
    const match = route.request().url().match(/\/api\/conversations\/([^/]+)\/title$/)
    const convId = match?.[1] || ''
    const body = JSON.parse(route.request().postData() || '{}') as { title?: string }
    const nextTitle = String(body.title || '').trim()
    if (!conversations[convId]) {
      await fulfillJson(route, { detail: 'conversation not found' }, 404)
      return
    }
    conversations[convId] = { ...conversations[convId], title: nextTitle }
    titleUpdates.push({ convId, title: nextTitle })
    await fulfillJson(route, { ok: true })
  })
  return { titleUpdates }
}

test('renaming an older conversation does not move it above newer activity', async ({ page }) => {
  const backend = await installBackend(page)
  await page.goto('/')

  const rootRows = page.locator('.kb-root-conversations .kb-conv-row')
  await expect(rootRows).toHaveCount(2)
  await expect(rootRows.nth(0)).toContainText('Newer Conversation')
  await expect(rootRows.nth(1)).toContainText('Older Conversation')

  const olderRow = page.locator('.kb-root-conversations .kb-conv-row').filter({ hasText: 'Older Conversation' })
  await olderRow.getByLabel('Conversation actions').click()
  await page.getByRole('menuitem', { name: 'Rename' }).click()
  const titleInput = page.getByPlaceholder('Enter conversation title')
  await titleInput.fill('Renamed Older Conversation')
  await titleInput.press('Enter')

  await expect.poll(() => backend.titleUpdates).toContainEqual({
    convId: OLD_CONV_ID,
    title: 'Renamed Older Conversation',
  })
  await expect(rootRows.nth(0)).toContainText('Newer Conversation')
  await expect(rootRows.nth(1)).toContainText('Renamed Older Conversation')
})
