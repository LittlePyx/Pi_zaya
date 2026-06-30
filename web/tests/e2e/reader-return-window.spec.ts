import { expect, test, type Page, type Route } from '@playwright/test'
import {
  CHAT_MAIN_WINDOW_NAME,
  READER_STANDALONE_WINDOW_NAME,
} from '../../src/components/chat/reader/readerTypes'
import { installAppShellMocks } from './mockAppShell'

const SESSION_ID = 'reader-return-session'
const CONV_ID = 'reader-return-conv'
const SOURCE_PATH = 'db/Fixture/ReaderReturn.md'

async function fulfillJson(route: Route, body: unknown) {
  await route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify(body),
  })
}

async function installReaderBackend(page: Page) {
  await installAppShellMocks(page)
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
        providers: {
          text: { target: 'text', has_api_key: true, status: 'ok', severity: 'ok', model: 'test-model', base_url: '', reason: 'Ready' },
          vision: { target: 'vision', has_api_key: true, status: 'ok', severity: 'ok', model: 'test-vision', base_url: '', reason: 'Ready' },
        },
        overall: { status: 'ok', reason: 'Ready', target: '' },
      },
      prefs: { ui_locale: 'en', theme: 'light' },
    })
  })
  await page.route(`**/api/reader/sessions/${SESSION_ID}`, async (route) => {
    await fulfillJson(route, {
      id: SESSION_ID,
      conversation_id: CONV_ID,
      title: 'Reader return fixture',
      payload: {
        sourcePath: SOURCE_PATH,
        sourceName: 'Reader Return Fixture.pdf',
        headingPath: 'Introduction',
        snippet: 'Reader return fixture text.',
      },
      state: {
        sourcePath: SOURCE_PATH,
        conversationId: CONV_ID,
        highlights: [],
      },
      created_at: 1,
      updated_at: 1,
    })
  })
  await page.route(`**/api/conversations/${CONV_ID}/reader-state**`, async (route) => {
    await fulfillJson(route, {
      conv_id: CONV_ID,
      source_path: SOURCE_PATH,
      state: { highlights: [] },
      created_at: 1,
      updated_at: 1,
    })
  })
  await page.route('**/api/references/reader/doc', async (route) => {
    await fulfillJson(route, {
      source_path: SOURCE_PATH,
      source_name: 'Reader Return Fixture.pdf',
      markdown: '# Reader Return Fixture\n\nReader return fixture text.',
      anchors: [],
      blocks: [],
      cite_details: [],
      reference_cite_details: [],
    })
  })
}

test('reader session page does not keep the chat main window name', async ({ page }) => {
  await installReaderBackend(page)
  await page.addInitScript((name) => {
    window.name = name
  }, CHAT_MAIN_WINDOW_NAME)

  await page.goto(`/reader/session/${SESSION_ID}?conversation=${CONV_ID}`)

  await expect.poll(async () => page.evaluate(() => window.name)).toBe(READER_STANDALONE_WINDOW_NAME)
  await expect(page.getByRole('heading', { name: 'Reader Return Fixture', exact: true })).toBeVisible()
})
