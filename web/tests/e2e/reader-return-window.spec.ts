import { expect, test, type Page, type Route } from '@playwright/test'
import {
  CHAT_MAIN_WINDOW_NAME,
  READER_STANDALONE_WINDOW_NAME,
} from '../../src/components/chat/reader/readerTypes'
import {
  buildReaderRegressionDocResponse,
  buildReaderRegressionPayload,
} from '../../src/testing/readerRegressionFixtures'
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

async function installReaderBackend(page: Page, options: {
  payload?: Record<string, unknown>
  document?: Record<string, unknown>
  state?: Record<string, unknown>
  statePatches?: Array<Record<string, unknown>>
} = {}) {
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
      payload: options.payload || {
        sourcePath: SOURCE_PATH,
        sourceName: 'Reader Return Fixture.pdf',
        headingPath: 'Introduction',
        snippet: 'Reader return fixture text.',
      },
      state: options.state || {
        sourcePath: SOURCE_PATH,
        conversationId: CONV_ID,
        highlights: [],
      },
      created_at: 1,
      updated_at: 1,
    })
  })
  await page.route(`**/api/reader/sessions/${SESSION_ID}/state`, async (route) => {
    options.statePatches?.push(route.request().postDataJSON() as Record<string, unknown>)
    await fulfillJson(route, {
      id: SESSION_ID,
      conversation_id: CONV_ID,
      title: 'Reader return fixture',
      payload: options.payload || {},
      state: (route.request().postDataJSON() as { state?: Record<string, unknown> }).state || {},
      created_at: 1,
      updated_at: 2,
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
    await fulfillJson(route, options.document || {
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

test('reader opened from a research note returns to that note and evidence card', async ({ page }) => {
  await installReaderBackend(page, {
    state: {
      sourcePath: SOURCE_PATH,
      highlights: [],
      researchNoteReturn: {
        version: 1,
        note_id: 'note-reader-return',
        source_index: 2,
      },
    },
  })

  await page.goto(`/reader/session/${SESSION_ID}`)
  await page.getByRole('button', { name: /Back|返回/ }).click()

  await expect(page).toHaveURL(/\/notes\?note=note-reader-return&evidence_source=2$/)
})

test('standalone reader automatically recovers a stale exact locator without a dead-end result', async ({ page }) => {
  const statePatches: Array<Record<string, unknown>> = []
  const modelRequests: string[] = []
  page.on('request', (request) => {
    const path = new URL(request.url()).pathname
    if (request.method() === 'POST' && /chat\/completions|\/api\/chat$/.test(path)) {
      modelRequests.push(path)
    }
  })
  await installReaderBackend(page, {
    payload: buildReaderRegressionPayload('strict-missing-exact') as unknown as Record<string, unknown>,
    document: buildReaderRegressionDocResponse('strict-missing-exact') as unknown as Record<string, unknown>,
    statePatches,
  })

  await page.goto(`/reader/session/${SESSION_ID}?conversation=${CONV_ID}`)

  await expect(page.getByTestId('reader-locate-resolution')).toHaveText(/^(Exact phrase|短语精确命中)$/)
  await expect(page.locator('.kb-reader-focus')).toContainText(
    'SCI compresses a short video into one coded measurement.',
  )
  await expect.poll(() => statePatches.some((patch) => {
    const state = patch.state as Record<string, unknown> | undefined
    const result = state?.locateResult as Record<string, unknown> | undefined
    return result?.status === 'exact' && result?.ok === true
  })).toBeTruthy()
  expect(statePatches.some((patch) => {
    const state = patch.state as Record<string, unknown> | undefined
    const result = state?.locateResult as Record<string, unknown> | undefined
    return result?.status === 'failed'
  })).toBe(false)
  expect(modelRequests).toEqual([])
})
