import { expect, test, type Page, type Route } from '@playwright/test'

const CONV_ID = 'conv-src-persistence'
const USER_MSG_ID = 301
const ASSISTANT_MSG_ID = 302
const SOURCE_MD_PATH = 'db/Fixture/Fixture.en.md'

const conversation = {
  id: CONV_ID,
  title: 'Src persistence regression',
  created_at: 1_780_000_000,
  updated_at: 1_780_000_060,
  project_id: null,
  mode: 'paper_guide',
  bound_source_path: SOURCE_MD_PATH,
  bound_source_name: 'Fixture Paper.pdf',
  bound_source_ready: true,
}

const userMessage = {
  id: USER_MSG_ID,
  role: 'user',
  content: 'Where is the claim supported?',
  created_at: 1_780_000_001,
}

const answerMarkdown = 'The claim is supported by the fixture evidence [1](#src-refresh-a1).'

const assistantMessage = {
  id: ASSISTANT_MSG_ID,
  role: 'assistant',
  refs_user_msg_id: USER_MSG_ID,
  content: answerMarkdown,
  cite_details: [
    {
      num: 1,
      anchor: 'src-refresh-a1',
      source_name: 'Fixture Paper.pdf',
      source_path: SOURCE_MD_PATH,
      title: 'Fixture evidence',
      heading_path: 'Results / Evidence',
      answer_claim: 'The answer cites fixture evidence.',
      evidence_quote: 'The fixture evidence supports the claim.',
      support_relation: 'This is the source chip backing the answer.',
      block_id: 'fixture-block-1',
      anchor_id: 'fixture-anchor-1',
      anchor_kind: 'paragraph',
      page_start: 3,
      is_inpaper: false,
    },
  ],
  meta: {
    paper_guide_contracts: {
      version: 1,
      render_packet: {
        answer_markdown: answerMarkdown,
        rendered_body: answerMarkdown,
        rendered_content: answerMarkdown,
        copy_text: answerMarkdown,
        copy_markdown: answerMarkdown,
        cite_details: [],
      },
    },
  },
  created_at: 1_780_000_002,
}

async function fulfillJson(route: Route, body: unknown) {
  await route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify(body),
  })
}

async function installMockBackend(page: Page, options: { messagesPageDelayMs?: number; hideConversationFromList?: boolean } = {}) {
  let renderPacketOnlyPageLoads = 0

  await page.route('**/api/settings', async (route) => {
    await fulfillJson(route, {
      model: 'test-model',
      base_url: '',
      has_api_key: true,
      connection: {
        text: { configured: true, connected: true, has_api_key: true, model: 'test-model', base_url: '' },
        vision: { configured: true, connected: true, has_api_key: true, model: 'test-vision', base_url: '' },
        auto_route: false,
      },
      readiness: {
        ok: true,
        overall: { status: 'ok', severity: 'ok', message: 'Ready' },
        providers: {
          text: { status: 'ok', severity: 'ok', configured: true, connected: true, has_api_key: true, model: 'test-model', base_url: '' },
          vision: { status: 'ok', severity: 'ok', configured: true, connected: true, has_api_key: true, model: 'test-vision', base_url: '' },
        },
        issues: [],
      },
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

  await page.route('**/api/settings/readiness', async (route) => {
    await fulfillJson(route, {
      ok: true,
      overall: { status: 'ok', severity: 'ok', message: 'Ready' },
      providers: {
        text: { status: 'ok', severity: 'ok', configured: true, connected: true, has_api_key: true, model: 'test-model', base_url: '' },
        vision: { status: 'ok', severity: 'ok', configured: true, connected: true, has_api_key: true, model: 'test-vision', base_url: '' },
      },
      issues: [],
    })
  })

  await page.route('**/api/projects', async (route) => {
    await fulfillJson(route, [])
  })

  await page.route(/\/api\/conversations(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, options.hideConversationFromList ? [] : [conversation])
  })

  await page.route(`**/api/conversations/${CONV_ID}`, async (route) => {
    await fulfillJson(route, conversation)
  })

  await page.route(`**/api/conversations/${CONV_ID}/messages_page**`, async (route) => {
    const url = new URL(route.request().url())
    if (url.searchParams.get('render_packet_only') === '1') renderPacketOnlyPageLoads += 1
    if (options.messagesPageDelayMs && options.messagesPageDelayMs > 0) {
      await new Promise((resolve) => {
        setTimeout(resolve, options.messagesPageDelayMs)
      })
    }
    await fulfillJson(route, {
      messages: [userMessage, assistantMessage],
      has_more_before: false,
      oldest_loaded_id: USER_MSG_ID,
      newest_loaded_id: ASSISTANT_MSG_ID,
    })
  })

  await page.route(new RegExp(`/api/conversations/${CONV_ID}/messages(?:\\?.*)?$`), async (route) => {
    await fulfillJson(route, [userMessage, assistantMessage])
  })

  await page.route(`**/api/references/conversation/${CONV_ID}`, async (route) => {
    await fulfillJson(route, {})
  })

  await page.route('**/api/chat/citation-shelf**', async (route) => {
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
  })

  await page.route('**/api/references/citation-meta', async (route) => {
    await fulfillJson(route, {})
  })

  await page.route('**/api/references/reader/doc', async (route) => {
    await fulfillJson(route, {
      source_path: SOURCE_MD_PATH,
      source_name: 'Fixture Paper.pdf',
      markdown: '# Fixture Paper\n\nThe fixture evidence supports the claim.',
      anchors: [],
      blocks: [],
      cite_details: [],
      reference_cite_details: [],
    })
  })

  await page.route('**/api/references/citation-card-polish', async (route) => {
    await fulfillJson(route, {})
  })

  await page.route('**/api/references/bibliometrics', async (route) => {
    await fulfillJson(route, { bibliometrics_checked: true })
  })

  return {
    renderPacketOnlyPageLoads: () => renderPacketOnlyPageLoads,
  }
}

async function openConversationAndExpectSrcChip(page: Page) {
  const conversationRow = page.locator('.kb-conv-row', { hasText: conversation.title })
  await expect(conversationRow).toBeVisible()
  await conversationRow.click()
  await expect(page.locator('body')).toContainText('The claim is supported by the fixture evidence')
  const chip = page.locator('.kb-cite-chip').first()
  await expect(chip).toBeVisible()
  await expect(chip).toHaveText('1')
}

test('paper guide src chips survive refresh when render packet only uses top-level cite details', async ({ page }) => {
  const backend = await installMockBackend(page)

  await page.goto('/')
  await openConversationAndExpectSrcChip(page)
  expect(backend.renderPacketOnlyPageLoads()).toBeGreaterThanOrEqual(1)

  await page.reload()
  await openConversationAndExpectSrcChip(page)
  expect(backend.renderPacketOnlyPageLoads()).toBeGreaterThanOrEqual(2)
})

test('paper guide src chips survive direct URL restore without sidebar conversation cache', async ({ page }) => {
  const backend = await installMockBackend(page, { hideConversationFromList: true })

  await page.goto(`/?conversation=${CONV_ID}`)
  await expect(page.getByTestId('research-context-state')).toHaveAttribute('data-research-conversation-id', CONV_ID)
  await expect(page.getByTestId('research-context-state')).toHaveAttribute('data-research-mode', 'paper_guide')
  await expect(page.locator('body')).toContainText('The claim is supported by the fixture evidence')
  const chip = page.locator('.kb-cite-chip').first()
  await expect(chip).toBeVisible()
  await expect(chip).toHaveText('1')
  expect(backend.renderPacketOnlyPageLoads()).toBeGreaterThanOrEqual(1)
})

test('chat activity strip exposes slow conversation loading and opt-in perf status', async ({ page }) => {
  await installMockBackend(page, { messagesPageDelayMs: 1000 })

  await page.goto('/?debug=1')
  await expect(page.getByTestId('chat-perf-panel')).toBeVisible()

  const conversationRow = page.locator('.kb-conv-row', { hasText: conversation.title })
  await expect(conversationRow).toBeVisible()
  await conversationRow.click()

  await expect(page.getByTestId('chat-activity-strip')).toBeVisible()
  await expect(page.getByTestId('chat-activity-messages')).toBeVisible()
  await expect(page.locator('body')).toContainText('The claim is supported by the fixture evidence')
})
