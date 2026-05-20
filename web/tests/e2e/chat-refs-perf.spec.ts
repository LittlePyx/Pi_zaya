import { expect, test, type Page, type Route } from '@playwright/test'

const CONV_ID = 'conv-live-refs'
const USER_MSG_ID = 101
const ASSISTANT_MSG_ID = 102
const SOURCE_PDF_PATH = '__chat_refs_perf__/Fixture.pdf'
const SOURCE_MD_PATH = '__chat_refs_perf__/Fixture.en.md'

const conversation = {
  id: CONV_ID,
  title: 'Live refs regression',
  created_at: 1_700_000_000,
  updated_at: 1_700_000_000,
  project_id: null,
  mode: 'paper_guide',
  bound_source_path: SOURCE_PDF_PATH,
  bound_source_name: 'Fixture Paper.pdf',
  bound_source_ready: true,
}

const userMessage = {
  id: USER_MSG_ID,
  role: 'user',
  content: 'Which paper compares Hadamard and Fourier single-pixel imaging?',
  created_at: 1_700_000_001,
}

const assistantMessage = {
  id: ASSISTANT_MSG_ID,
  role: 'assistant',
  refs_user_msg_id: USER_MSG_ID,
  content: 'The direct match is Fixture Paper because it compares Hadamard and Fourier single-pixel imaging.',
  rendered_body: 'The direct match is Fixture Paper because it compares Hadamard and Fourier single-pixel imaging.',
  copy_text: 'The direct match is Fixture Paper because it compares Hadamard and Fourier single-pixel imaging.',
  copy_markdown: 'The direct match is Fixture Paper because it compares Hadamard and Fourier single-pixel imaging.',
  created_at: 1_700_000_002,
}

const assistantMessageWithProvenance = {
  ...assistantMessage,
  provenance: {
    status: 'ready',
    source_path: SOURCE_MD_PATH,
    source_name: 'Fixture Paper.pdf',
    strict_identity_ready: true,
    mapping_mode: 'fast',
    must_locate_candidate_count: 1,
    must_locate_count: 1,
    strict_identity_count: 1,
    block_map: {
      'p-delayed': {
        block_id: 'p-delayed',
        anchor_id: 'a-p-delayed',
        kind: 'paragraph',
        heading_path: '3. Comparison / 3.1 Numerical simulations',
        text: 'Hadamard single-pixel imaging and Fourier single-pixel imaging are compared in the simulation section.',
      },
    },
    segments: [
      {
        segment_id: 'seg-delayed-provenance',
        segment_index: 0,
        kind: 'paragraph',
        segment_type: 'paragraph',
        claim_type: 'critical_fact_claim',
        must_locate: true,
        locate_policy: 'required',
        locate_surface_policy: 'primary',
        text: 'The paper compares Hadamard and Fourier single-pixel imaging in numerical simulations.',
        snippet_key: 'hadamard fourier single pixel imaging simulations',
        evidence_mode: 'direct',
        evidence_block_ids: ['p-delayed'],
        primary_block_id: 'p-delayed',
        primary_anchor_id: 'a-p-delayed',
        primary_heading_path: '3. Comparison / 3.1 Numerical simulations',
        evidence_quote: 'Hadamard single-pixel imaging and Fourier single-pixel imaging are compared in the simulation section.',
        evidence_confidence: 0.94,
        anchor_kind: 'paragraph',
        anchor_text: 'Hadamard single-pixel imaging and Fourier single-pixel imaging are compared in the simulation section.',
      },
    ],
  },
}

function refsPayload(state: 'pending' | 'ready') {
  const pending = state === 'pending'
  return {
    [String(USER_MSG_ID)]: {
      prompt: userMessage.content,
      display_state: pending ? 'pending' : 'ready',
      payload_mode: pending ? 'pending' : 'fast',
      enrichment_pending: pending,
      hits: [
        {
          text: 'Hadamard single-pixel imaging and Fourier single-pixel imaging are compared in the simulation section.',
          meta: {
            source_path: SOURCE_MD_PATH,
            ref_pack_state: pending ? 'pending' : 'ready',
          },
          ui_meta: {
            display_name: 'Fixture Paper.pdf',
            source_path: SOURCE_MD_PATH,
            heading_path: '3. Comparison / 3.1 Numerical simulations',
            summary_line: pending
              ? 'A provisional card is available while final reference copy is still being refined.'
              : 'The section compares Hadamard and Fourier single-pixel imaging in numerical simulations.',
            why_line: 'This card is tied to the same user question and should not require switching conversations to appear.',
            score: pending ? null : 8.6,
            score_pending: pending,
            reader_open: {
              sourcePath: SOURCE_MD_PATH,
              sourceName: 'Fixture Paper.pdf',
              headingPath: '3. Comparison / 3.1 Numerical simulations',
              snippet: 'Hadamard single-pixel imaging and Fourier single-pixel imaging are compared.',
              highlightSnippet: 'Hadamard single-pixel imaging and Fourier single-pixel imaging are compared.',
              strictLocate: !pending,
            },
          },
        },
      ],
    },
  }
}

async function fulfillJson(route: Route, body: unknown, headers?: Record<string, string>) {
  await route.fulfill({
    status: 200,
    contentType: 'application/json',
    headers,
    body: JSON.stringify(body),
  })
}

async function installMockChatBackend(page: Page) {
  let generatePosted = false
  let generationDone = false
  let refsCalls = 0
  let messagePageCallsAfterDone = 0

  await page.route('**/api/settings', async (route) => {
    if (route.request().method() === 'PATCH') {
      await fulfillJson(route, { ok: true })
      return
    }
    await fulfillJson(route, {
      model: 'test-model',
      base_url: '',
      has_api_key: true,
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

  await page.route('**/api/projects', async (route) => {
    await fulfillJson(route, [])
  })

  await page.route(/\/api\/conversations(?:\?.*)?$/, async (route) => {
    if (route.request().method() === 'POST') {
      await fulfillJson(route, { id: CONV_ID })
      return
    }
    await fulfillJson(route, [conversation])
  })

  await page.route(/\/api\/conversations\/conv-live-refs$/, async (route) => {
    await fulfillJson(route, conversation)
  })

  await page.route(/\/api\/conversations\/conv-live-refs\/messages_page(?:\?.*)?$/, async (route) => {
    if (generationDone) {
      messagePageCallsAfterDone += 1
    }
    const assistant = generationDone && messagePageCallsAfterDone >= 2
      ? assistantMessageWithProvenance
      : assistantMessage
    await fulfillJson(route, {
      messages: generationDone ? [userMessage, assistant] : [],
      has_more_before: false,
      oldest_loaded_id: generationDone ? USER_MSG_ID : null,
      newest_loaded_id: generationDone ? ASSISTANT_MSG_ID : null,
    })
  })

  await page.route(/\/api\/conversations\/conv-live-refs\/messages(?:\?.*)?$/, async (route) => {
    const assistant = generationDone && messagePageCallsAfterDone >= 2
      ? assistantMessageWithProvenance
      : assistantMessage
    await fulfillJson(route, generationDone ? [userMessage, assistant] : [])
  })

  await page.route('**/api/generate', async (route) => {
    generatePosted = true
    await fulfillJson(route, {
      session_id: 'session-live-refs',
      task_id: 'task-live-refs',
      user_msg_id: USER_MSG_ID,
      assistant_msg_id: ASSISTANT_MSG_ID,
    })
  })

  await page.route('**/api/generate/session-live-refs/stream', async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 4000))
    generationDone = true
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: [
        'data: {"stage":"drafting","partial":"Drafting answer...","done":false}',
        '',
        'data: {"stage":"done","partial":"Final answer is ready.","done":true}',
        '',
      ].join('\n'),
    })
  })

  await page.route(`**/api/references/conversation/${CONV_ID}`, async (route) => {
    refsCalls += 1
    const state = generatePosted && refsCalls >= 3 ? 'ready' : 'pending'
    await fulfillJson(route, generatePosted ? refsPayload(state) : {}, {
      'server-timing': `total;dur=${state === 'ready' ? 18 : 9}, fast_render;dur=4`,
      'x-kb-refs-mode': state === 'ready' ? 'fast' : 'pending',
      'x-kb-refs-counts': generatePosted
        ? `packs=1,hits=1,pending=${state === 'pending' ? 1 : 0}`
        : 'packs=0,hits=0,pending=0',
    })
  })

  await page.route('**/api/references/citation-meta', async (route) => {
    await fulfillJson(route, {})
  })

  return {
    getRefsCalls: () => refsCalls,
    getMessagePageCallsAfterDone: () => messagePageCallsAfterDone,
  }
}

test('refs cards render during generation and perf logs prove polling continued', async ({ page }) => {
  const backend = await installMockChatBackend(page)

  await page.goto('/')
  const conversationRow = page.locator('.kb-conv-row', { hasText: 'Live refs regression' })
  await expect(conversationRow).toHaveCount(1)
  await conversationRow.click()
  await expect(page.locator('textarea.kb-chat-textarea, .kb-chat-textarea textarea')).toBeVisible()
  await page.locator('textarea.kb-chat-textarea, .kb-chat-textarea textarea').fill(userMessage.content)
  await page.locator('button.kb-send-btn').click()

  await expect(page.locator('button.kb-stop-btn')).toBeVisible({ timeout: 5_000 })
  await expect(page.locator('.kb-refs-panel')).toBeVisible({ timeout: 5_000 })
  await page.locator('.kb-refs-panel .ant-collapse-header').click()
  await expect(page.locator('.kb-ref-title')).toContainText('Fixture Paper.pdf', { timeout: 5_000 })
  await expect(page.locator('button.kb-stop-btn')).toBeVisible()

  await expect(page.locator('button.kb-stop-btn')).toHaveCount(0, { timeout: 10_000 })
  await expect(page.locator('body')).toContainText('The direct match is Fixture Paper', { timeout: 5_000 })
  await expect.poll(
    () => backend.getMessagePageCallsAfterDone(),
    { timeout: 5_000 },
  ).toBeGreaterThanOrEqual(2)
  const locateChip = page.locator('.kb-prov-locate-chip').first()
  await expect(locateChip).toBeVisible({ timeout: 5_000 })
  await expect(locateChip).toHaveAttribute('data-kb-locate-block-id', 'p-delayed')
  await expect(page.locator('.kb-ref-title')).toContainText('Fixture Paper.pdf')
  await expect(page.locator('.kb-ref-score')).toContainText('相关分 8.60')

  const refsSummary = await page.evaluate(() => window.__kbRefsPerf?.summary())
  const refsLogs = await page.evaluate(() => window.__kbRefsPerf?.getLogs() || [])
  const pollSuccesses = refsLogs.filter((event) => event.phase === 'poll_success')
  const generationStartedEvents = refsLogs.filter((event) => event.reason === 'generation_started')

  expect(backend.getRefsCalls()).toBeGreaterThanOrEqual(3)
  expect(backend.getMessagePageCallsAfterDone()).toBeGreaterThanOrEqual(2)
  expect(refsSummary?.fetchSuccess).toBeGreaterThanOrEqual(2)
  expect(refsSummary?.lastMode).toBe('fast')
  expect(refsSummary?.lastCounts).toContain('packs=1')
  expect(generationStartedEvents.length).toBeGreaterThan(0)
  expect(pollSuccesses.some((event) => event.keepPolling === true)).toBe(true)
  expect(refsLogs.some((event) => event.summary?.pendingPackCount === 1)).toBe(true)
  expect(refsLogs.some((event) => event.summary?.fastPackCount === 1)).toBe(true)
})
