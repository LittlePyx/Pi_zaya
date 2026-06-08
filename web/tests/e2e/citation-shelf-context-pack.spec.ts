import { expect, test, type Page, type Route } from '@playwright/test'

const SHELF_ITEM = {
  key: 'ctx-ref-12',
  num: 12,
  displayNum: 12,
  anchor: 'ref-12',
  sourceName: 'Fixture Paper.pdf',
  sourcePath: 'db/Fixture/Fixture.en.md',
  title: 'Sparse 3-D transform-domain filtering',
  authors: 'K Dabov, A Foi, V Katkovnik',
  venue: 'IEEE Trans. Image Process.',
  year: '2007',
  doi: '10.1109/tip.2007.901238',
  summaryLine: 'A denoising baseline used for comparison.',
  shelfItemKind: 'reference',
  shelfOrigin: 'reader_references',
  shelfExcerpt: 'This reference is a baseline method for image denoising comparisons.',
  locationLabel: 'References / [12]',
  main: 'Sparse 3-D transform-domain filtering',
  tags: [],
  note: '',
}

const CONVERSATION = {
  id: 'conv-context-pack',
  title: 'Context pack conversation',
  created_at: 1,
  updated_at: 2,
  project_id: null,
  mode: 'normal',
}

const EXISTING_MESSAGES = [
  {
    id: 1,
    role: 'user',
    content: 'Collect the baseline reference.',
    created_at: 1,
  },
  {
    id: 2,
    role: 'assistant',
    content: 'The baseline reference is available in the research basket.',
    created_at: 2,
  },
]

async function fulfillJson(route: Route, body: unknown) {
  await route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify(body),
  })
}

async function installBackend(page: Page) {
  let created = true
  let generatePayload: Record<string, unknown> | null = null
  let generated = false
  let researchState: Record<string, unknown> = {}
  let resolveGenerate: (payload: Record<string, unknown>) => void = () => {}
  const generateSeen = new Promise<Record<string, unknown>>((resolve) => {
    resolveGenerate = resolve
  })

  await page.route('**/api/settings', async (route) => {
    await fulfillJson(route, {
      model: 'test-text-model',
      base_url: '',
      has_api_key: true,
      connection: {
        text: { configured: true, connected: true, has_api_key: true, model: 'test-text-model', base_url: '' },
        vision: { configured: true, connected: true, has_api_key: true, model: 'test-vision-model', base_url: '' },
        auto_route: false,
      },
      readiness: {
        overall: { status: 'ok', severity: 'ok', reason: 'Ready' },
        providers: {
          text: { status: 'ok', severity: 'ok', configured: true, connected: true, has_api_key: true, model: 'test-text-model', base_url: '' },
          vision: { status: 'ok', severity: 'ok', configured: true, connected: true, has_api_key: true, model: 'test-vision-model', base_url: '' },
        },
        issues: [],
      },
      prefs: {
        ui_locale: 'en',
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
      overall: { status: 'ok', severity: 'ok', reason: 'Ready' },
      providers: {
        text: { status: 'ok', severity: 'ok', configured: true, connected: true, has_api_key: true, model: 'test-text-model', base_url: '' },
        vision: { status: 'ok', severity: 'ok', configured: true, connected: true, has_api_key: true, model: 'test-vision-model', base_url: '' },
      },
      issues: [],
    })
  })
  await page.route('**/api/projects', async (route) => {
    await fulfillJson(route, [])
  })
  await page.route(/\/api\/conversations(?:\?.*)?$/, async (route) => {
    if (route.request().method() === 'POST') {
      created = true
      await fulfillJson(route, { id: 'conv-context-pack' })
      return
    }
    await fulfillJson(route, created ? [CONVERSATION] : [])
  })
  await page.route('**/api/conversations/conv-context-pack', async (route) => {
    await fulfillJson(route, CONVERSATION)
  })
  await page.route('**/api/conversations/conv-context-pack/research-state', async (route) => {
    const request = route.request()
    if (request.method() === 'PATCH') {
      const body = request.postDataJSON() as { state?: Record<string, unknown> } | null
      const patch = body?.state && typeof body.state === 'object' ? body.state : {}
      researchState = { ...researchState }
      for (const [key, value] of Object.entries(patch)) {
        if (value == null) {
          delete researchState[key]
        } else {
          researchState[key] = value
        }
      }
    }
    await fulfillJson(route, {
      conv_id: 'conv-context-pack',
      state: researchState,
      created_at: 1,
      updated_at: 2,
    })
  })
  await page.route('**/api/conversations/conv-context-pack/messages_page**', async (route) => {
    const generatedMessages = generated && generatePayload
      ? [
        ...EXISTING_MESSAGES,
        {
          id: 10,
          role: 'user',
          content: String(generatePayload.prompt || ''),
          created_at: 3,
          meta: { prompt_context: generatePayload.prompt_context },
        },
        {
          id: 11,
          role: 'assistant',
          content: 'The selected baseline is useful for comparison.',
          created_at: 4,
          meta: {
            paper_guide_contracts: {
              version: 1,
              selected_research_context: generatePayload.prompt_context,
            },
          },
        },
      ]
      : EXISTING_MESSAGES
    await fulfillJson(route, {
      messages: generatedMessages,
      has_more_before: false,
      oldest_loaded_id: 1,
      newest_loaded_id: generated ? 11 : 2,
    })
  })
  await page.route('**/api/references/conversation/**', async (route) => {
    await fulfillJson(route, {})
  })
  await page.route('**/api/chat/citation-shelf**', async (route) => {
    const request = route.request()
    const body = request.method() === 'PATCH'
      ? request.postDataJSON() as { items?: unknown[]; open?: boolean } | null
      : null
    await fulfillJson(route, {
      version: 1,
      scope: 'project',
      scope_id: '__default__',
      project_id: null,
      items: Array.isArray(body?.items) ? body?.items : [SHELF_ITEM],
      open: body?.open ?? true,
      revision: 3,
      created_at: 1,
      updated_at: 2,
    })
  })
  await page.route('**/api/references/citation-meta', async (route) => {
    await fulfillJson(route, {})
  })
  await page.route('**/api/references/bibliometrics', async (route) => {
    await fulfillJson(route, { bibliometrics_checked: true })
  })
  await page.route('**/api/library/quality/sources**', async (route) => {
    await fulfillJson(route, { items: [] })
  })
  await page.route('**/api/generate', async (route) => {
    generatePayload = route.request().postDataJSON() as Record<string, unknown>
    generated = true
    resolveGenerate(generatePayload)
    await fulfillJson(route, {
      session_id: 'session-context-pack',
      task_id: 'task-context-pack',
      trace_id: 'trace-context-pack',
      user_msg_id: 10,
      assistant_msg_id: 11,
      conversation_title: 'New context conversation',
    })
  })
  await page.route('**/api/generate/session-context-pack/stream', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: 'data: {"done":true,"status":"done","stage":"done","partial":"ok","answer":"ok","research_trace":{}}\n\n',
    })
  })

  return {
    generateSeen,
    getResearchState: () => researchState,
  }
}

test('selected citation shelf items are sent as next-turn prompt context', async ({ page }) => {
  const backend = await installBackend(page)
  await page.goto('/?conversation=conv-context-pack')

  await expect(page).toHaveURL(/conversation=conv-context-pack/)
  await expect(page.locator('body')).toContainText('The baseline reference is available')
  await expect(page.getByTestId('citation-shelf-item')).toHaveCount(1)
  await page.locator('.kb-shelf-check').first().check({ force: true })
  await expect(page.getByTestId('citation-shelf-use-context')).toBeVisible()
  await expect(page.getByTestId('citation-shelf-export-selected')).toBeVisible()
  await expect(page.getByTestId('citation-shelf-batch-organize-panel')).toHaveCount(0)
  await page.getByTestId('citation-shelf-batch-organize').click()
  await expect(page.getByTestId('citation-shelf-batch-organize-panel')).toBeVisible()
  await page.getByTestId('citation-shelf-use-context').click()
  await expect(page.getByTestId('chat-context-pack')).toContainText('1 excerpts')
  await expect(page.getByTestId('citation-shelf-context-badge')).toBeVisible()
  await expect.poll(() => Boolean(backend.getResearchState().selected_research_context)).toBe(true)

  await page.evaluate(() => {
    for (const key of Object.keys(window.localStorage)) {
      if (key.startsWith('kb:chat:selected-research-context:v1')) {
        window.localStorage.removeItem(key)
      }
    }
  })
  await page.reload()
  await expect(page).toHaveURL(/conversation=conv-context-pack/)
  await expect(page.locator('body')).toContainText('The baseline reference is available')
  await expect(page.getByTestId('chat-context-pack')).toContainText('1 excerpts')
  await expect(page.getByTestId('citation-shelf-context-badge')).toBeVisible()

  await page.getByTestId('chat-context-pack-clear').click()
  await expect(page.getByTestId('chat-context-pack')).toHaveCount(0)
  await expect.poll(() => Boolean(backend.getResearchState().selected_research_context)).toBe(false)
  await page.reload()
  await expect(page.getByTestId('chat-context-pack')).toHaveCount(0)

  await expect(page.getByTestId('citation-shelf-item')).toHaveCount(1)
  await page.locator('.kb-shelf-check').first().check({ force: true })
  await page.getByTestId('citation-shelf-use-context').click()
  await expect(page.getByTestId('chat-context-pack')).toContainText('1 excerpts')
  await expect.poll(() => Boolean(backend.getResearchState().selected_research_context)).toBe(true)

  await page.locator('textarea.kb-chat-textarea').fill('Compare with my selected baseline.')
  await page.getByRole('button', { name: 'Send' }).click()

  const payload = await backend.generateSeen
  const promptContext = payload.prompt_context as { items?: Array<Record<string, unknown>> } | undefined
  expect(promptContext?.items?.length).toBe(1)
  expect(promptContext?.items?.[0]?.title).toBe('Sparse 3-D transform-domain filtering')
  expect(promptContext?.items?.[0]?.summary).toContain('denoising baseline')
  expect(promptContext?.items?.[0]?.refNum).toBe(12)
  await expect(page.getByTestId('chat-context-pack')).toHaveCount(0)
  await expect.poll(() => Boolean(backend.getResearchState().selected_research_context)).toBe(false)

  await page.reload()
  await expect(page.getByTestId('chat-context-pack')).toHaveCount(0)

  await expect(page.getByTestId('research-context-receipt')).toContainText('Used research context (1)')
  await page.getByTestId('research-context-receipt-toggle').click()
  await expect(page.getByTestId('research-context-receipt-item')).toContainText('Sparse 3-D transform-domain filtering')
  await expect(page.getByTestId('research-context-receipt-item')).toContainText('References / [12]')

  await page.getByTestId('research-context-receipt-followup').click()
  await expect(page.getByTestId('chat-context-pack')).toContainText('1 excerpts')
  await expect(page.locator('textarea.kb-chat-textarea')).toHaveValue(/Sparse 3-D transform-domain filtering/)
})
