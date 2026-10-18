import { expect, test, type Page, type Route } from '@playwright/test'
import { installAppShellMocks, installEmptyCitationShelfMock } from './mockAppShell'

const SHELF_ITEM = {
  key: 'ctx-ref-12',
  num: 12,
  displayNum: 12,
  anchor: 'ref-12',
  sourceName: 'Fixture Paper.pdf',
  sourcePath: 'db/Fixture/Fixture.en.md',
  headingPath: 'References',
  blockId: 'ref-block-12',
  anchorId: 'ref-12',
  anchorKind: 'reference',
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
  libraryMatchPath: 'db/Library/Sparse3D.en.md',
  libraryMatchStatus: 'ready',
  libraryMatchTitle: 'Sparse 3-D transform-domain filtering',
  libraryMatchDoi: '10.1109/tip.2007.901238',
  libraryMatchYear: '2007',
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

async function installBackend(
  page: Page,
  options: {
    initialShelfItems?: Array<Record<string, unknown>>
    initialShelfOpen?: boolean
    initialShelfRevision?: number
    settingsGate?: Promise<void>
  } = {},
) {
  let created = true
  let generatePayload: Record<string, unknown> | null = null
  let generated = false
  let researchState: Record<string, unknown> = {}
  const bibliometricsPayloads: Array<Record<string, unknown>> = []
  const generatePayloads: Record<string, unknown>[] = []
  const generateWaiters: Array<{
    count: number
    resolve: (payload: Record<string, unknown>) => void
  }> = []
  const waitForGenerateCount = (count: number) => {
    if (generatePayloads.length >= count) {
      return Promise.resolve(generatePayloads[count - 1])
    }
    return new Promise<Record<string, unknown>>((resolve) => {
      generateWaiters.push({ count, resolve })
    })
  }
  const generateSeen = waitForGenerateCount(1)

  await installAppShellMocks(page, { rootConversations: [CONVERSATION] })
  const citationShelf = await installEmptyCitationShelfMock(page, {
    scopeId: '__default__',
    projectId: null,
    initialItems: options.initialShelfItems ?? [SHELF_ITEM],
    initialOpen: options.initialShelfOpen ?? true,
    initialRevision: options.initialShelfRevision,
  })

  await page.route('**/api/settings', async (route) => {
    await options.settingsGate
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
  await page.route('**/api/references/citation-meta', async (route) => {
    await fulfillJson(route, {})
  })
  await page.route('**/api/references/bibliometrics', async (route) => {
    bibliometricsPayloads.push(route.request().postDataJSON() as Record<string, unknown>)
    await fulfillJson(route, { bibliometrics_checked: true })
  })
  await page.route('**/api/library/quality/sources**', async (route) => {
    await fulfillJson(route, { items: [] })
  })
  await page.route('**/api/generate', async (route) => {
    generatePayload = route.request().postDataJSON() as Record<string, unknown>
    generatePayloads.push(generatePayload)
    generated = true
    for (let idx = generateWaiters.length - 1; idx >= 0; idx -= 1) {
      const waiter = generateWaiters[idx]
      if (generatePayloads.length < waiter.count) continue
      generateWaiters.splice(idx, 1)
      waiter.resolve(generatePayloads[waiter.count - 1])
    }
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
    citationShelf,
    generateSeen,
    waitForGenerateCount,
    getBibliometricsPayloads: () => bibliometricsPayloads.map((payload) => ({ ...payload })),
    getResearchState: () => researchState,
  }
}

test('pagehide cannot flush an open stale empty shelf over restored backend evidence', async ({ page }) => {
  const backend = await installBackend(page, {
    initialShelfItems: [],
    initialShelfOpen: true,
    initialShelfRevision: 4,
  })
  await page.goto('/?conversation=conv-context-pack')

  await expect(page).toHaveURL(/conversation=conv-context-pack/)
  await expect(page.locator('body')).toContainText('The baseline reference is available')
  await expect(page.getByTestId('citation-shelf-item')).toHaveCount(0)
  await expect.poll(backend.citationShelf.getGetCount).toBeGreaterThan(0)

  backend.citationShelf.seed([SHELF_ITEM], true)
  const patchesBeforePagehide = backend.citationShelf.getPatchPayloads().length
  await page.evaluate(() => window.dispatchEvent(new PageTransitionEvent('pagehide')))
  await page.waitForTimeout(250)

  const pagehidePatches = backend.citationShelf.getPatchPayloads().slice(patchesBeforePagehide)
  expect(pagehidePatches.filter((payload) => (
    Array.isArray(payload.items)
    && payload.items.length === 0
    && !payload.allow_empty_overwrite
  ))).toEqual([])
  expect(backend.citationShelf.getState().items).toHaveLength(1)

  await page.reload()
  await expect(page.getByTestId('citation-shelf-item')).toHaveCount(1)
  await expect(page.getByTestId('citation-shelf-item')).toContainText('Fixture Paper.pdf')
  await expect.poll(() => backend.getBibliometricsPayloads().length).toBeGreaterThan(0)
  expect(backend.getBibliometricsPayloads().every((payload) => payload.allow_model_calls === false)).toBe(true)
})

test('a conversation deep link loads an existing answer outside the sidebar window', async ({ page }) => {
  await installBackend(page)
  await page.route('**/api/sidebar**', route => fulfillJson(route, { projects: [], root_conversations: [], project_conversations: {} }))
  await page.route(/\/api\/conversations(?:\?.*)?$/, route => fulfillJson(route, []))
  const creates: string[] = []
  page.on('request', request => {
    if (request.method() === 'POST' && /\/api\/(?:conversations|chat)(?:\?|$)/.test(new URL(request.url()).pathname)) creates.push(request.url())
  })
  await page.goto('/?conversation=conv-context-pack')
  await expect(page.locator('.kb-msg-bubble-assistant')).toContainText('The baseline reference is available')
  await expect(page).toHaveURL(/conversation=conv-context-pack/)
  expect(creates).toEqual([])
})

for (const mode of ['normal', 'paper_guide']) {
  test(`a cold ${mode} conversation fetches messages while metadata is pending`, async ({ page }) => {
    await installBackend(page)
    await page.route('**/api/sidebar**', route => fulfillJson(route, { projects: [], root_conversations: [], project_conversations: {} }))
    let releaseMetadata = () => {}
    const metadataGate = new Promise<void>(resolve => { releaseMetadata = resolve })
    const messageQueries: string[] = []
    await page.route('**/api/conversations/conv-context-pack', async route => {
      await metadataGate
      await fulfillJson(route, { ...CONVERSATION, mode })
    })
    await page.route('**/api/conversations/conv-context-pack/messages_page**', async route => {
      messageQueries.push(new URL(route.request().url()).search)
      await fulfillJson(route, { messages: EXISTING_MESSAGES, has_more_before: false, oldest_loaded_id: 1, newest_loaded_id: 2 })
    })
    await page.goto('/?conversation=conv-context-pack')
    try {
      await expect.poll(() => messageQueries.length).toBe(1)
      // The server selects its stored conversation mode before metadata arrives.
      expect(new URLSearchParams(messageQueries[0]).has('render_packet_only')).toBe(false)
    } finally {
      releaseMetadata()
    }
    await expect(page.locator('.kb-msg-bubble-assistant')).toContainText('The baseline reference is available')
    expect(messageQueries).toHaveLength(1)
  })
}

test('initial settings hydration shares the conversation load and a later locale change waits for it', async ({ page }) => {
  let releaseSettings = () => {}
  let releaseMetadata = () => {}
  const settingsGate = new Promise<void>(resolve => { releaseSettings = resolve })
  const metadataGate = new Promise<void>(resolve => { releaseMetadata = resolve })
  await installBackend(page, { settingsGate })
  await page.route('**/api/sidebar**', route => fulfillJson(route, { projects: [], root_conversations: [], project_conversations: {} }))
  await page.route('**/api/conversations/conv-context-pack', async route => {
    await metadataGate
    await fulfillJson(route, CONVERSATION)
  })
  let messageRequests = 0
  await page.route('**/api/conversations/conv-context-pack/messages_page**', async route => {
    messageRequests += 1
    releaseSettings()
    await fulfillJson(route, { messages: EXISTING_MESSAGES, has_more_before: false, oldest_loaded_id: 1, newest_loaded_id: 2 })
  })
  await page.goto('/?conversation=conv-context-pack')
  try {
    await expect.poll(() => page.evaluate(async () => {
      const { useSettingsStore } = await import('/src/stores/settingsStore.ts')
      return useSettingsStore.getState().loaded
    })).toBe(true)
    await page.evaluate(async () => { await new Promise(requestAnimationFrame); await new Promise(requestAnimationFrame) })
    expect(messageRequests).toBe(1)
    await page.evaluate(async () => {
      const { useSettingsStore } = await import('/src/stores/settingsStore.ts')
      const state = useSettingsStore.getState()
      useSettingsStore.setState({ uiLocale: 'zh', refsCardLocale: 'zh', localePreferencesRevision: state.localePreferencesRevision + 1 })
      await new Promise(requestAnimationFrame); await new Promise(requestAnimationFrame)
    })
    expect(messageRequests).toBe(1)
  } finally {
    releaseSettings()
    releaseMetadata()
  }
  await expect(page.locator('.kb-msg-bubble-assistant')).toContainText('The baseline reference is available')
  await expect.poll(() => messageRequests).toBe(2)
})

test('an unavailable conversation deep link does not loop or create a conversation', async ({ page }) => {
  await installBackend(page)
  let lookups = 0
  let creates = 0
  await page.route('**/api/conversations/missing-history', async route => {
    lookups += 1
    await route.fulfill({ status: 404, contentType: 'application/json', body: '{"detail":"Not found"}' })
  })
  await page.route('**/api/conversations/missing-history/messages**', route => route.fulfill({ status: 404, contentType: 'application/json', body: '{"detail":"Not found"}' }))
  page.on('request', request => {
    if (request.method() === 'POST' && /\/api\/(?:conversations|chat)$/.test(new URL(request.url()).pathname)) creates += 1
  })
  await page.goto('/?conversation=missing-history')
  await expect.poll(() => lookups).toBe(1)
  await expect(page.locator('.kb-msg-bubble-assistant')).toHaveCount(0)
  await page.waitForTimeout(500)
  expect(lookups).toBe(1)
  expect(creates).toBe(0)
})

test('conversation deep links reject route fragments', async ({ page }) => {
  await installBackend(page)
  const requests: string[] = []
  page.on('request', request => {
    if (new URL(request.url()).pathname.startsWith('/api/conversations/')) requests.push(request.url())
  })
  await page.goto('/?conversation=..%2Fsettings')
  await expect(page.locator('.kb-sider-search-input')).toBeVisible()
  expect(requests).toEqual([])
})

test('a receipt click survives basket maintenance finishing between pointer down and up', async ({ page }) => {
  await installBackend(page)
  let releaseMaintenance = () => {}
  const maintenanceGate = new Promise<void>(resolve => { releaseMaintenance = resolve })
  for (const endpoint of ['reader-locators/maintain', 'metadata/repair']) {
    await page.route(`**/api/references/shelf/${endpoint}`, async route => {
      await maintenanceGate
      await fulfillJson(route, { items: [] })
    })
  }
  await page.route('**/api/conversations/conv-context-pack/messages_page**', route => fulfillJson(route, {
    messages: [EXISTING_MESSAGES[0], {
      ...EXISTING_MESSAGES[1],
      meta: { paper_guide_contracts: { selected_research_context: {
        version: 1, id: 'receipt-click', source: 'citation_shelf', createdAt: 1,
        conversationId: CONVERSATION.id,
        items: [{ ...SHELF_ITEM, kind: 'reference', refNum: 12, excerpt: SHELF_ITEM.shelfExcerpt }],
      } } },
    }],
    has_more_before: false, oldest_loaded_id: 1, newest_loaded_id: 2,
  }))
  await page.goto('/?conversation=conv-context-pack')
  const toggle = page.getByTestId('research-context-receipt-toggle')
  try {
    await expect(toggle).toBeVisible()
    await expect(page.getByTestId('chat-activity-shelf')).toBeVisible()
    const box = await toggle.boundingBox()
    expect(box).not.toBeNull()
    await page.mouse.move(box!.x + box!.width / 2, box!.y + box!.height / 2)
    await page.mouse.down()
    releaseMaintenance()
    await expect(page.getByTestId('chat-activity-shelf')).toHaveCount(0)
    await page.mouse.up()
    await expect(toggle).toHaveAttribute('aria-expanded', 'true')
    await expect(page.getByTestId('research-context-receipt-item')).toContainText(SHELF_ITEM.title)
    // Keyboard activation still toggles the same receipt after the task ends.
    await toggle.press('Enter')
    await expect(toggle).toHaveAttribute('aria-expanded', 'false')
  } finally {
    releaseMaintenance()
    await page.mouse.up()
  }
})

test('selected citation shelf items are sent as next-turn prompt context', async ({ page }) => {
  test.setTimeout(60_000)
  const backend = await installBackend(page)
  await page.goto('/?conversation=conv-context-pack')

  await expect(page).toHaveURL(/conversation=conv-context-pack/)
  await expect(page.locator('body')).toContainText('The baseline reference is available')
  await expect(page.getByTestId('citation-shelf-item')).toHaveCount(1)
  await page.locator('.kb-shelf-check').first().check()
  await expect(page.locator('.kb-shelf-check').first()).toBeChecked()
  await expect(page.getByTestId('citation-shelf-use-context')).toBeVisible()
  await expect(page.getByTestId('citation-shelf-export-selected')).toBeVisible()
  await expect(page.getByTestId('citation-shelf-batch-organize-panel')).toHaveCount(0)
  await page.getByTestId('citation-shelf-batch-organize').click()
  await expect(page.getByTestId('citation-shelf-batch-organize-panel')).toBeVisible()
  await page.getByTestId('citation-shelf-use-context').click()
  await expect(page.getByTestId('chat-context-pack')).toContainText('1 excerpts')
  await expect(page.getByTestId('citation-shelf-context-badge')).toBeVisible()
  await expect.poll(() => Boolean(backend.getResearchState().selected_research_context)).toBe(true)
  await expect.poll(() => {
    const pack = backend.getResearchState().selected_research_context as { items?: Array<{ summary?: string }> } | undefined
    return pack?.items?.[0]?.summary || ''
  }).toContain('denoising baseline')

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
  await page.locator('.kb-shelf-check').first().check()
  await expect(page.locator('.kb-shelf-check').first()).toBeChecked()
  await page.getByTestId('citation-shelf-use-context').click()
  await expect(page.getByTestId('chat-context-pack')).toContainText('1 excerpts')
  await expect.poll(() => Boolean(backend.getResearchState().selected_research_context)).toBe(true)
  await expect.poll(() => {
    const pack = backend.getResearchState().selected_research_context as { items?: Array<{ summary?: string }> } | undefined
    return pack?.items?.[0]?.summary || ''
  }).toContain('denoising baseline')

  await page.locator('textarea.kb-chat-textarea').fill('Compare with my selected baseline.')
  await page.getByRole('button', { name: 'Send' }).click()

  const payload = await backend.generateSeen
  const promptContext = payload.prompt_context as { items?: Array<Record<string, unknown>> } | undefined
  expect(promptContext?.items?.length).toBe(1)
  expect(promptContext?.items?.[0]?.title).toBe('Sparse 3-D transform-domain filtering')
  expect(promptContext?.items?.[0]?.summary).toContain('denoising baseline')
  expect(promptContext?.items?.[0]?.refNum).toBe(12)
  expect(promptContext?.items?.[0]?.blockId).toBe('ref-block-12')
  expect(promptContext?.items?.[0]?.libraryMatchPath).toBe('db/Library/Sparse3D.en.md')
  expect(payload.query_scope).toBe('basket')
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

  await page.getByRole('button', { name: 'Send' }).click()
  const followupPayload = await backend.waitForGenerateCount(2)
  const followupContext = followupPayload.prompt_context as { items?: Array<Record<string, unknown>> } | undefined
  expect(followupPayload.query_scope).toBe('basket')
  expect(followupContext?.items?.length).toBe(1)
  expect(followupContext?.items?.[0]?.title).toBe('Sparse 3-D transform-domain filtering')
})
