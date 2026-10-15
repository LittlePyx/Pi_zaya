import { expect, test, type Page, type Route } from '@playwright/test'
import {
  installAppShellMocks,
  installEmptyCitationShelfMock,
  installIdleReferenceMocks,
} from './mockAppShell'

const CONV_ID = 'conv-local-semantic'
const SOURCE_PATH = 'db/Fixture/Semantic.en.md'
const FALLBACK_TRACE = {
  version: 1,
  status: 'done',
  retrieval: {
    bypassed: false,
    local_retrieval: {
      requested_mode: 'hybrid_local',
      used_mode: 'bm25',
      fallback_reason_id: 'dense_query_failed',
    },
  },
}

const conversation = {
  id: CONV_ID,
  title: 'Local semantic fixture',
  created_at: 1_780_000_000,
  updated_at: 1_780_000_060,
  project_id: null,
  mode: 'paper_guide',
  bound_source_path: SOURCE_PATH,
  bound_source_name: 'Semantic Fixture.pdf',
  bound_source_ready: true,
}

type PublicState =
  | 'not_installed'
  | 'preparing'
  | 'ready'
  | 'paused'
  | 'needs_space'
  | 'bm25_available'
  | 'removing'

async function fulfillJson(route: Route, body: unknown) {
  await route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify(body),
  })
}

function lifecyclePayload(state: PublicState) {
  const actionByState = {
    not_installed: 'enable',
    preparing: 'cancel',
    ready: 'remove',
    paused: 'resume',
    needs_space: 'free_space',
    bm25_available: 'none',
    removing: 'none',
  } as const
  return {
    schema_version: 2,
    default_mode: 'bm25',
    available_modes: state === 'ready' ? ['bm25', 'hybrid_local'] : ['bm25'],
    show_mode_control: true,
    product_enabled: true,
    install_available: true,
    component: {
      consent_scope: 'install_local_semantic_retrieval_component',
      archive_bytes: 299_558_739,
      installed_bytes: 538_888_249,
      network_required_for_install: true,
      library_content_uploaded: false,
      external_content_or_model_knowledge_added: false,
      bm25_remains_available: true,
    },
    hybrid_local: {
      state,
      available: state === 'ready',
      mode: 'hybrid_local',
      reason_id: `semantic_component_${state}`,
      action_id: actionByState[state],
      task_id: state === 'preparing' ? 'semantic-task-1' : '',
      completed_bytes: state === 'preparing' ? 149_779_370 : 0,
      total_bytes: state === 'preparing' ? 299_558_739 : 0,
      eta_seconds: state === 'preparing' ? 8 : null,
      required_additional_bytes: state === 'needs_space' ? 157_286_400 : 0,
      bm25_available: true,
    },
  }
}

async function installBackend(
  page: Page,
  initialState: PublicState,
  options: { fallbackAnswer?: boolean; bundled?: boolean; indexing?: boolean } = {},
) {
  let state = initialState
  let answerDone = false
  const generatePayloads: Array<Record<string, unknown>> = []
  const componentRequests: Array<{ method: string; path: string; body: unknown }> = []

  await installAppShellMocks(page, { rootConversations: [conversation] })
  await installEmptyCitationShelfMock(page, { scopeId: '__default__', projectId: null })
  await installIdleReferenceMocks(page)

  await page.route('**/api/settings', async (route) => {
    await fulfillJson(route, {
      model: 'test-model',
      base_url: '',
      has_api_key: true,
      connection: {
        text: { configured: true, connected: true, has_api_key: true, model: 'test-model', base_url: '' },
        vision: { configured: true, connected: true, has_api_key: true, model: 'test-vision', base_url: '' },
        web_search: { available: true, enabled: true, has_api_key: true },
        auto_route: false,
      },
      readiness: {
        overall: { status: 'ok', severity: 'ok', message: 'Ready' },
        providers: {
          text: { status: 'ok', severity: 'ok', configured: true, connected: true, has_api_key: true },
          vision: { status: 'ok', severity: 'ok', configured: true, connected: true, has_api_key: true },
        },
        issues: [],
      },
      prefs: { ui_locale: 'zh', theme: 'light', top_k: 6, temperature: 0.2, max_tokens: 1216 },
    })
  })
  await page.route('**/api/settings/readiness', async (route) => {
    await fulfillJson(route, {
      overall: { status: 'ok', severity: 'ok', message: 'Ready' },
      providers: {
        text: { status: 'ok', severity: 'ok', configured: true, connected: true, has_api_key: true },
        vision: { status: 'ok', severity: 'ok', configured: true, connected: true, has_api_key: true },
      },
      issues: [],
    })
  })
  await page.route(`**/api/conversations/${CONV_ID}`, async (route) => {
    await fulfillJson(route, conversation)
  })
  await page.route(`**/api/conversations/${CONV_ID}/research-state`, async (route) => {
    await fulfillJson(route, { ok: true, state: null })
  })
  await page.route(`**/api/conversations/${CONV_ID}/messages_page**`, async (route) => {
    await fulfillJson(route, {
      messages: answerDone ? [
        { id: 901, role: 'user', content: '定位本文中的机制证据', created_at: 1_780_000_061 },
        {
          id: 902,
          role: 'assistant',
          content: '这是使用普通检索完成的文献证据回答。',
          created_at: 1_780_000_062,
          meta: { research_trace: FALLBACK_TRACE },
        },
      ] : [],
      has_more_before: false,
      oldest_loaded_id: null,
      newest_loaded_id: null,
    })
  })
  await page.route(new RegExp(`/api/conversations/${CONV_ID}/messages(?:\\?.*)?$`), async (route) => {
    await fulfillJson(route, [])
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
  await page.route('**/api/references/bibliometrics', async (route) => {
    await fulfillJson(route, { bibliometrics_checked: true })
  })
  await page.route('**/api/library/quality/sources**', async (route) => {
    await fulfillJson(route, { items: [] })
  })
  await page.route('**/api/user-issues', async (route) => {
    await fulfillJson(route, { ok: true })
  })
  await page.route('**/api/retrieval/local-component**', async (route) => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    const body = request.postDataJSON() as unknown
    componentRequests.push({ method: request.method(), path, body })
    if (path.endsWith('/enable')) state = 'preparing'
    else if (path.endsWith('/cancel')) state = 'paused'
    else if (path.endsWith('/resume')) state = 'ready'
    else if (request.method() === 'DELETE') state = 'not_installed'
    const payload = lifecyclePayload(state)
    if (options.bundled) payload.component.network_required_for_install = false
    if (options.indexing && state === 'preparing') {
      payload.hybrid_local.reason_id = 'semantic_index_preparing'
      payload.hybrid_local.completed_bytes = 25
      payload.hybrid_local.total_bytes = 100
    }
    await fulfillJson(route, payload)
  })
  await page.route('**/api/generate', async (route) => {
    generatePayloads.push(route.request().postDataJSON() as Record<string, unknown>)
    await fulfillJson(route, {
      session_id: 'semantic-session',
      task_id: 'semantic-generation',
      user_msg_id: 901,
      assistant_msg_id: 902,
      started: Boolean(options.fallbackAnswer),
      ...(options.fallbackAnswer ? {} : { start_error: 'generation_start_failed' }),
    })
  })
  await page.route('**/api/generate/*/stream', async (route) => {
    answerDone = true
    const event = {
      stage: 'done',
      status: 'done',
      partial: '这是使用普通检索完成的文献证据回答。',
      answer: '这是使用普通检索完成的文献证据回答。',
      research_trace: FALLBACK_TRACE,
      done: true,
    }
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: `data: ${JSON.stringify(event)}\n\n`,
    })
  })

  return {
    generatePayloads,
    componentRequests,
    setState(next: PublicState) {
      state = next
    },
  }
}

async function openPaper(page: Page) {
  await page.goto(`/?conversation=${CONV_ID}`)
  await expect(page.getByRole('radio', { name: '本文' })).toBeChecked()
  await expect(page.getByTestId('chat-local-retrieval-control')).toBeVisible()
}

test('semantic indexing shows completed work rather than an archive percentage', async ({ page }) => {
  await installBackend(page, 'preparing', { indexing: true })
  await page.goto(`/?conversation=${CONV_ID}`)
  await expect(page.getByTestId('chat-local-retrieval-control')).toContainText('建立本地索引 25/100')
})

test('bundled component discloses offline installation and waits for explicit consent', async ({ page }) => {
  const backend = await installBackend(page, 'not_installed', { bundled: true })
  await openPaper(page)
  await page.getByTestId('chat-local-retrieval-control').click()
  await page.getByText('启用本地语义检索', { exact: true }).click()
  await expect(page.getByTestId('local-semantic-consent')).toContainText('安装无需联网')
  await expect(page.getByRole('button', { name: '下载并启用' })).toHaveCount(0)
  expect(backend.componentRequests.some((item) => item.path.endsWith('/enable'))).toBe(false)
  await page.getByRole('button', { name: '安装并启用' }).click()
  await expect.poll(() => backend.componentRequests.filter((item) => item.path.endsWith('/enable')).length).toBe(1)
  expect(backend.componentRequests.find((item) => item.path.endsWith('/enable'))?.body).toEqual({ consent: true })
})

test('explicit consent, pause, resume, and selected-paper query use local semantic retrieval', async ({ page }) => {
  const backend = await installBackend(page, 'not_installed')
  await openPaper(page)

  await expect(page.getByTestId('chat-local-retrieval-control')).toContainText('BM25')
  await page.getByTestId('chat-local-retrieval-control').click()
  await page.getByText('启用本地语义检索', { exact: true }).click()
  const consent = page.getByTestId('local-semantic-consent')
  await expect(consent).toContainText('286 MiB')
  await expect(consent).toContainText('514 MiB')
  await expect(consent).toContainText('不会启用网页或模型知识补充')
  await expect(page.getByTestId('chat-external-supplement-control')).toContainText('仅文献库')
  await page.getByRole('button', { name: '下载并启用' }).click()

  await expect(page.getByTestId('chat-local-retrieval-control')).toContainText('50%')
  await page.getByTestId('chat-local-retrieval-control').click()
  await page.getByText('暂停准备并继续使用 BM25', { exact: true }).click()
  await expect(page.getByTestId('chat-local-retrieval-control')).toContainText('已暂停')

  await page.getByTestId('chat-local-retrieval-control').click()
  await page.getByText('继续准备本地语义', { exact: true }).click()
  await expect(page.getByTestId('chat-local-retrieval-control')).toContainText('本地语义检索')

  await page.locator('textarea.kb-chat-textarea').fill('定位本文中的机制证据')
  await page.getByRole('button', { name: '发送' }).click()
  await expect.poll(() => backend.generatePayloads.length).toBe(1)
  expect(backend.generatePayloads[0].retrieval_mode).toBe('hybrid_local')
  expect(backend.generatePayloads[0].query_scope).toBe('current_paper')
  expect(backend.generatePayloads[0].external_supplement_mode).toBe('disabled')
})

test('space and internal failure states give a resolvable or automatic BM25 outcome', async ({ page }) => {
  const backend = await installBackend(page, 'needs_space')
  await openPaper(page)

  await expect(page.getByTestId('chat-local-retrieval-control')).toContainText('需要空间')
  await page.getByTestId('chat-local-retrieval-control').click()
  await expect(page.getByText(/还需至少 150 MiB/)).toBeVisible()
  await page.getByText('释放空间后继续', { exact: true }).click()
  await expect(page.getByTestId('chat-local-retrieval-control')).toContainText('本地语义检索')

  backend.setState('bm25_available')
  await page.reload()
  await openPaper(page)
  await page.getByTestId('chat-local-retrieval-control').click()
  await expect(page.getByText('本地语义组件当前不可用，无需核验或重试；普通检索不受影响')).toBeVisible()
  await expect(page.getByRole('menuitem', { name: /^重试$/ })).toHaveCount(0)
})

test('library scope stays on BM25 and removal keeps the external-source choice separate', async ({ page }) => {
  const backend = await installBackend(page, 'ready')
  await page.addInitScript(() => {
    window.localStorage.setItem('pi_zaya_local_retrieval_mode_v1', 'hybrid_local')
  })
  await openPaper(page)
  await expect(page.getByTestId('chat-local-retrieval-control')).toContainText('本地语义检索')

  await page.getByText('全库', { exact: true }).click()
  await expect(page.getByTestId('chat-local-retrieval-control')).toContainText('BM25')
  await expect(page.getByTestId('chat-external-supplement-control')).toContainText('仅文献库')

  await page.getByText('本文', { exact: true }).click()
  await page.getByTestId('chat-local-retrieval-control').click()
  await page.getByText('移除本地语义组件', { exact: true }).click()
  await expect(page.getByText('只移除可重新下载的组件和向量索引')).toBeVisible()
  await page.getByRole('button', { name: '移除组件' }).click()
  await expect(page.getByTestId('chat-local-retrieval-control')).toContainText('BM25')
  expect(backend.componentRequests.some((item) => item.method === 'DELETE')).toBe(true)
})

test('a requested semantic turn reports automatic BM25 fallback without a user repair task', async ({ page }) => {
  const backend = await installBackend(page, 'ready', { fallbackAnswer: true })
  await page.addInitScript(() => {
    window.localStorage.setItem('pi_zaya_local_retrieval_mode_v1', 'hybrid_local')
  })
  await openPaper(page)

  await page.locator('textarea.kb-chat-textarea').fill('定位本文中的机制证据')
  await page.getByRole('button', { name: '发送' }).click()

  const outcome = page.getByTestId('assistant-retrieval-fallback')
  await expect(outcome).toHaveText('已自动使用 BM25')
  await expect(outcome).toHaveAttribute('title', /无需核验或重试/)
  await expect(page.getByText('这是使用普通检索完成的文献证据回答。')).toBeVisible()
  await expect(page.getByText(/dense_query_failed/)).toHaveCount(0)
  await expect(page.getByRole('button', { name: /重试/ })).toHaveCount(0)
  expect(backend.generatePayloads).toHaveLength(1)
  expect(backend.generatePayloads[0].retrieval_mode).toBe('hybrid_local')
})

test('ordinary BM25 and intentional scope preservation do not create fallback noise', async ({ page }) => {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  const outcomes = await page.evaluate(async () => {
    const { hasAutomaticLocalSemanticFallback } = await import(
      '/src/components/chat/localRetrievalOutcome.ts'
    )
    const trace = (requestedMode: string, reason: string) => ({
      retrieval: {
        local_retrieval: {
          requested_mode: requestedMode,
          used_mode: 'bm25',
          fallback_reason_id: reason,
        },
      },
    })
    return {
      ordinary: hasAutomaticLocalSemanticFallback(trace('bm25', '')),
      fullLibrary: hasAutomaticLocalSemanticFallback(
        trace('hybrid_local', 'local_semantic_scope_not_eligible'),
      ),
      structured: hasAutomaticLocalSemanticFallback(
        trace('hybrid_local', 'structured_section_preserved'),
      ),
      runtimeFailure: hasAutomaticLocalSemanticFallback(
        trace('hybrid_local', 'component_lifecycle_not_ready'),
      ),
    }
  })
  expect(outcomes).toEqual({
    ordinary: false,
    fullLibrary: false,
    structured: false,
    runtimeFailure: true,
  })
})
