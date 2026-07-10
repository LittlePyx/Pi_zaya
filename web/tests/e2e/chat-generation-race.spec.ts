import { expect, test, type Locator, type Page, type Route } from '@playwright/test'
import {
  installAppShellMocks,
  installEmptyCitationShelfMock,
  installIdleReferenceMocks,
} from './mockAppShell'

const CONV_A_ID = 'conv-generation-race-a'
const CONV_B_ID = 'conv-generation-race-b'
const A_GENERATED_ANSWER = 'ANSWER_FROM_A_GENERATION'
const A_CANCELED_ANSWER = 'Generation canceled by user.'
const A_START_FAILED_ANSWER = 'Generation could not be started. Please retry.'
const A_START_FAILED_ANSWER_ZH = '回答任务未能启动，请稍后重试。'
const A_STREAM_FAILED_ANSWER = 'Answer stream failed. Please retry.'
const A_STREAM_FAILED_ANSWER_ZH = '回答连接中断，请稍后重试。'
const A_STREAM_INCOMPLETE_ANSWER = 'Answer was interrupted before completion. Please retry.'
const A_REFRESH_FAILED_ANSWER = 'Answer finished, but the latest message could not be refreshed. Reopen the conversation to load it.'
const B_STABLE_ANSWER = 'Conversation B stable answer stays visible.'
const UPLOAD_PDF_NAME = 'switch-upload.pdf'
const UPLOAD_JOB_ID = 'job-upload-switch-a'

type ReportedIssue = {
  source?: string
  domain?: string
  severity?: string
  summary?: string
  detail?: string
  route?: string
  context?: Record<string, unknown>
  payload?: Record<string, unknown>
  fingerprint?: string
}

const convA = {
  id: CONV_A_ID,
  title: 'Generation Race A',
  created_at: 1,
  updated_at: 20,
  project_id: null,
  mode: 'normal',
}

const convB = {
  id: CONV_B_ID,
  title: 'Generation Race B',
  created_at: 2,
  updated_at: 19,
  project_id: null,
  mode: 'normal',
}

async function fulfillJson(route: Route, body: unknown, headers?: Record<string, string>) {
  await route.fulfill({
    status: 200,
    contentType: 'application/json',
    headers,
    body: JSON.stringify(body),
  })
}

function messagePage(messages: unknown[]) {
  return {
    messages,
    has_more_before: false,
    oldest_loaded_id: messages.length > 0 ? 1 : null,
    newest_loaded_id: messages.length > 0 ? messages.length : null,
  }
}

async function deleteConversationFromRow(page: Page, row: Locator) {
  const confirmButton = page.getByRole('button', { name: /^(OK|确定)$/ })
  for (let attempt = 0; attempt < 2; attempt += 1) {
    await row.getByRole('button', { name: 'Conversation actions' }).click()
    const deleteItem = page.getByRole('menuitem', { name: /Delete|删除/ })
    await expect(deleteItem).toBeVisible()
    await deleteItem.click({ force: attempt > 0 })
    try {
      await expect(confirmButton).toBeVisible({ timeout: 1_500 })
      await confirmButton.click()
      return
    } catch {
      await page.keyboard.press('Escape').catch(() => {})
    }
  }
  await expect(confirmButton).toBeVisible()
  await confirmButton.click()
}

async function installBackend(
  page: Page,
  opts?: {
    streamFailure?: boolean
    streamIncomplete?: boolean
    terminalStreamError?: boolean
    completionMessageFailure?: boolean
    generateStartFailure?: boolean
    uiLocale?: 'en' | 'zh'
  },
) {
  let releaseStream: (() => void) | null = null
  let generationDone = false
  let generationCanceled = false
  let generationStartFailed = false
  let convADonePageLoads = 0
  let generationStreamCalls = 0
  const generatePayloads: Array<Record<string, unknown>> = []
  const userIssueReports: ReportedIssue[] = []
  let uploadReady = false
  let uploadStatusCalls = 0
  let uploadCancelCalls = 0
  let generationCancelCalls = 0
  const deletedConversationIds = new Set<string>()
  const streamReleased = new Promise<void>((resolve) => {
    releaseStream = resolve
  })

  await installAppShellMocks(page, { rootConversations: [convA, convB] })
  await installEmptyCitationShelfMock(page, { scopeId: '__default__', projectId: null })
  await installIdleReferenceMocks(page)

  await page.route('**/api/user-issues', async (route) => {
    if (route.request().method() === 'POST') {
      userIssueReports.push(route.request().postDataJSON() as ReportedIssue)
    }
    await fulfillJson(route, { ok: true, issue: { id: 'issue-chat-generation-race' } })
  })

  await page.route('**/api/settings', async (route) => {
    if (route.request().method() === 'PATCH') {
      await fulfillJson(route, { ok: true })
      return
    }
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
        overall: { status: 'ok', severity: 'ok', message: 'Ready' },
        providers: {
          text: { status: 'ok', severity: 'ok', configured: true, connected: true, has_api_key: true, model: 'test-model', base_url: '' },
          vision: { status: 'ok', severity: 'ok', configured: true, connected: true, has_api_key: true, model: 'test-vision', base_url: '' },
        },
        issues: [],
      },
      prefs: {
        ui_locale: opts?.uiLocale || 'en',
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
      overall: { status: 'ok', severity: 'ok', message: 'Ready' },
      providers: {
        text: { status: 'ok', severity: 'ok', configured: true, connected: true, has_api_key: true, model: 'test-model', base_url: '' },
        vision: { status: 'ok', severity: 'ok', configured: true, connected: true, has_api_key: true, model: 'test-vision', base_url: '' },
      },
      issues: [],
    })
  })

  await page.route('**/api/sidebar**', async (route) => {
    await fulfillJson(route, {
      projects: [],
      root_conversations: [convA, convB].filter((conv) => !deletedConversationIds.has(conv.id)),
      project_conversations: {},
    })
  })

  await page.route(`**/api/conversations/${CONV_A_ID}`, async (route) => {
    if (route.request().method() === 'DELETE') {
      deletedConversationIds.add(CONV_A_ID)
      await fulfillJson(route, { ok: true })
      return
    }
    if (deletedConversationIds.has(CONV_A_ID)) {
      await route.fulfill({
        status: 404,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'conversation not found' }),
      })
      return
    }
    await fulfillJson(route, convA)
  })

  await page.route(`**/api/conversations/${CONV_B_ID}`, async (route) => {
    if (route.request().method() === 'DELETE') {
      deletedConversationIds.add(CONV_B_ID)
      await fulfillJson(route, { ok: true })
      return
    }
    if (deletedConversationIds.has(CONV_B_ID)) {
      await route.fulfill({
        status: 404,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'conversation not found' }),
      })
      return
    }
    await fulfillJson(route, convB)
  })

  await page.route(`**/api/conversations/${CONV_A_ID}/messages_page**`, async (route) => {
    if (generationDone || generationStartFailed) convADonePageLoads += 1
    if (generationDone && opts?.completionMessageFailure) {
      await route.fulfill({
        status: 503,
        contentType: 'text/plain',
        body: 'messages page temporarily unavailable',
      })
      return
    }
    await fulfillJson(route, messagePage(generationStartFailed
      ? [
          { id: 101, role: 'user', content: 'Question for A', created_at: 3 },
          { id: 102, role: 'assistant', content: A_START_FAILED_ANSWER, created_at: 4 },
        ]
      : generationDone
      ? [
          { id: 101, role: 'user', content: 'Question for A', created_at: 3 },
          { id: 102, role: 'assistant', content: generationCanceled ? A_CANCELED_ANSWER : A_GENERATED_ANSWER, created_at: 4 },
        ]
      : [
          { id: 1, role: 'user', content: 'Existing question A', created_at: 1 },
          { id: 2, role: 'assistant', content: 'Existing answer A', created_at: 2 },
        ]))
  })

  await page.route(new RegExp(`/api/conversations/${CONV_A_ID}/messages(?:\\?.*)?$`), async (route) => {
    if (generationDone && opts?.completionMessageFailure) {
      await route.fulfill({
        status: 503,
        contentType: 'text/plain',
        body: 'messages fallback temporarily unavailable',
      })
      return
    }
    await fulfillJson(route, generationStartFailed
      ? [
          { id: 101, role: 'user', content: 'Question for A', created_at: 3 },
          { id: 102, role: 'assistant', content: A_START_FAILED_ANSWER, created_at: 4 },
        ]
      : generationDone
      ? [
          { id: 101, role: 'user', content: 'Question for A', created_at: 3 },
          { id: 102, role: 'assistant', content: generationCanceled ? A_CANCELED_ANSWER : A_GENERATED_ANSWER, created_at: 4 },
        ]
      : [
          { id: 1, role: 'user', content: 'Existing question A', created_at: 1 },
          { id: 2, role: 'assistant', content: 'Existing answer A', created_at: 2 },
        ])
  })

  await page.route(`**/api/conversations/${CONV_B_ID}/messages_page**`, async (route) => {
    await fulfillJson(route, messagePage([
      { id: 11, role: 'user', content: 'Existing question B', created_at: 1 },
      { id: 12, role: 'assistant', content: B_STABLE_ANSWER, created_at: 2 },
    ]))
  })

  await page.route(`**/api/conversations/${CONV_A_ID}/research-state`, async (route) => {
    await fulfillJson(route, { conv_id: CONV_A_ID, state: {}, created_at: 1, updated_at: 1 })
  })

  await page.route(`**/api/conversations/${CONV_B_ID}/research-state`, async (route) => {
    await fulfillJson(route, { conv_id: CONV_B_ID, state: {}, created_at: 1, updated_at: 1 })
  })

  await page.route('**/api/chat/uploads/status**', async (route) => {
    uploadStatusCalls += 1
    await fulfillJson(route, {
      items: [{
        kind: 'pdf',
        status: 'saved',
        name: UPLOAD_PDF_NAME,
        sha1: 'upload-sha1',
        path: '/fake/switch-upload.pdf',
        md_path: uploadReady ? '/fake/switch-upload.md' : '',
        ingest_job_id: UPLOAD_JOB_ID,
        ready: uploadReady,
        ingest_status: uploadReady ? 'ready' : 'converting',
        quality_status: 'none',
      }],
    })
  })

  await page.route('**/api/chat/uploads/cancel', async (route) => {
    uploadCancelCalls += 1
    await fulfillJson(route, {
      item: {
        kind: 'pdf',
        status: 'error',
        name: UPLOAD_PDF_NAME,
        sha1: 'upload-sha1',
        path: '/fake/switch-upload.pdf',
        md_path: '',
        ingest_job_id: UPLOAD_JOB_ID,
        ready: false,
        ingest_status: 'cancelled',
        quality_status: 'cancelled',
      },
    })
  })

  await page.route('**/api/chat/uploads', async (route) => {
    await fulfillJson(route, {
      items: [{
        kind: 'pdf',
        status: 'saved',
        name: UPLOAD_PDF_NAME,
        sha1: 'upload-sha1',
        path: '/fake/switch-upload.pdf',
        md_path: '',
        ingest_job_id: UPLOAD_JOB_ID,
        ready: false,
        ingest_status: 'converting',
        quality_status: 'none',
      }],
    })
  })

  await page.route('**/api/generate', async (route) => {
    const body = route.request().postDataJSON() as { conv_id?: string } | null
    generatePayloads.push(body || {})
    expect(body?.conv_id).toBe(CONV_A_ID)
    if (opts?.generateStartFailure) {
      generationStartFailed = true
      await fulfillJson(route, {
        session_id: 'session-generation-race-a',
        task_id: 'task-generation-race-a',
        user_msg_id: 101,
        assistant_msg_id: 102,
        started: false,
        start_error: 'generation_start_failed',
      })
      return
    }
    await fulfillJson(route, {
      session_id: 'session-generation-race-a',
      task_id: 'task-generation-race-a',
      user_msg_id: 101,
      assistant_msg_id: 102,
    })
  })

  await page.route('**/api/generate/session-generation-race-a/cancel**', async (route) => {
    generationCancelCalls += 1
    generationCanceled = true
    generationDone = true
    await fulfillJson(route, { ok: true })
  })

  await page.route('**/api/generate/session-generation-race-a/stream', async (route) => {
    generationStreamCalls += 1
    if (opts?.streamFailure) {
      await route.fulfill({
        status: 502,
        contentType: 'text/plain',
        body: 'stream temporarily unavailable',
      })
      return
    }
    await streamReleased
    generationDone = true
    try {
      if (opts?.terminalStreamError) {
        await route.fulfill({
          status: 200,
          contentType: 'text/event-stream',
          body: [
            'data: {"stage":"error","status":"error","partial":"Generation stream failed.","error":"Generation stream failed.","done":true}',
            '',
          ].join('\n'),
        })
        return
      }
      if (opts?.streamIncomplete) {
        await route.fulfill({
          status: 200,
          contentType: 'text/event-stream',
          body: [
            `data: {"stage":"drafting","partial":"${A_GENERATED_ANSWER}","done":false}`,
            '',
          ].join('\n'),
        })
        return
      }
      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: [
          `data: {"stage":"drafting","partial":"${A_GENERATED_ANSWER}","done":false}`,
          '',
          `data: {"stage":"done","partial":"${A_GENERATED_ANSWER}","done":true}`,
          '',
        ].join('\n'),
      })
    } catch {
      // The client may abort the SSE request when cancelling or deleting the conversation.
    }
  })

  await page.route(`**/api/references/conversation/${CONV_A_ID}`, async (route) => {
    await fulfillJson(route, {}, {
      'server-timing': 'total;dur=1',
      'x-kb-refs-mode': 'empty',
      'x-kb-refs-counts': 'packs=0,hits=0,pending=0',
    })
  })

  await page.route(`**/api/references/conversation/${CONV_B_ID}`, async (route) => {
    await fulfillJson(route, {}, {
      'server-timing': 'total;dur=1',
      'x-kb-refs-mode': 'empty',
      'x-kb-refs-counts': 'packs=0,hits=0,pending=0',
    })
  })

  return {
    releaseStream: () => releaseStream?.(),
    getConvADonePageLoads: () => convADonePageLoads,
    getGenerationStreamCalls: () => generationStreamCalls,
    getGeneratePayloads: () => generatePayloads.slice(),
    getUserIssueReports: () => userIssueReports.slice(),
    markUploadReady: () => {
      uploadReady = true
    },
    getUploadStatusCalls: () => uploadStatusCalls,
    getUploadCancelCalls: () => uploadCancelCalls,
    getGenerationCancelCalls: () => generationCancelCalls,
  }
}

test('default chat send does not request research agent mode', async ({ page }) => {
  const backend = await installBackend(page)
  const prompt = 'Plain default question'

  await page.goto('/')
  await page.locator('.kb-conv-row', { hasText: 'Generation Race A' }).click()
  await expect(page.locator('body')).toContainText('Existing answer A')
  await expect(page.locator('button.kb-agent-mode-btn')).toContainText('Standard')

  await page.locator('textarea.kb-chat-textarea, .kb-chat-textarea textarea').fill(prompt)
  await page.locator('button.kb-send-btn').click()

  await expect.poll(() => backend.getGeneratePayloads().length, { timeout: 5_000 }).toBe(1)
  const payload = backend.getGeneratePayloads()[0]
  expect(payload.agent_mode).toBeUndefined()

  const userMeta = await page.evaluate(async (content) => {
    const { useChatStore } = await import('/src/stores/chatStore.ts')
    const message = useChatStore.getState().messages.find((item) => item.role === 'user' && item.content === content)
    return message?.meta || null
  }, prompt)
  expect(userMeta).not.toHaveProperty('agent_mode')
  expect(userMeta).not.toHaveProperty('agent_mode_requested')

  backend.releaseStream()
  await expect(page.locator('button.kb-stop-btn')).toHaveCount(0, { timeout: 5_000 })
})

test('agent mode URL parameter enables the composer toggle and request flag', async ({ page }) => {
  const backend = await installBackend(page)
  const prompt = 'Agent URL question'

  await page.goto('/?agent_mode=1')
  await page.locator('.kb-conv-row', { hasText: 'Generation Race A' }).click()
  await expect(page.locator('body')).toContainText('Existing answer A')
  await expect(page.locator('button.kb-agent-mode-btn')).toContainText('Research')

  await page.locator('textarea.kb-chat-textarea, .kb-chat-textarea textarea').fill(prompt)
  await page.locator('button.kb-send-btn').click()

  await expect.poll(() => backend.getGeneratePayloads().length, { timeout: 5_000 }).toBe(1)
  const payload = backend.getGeneratePayloads()[0]
  expect(payload.agent_mode).toBe(true)

  const userMeta = await page.evaluate(async (content) => {
    const { useChatStore } = await import('/src/stores/chatStore.ts')
    const message = useChatStore.getState().messages.find((item) => item.role === 'user' && item.content === content)
    return message?.meta || null
  }, prompt)
  expect(userMeta).toMatchObject({
    agent_mode: 'research_agent',
    agent_mode_requested: true,
  })

  backend.releaseStream()
  await expect(page.locator('button.kb-stop-btn')).toHaveCount(0, { timeout: 5_000 })
})

test('agent mode toggle is remembered per conversation without leaking across switches', async ({ page }) => {
  await installBackend(page)
  const agentButton = page.locator('button.kb-agent-mode-btn')

  await page.goto('/')
  await page.locator('.kb-conv-row', { hasText: 'Generation Race A' }).click()
  await expect(page.locator('body')).toContainText('Existing answer A')
  await expect(agentButton).toContainText('Standard')

  await agentButton.click()
  await expect(agentButton).toContainText('Research')

  await page.locator('.kb-conv-row', { hasText: 'Generation Race B' }).click()
  await expect(page.locator('body')).toContainText(B_STABLE_ANSWER)
  await expect(agentButton).toContainText('Standard')

  await page.locator('.kb-conv-row', { hasText: 'Generation Race A' }).click()
  await expect(page.locator('body')).toContainText('Existing answer A')
  await expect(agentButton).toContainText('Research')
})

test('stale generation stream cannot overwrite the active conversation after switching', async ({ page }) => {
  const backend = await installBackend(page)

  await page.goto('/')
  await page.locator('.kb-conv-row', { hasText: 'Generation Race A' }).click()
  await expect(page.locator('body')).toContainText('Existing answer A')

  await page.locator('textarea.kb-chat-textarea, .kb-chat-textarea textarea').fill('Question for A')
  await page.locator('button.kb-send-btn').click()
  await expect(page.locator('button.kb-stop-btn')).toBeVisible({ timeout: 5_000 })

  await page.locator('.kb-conv-row', { hasText: 'Generation Race B' }).click()
  await expect(page.locator('body')).toContainText(B_STABLE_ANSWER)
  await expect(page.locator('button.kb-stop-btn')).toHaveCount(0)

  backend.releaseStream()
  await expect.poll(
    () => backend.getConvADonePageLoads(),
    { timeout: 5_000 },
  ).toBeGreaterThanOrEqual(1)

  await expect(page.locator('body')).toContainText(B_STABLE_ANSWER)
  await expect(page.locator('body')).not.toContainText(A_GENERATED_ANSWER)
})

test('in-flight generation resumes when switching back to its conversation', async ({ page }) => {
  const backend = await installBackend(page)

  await page.goto('/')
  await page.locator('.kb-conv-row', { hasText: 'Generation Race A' }).click()
  await expect(page.locator('body')).toContainText('Existing answer A')

  await page.locator('textarea.kb-chat-textarea, .kb-chat-textarea textarea').fill('Question for A')
  await page.locator('button.kb-send-btn').click()
  await expect(page.locator('button.kb-stop-btn')).toBeVisible({ timeout: 5_000 })

  await page.locator('.kb-conv-row', { hasText: 'Generation Race B' }).click()
  await expect(page.locator('body')).toContainText(B_STABLE_ANSWER)
  await expect(page.locator('button.kb-stop-btn')).toHaveCount(0)

  await page.locator('.kb-conv-row', { hasText: 'Generation Race A' }).click()
  await expect(page.locator('button.kb-stop-btn')).toBeVisible({ timeout: 5_000 })

  backend.releaseStream()
  await expect.poll(
    () => backend.getConvADonePageLoads(),
    { timeout: 5_000 },
  ).toBeGreaterThanOrEqual(1)

  await expect(page.locator('button.kb-stop-btn')).toHaveCount(0, { timeout: 5_000 })
  await expect(page.locator('body')).toContainText(A_GENERATED_ANSWER)
})

test('deleting an in-flight generation cancels the backend task', async ({ page }) => {
  const backend = await installBackend(page)

  await page.goto('/')
  await page.locator('.kb-conv-row', { hasText: 'Generation Race A' }).click()
  await expect(page.locator('body')).toContainText('Existing answer A')

  await page.locator('textarea.kb-chat-textarea, .kb-chat-textarea textarea').fill('Question for A')
  await page.locator('button.kb-send-btn').click()
  await expect(page.locator('button.kb-stop-btn')).toBeVisible({ timeout: 5_000 })

  const conversationRow = page.locator('.kb-conv-row', { hasText: 'Generation Race A' })
  await deleteConversationFromRow(page, conversationRow)

  await expect.poll(() => backend.getGenerationCancelCalls(), { timeout: 5_000 }).toBe(1)
  await expect(page.locator('button.kb-stop-btn')).toHaveCount(0)
  await expect(page.locator('.kb-conv-row', { hasText: 'Generation Race A' })).toHaveCount(0)
  await expect(page.locator('body')).not.toContainText('Question for A')

  backend.releaseStream()
  await expect(page.locator('body')).not.toContainText(A_GENERATED_ANSWER)
})

test('canceling an in-flight generation refreshes the stored canceled assistant message', async ({ page }) => {
  const backend = await installBackend(page)

  await page.goto('/')
  await page.locator('.kb-conv-row', { hasText: 'Generation Race A' }).click()
  await expect(page.locator('body')).toContainText('Existing answer A')

  await page.locator('textarea.kb-chat-textarea, .kb-chat-textarea textarea').fill('Question for A')
  await page.locator('button.kb-send-btn').click()
  await expect(page.locator('button.kb-stop-btn')).toBeVisible({ timeout: 5_000 })

  await page.locator('button.kb-stop-btn').click()

  await expect.poll(() => backend.getGenerationCancelCalls(), { timeout: 5_000 }).toBe(1)
  await expect(page.locator('button.kb-stop-btn')).toHaveCount(0, { timeout: 5_000 })
  await expect(page.locator('body')).toContainText('Question for A')
  await expect(page.locator('body')).toContainText(A_CANCELED_ANSWER, { timeout: 5_000 })
  await expect(page.locator('body')).not.toContainText(A_GENERATED_ANSWER)
})

test('in-flight PDF upload stays scoped to its conversation while switching', async ({ page }) => {
  const backend = await installBackend(page)

  await page.goto('/')
  await page.locator('.kb-conv-row', { hasText: 'Generation Race A' }).click()
  await expect(page.locator('body')).toContainText('Existing answer A')

  await page.locator('input[type="file"]').setInputFiles({
    name: UPLOAD_PDF_NAME,
    mimeType: 'application/pdf',
    buffer: Buffer.from('%PDF-1.4\n% upload race test\n'),
  })
  await expect(page.locator('body')).toContainText(UPLOAD_PDF_NAME)
  await expect(page.locator('body')).toContainText('Converting PDF')

  await page.locator('.kb-conv-row', { hasText: 'Generation Race B' }).click()
  await expect(page.locator('body')).toContainText(B_STABLE_ANSWER)
  await expect(page.locator('body')).not.toContainText(UPLOAD_PDF_NAME)

  backend.markUploadReady()
  await expect.poll(
    async () => page.evaluate(async ({ convId, jobId }) => {
      const { useChatStore } = await import('/src/stores/chatStore.ts')
      const items = useChatStore.getState().conversationCacheById[convId]?.uploadItems || []
      return items.find((item) => item.ingest_job_id === jobId)?.ingest_status || ''
    }, { convId: CONV_A_ID, jobId: UPLOAD_JOB_ID }),
    { timeout: 5_000 },
  ).toBe('ready')
  await expect(page.locator('body')).toContainText(B_STABLE_ANSWER)
  await expect(page.locator('body')).not.toContainText(UPLOAD_PDF_NAME)

  await page.locator('.kb-conv-row', { hasText: 'Generation Race A' }).click()
  await expect(page.locator('body')).toContainText(UPLOAD_PDF_NAME)
  await expect.poll(
    () => backend.getUploadStatusCalls(),
    { timeout: 5_000 },
  ).toBeGreaterThanOrEqual(1)
  await expect(page.locator('body')).toContainText('PDF ingested')
})

test('deleting another conversation does not stop the active upload polling', async ({ page }) => {
  const backend = await installBackend(page)

  await page.goto('/')
  await page.locator('.kb-conv-row', { hasText: 'Generation Race B' }).click()
  await expect(page.locator('body')).toContainText(B_STABLE_ANSWER)

  await page.locator('input[type="file"]').setInputFiles({
    name: UPLOAD_PDF_NAME,
    mimeType: 'application/pdf',
    buffer: Buffer.from('%PDF-1.4\n% upload non-active delete test\n'),
  })
  await expect(page.locator('body')).toContainText(UPLOAD_PDF_NAME)
  await expect(page.locator('body')).toContainText('Converting PDF')

  const conversationA = page.locator('.kb-conv-row', { hasText: 'Generation Race A' })
  await deleteConversationFromRow(page, conversationA)
  await expect(page.locator('.kb-conv-row', { hasText: 'Generation Race A' })).toHaveCount(0)

  backend.markUploadReady()
  await expect.poll(
    () => backend.getUploadStatusCalls(),
    { timeout: 5_000 },
  ).toBeGreaterThanOrEqual(2)
  await expect(page.locator('body')).toContainText(UPLOAD_PDF_NAME)
  await expect(page.locator('body')).toContainText('PDF ingested')
})

test('deleting a cached conversation cancels its in-flight PDF upload job', async ({ page }) => {
  const backend = await installBackend(page)

  await page.goto('/')
  await page.locator('.kb-conv-row', { hasText: 'Generation Race A' }).click()
  await expect(page.locator('body')).toContainText('Existing answer A')

  await page.locator('input[type="file"]').setInputFiles({
    name: UPLOAD_PDF_NAME,
    mimeType: 'application/pdf',
    buffer: Buffer.from('%PDF-1.4\n% upload cached delete test\n'),
  })
  await expect(page.locator('body')).toContainText(UPLOAD_PDF_NAME)
  await expect(page.locator('body')).toContainText('Converting PDF')

  await page.locator('.kb-conv-row', { hasText: 'Generation Race B' }).click()
  await expect(page.locator('body')).toContainText(B_STABLE_ANSWER)
  await expect(page.locator('body')).not.toContainText(UPLOAD_PDF_NAME)

  const conversationA = page.locator('.kb-conv-row', { hasText: 'Generation Race A' })
  await deleteConversationFromRow(page, conversationA)

  await expect.poll(() => backend.getUploadCancelCalls(), { timeout: 5_000 }).toBe(1)
  await expect(page.locator('.kb-conv-row', { hasText: 'Generation Race A' })).toHaveCount(0)
  await expect(page.locator('body')).toContainText(B_STABLE_ANSWER)
  await expect(page.locator('body')).not.toContainText(UPLOAD_PDF_NAME)

  const cachedItems = await page.evaluate(async (convId) => {
    const { useChatStore } = await import('/src/stores/chatStore.ts')
    return useChatStore.getState().conversationCacheById[convId]?.uploadItems || []
  }, CONV_A_ID)
  expect(cachedItems).toEqual([])
})

test('failed generation stream clears the active running state', async ({ page }) => {
  const backend = await installBackend(page, { streamFailure: true })

  await page.goto('/')
  await page.locator('.kb-conv-row', { hasText: 'Generation Race A' }).click()
  await expect(page.locator('body')).toContainText('Existing answer A')

  await page.locator('textarea.kb-chat-textarea, .kb-chat-textarea textarea').fill('Question for A')
  await page.locator('button.kb-send-btn').click()

  await expect(page.locator('button.kb-stop-btn')).toHaveCount(0, { timeout: 5_000 })
  await expect(page.locator('body')).toContainText('Question for A')
  await expect(page.locator('body')).toContainText(A_STREAM_FAILED_ANSWER)
  await expect(page.locator('body')).not.toContainText('stream temporarily unavailable')
  await expect.poll(
    () => backend.getUserIssueReports().filter((item) => item.domain === 'chat_generation').length,
    { timeout: 5_000 },
  ).toBe(1)
  const report = backend.getUserIssueReports().find((item) => item.domain === 'chat_generation')
  expect(report?.summary).toBe('Chat send failed: generation_stream_failed')
  expect(report?.detail).toBe(A_STREAM_FAILED_ANSWER)
  expect(report?.context?.query_scope).toBe('library')
  expect(report?.context?.prompt_length).toBe('Question for A'.length)
  expect(report?.context?.prompt_empty).toBe(false)
  expect(report?.payload?.error_kind).toBe('generation_stream_failed')
  expect(JSON.stringify(report)).not.toContain('Question for A')
})

test('failed generation stream is localized in Chinese UI', async ({ page }) => {
  await installBackend(page, { streamFailure: true, uiLocale: 'zh' })

  await page.goto('/')
  await page.locator('.kb-conv-row', { hasText: 'Generation Race A' }).click()
  await expect(page.locator('body')).toContainText('Existing answer A')

  await page.locator('textarea.kb-chat-textarea, .kb-chat-textarea textarea').fill('Question for A')
  await page.locator('button.kb-send-btn').click()

  await expect(page.locator('button.kb-stop-btn')).toHaveCount(0, { timeout: 5_000 })
  await expect(page.locator('body')).toContainText('Question for A')
  await expect(page.locator('body')).toContainText(A_STREAM_FAILED_ANSWER_ZH)
  await expect(page.locator('body')).not.toContainText('stream temporarily unavailable')
})

test('incomplete generation stream shows interruption guidance', async ({ page }) => {
  const backend = await installBackend(page, { streamIncomplete: true })

  await page.goto('/')
  await page.locator('.kb-conv-row', { hasText: 'Generation Race A' }).click()
  await expect(page.locator('body')).toContainText('Existing answer A')

  await page.locator('textarea.kb-chat-textarea, .kb-chat-textarea textarea').fill('Question for A')
  await page.locator('button.kb-send-btn').click()

  backend.releaseStream()
  await expect(page.locator('button.kb-stop-btn')).toHaveCount(0, { timeout: 5_000 })
  await expect(page.locator('body')).toContainText('Question for A')
  await expect(page.locator('body')).toContainText(A_GENERATED_ANSWER)
  await expect(page.locator('body')).toContainText(A_STREAM_INCOMPLETE_ANSWER)
  await expect(page.locator('body')).not.toContainText('Generation stream ended before completion')
})

test('generation start failure refreshes persisted error without opening a stream', async ({ page }) => {
  const backend = await installBackend(page, { generateStartFailure: true })

  await page.goto('/')
  await page.locator('.kb-conv-row', { hasText: 'Generation Race A' }).click()
  await expect(page.locator('body')).toContainText('Existing answer A')

  await page.locator('textarea.kb-chat-textarea, .kb-chat-textarea textarea').fill('Question for A')
  await page.locator('button.kb-send-btn').click()

  await expect(page.locator('button.kb-stop-btn')).toHaveCount(0, { timeout: 5_000 })
  await expect(page.locator('body')).toContainText('Question for A')
  await expect(page.locator('body')).toContainText(A_START_FAILED_ANSWER)
  expect(backend.getGenerationStreamCalls()).toBe(0)
})

test('generation start failure is localized in Chinese UI', async ({ page }) => {
  const backend = await installBackend(page, { generateStartFailure: true, uiLocale: 'zh' })

  await page.goto('/')
  await page.locator('.kb-conv-row', { hasText: 'Generation Race A' }).click()
  await expect(page.locator('body')).toContainText('Existing answer A')

  await page.locator('textarea.kb-chat-textarea, .kb-chat-textarea textarea').fill('Question for A')
  await page.locator('button.kb-send-btn').click()

  await expect(page.locator('button.kb-stop-btn')).toHaveCount(0, { timeout: 5_000 })
  await expect(page.locator('body')).toContainText('Question for A')
  await expect(page.locator('body')).toContainText(A_START_FAILED_ANSWER_ZH)
  await expect(page.locator('body')).not.toContainText(A_START_FAILED_ANSWER)
  expect(backend.getGenerationStreamCalls()).toBe(0)
})

test('done generation stream clears running state when final message reload fails', async ({ page }) => {
  const backend = await installBackend(page, { completionMessageFailure: true })

  await page.goto('/')
  await page.locator('.kb-conv-row', { hasText: 'Generation Race A' }).click()
  await expect(page.locator('body')).toContainText('Existing answer A')

  await page.locator('textarea.kb-chat-textarea, .kb-chat-textarea textarea').fill('Question for A')
  await page.locator('button.kb-send-btn').click()
  await expect(page.locator('button.kb-stop-btn')).toBeVisible({ timeout: 5_000 })

  backend.releaseStream()
  await expect(page.locator('button.kb-stop-btn')).toHaveCount(0, { timeout: 5_000 })
  await expect(page.locator('body')).toContainText('Question for A')
  await expect(page.locator('body')).toContainText(A_GENERATED_ANSWER)
  await expect(page.locator('body')).toContainText(A_REFRESH_FAILED_ANSWER)
  await expect(page.locator('body')).not.toContainText('messages page temporarily unavailable')
  await expect(page.locator('body')).not.toContainText(A_STREAM_FAILED_ANSWER)
})

test('terminal error event is not treated as a successful answer and can retry the same prompt', async ({ page }) => {
  const backend = await installBackend(page, { terminalStreamError: true })
  const prompt = 'Question for terminal failure'

  await page.goto('/')
  await page.locator('.kb-conv-row', { hasText: 'Generation Race A' }).click()
  await expect(page.locator('body')).toContainText('Existing answer A')

  await page.locator('textarea.kb-chat-textarea, .kb-chat-textarea textarea').fill(prompt)
  await page.locator('button.kb-send-btn').click()
  await expect(page.locator('button.kb-stop-btn')).toBeVisible({ timeout: 5_000 })

  backend.releaseStream()
  await expect(page.locator('button.kb-stop-btn')).toHaveCount(0, { timeout: 5_000 })
  await expect(page.locator('body')).toContainText(A_STREAM_FAILED_ANSWER)
  await expect(page.locator('button.kb-generation-retry-btn')).toHaveText('Retry this question')
  expect(backend.getConvADonePageLoads()).toBe(0)

  await page.locator('button.kb-generation-retry-btn').click()
  await expect.poll(() => backend.getGeneratePayloads().length, { timeout: 5_000 }).toBe(2)
  expect(backend.getGeneratePayloads()[1].prompt).toBe(prompt)
})

test('chat store ignores duplicate send while generation start is still pending', async ({ page }) => {
  await page.goto('/__message_list_test__')

  const result = await page.evaluate(async () => {
    const { useChatStore } = await import('/src/stores/chatStore.ts')
    const originalFetch = window.fetch.bind(window)
    let generateCalls = 0
    let streamCalls = 0
    let resolveGenerateStart: (() => void) | null = null

    const jsonResponse = (body: unknown, headers?: Record<string, string>) => new Response(JSON.stringify(body), {
      status: 200,
      headers: {
        'content-type': 'application/json',
        ...(headers || {}),
      },
    })

    window.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      const rawUrl = typeof input === 'string'
        ? input
        : input instanceof URL
          ? input.toString()
          : input.url
      const url = new URL(rawUrl, window.location.origin)
      const path = url.pathname
      if (path === '/api/generate') {
        generateCalls += 1
        if (generateCalls === 1) {
          await new Promise<void>((resolve) => {
            resolveGenerateStart = resolve
          })
        }
        return jsonResponse({
          session_id: `session-duplicate-${generateCalls}`,
          task_id: `task-duplicate-${generateCalls}`,
          user_msg_id: 100 + generateCalls,
          assistant_msg_id: 200 + generateCalls,
        })
      }
      if (path === '/api/generate/session-duplicate-1/stream') {
        streamCalls += 1
        return new Response([
          'data: {"stage":"drafting","partial":"final duplicate guard answer","done":false}',
          '',
          'data: {"stage":"done","partial":"final duplicate guard answer","done":true}',
          '',
        ].join('\n'), {
          status: 200,
          headers: { 'content-type': 'text/event-stream' },
        })
      }
      if (path === '/api/conversations/conv-duplicate-start/messages_page') {
        return jsonResponse({
          messages: [
            { id: 101, role: 'user', content: 'First duplicate guard prompt', created_at: 3 },
            { id: 201, role: 'assistant', content: 'Final duplicate guard answer', created_at: 4 },
          ],
          has_more_before: false,
          oldest_loaded_id: 101,
          newest_loaded_id: 201,
        })
      }
      if (path === '/api/sidebar') {
        return jsonResponse({
          projects: [],
          root_conversations: [
            { id: 'conv-duplicate-start', title: 'Duplicate start guard', created_at: 1, updated_at: 2 },
          ],
          project_conversations: {},
        })
      }
      if (path === '/api/references/conversation/conv-duplicate-start') {
        return jsonResponse({}, {
          'server-timing': 'total;dur=1',
          'x-kb-refs-mode': 'empty',
          'x-kb-refs-counts': 'packs=0,hits=0,pending=0',
        })
      }
      if (path.startsWith('/api/')) {
        return jsonResponse({})
      }
      return originalFetch(input, init)
    }) as typeof window.fetch

    try {
      useChatStore.setState({
        activeConvId: 'conv-duplicate-start',
        activeConversation: {
          id: 'conv-duplicate-start',
          title: 'Duplicate start guard',
          project_id: null,
          created_at: 1,
          updated_at: 2,
        },
        rootConversations: [
          {
            id: 'conv-duplicate-start',
            title: 'Duplicate start guard',
            project_id: null,
            created_at: 1,
            updated_at: 2,
          },
        ],
        projectConversations: {},
        conversationCacheById: {},
        messages: [],
        refs: {},
        generation: null,
        sseController: null,
        uploadItems: [],
        pendingImages: [],
        conversationLoading: false,
      })

      const opts = {
        topK: 6,
        temperature: 0.2,
        maxTokens: 1216,
        deepRead: true,
        queryScope: 'library' as const,
      }
      const first = useChatStore.getState().sendMessage('First duplicate guard prompt', opts)
      const second = useChatStore.getState().sendMessage('Second duplicate guard prompt', opts)
      await new Promise((resolve) => setTimeout(resolve, 80))
      const generateCallsWhileFirstPending = generateCalls
      if (!resolveGenerateStart) throw new Error('first generate request did not start')
      resolveGenerateStart()
      await Promise.all([first, second])

      const state = useChatStore.getState()
      return {
        generateCalls,
        generateCallsWhileFirstPending,
        streamCalls,
        messages: state.messages.map((item) => ({
          role: item.role,
          content: item.content,
        })),
        generationActive: Boolean(state.generation),
      }
    } finally {
      window.fetch = originalFetch
    }
  })

  expect(result.generateCallsWhileFirstPending).toBe(1)
  expect(result.generateCalls).toBe(1)
  expect(result.streamCalls).toBe(1)
  expect(result.generationActive).toBe(false)
  expect(result.messages).toEqual([
    { role: 'user', content: 'First duplicate guard prompt' },
    { role: 'assistant', content: 'Final duplicate guard answer' },
  ])
})
