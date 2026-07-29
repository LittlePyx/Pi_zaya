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
    delayedCancelFinalize?: boolean
    uiLocale?: 'en' | 'zh'
    paperGuideMode?: boolean
    lateCitationHydration?: boolean
    legalZeroCitationTerminal?: boolean
    partialCitationCoverage?: boolean
    lateCitationPlanHydration?: boolean
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
  let cancelFinalizePageLoads = 0
  const deletedConversationIds = new Set<string>()
  const activeConvA = opts?.paperGuideMode
    ? {
        ...convA,
        mode: 'paper_guide' as const,
        bound_source_path: '/papers/generation-race-a.md',
        bound_source_name: 'Generation Race A.pdf',
        bound_source_ready: true,
      }
    : convA
  const streamReleased = new Promise<void>((resolve) => {
    releaseStream = resolve
  })

  await installAppShellMocks(page, { rootConversations: [activeConvA, convB] })
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
      root_conversations: [activeConvA, convB].filter((conv) => !deletedConversationIds.has(conv.id)),
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
    await fulfillJson(route, activeConvA)
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
    if (generationCanceled && opts?.delayedCancelFinalize && !generationDone) {
      cancelFinalizePageLoads += 1
      if (cancelFinalizePageLoads >= 2) {
        generationDone = true
      } else {
        await fulfillJson(route, messagePage([
          { id: 101, role: 'user', content: 'Question for A', created_at: 3 },
          { id: 102, role: 'assistant', content: 'Draft answer [10001]', created_at: 4 },
        ]))
        return
      }
    }
    if (generationDone || generationStartFailed) convADonePageLoads += 1
    if (generationDone && opts?.completionMessageFailure) {
      await route.fulfill({
        status: 503,
        contentType: 'text/plain',
        body: 'messages page temporarily unavailable',
      })
      return
    }
    const citationHydrationMessage = () => {
      const anchors = opts?.partialCitationCoverage
        ? ['partial-system-a-anchor-1', 'partial-system-a-anchor-2']
        : ['late-system-a-anchor']
      const latePlanReady = Boolean(opts?.lateCitationPlanHydration && convADonePageLoads >= 6)
      const linked = Boolean(
        opts?.partialCitationCoverage
        || latePlanReady
        || (opts?.lateCitationHydration && convADonePageLoads >= 6),
      )
      const citeDetails = opts?.legalZeroCitationTerminal
        ? []
        : opts?.lateCitationPlanHydration && !latePlanReady
          ? []
          : anchors.map((anchor, index) => ({
            num: index + 1,
            anchor,
            citation_route: 'system_a',
            source_name: 'Late citation source.pdf',
            source_path: '/papers/late-citation-source.md',
            evidence_quote: 'Late-arriving evidence for the generated answer.',
          }))
      const renderedBody = linked
        ? `${A_GENERATED_ANSWER} ${anchors.map((anchor, index) => `[${index + 1}](#${anchor})`).join(' ')}`
        : A_GENERATED_ANSWER
      const citationPlan = {
        system_a_enabled: true,
        budget: { system_a: opts?.partialCitationCoverage ? 3 : 1, system_b: 0 },
        ...(opts?.partialCitationCoverage
          ? { coverage_mode: 'per_entity', coverage_target_count: 3 }
          : {}),
        slots: anchors.map((_anchor, index) => ({
          preferred_system: 'system_a',
          candidate_hits: [index + 1],
        })),
      }
      const hasCitationPlan = !opts?.lateCitationPlanHydration || latePlanReady
      return {
        id: 102,
        role: 'assistant',
        content: A_GENERATED_ANSWER,
        rendered_body: renderedBody,
        cite_details: citeDetails,
        refs_user_msg_id: 101,
        created_at: 4,
        provenance: {
          status: 'ready',
          strict_identity_ready: true,
          segments: [{ segment_id: 'generation-race-evidence' }],
        },
        meta: {
          ...(hasCitationPlan ? { answer_quality: { citation_plan: citationPlan } } : {}),
          paper_guide_contracts: {
            render_packet: {
              answer_markdown: A_GENERATED_ANSWER,
              rendered_body: renderedBody,
              rendered_content: renderedBody,
              cite_details: citeDetails,
            },
          },
        },
      }
    }
    const completedAssistant = (
      opts?.lateCitationHydration
      || opts?.legalZeroCitationTerminal
      || opts?.partialCitationCoverage
      || opts?.lateCitationPlanHydration
    )
      ? citationHydrationMessage()
      : { id: 102, role: 'assistant', content: generationCanceled ? A_CANCELED_ANSWER : A_GENERATED_ANSWER, created_at: 4 }
    await fulfillJson(route, messagePage(generationStartFailed
      ? [
          { id: 101, role: 'user', content: 'Question for A', created_at: 3 },
          { id: 102, role: 'assistant', content: A_START_FAILED_ANSWER, created_at: 4 },
        ]
      : generationDone
      ? [
          { id: 101, role: 'user', content: 'Question for A', created_at: 3 },
          completedAssistant,
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
    generationDone = !opts?.delayedCancelFinalize
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
    const settledEmptyPack = {
      payload_mode: 'full',
      render_status: 'full',
      display_state: 'empty',
      suppression_reason: 'no_candidate_hits',
      enrichment_pending: false,
      pipeline_debug: { raw_hit_count: 0 },
      hits: [],
    }
    const refs = (
      opts?.lateCitationHydration
      || opts?.partialCitationCoverage
      || opts?.lateCitationPlanHydration
    )
      ? {
          101: {
            payload_mode: 'full',
            render_status: 'full',
            display_state: 'ready',
            enrichment_pending: false,
            hits: [{ text: 'Current-turn evidence is available.' }],
          },
          999: settledEmptyPack,
        }
      : opts?.legalZeroCitationTerminal
        ? {
            101: settledEmptyPack,
            999: {
              payload_mode: 'full',
              render_status: 'full',
              display_state: 'ready',
              enrichment_pending: false,
              hits: [{ text: 'Unrelated newer pack must not control this message.' }],
            },
          }
        : {}
    await fulfillJson(route, refs, {
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

test('cancel refresh waits for the finalized user-safe assistant message', async ({ page }) => {
  const backend = await installBackend(page, { delayedCancelFinalize: true })

  await page.goto('/')
  await page.locator('.kb-conv-row', { hasText: 'Generation Race A' }).click()
  await page.locator('textarea.kb-chat-textarea, .kb-chat-textarea textarea').fill('Question for A')
  await page.locator('button.kb-send-btn').click()
  await expect(page.locator('button.kb-stop-btn')).toBeVisible({ timeout: 5_000 })

  await page.locator('button.kb-stop-btn').click()

  await expect.poll(() => backend.getGenerationCancelCalls(), { timeout: 5_000 }).toBe(1)
  await expect(page.locator('body')).toContainText(A_CANCELED_ANSWER, { timeout: 5_000 })
  await expect(page.locator('body')).not.toContainText('Draft answer [10001]')
  await expect(page.locator('body')).not.toContainText('[10001]')
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

test('done generation stream keeps the streamed answer when final message hydration fails', async ({ page }) => {
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
  await expect(page.locator('body')).not.toContainText(A_REFRESH_FAILED_ANSWER)
  await expect(page.locator('body')).not.toContainText('messages page temporarily unavailable')
  await expect(page.locator('body')).not.toContainText(A_STREAM_FAILED_ANSWER)
})

test('paper-guide polling keeps waiting for a late same-message citation packet', async ({ page }) => {
  const backend = await installBackend(page, {
    paperGuideMode: true,
    lateCitationHydration: true,
  })

  await page.goto('/')
  await page.locator('.kb-conv-row', { hasText: 'Generation Race A' }).click()
  await expect(page.locator('body')).toContainText('Existing answer A')

  await page.locator('textarea.kb-chat-textarea, .kb-chat-textarea textarea').fill('Question for A')
  await page.locator('button.kb-send-btn').click()
  backend.releaseStream()

  const assistant = page.locator('[data-msg-id="102"] .kb-msg-bubble-assistant')
  await expect(page.locator('button.kb-stop-btn')).toHaveCount(0, { timeout: 5_000 })
  await expect(assistant).toContainText(A_GENERATED_ANSWER)
  await expect(assistant.locator('.kb-cite-chip')).toHaveCount(1, { timeout: 5_000 })
  await expect.poll(
    () => backend.getConvADonePageLoads(),
    { timeout: 5_000 },
  ).toBeGreaterThanOrEqual(6)
})

test('short postprocess tail merge preserves an already loaded long history', async ({ page }) => {
  await installBackend(page)
  await page.goto('/')

  const merged = await page.evaluate(async () => {
    const { mergeLatestMessagePage } = await import('/src/stores/chatStoreMessages.ts')
    const currentMessages = Array.from({ length: 30 }, (_item, index) => ({
      id: index + 1,
      role: (index + 1) % 2 === 0 ? 'assistant' : 'user',
      content: `history-${index + 1}`,
      created_at: index + 1,
    }))
    const tailMessages = [29, 30, 31, 32].map((id) => ({
      id,
      role: id % 2 === 0 ? 'assistant' : 'user',
      content: id === 30 ? 'history-30-with-late-citation' : `history-${id}`,
      created_at: id,
    }))
    const result = mergeLatestMessagePage(currentMessages, true, {
      messages: tailMessages,
      has_more_before: true,
      oldest_loaded_id: 29,
      newest_loaded_id: 32,
    })
    return {
      ids: result.messages.map((message) => Number(message.id || 0)),
      contents: result.messages.map((message) => String(message.content || '')),
      hasMoreBefore: result.hasMoreBefore,
      oldestLoadedMessageId: result.oldestLoadedMessageId,
    }
  })

  expect(merged.ids).toEqual(Array.from({ length: 32 }, (_item, index) => index + 1))
  expect(new Set(merged.ids).size).toBe(32)
  expect(merged.contents[0]).toBe('history-1')
  expect(merged.contents[29]).toBe('history-30-with-late-citation')
  expect(merged.hasMoreBefore).toBe(true)
  expect(merged.oldestLoadedMessageId).toBe(1)
})

test('paper-guide polling settles when per-entity coverage stabilizes below its target', async ({ page }) => {
  const backend = await installBackend(page, {
    paperGuideMode: true,
    partialCitationCoverage: true,
  })

  await page.goto('/')
  await page.locator('.kb-conv-row', { hasText: 'Generation Race A' }).click()
  await expect(page.locator('body')).toContainText('Existing answer A')

  await page.locator('textarea.kb-chat-textarea, .kb-chat-textarea textarea').fill('Question for A')
  await page.locator('button.kb-send-btn').click()
  const pollingStartedAt = Date.now()
  backend.releaseStream()

  const assistant = page.locator('[data-msg-id="102"] .kb-msg-bubble-assistant')
  await expect(page.locator('button.kb-stop-btn')).toHaveCount(0, { timeout: 5_000 })
  await expect(assistant.locator('.kb-cite-chip')).toHaveCount(2, { timeout: 5_000 })
  await expect.poll(
    () => backend.getConvADonePageLoads(),
    { timeout: 5_000 },
  ).toBeGreaterThanOrEqual(6)

  await page.waitForTimeout(1_600)
  const settledLoads = backend.getConvADonePageLoads()
  const elapsedMs = Date.now() - pollingStartedAt
  expect(settledLoads).toBe(6)
  expect(elapsedMs).toBeLessThan(7_000)
  await page.waitForTimeout(1_000)
  expect(backend.getConvADonePageLoads()).toBe(settledLoads)
})

test('paper-guide polling waits when the citation plan and citation packet arrive together late', async ({ page }) => {
  const backend = await installBackend(page, {
    paperGuideMode: true,
    lateCitationPlanHydration: true,
  })

  await page.goto('/')
  await page.locator('.kb-conv-row', { hasText: 'Generation Race A' }).click()
  await expect(page.locator('body')).toContainText('Existing answer A')

  await page.locator('textarea.kb-chat-textarea, .kb-chat-textarea textarea').fill('Question for A')
  await page.locator('button.kb-send-btn').click()
  const pollingStartedAt = Date.now()
  backend.releaseStream()

  const assistant = page.locator('[data-msg-id="102"] .kb-msg-bubble-assistant')
  await expect(page.locator('button.kb-stop-btn')).toHaveCount(0, { timeout: 5_000 })
  await expect(assistant.locator('.kb-cite-chip')).toHaveCount(1, { timeout: 5_000 })
  await expect.poll(
    () => backend.getConvADonePageLoads(),
    { timeout: 5_000 },
  ).toBeGreaterThanOrEqual(6)

  await page.waitForTimeout(500)
  const settledLoads = backend.getConvADonePageLoads()
  const elapsedMs = Date.now() - pollingStartedAt
  expect(settledLoads).toBe(6)
  expect(elapsedMs).toBeLessThan(6_000)
})

test('paper-guide polling stops on the exact turn legal zero-citation terminal', async ({ page }) => {
  const backend = await installBackend(page, {
    paperGuideMode: true,
    legalZeroCitationTerminal: true,
  })

  await page.goto('/')
  await page.locator('.kb-conv-row', { hasText: 'Generation Race A' }).click()
  await expect(page.locator('body')).toContainText('Existing answer A')

  await page.locator('textarea.kb-chat-textarea, .kb-chat-textarea textarea').fill('Question for A')
  await page.locator('button.kb-send-btn').click()
  backend.releaseStream()

  const assistant = page.locator('[data-msg-id="102"] .kb-msg-bubble-assistant')
  await expect(page.locator('button.kb-stop-btn')).toHaveCount(0, { timeout: 5_000 })
  await expect(assistant).toContainText(A_GENERATED_ANSWER)
  await expect(assistant.locator('.kb-cite-chip')).toHaveCount(0)
  await expect.poll(
    () => backend.getConvADonePageLoads(),
    { timeout: 5_000 },
  ).toBeGreaterThanOrEqual(5)

  await page.waitForTimeout(500)
  const settledLoads = backend.getConvADonePageLoads()
  await page.waitForTimeout(1_600)
  expect(backend.getConvADonePageLoads()).toBe(settledLoads)
})

test('terminal answer clears a stale render packet even when answer_markdown matches', async ({ page }) => {
  const backend = await installBackend(page, { completionMessageFailure: true })

  await page.goto('/')
  await page.locator('.kb-conv-row', { hasText: 'Generation Race A' }).click()
  await page.locator('textarea.kb-chat-textarea, .kb-chat-textarea textarea').fill('Question for A')
  await page.locator('button.kb-send-btn').click()
  await expect(page.locator('button.kb-stop-btn')).toBeVisible({ timeout: 5_000 })

  await page.evaluate(async ({ convId, finalAnswer }) => {
    const { useChatStore } = await import('/src/stores/chatStore.ts')
    useChatStore.setState((state) => {
      const staleAssistant: (typeof state.messages)[number] = {
        id: 102,
        role: 'assistant',
        content: finalAnswer,
        rendered_body: 'STALE_PACKET_BODY [9](#stale-packet-anchor)',
        rendered_content: 'STALE_PACKET_BODY [9](#stale-packet-anchor)',
        copy_text: 'STALE_PACKET_COPY',
        copy_markdown: 'STALE_PACKET_COPY',
        notice: 'STALE_PACKET_NOTICE',
        cite_details: [{ num: 9, anchor: 'stale-top-level-anchor' }],
        refs_user_msg_id: 101,
        render_cache_key: 'stale-render-cache-key',
        created_at: Date.now() / 1000,
        provenance: { status: 'ready', segments: [{ segment_id: 'stale-segment' }] },
        meta: {
          answer_quality: { retrieval: { low_confidence: true } },
          agent_source_summary: { label: 'STALE_SOURCE_SUMMARY', should_show: true },
          paper_guide_contracts: {
            render_packet: {
              answer_markdown: finalAnswer,
              rendered_body: 'STALE_PACKET_BODY [9](#stale-packet-anchor)',
              rendered_content: 'STALE_PACKET_BODY [9](#stale-packet-anchor)',
              copy_text: 'STALE_PACKET_COPY',
              copy_markdown: 'STALE_PACKET_COPY',
              notice: 'STALE_PACKET_NOTICE',
              cite_details: [{
                num: 9,
                anchor: 'stale-packet-anchor',
                source_name: 'STALE_PACKET_SOURCE',
                source_path: '/papers/stale-packet.md',
              }],
            },
          },
        },
      }
      const messages = [...state.messages, staleAssistant]
      const cached = state.conversationCacheById[convId]
      return {
        messages,
        conversationCacheById: {
          ...state.conversationCacheById,
          [convId]: { ...cached, messages },
        },
      }
    })
  }, { convId: CONV_A_ID, finalAnswer: A_GENERATED_ANSWER })

  const assistant = page.locator('[data-msg-id="102"] .kb-msg-bubble-assistant')
  await expect(assistant).toContainText('STALE_PACKET_BODY')
  await expect(assistant.locator('.kb-cite-chip')).toHaveCount(1)

  backend.releaseStream()
  await expect(page.locator('button.kb-stop-btn')).toHaveCount(0, { timeout: 5_000 })
  await expect(assistant).toContainText(A_GENERATED_ANSWER)
  await expect(assistant).not.toContainText('STALE_PACKET_BODY')
  await expect(assistant).not.toContainText('STALE_PACKET_NOTICE')
  await expect(assistant.locator('.kb-cite-chip')).toHaveCount(0)

  const storedPresentation = await page.evaluate(async () => {
    const { useChatStore } = await import('/src/stores/chatStore.ts')
    const message = useChatStore.getState().messages.find((item) => item.id === 102)
    const contracts = message?.meta?.paper_guide_contracts as Record<string, unknown> | undefined
    return {
      content: message?.content || '',
      renderedBody: message?.rendered_body || '',
      copyText: message?.copy_text || '',
      citeCount: message?.cite_details?.length || 0,
      hasRenderPacket: Boolean(contracts?.render_packet),
      hasAnswerQuality: Boolean(message?.meta?.answer_quality),
      hasStaleSourceSummary: Boolean(message?.meta?.agent_source_summary),
      hasProvenance: Boolean(message?.provenance),
    }
  })
  expect(storedPresentation).toEqual({
    content: A_GENERATED_ANSWER,
    renderedBody: '',
    copyText: '',
    citeCount: 0,
    hasRenderPacket: false,
    hasAnswerQuality: false,
    hasStaleSourceSummary: false,
    hasProvenance: false,
  })
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
