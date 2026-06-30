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

const PROJECT_A = { id: 'project-a', name: 'Paper Project', created_at: 1, updated_at: 4 }
const PROJECT_B = { id: 'project-b', name: 'Methods Project', created_at: 2, updated_at: 3 }
const CONV_PROJECT = 'conv-project-paper'
const CONV_ROOT = 'conv-root-paper'

async function fulfillJson(route: Route, body: unknown, status = 200, headers?: Record<string, string>) {
  await route.fulfill({
    status,
    contentType: 'application/json',
    headers,
    body: JSON.stringify(body),
  })
}

async function installBackend(page: Page) {
  const conversations: Record<string, Conversation> = {
    [CONV_PROJECT]: {
      id: CONV_PROJECT,
      title: 'Project Paper',
      created_at: 1,
      updated_at: 10,
      project_id: PROJECT_A.id,
      mode: 'normal',
    },
    [CONV_ROOT]: {
      id: CONV_ROOT,
      title: 'Root Paper',
      created_at: 1,
      updated_at: 9,
      project_id: null,
      mode: 'normal',
    },
  }
  let projects = [PROJECT_A, PROJECT_B]
  const projectIds = new Set(projects.map((project) => project.id))
  let hideProjectConversationFromSidebar = false
  let failSidebarRefreshAfterProjectDelete = false
  let failSidebarRefreshAfterProjectMove = false
  const projectUpdates: Array<{ convId: string; projectId: string | null }> = []
  const projectDeletes: string[] = []
  const generatePayloads: Record<string, unknown>[] = []

  await installAppShellMocks(page, { projects })
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
    if (failSidebarRefreshAfterProjectDelete && projectDeletes.length > 0) {
      await fulfillJson(route, { detail: 'sidebar refresh failed after project delete' }, 503)
      return
    }
    if (failSidebarRefreshAfterProjectMove && projectUpdates.length > 0) {
      await fulfillJson(route, { detail: 'sidebar refresh failed after project move' }, 503)
      return
    }
    const grouped: Record<string, Conversation[]> = {
      [PROJECT_A.id]: [],
      [PROJECT_B.id]: [],
    }
    const root: Conversation[] = []
    for (const conversation of Object.values(conversations)) {
      if (hideProjectConversationFromSidebar && conversation.id === CONV_PROJECT) continue
      if (conversation.project_id && grouped[conversation.project_id]) {
        grouped[conversation.project_id].push(conversation)
      } else {
        root.push({ ...conversation, project_id: null })
      }
    }
    await fulfillJson(route, {
      projects,
      root_conversations: root,
      project_conversations: grouped,
    })
  })
  await page.route(/\/api\/projects\/([^/?]+)$/, async (route) => {
    const match = route.request().url().match(/\/api\/projects\/([^/?]+)$/)
    const projectId = match?.[1] || ''
    if (route.request().method() !== 'DELETE') {
      await fulfillJson(route, { detail: 'not found' }, 404)
      return
    }
    if (!projectIds.has(projectId)) {
      await fulfillJson(route, { detail: 'project not found' }, 404)
      return
    }
    projectIds.delete(projectId)
    projects = projects.filter((project) => project.id !== projectId)
    for (const conversation of Object.values(conversations)) {
      if (conversation.project_id === projectId) {
        conversations[conversation.id] = { ...conversation, project_id: null, updated_at: conversation.updated_at + 1 }
      }
    }
    hideProjectConversationFromSidebar = true
    projectDeletes.push(projectId)
    await fulfillJson(route, { ok: true })
  })
  await page.route(/\/api\/conversations\/([^/]+)\/project$/, async (route) => {
    const match = route.request().url().match(/\/api\/conversations\/([^/]+)\/project$/)
    const convId = match?.[1] || ''
    const body = JSON.parse(route.request().postData() || '{}') as { project_id?: string | null }
    const target = body.project_id || null
    if (!conversations[convId]) {
      await fulfillJson(route, { detail: 'conversation not found' }, 404)
      return
    }
    if (target && !projectIds.has(target)) {
      await fulfillJson(route, { detail: 'project not found' }, 404)
      return
    }
    conversations[convId] = { ...conversations[convId], project_id: target, updated_at: conversations[convId].updated_at + 1 }
    projectUpdates.push({ convId, projectId: target })
    await fulfillJson(route, { ok: true })
  })
  await page.route(/\/api\/conversations\/([^/?]+)$/, async (route) => {
    const match = route.request().url().match(/\/api\/conversations\/([^/?]+)$/)
    const conv = conversations[match?.[1] || '']
    if (!conv) {
      await fulfillJson(route, { detail: 'conversation not found' }, 404)
      return
    }
    await fulfillJson(route, conv)
  })
  await page.route(/\/api\/conversations\/([^/]+)\/messages_page(?:\?.*)?$/, async (route) => {
    const match = route.request().url().match(/\/api\/conversations\/([^/]+)\/messages_page/)
    const convId = match?.[1] || ''
    await fulfillJson(route, {
      messages: [
        { id: 1, role: 'user', content: `Question for ${convId}`, created_at: 1 },
        { id: 2, role: 'assistant', content: `Answer for ${convId}`, created_at: 2 },
      ],
      has_more_before: false,
      oldest_loaded_id: 1,
      newest_loaded_id: 2,
    })
  })
  await page.route(/\/api\/conversations\/([^/]+)\/messages(?:\?.*)?$/, async (route) => {
    const match = route.request().url().match(/\/api\/conversations\/([^/]+)\/messages/)
    const convId = match?.[1] || ''
    await fulfillJson(route, [
      { id: 1, role: 'user', content: `Question for ${convId}`, created_at: 1 },
      { id: 2, role: 'assistant', content: `Answer for ${convId}`, created_at: 2 },
    ])
  })
  await page.route(/\/api\/conversations\/([^/]+)\/research-state$/, async (route) => {
    const match = route.request().url().match(/\/api\/conversations\/([^/]+)\/research-state$/)
    await fulfillJson(route, { conv_id: match?.[1] || '', state: {}, created_at: 1, updated_at: 1 })
  })
  await page.route(/\/api\/references\/conversation\/([^/?]+)(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, {}, 200, {
      'server-timing': 'total;dur=1',
      'x-kb-refs-mode': 'empty',
      'x-kb-refs-counts': 'packs=0,hits=0,pending=0',
    })
  })
  await page.route('**/api/generate', async (route) => {
    const payload = route.request().postDataJSON() as Record<string, unknown>
    generatePayloads.push(payload)
    await fulfillJson(route, {
      session_id: 'session-sidebar-project',
      task_id: 'task-sidebar-project',
      trace_id: 'trace-sidebar-project',
      user_msg_id: 101,
      assistant_msg_id: 102,
      conversation_title: 'Project Paper',
    })
  })
  await page.route('**/api/generate/session-sidebar-project/stream', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: 'data: {"done":true,"status":"done","stage":"done","partial":"ok","answer":"ok","research_trace":{}}\n\n',
    })
  })
  await page.route(/\/api\/conversations(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, Object.values(conversations))
  })
  return {
    projectUpdates,
    projectDeletes,
    generatePayloads,
    failSidebarRefreshAfterProjectDelete: () => {
      failSidebarRefreshAfterProjectDelete = true
    },
    failSidebarRefreshAfterProjectMove: () => {
      failSidebarRefreshAfterProjectMove = true
    },
  }
}

test('project conversations can move locally even when sidebar refresh fails', async ({ page }) => {
  const backend = await installBackend(page)
  await page.goto('/')

  await page.getByText('Paper Project', { exact: true }).click()
  const projectRow = page.locator('.kb-conv-row').filter({ hasText: 'Project Paper' })
  await expect(projectRow).toBeVisible()
  backend.failSidebarRefreshAfterProjectMove()
  await projectRow.getByLabel('Conversation actions').click()
  await page.getByText('Move to', { exact: true }).hover()
  await page.getByRole('menuitem', { name: 'Ungrouped' }).click()

  await expect.poll(() => backend.projectUpdates).toContainEqual({ convId: CONV_PROJECT, projectId: null })
  const rootRow = page.locator('.kb-root-conversations .kb-conv-row').filter({ hasText: 'Project Paper' })
  await expect(rootRow).toBeVisible()
  await expect(page.locator('body')).not.toContainText('sidebar refresh failed after project move')

  await rootRow.getByLabel('Conversation actions').click()
  await page.getByText('Move to', { exact: true }).hover()
  await page.getByRole('menuitem', { name: 'Methods Project' }).click()

  await expect.poll(() => backend.projectUpdates).toContainEqual({ convId: CONV_PROJECT, projectId: PROJECT_B.id })
  await page.getByRole('button', { name: 'Methods Project' }).click()
  const movedProjectRow = page.locator('.kb-project-card').filter({ hasText: 'Methods Project' }).locator('.kb-conv-row').filter({ hasText: 'Project Paper' })
  await expect(movedProjectRow).toBeVisible()
  await expect(page.locator('body')).not.toContainText('sidebar refresh failed after project move')
})

test('deleting the active project rehomes the active conversation shelf scope to ungrouped', async ({ page }) => {
  const backend = await installBackend(page)
  const staleProjectContextPack = {
    version: 1,
    id: 'stale-project-context',
    source: 'citation_shelf',
    createdAt: 1,
    conversationId: CONV_PROJECT,
    guideSourcePath: '',
    guideSourceName: '',
    itemCount: 1,
    tokenEstimate: 12,
    items: [{
      key: 'stale-project-ref',
      kind: 'reference',
      title: 'Project scoped context should not leak',
      sourceName: 'Project Source.pdf',
      sourcePath: 'db/project/source.en.md',
      locationLabel: 'Project / Ref',
      headingPath: 'References',
      blockId: 'project-block',
      anchorId: 'project-ref',
      anchorKind: 'reference',
      refNum: 1,
      doi: '',
      libraryMatchPath: '',
      libraryMatchStatus: '',
      libraryMatchTitle: '',
      libraryMatchDoi: '',
      libraryMatchYear: '',
      authors: '',
      year: '',
      summary: 'This old project-scoped context must be dropped after the project is deleted.',
      excerpt: '',
      note: '',
    }],
  }
  await page.addInitScript(
    ({ convId, projectId, pack }) => {
      window.localStorage.setItem(
        `kb:chat:selected-research-context:v1:${encodeURIComponent(convId)}:${encodeURIComponent(projectId)}`,
        JSON.stringify(pack),
      )
    },
    { convId: CONV_PROJECT, projectId: PROJECT_A.id, pack: staleProjectContextPack },
  )
  await page.goto('/')

  await page.getByText('Paper Project', { exact: true }).click()
  const projectRow = page.locator('.kb-project-card').filter({ hasText: 'Paper Project' }).locator('.kb-conv-row').filter({ hasText: 'Project Paper' })
  await projectRow.click()
  await expect(page.locator('body')).toContainText(`Answer for ${CONV_PROJECT}`)
  await expect(page.getByTestId('research-context-state')).toHaveAttribute('data-research-shelf-scope', PROJECT_A.id)
  await expect(page.getByTestId('chat-context-pack')).toContainText('1 excerpts')

  backend.failSidebarRefreshAfterProjectDelete()
  await page.locator('.kb-project-card').filter({ hasText: 'Paper Project' }).getByLabel('Project actions').click()
  await page.getByRole('menuitem', { name: 'Delete Project' }).click()
  await page.getByRole('button', { name: 'OK' }).click()

  await expect.poll(() => backend.projectDeletes).toContain(PROJECT_A.id)
  await expect(page.locator('.kb-project-card').filter({ hasText: 'Paper Project' })).toHaveCount(0)
  await expect(page.locator('body')).toContainText(`Answer for ${CONV_PROJECT}`)
  await expect(page.getByTestId('research-context-state')).toHaveAttribute('data-research-shelf-scope', '__default__')
  await expect(page.getByTestId('chat-context-pack')).toHaveCount(0)

  await page.locator('textarea.kb-chat-textarea, .kb-chat-textarea textarea').fill('Send after project delete.')
  await page.getByRole('button', { name: 'Send' }).click()
  await expect.poll(() => backend.generatePayloads.length).toBe(1)
  expect(backend.generatePayloads[0].prompt_context).toBeFalsy()
  expect(backend.generatePayloads[0].query_scope).toBe('library')
})
