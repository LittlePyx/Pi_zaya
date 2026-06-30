import { expect, test, type Page, type Route } from '@playwright/test'
import {
  installAppShellMocks,
  installEmptyCitationShelfMock,
  installIdleReferenceMocks,
} from './mockAppShell'

async function fulfillJson(route: Route, body: unknown, status = 200) {
  await route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(body),
  })
}

function settingsPayload() {
  return {
    model: 'test-model',
    base_url: '',
    has_api_key: true,
    db_dir: 'F:\\kb\\db',
    connection: {
      text: { has_api_key: true, base_url: '', model: 'test-model' },
      vision: { has_api_key: true, base_url: '', model: 'vision-model', uses_text_fallback: false },
      auto_route: true,
    },
    readiness: {
      providers: {
        text: { target: 'text', has_api_key: true, base_url: '', model: 'test-model', status: 'ok', severity: 'ok', reason: 'ready' },
        vision: { target: 'vision', has_api_key: true, base_url: '', model: 'vision-model', status: 'ok', severity: 'ok', reason: 'ready' },
      },
      overall: { status: 'ok', reason: 'ready', target: '' },
    },
    app_readiness: {
      status: 'ok',
      env: 'development',
      production: false,
      auth_required: false,
      items: [],
    },
    prefs: {
      pdf_dir: 'F:\\kb\\pdfs',
      md_dir: 'F:\\kb\\md',
      theme: 'light',
      ui_locale: 'en',
    },
  }
}

async function installLibraryUploadBackend(page: Page) {
  let inspectCalls = 0
  let commitCalls = 0
  let lastCommitBody = ''
  let committed = false
  await installAppShellMocks(page, {
    authStatus: {
      required: false,
      configured: false,
      authenticated: true,
      env: 'development',
      production: false,
    },
    readiness: { status: 'ok', env: 'development', production: false, auth_required: false, items: [] },
  })
  await installEmptyCitationShelfMock(page, { scopeId: '__default__', projectId: null })
  await installIdleReferenceMocks(page)

  await page.route('**/api/settings', async (route) => {
    await fulfillJson(route, settingsPayload())
  })
  await page.route('**/api/settings/readiness', async (route) => {
    await fulfillJson(route, settingsPayload().readiness)
  })
  await page.route('**/api/library/files**', async (route) => {
    await fulfillJson(route, {
      items: committed
        ? [{
          name: 'Suggested-Paper.pdf',
          path: 'F:\\kb\\pdfs\\Suggested-Paper.pdf',
          sha1: 'retry-sha1',
          md_exists: false,
          md_path: '',
          md_folder: '',
          category: 'pending',
          task_state: 'queued',
          status: 'pending',
          replace_task: false,
          queue_pos: 1,
          cur_page_done: 0,
          cur_page_total: 0,
          cur_page_msg: '',
        }]
        : [],
      counts: {
        total_view: committed ? 1 : 0,
        total_all: committed ? 1 : 0,
        pending: committed ? 1 : 0,
        converted: 0,
        queued: committed ? 1 : 0,
        running: 0,
        reconverting: 0,
        quality_review: 0,
        quality_ready: 0,
        index_quality_blocked: 0,
      },
      truncated: false,
      scope: '200',
      queue: { running: false, active_count: 0, active_tasks: [], current: '', done: 0, total: 0 },
    })
  })
  await page.route('**/api/library/upload/inspect', async (route) => {
    inspectCalls += 1
    if (inspectCalls === 1) {
      await fulfillJson(route, { detail: 'temporary scan outage' }, 503)
      return
    }
    await fulfillJson(route, {
      name: 'raw-upload.pdf',
      sha1: 'retry-sha1',
      duplicate: false,
      existing: '',
      existing_path: '',
      suggested_name: 'Suggested-Paper.pdf',
      suggested_stem: 'Suggested-Paper',
      display_full_name: 'Suggested Paper.pdf',
      meta: {
        basis_label: 'DOI metadata',
        basis_detail: 'Resolved title and venue from metadata',
        match_method: 'doi',
        year_source: 'crossref',
      },
    })
  })
  await page.route('**/api/library/upload/commit', async (route) => {
    commitCalls += 1
    lastCommitBody = route.request().postData() || ''
    committed = true
    await fulfillJson(route, {
      duplicate: false,
      path: 'F:\\kb\\pdfs\\Suggested-Paper.pdf',
      name: 'Suggested-Paper.pdf',
      sha1: 'retry-sha1',
      citation_meta: {},
      enqueued: true,
      task_id: 'task-retry',
    })
  })
  await page.route('**/api/library/convert/status', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: 'data: {"running":false,"done":true,"status":"idle","current":"","completed":0,"total":0,"active_count":0,"active_tasks":[]}\n\n',
    })
  })
  await page.route('**/api/library/quality/overview**', async (route) => {
    await fulfillJson(route, {
      ok: true,
      status: 'ok',
      summary: { converted: 0, assessed: 0, good: 0, review: 0, unknown: 0, avg_score: 0 },
      top_issues: [],
      recommended: [],
      domains: {},
      failure_cases: [],
      queue: { running: false, active_count: 0, active_tasks: [], current: '', done: 0, total: 0 },
      scope: 'all',
      truncated: false,
    })
  })

  return {
    inspectCalls: () => inspectCalls,
    commitCalls: () => commitCalls,
    lastCommitBody: () => lastCommitBody,
  }
}

test('retry and convert rescans an inspect-failed PDF before committing it', async ({ page }) => {
  const backend = await installLibraryUploadBackend(page)
  await page.goto('/library')

  const input = page.locator('.kb-lib-upload-dropzone input[type="file"]')
  await expect(input).toHaveCount(1)
  await input.setInputFiles({
    name: 'raw-upload.pdf',
    mimeType: 'application/pdf',
    buffer: Buffer.from('%PDF-1.4\n% upload retry regression\n'),
  })

  await expect(page.locator('.kb-lib-upload-draft-list')).toContainText('temporary scan outage')
  await expect.poll(() => backend.inspectCalls()).toBe(1)
  await expect.poll(() => backend.commitCalls()).toBe(0)

  await page.getByRole('button', { name: 'Retry & Convert' }).click()

  await expect.poll(() => backend.inspectCalls()).toBe(2)
  await expect.poll(() => backend.commitCalls()).toBe(1)
  await expect(page.locator('.kb-lib-upload-draft-list')).toContainText('Suggested-Paper.pdf')
  expect(backend.lastCommitBody()).toContain('name="base_name"')
  expect(backend.lastCommitBody()).toContain('Suggested-Paper')
  expect(backend.lastCommitBody()).toContain('name="convert_now"')
  expect(backend.lastCommitBody()).toContain('true')
})
