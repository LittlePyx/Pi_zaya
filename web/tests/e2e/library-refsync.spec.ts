import { expect, test, type Route } from '@playwright/test'
import { installAppShellMocks } from './mockAppShell'

async function fulfillJson(route: Route, body: unknown, status = 200) {
  await route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(body),
  })
}

test('reference sync metrics keep explicit zero progress instead of falling back to indexed stats', async ({ page }) => {
  await installAppShellMocks(page)
  await page.route('**/api/settings', async (route) => {
    await fulfillJson(route, {
      model: 'test-model',
      base_url: '',
      has_api_key: true,
      db_dir: 'F:\\kb\\db',
      prefs: {
        pdf_dir: 'F:\\kb\\pdfs',
        md_dir: 'F:\\kb\\md',
        theme: 'light',
        ui_locale: 'zh',
      },
    })
  })
  await page.route('**/api/conversations**', async (route) => {
    await fulfillJson(route, [])
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
  await page.route('**/api/library/files**', async (route) => {
    await fulfillJson(route, {
      items: [],
      counts: {
        total_view: 0,
        total_all: 0,
        pending: 0,
        converted: 0,
        queued: 0,
        running: 0,
        reconverting: 0,
        quality_review: 0,
        quality_ready: 0,
      },
      truncated: false,
      scope: '200',
      queue: {
        running: false,
        queued_count: 0,
        active_count: 0,
        active_tasks: [],
        current: '',
        done: 0,
        total: 0,
      },
    })
  })
  await page.route('**/api/library/quality/overview**', async (route) => {
    await fulfillJson(route, { ok: true, items: [], summary: {} })
  })
  await page.route('**/api/references/sync/status', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: [
        'data: {"running":false,"done":true,"status":"done","stage":"done","message":"done","current":"","docs_done":0,"docs_total":21,"stats":{"docs_total":21,"docs_indexed":6,"refs_total":1304,"refs_metadata_user_ready":416,"refs_metadata_ready":387,"refs_crossref_ok":140,"crossref_network_attempts":88,"elapsed_s":3.2}}',
        '',
        '',
      ].join('\n'),
    })
  })

  await page.goto('/library')

  const metrics = page.locator('.kb-lib-refsync-metrics')
  await expect(metrics).toContainText('0/21')
  await expect(metrics).not.toContainText('6/21')
})
