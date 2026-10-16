import { expect, test, type Page, type Route } from '@playwright/test'
import {
  installAppShellMocks,
  installEmptyCitationShelfMock,
  installIdleReferenceMocks,
} from './mockAppShell'

async function fulfillJson(route: Route, body: unknown) {
  await route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify(body),
  })
}

async function installRecoveryBackend(page: Page) {
  let recoverable = true
  let resumeAllCalls = 0
  await installAppShellMocks(page)
  await installEmptyCitationShelfMock(page, { scopeId: '__default__', projectId: null })
  await installIdleReferenceMocks(page)

  const readiness = {
    providers: {
      text: { target: 'text', has_api_key: true, model: 'test-model', status: 'ok', severity: 'ok', reason: 'ready' },
      vision: { target: 'vision', has_api_key: true, model: 'vision-model', status: 'ok', severity: 'ok', reason: 'ready' },
    },
    overall: { status: 'ok', reason: 'ready', target: '' },
  }
  await page.route('**/api/settings', async (route) => {
    await fulfillJson(route, {
      model: 'test-model',
      base_url: '',
      has_api_key: true,
      connection: { text: readiness.providers.text, vision: readiness.providers.vision, auto_route: true },
      readiness,
      app_readiness: { status: 'ok', env: 'test', production: false, auth_required: false, items: [] },
      db_dir: 'C:/Pi_zaya/data/db',
      library_paths: {
        pdf_dir: 'C:/Pi_zaya/data/pdfs',
        md_dir: 'C:/Pi_zaya/data/markdown',
        uses_managed_defaults: true,
      },
      prefs: { ui_locale: 'en', theme: 'light' },
    })
  })
  await page.route('**/api/settings/readiness', async (route) => {
    await fulfillJson(route, readiness)
  })
  await page.route('**/api/app/onboarding-status', async (route) => {
    await fulfillJson(route, {
      text_model_ready: true,
      imported_document_count: 1,
      ready_document_count: 1,
      grounded_answer_count: 1,
      current_step: 'completed',
      completed: true,
    })
  })
  await page.route('**/api/library/files**', async (route) => {
    const interruptedResult = {
      task_id: 'recover-task',
      name: 'recover.pdf',
      pdf: 'C:/Pi_zaya/data/pdfs/recover.pdf',
      outcome: 'interrupted',
      operation: 'conversion',
      message: 'Conversion was interrupted when Pi_zaya stopped.',
      detail: '',
      retry_action: 'resume',
      replace: true,
      speed_mode: 'balanced',
      started_at: 10,
      finished_at: 20,
      duration_s: 10,
      page_done: 4,
      page_total: 10,
    }
    await fulfillJson(route, {
      items: [{
        name: 'recover.pdf',
        path: 'C:/Pi_zaya/data/pdfs/recover.pdf',
        sha1: 'recover-sha',
        md_exists: false,
        md_path: '',
        md_folder: 'C:/Pi_zaya/data/markdown/recover',
        conversion_quality: null,
        category: 'pending',
        task_state: recoverable ? 'interrupted' : 'idle',
        status: recoverable ? 'interrupted' : 'pending',
        task_id: recoverable ? 'recover-task' : '',
        replace_task: recoverable,
        queue_pos: 0,
        cur_page_done: recoverable ? 4 : 0,
        cur_page_total: recoverable ? 10 : 0,
        cur_page_msg: '',
        conversion_stage: '',
        recoverable,
        cached_page_count: recoverable ? 4 : 0,
        reused_page_count: 0,
        recovery_message: recoverable ? interruptedResult.message : '',
        recovery_blocked_reason: '',
        last_conversion: recoverable ? interruptedResult : null,
        paper_category: '',
        reading_status: '',
        note: '',
        user_tags: [],
        has_suggestions: false,
        suggested_category: '',
        suggested_tags: [],
      }],
      counts: {
        total_view: 1,
        total_all: 1,
        pending: 1,
        converted: 0,
        queued: 0,
        running: 0,
        recoverable: recoverable ? 1 : 0,
        reconverting: 0,
        quality_review: 0,
        quality_ready: 0,
      },
      truncated: false,
      scope: '200',
      queue: {
        running: false,
        active_count: 0,
        active_tasks: [],
        current: '',
        done: 0,
        total: 0,
        recoverable_count: recoverable ? 1 : 0,
        recoverable_tasks: recoverable ? [{ task_id: 'recover-task' }] : [],
      },
    })
  })
  await page.route('**/api/library/convert/resume-all', async (route) => {
    resumeAllCalls += 1
    recoverable = false
    await fulfillJson(route, {
      ok: true,
      requested: 1,
      enqueued: 1,
      blocked: 0,
      skipped_busy: 0,
      items: [{ matched: true, enqueued: true, task_id: 'recover-task', state: 'queued' }],
    })
  })
  await page.route('**/api/library/convert/status', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: 'data: {"running":false,"done":true,"current":"","completed":1,"total":1,"active_count":0,"active_tasks":[],"recoverable_count":0}\n\n',
    })
  })
  await page.route('**/api/library/quality/overview**', async (route) => {
    await fulfillJson(route, {
      ok: true,
      status: 'ok',
      summary: { converted: 0, assessed: 0, good: 0, review: 0, unknown: 1, avg_score: 0 },
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
    resumeAllCalls: () => resumeAllCalls,
  }
}

test('interrupted conversions require an explicit click and offer cache-aware recovery', async ({ page }) => {
  const backend = await installRecoveryBackend(page)
  await page.goto('/library')

  const recovery = page.getByTestId('library-conversion-recovery')
  await expect(recovery).toContainText('1 conversion task(s) can be recovered')
  await expect(recovery).toContainText('does not call paid models automatically')
  await expect(page.getByTestId('library-resume-conversion')).toBeVisible()
  await expect(page.getByTestId('library-conversion-result')).toContainText('4 completed pages may be reused')
  await expect.poll(() => backend.resumeAllCalls()).toBe(0)

  await page.getByTestId('library-resume-all-conversions').click()

  await expect.poll(() => backend.resumeAllCalls()).toBe(1)
  await expect(page.getByTestId('library-conversion-recovery')).toHaveCount(0)
})

for (const scenario of ['ready', 'blocked', 'stale', 'confirmed', 'source-warning', 'replacement-failed'] as const) {
  test(`conversion history respects current source readiness: ${scenario}`, async ({ page }) => {
    await installRecoveryBackend(page)
    const ready = ['ready', 'confirmed', 'source-warning', 'replacement-failed'].includes(scenario)
    await page.route('**/api/library/files**', async route => {
      await fulfillJson(route, {
        items: [{
          name: 'history.pdf', path: 'C:/Pi_zaya/data/pdfs/history.pdf', sha1: 'history',
          md_exists: true, md_path: 'C:/Pi_zaya/data/markdown/history.en.md', md_folder: 'C:/Pi_zaya/data/markdown',
          index_state: ready ? 'ready' : scenario === 'blocked' ? 'quality_blocked' : 'index_stale',
          index_ready: ready, index_status: scenario === 'confirmed' ? 'quality_degraded' : 'ready',
          quality_gate: scenario === 'confirmed' ? { indexable: true, override_applied: true } : {},
          conversion_quality: scenario === 'blocked' ? {
            status: 'review', conversion_report: { needs_reconvert: true },
          } : scenario === 'source-warning' ? {
            has_review_issue: true, conversion_report: { remaining_issue_codes: ['source_text_loss'] },
          } : null,
          category: 'converted', task_state: 'idle', status: 'converted', replace_task: false,
          queue_pos: 0, cur_page_done: 0, cur_page_total: 0, conversion_stage: '',
          last_conversion: {
            task_id: 'old-attempt', name: 'history.pdf', outcome: scenario === 'replacement-failed' ? 'conversion_failed' : 'quality_blocked',
            operation: 'conversion', retry_action: 'reconvert', message: 'Original task failure retained.',
            detail: 'Original task failure retained.', finished_at: 20, duration_s: 10,
          },
          paper_category: '', reading_status: '', note: '', user_tags: [], has_suggestions: false, suggested_tags: [],
        }],
        counts: { total_view: 1, total_all: 1, converted: 1, pending: 0, queued: 0, running: 0, recoverable: 0 },
        scope: '200', queue: { running: false, active_tasks: [], recoverable_count: 0 },
      })
    })
    await page.goto('/library')
    if (scenario === 'ready') {
      const history = page.getByTestId('library-conversion-history')
      await expect(history).toContainText('currently ready for Q&A')
      await expect(page.getByTestId('library-conversion-result')).not.toBeVisible()
      await expect(page.getByTestId('library-retry-conversion')).toHaveCount(0)
      await history.locator('summary').click()
      await expect(page.getByTestId('library-conversion-result')).toBeVisible()
      await expect(page.getByTestId('library-conversion-result')).toContainText('Quality checks blocked indexing')
      await page.getByTestId('library-conversion-result').hover()
      await expect(page.getByRole('tooltip')).toContainText('Original task failure retained.')
    } else {
      await expect(page.getByTestId('library-conversion-history')).toHaveCount(0)
      await expect(page.getByTestId('library-conversion-result')).toBeVisible()
      await expect(page.getByTestId('library-retry-conversion')).toBeVisible()
    }
  })
}
