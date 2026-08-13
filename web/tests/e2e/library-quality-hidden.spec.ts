import { expect, test, type Page } from '@playwright/test'
import {
  installAppShellMocks,
  installEmptyCitationShelfMock,
  installIdleReferenceMocks,
} from './mockAppShell'

const libraryItem = {
  name: 'User visible paper.pdf',
  path: 'F:\\kb\\pdfs\\User visible paper.pdf',
  sha1: 'visible-paper-sha1',
  md_exists: true,
  md_path: 'F:\\kb\\md\\User visible paper.en.md',
  md_folder: 'F:\\kb\\md',
  conversion_quality: {
    status: 'warning',
    label: 'Needs review',
    score: 62,
    summary: 'Internal quality issue summary',
    has_review_issue: true,
    issues: [{ code: 'missing_images', label: 'Missing images', severity: 'warning', count: 2 }],
    metrics: {},
  },
  index_state: 'quality_blocked',
  index_status: 'quality_blocked',
  index_ready: false,
  category: 'converted',
  task_state: 'idle',
  status: 'converted',
  replace_task: false,
  queue_pos: 0,
  cur_page_done: 0,
  cur_page_total: 0,
  cur_page_msg: '',
  paper_category: 'Single-Photon Imaging',
  reading_status: 'reading',
  note: '',
  user_tags: ['fixture'],
  has_suggestions: false,
  suggested_category: '',
  suggested_tags: [],
}

const INTERNAL_QUALITY_BUILD =
  process.env.VITE_SHOW_USER_QUALITY_DIAGNOSTICS === '1'
  || process.env.VITE_ENABLE_INTERNAL_DEBUG !== '0'
  || process.env.VITE_ENABLE_INTERNAL_ROUTES !== '0'

async function installLibraryBackend(page: Page, options: { forceSessionQualityDiagnostics?: boolean } = {}) {
  let qualityOverviewCalls = 0
  await page.addInitScript((forceSessionQualityDiagnostics) => {
    window.localStorage.removeItem('kb.library.qualityRepairHistory.v1')
    if (forceSessionQualityDiagnostics) {
      window.sessionStorage.setItem('kb.internal.showQualityDiagnostics', '1')
    } else {
      window.sessionStorage.removeItem('kb.internal.showQualityDiagnostics')
    }
  }, Boolean(options.forceSessionQualityDiagnostics))
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
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
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
          theme: 'dark',
          ui_locale: 'en',
        },
      }),
    })
  })
  await page.route('**/api/library/files**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        items: [libraryItem],
        counts: {
          total_view: 1,
          total_all: 1,
          pending: 0,
          converted: 1,
          queued: 0,
          running: 0,
          reconverting: 0,
          quality_review: 1,
          quality_ready: 0,
          index_quality_blocked: 1,
        },
        truncated: false,
        scope: '200',
        queue: { running: false, active_count: 0, active_tasks: [], current: '', done: 0, total: 0 },
      }),
    })
  })
  await page.route('**/api/library/quality/overview**', async (route) => {
    qualityOverviewCalls += 1
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ok: true,
        status: 'warning',
        summary: {
          total_view: 1,
          total_all: 1,
          converted: 1,
          pending: 0,
          queued: 0,
          running: 0,
          assessed: 1,
          good: 0,
          review: 1,
          unknown: 0,
          avg_score: 62,
        },
        top_issues: [{ code: 'missing_images', label: 'Missing images', severity: 'warning', papers: 1, count: 2 }],
        recommended: [{ name: 'User visible paper.pdf', score: 62, issues: ['Missing images'] }],
        domains: { conversion: { available: true, status: 'warning', summary: {}, top_failures: [{ name: 'missing_images', count: 2 }] } },
        failure_cases: [{ id: 'case-1', question: 'Internal QA question', failures: [{ name: 'citation_missing' }] }],
        queue: { running: false, active_count: 0, active_tasks: [], current: '', done: 0, total: 0 },
        scope: 'all',
        truncated: false,
      }),
    })
  })
  return {
    qualityOverviewCalls: () => qualityOverviewCalls,
  }
}

test('library hides internal quality maintenance from ordinary users by default', async ({ page }) => {
  const backend = await installLibraryBackend(page)
  await page.goto('/library')

  await expect(page.getByTestId('library-file-row')).toHaveCount(1)
  expect(backend.qualityOverviewCalls()).toBe(0)
  await expect(page.getByTestId('library-file-row')).toContainText('User visible paper.pdf')
  await expect(page.getByTestId('library-file-quality-chip')).toHaveCount(0)
  await expect(page.getByTestId('library-file-source-readiness')).toHaveAttribute('data-source-readiness', 'review')
  await expect(page.getByTestId('library-file-row')).not.toContainText(/(?:Review|Repair) Q\d+/)
  await expect(page.getByTestId('library-quality-repair')).toBeVisible()
  await expect(page.getByTestId('library-quality-report')).toHaveCount(0)
  await expect(page.getByTestId('library-quality-center-summary')).toHaveCount(0)
  await expect(page.getByTestId('library-quality-center-toggle')).toHaveCount(0)
  await expect(page.getByTestId('library-quality-issues-filter')).toHaveCount(0)
  await expect(page.getByText('Internal quality issue summary')).toHaveCount(0)
  await expect(page.getByText('Internal QA question')).toHaveCount(0)
})

test('ordinary user build ignores stray quality diagnostics session flag', async ({ page }) => {
  test.skip(INTERNAL_QUALITY_BUILD, 'session diagnostics are intentionally honored only in internal/debug builds')

  const backend = await installLibraryBackend(page, { forceSessionQualityDiagnostics: true })
  await page.goto('/library')

  await expect(page.getByTestId('library-file-row')).toHaveCount(1)
  expect(backend.qualityOverviewCalls()).toBe(0)
  await expect(page.getByTestId('library-file-quality-chip')).toHaveCount(0)
  await expect(page.getByTestId('library-file-source-readiness')).toHaveAttribute('data-source-readiness', 'review')
  await expect(page.getByTestId('library-quality-repair')).toBeVisible()
  await expect(page.getByTestId('library-quality-report')).toHaveCount(0)
  await expect(page.getByTestId('library-quality-center-summary')).toHaveCount(0)
  await expect(page.getByText('Internal quality issue summary')).toHaveCount(0)
  await expect(page.getByText('Internal QA question')).toHaveCount(0)
})
