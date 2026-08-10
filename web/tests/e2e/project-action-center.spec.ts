import { expect, test, type Page, type Route } from '@playwright/test'
import {
  installAppShellMocks,
  installEmptyCitationShelfMock,
  installIdleReferenceMocks,
} from './mockAppShell'


const PROJECT = { id: 'project-status', name: 'Living Evidence Review', created_at: 1, updated_at: 2 }

async function fulfillJson(route: Route, body: unknown, status = 200) {
  await route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(body),
  })
}

const MATRIX = {
  id: 'matrix-status',
  project_id: PROJECT.id,
  source_conv_id: null,
  title: 'Audited comparison matrix',
  objective: 'Compare two imaging methods.',
  rows: [
    {
      id: 'row-a',
      source_item_key: 'paper-a',
      paper: 'Paper A',
      source_name: 'Paper A',
      source_path: '/papers/a.md',
      source_status: 'active',
      notes: '',
      cells: {},
    },
    {
      id: 'row-b',
      source_item_key: 'paper-b',
      paper: 'Paper B',
      source_name: 'Paper B',
      source_path: '/papers/b.md',
      source_status: 'active',
      notes: '',
      cells: {},
    },
  ],
  evidence: [],
  source_items: [],
  comparison_flags: [],
  comparison_audits: [],
  quality_status: 'verified',
  quality: { supported_cell_count: 10, populated_cell_count: 10 },
  revision: 3,
  created_at: 1,
  updated_at: 3,
}

function statusPayload() {
  return {
    contract_version: 1,
    project: { id: PROJECT.id, name: PROJECT.name },
    readiness: 'needs_review',
    stages: {
      sources: {
        status: 'ready',
        project_source_count: 2,
        shelf_source_count: 2,
        matrix_source_count: 2,
        changed_source_count: 0,
      },
      matrices: {
        status: 'ready',
        total: 1,
        verified: 1,
        needs_review: 0,
        latest_matrix_id: MATRIX.id,
        latest_matrix_title: MATRIX.title,
        latest_matrix_revision: MATRIX.revision,
      },
      evidence: {
        status: 'ready',
        active_gap_count: 0,
        unsupported_count: 0,
        missing_count: 0,
        matrix_review_count: 0,
      },
      comparisons: {
        status: 'needs_review',
        verified_count: 0,
        not_comparable_count: 0,
        boundary_gap_count: 0,
        pending_candidate_count: 2,
        eligible_matrix_count: 1,
        scanned_matrix_count: 1,
        skipped_stale_matrix_count: 0,
        scan_complete: true,
      },
      briefs: {
        status: 'not_started',
        total: 0,
        verified: 0,
        current: 0,
        stale: 0,
        lineage_blocked: 0,
        latest_brief_id: '',
        latest_brief_title: '',
        latest_brief_revision: 0,
      },
    },
    active_gap_count: 0,
    gap_counts: {},
    recommended_action: {
      code: 'review_comparison_candidates',
      target: 'evidence_matrix',
      priority: 75,
      reason: 'evidence_bound_comparisons_await_human_confirmation',
      matrix_id: MATRIX.id,
      matrix_title: MATRIX.title,
      matrix_revision: MATRIX.revision,
      brief_id: '',
      brief_title: '',
      brief_revision: 0,
      gap_count: 0,
      candidate_count: 2,
      workspace_tab: 'comparisons',
    },
    refreshed: true,
    generated_at: 10,
    comparison_scan: {
      candidate_count: 2,
      eligible_matrix_count: 1,
      scanned_matrix_count: 1,
      skipped_stale_matrix_count: 0,
      scan_complete: true,
      examined_row_pairs: 1,
      structured_observation_count: 4,
      elapsed_ms: 11.2,
      matrix_results: [],
    },
    phase_timings_ms: {
      load_artifacts: 3.1,
      scan_and_sync_gaps: 4.2,
      scan_comparison_candidates: 11.2,
      assemble: 0.3,
      total: 18.8,
    },
  }
}

async function installBackend(page: Page) {
  await installAppShellMocks(page, { projects: [PROJECT], projectConversations: { [PROJECT.id]: [] } })
  await installEmptyCitationShelfMock(page, {
    scopeId: PROJECT.id,
    projectId: PROJECT.id,
    initialItems: [
      { key: 'paper-a', anchor: 'a', sourceName: 'Paper A', sourcePath: '/papers/a.md', main: 'Paper A' },
      { key: 'paper-b', anchor: 'b', sourceName: 'Paper B', sourcePath: '/papers/b.md', main: 'Paper B' },
    ],
  })
  await installIdleReferenceMocks(page)
  await page.route('**/api/settings', async (route) => {
    await fulfillJson(route, {
      model: 'test-model',
      has_api_key: true,
      connection: {
        text: { has_api_key: true, model: 'test-model', base_url: '' },
        vision: { has_api_key: true, model: 'test-model', base_url: '', uses_text_fallback: false },
        auto_route: false,
      },
      readiness: { overall: { status: 'ok', severity: 'ok', message: 'Ready' }, providers: {}, issues: [] },
      app_readiness: { status: 'ok', env: 'test', production: false, auth_required: false, items: [] },
      prefs: { ui_locale: 'en', theme: 'light', sidebar_collapsed: false },
    })
  })
  await page.route(`**/api/projects/${PROJECT.id}/research-status/refresh`, async (route) => {
    await fulfillJson(route, statusPayload())
  })
  await page.route(new RegExp(`/api/projects/${PROJECT.id}/evidence-matrices(?:\\?.*)?$`), async (route) => {
    await fulfillJson(route, [MATRIX])
  })
  await page.route(`**/api/evidence-matrices/${MATRIX.id}`, async (route) => {
    await fulfillJson(route, MATRIX)
  })
  await page.route(`**/api/evidence-matrices/${MATRIX.id}/revisions**`, async (route) => {
    await fulfillJson(route, [MATRIX])
  })
  await page.route(`**/api/projects/${PROJECT.id}/evidence-changes/scan`, async (route) => {
    await fulfillJson(route, {
      items: [],
      summary: { total: 0, actionable: 0, metadata_only: 0, high_severity: 0, affected_matrix_count: 0, affected_brief_count: 0 },
    })
  })
}

test('project menu opens measured status and the primary action lands on comparison review', async ({ page }) => {
  await installBackend(page)
  await page.goto('/')

  await page.getByLabel('Project actions').click()
  await page.getByText('Research status', { exact: true }).click()

  await expect(page.getByTestId('project-action-center')).toBeVisible()
  await expect(page.getByTestId('project-status-project-name')).toHaveText(PROJECT.name)
  await expect(page.getByTestId('project-status-stage-comparisons')).toContainText('scanned 1/1')
  await expect(page.getByTestId('project-status-coverage')).toContainText('1 row pairs')
  await expect(page.getByTestId('project-status-primary-action')).toContainText('2 review candidates')

  await page.getByTestId('project-status-action-review_comparison_candidates').click()

  await expect(page.getByText('Project evidence matrices', { exact: true })).toBeVisible()
  await expect(page.getByTestId('evidence-comparison-find-candidates')).toBeVisible()
  await expect(page.getByText('Explicit comparison contract', { exact: true })).toBeVisible()
})
