import { expect, test, type Page, type Route } from '@playwright/test'
import {
  installAppShellMocks,
  installEmptyCitationShelfMock,
  installIdleReferenceMocks,
} from './mockAppShell'

const PROJECT = { id: 'project-research-gap', name: 'Research gap project', created_at: 1, updated_at: 2 }
const CONVERSATION = {
  id: 'conv-research-gap',
  title: 'Review evidence gaps',
  created_at: 1,
  updated_at: 2,
  project_id: PROJECT.id,
  mode: 'normal',
}
const SHELF_ITEM = {
  key: 'existing-paper',
  anchor: 'existing-anchor',
  sourceName: 'Existing paper',
  sourcePath: 'db/Library/existing.en.md',
  title: 'Existing paper',
  main: 'Existing paper',
  shelfItemKind: 'citation',
  shelfOrigin: 'answer',
  shelfExcerpt: 'Existing grounded evidence.',
  libraryMatchPath: 'db/Library/existing.en.md',
  libraryMatchStatus: 'ready',
  tags: [],
  note: '',
}

async function fulfillJson(route: Route, body: unknown, status = 200) {
  await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
}

async function installBackend(page: Page) {
  let status = 'open'
  const gap = () => ({
    id: 'gap-limitation',
    gap_key: 'gap-key-limitation',
    contract_version: 1,
    project_id: PROJECT.id,
    kind: 'missing_cell',
    status,
    severity: 'warning',
    priority: 'medium',
    priority_score: 58,
    title: 'Paper A: limitation',
    detail: 'The matrix has no source-grounded value for this field.',
    matrix_id: 'matrix-1',
    matrix_title: 'Imaging evidence matrix',
    matrix_revision: 3,
    brief_id: '',
    brief_title: '',
    brief_revision: 0,
    row_id: 'row-a',
    row_label: 'Paper A',
    field: 'limitation',
    comparison_id: '',
    source_path: SHELF_ITEM.sourcePath,
    source_name: 'Paper A',
    reasons: ['missing_cells'],
    impact: { affected_brief_count: 1, affected_citation_count: 2, affected_comparison_count: 1 },
    candidate_query: 'dynamic imaging limitation failure trade-off',
    candidate_searchable: true,
    dismissible: true,
    action: {},
    created_at: 1,
    updated_at: 2,
  })
  const candidate = {
    id: 'candidate-new-paper',
    gap_id: 'gap-limitation',
    gap_key: 'gap-key-limitation',
    source_path: 'db/Library/new-candidate.en.md',
    source_name: 'New candidate paper',
    title: 'New candidate paper',
    chunk_id: 'new-candidate:7',
    score: 4.2,
    evidence_quote: 'Dynamic reconstruction remains limited by motion and calibration errors.',
    heading_path: 'Discussion / Limitations',
    location_label: 'Discussion / Limitations',
    page_start: 7,
    page_end: 7,
    block_id: 'candidate-block-7',
    anchor_id: 'candidate-anchor-7',
    matched_terms: ['dynamic', 'limitation'],
    match_reason: 'Local indexed passage shares the deterministic gap query terms.',
  }

  await installAppShellMocks(page, {
    projects: [PROJECT],
    projectConversations: { [PROJECT.id]: [CONVERSATION] },
  })
  await installEmptyCitationShelfMock(page, {
    scopeId: PROJECT.id,
    projectId: PROJECT.id,
    initialItems: [SHELF_ITEM],
    initialOpen: true,
  })
  await installIdleReferenceMocks(page)

  await page.route('**/api/settings', async (route) => {
    await fulfillJson(route, {
      model: 'test-model',
      has_api_key: true,
      connection: { text: { configured: true, connected: true, has_api_key: true }, vision: { configured: false }, auto_route: false },
      readiness: { overall: { status: 'ok', severity: 'ok', reason: 'Ready' }, providers: {}, issues: [] },
      prefs: { ui_locale: 'en', theme: 'light', top_k: 6, temperature: 0.2, max_tokens: 1216, deep_read: false },
    })
  })
  await page.route('**/api/settings/readiness', async (route) => {
    await fulfillJson(route, { overall: { status: 'ok', severity: 'ok', reason: 'Ready' }, providers: {}, issues: [] })
  })
  await page.route(`**/api/conversations/${CONVERSATION.id}`, async (route) => { await fulfillJson(route, CONVERSATION) })
  await page.route(`**/api/conversations/${CONVERSATION.id}/messages_page**`, async (route) => {
    await fulfillJson(route, {
      messages: [
        { id: 1, role: 'user', content: 'Review the project evidence gaps.', created_at: 1 },
        { id: 2, role: 'assistant', content: 'The evidence audit is ready.', created_at: 2 },
      ],
      has_more_before: false,
      oldest_loaded_id: 1,
      newest_loaded_id: 2,
    })
  })
  await page.route(`**/api/conversations/${CONVERSATION.id}/research-state`, async (route) => {
    await fulfillJson(route, { conv_id: CONVERSATION.id, state: {}, created_at: 1, updated_at: 1 })
  })
  await page.route('**/api/references/conversation/**', async (route) => { await fulfillJson(route, {}) })
  await page.route('**/api/references/citation-meta', async (route) => { await fulfillJson(route, {}) })
  await page.route('**/api/references/bibliometrics', async (route) => { await fulfillJson(route, { bibliometrics_checked: true }) })
  await page.route('**/api/library/quality/sources**', async (route) => { await fulfillJson(route, { items: [] }) })

  await page.route(`**/api/projects/${PROJECT.id}/research-gaps/scan`, async (route) => {
    await fulfillJson(route, {
      items: [gap()],
      summary: { total: 1, open: status === 'open' ? 1 : 0, in_progress: status === 'in_progress' ? 1 : 0, high: 0, medium: 1, low: 0, searchable: 1, affected_matrix_count: 1, affected_brief_count: 1 },
      scanned_at: 3,
      matrix_count: 1,
      brief_count: 1,
      source_change_count: 0,
    })
  })
  await page.route(`**/api/projects/${PROJECT.id}/research-gaps/gap-limitation/candidates**`, async (route) => {
    await fulfillJson(route, { items: [candidate], query: gap().candidate_query, gap_id: gap().id })
  })
  await page.route(`**/api/projects/${PROJECT.id}/research-gaps/gap-limitation/candidates/${candidate.id}/confirm`, async (route) => {
    status = 'in_progress'
    await fulfillJson(route, {
      gap: gap(),
      candidate,
      shelf: {
        version: 1,
        scope: 'project',
        scope_id: PROJECT.id,
        project_id: PROJECT.id,
        items: [{
          key: `research-gap:${candidate.id}`,
          anchor: candidate.anchor_id,
          title: candidate.title,
          main: candidate.title,
          sourceName: candidate.source_name,
          sourcePath: candidate.source_path,
          shelfItemKind: 'citation',
          shelfOrigin: 'research_gap',
          shelfExcerpt: candidate.evidence_quote,
          evidenceQuote: candidate.evidence_quote,
          headingPath: candidate.heading_path,
          locationLabel: candidate.location_label,
          pageStart: candidate.page_start,
          blockId: candidate.block_id,
        }, SHELF_ITEM],
        open: true,
        revision: 2,
        created_at: 1,
        updated_at: 4,
      },
    })
  })
}

test('project gap queue exposes impact and requires human confirmation for candidate evidence', async ({ page }) => {
  const pageErrors: string[] = []
  page.on('pageerror', (error) => pageErrors.push(error.message))
  await installBackend(page)
  await page.goto(`/?conversation=${CONVERSATION.id}`)

  await page.getByTestId('citation-shelf-open-research-gaps').click()
  await expect.poll(() => pageErrors).toEqual([])
  await page.waitForTimeout(200)
  expect(await page.locator('.ant-message-notice-content').allTextContents()).toEqual([])
  const dialog = page.getByRole('dialog', { name: 'Project research gap queue' })
  await expect(dialog).toBeVisible()
  await expect(dialog.getByTestId('research-gap-card')).toContainText('Paper A: limitation')
  await expect(dialog.getByTestId('research-gap-card')).toContainText('1 briefs')
  await expect(dialog.getByTestId('research-gap-card')).toContainText('2 citations')

  await dialog.getByTestId('research-gap-find-candidates').click()
  await expect(dialog.getByTestId('research-gap-candidates')).toContainText('Dynamic reconstruction remains limited by motion and calibration errors.')
  await dialog.getByTestId('research-gap-confirm-candidate').click()
  await page.getByRole('button', { name: 'Confirm candidate', exact: true }).last().click()
  await expect(dialog.getByTestId('research-gap-card')).toContainText('Evidence selected')
  await dialog.getByRole('button', { name: 'Close' }).click()
  await expect(page.getByTestId('citation-shelf-item')).toHaveCount(2)
})
