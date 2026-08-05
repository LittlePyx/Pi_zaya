import { expect, test, type Page, type Route } from '@playwright/test'
import {
  installAppShellMocks,
  installEmptyCitationShelfMock,
  installIdleReferenceMocks,
} from './mockAppShell'

const PROJECT = {
  id: 'project-evidence-matrix',
  name: 'Evidence matrix project',
  created_at: 1,
  updated_at: 2,
}

const CONVERSATION = {
  id: 'conv-evidence-matrix',
  title: 'Compare imaging methods',
  created_at: 1,
  updated_at: 2,
  project_id: PROJECT.id,
  mode: 'normal',
}

const SHELF_ITEM = {
  key: 'matrix-paper-a',
  num: 1,
  anchor: 'paper-a-anchor',
  sourceName: 'Paper A',
  sourcePath: 'db/Library/PaperA.en.md',
  headingPath: 'Results',
  blockId: 'paper-a-block',
  title: 'Paper A',
  main: 'Paper A',
  authors: 'Ada Author',
  venue: 'Optics Letters',
  year: '2025',
  doi: '10.1000/paper-a',
  shelfItemKind: 'citation',
  shelfOrigin: 'answer',
  shelfExcerpt: 'Paper A reports measured reconstruction results.',
  libraryMatchPath: 'db/Library/PaperA.en.md',
  libraryMatchStatus: 'ready',
  libraryMatchTitle: 'Paper A',
  tags: [],
  note: '',
}

type Matrix = {
  id: string
  project_id: string
  source_conv_id: string
  title: string
  objective: string
  rows: Array<Record<string, unknown>>
  evidence: Array<Record<string, unknown>>
  source_items: Array<Record<string, unknown>>
  comparison_flags: Array<Record<string, unknown>>
  quality_status: string
  quality: Record<string, unknown>
  revision: number
  created_at: number
  updated_at: number
}

async function fulfillJson(route: Route, body: unknown, status = 200) {
  await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
}

async function installBackend(page: Page) {
  let matrix: Matrix | null = null
  let generatedPayload: Record<string, unknown> | null = null
  let savedPayload: Record<string, unknown> | null = null
  const revisions: Matrix[] = []

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
      model: 'test-text-model',
      has_api_key: true,
      connection: {
        text: { configured: true, connected: true, has_api_key: true, model: 'test-text-model', base_url: '' },
        vision: { configured: true, connected: true, has_api_key: true, model: 'test-vision-model', base_url: '' },
        auto_route: false,
      },
      readiness: { overall: { status: 'ok', severity: 'ok', reason: 'Ready' }, providers: {}, issues: [] },
      prefs: { ui_locale: 'en', theme: 'light', top_k: 6, temperature: 0.2, max_tokens: 1216, deep_read: false },
    })
  })
  await page.route('**/api/settings/readiness', async (route) => {
    await fulfillJson(route, { overall: { status: 'ok', severity: 'ok', reason: 'Ready' }, providers: {}, issues: [] })
  })
  await page.route(`**/api/conversations/${CONVERSATION.id}`, async (route) => {
    await fulfillJson(route, CONVERSATION)
  })
  await page.route(`**/api/conversations/${CONVERSATION.id}/messages_page**`, async (route) => {
    await fulfillJson(route, {
      messages: [
        { id: 1, role: 'user', content: 'Compare the selected papers.', created_at: 1 },
        { id: 2, role: 'assistant', content: 'The selected evidence is ready.', created_at: 2 },
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
  await page.route(`**/api/projects/${PROJECT.id}/research-briefs**`, async (route) => { await fulfillJson(route, []) })

  await page.route(`**/api/projects/${PROJECT.id}/evidence-matrices**`, async (route) => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    if (path.endsWith('/generate') && request.method() === 'POST') {
      generatedPayload = request.postDataJSON() as Record<string, unknown>
      matrix = {
        id: 'matrix-1',
        project_id: PROJECT.id,
        source_conv_id: CONVERSATION.id,
        title: String(generatedPayload.title || ''),
        objective: String(generatedPayload.objective || ''),
        rows: [{
          id: 'row-a',
          source_item_key: SHELF_ITEM.key,
          paper: 'Paper A',
          source_name: 'Paper A',
          source_path: SHELF_ITEM.sourcePath,
          year: '2025',
          notes: '',
          source_status: 'active',
          cells: {
            method: { field: 'method', value: 'The method uses a coded optical network.', support_status: 'grounded', evidence_ids: ['ev-method'], manual_override: false },
            dataset_or_experiment: { field: 'dataset_or_experiment', value: '', support_status: 'missing', evidence_ids: [], manual_override: false },
            metric: { field: 'metric', value: 'The evaluation metric is PSNR.', support_status: 'grounded', evidence_ids: ['ev-metric'], manual_override: false },
            key_result: { field: 'key_result', value: 'Results reach 31 dB PSNR.', support_status: 'grounded', evidence_ids: ['ev-result'], manual_override: false },
            limitation: { field: 'limitation', value: '', support_status: 'missing', evidence_ids: [], manual_override: false },
          },
        }],
        evidence: [{
          id: 'ev-method',
          field: 'method',
          source_name: 'Paper A',
          source_path: SHELF_ITEM.sourcePath,
          heading_path: 'Method / Architecture',
          block_id: 'block-method',
          evidence_quote: 'The method uses a coded optical network.',
        }],
        source_items: [{ key: SHELF_ITEM.key, title: 'Paper A', sourcePath: SHELF_ITEM.sourcePath }],
        comparison_flags: [{ code: 'metrics_differ', message: 'Reported metrics differ across sources.' }],
        quality_status: 'verified',
        quality: { supported_cell_count: 3, populated_cell_count: 3, missing_cell_count: 2, reasons: [] },
        revision: 1,
        created_at: 10,
        updated_at: 10,
      }
      revisions.splice(0, revisions.length, { ...matrix })
      await fulfillJson(route, matrix)
      return
    }
    await fulfillJson(route, matrix ? [matrix] : [])
  })

  await page.route(/\/api\/evidence-matrices\/matrix-1(?:\/.*)?(?:\?.*)?$/, async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    if (url.pathname.endsWith('/revisions')) {
      await fulfillJson(route, [...revisions].reverse())
      return
    }
    if (request.method() === 'PATCH') {
      savedPayload = request.postDataJSON() as Record<string, unknown>
      if (!matrix) {
        await fulfillJson(route, { detail: 'not found' }, 404)
        return
      }
      const rowUpdates = savedPayload.row_updates as Array<Record<string, unknown>>
      const firstUpdate = rowUpdates[0]
      const cells = firstUpdate.cells as Array<{ field: string; value: string }>
      const methodValue = cells.find((cell) => cell.field === 'method')?.value || ''
      const firstRow = matrix.rows[0] as Record<string, unknown>
      matrix = {
        ...matrix,
        rows: [{
          ...firstRow,
          notes: String(firstUpdate.notes || ''),
          cells: {
            ...(firstRow.cells as Record<string, unknown>),
            method: { field: 'method', value: methodValue, support_status: 'needs_review', evidence_ids: ['ev-method'], manual_override: true },
          },
        }],
        quality_status: 'needs_review',
        quality: { reasons: ['edited_after_verification'], supported_cell_count: 2, populated_cell_count: 3 },
        revision: 2,
        updated_at: 11,
      }
      revisions.push({ ...matrix })
      await fulfillJson(route, matrix)
      return
    }
    await fulfillJson(route, matrix || { detail: 'not found' }, matrix ? 200 : 404)
  })

  return {
    generatedPayload: () => generatedPayload,
    savedPayload: () => savedPayload,
  }
}

test('project basket becomes a persistent cell-audited evidence matrix', async ({ page }) => {
  const backend = await installBackend(page)
  await page.goto(`/?conversation=${CONVERSATION.id}`)

  await expect(page.getByTestId('citation-shelf-item')).toHaveCount(1)
  await page.getByTestId('citation-shelf-open-evidence-matrix').click()
  await expect(page.getByRole('dialog', { name: 'Project evidence matrices' })).toBeVisible()
  await page.getByRole('button', { name: 'New matrix' }).click()
  await page.getByTestId('evidence-matrix-title').fill('Imaging evidence matrix')
  await page.getByTestId('evidence-matrix-objective').fill('Compare methods without merging experimental conditions.')
  await page.getByTestId('evidence-matrix-generate').click()

  await expect(page.getByText('Cell-level evidence audit passed')).toBeVisible()
  await expect(page.getByTestId('project-evidence-matrix-row')).toHaveCount(1)
  await expect(page.getByTestId('project-evidence-matrix-row')).toContainText('coded optical network')
  await expect(page.getByTestId('project-evidence-matrix-row').locator('textarea').nth(1)).toHaveValue('')
  await expect(page.getByTestId('project-evidence-matrix-row').locator('textarea').nth(4)).toHaveValue('')
  await expect.poll(() => backend.generatedPayload()).toMatchObject({
    title: 'Imaging evidence matrix',
    source_conv_id: CONVERSATION.id,
    item_keys: [SHELF_ITEM.key],
  })

  const row = page.getByTestId('project-evidence-matrix-row')
  await row.locator('textarea').nth(0).fill('A human-edited method summary.')
  await row.locator('textarea').nth(5).fill('Keep this note on source refresh.')
  await page.getByTestId('evidence-matrix-save').click()
  await expect(page.getByText('This matrix needs review')).toBeVisible()
  await expect.poll(() => backend.savedPayload()).toMatchObject({ expected_revision: 1 })

  await page.getByRole('tab', { name: /Evidence/ }).click()
  await expect(page.getByTestId('project-evidence-matrix-evidence')).toContainText('Method / Architecture')
  await expect(page.getByTestId('project-evidence-matrix-evidence')).toContainText('coded optical network')
})
