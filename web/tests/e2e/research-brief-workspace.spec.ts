import { expect, test, type Page, type Route } from '@playwright/test'
import {
  installAppShellMocks,
  installEmptyCitationShelfMock,
  installIdleReferenceMocks,
} from './mockAppShell'

const PROJECT = {
  id: 'project-research-brief',
  name: 'Imaging evidence project',
  created_at: 1,
  updated_at: 2,
}

const CONVERSATION = {
  id: 'conv-research-brief',
  title: 'Compare denoising methods',
  created_at: 1,
  updated_at: 2,
  project_id: PROJECT.id,
  mode: 'normal',
}

const SHELF_ITEM = {
  key: 'brief-ref-12',
  num: 12,
  anchor: 'ref-12',
  sourceName: 'Review Paper.pdf',
  sourcePath: 'db/Review/Review.en.md',
  headingPath: 'References',
  blockId: 'ref-block-12',
  title: 'Sparse 3-D transform-domain filtering',
  authors: 'K Dabov, A Foi, V Katkovnik',
  venue: 'IEEE Trans. Image Process.',
  year: '2007',
  doi: '10.1109/tip.2007.901238',
  shelfItemKind: 'reference',
  shelfOrigin: 'reader_references',
  shelfExcerpt: 'A denoising baseline used for comparison.',
  libraryMatchPath: 'db/Library/Sparse3D.en.md',
  libraryMatchStatus: 'ready',
  libraryMatchTitle: 'Sparse 3-D transform-domain filtering',
  main: 'Sparse 3-D transform-domain filtering',
  tags: [],
  note: '',
}

const VERIFIED_MATRIX = {
  id: 'matrix-verified-1',
  project_id: PROJECT.id,
  source_conv_id: CONVERSATION.id,
  title: 'Verified imaging evidence',
  objective: 'Compare methods.',
  rows: [],
  evidence: [],
  source_items: [],
  comparison_flags: [],
  quality_status: 'verified',
  quality: { supported_cell_count: 4, populated_cell_count: 4 },
  revision: 2,
  created_at: 8,
  updated_at: 9,
}

type Brief = {
  id: string
  project_id: string
  source_conv_id: string
  title: string
  objective: string
  content_markdown: string
  evidence: Array<Record<string, unknown>>
  bibliography: Array<Record<string, unknown>>
  agent_trace: Record<string, unknown>
  quality_status: string
  quality: Record<string, unknown>
  revision: number
  created_at: number
  updated_at: number
}

async function fulfillJson(route: Route, body: unknown, status = 200) {
  await route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(body),
  })
}

async function installBackend(page: Page) {
  let brief: Brief | null = null
  let generatedPayload: Record<string, unknown> | null = null
  let savedPayload: Record<string, unknown> | null = null
  let exportedFormat = ''
  const revisions: Brief[] = []

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
      base_url: '',
      has_api_key: true,
      connection: {
        text: { configured: true, connected: true, has_api_key: true, model: 'test-text-model', base_url: '' },
        vision: { configured: true, connected: true, has_api_key: true, model: 'test-vision-model', base_url: '' },
        auto_route: false,
      },
      readiness: {
        overall: { status: 'ok', severity: 'ok', reason: 'Ready' },
        providers: {},
        issues: [],
      },
      prefs: {
        ui_locale: 'en',
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
      overall: { status: 'ok', severity: 'ok', reason: 'Ready' },
      providers: {},
      issues: [],
    })
  })
  await page.route(`**/api/conversations/${CONVERSATION.id}`, async (route) => {
    await fulfillJson(route, CONVERSATION)
  })
  await page.route(`**/api/conversations/${CONVERSATION.id}/messages_page**`, async (route) => {
    await fulfillJson(route, {
      messages: [
        { id: 1, role: 'user', content: 'Compare the selected methods.', created_at: 1 },
        { id: 2, role: 'assistant', content: 'The selected evidence is ready.', created_at: 2 },
      ],
      has_more_before: false,
      oldest_loaded_id: 1,
      newest_loaded_id: 2,
    })
  })
  await page.route(`**/api/conversations/${CONVERSATION.id}/research-state`, async (route) => {
    await fulfillJson(route, {
      conv_id: CONVERSATION.id,
      state: {},
      created_at: 1,
      updated_at: 1,
    })
  })
  await page.route('**/api/references/conversation/**', async (route) => {
    await fulfillJson(route, {})
  })
  await page.route('**/api/references/citation-meta', async (route) => {
    await fulfillJson(route, {})
  })
  await page.route('**/api/references/bibliometrics', async (route) => {
    await fulfillJson(route, { bibliometrics_checked: true })
  })
  await page.route('**/api/library/quality/sources**', async (route) => {
    await fulfillJson(route, { items: [] })
  })
  await page.route(`**/api/projects/${PROJECT.id}/evidence-matrices**`, async (route) => {
    await fulfillJson(route, [VERIFIED_MATRIX])
  })

  await page.route(`**/api/projects/${PROJECT.id}/research-briefs**`, async (route) => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    if (path.endsWith('/generate') && request.method() === 'POST') {
      generatedPayload = request.postDataJSON() as Record<string, unknown>
      brief = {
        id: 'brief-1',
        project_id: PROJECT.id,
        source_conv_id: CONVERSATION.id,
        title: String(generatedPayload.title || ''),
        objective: String(generatedPayload.objective || ''),
        content_markdown: '# Findings\n\nSparse 3-D filtering is the selected comparison baseline [1].',
        evidence: [{
          citation_number: 1,
          source_name: 'Sparse 3-D transform-domain filtering',
          source_path: 'db/Library/Sparse3D.en.md',
          heading_path: 'Method / Transform-domain filtering',
          block_id: 'block-method-1',
          anchor_id: 'method-transform',
          evidence_quote: 'The method groups similar image fragments before collaborative filtering.',
          score: 12.4,
        }],
        bibliography: [{
          citation_number: 1,
          title: 'Sparse 3-D transform-domain filtering',
          authors: 'K Dabov, A Foi, V Katkovnik',
          year: '2007',
          doi: '10.1109/tip.2007.901238',
        }],
        agent_trace: { status: 'done', evidence_status: 'grounded' },
        quality_status: 'verified',
        quality: {
          status: 'verified',
          total_claims: 1,
          supported_claims: 1,
          unsupported_claims: 0,
          support_ratio: 1,
          generation_mode: 'model_synthesis_repaired',
          claim_repair: {
            preserved_model_claims: 3,
            removed_unsupported_claims: 1,
            supplemented_source_claims: 0,
          },
          reasons: [],
        },
        revision: 1,
        created_at: 10,
        updated_at: 10,
      }
      revisions.splice(0, revisions.length, { ...brief })
      await fulfillJson(route, brief)
      return
    }
    await fulfillJson(route, brief ? [brief] : [])
  })

  await page.route(/\/api\/research-briefs\/brief-1(?:\/.*)?(?:\?.*)?$/, async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    if (url.pathname.endsWith('/export')) {
      const format = url.searchParams.get('format') || 'markdown'
      exportedFormat = format
      const suffix = format === 'markdown' ? 'md' : format
      await route.fulfill({
        status: 200,
        contentType: 'text/markdown',
        headers: { 'content-disposition': `attachment; filename="imaging-brief.${suffix}"` },
        body: brief?.content_markdown || '',
      })
      return
    }
    if (url.pathname.endsWith('/revisions')) {
      await fulfillJson(route, [...revisions].reverse())
      return
    }
    if (request.method() === 'PATCH') {
      savedPayload = request.postDataJSON() as Record<string, unknown>
      const previous = brief
      if (!previous) {
        await fulfillJson(route, { detail: 'not found' }, 404)
        return
      }
      brief = {
        ...previous,
        title: String(savedPayload.title ?? previous.title),
        objective: String(savedPayload.objective ?? previous.objective),
        content_markdown: String(savedPayload.content_markdown ?? previous.content_markdown),
        quality_status: 'draft',
        quality: { status: 'draft', reasons: ['edited_after_verification'] },
        revision: previous.revision + 1,
        updated_at: previous.updated_at + 1,
      }
      revisions.push({ ...brief })
      await fulfillJson(route, brief)
      return
    }
    if (!brief) {
      await fulfillJson(route, { detail: 'not found' }, 404)
      return
    }
    await fulfillJson(route, brief)
  })

  return {
    generatedPayload: () => generatedPayload,
    savedPayload: () => savedPayload,
    exportedFormat: () => exportedFormat,
  }
}

test('project basket becomes a versioned, audited, exportable research brief', async ({ page }) => {
  const backend = await installBackend(page)
  await page.goto(`/?conversation=${CONVERSATION.id}`)

  await expect(page).toHaveURL(new RegExp(`conversation=${CONVERSATION.id}`))
  await expect(page.getByTestId('citation-shelf-item')).toHaveCount(1)
  await page.getByTestId('citation-shelf-open-research-brief').click()

  await expect(page.getByRole('dialog', { name: 'Project research briefs' })).toBeVisible()
  await page.getByTestId('research-brief-new').click()
  await page.getByTestId('research-brief-title').fill('Imaging comparison brief')
  await page.getByTestId('research-brief-objective').fill('Compare the selected method without merging experimental conditions.')
  await page.getByTestId('research-brief-generate').click()

  await expect(page.getByText('Evidence audit passed')).toBeVisible()
  await expect(page.getByText('Targeted evidence repair applied')).toBeVisible()
  await expect(page.getByText(/retained 3 supported model claims, removed 1 unsupported or out-of-contract claim/)).toBeVisible()
  await expect(page.getByTestId('research-brief-preview')).toContainText('selected comparison baseline')
  await expect.poll(() => backend.generatedPayload()).toMatchObject({
    title: 'Imaging comparison brief',
    source_conv_id: CONVERSATION.id,
    item_keys: [SHELF_ITEM.key],
    matrix_id: VERIFIED_MATRIX.id,
  })

  await page.getByRole('tab', { name: /Evidence/ }).click()
  await expect(page.getByTestId('research-brief-evidence')).toContainText('Method / Transform-domain filtering')
  await expect(page.getByTestId('research-brief-evidence')).toContainText('groups similar image fragments')

  await page.getByRole('tab', { name: 'Edit' }).click()
  await page.getByTestId('research-brief-content').fill('# Revised findings\n\nA human-edited comparison [1].')
  await page.getByTestId('research-brief-save').click()
  await expect(page.getByText('draft', { exact: true })).toBeVisible()
  await expect.poll(() => backend.savedPayload()).toMatchObject({
    expected_revision: 1,
    content_markdown: '# Revised findings\n\nA human-edited comparison [1].',
  })

  const downloadPromise = page.waitForEvent('download')
  await page.getByTestId('research-brief-export-markdown').click()
  const download = await downloadPromise
  expect(download.suggestedFilename()).toMatch(/\.md$/)
  expect(backend.exportedFormat()).toBe('markdown')
})
