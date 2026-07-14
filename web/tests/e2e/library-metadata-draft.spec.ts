import { expect, test, type Page, type Route } from '@playwright/test'
import {
  installAppShellMocks,
  installEmptyCitationShelfMock,
  installIdleReferenceMocks,
} from './mockAppShell'

const PAPER_NAME = 'Metadata Draft Fixture.pdf'

type MetadataItem = {
  name: string
  path: string
  sha1: string
  md_exists: boolean
  md_path: string
  md_folder: string
  conversion_quality: null
  index_state: string
  index_status: string
  index_ready: boolean
  category: string
  task_state: string
  status: string
  replace_task: boolean
  queue_pos: number
  cur_page_done: number
  cur_page_total: number
  cur_page_msg: string
  paper_category: string
  reading_status: string
  note: string
  user_tags: string[]
  has_suggestions: boolean
  suggested_category: string
  suggested_tags: string[]
}

type MetaUpdatePayload = {
  pdf_name?: string
  paper_category?: string
  reading_status?: string
  note?: string
  user_tags?: string[]
}

async function fulfillJson(route: Route, body: unknown) {
  await route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify(body),
  })
}

function createLibraryItem(overrides: Partial<MetadataItem> = {}): MetadataItem {
  return {
    name: PAPER_NAME,
    path: `F:\\kb\\pdfs\\${PAPER_NAME}`,
    sha1: 'metadata-draft-fixture',
    md_exists: true,
    md_path: 'F:\\kb\\md\\Metadata Draft Fixture.md',
    md_folder: 'F:\\kb\\md',
    conversion_quality: null,
    index_state: 'ready',
    index_status: 'ready',
    index_ready: true,
    category: 'converted',
    task_state: 'idle',
    status: 'converted',
    replace_task: false,
    queue_pos: 0,
    cur_page_done: 0,
    cur_page_total: 0,
    cur_page_msg: '',
    paper_category: 'Existing category',
    reading_status: 'reading',
    note: 'Persisted note',
    user_tags: ['existing-tag'],
    has_suggestions: true,
    suggested_category: 'Suggested category',
    suggested_tags: ['suggested-tag', 'second-suggestion'],
    ...overrides,
  }
}

async function installLibraryMetadataBackend(
  page: Page,
  overrides: Partial<MetadataItem> = {},
  options: { suggestionApplyDelayMs?: number } = {},
) {
  let item = createLibraryItem(overrides)
  const metaUpdates: MetaUpdatePayload[] = []

  await page.addInitScript(() => {
    window.localStorage.removeItem('kb.library.qualityRepairHistory.v1')
    window.localStorage.removeItem('kb.settings')
    window.sessionStorage.removeItem('kb.internal.showQualityDiagnostics')
  })
  await installAppShellMocks(page)
  await installEmptyCitationShelfMock(page, { scopeId: '__default__', projectId: null })
  await installIdleReferenceMocks(page)

  await page.route('**/api/settings', async (route) => {
    await fulfillJson(route, {
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
      prefs: {
        pdf_dir: 'F:\\kb\\pdfs',
        md_dir: 'F:\\kb\\md',
        theme: 'light',
        ui_locale: 'zh',
      },
    })
  })
  await page.route('**/api/library/files**', async (route) => {
    await fulfillJson(route, {
      items: [item],
      counts: {
        total_view: 1,
        total_all: 1,
        pending: 0,
        converted: 1,
        queued: 0,
        running: 0,
        reconverting: 0,
        quality_review: 0,
        quality_ready: 1,
        index_ready: 1,
        index_quality_blocked: 0,
      },
      truncated: false,
      scope: '200',
      queue: { running: false, active_count: 0, active_tasks: [], current: '', done: 0, total: 0 },
    })
  })
  await page.route('**/api/library/quality/overview**', async (route) => {
    await fulfillJson(route, {
      ok: true,
      status: 'ready',
      summary: {
        total_view: 1,
        total_all: 1,
        converted: 1,
        pending: 0,
        queued: 0,
        running: 0,
        assessed: 0,
        good: 0,
        review: 0,
        unknown: 1,
        avg_score: 0,
      },
      top_issues: [],
      recommended: [],
      domains: {},
      failure_cases: [],
      queue: { running: false, active_count: 0, active_tasks: [], current: '', done: 0, total: 0 },
      scope: 'all',
      truncated: false,
    })
  })
  await page.route('**/api/library/meta/suggestions/apply', async (route) => {
    if (options.suggestionApplyDelayMs) {
      await new Promise((resolve) => setTimeout(resolve, options.suggestionApplyDelayMs))
    }
    const body = route.request().postDataJSON() as {
      category_action?: string
      accept_tags?: string[]
      dismiss_tags?: string[]
      accept_all_tags?: boolean
      dismiss_all_tags?: boolean
    }
    if (body.category_action === 'accept' && item.suggested_category) {
      item = { ...item, paper_category: item.suggested_category, suggested_category: '' }
    }
    const acceptedTags = body.accept_all_tags ? item.suggested_tags : (body.accept_tags || [])
    if (acceptedTags.length > 0) {
      const acceptedKeys = new Set(acceptedTags.map((tag) => tag.toLowerCase()))
      item = {
        ...item,
        user_tags: Array.from(new Set([...item.user_tags, ...acceptedTags])),
        suggested_tags: item.suggested_tags.filter((tag) => !acceptedKeys.has(tag.toLowerCase())),
      }
    }
    const dismissedTags = body.dismiss_all_tags ? item.suggested_tags : (body.dismiss_tags || [])
    if (dismissedTags.length > 0) {
      const dismissedKeys = new Set(dismissedTags.map((tag) => tag.toLowerCase()))
      item = {
        ...item,
        suggested_tags: item.suggested_tags.filter((tag) => !dismissedKeys.has(tag.toLowerCase())),
      }
    }
    item = { ...item, has_suggestions: Boolean(item.suggested_category || item.suggested_tags.length) }
    await fulfillJson(route, { ok: true, ...item })
  })
  await page.route('**/api/library/meta/suggestions/regenerate', async (route) => {
    item = {
      ...item,
      has_suggestions: true,
      suggested_category: 'Refreshed category',
      suggested_tags: ['refreshed-tag'],
    }
    await fulfillJson(route, { ok: true, updated: 1, items: [item] })
  })
  await page.route('**/api/library/meta/update', async (route) => {
    const body = route.request().postDataJSON() as MetaUpdatePayload
    metaUpdates.push(body)
    item = {
      ...item,
      paper_category: String(body.paper_category || ''),
      reading_status: String(body.reading_status || ''),
      note: String(body.note || ''),
      user_tags: Array.isArray(body.user_tags) ? body.user_tags : [],
    }
    await fulfillJson(route, { ok: true, ...item })
  })

  return {
    metaUpdates: () => metaUpdates,
  }
}

async function openMetadataDrawer(page: Page) {
  await page.goto('/library')
  const row = page.getByTestId('library-file-row').filter({ hasText: PAPER_NAME })
  await expect(row).toBeVisible()
  await row.locator('.kb-lib-file-action-main').click()
  await expect(metadataDrawer(page)).toBeVisible()
}

function metadataDrawer(page: Page) {
  return page.getByRole('dialog', { name: new RegExp(`文献元数据.*${PAPER_NAME.replace('.', '\\.')}`) })
}

async function addTag(page: Page, tag: string) {
  const select = page.getByTestId('library-meta-tags')
  await select.click()
  const input = select.locator('input')
  await input.fill(tag)
  await input.press('Enter')
  await expect(select).toContainText(tag)
}

function unsavedDialog(page: Page) {
  return page.getByRole('dialog').filter({ hasText: '保存这次整理吗？' })
}

test('accepting a suggestion preserves unsaved note, status, and unrelated tags', async ({ page }) => {
  const backend = await installLibraryMetadataBackend(page)
  await openMetadataDrawer(page)

  await page.getByTestId('library-meta-note').fill('Manual note before accepting')
  await addTag(page, 'manual-tag')
  const categorySuggestion = page.locator('.kb-lib-suggest-item').filter({ hasText: '建议分类' })
  await categorySuggestion.getByRole('button', { name: /接\s*受/ }).click()

  await expect(page.getByTestId('library-meta-note')).toHaveValue('Manual note before accepting')
  await expect(page.getByTestId('library-meta-tags')).toContainText('existing-tag')
  await expect(page.getByTestId('library-meta-tags')).toContainText('manual-tag')
  await expect(page.getByTestId('library-meta-category').locator('input')).toHaveValue('Suggested category')

  await page.locator('.kb-lib-meta-actions').getByRole('button', { name: /保\s*存/ }).click()
  await expect.poll(() => backend.metaUpdates().length).toBe(1)
  expect(backend.metaUpdates()[0]).toMatchObject({
    paper_category: 'Suggested category',
    reading_status: 'reading',
    note: 'Manual note before accepting',
    user_tags: ['existing-tag', 'manual-tag'],
  })
})

test('saving immediately after accepting all waits for suggestion category and tags', async ({ page }) => {
  const backend = await installLibraryMetadataBackend(page, {}, { suggestionApplyDelayMs: 350 })
  await openMetadataDrawer(page)

  const save = page.getByTestId('library-meta-save')
  const cancel = page.getByTestId('library-meta-cancel')
  await page.getByTestId('library-meta-accept-all').click()

  await expect(save).toBeDisabled()
  await expect(cancel).toBeDisabled()
  await expect(page.getByTestId('library-meta-note')).toBeDisabled()
  await expect(page.locator('.ant-drawer-close')).toHaveCount(0)

  // A rapid user click waits until accepting suggestions has updated the draft.
  await save.click()
  await expect(metadataDrawer(page)).toBeHidden()
  await expect.poll(() => backend.metaUpdates().length).toBe(1)
  expect(backend.metaUpdates()[0]).toMatchObject({
    paper_category: 'Suggested category',
    reading_status: 'reading',
    note: 'Persisted note',
    user_tags: ['existing-tag', 'suggested-tag', 'second-suggestion'],
  })
})

test('refreshing suggestions never replaces the current manual draft', async ({ page }) => {
  await installLibraryMetadataBackend(page)
  await openMetadataDrawer(page)

  await page.getByTestId('library-meta-note').fill('Manual note before refresh')
  await addTag(page, 'manual-refresh-tag')
  await page.getByRole('button', { name: '刷新建议', exact: true }).click()

  await expect(page.getByTestId('library-meta-note')).toHaveValue('Manual note before refresh')
  await expect(page.getByTestId('library-meta-tags')).toContainText('existing-tag')
  await expect(page.getByTestId('library-meta-tags')).toContainText('manual-refresh-tag')
  await expect(page.getByTestId('library-meta-category').locator('input')).toHaveValue('Existing category')
  await expect(page.locator('.kb-lib-suggest-item').filter({ hasText: 'Refreshed category' })).toBeVisible()
})

test('dismissing a suggestion preserves a manually added tag with the same text', async ({ page }) => {
  const backend = await installLibraryMetadataBackend(page)
  await openMetadataDrawer(page)

  await addTag(page, 'suggested-tag')
  const suggestedTag = page.locator('.kb-lib-suggest-item').filter({ hasText: 'suggested-tag' })
  await suggestedTag.getByRole('button').last().click()

  await expect(suggestedTag).toHaveCount(0)
  await expect(page.getByTestId('library-meta-tags')).toContainText('existing-tag')
  await expect(page.getByTestId('library-meta-tags')).toContainText('suggested-tag')

  await page.locator('.kb-lib-meta-actions').getByRole('button', { name: /保\s*存/ }).click()
  await expect.poll(() => backend.metaUpdates().length).toBe(1)
  expect(backend.metaUpdates()[0]?.user_tags).toEqual(['existing-tag', 'suggested-tag'])
})

test('dismissing a duplicate suggestion preserves the already persisted tag', async ({ page }) => {
  const backend = await installLibraryMetadataBackend(page, {
    user_tags: ['existing-tag', 'suggested-tag'],
    suggested_tags: ['suggested-tag', 'second-suggestion'],
  })
  await openMetadataDrawer(page)

  const suggestedTag = page.locator('.kb-lib-suggest-item').filter({ hasText: 'suggested-tag' })
  await suggestedTag.getByRole('button').last().click()

  await expect(suggestedTag).toHaveCount(0)
  await expect(page.getByTestId('library-meta-tags')).toContainText('existing-tag')
  await expect(page.getByTestId('library-meta-tags')).toContainText('suggested-tag')

  await page.locator('.kb-lib-meta-actions').getByRole('button', { name: /保\s*存/ }).click()
  await expect.poll(() => backend.metaUpdates().length).toBe(1)
  expect(backend.metaUpdates()[0]?.user_tags).toEqual(['existing-tag', 'suggested-tag'])
})

test('dirty drawer close paths offer continue, discard, and save without silent loss', async ({ page }) => {
  const backend = await installLibraryMetadataBackend(page)
  await openMetadataDrawer(page)

  const note = page.getByTestId('library-meta-note')
  await note.fill('Keep editing this note')
  await page.locator('.kb-lib-meta-actions').getByRole('button', { name: /取\s*消/ }).click()
  await expect(unsavedDialog(page)).toBeVisible()
  await unsavedDialog(page).getByRole('button', { name: '继续编辑', exact: true }).click()
  await expect(note).toHaveValue('Keep editing this note')

  await page.locator('.ant-drawer-close').click()
  await expect(unsavedDialog(page)).toBeVisible()
  await unsavedDialog(page).getByRole('button', { name: '放弃修改', exact: true }).click()
  await expect(metadataDrawer(page)).toBeHidden()
  expect(backend.metaUpdates()).toHaveLength(0)

  await openMetadataDrawer(page)
  await page.getByTestId('library-meta-note').fill('Save from escape path')
  await page.keyboard.press('Escape')
  await expect(unsavedDialog(page)).toBeVisible()
  await unsavedDialog(page).getByRole('button', { name: '保存并关闭', exact: true }).click()
  await expect(metadataDrawer(page)).toBeHidden()
  await expect.poll(() => backend.metaUpdates().length).toBe(1)
  expect(backend.metaUpdates()[0]?.note).toBe('Save from escape path')

  await openMetadataDrawer(page)
  await page.getByTestId('library-meta-note').fill('Mask close keeps draft')
  await page.locator('.ant-drawer-mask').click({ position: { x: 20, y: 20 } })
  await expect(unsavedDialog(page)).toBeVisible()
  await unsavedDialog(page).getByRole('button', { name: '继续编辑', exact: true }).click()
  await expect(page.getByTestId('library-meta-note')).toHaveValue('Mask close keeps draft')
})
