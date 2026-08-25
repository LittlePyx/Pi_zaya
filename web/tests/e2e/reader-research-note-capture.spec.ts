import { expect, test, type Page, type Route } from '@playwright/test'
import { installAppShellMocks } from './mockAppShell'

const SOURCE_SENTENCE = 'Our method exploits neural radiance fields (NeRF) for snapshot compressed imaging.'

async function fulfillJson(route: Route, body: unknown, status = 200) {
  await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
}

async function selectText(page: Page, needle: string) {
  await expect(page.getByTestId('reader-content')).toContainText(needle)
  const selected = await page.evaluate((text) => {
    const root = document.querySelector('[data-testid="reader-content"]')
    if (!root) return false
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT)
    while (walker.nextNode()) {
      const node = walker.currentNode as Text
      const content = String(node.textContent || '')
      const start = content.indexOf(text)
      if (start < 0) continue
      node.parentElement?.scrollIntoView({ block: 'center' })
      const range = document.createRange()
      range.setStart(node, start)
      range.setEnd(node, start + text.length)
      const selection = window.getSelection()
      selection?.removeAllRanges()
      selection?.addRange(range)
      root.dispatchEvent(new MouseEvent('mouseup', { bubbles: true }))
      return true
    }
    return false
  }, needle)
  expect(selected).toBeTruthy()
  await expect(page.getByTestId('reader-selection-bubble')).toBeVisible()
}

function noteRecord(overrides: Record<string, unknown> = {}) {
  return {
    id: 'note-existing',
    project_id: 'project-reader-note',
    source_conv_id: null,
    title: 'Existing evidence note',
    content_markdown: '## Existing finding\n\nKeep this conclusion.',
    source_state: { version: 1, links: [] },
    tags: ['existing'],
    pinned: false,
    archived: false,
    revision: 1,
    created_at: 1_780_000_000,
    updated_at: 1_780_000_000,
    ...overrides,
  }
}

test.beforeEach(async ({ page }) => {
  await installAppShellMocks(page, {
    projects: [{ id: 'project-reader-note', name: 'Reader notes', created_at: 1, updated_at: 1 }],
  })
})

test('selected source text saves to a new note with comment and exact locator', async ({ page }) => {
  let createdBody: Record<string, unknown> | null = null
  const renderLoopErrors: string[] = []
  page.on('console', (entry) => {
    if (entry.type() === 'error' && entry.text().includes('Maximum update depth exceeded')) {
      renderLoopErrors.push(entry.text())
    }
  })
  await page.route('**/api/chat/research-notes**', async (route) => {
    const request = route.request()
    if (request.method() === 'GET') {
      await fulfillJson(route, [])
      return
    }
    createdBody = request.postDataJSON() as Record<string, unknown>
    await fulfillJson(route, noteRecord({ id: 'note-created', ...createdBody }))
  })

  await page.goto('/__reader_test__?scenario=strict-quote')
  await selectText(page, SOURCE_SENTENCE)
  await page.getByTestId('reader-selection-note').click()
  const modal = page.getByTestId('reader-note-modal')
  await expect(modal).toBeVisible()
  await modal.getByTestId('reader-note-comment').fill('This sentence defines the reconstruction premise.')
  await page.getByTestId('reader-note-save').click()
  await expect(modal).not.toBeVisible()
  await expect.poll(() => createdBody).not.toBeNull()
  const sourceState = createdBody?.source_state as { links?: Array<Record<string, unknown>> }
  const link = sourceState.links?.[0]
  expect(String(createdBody?.content_markdown || '')).toContain('This sentence defines the reconstruction premise.')
  expect(link?.block_id).toBe('p-intro-1')
  expect(link?.anchor_id).toBe('a-p-intro-1')
  expect(Number(link?.start_offset)).toBeGreaterThanOrEqual(0)
  expect(Number(link?.end_offset)).toBeGreaterThan(Number(link?.start_offset))
  expect(String(link?.capture_id || '')).toMatch(/^reader:/)
  expect(renderLoopErrors).toEqual([])
})

test('existing note blocks duplicate capture and failed save keeps the full draft', async ({ page }) => {
  let current = noteRecord()
  let patchCount = 0
  let failPatch = false
  await page.route('**/api/chat/research-notes**', async (route) => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    if (request.method() === 'GET' && path === '/api/chat/research-notes') {
      await fulfillJson(route, [{ ...current, content_markdown: '' }])
      return
    }
    if (request.method() === 'GET') {
      await fulfillJson(route, current)
      return
    }
    if (request.method() === 'PATCH') {
      patchCount += 1
      if (failPatch) {
        await fulfillJson(route, { detail: 'temporary failure' }, 500)
        return
      }
      const body = request.postDataJSON() as Record<string, unknown>
      current = noteRecord({ ...current, ...body, revision: Number(current.revision) + 1 })
      await fulfillJson(route, current)
      return
    }
    await fulfillJson(route, current)
  })

  await page.goto('/__reader_test__?scenario=strict-quote')
  await selectText(page, SOURCE_SENTENCE)
  await page.getByTestId('reader-selection-note').click()
  await page.getByTestId('reader-note-target').click()
  await page.getByText('Existing evidence note', { exact: true }).click()
  await page.getByTestId('reader-note-comment').fill('First annotation')
  await page.getByTestId('reader-note-save').click()
  await expect.poll(() => patchCount).toBe(1)

  await page.getByTestId('reader-selection-note').click()
  await page.getByTestId('reader-note-target').click()
  await page.getByText('Existing evidence note', { exact: true }).click()
  await page.getByTestId('reader-note-save').click()
  await expect(page.getByTestId('reader-note-modal')).toContainText(/已在该笔记|already in that note/)
  expect(patchCount).toBe(1)

  await page.getByRole('button', { name: /取\s*消|Cancel/ }).click()
  failPatch = true
  await selectText(page, 'Conventional high-speed imaging systems often face challenges')
  await page.getByTestId('reader-selection-note').click()
  await page.getByTestId('reader-note-target').click()
  await page.getByText('Existing evidence note', { exact: true }).click()
  await page.getByTestId('reader-note-comment').fill('Keep this draft after a network failure.')
  await page.getByTestId('reader-note-save').click()
  await expect(page.getByTestId('reader-note-modal')).toBeVisible()
  await expect(page.getByTestId('reader-note-comment')).toHaveValue('Keep this draft after a network failure.')
  await expect(page.getByTestId('reader-note-modal')).toContainText(/保存失败|Save failed|temporary failure/)
})

test('table action saves the original Markdown table instead of flattened text', async ({ page }) => {
  let createdBody: Record<string, unknown> | null = null
  await page.route('**/api/chat/research-notes**', async (route) => {
    if (route.request().method() === 'GET') {
      await fulfillJson(route, [])
      return
    }
    createdBody = route.request().postDataJSON() as Record<string, unknown>
    await fulfillJson(route, noteRecord({ id: 'note-table', ...createdBody }))
  })

  await page.goto('/__reader_test__?scenario=strict-quote')
  const table = page.locator('.kb-table-wrap').last()
  await table.scrollIntoViewIfNeeded()
  await table.hover()
  await table.getByTestId('reader-block-note').click()
  await expect(page.getByTestId('reader-note-modal')).toContainText(/表格摘录|Table excerpt/)
  await page.getByTestId('reader-note-save').click()
  await expect.poll(() => createdBody).not.toBeNull()
  expect(String(createdBody?.content_markdown || '')).toContain('| Metric | Value |')
  expect(String(createdBody?.content_markdown || '')).toContain('| PSNR | 32.4 dB |')
  const sourceState = createdBody?.source_state as { links?: Array<Record<string, unknown>> }
  expect(sourceState.links?.[0]?.capture_kind).toBe('table')
})
