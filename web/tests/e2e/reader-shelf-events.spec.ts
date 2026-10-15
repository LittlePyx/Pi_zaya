import { expect, test, type Page } from '@playwright/test'
import { installAppShellMocks, installIdleReferenceMocks } from './mockAppShell'

async function broadcastBatch(page: Page, type: string, payloads: Record<string, unknown>[]) {
  return page.evaluate(async ({ type, payloads }) => {
    const channel = new BroadcastChannel(`kb:${type}`)
    const acknowledgements: string[] = []
    return new Promise<string[]>((resolve, reject) => {
      const timeout = window.setTimeout(() => {
        channel.close()
        reject(new Error('Accepted shelf event was not acknowledged'))
      }, 5000)
      channel.onmessage = (event) => {
        acknowledgements.push(String(event.data.requestId))
        if (event.data.requestId === String(payloads.length - 1)) {
          window.clearTimeout(timeout)
          channel.close()
          resolve(acknowledgements)
        }
      }
      payloads.forEach((payload, index) => channel.postMessage({ ...payload, type, requestId: String(index) }))
    })
  }, { type, payloads })
}

test('reader shelf subscriptions use committed scope, acknowledge acceptance, and clean up both transports', async ({ page }) => {
  await page.route('**/__reader_shelf_hook_test__', (route) => route.fulfill({
    contentType: 'text/html', body: '<!doctype html><title>Reader shelf hook regression</title>',
  }))
  await page.goto('/__reader_shelf_hook_test__')
  const result = await page.evaluate(async () => {
    const { runReaderShelfEventsSmoke } = await import('/src/testing/readerShelfEventsSmoke.ts')
    return runReaderShelfEventsSmoke()
  })
  expect(result.calls).toEqual([
    'first:citation:source-citation', 'first:selection:source quote',
    'latest:citation:source-citation', 'window-only:selection:source quote',
  ])
  expect(result.channels).toEqual([
    { name: 'kb:reader-citation-shelf', sent: [{ type: 'reader-citation-shelf-ack', requestId: 'accepted-b' }], closed: 1, detached: true },
    { name: 'kb:reader-selection-shelf', sent: [{ type: 'reader-selection-shelf-ack', requestId: 'accepted-a' }], closed: 1, detached: true },
  ])
})

test('reader evidence reaches the matching project basket through window and real broadcast events', async ({ page }) => {
  await installAppShellMocks(page)
  await installIdleReferenceMocks(page)
  const loadedScopes = new Set<string>()
  const shelves = new Map<string, { items: Record<string, unknown>[]; open: boolean; revision: number }>()
  await page.route('**/api/chat/citation-shelf**', async (route) => {
    const request = route.request()
    const body = request.method() === 'PATCH' ? request.postDataJSON() : {}
    const projectId = new URL(request.url()).searchParams.get('project_id') || body.project_id || 'a'
    const state = shelves.get(projectId) || { items: [], open: false, revision: 0 }
    if (request.method() === 'PATCH') {
      state.items = body.items || []
      state.open = Boolean(body.open)
      state.revision += 1
    } else {
      loadedScopes.add(projectId)
    }
    shelves.set(projectId, state)
    await route.fulfill({ contentType: 'application/json', body: JSON.stringify({
      version: 1, scope: 'project', scope_id: projectId, project_id: projectId, ...state,
    }) })
  })
  await page.goto('/__message_list_test__?project=a&switchProject=b')
  await expect(page.getByTestId('message-list-current-project')).toHaveText('a', { timeout: 15_000 })
  await expect.poll(() => loadedScopes.has('a')).toBe(true)
  const citation = { detail: {
    anchor: 'bridge-citation', title: 'Bridge citation paper', sourceName: 'Bridge citation paper',
    sourcePath: '/tmp/bridge-citation.md', raw: 'Exact citation evidence',
  } }
  const selection = {
    text: '| Evidence | Value |\n| --- | --- |\n| Recall | 0.95 |', sourcePath: '/tmp/bridge-table.md',
    sourceName: 'Bridge table paper', headingPath: 'Results', blockId: 'table-block',
    anchorId: 'table-anchor', anchorKind: 'table', conversationId: 'reader-origin',
  }
  await page.evaluate((detail) => {
    window.dispatchEvent(new CustomEvent('kb:reader-selection-shelf', { detail }))
  }, { ...selection, projectId: 'a' })
  expect(await broadcastBatch(page, 'reader-citation-shelf', [
    { ...citation, projectId: 'other' }, { ...citation, projectId: 'a' },
  ])).toEqual(['1'])
  await expect(page.getByTestId('citation-shelf-item')).toHaveCount(2)
  await expect(page.getByTestId('citation-shelf')).toContainText('Bridge table paper')

  await page.evaluate((detail) => {
    // Switch in the same task, before the debounced backend save can run.
    window.dispatchEvent(new CustomEvent('kb:reader-selection-shelf', { detail }))
    document.querySelector<HTMLButtonElement>('[data-testid="message-list-switch-project-scope"]')?.click()
  }, { ...selection, projectId: 'a', text: 'Final project A evidence before immediate switch' })
  await expect(page.getByTestId('message-list-current-project')).toHaveText('b')
  await expect.poll(() => loadedScopes.has('b')).toBe(true)
  await expect(page.getByTestId('citation-shelf-item')).toHaveCount(0)
  await page.evaluate((detail) => {
    window.dispatchEvent(new CustomEvent('kb:reader-selection-shelf', { detail }))
  }, { ...selection, projectId: 'a' })
  expect(await broadcastBatch(page, 'reader-citation-shelf', [
    { ...citation, projectId: 'a' }, { ...citation, projectId: 'b' },
  ])).toEqual(['1'])
  await expect(page.getByTestId('citation-shelf-item')).toHaveCount(1)
  expect(await broadcastBatch(page, 'reader-selection-shelf', [
    { ...selection, projectId: 'a' }, { text: 'invalid', projectId: 'b' }, { ...selection, projectId: 'b' },
  ])).toEqual(['2'])
  await expect(page.getByTestId('citation-shelf-item')).toHaveCount(2)
  const stored = await page.evaluate(() => JSON.parse(localStorage.getItem('kb_cite_shelf:project:b') || '{}'))
  // Basket excerpts use compact display text; original Markdown is owned by note capture.
  expect(stored.items).toEqual(expect.arrayContaining([expect.objectContaining({
    sourcePath: selection.sourcePath, shelfExcerpt: 'Evidence Value Recall 0.95', blockId: 'table-block',
    anchorId: 'table-anchor', anchorKind: 'table', traceConvId: 'reader-origin',
  })]))
  await expect.poll(() => shelves.get('b')?.items.length).toBe(2)
  expect(shelves.get('a')?.items).toHaveLength(2)
  expect(shelves.get('a')?.items).toEqual(expect.arrayContaining([expect.objectContaining({
    note: expect.stringContaining('Final project A evidence before immediate switch'),
  })]))
})
