import { expect, test, type Page, type Route } from '@playwright/test'
import { spawn, type ChildProcess } from 'node:child_process'
import { installAppShellMocks, installIdleReferenceMocks } from './mockAppShell'

let process: ChildProcess
let backend: { url: string; project: string; conversation: string }
let serverErrors = ''
let conflicts = 0
test.beforeAll(async () => {
  process = spawn('python', ['../tools/citation_shelf_test_server.py'], { windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] })
  process.stderr!.on('data', (data) => { serverErrors += String(data) })
  backend = await new Promise((resolve, reject) => {
    let output = ''
    process.stdout!.on('data', (data) => {
      output += String(data)
      const match = output.match(/SHELF_TEST_READY (.+)\r?\n/)
      if (match) resolve(JSON.parse(match[1]))
    })
    process.on('error', reject)
    process.on('exit', (code) => reject(new Error(`Shelf test backend exited: ${code}`)))
  })
})
test.afterAll(async () => {
  if (!process || process.exitCode !== null) return
  const exited = new Promise<void>((resolve) => process.once('exit', () => resolve()))
  process.stdin!.end('\n')
  await exited
})

async function attach(page: Page, controls: { hold: boolean; pending: Route[]; drop: boolean }) {
  await installAppShellMocks(page)
  await installIdleReferenceMocks(page)
  await page.route('**/api/chat/citation-shelf**', async (route) => {
    if (route.request().method() === 'PATCH' && controls.hold) { controls.pending.push(route); return }
    await forward(route, controls)
  })
  const hydrated = page.waitForResponse((response) => response.request().method() === 'GET' && response.url().includes('/api/chat/citation-shelf'))
  await page.goto(`/__message_list_test__?project=${backend.project}`)
  await expect(page.getByTestId('message-list-current-project')).toHaveText(backend.project)
  expect((await hydrated).status()).toBe(200)
  await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))))
}

async function forward(route: Route, controls: { drop: boolean; dropClear?: boolean }) {
  const url = new URL(route.request().url())
  // The UI harness uses a synthetic conversation; bind it to the real fixture conversation.
  if (url.searchParams.has('conv_id')) url.searchParams.set('conv_id', backend.conversation)
  const response = await route.fetch({ url: `${backend.url}${url.pathname}${url.search}` })
  if (response.status() === 409) conflicts += 1
  if (route.request().method() === 'GET') expect(response.status(), `${await response.text()}\n${serverErrors}`).toBe(200)
  const body = route.request().method() === 'PATCH' ? route.request().postDataJSON() : null
  const dropEligible = !controls.dropClear || body?.mutations?.some((op: { clear?: boolean }) => op.clear)
  if (controls.drop && dropEligible && route.request().method() === 'PATCH' && response.ok()) {
    controls.drop = false
    await route.abort('failed')
  } else await route.fulfill({ response })
}

async function add(page: Page, text: string) {
  await page.evaluate(({ project, text }) => window.dispatchEvent(new CustomEvent('kb:reader-selection-shelf', {
    detail: { projectId: project, text, sourcePath: `/tmp/${text}.md`, sourceName: text,
      blockId: `${text}-block`, anchorId: `${text}-anchor` },
  })), { project: backend.project, text })
}

async function journalSettled(page: Page) {
  await expect.poll(() => page.evaluate(() => Object.keys(localStorage).filter((key) => key.includes(':operation:'))
    .every((key) => !localStorage.getItem(key)))).toBe(true)
}

test('independent browser windows preserve concurrent additions, replay a lost response once, and reload authoritative SQLite state', async ({ page, browser, request }) => {
  const controls = { hold: true, pending: [] as Route[], drop: false }
  // Independent caches force genuine competing writes, rather than a shared journal coalescing them.
  const second = await browser.newPage({ baseURL: String(test.info().project.use.baseURL) })
  await attach(page, controls)
  await attach(second, controls)
  // Both requests are deliberately held until both windows have authored changes.
  await add(page, 'first-window-evidence')
  await expect.poll(() => controls.pending.some((route) => (JSON.stringify(route.request().postDataJSON().mutations) || '').includes('first-window-evidence'))).toBe(true)
  await add(second, 'second-window-evidence')
  await expect.poll(() => controls.pending.some((route) => (JSON.stringify(route.request().postDataJSON().mutations) || '').includes('second-window-evidence'))).toBe(true)
  controls.hold = false
  for (const route of controls.pending.splice(0)) await forward(route, controls)
  const load = async () => (await request.get(`${backend.url}/api/chat/citation-shelf?project_id=${backend.project}`)).json()
  await expect.poll(async () => (await load()).items.map((item: Record<string, unknown>) => item.sourceName).sort())
    .toEqual(['first-window-evidence', 'second-window-evidence'])
  expect(conflicts).toBeGreaterThan(0)
  await journalSettled(page)
  await journalSettled(second)
  controls.drop = true
  await add(page, 'committed-with-lost-response')
  await expect.poll(async () => (await load()).items.length).toBe(3)
  await expect.poll(() => controls.drop).toBe(false)
  await journalSettled(page)
  const final = await load()
  await page.reload()
  await expect(page.getByTestId('citation-shelf-item')).toHaveCount(3)
  await expect.poll(async () => (await load()).revision).toBe(final.revision)
  await journalSettled(page)
  await journalSettled(second)
  // Finish forwarded requests before disposing their browser contexts.
  await page.unrouteAll({ behavior: 'wait' })
  await second.unrouteAll({ behavior: 'wait' })
  await second.close()
})

test('a clear with a lost response cannot erase a subsequent append when the page reopens', async ({ page, request }) => {
  const evidence = (key: string) => ({ key, anchor: key, main: key, title: key, raw: key,
    sourceName: key, sourcePath: `/tmp/${key}.md`, shelfItemKind: 'reader_selection', shelfExcerpt: key })
  const url = `${backend.url}/api/chat/citation-shelf?project_id=${backend.project}`
  const seed = await request.post(`${backend.url}/api/chat/citation-shelf/items?project_id=${backend.project}`, {
    data: { item: evidence('before-clear'), open: true },
  })
  expect(seed.ok()).toBe(true)
  const controls = { hold: false, pending: [] as Route[], drop: true, dropClear: true }
  await attach(page, controls)
  await expect(page.getByTestId('citation-shelf-item')).not.toHaveCount(0)
  await page.getByTestId('citation-shelf-clear').click()
  await expect.poll(() => controls.drop).toBe(false)
  // Add after the committed clear, before its retry; a duplicate clear must be inert.
  const appended = await request.post(`${backend.url}/api/chat/citation-shelf/items?project_id=${backend.project}`, {
    data: { item: evidence('after-clear'), open: true },
  })
  expect(appended.ok()).toBe(true)
  await page.reload()
  await expect(page.getByTestId('citation-shelf-item')).toHaveCount(1)
  await expect(page.getByTestId('citation-shelf-item')).toContainText('after-clear')
  expect((await (await request.get(url)).json()).items.map((item: Record<string, unknown>) => item.key)).toEqual(['after-clear'])
})

test('more than 120 retained offline operations drain in bounded API batches', async ({ page, request }) => {
  const controls = { hold: false, pending: [] as Route[], drop: false }
  await attach(page, controls)
  const result = await page.evaluate(async (projectId) => {
    const modulePath = '/src/components/chat/citeShelfSaveQueue.ts'
    const { persistShelfWrite } = await import(/* @vite-ignore */ modulePath)
    const mutations = Array.from({ length: 125 }, (_, index) => ({ id: `offline-${index}`, clear: false,
      upserts: [{ key: 'edited-paper', main: 'edited-paper', title: 'edited-paper', note: String(index) }], order: [] }))
    return persistShelfWrite({ projectId, items: [], expectedRevision: 0, mutationId: 'offline-124',
      mutations, allowEmptyOverwrite: true }, null)
  }, backend.project)
  expect(result.items).toEqual([expect.objectContaining({ key: 'edited-paper', note: '124' })])
  expect(result.acknowledged_mutation_id).toBe('offline-124')
  const stored = await (await request.get(`${backend.url}/api/chat/citation-shelf?project_id=${backend.project}`)).json()
  expect(stored.revision).toBe(2)
  expect(stored.items).toEqual(result.items)
})
