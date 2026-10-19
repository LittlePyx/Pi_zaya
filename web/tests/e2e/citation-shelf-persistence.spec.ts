import { expect, test, type Page, type Route } from '@playwright/test'
import { installAppShellMocks, installIdleReferenceMocks } from './mockAppShell'

function evidence(key: string) {
  return { key, anchor: key, title: key, main: key, sourceName: key,
    sourcePath: `/tmp/${key}.md`, raw: `Evidence from ${key}`, shelfItemKind: 'reader_selection',
    shelfExcerpt: `Evidence from ${key}`, blockId: `${key}-block`, anchorId: `${key}-anchor` }
}

async function controlledShelves(page: Page, holdWrites = false) {
  await installAppShellMocks(page)
  await installIdleReferenceMocks(page)
  const reads: { project: string; route: Route }[] = []
  const writes: { project: string; items: Record<string, unknown>[]; allow_empty_overwrite?: boolean }[] = []
  const writeResponses: ((override?: Record<string, unknown>) => Promise<void>)[] = []
  const controls = { failWrites: 0, storageOffline: false }
  let revision = 10
  const handle = async (route: Route) => {
    const request = route.request()
    const project = new URL(request.url()).searchParams.get('project_id') || 'a'
    if (request.method() === 'GET') {
      reads.push({ project, route })
      return
    }
    const body = request.method() === 'PATCH' ? request.postDataJSON() : { items: [], open: false }
    writes.push({ project, ...body })
    if (controls.failWrites > 0 || controls.storageOffline) {
      if (controls.failWrites > 0) controls.failWrites -= 1
      await route.fulfill({ status: 503, json: { detail: 'Temporary storage failure' } })
      return
    }
    const record = { ...body, scope: 'project', scope_id: project, project_id: project, revision: ++revision }
    const respond = (override?: Record<string, unknown>) => route.fulfill({ json: { ...record, ...override } })
    writeResponses.push(respond)
    if (!holdWrites) await respond()
  }
  const attach = async (target: Page) => {
    await installAppShellMocks(target)
    await installIdleReferenceMocks(target)
    await target.route('**/api/chat/citation-shelf**', handle)
  }
  await page.route('**/api/chat/citation-shelf**', handle)
  const release = async (index: number, items: Record<string, unknown>[]) => {
    await expect.poll(() => reads.length).toBeGreaterThan(index)
    const { route, project } = reads[index]
    const page = route.request().frame().page()
    const response = page.waitForResponse((response) => response.request() === route.request())
    await route.fulfill({ json: { scope: 'project', scope_id: project, project_id: project,
      revision: 10, updated_at: 1, open: true, items } })
    await response
    // Let the response's React work commit before making negative assertions.
    await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))))
  }
  return { reads, writes, writeResponses, release, controls, attach }
}

async function openShelf(page: Page, initialItems: Record<string, unknown>[] = []) {
  await page.addInitScript((items) => {
    if (localStorage.getItem('kb_cite_shelf:project:a')) return
    localStorage.setItem('kb_cite_shelf:project:a', JSON.stringify({
      version: 4, revision: 2, updatedAt: Date.now(), open: true, items,
    }))
  }, initialItems)
  await page.goto('/__message_list_test__?project=a&switchProject=b')
  await expect(page.getByTestId('message-list-current-project')).toHaveText('a', { timeout: 15_000 })
}

async function addSelection(page: Page, project: string, text: string, actionSelector?: string) {
  await page.evaluate(({ project, text, actionSelector }) => {
    window.dispatchEvent(new CustomEvent('kb:reader-selection-shelf', { detail: {
      projectId: project, text, sourcePath: `/tmp/${project}-new.md`, sourceName: `${project} new paper`,
      blockId: `${project}-new-block`, anchorId: `${project}-new-anchor`,
    } }))
    if (actionSelector === 'pagehide') window.dispatchEvent(new PageTransitionEvent('pagehide'))
    else if (actionSelector) document.querySelector<HTMLButtonElement>(actionSelector)?.click()
  }, { project, text, actionSelector })
}

test('A to B to A ignores both old hydration responses and keeps local additions while current hydration waits', async ({ page }) => {
  const backend = await controlledShelves(page)
  await openShelf(page)
  await expect.poll(() => backend.reads.length).toBe(1)
  await addSelection(page, 'a', 'A accepted before switching', '[data-testid="message-list-switch-project-scope"]')
  await expect(page.getByTestId('message-list-current-project')).toHaveText('b')
  await expect.poll(() => backend.reads.length).toBe(2)
  await addSelection(page, 'b', 'B accepted before switching back', '[data-testid="message-list-switch-project-scope"]')
  await expect(page.getByTestId('message-list-current-project')).toHaveText('a')
  await expect.poll(() => backend.reads.length).toBe(3)
  await backend.release(0, [evidence('obsolete-a')])
  await backend.release(1, [evidence('obsolete-b')])
  await expect(page.getByTestId('citation-shelf-item')).toHaveCount(1)
  expect(backend.writes).toEqual([])
  await addSelection(page, 'a', 'A newest evidence while hydration waits')
  await backend.release(2, [evidence('backend-a')])
  await expect(page.getByTestId('citation-shelf-item')).toHaveCount(2)
  await expect(page.getByTestId('citation-shelf')).not.toContainText('obsolete')
  await expect.poll(() => backend.writes.some((write) => write.project === 'a' && write.items.some(
    (item) => String(item.note).includes('A newest evidence'),
  ))).toBe(true)
  const stored = await page.evaluate(() => JSON.parse(localStorage.getItem('kb_cite_shelf:project:b') || '{}'))
  expect(stored.items).toEqual([expect.objectContaining({ note: expect.stringContaining('B accepted') })])
})

for (const action of ['remove', 'clear'] as const) {
  test(`delayed hydration does not resurrect evidence after ${action}`, async ({ page }) => {
    const backend = await controlledShelves(page)
    await openShelf(page, [evidence('removed-paper')])
    await expect.poll(() => backend.reads.length).toBe(1)
    await expect(page.getByTestId('citation-shelf-item')).toHaveCount(1)
    await page.locator(action === 'remove' ? '.kb-shelf-item-remove' : '[data-testid="citation-shelf-clear"]').click()
    await expect(page.getByTestId('citation-shelf-item')).toHaveCount(0)
    await backend.release(0, [evidence('removed-paper')])
    await expect(page.getByTestId('citation-shelf-item')).toHaveCount(0)
    await expect.poll(() => backend.writes.some((write) => write.items.length === 0 && write.allow_empty_overwrite)).toBe(true)
    expect(backend.writes.every((write) => write.items.length === 0)).toBe(true)
  })
}

test('pagehide and component unmount flush the most recent accepted evidence before debounce', async ({ page }) => {
  const backend = await controlledShelves(page)
  await openShelf(page)
  await backend.release(0, [])
  await addSelection(page, 'a', 'pagehide evidence', 'pagehide')
  await expect.poll(() => backend.writes.some((write) => write.items.some((item) => String(item.note).includes('pagehide evidence')))).toBe(true)
  await addSelection(page, 'a', 'unmount evidence', '[data-testid="message-list-toggle-mount"]')
  await expect(page.getByTestId('citation-shelf')).toHaveCount(0)
  await expect.poll(() => backend.writes.some((write) => write.items.some((item) => String(item.note).includes('unmount evidence')))).toBe(true)
  const stored = await page.evaluate(() => JSON.parse(localStorage.getItem('kb_cite_shelf:project:a') || '{}'))
  expect(stored.items).toEqual([expect.objectContaining({ note: expect.stringContaining('unmount evidence') })])
  const writeCount = backend.writes.length
  await addSelection(page, 'a', 'ignored after unmount', 'pagehide')
  expect(backend.writes).toHaveLength(writeCount)
})

test('a late save acknowledgement cannot authorize writes before a returning project finishes hydration', async ({ page }) => {
  const backend = await controlledShelves(page, true)
  await openShelf(page)
  await backend.release(0, [])
  await addSelection(page, 'a', 'first A evidence', 'pagehide')
  await expect.poll(() => backend.writeResponses.length).toBeGreaterThan(0)
  await page.getByTestId('message-list-switch-project-scope').click()
  await expect.poll(() => backend.reads.length).toBe(2)
  await page.getByTestId('message-list-switch-project-scope').click()
  await expect.poll(() => backend.reads.length).toBe(3)
  const before = backend.writes.length
  await backend.writeResponses[0]()
  await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))))
  await addSelection(page, 'a', 'latest A evidence', 'pagehide')
  // The synchronous flush must still be refused until this visit's GET finishes.
  await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))))
  expect(backend.writes).toHaveLength(before)
  await backend.release(2, [evidence('backend-only-a')])
  await expect.poll(() => backend.writes.some((write) => write.items.length === 2 && write.items.some(
    (item) => String(item.note).includes('latest A evidence'),
  ))).toBe(true)
  await expect(page.getByTestId('citation-shelf-item')).toHaveCount(2)
  await backend.release(1, [evidence('obsolete-b')])
  await expect(page.getByTestId('citation-shelf')).not.toContainText('obsolete-b')
})

test('typed shelf contract keeps unknown backend evidence, explicit clear intent, and save ownership distinct', async ({ page }) => {
  await page.route('**/__shelf_contract_test__', (route) => route.fulfill({ contentType: 'text/html', body: '<!doctype html><title>Shelf contract</title>' }))
  await page.goto('/__shelf_contract_test__')
  const result = await page.evaluate(async (fixtures) => {
    const { buildShelfPendingSave, reconcileShelfHydration, shelfHydrationIsCurrent } = await import('/src/components/chat/citeShelfPersistence.ts')
    const { restoreShelfItems } = await import('/src/components/chat/citeShelfStorage.ts')
    const items = restoreShelfItems(fixtures)
    const initial = { projectId: 'a', convId: 'conversation-a', open: true, items: items.slice(0, 2) }
    const request = { storageKey: 'kb_cite_shelf:project:a', sequence: 3, snapshot: initial, clearRevision: 0 }
    const record = { version: 1, scope: 'project', scope_id: 'a', revision: 5, open: true,
      created_at: 1, updated_at: 1, items: fixtures.slice(0, 3) }
    const reconcile = (currentItems: typeof items, clearRevision = 0) => reconcileShelfHydration({
      request, current: { ...initial, open: false, items: currentItems }, record, localUpdatedAt: 0, clearRevision,
    })
    const partial = reconcile([items[1]])
    const clearThenAdd = reconcile([items[3]], 1)
    const empty = { ...initial, items: [] }
    const explicitEmpty = reconcileShelfHydration({ request, current: empty,
      record: { ...record, items: fixtures.slice(0, 2) }, localUpdatedAt: 0, clearRevision: 0 })
    const untouchedEmpty = reconcileShelfHydration({ request: { ...request, snapshot: empty }, current: empty,
      record, localUpdatedAt: Date.now() + 60000, clearRevision: 0 })
    return {
      partial: partial.snapshot.items.map((item) => item.key), partialOpen: partial.snapshot.open,
      clearThenAdd: clearThenAdd.snapshot.items.map((item) => item.key),
      explicitEmpty: [explicitEmpty.snapshot.items.length, explicitEmpty.shouldSaveBackend, explicitEmpty.allowEmptyOverwrite],
      untouchedEmpty: untouchedEmpty.snapshot.items.map((item) => item.key),
      ownership: [shelfHydrationIsCurrent(request, 3, initial), shelfHydrationIsCurrent(request, 4, initial),
        shelfHydrationIsCurrent(request, 3, { ...initial, projectId: 'b' })],
      blocked: [buildShelfPendingSave(initial, { hydrated: false, backendRevision: 5, allowEmptyOverwrite: false }),
        buildShelfPendingSave(empty, { hydrated: true, backendRevision: 5, allowEmptyOverwrite: false })],
      explicitSave: buildShelfPendingSave(empty, { hydrated: true, backendRevision: 5, allowEmptyOverwrite: true }),
      unchangedRequest: request.snapshot.items.map((item) => item.key),
    }
  }, ['deleted', 'retained', 'backend-only', 'added-after-clear'].map(evidence))
  expect(result.partial).toEqual(['retained', 'backend-only'])
  expect(result.partialOpen).toBe(false)
  expect(result.clearThenAdd).toEqual(['added-after-clear'])
  expect(result.explicitEmpty).toEqual([0, true, true])
  expect(result.untouchedEmpty).toEqual(['deleted', 'retained', 'backend-only'])
  expect(result.ownership).toEqual([true, false, false])
  expect(result.blocked).toEqual([null, null])
  expect(result.explicitSave).toEqual({ storageKey: 'kb_cite_shelf:project:a', body: {
    convId: 'conversation-a', projectId: 'a', scope: 'project', open: true, items: [], allowEmptyOverwrite: true, expectedRevision: 5,
  } })
  expect(result.unchangedRequest).toEqual(['deleted', 'retained'])
})

test('removal before hydration survives a full page reload and keeps unseen backend evidence', async ({ page }) => {
  const backend = await controlledShelves(page)
  await openShelf(page, [evidence('removed-before-reload')])
  await expect.poll(() => backend.reads.length).toBe(1)
  await page.locator('.kb-shelf-item-remove').click()
  await expect(page.getByTestId('citation-shelf-item')).toHaveCount(0)
  await page.reload()
  await expect.poll(() => backend.reads.length).toBe(2)
  await backend.release(1, [evidence('removed-before-reload'), evidence('unseen-backend-paper')])
  await expect(page.getByTestId('citation-shelf-item')).toHaveCount(1)
  await expect(page.getByTestId('citation-shelf-item')).toContainText('unseen-backend-paper')
  await expect.poll(() => backend.writes.some((write) => write.items.length === 1 && write.items[0].key === 'unseen-backend-paper')).toBe(true)
})

test('failed save retries automatically against fresh backend evidence without another user edit', async ({ page }) => {
  const backend = await controlledShelves(page)
  await openShelf(page)
  await backend.release(0, [])
  backend.controls.failWrites = 1
  await addSelection(page, 'a', 'recover this accepted evidence', 'pagehide')
  await expect.poll(() => backend.controls.failWrites).toBe(0)
  await expect.poll(() => backend.reads.length).toBe(2)
  await backend.release(1, [evidence('arrived-during-outage')])
  await expect.poll(() => backend.writes.some((write) => write.items.length === 2 && write.items.some(
    (item) => String(item.note).includes('recover this accepted evidence'),
  ))).toBe(true)
  await expect.poll(() => page.evaluate(() => localStorage.getItem('kb_cite_shelf:project:a:pending_mutations:v1') || '')).toBe('')
})

test('clear and subsequent addition queue behind an in-flight save and only its own acknowledgement clears intent', async ({ page }) => {
  const backend = await controlledShelves(page, true)
  await openShelf(page)
  await backend.release(0, [])
  await addSelection(page, 'a', 'old evidence before clear', 'pagehide')
  await expect.poll(() => backend.writeResponses.length).toBe(1)
  await page.getByTestId('citation-shelf-clear').click()
  await addSelection(page, 'a', 'new evidence after clear', 'pagehide')
  expect(backend.writes).toHaveLength(1)
  await backend.writeResponses[0]()
  await expect.poll(() => backend.writeResponses.length).toBe(2)
  expect(backend.writes[1].items).toEqual([expect.objectContaining({ note: expect.stringContaining('new evidence after clear') })])
  expect(JSON.stringify(backend.writes[1].items)).not.toContain('old evidence before clear')
  expect(await page.evaluate(() => localStorage.getItem('kb_cite_shelf:project:a:pending_mutations:v1'))).toBeTruthy()
  await backend.writeResponses[1]()
  await expect.poll(() => page.evaluate(() => localStorage.getItem('kb_cite_shelf:project:a:pending_mutations:v1') || '')).toBe('')
})

test('initial hydration failure retries while retaining a deletion made during the outage', async ({ page }) => {
  const backend = await controlledShelves(page)
  await openShelf(page, [evidence('removed-during-load-failure')])
  await expect.poll(() => backend.reads.length).toBe(1)
  await backend.reads[0].route.fulfill({ status: 503, json: { detail: 'Temporary read failure' } })
  await page.locator('.kb-shelf-item-remove').click()
  await expect(page.getByTestId('citation-shelf-item')).toHaveCount(0)
  await expect.poll(() => backend.reads.length).toBe(2)
  await backend.release(1, [evidence('removed-during-load-failure'), evidence('unseen-during-load-failure')])
  await expect(page.getByTestId('citation-shelf-item')).toHaveCount(1)
  await expect(page.getByTestId('citation-shelf-item')).toContainText('unseen-during-load-failure')
  await expect.poll(() => backend.writes.some((write) => write.items.length === 1 && write.items[0].key === 'unseen-during-load-failure')).toBe(true)
})

test('failed clear survives closing and reopening the page until its empty snapshot reaches storage', async ({ page, context }) => {
  const backend = await controlledShelves(page)
  await openShelf(page)
  await backend.release(0, [evidence('clear-across-reload')])
  await expect(page.getByTestId('citation-shelf-item')).toHaveCount(1)
  backend.controls.failWrites = 1
  backend.controls.storageOffline = true
  await page.getByTestId('citation-shelf-clear').click()
  await page.evaluate(() => window.dispatchEvent(new PageTransitionEvent('pagehide')))
  await expect.poll(() => backend.controls.failWrites).toBe(0)
  expect(await page.evaluate(() => localStorage.getItem('kb_cite_shelf:project:a:pending_mutations:v1'))).toBeTruthy()
  await page.close()
  backend.controls.storageOffline = false
  const oldReadCount = backend.reads.length
  const reopened = await context.newPage()
  await backend.attach(reopened)
  await reopened.goto('/__message_list_test__?project=a&switchProject=b')
  await expect.poll(() => backend.reads.length).toBe(oldReadCount + 1)
  await backend.release(oldReadCount, [evidence('clear-across-reload')])
  await expect(reopened.getByTestId('citation-shelf-item')).toHaveCount(0)
  await expect.poll(() => reopened.evaluate(() => localStorage.getItem('kb_cite_shelf:project:a:pending_mutations:v1') || '')).toBe('')
  expect(backend.writes.filter((write) => write.allow_empty_overwrite)).toEqual(expect.arrayContaining([
    expect.objectContaining({ project: 'a', items: [], allow_empty_overwrite: true }),
  ]))
})

test('journal replay preserves order, explicit blank annotations, scope, and newer unacknowledged changes', async ({ page }) => {
  await page.route('**/__shelf_journal_test__', (route) => route.fulfill({ contentType: 'text/html', body: '<!doctype html><title>Shelf journal</title>' }))
  await page.goto('/__shelf_journal_test__')
  const result = await page.evaluate(async (fixtures) => {
    const { restoreShelfItems } = await import('/src/components/chat/citeShelfStorage.ts')
    const { recordShelfMutation, readShelfMutationIntent, applyShelfMutationIntent, acknowledgeShelfMutation } = await import('/src/components/chat/citeShelfMutationJournal.ts')
    const items = restoreShelfItems(fixtures)
    const original = { projectId: 'a', open: true, items }
    const removed = { ...original, items: [items[1]] }
    recordShelfMutation(original, removed)
    const oldId = readShelfMutationIntent('a')!.id
    const edited = { ...original, items: [{ ...items[0], note: '', tags: [] }, items[1]] }
    recordShelfMutation(removed, edited)
    const intent = readShelfMutationIntent('a')!
    acknowledgeShelfMutation('a', oldId)
    const staleAckRetained = readShelfMutationIntent('a')?.id === intent.id
    const replay = applyShelfMutationIntent(original, readShelfMutationIntent('a'))
    const other = applyShelfMutationIntent({ ...original, projectId: 'b' }, intent)
    acknowledgeShelfMutation('a', intent.id)
    localStorage.setItem('kb_cite_shelf:project:b:pending_mutations:v1', JSON.stringify({ ...intent, projectId: 'a', clear: true }))
    return { order: replay.items.map((item) => item.key), note: replay.items[0].note, tags: replay.items[0].tags,
      staleAckRetained, cleared: readShelfMutationIntent('a') === null,
      otherUnchanged: other.items[0].note === items[0].note, malformedScopeIgnored: readShelfMutationIntent('b') === null }
  }, [{ ...evidence('first'), note: 'obsolete annotation', tags: ['old'] }, evidence('second')])
  expect(result).toEqual({ order: ['first', 'second'], note: '', tags: [], staleAckRetained: true,
    cleared: true, otherUnchanged: true, malformedScopeIgnored: true })
})

test('a successful HTTP status without the saved evidence retains intent and retries', async ({ page }) => {
  const backend = await controlledShelves(page, true)
  await openShelf(page)
  await backend.release(0, [])
  await addSelection(page, 'a', 'must actually be stored', 'pagehide')
  await expect.poll(() => backend.writeResponses.length).toBe(1)
  await backend.writeResponses[0]({ items: [] })
  await expect.poll(() => backend.reads.length).toBe(2)
  expect(await page.evaluate(() => localStorage.getItem('kb_cite_shelf:project:a:pending_mutations:v1'))).toBeTruthy()
  await backend.release(1, [])
  await expect.poll(() => backend.writeResponses.length).toBe(2)
  await backend.writeResponses[1]()
  await expect.poll(() => page.evaluate(() => localStorage.getItem('kb_cite_shelf:project:a:pending_mutations:v1') || '')).toBe('')
})

test('backend whitespace normalization acknowledges saved evidence without changing local note formatting', async ({ page }) => {
  const backend = await controlledShelves(page, true)
  await openShelf(page)
  await backend.release(0, [])
  await addSelection(page, 'a', 'multiline evidence\nwith another line', 'pagehide')
  await expect.poll(() => backend.writeResponses.length).toBe(1)
  const localNote = String(backend.writes[0].items[0].note)
  expect(localNote).toContain('\n')
  await backend.writeResponses[0]({ items: backend.writes[0].items.map((item) => ({ ...item,
    note: String(item.note).replace(/\s+/g, ' ').trim(),
  })) })
  await expect.poll(() => page.evaluate(() => localStorage.getItem('kb_cite_shelf:project:a:pending_mutations:v1') || '')).toBe('')
  const persisted = await page.evaluate(() => JSON.parse(localStorage.getItem('kb_cite_shelf:project:a') || '{}'))
  expect(persisted.items[0].note).toBe(localNote)
})

test('save retries have a finite backoff budget and resume on reconnection without losing the journal', async ({ page }) => {
  await page.clock.install({ time: new Date('2026-09-04T08:00:00Z') })
  await page.route('**/__shelf_retry_test__', (route) => route.fulfill({ contentType: 'text/html', body: '<!doctype html><title>Shelf retry budget</title>' }))
  await page.goto('/__shelf_retry_test__')
  await page.clock.pauseAt(new Date('2026-09-04T10:00:00Z'))
  await page.evaluate(async (fixture) => {
    const { chatApi } = await import('/src/api/chat.ts')
    const { restoreShelfItems } = await import('/src/components/chat/citeShelfStorage.ts')
    const { recordShelfMutation } = await import('/src/components/chat/citeShelfMutationJournal.ts')
    const { enqueueShelfSave } = await import('/src/components/chat/citeShelfSaveQueue.ts')
    const { buildShelfPendingSave } = await import('/src/components/chat/citeShelfPersistence.ts')
    const probe = { reads: 0, writes: 0, saved: 0, healthy: false }
    Object.assign(window, { shelfRetryProbe: probe })
    const empty = { version: 1, scope: 'project', scope_id: 'a', items: [], open: false, revision: 1, created_at: 0, updated_at: 0 }
    chatApi.getCitationShelf = async () => {
      probe.reads += 1
      if (!probe.healthy) throw new Error('offline')
      return empty
    }
    chatApi.saveCitationShelf = async (body) => {
      probe.writes += 1
      if (!probe.healthy) throw new Error('offline')
      return { ...empty, ...body, open: Boolean(body.open) }
    }
    const state = { projectId: 'a', open: true, items: restoreShelfItems([fixture]) }
    recordShelfMutation({ ...state, items: [], open: false }, state)
    enqueueShelfSave(buildShelfPendingSave(state, { hydrated: true, backendRevision: 1, allowEmptyOverwrite: false })!, () => { probe.saved += 1 })
  }, evidence('retry-budget-paper'))
  const readProbe = () => page.evaluate(() => (window as Window & { shelfRetryProbe?: { reads: number; writes: number; saved: number } }).shelfRetryProbe!)
  await expect.poll(async () => (await readProbe()).writes).toBe(1)
  for (const [index, delay] of [1000, 2000, 4000, 8000, 16000].entries()) {
    await page.clock.runFor(delay)
    await expect.poll(async () => (await readProbe()).reads).toBe(index + 1)
  }
  await page.clock.runFor(60_000)
  expect(await readProbe()).toMatchObject({ reads: 5, writes: 1, saved: 0 })
  expect(await page.evaluate(() => localStorage.getItem('kb_cite_shelf:project:a:pending_mutations:v1'))).toBeTruthy()
  await page.evaluate(() => {
    const probe = (window as Window & { shelfRetryProbe?: { healthy: boolean } }).shelfRetryProbe!
    probe.healthy = true
    window.dispatchEvent(new Event('online'))
  })
  await expect.poll(async () => (await readProbe()).saved).toBe(1)
  await page.clock.runFor(60_000)
  expect(await readProbe()).toMatchObject({ reads: 6, writes: 2, saved: 1 })
  expect(await page.evaluate(() => localStorage.getItem('kb_cite_shelf:project:a:pending_mutations:v1') || '')).toBe('')
})
