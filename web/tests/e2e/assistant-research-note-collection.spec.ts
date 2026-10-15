import { expect, test, type Route } from '@playwright/test'

const INTERNAL_ROUTE_READY_TIMEOUT_MS = 15_000

function noteRecord(id: string, title: string, projectId: string, updatedAt: number) {
  return {
    id,
    project_id: projectId,
    source_conv_id: null,
    title,
    content_markdown: '',
    source_state: { version: 1, links: [] },
    tags: ['fixture'],
    pinned: false,
    archived: false,
    revision: 1,
    created_at: updatedAt - 10,
    updated_at: updatedAt,
  }
}

async function fulfillJson(route: Route, body: unknown) {
  await route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify(body),
  })
}

test('research-note collection rejects stale scope loads and refreshes and deletes without stale cards', async ({ page }) => {
  let delayNextProjectAGet = false
  let delayedProjectAStarted = false
  let delayedProjectAResolved = false
  let delayNextProjectBGet = false
  let delayedProjectBStarted = false
  let delayedProjectBResolved = false
  let deleteCount = 0
  let projectBNotes = [noteRecord('note-b', 'Project B evidence note', 'notes-project-b', 200)]

  await page.route('**/api/chat/research-notes**', async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    if (request.method() === 'DELETE') {
      deleteCount += 1
      projectBNotes = []
      await fulfillJson(route, { ok: true })
      return
    }
    if (request.method() !== 'GET' || url.pathname !== '/api/chat/research-notes') {
      await fulfillJson(route, {})
      return
    }
    if (url.searchParams.get('project_id') === 'notes-project-a') {
      if (delayNextProjectAGet) {
        delayNextProjectAGet = false
        delayedProjectAStarted = true
        await new Promise((resolve) => setTimeout(resolve, 450))
        delayedProjectAResolved = true
      }
      await fulfillJson(route, [noteRecord('note-a', 'Stale project A note', 'notes-project-a', 100)])
      return
    }
    if (delayNextProjectBGet) {
      delayNextProjectBGet = false
      delayedProjectBStarted = true
      const staleSnapshot = [...projectBNotes]
      await new Promise((resolve) => setTimeout(resolve, 450))
      delayedProjectBResolved = true
      await fulfillJson(route, staleSnapshot)
      return
    }
    await fulfillJson(route, projectBNotes)
  })

  await page.goto(
    '/__message_list_test__?notes=1&project=notes-project-a&switchProject=notes-project-b',
  )
  await expect(page.getByTestId('message-list-test-scenario')).toBeVisible({
    timeout: INTERNAL_ROUTE_READY_TIMEOUT_MS,
  })
  const panel = page.getByTestId('research-notes-panel')
  await expect(panel).toContainText('Stale project A note')
  delayNextProjectAGet = true
  await page.evaluate(() => {
    window.dispatchEvent(new CustomEvent('kb:research-notes-changed'))
  })
  await expect.poll(() => delayedProjectAStarted).toBe(true)
  await page.getByTestId('message-list-switch-project-scope').click()

  await expect(panel).not.toContainText('Stale project A note')
  await expect(panel).toContainText('Project B evidence note')
  await expect.poll(() => delayedProjectAResolved).toBe(true)
  await expect(panel).not.toContainText('Stale project A note')
  await expect(page.getByTestId('message-list-research-notes-count')).toHaveText('1')

  projectBNotes = [noteRecord('note-b', 'Refreshed project B evidence note', 'notes-project-b', 300)]
  await page.evaluate(() => {
    window.dispatchEvent(new CustomEvent('kb:research-notes-changed'))
  })
  await expect(panel).toContainText('Refreshed project B evidence note')

  delayNextProjectBGet = true
  await page.evaluate(() => {
    window.dispatchEvent(new CustomEvent('kb:research-notes-changed'))
  })
  await expect.poll(() => delayedProjectBStarted).toBe(true)
  const noteCard = page.getByTestId('research-note-card-note-b')
  await noteCard.getByRole('button', { name: /Delete|删除/ }).click()
  await page.getByRole('tooltip').getByRole('button', { name: /Delete|删\s*除/ }).click()
  await expect(noteCard).toHaveCount(0)
  await expect(page.getByTestId('message-list-research-notes-count')).toHaveText('0')
  await expect.poll(() => delayedProjectBResolved).toBe(true)
  await expect(noteCard).toHaveCount(0)
  await expect(page.getByTestId('message-list-research-notes-count')).toHaveText('0')
  expect(deleteCount).toBe(1)
})
