import { expect, test, type Route } from '@playwright/test'
import { downloadFilename, researchNoteDownloadFilename } from '../../src/api/chat'
import { installAppShellMocks } from './mockAppShell'

const PROJECT_ID = 'project-notes'
const CONVERSATION_ID = 'conversation-notes'
const SOURCE_PATH = 'db/paper/example.md'

async function fulfillJson(route: Route, body: unknown, status = 200) {
  await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
}

test('research note export prefers the UTF-8 title filename', () => {
  expect(downloadFilename(
    "attachment; filename=\"research-note.docx\"; filename*=UTF-8''%E5%8D%95%E5%85%89%E5%AD%90%20%E7%AC%94%E8%AE%B0.docx",
    'fallback.docx',
  )).toBe('单光子 笔记.docx')
  expect(researchNoteDownloadFilename('  单光子：结果 / 对比？  ')).toBe('单光子：结果 - 对比？.docx')
})

test('research notes workspace edits, combines, filters, and follows traceable sources', async ({ page }) => {
  let readerPayload: Record<string, unknown> | null = null
  let wordExportBody: Record<string, unknown> | null = null
  let records = [
    {
      id: 'note-a',
      project_id: PROJECT_ID,
      source_conv_id: CONVERSATION_ID,
      title: 'SPAD noise conclusion',
      content_markdown: '## Finding\n\nShot noise follows photon arrival statistics.',
      source_state: {
        version: 1,
        links: [
          {
            kind: 'answer',
            label: 'Why model SPAD noise?',
            conversation_id: CONVERSATION_ID,
            message_id: 22,
          },
          {
            kind: 'source',
            label: 'Noise modeling of SPAD arrays',
            source_name: 'Example SPAD paper',
            source_path: SOURCE_PATH,
            heading_path: 'Methods / Noise modeling',
            evidence_quote: 'Photon arrival is a stochastic process.',
            block_id: 'block-noise',
            page_start: 3,
          },
        ],
      },
      tags: ['SPAD', 'noise'],
      pinned: false,
      archived: false,
      revision: 1,
      created_at: 1_780_000_000,
      updated_at: 1_780_000_100,
    },
    {
      id: 'note-b',
      project_id: null,
      source_conv_id: null,
      title: 'Detector trade-off',
      content_markdown: '## Finding\n\nDynamic range constrains the detector.',
      source_state: { version: 1, links: [] },
      tags: ['detector'],
      pinned: false,
      archived: false,
      revision: 1,
      created_at: 1_780_000_010,
      updated_at: 1_780_000_090,
    },
  ]

  await installAppShellMocks(page, {
    projects: [{ id: PROJECT_ID, name: 'Single-photon imaging', created_at: 1, updated_at: 1 }],
  })

  await page.route('**/api/chat/research-notes**', async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname
    const noteId = path.split('/').at(-1) || ''

    if (request.method() === 'GET' && path === '/api/chat/research-notes') {
      const query = String(url.searchParams.get('query') || '').toLowerCase()
      const archived = url.searchParams.get('archived') === 'archived'
      const listed = records
        .filter(record => record.archived === archived)
        .filter(record => !query || `${record.title} ${record.content_markdown} ${record.tags.join(' ')}`.toLowerCase().includes(query))
        .map(record => ({ ...record, content_markdown: '' }))
      await fulfillJson(route, listed)
      return
    }
    if (request.method() === 'GET') {
      const record = records.find(item => item.id === noteId)
      await fulfillJson(route, record || { detail: 'not found' }, record ? 200 : 404)
      return
    }
    if (request.method() === 'PATCH') {
      const index = records.findIndex(item => item.id === noteId)
      const body = request.postDataJSON() as Record<string, unknown>
      const current = records[index]
      records[index] = {
        ...current,
        ...body,
        project_id: Object.prototype.hasOwnProperty.call(body, 'project_id') ? body.project_id as string | null : current.project_id,
        tags: Array.isArray(body.tags) ? body.tags as string[] : current.tags,
        revision: current.revision + 1,
        updated_at: current.updated_at + 1,
      }
      await fulfillJson(route, records[index])
      return
    }
    if (request.method() === 'POST') {
      const body = request.postDataJSON() as Record<string, unknown>
      const record = {
        id: `note-${records.length + 1}`,
        project_id: body.project_id as string | null || null,
        source_conv_id: null,
        title: String(body.title || ''),
        content_markdown: String(body.content_markdown || ''),
        source_state: body.source_state || { version: 1, links: [] },
        tags: [],
        pinned: false,
        archived: false,
        revision: 1,
        created_at: 1_780_000_200,
        updated_at: 1_780_000_200,
      }
      records = [...records, record]
      await fulfillJson(route, record)
      return
    }
    await fulfillJson(route, { ok: true })
  })

  await page.route('**/api/reader/sessions', async (route) => {
    const body = route.request().postDataJSON() as { payload?: Record<string, unknown> }
    readerPayload = body.payload || null
    await fulfillJson(route, {
      id: 'reader-note-source',
      conversation_id: CONVERSATION_ID,
      message_id: 22,
      payload: body.payload,
      state: {},
    })
  })

  await page.route('**/api/chat/research-note/export', async (route) => {
    wordExportBody = route.request().postDataJSON() as Record<string, unknown>
    await route.fulfill({
      status: 200,
      contentType: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      headers: {
        'content-disposition': "attachment; filename=\"SPAD noise evidence note.docx\"; filename*=UTF-8''SPAD%20noise%20evidence%20note.docx",
      },
      body: 'PK\u0003\u0004research-note-docx-fixture',
    })
  })

  await page.goto('/notes')
  const workspace = page.getByTestId('research-notes-workspace')
  await expect(workspace).toBeVisible()
  await expect(page.getByRole('menuitem', { name: /研究笔记/ })).toHaveClass(/ant-menu-item-selected/)
  await expect(workspace.getByText('SPAD noise conclusion', { exact: true })).toBeVisible()
  await expect(workspace.locator('.kb-notes-source-card blockquote')).toContainText('Photon arrival is a stochastic process.')

  const titleInput = workspace.locator('.kb-notes-title-input')
  await titleInput.fill('SPAD noise evidence note')
  await workspace.locator('.kb-notes-editor-body textarea').fill('## Finding\n\nEdited, traceable conclusion.')
  await expect.poll(() => records[0]?.title).toBe('SPAD noise evidence note')
  await expect.poll(() => records[0]?.content_markdown).toContain('Edited, traceable conclusion')

  const markdownDownloadPromise = page.waitForEvent('download')
  await workspace.getByRole('button', { name: '导出 Markdown' }).click()
  const markdownDownload = await markdownDownloadPromise
  expect(markdownDownload.suggestedFilename()).toBe('SPAD noise evidence note.md')

  const wordDownloadPromise = page.waitForEvent('download')
  await workspace.getByRole('button', { name: '导出 Word' }).click()
  const wordDownload = await wordDownloadPromise
  expect(wordDownload.suggestedFilename()).toBe('SPAD noise evidence note.docx')
  await expect.poll(() => wordExportBody?.title).toBe('SPAD noise evidence note')
  expect(String(wordExportBody?.content_markdown || '')).toContain('Edited, traceable conclusion')

  await workspace.getByRole('button', { name: 'pushpin', exact: true }).click()
  await expect.poll(() => records[0]?.pinned).toBe(true)

  const checkboxes = workspace.getByRole('checkbox', { name: '选择用于组合' })
  await checkboxes.nth(0).check()
  await checkboxes.nth(1).check()
  await workspace.getByRole('button', { name: '组合写作提纲' }).click()
  const outline = page.getByRole('dialog', { name: '组合写作提纲' })
  await expect(outline.getByRole('textbox').nth(1)).toHaveValue(/SPAD noise evidence note/)
  await expect(outline.getByRole('textbox').nth(1)).toHaveValue(/Detector trade-off/)
  await outline.getByRole('button', { name: 'Close' }).click()

  await workspace.getByRole('button', { name: '打开原文' }).click()
  await expect.poll(() => readerPayload?.sourcePath).toBe(SOURCE_PATH)
  await expect(page).toHaveURL(/\/reader\/session\/reader-note-source/)

  await page.goto('/notes')
  await workspace.getByRole('button', { name: '回到回答' }).click()
  await expect(page).toHaveURL(new RegExp(`conversation=${CONVERSATION_ID}.*note_message=22`))
})
