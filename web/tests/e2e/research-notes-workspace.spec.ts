import { expect, test, type Route } from '@playwright/test'
import {
  downloadFilename,
  researchNoteDownloadFilename,
  type ResearchNoteRecord,
} from '../../src/api/chat'
import {
  researchNoteOutlineHasMixedProjects,
  researchNoteOutlineProjectId,
  researchNoteOutlineSourceState,
  researchNoteOutlineTags,
} from '../../src/pages/researchNoteOutline'
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

test('outline helpers inherit a shared project and deduplicate traceable sources', () => {
  const source = {
    kind: 'source',
    label: 'Shared source',
    source_path: SOURCE_PATH,
    block_id: 'block-shared',
  }
  const notes: ResearchNoteRecord[] = [
    {
      id: 'outline-a',
      project_id: PROJECT_ID,
      source_conv_id: CONVERSATION_ID,
      title: 'Outline A',
      content_markdown: 'A',
      source_state: { version: 1, links: [source] },
      tags: ['SPAD', 'Noise'],
      pinned: false,
      archived: false,
      revision: 2,
      created_at: 1,
      updated_at: 2,
    },
    {
      id: 'outline-b',
      project_id: PROJECT_ID,
      source_conv_id: CONVERSATION_ID,
      title: 'Outline B',
      content_markdown: 'B',
      source_state: { version: 1, links: [{ ...source }] },
      tags: ['noise', 'detector'],
      pinned: false,
      archived: false,
      revision: 3,
      created_at: 1,
      updated_at: 3,
    },
  ]

  expect(researchNoteOutlineProjectId(notes)).toBe(PROJECT_ID)
  expect(researchNoteOutlineHasMixedProjects(notes)).toBe(false)
  expect(researchNoteOutlineTags(notes)).toEqual(['SPAD', 'Noise', 'detector'])
  expect(researchNoteOutlineSourceState(notes).links).toHaveLength(1)
  expect(researchNoteOutlineProjectId([{ ...notes[0], project_id: null }, notes[1]])).toBeNull()
  expect(researchNoteOutlineHasMixedProjects([{ ...notes[0], project_id: null }, notes[1]])).toBe(true)
})

test('research notes workspace edits, combines, filters, and follows traceable sources', async ({ page }) => {
  let readerPayload: Record<string, unknown> | null = null
  let wordExportBody: Record<string, unknown> | null = null
  let records: ResearchNoteRecord[] = [
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
      source_state: {
        version: 1,
        links: [
          {
            kind: 'source',
            label: 'Noise modeling duplicate',
            source_name: 'Example SPAD paper',
            source_path: SOURCE_PATH,
            heading_path: 'Methods / Noise modeling',
            evidence_quote: 'Photon arrival is a stochastic process.',
            block_id: 'block-noise',
            page_start: 3,
          },
          {
            kind: 'source',
            label: 'Detector dynamic range',
            source_name: 'Detector paper',
            source_path: 'db/paper/detector.md',
            heading_path: 'Results / Dynamic range',
            evidence_quote: 'Dynamic range constrains the detector.',
            block_id: 'block-detector',
            page_start: 5,
          },
        ],
      },
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
        source_conv_id: body.source_conv_id as string | null || null,
        title: String(body.title || ''),
        content_markdown: String(body.content_markdown || ''),
        source_state: body.source_state as ResearchNoteRecord['source_state'] || { version: 1, links: [] },
        tags: Array.isArray(body.tags) ? body.tags as string[] : [],
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
  await expect(outline).toContainText('所选笔记来自不同项目')
  await outline.getByRole('textbox').nth(0).fill('Integrated SPAD writing outline')
  await expect(outline.getByRole('textbox').nth(1)).toHaveValue(/^# Integrated SPAD writing outline/)
  const noteARevisionAtComposition = records.find(record => record.id === 'note-a')?.revision
  await page.getByTestId('research-note-outline-save').click()
  await expect(outline).toBeHidden()

  const outlineRecord = records.find(record => record.title === 'Integrated SPAD writing outline')
  expect(outlineRecord?.project_id).toBeNull()
  expect(outlineRecord?.tags).toEqual(['SPAD', 'noise', 'detector'])
  expect(outlineRecord?.source_state.links).toHaveLength(3)
  expect(outlineRecord?.source_state.composition).toEqual({
    kind: 'research_note_outline',
    version: 1,
    source_notes: [
      { id: 'note-a', title: 'SPAD noise evidence note', revision: noteARevisionAtComposition },
      { id: 'note-b', title: 'Detector trade-off', revision: 1 },
    ],
  })
  await expect(titleInput).toHaveValue('Integrated SPAD writing outline')
  await expect(page.getByTestId('research-note-outline-sources')).toContainText('组成笔记（2）')
  await expect(workspace.locator('.kb-notes-source-card')).toHaveCount(3)

  await page.reload()
  await page.getByTestId(`workspace-note-${outlineRecord?.id}`).click()
  await expect(titleInput).toHaveValue('Integrated SPAD writing outline')
  await expect(workspace.locator('.kb-notes-editor-body textarea')).toHaveValue(/Edited, traceable conclusion/)
  await page.getByTestId('research-note-outline-source-note-a').click()
  await expect(titleInput).toHaveValue('SPAD noise evidence note')

  await workspace.getByRole('button', { name: '打开原文' }).click()
  await expect.poll(() => readerPayload?.sourcePath).toBe(SOURCE_PATH)
  await expect(page).toHaveURL(/\/reader\/session\/reader-note-source/)

  await page.goto('/notes')
  await workspace.getByRole('button', { name: '回到回答' }).click()
  await expect(page).toHaveURL(new RegExp(`conversation=${CONVERSATION_ID}.*note_message=22`))
})

test('workspace preserves fast-switch drafts and recovers safely from save failures and conflicts', async ({ page }) => {
  let failSave = false
  let conflictNextSave = false
  let records = [
    {
      id: 'note-safe-a',
      project_id: PROJECT_ID,
      source_conv_id: CONVERSATION_ID,
      title: 'Safety note A',
      content_markdown: 'Initial evidence A.',
      source_state: { version: 1, links: [] },
      tags: ['safety'],
      pinned: false,
      archived: false,
      revision: 1,
      created_at: 1_780_100_000,
      updated_at: 1_780_100_100,
    },
    {
      id: 'note-safe-b',
      project_id: null,
      source_conv_id: null,
      title: 'Safety note B',
      content_markdown: 'Initial evidence B.',
      source_state: { version: 1, links: [] },
      tags: [],
      pinned: false,
      archived: false,
      revision: 1,
      created_at: 1_780_100_010,
      updated_at: 1_780_100_090,
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
      await fulfillJson(route, records
        .filter(record => !record.archived)
        .map(record => ({ ...record, content_markdown: '' })))
      return
    }
    if (request.method() === 'GET') {
      const record = records.find(item => item.id === noteId)
      await fulfillJson(route, record || { detail: 'not found' }, record ? 200 : 404)
      return
    }
    if (request.method() === 'PATCH') {
      const index = records.findIndex(item => item.id === noteId)
      const current = records[index]
      if (!current) {
        await fulfillJson(route, { detail: 'not found' }, 404)
        return
      }
      if (failSave) {
        await fulfillJson(route, { detail: 'temporary outage' }, 503)
        return
      }
      if (conflictNextSave) {
        conflictNextSave = false
        records[index] = {
          ...current,
          title: 'External revision title',
          content_markdown: 'Content saved by another window.',
          revision: current.revision + 1,
          updated_at: current.updated_at + 1,
        }
        await fulfillJson(route, { detail: 'research note revision conflict' }, 409)
        return
      }
      const body = request.postDataJSON() as Record<string, unknown>
      records[index] = {
        ...current,
        title: String(body.title || current.title),
        content_markdown: String(body.content_markdown || current.content_markdown),
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
        id: `note-safe-${records.length + 1}`,
        project_id: body.project_id as string | null || null,
        source_conv_id: body.source_conv_id as string | null || null,
        title: String(body.title || ''),
        content_markdown: String(body.content_markdown || ''),
        source_state: body.source_state as { version: number; links: never[] } || { version: 1, links: [] },
        tags: Array.isArray(body.tags) ? body.tags as string[] : [],
        pinned: false,
        archived: false,
        revision: 1,
        created_at: 1_780_100_200,
        updated_at: 1_780_100_200,
      }
      records = [record, ...records]
      await fulfillJson(route, record)
      return
    }
    await fulfillJson(route, { ok: true })
  })

  await page.goto('/notes')
  const workspace = page.getByTestId('research-notes-workspace')
  const titleInput = workspace.locator('.kb-notes-title-input')
  const editor = workspace.locator('.kb-notes-editor-body textarea')
  await expect(titleInput).toHaveValue('Safety note A')

  await titleInput.fill('Draft kept during a fast switch')
  await page.getByTestId('workspace-note-note-safe-b').click()
  await expect(titleInput).toHaveValue('Safety note B')
  await page.getByTestId('workspace-note-note-safe-a').click()
  await expect(titleInput).toHaveValue('Draft kept during a fast switch')
  await expect.poll(() => records.find(record => record.id === 'note-safe-a')?.title)
    .toBe('Draft kept during a fast switch')
  await expect.poll(() => page.evaluate(() => localStorage.getItem('kb:research-note-workspace-draft:note-safe-a')))
    .toBeNull()

  failSave = true
  await editor.fill('This edit must survive a temporary save outage.')
  await expect(page.getByTestId('research-note-save-state')).toContainText('保存失败')
  await expect(editor).toHaveValue('This edit must survive a temporary save outage.')
  failSave = false
  await page.getByTestId('research-note-save-now').click()
  await expect.poll(() => records.find(record => record.id === 'note-safe-a')?.content_markdown)
    .toBe('This edit must survive a temporary save outage.')
  await expect(page.getByTestId('research-note-save-state')).toContainText('已保存')

  conflictNextSave = true
  await editor.fill('Keep this local conclusion instead of overwriting it.')
  await expect(page.getByTestId('research-note-save-state')).toContainText('本机草稿仍保留')
  await expect(editor).toHaveValue('Keep this local conclusion instead of overwriting it.')
  expect(records.find(record => record.id === 'note-safe-a')?.content_markdown)
    .toBe('Content saved by another window.')

  await page.getByTestId('research-note-save-copy').click()
  await expect(titleInput).toHaveValue(/冲突副本/)
  await expect(editor).toHaveValue('Keep this local conclusion instead of overwriting it.')
  expect(records[0]?.content_markdown).toBe('Keep this local conclusion instead of overwriting it.')
  expect(records.find(record => record.id === 'note-safe-a')?.content_markdown)
    .toBe('Content saved by another window.')
  await expect.poll(() => page.evaluate(() => localStorage.getItem('kb:research-note-workspace-draft:note-safe-a')))
    .toBeNull()

  const copyId = records[0]?.id || ''
  conflictNextSave = true
  await editor.fill('A second local edit that the user chooses to discard.')
  await expect(page.getByTestId('research-note-save-state')).toContainText('本机草稿仍保留')
  await page.getByTestId('research-note-load-latest').click()
  await page.locator('.ant-popconfirm').getByRole('button', { name: '载入最新版' }).click()
  await expect(editor).toHaveValue('Content saved by another window.')
  await expect(page.getByTestId('research-note-save-state')).toContainText('已保存')
  await expect.poll(() => page.evaluate(
    noteId => localStorage.getItem(`kb:research-note-workspace-draft:${noteId}`),
    copyId,
  )).toBeNull()
})
