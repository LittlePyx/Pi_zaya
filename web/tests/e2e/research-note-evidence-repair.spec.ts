import { expect, test, type Route } from '@playwright/test'
import type { ResearchNoteRecord } from '../../src/api/chat'
import {
  READER_REGRESSION_SOURCE_NAME,
  READER_REGRESSION_SOURCE_PATH,
  buildReaderRegressionDocResponse,
} from '../../src/testing/readerRegressionFixtures'
import {
  installAppShellMocks,
  installEmptyCitationShelfMock,
  installIdleReferenceMocks,
} from './mockAppShell'

async function fulfillJson(route: Route, body: unknown, status = 200) {
  await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
}

test('note evidence health check prioritizes issues, repairs one locator, and verifies it again', async ({ page }) => {
  let repairCalls = 0
  let repairBody: Record<string, unknown> | null = null
  let readerSession: Record<string, unknown> | null = null
  const evidenceContextBodies: Record<string, unknown>[] = []
  const exactSourcePath = 'db/paper/exact.md'
  const changedSourcePath = 'db/paper/changed.md'
  const missingSourcePath = 'db/paper/missing.md'
  const failingSourcePath = 'db/paper/failure.md'
  const answerSourcePath = 'answer://must-not-be-reviewed'
  let note: ResearchNoteRecord = {
    id: 'note-repair',
    project_id: null,
    source_conv_id: null,
    title: 'Detector evidence',
    content_markdown: '## Finding\n\nA saved source location needs review.',
    source_state: {
      version: 1,
      links: [
        {
          kind: 'source',
          label: READER_REGRESSION_SOURCE_NAME,
          source_name: READER_REGRESSION_SOURCE_NAME,
          source_path: READER_REGRESSION_SOURCE_PATH,
          heading_path: 'Fixture Paper / Old section',
          location_label: 'Old section',
          evidence_quote: 'SCI records one compressed video measurement.',
          block_id: 'stale-block',
          anchor_id: 'stale-anchor',
          capture_id: 'capture-repair-1',
          capture_kind: 'selection',
          start_offset: 4,
          end_offset: 49,
        },
        {
          kind: 'source',
          label: 'Exact source',
          source_name: 'Exact source',
          source_path: exactSourcePath,
          evidence_quote: 'This locator is still exact.',
          block_id: 'exact-block',
        },
        {
          kind: 'source',
          label: 'Changed source',
          source_name: 'Changed source',
          source_path: changedSourcePath,
          evidence_quote: 'This source changed after capture.',
          block_id: 'changed-block',
          source_fingerprint: 'sha256:old-source',
        },
        {
          kind: 'source',
          label: 'Unavailable source',
          source_name: 'Unavailable source',
          source_path: missingSourcePath,
          evidence_quote: 'This converted source is unavailable.',
        },
        {
          kind: 'source',
          label: 'Failed source',
          source_name: 'Failed source',
          source_path: failingSourcePath,
          evidence_quote: 'This request fails.',
        },
        {
          kind: 'answer',
          label: 'Saved answer',
          source_name: 'Saved answer',
          source_path: answerSourcePath,
          evidence_quote: 'Answer links are outside evidence health checks.',
        },
      ],
    },
    tags: ['evidence'],
    pinned: false,
    archived: false,
    revision: 1,
    created_at: 1_800_000_000,
    updated_at: 1_800_000_000,
  }

  await installAppShellMocks(page)
  await installEmptyCitationShelfMock(page, { scopeId: 'note-repair' })
  await installIdleReferenceMocks(page)

  await page.route('**/api/chat/research-notes**', async (route) => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    if (request.method() === 'GET' && path === '/api/chat/research-notes') {
      await fulfillJson(route, [{ ...note, content_markdown: '' }])
      return
    }
    if (request.method() === 'GET' && path === `/api/chat/research-notes/${note.id}`) {
      await fulfillJson(route, note)
      return
    }
    if (request.method() === 'POST' && path === `/api/chat/research-notes/${note.id}/evidence-repair`) {
      repairCalls += 1
      repairBody = request.postDataJSON() as Record<string, unknown>
      const previous = note.source_state.links?.[0] || {}
      note = {
        ...note,
        source_state: {
          ...note.source_state,
          links: (note.source_state.links || []).map((link, index) => index === 0 ? {
            ...previous,
            heading_path: 'Fixture Paper / 1. Introduction',
            evidence_quote: 'SCI compresses a short video into one coded measurement.',
            block_id: 'quote-1',
            anchor_id: 'a-quote-1',
            anchor_kind: 'blockquote',
            line_start: 9,
            line_end: 9,
            location_label: undefined,
            start_offset: undefined,
            end_offset: undefined,
            evidence_verification: {
              status: 'exact',
              reason: 'confirmed_candidate',
              verified_at: 1_800_000_100,
              source_fingerprint: 'sha256:fixture',
            },
            locator_repair_history: [{
              version: 1,
              repaired_at: 1_800_000_100,
              source_fingerprint: 'sha256:fixture',
              from: { block_id: 'stale-block', anchor_id: 'stale-anchor' },
              to: { block_id: 'quote-1', anchor_id: 'a-quote-1' },
            }],
          } : link),
        },
        revision: 2,
        updated_at: 1_800_000_100,
      }
      await fulfillJson(route, note)
      return
    }
    await fulfillJson(route, { detail: 'unexpected research note request' }, 404)
  })

  await page.route('**/api/chat/research-note/evidence-context', async (route) => {
    const body = route.request().postDataJSON() as Record<string, unknown>
    evidenceContextBodies.push(body)
    const sourcePath = String(body.source_path || '')
    if (sourcePath === failingSourcePath) {
      await fulfillJson(route, { detail: 'fixture evidence lookup failed' }, 503)
      return
    }
    const repaired = String(body.block_id || '') === 'quote-1'
    const readerOpen = {
      sourcePath: READER_REGRESSION_SOURCE_PATH,
      sourceName: READER_REGRESSION_SOURCE_NAME,
      headingPath: 'Fixture Paper / 1. Introduction',
      snippet: 'SCI compresses a short video into one coded measurement.',
      highlightSnippet: 'SCI compresses a short video into one coded measurement.',
      blockId: 'quote-1',
      anchorId: 'a-quote-1',
      anchorKind: 'blockquote',
      strictLocate: true,
      locateTarget: {
        headingPath: 'Fixture Paper / 1. Introduction',
        snippet: 'SCI compresses a short video into one coded measurement.',
        highlightSnippet: 'SCI compresses a short video into one coded measurement.',
        evidenceQuote: 'SCI compresses a short video into one coded measurement.',
        blockId: 'quote-1',
        anchorId: 'a-quote-1',
        anchorKind: 'blockquote',
      },
    }
    const target = {
      block_id: 'quote-1',
      anchor_id: 'a-quote-1',
      kind: 'blockquote',
      heading_path: 'Fixture Paper / 1. Introduction',
      line_start: 9,
      line_end: 9,
      text: 'SCI compresses a short video into one coded measurement.',
      markdown: '> SCI compresses a short video into one coded measurement.',
    }
    if (sourcePath === missingSourcePath) {
      await fulfillJson(route, {
        status: 'unavailable',
        reason: 'converted_markdown_missing',
        source_name: 'Unavailable source',
        heading_path: '',
        before: [],
        target: null,
        after: [],
        candidates: [],
        can_open_exact: false,
        reader_open: null,
        source_fingerprint: '',
        source_changed: false,
      })
      return
    }
    if (sourcePath === exactSourcePath || sourcePath === changedSourcePath) {
      const sourceChanged = sourcePath === changedSourcePath
      await fulfillJson(route, {
        status: 'exact',
        reason: 'exact_block',
        source_name: sourceChanged ? 'Changed source' : 'Exact source',
        heading_path: sourceChanged ? 'Changed section' : 'Exact section',
        before: [],
        target: {
          ...target,
          block_id: sourceChanged ? 'changed-block' : 'exact-block',
          text: sourceChanged ? 'This source changed after capture.' : 'This locator is still exact.',
          markdown: sourceChanged ? 'This source changed after capture.' : 'This locator is still exact.',
        },
        after: [],
        candidates: [],
        can_open_exact: true,
        reader_open: readerOpen,
        source_fingerprint: sourceChanged ? 'sha256:new-source' : 'sha256:exact-source',
        source_changed: sourceChanged,
      })
      return
    }
    await fulfillJson(route, repaired ? {
      status: 'exact',
      reason: 'exact_block',
      source_name: READER_REGRESSION_SOURCE_NAME,
      heading_path: target.heading_path,
      before: [],
      target,
      after: [],
      candidates: [],
      can_open_exact: true,
      reader_open: readerOpen,
      source_fingerprint: 'sha256:fixture',
      source_changed: false,
    } : {
      status: 'candidate',
      reason: 'saved_target_missing',
      source_name: READER_REGRESSION_SOURCE_NAME,
      heading_path: 'Fixture Paper / Old section',
      before: [],
      target: null,
      after: [],
      candidates: [{ ...target, score: 0.82, reader_open: readerOpen }],
      can_open_exact: false,
      reader_open: null,
      source_fingerprint: 'sha256:fixture',
      source_changed: false,
    })
  })

  await page.route('**/api/reader/sessions**', async (route) => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    if (request.method() === 'POST' && path === '/api/reader/sessions') {
      const body = request.postDataJSON() as Record<string, unknown>
      readerSession = {
        id: 'repair-session',
        title: body.title,
        conversation_id: '',
        message_id: null,
        payload: body.payload,
        state: body.state,
      }
      await fulfillJson(route, readerSession)
      return
    }
    if (request.method() === 'GET' && path === '/api/reader/sessions/repair-session') {
      await fulfillJson(route, readerSession)
      return
    }
    if (request.method() === 'PATCH' && path === '/api/reader/sessions/repair-session/state') {
      const body = request.postDataJSON() as { state?: Record<string, unknown> }
      readerSession = { ...readerSession, state: { ...(readerSession?.state as object || {}), ...(body.state || {}) } }
      await fulfillJson(route, readerSession)
      return
    }
    await fulfillJson(route, { detail: 'unexpected reader session request' }, 404)
  })

  await page.route('**/api/references/reader/doc', async (route) => {
    await fulfillJson(route, buildReaderRegressionDocResponse('strict-quote'))
  })

  await page.goto('/notes?note=note-repair')
  await expect(page.locator('.kb-notes-title-input')).toHaveValue('Detector evidence')
  await page.getByTestId('research-note-evidence-check-all').click()
  const health = page.getByTestId('research-note-evidence-health')
  await expect(health).toContainText('已核查 5/5 条文献证据')
  await expect(health).toContainText('准确 1')
  await expect(health).toContainText('原文变化 1')
  await expect(health).toContainText('需核查 1')
  await expect(health).toContainText('不可用 1')
  await expect(health).toContainText('失败 1')
  await expect(page.locator('.kb-notes-source-card')).toHaveCount(4)
  await expect(page.getByTestId('research-note-source-0')).toBeVisible()
  await expect(page.getByTestId('research-note-source-1')).toHaveCount(0)
  await expect(page.getByTestId('research-note-source-2')).toBeVisible()
  await expect(page.getByTestId('research-note-source-3')).toBeVisible()
  await expect(page.getByTestId('research-note-source-4')).toBeVisible()
  await expect(page.getByTestId('research-note-source-5')).toHaveCount(0)
  await expect(page.getByTestId('research-note-evidence-health-status-0')).toHaveText('需核查')
  await expect(page.getByTestId('research-note-evidence-health-status-2')).toHaveText('原文变化')
  await expect(page.getByTestId('research-note-evidence-health-status-3')).toHaveText('不可用')
  await expect(page.getByTestId('research-note-evidence-health-status-4')).toHaveText('失败')

  await page.getByText('全部来源 6', { exact: true }).click()
  await expect(page.locator('.kb-notes-source-card')).toHaveCount(6)
  await expect(page.getByTestId('research-note-source-1')).toBeVisible()
  await expect(page.getByTestId('research-note-source-5')).toBeVisible()
  await page.getByText('需处理 4', { exact: true }).click()
  await expect(page.locator('.kb-notes-source-card')).toHaveCount(4)

  await expect(page.getByTestId('research-note-evidence-result-0')).toContainText('需要核查')
  await page.getByTestId('research-note-evidence-repair-candidate-0-0').click()

  await expect(page).toHaveURL(/\/reader\/session\/repair-session/)
  await expect(page.getByTestId('reader-evidence-repair-panel')).toBeVisible()
  await expect(page.getByTestId('reader-content')).toContainText('Fixture Paper')
  const confirm = page.getByTestId('reader-confirm-evidence-repair')
  await expect(confirm).toBeEnabled()
  expect(repairCalls).toBe(0)
  await confirm.click()

  await expect(page).toHaveURL(/\/notes\?note=note-repair/)
  await expect(page.locator('.kb-notes-title-input')).toHaveValue('Detector evidence')
  await expect(page.getByTestId('research-note-source-0')).toContainText('SCI compresses a short video into one coded measurement.')
  expect(repairCalls).toBe(1)
  expect(repairBody).toEqual(expect.objectContaining({
    source_index: 0,
    source_path: READER_REGRESSION_SOURCE_PATH,
    capture_id: 'capture-repair-1',
    old_block_id: 'stale-block',
    candidate_block_id: 'quote-1',
    candidate_anchor_id: 'a-quote-1',
    source_fingerprint: 'sha256:fixture',
  }))
  expect(note.source_state.links?.[0]?.locator_repair_history?.[0]?.from).toEqual(
    expect.objectContaining({ block_id: 'stale-block' }),
  )

  await page.getByTestId('research-note-evidence-check-all').click()
  const repairedHealth = page.getByTestId('research-note-evidence-health')
  await expect(repairedHealth).toContainText('已核查 5/5 条文献证据')
  await expect(repairedHealth).toContainText('准确 2')
  await expect(repairedHealth).toContainText('原文变化 1')
  await expect(repairedHealth).toContainText('需核查 0')
  await expect(repairedHealth).toContainText('不可用 1')
  await expect(repairedHealth).toContainText('失败 1')
  await expect(page.getByText('需处理 3', { exact: true })).toBeVisible()
  await expect(page.getByTestId('research-note-source-0')).toHaveCount(0)
  await page.getByText('全部来源 6', { exact: true }).click()
  await expect(page.getByTestId('research-note-evidence-result-0')).toContainText('准确定位')
  await expect(page.getByTestId('research-note-evidence-open-0')).toBeVisible()
  expect(evidenceContextBodies).toHaveLength(10)
  expect(evidenceContextBodies.every(body => String(body.source_path || '') !== answerSourcePath)).toBe(true)
})
