import { expect, test, type Route } from '@playwright/test'
import type { ResearchNoteRecord } from '../../src/api/chat'
import {
  researchNoteEvidenceMaintenanceDue,
  researchNoteEvidenceMaintenanceTracked,
  researchNoteEvidenceNextRetryAt,
  researchNoteEvidenceReviewIssues,
  researchNoteEvidenceReviewSummary,
  researchNoteEvidenceSystemRetryRequired,
} from '../../src/pages/researchNoteEvidenceReview'
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

test('research note evidence review summary counts only unresolved saved checks', () => {
  const sourceState = {
    links: [
      {
        kind: 'source',
        source_name: 'Needs review',
        source_path: 'db/paper/needs-review.md',
        evidence_baseline_check: { status: 'skipped', reason: 'candidate', checked_at: 10 },
      },
      {
        kind: 'source',
        source_name: 'Check failed',
        source_path: 'db/paper/check-failed.md',
        evidence_baseline_check: { status: 'failed', reason: 'verification_failed', checked_at: 20 },
      },
      {
        kind: 'source',
        source_name: 'Already resolved',
        source_path: 'db/paper/resolved.md',
        evidence_context_snapshot: { source_fingerprint: 'sha256:current' },
        evidence_baseline_check: { status: 'skipped', reason: 'candidate' },
      },
      {
        kind: 'source',
        source_name: 'Explicitly invalid',
        source_path: 'db/paper/invalid.md',
        evidence_verification: { status: 'invalid' },
        evidence_baseline_check: { status: 'skipped', reason: 'source_changed' },
      },
      {
        kind: 'answer',
        evidence_baseline_check: { status: 'failed', reason: 'verification_failed' },
      },
      {
        kind: 'source',
        source_name: 'External reference without local full text',
        evidence_baseline_check: { status: 'failed', reason: 'unavailable' },
      },
    ],
  }

  expect(researchNoteEvidenceReviewIssues(sourceState).map(issue => issue.sourceIndex)).toEqual([0])
  expect(researchNoteEvidenceReviewSummary(sourceState)).toEqual({
    needsReview: 1,
    failed: 0,
    total: 1,
    firstSourceIndex: 0,
  })
  expect(researchNoteEvidenceSystemRetryRequired(sourceState)).toBe(true)
  expect(researchNoteEvidenceMaintenanceTracked(sourceState)).toBe(true)
  expect(researchNoteEvidenceMaintenanceDue(sourceState, 100)).toBe(true)

  const scheduledRetryState = {
    links: [{
      kind: 'source',
      source_path: 'db/paper/check-failed.md',
      evidence_baseline_check: {
        status: 'failed',
        reason: 'verification_failed',
        maintenance_version: 2,
        retry_count: 1,
        retry_after: 120,
      },
    }],
  }
  expect(researchNoteEvidenceMaintenanceDue(scheduledRetryState, 100)).toBe(false)
  expect(researchNoteEvidenceMaintenanceDue(scheduledRetryState, 120)).toBe(true)
  expect(researchNoteEvidenceNextRetryAt(scheduledRetryState)).toBe(120)

  const legacyContextDecisionState = {
    links: [{
      kind: 'source',
      source_path: 'db/paper/repeated.md',
      evidence_context_snapshot: {
        version: 1,
        source_fingerprint: 'sha256:saved',
        target: { block_id: 'old-repeat', text: 'The same sentence appears more than once.' },
      },
      evidence_baseline_check: {
        status: 'skipped',
        reason: 'candidate',
        maintenance_version: 1,
      },
    }],
  }
  expect(researchNoteEvidenceMaintenanceDue(legacyContextDecisionState, 100)).toBe(true)
  legacyContextDecisionState.links[0].evidence_baseline_check.maintenance_version = 2
  expect(researchNoteEvidenceMaintenanceDue(legacyContextDecisionState, 100)).toBe(false)

  const savedContextRetryState = {
    links: [{
      kind: 'source',
      source_path: 'db/paper/repeated.md',
      evidence_context_snapshot: {
        version: 1,
        source_fingerprint: 'sha256:saved',
        target: { block_id: 'old-repeat', text: 'The same sentence appears more than once.' },
      },
      evidence_baseline_check: {
        status: 'failed',
        reason: 'verification_failed',
        maintenance_version: 2,
        retry_after: 120,
      },
    }],
  }
  expect(researchNoteEvidenceMaintenanceDue(savedContextRetryState, 100)).toBe(false)
  expect(researchNoteEvidenceMaintenanceDue(savedContextRetryState, 120)).toBe(true)
})

test('note evidence health check prioritizes issues, repairs one locator, and verifies it again', async ({ page }) => {
  let repairCalls = 0
  let repairBody: Record<string, unknown> | null = null
  const evidenceReviewBodies: Record<string, unknown>[] = []
  let readerSession: Record<string, unknown> | null = null
  const evidenceContextBodies: Record<string, unknown>[] = []
  const exactSourcePath = 'db/paper/exact.md'
  const changedSourcePath = 'db/paper/changed.md'
  const missingSourcePath = 'db/paper/missing.md'
  const failingSourcePath = 'db/paper/failure.md'
  const answerSourcePath = 'answer://must-not-be-reviewed'
  const changedSavedContext = {
    version: 1 as const,
    captured_at: 1_799_999_900,
    source_fingerprint: 'sha256:old-source',
    reason: 'exact_block',
    before: [{ block_id: 'before-change', text: 'Earlier calibration context.' }],
    target: {
      block_id: 'quote-1',
      anchor_id: 'a-quote-1',
      kind: 'blockquote',
      text: 'SCI compresses a short video into one coded measurement.',
    },
    after: [{ block_id: 'after-change', text: 'The following limitation remains unchanged.' }],
  }
  const changedCurrentContext = {
    version: 1 as const,
    captured_at: 1_800_000_200,
    source_fingerprint: 'sha256:new-source',
    reason: 'exact_block',
    before: [{ block_id: 'before-change', text: 'Updated calibration context.' }],
    target: {
      block_id: 'quote-1',
      anchor_id: 'a-quote-1',
      kind: 'blockquote',
      text: 'SCI compresses a short video into one coded measurement.',
    },
    after: [{ block_id: 'after-change', text: 'The following limitation remains unchanged.' }],
  }
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
          evidence_context_snapshot: {
            version: 1,
            captured_at: 1_799_999_800,
            source_fingerprint: 'sha256:fixture',
            reason: 'exact_block',
            before: [{ block_id: 'before-old', text: 'The coded exposure follows detector calibration.' }],
            target: { block_id: 'stale-block', text: 'SCI records one compressed video measurement.' },
            after: [{ block_id: 'after-old', text: 'The reconstruction is evaluated on the same sequence.' }],
          },
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
          evidence_quote: 'SCI compresses a short video into one coded measurement.',
          block_id: 'quote-1',
          anchor_id: 'a-quote-1',
          anchor_kind: 'blockquote',
          evidence_verification: {
            status: 'exact',
            reason: 'confirmed_candidate',
            verified_at: 1_799_999_900,
            source_fingerprint: 'sha256:old-source',
          },
          evidence_context_snapshot: changedSavedContext,
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
            evidence_context_snapshot: {
              version: 1,
              captured_at: 1_800_000_100,
              source_fingerprint: 'sha256:fixture',
              reason: 'exact_block',
              before: [],
              target: {
                block_id: 'quote-1',
                anchor_id: 'a-quote-1',
                kind: 'blockquote',
                text: 'SCI compresses a short video into one coded measurement.',
              },
              after: [],
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
        revision: note.revision + 1,
        updated_at: 1_800_000_100,
      }
      await fulfillJson(route, note)
      return
    }
    if (request.method() === 'POST' && path === `/api/chat/research-notes/${note.id}/evidence-review`) {
      const body = request.postDataJSON() as Record<string, unknown>
      evidenceReviewBodies.push(body)
      const decision = String(body.decision || '')
      const previous = note.source_state.links?.[2] || {}
      const previousVerification = previous.evidence_verification || {}
      note = {
        ...note,
        source_state: {
          ...note.source_state,
          links: (note.source_state.links || []).map((link, index) => index === 2 ? {
            ...link,
            evidence_context_snapshot: decision === 'confirmed'
              ? changedCurrentContext
              : link.evidence_context_snapshot,
            evidence_verification: {
              status: decision === 'invalid' ? 'invalid' : 'exact',
              reason: decision === 'invalid' ? 'marked_invalid' : 'confirmed_source_change',
              verified_at: 1_800_000_200 + evidenceReviewBodies.length,
              source_fingerprint: 'sha256:new-source',
            },
            evidence_review_history: [
              ...(link.evidence_review_history || []),
              {
                version: 1,
                reviewed_at: 1_800_000_200 + evidenceReviewBodies.length,
                decision,
                previous_verification: previousVerification,
                saved_locator: { block_id: link.block_id, anchor_id: link.anchor_id },
                previous_context_snapshot: link.evidence_context_snapshot,
                reviewed_context_snapshot: changedCurrentContext,
              },
            ],
          } : link),
        },
        revision: note.revision + 1,
        updated_at: 1_800_000_200 + evidenceReviewBodies.length,
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
      const currentFingerprint = String(body.source_fingerprint || '')
      const currentReaderOpen = sourceChanged ? {
        ...readerOpen,
        sourcePath: changedSourcePath,
        sourceName: 'Changed source',
      } : readerOpen
      await fulfillJson(route, {
        status: 'exact',
        reason: 'exact_block',
        source_name: sourceChanged ? 'Changed source' : 'Exact source',
        heading_path: sourceChanged ? 'Changed section' : 'Exact section',
        before: [],
        target: {
          ...target,
          block_id: sourceChanged ? 'quote-1' : 'exact-block',
          anchor_id: sourceChanged ? 'a-quote-1' : target.anchor_id,
          text: sourceChanged ? 'SCI compresses a short video into one coded measurement.' : 'This locator is still exact.',
          markdown: sourceChanged ? '> SCI compresses a short video into one coded measurement.' : 'This locator is still exact.',
        },
        after: [],
        candidates: [],
        can_open_exact: true,
        reader_open: currentReaderOpen,
        source_fingerprint: sourceChanged ? 'sha256:new-source' : 'sha256:exact-source',
        source_changed: sourceChanged && currentFingerprint !== 'sha256:new-source',
        context_snapshot: sourceChanged ? changedCurrentContext : {
          version: 1,
          captured_at: 1_800_000_200,
          source_fingerprint: 'sha256:exact-source',
          reason: 'exact_block',
          before: [],
          target: {
            block_id: 'exact-block',
            text: 'This locator is still exact.',
          },
          after: [],
        },
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
      context_snapshot: {
        version: 1,
        captured_at: 1_800_000_200,
        source_fingerprint: 'sha256:fixture',
        reason: 'exact_block',
        before: [],
        target,
        after: [],
      },
    } : {
      status: 'candidate',
      reason: 'saved_target_missing',
      source_name: READER_REGRESSION_SOURCE_NAME,
      heading_path: 'Fixture Paper / Old section',
      before: [{
        block_id: 'before-candidate',
        kind: 'paragraph',
        text: 'The coded exposure follows detector calibration.',
        markdown: 'The coded exposure follows detector calibration.',
      }],
      target: null,
      after: [{
        block_id: 'after-candidate',
        kind: 'paragraph',
        text: 'The reconstruction is evaluated on the same sequence.',
        markdown: 'The reconstruction is evaluated on the same sequence.',
      }],
      candidates: [{
        ...target,
        score: 0.82,
        before: [{
          block_id: 'before-candidate',
          kind: 'paragraph',
          text: 'The coded exposure follows detector calibration.',
          markdown: 'The coded exposure follows detector calibration.',
        }],
        after: [{
          block_id: 'after-candidate',
          kind: 'paragraph',
          text: 'The reconstruction is evaluated on the same sequence.',
          markdown: 'The reconstruction is evaluated on the same sequence.',
        }],
        context_match: {
          available_fields: ['before', 'after', 'heading'],
          matched_fields: ['before', 'after'],
        },
        reader_open: readerOpen,
      }],
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

  await expect(page.getByTestId('research-note-evidence-review-change-2')).toHaveText('在 Reader 中对照核实')
  await page.getByTestId('research-note-evidence-review-change-2').click()
  await expect(page).toHaveURL(/\/reader\/session\/repair-session/)
  await expect(page.getByTestId('reader-evidence-review-panel')).toBeVisible()
  await expect(page.getByTestId('reader-evidence-review-saved')).toContainText(
    'Earlier calibration context.',
  )
  await expect(page.getByTestId('reader-evidence-review-current')).toContainText(
    'Updated calibration context.',
  )
  await expect(page.getByTestId('reader-evidence-review-saved-before').locator('mark.is-previous')).toBeVisible()
  await expect(page.getByTestId('reader-evidence-review-current-before').locator('mark.is-current')).toBeVisible()
  const invalidate = page.getByTestId('reader-invalidate-evidence-review')
  await expect(invalidate).toBeEnabled()
  await invalidate.click()

  await expect(page).toHaveURL(/\/notes\?note=note-repair/)
  expect(note.source_state.links?.[2]?.evidence_verification?.status).toBe('invalid')
  expect(note.source_state.links?.[2]?.evidence_review_history?.[0]?.decision).toBe('invalid')
  expect(note.source_state.links?.[2]?.evidence_context_snapshot?.source_fingerprint).toBe('sha256:old-source')
  await page.getByTestId('research-note-evidence-check-all').click()
  const invalidHealth = page.getByTestId('research-note-evidence-health')
  await expect(invalidHealth).toContainText('已核查 5/5 条文献证据')
  await expect(invalidHealth).toContainText('准确 1')
  await expect(invalidHealth).toContainText('原文变化 0')
  await expect(invalidHealth).toContainText('需核查 1')
  await expect(invalidHealth).toContainText('已失效 1')
  await expect(page.getByTestId('research-note-evidence-health-status-2')).toHaveText('已失效')
  await expect(page.getByTestId('research-note-evidence-review-change-2')).toHaveText('在 Reader 中重新核实')
  await page.getByTestId('research-note-evidence-review-change-2').click()

  await expect(page.getByTestId('reader-evidence-review-panel')).toContainText('这条证据此前已标记失效')
  const reconfirm = page.getByTestId('reader-confirm-evidence-review')
  await expect(reconfirm).toBeEnabled()
  await reconfirm.click()
  await expect(page).toHaveURL(/\/notes\?note=note-repair/)
  expect(note.source_state.links?.[2]?.evidence_verification?.status).toBe('exact')
  expect(note.source_state.links?.[2]?.evidence_context_snapshot?.source_fingerprint).toBe('sha256:new-source')
  expect(note.source_state.links?.[2]?.evidence_review_history?.map(row => row.decision)).toEqual([
    'invalid',
    'confirmed',
  ])
  await page.getByTestId('research-note-evidence-check-all').click()
  const confirmedHealth = page.getByTestId('research-note-evidence-health')
  await expect(confirmedHealth).toContainText('准确 2')
  await expect(confirmedHealth).toContainText('原文变化 0')
  await expect(confirmedHealth).toContainText('已失效 0')
  await expect(page.getByText('需处理 3', { exact: true })).toBeVisible()

  await expect(page.getByTestId('research-note-evidence-result-0')).toContainText('需要核查')
  await expect(page.getByTestId('research-note-evidence-candidate-matches-0-0')).toContainText('前文')
  await expect(page.getByTestId('research-note-evidence-candidate-matches-0-0')).toContainText('后文')
  await expect(page.getByTestId('research-note-evidence-result-0')).toContainText(
    'The coded exposure follows detector calibration.',
  )
  await expect(page.getByTestId('research-note-evidence-result-0')).toContainText(
    'The reconstruction is evaluated on the same sequence.',
  )
  expect(evidenceContextBodies.find(body => body.source_path === READER_REGRESSION_SOURCE_PATH)).toEqual(
    expect.objectContaining({
      evidence_context_snapshot: expect.objectContaining({
        source_fingerprint: 'sha256:fixture',
      }),
    }),
  )
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
  await expect(repairedHealth).toContainText('准确 3')
  await expect(repairedHealth).toContainText('原文变化 0')
  await expect(repairedHealth).toContainText('需核查 0')
  await expect(repairedHealth).toContainText('已失效 0')
  await expect(repairedHealth).toContainText('不可用 1')
  await expect(repairedHealth).toContainText('失败 1')
  await expect(page.getByText('需处理 2', { exact: true })).toBeVisible()
  await expect(page.getByTestId('research-note-source-0')).toHaveCount(0)
  await page.getByText('全部来源 6', { exact: true }).click()
  await expect(page.getByTestId('research-note-evidence-result-0')).toContainText('准确定位')
  await expect(page.getByTestId('research-note-evidence-open-0')).toBeVisible()
  expect(evidenceReviewBodies.map(body => body.decision)).toEqual(['invalid', 'confirmed'])
  expect(evidenceReviewBodies[0]).toEqual(expect.objectContaining({
    source_index: 2,
    expected_source_fingerprint: 'sha256:old-source',
    source_fingerprint: 'sha256:new-source',
    current_block_id: 'quote-1',
  }))
  expect(evidenceContextBodies).toHaveLength(20)
  expect(evidenceContextBodies.every(body => String(body.source_path || '') !== answerSourcePath)).toBe(true)
})

test('legacy evidence review labels a missing historical context without fabricating it', async ({ page }) => {
  await installAppShellMocks(page)
  await installEmptyCitationShelfMock(page, { scopeId: 'legacy-context-review' })
  await installIdleReferenceMocks(page)

  const quote = 'SCI compresses a short video into one coded measurement.'
  const readerOpen = {
    sourcePath: READER_REGRESSION_SOURCE_PATH,
    sourceName: READER_REGRESSION_SOURCE_NAME,
    headingPath: 'Fixture Paper / 1. Introduction',
    snippet: quote,
    highlightSnippet: quote,
    blockId: 'quote-1',
    anchorId: 'a-quote-1',
    anchorKind: 'blockquote',
    strictLocate: true,
    locateTarget: {
      headingPath: 'Fixture Paper / 1. Introduction',
      snippet: quote,
      highlightSnippet: quote,
      evidenceQuote: quote,
      blockId: 'quote-1',
      anchorId: 'a-quote-1',
      anchorKind: 'blockquote',
    },
  }
  await page.route('**/api/reader/sessions/legacy-context-review', async (route) => {
    await fulfillJson(route, {
      id: 'legacy-context-review',
      title: READER_REGRESSION_SOURCE_NAME,
      conversation_id: '',
      message_id: null,
      payload: readerOpen,
      state: {
        researchNoteEvidenceReview: {
          version: 1,
          note_id: 'legacy-note',
          note_title: 'Legacy note',
          source_label: READER_REGRESSION_SOURCE_NAME,
          saved_evidence_quote: quote,
          current_evidence_quote: quote,
          previously_invalid: false,
          current_context_snapshot: {
            version: 1,
            captured_at: 1_800_000_000,
            source_fingerprint: 'sha256:new-source',
            reason: 'exact_block',
            before: [{ block_id: 'before', text: 'Current preceding context.' }],
            target: { block_id: 'quote-1', anchor_id: 'a-quote-1', kind: 'blockquote', text: quote },
            after: [{ block_id: 'after', text: 'Current following context.' }],
          },
          review: {
            source_index: 0,
            source_path: READER_REGRESSION_SOURCE_PATH,
            old_block_id: 'quote-1',
            old_anchor_id: 'a-quote-1',
            old_evidence_quote: quote,
            expected_source_fingerprint: 'sha256:old-source',
            current_block_id: 'quote-1',
            current_anchor_id: 'a-quote-1',
            current_anchor_kind: 'blockquote',
            current_evidence_quote: quote,
            source_fingerprint: 'sha256:new-source',
          },
        },
      },
    })
  })
  await page.route('**/api/references/reader/doc', async (route) => {
    await fulfillJson(route, buildReaderRegressionDocResponse('strict-quote'))
  })

  await page.goto('/reader/session/legacy-context-review')
  const panel = page.getByTestId('reader-evidence-review-panel')
  await expect(panel).toContainText('这条旧笔记没有保存时的语境快照')
  await expect(page.getByTestId('reader-evidence-review-saved')).toContainText('仅保留了摘录')
  await expect(page.getByTestId('reader-evidence-review-current')).toContainText('Current preceding context.')
  await expect(page.getByTestId('reader-confirm-evidence-review')).toBeEnabled()
})

test('legacy note automatically resolves deterministic evidence and leaves only real decisions', async ({ page }) => {
  await page.setViewportSize({ width: 1024, height: 800 })
  await installAppShellMocks(page)
  await installEmptyCitationShelfMock(page, { scopeId: 'note-baseline' })
  await installIdleReferenceMocks(page)

  const quote = 'SCI compresses a short video into one coded measurement.'
  let baselineBody: Record<string, unknown> | null = null
  const maintenanceBodies: Array<{ notes?: Array<{ note_id: string; expected_revision: number }> }> = []
  const evidenceContextBodies: Record<string, unknown>[] = []
  let markBaselineStarted: (() => void) | undefined
  let finishBaselineResponse: (() => void) | undefined
  const baselineStarted = new Promise<void>((resolve) => { markBaselineStarted = resolve })
  const baselineResponseReleased = new Promise<void>((resolve) => { finishBaselineResponse = resolve })
  let note: ResearchNoteRecord = {
    id: 'note-baseline',
    project_id: null,
    source_conv_id: null,
    title: 'Legacy evidence baseline',
    content_markdown: 'A note body that must not be rewritten.',
    source_state: {
      version: 1,
      links: [
        {
          kind: 'source',
          label: READER_REGRESSION_SOURCE_NAME,
          source_name: READER_REGRESSION_SOURCE_NAME,
          source_path: READER_REGRESSION_SOURCE_PATH,
          block_id: 'quote-1',
          anchor_id: 'a-quote-1',
          anchor_kind: 'blockquote',
          evidence_quote: quote,
        },
        {
          kind: 'source',
          label: 'Changed paper',
          source_name: 'Changed paper',
          source_path: 'db/paper/changed.md',
          block_id: 'changed-block',
          evidence_quote: 'Changed evidence must be skipped.',
          evidence_verification: {
            status: 'exact',
            source_fingerprint: 'sha256:old-source',
          },
          evidence_baseline_check: {
            version: 1,
            checked_at: 1_799_999_900,
            status: 'skipped',
            reason: 'source_changed',
            source_fingerprint: 'sha256:new-source',
            locator: { block_id: 'changed-block' },
          },
        },
        {
          kind: 'source',
          label: 'Invalid evidence',
          source_name: 'Invalid evidence',
          source_path: 'db/paper/invalid.md',
          block_id: 'invalid-block',
          evidence_quote: 'This evidence was explicitly marked invalid.',
          evidence_verification: {
            status: 'invalid',
            reason: 'marked_invalid',
            source_fingerprint: 'sha256:invalid-source',
          },
        },
      ],
    },
    tags: [],
    pinned: false,
    archived: false,
    revision: 1,
    created_at: 1_800_000_000,
    updated_at: 1_800_000_000,
  }
  let failedNote: ResearchNoteRecord = {
    ...note,
    id: 'note-failed-check',
    title: 'Evidence check failed',
    content_markdown: 'A separate saved note.',
    source_state: {
      version: 1,
      links: [{
        kind: 'source',
        source_name: 'Temporarily unavailable paper',
        source_path: 'db/paper/unavailable.md',
        evidence_baseline_check: {
          version: 1,
          checked_at: 1_800_000_050,
          status: 'failed',
          reason: 'verification_failed',
        },
      }],
    },
    updated_at: 1_799_999_990,
  }
  const cleanNote: ResearchNoteRecord = {
    ...note,
    id: 'note-clean-check',
    title: 'Evidence already checked',
    content_markdown: 'A checked saved note.',
    source_state: {
      version: 1,
      links: [{
        kind: 'source',
        source_name: 'Checked paper',
        source_path: 'db/paper/checked.md',
        evidence_context_snapshot: {
          version: 1,
          captured_at: 1_800_000_000,
          source_fingerprint: 'sha256:checked',
          reason: 'exact_block',
          before: [],
          target: { block_id: 'checked', text: 'Checked evidence.' },
          after: [],
        },
      }],
    },
    updated_at: 1_799_999_980,
  }
  let needsDecisionNote: ResearchNoteRecord = {
    ...note,
    id: 'note-needs-decision',
    title: 'Ambiguous source evidence',
    content_markdown: 'A saved quote with multiple possible locations.',
    source_state: {
      version: 1,
      links: [{
        kind: 'source',
        source_name: 'Repeated evidence paper',
        source_path: 'db/paper/repeated.md',
        block_id: 'stale-repeat',
        evidence_quote: 'The same sentence appears more than once.',
        evidence_baseline_check: {
          version: 1,
          checked_at: 1_800_000_040,
          status: 'skipped',
          reason: 'candidate',
        },
      }],
    },
    updated_at: 1_799_999_985,
  }
  await page.route('**/api/chat/research-notes**', async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    if (request.method() === 'GET' && url.pathname === '/api/chat/research-notes') {
      await fulfillJson(route, [
        { ...note, content_markdown: '' },
        { ...failedNote, content_markdown: '' },
        { ...needsDecisionNote, content_markdown: '' },
        { ...cleanNote, content_markdown: '' },
      ])
      return
    }
    if (request.method() === 'GET' && url.pathname === `/api/chat/research-notes/${note.id}`) {
      await fulfillJson(route, note)
      return
    }
    if (request.method() === 'GET' && url.pathname === `/api/chat/research-notes/${failedNote.id}`) {
      await fulfillJson(route, failedNote)
      return
    }
    if (request.method() === 'GET' && url.pathname === `/api/chat/research-notes/${cleanNote.id}`) {
      await fulfillJson(route, cleanNote)
      return
    }
    if (request.method() === 'GET' && url.pathname === `/api/chat/research-notes/${needsDecisionNote.id}`) {
      await fulfillJson(route, needsDecisionNote)
      return
    }
    await fulfillJson(route, { detail: 'unexpected research note request' }, 404)
  })
  await page.route('**/api/chat/research-notes/evidence-maintenance', async (route) => {
    const body = route.request().postDataJSON() as {
      notes?: Array<{ note_id: string; expected_revision: number }>
    }
    maintenanceBodies.push(body)
    expect(body.notes).toEqual([
      { note_id: failedNote.id, expected_revision: 1 },
      { note_id: needsDecisionNote.id, expected_revision: 1 },
    ])

    failedNote = {
      ...failedNote,
      revision: 2,
      source_state: {
        ...failedNote.source_state,
        links: (failedNote.source_state.links || []).map(link => ({
          ...link,
          evidence_baseline_check: {
            ...link.evidence_baseline_check,
            version: 1,
            checked_at: 1_800_000_100,
            status: 'failed',
            reason: 'verification_failed',
            maintenance_version: 2,
            retry_count: 1,
            retry_after: 1_900_000_000,
          },
        })),
      },
    }
    needsDecisionNote = {
      ...needsDecisionNote,
      revision: 2,
      source_state: {
        ...needsDecisionNote.source_state,
        links: (needsDecisionNote.source_state.links || []).map(link => ({
          ...link,
          evidence_baseline_check: {
            ...link.evidence_baseline_check,
            version: 1,
            checked_at: 1_800_000_100,
            status: 'skipped',
            reason: 'candidate',
            maintenance_version: 2,
          },
        })),
      },
    }
    await fulfillJson(route, {
      notes: [
        { ...failedNote, content_markdown: '' },
        { ...needsDecisionNote, content_markdown: '' },
      ],
      summary: {
        requested: 2,
        processed: 2,
        updated: 2,
        deferred: 0,
        conflicted: 0,
        not_found: 0,
        errors: 0,
        resolved_links: 0,
        decision_links: 1,
        has_more: false,
      },
      items: [
        { note_id: failedNote.id, status: 'processed', revision: 2 },
        { note_id: needsDecisionNote.id, status: 'processed', revision: 2 },
      ],
    })
  })
  await page.route(`**/api/chat/research-notes/${note.id}/evidence-context-baseline`, async (route) => {
    baselineBody = route.request().postDataJSON() as Record<string, unknown>
    markBaselineStarted?.()
    await baselineResponseReleased
    const originalLink = note.source_state.links?.[0]
    expect(originalLink).toBeTruthy()
    const snapshot = {
      version: 1 as const,
      captured_at: 1_800_000_100,
      source_fingerprint: 'sha256:current-source',
      reason: 'exact_block',
      before: [{ block_id: 'before', text: 'Current context before.' }],
      target: { block_id: 'quote-1', anchor_id: 'a-quote-1', kind: 'blockquote', text: quote },
      after: [{ block_id: 'after', text: 'Current context after.' }],
    }
    note = {
      ...note,
      revision: 2,
      source_state: {
        ...note.source_state,
        links: (note.source_state.links || []).map((link, index) => {
          if (index === 0) return {
            ...link,
            evidence_context_snapshot: snapshot,
            evidence_verification: {
              status: 'exact',
              reason: 'baseline_backfilled',
              verified_at: snapshot.captured_at,
              source_fingerprint: snapshot.source_fingerprint,
            },
            evidence_context_history: [{
              version: 1,
              action: 'baseline_backfilled',
              captured_at: snapshot.captured_at,
              source_fingerprint: snapshot.source_fingerprint,
            }],
          }
          if (index === 1) return {
            ...link,
            evidence_baseline_check: undefined,
            evidence_context_snapshot: {
              version: 1,
              captured_at: 1_800_000_100,
              source_fingerprint: 'sha256:new-source',
              reason: 'exact_block',
              before: [],
              target: {
                block_id: 'changed-block',
                kind: 'paragraph',
                text: 'Changed evidence must be skipped.',
              },
              after: [],
            },
            evidence_verification: {
              status: 'exact',
              reason: 'source_change_auto_verified',
              verified_at: 1_800_000_100,
              source_fingerprint: 'sha256:new-source',
            },
            evidence_auto_resolution_history: [{
              version: 1,
              resolved_at: 1_800_000_100,
              action: 'source_change_auto_verified',
              match_reason: 'exact_block',
              source_fingerprint: 'sha256:new-source',
            }],
          }
          return link
        }),
      },
    }
    await fulfillJson(route, {
      note,
      summary: {
        total: 3,
        created: 2,
        skipped: 1,
        failed: 0,
        reasons: { baseline_created: 1, source_change_auto_verified: 1, invalid: 1 },
      },
      items: [
        { source_index: 0, source_name: READER_REGRESSION_SOURCE_NAME, status: 'created', reason: 'baseline_created' },
        {
          source_index: 1,
          source_name: 'Changed paper',
          status: 'created',
          reason: 'source_change_auto_verified',
        },
        { source_index: 2, source_name: 'Invalid evidence', status: 'skipped', reason: 'invalid' },
      ],
    })
  })
  await page.route('**/api/chat/research-note/evidence-context', async (route) => {
    const body = route.request().postDataJSON() as Record<string, unknown>
    evidenceContextBodies.push(body)
    await fulfillJson(route, {
      status: 'exact',
      reason: 'exact_block',
      source_name: 'Changed paper',
      heading_path: 'Changed paper / Results',
      before: [],
      target: {
        block_id: 'changed-block',
        anchor_id: '',
        kind: 'paragraph',
        heading_path: 'Changed paper / Results',
        line_start: 12,
        line_end: 12,
        text: 'Changed evidence must be skipped.',
        markdown: 'Changed evidence must be skipped.',
      },
      after: [],
      candidates: [],
      can_open_exact: true,
      reader_open: {
        sourcePath: 'db/paper/changed.md',
        sourceName: 'Changed paper',
        headingPath: 'Changed paper / Results',
        snippet: 'Changed evidence must be skipped.',
        highlightSnippet: 'Changed evidence must be skipped.',
        blockId: 'changed-block',
        strictLocate: true,
      },
      source_fingerprint: 'sha256:new-source',
      source_changed: true,
    })
  })

  const navigation = page.goto('/notes?note=note-baseline')
  await baselineStarted
  finishBaselineResponse?.()
  await navigation

  const editor = page.locator('.kb-notes-editor-body textarea')
  await expect(editor).toHaveValue('A note body that must not be rewritten.')
  await expect(page.getByTestId('workspace-note-evidence-review-note-failed-check')).toHaveCount(0)
  await expect(page.getByTestId('workspace-note-evidence-review-count-note-needs-decision')).toHaveText('需判断 1')
  await expect(page.getByTestId('research-notes-evidence-filter')).toContainText('需处理（1）')
  await expect(page.getByTestId('workspace-note-evidence-review-note-baseline')).toHaveCount(0)
  await expect(page.getByTestId('research-note-evidence-baseline')).toHaveCount(0)
  await page.getByTestId('research-notes-evidence-filter').getByText('需处理（1）', { exact: true }).click()
  await expect(page.getByTestId('workspace-note-note-baseline')).toHaveCount(0)
  await expect(page.getByTestId('workspace-note-note-failed-check')).toHaveCount(0)
  await expect(page.getByTestId('workspace-note-note-needs-decision')).toBeVisible()
  await expect(page.getByTestId('workspace-note-note-clean-check')).toHaveCount(0)
  await page.getByTestId('research-notes-evidence-filter').getByText('全部证据状态', { exact: true }).click()
  await page.getByTestId('workspace-note-note-clean-check').click()
  await expect(page.locator('.kb-notes-title-input')).toHaveValue('Evidence already checked')
  await page.getByTestId('workspace-note-note-needs-decision').click()
  await expect(page.locator('.kb-notes-title-input')).toHaveValue('Ambiguous source evidence')
  await expect(page.getByTestId('research-note-source-0')).toBeFocused()
  await page.getByTestId('workspace-note-note-baseline').click()
  await expect(page.locator('.kb-notes-title-input')).toHaveValue('Legacy evidence baseline')
  expect(baselineBody).toEqual({ expected_revision: 1 })
  expect(note.content_markdown).toBe('A note body that must not be rewritten.')
  expect(note.source_state.links?.[0]?.block_id).toBe('quote-1')
  expect(note.source_state.links?.[0]?.evidence_quote).toBe(quote)
  expect(note.source_state.links?.[0]?.evidence_context_history?.[0]?.action).toBe('baseline_backfilled')
  expect(note.source_state.links?.[1]?.evidence_baseline_check).toBeUndefined()
  expect(note.source_state.links?.[1]?.evidence_verification?.reason).toBe('source_change_auto_verified')
  expect(note.source_state.links?.[1]?.evidence_auto_resolution_history?.[0]?.action).toBe('source_change_auto_verified')
  expect(maintenanceBodies).toHaveLength(1)
  expect(failedNote.content_markdown).toBe('A separate saved note.')
  expect(failedNote.updated_at).toBe(1_799_999_990)
  expect(failedNote.source_state.links?.[0]?.evidence_baseline_check?.retry_count).toBe(1)
  expect(failedNote.source_state.links?.[0]?.evidence_baseline_check?.retry_after).toBe(1_900_000_000)
  expect(needsDecisionNote.content_markdown).toBe('A saved quote with multiple possible locations.')
  expect(needsDecisionNote.updated_at).toBe(1_799_999_985)
  expect(needsDecisionNote.source_state.links?.[0]?.evidence_baseline_check?.maintenance_version).toBe(2)
  expect(evidenceContextBodies).toHaveLength(0)

  await page.reload()
  await expect(page.locator('.kb-notes-editor-body textarea')).toHaveValue('A note body that must not be rewritten.')
  await expect(page.getByTestId('workspace-note-evidence-review-note-baseline')).toHaveCount(0)
  await expect(page.getByTestId('research-notes-evidence-filter')).toContainText('需处理（1）')
  await expect(page.getByTestId('research-note-evidence-baseline')).toHaveCount(0)
  expect(maintenanceBodies).toHaveLength(1)
})
