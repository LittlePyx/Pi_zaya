import { expect, test } from '@playwright/test'
import {
  READER_REGRESSION_SOURCE_PATH,
} from '../../src/testing/readerRegressionFixtures'
import {
  installAppShellMocks,
  installEmptyCitationShelfMock,
  installIdleReferenceMocks,
} from './mockAppShell'

test.beforeEach(async ({ page }) => {
  await installAppShellMocks(page)
  await installIdleReferenceMocks(page)
  await installEmptyCitationShelfMock(page, {
    scopeId: 'message-list-regression-project',
    projectId: 'message-list-regression-project',
  })
})

test('research agent trace references can open and enter the literature basket', async ({ page }) => {
  await page.goto('/__message_list_test__?scenario=agent-trace-reference-actions')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('agent-trace-reference-actions')
  const traceSummary = page.locator('.kb-agent-trace > summary').first()
  await expect(traceSummary).toContainText('Research Agent Trace')
  await expect(traceSummary).toContainText('Needs review')
  await expect(traceSummary).toContainText('Review 1/2')
  await expect(traceSummary).not.toContainText('reference_followup')
  await expect(traceSummary).not.toContainText('done')
  await page.getByText('Research Agent Trace').click()
  await expect(page.getByTestId('agent-trace-evidence-status')).toContainText('Needs review')
  await expect(page.getByTestId('agent-trace-unsupported-claim')).toContainText('fully solves every downstream limitation')
  await expect(page.getByTestId('agent-trace-unsupported-claim')).toContainText('Citation does not match retrieved evidence')
  await expect(page.getByText('Resolved 1 upstream reference from 1 citing source paper.')).toBeHidden()
  await page.getByText('Execution Details').click()
  await expect(page.getByText('Resolved 1 upstream reference from 1 citing source paper.')).toBeVisible()

  const ref = page.getByTestId('agent-trace-reference').first()
  await expect(ref.getByTestId('agent-trace-ref-title')).toContainText('Fast hyperspectral single-pixel imaging')

  await ref.getByTestId('agent-trace-ref-open').click()
  const payload = page.getByTestId('message-list-open-payload')
  await expect(payload).toContainText(READER_REGRESSION_SOURCE_PATH)
  await expect(payload).toContainText('Fixture Paper / Related Work')
  await expect(payload).toContainText('frequency-division multiplexed illumination')

  await ref.getByTestId('agent-trace-ref-add').click()
  await expect(page.getByTestId('citation-shelf')).toHaveClass(/translate-x-0/)
  await expect(page.getByTestId('citation-shelf-item')).toHaveCount(1)
  await expect(page.getByTestId('citation-shelf-item-title')).toContainText('Fast hyperspectral single-pixel imaging')
})

test('research agent trace can be loaded from stored audit endpoint on demand', async ({ page }) => {
  let requested = false
  await page.route('**/api/messages/9301/agent-trace**', async (route) => {
    requested = true
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        message_id: 9301,
        conv_id: 'message-list-regression:agent-trace-lazy-audit',
        available: true,
        agent_trace: {
          mode: 'research_agent',
          question_type: 'single_paper_qa',
          status: 'done',
          context: { query_scope: 'raw-trace-context' },
          plan: [
            { goal: 'Retrieve compact evidence.', tool: 'retrieve_evidence', status: 'done' },
          ],
          steps: [
            {
              tool: 'retrieve_evidence',
              status: 'done',
              observation: 'Stored audit trace was loaded on demand.',
              output: {},
            },
          ],
          verification: {
            total_claims: 9,
            supported_claims: 3,
            unsupported_claims: 0,
            evidence_status: 'grounded',
            evidence_hit_count: 3,
            evidence_status_reasons: [],
            claims: [],
          },
          errors: [],
        },
        summary: {
          available: true,
          question_type: 'single_paper_qa',
          total_claims: 1,
          supported_claims: 1,
          unsupported_claims: 0,
          evidence_status: 'grounded',
          evidence_hit_count: 3,
          evidence_status_reasons: [],
          tool_call_count: 1,
          has_errors: false,
          query_scope: 'library',
        },
      }),
    })
  })

  await page.goto('/__message_list_test__?scenario=agent-trace-lazy-audit')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('agent-trace-lazy-audit')
  await expect(page.getByText('Stored audit trace was loaded on demand.')).toBeHidden()
  await page.getByText('Research Agent Trace').click()
  await expect(page.getByText('Stored audit trace was loaded on demand.')).toBeHidden()
  await expect(page.locator('.kb-agent-trace > summary').first()).toContainText('Evidence grounded')
  await expect(page.locator('.kb-agent-trace > summary').first()).toContainText('Checked 1/1')
  await expect(page.getByTestId('agent-trace-evidence-status')).toContainText('Evidence grounded')
  await expect(page.locator('.kb-agent-trace-summary strong', { hasText: '1/1' })).toBeVisible()
  await expect(page.locator('.kb-agent-trace-summary strong', { hasText: 'library' })).toBeVisible()
  await expect(page.getByText('3/9')).toHaveCount(0)
  await page.getByText('Execution Details').click()
  await expect(page.getByText('Stored audit trace was loaded on demand.')).toBeVisible()
  expect(requested).toBe(true)
})

test('research agent debug sections stay out of the answer body', async ({ page }) => {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('agent-trace-clean-answer')
  await expect(page.getByText('The answer should stay focused on the evidence-backed conclusion.')).toBeVisible()
  await expect(page.getByText('retrieve_evidence debug detail leaked')).toHaveCount(0)
  await expect(page.getByText('verify_answer_citations debug detail leaked')).toHaveCount(0)
  await expect(page.getByText('supported_claims: 1')).toHaveCount(0)
})

test('streaming research agent partial hides appended trace json', async ({ page }) => {
  await page.goto('/__message_list_test__?scenario=agent-trace-streaming-clean')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('agent-trace-streaming-clean')
  await expect(page.getByText('Streaming answer stays focused on the conclusion.')).toBeVisible()
  await expect(page.getByText('stream trace leaked')).toHaveCount(0)
  await expect(page.getByText('agent_trace')).toHaveCount(0)
})
