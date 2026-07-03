import { expect, test } from '@playwright/test'
import {
  READER_REGRESSION_SOURCE_PATH,
} from '../../src/testing/readerRegressionFixtures'
import {
  installAppShellMocks,
  installEmptyCitationShelfMock,
  installIdleReferenceMocks,
} from './mockAppShell'
const LOCAL_KB_RE = /Local KB|本地文献库|鏈湴鏂囩尞搴?/

const SOURCE_PANEL_RE = /Sources & evidence|依据与来源/
const NEEDS_REVIEW_RE = /Needs review|需要核对/
const REVIEW_FRACTION_RE = /Review 1\/2|需核对 1\/2/
const CHECKED_FRACTION_RE = /Checked 1\/1|已核对 1\/1/
const GROUNDED_RE = /Evidence grounded|文献证据充分/
const ANSWER_QUALITY_RE = /Answer quality|回答质量/
const REPAIRED_RE = /Repaired|已修正/
const EVIDENCE_MAP_RE = /Evidence map|证据地图/
const MISMATCH_RE = /Citation does not match retrieved evidence|引用与检索证据不匹配/
const DIAGNOSTICS_RE = /Diagnostics|诊断信息/

const NOT_FROM_KB_RE = /Not from KB|非本地文献库|闈炴湰鍦版枃鐚簱/
const LOCAL_EXTERNAL_RE = /Local \+ external|文献库 \+ 外部补充|鏂囩尞搴?.*澶栭儴琛ュ厖/

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
  await expect(traceSummary).toContainText(SOURCE_PANEL_RE)
  await expect(traceSummary).toContainText(NEEDS_REVIEW_RE)
  await expect(traceSummary).toContainText(REVIEW_FRACTION_RE)
  await expect(traceSummary).not.toContainText('reference_followup')
  await expect(traceSummary).not.toContainText('done')
  await expect(traceSummary).not.toContainText(ANSWER_QUALITY_RE)
  await expect(page.getByText('Research Agent Trace')).toHaveCount(0)
  await expect(page.getByText('Tool Calls')).toHaveCount(0)
  await page.getByText(SOURCE_PANEL_RE).click()
  await expect(page.getByTestId('agent-trace-evidence-status')).toContainText(NEEDS_REVIEW_RE)
  await expect(page.getByTestId('agent-trace-quality-gate')).toContainText(ANSWER_QUALITY_RE)
  await expect(page.getByTestId('agent-trace-quality-gate')).toContainText(REPAIRED_RE)
  await expect(page.getByTestId('agent-evidence-matrix')).toContainText(EVIDENCE_MAP_RE)
  await expect(page.getByTestId('agent-evidence-matrix-row').first()).toContainText('Fast hyperspectral single-pixel imaging')
  await expect(page.getByTestId('agent-evidence-matrix-row').first()).toContainText('frequency-division multiplexed illumination')
  await expect(page.getByTestId('agent-trace-unsupported-claim')).toContainText('fully solves every downstream limitation')
  await expect(page.getByTestId('agent-trace-unsupported-claim')).toContainText(MISMATCH_RE)
  await expect(page.getByText('Resolved 1 upstream reference from 1 citing source paper.')).toBeHidden()
  await page.getByText(DIAGNOSTICS_RE).click()
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
  await page.getByText(SOURCE_PANEL_RE).click()
  await expect(page.getByText('Stored audit trace was loaded on demand.')).toBeHidden()
  await expect(page.locator('.kb-agent-trace > summary').first()).toContainText(GROUNDED_RE)
  await expect(page.locator('.kb-agent-trace > summary').first()).toContainText(CHECKED_FRACTION_RE)
  await expect(page.getByTestId('agent-trace-evidence-status')).toContainText(GROUNDED_RE)
  await expect(page.locator('.kb-agent-trace-summary strong', { hasText: '1/1' })).toBeVisible()
  await expect(page.locator('.kb-agent-trace-summary strong', { hasText: 'library' })).toBeVisible()
  await expect(page.getByText('3/9')).toHaveCount(0)
  await expect(page.getByText('Research Agent Trace')).toHaveCount(0)
  await expect(page.getByText('Tool Calls')).toHaveCount(0)
  await page.getByText(DIAGNOSTICS_RE).click()
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

test('agent source summary shows a compact local source badge', async ({ page }) => {
  await page.goto('/__message_list_test__?scenario=agent-source-summary-local')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('agent-source-summary-local')
  await expect(page.getByText('The local paper reports that retrieval happens before generation')).toBeVisible()
  await expect(page.getByTestId('assistant-source-notice')).toContainText(LOCAL_KB_RE)
  await expect(page.getByText('agent_trace')).toHaveCount(0)
  await expect(page.getByText('tool calls')).toHaveCount(0)
})

test('answer contract source badge wins over legacy source summary', async ({ page }) => {
  await page.goto('/__message_list_test__?scenario=agent-answer-contract-source-precedence')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('agent-answer-contract-source-precedence')
  await expect(page.getByText('The contract-backed answer keeps the visible response focused on the paper evidence')).toBeVisible()
  await expect(page.getByTestId('assistant-source-notice')).toContainText(LOCAL_KB_RE)
  await expect(page.getByTestId('assistant-source-notice')).not.toContainText(NOT_FROM_KB_RE)
  await expect(page.getByText('Legacy external source')).toHaveCount(0)
  await expect(page.getByText('runtime_check')).toHaveCount(0)
  await expect(page.getByText('needs_review_count')).toHaveCount(0)
})

test('external source notice is compacted outside the answer body', async ({ page }) => {
  await page.goto('/__message_list_test__?scenario=agent-trace-external-notice')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('agent-trace-external-notice')
  await expect(page.getByText('External context can still explain the concept clearly')).toBeVisible()
  await expect(page.getByTestId('assistant-source-notice')).toContainText(NOT_FROM_KB_RE)
  await expect(page.getByText('no matching local knowledge-base evidence was found')).toHaveCount(0)
  await expect(page.locator('.kb-markdown-chat')).not.toContainText('knowledge-base-grounded answer')
  await expect(page.locator('.kb-agent-trace > summary').first()).toContainText(NOT_FROM_KB_RE)
})

test('hybrid source notice is compacted outside the answer body', async ({ page }) => {
  await page.goto('/__message_list_test__?scenario=agent-trace-hybrid-notice')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('agent-trace-hybrid-notice')
  await expect(page.getByText('Local evidence: the paper uses retrieval before generation')).toBeVisible()
  await expect(page.getByText('External context: RAG often uses retrieved context')).toBeVisible()
  await expect(page.getByTestId('assistant-source-notice')).toContainText(LOCAL_EXTERNAL_RE)
  await expect(page.getByText('local citations [n] come from the knowledge base')).toHaveCount(0)
  await expect(page.locator('.kb-markdown-chat')).not.toContainText('uncited background may use external model context')
  await expect(page.locator('.kb-agent-trace > summary').first()).toContainText(NEEDS_REVIEW_RE)
})

test('streaming research agent partial hides appended trace json', async ({ page }) => {
  await page.goto('/__message_list_test__?scenario=agent-trace-streaming-clean')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('agent-trace-streaming-clean')
  await expect(page.getByText('Streaming answer stays focused on the conclusion.')).toBeVisible()
  await expect(page.getByTestId('assistant-source-notice')).toContainText(NOT_FROM_KB_RE)
  await expect(page.getByText('no matching local knowledge-base evidence was found')).toHaveCount(0)
  await expect(page.getByText('stream trace leaked')).toHaveCount(0)
  await expect(page.getByText('agent_trace')).toHaveCount(0)
})
