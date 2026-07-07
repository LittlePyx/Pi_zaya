import { expect, test, type Page } from '@playwright/test'
import {
  READER_REGRESSION_SOURCE_PATH,
} from '../../src/testing/readerRegressionFixtures'
import type { AgentTraceHeaderSummaryInput } from '../../src/components/chat/agentTraceHeaderSummary'
import type { AgentTraceMetricCountInput } from '../../src/components/chat/agentTraceMetricCounts'
import type { AgentTraceQualityGateTitleInput } from '../../src/components/chat/agentTraceQualityGate'
import type { AgentTraceScopeSummaryInput } from '../../src/components/chat/agentTraceScopeSummary'
import type { AgentTraceSourceRowsInput } from '../../src/components/chat/agentTraceSourceRows'
import type { AgentTraceSourceStatusInput } from '../../src/components/chat/agentTraceSourceStatus'
import type { AgentSourceSummaryViewModel } from '../../src/components/chat/useAgentTraceViewModel'
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

function agentSummaryViewModel(overrides: Partial<AgentSourceSummaryViewModel> = {}): AgentSourceSummaryViewModel {
  return {
    evidenceLabel: 'Evidence grounded',
    evidenceStatus: 'grounded',
    totalClaims: 2,
    supportedClaims: 1,
    unsupportedClaims: 1,
    qualityGateStatus: 'repaired',
    qualityGateTitle: 'citation repair applied',
    taskLabel: 'Single paper',
    scopeSummary: 'library / 2 selected',
    hasErrors: true,
    researchRunStatus: 'done',
    evidenceMatrixRows: 3,
    sourcePolicy: 'local_only',
    evidenceMatrix: [],
    subtaskCount: 0,
    unsupportedClaimRows: [],
    references: [],
    ...overrides,
  }
}

type SummaryChipSnapshot = {
  id: string
  className?: string
  label: string | number
  testId?: string
  title?: string
  value: string | number
}

async function agentHeaderSummary(page: Page, input: AgentTraceHeaderSummaryInput) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async (summaryInput) => {
    const { buildAgentTraceHeaderSummary } = await import('/src/components/chat/agentTraceHeaderSummary.ts')
    return buildAgentTraceHeaderSummary({}, summaryInput)
  }, input)
}

async function agentMetricCounts(page: Page, input: AgentTraceMetricCountInput) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async (summaryInput) => {
    const { buildAgentTraceMetricCounts } = await import('/src/components/chat/agentTraceMetricCounts.ts')
    return buildAgentTraceMetricCounts(summaryInput)
  }, input)
}

async function agentScopeSummary(page: Page, input: AgentTraceScopeSummaryInput) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async (summaryInput) => {
    const { buildAgentTraceScopeSummary } = await import('/src/components/chat/agentTraceScopeSummary.ts')
    return buildAgentTraceScopeSummary(summaryInput)
  }, input)
}

async function agentSourceStatus(page: Page, input: Omit<AgentTraceSourceStatusInput, 'labels'>) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async (summaryInput) => {
    const { buildAgentTraceSourceStatus } = await import('/src/components/chat/agentTraceSourceStatus.ts')
    return buildAgentTraceSourceStatus({
      ...summaryInput,
      labels: {},
    })
  }, input)
}

async function agentSourceRows(page: Page, input: AgentTraceSourceRowsInput) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async (rowsInput) => {
    const { buildAgentTraceSourceRows } = await import('/src/components/chat/agentTraceSourceRows.ts')
    return buildAgentTraceSourceRows(rowsInput)
  }, input)
}

async function agentQualityGateTitle(page: Page, input: AgentTraceQualityGateTitleInput) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async (summaryInput) => {
    const { buildAgentTraceQualityGateTitle } = await import('/src/components/chat/agentTraceQualityGate.ts')
    return buildAgentTraceQualityGateTitle(summaryInput)
  }, input)
}

async function visibleSummaryChips(page: Page, viewModel: AgentSourceSummaryViewModel): Promise<SummaryChipSnapshot[]> {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async (model) => {
    const { buildAgentTraceSummaryChips } = await import('/src/components/chat/agentTraceSummaryChips.ts')
    return buildAgentTraceSummaryChips({}, model)
      .filter((chip) => chip.visible !== false)
      .map((chip) => ({
        id: chip.id,
        className: chip.className,
        label: typeof chip.label === 'number' || typeof chip.label === 'string' ? chip.label : String(chip.label ?? ''),
        testId: chip.testId,
        title: chip.title,
        value: typeof chip.value === 'number' || typeof chip.value === 'string' ? chip.value : String(chip.value ?? ''),
      }))
  }, viewModel)
}

test.beforeEach(async ({ page }) => {
  await installAppShellMocks(page)
  await installIdleReferenceMocks(page)
  await installEmptyCitationShelfMock(page, {
    scopeId: 'message-list-regression-project',
    projectId: 'message-list-regression-project',
  })
})

test('agent trace metric counts prefer summary values when present', async ({ page }) => {
  const counts = await agentMetricCounts(page, {
    summary: {
      total_claims: 4,
      supported_claims: 3,
      unsupported_claims: 1,
      plan_step_count: 5,
      tool_call_count: 6,
      has_errors: true,
      evidence_matrix_rows: 7,
      subtask_count: 8,
    },
    verification: {
      total_claims: 1,
      supported_claims: 1,
      unsupported_claims: 0,
    },
    planCount: 1,
    stepCount: 2,
    errorCount: 0,
    evidenceMatrixCount: 3,
    researchSubtaskCount: 4,
  })

  expect(counts).toEqual({
    totalClaims: 4,
    supportedClaims: 3,
    unsupportedClaims: 1,
    planStepCount: 5,
    toolCallCount: 6,
    hasErrors: true,
    evidenceMatrixRows: 7,
    subtaskCount: 8,
  })
})

test('agent trace metric counts fall back to source records and lengths', async ({ page }) => {
  const counts = await agentMetricCounts(page, {
    summary: {},
    verification: {
      total_claims: 2,
      supported_claims: 1,
      unsupported_claims: 1,
    },
    planCount: 3,
    stepCount: 4,
    errorCount: 1,
    evidenceMatrixCount: 5,
    researchSubtaskCount: 6,
  })

  expect(counts).toEqual({
    totalClaims: 2,
    supportedClaims: 1,
    unsupportedClaims: 1,
    planStepCount: 3,
    toolCallCount: 4,
    hasErrors: true,
    evidenceMatrixRows: 5,
    subtaskCount: 6,
  })
})

test('agent trace source status prefers summary values and keeps not-applicable tasks general', async ({ page }) => {
  const status = await agentSourceStatus(page, {
    summary: {
      evidence_status: 'not_applicable',
      question_type: 'single_paper_qa',
      quality_gate_status: ' Repaired ',
      quality_gate_reasons: ['citation repair applied'],
      research_run_status: 'done',
      source_policy: 'external_allowed_with_notice',
    },
    verification: {
      evidence_status: 'grounded',
    },
    researchRun: {
      status: 'fallback',
      source_policy: 'local_only',
    },
    traceQuestionType: 'reading_guide',
  })

  expect(status).toEqual({
    evidenceStatus: 'not_applicable',
    evidenceLabel: 'Not from KB',
    qualityGateStatus: 'repaired',
    qualityGateTitle: 'citation repair applied',
    taskLabel: 'General',
    researchRunStatus: 'done',
    sourcePolicy: 'external_allowed_with_notice',
  })
})

test('agent trace source status falls back to verification, research run, and trace task type', async ({ page }) => {
  const status = await agentSourceStatus(page, {
    summary: {},
    verification: {
      evidence_status: 'grounded',
    },
    researchRun: {
      status: 'done',
      source_policy: 'local_only',
    },
    traceQuestionType: 'reading_guide',
  })

  expect(status).toEqual({
    evidenceStatus: 'grounded',
    evidenceLabel: 'Evidence grounded',
    qualityGateStatus: '',
    qualityGateTitle: '',
    taskLabel: 'Reading guide',
    researchRunStatus: 'done',
    sourcePolicy: 'local_only',
  })
})

test('agent trace source rows limit unsupported claims and dedupe step references', async ({ page }) => {
  const rows = await agentSourceRows(page, {
    verification: {
      claims: [
        { text: 'supported claim', supported: true },
        { text: 'unsupported explicit', supported: false },
        { text: 'unsupported reason only', unsupported_reason: 'missing_evidence_overlap' },
        { text: 'unsupported second', supported: false },
        { text: 'unsupported third omitted by limit', supported: false },
      ],
    },
    steps: [
      {
        output: {
          references: [
            { title: 'Ref A', source_path: '/tmp/a.md', ref_num: 1 },
            { title: 'Ref A', source_path: '/tmp/a.md', ref_num: 1 },
            { title: 'Ref B', source_path: '/tmp/b.md', ref_num: 2 },
          ],
        },
      },
      {
        output: {
          references: [
            { title: 'Ref C', source_path: '/tmp/c.md', ref_num: 3 },
            { title: 'Ref D omitted by limit', source_path: '/tmp/d.md', ref_num: 4 },
          ],
        },
      },
    ],
    unsupportedClaimLimit: 3,
    referenceLimit: 3,
  })

  expect(rows.unsupportedClaimRows.map((claim) => claim.text)).toEqual([
    'unsupported explicit',
    'unsupported reason only',
    'unsupported second',
  ])
  expect(rows.references.map((ref) => ref.title)).toEqual(['Ref A', 'Ref B', 'Ref C'])
})

test('agent trace quality gate title combines reasons before warnings', async ({ page }) => {
  const title = await agentQualityGateTitle(page, {
    reasons: ['missing citation repaired', 'claim overlap checked'],
    warnings: ['fallback citation used'],
  })

  expect(title).toBe('missing citation repaired / claim overlap checked / fallback citation used')
})

test('agent trace quality gate title trims and limits noisy lists', async ({ page }) => {
  const title = await agentQualityGateTitle(page, {
    reasons: [
      ' first reason ',
      'second   reason',
      'third reason',
      'fourth reason',
      'fifth reason omitted',
    ],
    warnings: 'not an array',
  })

  expect(title).toBe('first reason / second reason / third reason / fourth reason')
})

test('agent trace scope summary includes requested scope and selected count', async ({ page }) => {
  const summary = await agentScopeSummary(page, {
    queryScope: 'library',
    requestedScope: 'current_paper',
    selectedCount: 2,
    currentSource: 'Current source should not appear outside current_paper scope',
  })

  expect(summary).toBe('library / requested current_paper / 2 selected')
})

test('agent trace scope summary shows current paper source only for current-paper scope', async ({ page }) => {
  const currentPaperSummary = await agentScopeSummary(page, {
    queryScope: 'current_paper',
    requestedScope: 'current_paper',
    selectedCount: 1,
    currentSource: 'Fast hyperspectral single-pixel imaging fixture source',
  })
  const librarySummary = await agentScopeSummary(page, {
    queryScope: 'library',
    requestedScope: 'library',
    selectedCount: 0,
    currentSource: 'Fast hyperspectral single-pixel imaging fixture source',
  })

  expect(currentPaperSummary).toBe('current_paper / 1 selected / Fast hyperspectral single-pixel imaging fixture source')
  expect(librarySummary).toBe('library')
})

test('agent trace header summary keeps evidence status primary and claims as context', async ({ page }) => {
  const summary = await agentHeaderSummary(page, {
    evidenceLabel: 'Evidence grounded',
    totalClaims: 2,
    supportedClaims: 1,
    unsupportedClaims: 1,
    hasErrors: false,
    scopeSummary: 'library / 2 selected',
    taskLabel: 'Single paper',
  })

  expect(summary).toEqual({
    headerEvidence: 'Evidence grounded',
    headerContext: 'Review 1/2',
  })
})

test('agent trace header summary falls back from scope to task without evidence', async ({ page }) => {
  const scopedSummary = await agentHeaderSummary(page, {
    evidenceLabel: '',
    totalClaims: 0,
    supportedClaims: 0,
    unsupportedClaims: 0,
    hasErrors: false,
    scopeSummary: 'library / 2 selected',
    taskLabel: 'Single paper',
  })
  const taskSummary = await agentHeaderSummary(page, {
    evidenceLabel: '',
    totalClaims: 0,
    supportedClaims: 0,
    unsupportedClaims: 0,
    hasErrors: false,
    scopeSummary: '',
    taskLabel: 'Single paper',
  })

  expect(scopedSummary).toEqual({
    headerEvidence: 'Source check available',
    headerContext: 'library / 2 selected',
  })
  expect(taskSummary).toEqual({
    headerEvidence: 'Source check available',
    headerContext: 'Single paper',
  })
})

test('agent trace summary chip builder preserves compact chip order and test ids', async ({ page }) => {
  const chips = await visibleSummaryChips(page, agentSummaryViewModel())

  expect(chips.map((chip) => chip.id)).toEqual([
    'evidence',
    'claims',
    'unsupported-claims',
    'quality-gate',
    'task',
    'scope',
    'run-errors',
    'research-run',
    'source-policy',
  ])
  expect(chips.find((chip) => chip.id === 'evidence')).toMatchObject({
    className: 'kb-agent-trace-evidence-status is-grounded',
    testId: 'agent-trace-evidence-status',
    value: 'Evidence grounded',
  })
  expect(chips.find((chip) => chip.id === 'quality-gate')).toMatchObject({
    className: 'is-warning',
    testId: 'agent-trace-quality-gate',
    title: 'citation repair applied',
    value: 'Repaired',
  })
  expect(chips.find((chip) => chip.id === 'research-run')?.value).toBe('done / 3 rows')
  expect(chips.find((chip) => chip.id === 'source-policy')?.value).toBe('Local KB')
})

test('agent trace summary chip builder hides empty optional chips but keeps task', async ({ page }) => {
  const chips = await visibleSummaryChips(page, agentSummaryViewModel({
    evidenceLabel: '',
    evidenceStatus: '',
    totalClaims: 0,
    supportedClaims: 0,
    unsupportedClaims: 0,
    qualityGateStatus: '',
    qualityGateTitle: '',
    scopeSummary: '',
    hasErrors: false,
    researchRunStatus: '',
    evidenceMatrixRows: 0,
    sourcePolicy: '',
  }))

  expect(chips.map((chip) => chip.id)).toEqual(['task'])
  expect(chips[0]).toMatchObject({
    label: 'Task',
    value: 'Single paper',
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
  await page.getByTestId('assistant-source-notice').click()
  await expect(page.getByTestId('evidence-drawer')).toBeVisible()
  await expect(page.getByTestId('evidence-source-summary')).toContainText(LOCAL_KB_RE)
  await expect(page.getByTestId('evidence-drawer-item')).toContainText('The answer is supported by the local fixture paper')
  await expect(page.getByTestId('evidence-drawer-item')).toContainText('frequency-division multiplexed illumination')
  await expect(page.getByTestId('evidence-drawer-item')).toContainText('Fixture Paper / Method')
  await expect(page.getByTestId('evidence-drawer')).not.toContainText('runtime_check')
  await expect(page.getByTestId('evidence-drawer')).not.toContainText('needs_review_count')
  await page.getByTestId('evidence-open-source').click()
  await expect(page.getByTestId('message-list-open-payload')).toContainText(READER_REGRESSION_SOURCE_PATH)
  await page.locator('.kb-evidence-drawer .ant-drawer-close').click()
  await page.locator('.kb-cite-chip').first().click()
  await expect(page.getByTestId('citation-popover-system-a-evidence')).toContainText('frequency-division multiplexed illumination')
  await expect(page.locator('.kb-cite-pop')).toContainText('Fixture Paper')
  await expect(page.locator('.kb-cite-pop')).toContainText('Method')
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
