import { readFile } from 'node:fs/promises'
import { expect, test, type Page, type Route } from '@playwright/test'
import {
  installAppShellMocks,
  installEmptyCitationShelfMock,
  installIdleReferenceMocks,
} from './mockAppShell'

const CONV_ID = 'public-surface-isolation'
const ABSOLUTE_SOURCE_PATH = 'F:\\private\\research\\Internal Quality Paper.en.md'
const ABSOLUTE_LIBRARY_MATCH_PATH = '/srv/private/research/Matched Internal Quality Paper.en.md'
const ABSOLUTE_REFS_FALLBACK_PATH = '/srv/private/research/Refs Unix Fallback.en.md'
const REFS_FALLBACK_BASENAME = 'Refs Unix Fallback.en.md'
const SOURCE_NAME = 'Internal Quality Paper.pdf'
const LIBRARY_MATCH_BASENAME = 'Matched Internal Quality Paper.en.md'
const PATH_SOURCE_NAME = '/home/private/research/Path Named Source.en.md'
const PATH_SOURCE_NAME_BASENAME = 'Path Named Source.en.md'
const EMPTY_NAME_SOURCE_PATH = 'C:\\private\\research\\Empty Name Source.en.md'
const EMPTY_NAME_SOURCE_BASENAME = 'Empty Name Source.en.md'
const WEAK_PATH_SOURCE_NAME = '/home/private/research/Weak Path Source.en.md'
const WEAK_PATH_SOURCE_PATH = '/mnt/private/research/weak-path-source.en.md'
const WEAK_PATH_SOURCE_BASENAME = 'Weak Path Source.en.md'
const STORED_CARD_VIEW_PATH = '/srv/private/research/Stored Card View Title.pdf'
const STORED_CARD_VIEW_BASENAME = 'Stored Card View Title.pdf'
const STORED_CARD_TITLE_PATH = 'C:\\private\\research\\Stored Card Title.pdf'
const STORED_CARD_TITLE_BASENAME = 'Stored Card Title.pdf'
const INTERNAL_TRACE_OBSERVATION = 'internal trace observation: quality gate repair executed'
const PUBLIC_SURFACE_BUILD = process.env.PW_PUBLIC_SURFACE === '1'

test.skip(!PUBLIC_SURFACE_BUILD, 'run through npm run test:e2e:public-surface')

const conversation = {
  id: CONV_ID,
  title: 'Public surface isolation fixture',
  created_at: 1_780_000_000,
  updated_at: 1_780_000_010,
  project_id: null,
  mode: 'normal',
  bound_source_path: '',
  bound_source_name: '',
  bound_source_ready: false,
}

const citationMeta = {
  num: 1,
  anchor: 'public-source-1',
  source_name: SOURCE_NAME,
  source_path: ABSOLUTE_SOURCE_PATH,
  title: 'Trustworthy evidence presentation for ordinary users',
  authors: 'Public A, Reader B',
  year: '2026',
  venue: 'Journal of Public Research UX',
  doi: '10.1234/public.surface.2026',
  doi_url: 'https://doi.org/10.1234/public.surface.2026',
  citation_count: 42,
  journal_if: '8.2',
  journal_quartile: 'Q1',
  bibliometrics_checked: true,
  metadata_quality: { status: 'ready', ok: true, score: 94, issues: [] },
  metadata_export_acceptance: { export_ready: true, missing_fields: [], issue_codes: [] },
  summary_line: 'The article explains how evidence can stay useful without exposing internal evaluation machinery.',
  summary_source: 'abstract',
  summary_provider: 'crossref',
  summary_quality: {
    ok: true,
    status: 'grounded',
    score: 94,
    source: 'abstract',
    provider: 'crossref',
    identity_title: 'Trustworthy evidence presentation for ordinary users',
    identity_doi: '10.1234/public.surface.2026',
  },
  trace_conv_id: 'internal-trace-conversation-id',
  trace_assistant_msg_id: 102,
  trace_assistant_order: 1,
  trace_user_msg_id: 101,
  source_open_status: 'verified',
  source_open_precision: 'exact_anchor',
  source_open_reason: 'internal exact-anchor quality reason',
  library_match_status: 'in_library',
  library_match_method: 'title',
  library_match_path: ABSOLUTE_LIBRARY_MATCH_PATH,
  shelf_item_kind: 'citation',
  shelf_origin: 'answer_citation',
  answer_claim: 'The public answer should show the conclusion and its source.',
  why_line: 'Collected because it directly supports the user-facing conclusion.',
  heading_path: 'Results / Evidence presentation',
  location_label: 'Results / Evidence presentation · p. 3',
  page_start: 3,
  page_end: 3,
  block_id: 'blk-public-source-1',
  anchor_id: 'p-public-source-1',
  anchor_kind: 'paragraph',
}

const pathNamedCitationMeta = {
  ...citationMeta,
  num: 2,
  anchor: 'public-source-path-name',
  source_name: PATH_SOURCE_NAME,
  source_path: '/mnt/private/research/path-named-source.en.md',
  title: 'Source labels remain safe when the source name is a path',
  doi: '10.1234/public.surface.path-name',
  doi_url: 'https://doi.org/10.1234/public.surface.path-name',
  summary_line: '',
  summary_source: '',
  summary_quality: {},
  library_match_status: 'unknown',
  library_match_path: '',
  card_view: {
    version: 1,
    route: 'system_a',
    kind: 'answer_evidence',
    header: { title: STORED_CARD_VIEW_PATH, subtitle: 'Results / Path-safe title' },
    sections: [],
  },
}

const emptyNamedCitationMeta = {
  ...citationMeta,
  num: 3,
  anchor: 'public-source-empty-name',
  source_name: '',
  source_path: EMPTY_NAME_SOURCE_PATH,
  title: 'Source grouping remains safe without a source name',
  doi: '10.1234/public.surface.empty-name',
  doi_url: 'https://doi.org/10.1234/public.surface.empty-name',
  summary_line: '',
  summary_source: '',
  summary_quality: {},
  library_match_status: 'unknown',
  library_match_path: '',
  card_title: STORED_CARD_TITLE_PATH,
}

const weakPathCitationMeta = {
  ...citationMeta,
  num: 4,
  anchor: 'public-source-weak-path',
  source_name: WEAK_PATH_SOURCE_NAME,
  source_path: WEAK_PATH_SOURCE_PATH,
  title: '',
  authors: '',
  year: '',
  venue: '',
  doi: '',
  doi_url: '',
  summary_line: '',
  summary_source: '',
  summary_quality: {},
  library_match_status: 'unknown',
  library_match_path: '',
}

const messages = [
  {
    id: 101,
    role: 'user',
    content: 'How should evidence be presented to an ordinary user?',
    created_at: 1_780_000_001,
  },
  {
    id: 102,
    role: 'assistant',
    refs_user_msg_id: 101,
    content: 'Show the conclusion, a useful source explanation, and a direct path to the evidence [1] [2] [3].',
    rendered_body: 'Show the conclusion, a useful source explanation, and a direct path to the evidence [1] [2] [3].',
    copy_text: 'Show the conclusion, a useful source explanation, and a direct path to the evidence.',
    copy_markdown: 'Show the conclusion, a useful source explanation, and a direct path to the evidence.',
    cite_details: [citationMeta, pathNamedCitationMeta, emptyNamedCitationMeta],
    created_at: 1_780_000_002,
    meta: {
      agent_trace: {
        mode: 'research_agent',
        question_type: 'reference_followup',
        status: 'done',
        context: {
          query_scope: 'library',
          requested_query_scope: 'current_paper',
          selected_source_count: 2,
        },
        summary: {
          question_type: 'reference_followup',
          status: 'done',
          research_run_status: 'verified',
          source_policy: 'local_only',
          subtask_count: 3,
          evidence_matrix_rows: 1,
          query_scope: 'library',
          requested_query_scope: 'current_paper',
          selected_source_count: 2,
          total_claims: 2,
          supported_claims: 1,
          unsupported_claims: 1,
          evidence_status: 'needs_review',
          evidence_hit_count: 1,
          evidence_status_reasons: ['unsupported_claims'],
          quality_gate_status: 'repaired',
          quality_gate_reasons: ['internal_quality_gate_reason'],
          quality_gate_warnings: ['internal_quality_gate_warning'],
          has_errors: true,
        },
        steps: [
          {
            tool: 'verify_answer_citations',
            status: 'done',
            observation: INTERNAL_TRACE_OBSERVATION,
            elapsed_ms: 17,
            output: { internal_score: 94 },
          },
        ],
        verification: {
          total_claims: 2,
          supported_claims: 1,
          unsupported_claims: 1,
          evidence_status: 'needs_review',
          evidence_hit_count: 1,
          claims: [
            {
              index: 2,
              claim_text: 'One public caveat remains useful to the reader.',
              supported: false,
              unsupported_reason: 'citation_evidence_mismatch',
              matched_evidence_count: 0,
            },
          ],
        },
        research_run: {
          run_id: 'internal-public-surface-run',
          status: 'verified',
          source_policy: 'local_only',
          query_scope: 'library',
          question: 'How should evidence be presented?',
          subtasks: [
            { goal: 'retrieve', tool: 'retrieve_evidence', status: 'done' },
            { goal: 'verify', tool: 'verify_answer_citations', status: 'done' },
            { goal: 'report', tool: 'compose_answer', status: 'done' },
          ],
          evidence_matrix: [
            {
              paper: SOURCE_NAME,
              source_name: SOURCE_NAME,
              source_path: ABSOLUTE_SOURCE_PATH,
              method: 'Separate useful evidence from internal evaluation details.',
              key_result: 'The ordinary-user surface remains concise.',
              limitation: 'One claim still needs a direct source.',
              evidence_quote: 'Evidence remains useful without exposing internal evaluation machinery.',
              citation: '[1]',
              heading_path: 'Results / Evidence presentation',
              support_status: 'partial',
            },
          ],
        },
      },
    },
  },
]

const refsPayload = {
  '101': {
    display_state: 'ready',
    payload_mode: 'full',
    render_status: 'full',
    pending: false,
    hits: [
      {
        meta: {
          ref_pack_state: 'ready',
          source_path: ABSOLUTE_SOURCE_PATH,
        },
        ui_meta: {
          source_path: ABSOLUTE_SOURCE_PATH,
          display_name: SOURCE_NAME,
          heading_path: 'Results / Evidence presentation',
          score: 9.2,
          score_pending: false,
          polish_status: 'full',
          polish_detail: 'summary:llm_grounded->full;why:llm_grounded->full',
          summary_line: 'The reference provides the evidence used in the public answer.',
          why_line: 'It directly supports the user-facing conclusion.',
          can_open: true,
          citation_meta: citationMeta,
          reader_open: {
            sourcePath: ABSOLUTE_SOURCE_PATH,
            sourceName: SOURCE_NAME,
            headingPath: 'Results / Evidence presentation',
            snippet: 'Evidence remains useful without exposing internal evaluation machinery.',
            highlightSnippet: 'Evidence remains useful without exposing internal evaluation machinery.',
            strictLocate: true,
          },
        },
      },
      {
        meta: {
          ref_pack_state: 'ready',
          source_path: ABSOLUTE_REFS_FALLBACK_PATH,
        },
        ui_meta: {
          source_path: ABSOLUTE_REFS_FALLBACK_PATH,
          heading_path: 'Results / Unix fallback',
          summary_line: 'The fallback source label must remain a basename.',
          why_line: 'It verifies path-safe reference rendering.',
          can_open: false,
        },
      },
    ],
  },
}

function fulfillJson(route: Route, body: unknown) {
  return route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify(body),
  })
}

async function injectAllDebugFlags(page: Page) {
  await page.addInitScript(() => {
    window.localStorage.setItem('kb:chat-perf-panel', '1')
    window.sessionStorage.setItem('kb.internal.debug', '1')
    window.sessionStorage.setItem('kb.internal.showSettingsDiagnostics', '1')
    window.sessionStorage.setItem('kb.internal.showQualityDiagnostics', '1')
  })
}

async function installSettingsMock(page: Page) {
  await page.route('**/api/settings', async (route) => {
    if (route.request().method() === 'PATCH') {
      await fulfillJson(route, { ok: true })
      return
    }
    await fulfillJson(route, {
      model: 'test-model',
      base_url: '',
      has_api_key: true,
      db_dir: 'F:\\private\\research\\db',
      prefs: {
        ui_locale: 'en',
        theme: 'light',
        top_k: 6,
        temperature: 0.2,
        max_tokens: 1216,
        deep_read: false,
      },
    })
  })
}

async function installPublicChatBackend(page: Page) {
  let sourceQualityCalls = 0
  const citationMetaForRequest = (route: Route) => {
    const payload = route.request().postDataJSON() as Record<string, unknown> | null
    const nestedMeta = payload?.meta && typeof payload.meta === 'object'
      ? payload.meta as Record<string, unknown>
      : {}
    const sourcePath = String(
      payload?.source_path
      || payload?.sourcePath
      || nestedMeta.source_path
      || nestedMeta.sourcePath
      || '',
    ).trim()
    if (sourcePath === pathNamedCitationMeta.source_path) return pathNamedCitationMeta
    if (sourcePath === emptyNamedCitationMeta.source_path) return emptyNamedCitationMeta
    if (sourcePath === weakPathCitationMeta.source_path) return weakPathCitationMeta
    return citationMeta
  }
  await installAppShellMocks(page, { rootConversations: [conversation] })
  await installSettingsMock(page)
  await installIdleReferenceMocks(page)
  await installEmptyCitationShelfMock(page, {
    scopeId: '__default__',
    projectId: null,
    initialItems: [citationMeta, pathNamedCitationMeta, emptyNamedCitationMeta, weakPathCitationMeta],
    initialOpen: true,
  })

  await page.route(/\/api\/conversations(?:\?.*)?$/, async (route) => {
    if (route.request().method() === 'POST') {
      await fulfillJson(route, { id: CONV_ID })
      return
    }
    await fulfillJson(route, [conversation])
  })
  await page.route(`**/api/conversations/${CONV_ID}`, async (route) => {
    await fulfillJson(route, conversation)
  })
  await page.route(`**/api/conversations/${CONV_ID}/research-state`, async (route) => {
    await fulfillJson(route, { ok: true, state: null })
  })
  await page.route(new RegExp(`/api/conversations/${CONV_ID}/messages(?:\\?.*)?$`), async (route) => {
    await fulfillJson(route, messages)
  })
  await page.route(`**/api/conversations/${CONV_ID}/messages_page*`, async (route) => {
    await fulfillJson(route, {
      messages,
      has_more_before: false,
      oldest_loaded_id: 101,
      newest_loaded_id: 102,
    })
  })
  await page.route(`**/api/references/conversation/${CONV_ID}`, async (route) => {
    await fulfillJson(route, refsPayload)
  })
  await page.route('**/api/references/citation-meta', async (route) => {
    await fulfillJson(route, citationMetaForRequest(route))
  })
  await page.route('**/api/references/citation-card-polish', async (route) => {
    await fulfillJson(route, citationMetaForRequest(route))
  })
  await page.route('**/api/references/bibliometrics', async (route) => {
    await fulfillJson(route, citationMetaForRequest(route))
  })
  await page.route('**/api/references/reader/doc', async (route) => {
    await fulfillJson(route, {
      source_name: SOURCE_NAME,
      markdown: '# Evidence presentation\n\nEvidence remains useful without exposing internal evaluation machinery.',
      anchors: [],
      blocks: [],
      cite_details: [],
    })
  })
  await page.route('**/api/library/quality/sources', async (route) => {
    sourceQualityCalls += 1
    await fulfillJson(route, {
      ok: true,
      requested: 1,
      review_count: 1,
      items: [
        {
          source_path: ABSOLUTE_SOURCE_PATH,
          source_name: SOURCE_NAME,
          conversion_quality: {
            status: 'warning',
            score: 62,
            summary: 'Internal conversion quality warning',
            has_review_issue: true,
            issues: [{ code: 'internal_quality_issue', label: 'Internal quality issue' }],
          },
        },
      ],
    })
  })
  return { sourceQualityCalls: () => sourceQualityCalls }
}

test.beforeAll(() => {
  if (!PUBLIC_SURFACE_BUILD) return
  const expected = {
    PW_PUBLIC_SURFACE: '1',
    VITE_ENABLE_INTERNAL_DEBUG: '0',
    VITE_ENABLE_INTERNAL_ROUTES: '0',
    VITE_SHOW_USER_QUALITY_DIAGNOSTICS: '0',
    VITE_SHOW_INTERNAL_SETTINGS: '0',
    VITE_ENABLE_EVIDENCE_MATRIX_WORKSPACE: '0',
  }
  for (const [name, value] of Object.entries(expected)) {
    if (process.env[name] !== value) {
      throw new Error(`public-surface.spec.ts must run through npm run test:e2e:public-surface (${name}=${process.env[name] || ''})`)
    }
  }
})

test('ordinary-user chat ignores debug flags and exports no diagnostic surface', async ({ page }) => {
  await page.context().grantPermissions(['clipboard-read', 'clipboard-write'])
  await injectAllDebugFlags(page)
  const backend = await installPublicChatBackend(page)

  await page.goto('/?debug=1&perf=1&kb_debug=1')
  const conversationRow = page.locator('.kb-conv-row', { hasText: conversation.title })
  await expect(conversationRow).toHaveCount(1)
  await conversationRow.click()

  const injectedFlags = await page.evaluate(() => ({
    debugQuery: new URLSearchParams(window.location.search).get('debug'),
    perfQuery: new URLSearchParams(window.location.search).get('perf'),
    kbDebugQuery: new URLSearchParams(window.location.search).get('kb_debug'),
    perfPanel: window.localStorage.getItem('kb:chat-perf-panel'),
    internalDebug: window.sessionStorage.getItem('kb.internal.debug'),
    settingsDiagnostics: window.sessionStorage.getItem('kb.internal.showSettingsDiagnostics'),
    qualityDiagnostics: window.sessionStorage.getItem('kb.internal.showQualityDiagnostics'),
  }))
  expect(injectedFlags).toEqual({
    debugQuery: '1',
    perfQuery: '1',
    kbDebugQuery: '1',
    perfPanel: '1',
    internalDebug: '1',
    settingsDiagnostics: '1',
    qualityDiagnostics: '1',
  })

  await expect(page.getByText('Show the conclusion, a useful source explanation')).toBeVisible()
  await expect(page.getByTestId('chat-perf-panel')).toHaveCount(0)
  await expect(page.locator('body')).not.toContainText(ABSOLUTE_SOURCE_PATH)
  await expect(page.locator('body')).not.toContainText(ABSOLUTE_REFS_FALLBACK_PATH)
  await expect(page.locator('body')).not.toContainText(STORED_CARD_VIEW_PATH)
  await expect(page.locator('body')).not.toContainText(STORED_CARD_TITLE_PATH)
  await expect(page.locator('body')).not.toContainText('Q94')
  expect(backend.sourceQualityCalls()).toBe(0)

  const citeChips = page.locator('.kb-cite-chip')
  await expect(citeChips).toHaveCount(3)
  await citeChips.nth(1).click()
  const citationPopover = page.getByTestId('citation-popover')
  await expect(citationPopover).toBeVisible()
  await expect(citationPopover).toContainText(STORED_CARD_VIEW_BASENAME)
  await expect(citationPopover).not.toContainText(PATH_SOURCE_NAME)
  await expect(citationPopover).not.toContainText(pathNamedCitationMeta.source_path)
  await citationPopover.locator('.kb-cite-pop-close').click()

  const refsPanel = page.locator('.kb-refs-panel')
  await expect(refsPanel).toHaveCount(1)
  const refsHeader = refsPanel.locator('.ant-collapse-header')
  const refsTitle = refsPanel.locator('.kb-ref-title')
  await expect(async () => {
    if (await refsTitle.count() === 0) await refsHeader.click()
    await expect(refsTitle).toContainText([SOURCE_NAME, REFS_FALLBACK_BASENAME])
  }).toPass({ timeout: 8_000 })
  await expect(refsPanel).not.toContainText(ABSOLUTE_REFS_FALLBACK_PATH)
  await expect(refsPanel.locator('.kb-ref-score')).toHaveCount(0)
  await expect(refsPanel.locator('[data-testid^="refs-panel-polish-status-"]')).toHaveCount(0)
  await expect(refsPanel).not.toContainText('Score 9.20')
  await expect(refsPanel).not.toContainText('LLM polished')

  const tracePanel = page.locator('.kb-agent-trace')
  await expect(tracePanel).toHaveCount(1)
  await tracePanel.locator('summary').click()
  await expect(tracePanel).not.toContainText('Answer quality')
  await expect(tracePanel).not.toContainText('Task')
  await expect(tracePanel).not.toContainText('Scope')
  await expect(tracePanel).not.toContainText('Research run')
  await expect(tracePanel).not.toContainText('Source policy')
  await expect(tracePanel).not.toContainText('Diagnostics')
  await expect(tracePanel).not.toContainText('internal_quality_gate_reason')
  await expect(tracePanel).not.toContainText(INTERNAL_TRACE_OBSERVATION)
  await expect(tracePanel.getByTestId('agent-trace-quality-gate')).toHaveCount(0)

  const shelf = page.getByTestId('citation-shelf')
  await expect(shelf).toHaveClass(/is-visible/)
  await expect(shelf.getByTestId('citation-shelf-item')).toHaveCount(4)
  await expect(shelf).not.toContainText(ABSOLUTE_SOURCE_PATH)
  await expect(shelf).not.toContainText(ABSOLUTE_LIBRARY_MATCH_PATH)
  await expect(shelf).not.toContainText(PATH_SOURCE_NAME)
  await expect(shelf).not.toContainText(EMPTY_NAME_SOURCE_PATH)
  await expect(shelf).not.toContainText(WEAK_PATH_SOURCE_NAME)
  await expect(shelf).not.toContainText(WEAK_PATH_SOURCE_PATH)
  await expect(shelf).not.toContainText(STORED_CARD_VIEW_PATH)
  await expect(shelf).not.toContainText(STORED_CARD_TITLE_PATH)
  await expect(shelf.getByTestId('citation-shelf-summary-quality')).toHaveCount(0)

  await shelf.getByTestId('citation-shelf-organize-toggle').click()
  const primaryShelfItem = shelf.getByTestId('citation-shelf-item').first()
  await primaryShelfItem.click()
  await expect(shelf.getByTestId('citation-shelf-trace-row-source')).toHaveAttribute('title', SOURCE_NAME)
  await expect(shelf.getByTestId('citation-shelf-trace-row-fulltext')).toHaveAttribute('title', LIBRARY_MATCH_BASENAME)
  await shelf.locator('.kb-shelf-advanced-toggle').click()
  await shelf.locator('.kb-shelf-filter-segments[role="group"]').first().locator('button').nth(2).click()
  const groupTitles = shelf.locator('.kb-shelf-group-title')
  await expect(groupTitles).toContainText([SOURCE_NAME, PATH_SOURCE_NAME_BASENAME, EMPTY_NAME_SOURCE_BASENAME, WEAK_PATH_SOURCE_BASENAME])
  const groupTitleText = (await groupTitles.allTextContents()).join('\n')
  expect(groupTitleText).not.toContain(PATH_SOURCE_NAME)
  expect(groupTitleText).not.toContain(EMPTY_NAME_SOURCE_PATH)
  expect(groupTitleText).not.toContain(WEAK_PATH_SOURCE_NAME)
  expect(groupTitleText).not.toContain(WEAK_PATH_SOURCE_PATH)
  const pathNamedShelfItem = shelf.getByTestId('citation-shelf-item').filter({ hasText: STORED_CARD_VIEW_BASENAME })
  await pathNamedShelfItem.click()
  await expect(pathNamedShelfItem.getByTestId('citation-shelf-trace-row-source')).toHaveAttribute('title', PATH_SOURCE_NAME_BASENAME)
  const emptyNamedShelfItem = shelf.getByTestId('citation-shelf-item').filter({ hasText: STORED_CARD_TITLE_BASENAME })
  await emptyNamedShelfItem.click()
  await expect(emptyNamedShelfItem.getByTestId('citation-shelf-trace-row-source')).toHaveAttribute('title', EMPTY_NAME_SOURCE_BASENAME)
  const shelfTitleAttributes = await shelf.locator('[title]').evaluateAll((nodes) => (
    nodes.map((node) => node.getAttribute('title') || '')
  ))
  expect(shelfTitleAttributes.join('\n')).not.toContain(ABSOLUTE_SOURCE_PATH)
  expect(shelfTitleAttributes.join('\n')).not.toContain(ABSOLUTE_LIBRARY_MATCH_PATH)
  expect(shelfTitleAttributes.join('\n')).not.toContain(PATH_SOURCE_NAME)
  expect(shelfTitleAttributes.join('\n')).not.toContain(EMPTY_NAME_SOURCE_PATH)
  expect(shelfTitleAttributes.join('\n')).not.toContain(WEAK_PATH_SOURCE_NAME)
  expect(shelfTitleAttributes.join('\n')).not.toContain(WEAK_PATH_SOURCE_PATH)
  const shelfAriaLabels = await shelf.locator('[aria-label]').evaluateAll((nodes) => (
    nodes.map((node) => node.getAttribute('aria-label') || '')
  ))
  expect(shelfAriaLabels.join('\n')).not.toContain(PATH_SOURCE_NAME)
  expect(shelfAriaLabels.join('\n')).not.toContain(EMPTY_NAME_SOURCE_PATH)
  expect(shelfAriaLabels.join('\n')).not.toContain(WEAK_PATH_SOURCE_NAME)
  expect(shelfAriaLabels.join('\n')).not.toContain(WEAK_PATH_SOURCE_PATH)

  await shelf.getByTestId('citation-shelf-export-toggle').click()
  const downloadPromise = page.waitForEvent('download')
  await shelf.getByTestId('citation-shelf-export-main-csv').click()
  const download = await downloadPromise
  const downloadPath = await download.path()
  expect(downloadPath).not.toBeNull()
  if (downloadPath) {
    const csv = await readFile(downloadPath, 'utf8')
    expect(csv).toContain('title,authors,year,venue,doi,source')
    expect(csv).toContain(SOURCE_NAME)
    expect(csv).toContain(PATH_SOURCE_NAME_BASENAME)
    expect(csv).toContain(EMPTY_NAME_SOURCE_BASENAME)
    expect(csv).toContain(WEAK_PATH_SOURCE_BASENAME)
    expect(csv).toContain(STORED_CARD_VIEW_BASENAME)
    expect(csv).toContain(STORED_CARD_TITLE_BASENAME)
    expect(csv).not.toContain(ABSOLUTE_SOURCE_PATH)
    expect(csv).not.toContain(PATH_SOURCE_NAME)
    expect(csv).not.toContain(EMPTY_NAME_SOURCE_PATH)
    expect(csv).not.toContain(WEAK_PATH_SOURCE_NAME)
    expect(csv).not.toContain(WEAK_PATH_SOURCE_PATH)
    expect(csv).not.toContain(STORED_CARD_VIEW_PATH)
    expect(csv).not.toContain(STORED_CARD_TITLE_PATH)
    expect(csv).not.toContain('trace_conversation_id')
    expect(csv).not.toContain('source_open_status')
    expect(csv).not.toContain('source_quality_status')
    expect(csv).not.toContain('summary_quality_score')
    expect(csv).not.toContain('internal-trace-conversation-id')
    expect(csv).not.toContain('internal exact-anchor quality reason')
    expect(csv).not.toContain('Q94')
  }

  const markdownDownloadPromise = page.waitForEvent('download')
  await shelf.getByTestId('citation-shelf-export-main-md').click()
  const markdownPath = await (await markdownDownloadPromise).path()
  expect(markdownPath).not.toBeNull()
  if (markdownPath) {
    const markdown = await readFile(markdownPath, 'utf8')
    expect(markdown).toContain(WEAK_PATH_SOURCE_BASENAME)
    expect(markdown).not.toContain(WEAK_PATH_SOURCE_NAME)
    expect(markdown).not.toContain(WEAK_PATH_SOURCE_PATH)
  }

  await shelf.getByTestId('citation-shelf-export-copy-bibtex').click()
  await expect.poll(async () => page.evaluate(() => navigator.clipboard.readText())).toContain(WEAK_PATH_SOURCE_BASENAME)
  const copiedBibtex = await page.evaluate(() => navigator.clipboard.readText())
  expect(copiedBibtex).not.toContain(WEAK_PATH_SOURCE_NAME)
  expect(copiedBibtex).not.toContain(WEAK_PATH_SOURCE_PATH)

  await citeChips.nth(0).click()
  await citationPopover.locator('.kb-cite-pop-action-primary').click()
  const readerTitle = page.locator('.kb-reader-shell-title')
  await expect(readerTitle).toBeVisible()
  await expect(readerTitle).toHaveAttribute('title', SOURCE_NAME)
  const readerTitleAttributes = await page.locator('.kb-reader-shell [title]').evaluateAll((nodes) => (
    nodes.map((node) => node.getAttribute('title') || '')
  ))
  expect(readerTitleAttributes.join('\n')).not.toContain(ABSOLUTE_SOURCE_PATH)
})

test('ordinary-user build excludes internal regression routes despite debug storage', async ({ page }) => {
  await injectAllDebugFlags(page)
  await installAppShellMocks(page)
  await installSettingsMock(page)
  await installIdleReferenceMocks(page)
  await installEmptyCitationShelfMock(page, {
    scopeId: '__default__',
    projectId: null,
  })

  await page.goto('/__message_list_test__?debug=1&perf=1&kb_debug=1')
  await expect.poll(() => new URL(page.url()).pathname).toBe('/')
  await expect(page.getByTestId('message-list-test-scenario')).toHaveCount(0)
  await expect(page.getByTestId('chat-perf-panel')).toHaveCount(0)
})
