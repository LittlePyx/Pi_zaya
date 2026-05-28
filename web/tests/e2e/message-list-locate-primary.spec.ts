import { readFile } from 'node:fs/promises'
import { expect, test, type Page } from '@playwright/test'
import {
  READER_REGRESSION_SOURCE_PATH,
  readerRegressionDocResponse,
} from '../../src/testing/readerRegressionFixtures'

test.beforeEach(async ({ page }) => {
  await page.route('**/api/references/citation-card-polish', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({}),
    })
  })
})

async function mockReaderDoc(page: Page) {
  await page.route('**/api/references/citation-meta', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({}),
    })
  })
  await page.route('**/api/references/bibliometrics', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({}),
    })
  })
  await page.route('**/api/references/reader/doc', async (route) => {
    const req = route.request()
    const payload = req.postDataJSON() as { source_path?: string } | undefined
    if (String(payload?.source_path || '').trim() !== READER_REGRESSION_SOURCE_PATH) {
      await route.fulfill({
        status: 404,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'unexpected source path' }),
      })
      return
    }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(readerRegressionDocResponse),
    })
  })
}

function shelfMetadataRepairFixture(items: Array<Record<string, unknown>>, ready: boolean) {
  const readyQuality = {
    contract_version: 1,
    ok: true,
    status: 'ready',
    score: 100,
    missing_fields: [],
    issues: [],
    repairable: true,
    retryable: false,
    doi: '10.1109/TASSP.1988.1164940',
  }
  const unresolvedQuality = {
    contract_version: 1,
    ok: false,
    status: 'error',
    score: 44,
    missing_fields: ['doi', 'authors', 'venue'],
    issues: [
      { code: 'missing_doi', label: 'Missing DOI', field: 'doi', severity: 'warning' },
      { code: 'missing_authors', label: 'Missing authors', field: 'authors', severity: 'warning' },
      { code: 'missing_venue', label: 'Missing venue', field: 'venue', severity: 'warning' },
    ],
    repairable: true,
    retryable: true,
  }
  const quality = ready ? readyQuality : unresolvedQuality
  return {
    ok: true,
    requested: items.length,
    ready: ready ? items.length : 0,
    partial: ready ? 0 : items.length,
    retryable: ready ? 0 : items.length,
    failed: 0,
    changed: ready ? items.length : 0,
    persisted: ready ? 1 : 0,
    export_ready: ready ? items.length : 0,
    unresolved: ready ? 0 : items.length,
    impact: {
      requested: items.length,
      ready_before: 0,
      ready_after: ready ? items.length : 0,
      ready_delta: ready ? items.length : 0,
      export_ready_before: 0,
      export_ready_after: ready ? items.length : 0,
      export_ready_delta: ready ? items.length : 0,
      unresolved_after: ready ? 0 : items.length,
      summary_export_ready_after: ready ? items.length : 0,
      changed: ready ? items.length : 0,
      persisted: ready ? 1 : 0,
      before_avg_score: 44,
      after_avg_score: ready ? 100 : 44,
      score_delta: ready ? 56 : 0,
      fixed_issue_codes: ready ? [{ name: 'missing_doi', count: 1 }] : [],
      remaining_issue_codes: ready ? [] : [{ name: 'missing_doi', count: 1 }],
      changed_fields: ready ? [{ name: 'doi', count: 1 }] : [],
      repair_sources: ready ? [{ name: 'reference_index', count: 1 }] : [],
    },
    items: items.map((item, idx) => ({
      key: String(item.key || item.anchor || `repair-${idx}`),
      ok: ready,
      changed: ready,
      changed_fields: ready ? ['title', 'authors', 'venue', 'year', 'doi', 'doi_url'] : [],
      repair_status: ready ? 'repaired' : 'retryable',
      retryable: !ready,
      fixed_issue_codes: ready ? ['missing_doi', 'missing_authors', 'missing_venue'] : [],
      remaining_issue_codes: ready ? [] : ['missing_doi', 'missing_authors', 'missing_venue'],
      repair_sources: ready ? ['reference_index'] : [],
      before: unresolvedQuality,
      after: quality,
      meta: ready
        ? {
            ...item,
            title: 'The missing cone problem and low-pass distortion in optical serial sectioning microscopy',
            authors: 'Macias-Garza F, Bovik A C, Diller K R',
            venue: 'IEEE Transactions on Acoustics, Speech, and Signal Processing',
            year: '1988',
            doi: '10.1109/TASSP.1988.1164940',
            doi_url: 'https://doi.org/10.1109/TASSP.1988.1164940',
            summary_line: 'The abstract explains how missing spatial frequencies create low-pass distortion in optical serial sectioning microscopy.',
            summary_source: 'abstract',
            summary_provider: 'crossref',
            summary_quality: { contract_version: 1, ok: true, status: 'grounded', score: 94, source: 'abstract', provider: 'crossref', issues: [], export_ready: true },
            bibliometrics_checked: true,
            metadata_quality: readyQuality,
            metadata_repair_status: 'repaired',
            metadata_changed_fields: ['title', 'authors', 'venue', 'year', 'doi', 'doi_url'],
            metadata_repair_sources: ['reference_index'],
          }
        : {
            ...item,
            metadata_quality: unresolvedQuality,
            metadata_repair_status: 'retryable',
            metadata_changed_fields: [],
            metadata_repair_sources: [],
          },
      persisted: ready && idx === 0,
      persisted_targets: ready && idx === 0 ? ['reference_index'] : [],
    })),
  }
}

test('structured locate chip prefers the best evidence block over a wrong raw primary block', async ({ page }) => {
  await mockReaderDoc(page)
  await page.goto('/__message_list_test__')

  const chip = page.locator('.kb-prov-locate-chip').first()
  await expect(chip).toBeVisible()
  await expect(chip).toHaveAttribute('data-kb-locate-block-id', 'eq-1')
  await expect(chip).toHaveAttribute('data-kb-locate-anchor-id', 'a-eq-1')

  await chip.click()

  const payload = page.getByTestId('message-list-open-payload')
  await expect(payload).toContainText('"blockId": "eq-1"')
  await expect(payload).toContainText('"anchorId": "a-eq-1"')
  await expect(payload).toContainText('"anchorKind": "equation"')
})

test('required segment without explicit anchor_kind still renders a strict locate chip', async ({ page }) => {
  await mockReaderDoc(page)
  await page.goto('/__message_list_test__?scenario=required-fallback-anchor')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('required-fallback-anchor')
  const chip = page.locator('.kb-prov-locate-chip').first()
  await expect(chip).toBeVisible()
  await expect(chip).toHaveAttribute('data-kb-locate-block-id', 'p-1')
  await expect(chip).toHaveAttribute('data-kb-locate-anchor-id', 'a-p-1')

  await chip.click()

  const payload = page.getByTestId('message-list-open-payload')
  await expect(payload).toContainText('"blockId": "p-1"')
  await expect(payload).toContainText('"anchorId": "a-p-1"')
  await expect(payload).toContainText('"visibleAlternatives"')
  await expect(payload).toContainText('"evidenceAlternatives"')
  await expect(payload).toContainText('"blockId": "p-2"')
})

test('figure panel locate chip remaps to the guide figure anchor', async ({ page }) => {
  await mockReaderDoc(page)
  await page.goto('/__message_list_test__?scenario=guide-figure-remap')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('guide-figure-remap')
  const chip = page.locator('.kb-prov-locate-chip').first()
  await expect(chip).toBeVisible()
  await expect(chip).toHaveAttribute('data-kb-locate-block-id', 'fig-1')
  await expect(chip).toHaveAttribute('data-kb-locate-anchor-id', 'a-fig-1')

  await chip.click()

  const payload = page.getByTestId('message-list-open-payload')
  await expect(payload).toContainText('"blockId": "fig-1"')
  await expect(payload).toContainText('"anchorId": "a-fig-1"')
  await expect(payload).toContainText('"anchorKind": "figure"')
})

test('formula locate chip remaps to the guide equation anchor', async ({ page }) => {
  await mockReaderDoc(page)
  await page.goto('/__message_list_test__?scenario=guide-formula-remap')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('guide-formula-remap')
  const chip = page.locator('.kb-prov-locate-chip').first()
  await expect(chip).toBeVisible()
  await expect(chip).toHaveAttribute('data-kb-locate-block-id', 'eq-1')
  await expect(chip).toHaveAttribute('data-kb-locate-anchor-id', 'a-eq-1')

  await chip.click()

  const payload = page.getByTestId('message-list-open-payload')
  await expect(payload).toContainText('"blockId": "eq-1"')
  await expect(payload).toContainText('"anchorId": "a-eq-1"')
  await expect(payload).toContainText('"anchorKind": "equation"')
})

test('render packet contract can drive body render and strict locate without top-level render fields', async ({ page }) => {
  await mockReaderDoc(page)
  await page.goto('/__message_list_test__?scenario=render-packet-contract')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('render-packet-contract')
  await expect(page.locator('body')).not.toContainText('[[CITE:')
  const citeChip = page.locator('.kb-cite-chip-sysb')
  await expect(citeChip).toHaveCount(1)
  await expect(citeChip.first()).toHaveText('[R1]')
  await citeChip.first().click()
  await expect(page.locator('.kb-cite-pop')).toBeVisible()
  await expect(page.locator('.kb-cite-pop')).toContainText('上游引用')
  await expect(page.getByTestId('citation-popover-explain')).toHaveCount(0)
  await expect(page.getByTestId('citation-popover-flow')).toHaveCount(0)
  await expect(page.getByTestId('citation-popover-system-b-claim')).toHaveCount(0)
  await expect(page.getByTestId('citation-popover-system-b-role')).toHaveCount(0)
  await expect(page.getByTestId('citation-popover-system-b-relation')).toHaveCount(0)
  await expect(page.getByTestId('citation-popover-system-b-citing-source')).toHaveCount(0)
  await expect(page.getByTestId('citation-popover-system-b-cited-source')).toHaveCount(0)
  await expect(page.getByTestId('citation-popover-system-b-takeaway')).toContainText('单次压缩光谱成像')
  await expect(page.getByTestId('citation-popover-system-b-trace')).toHaveCount(0)
  await expect(page.getByTestId('citation-popover-system-b-context')).toHaveCount(0)
  await expect(page.getByTestId('citation-popover-system-b-location')).toContainText('引用所在论文')
  await expect(page.getByTestId('citation-popover-system-b-location')).toContainText('仅定位到当前论文')
  await expect(page.getByTestId('citation-popover-system-b-support')).toContainText('cited prior work')
  await expect(page.locator('.kb-cite-pop')).toContainText('Single-shot compressive spectral imaging')
  await expect(page.locator('.kb-cite-pop')).toContainText('DOI 10.1364/OE.15.014013')
  await expect(page.locator('.kb-cite-pop')).toContainText('被引 123')
  await expect(page.locator('.kb-cite-pop')).toContainText('IF 3.8')
  await expect(page.locator('.kb-cite-pop')).toContainText('JCR Q2')
  await expect(page.locator('.kb-cite-pop')).not.toContainText('Wrong Reference')
  await expect(page.locator('body')).toContainText('RenderPacket notice: this message should show notice without top-level fields.')
  const chip = page.locator('.kb-prov-locate-chip').first()
  await expect(chip).toBeVisible()
  await expect(chip).toHaveAttribute('data-kb-locate-block-id', 'eq-1')
  await expect(chip).toHaveAttribute('data-kb-locate-anchor-id', 'a-eq-1')

  await chip.click()

  const payload = page.getByTestId('message-list-open-payload')
  await expect(payload).toContainText('"blockId": "eq-1"')
  await expect(payload).toContainText('"anchorId": "a-eq-1"')
  await expect(payload).toContainText('"anchorKind": "equation"')

  await page.reload()
  await expect(page.getByTestId('message-list-test-scenario')).toContainText('render-packet-contract')
  await expect(page.locator('body')).not.toContainText('[[CITE:')
  await expect(page.locator('.kb-cite-chip-sysb')).toHaveCount(1)
  await expect(page.locator('.kb-cite-chip-sysb').first()).toHaveText('[R1]')
})

test('system B upstream reference citation is explicitly clickable and opens its card', async ({ page }) => {
  await mockReaderDoc(page)
  await page.goto('/__message_list_test__?scenario=render-packet-contract')

  const systemBChip = page.locator('.kb-cite-chip-sysb').first()
  await expect(systemBChip).toBeVisible()
  await expect(systemBChip).toHaveAttribute('href', /#kb-cite-/)
  await systemBChip.click()

  const popover = page.locator('.kb-cite-pop')
  await expect(popover).toBeVisible()
  await expect(popover).toHaveClass(/kb-cite-pop-system-b/)
  await expect(page.getByTestId('citation-popover-system-b-takeaway')).toContainText('单次压缩光谱成像')
  await expect(page.getByTestId('citation-popover-system-b-context')).toHaveCount(0)
  await expect(page.getByTestId('citation-popover-system-b-location')).toContainText('引用所在论文')
  await expect(page.getByTestId('citation-popover-system-b-location')).toContainText('仅定位到当前论文')
  await expect(page.getByTestId('citation-popover-system-b-reference')).toHaveCount(0)
  await expect(popover).toContainText('DOI 10.1364/OE.15.014013')
  await expect(popover).toContainText('被引 123')
  await expect(popover).toContainText('IF 3.8')
  await expect(popover).toContainText('JCR Q2')
})

test('system B popover can show LLM citation-context summary while hiding answer-context raw text', async ({ page }) => {
  await mockReaderDoc(page)
  await page.route('**/api/references/citation-card-polish', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        citation_card_polish_status: 'full',
        citation_card_polish_source: 'llm',
        citation_card_polish_checked: true,
        citation_card_polish_route: 'system_b',
        citation_card_polish_fields: ['card_context_summary'],
        card_context_summary: '当前论文在介绍 SCI 背景时引用这项工作，用它补足单次压缩光谱成像的上游来源。',
      }),
    })
  })
  await page.goto('/__message_list_test__?scenario=render-packet-contract')

  const systemBChip = page.locator('.kb-cite-chip-sysb').first()
  await expect(systemBChip).toBeVisible()
  await systemBChip.click()

  await expect(page.getByTestId('citation-popover-system-b-context-summary')).toContainText('单次压缩光谱成像的上游来源')
  await expect(page.getByTestId('citation-popover-system-b-context')).toHaveCount(0)
})

test('system B popover suppresses weak raw context without dropping metrics', async ({ page }) => {
  await mockReaderDoc(page)
  await page.goto('/__message_list_test__?scenario=weak-system-b-popover')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('weak-system-b-popover')
  const systemBChip = page.locator('.kb-cite-chip-sysb').first()
  await expect(systemBChip).toBeVisible()
  await systemBChip.click()

  const popover = page.locator('.kb-cite-pop')
  await expect(popover).toBeVisible()
  await expect(popover).toHaveClass(/kb-cite-pop-system-b/)
  await expect(popover).toContainText('上游参考文献')
  await expect(page.getByTestId('citation-popover-system-b-context')).toHaveCount(0)
  await expect(page.getByTestId('citation-popover-system-b-reference')).toContainText('The missing cone problem')
  await expect(popover).not.toContainText('##')
  await expect(popover).not.toContainText('Alessandro Zunino')
  await expect(popover).not.toContainText('No useful citation context')
  await expect(popover).toContainText('DOI 10.1117/12.7976703')
  await expect(popover).toContainText('22')
  await expect(popover).toContainText('IF 1.2')
  await expect(popover).toContainText('JCR Q4')
})

test('render packet hidden locate does not leak a visible locate chip', async ({ page }) => {
  await mockReaderDoc(page)
  await page.goto('/__message_list_test__?scenario=render-packet-hidden-locate')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('render-packet-hidden-locate')
  await expect(page.locator('body')).toContainText('This answer should not expose a hidden locate target.')
  await expect(page.locator('.kb-prov-locate-chip')).toHaveCount(0)
})

test('citation popover ignores stale async metadata after switching citations', async ({ page }) => {
  await page.route('**/api/references/citation-meta', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({}),
    })
  })
  await page.route('**/api/references/bibliometrics', async (route) => {
    const payload = route.request().postDataJSON() as { source_path?: string } | undefined
    const meta = (payload && typeof payload === 'object' && 'meta' in payload)
      ? (payload as { meta?: { sourcePath?: string, source_path?: string } }).meta
      : undefined
    const sourcePath = String(meta?.sourcePath || meta?.source_path || payload?.source_path || '')
    if (sourcePath.includes('slow-a')) {
      await new Promise((resolve) => setTimeout(resolve, 500))
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          title: 'Slow Metadata A',
          doi: '10.0000/slow-a',
          doi_url: 'https://doi.org/10.0000/slow-a',
        }),
      })
      return
    }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        title: 'Fast Metadata B',
        doi: '10.0000/fast-b',
        doi_url: 'https://doi.org/10.0000/fast-b',
      }),
    })
  })

  await page.goto('/__message_list_test__?scenario=citation-hover-race')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('citation-hover-race')
  const citeChips = page.locator('.kb-cite-chip-sysb')
  await expect(citeChips).toHaveCount(2)
  await citeChips.nth(0).click()
  await expect(page.locator('.kb-cite-pop')).toBeVisible()
  await expect(page.locator('.kb-cite-pop')).toHaveClass(/kb-cite-pop-system-b/)
  await citeChips.nth(1).click()
  await expect(page.locator('.kb-cite-pop')).toContainText('Fast Metadata B')
  await page.waitForTimeout(650)
  await expect(page.locator('.kb-cite-pop')).toContainText('Fast Metadata B')
  await expect(page.locator('.kb-cite-pop')).not.toContainText('Slow Metadata A')
})

test('system A citation popover shows source location, evidence quote, and opens strict reader target', async ({ page }) => {
  await mockReaderDoc(page)
  await page.goto('/__message_list_test__?scenario=system-a-citation-popover')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('system-a-citation-popover')
  const citeChip = page.locator('.kb-cite-chip').first()
  await expect(citeChip).toBeVisible()
  await expect(citeChip).toHaveText('1')

  await citeChip.click()
  const popover = page.locator('.kb-cite-pop')
  await expect(popover).toBeVisible()
  await expect(popover).toHaveClass(/kb-cite-pop-system-a/)
  await expect(page.getByTestId('citation-popover-system-a-location')).toContainText('2. Method')
  await expect(page.getByTestId('citation-popover-system-a-location')).not.toContainText('Fixture Paper / 2. Method')
  await expect(page.getByTestId('citation-popover-system-a-location')).toContainText('sentence')
  await expect(page.getByTestId('citation-popover-system-a-evidence')).toContainText('Given a set of input multi-view images')
  await expect(page.getByTestId('citation-popover-system-a-claim')).toHaveCount(0)
  await expect(page.getByTestId('citation-popover-system-a-support')).toHaveCount(0)
  await expect(page.getByTestId('citation-popover-binding-status')).toHaveCount(0)
  await expect(page.getByTestId('citation-popover-system-a-source')).toHaveCount(0)
  await expect(page.getByTestId('citation-popover-system-a-anchor-kind')).toHaveCount(0)
  await expect(popover).toContainText('Fixture Paper')
  await expect(popover).toContainText('位置')
  await expect(popover).toContainText('2. Method')
  await expect(popover).not.toContainText('Fixture Paper / 2. Method')
  await expect(popover).toContainText('原文证据')
  await expect(popover).toContainText('Given a set of input multi-view images')
  await expect(popover).not.toContainText('Method section states the exact mechanism')

  await popover.getByRole('button', { name: '打开答案依据' }).click()
  const payload = page.getByTestId('message-list-open-payload')
  await expect(payload).toContainText('"blockId": "p-method-1"')
  await expect(payload).toContainText('"anchorId": "a-p-method-1"')
  await expect(payload).toContainText('"anchorKind": "sentence"')
  await expect(payload).toContainText('"strictLocate": true')
})

test('citation shelf reflects actual reader locate result after opening source', async ({ page }) => {
  await mockReaderDoc(page)
  await page.route('**/api/library/quality/reader-locate', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ok: true,
        item: {},
        summary: {
          available: true,
          status: 'good',
          summary: { total: 1, exact: 1, block: 0, degraded: 0, failed: 0, repairable: 0, strict_miss: 0, affected_sources: 1 },
          top_failures: [],
          recommended_sources: [],
        },
      }),
    })
  })
  await page.route('**/api/library/source-quality', async (route) => {
    const payload = route.request().postDataJSON() as { sources?: Array<{ source_path?: string, source_name?: string }> }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ok: true,
        requested: payload.sources?.length || 0,
        review_count: 0,
        items: (payload.sources || []).map((source) => ({
          source_path: source.source_path || '',
          source_name: source.source_name || '',
          conversion_quality: { status: 'good', has_review_issue: false, score: 98, issues: [] },
        })),
      }),
    })
  })

  await page.goto('/__message_list_test__?scenario=system-a-citation-popover&reader=1')
  await expect(page.getByTestId('message-list-test-scenario')).toContainText('system-a-citation-popover')

  const citeChip = page.locator('.kb-cite-chip').first()
  await expect(citeChip).toBeVisible()
  await citeChip.click()
  const popover = page.locator('.kb-cite-pop')
  await expect(popover).toBeVisible()
  await popover.locator('.kb-cite-pop-add').click()
  await popover.locator('.kb-cite-pop-open-shelf').last().click()
  await expect(page.getByTestId('citation-shelf')).toHaveClass(/translate-x-0/)
  await expect(page.getByTestId('citation-shelf-source-open-quality')).toHaveClass(/is-ready/)
  await popover.locator('.kb-cite-pop-close').click()

  const locateQualityRequest = page.waitForRequest('**/api/library/quality/reader-locate')
  await page.getByTestId('citation-shelf-open-source').click()
  await expect(page.getByTestId('reader-locate-resolution')).toHaveText(/Exact target|Bound block/)
  const locateQualityPayload = (await locateQualityRequest).postDataJSON() as Record<string, unknown>
  expect(locateQualityPayload.source_path).toBe(READER_REGRESSION_SOURCE_PATH)
  expect(locateQualityPayload.status).toBe('exact')
  expect(locateQualityPayload.precision).toBe('phrase')
  expect(locateQualityPayload.locate_feedback_key).toBeTruthy()
  await expect(page.getByTestId('citation-shelf-source-open-quality')).toHaveClass(/is-ready/)

  await page.getByTestId('citation-shelf-add-visible').click()
  const csvDownloadPromise = page.waitForEvent('download')
  await page.getByTestId('citation-shelf-export-csv').click()
  if (await page.getByTestId('citation-shelf-export-preflight').isVisible().catch(() => false)) {
    await page.getByTestId('citation-shelf-export-preflight-continue').click()
  }
  const csvDownload = await csvDownloadPromise
  const csvPath = await csvDownload.path()
  expect(csvPath, 'CSV export should produce a downloadable file').not.toBeNull()
  if (csvPath) {
    const csv = await readFile(csvPath, 'utf8')
    expect(csv).toContain('source_open_status,source_open_precision,source_open_reason')
    expect(csv).toContain('verified,phrase')
  }
})

test('citation shelf advances source repair run and refreshes repaired locate state', async ({ page }) => {
  await page.route('**/api/references/citation-meta', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({}),
    })
  })
  await page.route('**/api/references/bibliometrics', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({}),
    })
  })
  await page.route('**/api/references/reader/doc', async (route) => {
    await route.fulfill({
      status: 404,
      contentType: 'application/json',
      body: JSON.stringify({ detail: 'reader source block index is stale' }),
    })
  })
  await page.route('**/api/library/quality/reader-locate', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ok: true,
        item: {},
        summary: {
          available: true,
          status: 'warning',
          summary: { total: 1, exact: 0, block: 0, degraded: 0, failed: 1, repairable: 1, strict_miss: 1, affected_sources: 1 },
          top_failures: [],
          recommended_sources: [],
        },
      }),
    })
  })

  let repaired = false
  let repairCalls = 0
  let advanceCalls = 0
  await page.route('**/api/library/quality/sources', async (route) => {
    const payload = route.request().postDataJSON() as { sources?: Array<{ source_path?: string, source_name?: string }> }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ok: true,
        requested: payload.sources?.length || 0,
        review_count: 0,
        items: (payload.sources || []).map((source) => ({
          source_path: source.source_path || '',
          source_name: source.source_name || '',
          pdf_path: '',
          md_path: source.source_path || '',
          md_exists: true,
          conversion_quality: {
            status: 'good',
            label: 'Ready',
            score: 98,
            summary: 'Ready',
            has_review_issue: false,
            issues: [],
            metrics: {},
            conversion_report: repaired
              ? {
                  available: true,
                  latest_repair_attempt: {
                    event: 'reader_locate_reindex_required',
                    status: 'reindex_pending',
                    action: 'reindex',
                    source: 'reader_locate_quality',
                    extra: { reader_locate_problem_count: 1 },
                  },
                }
              : { available: true, latest_repair_attempt: null },
          },
        })),
      }),
    })
  })
  await page.route('**/api/library/quality/repair', async (route) => {
    repairCalls += 1
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ok: true,
        repair_run_id: 'reader-locate-run-1',
        repair_run: {
          run_id: 'reader-locate-run-1',
          status: 'reindex_pending',
          phase: 'reindex_pending',
          created_at: 1,
          updated_at: 1,
          requested: 1,
          enqueued: 0,
          repaired: 0,
          failed: 0,
          skipped_busy: 0,
          needs_reindex: true,
          target_names: ['Fixture Paper'],
          target_sources: [READER_REGRESSION_SOURCE_PATH],
          detail: 'Reader locate source anchors need index refresh.',
        },
        requested: 1,
        enqueued: 0,
        repaired: 0,
        needs_reindex: true,
        skipped_busy: 0,
        failed: 0,
        impact: {
          requested: 1,
          repaired: 0,
          improved: 0,
          enqueued: 0,
          skipped_busy: 0,
          failed: 0,
          needs_reindex: true,
          reader_locate_reindex: 1,
          before_avg_score: 98,
          after_avg_score: 98,
          score_delta: 0,
        },
        items: [{
          source_path: READER_REGRESSION_SOURCE_PATH,
          source_name: 'Fixture Paper',
          pdf_name: '',
          pdf_path: '',
          md_path: READER_REGRESSION_SOURCE_PATH,
          ok: true,
          enqueued: false,
          skipped_busy: false,
          error: '',
          task_id: '',
          reader_locate_reindex_required: true,
        }],
      }),
    })
  })
  await page.route('**/api/library/quality/repair-runs/*/advance', async (route) => {
    advanceCalls += 1
    repaired = true
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ok: true,
        advanced: true,
        waiting: false,
        item: {
          run_id: 'reader-locate-run-1',
          status: 'completed',
          phase: 'verification_passed',
          created_at: 1,
          updated_at: 2,
          requested: 1,
          enqueued: 0,
          repaired: 0,
          failed: 0,
          skipped_busy: 0,
          needs_reindex: true,
          reindexed: true,
          target_names: ['Fixture Paper'],
          target_sources: [READER_REGRESSION_SOURCE_PATH],
          verification: { type: 'reader_locate_repair', status: 'passed', quality_ok: true },
          detail: 'Reader locate verification passed.',
        },
        reindex: { ok: true },
        detail: 'quality repair run advanced',
      }),
    })
  })
  await page.route('**/api/library/reindex', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ ok: true }),
    })
  })

  await page.goto('/__message_list_test__?scenario=system-a-citation-popover&reader=1')
  await expect(page.getByTestId('message-list-test-scenario')).toContainText('system-a-citation-popover')

  const citeChip = page.locator('.kb-cite-chip').first()
  await expect(citeChip).toBeVisible()
  await citeChip.click()
  const popover = page.locator('.kb-cite-pop')
  await expect(popover).toBeVisible()
  await popover.locator('.kb-cite-pop-add').click()
  await popover.locator('.kb-cite-pop-open-shelf').last().click()
  await expect(page.getByTestId('citation-shelf')).toHaveClass(/translate-x-0/)
  await popover.locator('.kb-cite-pop-close').click()

  await page.getByTestId('citation-shelf-open-source').click()
  await expect.poll(() => repairCalls).toBeGreaterThan(0)
  await expect.poll(() => advanceCalls).toBeGreaterThan(0)
  await expect(page.getByTestId('citation-shelf-source-open-quality')).toContainText('已修复，重新校验')
  await expect(page.getByTestId('citation-shelf-source-open-quality')).toHaveClass(/is-partial/)
})

test('citation shelf hydrates persisted quality-center metadata on open', async ({ page }) => {
  await page.route('**/api/references/citation-meta', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({}),
    })
  })
  await page.route('**/api/library/source-quality', async (route) => {
    const payload = route.request().postDataJSON() as { sources?: Array<{ source_path?: string, source_name?: string }> }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ok: true,
        requested: payload.sources?.length || 0,
        review_count: 0,
        items: (payload.sources || []).map((source) => ({
          source_path: source.source_path || '',
          source_name: source.source_name || '',
          conversion_quality: { status: 'good', has_review_issue: false, score: 98, issues: [] },
        })),
      }),
    })
  })
  let bibliometricsRequests = 0
  await page.route('**/api/references/bibliometrics', async (route) => {
    bibliometricsRequests += 1
    const payload = route.request().postDataJSON() as { meta?: Record<string, unknown> } | undefined
    const meta = payload?.meta || {}
    if (bibliometricsRequests === 1) {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({}),
      })
      return
    }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ...meta,
        title: 'The missing cone problem and low-pass distortion in optical serial sectioning microscopy',
        authors: 'Macias-Garza F, Bovik A C, Diller K R',
        venue: 'IEEE Transactions on Acoustics, Speech, and Signal Processing',
        year: '1988',
        doi: '10.1109/TASSP.1988.1164940',
        doi_url: 'https://doi.org/10.1109/TASSP.1988.1164940',
        summary_line: 'The abstract explains how missing spatial frequencies create low-pass distortion in optical serial sectioning microscopy.',
        summary_source: 'abstract',
        summary_provider: 'crossref',
        summary_quality: { contract_version: 1, ok: true, status: 'grounded', score: 94, source: 'abstract', provider: 'crossref', issues: [], export_ready: true },
        bibliometrics_checked: true,
        metadata_quality: { contract_version: 1, ok: true, status: 'ready', score: 100, missing_fields: [], issues: [], repairable: true, retryable: false, doi: '10.1109/TASSP.1988.1164940' },
        metadata_repair_status: 'ready',
        metadata_changed_fields: ['title', 'authors', 'venue', 'year', 'doi', 'doi_url'],
        metadata_repair_sources: ['reference_index'],
      }),
    })
  })
  let repairRequests = 0
  await page.route('**/api/references/shelf/metadata/repair', async (route) => {
    repairRequests += 1
    const payload = route.request().postDataJSON() as { items?: Array<Record<string, unknown>> }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(shelfMetadataRepairFixture(payload.items || [], false)),
    })
  })

  await page.goto('/__message_list_test__?scenario=weak-system-b-popover')
  await expect(page.getByTestId('message-list-test-scenario')).toContainText('weak-system-b-popover')
  const citeChip = page.locator('.kb-cite-chip-sysb').first()
  await expect(citeChip).toBeVisible()
  await citeChip.click()
  const popover = page.locator('.kb-cite-pop')
  await expect(popover).toBeVisible()
  await expect.poll(() => bibliometricsRequests).toBeGreaterThan(0)
  await popover.locator('.kb-cite-pop-add').click()
  await popover.locator('.kb-cite-pop-open-shelf').nth(2).click()
  const shelf = page.getByTestId('citation-shelf')
  await expect(shelf).toHaveClass(/translate-x-0/)
  await expect.poll(() => bibliometricsRequests).toBeGreaterThan(1)
  await expect(page.getByTestId('citation-shelf-item-title')).toContainText('The missing cone problem')
  await expect(shelf.locator('.kb-shelf-doi-link')).toContainText('10.1109/TASSP.1988.1164940')
  await expect(page.getByTestId('citation-shelf-readiness')).toContainText(/1\/1/)
  await expect(shelf.locator('.kb-shelf-quality-chip')).toHaveCount(0)
  await expect(shelf.getByTestId('citation-shelf-repair')).toHaveCount(0)
  await expect(page.getByTestId('citation-shelf-summary-quality')).toContainText(/Q94/)
  expect(repairRequests).toBeLessThanOrEqual(1)
})

test('citation popover upgrades to waited LLM polish when it is ready', async ({ page }) => {
  await mockReaderDoc(page)
  let observedWaitSeconds = 0
  await page.route('**/api/references/citation-card-polish', async (route) => {
    const payload = route.request().postDataJSON() as { wait_s?: number } | undefined
    observedWaitSeconds = Number(payload?.wait_s || 0)
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        citation_card_polish_status: 'full',
        citation_card_polish_source: 'llm',
        citation_card_polish_checked: true,
        card_takeaway: 'LLM polished: this evidence explains how multi-view images become a checkable reconstruction basis.',
        card_view: {
          version: 1,
          route: 'system_a',
          kind: 'answer_evidence',
          header: { kicker: '答案依据', title: 'Fixture Paper', subtitle: '2. Method' },
          sections: [
            {
              id: 'takeaway',
              label: 'LLM section',
              text: 'LLM polished: this evidence explains how multi-view images become a checkable reconstruction basis.',
              kind: 'insight',
              hint: '',
              tone: 'primary',
            },
            { id: 'locator', label: '原文位置', text: '2. Method', kind: 'locator', hint: '', tone: '' },
            {
              id: 'evidence',
              label: '原文证据',
              text: 'Given a set of input multi-view images, the method builds a reconstruction basis.',
              kind: 'quote',
              hint: '',
              tone: '',
            },
          ],
          summary: 'LLM polished: this evidence explains how multi-view images become a checkable reconstruction basis.',
          quality: { label: 'polished', score: 0.9, flags: [], warning: '' },
        },
      }),
    })
  })
  await page.goto('/__message_list_test__?scenario=system-a-citation-popover')

  const citeChip = page.locator('.kb-cite-chip').first()
  await expect(citeChip).toBeVisible()
  await citeChip.click()

  await expect(page.getByTestId('citation-popover-system-a-takeaway')).toContainText('LLM section')
  await expect(page.getByTestId('citation-popover-system-a-takeaway')).toContainText('LLM polished')
  await expect(page.getByTestId('citation-popover-system-a-evidence')).toContainText('Given a set of input multi-view images')
  expect(observedWaitSeconds).toBeGreaterThan(0)
})

test('citation popover and shelf prefer card_view over legacy fallback fields', async ({ page }) => {
  await mockReaderDoc(page)
  await page.goto('/__message_list_test__?scenario=card-view-priority-popover')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('card-view-priority-popover')
  const citeChip = page.locator('.kb-cite-chip').first()
  await expect(citeChip).toBeVisible()
  await citeChip.click()

  const popover = page.locator('.kb-cite-pop')
  await expect(popover).toBeVisible()
  await expect(popover).toContainText('Clean Card Title')
  await expect(popover).toContainText('Clean Method Section')
  await expect(page.getByTestId('citation-popover-system-a-takeaway')).toContainText('Key point')
  await expect(page.getByTestId('citation-popover-system-a-takeaway')).toContainText('Polished card-view takeaway')
  await expect(page.getByTestId('citation-popover-system-a-location')).toContainText('Clean Method Section')
  await expect(page.getByTestId('citation-popover-system-a-evidence')).toContainText('Source evidence')
  await expect(page.getByTestId('citation-popover-system-a-evidence')).toContainText('calibrated measurements')
  await expect(popover).not.toContainText('Legacy fallback takeaway')
  await expect(popover).not.toContainText('Legacy markdown evidence')

  await popover.locator('.kb-cite-pop-add').click()
  await popover.locator('.kb-cite-pop-open-shelf').nth(2).click()
  await expect(page.locator('.kb-shelf-item')).toContainText('Clean Card Title')
  await page.locator('.kb-shelf-item').first().click()
  await expect(page.locator('.kb-shelf-summary')).toContainText('证据卡片')
  await expect(page.locator('.kb-shelf-summary')).toContainText('Polished card-view takeaway')
})

test('citation shelf export auto-completes metadata before download', async ({ page }) => {
  await mockReaderDoc(page)
  await page.route('**/api/library/source-quality', async (route) => {
    const payload = route.request().postDataJSON() as { sources?: Array<{ source_path?: string, source_name?: string }> }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ok: true,
        requested: payload.sources?.length || 0,
        review_count: 0,
        items: (payload.sources || []).map((source) => ({
          source_path: source.source_path || '',
          source_name: source.source_name || '',
          conversion_quality: { status: 'good', has_review_issue: false, score: 96, issues: [] },
        })),
      }),
    })
  })

  let repairCalls = 0
  await page.route('**/api/references/shelf/metadata/repair', async (route) => {
    repairCalls += 1
    const payload = route.request().postDataJSON() as { items?: Array<Record<string, unknown>> }
    const items = Array.isArray(payload.items) ? payload.items : []
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(shelfMetadataRepairFixture(items, repairCalls >= 2)),
    })
  })

  await page.goto('/__message_list_test__?scenario=weak-system-b-popover')
  await expect(page.getByTestId('message-list-test-scenario')).toContainText('weak-system-b-popover')

  const citeChip = page.locator('.kb-cite-chip-sysb').first()
  await expect(citeChip).toBeVisible()
  await citeChip.click()
  const popover = page.locator('.kb-cite-pop')
  await expect(popover).toBeVisible()

  const firstRepairRequest = page.waitForRequest('**/api/references/shelf/metadata/repair')
  await popover.locator('.kb-cite-pop-add').click()
  await popover.locator('.kb-cite-pop-open-shelf').nth(2).click()
  await firstRepairRequest

  const shelf = page.getByTestId('citation-shelf')
  await expect(shelf).toHaveClass(/translate-x-0/)
  await expect(shelf.getByTestId('citation-shelf-repair')).toBeVisible()
  await popover.locator('.kb-cite-pop-close').click()
  await expect(popover).toBeHidden()

  await page.getByTestId('citation-shelf-add-visible').click()
  await expect(page.getByTestId('citation-shelf-batch-count')).toContainText('1')
  await page.getByTestId('citation-shelf-export-bib').click()
  await expect(page.getByTestId('citation-shelf-export-preflight')).toBeVisible()

  const exportRepairRequest = page.waitForRequest('**/api/references/shelf/metadata/repair')
  const bibDownloadPromise = page.waitForEvent('download')
  await page.getByTestId('citation-shelf-export-preflight-continue').click()
  const exportRepairPayload = (await exportRepairRequest).postDataJSON() as { items?: Array<Record<string, unknown>> }
  expect(exportRepairPayload.items?.length || 0).toBeGreaterThan(0)
  const bibDownload = await bibDownloadPromise
  const bibPath = await bibDownload.path()
  expect(bibPath, 'BibTeX export should produce a downloadable file').not.toBeNull()
  if (bibPath) {
    const bib = await readFile(bibPath, 'utf8')
    expect(bib).toContain('title={The missing cone problem and low-pass distortion in optical serial sectioning microscopy}')
    expect(bib).toContain('author={Macias-Garza F and Bovik A C and Diller K R}')
    expect(bib).toContain('journal={IEEE Transactions on Acoustics, Speech, and Signal Processing}')
    expect(bib).toContain('doi={10.1109/tassp.1988.1164940}')
  }
  await expect(shelf.locator('.kb-shelf-doi-link')).toContainText('10.1109/TASSP.1988.1164940')
  await expect(page.getByTestId('citation-shelf-export-preflight')).toHaveCount(0)
})

test('citation shelf consumes metadata repair quality and clears review chips', async ({ page }) => {
  await mockReaderDoc(page)
  await page.route('**/api/library/source-quality', async (route) => {
    const payload = route.request().postDataJSON() as { sources?: Array<{ source_path?: string, source_name?: string }> }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ok: true,
        requested: payload.sources?.length || 0,
        review_count: 0,
        items: (payload.sources || []).map((source) => ({
          source_path: source.source_path || '',
          source_name: source.source_name || '',
          conversion_quality: { status: 'good', has_review_issue: false, score: 96, issues: [] },
        })),
      }),
    })
  })
  let repairRequestCount = 0
  await page.route('**/api/references/shelf/metadata/repair', async (route) => {
    repairRequestCount += 1
    const payload = route.request().postDataJSON() as { items?: Array<Record<string, unknown>> }
    const items = Array.isArray(payload.items) ? payload.items : []
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ok: true,
        requested: items.length,
        ready: items.length,
        partial: 0,
        retryable: 0,
        failed: 0,
        changed: items.length,
        persisted: 1,
        export_ready: items.length,
        unresolved: 0,
        acceptance: {
          contract_version: 1,
          requested: items.length,
          quality_ok: true,
          metadata_ready_before: 0,
          metadata_ready_after: items.length,
          metadata_ready_delta: items.length,
          export_ready_before: 0,
          export_ready_after: items.length,
          export_ready_delta: items.length,
          summary_export_ready_after: items.length,
          retryable: 0,
          failed: 0,
          unresolved_after: 0,
          remaining_fields: [],
          remaining_issue_codes: [],
        },
        verification: {
          type: 'shelf_metadata_repair',
          status: 'passed',
          quality_ok: true,
          target_count: items.length,
          metadata_ready_after: items.length,
          export_ready_after: items.length,
          changed: items.length,
          retryable: 0,
          failed: 0,
          unresolved_after: 0,
          detail: `Metadata export verified for ${items.length}/${items.length} shelf items.`,
        },
        impact: {
          requested: items.length,
          ready_before: 0,
          ready_after: items.length,
          ready_delta: items.length,
          export_ready_before: 0,
          export_ready_after: items.length,
          export_ready_delta: items.length,
          unresolved_after: 0,
          summary_export_ready_after: items.length,
          changed: items.length,
          persisted: 1,
          before_avg_score: 44,
          after_avg_score: 100,
          score_delta: 56,
          fixed_issue_codes: [
            { name: 'weak_or_missing_title', count: 1 },
            { name: 'missing_doi', count: 1 },
          ],
          remaining_issue_codes: [],
          changed_fields: [
            { name: 'title', count: 1 },
            { name: 'doi', count: 1 },
          ],
          repair_sources: [{ name: 'reference_index', count: 1 }],
        },
        items: items.map((item, idx) => ({
          key: String(item.key || item.anchor || `repair-${idx}`),
          ok: true,
          changed: true,
          changed_fields: ['title', 'authors', 'venue', 'year', 'doi', 'doi_url'],
          repair_status: 'repaired',
          retryable: false,
          fixed_issue_codes: ['weak_or_missing_title', 'missing_doi', 'missing_authors', 'missing_venue'],
          remaining_issue_codes: [],
          repair_sources: ['reference_index'],
          before: { contract_version: 1, ok: false, status: 'error', score: 44, missing_fields: ['doi'], issues: [{ code: 'missing_doi', label: 'Missing DOI', field: 'doi', severity: 'warning' }], repairable: true, retryable: true },
          after: { contract_version: 1, ok: true, status: 'ready', score: 100, missing_fields: [], issues: [], repairable: true, retryable: false, doi: '10.1109/TASSP.1988.1164940' },
          meta: {
            ...item,
            title: 'The missing cone problem and low-pass distortion in optical serial sectioning microscopy',
            authors: 'Macias-Garza F, Bovik A C, Diller K R',
            venue: 'IEEE Transactions on Acoustics, Speech, and Signal Processing',
            year: '1988',
            doi: '10.1109/TASSP.1988.1164940',
            doi_url: 'https://doi.org/10.1109/TASSP.1988.1164940',
            summary_line: 'The abstract explains how missing spatial frequencies create low-pass distortion in optical serial sectioning microscopy.',
            summary_source: 'abstract',
            summary_provider: 'crossref',
            summary_quality: { contract_version: 1, ok: true, status: 'grounded', score: 94, source: 'abstract', provider: 'crossref', issues: [], export_ready: true },
            bibliometrics_checked: true,
            metadata_quality: { contract_version: 1, ok: true, status: 'ready', score: 100, missing_fields: [], issues: [], repairable: true, retryable: false, doi: '10.1109/TASSP.1988.1164940' },
            metadata_repair_status: 'repaired',
            metadata_changed_fields: ['title', 'authors', 'venue', 'year', 'doi', 'doi_url'],
            metadata_repair_sources: ['reference_index'],
          },
          persisted: idx === 0,
          persisted_targets: idx === 0 ? ['reference_index'] : [],
        })),
      }),
    })
  })

  await page.goto('/__message_list_test__?scenario=weak-system-b-popover')
  await expect(page.getByTestId('message-list-test-scenario')).toContainText('weak-system-b-popover')

  const citeChip = page.locator('.kb-cite-chip-sysb').first()
  await expect(citeChip).toBeVisible()
  await citeChip.click()
  const popover = page.locator('.kb-cite-pop')
  await expect(popover).toBeVisible()

  const repairRequest = page.waitForRequest('**/api/references/shelf/metadata/repair')
  await popover.locator('.kb-cite-pop-add').click()
  await popover.locator('.kb-cite-pop-open-shelf').nth(2).click()
  await expect.poll(async () => {
    const payload = (await repairRequest).postDataJSON() as { items?: Array<Record<string, unknown>> }
    return payload.items?.length || 0
  }).toBeGreaterThan(0)

  const shelf = page.getByTestId('citation-shelf')
  await expect(shelf).toHaveClass(/translate-x-0/)
  await expect(page.getByTestId('citation-shelf-item-title')).toContainText('The missing cone problem')
  await expect(page.locator('.kb-shelf-doi-link')).toContainText('10.1109/TASSP.1988.1164940')
  await expect(page.getByTestId('citation-shelf-readiness')).toContainText(/1\/1/)
  await expect(page.getByTestId('citation-shelf-repair-impact')).toContainText('doi')
  await expect(shelf.locator('.kb-shelf-quality-chip')).toHaveCount(0)
  await expect(shelf.getByTestId('citation-shelf-repair')).toHaveCount(0)
  await expect(page.getByTestId('citation-shelf-summary-quality')).toContainText(/Q94/)
  await expect(page.getByTestId('citation-shelf-source-open-quality')).toHaveClass(/is-partial/)
  const repairCountAfterAutoFill = repairRequestCount

  await popover.locator('.kb-cite-pop-close').click()
  await page.getByTestId('citation-shelf-open-source').click()
  const openPayload = page.getByTestId('message-list-open-payload')
  await expect(openPayload).toContainText(READER_REGRESSION_SOURCE_PATH)
  await expect(openPayload).toContainText('"strictLocate": false')

  await page.getByTestId('citation-shelf-add-visible').click()
  await expect(page.getByTestId('citation-shelf-batch-count')).toContainText('1')
  await expect(page.getByTestId('citation-shelf-export-preflight')).toHaveCount(0)

  const bibDownloadPromise = page.waitForEvent('download')
  await page.getByTestId('citation-shelf-export-bib').click()
  const bibDownload = await bibDownloadPromise
  expect(bibDownload.suggestedFilename()).toMatch(/^cite_shelf_selected_\d{8}_\d{4}\.bib$/)
  const bibPath = await bibDownload.path()
  expect(bibPath, 'BibTeX export should produce a downloadable file').not.toBeNull()
  if (bibPath) {
    const bib = await readFile(bibPath, 'utf8')
    expect(bib).toContain('title={The missing cone problem and low-pass distortion in optical serial sectioning microscopy}')
    expect(bib).toContain('author={Macias-Garza F and Bovik A C and Diller K R}')
    expect(bib).toContain('journal={IEEE Transactions on Acoustics, Speech, and Signal Processing}')
    expect(bib).toContain('doi={10.1109/tassp.1988.1164940}')
  }

  const risDownloadPromise = page.waitForEvent('download')
  await page.getByTestId('citation-shelf-export-ris').click()
  const risDownload = await risDownloadPromise
  expect(risDownload.suggestedFilename()).toMatch(/^cite_shelf_selected_\d{8}_\d{4}\.ris$/)
  const risPath = await risDownload.path()
  expect(risPath, 'RIS export should produce a downloadable file').not.toBeNull()
  if (risPath) {
    const ris = await readFile(risPath, 'utf8')
    expect(ris).toContain('TI  - The missing cone problem and low-pass distortion in optical serial sectioning microscopy')
    expect(ris).toContain('AU  - Macias-Garza F')
    expect(ris).toContain('AU  - Bovik A C')
    expect(ris).toContain('AU  - Diller K R')
    expect(ris).toContain('JO  - IEEE Transactions on Acoustics, Speech, and Signal Processing')
    expect(ris).toContain('DO  - 10.1109/tassp.1988.1164940')
    expect(ris).toContain('UR  - https://doi.org/10.1109/TASSP.1988.1164940')
  }

  const csvDownloadPromise = page.waitForEvent('download')
  await page.getByTestId('citation-shelf-export-csv').click()
  const csvDownload = await csvDownloadPromise
  expect(csvDownload.suggestedFilename()).toMatch(/^cite_shelf_selected_\d{8}_\d{4}\.csv$/)
  const csvPath = await csvDownload.path()
  expect(csvPath, 'CSV export should produce a downloadable file').not.toBeNull()
  if (csvPath) {
    const csv = await readFile(csvPath, 'utf8')
    expect(csv).toContain('title,authors,year,venue,doi,source,source_quality_status,source_quality_issues')
    expect(csv).toContain('source_open_status,source_open_precision,source_open_reason')
    expect(csv).toContain('summary_source,summary_provider,summary_quality_status,summary_quality_score,summary')
    expect(csv).toContain('The missing cone problem and low-pass distortion in optical serial sectioning microscopy')
    expect(csv).toContain('Macias-Garza F, Bovik A C, Diller K R')
    expect(csv).toContain('IEEE Transactions on Acoustics, Speech, and Signal Processing')
    expect(csv).toContain('10.1109/tassp.1988.1164940')
    expect(csv).toContain('partial,section')
    expect(csv).toContain('abstract,crossref,grounded,94')
  }
  expect(repairRequestCount).toBe(repairCountAfterAutoFill)
})

test('old repeated system A citations use clicked answer line and clean markdown source', async ({ page }) => {
  await mockReaderDoc(page)
  await page.goto('/__message_list_test__?scenario=repeated-system-a-old-packet')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('repeated-system-a-old-packet')
  const citeChips = page.locator('.kb-cite-chip')
  await expect(citeChips).toHaveCount(3)

  await citeChips.nth(0).click()
  let popover = page.locator('.kb-cite-pop')
  await expect(popover).toBeVisible()
  await expect(page.getByTestId('citation-popover-system-a-claim')).toContainText('自适应采样策略')
  await expect(page.getByTestId('citation-popover-system-a-claim')).not.toContainText('实际系统搭建')
  await expect(page.getByTestId('citation-popover-system-a-evidence')).toContainText('Foveated single-pixel imaging')
  await expect(page.getByTestId('citation-popover-system-a-evidence')).not.toContainText('##')

  await page.keyboard.press('Escape')
  await citeChips.nth(1).click()
  popover = page.locator('.kb-cite-pop')
  await expect(popover).toBeVisible()
  await expect(page.getByTestId('citation-popover-system-a-claim')).toContainText('实际系统搭建')
  await expect(page.getByTestId('citation-popover-system-a-claim')).not.toContainText('自适应采样策略')
  await expect(page.getByTestId('citation-popover-system-a-evidence')).not.toContainText('##')
})

test('old low-quality system A card hides label claim and strips paper metadata from evidence', async ({ page }) => {
  await mockReaderDoc(page)
  await page.goto('/__message_list_test__?scenario=low-quality-system-a-old-packet')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('low-quality-system-a-old-packet')
  const citeChip = page.locator('.kb-cite-chip').first()
  await expect(citeChip).toBeVisible()

  await citeChip.click()
  const popover = page.locator('.kb-cite-pop')
  await expect(popover).toBeVisible()
  await expect(page.getByTestId('citation-popover-system-a-claim')).toHaveCount(0)
  await expect(page.getByTestId('citation-popover-system-a-takeaway')).toContainText('单像素成像可以覆盖')
  await expect(page.getByTestId('citation-popover-system-a-takeaway')).not.toContainText('Deep learning review')
  await expect(page.getByTestId('citation-popover-system-a-evidence')).toContainText('Single-pixel imaging technology can capture')
  await expect(page.getByTestId('citation-popover-system-a-evidence')).toContainText('limited image quality')
  await expect(page.getByTestId('citation-popover-system-a-evidence')).not.toContainText('Kai Song')
  await expect(page.getByTestId('citation-popover-system-a-evidence')).not.toContainText('Yaoxing')
  await expect(page.getByTestId('citation-popover-system-a-evidence')).not.toContainText('Advances and Challenges')
  await expect(page.getByTestId('citation-popover-system-a-evidence')).not.toContainText('\\')
})

test('old fragmentary system A card starts from a readable evidence sentence', async ({ page }) => {
  await mockReaderDoc(page)
  await page.goto('/__message_list_test__?scenario=fragmentary-system-a-old-packet')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('fragmentary-system-a-old-packet')
  const citeChip = page.locator('.kb-cite-chip').first()
  await expect(citeChip).toBeVisible()

  await citeChip.click()
  await expect(page.getByTestId('citation-popover-system-a-takeaway')).toContainText('DMD')
  await expect(page.getByTestId('citation-popover-system-a-takeaway')).toContainText('单像素相机')
  const evidence = page.getByTestId('citation-popover-system-a-evidence')
  await expect(evidence).toContainText('A DMD can be used to spatially filter light')
  await expect(evidence).not.toContainText('rson can be described')
  await expect(evidence).not.toContainText('targeted questions')
  await expect(evidence).not.toContainText('Computational imaging configurations')
  await expect(evidence).not.toContainText('a, Single-pixel camera configuration')
})

test('plain numeric citations become clickable from refs hits when cite details are absent', async ({ page }) => {
  await mockReaderDoc(page)
  await page.goto('/__message_list_test__?scenario=plain-citation-refs-fallback')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('plain-citation-refs-fallback')
  const citeChips = page.locator('.kb-cite-chip')
  await expect(citeChips).toHaveCount(2)
  await expect(citeChips.nth(0)).toHaveText('1')
  await expect(citeChips.nth(1)).toHaveText('2')

  await citeChips.nth(0).click()
  await expect(page.locator('.kb-cite-pop')).toHaveClass(/kb-cite-pop-system-a/)
  await expect(page.locator('.kb-cite-pop')).toContainText('Deep Learning SPI Review')
  const firstClaim = page.getByTestId('citation-popover-system-a-claim')
  await expect(firstClaim).toContainText('deep learning SPI improves reconstruction quality')
  await expect(firstClaim).not.toContainText('[1]')
  await expect(firstClaim).not.toContainText('PILN uses')
  expect((await firstClaim.innerText()).length).toBeLessThan(220)
  await expect(page.getByTestId('citation-popover-system-a-evidence')).toContainText('improves reconstruction quality')

  await page.keyboard.press('Escape')
  await expect(page.locator('.kb-cite-pop')).toHaveCount(0)
  await citeChips.nth(1).click()
  await expect(page.locator('.kb-cite-pop')).toContainText('PILN Paper')
  await expect(page.getByTestId('citation-popover-system-a-evidence')).toContainText('part-based image-loop network')
})

test('guide refs remain renderable when only the bound source was filtered out', async ({ page }) => {
  await mockReaderDoc(page)
  await page.goto('/__message_list_test__?scenario=guide-filter-empty-external')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('guide-filter-empty-external')
  await expect(page.locator('.kb-refs-panel')).toBeVisible()
  await page.locator('.kb-refs-panel .ant-collapse-header').click()
  await expect(page.getByTestId('refs-panel-guide-filter-note')).toContainText('已过滤当前阅读指导文献')
  await expect(page.locator('.kb-prov-locate-chip')).toHaveCount(0)
})

test('refs render after the latest user message while assistant is still streaming', async ({ page }) => {
  await mockReaderDoc(page)
  await page.goto('/__message_list_test__?scenario=live-user-pending-refs')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('live-user-pending-refs')
  await expect(page.locator('.kb-refs-panel')).toBeVisible()
  await page.locator('.kb-refs-panel .ant-collapse-header').click()
  await expect(page.getByTestId('refs-panel-pending-note')).toBeVisible()
  await expect(page.locator('.kb-ref-title')).toContainText('Fixture Paper')
  await expect(page.locator('.kb-ref-score')).toContainText('相关分评估中')
})

test('negative evidence-note locate is suppressed instead of showing a misleading jump', async ({ page }) => {
  await mockReaderDoc(page)
  await page.goto('/__message_list_test__?scenario=negative-evidence-locate')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('negative-evidence-locate')
  await expect(page.locator('body')).toContainText('does not mention ADMM')
  await expect(page.locator('.kb-prov-locate-chip')).toHaveCount(0)
})

test('normal multi-doc answer suppresses ambiguous inline locate buttons without structured bindings', async ({ page }) => {
  await mockReaderDoc(page)
  await page.goto('/__message_list_test__?scenario=normal-multi-doc-ambiguous-inline-locate')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('normal-multi-doc-ambiguous-inline-locate')
  await expect(page.locator('body')).toContainText('DOC-1')
  await expect(page.locator('body')).toContainText('DOC-2')
  await expect(page.locator('.kb-md-locate-inline-btn')).toHaveCount(0)
})
