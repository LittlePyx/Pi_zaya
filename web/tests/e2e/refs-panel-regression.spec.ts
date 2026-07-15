import { expect, test } from '@playwright/test'

test.beforeEach(async ({ page }) => {
  await page.route('**/api/references/citation-meta', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({}),
    })
  })
})

test('refs panel preserves backend reader_open candidates when opening the reader', async ({ page }) => {
  await page.goto('/__refs_panel_test__')

  await expect(page.getByTestId('refs-panel-test-scenario')).toHaveText('rich-reader-open')
  await page.getByRole('button').first().click()
  await expect(page.locator('.kb-ref-title')).toContainText('Fixture Paper')
  await expect(page.locator('.kb-ref-action').first()).toBeVisible()
  await expect(page.getByTestId('refs-panel-locate-overview-0')).toHaveCount(0)
  await expect(page.getByTestId('refs-panel-locate-path-0')).toHaveCount(0)
  await expect(page.getByTestId('refs-panel-locate-snippet-0')).toHaveCount(0)
  await expect(page.getByTestId('refs-panel-summary-basis')).toHaveCount(0)
  await expect(page.getByTestId('refs-panel-why-basis')).toHaveCount(0)

  await page.locator('.kb-ref-action').first().click()

  const payload = page.getByTestId('refs-panel-open-payload')
  await expect(payload).toContainText('"sourcePath": "__reader_regression__/fixture.md"')
  await expect(payload).toContainText('"blockId": "eq-1"')
  await expect(payload).toContainText('"anchorId": "a-eq-1"')
  await expect(payload).toContainText('"anchorKind": "equation"')
  await expect(payload).toContainText('"anchorNumber": 1')
  await expect(payload).toContainText('"strictLocate": true')
  await expect(payload).toContainText('"locateTarget"')
  await expect(payload).toContainText('"relatedBlockIds"')
  await expect(payload).toContainText('"alternatives"')
  await expect(payload).toContainText('"visibleAlternatives"')
  await expect(payload).toContainText('"evidenceAlternatives"')
  await expect(payload).toContainText('"initialAltIndex": 0')
  await expect(payload).toContainText('2.2 Optimization')
  await expect(payload).toContainText('Experimental analysis reuses the same rendering loss')
})

test('refs panel explains when guide mode filtered the current paper and no external hit remains', async ({ page }) => {
  await page.goto('/__refs_panel_test__?scenario=guide-filter-note')

  await expect(page.getByTestId('refs-panel-test-scenario')).toHaveText('guide-filter-note')
  await page.getByRole('button').first().click()
  await expect(page.getByTestId('refs-panel-guide-filter-note')).toBeVisible()
})

test('refs panel renders provisional cards while refs enrichment is pending', async ({ page }) => {
  await page.goto('/__refs_panel_test__?scenario=pending-with-hits')

  await expect(page.getByTestId('refs-panel-test-scenario')).toHaveText('pending-with-hits')
  await page.getByRole('button').first().click()
  await expect(page.getByTestId('refs-panel-pending-note')).toBeVisible()
  await expect(page.locator('.kb-ref-title')).toContainText('Fixture Paper')
  await expect(page.locator('.kb-ref-score')).toHaveCount(0)
  await page.locator('.kb-ref-action').first().click()
  await expect(page.getByTestId('refs-panel-open-payload')).toContainText('"strictLocate": false')
})

test('refs panel renders synthetic research basket evidence as non-openable context', async ({ page }) => {
  await page.goto('/__refs_panel_test__?scenario=research-basket-synthetic')

  await expect(page.getByTestId('refs-panel-test-scenario')).toHaveText('research-basket-synthetic')
  await page.getByRole('button').first().click()
  await expect(page.locator('.kb-ref-title')).toContainText('Research basket: A hard to find preprint')
  await expect(page.locator('.kb-ref-card').first()).toContainText('Research basket')
  await expect(page.locator('.kb-ref-card').first()).toContainText('Selected Context')
  await expect(page.locator('.kb-ref-card').first()).toContainText('10.1234/example.1')
  await expect(page.locator('.kb-ref-score')).toHaveCount(0)

  const actions = page.locator('.kb-ref-action')
  await expect(actions).toHaveCount(4)
  await expect(actions.nth(0)).toBeDisabled()
  await expect(actions.nth(1)).toBeDisabled()
  await expect(actions.nth(2)).toBeDisabled()
  await expect(actions.nth(3)).toBeDisabled()
  await expect(page.getByTestId('refs-panel-open-payload')).toContainText('(empty)')
})

test('refs panel hides reference scoring and polish diagnostics from ordinary users', async ({ page }) => {
  await page.goto('/__refs_panel_test__?scenario=polish-status')

  await expect(page.getByTestId('refs-panel-test-scenario')).toHaveText('polish-status')
  await page.getByRole('button').first().click()
  await expect(page.locator('.kb-ref-score')).toHaveCount(0)
  await expect(page.locator('[data-testid^="refs-panel-polish-status-"]')).toHaveCount(0)
  await expect(page.locator('.kb-refs-panel')).not.toContainText('LLM polished')
  await expect(page.locator('.kb-refs-panel')).not.toContainText('LLM 润色')
})

test('refs panel keeps scoring and polish diagnostics behind the internal debug switch', async ({ page }) => {
  await page.goto('/__refs_panel_test__?scenario=polish-status&debug=1')

  await expect(page.getByTestId('refs-panel-test-scenario')).toHaveText('polish-status')
  await page.getByRole('button').first().click()
  await expect(page.getByTestId('refs-panel-polish-status-0')).toHaveAttribute('data-status', 'full')
  await expect(page.getByTestId('refs-panel-polish-status-0')).toContainText('LLM')
  await expect(page.getByTestId('refs-panel-polish-status-1')).toHaveAttribute('data-status', 'heuristic')
})

test('refs panel keeps relevance scores behind the internal debug switch', async ({ page }) => {
  await page.goto('/__refs_panel_test__?scenario=research-basket-synthetic&debug=1')

  await page.getByRole('button').first().click()
  await expect(page.locator('.kb-ref-score')).toContainText(/Score 9\.20|相关分 9\.20/)
})

test('refs panel prefers the card_view contract over legacy card fields', async ({ page }) => {
  await page.goto('/__refs_panel_test__?scenario=card-view-contract')

  await expect(page.getByTestId('refs-panel-test-scenario')).toHaveText('card-view-contract')
  await page.getByRole('button').first().click()
  await expect(page.locator('.kb-ref-card').first()).toContainText('Guide')
  await expect(page.locator('.kb-ref-card').first()).toContainText('What this section gives you')
  await expect(page.locator('.kb-ref-card').first()).toContainText('This section explains the method at the level a first reading needs.')
  await expect(page.locator('.kb-ref-card').nth(1)).toContainText('Relevance')
  await expect(page.locator('.kb-ref-card').nth(1)).toContainText('Why it matches your question')
  await expect(page.locator('.kb-ref-card').nth(1)).toContainText('It is a good first stop because it connects the paper title to the concrete method steps.')
  await expect(page.getByText('Old fallback summary should not be rendered when card_view is present.')).toHaveCount(0)
  await expect(page.getByText('Old fallback reason should not be rendered when card_view is present.')).toHaveCount(0)
})

test('refs panel can render a section-level strict locate card directly in the page', async ({ page }) => {
  await page.goto('/__refs_panel_test__?scenario=section-target')

  await expect(page.getByTestId('refs-panel-test-scenario')).toHaveText('section-target')
  await page.getByRole('button').first().click()
  await expect(page.getByTestId('refs-panel-locate-overview-0')).toHaveCount(0)
  await expect(page.getByTestId('refs-panel-locate-path-0')).toHaveCount(0)
  await expect(page.getByTestId('refs-panel-locate-snippet-0')).toHaveCount(0)
})

test('refs panel suppresses misleading negative-evidence cards', async ({ page }) => {
  await page.goto('/__refs_panel_test__?scenario=negative-suppressed')

  await expect(page.getByTestId('refs-panel-test-scenario')).toHaveText('negative-suppressed')
  await page.getByRole('button').first().click()
  await expect(page.getByTestId('refs-panel-negative-suppressed-note')).toContainText('已隐藏可能误导的参考定位卡片')
  await expect(page.locator('.kb-ref-title')).toHaveCount(0)
  await expect(page.locator('.kb-ref-action')).toHaveCount(0)
})

test('refs panel auto-fetches citation meta for visible cards without clicking Cite', async ({ page }) => {
  await page.route('**/api/references/citation-meta', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        title: 'CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image',
        venue: 'CVPR',
        year: '2024',
        citation_count: 3,
        citation_source: 'OpenAlex',
        venue_kind: 'conference',
        conference_name: 'CVPR',
        conference_acronym: 'CVPR',
        conference_tier: 'A*',
        conference_rank_source: 'ICORE2026',
        conference_ccf: 'A',
        conference_ccf_source: 'CORE tier proxy',
        bibliometrics_checked: true,
      }),
    })
  })

  await page.goto('/__refs_panel_test__?scenario=auto-citation-meta')

  await expect(page.getByTestId('refs-panel-test-scenario')).toHaveText('auto-citation-meta')
  await page.getByRole('button').first().click()
  await expect(page.getByTestId('refs-panel-metrics-0')).toContainText('被引 3 (OpenAlex)')
  await expect(page.getByTestId('refs-panel-metrics-0')).toContainText('CORE A* (ICORE2026)')
  await expect(page.getByTestId('refs-panel-metrics-0')).toContainText('CCF A (CORE tier proxy)')
})

test('refs panel keeps delayed citation metadata bound to its source after reorder', async ({ page }) => {
  await page.route('**/api/references/citation-meta', async (route) => {
    const sourcePath = String(route.request().postDataJSON()?.source_path || '')
    const isPaperA = sourcePath.includes('Paper-A.en.md')
    if (isPaperA) await new Promise((resolve) => setTimeout(resolve, 300))
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        title: isPaperA ? 'Paper A' : 'Paper B',
        year: isPaperA ? '2021' : '2022',
        citation_count: isPaperA ? 11 : 22,
        citation_source: 'OpenAlex',
        bibliometrics_checked: true,
      }),
    })
  })

  await page.goto('/__refs_panel_test__?scenario=citation-meta-reorder')
  await page.locator('.kb-refs-panel .ant-collapse-header').click()
  await page.getByRole('button', { name: 'Swap reference order' }).click()

  await expect(page.locator('.kb-ref-title').nth(0)).toContainText('Paper B.pdf')
  await expect(page.getByTestId('refs-panel-metrics-0')).toContainText('22')
  await expect(page.locator('.kb-ref-title').nth(1)).toContainText('Paper A.pdf')
  await expect(page.getByTestId('refs-panel-metrics-1')).toContainText('11')
})
