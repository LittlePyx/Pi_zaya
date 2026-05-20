import { expect, test, type Page } from '@playwright/test'
import {
  READER_REGRESSION_SOURCE_PATH,
  readerRegressionDocResponse,
} from '../../src/testing/readerRegressionFixtures'

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
  await expect(page.locator('.kb-cite-pop')).toContainText('文内参考')
  await expect(page.getByTestId('citation-popover-system-b-claim')).toContainText('回答中的判断')
  await expect(page.getByTestId('citation-popover-system-b-context')).toContainText('当前论文引用语境')
  await expect(page.getByTestId('citation-popover-system-b-role')).toContainText('上游文献角色')
  await expect(page.getByTestId('citation-popover-system-b-relation')).toContainText('为什么与这个问题有关')
  await expect(page.locator('.kb-cite-pop')).toContainText('Cited prior work')
  await expect(page.locator('.kb-cite-pop')).toContainText('upstream source for single-shot')
  await expect(page.locator('.kb-cite-pop')).toContainText('Single-shot compressive spectral imaging')
  await expect(page.locator('.kb-cite-pop')).toContainText('10.1364/OE.15.014013')
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
  await expect(page.getByTestId('citation-popover-system-a-claim')).toContainText('The method details are grounded')
  await expect(page.getByTestId('citation-popover-system-a-location')).toContainText('Fixture Paper / 2. Method')
  await expect(page.getByTestId('citation-popover-system-a-location')).toContainText('sentence')
  await expect(page.getByTestId('citation-popover-system-a-evidence')).toContainText('Given a set of input multi-view images')
  await expect(page.getByTestId('citation-popover-system-a-support')).toContainText('Method section states the exact mechanism')
  await expect(popover).toContainText('原文位置')
  await expect(popover).toContainText('Fixture Paper / 2. Method')
  await expect(popover).toContainText('命中原文')
  await expect(popover).toContainText('Given a set of input multi-view images')
  await expect(popover).toContainText('为什么链接到这里')
  await expect(popover).toContainText('Method section states the exact mechanism')

  await popover.getByRole('button', { name: '打开原文证据' }).click()
  const payload = page.getByTestId('message-list-open-payload')
  await expect(payload).toContainText('"blockId": "p-method-1"')
  await expect(payload).toContainText('"anchorId": "a-p-method-1"')
  await expect(payload).toContainText('"anchorKind": "sentence"')
  await expect(payload).toContainText('"strictLocate": true')
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
