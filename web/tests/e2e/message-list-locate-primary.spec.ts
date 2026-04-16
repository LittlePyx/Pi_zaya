import { expect, test, type Page } from '@playwright/test'
import {
  READER_REGRESSION_SOURCE_PATH,
  readerRegressionDocResponse,
} from '../../src/testing/readerRegressionFixtures'

async function mockReaderDoc(page: Page) {
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
})

test('render packet hidden locate does not leak a visible locate chip', async ({ page }) => {
  await mockReaderDoc(page)
  await page.goto('/__message_list_test__?scenario=render-packet-hidden-locate')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('render-packet-hidden-locate')
  await expect(page.locator('body')).toContainText('This answer should not expose a hidden locate target.')
  await expect(page.locator('.kb-prov-locate-chip')).toHaveCount(0)
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

test('negative evidence-note locate is suppressed instead of showing a misleading jump', async ({ page }) => {
  await mockReaderDoc(page)
  await page.goto('/__message_list_test__?scenario=negative-evidence-locate')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('negative-evidence-locate')
  await expect(page.locator('body')).toContainText('does not mention ADMM')
  await expect(page.locator('.kb-prov-locate-chip')).toHaveCount(0)
})
