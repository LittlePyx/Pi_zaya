import { expect, test } from '@playwright/test'

test('refs panel preserves backend reader_open candidates when opening the reader', async ({ page }) => {
  await page.goto('/__refs_panel_test__')

  await expect(page.getByTestId('refs-panel-test-scenario')).toHaveText('rich-reader-open')
  await page.getByRole('button').first().click()
  await expect(page.locator('.kb-ref-title')).toContainText('Fixture Paper')
  await expect(page.locator('.kb-ref-action').first()).toBeVisible()
  await expect(page.getByTestId('refs-panel-summary-basis')).toContainText('LLM')
  await expect(page.getByTestId('refs-panel-why-basis')).toContainText('LLM')

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

test('refs panel suppresses misleading negative-evidence cards', async ({ page }) => {
  await page.goto('/__refs_panel_test__?scenario=negative-suppressed')

  await expect(page.getByTestId('refs-panel-test-scenario')).toHaveText('negative-suppressed')
  await page.getByRole('button').first().click()
  await expect(page.getByTestId('refs-panel-negative-suppressed-note')).toContainText('已隐藏可能误导的参考定位卡片')
  await expect(page.locator('.kb-ref-title')).toHaveCount(0)
  await expect(page.locator('.kb-ref-action')).toHaveCount(0)
})
