import { expect, test } from '@playwright/test'

test('structured locate chip prefers the best evidence block over a wrong raw primary block', async ({ page }) => {
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
})
