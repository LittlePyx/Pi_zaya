import { expect, test } from '@playwright/test'

const INTERNAL_ROUTE_READY_TIMEOUT_MS = 15_000

test('unlinked reference candidate keeps its reader and basket actions bound to the same evidence', async ({ page }) => {
  await page.goto('/__message_list_test__?scenario=unlinked-reference-candidate')
  await expect(page.getByTestId('message-list-test-scenario')).toContainText(
    'unlinked-reference-candidate',
    { timeout: INTERNAL_ROUTE_READY_TIMEOUT_MS },
  )

  const candidates = page.getByTestId('unlinked-reference-candidates-1')
  await expect(candidates).toContainText('Fast rotation-shearing single-pixel imaging')

  await candidates.getByRole('button', { name: /Open|打开/ }).click()
  const openPayload = page.getByTestId('message-list-open-payload')
  await expect(openPayload).toContainText('"sourcePath": "__reader_regression__/fixture.md"')
  await expect(openPayload).toContainText('Fast rotation-shearing single-pixel imaging')

  await candidates.getByRole('button', { name: /Add|入篮/ }).click()
  const shelf = page.getByTestId('citation-shelf')
  await expect(shelf).toHaveClass(/translate-x-0/)
  await expect(shelf.getByTestId('citation-shelf-item')).toHaveCount(1)
  await expect(shelf.getByTestId('citation-shelf-item-title')).toContainText(
    'Fast rotation-shearing single-pixel imaging',
  )
})
