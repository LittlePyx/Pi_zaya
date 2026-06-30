import { expect, test } from '@playwright/test'
import {
  READER_REGRESSION_SOURCE_PATH,
} from '../../src/testing/readerRegressionFixtures'
import {
  installAppShellMocks,
  installEmptyCitationShelfMock,
  installIdleReferenceMocks,
} from './mockAppShell'

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
  await page.getByText('Research Agent Trace').click()
  await expect(page.getByTestId('agent-trace-unsupported-claim')).toContainText('fully solves every downstream limitation')
  await expect(page.getByTestId('agent-trace-unsupported-claim')).toContainText('Citation does not match retrieved evidence')

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
