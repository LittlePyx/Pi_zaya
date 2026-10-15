import { expect, test } from '@playwright/test'

test('completed local answer keeps exact copy payloads and evidence-note handoff', async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: {
        writeText: async (value: string) => {
          ;(window as unknown as { __assistantActionClipboard?: string }).__assistantActionClipboard = value
        },
      },
    })
  })
  await page.goto('/__message_list_test__?scenario=agent-answer-contract-source-precedence')

  const actions = page.getByTestId('assistant-message-actions-9471')
  await expect(actions).toBeVisible()
  await actions.getByRole('button', { name: /Copy Text|复制文本/ }).click()
  await expect.poll(() => page.evaluate(
    () => (window as unknown as { __assistantActionClipboard?: string }).__assistantActionClipboard || '',
  )).toBe('The contract-backed answer keeps the visible response focused on the paper evidence [1].')

  await actions.getByRole('button', { name: /Copy Markdown|复制 Markdown/ }).click()
  await expect.poll(() => page.evaluate(
    () => (window as unknown as { __assistantActionClipboard?: string }).__assistantActionClipboard || '',
  )).toBe('The contract-backed answer keeps the visible response focused on the paper evidence [1].')

  await page.getByTestId('research-note-open-9471').click()
  await expect(page.locator('.kb-research-note-modal')).toBeVisible()
})

test('failed generation retries the matching question and never becomes an evidence note', async ({ page }) => {
  await page.goto('/__message_list_test__?scenario=assistant-action-retry')

  const retry = page.locator('button.kb-generation-retry-btn')
  await expect(retry).toBeEnabled()
  await expect(page.getByTestId('research-note-open-9602')).toHaveCount(0)
  await retry.click()
  await expect(page.getByTestId('assistant-action-retry-prompt')).toHaveText('Find the local evidence again.')

  await page.goto('/__message_list_test__?scenario=assistant-action-retry&busy=1')
  await expect(page.locator('button.kb-generation-retry-btn')).toBeDisabled()
  await expect(page.getByTestId('assistant-action-retry-prompt')).toHaveText('(empty)')
})
