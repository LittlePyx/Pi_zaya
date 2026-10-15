import { expect, test } from '@playwright/test'


test('empty assistant stream keeps activity status and typing affordance', async ({ page }) => {
  await page.goto('/__message_list_test__?scenario=assistant-streaming-empty')

  const bubble = page.locator('.kb-msg-bubble.is-streaming')
  await expect(bubble).toBeVisible()
  await expect(bubble).toContainText(/正在回答|Answering/)
  await expect(bubble.locator('.typing-dot')).toHaveCount(3)
  await expect(page.getByTestId('streaming-markdown-answer')).toHaveCount(0)
})


test('streaming answer keeps source summary and automatic fallback outside body', async ({ page }) => {
  await page.goto('/__message_list_test__?scenario=assistant-streaming-summary')

  const bubble = page.locator('.kb-msg-bubble.is-streaming')
  await expect(page.getByTestId('assistant-source-notice')).toHaveAttribute(
    'title',
    'Streaming evidence is from the local library.',
  )
  await expect(page.getByTestId('assistant-retrieval-fallback')).toHaveText(
    /已自动使用 BM25|Automatically used BM25/,
  )
  await expect(page.getByTestId('streaming-markdown-answer')).toHaveText(
    'Streaming local evidence answer.',
  )
  await expect(bubble.locator('.typing-dot')).toHaveCount(0)
  await expect(page.getByTestId('streaming-markdown-answer')).not.toContainText(
    'Streaming evidence is from the local library.',
  )
})
