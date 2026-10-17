import { expect, test } from '@playwright/test'

const INTERNAL_ROUTE_READY_TIMEOUT_MS = 15_000

test('completed legacy answer exposes its bound evidence without source-summary metadata', async ({ page }) => {
  await page.goto('/__message_list_test__?scenario=system-a-citation-popover')
  await expect(page.getByTestId('message-list-test-scenario')).toContainText('system-a-citation-popover', { timeout: INTERNAL_ROUTE_READY_TIMEOUT_MS })
  await page.getByTestId('assistant-source-notice').click()
  const drawer = page.getByTestId('evidence-drawer')
  await expect(drawer).toBeVisible()
  await expect(drawer.getByTestId('evidence-drawer-item')).toHaveCount(1)
  await expect(drawer.locator('blockquote')).toContainText('NeRF transfers the pixels')
  await drawer.getByTestId('evidence-open-source').click()
  await expect(page.getByTestId('message-list-open-payload')).toContainText('"blockId": "p-method-1"')
})

test('an empty atomic packet cannot expose stale legacy cards through an evidence entry', async ({ page }) => {
  await page.goto('/__message_list_test__?scenario=render-packet-empty-citations')
  await expect(page.getByTestId('message-list-test-scenario')).toContainText('render-packet-empty-citations', { timeout: INTERNAL_ROUTE_READY_TIMEOUT_MS })
  await expect(page.getByTestId('assistant-source-notice')).toHaveCount(0)
  await expect(page.getByTestId('evidence-drawer')).toHaveCount(0)
})

test('completed answer keeps source notice outside Markdown and citations interactive', async ({ page }) => {
  await page.goto('/__message_list_test__?scenario=agent-answer-contract-source-precedence')
  await expect(page.getByTestId('message-list-test-scenario')).toContainText(
    'agent-answer-contract-source-precedence',
    { timeout: INTERNAL_ROUTE_READY_TIMEOUT_MS },
  )

  const markdown = page.locator('.kb-markdown-chat').first()
  await expect(markdown).toContainText('The contract-backed answer keeps the visible response focused on the paper evidence')
  await expect(page.getByTestId('assistant-source-notice')).toBeVisible()
  await expect(markdown).not.toContainText('Answer uses local knowledge-base evidence.')

  await page.locator('.kb-cite-chip').first().click()
  await expect(page.getByTestId('citation-popover-system-a-evidence')).toContainText(
    'frequency-division multiplexed illumination',
  )
})

test('completed answer preserves the typed strict-locate contract', async ({ page }) => {
  await page.goto('/__message_list_test__?scenario=required-fallback-anchor')
  await expect(page.getByTestId('message-list-test-scenario')).toContainText(
    'required-fallback-anchor',
    { timeout: INTERNAL_ROUTE_READY_TIMEOUT_MS },
  )

  const locate = page.locator('.kb-prov-locate-chip').first()
  await expect(locate).toHaveAttribute('data-kb-locate-block-id', 'p-1')
  await expect(locate).toHaveAttribute('data-kb-locate-anchor-id', 'a-p-1')
  await locate.click()
  await expect(page.getByTestId('message-list-open-payload')).toContainText('"blockId": "p-1"')
})

test('completed answer state separates a diagnosed low-confidence notice from the body', async ({ page }) => {
  await page.goto('/__message_list_test__')
  const result = await page.evaluate(async () => {
    const { buildAssistantCompletedContentState } = await import(
      '/src/components/chat/assistantCompletedContentState.ts'
    )
    return buildAssistantCompletedContentState(
      {
        id: 41,
        role: 'assistant',
        content: 'Stored answer',
        meta: {
          answer_quality: {
            retrieval_confidence: {
              low_confidence: true,
              low_confidence_reason: 'empty_hits',
              candidate_refs_for_notice: [3, 3, 7],
            },
          },
        },
      } as never,
      'Note: this answer is based on lower-confidence evidence matching.\n\nGrounded answer body.',
      { msg_empty_hits: 'no scoped evidence was retrieved' },
    )
  })

  expect(result.bodyContent).toBe('Grounded answer body.')
  expect(result.lowConfidenceMeta).toMatchObject({
    reasonCode: 'empty_hits',
    candidateRefs: [3, 7],
  })
})
