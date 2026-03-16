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

async function openHarness(page: Page, scenario: string) {
  await mockReaderDoc(page)
  await page.goto(`/__reader_test__?scenario=${scenario}`)
  await expect(page.getByTestId('reader-content')).toContainText('Fixture Paper')
}

async function openSplitHarness(page: Page) {
  await mockReaderDoc(page)
  await page.goto('/__reader_split_test__')
  await expect(page.getByTestId('split-reader-pane')).toBeVisible()
}

async function selectText(page: Page, startText: string, endText?: string) {
  const result = await page.evaluate(({ startText, endText }) => {
    const root = document.querySelector('[data-testid="reader-content"]')
    if (!root) return false
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT)
    const nodes: Text[] = []
    while (walker.nextNode()) {
      const node = walker.currentNode as Text
      if (!String(node.textContent || '').trim()) continue
      nodes.push(node)
    }
    const startNeedle = String(startText || '')
    const endNeedle = String(endText || startNeedle)
    let startNode: Text | null = null
    let startOffset = -1
    let endNode: Text | null = null
    let endOffset = -1
    for (const node of nodes) {
      const text = String(node.textContent || '')
      const idx = text.indexOf(startNeedle)
      if (idx >= 0) {
        startNode = node
        startOffset = idx
        break
      }
    }
    if (!startNode || startOffset < 0) return false
    if (endNeedle === startNeedle) {
      endNode = startNode
      endOffset = startOffset + startNeedle.length
    } else {
      let seenStart = false
      for (const node of nodes) {
        if (node === startNode) seenStart = true
        if (!seenStart) continue
        const text = String(node.textContent || '')
        const idx = text.indexOf(endNeedle)
        if (idx >= 0) {
          endNode = node
          endOffset = idx + endNeedle.length
          break
        }
      }
    }
    if (!endNode || endOffset <= 0) return false
    const range = document.createRange()
    range.setStart(startNode, startOffset)
    range.setEnd(endNode, endOffset)
    const selection = window.getSelection()
    selection?.removeAllRanges()
    selection?.addRange(range)
    root.dispatchEvent(new MouseEvent('mouseup', { bubbles: true }))
    return true
  }, { startText, endText })
  expect(result).toBeTruthy()
  await expect(page.getByTestId('reader-selection-bubble')).toBeVisible()
}

test('strict quote locate keeps the exact phrase target', async ({ page }) => {
  await openHarness(page, 'strict-quote')
  await expect(page.getByTestId('reader-locate-status')).toHaveText('Exact phrase')
  await expect(page.locator('.kb-reader-inline-hit')).toContainText('SCI compresses a short video into one coded measurement.')
})

test('outline jump lands on the selected section heading', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 560 })
  await openHarness(page, 'strict-quote')
  await expect(page.getByTestId('reader-outline-panel')).toBeVisible()
  const reader = page.getByTestId('reader-content')
  await page.getByTestId('reader-outline-item-2').click()
  await expect.poll(async () => reader.evaluate((node) => (node as HTMLDivElement).scrollTop)).toBeGreaterThan(120)
  await expect(page.getByRole('heading', { name: '3. Conclusion' })).toBeInViewport()
  await expect(page.getByTestId('reader-outline-item-2')).toHaveClass(/is-active/)
})

test('outline active section follows reader scroll position', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 560 })
  await openHarness(page, 'strict-quote')
  const reader = page.getByTestId('reader-content')
  await expect.poll(async () => reader.evaluate((node) => {
    const el = node as HTMLDivElement
    return el.scrollHeight > el.clientHeight + 24
  })).toBeTruthy()
  await expect(page.getByTestId('reader-outline-item-0')).toHaveClass(/is-active/)

  await reader.evaluate((node) => {
    ;(node as HTMLDivElement).scrollTop = (node as HTMLDivElement).scrollHeight
  })
  await expect(page.getByTestId('reader-outline-item-2')).toHaveClass(/is-active/)

  await reader.evaluate((node) => {
    ;(node as HTMLDivElement).scrollTop = 0
  })
  await expect(page.getByTestId('reader-outline-item-0')).toHaveClass(/is-active/)
})

test('structured fallback switches to the resolved alternative instead of re-ranking blindly', async ({ page }) => {
  await openHarness(page, 'candidate-fallback')
  await expect(page.getByTestId('reader-locate-status')).toHaveText('Exact phrase')
  await expect(page.getByTestId('reader-candidate-chip-1')).toHaveClass(/is-active/)
  await expect(page.getByTestId('reader-evidence-nav')).toHaveCount(0)
})

test('ask bubble appends the selected source text back to the session input log', async ({ page }) => {
  await openHarness(page, 'strict-quote')
  await selectText(page, 'Our method exploits neural radiance fields (NeRF) for snapshot compressed imaging.')
  await page.getByTestId('reader-selection-ask').click()
  await expect(page.getByTestId('append-output')).toContainText('> Our method exploits neural radiance fields (NeRF) for snapshot compressed imaging.')
  await expect(page.getByTestId('append-output')).toContainText('> Source: Fixture Paper / Fixture Paper / 1. Introduction')
})

test('same-paragraph highlight is stored in session state', async ({ page }) => {
  await openHarness(page, 'strict-quote')
  await selectText(page, 'Our method exploits neural radiance fields (NeRF) for snapshot compressed imaging.')
  await page.getByTestId('reader-selection-highlight').click()
  await expect(page.getByTestId('highlight-count')).toHaveText('1 highlights')
  await expect(page.getByTestId('highlight-list')).toContainText('Our method exploits neural radiance fields (NeRF) for snapshot compressed imaging.')
})

test('cross-paragraph highlight uses the same range path as a single sentence', async ({ page }) => {
  await openHarness(page, 'strict-quote')
  await selectText(
    page,
    'Our method exploits neural radiance fields (NeRF) for snapshot compressed imaging.',
    'Conventional high-speed imaging systems often face challenges such as high hardware cost and storage requirements.',
  )
  await page.getByTestId('reader-selection-highlight').click()
  await expect(page.getByTestId('highlight-count')).toHaveText('1 highlights')
  await expect(page.getByTestId('highlight-list')).toContainText('Our method exploits neural radiance fields (NeRF) for snapshot compressed imaging.')
  await expect(page.getByTestId('highlight-list')).toContainText('Conventional high-speed imaging systems often face challenges such as high hardware cost and storage requirements.')
})

test('highlights workspace can jump back to a saved session highlight', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 560 })
  await openHarness(page, 'strict-quote')
  const reader = page.getByTestId('reader-content')
  await reader.evaluate((node) => {
    ;(node as HTMLDivElement).scrollTop = (node as HTMLDivElement).scrollHeight
  })
  await selectText(page, 'Our method achieves stable reconstruction from a single snapshot.')
  await page.getByTestId('reader-selection-highlight').click()
  await expect(page.getByTestId('reader-highlights-toggle')).toContainText('1 highlight')

  await reader.evaluate((node) => {
    ;(node as HTMLDivElement).scrollTop = 0
  })
  await page.getByTestId('reader-highlights-toggle').click()
  await expect(page.getByTestId('reader-highlights-panel')).toBeVisible()
  await page.getByTestId('reader-highlight-item-0').click()

  await expect.poll(async () => reader.evaluate((node) => (node as HTMLDivElement).scrollTop)).toBeGreaterThan(120)
  await expect(page.getByTestId('reader-highlight-item-0')).toHaveClass(/is-active/)
})

test('evidence navigation walks a stable ordered list under strict locate', async ({ page }) => {
  await openHarness(page, 'evidence-nav')
  await expect(page.getByTestId('reader-evidence-nav')).toBeVisible()
  await expect(page.getByTestId('reader-evidence-position')).toHaveText('1 / 3')
  await expect(page.getByTestId('reader-locate-status')).toHaveText('Exact phrase')

  await page.getByTestId('reader-evidence-next').click()
  await expect(page.getByTestId('reader-evidence-position')).toHaveText('2 / 3')
  await expect(page.getByTestId('reader-locate-status')).toHaveText('Equation block')
  await expect(page.locator('.kb-reader-focus')).toContainText('C(r)')

  await page.getByTestId('reader-evidence-next').click()
  await expect(page.getByTestId('reader-evidence-position')).toHaveText('3 / 3')
  await expect(page.getByTestId('reader-locate-status')).toHaveText('Figure block')
  await expect(page.locator('.kb-reader-focus[data-kb-block-id="fig-1"]')).toHaveCount(1)

  await page.getByTestId('reader-evidence-prev').click()
  await expect(page.getByTestId('reader-evidence-position')).toHaveText('2 / 3')
  await expect(page.getByTestId('reader-locate-status')).toHaveText('Equation block')
})

test('equation and figure fixtures resolve through the same structured target contract', async ({ page }) => {
  await openHarness(page, 'equation')
  await expect(page.getByTestId('reader-locate-status')).toHaveText('Equation block')
  await expect(page.locator('.kb-reader-focus')).toContainText('C(r)')

  await openHarness(page, 'figure')
  await expect(page.getByTestId('reader-locate-status')).toHaveText('Figure block')
  await expect(page.locator('.kb-reader-focus[data-kb-block-id="fig-1"]')).toHaveCount(1)
})

test('split-pane resize keeps a live preview and commits width on release', async ({ page }) => {
  await page.setViewportSize({ width: 1600, height: 960 })
  await openSplitHarness(page)
  const handle = page.getByTestId('split-resize-handle')
  const pane = page.getByTestId('split-reader-pane')
  const previewLabel = page.getByTestId('split-preview-width')
  const committedLabel = page.getByTestId('split-committed-width')

  await expect(committedLabel).toHaveText('560')
  const box = await handle.boundingBox()
  if (!box) throw new Error('resize handle not available')

  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2)
  await page.mouse.down()
  await page.mouse.move(box.x - 120, box.y + box.height / 2, { steps: 6 })

  const previewWidth = Number(await previewLabel.textContent())
  expect(previewWidth).toBeGreaterThan(640)
  await expect(committedLabel).toHaveText('560')
  await expect(page.getByTestId('split-resize-guide')).toHaveClass(/opacity-100/)

  await page.mouse.up()

  await expect.poll(async () => Number(await committedLabel.textContent())).toBe(previewWidth)
  const paneWidth = await pane.evaluate((node) => (node as HTMLDivElement).clientWidth)
  expect(Math.abs(paneWidth - previewWidth)).toBeLessThanOrEqual(2)
})
