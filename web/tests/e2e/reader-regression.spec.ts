import { expect, test, type Page } from '@playwright/test'
import {
  READER_REGRESSION_SOURCE_PATH,
  buildReaderRegressionDocResponse,
  type ReaderRegressionScenario,
} from '../../src/testing/readerRegressionFixtures'
import {
  installAppShellMocks,
  installEmptyCitationShelfMock,
  installIdleReferenceMocks,
} from './mockAppShell'

const STRICT_LOCATE_LABEL_RE = /^(Strict locate|Evidence locate|证据定位)$/
const EXACT_TARGET_LABEL_RE = /^(Exact target|精确命中)$/
const SECTION_ONLY_LABEL_RE = /^(Section only|仅到章节)$/
const UNRESOLVED_LABEL_RE = /^(Unresolved|未定位)$/
const AUTO_SWITCHED_LABEL_RE = /^(Auto-switched|已自动切换)$/
const ONE_HIGHLIGHT_LABEL_RE = /^1 (highlight|highlights|条高亮)$/
const REQUESTED_LABEL_RE = /Requested|请求/
const RESOLVED_LABEL_RE = /Resolved|已解析/

test.describe.configure({ timeout: 60_000 })

test.beforeEach(async ({ page }) => {
  await installAppShellMocks(page)
  await installEmptyCitationShelfMock(page, { scopeId: 'reader-regression-project' })
  await installIdleReferenceMocks(page)
})

async function mockReaderDoc(page: Page, scenario: ReaderRegressionScenario = 'strict-quote') {
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
      body: JSON.stringify(buildReaderRegressionDocResponse(scenario)),
    })
  })
}

async function openHarness(page: Page, scenario: ReaderRegressionScenario) {
  await mockReaderDoc(page, scenario)
  await page.goto(`/__reader_test__?scenario=${scenario}`, { waitUntil: 'domcontentloaded' })
  const expectedTitle = scenario === 'citation-links'
    ? 'Citation Fixture'
    : scenario === 'render-polish'
      ? 'Render Polish Fixture'
      : scenario === 'image-anchor-mismatch'
        ? 'Anchor Mismatch Fixture'
      : 'Fixture Paper'
  await expect(page.getByTestId('reader-content')).toContainText(expectedTitle)
}

async function openSplitHarness(page: Page) {
  await mockReaderDoc(page)
  await page.goto('/__reader_split_test__', { waitUntil: 'domcontentloaded' })
  await expect(page.getByTestId('split-reader-pane')).toBeVisible()
}

async function selectText(page: Page, startText: string, endText?: string) {
  await expect(page.getByTestId('reader-locate-result-json')).not.toHaveText('(empty)')
  const scrolled = await page.evaluate(({ startText }) => {
    const root = document.querySelector('[data-testid="reader-content"]')
    if (!root) return false
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT)
    while (walker.nextNode()) {
      const node = walker.currentNode as Text
      const text = String(node.textContent || '')
      if (!text.includes(String(startText || ''))) continue
      node.parentElement?.scrollIntoView({ block: 'center', inline: 'nearest' })
      return true
    }
    return false
  }, { startText })
  expect(scrolled).toBeTruthy()
  await page.waitForTimeout(80)
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
    if (startNode.parentElement) {
      startNode.parentElement.scrollIntoView({ block: 'center', inline: 'nearest' })
    }
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
  await expect(page.getByTestId('reader-locate-mode')).toHaveText(STRICT_LOCATE_LABEL_RE)
  await expect(page.getByTestId('reader-locate-resolution')).toHaveText(EXACT_TARGET_LABEL_RE)
  await expect(page.getByTestId('reader-locate-status')).toHaveText('Exact phrase')
  await expect(page.getByTestId('reader-locate-result-json')).toContainText('"status": "exact"')
  await expect(page.getByTestId('reader-locate-result-json')).toContainText('"ok": true')
  await expect(page.locator('.kb-reader-inline-hit')).toContainText('SCI compresses a short video into one coded measurement.')
})

test('reader clears rendered markdown when the source payload becomes empty', async ({ page }) => {
  await openHarness(page, 'strict-quote')
  await expect(page.getByTestId('reader-content')).toContainText('Fixture Paper')

  await page.getByTestId('reader-toggle-source').click()
  await expect(page.getByTestId('reader-source-state')).toHaveText('empty')
  await expect(page.getByTestId('reader-content')).toHaveCount(0)

  await page.getByTestId('reader-toggle-source').click()
  await expect(page.getByTestId('reader-source-state')).toHaveText('source')
  await expect(page.getByTestId('reader-content')).toContainText('Fixture Paper')
})

test('multi-panel caption locate highlights the combined target snippet', async ({ page }) => {
  await openHarness(page, 'multi-panel')
  await expect(page.getByTestId('reader-locate-status')).toHaveText('Exact phrase')
  await expect(page.locator('.kb-reader-focus[data-kb-block-id="p-fig-panels"]')).toHaveCount(1)
  await expect(page.locator('.kb-reader-inline-hit')).toContainText('g Line profiles of the iPSF')
})

test('discussion-only locate can open the reader at section level without an exact block id', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 640 })
  await openHarness(page, 'discussion-only')
  await expect(page.getByTestId('reader-locate-mode')).toHaveText(STRICT_LOCATE_LABEL_RE)
  await expect(page.getByTestId('reader-locate-resolution')).toHaveText(SECTION_ONLY_LABEL_RE)
  await expect(page.getByTestId('reader-locate-status')).toHaveText('Heading match')
  await expect(page.locator('.kb-reader-focus')).toContainText('4. Discussion')
  await expect(page.getByRole('heading', { name: '4. Discussion' })).toBeInViewport()
  await expect(page.getByTestId('reader-outline-item-3')).toContainText('4. Discussion')
})

test('limitations-only locate can open the reader at section level without an exact block id', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 640 })
  await openHarness(page, 'limitations-only')
  await expect(page.getByTestId('reader-locate-status')).toHaveText('Heading match')
  await expect(page.locator('.kb-reader-focus')).toContainText('5. Limitations')
  await expect(page.getByRole('heading', { name: '5. Limitations' })).toBeInViewport()
  await expect(page.getByTestId('reader-outline-item-4')).toContainText('5. Limitations')
})

test('future-work-only locate can open the reader at section level without an exact block id', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 640 })
  await openHarness(page, 'future-work-only')
  await expect(page.getByTestId('reader-locate-status')).toHaveText('Heading match')
  await expect(page.locator('.kb-reader-focus')).toContainText('6. Future Work')
  await expect(page.getByRole('heading', { name: '6. Future Work' })).toBeInViewport()
  await expect(page.getByTestId('reader-outline-item-5')).toContainText('6. Future Work')
})

test('outline jump lands on the selected section heading', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 560 })
  await openHarness(page, 'strict-quote')
  await expect(page.getByTestId('reader-outline-panel')).toBeVisible()
  const reader = page.getByTestId('reader-content')
  await page.getByTestId('reader-outline-item-2').click()
  await expect.poll(async () => reader.evaluate((node) => (node as HTMLDivElement).scrollTop)).toBeGreaterThan(120)
  await expect(page.getByTestId('reader-outline-item-2')).toContainText('3. Conclusion')
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
  await expect(page.getByTestId('reader-outline-item-5')).toHaveClass(/is-active/)

  await reader.evaluate((node) => {
    ;(node as HTMLDivElement).scrollTop = 0
  })
  await expect(page.getByTestId('reader-outline-item-0')).toHaveClass(/is-active/)
})

test('structured fallback switches to the resolved alternative instead of re-ranking blindly', async ({ page }) => {
  await openHarness(page, 'candidate-fallback')
  await expect(page.getByTestId('reader-locate-switch')).toHaveText(AUTO_SWITCHED_LABEL_RE)
  await expect(page.getByTestId('reader-locate-decision')).toHaveText(/best backup evidence|最接近的备用证据/)
  await expect(page.getByTestId('reader-locate-status')).toHaveText('Exact phrase')
  await expect(page.getByTestId('reader-candidate-chip-0')).toContainText(REQUESTED_LABEL_RE)
  await expect(page.getByTestId('reader-candidate-chip-1')).toHaveClass(/is-active/)
  await expect(page.getByTestId('reader-candidate-chip-1')).toContainText(RESOLVED_LABEL_RE)
  await expect(page.getByTestId('reader-candidate-chip-2')).toBeVisible()
  await expect(page.getByTestId('reader-evidence-nav')).toHaveCount(0)
})

test('strict exact locate does not degrade to heading fallback when direct identity is missing', async ({ page }) => {
  await openHarness(page, 'strict-missing-exact')
  await expect(page.getByTestId('reader-locate-resolution')).toHaveText(UNRESOLVED_LABEL_RE)
  await expect(page.getByTestId('reader-locate-status')).toHaveText('Strict stopped')
  await expect(page.getByTestId('reader-locate-result-json')).toContainText('"status": "failed"')
  await expect(page.getByTestId('reader-locate-result-json')).toContainText('"repairable": true')
  await expect(page.locator('.kb-reader-focus')).toHaveCount(0)
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
  await selectText(page, 'Looking ahead, the most direct extension would be to combine the current pipeline with adaptive masking')
  await page.getByTestId('reader-selection-highlight').click()
  await expect(page.getByTestId('reader-highlights-toggle')).toHaveText(ONE_HIGHLIGHT_LABEL_RE)

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

test('duplicate section alternatives collapse to distinct visible entries', async ({ page }) => {
  await openHarness(page, 'duplicate-sections')
  await expect(page.getByTestId('reader-evidence-nav')).toBeVisible()
  await expect(page.getByTestId('reader-evidence-position')).toHaveText('1 / 3')
  await expect(page.getByTestId('reader-candidate-toggle')).toHaveText(/^(3 candidates|3 个候选)$/)

  await page.getByTestId('reader-candidate-toggle').click()
  await expect(page.getByTestId('reader-candidate-chip-0')).toContainText('2. Method')
  await expect(page.getByTestId('reader-candidate-chip-1')).toContainText('Eq. (1)')
  await expect(page.getByTestId('reader-candidate-chip-2')).toContainText('Figure 1')
  await expect(page.getByTestId('reader-candidate-chip-3')).toHaveCount(0)
})

test('reader suppresses repeated identical figure assets in the same document render', async ({ page }) => {
  await openHarness(page, 'duplicate-images')
  await expect(page.locator('.kb-md-image')).toHaveCount(1)
  const figureShell = page.locator('.kb-md-figure-shell').first()
  const figureTail = figureShell.locator('> .kb-md-reader-block-shelf-tail')
  const figureButton = figureShell.locator('[data-testid="reader-block-shelf"][data-kb-reader-block-kind="figure"]')
  await expect(figureButton).toHaveCount(1)
  await expect(figureTail).toHaveCSS('opacity', '0')
  await figureShell.hover()
  await expect(figureTail).toHaveCSS('opacity', '1')

  await figureButton.click()
  await expect(page.getByTestId('reader-selection-shelf-count')).toHaveText('1 selections')
  await expect(page.getByTestId('reader-selection-shelf-kind-0')).toHaveText('figure')
  await expect(page.getByTestId('reader-selection-shelf-list')).toContainText('Figure 1')
})

test('reader image shelf button survives non-figure anchor mismatch', async ({ page }) => {
  await openHarness(page, 'image-anchor-mismatch')
  const image = page.locator('.kb-md-image')
  await expect(image).toHaveCount(1)
  const figureShell = page.locator('.kb-md-figure-shell')
  const figureTail = figureShell.locator('> .kb-md-reader-block-shelf-tail')
  const figureButton = figureShell.locator('[data-testid="reader-block-shelf"][data-kb-reader-block-kind="figure"]')
  await expect(figureButton).toHaveCount(1)
  await expect(figureTail).toHaveCSS('opacity', '0')
  await figureShell.hover()
  await expect(figureTail).toHaveCSS('opacity', '1')
})

test('reader normalizes glued microsecond latex units before KaTeX render', async ({ page }) => {
  await openHarness(page, 'render-polish')
  const reader = page.getByTestId('reader-content')
  await expect(reader).toContainText('0.02')
  await expect(reader).toContainText('20')
  await expect(reader).not.toContainText('\\mus')
  await expect(reader.locator('.katex-error')).toHaveCount(0)
  const referenceEntries = page.locator('.kb-md-reference-entry')
  await expect(referenceEntries).toHaveCount(3)
  await expect(referenceEntries.nth(0)).toContainText('First reference title')
  await expect(referenceEntries.nth(1)).toContainText('Second reference should split')
  await expect(referenceEntries.nth(2)).toContainText('Third reference already starts')
})

test('reader in-paper citations and reference entries open system-b cards', async ({ page }) => {
  await page.route('**/api/references/bibliometrics', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ bibliometrics_checked: true }),
    })
  })
  await page.route('**/api/references/citation-card-polish', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ citation_card_polish_status: 'disabled', citation_card_polish_checked: true }),
    })
  })
  await openHarness(page, 'citation-links')

  const citeChips = page.locator('[data-testid="reader-content"] .kb-cite-chip-sysb')
  await expect(citeChips).toHaveCount(6)
  await expect(citeChips.first()).toHaveText('[1]')
  await expect(citeChips.nth(2)).toHaveText('[1]')
  await expect(citeChips.nth(3)).toHaveText('[2]')
  await expect(citeChips.nth(4)).toHaveText('[3]')
  await expect(citeChips.nth(5)).toHaveText('[4]')
  await expect(page.locator('.kb-md-reference-entry')).toHaveCount(4)
  await expect(page.locator('.kb-md-reference-entry .kb-cite-chip-sysb')).toHaveCount(0)
  const referenceActions = page.locator('.kb-md-reference-entry-action')
  await expect(referenceActions).toHaveCount(4)
  await expect(referenceActions.first()).toHaveCSS('opacity', '0')
  await page.locator('.kb-md-reference-entry').first().hover()
  await expect(referenceActions.first()).toHaveCSS('opacity', '1')

  await citeChips.first().click()
  const popover = page.locator('.kb-cite-pop')
  await expect(popover).toBeVisible()
  await expect(popover).toHaveClass(/kb-cite-pop-system-b/)
  await expect(page.getByTestId('citation-popover-system-b-card')).toBeVisible()
  await expect(page.getByTestId('citation-popover-system-b-overview')).toBeVisible()
  await expect(popover).toContainText('Single-shot compressive spectral imaging')
  await expect(popover).toContainText('This paper introduces a dual-disperser architecture')
  await expect(popover).not.toContainText('Snapshot compressive imaging often builds')
  await expect(page.getByTestId('citation-popover-system-b-reference')).toHaveCount(0)
  await popover.locator('.kb-cite-pop-close').click()
  await expect(popover).toHaveCount(0)

  await page.locator('.kb-md-reference-entry').first().click({ position: { x: 220, y: 12 } })
  await expect(popover).toBeVisible()
  await expect(popover).toHaveClass(/kb-cite-pop-system-b/)
  await expect(page.getByTestId('citation-popover-system-b-card')).toBeVisible()
  await expect(popover).toContainText('Single-shot compressive spectral imaging')
  await expect(popover).not.toContainText('The opened paper cites this upstream work as reference [1].')

  await popover.locator('.kb-cite-pop-add').click()
  await expect(page.getByTestId('reader-citation-shelf-count')).toHaveText('1 citation refs')
})

test('reader figure, equation, and table blocks can be added directly to the research basket', async ({ page }) => {
  await openHarness(page, 'strict-quote')

  await page.locator('.katex-display').first().hover()
  const equationButton = page.locator('[data-testid="reader-block-shelf"][data-kb-reader-block-kind="equation"]').first()
  const figureShell = page.locator('.kb-md-figure-shell').first()
  const figureTail = figureShell.locator('> .kb-md-reader-block-shelf-tail')
  const figureButton = figureShell.locator('[data-testid="reader-block-shelf"][data-kb-reader-block-kind="figure"]')
  const tableWrap = page.locator('.kb-md-table-action-host').first()
  const tableTail = tableWrap.locator('> .kb-md-reader-block-shelf-tail')
  const tableButton = tableWrap.locator('[data-testid="reader-block-shelf"][data-kb-reader-block-kind="table"]')
  await expect(equationButton).toBeVisible()
  await expect(figureButton).toBeVisible()
  await expect(figureTail).toHaveCSS('opacity', '0')
  await expect(tableButton).toHaveCount(1)
  await expect(tableTail).toHaveCSS('opacity', '0')

  await equationButton.click()
  await expect(page.getByTestId('reader-selection-shelf-count')).toHaveText('1 selections')
  await expect(page.getByTestId('reader-selection-shelf-kind-0')).toHaveText('equation')
  await expect(page.getByTestId('reader-selection-shelf-list')).toContainText('C(r)')

  await figureShell.hover()
  await expect(figureTail).toHaveCSS('opacity', '1')
  await figureButton.click()
  await expect(page.getByTestId('reader-selection-shelf-count')).toHaveText('2 selections')
  await expect(page.getByTestId('reader-selection-shelf-kind-0')).toHaveText('figure')
  await expect(page.getByTestId('reader-selection-shelf-list')).toContainText('Figure 1. SCI system pipeline.')

  await tableWrap.hover()
  await expect(tableTail).toHaveCSS('opacity', '1')
  await tableButton.click()
  await expect(page.getByTestId('reader-selection-shelf-count')).toHaveText('3 selections')
  await expect(page.getByTestId('reader-selection-shelf-kind-0')).toHaveText('table')
  await expect(page.getByTestId('reader-selection-shelf-list')).toContainText('PSNR')
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
