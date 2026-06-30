import fs from 'node:fs'
import path from 'node:path'

import { expect, test, type Page } from '@playwright/test'

const PAPER_NAME = 'NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf'
const PAPER_SEARCH = 'NatPhoton-2019'

test.use({
  // The flow is long and we want a full recording.
  video: 'on',
  trace: 'on',
  screenshot: 'on',
  viewport: { width: 1440, height: 900 },
})

type GoldenCase = {
  id: string
  question: string
  answerContainsAny?: string[]
  answerNotContains?: string[]
  locateBlockIdsAny?: string[]
  minLocateButtons?: number
  notes?: string
}

type LocateAudit = {
  count: number
  targets: Array<{
    blockId: string
    heading: string
  }>
}

type GoldenCaseReport = {
  id: string
  question: string
  answerDoneMs: number
  locateReadyMs: number
  locateButtonCount: number
  locateBlockIds: string[]
  checks: Record<string, boolean>
  answerPreview: string
}

const GOLDEN_CASES_PATH = path.join(process.cwd(), 'tests', 'fixtures', 'paper-guide-golden-cases.json')
const GOLDEN_REPORT_PATH = path.join(process.cwd(), 'test-results', 'paper-guide-golden-report.json')
const GOLDEN_CASES: GoldenCase[] = JSON.parse(fs.readFileSync(GOLDEN_CASES_PATH, 'utf8'))

const QUESTION_LIMIT_RAW = Number(process.env.PW_QUESTION_LIMIT || 0)
const QUESTION_LIMIT = Number.isFinite(QUESTION_LIMIT_RAW) && QUESTION_LIMIT_RAW > 0
  ? Math.max(1, Math.floor(QUESTION_LIMIT_RAW))
  : 0
const QUESTION_OFFSET_RAW = Number(process.env.PW_QUESTION_OFFSET || 0)
const QUESTION_OFFSET = Number.isFinite(QUESTION_OFFSET_RAW) && QUESTION_OFFSET_RAW > 0
  ? Math.max(0, Math.floor(QUESTION_OFFSET_RAW))
  : 0
const RUN_RECORDED_FLOW = process.env.PW_RUN_PAPER_GUIDE_RECORDED === '1'
const ACTIVE_QUESTIONS = (() => {
  const sliced = GOLDEN_CASES.slice(QUESTION_OFFSET)
  return QUESTION_LIMIT > 0 ? sliced.slice(0, QUESTION_LIMIT) : sliced
})()

function writeGoldenReport(records: GoldenCaseReport[]) {
  fs.mkdirSync(path.dirname(GOLDEN_REPORT_PATH), { recursive: true })
  fs.writeFileSync(
    GOLDEN_REPORT_PATH,
    JSON.stringify({
      paper: PAPER_NAME,
      generatedAt: new Date().toISOString(),
      caseCount: records.length,
      passCount: records.filter((item) => Object.values(item.checks).every(Boolean)).length,
      records,
    }, null, 2),
    'utf8',
  )
}

async function startPaperGuideFromLibrary(page: Page) {
  await page.goto('/library')

  // Filter for the paper by keyword.
  const search = page.getByPlaceholder('搜索标题、分类、标签或备注')
  await expect(search).toBeVisible()
  await search.fill(PAPER_SEARCH)

  const row = page.locator('.kb-lib-file-row', { hasText: PAPER_NAME })
  await expect(row).toHaveCount(1)

  // "阅读" starts a paper-guide conversation and navigates to chat page.
  await row.getByRole('button', { name: '阅读' }).click()
  await expect(page).toHaveURL('/')

  // Wait until paper-guide binding shows up.
  const guideMeta = page.locator('.kb-chat-meta-inline-guide')
  await expect(guideMeta).toBeVisible({ timeout: 30_000 })
  await expect(guideMeta).toContainText('阅读指导')
  await expect(guideMeta).toContainText('NatPhoton-2019')
}

async function waitForGenerationDone(page: Page) {
  // When generating, the stop button exists; when done, the send button is enabled.
  await expect(page.locator('button.kb-stop-btn')).toHaveCount(0, { timeout: 240_000 })
  const sendBtn = page.locator('button.kb-send-btn')
  await expect(sendBtn).toBeVisible({ timeout: 30_000 })
}

function assistantMessages(page: Page) {
  return page.locator('div[data-msg-id]', {
    has: page.locator('img[alt="Pi assistant"]'),
  })
}

async function clickLocateButtonsAndAssert(
  page: Page,
  assistantMsg: ReturnType<Page['locator']>,
  stepKey: string,
  attach: (name: string, buffer: Buffer) => Promise<void>,
  expectedBlockIdsAny: string[] = [],
): Promise<LocateAudit> {
  const locateBtns = assistantMsg.locator('button[aria-label="定位到原文证据"]')
  await expect.poll(async () => locateBtns.count(), {
    timeout: 30_000,
    intervals: [250, 500, 750, 1000, 1500, 2000],
  }).toBeGreaterThan(0)
  const expectedBlocks = new Set(expectedBlockIdsAny.map((item) => String(item || '').trim()).filter(Boolean))
  if (expectedBlocks.size > 0) {
    await expect.poll(async () => {
      const targets = await collectLocateTargets(locateBtns)
      return targets.some((item) => expectedBlocks.has(item.blockId))
    }, {
      message: `${stepKey}: locate buttons never reached expected block set`,
      timeout: 45_000,
      intervals: [500, 750, 1000, 1500, 2000, 3000],
    }).toBeTruthy()
  }
  const count = await locateBtns.count()
  const targets = await collectLocateTargets(locateBtns)
  expect.soft(count, `no locate buttons rendered for step=${stepKey}`).toBeGreaterThan(0)
  if (count <= 0) {
    const shot = await page.screenshot({ fullPage: false })
    await attach(`no-locate-${stepKey}.png`, shot)
    return { count, targets }
  }

  const maxClicks = Math.min(3, count)
  for (let i = 0; i < maxClicks; i += 1) {
    const btn = locateBtns.nth(i)
    const expectedBlockId = (await btn.getAttribute('data-kb-locate-block-id')) || ''
    const expectedHeading = (await btn.getAttribute('data-kb-locate-heading')) || ''
    await btn.click()

    const reader = page.locator('[data-testid="reader-content"], .kb-reader-content').first()
    await expect(reader).toBeVisible({ timeout: 30_000 })

    const locateMeta = page.locator([
      '[data-testid="reader-locate-status"]',
      '[data-testid="reader-locate-resolution"]',
      '[data-testid="reader-locate-mode"]',
    ].join(', '))
    await expect.poll(async () => locateMeta.count(), {
      timeout: 30_000,
      intervals: [250, 500, 750, 1000],
    }).toBeGreaterThan(0)
    await expect(locateMeta.first()).toBeVisible({ timeout: 30_000 })

    // Should not degrade into fuzzy locate for strict provenance locate.
    const locateMetaText = (await locateMeta.allInnerTexts()).join(' ')
    expect.soft(locateMetaText).not.toMatch(/Fuzzy|fuzzy/i)

    // Must have a focused block in reader.
    const focus = page.locator('.kb-reader-focus')
    await expect.soft(focus).toHaveCount(1, { timeout: 12_000 })

    if (expectedBlockId && expectedBlockId.trim()) {
      const focusedBlockId = await focus.first().evaluate((node) => {
        const el = node as HTMLElement
        const direct = el.getAttribute('data-kb-block-id') || ''
        if (direct) return direct
        const parent = el.closest('[data-kb-block-id]') as HTMLElement | null
        return parent?.getAttribute('data-kb-block-id') || ''
      }).catch(() => '')
      expect.soft(focusedBlockId).toBe(expectedBlockId.trim())
    } else if (expectedHeading && expectedHeading.trim()) {
      // Fallback sanity check when block id is missing.
      const meta = page.locator('.kb-reader-meta-location')
      const metaText = await meta.innerText().catch(() => '')
      expect.soft(String(metaText || '')).toContain(expectedHeading.trim().split('/').pop()!.trim())
    }

    const shot = await page.screenshot({ fullPage: false })
    await attach(`locate-${stepKey}-${i + 1}.png`, shot)
  }
  return { count, targets }
}

async function collectLocateTargets(locateBtns: ReturnType<Page['locator']>): Promise<LocateAudit['targets']> {
  const count = await locateBtns.count()
  const targets: LocateAudit['targets'] = []
  for (let i = 0; i < count; i += 1) {
    const btn = locateBtns.nth(i)
    targets.push({
      blockId: ((await btn.getAttribute('data-kb-locate-block-id')) || '').trim(),
      heading: ((await btn.getAttribute('data-kb-locate-heading')) || '').trim(),
    })
  }
  return targets
}

test.describe.serial('paper guide locate flow (recorded)', () => {
  test.skip(!RUN_RECORDED_FLOW, 'Set PW_RUN_PAPER_GUIDE_RECORDED=1 with a live backend and local NatPhoton-2019 library data to run this recorded flow.')
  // Full end-to-end (12 natural questions + strict locate clicks) can take a long time when deep-read is enabled.
  test.setTimeout(90 * 60_000)

  test('NatPhoton-2019: natural questions with locate jumps', async ({ page }, testInfo) => {
    await startPaperGuideFromLibrary(page)
    const reportRecords: GoldenCaseReport[] = []
    writeGoldenReport(reportRecords)

    const attach = async (name: string, buffer: Buffer) => {
      await testInfo.attach(name, { body: buffer, contentType: 'image/png' })
    }

    for (let idx = 0; idx < ACTIVE_QUESTIONS.length; idx += 1) {
      const goldenCase = ACTIVE_QUESTIONS[idx]
      const q = goldenCase.question
      const stepKey = `q${String(QUESTION_OFFSET + idx + 1).padStart(2, '0')}`

      const beforeCount = await assistantMessages(page).count()
      const startedAt = Date.now()

      const input = page.locator('textarea.kb-chat-textarea, .kb-chat-textarea textarea')
      await expect(input).toBeVisible({ timeout: 30_000 })
      await input.fill(q)
      await page.locator('button.kb-send-btn').click()

      // Wait a new assistant message arrives and generation is complete.
      await expect.poll(async () => assistantMessages(page).count(), {
        timeout: 240_000,
      }).toBeGreaterThan(beforeCount)

      await waitForGenerationDone(page)

      const msg = assistantMessages(page).last()
      await expect(msg).toBeVisible({ timeout: 30_000 })
      await expect(msg).toContainText(/./, { timeout: 30_000 })
      const answerDoneMs = Date.now() - startedAt
      const answerText = await msg.innerText()

      const locateAudit = await clickLocateButtonsAndAssert(
        page,
        msg,
        stepKey,
        attach,
        goldenCase.locateBlockIdsAny || [],
      )
      const locateReadyMs = Date.now() - startedAt
      const answerLower = answerText.toLowerCase()
      const expectedTerms = listLower(goldenCase.answerContainsAny)
      const forbiddenTerms = listLower(goldenCase.answerNotContains)
      const expectedBlocks = new Set((goldenCase.locateBlockIdsAny || []).map((item) => String(item || '').trim()).filter(Boolean))
      const locateBlockIds = locateAudit.targets.map((item) => item.blockId).filter(Boolean)
      const checks = {
        answerContainsAny: expectedTerms.length <= 0 || expectedTerms.some((term) => answerLower.includes(term)),
        forbiddenTextAbsent: forbiddenTerms.every((term) => !answerLower.includes(term)),
        locateButtonCount: locateAudit.count >= Math.max(1, Number(goldenCase.minLocateButtons || 1)),
        expectedBlockMatched: expectedBlocks.size <= 0 || locateBlockIds.some((blockId) => expectedBlocks.has(blockId)),
      }
      expect.soft(checks.answerContainsAny, `${goldenCase.id}: answer missing expected terms`).toBeTruthy()
      expect.soft(checks.forbiddenTextAbsent, `${goldenCase.id}: answer contains forbidden internal text`).toBeTruthy()
      expect.soft(checks.locateButtonCount, `${goldenCase.id}: locate button count below threshold`).toBeTruthy()
      expect.soft(checks.expectedBlockMatched, `${goldenCase.id}: locate blocks did not match expected set`).toBeTruthy()

      reportRecords.push({
        id: goldenCase.id,
        question: q,
        answerDoneMs,
        locateReadyMs,
        locateButtonCount: locateAudit.count,
        locateBlockIds,
        checks,
        answerPreview: answerText.replace(/\s+/g, ' ').slice(0, 600),
      })
      writeGoldenReport(reportRecords)
    }
  })
})

function listLower(values: string[] | undefined): string[] {
  return (values || [])
    .map((item) => String(item || '').trim().toLowerCase())
    .filter(Boolean)
}
