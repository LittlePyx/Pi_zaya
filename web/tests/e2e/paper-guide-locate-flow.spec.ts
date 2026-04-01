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

const QUESTIONS: string[] = [
  '这篇文章想解决的核心问题是什么，为什么传统方案不太好？',
  '单像素成像的基本工作流程可以用一段话讲清楚吗？（从采集到重建）',
  '文中提到的几类主流重建方法分别有什么优缺点，适用场景怎么选？',
  '作者强调的主要瓶颈是什么：噪声、速度、分辨率还是硬件复杂度？为什么？',
  '文中有没有给出提升成像速度的关键思路？它的代价是什么？',
  '文章里对“压缩率/采样数”和重建质量的关系是怎么讨论的？',
  '作者如何评价深度学习方法在单像素成像里的作用？它解决了什么，又带来了什么风险？',
  '文中提到哪些硬件设计会显著影响成像质量或速度？',
  '这篇综述里有没有对未来方向做出明确判断？最值得追的 2-3 个方向是什么？',
  '如果我要复现一个最基础的 baseline，作者给的建议路线是什么？',
  '文章有没有提到和其它成像范式（比如计算成像/压缩感知相关方向）的联系或区别？',
  '这篇文章里有没有一处“同一句话提到多个相关工作”的地方？作者想表达的对比点是什么？',
]

const QUESTION_LIMIT_RAW = Number(process.env.PW_QUESTION_LIMIT || 0)
const QUESTION_LIMIT = Number.isFinite(QUESTION_LIMIT_RAW) && QUESTION_LIMIT_RAW > 0
  ? Math.max(1, Math.floor(QUESTION_LIMIT_RAW))
  : 0
const QUESTION_OFFSET_RAW = Number(process.env.PW_QUESTION_OFFSET || 0)
const QUESTION_OFFSET = Number.isFinite(QUESTION_OFFSET_RAW) && QUESTION_OFFSET_RAW > 0
  ? Math.max(0, Math.floor(QUESTION_OFFSET_RAW))
  : 0
const ACTIVE_QUESTIONS = (() => {
  const sliced = QUESTIONS.slice(QUESTION_OFFSET)
  return QUESTION_LIMIT > 0 ? sliced.slice(0, QUESTION_LIMIT) : sliced
})()

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
) {
  const locateBtns = assistantMsg.locator('button[aria-label="定位到原文证据"]')
  const count = await locateBtns.count()
  expect.soft(count, `no locate buttons rendered for step=${stepKey}`).toBeGreaterThan(0)
  if (count <= 0) {
    const shot = await page.screenshot({ fullPage: false })
    await attach(`no-locate-${stepKey}.png`, shot)
    return
  }

  const maxClicks = Math.min(3, count)
  for (let i = 0; i < maxClicks; i += 1) {
    const btn = locateBtns.nth(i)
    const expectedBlockId = (await btn.getAttribute('data-kb-locate-block-id')) || ''
    const expectedHeading = (await btn.getAttribute('data-kb-locate-heading')) || ''
    await btn.click()

    const reader = page.getByTestId('reader-content')
    await expect(reader).toBeVisible({ timeout: 30_000 })

    const status = page.getByTestId('reader-locate-status')
    await expect(status).toBeVisible({ timeout: 30_000 })

    // Should not degrade into fuzzy locate for strict provenance locate.
    await expect.soft(status).not.toContainText(/Fuzzy|fuzzy/i)

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
}

test.describe.serial('paper guide locate flow (recorded)', () => {
  // Full end-to-end (12 natural questions + strict locate clicks) can take a long time when deep-read is enabled.
  test.setTimeout(90 * 60_000)

  test('NatPhoton-2019: natural questions with locate jumps', async ({ page }, testInfo) => {
    await startPaperGuideFromLibrary(page)

    const attach = async (name: string, buffer: Buffer) => {
      await testInfo.attach(name, { body: buffer, contentType: 'image/png' })
    }

    for (let idx = 0; idx < ACTIVE_QUESTIONS.length; idx += 1) {
      const q = ACTIVE_QUESTIONS[idx]
      const stepKey = `q${String(QUESTION_OFFSET + idx + 1).padStart(2, '0')}`

      const beforeCount = await assistantMessages(page).count()

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

      await clickLocateButtonsAndAssert(page, msg, stepKey, attach)
    }
  })
})
